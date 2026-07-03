#!/usr/bin/env python3

"""
textrender.py — render a plain-text file to PNG.

Python port of textrender.cpp using the same underlying C libraries:
  linebreak.py   — Knuth-Plass line-breaking
  uharfbuzz      — HarfBuzz text shaping
  freetype-py    — FreeType font loading
  pycairo        — Cairo PNG rendering
  pyphen         — English word hyphenation
  uniseg         — UAX #14 Unicode line-break segmentation
"""

import sys
import os
import ctypes
import argparse
from pathlib import Path

import uharfbuzz as hb
import cairo
import freetype
import pyphen
from uniseg.linebreak import line_break_units

sys.path.insert(0, str(Path(__file__).parent))
from linebreak import Box, Glue, Penalty, Params, LineSpec, break_paragraph, INF_PENALTY

# ---------------------------------------------------------------------------
# ctypes bridge: FreeType→Cairo font face
# pycairo 1.29 lacks cairo.FTFontFace, so we call libcairo directly.
# We also extract raw pointers from pycairo objects via CPython struct layout:
#   PycairoObject { ob_refcnt, ob_type, <cairo ptr>, ... }
# ---------------------------------------------------------------------------

_libcairo = ctypes.CDLL("libcairo.so.2")
_libcairo.cairo_ft_font_face_create_for_ft_face.restype  = ctypes.c_void_p
_libcairo.cairo_ft_font_face_create_for_ft_face.argtypes = [ctypes.c_void_p, ctypes.c_int]
_libcairo.cairo_set_font_face.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_libcairo.cairo_set_font_size.argtypes = [ctypes.c_void_p, ctypes.c_double]
_libcairo.cairo_show_glyphs.argtypes   = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_libcairo.cairo_font_face_destroy.argtypes = [ctypes.c_void_p]

_PTR_SIZE = ctypes.sizeof(ctypes.c_void_p)


class _CairoGlyph(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_ulong),
        ("x",     ctypes.c_double),
        ("y",     ctypes.c_double),
    ]


def _raw_ptr(pycairo_obj) -> int:
    """Extract the raw Cairo pointer from a pycairo wrapper object."""
    return ctypes.c_void_p.from_address(id(pycairo_obj) + 2 * _PTR_SIZE).value


def _ft_face_ptr(ft_face: freetype.Face) -> int:
    """Extract the raw FT_Face pointer from a freetype-py Face object."""
    return ctypes.cast(ft_face._FT_Face, ctypes.c_void_p).value


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_FONT     = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
DEFAULT_DICT     = "/usr/share/hyphen/hyph_en_US.dic"

# ---------------------------------------------------------------------------
# Script detection (CSS text-autospace: normal)
# ---------------------------------------------------------------------------

def _is_cjk(cp: int) -> bool:
    """True for CJK ideographs and adjacent blocks (matches C++ is_cjk ranges)."""
    return ((0x3000  <= cp <= 0x9FFF)  or
            (0xF900  <= cp <= 0xFAFF)  or
            (0x20000 <= cp <= 0x2FA1F))


def _is_cjk_segmentation(cp: int) -> bool:
    """True for characters that ICU UAX#14 treats as individually breakable CJK.
    Extends _is_cjk to include fullwidth forms (FF0C '，', 3002 '。', etc.)
    which ICU gives line-break class ID (Ideographic).
    """
    return (_is_cjk(cp) or
            (0x2E80 <= cp <= 0x2FFF) or   # CJK radicals, Kangxi
            (0xFF01 <= cp <= 0xFF60) or   # fullwidth ASCII/punct
            (0xFFE0 <= cp <= 0xFFE6))


def _is_latin_or_digit(cp: int) -> bool:
    return ((0x41  <= cp <= 0x5A) or
            (0x61  <= cp <= 0x7A) or
            (0x30  <= cp <= 0x39) or
            (0xC0  <= cp <= 0x24F))

# ---------------------------------------------------------------------------
# HarfBuzz helpers
# ---------------------------------------------------------------------------

def _hb_advance_px(font: hb.Font, text: str) -> float:
    """Measure shaped advance in pixels."""
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    return sum(p.x_advance for p in buf.glyph_positions) / 64.0


def _shape_to_glyphs(font: hb.Font, text: str, pen_x: float, pen_y: float):
    """Return [(glyph_id, x, y), ...].
    Always uses HarfBuzz x_advance for pen advancement (includes GPOS kerning).
    """
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    glyphs = []
    x, y = pen_x, pen_y
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        glyphs.append((info.codepoint,
                       x + pos.x_offset / 64.0,
                       y - pos.y_offset / 64.0))
        x += pos.x_advance / 64.0
        y -= pos.y_advance / 64.0
    return glyphs

# ---------------------------------------------------------------------------
# Hyphenation (pyphen)
# ---------------------------------------------------------------------------

def _hyphen_breaks(dic, word: str):
    """Return 0-based byte indices i after which a hyphen may be inserted."""
    if dic is None or len(word) < 4:
        return []
    n = len(word)
    # pyphen position p → break before word[p], i.e. after word[p-1]
    return [p - 1 for p in dic.positions(word.lower()) if 2 <= p <= n - 2]

# ---------------------------------------------------------------------------
# Paragraph segmentation (UAX #14 via uniseg)
# ---------------------------------------------------------------------------

def _segment_paragraph(text: str):
    """
    Yield (word, has_trailing_space) tuples using UAX #14 line-break units
    (uniseg.linebreak.line_break_units).  Each chunk from uniseg is stripped of
    its trailing space; has_trailing_space is True when a space was present.
    """
    for chunk in line_break_units(text):
        word = chunk.rstrip(" ")
        if not word:
            continue
        has_space = len(word) < len(chunk)
        yield word, has_space

# ---------------------------------------------------------------------------
# Build Knuth-Plass item list from a paragraph
# ---------------------------------------------------------------------------

def _build_para_items(hb_font, para: str, space_w: float, space_s: float,
                      space_k: float, pt_size: float, dic) -> list:
    cjk_stretch = space_w * 0.5
    autospace   = pt_size * 0.25
    autospace_s = pt_size * 0.05

    UNKNOWN, CJK, LATIN_DIGIT = 0, 1, 2
    last_script = UNKNOWN
    items: list = []

    for word, has_space in _segment_paragraph(para):
        cp = ord(word[0])
        cur_script = (CJK        if _is_cjk(cp)            else
                      LATIN_DIGIT if _is_latin_or_digit(cp) else
                      UNKNOWN)

        # CSS text-autospace: widen the preceding Glue at script boundaries.
        if (last_script != UNKNOWN and cur_script != UNKNOWN
                and last_script != cur_script and items):
            for k in range(len(items) - 1, -1, -1):
                if isinstance(items[k], Glue):
                    g = items[k]
                    items[k] = Glue(g.width + autospace,
                                    g.stretch + autospace_s,
                                    g.shrink)
                    break
                if isinstance(items[k], Box):
                    items.append(Glue(autospace, autospace_s, 0.0))
                    break

        if cur_script != UNKNOWN:
            last_script = cur_script

        if has_space:
            # Latin word (with trailing space): optionally hyphenate.
            if dic:
                hbreaks = _hyphen_breaks(dic, word)
                if hbreaks:
                    prev = 0
                    for bp in hbreaks:
                        syl = word[prev:bp + 1]
                        items.append(Box(_hb_advance_px(hb_font, syl), syl))
                        items.append(Penalty(50.0, True))
                        prev = bp + 1
                    last_syl = word[prev:]
                    items.append(Box(_hb_advance_px(hb_font, last_syl), last_syl))
                else:
                    items.append(Box(_hb_advance_px(hb_font, word), word))
            else:
                items.append(Box(_hb_advance_px(hb_font, word), word))
            items.append(Glue(space_w, space_s, space_k))
        else:
            # CJK char or last Latin word (no trailing space).
            w = _hb_advance_px(hb_font, word)
            items.append(Box(w, word))
            if _is_cjk_segmentation(cp):
                items.append(Glue(0.0, cjk_stretch, 0.0))

    # Strip trailing Glue; add paragraph terminator (parfillskip + forced break).
    while items and isinstance(items[-1], Glue):
        items.pop()
    items.append(Glue(0.0, float(INF_PENALTY), 0.0))
    items.append(Penalty(float(-INF_PENALTY)))
    return items

# ---------------------------------------------------------------------------
# Split text into paragraphs (blank-line separated)
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str):
    paras, cur = [], []
    for line in text.splitlines():
        if not line.strip():
            if cur:
                paras.append(" ".join(cur))
                cur = []
        else:
            cur.append(line.rstrip())
    if cur:
        paras.append(" ".join(cur))
    return paras

def _first_fit(items: list, text_w: float, hyphen_w: float = 0.0) -> list[int]:
    """Simple first-fit line breaking."""
    breaks = []
    acc = 0.0
    last_break_opportunity = -1
    
    i = 0
    n = len(items)
    while i < n:
        it = items[i]
        if isinstance(it, Box):
            acc += it.width
            i += 1
        elif isinstance(it, Glue):
            if acc + it.width > text_w:
                if last_break_opportunity != -1:
                    breaks.append(last_break_opportunity)
                    i = last_break_opportunity + 1
                    while i < n and isinstance(items[i], Glue):
                        i += 1
                    acc = 0.0
                    last_break_opportunity = -1
                else:
                    breaks.append(i)
                    i += 1
                    acc = 0.0
                    last_break_opportunity = -1
            else:
                acc += it.width
                last_break_opportunity = i
                i += 1
        elif isinstance(it, Penalty):
            if it.value <= -INF_PENALTY:
                if acc > text_w and last_break_opportunity != -1:
                    breaks.append(last_break_opportunity)
                    i = last_break_opportunity + 1
                    while i < n and isinstance(items[i], Glue):
                        i += 1
                    acc = 0.0
                    last_break_opportunity = -1
                else:
                    breaks.append(i)
                    i += 1
                    acc = 0.0
                    last_break_opportunity = -1
            elif it.value < INF_PENALTY:
                extra_w = hyphen_w if it.flagged else 0.0
                if acc + extra_w > text_w:
                    if last_break_opportunity != -1:
                        breaks.append(last_break_opportunity)
                        i = last_break_opportunity + 1
                        while i < n and isinstance(items[i], Glue):
                            i += 1
                        acc = 0.0
                        last_break_opportunity = -1
                    else:
                        breaks.append(i)
                        i += 1
                        acc = 0.0
                        last_break_opportunity = -1
                else:
                    last_break_opportunity = i
                    i += 1
            else:
                i += 1
        else:
            i += 1
            
    if items and (not breaks or breaks[-1] != n - 1):
        breaks.append(n - 1)
        
    return breaks


def render_to_io(text, font_path, pt_size, page_w, margin=10, leading=1.4, tolerance=200.0, hyphen_on=True, fg=(0.0, 0.0, 0.0), bg=(1.0, 1.0, 1.0), font_index=0, justification=True):
    import io
    
    # --- FreeType ---
    ft_face = freetype.Face(font_path, index=font_index)
    ft_face.set_char_size(int(pt_size * 64), 0, 72, 72)

    # --- HarfBuzz ---
    font_data = Path(font_path).read_bytes()
    hb_blob = hb.Blob(font_data)
    hb_face = hb.Face(hb_blob, font_index)
    hb_font = hb.Font(hb_face)
    hb_font.scale = (int(pt_size * 64), int(pt_size * 64))

    # --- Spacing ---
    space_w = _hb_advance_px(hb_font, " ")
    space_s = space_w / 2.0
    space_k = space_w / 3.0
    text_w  = page_w - 2 * margin
    line_h  = pt_size * leading
    hyphen_w = _hb_advance_px(hb_font, "-")

    # --- Hyphenation ---
    dic = None
    if hyphen_on:
        try:
            dic = pyphen.Pyphen(lang="en_US")
        except:
            dic = None

    # --- Knuth-Plass params ---
    kp_params = Params(tolerance=tolerance, emergency_stretch=pt_size)

    # First pass: calculate height and store layout info
    paras = _split_paragraphs(text)
    prepared_paras = []
    estimated_height = margin * 2 + pt_size
    
    for para in paras:
        items = _build_para_items(hb_font, para, space_w, space_s, space_k, pt_size, dic)
        if len(items) <= 2:
            continue
        
        spec = LineSpec.uniform(text_w)
        if justification:
            try:
                breaks = break_paragraph(items, spec, kp_params)
            except ValueError:
                # simple greedy fallback
                print("Fallback")
                breaks = _first_fit(items, text_w, hyphen_w)
        else:
            breaks = _first_fit(items, text_w, hyphen_w)
        
        prepared_paras.append((items, breaks))
        num_lines = len(breaks)
        estimated_height += num_lines * line_h + line_h * 0.5
    
    page_h = int(estimated_height + margin)

    # --- Cairo surface ---
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, page_w, page_h)
    cr = cairo.Context(surface)
    cr.set_source_rgb(*bg)
    cr.paint()
    cr.set_source_rgb(*fg)

    # --- Cairo font face via ctypes FT bridge ---
    raw_cr   = _raw_ptr(cr)
    ff_ptr   = _libcairo.cairo_ft_font_face_create_for_ft_face(_ft_face_ptr(ft_face), 0)
    _libcairo.cairo_set_font_face(raw_cr, ff_ptr)
    _libcairo.cairo_set_font_size(raw_cr, ctypes.c_double(pt_size))

    baseline_y = margin + pt_size
    for items, breaks in prepared_paras:
        starts = [-1] + breaks
        ends   = breaks + [len(items)]

        for seg_start_br, seg_end in zip(starts, ends):
            seg_start = seg_start_br + 1
            while seg_start < seg_end and isinstance(items[seg_start], Glue):
                seg_start += 1
            if seg_start >= seg_end: continue

            line_ends_hyphen = False
            line_hyphen_w = 0.0
            if (seg_end < len(items) and isinstance(items[seg_end], Penalty) and items[seg_end].flagged):
                line_ends_hyphen = True
                line_hyphen_w = hyphen_w

            nat = line_hyphen_w
            stretch = shrink = 0.0
            for it in items[seg_start:seg_end]:
                if isinstance(it, Box): nat += it.width
                elif isinstance(it, Glue):
                    nat += it.width
                    stretch += it.stretch
                    shrink  += it.shrink

            sfall = text_w - nat
            ratio = 0.0
            if justification:
                if sfall > 0 and stretch > 0: ratio = sfall / stretch
                elif sfall < 0 and shrink > 0: ratio = sfall / shrink

            pen_x  = float(margin)
            glyphs = []
            for it in items[seg_start:seg_end]:
                if isinstance(it, Box):
                    g = _shape_to_glyphs(hb_font, it.text, pen_x, baseline_y)
                    glyphs.extend(g)
                    pen_x += it.width
                elif isinstance(it, Glue):
                    w = it.width + ratio * (it.stretch if ratio >= 0 else it.shrink)
                    pen_x += w

            if line_ends_hyphen:
                g = _shape_to_glyphs(hb_font, "-", pen_x, baseline_y)
                glyphs.extend(g)

            if glyphs:
                arr = (_CairoGlyph * len(glyphs))(*(_CairoGlyph(int(gid), x, y) for gid, x, y in glyphs))
                _libcairo.cairo_show_glyphs(raw_cr, arr, len(glyphs))

            baseline_y += line_h
        baseline_y += line_h * 0.5

    _libcairo.cairo_font_face_destroy(ff_ptr)

    img_io = io.BytesIO()
    surface.write_to_png(img_io)
    img_io.seek(0)
    return img_io

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Render a plain-text file to PNG (Knuth-Plass + HarfBuzz + Cairo)")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--font",      default=DEFAULT_FONT)
    ap.add_argument("--size",      type=float, default=24.0,  metavar="N")
    ap.add_argument("--width",     type=int,   default=600,   metavar="N")
    ap.add_argument("--margin",    type=int,   default=10,    metavar="N")
    ap.add_argument("--leading",   type=float, default=1.4,   metavar="N")
    ap.add_argument("--tolerance", type=float, default=200.0, metavar="N")
    ap.add_argument("--nohyphen",  action="store_true")
    ap.add_argument("--hyphen-dict", default="", metavar="PATH")
    ap.add_argument("--tracingparagraphs", action="store_true")
    args = ap.parse_args()

    pt_size   = args.size
    page_w    = args.width
    margin    = args.margin
    leading   = args.leading
    tolerance = args.tolerance

    text = Path(args.input).read_text(encoding="utf-8")

    img_io = render_to_io(
        text=text,
        font_path=args.font,
        pt_size=pt_size,
        page_w=page_w,
        margin=margin,
        leading=leading,
        tolerance=tolerance,
        hyphen_on=not args.nohyphen,
    )

    with open(args.output, "wb") as f:
        f.write(img_io.getvalue())
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
