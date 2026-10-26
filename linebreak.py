#!/usr/bin/env python3

"""
Knuth-Plass optimal line-breaking algorithm.

Extracted and reimplemented from tex.p (TeX by D.E. Knuth).
Mirrors the structure of trybreak / postlinebreak / the active-list scan
as described in:
  Knuth & Plass, "Breaking Paragraphs into Lines",
  Software—Practice and Experience, 11(11):1119-1184, 1981.

Usage:
    python3 linebreak.py                      # run built-in demo
    python3 linebreak.py "your text" WIDTH    # break custom text
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INF_PENALTY = 10_000        # forbidden / forced break threshold (TeX: 10000)
INF_BAD     = 10_000        # maximum badness
AWFUL_BAD   = float("inf")  # demerits sentinel

VERY_LOOSE = 0
LOOSE      = 1
DECENT     = 2
TIGHT      = 3


# ---------------------------------------------------------------------------
# Paragraph item types
# ---------------------------------------------------------------------------

@dataclass
class Box:
    """Rigid item (character, word fragment, rule). Cannot stretch or shrink."""
    width: float
    text: str = ""


@dataclass
class Glue:
    """Flexible space: natural width w, max stretch +y, max shrink -z."""
    width: float
    stretch: float
    shrink: float


@dataclass
class Penalty:
    """Break desirability hint.
      >= +INF_PENALTY → no break allowed
      <= -INF_PENALTY → forced break
      0               → neutral
    """
    value: float
    flagged: bool = False    # True = hyphen-like (affects double-hyphen demerits)
    width: float = 0.0       # added to the line only if it breaks here (hyphen)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Params:
    line_penalty:          float = 10      # added to badness each line
    hyphen_penalty:        float = 50      # extra demerits at hyphen break
    ex_hyphen_penalty:     float = 50      # extra at explicit hyphen
    double_hyphen_demerits: float = 10000  # consecutive hyphen breaks
    adj_demerits:          float = 10000   # adjacent incompatible fit classes
    tolerance:             float = 200     # max badness before pass 2
    pre_tolerance:         float = 100     # max badness in first pass
    emergency_stretch:     float = 0       # extra stretch on final pass


# ---------------------------------------------------------------------------
# Line-width specification
# ---------------------------------------------------------------------------

@dataclass
class LineSpec:
    """Width for each line of the paragraph."""
    widths: List[float]    # widths[i] is the width of line i+1
    last: float = 0.0      # width for all lines beyond widths list

    @classmethod
    def uniform(cls, w: float) -> "LineSpec":
        return cls(widths=[], last=w)

    def width(self, line: int) -> float:
        """Return target width for the given 1-based line number."""
        if 1 <= line <= len(self.widths):
            return self.widths[line - 1]
        return self.last


# ---------------------------------------------------------------------------
# Active node (one feasible break point under consideration)
# Mirrors TeX's active node (3 halfwords of mem[]).
# ---------------------------------------------------------------------------

@dataclass
class _Active:
    pos:       int    # index into items[] of this break (-1 = start of para)
    line:      int    # line number that ends at this break (0 = before para)
    fit:       int    # fit class (VERY_LOOSE..TIGHT) of this break's line
    flagged:   bool   # was this break flagged (hyphen)?
    demerits:  float  # total accumulated demerits
    # cumulative widths where the next line starts: everything up to and
    # including the break item and the discardable items after it (§837)
    tw: float = 0.0   # total natural width
    ts: float = 0.0   # total stretch
    tk: float = 0.0   # total shrink
    prev: Optional[_Active] = None


# ---------------------------------------------------------------------------
# badness(shortfall, total_stretch_or_shrink) — direct from TeX §108
# ---------------------------------------------------------------------------

def _badness(t: float, s: float) -> float:
    # The caller handles a line shrunk beyond its shrink (§851).
    if t == 0:
        return 0.0
    if s <= 0:
        return float(INF_BAD)
    r = t / s
    b = math.floor(100.0 * r ** 3 + 0.5)   # match C++ std::round (half away from zero)
    return min(float(INF_BAD), float(b))


# ---------------------------------------------------------------------------
# The main algorithm
# ---------------------------------------------------------------------------

def _try_break(
    active: List[_Active],
    pos: int,
    pi: float,
    is_flag: bool,
    pen_w: float,
    bw: float,
    bs: float,
    bk: float,
    aw: float,
    as_: float,
    ak: float,
    spec: LineSpec,
    params: Params,
    tolerance: float,
    extra_stretch: float,
    final_pass: bool,
) -> List[_Active]:
    """
    For each active node, consider breaking at the current position (pos).
    Mirrors procedure trybreak in tex.p §829.

    bw/bs/bk are the cumulative totals before the break item, pen_w is the
    width of a penalty (hyphen) added to a line ending here, and aw/as_/ak
    are the totals where the next line starts.
    """
    if pi <= -INF_PENALTY:
        pi = float(-INF_PENALTY)    # clamp forced break
    forced = (pi == -INF_PENALTY)

    new_active: List[_Active] = []
    survivors:  List[_Active] = []

    min_dem    = [AWFUL_BAD] * 4
    best_prev: List[Optional[_Active]] = [None] * 4
    best_line  = [0] * 4
    global_min = AWFUL_BAD

    for idx, a in enumerate(active):
        # Width of candidate line from a.pos to here
        line_w = bw - a.tw + pen_w
        line_s = bs - a.ts + extra_stretch
        line_k = bk - a.tk

        line_num = a.line + 1
        target   = spec.width(line_num)
        shortfall = target - line_w

        # Classify and compute badness (§851-853)
        if shortfall > 0:
            b = _badness(shortfall, line_s)
            if   b > 99: fc = VERY_LOOSE
            elif b > 12: fc = LOOSE
            else:        fc = DECENT
        elif shortfall < 0:
            if -shortfall > line_k:
                b = float(INF_BAD + 1)    # cannot shrink enough (§851)
            else:
                b = _badness(-shortfall, line_k)
            fc = TIGHT if b > 12 else DECENT
        else:
            b  = 0.0
            fc = DECENT

        # Deactivation rule (§851, §854 in tex.p):
        # - b > INF_BAD (overfull beyond shrink): future items only add
        #   width, so deactivate the node.
        # - Forced break: no line may span it, so deactivate every node.
        # - Otherwise keep the node active, even if b > tolerance, since
        #   future items may fill the line.
        # A deactivated node may still end a line here if b <= tolerance.
        # On the final pass, the last remaining node is never lost: it breaks
        # here anyway with artificial demerits 0 (an overfull or underfull
        # line), so a paragraph can always be set.
        artificial = False
        if b > INF_BAD or forced:
            if (final_pass and global_min == AWFUL_BAD and not survivors
                    and idx == len(active) - 1):
                artificial = True
            elif b > tolerance:
                continue    # deactivate
        else:
            survivors.append(a)
            if b > tolerance:
                continue    # too bad to break here, but stays active

        # Demerits (§855, §859)
        if artificial:
            d = 0.0
        else:
            d = (params.line_penalty + b) ** 2
            if pi > 0:
                d += pi * pi
            elif pi > -INF_PENALTY:
                d -= pi * pi
            if is_flag and a.flagged:
                d += params.double_hyphen_demerits
            if abs(fc - a.fit) > 1:
                d += params.adj_demerits

        d += a.demerits

        if d <= min_dem[fc]:
            min_dem[fc]   = d
            best_prev[fc] = a
            best_line[fc] = line_num
            if d < global_min:
                global_min = d

    # NOTE: TeX also groups nodes by line number (§835) so that varying
    # LineSpec widths work; here all line numbers share min_dem.
    # Create new active nodes for the best break in each fit class (§845)
    if global_min < AWFUL_BAD:
        for fc in range(4):
            if min_dem[fc] <= global_min + params.adj_demerits:
                node = _Active(
                    pos      = pos,
                    line     = best_line[fc],
                    fit      = fc,
                    flagged  = is_flag,
                    demerits = min_dem[fc],
                    tw = aw,
                    ts = as_,
                    tk = ak,
                    prev = best_prev[fc],
                )
                new_active.append(node)

    return survivors + new_active


def _linebreak(
    items: list,
    spec: LineSpec,
    params: Params,
    tolerance: float,
    extra_stretch: float = 0.0,
    final_pass: bool = False,
) -> Optional[List[_Active]]:
    """
    Single pass of the Knuth-Plass algorithm with given tolerance.

    Returns the final active list (non-empty = success) or None.
    Each active node that survives until the forced end-break is a
    candidate for the optimal solution.

    The passive chain is encoded directly in the _Active objects:
    we store a `prev` pointer to enable traceback.
    """
    # Seed the active list: one node representing the start of the paragraph.
    seed = _Active(pos=-1, line=0, fit=DECENT, flagged=False, demerits=0, prev=None)
    active: List[_Active] = [seed]

    # Running sums of natural width / stretch / shrink from start of paragraph
    # to the position just *before* the current item (updated as we scan).
    cum_w = cum_s = cum_k = 0.0

    for i, item in enumerate(items):
        # -----------------------------------------------------------------
        # Determine if this position is a legal break point, and if so
        # what penalty applies.
        # Legal breaks (mirroring TeX §813):
        #   - Glue node, if preceded by a Box
        #   - Penalty node with value < INF_PENALTY
        # -----------------------------------------------------------------
        is_break = False
        pi       = 0.0
        is_flag  = False

        if isinstance(item, Glue):
            # Legal only after a box (TeX rule)
            if i > 0 and isinstance(items[i - 1], Box):
                is_break = True
                pi       = 0.0
                is_flag  = False
        elif isinstance(item, Penalty):
            if item.value < INF_PENALTY:
                is_break = True
                pi       = item.value
                is_flag  = item.flagged

        # -----------------------------------------------------------------
        # A line ending HERE has the cumulative totals cum_* (the break item
        # has not been added yet), plus the penalty width for a hyphen.
        # Compute break_width (§837): the break item and the glue/penalties
        # after it, up to the next box, are discarded, so the next line
        # starts after them.  They are added to cum_* as the scan goes on,
        # so the new node's totals must include them.
        # -----------------------------------------------------------------
        if is_break:
            aw, as_, ak = cum_w, cum_s, cum_k
            j = i
            while j < len(items) and not isinstance(items[j], Box):
                if isinstance(items[j], Glue):
                    aw  += items[j].width
                    as_ += items[j].stretch
                    ak  += items[j].shrink
                j += 1
            pen_w = item.width if isinstance(item, Penalty) else 0.0
            active = _try_break(
                active, i, pi, is_flag, pen_w, cum_w, cum_s, cum_k,
                aw, as_, ak, spec, params, tolerance, extra_stretch, final_pass
            )

        # -----------------------------------------------------------------
        # Update running sums (AFTER try_break, matching TeX's scan order)
        # -----------------------------------------------------------------
        if isinstance(item, Box):
            cum_w += item.width
        elif isinstance(item, Glue):
            cum_w += item.width
            cum_s += item.stretch
            cum_k += item.shrink
        # Penalty has no width

    # ---------------------------------------------------------------------
    # Collect complete-paragraph nodes.
    # text_to_items() terminates with Glue(INF stretch) + Penalty(-INF).
    # The try_break for Penalty(-INF) creates active nodes whose position
    # is that final penalty index — these represent complete paragraphs.
    # All nodes in `active` whose `pos` equals the final item index are
    # end-of-paragraph nodes; pick the one with minimum total demerits.
    # If there are none, this pass failed (never on the final pass).
    # ---------------------------------------------------------------------
    final_penalty_pos = len(items) - 1   # index of the Penalty(-INF) item

    # Nodes created at the final penalty position are complete paragraphs.
    final_nodes = [a for a in active if a.pos == final_penalty_pos]
    return final_nodes or None


def _traceback(final_node: _Active) -> List[int]:
    """Follow prev pointers from final_node back to the seed, collect break positions."""
    breaks  = []
    cur: Optional[_Active] = final_node
    while cur is not None:
        if cur.pos >= 0:
            breaks.append(cur.pos)
        cur = cur.prev
    breaks.reverse()
    return breaks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def break_paragraph(
    items: list,
    spec: LineSpec,
    params: Optional[Params] = None,
) -> List[int]:
    """
    Find optimal line breaks for a paragraph represented as Box/Glue/Penalty items.

    Returns a list of item indices where breaks occur.  Each index points to
    the Glue or Penalty item at which the break is taken (glue is discarded;
    a flagged Penalty represents a hyphen).

    Tries up to three passes (pre_tolerance → tolerance → emergency_stretch),
    mirroring TeX §828 and §863: the first pass is skipped if pre_tolerance
    < 0, and the second is final if emergency_stretch <= 0.  The final pass
    always sets the paragraph, with overfull or underfull lines if needed.

    Raises ValueError only if items do not end with a forced break.
    """
    if params is None:
        params = Params()

    passes = []
    if params.pre_tolerance >= 0:
        passes.append((params.pre_tolerance, 0.0, False))
    passes.append((params.tolerance, 0.0, params.emergency_stretch <= 0))
    if params.emergency_stretch > 0:
        passes.append((params.tolerance, params.emergency_stretch, True))

    for tolerance, extra, final_pass in passes:
        result = _linebreak(items, spec, params, tolerance, extra, final_pass)
        if result:
            # Pick the node with lowest total demerits
            best = min(result, key=lambda n: n.demerits)
            return _traceback(best)

    raise ValueError(
        "Cannot break paragraph: items must end with a forced break."
    )


# ---------------------------------------------------------------------------
# post_line_break: turn break indices into rendered lines
# Mirrors procedure postlinebreak in tex.p §877.
# ---------------------------------------------------------------------------

def post_line_break(
    items: list,
    breaks: List[int],
    spec: LineSpec,
) -> List[str]:
    """
    Render paragraph into lines given break positions.

    Returns a list of (line_string, badness, fit_description) tuples
    as formatted strings.
    """
    lines   = []
    starts  = [-1] + breaks        # break position before each line
    ends    = breaks + [len(items)]

    for line_num, (start, end) in enumerate(zip(starts, ends), start=1):
        # Collect items for this line
        # Start after the break item; end before/at the break item.
        seg_start = start + 1
        seg_end   = end   # break item itself is not on the line (glue discarded)

        segment = items[seg_start:seg_end]

        # Strip leading glue (§879 in tex.p)
        while segment and isinstance(segment[0], Glue):
            segment = segment[1:]
        # Strip trailing Penalty only; keep trailing Glue (parfillskip absorbs shortfall)

        # Measure
        nat     = sum(i.width   if isinstance(i, (Box, Glue)) else 0 for i in segment)
        stretch = sum(i.stretch if isinstance(i, Glue)        else 0 for i in segment)
        shrink  = sum(i.shrink  if isinstance(i, Glue)        else 0 for i in segment)
        if end < len(items) and isinstance(items[end], Penalty):
            nat += items[end].width    # hyphen width when breaking at a penalty

        target    = spec.width(line_num)
        shortfall = target - nat

        if shortfall > 0 and stretch > 0:
            ratio   = shortfall / stretch
            badness = _badness(shortfall, stretch)
            fit     = "very loose" if badness > 99 else ("loose" if badness > 12 else "decent")
        elif shortfall < 0 and -shortfall > shrink:
            ratio   = -1.0     # shrink all it can; the rest sticks out
            badness = float(INF_BAD + 1)
            fit     = "overfull"
        elif shortfall < 0:
            ratio   = shortfall / shrink   # negative
            badness = _badness(-shortfall, shrink)
            fit     = "tight"
        else:
            ratio   = 0.0
            badness = 0.0
            fit     = "perfect"

        # Render
        is_last = (end == len(items))
        hyphen  = ""
        if not is_last and end < len(items) and isinstance(items[end], Penalty):
            if items[end].flagged:
                hyphen = "-"

        if not segment:
            continue   # skip empty trailing "lines" from the paragraph terminator
        rendered = _render(segment, ratio) + hyphen
        lines.append(f"{rendered:<{int(target)}}  [bad={badness:5.0f}  {fit}]")

    return lines


def _render(segment: list, ratio: float) -> str:
    parts = []
    for item in segment:
        if isinstance(item, Box):
            parts.append(item.text)
        elif isinstance(item, Glue):
            if ratio >= 0:
                w = item.width + ratio * item.stretch
            else:
                w = item.width + ratio * item.shrink
            parts.append(" " * max(1, math.floor(w + 0.5)))
        # A Penalty inside a line is a break not taken: draw nothing.  The
        # hyphen of a line ending at a flagged Penalty is added by the caller.
    return "".join(parts)


# ---------------------------------------------------------------------------
# Convenience: build items from plain text (word-level, monospaced)
# ---------------------------------------------------------------------------

def text_to_items(
    text: str,
    space_width:   float = 1.0,
    space_stretch: float = 0.5,
    space_shrink:  float = 0.333,
) -> list:
    """
    Convert a plain string into Box/Glue/Penalty items.
    Words are single Boxes; spaces become Glue.
    A final infinite-stretch Glue + forced Penalty terminates the paragraph.
    """
    items: list = []
    words = text.split()
    for wi, word in enumerate(words):
        items.append(Box(width=float(len(word)), text=word))
        if wi < len(words) - 1:
            items.append(Glue(space_width, space_stretch, space_shrink))
    # paragraph terminator
    items.append(Glue(0, float(INF_PENALTY), 0))
    items.append(Penalty(float(-INF_PENALTY)))
    return items


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

_DEMO_TEXT = (
    "In olden times when wishing still helped one there lived a king "
    "whose daughters were all beautiful but the youngest was so beautiful "
    "that the sun itself which has seen so much was astonished whenever it "
    "shone in her face. Close by the kings castle lay a great dark forest "
    "and under an old lime tree in the forest was a well and when the day "
    "was very warm the kings child went out into the forest and sat down "
    "by the side of the cool fountain and when she was bored she took a "
    "golden ball and threw it up on high and caught it and this ball was "
    "her greatest delight."
)


def demo(text: str = _DEMO_TEXT, width: int = 72):
    items  = text_to_items(text)
    spec   = LineSpec.uniform(float(width))
    params = Params(tolerance=200, pre_tolerance=100, line_penalty=10)

    breaks = break_paragraph(items, spec, params)
    lines  = post_line_break(items, breaks, spec)

    print(f"Paragraph broken into {len(lines)} lines  (target width = {width})")
    print("─" * (width + 22))
    for ln, line in enumerate(lines, 1):
        print(f"  {ln:2d}: {line}")
    print("─" * (width + 22))
    words_used = len([i for i in items if isinstance(i, Box)])
    print(f"  {words_used} words, {len(breaks)} break points: {breaks}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        t = sys.argv[1]
        w = int(sys.argv[2]) if len(sys.argv) >= 3 else 72
        demo(t, w)
    else:
        demo()
