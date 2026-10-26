#pragma once
// Knuth-Plass optimal line-breaking algorithm — public API.
//
// Extracted from tex/linebreak.cpp.  Include this header in translation units
// that need break_paragraph() but not the text-rendering helpers.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <variant>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr double INF_PENALTY = 10'000.0;
static constexpr double INF_BAD     = 10'000.0;
static constexpr double AWFUL_BAD   = std::numeric_limits<double>::infinity();  // demerits sentinel

enum FitClass { VERY_LOOSE = 0, LOOSE = 1, DECENT = 2, TIGHT = 3 };

// ---------------------------------------------------------------------------
// Paragraph item types
// ---------------------------------------------------------------------------

struct Box {
    double      width;
    std::string text;
};

struct Glue {
    double width;
    double stretch;
    double shrink;
};

struct Penalty {
    double value;
    bool   flagged = false;
    double width   = 0.0;   // added to the line only if it breaks here (hyphen)
};

using Item = std::variant<Box, Glue, Penalty>;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct Params {
    double line_penalty           = 10.0;
    double double_hyphen_demerits = 10'000.0;
    double adj_demerits           = 10'000.0;
    double tolerance              = 200.0;
    double pre_tolerance          = 100.0;
    double emergency_stretch      = 0.0;
};

// ---------------------------------------------------------------------------
// Line-width specification
// ---------------------------------------------------------------------------

struct LineSpec {
    std::vector<double> widths;
    double              last;

    static LineSpec uniform(double w) { return {{}, w}; }

    double width(int line) const {
        if (line >= 1 && line <= static_cast<int>(widths.size()))
            return widths[static_cast<size_t>(line - 1)];
        return last;
    }
};

// ---------------------------------------------------------------------------
// Active node
// ---------------------------------------------------------------------------

struct Active {
    int    pos;
    int    line;
    int    fit;
    bool   flagged;
    double demerits;
    // cumulative widths where the next line starts: everything up to and
    // including the break item and the discardable items after it (§837)
    double tw, ts, tk;
    int    serial = 0;          // tracing: serial number (@@N in TeX output)
    std::shared_ptr<Active> prev;
};

using ActivePtr = std::shared_ptr<Active>;

// ---------------------------------------------------------------------------
// badness(shortfall, total_stretch_or_shrink) — TeX §108
// ---------------------------------------------------------------------------

inline double badness(double t, double s) {
    // The caller handles a line shrunk beyond its shrink (§851).
    if (t == 0.0) return 0.0;
    if (s <= 0.0) return INF_BAD;
    double r = t / s;
    return std::min(INF_BAD, std::round(100.0 * r * r * r));
}

// ---------------------------------------------------------------------------
// Single pass of the Knuth-Plass algorithm.
//
// final_pass: never lose the last active node, so a paragraph can always be
// set (with overfull or underfull lines if needed).  Returns the nodes that
// end the paragraph, or none if this pass failed.
//
// trace_out: if non-null, emit \tracingparagraphs-style diagnostics.
// ---------------------------------------------------------------------------

inline std::vector<ActivePtr> linebreak_pass(
    const std::vector<Item>& items,
    const LineSpec&           spec,
    const Params&             params,
    double                    tolerance,
    double                    extra_stretch,
    bool                      final_pass,
    std::ostream*             trace_out = nullptr)
{
    // seed node: serial 0 (@@0 = paragraph start)
    auto seed = std::make_shared<Active>(Active{
        -1, 0, DECENT, false, 0.0, 0.0, 0.0, 0.0, 0, nullptr
    });
    std::vector<ActivePtr> active{seed};

    int serial_ctr = 1;  // next serial to assign; 0 is the seed

    double cum_w = 0.0, cum_s = 0.0, cum_k = 0.0;
    const int n = static_cast<int>(items.size());

    for (int i = 0; i < n; ++i) {
        const Item& item = items[static_cast<size_t>(i)];

        bool   is_break = false;
        double pi       = 0.0;
        bool   is_flag  = false;

        if (auto* g = std::get_if<Glue>(&item)) {
            (void)g;
            if (i > 0 && std::holds_alternative<Box>(items[static_cast<size_t>(i - 1)]))
                is_break = true;
        } else if (auto* pen = std::get_if<Penalty>(&item)) {
            if (pen->value < INF_PENALTY) {
                is_break = true;
                pi       = pen->value;
                is_flag  = pen->flagged;
            }
        }

        if (is_break) {
            // A line ending HERE has the cumulative totals cum_* (the break
            // item has not been added yet), plus the penalty width for a
            // hyphen.  Compute break_width (§837): the break item and the
            // glue/penalties after it, up to the next box, are discarded, so
            // the next line starts after them.  They are added to cum_* as
            // the scan goes on, so the new node's totals must include them.
            const double bw = cum_w, bs = cum_s, bk = cum_k;
            double aw = cum_w, as = cum_s, ak = cum_k;
            for (int j = i; j < n && !std::holds_alternative<Box>(items[static_cast<size_t>(j)]); ++j) {
                if (auto* g = std::get_if<Glue>(&items[static_cast<size_t>(j)])) {
                    aw += g->width;
                    as += g->stretch;
                    ak += g->shrink;
                }
            }
            const auto*  pen_item = std::get_if<Penalty>(&item);
            const double pen_w    = pen_item ? pen_item->width : 0.0;
            if (pi <= -INF_PENALTY) pi = -INF_PENALTY;
            const bool forced = (pi == -INF_PENALTY);

            const char* itype = std::holds_alternative<Glue>(item)
                ? "\\glue" : "\\penalty";

            std::vector<ActivePtr> survivors;
            std::vector<ActivePtr> new_nodes;

            double    min_dem[4]   = {AWFUL_BAD, AWFUL_BAD, AWFUL_BAD, AWFUL_BAD};
            ActivePtr best_prev[4] = {};
            int       best_line[4] = {};
            double    global_min   = AWFUL_BAD;

            for (size_t idx = 0; idx < active.size(); ++idx) {
                const ActivePtr& a = active[idx];
                const double line_w = bw - a->tw + pen_w;
                const double line_s = bs - a->ts + extra_stretch;
                const double line_k = bk - a->tk;
                const int    lnum   = a->line + 1;
                const double target = spec.width(lnum);
                const double sfall  = target - line_w;

                double b  = 0.0;
                int    fc = DECENT;
                if (sfall > 0.0) {
                    b  = badness(sfall, line_s);
                    fc = (b > 99.0) ? VERY_LOOSE : (b > 12.0) ? LOOSE : DECENT;
                } else if (sfall < 0.0) {
                    if (-sfall > line_k)
                        b = INF_BAD + 1.0;  // cannot shrink enough (§851)
                    else
                        b = badness(-sfall, line_k);
                    fc = (b > 12.0) ? TIGHT : DECENT;
                }

                // Deactivation rule (§851, §854 in tex.p):
                // - b > INF_BAD (overfull beyond shrink): future items only
                //   add width, so deactivate the node.
                // - Forced break: no line may span it, so deactivate every node.
                // - Otherwise keep the node active, even if b > tolerance,
                //   since future items may fill the line.
                // A deactivated node may still end a line here if b <= tolerance.
                // On the final pass, the last remaining node is never lost: it
                // breaks here anyway with artificial demerits 0 (an overfull or
                // underfull line), so a paragraph can always be set.
                bool artificial = false;
                if (b > INF_BAD || forced) {
                    if (final_pass && global_min == AWFUL_BAD && survivors.empty()
                            && idx == active.size() - 1)
                        artificial = true;
                    else if (b > tolerance)
                        continue;  // deactivated — don't trace
                } else {
                    survivors.push_back(a);
                    if (b > tolerance)
                        continue;  // too bad to break here, but stays active
                }

                double d = 0.0;
                if (!artificial) {
                    d = (params.line_penalty + b) * (params.line_penalty + b);
                    if (pi > 0.0)               d += pi * pi;
                    else if (pi > -INF_PENALTY)  d -= pi * pi;
                    if (is_flag && a->flagged)   d += params.double_hyphen_demerits;
                    if (std::abs(fc - a->fit) > 1) d += params.adj_demerits;
                }
                d += a->demerits;

                // Tracing: @\glue via @@N b=B p=P d=D
                if (trace_out) {
                    *trace_out << "@" << itype
                               << " via @@" << a->serial
                               << " b=" << static_cast<long long>(b)
                               << " p=" << static_cast<long long>(pi)
                               << " d=";
                    if (artificial)
                        *trace_out << "*";
                    else
                        *trace_out << static_cast<long long>(d - a->demerits);
                    *trace_out << "\n";
                }

                if (d <= min_dem[fc]) {
                    min_dem[fc]   = d;
                    best_prev[fc] = a;
                    best_line[fc] = lnum;
                    if (d < global_min) global_min = d;
                }
            }

            // NOTE: TeX also groups nodes by line number (§835) so that
            // varying LineSpec widths work; here all line numbers share min_dem.
            if (global_min < AWFUL_BAD) {
                for (int fc = 0; fc < 4; ++fc) {
                    if (min_dem[fc] <= global_min + params.adj_demerits) {
                        auto node = std::make_shared<Active>(Active{
                            i, best_line[fc], fc, is_flag,
                            min_dem[fc], aw, as, ak,
                            serial_ctr,
                            best_prev[fc]
                        });
                        // Tracing: @@N: line L.F[−] t=T -> @@M
                        if (trace_out) {
                            int ps = best_prev[fc] ? best_prev[fc]->serial : 0;
                            *trace_out << "@@" << serial_ctr
                                       << ": line " << best_line[fc]
                                       << "." << fc;
                            if (is_flag) *trace_out << "-";
                            *trace_out << " t="
                                       << static_cast<long long>(min_dem[fc])
                                       << " -> @@" << ps << "\n";
                        }
                        ++serial_ctr;
                        new_nodes.push_back(std::move(node));
                    }
                }
            }

            survivors.insert(survivors.end(),
                std::make_move_iterator(new_nodes.begin()),
                std::make_move_iterator(new_nodes.end()));
            active = std::move(survivors);
        }

        if (auto* b = std::get_if<Box>(&item)) {
            cum_w += b->width;
        } else if (auto* g = std::get_if<Glue>(&item)) {
            cum_w += g->width;
            cum_s += g->stretch;
            cum_k += g->shrink;
        }
    }

    const int final_pos = n - 1;
    std::vector<ActivePtr> finals;
    for (const ActivePtr& a : active)
        if (a->pos == final_pos) finals.push_back(a);
    return finals;
}

// ---------------------------------------------------------------------------
// Traceback
// ---------------------------------------------------------------------------

inline std::vector<int> traceback(const ActivePtr& node) {
    std::vector<int> breaks;
    for (const Active* cur = node.get(); cur; cur = cur->prev.get())
        if (cur->pos >= 0) breaks.push_back(cur->pos);
    std::reverse(breaks.begin(), breaks.end());
    return breaks;
}

// ---------------------------------------------------------------------------
// Public API: break_paragraph
//
// trace_out: if non-null, emit TeX \tracingparagraphs-style diagnostics.
// ---------------------------------------------------------------------------

inline std::vector<int> break_paragraph(
    const std::vector<Item>& items,
    const LineSpec&           spec,
    const Params&             params    = {},
    std::ostream*             trace_out = nullptr)
{
    // Passes mirror TeX §828 and §863: the first pass is skipped if
    // pre_tolerance < 0, and the second is final if emergency_stretch <= 0.
    struct Pass { const char* name; double tolerance, extra; bool final_pass; };
    std::vector<Pass> passes;
    if (params.pre_tolerance >= 0.0)
        passes.push_back({"@firstpass", params.pre_tolerance, 0.0, false});
    passes.push_back({"@secondpass", params.tolerance, 0.0,
                      params.emergency_stretch <= 0.0});
    if (params.emergency_stretch > 0.0)
        passes.push_back({"@emergencypass", params.tolerance,
                          params.emergency_stretch, true});

    for (const Pass& pass : passes) {
        if (trace_out) *trace_out << pass.name << "\n";
        auto finals = linebreak_pass(items, spec, params, pass.tolerance,
                                     pass.extra, pass.final_pass, trace_out);
        if (!finals.empty()) {
            auto best = *std::min_element(finals.begin(), finals.end(),
                [](const ActivePtr& a, const ActivePtr& b) {
                    return a->demerits < b->demerits;
                });
            if (trace_out)
                *trace_out << "\ntotal demerits = "
                           << static_cast<long long>(best->demerits) << "\n\n";
            return traceback(best);
        }
    }

    // The final pass always sets the paragraph, unless items do not end
    // with a forced break.
    throw std::runtime_error(
        "Cannot break paragraph: items must end with a forced break.");
}
