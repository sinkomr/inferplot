"""The figure check ``level_labels_alike`` (SPEC section 8), for every figure.

A figure writes each level's name where it draws the level: under its place on
the x axis (betweenstats, withinstats; barstats' bars, the levels of y) and in
the legend (barstats' levels of x, whose colours the legend is the only key
to). Long names are wrapped and shortened, keeping names that would shorten
alike apart (D77); this check reads the labels as drawn, after anything the
user changed, and reports two different levels whose labels read alike (line
breaks and runs of spaces read as one space; the n line under a tick label is
not part of the name). Figures that draw no level names report nothing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from inferplot.check import Issue
from inferplot.render.comparison import reads

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.text import Text

    from inferplot.render.figure import StatsFigure

__all__ = ["level_label_issue"]

_N_LINE: Final = re.compile(r"\n\(n = [^\n]*\)\s*$")
_MAX_LISTED: Final = 5


def _drawn(text: Text) -> bool:
    return text.get_visible() and bool(text.get_text().strip())


def _tick_names(ax: Axes, names: Sequence[str]) -> list[tuple[str, str]]:
    """(level, how its tick label reads) for each level whose tick label is
    drawn: the tick at the level's place, 0, 1, ..., within the view."""
    if not ax.xaxis.get_visible():
        return []
    lo, hi = sorted(float(v) for v in ax.xaxis.get_view_interval())
    out: list[tuple[str, str]] = []
    for tick in ax.xaxis.get_major_ticks():
        loc = float(tick.get_loc())
        index = round(loc)
        label = tick.label1
        if (
            abs(loc - index) > 1e-9
            or not 0 <= index < len(names)
            or not lo - 1e-9 <= loc <= hi + 1e-9
            or not tick.get_visible()
            or not _drawn(label)
        ):
            continue
        out.append((names[index], reads(_N_LINE.sub("", label.get_text()))))
    return out


def _legend_names(ax: Axes, names: Sequence[str]) -> list[tuple[str, str]]:
    """(level, how its legend entry reads) for each drawn legend entry, in the
    order of the levels."""
    legend = ax.get_legend()
    if legend is None or not legend.get_visible():
        return []
    texts = legend.get_texts()
    if len(texts) != len(names):
        return []
    return [
        (name, reads(text.get_text()))
        for name, text in zip(names, texts, strict=True)
        if _drawn(text)
    ]


def _alike(drawn: Sequence[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    """Groups of different levels whose labels read alike: (the text, levels)."""
    groups: dict[str, list[str]] = {}
    for name, text in drawn:
        levels = groups.setdefault(text, [])
        if name not in levels:
            levels.append(name)
    return [(text, levels) for text, levels in groups.items() if len(levels) > 1]


def _and(names: Sequence[str]) -> str:
    shown = [repr(n) for n in names]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def level_label_issue(fig: StatsFigure) -> Issue | None:
    """``level_labels_alike``: two different levels drawn with labels that read
    alike, in the tick labels or the legend; None where every level reads as
    its own, and for figures that draw no level names."""
    result = fig.result
    ax = fig.axes.get("main")
    if ax is None:
        return None
    found: list[str] = []
    if result.design in ("between", "within"):
        where = [("tick labels", result.x, _tick_names(ax, result.levels))]
    elif result.design == "contingency":
        bars = [b.level for b in result.bars]
        where = [("legend entries", result.x, _legend_names(ax, result.levels))]
        if result.y is not None and all(b is not None for b in bars):
            named = [b for b in bars if b is not None]
            where.insert(0, ("tick labels", result.y, _tick_names(ax, named)))
    else:
        return None
    for place, column, drawn in where:
        for text, levels in _alike(drawn):
            found.append(
                f"the {place} of the levels {_and(levels)} of {column!r} all read "
                f"{text!r}"
            )
    if not found:
        return None
    listed = "; ".join(found[:_MAX_LISTED])
    if len(found) > _MAX_LISTED:
        listed += f"; and {len(found) - _MAX_LISTED} more"
    full = (
        "fig.result.levels (the levels of x) and fig.result.y_levels (of y)"
        if result.design == "contingency"
        else "fig.result.levels"
    )
    return Issue(
        code="level_labels_alike",
        severity="warning",
        message=(
            f"Different levels are drawn with labels that read alike, so a reader "
            f"cannot tell them apart: {listed}."
        ),
        fix=(
            "Give the levels names that differ in what is drawn: rename them in the "
            "data before the analysis (names that differ only in spaces or line "
            "breaks read alike), and leave the drawn tick labels and legend entries "
            f"as inferplot writes them. The full names are in {full}."
        ),
    )
