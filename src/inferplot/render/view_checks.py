# matplotlib's stubs type Line2D's data getters as partly Unknown.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Figure-level checks of the view inferplot drew into (SPEC section 8).

Measured on the drawn figure, after anything the user changed:

* ``axis_changed`` (warning): an inverted axis or a non-linear scale; the
  brackets, labels and violins were placed for an upright, linear axis;
* ``data_outside_view`` (warning): values, levels or brackets outside the axis
  limits, whoever changed them (``set_ylim``, or a later plot into an Axes that
  shares the y axis);
* ``flat_data`` (warning): values that differ drawn within two pixels of height
  (axis limits far wider than the data);
* ``centrality_label_overlap`` (warning): a centrality point at least half
  covered by a text's box (a label placed over it, or moved there later);
* ``glyph_missing`` (warning): characters no font of a text can draw (shown as
  empty boxes);
* ``low_contrast`` (warning): a level's colour or a text that can hardly be told
  from what it is drawn on (WCAG 2.1 contrast ratio; thresholds below).
"""

from __future__ import annotations

import functools
import re
import unicodedata
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgba

from inferplot.check import Issue
from inferplot.render import theme
from inferplot.render.figure import axes_background, figure_background, root_figure

if TYPE_CHECKING:
    from matplotlib.font_manager import FontProperties
    from matplotlib.ft2font import FT2Font
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    from inferplot.render.figure import StatsFigure

_DOLLAR: Final = re.compile(r"(?<!\\)\$")

# Contrast ratios (WCAG 2.1) below which something counts as hard to see. Text:
# 3:1, WCAG's minimum for large text (SC 1.4.3) and for graphical objects (SC
# 1.4.11); inferplot's own ink has about 16:1. A level's colour: 1.1:1, nearly the
# background itself (white on white is 1:1, the light end of "Blues" 1.04:1, pure
# yellow on white 1.07:1). WCAG's 3:1 would report the colour-blind-safe default
# Okabe-Ito palette (its yellow has 1.32:1 on white, its orange 2.25:1) and most
# of matplotlib's qualitative colormaps, so marks get the lower floor.
TEXT_CONTRAST: Final = 3.0
COLOR_CONTRAST: Final = 1.1

__all__ = ["view_issues"]


def _axis_changed(fig: StatsFigure) -> Issue | None:
    ax = fig.axes["main"]
    found: list[str] = []
    undo: list[str] = []
    if ax.yaxis_inverted():
        found.append(
            "the y axis is inverted, so each bracket's label is drawn through its "
            "own bracket and the centrality labels sit below their points' height"
        )
        undo.append("ax.invert_yaxis()")
    if ax.xaxis_inverted():
        found.append(
            "the x axis is inverted, so the centrality labels land on the points "
            "they were placed beside"
        )
        undo.append("ax.invert_xaxis()")
    for name, scale, setter in (
        ("y", ax.get_yscale(), "set_yscale"),
        ("x", ax.get_xscale(), "set_xscale"),
    ):
        if scale != "linear":
            found.append(
                f"the {name} axis has a {scale!r} scale, while the violins, "
                "brackets and labels were placed for a linear one"
            )
            undo.append(f"ax.{setter}('linear')")
    if not found:
        return None
    return Issue(
        code="axis_changed",
        severity="warning",
        message=(
            "The axes were changed after inferplot drew them: " + "; ".join(found) + "."
        ),
        fix=(
            f"Undo it with {' and '.join(undo)}. To show the levels in another "
            "order, pass order= to the analysis; to show the values on another "
            "scale, transform the column before the analysis."
        ),
    )


def _interval(limits: tuple[float, float]) -> tuple[float, float]:
    """Sorted limits widened by a relative 1e-9 (rounding of the limits)."""
    lo, hi = sorted(float(v) for v in limits)
    slack = (hi - lo) * 1e-9
    return lo - slack, hi + slack


def _restore(fig: StatsFigure) -> str:
    """How to set back the limits inferplot set, in words."""
    view = fig.view
    if view is None:
        return "restore the limits inferplot set"
    return (
        f"restore the limits inferplot set with ax.set_ylim({view.ylim[0]:.6g}, "
        f"{view.ylim[1]:.6g}) and ax.set_xlim({view.xlim[0]:.6g}, "
        f"{view.xlim[1]:.6g})"
    )


def _out_of_view(fig: StatsFigure) -> Issue | None:
    """Values, levels and brackets that lie outside the view."""
    ax = fig.axes["main"]
    x_lo, x_hi = _interval(ax.get_xlim())
    y_lo, y_hi = _interval(ax.get_ylim())
    result = fig.result
    found: list[str] = []
    total = sum(len(v) for v in result.values)
    out = 0
    levels: list[str] = []
    for i, (level, values) in enumerate(zip(result.levels, result.values, strict=True)):
        y = np.asarray(values, dtype=np.float64)
        if not x_lo <= i <= x_hi:
            out += y.size
            levels.append(level)
            continue
        out += int(((y < y_lo) | (y > y_hi)).sum())
    if out:
        found.append(f"{out:,} of {total:,} values")
    if levels:
        names = ", ".join(repr(level) for level in levels)
        found.append(f"the level{'s' if len(levels) > 1 else ''} {names} (x)")
    brackets = 0
    for _, line, _ in fig.brackets:
        if not line.get_visible() or line.axes is None:
            continue
        xs = np.asarray(line.get_xdata(), dtype=np.float64)  # pyright: ignore[reportArgumentType]
        ys = np.asarray(line.get_ydata(), dtype=np.float64)  # pyright: ignore[reportArgumentType]
        if (
            float(xs.min()) < x_lo
            or float(xs.max()) > x_hi
            or float(ys.min()) < y_lo
            or float(ys.max()) > y_hi
        ):
            brackets += 1
    if brackets:
        found.append(f"{brackets} bracket{'s' if brackets > 1 else ''}")
    if not found:
        return None
    restore = _restore(fig)
    return Issue(
        code="data_outside_view",
        severity="warning",
        message=(
            "Part of the plot lies outside the axis limits and is not shown: "
            + ", ".join(found)
            + f" (the view shows y from {y_lo:.4g} to {y_hi:.4g}). The limits were "
            "changed after inferplot drew: by ax.set_ylim or ax.set_xlim, or by a "
            "later plot into an Axes that shares this one's y axis."
        ),
        fix=(
            f"{restore[0].upper()}{restore[1:]}, or for panels that share a y axis "
            "draw everything into "
            "them before setting limits."
        ),
    )


def _hex(rgb: tuple[float, float, float]) -> str:
    return to_hex(rgb)


def _low_contrast(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> Issue | None:
    """Levels' colours and texts with too little contrast with their background."""
    ax = fig.axes["main"]
    inside = axes_background(ax)
    outside = figure_background(root_figure(fig.figure))
    found: list[str] = []
    for level, color in zip(fig.result.levels, fig.colors, strict=False):
        drawn = theme.over(color, inside)
        ratio = theme.contrast(drawn, inside)
        if ratio < COLOR_CONTRAST:
            found.append(
                f"the colour of {level!r} ({_hex(drawn)}) on the Axes background "
                f"{_hex(inside)} has {ratio:.2f}:1, below {COLOR_CONTRAST}:1"
            )
    frame = ax.get_window_extent()
    for text, described, box in extents:
        patch = text.get_bbox_patch()
        x, y = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
        within = frame.x0 <= x <= frame.x1 and frame.y0 <= y <= frame.y1
        below = inside if within else outside
        if patch is not None and patch.get_visible() and patch.get_fill():
            below = theme.over(patch.get_facecolor(), below)
        alpha = text.get_alpha()
        r, g, b, a = (float(c) for c in to_rgba(text.get_color()))
        ink = theme.over((r, g, b, a * (1.0 if alpha is None else alpha)), below)
        ratio = theme.contrast(ink, below)
        if ratio < TEXT_CONTRAST:
            found.append(
                f"{described} is drawn in {_hex(ink)} on {_hex(below)}, "
                f"{ratio:.2f}:1, below {TEXT_CONTRAST:.0f}:1"
            )
    if not found:
        return None
    shown = "; ".join(found[:5]) + (
        f"; and {len(found) - 5} more" if len(found) > 5 else ""
    )
    return Issue(
        code="low_contrast",
        severity="warning",
        message=f"Some of the plot can hardly be seen on its background: {shown}.",
        fix=(
            "Choose palette colours that differ from the background (the default "
            "palette, or a colormap name, which inferplot samples away from its "
            "light end), give texts a colour with at least 3:1 contrast, or set the "
            "background back (ax.set_facecolor, figure.set_facecolor)."
        ),
    )


# The data drawn over fewer pixels than this (top to bottom) are a flat line.
FLAT_PX: Final = 2.0


def _flat(fig: StatsFigure) -> Issue | None:
    """Values that differ but are drawn within FLAT_PX pixels of height."""
    ax = fig.axes["main"]
    values = [np.asarray(v, dtype=np.float64) for v in fig.result.values if len(v)]
    if not values:
        return None
    lo = min(float(v.min()) for v in values)
    hi = max(float(v.max()) for v in values)
    if not hi > lo or ax.get_yscale() != "linear":
        return None
    ends = np.asarray(ax.transData.transform([(0.0, lo), (0.0, hi)]), dtype=np.float64)
    height = abs(float(ends[1, 1] - ends[0, 1]))
    if not np.isfinite(height) or height >= FLAT_PX:
        return None
    y_lo, y_hi = sorted(float(v) for v in ax.get_ylim())
    return Issue(
        code="flat_data",
        severity="warning",
        message=(
            f"The values, from {lo:.6g} to {hi:.6g}, are drawn within "
            f"{height:.2g} pixels of height, so the plot shows a flat line: the "
            f"y axis spans {y_lo:.4g} to {y_hi:.4g}, far more than the values."
        ),
        fix=(
            f"If the axis limits were changed after drawing: {_restore(fig)}. If "
            "the values differ only "
            "far beyond their leading digits (1e15 plus a little), subtract a "
            "common offset from y before the analysis; the statistics do not "
            "change."
        ),
    )


def _fonts(prop: FontProperties) -> list[FT2Font]:
    """The font files matplotlib draws a text of ``prop`` with: its families in
    order, as its font fallback uses them."""
    manager: Any = font_manager.fontManager
    paths = manager._find_fonts_by_props(prop)  # matplotlib >= 3.6, used by Agg
    return [font_manager.get_font(path) for path in paths]


@functools.cache
def _math_fonts() -> tuple[FT2Font, ...]:
    """The fonts of inferplot's mathtext: DejaVu Sans, then STIX (the fallback
    of matplotlib's "dejavusans" fontset)."""
    return tuple(
        font_manager.get_font(font_manager.findfont(name))
        for name in ("DejaVu Sans", "STIXGeneral")
    )


def _is_math(text: Text) -> bool:
    shown = text.get_text()
    return bool(text.get_parse_math()) and len(_DOLLAR.findall(shown)) >= 2


def _missing_glyphs(text: Text) -> list[str]:
    """Characters of ``text`` that none of the fonts it is drawn with has.

    Plain text uses its families with matplotlib's font fallback; a mathtext
    string (all of it, also outside the dollar signs) uses the mathtext fonts,
    whose ASCII is complete, so only its other characters are looked up.
    """
    math = _is_math(text)
    fonts = list(_math_fonts()) if math else _fonts(text.get_fontproperties())
    found: list[str] = []
    for char in dict.fromkeys(text.get_text()):
        if unicodedata.category(char)[0] in "CZ" or (math and char.isascii()):
            continue
        if not any(font.get_char_index(ord(char)) for font in fonts):
            found.append(char)
    return found


def _glyph_missing(extents: list[tuple[Text, str, Bbox]]) -> Issue | None:
    found: list[str] = []
    chars: dict[str, None] = {}
    for text, described, _ in extents:
        missing = _missing_glyphs(text)
        if missing:
            chars.update(dict.fromkeys(missing))
            shown = ", ".join(repr(c) for c in missing[:5])
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            found.append(f"{shown}{more} in the {described}")
    if not found:
        return None
    listed = "; ".join(found[:5]) + (
        f"; and {len(found) - 5} more texts" if len(found) > 5 else ""
    )
    return Issue(
        code="glyph_missing",
        severity="warning",
        message=(
            f"{len(chars)} character{'s' if len(chars) > 1 else ''} cannot be drawn "
            "by any font the texts use (DejaVu Sans, then the fonts listed under "
            "'font.sans-serif' in matplotlib's rcParams beyond matplotlib's "
            "default list; inside $...$ DejaVu Sans and STIX), so they show as "
            f"empty boxes: {listed}."
        ),
        fix=(
            "Install a font that has them and list it under 'font.sans-serif' in "
            "matplotlib's rcParams before drawing, for example "
            "matplotlib.rcParams.update({'font.sans-serif': ['Noto Sans CJK JP']}); "
            "inferplot draws Latin text with DejaVu Sans and the rest with the "
            "first of those fonts that has it. Characters inside $...$ need "
            "DejaVu Sans or STIX."
        ),
    )


# A centrality point counts as covered when a text covers this share of it.
COVERED_SHARE: Final = 0.5


def _covered_centers(
    fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]
) -> Issue | None:
    """Centrality points at least half covered by a text's box."""
    dpi = float(root_figure(fig.figure).dpi)
    found: list[str] = []
    for level, point in fig.centers:
        if not point.get_visible() or point.axes is None:
            continue
        offsets = np.asarray(point.get_offsets(), dtype=np.float64)
        if offsets.size == 0:
            continue
        x, y = (float(v) for v in point.get_offset_transform().transform(offsets)[0])
        sizes = np.asarray(point.get_sizes(), dtype=np.float64)
        half = float(np.sqrt(sizes[0] if sizes.size else 36.0)) / 2 * dpi / 72.0
        area = (2 * half) ** 2
        covers: list[tuple[float, str]] = []
        for _, described, box in extents:
            width = min(box.x1, x + half) - max(box.x0, x - half)
            height = min(box.y1, y + half) - max(box.y0, y - half)
            if width > 0 and height > 0:
                covers.append((width * height / area, described))
        share = min(1.0, sum(c for c, _ in covers))
        if share >= COVERED_SHARE:
            by = ", the ".join(d for _, d in covers)
            found.append(
                f"the centrality point of {level!r} lies {share:.0%} under the {by}"
            )
    if not found:
        return None
    listed = "; ".join(found[:5]) + (
        f"; and {len(found) - 5} more" if len(found) > 5 else ""
    )
    return Issue(
        code="centrality_label_overlap",
        severity="warning",
        message=f"A text hides a centrality point: {listed}.",
        fix=(
            "Give the plot more room (a larger Axes passed as ax=, or inferplot's "
            "own figure by leaving ax out) so the labels can be placed beside the "
            "points, or move the text that covers it."
        ),
    )


def view_issues(fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]) -> list[Issue]:
    """The view checks, in order; call after drawing."""
    issues: list[Issue] = []
    for check in (_axis_changed, _out_of_view, _flat):
        issue = check(fig)
        if issue is not None:
            issues.append(issue)
    for issue in (
        _covered_centers(fig, extents),
        _low_contrast(fig, extents),
        _glyph_missing(extents),
    ):
        if issue is not None:
            issues.append(issue)
    return issues
