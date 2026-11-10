# matplotlib's stubs type Line2D's data getters as partly Unknown.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Figure-level checks of the view inferplot drew into (SPEC section 8).

Measured on the drawn figure, after anything the user changed:

* ``axis_changed`` (warning): an inverted axis or a non-linear scale; the
  brackets, labels and violins were placed for an upright, linear axis;
* ``data_outside_view`` (warning): values, levels or brackets outside the axis
  limits, whoever changed them (``set_ylim``, or a later plot into an Axes that
  shares the y axis);
* ``low_contrast`` (warning): a level's colour or a text that can hardly be told
  from what it is drawn on (WCAG 2.1 contrast ratio; thresholds below).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from matplotlib.colors import to_hex, to_rgba

from inferplot.check import Issue
from inferplot.render import theme
from inferplot.render.figure import axes_background, figure_background, root_figure

if TYPE_CHECKING:
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox

    from inferplot.render.figure import StatsFigure

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
    view = fig.view
    restore = (
        "Restore the limits inferplot set"
        if view is None
        else (
            f"Restore the limits inferplot set with ax.set_ylim({view.ylim[0]:.6g}, "
            f"{view.ylim[1]:.6g}) and ax.set_xlim({view.xlim[0]:.6g}, "
            f"{view.xlim[1]:.6g})"
        )
    )
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
            f"{restore}, or for panels that share a y axis draw everything into "
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


def view_issues(fig: StatsFigure, extents: list[tuple[Text, str, Bbox]]) -> list[Issue]:
    """The view checks, in order; call after drawing."""
    issues: list[Issue] = []
    for check in (_axis_changed, _out_of_view):
        issue = check(fig)
        if issue is not None:
            issues.append(issue)
    low = _low_contrast(fig, extents)
    if low is not None:
        issues.append(low)
    return issues
