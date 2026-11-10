# matplotlib's stubs type Line2D's data getters as partly Unknown.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Figure-level checks of the view inferplot drew into (SPEC section 8).

Measured on the drawn figure, after anything the user changed:

* ``axis_changed`` (warning): an inverted axis or a non-linear scale; the
  brackets, labels and violins were placed for an upright, linear axis;
* ``data_outside_view`` (warning): values, levels or brackets outside the axis
  limits, whoever changed them (``set_ylim``, or a later plot into an Axes that
  shares the y axis).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from inferplot.check import Issue

if TYPE_CHECKING:
    from inferplot.render.figure import StatsFigure

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


def view_issues(fig: StatsFigure) -> list[Issue]:
    """The view checks, in order; call after drawing."""
    issues: list[Issue] = []
    for check in (_axis_changed, _out_of_view):
        issue = check(fig)
        if issue is not None:
            issues.append(issue)
    return issues
