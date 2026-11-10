"""Figure-level checks of the view inferplot drew into (SPEC section 8).

Measured on the drawn figure, after anything the user changed:

* ``axis_changed`` (warning): an inverted axis or a non-linear scale; the
  brackets, labels and violins were placed for an upright, linear axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def view_issues(fig: StatsFigure) -> list[Issue]:
    """The view checks, in order; call after drawing."""
    issues: list[Issue] = []
    changed = _axis_changed(fig)
    if changed is not None:
        issues.append(changed)
    return issues
