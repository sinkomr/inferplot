"""Draw a scatterstats result (entry point only; the drawing is not implemented
yet).

The signature is the drawing function's: ``marginal`` (histograms of x and y in
panels beside the plot), ``smooth`` (the least-squares line with its
``conf_level`` band), the user texts, ``palette``, ``point_args``, ``smooth_args``
and ``marginal_args`` (matplotlib properties of the points, the line and the
marginal histograms) and ``ax`` (not with marginal panels, SPEC.md section 2).
Every argument is checked, then the call raises ``NotImplementedError``
(CLAUDE.md rule 3: no partial figure).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from inferplot import _validate as v
from inferplot.render.histostats import checked_result
from inferplot.result import StatsResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from inferplot.render.figure import StatsFigure

__all__ = ["check_drawing", "scatterstats"]


def check_drawing(
    *,
    marginal: bool,
    smooth: bool,
    xlab: str | None,
    ylab: str | None,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    point_args: Mapping[str, object] | None,
    smooth_args: Mapping[str, object] | None,
    marginal_args: Mapping[str, object] | None,
    ax: object,
) -> None:
    """The checks of the drawing arguments that need no figure: ``ip.scatterstats``
    runs them before the analysis, so a bad one is reported first."""
    with_marginal = v.check_bool(marginal, "marginal")
    v.check_bool(smooth, "smooth")
    for text, name in (
        (xlab, "xlab"),
        (ylab, "ylab"),
        (title, "title"),
        (subtitle, "subtitle"),
        (caption, "caption"),
    ):
        v.check_optional_text(text, name)
    for args, name in (
        (point_args, "point_args"),
        (smooth_args, "smooth_args"),
        (marginal_args, "marginal_args"),
    ):
        v.check_artist_args(args, name)
    if ax is not None and with_marginal:
        raise ValueError(
            "ax= draws into an existing Axes, which has no room for the marginal "
            "histograms (marginal=True draws them in panels of their own). Pass "
            "marginal=False to draw into ax, or leave ax out."
        )


@v.checked
def scatterstats(
    result: StatsResult,
    *,
    marginal: bool = True,
    smooth: bool = True,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    palette: str | Sequence[object] | None = None,
    point_args: Mapping[str, object] | None = None,
    smooth_args: Mapping[str, object] | None = None,
    marginal_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.scatterstats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        marginal: draw histograms of x and y beside the plot.
        smooth: draw the least-squares line with its ``conf_level`` band.
        xlab, ylab, title, subtitle, caption: texts, as for
            ``ip.render.betweenstats``.
        palette: a matplotlib colormap name or a list of colors.
        point_args, smooth_args, marginal_args: matplotlib properties applied to
            the points, the line and the marginal histograms.
        ax: draw into this Axes instead of a new figure; needs
            ``marginal=False``.

    Raises:
        NotImplementedError: drawing a scatterstats result is not implemented in
            this version of inferplot.
    """
    checked_result(result, "scatterstats")
    check_drawing(
        marginal=marginal,
        smooth=smooth,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        point_args=point_args,
        smooth_args=smooth_args,
        marginal_args=marginal_args,
        ax=ax,
    )
    del palette
    raise NotImplementedError(
        "render.scatterstats(): drawing a scatterstats result is not implemented yet "
        "in this version of inferplot. The result's numbers and texts are in "
        "result.to_json() and result.text."
    )
