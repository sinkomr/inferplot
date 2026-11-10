"""Draw a histostats result (entry point only; the drawing is not implemented yet).

The signature is the drawing function's: ``binwidth`` (the width of the bars;
ggstatsplot's rule (max - min) / sqrt(n) by default), ``centrality_plotting`` (the
centrality line and its label), the user texts, ``palette``, ``bar_args``
(matplotlib properties of the bars) and ``ax``. Every argument is checked, then the
call raises ``NotImplementedError`` (CLAUDE.md rule 3: no partial figure).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from inferplot import _validate as v
from inferplot.result import StatsResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from inferplot.render.figure import StatsFigure

__all__ = ["check_drawing", "histostats"]


def check_drawing(
    *,
    binwidth: float | None,
    centrality_plotting: bool,
    xlab: str | None,
    ylab: str | None,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    bar_args: Mapping[str, object] | None,
) -> None:
    """The checks of the drawing arguments that need no figure: ``ip.histostats``
    runs them before the analysis, so a bad one is reported first."""
    if binwidth is not None:
        v.check_positive(binwidth, "binwidth", example="2")
    v.check_bool(centrality_plotting, "centrality_plotting")
    for text, name in (
        (xlab, "xlab"),
        (ylab, "ylab"),
        (title, "title"),
        (subtitle, "subtitle"),
        (caption, "caption"),
    ):
        v.check_optional_text(text, name)
    v.check_artist_args(bar_args, "bar_args")


def checked_result(result: StatsResult, analysis: str) -> None:
    """A result of ``analysis``, or ``TypeError`` naming the function that draws
    it."""
    if not isinstance(result, StatsResult):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"render.{analysis}() draws the result of ip.analyze.{analysis}(); "
            f"got {type(result).__name__}."
        )
    if result.analysis != analysis:
        raise TypeError(
            f"render.{analysis}() draws results of ip.analyze.{analysis}(); this "
            f"result comes from ip.analyze.{result.analysis}(). Draw it with "
            f"ip.render.{result.analysis}(result)."
        )


@v.checked
def histostats(
    result: StatsResult,
    *,
    binwidth: float | None = None,
    centrality_plotting: bool = True,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    palette: str | Sequence[object] | None = None,
    bar_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.histostats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        binwidth: the width of the bars; by default (max - min) / sqrt(n), as in
            ggstatsplot.
        centrality_plotting: draw the centrality line and its label.
        xlab, ylab, title, subtitle, caption: texts, as for
            ``ip.render.betweenstats``.
        palette: a matplotlib colormap name or a list of colors.
        bar_args: matplotlib properties applied to the bars.
        ax: draw into this Axes instead of a new figure.

    Raises:
        NotImplementedError: drawing a histostats result is not implemented in
            this version of inferplot.
    """
    checked_result(result, "histostats")
    check_drawing(
        binwidth=binwidth,
        centrality_plotting=centrality_plotting,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        bar_args=bar_args,
    )
    del palette, ax
    raise NotImplementedError(
        "render.histostats(): drawing a histostats result is not implemented yet in "
        "this version of inferplot. The result's numbers and texts are in "
        "result.to_json() and result.text."
    )
