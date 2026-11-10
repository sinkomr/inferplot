"""Draw a withinstats result (entry point only; the drawing is not implemented yet).

The signature is final: betweenstats' drawing keywords plus ``point_path`` (join each
subject's points across the levels) and ``centrality_path`` (join the centrality
points). Every argument is checked as ``render.betweenstats`` checks it, then the
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

__all__ = ["withinstats"]


@v.checked
def withinstats(
    result: StatsResult,
    *,
    point_path: bool = True,
    centrality_path: bool = True,
    centrality_plotting: bool = True,
    layers: Sequence[str] = ("violin", "box", "points"),
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    palette: str | Sequence[object] | None = None,
    point_args: Mapping[str, object] | None = None,
    boxplot_args: Mapping[str, object] | None = None,
    violin_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.withinstats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        point_path: join each subject's points across the levels of x.
        centrality_path: join the levels' centrality points.
        centrality_plotting: draw each level's centrality point and label.
        layers, xlab, ylab, title, subtitle, caption, palette, point_args,
            boxplot_args, violin_args, ax: as for ``ip.render.betweenstats``.

    Raises:
        NotImplementedError: drawing a within-subjects result is not implemented
            in this version of inferplot.
    """
    if not isinstance(result, StatsResult):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            "render.withinstats() draws the result of ip.analyze.withinstats(); "
            f"got {type(result).__name__}."
        )
    if result.analysis != "withinstats":
        raise TypeError(
            "render.withinstats() draws results of ip.analyze.withinstats(); this "
            f"result comes from ip.analyze.{result.analysis}(). Draw it with "
            f"ip.render.{result.analysis}(result)."
        )
    v.check_bool(point_path, "point_path")
    v.check_bool(centrality_path, "centrality_path")
    v.check_bool(centrality_plotting, "centrality_plotting")
    v.check_layers(layers)
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
        (boxplot_args, "boxplot_args"),
        (violin_args, "violin_args"),
    ):
        v.check_artist_args(args, name)
    del palette, ax
    raise NotImplementedError(
        "render.withinstats(): drawing a within-subjects result is not implemented "
        "yet in this version of inferplot. The result's numbers and texts are in "
        "result.to_json() and result.text."
    )
