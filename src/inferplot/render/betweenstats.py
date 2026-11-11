"""Draw a betweenstats result: violins, boxes, points, centrality labels, brackets.

The drawing itself is :class:`inferplot.render.comparison.Drawing`, shared with
withinstats; this module checks the arguments and hands them over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from inferplot import _validate as v
from inferplot.render import comparison
from inferplot.render.comparison import MAX_POINTS, SUBSAMPLE_SIZE
from inferplot.result import StatsResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from inferplot.render.figure import StatsFigure

__all__ = ["MAX_POINTS", "SUBSAMPLE_SIZE", "betweenstats"]


@v.checked
def betweenstats(
    result: StatsResult,
    *,
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
    """Draw a result of ``ip.analyze.betweenstats``.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        centrality_plotting: draw each level's centrality point and label.
        layers: any of "violin", "box" and "points", drawn back to front.
        xlab, ylab: axis labels; default to the column names.
        title: a title above the subtitle.
        subtitle: your own subtitle; allowed only for a result computed with
            ``results_subtitle=False`` (otherwise the results subtitle is drawn).
        caption: your own caption, drawn below the generated caption lines.
        palette: a matplotlib colormap name or a list of at least one color per
            level; default Okabe-Ito (``tab20`` for more than 8 levels).
        point_args, boxplot_args, violin_args: matplotlib properties applied with
            ``Artist.set`` to each level's points, box and violin. Each needs its
            layer ("points", "box", "violin") in ``layers``; unknown properties and
            values matplotlib rejects raise before anything is drawn.
        ax: draw into this Axes instead of a new figure; the subtitle and caption
            then go inside it.

    ``title``, ``subtitle``, ``caption``, ``xlab`` and ``ylab`` are matplotlib
    text, which follows matplotlib's convention: text between two dollar signs is
    a mathtext expression (``r"Weight ($\\mu$g)"``). Write ``\\$`` for a literal
    dollar sign (``r"Revenue in \\$"``). An expression matplotlib cannot read
    raises ``ValueError`` at the call. Column names drawn as default labels are
    never read as mathtext.

    Returns:
        A :class:`~inferplot.render.figure.StatsFigure`; its ``result`` adds any
        rendering warnings to ``result.warnings``.

    Example:
        >>> import pandas as pd
        >>> import inferplot as ip
        >>> df = pd.DataFrame({
        ...     "group": ["a"] * 5 + ["b"] * 5,
        ...     "score": [5.1, 4.8, 6.0, 5.5, 5.2, 6.3, 6.9, 5.9, 7.1, 6.6],
        ... })
        >>> result = ip.analyze.betweenstats(df, "group", "score")
        >>> fig = ip.render.betweenstats(result, layers=("box", "points"))
        >>> fig.result.text.apa == result.text.apa
        True
        >>> fig.check()
        ()
    """
    comparison.checked_result(
        result, "betweenstats", "ip.analyze.betweenstats(data, x, y, ...)"
    )
    options = comparison.Options(
        centrality=v.check_bool(centrality_plotting, "centrality_plotting"),
        layers=v.check_layers(layers),
        xlab=v.check_optional_text(xlab, "xlab"),
        ylab=v.check_optional_text(ylab, "ylab"),
        title=v.check_optional_text(title, "title"),
        subtitle=v.check_optional_text(subtitle, "subtitle"),
        caption=v.check_optional_text(caption, "caption"),
        point_args=v.check_artist_args(point_args, "point_args") or {},
        boxplot_args=v.check_artist_args(boxplot_args, "boxplot_args") or {},
        violin_args=v.check_artist_args(violin_args, "violin_args") or {},
    )
    comparison.check_options(result, options, ax)
    return comparison.draw(comparison.Drawing, result, options, palette, ax)
