"""Draw a barstats result (entry point only; the drawing is not implemented yet).

The signature is the drawing function's: the user texts (``xlab``, ``ylab``,
``title``, ``subtitle``, ``caption``), ``legend_title`` (the title of the legend
of the levels of x; ggstatsplot's ``legend.title``, by default the name of x),
``palette`` (one color per level of x), ``bar_args`` (matplotlib properties of
every bar segment) and ``ax``. Every argument is checked, then the call raises
``NotImplementedError`` (CLAUDE.md rule 3: no partial figure).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from inferplot import _validate as v
from inferplot.render import base, theme
from inferplot.result import StatsResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from inferplot.render.figure import StatsFigure

__all__ = ["barstats", "check_drawing"]


@dataclass(frozen=True, slots=True)
class Options:
    """The checked drawing arguments."""

    xlab: str | None
    ylab: str | None
    title: str | None
    subtitle: str | None
    caption: str | None
    legend_title: str | None
    bar_args: dict[str, object]


def check_drawing(
    *,
    xlab: str | None,
    ylab: str | None,
    title: str | None,
    subtitle: str | None,
    caption: str | None,
    legend_title: str | None,
    bar_args: Mapping[str, object] | None,
) -> Options:
    """The checks of the drawing arguments that need no figure: ``ip.barstats``
    runs them before the analysis, so a bad one is reported first."""
    return Options(
        xlab=v.check_optional_text(xlab, "xlab"),
        ylab=v.check_optional_text(ylab, "ylab"),
        title=v.check_optional_text(title, "title"),
        subtitle=v.check_optional_text(subtitle, "subtitle"),
        caption=v.check_optional_text(caption, "caption"),
        legend_title=v.check_optional_text(legend_title, "legend_title"),
        bar_args=v.check_artist_args(bar_args, "bar_args") or {},
    )


@v.checked
def barstats(
    result: StatsResult,
    *,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    caption: str | None = None,
    legend_title: str | None = None,
    palette: str | Sequence[object] | None = None,
    bar_args: Mapping[str, object] | None = None,
    ax: Axes | None = None,
) -> StatsFigure:
    """Draw a result of ``ip.analyze.barstats``: one stacked bar per level of y
    (one bar without y), filled by the shares of the levels of x.

    Args:
        result: the :class:`~inferplot.result.StatsResult` to draw.
        xlab, ylab, title, subtitle, caption: texts, as for
            ``ip.render.betweenstats``.
        legend_title: the title of the legend of the levels of x (by default the
            name of x).
        palette: a matplotlib colormap name or a list of colors, one per level of
            x.
        bar_args: matplotlib properties applied to every bar segment.
        ax: draw into this Axes instead of a new figure.

    Raises:
        NotImplementedError: drawing a barstats result is not implemented in
            this version of inferplot.
    """
    base.check_result_type(result, "barstats")
    options = check_drawing(
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        legend_title=legend_title,
        bar_args=bar_args,
    )
    base.check_subtitle(result, options.subtitle)
    base.check_given_axes(ax)
    theme.resolve_palette(palette, len(result.levels))
    raise NotImplementedError(
        "render.barstats(): drawing a barstats result is not implemented yet in "
        "this version of inferplot. The result's numbers and texts are in "
        "result.to_json() and result.text."
    )
