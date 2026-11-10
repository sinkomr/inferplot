"""The top-level functions: analyze, then draw (SPEC.md section 2).

Each function here validates its keyword names (``checked``), runs the matching
``inferplot.analyze`` function and draws the result with the matching
``inferplot.render`` function. matplotlib is imported only when drawing, so
``import inferplot`` stays light.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from inferplot import analyze
from inferplot._validate import checked

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from inferplot.render import StatsFigure

__all__ = ["betweenstats", "withinstats"]


@checked
def betweenstats(
    data: object,
    x: str,
    y: str,
    *,
    type: str = "parametric",
    pairwise_display: str = "significant",
    p_adjust_method: str = "holm",
    pairwise_alpha: float = 0.05,
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    alternative: str = "two-sided",
    seed: int = 0,
    order: Iterable[str | int | float] | None = None,
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
    """Compare ``y`` across the groups of ``x``, test it, and draw the result.

    The Python counterpart of ggstatsplot's ``ggbetweenstats()``: ``data`` is a
    pandas, polars or pyarrow data frame in long format, ``x`` the column of group
    labels and ``y`` the numeric column. The analysis arguments are those of
    :func:`inferplot.analyze.betweenstats`; the drawing arguments those of
    :func:`inferplot.render.betweenstats`.

    ``title``, ``subtitle``, ``caption``, ``xlab`` and ``ylab`` are matplotlib
    text, which follows matplotlib's convention: text between two dollar signs is
    a mathtext expression (``r"Weight ($\\mu$g)"``). Write ``\\$`` for a literal
    dollar sign (``r"Revenue in \\$"``). An expression matplotlib cannot read
    raises ``ValueError`` at the call. Column names drawn as default labels are
    never read as mathtext.

    Returns:
        A ``StatsFigure``: ``fig.result`` holds every number (``fig.result.to_json()``,
        ``fig.result.text.apa``), ``fig.check()`` lists statistical and rendering
        problems, and ``fig.save("plot.svg")`` writes an editable vector file.
    """
    from inferplot import render  # imports matplotlib

    result = analyze.betweenstats(
        data,
        x,
        y,
        type=type,
        pairwise_display=pairwise_display,
        p_adjust_method=p_adjust_method,
        pairwise_alpha=pairwise_alpha,
        conf_level=conf_level,
        digits=digits,
        tr=tr,
        bf_prior=bf_prior,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        alternative=alternative,
        seed=seed,
        order=order,
    )
    return render.betweenstats(
        result,
        centrality_plotting=centrality_plotting,
        layers=layers,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        palette=palette,
        point_args=point_args,
        boxplot_args=boxplot_args,
        violin_args=violin_args,
        ax=ax,
    )


@checked
def withinstats(
    data: object,
    x: str,
    y: str,
    *,
    subject_id: str,
    type: str = "parametric",
    pairwise_display: str = "significant",
    p_adjust_method: str = "holm",
    pairwise_alpha: float = 0.05,
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    alternative: str = "two-sided",
    seed: int = 0,
    order: Iterable[str | int | float] | None = None,
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
    """Compare ``y`` across the levels of ``x`` measured on the same subjects, test
    it, and draw the result.

    The Python counterpart of ggstatsplot's ``ggwithinstats()``: ``data`` is a
    pandas, polars or pyarrow data frame in long format (one row per subject and
    level of ``x``), and ``subject_id`` names the column that identifies each
    subject; it is required, because measurements are paired by subject, never by
    row order. The analysis arguments are those of
    :func:`inferplot.analyze.withinstats`; the drawing arguments those of
    :func:`inferplot.render.withinstats`.

    Returns:
        A ``StatsFigure``, as :func:`betweenstats` returns.
    """
    from inferplot import render  # imports matplotlib

    result = analyze.withinstats(
        data,
        x,
        y,
        subject_id=subject_id,
        type=type,
        pairwise_display=pairwise_display,
        p_adjust_method=p_adjust_method,
        pairwise_alpha=pairwise_alpha,
        conf_level=conf_level,
        digits=digits,
        tr=tr,
        bf_prior=bf_prior,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        alternative=alternative,
        seed=seed,
        order=order,
    )
    return render.withinstats(
        result,
        point_path=point_path,
        centrality_path=centrality_path,
        centrality_plotting=centrality_plotting,
        layers=layers,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        palette=palette,
        point_args=point_args,
        boxplot_args=boxplot_args,
        violin_args=violin_args,
        ax=ax,
    )
