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

__all__ = ["barstats", "betweenstats", "histostats", "scatterstats", "withinstats"]


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

    ``alternative="greater"`` tests whether the first level of ``x`` is greater
    than the second, and ``"less"`` whether it is less (two groups only). The first
    level is the first in ``order`` when given, else the first declared category,
    else the first in sorted order: pass ``order`` to fix the direction. The APA
    sentence and the alt text name it ("Welch's t-test (one-sided: control greater
    than treatment)"); the subtitle keeps ggstatsplot's text.

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
    point_path: bool | None = None,
    centrality_path: bool | None = None,
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

    What each ``type`` computes (ggstatsplot 1.1.4's ``ggwithinstats()``; two
    conditions / three or more): "parametric", the paired t-test with Hedges' g
    and the paired Bayes factor caption / the repeated-measures ANOVA
    (Greenhouse-Geisser) with partial omega squared and paired t pairs, means;
    "nonparametric", the Wilcoxon signed-rank test with the rank-biserial r /
    Friedman's test with Kendall's W and Durbin-Conover pairs, medians; "robust",
    Yuen's dependent trimmed-means test with the AKP effect size / the
    trimmed-means repeated-measures ANOVA with the average AKP and rmmcp pairs,
    trimmed means; "bayes", the paired Bayes factor / with
    ``results_subtitle=False``, a Bayes factor for every pair, MAP estimates. The
    figure joins each subject's points with two conditions (``point_path``) and
    the centrality points (``centrality_path``), as ggwithinstats does.

    ``alternative="greater"`` tests whether the first level of ``x`` is greater
    than the second (the differences first minus second are positive), and
    ``"less"`` whether it is less (two conditions only). The first level is the
    first in ``order`` when given, else the first declared category, else the
    first in sorted order: pass ``order`` to fix the direction. The APA sentence
    and the alt text name it; the subtitle keeps ggstatsplot's text.

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


@checked
def histostats(
    data: object,
    x: str,
    *,
    type: str = "parametric",
    test_value: float = 0.0,
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    alternative: str = "two-sided",
    seed: int = 0,
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
    """Test whether one numeric column differs from ``test_value``, and draw its
    histogram with the result.

    The Python counterpart of ggstatsplot's ``gghistostats()``. The analysis
    arguments are those of :func:`inferplot.analyze.histostats`; the drawing
    arguments those of :func:`inferplot.render.histostats`. The drawing arguments
    are checked first, then the analysis runs, then the figure is drawn.

    Returns:
        A ``StatsFigure``, as :func:`betweenstats` returns.

    Raises:
        NotImplementedError: the statistics of the given type are not in this
            version yet.
    """
    from inferplot import render  # imports matplotlib
    from inferplot.render.base import check_given_axes, check_one_call_subtitle
    from inferplot.render.histostats import check_drawing

    check_one_call_subtitle(subtitle, results_subtitle)
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
    check_given_axes(ax)
    result = analyze.histostats(
        data,
        x,
        type=type,
        test_value=test_value,
        conf_level=conf_level,
        digits=digits,
        tr=tr,
        bf_prior=bf_prior,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        alternative=alternative,
        seed=seed,
    )
    return render.histostats(
        result,
        binwidth=binwidth,
        centrality_plotting=centrality_plotting,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        palette=palette,
        bar_args=bar_args,
        ax=ax,
    )


@checked
def scatterstats(
    data: object,
    x: str,
    y: str,
    *,
    type: str = "parametric",
    conf_level: float = 0.95,
    digits: int = 2,
    tr: float = 0.2,
    bf_prior: float = 0.707,
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    label_var: str | None = None,
    seed: int = 0,
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
    """Test the correlation of two numeric columns, and draw them as a scatter
    plot with the result.

    The Python counterpart of ggstatsplot's ``ggscatterstats()``. The analysis
    arguments are those of :func:`inferplot.analyze.scatterstats`; the drawing
    arguments those of :func:`inferplot.render.scatterstats`. The drawing
    arguments are checked first, then the analysis runs, then the figure is
    drawn. Correlations are always tested two-sided.

    Returns:
        A ``StatsFigure``, as :func:`betweenstats` returns.

    Raises:
        NotImplementedError: the statistics of the given type are not in this
            version yet.
    """
    from inferplot import render  # imports matplotlib
    from inferplot.render.base import check_given_axes, check_one_call_subtitle
    from inferplot.render.scatterstats import check_drawing

    check_one_call_subtitle(subtitle, results_subtitle)
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
    check_given_axes(ax)
    result = analyze.scatterstats(
        data,
        x,
        y,
        type=type,
        conf_level=conf_level,
        digits=digits,
        tr=tr,
        bf_prior=bf_prior,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        label_var=label_var,
        seed=seed,
    )
    return render.scatterstats(
        result,
        marginal=marginal,
        smooth=smooth,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        palette=palette,
        point_args=point_args,
        smooth_args=smooth_args,
        marginal_args=marginal_args,
        ax=ax,
    )


@checked
def barstats(
    data: object,
    x: str,
    y: str | None = None,
    *,
    counts: str | None = None,
    paired: bool = False,
    ratio: Sequence[float] | None = None,
    label: str = "percentage",
    proportion_test: bool | None = None,
    p_adjust_method: str = "holm",
    digits_perc: int = 0,
    type: str = "parametric",
    conf_level: float = 0.95,
    digits: int = 2,
    alternative: str = "two-sided",
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    seed: int = 0,
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
    """Count the categories of ``x`` within each level of ``y``, test them, and
    draw the stacked bars with the result.

    The Python counterpart of ggstatsplot's ``ggbarstats()``. The analysis
    arguments are those of :func:`inferplot.analyze.barstats`; the drawing
    arguments those of :func:`inferplot.render.barstats`. The drawing arguments
    are checked first, then the analysis runs, then the figure is drawn.

    Returns:
        A ``StatsFigure``, as :func:`betweenstats` returns.

    Raises:
        NotImplementedError: the statistics and the drawing are not in this
            version yet.
    """
    from inferplot import render  # imports matplotlib
    from inferplot.render.barstats import check_drawing

    check_drawing(
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        legend_title=legend_title,
        bar_args=bar_args,
    )
    result = analyze.barstats(
        data,
        x,
        y,
        counts=counts,
        paired=paired,
        ratio=ratio,
        label=label,
        proportion_test=proportion_test,
        p_adjust_method=p_adjust_method,
        digits_perc=digits_perc,
        type=type,
        conf_level=conf_level,
        digits=digits,
        alternative=alternative,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        seed=seed,
    )
    return render.barstats(
        result,
        xlab=xlab,
        ylab=ylab,
        title=title,
        subtitle=subtitle,
        caption=caption,
        legend_title=legend_title,
        palette=palette,
        bar_args=bar_args,
        ax=ax,
    )
