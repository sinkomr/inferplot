"""``ip.render.histostats``: every gghistostats fixture drawn, and every argument.

Each fixture's call is analyzed by ``ip.analyze.histostats``
(``tests/no_level_cases.analyzed``) and drawn with the fixture's drawing arguments
(``binwidth``, ``centrality_plotting``). For each figure: the bars are
ggstatsplot's built histogram layer (edges and counts), the percent axis is
count / n, the centrality line sits at the result's centrality value with the
label ``figure_texts`` gives, the test value is marked, and ``fig.check()`` reports
only what the result itself reports.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.render.histostats import MAX_BARS, TEST_VALUE_REACH
from inferplot.result import StatsResult
from inferplot.stats import histogram_bins
from inferplot.text import figure_alt_text, figure_texts, r_number_text
from tests.no_level_cases import REFUSALS, analyzed
from tests.oracle_args import split
from tests.oracle_fixtures import OracleCase, cases_for, number, require_cases
from tests.render._helpers import by_gid, gids, numbered_count, numbered_texts, text_of
from tests.render._no_levels import hist_result

CASES = require_cases(cases_for("gghistostats"), "gghistostats fixtures")
BY_NAME = {c.name: c for c in CASES}
FIGURE_CODES = frozenset(
    {
        "label_overlap",
        "text_clipped",
        "points_subsampled",
        "low_contrast",
        "glyph_missing",
        "axis_changed",
        "data_outside_view",
        "flat_data",
        "centrality_label_overlap",
        "centrality_label_ambiguous",
        "path_hidden",
        "paths_crowded",
        "point_labels_crowded",
    }
)


def drawing(case: OracleCase) -> dict[str, Any]:
    return dict(split(case.function, case.args).drawing)


@functools.cache
def result_of(name: str) -> StatsResult:
    """``ip.analyze.histostats`` on the fixture's call, or on its replay where
    inferplot refuses the call (a one-sided alternative with type robust or
    bayes, which ggstatsplot ignores; ``tests/no_level_cases.REFUSALS``)."""
    result = analyzed(name)
    assert result is not None, name
    return result


@functools.cache
def figure(name: str) -> StatsFigure:
    return ip.render.histostats(result_of(name), **drawing(BY_NAME[name]))


def figure_codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check() if i.code in FIGURE_CODES]


def bars(fig: StatsFigure) -> list[Rectangle]:
    found = [by_gid(fig, f"bar-{i}") for i in range(numbered_count(fig, "bar"))]
    assert all(isinstance(b, Rectangle) for b in found)
    return [b for b in found if isinstance(b, Rectangle)]


def tooth(**kwargs: Any) -> StatsResult:
    """ToothGrowth's len against 18, or ``ip.analyze.histostats`` with
    ``kwargs``."""
    if not kwargs:
        return result_of("gghistostats-ToothGrowth-parametric")
    return ip.analyze.histostats(_tooth_frame(), "len", **{"test_value": 18, **kwargs})


@functools.cache
def _tooth_frame() -> pd.DataFrame:
    case = BY_NAME["gghistostats-ToothGrowth-parametric"]
    return pd.DataFrame({"len": case.analyzed_values("x")})


def message(error: type[Exception], call: Any) -> str:
    with pytest.raises(error) as caught:
        call()
    return str(caught.value)


# --------------------------------------------------------------------------- #
# every fixture
# --------------------------------------------------------------------------- #


def drawn_cases() -> list[Any]:
    """The fixtures inferplot analyzes: all but constant data with types
    parametric, bayes and robust, which it refuses (``REFUSALS``; edge_one_constant
    of the barstats oracle run). Their nonparametric cases are drawn: one bar."""
    return [
        pytest.param(c.name, id=c.name)
        for c in CASES
        if not (c.name in REFUSALS and REFUSALS[c.name].replay is None)
    ]


@pytest.mark.parametrize("name", drawn_cases())
def test_fixture_figure(name: str) -> None:
    case = BY_NAME[name]
    fig = figure(name)
    result = fig.result
    ax = fig.axes["main"]
    assert set(fig.axes) == {"main"}
    # The bars: ggstatsplot's built histogram, edges and counts.
    x = case.analyzed_values("x")
    (built,) = case.layer_built("GeomBar")
    drawn = bars(fig)
    assert len(drawn) == len(built)
    for bar, row in zip(drawn, built, strict=True):
        assert bar.get_height() == int(row["count"])
        assert math.isclose(
            bar.get_x(), number(row["xmin"]), rel_tol=1e-13, abs_tol=1e-13
        )
        right = bar.get_x() + bar.get_width()
        assert math.isclose(right, number(row["xmax"]), rel_tol=1e-13, abs_tol=1e-13)
        assert bar.get_y() == 0.0
    assert sum(b.get_height() for b in drawn) == len(x) == result.n_obs
    # The same bins as plot_geometry, exactly.
    bins = histogram_bins(x, binwidth=drawing(case).get("binwidth"))
    assert [b.get_x() for b in drawn] == list(bins.edges[:-1])
    # The percent axis: count / n as a percentage.
    (percent,) = ax.child_axes
    fig.draw()
    low, high = ax.get_ylim()
    assert percent.get_ylim() == pytest.approx(
        (low * 100 / len(x), high * 100 / len(x))
    )
    ticks = [t.get_text() for t in percent.get_yticklabels() if t.get_text()]
    assert ticks
    assert all(t.endswith("%") for t in ticks)
    # The centrality line and its label, where drawn.
    line = by_gid(fig, "centrality-line")
    label = text_of(fig, "centrality-label")
    center = result.descriptives[0].value
    if case.args.get("centrality.plotting") is False:
        assert line is None
        assert label is None
    else:
        assert isinstance(line, Line2D)
        assert center is not None
        assert list(line.get_xdata()) == [center, center]  # pyright: ignore[reportArgumentType]
        assert label == figure_texts(result).centrality_labels[0]
        # ggstatsplot's centrality line is the fixture's GeomVline.
        (vline,) = case.layer_built("GeomVline")
        assert math.isclose(center, number(vline[0]["xintercept"]), rel_tol=1e-12)
    # The test value: a line close to the bars, else an arrow at the edge.
    tv = result.settings.test_value
    assert tv is not None
    reach = TEST_VALUE_REACH * (bins.edges[-1] - bins.edges[0])
    words = text_of(fig, "test-value-label")
    assert words is not None
    if bins.edges[0] - reach <= tv <= bins.edges[-1] + reach:
        tv_line = by_gid(fig, "test-value-line")
        assert isinstance(tv_line, Line2D)
        assert list(tv_line.get_xdata()) == [tv, tv]  # pyright: ignore[reportArgumentType]
        assert words == f"test value {r_number_text(tv)}"
    else:
        assert by_gid(fig, "test-value-line") is None
        assert r_number_text(tv) in words
        assert words.startswith("←") or words.endswith("→")
    assert ax.get_xlabel() == result.x
    assert ax.get_ylabel() == "count"
    assert figure_codes(fig) == [], fig.check()
    # The result drawn is the result given, with the alt text of this figure.
    given = result_of(name)
    assert result == dataclasses.replace(
        given, text=dataclasses.replace(given.text, alt_text=result.text.alt_text)
    )
    assert result.text.alt_text == figure_alt_text(
        given,
        centrality=line is not None,
        test_value_line=by_gid(fig, "test-value-line") is not None,
        bins=len(drawn),
    )


def test_fixtures_include_both_test_value_marks() -> None:
    names = [p.values[0] for p in drawn_cases()]
    # 76, and 10 of the 16 from the barstats oracle run (the 6 refused not drawn).
    assert len(names) == 86
    marks = {by_gid(figure(n), "test-value-line") is None for n in names}
    assert marks == {True, False}


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def test_binwidth_sets_the_bins() -> None:
    fig = ip.render.histostats(tooth(), binwidth=5)
    widths = {round(b.get_width(), 12) for b in bars(fig)}
    assert widths == {5.0}
    edges = [b.get_x() for b in bars(fig)]
    # On the grid (k + 1/2) w, as ggplot2's stat_bin lays them.
    assert all(abs((e / 5 - 0.5) - round(e / 5 - 0.5)) < 1e-9 for e in edges)


def test_too_many_bars_are_refused_before_drawing() -> None:
    figure_ = Figure()
    FigureCanvasAgg(figure_)
    ax = figure_.add_subplot()
    text = message(
        ValueError, lambda: ip.render.histostats(tooth(), binwidth=0.001, ax=ax)
    )
    assert f"more than the {MAX_BARS:,} inferplot draws" in text
    assert "binwidth=0.297 (100 bars)" in text  # (33.9 - 4.2) / 100
    assert not ax.has_data()
    assert not ax.texts


def test_constant_column_draws_one_bar() -> None:
    # ggstatsplot's default for zero-range data: one bar 0.1 wide centred on the
    # value (edge_one_constant); a given binwidth sets its width.
    result = hist_result(pd.DataFrame({"v": [3.0] * 6}), "v")
    fig = ip.render.histostats(result)
    (bar,) = bars(fig)
    assert bar.get_height() == 6.0
    assert math.isclose(bar.get_x(), 2.95, rel_tol=1e-13)
    assert math.isclose(bar.get_width(), 0.1, rel_tol=1e-12)
    assert figure_codes(fig) == [], fig.check()
    wide = ip.render.histostats(result, binwidth=0.5)
    (bar,) = bars(wide)
    assert (bar.get_height(), bar.get_x(), bar.get_width()) == (6.0, 2.75, 0.5)


def test_an_arrow_label_does_not_widen_the_figure() -> None:
    # The "← test value" label once started at data x = 0 and the own figure grew
    # to reach it: ten values of 5 against 3 (edge_one_constant, a view 0.2 wide)
    # drew a figure 239 inches wide, mtcars mpg against 0 one 7.7 inches wide.
    sizes = [
        figure(name).figure.get_size_inches().tolist()
        for name in (
            "gghistostats-ToothGrowth-parametric",
            "gghistostats-edge_one_constant-nonparametric",
            "gghistostats-edge_one_constant-binwidth1-nonparametric",
            "gghistostats-mtcars-parametric",
        )
    ]
    widths = [w for w, _ in sizes]
    assert max(widths) - min(widths) < 0.3, sizes
    for shift in (0.0, 1e4):
        values = [v + shift for v in (1.0, 2.0, 2.5, 3.0, 4.0, 2.2)]
        result = ip.analyze.histostats(
            pd.DataFrame({"v": values}), "v", test_value=shift - 50
        )
        fig = ip.render.histostats(result)
        assert (
            text_of(fig, "test-value-label")
            == f"← test value {r_number_text(shift - 50)}"
        )
        assert fig.figure.get_size_inches()[0] < 7.0
        assert figure_codes(fig) == []


def test_a_missing_map_draws_no_centrality_line() -> None:
    # A MAP without a density (D55): the result has no centrality value.
    result = tooth(type="bayes")
    (d,) = result.descriptives
    missing = dataclasses.replace(
        result, descriptives=(dataclasses.replace(d, value=None),)
    )
    fig = ip.render.histostats(missing)
    assert by_gid(fig, "centrality-line") is None
    assert by_gid(fig, "centrality-label") is None
    assert by_gid(fig, "test-value-line") is not None
    assert figure_codes(fig) == []


def test_alt_text_describes_the_figure() -> None:
    result = tooth()
    fig = ip.render.histostats(result)
    alt = fig.result.text.alt_text
    assert alt != result.text.alt_text  # the analysis' figure draws no test value line
    assert "test value, 18" in alt
    assert "9 bins" in alt
    off = ip.render.histostats(result, centrality_plotting=False, binwidth=5)
    assert "mean, 18.81" not in off.result.text.alt_text
    assert "test value, 18" in off.result.text.alt_text
    assert off.result.text.alt_text == figure_alt_text(
        result, centrality=False, test_value_line=True, bins=len(bars(off))
    )
    far = figure("gghistostats-ToothGrowth-tv100-parametric")
    assert far.result.text.alt_text == figure_alt_text(
        far.result, test_value_line=False, bins=len(bars(far))
    )


def test_centrality_plotting_false_leaves_line_and_label_out() -> None:
    fig = ip.render.histostats(tooth(), centrality_plotting=False)
    assert by_gid(fig, "centrality-line") is None
    assert by_gid(fig, "centrality-label") is None
    assert by_gid(fig, "test-value-line") is not None


def test_bar_args_style_every_bar() -> None:
    fig = ip.render.histostats(tooth(), bar_args={"alpha": 0.3, "edgecolor": "black"})
    for bar in bars(fig):
        assert bar.get_alpha() == 0.3
        assert bar.get_edgecolor() == to_rgba("black", 0.3)


@pytest.mark.parametrize("prop", ["height", "width", "x", "xy", "bounds", "transform"])
def test_bar_args_may_not_move_the_bins(prop: str) -> None:
    text = message(
        ValueError, lambda: ip.render.histostats(tooth(), bar_args={prop: 1.0})
    )
    assert text.startswith(f"bar_args sets {prop!r}, which would")


def test_bar_args_errors() -> None:
    assert "Did you mean 'alpha'" in message(
        TypeError, lambda: ip.render.histostats(tooth(), bar_args={"alpah": 0.3})
    )
    assert message(
        ValueError, lambda: ip.render.histostats(tooth(), bar_args={"linewidth": "x"})
    ).startswith("bar_args has a value matplotlib rejects")


def test_invisible_bars_are_named_in_warnings() -> None:
    fig = ip.render.histostats(tooth(), bar_args={"alpha": 0})
    (last,) = fig.result.warnings[-1:]
    assert last.startswith("bar_args={alpha=0} makes the bars invisible")


def test_palette_colours_the_bars() -> None:
    fig = ip.render.histostats(tooth(), palette=["#0072B2"])
    face = bars(fig)[0].get_facecolor()
    assert face[:3] == pytest.approx(to_rgba("#0072B2")[:3])
    fig = ip.render.histostats(tooth(), palette="viridis")
    assert bars(fig)[0].get_facecolor()[:3] == pytest.approx(
        to_rgba(__import__("matplotlib").colormaps["viridis"](0.5))[:3]
    )
    assert "not a matplotlib colormap name" in message(
        ValueError, lambda: ip.render.histostats(tooth(), palette="nope")
    )


def test_user_texts() -> None:
    fig = ip.render.histostats(
        tooth(),
        title="Tooth length",
        xlab=r"Length ($\mu$m)",
        ylab="guinea pigs",
        caption="Source: Crampton (1947).",
    )
    ax = fig.axes["main"]
    assert text_of(fig, "title") == "Tooth length"
    assert ax.get_xlabel() == r"Length ($\mu$m)"
    assert ax.get_ylabel() == "guinea pigs"
    assert numbered_texts(fig, "caption")[-1] == "Source: Crampton (1947)."
    assert figure_codes(fig) == []
    assert "matplotlib cannot read it" in message(
        ValueError, lambda: ip.render.histostats(tooth(), title=r"$\frac{1$")
    )


def test_default_x_label_is_never_mathtext() -> None:
    result = hist_result(
        pd.DataFrame({"cost $ in $ USD": [1.0, 2, 4, 4, 7]}), "cost $ in $ USD"
    )
    fig = ip.render.histostats(result)
    label = fig.axes["main"].xaxis.label
    assert label.get_text() == "cost $ in $ USD"
    assert not label.get_parse_math()


def test_subtitle_needs_results_subtitle_false() -> None:
    assert "subtitle= would replace the results subtitle" in message(
        ValueError, lambda: ip.render.histostats(tooth(), subtitle="mine")
    )
    result = tooth(results_subtitle=False)
    fig = ip.render.histostats(result, subtitle="mine")
    assert text_of(fig, "subtitle") == "mine"


def test_wrong_results_and_axes() -> None:
    between = ip.analyze.betweenstats(
        pd.DataFrame({"g": ["a"] * 5 + ["b"] * 5, "v": np.arange(10.0)}), "g", "v"
    )
    assert "Draw it with ip.render.betweenstats(result)" in message(
        TypeError, lambda: ip.render.histostats(between)
    )
    saved = StatsResult.from_json(tooth().to_json(include_values=False))
    assert "holds no observations to draw" in message(
        ValueError, lambda: ip.render.histostats(saved)
    )
    not_axes: Any = "ax"
    assert message(TypeError, lambda: ip.render.histostats(tooth(), ax=not_axes)) == (
        "ax must be a matplotlib Axes or None; got str."
    )
    polar = Figure().add_subplot(projection="polar")
    assert "is a 'polar' Axes" in message(
        ValueError, lambda: ip.render.histostats(tooth(), ax=polar)
    )


def test_values_too_large_to_draw_are_refused() -> None:
    result = hist_result(pd.DataFrame({"v": [1e307, 2e307, 3e307, 5e307]}), "v")
    assert "divide x by 1e307" in message(
        ValueError, lambda: ip.render.histostats(result)
    )


# --------------------------------------------------------------------------- #
# the test value and the labels
# --------------------------------------------------------------------------- #


def _boxes(fig: StatsFigure, *gid: str) -> list[Any]:
    fig.draw()
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    out: list[Any] = []
    for name in gid:
        text = by_gid(fig, name)
        assert isinstance(text, Text)
        out.append(text.get_window_extent(renderer))
    return out


def test_labels_point_away_from_each_other() -> None:
    # Mean 18.81 above the test value 10: the centrality label right of its line,
    # the test value's label left of its own.
    fig = ip.render.histostats(tooth(test_value=10))
    center, tv = _boxes(fig, "centrality-label", "test-value-label")
    ax = fig.axes["main"]
    mean_px = ax.transData.transform((18.8133, 0))[0]
    tv_px = ax.transData.transform((10, 0))[0]
    assert center.x0 > mean_px
    assert tv.x1 < tv_px
    fig = ip.render.histostats(tooth(test_value=25))
    center, tv = _boxes(fig, "centrality-label", "test-value-label")
    assert center.x1 < ax.transData.transform((18.8133, 0))[0] + 1
    assert figure_codes(fig) == []


def test_labels_sit_above_the_tallest_bar() -> None:
    fig = figure("gghistostats-ToothGrowth-parametric")
    center, tv = _boxes(fig, "centrality-label", "test-value-label")
    ax = fig.axes["main"]
    tallest = max(b.get_height() for b in bars(fig))
    top_px = ax.transData.transform((0, tallest))[1]
    assert min(center.y0, tv.y0) > top_px
    assert max(center.y1, tv.y1) <= ax.bbox.y1


def test_close_labels_take_two_rows() -> None:
    # The trimmed mean (10.39) next to the test value (10) near the left edge:
    # the test value's label cannot go left, so it takes a second row.
    fig = figure("gghistostats-edge_one_outlier-robust")
    center, tv = _boxes(fig, "centrality-label", "test-value-label")
    assert tv.y1 <= center.y0 + 0.5
    line = by_gid(fig, "centrality-line")
    assert isinstance(line, Line2D)
    # It starts beyond the centrality line, which does not run through it.
    x_line = fig.axes["main"].transData.transform((line.get_xdata()[0], 0))[0]  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]
    assert tv.x0 > x_line
    assert figure_codes(fig) == []


def test_far_test_value_is_named_at_the_edge() -> None:
    fig = figure("gghistostats-ToothGrowth-tv100-parametric")
    assert by_gid(fig, "test-value-line") is None
    assert text_of(fig, "test-value-label") == "test value 100 →"
    ax = fig.axes["main"]
    (box,) = _boxes(fig, "test-value-label")
    assert box.x1 <= ax.bbox.x1
    assert box.x1 > ax.bbox.x1 - 10
    # The view is not stretched to reach it.
    assert ax.get_xlim()[1] < 50


# --------------------------------------------------------------------------- #
# ax=
# --------------------------------------------------------------------------- #


def _grid(
    width: float = 17.0, height: float = 5.5, **kwargs: Any
) -> tuple[Figure, Any]:
    fig = Figure(figsize=(width, height), dpi=100)
    FigureCanvasAgg(fig)
    return fig, fig.subplots(1, 2, gridspec_kw={"wspace": 0.45}, **kwargs)


def test_drawn_into_a_grid_without_resizing() -> None:
    figure_, (a, b) = _grid(width=20.0)
    left = ip.render.histostats(tooth(), ax=a, title="left")
    right = ip.render.histostats(result_of("gghistostats-mtcars-robust"), ax=b)
    assert figure_.get_size_inches().tolist() == [20.0, 5.5]
    assert left.figure is figure_
    assert right.axes["main"] is b
    assert figure_codes(left) == []
    assert figure_codes(right) == []
    # The title is the Axes' own title; the labels are inside it.
    assert a.get_title(loc="left") == "left"
    # The lines stop below the subtitle and above the caption, which ax= puts
    # inside the Axes.
    fig = left
    fig.draw()
    renderer = figure_.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]

    def in_left(gid: str) -> Any:
        (found,) = [x for x in a.findobj() if x.get_gid() == gid]
        return found

    subtitle, caption = in_left("subtitle"), in_left("caption-0")
    assert isinstance(subtitle, Text)
    assert isinstance(caption, Text)
    for gid in ("centrality-line", "test-value-line"):
        line = in_left(gid)
        assert isinstance(line, Line2D)
        box = line.get_window_extent(renderer)
        assert box.y1 <= subtitle.get_window_extent(renderer).y0 + 1
        assert box.y0 >= caption.get_window_extent(renderer).y1 - 1


def test_shared_axes_fit_both_panels() -> None:
    figure_, (a, b) = _grid(sharex=True, sharey=True)
    first = ip.render.histostats(tooth(), ax=a)
    second = ip.render.histostats(result_of("gghistostats-mtcars-parametric"), ax=b)
    del figure_
    # One scale fits both: nothing of the first panel is cut by the second.
    for fig in (first, second):
        assert "data_outside_view" not in [i.code for i in fig.check()]
    assert a.get_xlim() == b.get_xlim()


def test_small_axes_reports_what_does_not_fit() -> None:
    figure_ = Figure(figsize=(2.2, 1.6), dpi=100)
    FigureCanvasAgg(figure_)
    fig = ip.render.histostats(tooth(), ax=figure_.add_subplot(), title="A long title")
    codes = figure_codes(fig)
    assert "text_clipped" in codes or "label_overlap" in codes
    assert figure_.get_size_inches().tolist() == pytest.approx([2.2, 1.6])


def test_dark_background_ink() -> None:
    figure_ = Figure(figsize=(6, 4.5), dpi=100, facecolor="black")
    FigureCanvasAgg(figure_)
    ax = figure_.add_subplot()
    ax.set_facecolor("#111111")
    fig = ip.render.histostats(tooth(), ax=ax, caption="note")
    words = by_gid(fig, "test-value-label")
    assert isinstance(words, Text)
    assert words.get_color() == "#EEEEEE"
    assert ax.xaxis.label.get_color() == "#EEEEEE"
    (percent,) = ax.child_axes
    assert percent.yaxis.label.get_color() == "#EEEEEE"
    assert "low_contrast" not in [i.code for i in fig.check()]


# --------------------------------------------------------------------------- #
# checks that fire on a figure made to trigger them
# --------------------------------------------------------------------------- #


def _codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check()]


def test_view_checks() -> None:
    fig = ip.render.histostats(tooth())
    ax = fig.axes["main"]
    ax.set_xlim(10, 20)
    issue = next(i for i in fig.check() if i.code == "data_outside_view")
    assert "of 60 values" in issue.message
    assert "bars" in issue.message
    ax.set_xlim(-1e6, 1e6)
    assert "flat_data" in _codes(fig)
    fig = ip.render.histostats(tooth())
    fig.axes["main"].set_ylim(0, 5)
    issue = next(i for i in fig.check() if i.code == "data_outside_view")
    assert "bars" in issue.message
    fig = ip.render.histostats(tooth())
    fig.axes["main"].invert_xaxis()
    issue = next(i for i in fig.check() if i.code == "axis_changed")
    assert "ax.invert_xaxis()" in issue.fix
    fig = ip.render.histostats(tooth())
    fig.axes["main"].set_xscale("log")
    assert "axis_changed" in _codes(fig)


def test_low_contrast_and_missing_glyphs() -> None:
    fig = ip.render.histostats(tooth(), palette=["white"])
    issue = next(i for i in fig.check() if i.code == "low_contrast")
    assert "the colour of the bars" in issue.message
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fig = ip.render.histostats(tooth(), title="歯の長さ")
        codes = _codes(fig)
    assert "glyph_missing" in codes


def test_checks_of_other_figures_do_not_apply() -> None:
    fig = figure("gghistostats-ToothGrowth-parametric")
    # Read by path_hidden, paths_crowded, centrality_label_overlap,
    # centrality_label_ambiguous, the bracket crossings, points_subsampled and
    # point_labels_crowded: a histogram has none of these.
    assert fig.paths is None
    assert fig.centers == ()
    assert fig.labels == ()
    assert fig.brackets == ()
    assert fig.subsample is None
    assert fig.point_labels is None


# --------------------------------------------------------------------------- #
# the three forms, saving
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "type_name", ["parametric", "nonparametric", "robust", "bayes"]
)
def test_three_forms_agree(type_name: str, tmp_path: Any) -> None:
    """``ip.histostats`` draws what ``ip.render.histostats`` draws from the result
    of ``ip.analyze.histostats``, byte for byte."""
    frame = _tooth_frame()
    top = ip.histostats(
        frame, "len", type=type_name, test_value=18, binwidth=3, title="t"
    )
    result = ip.analyze.histostats(frame, "len", type=type_name, test_value=18)
    drawn = ip.render.histostats(result, binwidth=3, title="t")
    assert top.result == drawn.result
    for suffix in ("png", "svg"):
        top.save(tmp_path / f"a.{suffix}")
        drawn.save(tmp_path / f"b.{suffix}")
        assert (tmp_path / f"a.{suffix}").read_bytes() == (
            tmp_path / f"b.{suffix}"
        ).read_bytes()


def test_saved_files(tmp_path: Any) -> None:
    fig = ip.render.histostats(tooth(), title="Tooth length")
    for suffix in ("svg", "pdf", "png"):
        fig.save(tmp_path / f"a.{suffix}")
        fig.save(tmp_path / f"b.{suffix}")
        assert (tmp_path / f"a.{suffix}").read_bytes() == (
            tmp_path / f"b.{suffix}"
        ).read_bytes()
    import xml.etree.ElementTree as ET

    root = ET.parse(tmp_path / "a.svg").getroot()
    texts = [
        "".join(el.itertext()) for el in root.iter("{http://www.w3.org/2000/svg}text")
    ]
    assert "test value 18" in texts
    assert "Tooth length" in texts
    from tests.render._helpers import svg_description

    embedded = StatsResult.from_json(svg_description(tmp_path / "a.svg"))
    assert embedded.values == ()
    assert embedded.n_obs == 60
    assert {"bar-0", "centrality-line", "test-value-line"} <= gids(fig)


def test_repr_names_the_values() -> None:
    assert repr(figure("gghistostats-ToothGrowth-parametric")).startswith(
        "<StatsFigure histostats: len, 60 values;"
    )


def test_given_axes_is_a_matplotlib_axes() -> None:
    fig = figure("gghistostats-ToothGrowth-parametric")
    assert isinstance(fig.axes["main"], Axes)
