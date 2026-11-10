"""``ip.render.barstats``: every answerable ggbarstats fixture drawn, and every
argument.

Each fixture's result is assembled from its own numbers (``tests/bar_cases``;
the analysis' statistics are another package). For each figure: one bar per
level of y with observations, its segments' heights the cells' shares (to
1e-12), stacked with the first level of x on top as ggplot2 stacks them; the
cell labels, the labels above the bars and the n under each bar equal
ggstatsplot's built layers; the legend names the levels of x under the title
ggstatsplot gives it; and ``fig.check()`` reports nothing about the figure.
"""

from __future__ import annotations

import functools
import io
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

import inferplot as ip
from inferplot.render import StatsFigure, theme
from inferplot.result import StatsResult
from inferplot.text import figure_alt_text, figure_texts
from tests import bar_cases as bc
from tests.oracle_fixtures import OracleCase
from tests.render._bars import table_result
from tests.render._helpers import by_gid, svg_description

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
        "cell_labels_hidden",
        "figure_oversized",
    }
)
draw: Any = ip.render.barstats  # for calls with wrong argument types
BAR_CASES = [c for c in bc.ANSWERED if c.function == "ggbarstats"]


@functools.cache
def result_of(name: str) -> StatsResult:
    return bc.assembled(bc.BY_NAME[name])


@functools.cache
def figure(name: str) -> StatsFigure:
    return ip.render.barstats(result_of(name))


def figure_codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check() if i.code in FIGURE_CODES]


def codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check()]


def message(error: type[Exception], call: Any) -> str:
    with pytest.raises(error) as caught:
        call()
    return str(caught.value)


def mtcars() -> StatsResult:
    return result_of("ggbarstats-mtcars-cyl-am-parametric")


def segment(fig: StatsFigure, j: int, i: int) -> Rectangle | None:
    found = by_gid(fig, f"segment-{j}-{i}")
    assert found is None or isinstance(found, Rectangle)
    return found


def cell_label(fig: StatsFigure, j: int, i: int) -> Text | None:
    found = by_gid(fig, f"cell-label-{j}-{i}")
    assert found is None or isinstance(found, Text)
    return found


def legend_of(fig: StatsFigure) -> Legend:
    legend = fig.axes["main"].get_legend()
    assert legend is not None
    return legend


def tick_texts(fig: StatsFigure) -> list[str]:
    return [t.get_text() for t in fig.axes["main"].get_xticklabels()]


def tiny_cells() -> StatsResult:
    """A 4 x 3 table with a zero cell, two cells of 1 and one of 2."""
    return table_result(
        [[120, 0, 40], [1, 30, 2], [60, 50, 1], [3, 0, 57]],
        ["often", "sometimes", "rarely", "never"],
        ["A", "B", "C"],
    )


# --------------------------------------------------------------------------- #
# ggstatsplot's built layers
# --------------------------------------------------------------------------- #


def built_cell_labels(case: OracleCase) -> dict[tuple[int, int], str | None]:
    """ggstatsplot's cell labels by (bar, level of x), from 0: the built rows of
    the label layer, bar by bar in the levels' order."""
    (rows,) = case.layer_built("GeomLabel")
    out: dict[tuple[int, int], str | None] = {}
    seen: dict[int, int] = {}
    for row in rows:
        j = int(row["x"]) - 1
        i = seen.get(j, 0)
        seen[j] = i + 1
        out[(j, i)] = row["label"]
    return out


def built_texts(case: OracleCase) -> tuple[dict[int, str], dict[int, str]]:
    """ggstatsplot's labels above the bars (the parsed layer: the p-value as
    written in ``italic(p)=='0.09'``) and under them ("(n = 13)"), by bar."""
    above: dict[int, str] = {}
    under: dict[int, str] = {}
    for rows in case.layer_built("GeomText"):
        for row in rows:
            label = str(row["label"])
            j = int(row["x"]) - 1
            found = re.fullmatch(r"list\(~italic\(p\)=='(.*)'\)", label)
            if found is not None:
                above[j] = found.group(1)
            else:
                assert label.startswith("(n = "), label
                under[j] = label
    return above, under


# --------------------------------------------------------------------------- #
# every fixture
# --------------------------------------------------------------------------- #


def test_every_answerable_fixture_is_drawn() -> None:
    assert len(bc.ANSWERED) == 68
    assert len(BAR_CASES) == 65  # and 3 of the 5 ggpiestats cross-checks


@pytest.mark.parametrize("name", [pytest.param(c.name, id=c.name) for c in bc.ANSWERED])
def test_fixture_figure(name: str) -> None:
    result = result_of(name)
    fig = figure(name)
    k = len(result.levels)
    bars = result.bars
    # One bar per level of y with observations, its n under it.
    assert len(tick_texts(fig)) == len(bars)
    for j, (bar, tick) in enumerate(zip(bars, tick_texts(fig), strict=True)):
        assert tick.endswith(f"(n = {bar.n:,})")
        if bar.level is None:
            assert tick == f"(n = {bar.n:,})"
        top = None
        for i in range(k):
            cell = result.cells[j * k + i]
            drawn = segment(fig, j, i)
            label = cell_label(fig, j, i)
            if cell.count == 0:
                assert drawn is None
                assert label is None
                continue
            assert drawn is not None
            assert label is not None
            # The share, to 1e-12; stacked with the first level on top.
            assert abs(drawn.get_height() - cell.percent) <= 1e-12
            if top is not None:
                assert abs(drawn.get_y() + drawn.get_height() - top) <= 1e-9
            top = drawn.get_y()
        assert top is not None
        assert abs(top) <= 1e-9  # the last level drawn starts at 0%
    # The legend: the levels of x, first on top, under the name of x.
    legend = legend_of(fig)
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == list(result.levels)
    assert legend.get_title().get_text() == result.x
    assert fig.axes["main"].get_xlabel() == (result.y or "")
    assert fig.axes["main"].get_ylabel() == ""
    assert figure_codes(fig) == [], fig.check()
    # Percent ticks 0% to 100% by 10, none above 100% (Y7).
    ticks = [t.get_text() for t in fig.axes["main"].get_yticklabels()]
    assert ticks == [f"{v}%" for v in range(0, 101, 10)]


@pytest.mark.parametrize("name", [pytest.param(c.name, id=c.name) for c in BAR_CASES])
def test_fixture_labels_equal_ggstatsplots(name: str) -> None:
    """The cell labels, the bars' test labels and the n labels as ggstatsplot
    built them; the legend title and the axis title as it set them."""
    case = bc.BY_NAME[name]
    result = result_of(name)
    fig = figure(name)
    k = len(result.levels)
    expected = built_cell_labels(case)
    assert len(expected) == len(result.bars) * k
    for (j, i), label in expected.items():
        drawn = cell_label(fig, j, i)
        if label is None:
            assert drawn is None, (j, i)
        else:
            assert drawn is not None, (j, i)
            assert drawn.get_text() == label
    above, under = built_texts(case)
    drawn_texts = figure_texts(result)
    for j, tick in enumerate(tick_texts(fig)):
        assert tick.split("\n")[-1] == under[j]
        text = by_gid(fig, f"bar-label-{j}")
        if j in above:
            assert isinstance(text, Text)
            assert text.get_text() == drawn_texts.bar_labels[j]
            assert result.text.bar_labels[j] == f"p = {above[j]}"
        else:
            assert text is None
    texts = case.plot_texts
    assert legend_of(fig).get_title().get_text() == texts["legend_title"]
    assert fig.axes["main"].get_xlabel() == (texts["labels"]["x"] or "")


def test_fixtures_cover_every_kind_of_label() -> None:
    """The fixtures drawn include zero cells, labels moved beside their bar,
    counts and both, and bars without a test label."""
    places = {c.place for name in bc.ANSWERED for c in figure(name.name).cell_labels}
    assert "inside" in places
    labels = {result_of(c.name).settings.label for c in bc.ANSWERED}
    assert labels == {"percentage", "counts", "both"}
    zero = [
        c for c in bc.ANSWERED if any(x.count == 0 for x in result_of(c.name).cells)
    ]
    assert zero
    without = [c for c in bc.ANSWERED if not result_of(c.name).settings.proportion_test]
    assert without


def test_a_figure_from_a_saved_copy_is_the_same() -> None:
    """The copy of a result a saved figure embeds keeps the counts, cells and
    bars, so it draws the same figure."""
    result = mtcars()
    copy = StatsResult.from_json(result.to_json(include_values=False))
    one, two = ip.render.barstats(result), ip.render.barstats(copy)
    a, b = io.BytesIO(), io.BytesIO()
    one.figure.savefig(a, format="png")
    two.figure.savefig(b, format="png")
    assert a.getvalue() == b.getvalue()


# --------------------------------------------------------------------------- #
# labels that do not fit their segment
# --------------------------------------------------------------------------- #


def test_small_cells_are_labelled_beside_their_bar() -> None:
    fig = ip.render.barstats(tiny_cells())
    assert figure_codes(fig) == []
    fig.draw()
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    places = {c.name: c.place for c in fig.cell_labels}
    assert places["'sometimes' in 'A'"] == "outside"
    assert places["'never' in 'A'"] == "outside"
    assert places["'often' in 'A'"] == "inside"
    assert "omitted" not in places.values()
    for label in fig.cell_labels:
        box = label.text.get_window_extent(renderer)
        own = label.segment.get_window_extent(renderer)
        if label.place == "inside":
            assert own.x0 <= box.x0
            assert box.x1 <= own.x1
            assert own.y0 <= box.y0
            assert box.y1 <= own.y1
            continue
        # Beside its own bar, right of it and before the next one; joined by a
        # line that starts at its segment.
        assert box.x0 > own.x1
    leaders = [
        a for a in fig.figure.findobj(Line2D) if str(a.get_gid()).startswith("leader-")
    ]
    assert len(leaders) == sum(1 for c in fig.cell_labels if c.place == "outside")
    for line in leaders:
        _, j, i = str(line.get_gid()).rsplit("-", 2)
        drawn = segment(fig, int(j), int(i))
        assert drawn is not None
        xs = list(line.get_xdata())  # pyright: ignore[reportArgumentType]
        ys = list(line.get_ydata())  # pyright: ignore[reportArgumentType]
        assert xs[0] == pytest.approx(drawn.get_x() + drawn.get_width())
        assert ys[0] == pytest.approx(drawn.get_y() + drawn.get_height() / 2)


def test_moved_labels_stack_in_order_without_overlap() -> None:
    """Five cells of 1 next to 5,000: every label is drawn, beside the bar, in
    the stacking order, none over another."""
    result = table_result(
        [[1, 2], [1, 3], [5000, 4000], [2, 1], [1, 1]], ["a", "b", "c", "d", "e"]
    )
    fig = ip.render.barstats(result)
    assert figure_codes(fig) == []
    fig.draw()
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    for j in range(2):
        moved = [c for c in fig.cell_labels if c.bar == j and c.place == "outside"]
        assert len(moved) == 4
        tops = [c.text.get_window_extent(renderer).y1 for c in moved]
        assert tops == sorted(tops, reverse=True)


def test_own_figure_grows_for_many_moved_labels() -> None:
    """Twelve levels, eleven of them single counts next to 10,000: the bars grow
    taller so that every label fits beside them."""
    counts = [[1, 1]] * 11 + [[10_000, 9_000]]
    fig = ip.render.barstats(
        table_result(counts, [f"level {i}" for i in range(12)], label="both")
    )
    assert figure_codes(fig) == []
    assert all(c.place != "omitted" for c in fig.cell_labels)
    _width, height = fig.figure.get_size_inches()
    assert height < 9.0


def test_given_axes_too_small_leaves_labels_out_and_says_so() -> None:
    figure_ = Figure(figsize=(3.0, 2.2), dpi=100)
    FigureCanvasAgg(figure_)
    fig = ip.render.barstats(tiny_cells(), ax=figure_.add_subplot())
    issue = next(i for i in fig.check() if i.code == "cell_labels_hidden")
    assert "is left out for lack of room" in issue.message
    assert "fig.result.cells" in issue.fix
    assert figure_.get_size_inches().tolist() == pytest.approx([3.0, 2.2])


def test_a_label_moved_off_its_segment_is_reported() -> None:
    fig = ip.render.barstats(mtcars())
    label = cell_label(fig, 0, 0)
    assert label is not None
    label.set_position((0.0, 50.0))  # over another segment
    issue = next(i for i in fig.check() if i.code == "cell_labels_hidden")
    assert "the label of '4' in '0' ('16%') reaches beyond its segment" in issue.message
    label.set_visible(False)
    issue = next(i for i in fig.check() if i.code == "cell_labels_hidden")
    assert "is not drawn" in issue.message


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def test_label_kinds_and_digits() -> None:
    counts = [[1234, 300], [56, 2000], [700, 7]]
    both = ip.render.barstats(table_result(counts, label="both"))
    label = cell_label(both, 0, 0)
    assert label is not None
    assert label.get_text() == "1,234\n(62%)"
    only = ip.render.barstats(table_result(counts, label="counts"))
    label = cell_label(only, 1, 1)
    assert label is not None
    assert label.get_text() == "2,000"
    digits = ip.render.barstats(table_result(counts, digits_perc=2))
    label = cell_label(digits, 0, 1)
    assert label is not None
    assert label.get_text() == f"{round(56 / 1990 * 100, 2)}%"
    for fig in (both, only, digits):
        assert figure_codes(fig) == []


def test_bar_labels_follow_proportion_test() -> None:
    with_tests = ip.render.barstats(table_result([[3, 5], [4, 1]]))
    assert by_gid(with_tests, "bar-label-0") is not None
    without = ip.render.barstats(table_result([[3, 5], [4, 1]], proportion_test=False))
    assert by_gid(without, "bar-label-0") is None
    one_way = ip.render.barstats(table_result([[3], [4], [9]], one_way=True))
    assert by_gid(one_way, "bar-label-0") is None
    assert tick_texts(one_way) == ["(n = 16)"]
    assert one_way.axes["main"].get_xlabel() == ""


def test_user_texts_and_legend_title() -> None:
    fig = ip.render.barstats(
        mtcars(),
        title="Engines",
        xlab="transmission",
        ylab=r"share ($\%$)",
        caption="note",
        legend_title=r"$\alpha$ cylinders",
    )
    ax = fig.axes["main"]
    assert ax.get_xlabel() == "transmission"
    assert ax.get_ylabel() == r"share ($\%$)"
    title = legend_of(fig).get_title()
    assert title.get_text() == r"$\alpha$ cylinders"
    assert title.get_parse_math()
    assert by_gid(fig, "title") is not None
    assert by_gid(fig, "caption-0") is not None
    assert figure_codes(fig) == []
    # Default texts from column names are never mathtext.
    named = ip.render.barstats(table_result([[3, 5], [4, 1]], x="cost $", y="a_b"))
    assert not legend_of(named).get_title().get_parse_math()
    assert not named.axes["main"].xaxis.label.get_parse_math()
    error = message(
        ValueError, lambda: ip.render.barstats(mtcars(), legend_title="$\\frac$")
    )
    assert error.startswith("legend_title is drawn as matplotlib mathtext")


def test_subtitle_needs_results_subtitle_false() -> None:
    assert "subtitle= would replace" in message(
        ValueError, lambda: ip.render.barstats(mtcars(), subtitle="mine")
    )
    quiet = result_of("ggbarstats-mtcars-cyl-am-nosubtitle-parametric")
    fig = ip.render.barstats(quiet, subtitle="mine")
    subtitle = by_gid(fig, "subtitle")
    assert isinstance(subtitle, Text)
    assert subtitle.get_text() == "mine"


def test_palette_and_many_levels() -> None:
    fig = ip.render.barstats(mtcars(), palette=["red", "green", "blue"])
    drawn = segment(fig, 0, 1)
    assert drawn is not None
    assert to_hex(drawn.get_facecolor()) == "#008000"
    fig = ip.render.barstats(mtcars(), palette="viridis")
    assert figure_codes(fig) == []
    assert "palette has 2 colors" in message(
        ValueError, lambda: ip.render.barstats(mtcars(), palette=["red", "blue"])
    )
    many = table_result([[i + 1, 12 - i] for i in range(12)])
    fig = ip.render.barstats(many)
    assert any("tab20" in w for w in fig.result.warnings)
    assert figure_codes(fig) == []


def test_bar_args_style_every_segment_and_the_legend() -> None:
    fig = ip.render.barstats(mtcars(), bar_args={"alpha": 0.5, "hatch": "//"})
    for j in range(2):
        for i in range(3):
            drawn = segment(fig, j, i)
            assert drawn is not None
            assert drawn.get_alpha() == 0.5
            assert drawn.get_hatch() == "//"
    keys = legend_of(fig).legend_handles
    assert all(isinstance(k, Rectangle) and k.get_hatch() == "//" for k in keys)
    assert figure_codes(fig) == []


@pytest.mark.parametrize("prop", ["x", "y", "width", "height", "bounds", "xy"])
def test_bar_args_may_not_move_the_segments(prop: str) -> None:
    assert "Remove" in message(
        ValueError, lambda: ip.render.barstats(mtcars(), bar_args={prop: 1})
    )


def test_bar_args_errors_and_invisible_segments() -> None:
    assert "Did you mean 'facecolor'" in message(
        TypeError, lambda: ip.render.barstats(mtcars(), bar_args={"facecolour": "r"})
    )
    assert "rejects" in message(
        ValueError, lambda: ip.render.barstats(mtcars(), bar_args={"alpha": "much"})
    )
    fig = ip.render.barstats(mtcars(), bar_args={"visible": False})
    assert any("makes the bar segments invisible" in w for w in fig.result.warnings)


def test_wrong_results_and_axes() -> None:
    histo = ip.analyze.histostats(pd.DataFrame({"v": [1.0, 2.0, 4.0]}), "v")
    assert "comes from ip.analyze.histostats()" in message(
        TypeError, lambda: ip.render.barstats(histo)
    )
    assert "matplotlib Axes" in message(TypeError, lambda: draw(mtcars(), ax="axes"))
    figure_ = Figure()
    FigureCanvasAgg(figure_)
    polar = figure_.add_subplot(projection="polar")
    assert "plain rectangular" in message(
        ValueError, lambda: ip.render.barstats(mtcars(), ax=polar)
    )
    a, b = Figure().subplots(1, 2, sharex=True)
    del b
    assert "shares its x axis" in message(
        ValueError, lambda: ip.render.barstats(mtcars(), ax=a)
    )


def test_long_names_wrap_and_the_result_keeps_them() -> None:
    long_x = ["strongly agree with the statement", "neither agree nor disagree"]
    long_y = ["respondents from the northern region", "the south"]
    result = table_result([[30, 12], [5, 40]], long_x, long_y)
    fig = ip.render.barstats(result)
    assert result.levels == tuple(long_x)
    first = tick_texts(fig)[0]
    assert first.count("\n") >= 2
    assert first.endswith("(n = 35)")
    entries = [t.get_text() for t in legend_of(fig).get_texts()]
    assert entries[0] != long_x[0]
    assert entries[0].replace("\n", " ") == long_x[0]
    assert figure_codes(fig) == []


def test_control_characters_and_missing_glyphs(tmp_path: Path) -> None:
    result = table_result([[3, 5], [4, 1]], ["bell\x07", "tab\there"], ["歯", "b"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = ip.render.barstats(result)
        found = codes(fig)
        path = tmp_path / "control.svg"
        fig.save(path)
        assert not [w for w in caught if str(w.message).startswith("Glyph 27503")]
    entries = [t.get_text() for t in legend_of(fig).get_texts()]
    assert entries == ["bell�", "tab here"]
    assert "glyph_missing" in found
    ET.parse(path)  # well-formed XML


# --------------------------------------------------------------------------- #
# ax=
# --------------------------------------------------------------------------- #


def grid(width: float = 12.0, height: float = 5.0, **kwargs: Any) -> tuple[Figure, Any]:
    figure_ = Figure(figsize=(width, height), dpi=100)
    FigureCanvasAgg(figure_)
    return figure_, figure_.subplots(1, 2, gridspec_kw={"wspace": 0.3}, **kwargs)


def test_drawn_into_a_grid_without_resizing() -> None:
    quiet = result_of("ggbarstats-mtcars-cyl-am-nosubtitle-parametric")
    other = result_of("ggbarstats-edge_bar_zero_cell-parametric")
    figure_, (a, b) = grid(width=14.0, height=6.0)
    left = ip.render.barstats(quiet, ax=a, title="left")
    right = ip.render.barstats(other, ax=b)
    assert figure_.get_size_inches().tolist() == [14.0, 6.0]
    assert left.figure is figure_
    assert right.axes["main"] is b
    assert a.get_title(loc="left") == "left"
    assert figure_codes(left) == []
    assert figure_codes(right) == []
    # The legend inside the Axes, right of the bars.
    left.draw()
    renderer = figure_.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    box = legend_of(left).get_window_extent(renderer)
    frame = a.get_window_extent(renderer)
    assert frame.x0 < box.x0
    assert box.x1 <= frame.x1
    bars = [r for r in a.findobj(Rectangle) if str(r.get_gid()).startswith("segment-")]
    assert len(bars) == 6
    assert all(r.get_window_extent(renderer).x1 < box.x0 for r in bars)


def test_shared_y_axes_fit_both_panels() -> None:
    figure_, (a, b) = grid(sharey=True)
    del figure_
    first = ip.render.barstats(mtcars(), ax=a)
    second = ip.render.barstats(tiny_cells(), ax=b)
    for fig in (first, second):
        assert "data_outside_view" not in codes(fig)


def test_dark_background_ink() -> None:
    figure_ = Figure(figsize=(7, 5), dpi=100, facecolor="black")
    FigureCanvasAgg(figure_)
    ax = figure_.add_subplot()
    ax.set_facecolor("#111111")
    fig = ip.render.barstats(tiny_cells(), ax=ax, caption="note")
    assert ax.xaxis.label.get_color() == theme.LIGHT_INK
    entries = legend_of(fig).get_texts()
    assert all(t.get_color() == theme.LIGHT_INK for t in entries)
    moved = [c for c in fig.cell_labels if c.place == "outside"]
    assert moved
    assert all(c.text.get_color() == theme.LIGHT_INK for c in moved)
    # Segments are separated in the background's colour.
    drawn = segment(fig, 0, 0)
    assert drawn is not None
    assert to_hex(drawn.get_edgecolor()) == "#111111"
    assert "low_contrast" not in codes(fig)


def test_cell_label_ink_contrasts_with_its_segment() -> None:
    fig = ip.render.barstats(mtcars(), palette=["#000000", "#FFFFFF", "#0072B2"])
    dark = cell_label(fig, 0, 0)
    light = cell_label(fig, 0, 1)
    assert dark is not None
    assert light is not None
    assert dark.get_color() == theme.LIGHT_INK
    assert light.get_color() == theme.INK
    # White segments on white: the colour is reported, the labels are not.
    issue = next(i for i in fig.check() if i.code == "low_contrast")
    assert "the colour of '6'" in issue.message
    assert "cell label" not in issue.message


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def test_view_checks() -> None:
    fig = ip.render.barstats(mtcars())
    ax = fig.axes["main"]
    ax.set_ylim(0, 50)
    issue = next(i for i in fig.check() if i.code == "data_outside_view")
    assert "of 6 bar segments" in issue.message
    ax.set_ylim(-1e9, 1e9)
    assert "flat_data" in codes(fig)
    fig = ip.render.barstats(mtcars())
    fig.axes["main"].invert_yaxis()
    issue = next(i for i in fig.check() if i.code == "axis_changed")
    assert "the bars stacked" in issue.message
    assert "ax.invert_yaxis()" in issue.fix
    fig = ip.render.barstats(mtcars())
    fig.axes["main"].set_yscale("log")
    assert "axis_changed" in codes(fig)
    fig = ip.render.barstats(mtcars())
    fig.axes["main"].set_xlim(0.5, 3)
    issue = next(i for i in fig.check() if i.code == "data_outside_view")
    assert "bar segments" in issue.message


def test_overlap_and_clipping_in_a_small_axes() -> None:
    figure_ = Figure(figsize=(2.4, 1.8), dpi=100)
    FigureCanvasAgg(figure_)
    fig = ip.render.barstats(mtcars(), ax=figure_.add_subplot(), title="A long title")
    found = codes(fig)
    assert "text_clipped" in found or "label_overlap" in found


def test_checks_of_other_figures_do_not_apply() -> None:
    fig = ip.render.barstats(mtcars())
    # Read by path_hidden, paths_crowded, centrality_label_overlap,
    # centrality_label_ambiguous, the bracket crossings, points_subsampled and
    # point_labels_crowded: a bar chart has none of these.
    assert fig.paths is None
    assert fig.centers == ()
    assert fig.labels == ()
    assert fig.brackets == ()
    assert fig.subsample is None
    assert fig.point_labels is None


# --------------------------------------------------------------------------- #
# saved files and the three forms
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("suffix", [".svg", ".pdf", ".png"])
def test_saved_files_are_reproducible_and_embed_the_result(
    suffix: str, tmp_path: Path
) -> None:
    fig = ip.render.barstats(tiny_cells())
    first, second = tmp_path / f"a{suffix}", tmp_path / f"b{suffix}"
    fig.save(first)
    fig.save(second)
    assert first.read_bytes() == second.read_bytes()
    if suffix == ".svg":
        embedded = StatsResult.from_json(svg_description(first))
        assert embedded.counts == fig.result.counts
        assert embedded.cells == fig.result.cells


def test_three_forms() -> None:
    """``ip.barstats``, ``ip.analyze.barstats`` then ``ip.render.barstats``:
    the same figure. Runs once the analysis computes its statistics (until then
    ``ip.analyze.barstats`` raises NotImplementedError and the test is skipped
    with that reason)."""
    frame = pd.DataFrame({"cyl": list("4466888846") * 3, "am": list("0101010011") * 3})
    try:
        result = ip.analyze.barstats(frame, "cyl", "am")
    except NotImplementedError as error:
        pytest.skip(f"the barstats statistics are not joined yet: {error}")
    one = ip.barstats(frame, "cyl", "am")
    two = ip.render.barstats(result)
    a, b = io.BytesIO(), io.BytesIO()
    one.figure.savefig(a, format="png")
    two.figure.savefig(b, format="png")
    assert a.getvalue() == b.getvalue()
    assert one.result == two.result
    assert figure_codes(one) == []


def test_alt_text_describes_the_figure() -> None:
    """The alt text is ``figure_alt_text`` told what the figure draws: the
    default figure gives the analysis' own alt text; without the bars' tests
    it does not mention them; with every cell label left out it says none are
    drawn."""
    result = mtcars()
    assert ip.render.barstats(result).result.text.alt_text == result.text.alt_text
    quiet = result_of("ggbarstats-mtcars-cyl-am-noproptest-parametric")
    fig = ip.render.barstats(quiet)
    assert by_gid(fig, "bar-label-0") is None
    assert fig.result.text.alt_text == figure_alt_text(
        quiet, cell_labels=True, bar_tests=False, legend=True
    )
    one_way = result_of("ggbarstats-mtcars-cyl-parametric")
    assert ip.render.barstats(one_way).result.text.alt_text == (one_way.text.alt_text)
    figure_ = Figure(figsize=(1.6, 1.2), dpi=72)
    FigureCanvasAgg(figure_)
    tiny = table_result([[1, 1], [500, 600], [1, 2]], ["a", "b", "c"])
    fig = ip.render.barstats(tiny, ax=figure_.add_subplot())
    assert {c.place for c in fig.cell_labels} == {"omitted"}
    assert fig.result.text.alt_text == figure_alt_text(
        tiny, cell_labels=False, bar_tests=True, legend=True
    )
    assert "Labels show" not in fig.result.text.alt_text


def test_given_axes_with_a_log_or_inverted_axis_is_refused() -> None:
    for change in ("log", "invert"):
        figure_ = Figure()
        FigureCanvasAgg(figure_)
        ax = figure_.add_subplot()
        if change == "log":
            ax.set_yscale("log")
        else:
            ax.invert_xaxis()
        error = message(ValueError, lambda ax=ax: ip.render.barstats(mtcars(), ax=ax))
        assert "Undo it before the call" in error
        assert not ax.patches  # nothing drawn
    figure_ = Figure()
    FigureCanvasAgg(figure_)
    ax = figure_.add_subplot()
    ax.set_yscale("log")
    frame = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    # The one-call form refuses it before the analysis.
    assert "Undo it before the call" in message(
        ValueError, lambda: ip.barstats(frame, "a", "b", ax=ax)
    )
    assert "results_subtitle=False in this call" in message(
        ValueError, lambda: ip.barstats(frame, "a", "b", subtitle="mine")
    )


def test_segments_the_colour_of_the_background_are_named() -> None:
    fig = ip.render.barstats(mtcars(), bar_args={"facecolor": "white"})
    (warning,) = [w for w in fig.result.warnings if "bar_args=" in w]
    assert "makes the bar segments invisible" in warning
    assert "cannot be told apart" in warning


def test_unused_palette_colours_are_named() -> None:
    fig = ip.render.barstats(mtcars(), palette=["red", "green", "blue", "black"])
    assert any("uses the first 3, not the other 1" in w for w in fig.result.warnings)


def test_forty_bars_widen_the_figure_within_its_bounds() -> None:
    """40 bars: a slot of at least 0.8 in each, and no figure_oversized."""
    counts = [
        [(j * 7) % 13 + 1 for j in range(40)],
        [(j * 5) % 11 + 2 for j in range(40)],
    ]
    fig = ip.render.barstats(table_result(counts, ["a", "b"]))
    width, _height = fig.figure.get_size_inches()
    assert 40 * 0.8 <= width <= 4.0 + 1.25 * 40
    assert figure_codes(fig) == []
