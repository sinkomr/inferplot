"""``ip.render.scatterstats``: every ggscatterstats fixture drawn, and every argument.

Each fixture's call is analyzed by ``ip.analyze.scatterstats``
(``tests/no_level_cases.analyzed``) and drawn with the fixture's drawing arguments
(``marginal``). For each figure: the points are the result's pairs, the line and
band are ``ols_band``'s and ggstatsplot's built smooth layer, the marginal
histograms are ggside's built layers in panels that share the plot's axes exactly,
the point labels are the result's, and ``fig.check()`` reports only what the
result itself reports.
"""

from __future__ import annotations

import functools
import math
import warnings
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection, PolyCollection, QuadMesh
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.text import Text

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.render.comparison import MAX_POINTS, SUBSAMPLE_SIZE
from inferplot.render.scatterstats import MAX_LABELS
from inferplot.result import StatsResult
from inferplot.stats import marginal_bins, ols_band
from tests.no_level_cases import REFUSALS, analyzed
from tests.oracle_args import split
from tests.oracle_fixtures import OracleCase, cases_for, number, require_cases
from tests.render._helpers import by_gid, gids, numbered_count, svg_description
from tests.render._no_levels import scatter_result

CASES = require_cases(cases_for("ggscatterstats"), "ggscatterstats fixtures")
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
    """``ip.analyze.scatterstats`` on the fixture's call
    (``tests/no_level_cases.analyzed``)."""
    result = analyzed(name)
    assert result is not None, name
    return result


def drawn_cases() -> list[Any]:
    """The fixtures inferplot analyzes: all but a constant x, which it refuses
    (``REFUSALS``; ggstatsplot prints NA for every number)."""
    return [
        pytest.param(c.name, id=c.name)
        for c in CASES
        if not (c.name in REFUSALS and REFUSALS[c.name].replay is None)
    ]


@functools.cache
def figure(name: str) -> StatsFigure:
    return ip.render.scatterstats(result_of(name), **drawing(BY_NAME[name]))


def figure_codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check() if i.code in FIGURE_CODES]


def codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check()]


def mtcars(**kwargs: Any) -> StatsResult:
    """mtcars' mpg by wt, or ``ip.analyze.scatterstats`` with ``kwargs``."""
    if not kwargs:
        return result_of("ggscatterstats-mtcars-parametric")
    return ip.analyze.scatterstats(_mtcars(), "wt", "mpg", **kwargs)


@functools.cache
def _mtcars() -> pd.DataFrame:
    case = BY_NAME["ggscatterstats-mtcars-labels-parametric"]
    return pd.DataFrame(
        {
            "wt": case.analyzed_values("x"),
            "mpg": case.analyzed_values("y"),
            "car": result_of(case.name).point_labels,
        }
    )


def message(error: type[Exception], call: Any) -> str:
    with pytest.raises(error) as caught:
        call()
    return str(caught.value)


def points_of(fig: StatsFigure) -> PathCollection:
    points = by_gid(fig, "points")
    assert isinstance(points, PathCollection)
    return points


def labels_of(fig: StatsFigure) -> list[Text]:
    found = [
        by_gid(fig, f"point-label-{i}")
        for i in range(numbered_count(fig, "point-label"))
    ]
    return [t for t in found if isinstance(t, Text)]


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12)


# --------------------------------------------------------------------------- #
# every fixture
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", drawn_cases())
def test_fixture_figure(name: str) -> None:
    case = BY_NAME[name]
    fig = figure(name)
    result = fig.result
    ax = fig.axes["main"]
    x, y = case.analyzed_values("x"), case.analyzed_values("y")
    # The points: every complete pair, in row order.
    offsets = np.asarray(points_of(fig).get_offsets())
    assert offsets[:, 0].tolist() == x
    assert offsets[:, 1].tolist() == y
    # The line and band: ols_band's, and ggstatsplot's built lm smooth.
    line = by_gid(fig, "smooth-line")
    band = by_gid(fig, "smooth-band")
    built = case.layer_built("GeomSmooth")
    # (Only a constant x has none, and the analysis refuses it.)
    assert built != [[]]
    assert isinstance(line, Line2D)
    assert isinstance(band, PolyCollection)
    level = result.settings.conf_level
    mine = ols_band(x, y, conf_level=level)
    assert list(line.get_xdata()) == list(mine.x)  # pyright: ignore[reportArgumentType]
    assert list(line.get_ydata()) == list(mine.fit)  # pyright: ignore[reportArgumentType]
    vertices = np.asarray(band.get_paths()[0].vertices, dtype=np.float64)
    lows = {(float(a), float(b)) for a, b in zip(mine.x, mine.conf_low, strict=True)}
    highs = {(float(a), float(b)) for a, b in zip(mine.x, mine.conf_high, strict=True)}
    drawn = {(float(a), float(b)) for a, b in vertices}
    assert lows <= drawn
    assert highs <= drawn
    method = (case.args.get("smooth.line.args") or {}).get("method", "lm")
    if method == "lm":
        (rows,) = built
        for i, row in enumerate(rows):
            assert _close(mine.fit[i], number(row["y"]))
            assert _close(mine.conf_low[i], number(row["ymin"]))
            assert _close(mine.conf_high[i], number(row["ymax"]))
    # The marginal histograms: ggside's built layers, sharing the plot's axes.
    if case.args.get("marginal") is False:
        assert set(fig.axes) == {"main"}
    else:
        assert set(fig.axes) == {"main", "xside", "yside"}
        xside, yside = fig.axes["xside"], fig.axes["yside"]
        assert xside.get_shared_x_axes().joined(ax, xside)
        assert yside.get_shared_y_axes().joined(ax, yside)
        # Aligned with the plot to the pixel.
        assert xside.bbox.x0 == ax.bbox.x0
        assert xside.bbox.x1 == ax.bbox.x1
        assert yside.bbox.y0 == ax.bbox.y0
        assert yside.bbox.y1 == ax.bbox.y1
        for panel, column, geom, low, high in (
            ("xside", x, "GeomXsidebar", "xmin", "xmax"),
            ("yside", y, "GeomYsidebar", "ymin", "ymax"),
        ):
            bins = marginal_bins(column)
            bars = [by_gid(fig, f"{panel}-bar-{i}") for i in range(len(bins.counts))]
            assert numbered_count(fig, f"{panel}-bar") == len(bins.counts)
            (rows,) = [lay["built"] for lay in case.plot_layers if lay["geom"] == geom]
            for bar, count, row in zip(bars, bins.counts, rows, strict=True):
                assert isinstance(bar, Rectangle)
                along = bar.get_x() if panel == "xside" else bar.get_y()
                size = bar.get_height() if panel == "xside" else bar.get_width()
                assert size == count == int(row["count"])
                assert math.isclose(
                    along, number(row[low]), rel_tol=1e-13, abs_tol=1e-13
                )
                end = along + (
                    bar.get_width() if panel == "xside" else bar.get_height()
                )
                assert math.isclose(
                    end, number(row[high]), rel_tol=1e-13, abs_tol=1e-13
                )
    # The point labels: the result's non-empty labels, in row order.
    texts = [t.get_text() for t in labels_of(fig)]
    assert texts == [label for label in result.point_labels if label]
    if case.args.get("label.var") is not None:
        assert texts
    assert ax.get_xlabel() == result.x
    assert ax.get_ylabel() == result.y
    assert figure_codes(fig) == [], fig.check()


def test_constant_x_is_refused_and_draws_no_line() -> None:
    """The analysis refuses a constant x (edge_corr_constant_x); a result with
    one, built by hand, draws the points and marginals but no line, and
    ``warnings`` says so."""
    for case in CASES:
        if "constant_x" in case.name:
            assert analyzed(case.name) is None
            assert REFUSALS[case.name].replay is None
    frame = pd.DataFrame({"x": [5.0] * 8, "y": [1.0, 3, 2, 5, 4, 6, 8, 7]})
    fig = ip.render.scatterstats(scatter_result(frame, "x", "y"))
    assert by_gid(fig, "smooth-line") is None
    assert by_gid(fig, "smooth-band") is None
    assert fig.result.warnings[-1].startswith("No least-squares line is drawn")
    assert numbered_count(fig, "xside-bar") == 1
    assert figure_codes(fig) == []


def test_labelled_fixtures() -> None:
    labels = figure("ggscatterstats-mtcars-labels-parametric")
    selected = figure("ggscatterstats-mtcars-labelexpr-parametric")
    assert len(labels_of(labels)) == 32
    assert len(labels_of(selected)) == 9
    # Each label lies next to its own point, or is joined to it by a line.
    for fig in (labels, selected):
        fig.draw()
        renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        ax = fig.axes["main"]
        links = ax.findobj(ConnectionPatch)
        near = 0
        for text in labels_of(fig):
            box = text.get_window_extent(renderer)
            x, y = text.get_position()
            px, py = ax.transData.transform((x, y))
            gap = max(box.x0 - px, px - box.x1, box.y0 - py, py - box.y1)
            near += gap < 8.0
        assert near + len(links) == len(labels_of(fig))


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def test_smooth_false_draws_no_line() -> None:
    fig = ip.render.scatterstats(mtcars(), smooth=False)
    assert by_gid(fig, "smooth-line") is None
    assert by_gid(fig, "smooth-band") is None
    assert "smooth_args styles the least-squares line, which smooth=False" in message(
        ValueError,
        lambda: ip.render.scatterstats(
            mtcars(), smooth=False, smooth_args={"color": "red"}
        ),
    )


def test_smooth_args_style_the_line_and_band_follows() -> None:
    fig = ip.render.scatterstats(
        mtcars(), smooth_args={"color": "#D55E00", "linewidth": 3}
    )
    line = by_gid(fig, "smooth-line")
    band = by_gid(fig, "smooth-band")
    assert isinstance(line, Line2D)
    assert isinstance(band, PolyCollection)
    assert line.get_linewidth() == 3
    face = np.asarray(band.get_facecolor(), dtype=np.float64)
    assert face[0, :3] == pytest.approx(to_rgba("#D55E00")[:3])
    assert message(
        ValueError, lambda: ip.render.scatterstats(mtcars(), smooth_args={"xdata": [1]})
    ).startswith("smooth_args sets 'xdata'")


def test_band_level_is_the_results() -> None:
    fig = ip.render.scatterstats(mtcars(conf_level=0.9))
    band = by_gid(fig, "smooth-band")
    assert isinstance(band, PolyCollection)
    frame = _mtcars()
    mine = ols_band(
        frame["wt"].to_numpy(dtype=np.float64),
        frame["mpg"].to_numpy(dtype=np.float64),
        conf_level=0.9,
    )
    vertices = np.asarray(band.get_paths()[0].vertices, dtype=np.float64)
    drawn = {(float(a), float(b)) for a, b in vertices}
    assert {
        (float(a), float(b)) for a, b in zip(mine.x, mine.conf_high, strict=True)
    } <= drawn


def test_marginal_false_and_marginal_args() -> None:
    fig = ip.render.scatterstats(mtcars(), marginal=False)
    assert set(fig.axes) == {"main"}
    assert "marginal_args styles the marginal histograms" in message(
        ValueError,
        lambda: ip.render.scatterstats(
            mtcars(), marginal=False, marginal_args={"alpha": 1}
        ),
    )
    fig = ip.render.scatterstats(mtcars(), marginal_args={"facecolor": "grey"})
    bar = by_gid(fig, "xside-bar-0")
    assert isinstance(bar, Rectangle)
    assert bar.get_facecolor() == to_rgba("grey")
    assert message(
        ValueError,
        lambda: ip.render.scatterstats(mtcars(), marginal_args={"height": 2}),
    ).startswith("marginal_args sets 'height'")


def test_ax_needs_marginal_false() -> None:
    figure_ = Figure()
    ax = figure_.add_subplot()
    assert "Pass marginal=False to draw into ax" in message(
        ValueError, lambda: ip.render.scatterstats(mtcars(), ax=ax)
    )
    assert not ax.has_data()


def test_point_args() -> None:
    fig = ip.render.scatterstats(mtcars(), point_args={"alpha": 1.0, "sizes": [40]})
    points = points_of(fig)
    assert points.get_alpha() == 1.0
    assert list(points.get_sizes()) == [40]
    assert "point_args={'sizes': [30]}" in message(
        TypeError, lambda: ip.render.scatterstats(mtcars(), point_args={"s": 30})
    )
    fig = ip.render.scatterstats(mtcars(), point_args={"alpha": 0})
    assert fig.result.warnings[-1].startswith(
        "point_args={alpha=0} makes the points invisible"
    )


def test_palette_colours_points_and_marginals() -> None:
    fig = ip.render.scatterstats(mtcars(), palette=["#009E73"])
    assert np.asarray(points_of(fig).get_facecolor())[0, :3] == pytest.approx(
        to_rgba("#009E73")[:3]
    )
    bar = by_gid(fig, "yside-bar-0")
    assert isinstance(bar, Rectangle)
    assert bar.get_facecolor()[:3] == pytest.approx(to_rgba("#009E73")[:3])


def test_user_texts_and_subtitle() -> None:
    fig = ip.render.scatterstats(
        mtcars(), title="Weight and economy", xlab="weight (1,000 lb)", ylab="mpg",
        caption="Motor Trend, 1974",
    )  # fmt: skip
    assert fig.axes["main"].get_xlabel() == "weight (1,000 lb)"
    assert figure_codes(fig) == []
    assert "subtitle= would replace the results subtitle" in message(
        ValueError, lambda: ip.render.scatterstats(mtcars(), subtitle="mine")
    )
    fig = ip.render.scatterstats(mtcars(results_subtitle=False), subtitle="mine")
    subtitle = by_gid(fig, "subtitle")
    assert isinstance(subtitle, Text)
    assert subtitle.get_text() == "mine"


def test_wrong_results() -> None:
    hist = ip.render.histostats  # noqa: F841 (the other no-level function)
    saved = StatsResult.from_json(mtcars().to_json(include_values=False))
    assert "holds no observations to draw" in message(
        ValueError, lambda: ip.render.scatterstats(saved)
    )


def test_labels_with_control_characters_are_drawn_replaced() -> None:
    frame = _mtcars().assign(car=["tab\there", *([""] * 31)])
    fig = ip.render.scatterstats(scatter_result(frame, "wt", "mpg", label_var="car"))
    (text,) = labels_of(fig)
    assert text.get_text() == "tab here"
    assert any("control characters" in w for w in fig.result.warnings)
    assert fig.result.point_labels[0] == "tab\there"


# --------------------------------------------------------------------------- #
# many points, many labels
# --------------------------------------------------------------------------- #


@functools.cache
def _many(n: int = 6_000, labels: int = 3) -> StatsResult:
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": 0.5 * x + rng.normal(0.0, 1.0, n),
            "name": [f"p{i}" if i < labels else "" for i in range(n)],
        }
    )
    return scatter_result(frame, "x", "y", label_var="name")


def test_many_points_are_subsampled_over_a_density_layer() -> None:
    result = _many()
    assert result.n_obs > MAX_POINTS
    fig = ip.render.scatterstats(result)
    drawn = np.asarray(points_of(fig).get_offsets())
    assert drawn.shape[0] == SUBSAMPLE_SIZE
    x, y = (np.asarray(v) for v in result.values)
    pairs = {(float(a), float(b)) for a, b in zip(x, y, strict=True)}
    assert {(float(a), float(b)) for a, b in drawn} <= pairs
    for index in (x.argmin(), x.argmax(), y.argmin(), y.argmax(), 0, 1, 2):
        assert (float(x[index]), float(y[index])) in {
            (float(a), float(b)) for a, b in drawn
        }
    assert isinstance(by_gid(fig, "density"), QuadMesh)
    (issue,) = [i for i in fig.check() if i.code == "points_subsampled"]
    assert issue.severity == "info"
    assert "2,000 of the 6,000 pairs" in issue.message
    assert any("subsample of 2,000 (seed 0)" in w for w in fig.result.warnings)
    # Deterministic: the same seed draws the same points.
    again = ip.render.scatterstats(result)
    assert np.array_equal(np.asarray(points_of(again).get_offsets()), drawn)


def test_too_many_labels_are_reported() -> None:
    result = _many(n=400, labels=MAX_LABELS + 50)
    fig = ip.render.scatterstats(result)
    assert len(labels_of(fig)) == MAX_LABELS + 50
    (issue,) = [i for i in fig.check() if i.code == "point_labels_crowded"]
    assert f"more than the {MAX_LABELS:,} inferplot places" in issue.message
    assert "label_var" in issue.fix


def test_crowded_labels_are_reported() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"x": rng.normal(size=60), "y": rng.normal(size=60)}).assign(
        name=[f"a rather long name {i}" for i in range(60)]
    )
    figure_ = Figure(figsize=(3.5, 3.0), dpi=100)
    FigureCanvasAgg(figure_)
    fig = ip.render.scatterstats(
        scatter_result(frame, "x", "y", label_var="name"),
        ax=figure_.add_subplot(),
        marginal=False,
    )
    (issue,) = [i for i in fig.check() if i.code == "point_labels_crowded"]
    assert "overlap" in issue.message or "lie under a point label" in issue.message
    # Reported once: overlaps of two point labels are not also label_overlap.
    for issue in fig.check():
        if issue.code != "label_overlap":
            continue
        pairs = issue.message.split(": ", 1)[1].split("; ")
        for pair in pairs:
            first, _, second = pair.partition(" and ")
            assert not (
                first.startswith("point label") and second.startswith("point label")
            )


# --------------------------------------------------------------------------- #
# ax=, checks
# --------------------------------------------------------------------------- #


def test_drawn_into_a_grid_without_resizing() -> None:
    figure_ = Figure(figsize=(12, 5), dpi=100)
    FigureCanvasAgg(figure_)
    a, b = figure_.subplots(1, 2, gridspec_kw={"wspace": 0.3})
    left = ip.render.scatterstats(mtcars(), ax=a, marginal=False)
    right = ip.render.scatterstats(
        result_of("ggscatterstats-iris-parametric"), ax=b, marginal=False
    )
    assert figure_.get_size_inches().tolist() == [12.0, 5.0]
    assert figure_codes(left) == []
    assert figure_codes(right) == []
    # A given Axes keeps its own grid.
    assert not any(line.get_visible() for line in a.get_xgridlines())


def test_dark_background_ink() -> None:
    figure_ = Figure(figsize=(6, 4.5), dpi=100, facecolor="black")
    FigureCanvasAgg(figure_)
    ax = figure_.add_subplot()
    ax.set_facecolor("#111111")
    fig = ip.render.scatterstats(
        result_of("ggscatterstats-mtcars-labelexpr-parametric"), ax=ax, marginal=False
    )
    for text in labels_of(fig):
        assert text.get_color() == "#EEEEEE"
    line = by_gid(fig, "smooth-line")
    assert isinstance(line, Line2D)
    assert line.get_color() == "#EEEEEE"
    assert "low_contrast" not in codes(fig)


def test_view_checks() -> None:
    fig = ip.render.scatterstats(mtcars())
    ax = fig.axes["main"]
    ax.set_xlim(2, 4)
    issue = next(i for i in fig.check() if i.code == "data_outside_view")
    assert "of 32 values" in issue.message
    assert "the least-squares line" in issue.message
    ax.set_xlim(-1e7, 1e7)
    assert "flat_data" in codes(fig)
    fig = ip.render.scatterstats(mtcars())
    fig.axes["xside"].set_ylim(0, 1)
    issue = next(i for i in fig.check() if i.code == "data_outside_view")
    assert "marginal histogram of x" in issue.message
    fig = ip.render.scatterstats(mtcars())
    fig.axes["main"].invert_yaxis()
    assert "axis_changed" in codes(fig)


def test_low_contrast_and_missing_glyphs() -> None:
    fig = ip.render.scatterstats(mtcars(), palette=["white"])
    issue = next(i for i in fig.check() if i.code == "low_contrast")
    assert "the colour of the points" in issue.message
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        fig = ip.render.scatterstats(mtcars(), title="重さ")
        found = codes(fig)
    assert "glyph_missing" in found


def test_checks_of_other_figures_do_not_apply() -> None:
    fig = figure("ggscatterstats-mtcars-labels-parametric")
    assert fig.paths is None
    assert fig.centers == ()
    assert fig.labels == ()
    assert fig.brackets == ()


# --------------------------------------------------------------------------- #
# the three forms, saving
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "type_name", ["parametric", "nonparametric", "robust", "bayes"]
)
def test_three_forms_agree(type_name: str, tmp_path: Any) -> None:
    """``ip.scatterstats`` draws what ``ip.render.scatterstats`` draws from the
    result of ``ip.analyze.scatterstats``, byte for byte."""
    frame = _mtcars()
    top = ip.scatterstats(
        frame, "wt", "mpg", type=type_name, label_var="car", title="t"
    )
    result = ip.analyze.scatterstats(
        frame, "wt", "mpg", type=type_name, label_var="car"
    )
    drawn = ip.render.scatterstats(result, title="t")
    assert top.result == drawn.result
    for suffix in ("png", "svg"):
        top.save(tmp_path / f"a.{suffix}")
        drawn.save(tmp_path / f"b.{suffix}")
        assert (tmp_path / f"a.{suffix}").read_bytes() == (
            tmp_path / f"b.{suffix}"
        ).read_bytes()


def test_saved_files(tmp_path: Any) -> None:
    fig = figure("ggscatterstats-mtcars-labelexpr-parametric")
    for suffix in ("svg", "pdf", "png"):
        fig.save(tmp_path / f"a.{suffix}")
        fig.save(tmp_path / f"b.{suffix}")
        assert (tmp_path / f"a.{suffix}").read_bytes() == (
            tmp_path / f"b.{suffix}"
        ).read_bytes()
    root = ET.parse(tmp_path / "a.svg").getroot()
    texts = [
        "".join(el.itertext()) for el in root.iter("{http://www.w3.org/2000/svg}text")
    ]
    for label in labels_of(fig):
        assert label.get_text() in texts
    embedded = StatsResult.from_json(svg_description(tmp_path / "a.svg"))
    assert embedded.values == ()
    assert embedded.point_labels == ()
    assert {"points", "smooth-line", "smooth-band", "xside-bar-0"} <= gids(fig)


def test_repr_names_the_pairs() -> None:
    assert repr(figure("ggscatterstats-mtcars-parametric")).startswith(
        "<StatsFigure scatterstats: mpg and wt, 32 pairs;"
    )
