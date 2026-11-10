"""inferplot's own figure keeps its size, whatever the data.

The histogram once sized its figure to reach the test value: with the default
test value 0 for body temperatures (36.8 +/- 0.4) it was 80 to 100 in wide, for
Unix timestamps FreeType failed, and ``fig.check()`` said nothing (the edge
label was created at the data coordinate 0 and the layout measured it there). A
layout bug of this kind shows as a figure far larger than its plotting area, so
here every histostats and scatterstats figure over data sets from -1e6 to 1e9 in
location, 1e-6 to 1e6 in scale and 2 to 2,000 values, every type, and test
values at the default, inside and far beyond the data on both sides, must stay
within these bounds, and ``fig.check()`` must report ``figure_oversized`` when
a figure does not (here forced by resizing it).

Constant columns are left out: ggstatsplot's bin width for them is being
settled with a new oracle case (they raise for now).

The bounds hold with room for other systems' fonts: the largest measured over
300 data sets (macOS, matplotlib 3.10) were 6.5 x 4.7 in (histostats) and 7.6 x
5.7 in (scatterstats), with the plotting area filling at least 55% of the
figure.
"""

from __future__ import annotations

import itertools
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot.render import StatsFigure
from tests.render._no_levels import hist_result, scatter_result

# (width, height) in inches, most for each function.
MAX_SIZE_IN: Final = {"hist": (8.5, 6.0), "scatter": (9.0, 7.0)}
MIN_PLOT_SHARE: Final = 0.45

LOCATIONS: Final = (-1e6, -300.0, 0.0, 36.8, 1e4, 1.7e9)
SCALES: Final = (1e-6, 1e-3, 1.0, 1e3, 1e6)
SIZES: Final = (2, 3, 20, 200, 2_000)
TYPES: Final = ("parametric", "nonparametric", "robust", "bayes")
# The test value: the default 0, the location, and far beyond the data on
# either side (in units of the scale).
TEST_VALUES: Final = ("default", 0.0, 50.0, -50.0, 1e6, -1e6)

GRID: Final = list(itertools.product(LOCATIONS, SCALES, SIZES, TYPES, TEST_VALUES))
_PICK = np.random.default_rng(20261110).permutation(len(GRID))
QUICK: Final = [GRID[i] for i in _PICK[:10]]
SLOW: Final = [GRID[i] for i in _PICK[10:310]]


def plot_share(fig: StatsFigure) -> float:
    """The share of the figure's area the panels take (their positions are
    fractions of the figure)."""
    boxes = [ax.get_position() for ax in fig.axes.values()]
    x0, y0 = min(b.x0 for b in boxes), min(b.y0 for b in boxes)
    x1, y1 = max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    return float((x1 - x0) * (y1 - y0))


def assert_size(fig: StatsFigure, which: str, case: Any) -> None:
    width, height = (float(v) for v in fig.figure.get_size_inches())
    most_w, most_h = MAX_SIZE_IN[which]
    assert width <= most_w, (case, width, height)
    assert height <= most_h, (case, width, height)
    assert plot_share(fig) >= MIN_PLOT_SHARE, (case, plot_share(fig))
    assert "figure_oversized" not in [i.code for i in fig.check()], case


def _values(location: float, scale: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng([seed, n])
    return location + scale * rng.normal(0.0, 1.0, n)


def _histogram(case: tuple[float, float, int, str, object]) -> StatsFigure:
    location, scale, n, type_name, where = case
    x = _values(location, scale, n, 1)
    kwargs: dict[str, Any] = {"type": type_name}
    if where != "default":
        assert isinstance(where, float)
        kwargs["test_value"] = location + where * scale if where else location
    return ip.render.histostats(hist_result(pd.DataFrame({"x": x}), "x", **kwargs))


def _scatter(case: tuple[float, float, int, str, object]) -> StatsFigure:
    location, scale, n, type_name, _ = case
    n = max(n, 3)  # a correlation needs three pairs
    x = _values(location, scale, n, 2)
    y = location + 0.5 * (x - location) + _values(0.0, scale, n, 3)
    frame = pd.DataFrame({"x": x, "y": y})
    return ip.render.scatterstats(scatter_result(frame, "x", "y", type=type_name))


@pytest.mark.parametrize("case", QUICK, ids=str)
def test_histogram_size(case: Any) -> None:
    assert_size(_histogram(case), "hist", case)


@pytest.mark.parametrize("case", QUICK, ids=str)
def test_scatter_size(case: Any) -> None:
    assert_size(_scatter(case), "scatter", case)


@pytest.mark.slow
@pytest.mark.parametrize("case", SLOW, ids=str)
def test_histogram_size_many(case: Any) -> None:
    assert_size(_histogram(case), "hist", case)


@pytest.mark.slow
@pytest.mark.parametrize("case", SLOW[::2], ids=str)
def test_scatter_size_many(case: Any) -> None:
    assert_size(_scatter(case), "scatter", case)


@pytest.mark.parametrize(
    ("location", "scale"),
    [(36.8, 0.4), (170.0, 10.0), (-300.0, 10.0), (1e4, 0.5), (1.7e9, 3600.0)],
)
def test_default_test_value_far_from_the_data(location: float, scale: float) -> None:
    """The reviewer's cases, analyzed for real: body temperatures, heights,
    negative data, weights near 10,000 and Unix timestamps against the default
    test value 0, which the histogram names with an arrow at its edge."""
    x = location + scale * np.random.default_rng(21).normal(0.0, 1.0, 50)
    fig = ip.histostats(pd.DataFrame({"x": x}), "x")
    label = fig.axes["main"].texts[-1].get_text()
    assert "test value 0" in label
    assert label.startswith("←") or label.endswith("→")
    assert_size(fig, "hist", (location, scale))


def test_figure_oversized_reports_a_broken_layout() -> None:
    fig = ip.histostats(pd.DataFrame({"x": _values(36.8, 0.4, 50, 4)}), "x")
    assert "figure_oversized" not in [i.code for i in fig.check()]
    fig.figure.set_size_inches(80.0, 4.4)
    (issue,) = [i for i in fig.check() if i.code == "figure_oversized"]
    assert issue.severity == "warning"
    assert "80.0 x 4.4 in" in issue.message
    assert "layout went wrong" in issue.message
    assert "ax=" in issue.fix
    # A plotting area far smaller than its figure.
    fig = ip.scatterstats(
        pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 6.0], "y": [2.0, 1.0, 4.0, 3.0, 5.0]}),
        "x",
        "y",
    )
    width, height = (float(v) for v in fig.figure.get_size_inches())
    for ax in fig.axes.values():
        x0, y0, w, h = ax.get_position().bounds
        ax.set_position((x0 * 0.3, y0 * 0.3, w * 0.3, h * 0.3))
    (issue,) = [i for i in fig.check() if i.code == "figure_oversized"]
    assert f"{width:.1f} x {height:.1f} in" in issue.message
    assert "fills 5%" in issue.message or "fills 6%" in issue.message


def test_figure_oversized_names_a_long_text() -> None:
    caption = " ".join(f"word{i}" for i in range(80))
    fig = ip.histostats(
        pd.DataFrame({"x": _values(10.0, 2.0, 30, 5)}),
        "x",
        test_value=10,
        caption=caption,
    )
    (issue,) = [i for i in fig.check() if i.code == "figure_oversized"]
    assert "the figure grows to hold the caption line" in issue.message
    assert "Shorten those texts" in issue.fix


def test_figure_oversized_is_for_inferplot_own_figure_only() -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(60.0, 5.0), dpi=50)
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.45, 0.2, 0.1, 0.6))
    fig = ip.histostats(
        pd.DataFrame({"x": _values(10.0, 2.0, 30, 6)}), "x", test_value=10, ax=ax
    )
    assert "figure_oversized" not in [i.code for i in fig.check()]


LONG_LABEL: Final = " ".join(f"part{i}" for i in range(70))  # about 400 characters


def _drawn_with(function: str, **kwargs: Any) -> StatsFigure:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "g": ["a", "b"] * 15,
            "s": np.repeat(np.arange(15), 2),
            "x": rng.normal(10.0, 2.0, 30),
            "y": rng.normal(5.0, 1.0, 30),
        }
    )
    if function == "histostats":
        return ip.histostats(frame, "x", test_value=10, **kwargs)
    if function == "scatterstats":
        return ip.scatterstats(frame, "x", "y", **kwargs)
    if function == "betweenstats":
        return ip.betweenstats(frame, "g", "x", **kwargs)
    return ip.withinstats(frame, "g", "x", subject_id="s", **kwargs)


FUNCTIONS: Final = ("histostats", "scatterstats", "betweenstats", "withinstats")


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("which", ["xlab", "ylab"])
def test_long_axis_label_wraps(function: str, which: str) -> None:
    """A 400-character axis label wraps on its spaces to the plotting area's
    side instead of widening the figure to its length (it made a histogram 36 in
    wide); the words are all there, in order."""
    plain = _drawn_with(function)
    fig = _drawn_with(function, **{which: LONG_LABEL})
    ax = fig.axes["main"]
    label = ax.xaxis.label if which == "xlab" else ax.yaxis.label
    shown = label.get_text()
    assert "\n" in shown
    assert shown.split() == LONG_LABEL.split()
    fig.draw()
    box = label.get_window_extent()
    side = ax.get_window_extent()
    along = box.width if which == "xlab" else box.height
    assert along <= (side.width if which == "xlab" else side.height) + 1.0
    width, height = (float(v) for v in fig.figure.get_size_inches())
    w0, h0 = (float(v) for v in plain.figure.get_size_inches())
    assert width <= w0 + (3.0 if which == "ylab" else 0.5)
    assert height <= h0 + (3.0 if which == "xlab" else 0.5)
    codes = {"text_clipped", "label_overlap", "figure_oversized"}
    assert [i.code for i in fig.check() if i.code in codes] == []


@pytest.mark.parametrize("function", FUNCTIONS)
def test_short_axis_labels_are_drawn_as_given(function: str) -> None:
    fig = _drawn_with(function, xlab="dose (mg) per $m^2$", ylab="weight gain (g)")
    ax = fig.axes["main"]
    assert ax.get_xlabel() == "dose (mg) per $m^2$"
    assert ax.get_ylabel() == "weight gain (g)"


def test_wrapping_keeps_mathtext_whole() -> None:
    from inferplot.render.base import wrap_words

    def fits(line: str) -> bool:
        return len(line) <= 12

    assert wrap_words("mass in $\\mu g \\cdot m^{-2}$ per day", fits) == (
        "mass in\n$\\mu g \\cdot m^{-2}$\nper day"
    )
    assert wrap_words("short", fits) == "short"
    assert wrap_words("a verylongwordindeed b", fits) == "a\nverylongwordindeed\nb"
