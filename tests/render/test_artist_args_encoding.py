"""Artist dicts may style a layer but never change what the chart encodes.

One principle for every artist dict (``bar_args``, ``point_args``,
``boxplot_args``, ``violin_args``, ``smooth_args``, ``marginal_args``): a
property that would change what the figure encodes is refused before anything
is drawn, with a message that names the argument that does it properly or says
there is none. What a figure encodes: the drawing order that keeps the labels
above the layers readable (``zorder``, every dict); a mark's position, size or
shape (``angle`` of a rectangle, ``drawstyle`` of the
least-squares line); and in barstats, where colour is the only key to the levels
of x, each level's colour (``color``, ``facecolor``, ``fill``), an edge so wide
it covers the segments and the legend keys (``linewidth`` above
``MAX_SEGMENT_EDGE_PT``) and a transparency that fades the colours toward the
background (``alpha`` below ``MIN_SEGMENT_ALPHA``). Properties that only style
are accepted. ``fig.check()`` reports two levels of barstats drawn in colours
that cannot be told apart (``level_colors_alike``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot.render import StatsFigure

TABLE = pd.DataFrame(
    {
        "x": pd.Categorical(["a", "b", "c"] * 3, categories=["a", "b", "c"]),
        "y": pd.Categorical(["u"] * 3 + ["v"] * 3 + ["w"] * 3),
        "n": [30, 15, 10, 12, 25, 18, 20, 10, 30],
    }
)


def bars(**kwargs: Any) -> StatsFigure:
    return ip.barstats(TABLE, "x", "y", counts="n", bf_message=False, **kwargs)


def refusal(call: Any) -> str:
    with pytest.raises(ValueError, match="sets") as caught:
        call()
    return str(caught.value)


# -- barstats -------------------------------------------------------------------


@pytest.mark.parametrize("prop", ["color", "facecolor", "fc", "fill"])
def test_bar_args_may_not_colour_every_level_alike(prop: str) -> None:
    value: object = False if prop == "fill" else "red"
    text = refusal(lambda: bars(bar_args={prop: value}))
    assert f"bar_args sets {prop!r}" in text
    assert "palette=" in text
    assert "legend" in text


def test_bar_args_may_not_hide_the_cell_labels_by_drawing_order() -> None:
    text = refusal(lambda: bars(bar_args={"zorder": 10}))
    assert "bar_args sets 'zorder'" in text
    assert "labels" in text


def test_bar_args_may_not_rotate_the_segments() -> None:
    assert "bar_args sets 'angle'" in refusal(lambda: bars(bar_args={"angle": 30}))


def test_bar_args_edge_width_is_bounded() -> None:
    text = refusal(lambda: bars(bar_args={"linewidth": 25}))
    assert "linewidth" in text
    assert "2 points" in text
    assert "lw" in refusal(lambda: bars(bar_args={"lw": 2.5, "ec": "black"}))
    fig = bars(bar_args={"linewidth": 2, "edgecolor": "black"})
    assert not [i for i in fig.check() if i.severity == "warning"]


def test_bar_args_alpha_has_a_floor() -> None:
    text = refusal(lambda: bars(bar_args={"alpha": 0.02}))
    assert "alpha" in text
    assert "0.4" in text
    assert "alpha" in refusal(lambda: bars(bar_args={"alpha": 0}))
    fig = bars(bar_args={"alpha": 0.4})
    assert not [i for i in fig.check() if i.severity == "warning"]


@pytest.mark.parametrize(
    "args",
    [
        {"edgecolor": "black"},
        {"hatch": "xx"},
        {"linestyle": "--", "edgecolor": "black"},
        {"alpha": 0.8},
        {"linewidth": 1.5},
        {"rasterized": True},
        {"label": "mine"},
    ],
)
def test_bar_args_that_only_style_are_accepted(args: dict[str, object]) -> None:
    fig = bars(bar_args=args)
    assert not [i for i in fig.check() if i.severity == "warning"]


@pytest.mark.parametrize("keyword", ["color", "colors", "colour"])
def test_the_colour_keyword_points_to_palette_only(keyword: str) -> None:
    with pytest.raises(TypeError) as caught:
        bars(**{keyword: "red"})
    text = str(caught.value)
    assert "palette=" in text
    assert "or bar_args=" not in text


# -- colours that cannot be told apart ----------------------------------------------


@pytest.mark.parametrize(
    "palette",
    [
        ["red", "red", "red"],
        ["#777777", "#787878", "#797979"],
        ["#1f77b4", "#1f77b5", "#d62728"],
    ],
)
def test_levels_drawn_in_colours_that_cannot_be_told_apart(palette: list[str]) -> None:
    fig = bars(palette=palette)
    (issue,) = [i for i in fig.check() if i.code == "level_colors_alike"]
    assert issue.severity == "warning"
    assert "'a' and 'b'" in issue.message
    assert "palette=" in issue.fix


@pytest.mark.parametrize("palette", [None, "viridis", "Set2", ["red", "green", "blue"]])
def test_distinct_colours_report_nothing(palette: object) -> None:
    fig = bars(palette=palette)
    assert "level_colors_alike" not in [i.code for i in fig.check()]


def test_colours_changed_after_drawing_are_measured() -> None:
    fig = bars()
    ax = fig.axes["main"]
    for patch in ax.patches:
        if str(patch.get_gid()).startswith("segment-"):
            patch.set_facecolor("grey")
    assert "level_colors_alike" in [i.code for i in fig.check()]


# -- the other figures' dicts --------------------------------------------------------


def _numbers() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "g": np.repeat(["a", "b", "c"], 10),
            "v": rng.normal(size=30),
            "w": rng.normal(size=30),
            "id": np.tile(np.arange(10), 3),
        }
    )


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda a: ip.histostats(_numbers(), "v", bar_args=a), "bar_args"),
        (lambda a: ip.scatterstats(_numbers(), "v", "w", point_args=a), "point_args"),
        (lambda a: ip.scatterstats(_numbers(), "v", "w", smooth_args=a), "smooth_args"),
        (
            lambda a: ip.scatterstats(_numbers(), "v", "w", marginal_args=a),
            "marginal_args",
        ),
        (lambda a: ip.betweenstats(_numbers(), "g", "v", point_args=a), "point_args"),
        (
            lambda a: ip.betweenstats(_numbers(), "g", "v", boxplot_args=a),
            "boxplot_args",
        ),
        (
            lambda a: ip.betweenstats(_numbers(), "g", "v", violin_args=a),
            "violin_args",
        ),
        (
            lambda a: ip.withinstats(
                _numbers(), "g", "v", subject_id="id", point_args=a
            ),
            "point_args",
        ),
    ],
)
def test_no_artist_dict_changes_the_drawing_order(call: Any, name: str) -> None:
    text = refusal(lambda: call({"zorder": 10}))
    assert f"{name} sets 'zorder'" in text


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda a: ip.histostats(_numbers(), "v", bar_args=a), "bar_args"),
        (
            lambda a: ip.scatterstats(_numbers(), "v", "w", marginal_args=a),
            "marginal_args",
        ),
    ],
)
def test_no_rectangle_is_rotated(call: Any, name: str) -> None:
    assert f"{name} sets 'angle'" in refusal(lambda: call({"angle": 30}))


def test_the_least_squares_line_keeps_its_shape() -> None:
    steps = {"drawstyle": "steps"}
    text = refusal(lambda: ip.scatterstats(_numbers(), "v", "w", smooth_args=steps))
    assert "smooth_args sets 'drawstyle'" in text


def test_colour_stays_a_style_where_the_axis_names_the_levels() -> None:
    # betweenstats' x axis names each level, so one colour for all points hides
    # nothing; histostats' bars have no levels.
    fig = ip.betweenstats(_numbers(), "g", "v", point_args={"color": "black"})
    assert not [i for i in fig.check() if i.severity == "warning"]
    fig = ip.histostats(_numbers(), "v", bar_args={"color": "grey"})
    assert not [i for i in fig.check() if i.severity == "warning"]
