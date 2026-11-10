"""The review's low items for histostats and scatterstats (Y7, L1 to L8).

L1 the percent axis above 100% and the count axis above n; L2 save() at a dpi
FreeType cannot draw; L4 one message for any bin width that gives too many bars;
L5 a layer made invisible by its colours; L6 whitespace-only point labels; L7
the alt text naming a layer the caller made invisible; L8 wording (``c=`` in
scatterstats, an empty or long palette, the render function's name, a log or
inverted axis on a given Axes, ``subtitle=`` in the one-call form). L3 (the test
value's digits) is in test_histostats_line_labels.py.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import inferplot as ip
from inferplot.render import StatsFigure

rng = np.random.default_rng(3)
FRAME = pd.DataFrame({"x": rng.normal(10.0, 2.0, 30), "y": rng.normal(5.0, 1.0, 30)})


def _message(error: type[Exception], call: Any) -> str:
    with pytest.raises(error) as caught:
        call()
    return str(caught.value)


def _tick_values(fig: StatsFigure) -> tuple[list[float], list[str]]:
    fig.draw()
    ax = fig.axes["main"]
    low, high = ax.get_ylim()
    counts = [float(t) for t in ax.get_yticks() if low <= t <= high]
    (percent,) = ax.child_axes
    shares = [t.get_text() for t in percent.get_yticklabels() if t.get_text()]
    return counts, shares


# --- L1 ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 5, 12])
def test_axes_stop_at_n_and_100_percent(n: int) -> None:
    # Every value in one bar: the tallest bar is n, 100%.
    values = pd.DataFrame({"x": [3.0] * (n - 1) + [3.001]})
    fig = ip.histostats(values, "x", test_value=3, binwidth=1.0)
    counts, shares = _tick_values(fig)
    assert counts
    assert max(counts) <= n
    assert all(float(s.rstrip("%")) <= 100.0 for s in shares), shares


def test_shared_count_axis_says_whose_percentages() -> None:
    figure = Figure(figsize=(12.0, 5.0), dpi=100)
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2, sharey=True, gridspec_kw={"wspace": 0.5})
    ip.histostats(FRAME, "x", test_value=10, ax=left)
    fig = ip.histostats(FRAME.head(20), "y", test_value=5, ax=right)
    (percent,) = fig.axes["main"].child_axes
    assert percent.get_ylabel() == "percent of the 20 values"
    own = ip.histostats(FRAME, "x", test_value=10)
    (percent,) = own.axes["main"].child_axes
    assert percent.get_ylabel() == "percent of values"


# --- L2 ----------------------------------------------------------------------


@pytest.mark.parametrize("dpi", [0.5, 1, 6, 9.9])
def test_png_below_ten_dpi_is_refused(dpi: float, tmp_path: Path) -> None:
    fig = ip.histostats(FRAME, "x", test_value=10)
    text = _message(ValueError, lambda: fig.save(tmp_path / "f.png", dpi=dpi))
    assert text.startswith(f"dpi={dpi!r} is less than 10, the least save() writes")
    assert not (tmp_path / "f.png").exists()
    fig.save(tmp_path / "f.svg", dpi=dpi)  # vector formats do not depend on dpi
    fig.save(tmp_path / "f.png", dpi=10)


# --- L4 ----------------------------------------------------------------------


@pytest.mark.parametrize("binwidth", [0.0001, 0.001, 1e-9])
def test_one_message_for_too_many_bars(binwidth: float) -> None:
    for call in (
        lambda: ip.histostats(FRAME, "x", binwidth=binwidth),
        lambda: ip.render.histostats(
            ip.analyze.histostats(FRAME, "x"), binwidth=binwidth
        ),
    ):
        text = _message(ValueError, call)
        assert text.startswith(f"binwidth={binwidth!r} gives at least ")
        assert "bars, more than the 10,000 inferplot draws" in text
        assert "(100 bars)" in text
        assert "ggplot2" not in text


# --- L5 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("draw", "args", "noun"),
    [
        ("histostats", {"bar_args": {"color": "white"}}, "bars"),
        ("scatterstats", {"point_args": {"color": "white"}}, "points"),
        ("scatterstats", {"smooth_args": {"color": "#fefefe"}}, "least-squares line"),
        ("scatterstats", {"marginal_args": {"facecolor": "white"}}, "marginal"),
    ],
)
def test_white_on_white_is_invisible(draw: str, args: Any, noun: str) -> None:
    if draw == "histostats":
        fig = ip.histostats(FRAME, "x", test_value=10, **args)
    else:
        fig = ip.scatterstats(FRAME, "x", "y", **args)
    (warning,) = [w for w in fig.result.warnings if "invisible" in w]
    assert f"makes the {noun}" in warning
    assert "on the background #ffffff" in warning


def test_colour_is_judged_on_the_given_axes() -> None:
    figure = Figure(figsize=(6.0, 4.5), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    ax.set_facecolor("black")
    fig = ip.histostats(FRAME, "x", test_value=10, bar_args={"color": "white"}, ax=ax)
    assert not [w for w in fig.result.warnings if "invisible" in w]
    figure = Figure(figsize=(6.0, 4.5), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    ax.set_facecolor("black")
    fig = ip.histostats(FRAME, "x", test_value=10, bar_args={"color": "black"}, ax=ax)
    (warning,) = [w for w in fig.result.warnings if "invisible" in w]
    assert "on the background #000000" in warning


# --- L6 and L7 ---------------------------------------------------------------


def test_whitespace_labels_are_no_labels() -> None:
    labels = [" ", "\t", "  "] + [None] * 27
    fig = ip.scatterstats(FRAME.assign(lab=labels), "x", "y", label_var="lab")
    assert fig.point_labels is None
    assert "labels from" not in fig.result.text.alt_text
    assert not [t for t in fig.axes["main"].texts if t.get_text().strip() == ""]


@pytest.mark.parametrize(
    ("args", "absent"),
    [
        ({"smooth_args": {"alpha": 0}}, "least-squares line"),
        ({"marginal_args": {"alpha": 0}}, "marginal histograms"),
        ({"smooth_args": {"color": "white"}}, "least-squares line"),
    ],
)
def test_alt_text_leaves_out_invisible_layers(args: Any, absent: str) -> None:
    fig = ip.scatterstats(FRAME, "x", "y", **args)
    assert absent not in fig.result.text.alt_text
    plain = ip.scatterstats(FRAME, "x", "y")
    assert absent in plain.result.text.alt_text


def test_alt_text_of_invisible_bars() -> None:
    fig = ip.histostats(FRAME, "x", test_value=10, bar_args={"alpha": 0})
    assert " bins)" not in fig.result.text.alt_text
    plain = ip.histostats(FRAME, "x", test_value=10)
    assert " bins)" in plain.result.text.alt_text


# --- L8 ----------------------------------------------------------------------


def test_scatter_c_speaks_of_points_not_levels() -> None:
    text = _message(
        TypeError, lambda: ip.scatterstats(FRAME, "x", "y", point_args={"c": "red"})
    )
    assert "point_args={'color': 'black'}" in text
    assert "level" not in text
    between = _message(
        TypeError,
        lambda: ip.betweenstats(
            FRAME.assign(g=["a", "b"] * 15), "g", "y", point_args={"c": "red"}
        ),
    )
    assert "by level" in between


def test_palettes_of_one_colour_figures() -> None:
    for call in (
        lambda: ip.scatterstats(FRAME, "x", "y", palette=[]),
        lambda: ip.histostats(FRAME, "x", palette=[]),
    ):
        text = _message(ValueError, call)
        assert text.startswith("palette is empty; pass a list with at least one")
        assert "level" not in text
    fig = ip.scatterstats(FRAME, "x", "y", palette=["red", "blue", "green"])
    (warning,) = [w for w in fig.result.warnings if w.startswith("palette")]
    assert warning == (
        "palette has 3 colors; the figure uses only the first, for the points and "
        "the marginal histograms, and not the other 2."
    )
    fig = ip.histostats(FRAME, "x", test_value=10, palette=["red"])
    assert not [w for w in fig.result.warnings if w.startswith("palette")]


def test_render_functions_name_themselves() -> None:
    result = ip.analyze.histostats(FRAME, "x")
    # Calls that are wrong on purpose, through Any so the type checker lets them.
    render: Any = ip.render
    call: Any = ip.histostats
    text = _message(TypeError, lambda: render.histostats(result, layers=("box",)))
    assert text.startswith("render.histostats() got an unexpected keyword argument")
    text = _message(TypeError, lambda: render.histostats(result, 3))
    assert text.startswith("render.histostats() takes 1 positional argument")
    text = _message(TypeError, lambda: call(FRAME, "x", layers=("box",)))
    assert text.startswith("histostats() got an unexpected keyword argument")


@pytest.mark.parametrize("change", ["xlog", "ylog", "xinv", "yinv"])
@pytest.mark.parametrize("draw", ["histostats", "scatterstats"])
def test_a_changed_axis_is_refused_before_drawing(draw: str, change: str) -> None:
    figure = Figure(figsize=(6.0, 4.5), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    {
        "xlog": lambda: ax.set_xscale("log"),
        "ylog": lambda: ax.set_yscale("log"),
        "xinv": ax.invert_xaxis,
        "yinv": ax.invert_yaxis,
    }[change]()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if draw == "histostats":
            text = _message(ValueError, lambda: ip.histostats(FRAME, "x", ax=ax))
        else:
            text = _message(
                ValueError,
                lambda: ip.scatterstats(FRAME, "x", "y", ax=ax, marginal=False),
            )
        assert [str(w.message) for w in caught] == []
    letter = change[0]
    if change.endswith("log"):
        assert f"ax has a 'log' {letter} scale" in text
        assert f"ax.set_{letter}scale('linear')" in text
    else:
        assert f"ax has an inverted {letter} axis" in text
        assert f"ax.invert_{letter}axis()" in text
    assert not ax.has_data()  # nothing was drawn


@pytest.mark.parametrize("draw", ["histostats", "scatterstats"])
def test_subtitle_in_the_one_call_form(draw: str) -> None:
    args = ("x",) if draw == "histostats" else ("x", "y")
    text = _message(
        ValueError, lambda: getattr(ip, draw)(FRAME, *args, subtitle="mine")
    )
    assert "pass results_subtitle=False in this call as well" in text
    fig = getattr(ip, draw)(FRAME, *args, subtitle="mine", results_subtitle=False)
    assert any(t.get_text() == "mine" for t in fig.figure.texts)
