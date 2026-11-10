"""inferplot's own figure draws no axis offset text; with ax= the check sees it.

A review found the y axis offset text ("1e12", "+1e7") of scatterstats drawn on
the top marginal histogram. In inferplot's own figure (all four functions) an
offset or multiplier is now hidden and follows the axis label in parentheses,
as matplotlib prints it ("y (1e12)", "x (+1.7e9)"); in an Axes of the user's it
stays where matplotlib puts it, and ``fig.check()`` counts it among the texts
that may overlap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import inferplot as ip
from inferplot.render import StatsFigure

rng = np.random.default_rng(1)
X = rng.normal(0.0, 1.0, 40)
NOISE = rng.normal(0.0, 1.0, 40)
MINUS = "\N{MINUS SIGN}"  # as matplotlib prints negative exponents


def _visible_offsets(fig: StatsFigure) -> list[str]:
    fig.draw()
    texts = []
    for ax in fig.figure.axes:
        for axis in (ax.xaxis, ax.yaxis):
            text = axis.get_offset_text()
            if text.get_visible() and text.get_text():
                texts.append(text.get_text())
    return texts


@pytest.mark.parametrize(
    ("y", "suffix"),
    [
        (NOISE * 1e12, "(1e12)"),
        (1e7 + NOISE, "(+1e7)"),
        (NOISE * 1e-9, f"(1e{MINUS}9)"),
        (1e6 * NOISE, "(1e6)"),
    ],
)
def test_scatterstats_folds_the_offset_into_the_label(y: Any, suffix: str) -> None:
    fig = ip.scatterstats(pd.DataFrame({"dose": X, "response": y}), "dose", "response")
    assert _visible_offsets(fig) == []
    assert fig.axes["main"].get_ylabel() == f"response {suffix}"
    assert fig.axes["main"].get_xlabel() == "dose"
    codes = {"label_overlap", "text_clipped"}
    assert [i.code for i in fig.check() if i.code in codes] == []


def test_histostats_x_axis() -> None:
    fig = ip.histostats(pd.DataFrame({"t": 1.7e9 + 3600.0 * NOISE}), "t")
    assert _visible_offsets(fig) == []
    assert fig.axes["main"].get_xlabel() == "t (+1.7e9)"


def test_comparison_figures() -> None:
    frame = pd.DataFrame({"g": ["a", "b"] * 20, "y": 1e12 * (3.0 + NOISE)})
    for fig in (
        ip.betweenstats(frame, "g", "y"),
        ip.withinstats(
            frame.assign(s=np.repeat(np.arange(20), 2)), "g", "y", subject_id="s"
        ),
    ):
        assert _visible_offsets(fig) == []
        assert fig.axes["main"].get_ylabel() == "y (1e12)"


def test_ordinary_labels_are_unchanged() -> None:
    fig = ip.scatterstats(
        pd.DataFrame({"dose": X, "response": NOISE}), "dose", "response"
    )
    assert fig.axes["main"].get_ylabel() == "response"
    assert fig.axes["main"].get_xlabel() == "dose"


def test_given_axes_keep_their_offset_and_check_it() -> None:
    # Two panels close together: the offset text above the lower one reaches
    # the tick labels of the upper one.
    figure = Figure(figsize=(6.0, 8.0), dpi=100)
    FigureCanvasAgg(figure)
    upper, lower = figure.subplots(2, 1, gridspec_kw={"hspace": 0.08})
    upper.plot([0.0, 1.0], [0.0, 1.0])
    fig = ip.scatterstats(
        pd.DataFrame({"dose": X, "response": NOISE * 1e12}),
        "dose",
        "response",
        ax=lower,
        marginal=False,
        results_subtitle=False,
    )
    assert _visible_offsets(fig) == ["1e12"]
    assert lower.get_ylabel() == "response"
    (issue,) = [i for i in fig.check() if i.code == "label_overlap"]
    assert "y-axis offset text '1e12' and x tick label" in issue.message
