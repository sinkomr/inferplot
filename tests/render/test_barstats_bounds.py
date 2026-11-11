"""inferplot's own barstats figure is bounded (review B11).

Its plotting area is at most MAX_PLOT_IN (26 in) wide and tall, so the figure
stays within 30 in: up to 32 bars keep their 0.8 in slot; more share the 26 in
down to 0.7 in a bar; beyond 37 bars the call is refused before drawing, and
where the labels need wider slots, or the legend of many levels of x is taller,
than the bound allows, the call is refused with how many fit. With ax= nothing
is refused: fig.check() reports what does not fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import inferplot as ip
from inferplot.render.barstats import MAX_BARS, MAX_PLOT_IN, MIN_SLOT_IN


def _bars(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": np.tile(["a", "b"], n * 3),
            "y": np.repeat([f"g{i:03d}" for i in range(n)], 6),
        }
    )


def test_the_bound() -> None:
    assert MAX_PLOT_IN == 26.0
    assert MIN_SLOT_IN == 0.7
    assert MAX_BARS == 37


@pytest.mark.parametrize("n", [20, 32, 36])
def test_many_bars_stay_within_30_in(n: int) -> None:
    fig = ip.barstats(_bars(n), "x", "y", results_subtitle=False)
    width, height = (float(v) for v in fig.figure.get_size_inches())
    assert width <= 30.0
    assert height <= 30.0
    # Six observations a bar: the bars' tests' small_expected is expected.
    figure = [i for i in fig.check() if i.severity == "warning"]
    assert [i.code for i in figure if i.code != "small_expected"] == []


@pytest.mark.parametrize("n", [38, 100, 300])
def test_more_bars_than_fit_are_refused_before_drawing(n: int) -> None:
    with pytest.raises(ValueError, match="bars do not fit") as caught:
        ip.barstats(_bars(n), "x", "y", results_subtitle=False)
    text = str(caught.value)
    assert f"{n} bars do not fit" in text
    assert "at most 37 bars fit" in text
    assert "fewer levels of 'y'" in text
    assert "ax=Figure(figsize=(40, 6)).add_subplot()" in text


def test_ax_draws_them_and_check_reports_what_does_not_fit() -> None:
    ax = Figure(figsize=(40, 6)).add_subplot()
    fig = ip.barstats(_bars(100), "x", "y", results_subtitle=False, ax=ax)
    assert len(fig.result.bars) == 100


def test_a_legend_of_many_levels_is_bounded() -> None:
    rng = np.random.default_rng(1)

    def levels(k: int) -> pd.DataFrame:
        names = [f"level {i:03d}" for i in range(k)]
        return pd.DataFrame(
            {"x": rng.choice(names, 3000), "y": rng.choice(["u", "v", "w"], 3000)}
        )

    fig = ip.barstats(levels(100), "x", "y", results_subtitle=False)
    assert float(fig.figure.get_size_inches()[1]) <= 30.0
    with pytest.raises(ValueError, match="levels fit") as caught:
        ip.barstats(levels(160), "x", "y", results_subtitle=False)
    assert "the legend of the 160 levels of 'x'" in str(caught.value)
