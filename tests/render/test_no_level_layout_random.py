"""Random histostats and scatterstats data sets: nothing clipped or overlapping.

Each data set has a random size (5 to 400 values or pairs), scale (1e-3 to 1e4),
shape, column names of random length, test value (inside, near or far from the
data) and, for scatterstats, up to 20 point labels of random length. It is drawn
in inferplot's own figure and into a given Axes at 72, 100 and 200 dpi, sized from
the own figure times a random factor from 1.05 to 1.5 (with room for the margins
``add_subplot`` leaves). ``fig.check()`` must report neither ``label_overlap`` nor
``text_clipped``, and with no more labels than the plot has room for, no
``point_labels_crowded`` in inferplot's own figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from inferplot.render import StatsFigure, histostats, scatterstats
from inferplot.result import StatsResult
from tests.render._no_levels import hist_result, scatter_result

N_DATA_SETS = 60
QUICK = (0, 1, 2)
CODES = frozenset({"label_overlap", "text_clipped"})
TYPES = ("parametric", "nonparametric", "robust", "bayes")


def _name(rng: np.random.Generator, stem: str) -> str:
    return f"{stem}{'_x' * int(rng.integers(0, 12))}"


def _values(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    shape = int(rng.integers(0, 3))
    if shape == 0:
        return rng.normal(5.0, 1.0, n) * scale
    if shape == 1:
        return rng.lognormal(0.0, 0.8, n) * scale
    return np.round(rng.normal(3.0, 1.5, n)) * scale  # heavy ties


def _histogram(seed: int) -> StatsResult:
    rng = np.random.default_rng([20261110, seed])
    n = int(rng.integers(5, 400))
    scale = 10.0 ** rng.uniform(-3.0, 4.0)
    values = _values(rng, n, scale)
    name = _name(rng, "value")
    where = float(rng.choice([0.0, 0.5, 1.0, 3.0]))
    lo, hi = float(values.min()), float(values.max())
    test_value = lo + where * (hi - lo) if hi > lo else lo + 1.0
    return hist_result(
        pd.DataFrame({name: values}),
        name,
        type=str(rng.choice(TYPES)),
        test_value=float(f"{test_value:.3g}"),
    )


def _scatter(seed: int) -> StatsResult:
    rng = np.random.default_rng([20261111, seed])
    n = int(rng.integers(5, 400))
    scale = 10.0 ** rng.uniform(-3.0, 4.0)
    x = _values(rng, n, scale)
    y = 0.6 * x + _values(rng, n, scale)
    labelled = min(n, int(rng.integers(0, 21)))
    words = [
        f"{'item' * int(rng.integers(1, 4))} {i}" if i < labelled else ""
        for i in range(n)
    ]
    order = rng.permutation(n)
    xn, yn = _name(rng, "dose"), _name(rng, "response")
    frame = pd.DataFrame({xn: x, yn: y, "label": [words[i] for i in order]})
    return scatter_result(frame, xn, yn, type=str(rng.choice(TYPES)), label_var="label")


def _figures(
    make: object, result: StatsResult, seed: int
) -> list[tuple[str, StatsFigure]]:
    draw = histostats if make == "hist" else scatterstats
    own = draw(result)
    width, height = (float(v) for v in own.figure.get_size_inches())
    rng = np.random.default_rng([seed, 2])
    figures = [("own figure", own)]
    for dpi in (72, 100, 200):
        f = float(rng.uniform(1.05, 1.5))
        figure = Figure(figsize=(width * f / 0.775, height * f / 0.77), dpi=dpi)
        FigureCanvasAgg(figure)
        ax = figure.add_subplot()
        drawn = (
            histostats(result, ax=ax)
            if make == "hist"
            else scatterstats(result, ax=ax, marginal=False)
        )
        figures.append((f"ax= at {dpi} dpi", drawn))
    return figures


def _check(make: str, seed: int) -> None:
    result = _histogram(seed) if make == "hist" else _scatter(seed)
    for where, fig in _figures(make, result, seed):
        found = [i for i in fig.check() if i.code in CODES]
        assert found == [], (make, seed, where, found)
        if where == "own figure":
            crowded = [i for i in fig.check() if i.code == "point_labels_crowded"]
            assert crowded == [], (make, seed, crowded)


@pytest.mark.parametrize("make", ["hist", "scatter"])
@pytest.mark.parametrize("seed", QUICK)
def test_random_data_sets(make: str, seed: int) -> None:
    _check(make, seed)


@pytest.mark.slow
@pytest.mark.parametrize("make", ["hist", "scatter"])
@pytest.mark.parametrize("seed", [s for s in range(N_DATA_SETS) if s not in QUICK])
def test_random_data_sets_slow(make: str, seed: int) -> None:
    _check(make, seed)
