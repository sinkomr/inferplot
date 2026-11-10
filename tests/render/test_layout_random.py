"""Random data sets with every pair drawn: nothing clipped, overlapping or crossed.

Each data set has 3 to 8 groups of 4 to 39 values at a random scale and level
names of random length, with ``pairwise_display="all"``. It is drawn in
inferplot's own figure and into a given Axes at 72, 100 and 200 dpi, sized from
the own figure times a random factor from 1.05 to 1.5 (with room for the margins
``add_subplot`` leaves). ``fig.check()`` must report neither ``label_overlap`` nor
``text_clipped``, and the dense sampling of the drawn bracket lines
(``stroke_reach``, independent of ``check()``) must find no line inside another
bracket's label box and at least a pixel of clearance.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from inferplot.render import betweenstats
from inferplot.render.comparison import MIN_DATA_HEIGHT_IN
from inferplot.result import StatsResult
from tests.render._helpers import figure_codes, stroke_reach
from tests.render._results import frame_result

N_DATA_SETS = 120
# Run without the slow marker: 3, 4 and 5 groups, under 5 s each. The others
# (up to 28 brackets, drawn four times) take up to 10 s.
QUICK = (0, 3, 12)
MIN_CLEARANCE_PX = 1.0


def _random_result(seed: int) -> StatsResult:
    rng = np.random.default_rng([20261108, seed])
    k = int(rng.integers(3, 9))
    scale = 10.0 ** rng.uniform(-3.0, 4.0)
    names = [f"grp{'x' * int(rng.integers(0, 10))}{i}" for i in range(k)]
    groups = [
        rng.normal(
            rng.normal(0.0, 1.5) * scale,
            rng.uniform(0.3, 2.0) * scale,
            int(rng.integers(4, 40)),
        )
        for _ in range(k)
    ]
    return frame_result(names, groups, pairwise_display="all")


def _check_drawn(seed: int) -> None:
    result = _random_result(seed)
    own = betweenstats(result)
    width, height = (float(v) for v in own.figure.get_size_inches())
    assert own.axes["main"].bbox.height >= MIN_DATA_HEIGHT_IN * own.figure.dpi
    rng = np.random.default_rng([seed, 1])
    figures = [("own figure", own)]
    for dpi in (72, 100, 200):
        f = float(rng.uniform(1.05, 1.5))
        figure = Figure(figsize=(width * f / 0.775, height * f / 0.77), dpi=dpi)
        FigureCanvasAgg(figure)
        figures.append(
            (f"ax= at {dpi} dpi", betweenstats(result, ax=figure.add_subplot()))
        )
    for where, fig in figures:
        assert figure_codes(fig) == [], (seed, where, fig.check())
        deepest, clearance = stroke_reach(fig)
        assert deepest == 0.0, (seed, where)
        assert clearance >= MIN_CLEARANCE_PX, (seed, where, clearance)


@pytest.mark.parametrize("seed", QUICK)
def test_random_data_sets(seed: int) -> None:
    _check_drawn(seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [s for s in range(N_DATA_SETS) if s not in QUICK])
def test_random_data_sets_slow(seed: int) -> None:
    _check_drawn(seed)
