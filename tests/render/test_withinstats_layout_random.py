"""Random within data sets: no text clipped, overlapping or drawn over a path.

Each data set has 2 to 6 conditions of 4 to 60 subjects (a subject effect plus
noise) at a random scale, level names of random length and every pair drawn. It is
drawn in inferplot's own figure and into a given Axes at 72, 100 and 200 dpi, sized
as in test_layout_random. ``fig.check()`` must report none of ``label_overlap``,
``text_clipped``, ``path_hidden`` and ``paths_crowded``; the dense sampling of the
drawn paths (``path_reach``) and bracket lines (``stroke_reach``), independent of
``check()``, must find no path or bracket line inside a text's box.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from inferplot.render import withinstats
from inferplot.result import StatsResult
from tests.render._helpers import stroke_reach
from tests.render._within import within_result
from tests.render.test_withinstats_render import FIGURE_CODES, path_reach

N_DATA_SETS = 40
# Run without the slow marker: 2, 3 and 4 conditions. Each data set takes under
# 3 s; the other 37 carry the marker to keep the default run short.
QUICK = (0, 1, 2)


def _random_result(seed: int) -> StatsResult:
    rng = np.random.default_rng([20261109, seed])
    k = 2 + seed % 5
    n = int(rng.integers(4, 61))
    scale = 10.0 ** rng.uniform(-3.0, 4.0)
    names = [f"cond{'x' * int(rng.integers(0, 10))}{i}" for i in range(k)]
    means = rng.normal(0.0, 1.5, size=k) * scale
    subject = rng.normal(0.0, rng.uniform(0.0, 2.0) * scale, size=(n, 1))
    noise = rng.normal(0.0, rng.uniform(0.3, 2.0) * scale, size=(n, k))
    return within_result(
        names, means[None, :] + subject + noise, pairwise_display="all"
    )


def _check_drawn(seed: int) -> None:
    result = _random_result(seed)
    own = withinstats(result)
    width, height = (float(v) for v in own.figure.get_size_inches())
    rng = np.random.default_rng([seed, 1])
    figures = [("own figure", own)]
    for dpi in (72, 100, 200):
        f = float(rng.uniform(1.05, 1.5))
        figure = Figure(figsize=(width * f / 0.775, height * f / 0.77), dpi=dpi)
        FigureCanvasAgg(figure)
        figures.append(
            (f"ax= at {dpi} dpi", withinstats(result, ax=figure.add_subplot()))
        )
    for where, fig in figures:
        issues = [i for i in fig.check() if i.code in FIGURE_CODES]
        assert issues == [], (seed, where, issues)
        assert path_reach(fig) == 0.0, (seed, where)
        assert stroke_reach(fig)[0] == 0.0, (seed, where)


@pytest.mark.parametrize("seed", QUICK)
def test_random_within_data_sets(seed: int) -> None:
    _check_drawn(seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [s for s in range(N_DATA_SETS) if s not in QUICK])
def test_random_within_data_sets_slow(seed: int) -> None:
    _check_drawn(seed)
