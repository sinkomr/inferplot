"""Random barstats tables: nothing clipped, overlapping or hidden; the figure's
size within stated bounds.

Each table has 2 to 12 levels of x and no y (one bar) or 2 to 20 levels of y,
n from 2 to 1e6 (log-uniform), and one of four shapes: even shares, one dominant
cell, tiny cells (a few counts of 1 next to large ones) and zero cells; level
and column names of random length, and a random cell label ("percentage",
"counts", "both"). It is drawn in inferplot's own figure and into a given Axes
at 72, 100 and 200 dpi, sized from the own figure times a random factor from
1.05 to 1.5 (with room for the margins ``add_subplot`` leaves). ``fig.check()``
must report no figure issue, and inferplot's own figure must stay within the
bounds below (Y1 of the Phase 4 review: an artist in data coordinates outside
the view made an 80-inch figure).
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from inferplot.render import StatsFigure, barstats
from inferplot.result import StatsResult
from tests.render._bars import table_result

N_TABLES = 120
QUICK = (0, 1, 2, 3)
# Figure issues: every code fig.check() reports about the drawing.
CODES = frozenset(
    {
        "label_overlap",
        "text_clipped",
        "cell_labels_hidden",
        "low_contrast",
        "glyph_missing",
        "axis_changed",
        "data_outside_view",
        "flat_data",
    }
)
# inferplot's own figure (inches): at most WIDTH_FIXED + WIDTH_PER_BAR per bar
# wide (or SUBTITLE_IN, which a wide subtitle may reach with few bars), at most
# MAX_HEIGHT high, and the plotting area at least MIN_AXES_SHARE of the figure.
# Measured over these 120 tables on macOS (1 to 20 bars, 2 to 12 levels of x, n
# 3 to 910,625): 5.3 to 25.7 in wide, at most 4 + 1.09 in per bar with three or
# more bars and 6.26 in with one or two, 3.9 to 6.6 in high (12 levels with
# names wrapped into three lines make the legend, and so the bars, taller), the
# plotting area at least 0.35 of the figure. The bounds leave room for fonts
# that rasterize differently on Linux and Windows.
WIDTH_FIXED = 4.0
WIDTH_PER_BAR = 1.25
SUBTITLE_IN = 7.5
MAX_HEIGHT = 8.0
MIN_AXES_SHARE = 0.25


def _name(rng: np.random.Generator, stem: str) -> str:
    words = int(rng.integers(0, 6))
    return " ".join([stem] + ["word" * int(rng.integers(1, 3)) for _ in range(words)])


def table(seed: int) -> StatsResult:
    """A random table of counts and its result (stand-in numbers)."""
    rng = np.random.default_rng([20261112, seed])
    rows = int(rng.integers(2, 13))
    one_way = bool(rng.random() < 0.15)
    columns = 1 if one_way else int(rng.integers(2, 21))
    n = round(10.0 ** rng.uniform(np.log10(2), 6.0))
    n = max(n, columns + 1)
    shape = int(rng.integers(0, 4))
    p = rng.dirichlet(np.ones(rows * columns)).reshape(rows, columns)
    if shape == 1:  # one dominant cell
        p[int(rng.integers(rows)), int(rng.integers(columns))] += 20.0
    elif shape == 2:  # tiny cells
        p *= rng.random((rows, columns)) < 0.5
        p += 1e-9
    elif shape == 3:  # zero cells
        p *= rng.random((rows, columns)) < 0.6
    p = p / p.sum()
    counts = rng.multinomial(n - columns - 1, p.ravel()).reshape(rows, columns)
    counts[0, :] += 1  # every bar holds an observation
    counts[1, 0] += 1  # two levels of x hold observations
    if shape == 2:
        for _ in range(int(rng.integers(1, 4))):
            counts[int(rng.integers(rows)), int(rng.integers(columns))] = 1
    x_levels = [_name(rng, f"x{i}") for i in range(rows)]
    y_levels = None if one_way else [_name(rng, f"y{j}") for j in range(columns)]
    return table_result(
        counts.tolist(),
        x_levels,
        y_levels,
        x=_name(rng, "fill"),
        y=_name(rng, "bars"),
        one_way=one_way,
        label=str(rng.choice(["percentage", "counts", "both"])),
    )


def figures(result: StatsResult, seed: int) -> list[tuple[str, StatsFigure]]:
    """The own figure, and the result drawn into a given Axes ``f`` times as
    large as the own figure without its left and bottom decorations (a given
    Axes holds the legend, the subtitle and the caption inside), with the own
    figure's room for the tick labels left and below it."""
    own = barstats(result)
    width, height = (float(v) for v in own.figure.get_size_inches())
    where = own.axes["main"].get_position()
    left, bottom = where.x0 * width, where.y0 * height
    rng = np.random.default_rng([seed, 2])
    out = [("own figure", own)]
    for dpi in (72, 100, 200):
        f = float(rng.uniform(1.05, 1.5))
        w, h = left + width * f, bottom + height * f
        figure = Figure(figsize=(w, h), dpi=dpi)
        FigureCanvasAgg(figure)
        margin = 0.15
        ax = figure.add_axes(
            (
                left / w,
                bottom / h,
                1.0 - (left + margin) / w,
                1.0 - (bottom + margin) / h,
            )
        )
        out.append((f"ax= at {dpi} dpi", barstats(result, ax=ax)))
    return out


def bounds_problems(fig: StatsFigure) -> list[str]:
    """What breaks the stated bounds of inferplot's own figure."""
    width, height = (float(v) for v in fig.figure.get_size_inches())
    bars = len(fig.result.bars)
    allowed = max(WIDTH_FIXED + WIDTH_PER_BAR * bars, SUBTITLE_IN)
    position = fig.axes["main"].get_position()
    share = float(position.width * position.height)
    found: list[str] = []
    if width > allowed:
        found.append(f"{width:.2f} in wide, more than {allowed:.2f}")
    if height > MAX_HEIGHT:
        found.append(f"{height:.2f} in high, more than {MAX_HEIGHT}")
    if share < MIN_AXES_SHARE:
        found.append(f"plotting area {share:.2f} of the figure")
    return found


def _check(seed: int) -> None:
    result = table(seed)
    for where, fig in figures(result, seed):
        found = [(i.code, i.message) for i in fig.check() if i.code in CODES]
        assert found == [], (seed, where, found)
        if where == "own figure":
            assert bounds_problems(fig) == [], seed


@pytest.mark.parametrize("seed", QUICK)
def test_random_tables(seed: int) -> None:
    _check(seed)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [s for s in range(N_TABLES) if s not in QUICK])
def test_random_tables_slow(seed: int) -> None:
    _check(seed)
