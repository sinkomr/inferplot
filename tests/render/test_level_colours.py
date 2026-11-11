"""Each level's colour is tied to its palette entry and to what names it.

With an asymmetric palette and asymmetric data, for every level: barstats'
segments of level i are drawn in palette[i] in every bar, with the height of
that level's share of the bar; the legend's i-th key has palette[i] and its
i-th entry names level i. In betweenstats and withinstats every layer of level
i (points, box, violin) is drawn in palette[i] at x = i, whose tick label names
level i. A swap of two colours, or of two legend entries, fails these tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon, Rectangle

import inferplot as ip
from inferplot.render import StatsFigure

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
LEVELS = ["often", "sometimes", "rarely", "never"]


def _rgb(color: object) -> tuple[float, float, float]:
    return to_rgb(color)  # pyright: ignore[reportArgumentType]


def _gid(fig: StatsFigure, gid: str) -> object:
    found = [a for a in fig.figure.findobj() if a.get_gid() == gid]
    assert len(found) == 1, gid
    return found[0]


def test_barstats_colour_legend_and_palette_agree_for_every_level() -> None:
    counts = [[40, 3, 11], [7, 22, 1], [2, 9, 30], [15, 15, 5]]
    bars = ["east", "north", "west"]
    frame = pd.DataFrame(
        [
            (LEVELS[i], bars[j], counts[i][j])
            for i in range(len(LEVELS))
            for j in range(len(bars))
        ],
        columns=["answer", "region", "n"],
    )
    frame["answer"] = pd.Categorical(frame["answer"], categories=LEVELS)
    fig = ip.barstats(
        frame, "answer", "region", counts="n", palette=PALETTE, bf_message=False
    )
    result = fig.result
    assert result.levels == tuple(LEVELS)
    ax = fig.axes["main"]
    legend = ax.get_legend()
    assert legend is not None
    keys = legend.legend_handles
    names = [t.get_text() for t in legend.get_texts()]
    assert names == LEVELS
    for i, level in enumerate(LEVELS):
        key = keys[i]
        assert isinstance(key, Rectangle)
        assert _rgb(key.get_facecolor()) == pytest.approx(_rgb(PALETTE[i]))
        for j in range(len(bars)):
            cell = result.cells[j * len(LEVELS) + i]
            assert cell.level == level
            assert cell.count == counts[i][j]
            segment = _gid(fig, f"segment-{j}-{i}")
            assert isinstance(segment, Rectangle)
            assert _rgb(segment.get_facecolor()) == pytest.approx(_rgb(PALETTE[i]))
            assert segment.get_height() == pytest.approx(cell.percent, abs=1e-12)
            assert segment.get_x() + segment.get_width() / 2 == pytest.approx(j)


def _numbers() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    return pd.DataFrame(
        {
            "g": np.repeat(LEVELS, [6, 9, 12, 15]),
            "v": np.concatenate(
                [
                    rng.normal(m, 1, n)
                    for m, n in zip([0, 3, 1, 5], [6, 9, 12, 15], strict=True)
                ]
            ),
        }
    )


def _within() -> pd.DataFrame:
    rng = np.random.default_rng(10)
    n = 12
    return pd.DataFrame(
        {
            "g": np.tile(LEVELS, n),
            "id": np.repeat(np.arange(n), len(LEVELS)),
            "v": rng.normal(size=n * len(LEVELS)) + np.tile([0, 3, 1, 5], n),
        }
    )


@pytest.mark.parametrize("function", ["betweenstats", "withinstats"])
def test_comparison_layers_follow_the_palette_for_every_level(function: str) -> None:
    if function == "betweenstats":
        fig = ip.betweenstats(_numbers(), "g", "v", palette=PALETTE)
    else:
        fig = ip.withinstats(_within(), "g", "v", subject_id="id", palette=PALETTE)
    levels = fig.result.levels
    assert sorted(levels) == sorted(LEVELS)
    ticks = [t.get_text().split("\n")[0] for t in fig.axes["main"].get_xticklabels()]
    assert ticks == list(levels)
    for i, level in enumerate(levels):
        want = _rgb(PALETTE[i])
        points = _gid(fig, f"points-{i}")
        colors = np.asarray(points.get_facecolor(), dtype=float)  # type: ignore[attr-defined]
        assert colors.shape[0] >= 1
        assert all(tuple(c[:3]) == pytest.approx(want) for c in colors), level
        offsets = np.asarray(points.get_offsets(), dtype=float)  # type: ignore[attr-defined]
        assert np.all(np.abs(offsets[:, 0] - i) < 0.5), level
        # The violin is drawn in the level's colour (the box is neutral).
        violin = _gid(fig, f"violin-{i}")
        assert isinstance(violin, Polygon)
        assert _rgb(violin.get_edgecolor()) == pytest.approx(want), level
        assert _rgb(violin.get_facecolor()) == pytest.approx(want), level
        xs = np.asarray(violin.get_xy(), dtype=float)[:, 0]
        assert abs(float(xs.min() + xs.max()) / 2 - i) < 1e-9, level
