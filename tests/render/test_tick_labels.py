"""Long level names: tick labels wrap on spaces and shorten single long words, so
one long name does not widen every level."""

from __future__ import annotations

import numpy as np
import pytest

from inferplot.render import betweenstats
from inferplot.render.comparison import TICK_LINES, wrap_tick_label
from tests.render._results import frame_result


def _six(first: str) -> object:
    rng = np.random.default_rng(7)
    levels = [first, "b", "c", "d", "e", "f"]
    return frame_result(levels, [rng.normal(i * 0.8, 1, 20) for i in range(6)])


@pytest.mark.parametrize(
    ("label", "drawn"),
    [
        ("short\n(n = 20)", "short\n(n = 20)"),
        ("x" * 16 + "\n(n = 5)", "x" * 16 + "\n(n = 5)"),
        ("x" * 17 + "\n(n = 5)", "x" * 15 + "…\n(n = 5)"),
        (
            "control group receiving placebo\n(n = 8)",
            "control group\nreceiving\nplacebo\n(n = 8)",
        ),
        (
            "a very long level name that keeps going and going\n(n = 8)",
            "a very long\nlevel name that\nkeeps going and…\n(n = 8)",
        ),
    ],
)
def test_wrap_tick_label(label: str, drawn: str) -> None:
    # Lines of at most 16 characters stand in for the measured width.
    assert wrap_tick_label(label, lambda line: len(line) <= 16) == drawn
    lines = drawn.split("\n")[:-1]
    assert len(lines) <= TICK_LINES
    assert all(len(line) <= 16 for line in lines)


@pytest.mark.parametrize("chars", [20, 40, 60])
def test_one_long_name_does_not_widen_every_level(chars: int) -> None:
    short = betweenstats(_six("a"))  # pyright: ignore[reportArgumentType]
    long_name = " ".join(["word"] * (chars // 5)) if chars > 20 else "x" * chars
    fig = betweenstats(_six(long_name))  # pyright: ignore[reportArgumentType]
    width = float(fig.figure.get_size_inches()[0])
    assert width <= float(short.figure.get_size_inches()[0]) + 0.5
    # The full name stays in the result.
    assert fig.result.levels[0] == long_name
    assert fig.result.text.tick_labels[0].startswith(long_name)
    drawn = fig.axes["main"].get_xticklabels()[0].get_text()
    assert drawn != fig.result.text.tick_labels[0]
    assert drawn.endswith("\n(n = 20)")
    assert not [i for i in fig.check() if i.code in ("label_overlap", "text_clipped")]
