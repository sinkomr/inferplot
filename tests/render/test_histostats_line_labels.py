"""histostats: each line's label reads as its own line's, and the test value is
printed by one rule everywhere.

A review found the test value's label drawn beside the centrality line, 40 to 88
pt from its own (15 of 120 figures of test values a user would write), and
``fig.check()`` silent. Measured here independently of the renderer: from the
drawn lines and the rendered label boxes, every label must lie within a few
points of its own line, no nearer to the other line, and no line may run
through the other's label; ``line_label_ambiguous`` must report a label moved
next to the other line.
"""

from __future__ import annotations

import itertools
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest
from matplotlib.lines import Line2D
from matplotlib.text import Text

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.text import shown_test_value
from tests.render._helpers import by_gid


def _pieces(fig: StatsFigure, gid: str) -> np.ndarray[Any, Any] | None:
    line = by_gid(fig, gid)
    if line is None:
        return None
    assert isinstance(line, Line2D)
    data = np.asarray(line.get_xydata(), dtype=np.float64)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return np.asarray(line.get_transform().transform(data), dtype=np.float64)


def _distance(box: Any, ends: np.ndarray[Any, Any]) -> float:
    """Pixels from a box to a vertical line drawn from ends[0] to ends[1]."""
    (x, y0), (_, y1) = ends
    low, high = sorted((float(y0), float(y1)))
    dx = max(box.x0 - x, x - box.x1, 0.0)
    dy = max(box.y0 - high, low - box.y1, 0.0)
    return float(np.hypot(dx, dy))


def _label_box(fig: StatsFigure, gid: str) -> Any:
    text = by_gid(fig, gid)
    assert isinstance(text, Text)
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    box = text.get_window_extent(renderer)
    patch = text.get_bbox_patch()
    if patch is not None:
        from matplotlib.transforms import Bbox

        box = Bbox.union([box, patch.get_window_extent(renderer)])
    return box


def assert_labels_read_as_their_lines(fig: StatsFigure, case: object) -> None:
    fig.draw()
    to_pt = 72.0 / float(fig.figure.dpi)
    lines = {
        "centrality-label": _pieces(fig, "centrality-line"),
        "test-value-label": _pieces(fig, "test-value-line"),
    }
    drawn = {gid: ends for gid, ends in lines.items() if ends is not None}
    for gid, ends in drawn.items():
        box = _label_box(fig, gid)
        own = _distance(box, ends) * to_pt
        # Beside its own line: 3 pt away, 0.5 pt of rounding.
        assert own <= 3.5, (case, gid, own)
        for other, other_ends in drawn.items():
            if other == gid:
                continue
            near = _distance(box, other_ends) * to_pt
            assert near > 0.0, (case, gid, "the other line runs through it")
            assert near >= own - 0.5, (case, gid, near, own)
    assert "line_label_ambiguous" not in [i.code for i in fig.check()], case


# Test values a user would write (one decimal) at quantiles of the data, as the
# review drew them: normal, small whole numbers, skewed.
def _data(seed: int) -> tuple[np.ndarray[Any, Any], float]:
    rng = np.random.default_rng([20261110, seed])
    m = int(rng.choice([8, 20, 60, 200]))
    kind = seed % 3
    if kind == 0:
        values = rng.normal(50, 10, m)
    elif kind == 1:
        values = rng.integers(1, 8, m).astype(float)
    else:
        values = rng.lognormal(3, 0.6, m)
    return values, round(float(np.quantile(values, rng.uniform(0.02, 0.98))), 1)


QUICK: Final = range(8)
SLOW: Final = range(8, 120)


def _check(seed: int, type_name: str) -> None:
    values, tv = _data(seed)
    fig = ip.histostats(pd.DataFrame({"x": values}), "x", test_value=tv, type=type_name)
    assert_labels_read_as_their_lines(fig, (seed, type_name, tv))


@pytest.mark.parametrize(
    ("seed", "type_name"), list(itertools.product(QUICK, ["p", "np", "r", "bf"]))
)
def test_each_label_at_its_own_line(seed: int, type_name: str) -> None:
    _check(seed, type_name)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("seed", "type_name"), list(itertools.product(SLOW, ["p", "np"]))
)
def test_each_label_at_its_own_line_many(seed: int, type_name: str) -> None:
    _check(seed, type_name)


@pytest.mark.parametrize(
    ("values", "test_value"),
    [
        # Lines 7 px apart near the left edge: the test value's label reaches
        # across the centrality line (the review's detached-label cases).
        ([0.0] * 60 + [0.5, 1.2, 2.0, 3.1, 4.4, 6.0, 8.5, 12.0, 17.0, 18.0], 0.04),
        ([10.0] * 19 + [60.0], 10.0),
        # The test value's label right of the data, the centrality line near.
        ([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 9.0], 1 / 3),
        ([1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 9.6], 9.6),
    ],
)
# Type robust only where the winsorized values differ.
@pytest.mark.parametrize("type_name", ["p", "np"])
def test_lines_close_together(
    values: list[float], test_value: float, type_name: str
) -> None:
    fig = ip.histostats(
        pd.DataFrame({"x": values}), "x", test_value=test_value, type=type_name
    )
    assert_labels_read_as_their_lines(fig, (values[:3], test_value, type_name))


def test_a_label_moved_beside_the_other_line_is_reported() -> None:
    values, _ = _data(0)
    fig = ip.histostats(pd.DataFrame({"x": values}), "x", test_value=40)
    assert "line_label_ambiguous" not in [i.code for i in fig.check()]
    label = by_gid(fig, "test-value-label")
    assert isinstance(label, Text)
    center = fig.result.descriptives[0].value
    assert center is not None
    # Moved over the centrality line: that line now runs through it.
    label.set_horizontalalignment("left")
    label.set_x(center)
    (issue,) = [i for i in fig.check() if i.code == "line_label_ambiguous"]
    assert issue.severity == "warning"
    assert "test value label 'test value 40'" in issue.message
    assert "the centrality line" in issue.message
    assert "beside its own line" in issue.fix


# --------------------------------------------------------------------------- #
# the test value's text (low item L3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "digits", "shown", "rounded"),
    [
        (18.0, 2, "18", False),
        (5.8, 2, "5.8", False),
        (-10.0, 2, "-10", False),
        (0.001, 2, "0.001", False),
        (1.7e9, 2, "1.7e+09", False),
        (123456.7, 2, "123456.7", False),
        (12345678.0, 2, "12345678", False),
        (-0.1 - 0.2, 2, "-0.3", False),  # R's 15 digits: -0.3
        (1 / 3, 2, "0.33", True),
        (1 / 3, 4, "0.3333", True),
        (36.123456789, 2, "36.12", True),
        (123456.789, 2, "1.23e+05", True),
        (np.pi * 100, 3, "314.159", True),
        (5e-324, 2, "4.94e-324", True),
    ],
)
def test_test_value_rule(value: float, digits: int, shown: str, rounded: bool) -> None:
    assert shown_test_value(value, digits) == (shown, rounded)


@pytest.mark.parametrize(
    ("test_value", "label", "words"),
    [
        (18.0, "test value 18", "18"),
        (1 / 3, "test value ≈ 0.33", "about 0.33"),
        (100 / 3, "test value ≈ 33.33", "about 33.33"),
    ],
)
def test_test_value_printed_alike_everywhere(
    test_value: float, label: str, words: str
) -> None:
    rng = np.random.default_rng(5)
    values = test_value + rng.normal(0.0, 1.0, 30)
    fig = ip.histostats(pd.DataFrame({"x": values}), "x", test_value=test_value)
    text = by_gid(fig, "test-value-label")
    assert isinstance(text, Text)
    assert text.get_text() == label
    result = fig.result
    assert f"One-sample t-test against {words}: " in result.text.apa
    assert f"and the test value, {words}." in result.text.alt_text
    assert result.settings.test_value == test_value  # the exact value is kept
    # The arrow at the edge prints it alike.
    far = ip.histostats(pd.DataFrame({"x": values + 100.0}), "x", test_value=test_value)
    arrow = by_gid(far, "test-value-label")
    assert isinstance(arrow, Text)
    assert arrow.get_text() == f"← {label}"
