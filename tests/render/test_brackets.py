"""The bracket stacking and y-axis layout (pure geometry)."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inferplot.render.brackets import (
    LEVEL_STEP,
    TIP,
    Bracket,
    stack,
    vertical_layout,
)


def test_shortest_spans_lowest() -> None:
    brackets = [Bracket(0, 2, 0.5), Bracket(0, 1, 0.5), Bracket(1, 2, 0.5)]
    assert stack(brackets, gap=0.1) == (2, 0, 1)


def test_disjoint_brackets_share_a_level() -> None:
    brackets = [Bracket(0, 1, 0.5), Bracket(2, 3, 0.5), Bracket(0, 3, 0.5)]
    assert stack(brackets, gap=0.1) == (0, 0, 1)


def test_wide_labels_push_brackets_up() -> None:
    # The labels of (0, 1) and (2, 3) would overlap if 2.5 units wide.
    brackets = [Bracket(0, 1, 2.5), Bracket(2, 3, 2.5)]
    assert stack(brackets, gap=0.1) == (0, 1)


@given(
    st.lists(
        st.tuples(st.integers(0, 7), st.integers(0, 7), st.floats(0.1, 3.0)).filter(
            lambda t: t[0] < t[1]
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda t: (t[0], t[1]),
    )
)
def test_stacking_never_overlaps_and_nests_upward(
    items: list[tuple[int, int, float]],
) -> None:
    brackets = [Bracket(i, j, w) for i, j, w in items]
    gap = 0.1
    levels = stack(brackets, gap=gap)
    for (a, la), (b, lb) in itertools.combinations(
        zip(brackets, levels, strict=True), 2
    ):
        a_lo, a_hi = a.extent()
        b_lo, b_hi = b.extent()
        if la == lb:
            assert a_hi + gap <= b_lo or b_hi + gap <= a_lo
        if a.i <= b.i and b.j <= a.j:
            assert la > lb
        if b.i <= a.i and a.j <= b.j:
            assert lb > la


@pytest.mark.parametrize("n_levels", [0, 1, 2, 5])
def test_layout_leaves_room_for_labels(n_levels: int) -> None:
    label, pad, half = 14.0, 4.0, 9.0
    layout = vertical_layout(
        data_min=2.0,
        data_max=12.0,
        centers=[5.0, 11.5],
        n_levels=n_levels,
        center_half_px=half,
        label_px=label,
        pad_px=pad,
        min_height_px=320.0,
    )
    r = 10.0
    assert layout.height_px >= 320.0 - 1e-6
    assert layout.bottom <= 2.0 - 0.05 * r
    assert layout.tip == pytest.approx(TIP * r)
    ppu = layout.ppu
    # Centrality labels stay inside the limits and below the first bracket's legs.
    assert layout.bottom <= 5.0 - (half + pad) / ppu
    if n_levels:
        first = layout.levels[0]
        assert first >= 12.0 + LEVEL_STEP * r
        assert first - layout.tip >= 11.5 + (half + pad) / ppu - 1e-9
        for low, high in itertools.pairwise(layout.levels):
            assert high - low == pytest.approx(LEVEL_STEP * r)
            # A label (with padding) fits below the next level's legs.
            assert (high - layout.tip - low) * ppu >= label + 2 * pad - 1e-6
        top_label = layout.levels[-1] + (label + pad) / ppu
        assert layout.top >= top_label
    else:
        assert layout.top >= 11.5 + (half + pad) / ppu


def test_layout_fits_a_fixed_height() -> None:
    layout = vertical_layout(
        data_min=0.0,
        data_max=1.0,
        centers=[0.5],
        n_levels=2,
        center_half_px=8.0,
        label_px=12.0,
        pad_px=3.0,
        top_px=30.0,
        bottom_px=20.0,
        height_px=400.0,
    )
    assert layout.height_px == pytest.approx(400.0, rel=1e-9)


def test_layout_of_constant_data() -> None:
    layout = vertical_layout(
        data_min=3.0,
        data_max=3.0,
        centers=[3.0],
        n_levels=0,
        center_half_px=8.0,
        label_px=0.0,
        pad_px=3.0,
        min_height_px=300.0,
    )
    assert layout.bottom < 3.0 < layout.top
    assert layout.height_px >= 300.0 - 1e-6


def test_layout_needs_a_height() -> None:
    with pytest.raises(ValueError, match="height"):
        vertical_layout(
            data_min=0.0,
            data_max=1.0,
            centers=[],
            n_levels=0,
            center_half_px=0.0,
            label_px=0.0,
            pad_px=0.0,
        )
