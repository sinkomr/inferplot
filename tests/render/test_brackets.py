"""The bracket stacking and y-axis layout (pure geometry)."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inferplot.render.brackets import (
    DATA_PAD,
    MIN_DATA_SHARE,
    Bracket,
    level_step_px,
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


# Pixel sizes for the layout tests: label, padding, legs, line width.
LABEL, PAD, TIP, LINE = 14.0, 4.0, 5.5, 1.1
STEP = level_step_px(label_px=LABEL, pad_px=PAD, tip_px=TIP, line_px=LINE)


def test_level_step_holds_label_padding_and_legs() -> None:
    # pad, label, pad, line width, legs: 4 + 14 + 4 + 1.1 + 5.5
    step = level_step_px(label_px=14.0, pad_px=4.0, tip_px=5.5, line_px=1.1)
    assert step == pytest.approx(28.6)


@pytest.mark.parametrize("n_levels", [0, 1, 2, 5, 12])
def test_layout_leaves_room_for_labels(n_levels: int) -> None:
    half = 9.0
    layout = vertical_layout(
        data_min=2.0,
        data_max=12.0,
        centers=[5.0, 11.5],
        n_levels=n_levels,
        center_half_px=half,
        label_px=LABEL,
        pad_px=PAD,
        tip_px=TIP,
        line_px=LINE,
        min_height_px=320.0,
        data_height_px=250.0,
    )
    r = 10.0
    ppu = layout.ppu
    assert layout.height_px >= 320.0 - 1e-6
    assert layout.data_px >= 250.0 - 1e-6
    assert layout.bottom <= 2.0 - DATA_PAD * r
    assert layout.tip * ppu == pytest.approx(TIP)
    assert layout.label_pad * ppu == pytest.approx(PAD)
    # Centrality labels stay inside the limits and below the first bracket's legs.
    assert layout.bottom <= 5.0 - (half + PAD) / ppu
    if n_levels:
        first = layout.levels[0]
        legs = first - layout.tip - LINE / 2 / ppu  # the caps included
        assert legs >= 12.0 + DATA_PAD * r - 1e-9
        assert legs >= 11.5 + (half + PAD) / ppu - 1e-9
        assert layout.step_px == pytest.approx(STEP)
        for low, high in itertools.pairwise(layout.levels):
            assert (high - low) * ppu == pytest.approx(STEP)
            # A label and its padding fit below the next level's legs and caps,
            # with a pad and half a line width of clearance.
            gap = (high - layout.tip - low) * ppu - LINE / 2 - (PAD + LABEL)
            assert gap == pytest.approx(PAD + LINE / 2)
        top_label = layout.levels[-1] + (LABEL + 2 * PAD) / ppu
        assert layout.top == pytest.approx(top_label)
    else:
        assert layout.top >= 11.5 + (half + PAD) / ppu


@pytest.mark.parametrize("n_levels", [0, 1, 3, 8, 20])
@pytest.mark.parametrize("data_max", [12.0, 1e-6, 4e8])
def test_brackets_add_only_their_own_height(n_levels: int, data_max: float) -> None:
    # The data region keeps its height at any data range; the Axes grows by the
    # stack of levels and nothing else.
    data_min = 2.0 if data_max > 2.0 else 0.0
    layout = vertical_layout(
        data_min=data_min,
        data_max=data_max,
        centers=[],
        n_levels=n_levels,
        center_half_px=0.0,
        label_px=LABEL,
        pad_px=PAD,
        tip_px=TIP,
        line_px=LINE,
        min_height_px=320.0,
        data_height_px=250.0,
    )
    stack_px = 0.0
    if n_levels:
        stack_px = TIP + LINE / 2 + (n_levels - 1) * STEP + 2 * PAD + LABEL
    assert layout.height_px == pytest.approx(max(320.0, 250.0 + stack_px))
    if 250.0 + stack_px > 320.0:
        assert layout.data_px == pytest.approx(250.0)


def test_layout_fits_a_fixed_height() -> None:
    layout = vertical_layout(
        data_min=0.0,
        data_max=1.0,
        centers=[0.5],
        n_levels=2,
        center_half_px=8.0,
        label_px=12.0,
        pad_px=3.0,
        tip_px=TIP,
        line_px=LINE,
        top_px=30.0,
        bottom_px=20.0,
        data_height_px=250.0,
        height_px=400.0,
    )
    assert layout.height_px == pytest.approx(400.0, rel=1e-9)
    assert layout.step_px == pytest.approx(
        level_step_px(label_px=12.0, pad_px=3.0, tip_px=TIP, line_px=LINE)
    )


def test_a_short_fixed_height_pushes_levels_closer() -> None:
    # 10 levels need about 300 px; the Axes has 300 px in all, 250 px after the
    # subtitle and caption: the data keep half of that and the levels the rest.
    layout = vertical_layout(
        data_min=0.0,
        data_max=1.0,
        centers=[],
        n_levels=10,
        center_half_px=0.0,
        label_px=LABEL,
        pad_px=PAD,
        tip_px=TIP,
        line_px=LINE,
        top_px=30.0,
        bottom_px=20.0,
        data_height_px=400.0,
        height_px=300.0,
    )
    assert layout.height_px == pytest.approx(300.0, rel=1e-9)
    assert layout.data_px == pytest.approx(MIN_DATA_SHARE * 250.0)
    assert 0.0 < layout.step_px < STEP


def test_a_tall_fixed_height_keeps_full_spacing() -> None:
    # The data get less than data_height_px but at least half the room: the
    # levels keep their full spacing.
    layout = vertical_layout(
        data_min=0.0,
        data_max=1.0,
        centers=[],
        n_levels=10,
        center_half_px=0.0,
        label_px=LABEL,
        pad_px=PAD,
        tip_px=TIP,
        line_px=LINE,
        data_height_px=400.0,
        height_px=600.0,
    )
    assert layout.step_px == pytest.approx(STEP)
    assert MIN_DATA_SHARE * 600.0 < layout.data_px < 400.0


def test_layout_of_constant_data() -> None:
    layout = vertical_layout(
        data_min=3.0,
        data_max=3.0,
        centers=[3.0],
        n_levels=0,
        center_half_px=8.0,
        label_px=0.0,
        pad_px=3.0,
        tip_px=TIP,
        line_px=LINE,
        min_height_px=300.0,
        data_height_px=250.0,
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
            tip_px=TIP,
            line_px=LINE,
            data_height_px=250.0,
        )
