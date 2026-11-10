"""The point label placement (pure geometry in pixels): clear of each other and of
the points where a place exists, inside the plotting area, deterministic."""

from __future__ import annotations

import numpy as np
import pytest

from inferplot.render.point_labels import DIRECTIONS, Placement, place_labels

FRAME = (0.0, 0.0, 600.0, 400.0)


def _boxes(
    anchors: np.ndarray, sizes: np.ndarray, places: list[Placement]
) -> np.ndarray:
    out = []
    for (x, y), (w, h), p in zip(anchors, sizes, places, strict=True):
        ax_, ay_ = x + p.dx, y + p.dy
        x0 = {"left": ax_, "right": ax_ - w, "center": ax_ - w / 2}[p.ha]
        y0 = {"bottom": ay_, "top": ay_ - h, "center": ay_ - h / 2}[p.va]
        out.append((x0, y0, x0 + w, y0 + h))
    return np.array(out)


def _place(points: np.ndarray, labelled: list[int], width: float = 60.0) -> tuple[
    np.ndarray, np.ndarray, list[Placement]
]:  # fmt: skip
    anchors = points[labelled]
    sizes = np.tile([width, 12.0], (len(labelled), 1))
    places = place_labels(
        anchors=anchors,
        sizes=sizes,
        points=points,
        marker=2.8,
        gap=2.0,
        frame=FRAME,
    )
    return anchors, sizes, places


@pytest.mark.parametrize("seed", range(10))
def test_sparse_labels_cover_nothing(seed: int) -> None:
    rng = np.random.default_rng(seed)
    points = np.column_stack([rng.uniform(40, 560, 40), rng.uniform(30, 370, 40)])
    labelled = list(range(12))
    anchors, sizes, places = _place(points, labelled)
    boxes = _boxes(anchors, sizes, places)
    for i, a in enumerate(boxes):
        assert FRAME[0] <= a[0]
        assert a[2] <= FRAME[2]
        assert FRAME[1] <= a[1]
        assert a[3] <= FRAME[3]
        for b in boxes[i + 1 :]:
            assert min(a[2], b[2]) - max(a[0], b[0]) <= 0 or (
                min(a[3], b[3]) - max(a[1], b[1]) <= 0
            )
        inside = (
            (points[:, 0] > a[0] - 2.3)
            & (points[:, 0] < a[2] + 2.3)
            & (points[:, 1] > a[1] - 2.3)
            & (points[:, 1] < a[3] + 2.3)
        )
        assert not inside.any(), (seed, i)


def test_deterministic() -> None:
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.uniform(0, 600, 300), rng.uniform(0, 400, 300)])
    first = _place(points, list(range(50)))[2]
    assert first == _place(points.copy(), list(range(50)))[2]


def test_label_next_to_a_lone_point_goes_right() -> None:
    points = np.array([[300.0, 200.0]])
    (place,) = _place(points, [0])[2]
    assert (place.ha, place.va, place.ring) == ("left", "center", 0)
    assert place.dx == pytest.approx(4.8)
    assert place.dy == 0.0


def test_moves_away_from_a_neighbour() -> None:
    # A point just right of the labelled one: the label goes elsewhere.
    points = np.array([[300.0, 200.0], [330.0, 200.0]])
    (place,) = _place(points, [0])[2]
    assert (place.dx, place.dy) != (pytest.approx(4.8), 0.0)
    assert place.ring == 0


def test_at_the_right_edge_the_label_goes_left() -> None:
    points = np.array([[590.0, 200.0]])
    (place,) = _place(points, [0])[2]
    assert place.ha == "right"


def test_crowded_labels_move_to_outer_rings() -> None:
    # Eight labelled points on top of each other: all cannot sit next to them.
    points = np.tile([[300.0, 200.0]], (8, 1)) + np.arange(8)[:, None] * 0.5
    anchors, sizes, places = _place(points, list(range(8)))
    assert max(p.ring for p in places) >= 1
    boxes = _boxes(anchors, sizes, places)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert min(a[2], b[2]) - max(a[0], b[0]) <= 0 or (
                min(a[3], b[3]) - max(a[1], b[1]) <= 0
            )


def test_above_the_limit_every_label_goes_right() -> None:
    points = np.column_stack([np.linspace(10, 590, 30), np.full(30, 200.0)])
    places = place_labels(
        anchors=points,
        sizes=np.tile([40.0, 12.0], (30, 1)),
        points=points,
        marker=2.8,
        gap=2.0,
        frame=FRAME,
        limit=20,
    )
    assert {(p.ha, p.va, p.ring, p.dy) for p in places} == {("left", "center", 0, 0.0)}


def test_directions_are_unit_steps() -> None:
    for ux, uy in DIRECTIONS:
        assert np.hypot(ux, uy) == pytest.approx(1.0, abs=1e-4)
