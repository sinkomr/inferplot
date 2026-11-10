# NumPy's stubs type several functions taking array-likes as partly Unknown, which
# strict mode reports on every call; the arguments we pass are all float arrays.
# pyright: reportUnknownMemberType=false
"""Where scatterstats puts its point labels: pure geometry in pixels.

Each label is tried at candidate places around its point, nearest first: eight
directions (right, upper right, lower right, left, upper left, lower left, above,
below) on rings of growing distance. A place is allowed when the label's box lies
inside the plotting area and clear of every label placed before it (and of the
texts drawn inside the Axes); among those, the first that covers no point is
taken, else the one covering fewest points, nearest first. A label on an outer
ring is joined to its point by a thin line. Labels are placed most crowded first
(the most points near them), then in data order, so the result is deterministic
and needs no random numbers. Where no place is allowed the least bad one is
taken (least overlap with other labels and the edges), and ``fig.check()``
reports what is left (``point_labels_crowded``, ``text_clipped``).

Above ``limit`` labels the search is skipped: every label goes right of its point
and ``fig.check()`` says so.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["DIRECTIONS", "Placement", "place_labels"]

FloatArray = NDArray[np.float64]

# (x, y) unit steps of the eight directions, in order of preference.
DIRECTIONS: Final[tuple[tuple[float, float], ...]] = (
    (1.0, 0.0),
    (0.7071, 0.7071),
    (0.7071, -0.7071),
    (-1.0, 0.0),
    (-0.7071, 0.7071),
    (-0.7071, -0.7071),
    (0.0, 1.0),
    (0.0, -1.0),
)
# Ring distances beyond the first, in label heights.
RINGS: Final = (0.0, 1.2, 2.4, 3.6, 5.0, 7.0)
_TOLERANCE: Final = 0.5  # pixels of contact that do not count (anti-aliasing)
SPACING: Final = 1.0  # pixels kept between two labels

Horizontal = Literal["left", "right", "center"]
Vertical = Literal["bottom", "top", "center"]


@dataclasses.dataclass(frozen=True, slots=True)
class Placement:
    """A label's place: its anchor ``dx``, ``dy`` pixels from its point, the
    alignment of the text at the anchor, and its ring (0: next to the point; an
    outer ring gets a line to the point)."""

    dx: float
    dy: float
    ha: Horizontal
    va: Vertical
    ring: int


def _align(u: float) -> float:
    """Where the anchor lies along a label's side for a step ``u``: its start
    (0), its end (-1) or its middle (-0.5), in label widths or heights."""
    if u > 0.0:
        return 0.0
    if u < 0.0:
        return -1.0
    return -0.5


def _ha(ux: float) -> Horizontal:
    return "left" if ux > 0.0 else "right" if ux < 0.0 else "center"


def _va(uy: float) -> Vertical:
    return "bottom" if uy > 0.0 else "top" if uy < 0.0 else "center"


def _box(
    x: float, y: float, w: float, h: float, ux: float, uy: float, r: float
) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of a w x h label anchored r pixels from (x, y) in
    direction (ux, uy)."""
    ax_, ay_ = x + r * ux, y + r * uy
    x0, y0 = ax_ + _align(ux) * w, ay_ + _align(uy) * h
    return x0, y0, x0 + w, y0 + h


def _overlap_area(box: tuple[float, float, float, float], boxes: FloatArray) -> float:
    """Area of ``box``, grown by SPACING on every side, shared with each of
    ``boxes`` (k, 4), summed: labels keep a pixel apart."""
    if boxes.shape[0] == 0:
        return 0.0
    x0, y0, x1, y1 = (
        box[0] - SPACING,
        box[1] - SPACING,
        box[2] + SPACING,
        box[3] + SPACING,
    )
    w = np.minimum(x1, boxes[:, 2]) - np.maximum(x0, boxes[:, 0])
    h = np.minimum(y1, boxes[:, 3]) - np.maximum(y0, boxes[:, 1])
    hit = (w > 0.0) & (h > 0.0)
    return float((w[hit] * h[hit]).sum())


def place_labels(
    *,
    anchors: FloatArray,
    sizes: FloatArray,
    points: FloatArray,
    marker: float,
    gap: float,
    frame: tuple[float, float, float, float],
    obstacles: Sequence[tuple[float, float, float, float]] = (),
    limit: int = 200,
) -> list[Placement]:
    """A place for each label (see the module docstring).

    ``anchors`` (m, 2): each labelled point in pixels; ``sizes`` (m, 2): each
    label's width and height in pixels; ``points`` (n, 2): every drawn point in
    pixels; ``marker``: the points' half size in pixels; ``gap``: the room
    between a point's marker and the nearest label, in pixels; ``frame``: the plotting
    area (x0, y0, x1, y1); ``obstacles``: boxes of other texts inside it.
    """
    m = anchors.shape[0]
    if m == 0:
        return []
    first = marker + gap
    if m > limit:
        return [
            Placement(dx=first, dy=0.0, ha="left", va="center", ring=0)
            for _ in range(m)
        ]
    # Most crowded first: the points within a label's reach of each anchor.
    reach = (sizes[:, 0] + sizes[:, 1])[:, None]
    dx = np.abs(points[None, :, 0] - anchors[:, None, 0])
    dy = np.abs(points[None, :, 1] - anchors[:, None, 1])
    crowd = ((dx < reach) & (dy < reach)).sum(axis=1)
    order = sorted(range(m), key=lambda i: (-int(crowd[i]), i))
    placed = np.array(list(obstacles), dtype=np.float64).reshape(-1, 4)
    fx0, fy0, fx1, fy1 = frame
    out: list[Placement | None] = [None] * m
    for i in order:
        x, y = float(anchors[i, 0]), float(anchors[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        best: tuple[tuple[float, ...], Placement, tuple[float, float, float, float]]
        best = ((np.inf,), Placement(first, 0.0, "left", "center", 0), (0, 0, 0, 0))
        found = False
        for ring, step in enumerate(RINGS):
            r = first + step * h
            for ux, uy in DIRECTIONS:
                box = _box(x, y, w, h, ux, uy, r)
                outside = (
                    max(0.0, fx0 - box[0] - _TOLERANCE)
                    + max(0.0, box[2] - fx1 - _TOLERANCE)
                ) * h + (
                    max(0.0, fy0 - box[1] - _TOLERANCE)
                    + max(0.0, box[3] - fy1 - _TOLERANCE)
                ) * w
                clash = _overlap_area(box, placed)
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                covered = int(
                    (
                        (np.abs(points[:, 0] - cx) < w / 2 + marker - _TOLERANCE)
                        & (np.abs(points[:, 1] - cy) < h / 2 + marker - _TOLERANCE)
                    ).sum()
                )
                place = Placement(
                    dx=r * ux, dy=r * uy, ha=_ha(ux), va=_va(uy), ring=ring
                )
                bad = outside + clash
                score = (float(bad > 0.0), bad, float(covered), float(ring))
                if score < best[0]:
                    best = (score, place, box)
                if bad == 0.0 and covered == 0:
                    found = True
                    break
            if found:
                break
        out[i] = best[1]
        placed = np.vstack([placed, np.array(best[2], dtype=np.float64)[None, :]])
    return [p for p in out if p is not None]
