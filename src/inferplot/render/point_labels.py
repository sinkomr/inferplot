# NumPy's stubs type several functions taking array-likes as partly Unknown, which
# strict mode reports on every call; the arguments we pass are all float arrays.
# pyright: reportUnknownMemberType=false
"""Where scatterstats puts its point labels: pure geometry in pixels.

Each label is tried at candidate places around its point, nearest first: eight
directions (right, upper right, lower right, left, upper left, lower left, above,
below) on rings of growing distance. A place is allowed when the label's box lies
inside the plotting area and clear of every label placed before it (and of the
texts drawn inside the Axes); among those, the first that no line crosses (the
least-squares line, or the link of a label placed before) and whose own link
crosses no label, and that covers no point, is taken; else the one with fewest
such crossings, then covering fewest points, nearest first. A label on an outer
ring is joined to its point by a thin line (its link). Labels are placed most
crowded first (the most points near them), then in data order; then each label
still crossed by a line or link, or whose link crosses a label, is placed again
where that is better (two passes), so the result is deterministic
and needs no random numbers. Where no place is allowed the least bad one is
taken (least overlap with other labels and the edges), and ``fig.check()``
reports what is left (``point_labels_crowded``, ``text_clipped``).

Above ``limit`` labels the search is skipped: every label goes right of its point
and ``fig.check()`` says so.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CANDIDATES",
    "DIRECTIONS",
    "RINGS",
    "Placement",
    "crossing",
    "place_labels",
]

FloatArray = NDArray[np.float64]
Horizontal = Literal["left", "right", "center"]
Vertical = Literal["bottom", "top", "center"]


def _ha_of(ux: float) -> Horizontal:
    """The alignment of a label at an anchor reached in direction ``ux``."""
    return "left" if ux > 0.0 else "right" if ux < 0.0 else "center"


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
# The candidate places on a ring: each direction with the label's alignment at its
# anchor, then straight above and below with the label starting or ending at the
# point (a wide label then reaches past its neighbours on one side only).
CANDIDATES: Final[tuple[tuple[float, float, Horizontal], ...]] = (
    *((ux, uy, _ha_of(ux)) for ux, uy in DIRECTIONS),
    (0.0, 1.0, "left"),
    (0.0, -1.0, "left"),
    (0.0, 1.0, "right"),
    (0.0, -1.0, "right"),
)
# Ring distances beyond the first, in label heights.
RINGS: Final = (0.0, 1.2, 2.4, 3.6, 5.0, 7.0, 9.5, 12.5)
_TOLERANCE: Final = 0.5  # pixels of contact that do not count (anti-aliasing)
SPACING: Final = 1.0  # pixels kept between two labels


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
    """Where the anchor lies along a label's height for a step ``u``: its bottom
    (0), its top (-1) or its middle (-0.5), in label heights."""
    if u > 0.0:
        return 0.0
    if u < 0.0:
        return -1.0
    return -0.5


_START: Final[dict[str, float]] = {"left": 0.0, "right": -1.0, "center": -0.5}


def _va(uy: float) -> Vertical:
    return "bottom" if uy > 0.0 else "top" if uy < 0.0 else "center"


def crossing(segments: FloatArray, boxes: FloatArray) -> NDArray[np.bool_]:
    """(k, j): whether each of ``segments`` ((k, 5): x0, y0, x1, y1 and half the
    stroke's width, in pixels) crosses each of ``boxes`` ((j, 4): x0, y0, x1,
    y1), reaching into it by more than _TOLERANCE with its stroke (Liang-Barsky
    clipping of the segment against the box grown by the half width)."""
    k, j = segments.shape[0], boxes.shape[0]
    out = np.zeros((k, j), dtype=bool)
    if k == 0 or j == 0:
        return out
    # Only segments and boxes whose extents meet can cross (NaN rows: none).
    reach = segments[:, 4]
    s_lo_x = np.minimum(segments[:, 0], segments[:, 2]) - reach
    s_hi_x = np.maximum(segments[:, 0], segments[:, 2]) + reach
    s_lo_y = np.minimum(segments[:, 1], segments[:, 3]) - reach
    s_hi_y = np.maximum(segments[:, 1], segments[:, 3]) + reach
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        b_lo_x, b_hi_x = np.nanmin(boxes[:, 0]), np.nanmax(boxes[:, 2])
        b_lo_y, b_hi_y = np.nanmin(boxes[:, 1]), np.nanmax(boxes[:, 3])
        rows = np.flatnonzero(
            (s_hi_x >= b_lo_x) & (s_lo_x <= b_hi_x) & (s_hi_y >= b_lo_y)
            & (s_lo_y <= b_hi_y)
        )  # fmt: skip
        if rows.size == 0:
            return out
        cols = np.flatnonzero(
            (boxes[:, 2] >= np.nanmin(s_lo_x[rows]))
            & (boxes[:, 0] <= np.nanmax(s_hi_x[rows]))
            & (boxes[:, 3] >= np.nanmin(s_lo_y[rows]))
            & (boxes[:, 1] <= np.nanmax(s_hi_y[rows]))
        )
    if cols.size == 0:
        return out
    out[np.ix_(rows, cols)] = _crossing(segments[rows], boxes[cols])
    return out


def _crossing(segments: FloatArray, boxes: FloatArray) -> NDArray[np.bool_]:
    """:func:`crossing` on every pair, by separating axes: a segment and a box
    (grown by the stroke's half width less the tolerance) meet when their
    extents overlap on both axes and the box's corners do not all lie on one
    side of the segment's line."""
    grow = (segments[:, 4] - _TOLERANCE)[:, None]
    bx0, by0 = boxes[None, :, 0] - grow, boxes[None, :, 1] - grow
    bx1, by1 = boxes[None, :, 2] + grow, boxes[None, :, 3] + grow
    x0, y0 = segments[:, 0, None], segments[:, 1, None]
    x1, y1 = segments[:, 2, None], segments[:, 3, None]
    meet = (
        (np.maximum(x0, x1) > bx0)
        & (np.minimum(x0, x1) < bx1)
        & (np.maximum(y0, y1) > by0)
        & (np.minimum(y0, y1) < by1)
        & (bx1 > bx0)
        & (by1 > by0)
    )
    dx, dy = x1 - x0, y1 - y0

    def side(cx: FloatArray, cy: FloatArray) -> FloatArray:
        return (cx - x0) * dy - (cy - y0) * dx

    corners = (side(bx0, by0), side(bx1, by0), side(bx0, by1), side(bx1, by1))
    low = np.minimum(np.minimum(corners[0], corners[1]), np.minimum(*corners[2:]))
    high = np.maximum(np.maximum(corners[0], corners[1]), np.maximum(*corners[2:]))
    return meet & (low < 0.0) & (high > 0.0)


def _segments_cross(a: FloatArray, b: FloatArray) -> NDArray[np.bool_]:
    """(k, j): whether segment k of ``a`` ((k, 4): x0, y0, x1, y1) and segment
    j of ``b`` ((j, 4)) cross properly (each one's ends on either side of the
    other's line)."""

    def side(s: FloatArray, px: FloatArray, py: FloatArray) -> FloatArray:
        return (px - s[:, 0, None]) * (s[:, 3, None] - s[:, 1, None]) - (
            py - s[:, 1, None]
        ) * (s[:, 2, None] - s[:, 0, None])

    a_of_b = side(a, b[None, :, 0], b[None, :, 1]) * side(
        a, b[None, :, 2], b[None, :, 3]
    )
    b_of_a = side(b, a[None, :, 0], a[None, :, 1]) * side(
        b, a[None, :, 2], a[None, :, 3]
    )
    return (a_of_b < 0.0) & (b_of_a.T < 0.0)


def place_labels(
    *,
    anchors: FloatArray,
    sizes: FloatArray,
    points: FloatArray,
    marker: float,
    gap: float,
    frame: tuple[float, float, float, float],
    obstacles: Sequence[tuple[float, float, float, float]] = (),
    lines: FloatArray | None = None,
    link_width: float = 0.0,
    limit: int = 200,
) -> list[Placement]:
    """A place for each label (see the module docstring).

    ``anchors`` (m, 2): each labelled point in pixels; ``sizes`` (m, 2): each
    label's width and height in pixels; ``points`` (n, 2): every drawn point in
    pixels; ``marker``: the points' half size in pixels; ``gap``: the room
    between a point's marker and the nearest label, in pixels; ``frame``: the plotting
    area (x0, y0, x1, y1); ``obstacles``: boxes of other texts inside it;
    ``lines`` ((k, 5), as in :func:`crossings`): straight pieces of lines drawn
    in the plot (the least-squares line), which a label should not lie across;
    ``link_width``: the width of the lines joining moved labels to their points,
    in pixels: a link may not cross another label, nor a label a link.
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
    search = _Search(
        anchors=anchors,
        sizes=sizes,
        points=points,
        marker=marker,
        first=first,
        frame=frame,
        obstacles=np.array(list(obstacles), dtype=np.float64).reshape(-1, 4),
        lines=(
            np.empty((0, 5), dtype=np.float64)
            if lines is None
            else np.asarray(lines, dtype=np.float64).reshape(-1, 5)
        ),
        half_link=link_width / 2.0,
    )
    for i in order:
        search.place(i)
    # A label placed early cannot see the links of labels placed after it:
    # place again, in the same order, each label that a line or link still
    # crosses or whose link crosses a label, keeping a new place only where it
    # is better with every other label where it now is.
    for _ in range(_REPAIRS):
        moved = False
        for i in order:
            if search.crossed(i):
                moved |= search.place(i, only_if_better=True)
        if not moved:
            break
    return search.placements()


_REPAIRS: Final = 2  # passes of placing crossed labels again

# Every candidate place in search order (ring by ring, CANDIDATES within a
# ring): its ring, step (label heights beyond the first ring), direction,
# horizontal start and vertical anchor (in label sizes).
_RING_OF: Final = np.repeat(np.arange(len(RINGS)), len(CANDIDATES))
_STEP: Final = np.repeat(np.asarray(RINGS, dtype=np.float64), len(CANDIDATES))
_UX: Final = np.tile(np.array([c[0] for c in CANDIDATES]), len(RINGS))
_UY: Final = np.tile(np.array([c[1] for c in CANDIDATES]), len(RINGS))
_HA: Final[list[Horizontal]] = [c[2] for c in CANDIDATES] * len(RINGS)
_START_OF: Final = np.array([_START[ha] for ha in _HA])
_ALIGN_OF: Final = np.array([_align(float(u)) for u in _UY])


@dataclasses.dataclass(slots=True)
class _Search:
    """The labels placed so far, and the scoring of a label's candidate places
    against them (the module docstring's order, all candidates at once)."""

    anchors: FloatArray
    sizes: FloatArray
    points: FloatArray
    marker: float
    first: float
    frame: tuple[float, float, float, float]
    obstacles: FloatArray
    lines: FloatArray
    half_link: float
    # Rows: the obstacles (texts) then each label, NaN where not placed; the
    # plot's lines then each label's link, NaN where it has none. NaN rows
    # overlap and cross nothing.
    boxes: FloatArray = dataclasses.field(init=False)
    strokes: FloatArray = dataclasses.field(init=False)
    chosen: dict[int, Placement] = dataclasses.field(
        default_factory=lambda: dict[int, Placement]()
    )

    def __post_init__(self) -> None:
        m = self.anchors.shape[0]
        self.boxes = np.vstack([self.obstacles, np.full((m, 4), np.nan)])
        self.strokes = np.vstack([self.lines, np.full((m, 5), np.nan)])

    def box_row(self, i: int) -> int:
        return self.obstacles.shape[0] + i

    def link_row(self, i: int) -> int:
        return self.lines.shape[0] + i

    def others(self, i: int) -> tuple[FloatArray, FloatArray]:
        """The boxes (texts, labels) and strokes (lines, links) of everything
        but label i."""
        boxes, strokes = self.boxes.copy(), self.strokes.copy()
        boxes[self.box_row(i)] = np.nan
        strokes[self.link_row(i)] = np.nan
        return boxes, strokes

    def scores(
        self, i: int, others: FloatArray, strokes: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """For every candidate place of label i: its box (c, 4), its link (c, 5;
        NaN on the first ring) and its score (c, 6): whether it overlaps another
        label or the edge, by how much, the lines across it and the labels its
        link crosses, the plot's lines its link crosses, the points it covers,
        its ring."""
        x, y = float(self.anchors[i, 0]), float(self.anchors[i, 1])
        w, h = float(self.sizes[i, 0]), float(self.sizes[i, 1])
        r = self.first + _STEP * h
        ax_, ay_ = x + r * _UX, y + r * _UY
        x0 = ax_ + _START_OF * w
        y0 = ay_ + _ALIGN_OF * h
        boxes = np.column_stack([x0, y0, x0 + w, y0 + h])
        fx0, fy0, fx1, fy1 = self.frame
        outside = (
            np.maximum(0.0, fx0 - boxes[:, 0] - _TOLERANCE)
            + np.maximum(0.0, boxes[:, 2] - fx1 - _TOLERANCE)
        ) * h + (
            np.maximum(0.0, fy0 - boxes[:, 1] - _TOLERANCE)
            + np.maximum(0.0, boxes[:, 3] - fy1 - _TOLERANCE)
        ) * w
        clash = np.zeros(boxes.shape[0])
        if others.shape[0]:
            # Each box grown by SPACING: labels keep a pixel apart.
            ow = np.minimum(boxes[:, None, 2] + SPACING, others[None, :, 2])
            ow = ow - np.maximum(boxes[:, None, 0] - SPACING, others[None, :, 0])
            oh = np.minimum(boxes[:, None, 3] + SPACING, others[None, :, 3])
            oh = oh - np.maximum(boxes[:, None, 1] - SPACING, others[None, :, 1])
            clash = np.where((ow > 0.0) & (oh > 0.0), ow * oh, 0.0).sum(axis=1)
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        near_x = np.abs(self.points[None, :, 0] - cx[:, None])
        near_y = np.abs(self.points[None, :, 1] - cy[:, None])
        covered = (
            (near_x < w / 2 + self.marker - _TOLERANCE)
            & (near_y < h / 2 + self.marker - _TOLERANCE)
        ).sum(axis=1)
        crossed = crossing(strokes, boxes).sum(axis=0).astype(np.float64)
        links = np.column_stack(
            [
                x + _UX * self.marker,
                y + _UY * self.marker,
                ax_,
                ay_,
                np.full(boxes.shape[0], self.half_link),
            ]
        )
        moved = _RING_OF > 0
        crossed[moved] += crossing(links[moved], others).sum(axis=1)
        # A link across the plot's lines (the least-squares line) clutters it,
        # less than a line through a label.
        across = np.zeros(boxes.shape[0])
        lines = self.lines[:, :4]
        if lines.shape[0]:
            across[moved] = _segments_cross(links[moved, :4], lines).sum(axis=1)
        links[~moved] = np.nan
        bad = outside + clash
        score = np.column_stack(
            [(bad > 0.0).astype(np.float64), bad, crossed, across, covered, _RING_OF]
        ).astype(np.float64)
        return boxes, links, score

    def place(self, i: int, *, only_if_better: bool = False) -> bool:
        """Put label i at its best place: the first that overlaps nothing, is
        crossed by nothing and covers no point, else the least bad (nearest
        first among equals); with ``only_if_better`` only where that beats its
        current place, scored against the same labels. Whether it moved."""
        others, strokes = self.others(i)
        boxes, links, score = self.scores(i, others, strokes)
        perfect = np.flatnonzero(np.all(score[:, [0, 2, 3, 4]] == 0.0, axis=1))
        if perfect.size:
            k = int(perfect[0])
        else:  # the least bad; the earliest among equals
            # lexsort sorts by its last key first; the index breaks ties.
            index = np.arange(score.shape[0], dtype=np.float64)
            keys: list[FloatArray] = [index]
            keys += [score[:, c] for c in range(score.shape[1] - 1, -1, -1)]
            k = int(np.lexsort(keys)[0])
        if only_if_better:
            current = self.boxes[self.box_row(i)]
            here = np.flatnonzero(np.all(boxes == current, axis=1))
            if here.size and not tuple(score[k]) < tuple(score[int(here[0])]):
                return False
        ux, uy = float(_UX[k]), float(_UY[k])
        r = self.first + float(_STEP[k]) * float(self.sizes[i, 1])
        place = Placement(
            dx=r * ux, dy=r * uy, ha=_HA[k], va=_va(uy), ring=int(_RING_OF[k])
        )
        self.chosen[i] = place
        self.boxes[self.box_row(i)] = boxes[k]
        self.strokes[self.link_row(i)] = links[k]  # NaN on the first ring
        return True

    def crossed(self, i: int) -> bool:
        """Whether a line or another label's link crosses label i where it is,
        or its own link crosses another label."""
        others, strokes = self.others(i)
        if crossing(strokes, self.boxes[self.box_row(i)][None]).any():
            return True
        link = self.strokes[self.link_row(i)]
        if bool(crossing(link[None], others).any()):
            return True
        lines = self.lines[:, :4]
        return bool(lines.shape[0] and _segments_cross(link[None, :4], lines).any())

    def placements(self) -> list[Placement]:
        return [self.chosen[i] for i in range(self.anchors.shape[0])]
