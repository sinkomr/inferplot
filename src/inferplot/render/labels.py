# NumPy's stubs type several functions taking array-likes as partly Unknown, which
# strict mode reports on every call; the arguments we pass are all float arrays.
# pyright: reportUnknownMemberType=false
"""When a centrality label reads as another level's (SPEC section 5).

Pure geometry in pixels, shared by the drawing (which places each label where it
reads as its own) and ``fig.check()`` (``centrality_label_ambiguous``, which reports
a label that does not, wherever it ended up).

Every centrality label has a usual side: right of its point, as in ggstatsplot,
except the first of two within-subjects conditions (left, the space between the
two being the paths'). On that side a label may reach across the gap to the
neighbouring point, as every label does in the same direction, and is read as its
own because each point has its label on the same side. A label reads as another
level's when

* on its usual side, its box's middle lies beyond the neighbouring point (most of
  the label sits on that point's far side);
* on the other side, its box reaches past the middle between its point and the
  neighbouring one (it sits beside the neighbour, where that neighbour's own
  label would be expected);
* it covers part of the centrality path between two other conditions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from matplotlib.transforms import BboxBase

__all__ = ["Ambiguity", "ambiguities", "covered"]

FloatArray = NDArray[np.float64]
Side = Literal["right", "left"]

# Pixels a box may pass a line or a middle before it counts (anti-aliasing slack).
_TOLERANCE: Final = 0.5
# Pixels of a path's length a box may cover before it counts.
_PATH_TOLERANCE: Final = 1.0
_STEP: Final = 0.25  # pixels between the samples along a path


def covered(pieces: FloatArray, boxes: Sequence[BboxBase]) -> FloatArray:
    """Pixels of each straight piece that lie under each box.

    ``pieces`` is (m, 2, 2): m pieces from (x0, y0) to (x1, y1) in pixels. Returns
    an (m, len(boxes)) array of covered lengths, sampled at the midpoints of
    stretches of at most a quarter pixel; a piece with a NaN end (where a path
    breaks) covers nothing. A box counts only where a sample is more than half a
    pixel inside it.
    """
    out = np.zeros((pieces.shape[0], len(boxes)), dtype=np.float64)
    finite = np.isfinite(pieces).all(axis=(1, 2))
    if not boxes or not finite.any():
        return out
    start, stop = pieces[finite, 0, :], pieces[finite, 1, :]
    delta = stop - start
    length = np.hypot(delta[:, 0], delta[:, 1])
    steps = max(1, int(np.ceil(float(length.max()) / _STEP)))
    t = (np.arange(steps) + 0.5) / steps
    xs = start[:, None, 0] + delta[:, None, 0] * t[None, :]
    ys = start[:, None, 1] + delta[:, None, 1] * t[None, :]
    found = np.empty((start.shape[0], len(boxes)), dtype=np.float64)
    tol = _TOLERANCE
    for b, box in enumerate(boxes):
        inside = (
            (xs > box.x0 + tol)
            & (xs < box.x1 - tol)
            & (ys > box.y0 + tol)
            & (ys < box.y1 - tol)
        )
        found[:, b] = inside.mean(axis=1) * length
    out[finite] = found
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class Ambiguity:
    """Why a label reads as another level's: ``kind`` "beyond" (on its usual side,
    mostly past the point of level ``other``), "beside" (on the other side, past
    the middle toward the point of ``other``) or "path" (over the centrality path
    from level ``other`` to level ``other + 1``)."""

    kind: Literal["beyond", "beside", "path"]
    other: int


def ambiguities(
    i: int,
    box: BboxBase,
    usual: Side,
    points: Mapping[int, float],
    path: FloatArray | None = None,
) -> list[Ambiguity]:
    """Why the label of level ``i``, with its box at ``box``, reads as another
    level's; empty when it reads as its own.

    ``points`` maps each level with a drawn centrality point to the point's x in
    pixels (level ``i`` among them); ``path`` is the centrality path's pieces in
    pixels, (k - 1, 2, 2), piece ``j`` from level ``j`` to level ``j + 1`` (NaN
    where it breaks), or None without one.
    """
    x = points[i]
    middle = (box.x0 + box.x1) / 2
    found: list[Ambiguity] = []
    right = [j for j in points if points[j] > x]
    left = [j for j in points if points[j] < x]
    if right:
        j = min(right, key=lambda n: points[n])
        if usual == "right":
            if middle > points[j] + _TOLERANCE:
                found.append(Ambiguity("beyond", j))
        elif box.x1 > (x + points[j]) / 2 + _TOLERANCE:
            found.append(Ambiguity("beside", j))
    if left:
        j = max(left, key=lambda n: points[n])
        if usual == "left":
            if middle < points[j] - _TOLERANCE:
                found.append(Ambiguity("beyond", j))
        elif box.x0 < (x + points[j]) / 2 - _TOLERANCE:
            found.append(Ambiguity("beside", j))
    if path is not None and path.shape[0] > 0:
        lengths = covered(path, [box])[:, 0]
        for j in np.flatnonzero(lengths > _PATH_TOLERANCE):
            if i not in (int(j), int(j) + 1):
                found.append(Ambiguity("path", int(j)))
    return found
