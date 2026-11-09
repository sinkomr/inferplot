"""Where significance brackets go: stacking levels and the y-axis layout.

Pure geometry, no matplotlib. Our own algorithm (rendering rules):

* brackets are processed shortest span first and each takes the lowest level
  above every bracket it encloses at which neither it nor its label comes within
  ``gap`` of another bracket or label on that level;
* levels are spaced by 7% of the data range, starting above the data and above
  every centrality label;
* the y-limits are extended so that every label fits, and the axes are made tall
  enough (pixels per data unit) that a label never reaches the next level.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "LEVEL_STEP",
    "TIP",
    "Bracket",
    "VerticalLayout",
    "stack",
    "vertical_layout",
]

# Fractions of the data range.
LEVEL_STEP: Final = 0.07  # between bracket levels (rendering rules)
TIP: Final = 0.02  # length of a bracket's legs
DATA_PAD: Final = 0.05  # space below the data and above it when there is no bracket


@dataclass(frozen=True, slots=True)
class Bracket:
    """A bracket from x position ``i`` to ``j``; its label is ``width`` data units."""

    i: int
    j: int
    width: float

    def extent(self) -> tuple[float, float]:
        """Horizontal extent of the bracket and its centered label."""
        mid = (self.i + self.j) / 2
        return min(self.i, mid - self.width / 2), max(self.j, mid + self.width / 2)


def stack(brackets: Sequence[Bracket], *, gap: float) -> tuple[int, ...]:
    """Level (0 = lowest) of each bracket, in input order.

    Shortest spans first (ties: leftmost first). Each bracket takes the lowest level
    above every bracket it encloses at which it and its label keep at least ``gap``
    data units from the other items on that level, so brackets sharing an end point
    never share a level.
    """
    order = sorted(
        range(len(brackets)),
        key=lambda n: (brackets[n].j - brackets[n].i, brackets[n].i, brackets[n].j),
    )
    occupied: list[list[tuple[float, float]]] = []
    levels = [0] * len(brackets)
    placed: list[int] = []
    for n in order:
        this = brackets[n]
        lo, hi = this.extent()
        floor = 1 + max(
            (
                levels[m]
                for m in placed
                if this.i <= brackets[m].i and brackets[m].j <= this.j
            ),
            default=-1,
        )
        level = floor
        while level < len(occupied) and not all(
            hi + gap <= a or b + gap <= lo for a, b in occupied[level]
        ):
            level += 1
        while level >= len(occupied):
            occupied.append([])
        occupied[level].append((lo, hi))
        levels[n] = level
        placed.append(n)
    return tuple(levels)


@dataclass(frozen=True, slots=True)
class VerticalLayout:
    """y-limits, bracket level heights and the scale (pixels per data unit)."""

    bottom: float
    top: float
    ppu: float
    levels: tuple[float, ...]  # y of each bracket level
    tip: float  # leg length in data units
    label_pad: float  # data units between a bracket and its label

    @property
    def height_px(self) -> float:
        return (self.top - self.bottom) * self.ppu


def _data_range(data_min: float, data_max: float) -> float:
    span = data_max - data_min
    if span > 0 and math.isfinite(span):
        return span
    return max(abs(data_max) * 0.1, 1.0)


def _solve(height: Callable[[float], float], target: float) -> float | None:
    """ppu with height(ppu) == target for an increasing ``height``; None if none."""
    lo, hi = 1e-300, 1.0
    if height(lo) >= target:
        return None
    while height(hi) < target:
        hi *= 2.0
        if hi > 1e300:
            return None
    for _ in range(200):
        mid = math.sqrt(lo * hi) if lo > 0 else hi / 2
        if height(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12 * hi:
            break
    return hi


def vertical_layout(
    *,
    data_min: float,
    data_max: float,
    centers: Sequence[float],
    n_levels: int,
    center_half_px: float,
    label_px: float,
    pad_px: float,
    top_px: float = 0.0,
    bottom_px: float = 0.0,
    min_height_px: float | None = None,
    height_px: float | None = None,
) -> VerticalLayout:
    """The y-axis layout for data in [data_min, data_max].

    ``centers`` are the centrality values whose labels (``center_half_px`` half
    height) must stay below the brackets; ``label_px`` is the bracket label height;
    ``top_px`` / ``bottom_px`` reserve room inside the axes (a subtitle or caption
    drawn inside a given Axes). Give ``min_height_px`` to choose the axes height
    (at least that, and tall enough that labels never reach the next level), or
    ``height_px`` to fit an Axes of fixed height.
    """
    r = _data_range(data_min, data_max)
    step, tip = LEVEL_STEP * r, TIP * r
    has_centers = len(centers) > 0
    c_hi = max(centers) if has_centers else data_max
    c_lo = min(centers) if has_centers else data_min
    reach = center_half_px + pad_px if has_centers else 0.0

    def y0(ppu: float) -> float:
        base = data_max + step
        return max(base, c_hi + reach / ppu + tip) if has_centers else base

    def top(ppu: float) -> float:
        if n_levels > 0:
            value = y0(ppu) + (n_levels - 1) * step + (2 * pad_px + label_px) / ppu
        else:
            value = data_max + DATA_PAD * r
            if has_centers:
                value = max(value, c_hi + reach / ppu)
        return value + top_px / ppu

    def bottom(ppu: float) -> float:
        value = data_min - DATA_PAD * r
        if has_centers:
            value = min(value, c_lo - reach / ppu)
        return value - bottom_px / ppu

    def height(ppu: float) -> float:
        return (top(ppu) - bottom(ppu)) * ppu

    if height_px is not None:
        found = _solve(height, height_px)
        # The reserved pixels alone overflow the Axes: give the data half of it and
        # let check() report what does not fit.
        ppu = found if found is not None else 0.5 * height_px / (1.2 * r)
    else:
        if min_height_px is None:
            raise ValueError("give min_height_px or height_px")
        # A label plus its padding must fit between two levels (legs included).
        need = (2 * pad_px + label_px) / ((LEVEL_STEP - TIP) * r)
        ppu = need if n_levels > 1 else 0.0
        found = _solve(height, min_height_px)
        if found is not None:
            ppu = max(ppu, found)
        if ppu <= 0.0:
            ppu = min_height_px / (1.2 * r)
    levels = tuple(y0(ppu) + n * step for n in range(n_levels))
    return VerticalLayout(
        bottom=bottom(ppu),
        top=top(ppu),
        ppu=ppu,
        levels=levels,
        tip=tip,
        label_pad=pad_px / ppu,
    )
