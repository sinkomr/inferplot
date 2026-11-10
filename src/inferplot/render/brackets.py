"""Where significance brackets go: stacking levels and the y-axis layout.

Pure geometry, no matplotlib. Our own algorithm (rendering rules, D38):

* brackets are processed shortest span first and each takes the lowest level
  above every bracket it encloses at which neither it nor its label comes within
  ``gap`` of another bracket or label on that level;
* the bracket stack is laid out in pixels, not in data units: a level is as tall
  as a bracket's legs, its label, the padding on both sides of the label and the
  line width, so a label never reaches the legs of the level above at any data
  range;
* the first level sits above the data and above every centrality label; the
  y-limits are extended so that every label fits.

With inferplot's own figure the data region gets a fixed minimum height and the
Axes grows by exactly what the stack needs. In a given Axes of fixed height the
stack keeps its full pixel spacing while the data keep that minimum height or
half of the height left after the subtitle and caption, whichever is less; below
that the levels are pushed closer together and ``check()`` reports the labels
that then cross a line. So an Axes as tall as inferplot's own never pushes them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DATA_PAD",
    "MIN_DATA_SHARE",
    "Bracket",
    "VerticalLayout",
    "level_step_px",
    "stack",
    "vertical_layout",
]

DATA_PAD: Final = 0.05  # data range fraction of space below and above the data
# In a given Axes, the data keep the minimum data height or this share of the
# height left after the subtitle and caption, whichever is less, before the bracket
# levels are pushed closer together.
MIN_DATA_SHARE: Final = 0.5
# Data ranges laid out as they are; others are scaled by a power of two first.
_UNSCALED: Final = (2.0**-60, 2.0**60)


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


def level_step_px(
    *, label_px: float, pad_px: float, tip_px: float, line_px: float
) -> float:
    """Pixels between two bracket levels.

    From a bar upwards: ``pad_px`` to its label, the label, ``pad_px`` plus half a
    line width of clearance, and the legs of the next bracket (``tip_px`` long,
    their projecting caps half a line width more).
    """
    return pad_px + label_px + pad_px + line_px + tip_px


@dataclass(frozen=True, slots=True)
class VerticalLayout:
    """y-limits, bracket level heights and the scale (pixels per data unit)."""

    bottom: float
    top: float
    ppu: float
    levels: tuple[float, ...]  # y of each bracket level
    tip: float  # leg length in data units
    label_pad: float  # data units between a bracket and its label
    step_px: float  # pixels between two levels as laid out (full or pushed closer)
    data_px: float  # pixels of the data region (data, padding, centrality labels)

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
        product = lo * hi
        # The geometric mean; for scales near the ends of the double range the
        # product underflows, so take the roots first there.
        mid = math.sqrt(product) if product > 0 else math.sqrt(lo) * math.sqrt(hi)
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
    tip_px: float,
    line_px: float,
    top_px: float = 0.0,
    bottom_px: float = 0.0,
    min_height_px: float | None = None,
    data_height_px: float,
    height_px: float | None = None,
    ppu: float | None = None,
) -> VerticalLayout:
    """The y-axis layout for data in [data_min, data_max].

    ``centers`` are the centrality values whose labels (``center_half_px`` half
    height) must stay below the brackets; ``label_px`` is the bracket label height,
    ``pad_px`` the padding around it, ``tip_px`` the leg length and ``line_px`` the
    bracket line width, all in pixels. ``top_px`` / ``bottom_px`` reserve room
    inside the axes (a subtitle or caption drawn inside a given Axes).

    The data region (data, padding and centrality labels) gets at least
    ``data_height_px`` and the bracket levels sit :func:`level_step_px` apart on
    top of it. Give ``min_height_px`` to choose the axes height (at least that), or
    ``height_px`` to fit an Axes of fixed height; there the levels keep their full
    spacing while the data keep ``data_height_px`` or :data:`MIN_DATA_SHARE` of
    the height left after ``top_px`` and ``bottom_px``, whichever is less. Give
    ``ppu`` (pixels per data unit) to lay everything out at that scale with the
    levels at full spacing (panels sharing a y axis share one scale).
    """
    r = _data_range(data_min, data_max)
    if not _UNSCALED[0] <= r <= _UNSCALED[1]:
        # Far from 1 (y x 1e200, y x 1e-200), the search over pixels per data
        # unit leaves the double range: lay out the data scaled by a power of two
        # near 1 / r, which is exact, and scale the layout back.
        scale = math.ldexp(1.0, -math.frexp(r)[1])
        inner = vertical_layout(
            data_min=data_min * scale,
            data_max=data_max * scale,
            centers=[c * scale for c in centers],
            n_levels=n_levels,
            center_half_px=center_half_px,
            label_px=label_px,
            pad_px=pad_px,
            tip_px=tip_px,
            line_px=line_px,
            top_px=top_px,
            bottom_px=bottom_px,
            min_height_px=min_height_px,
            data_height_px=data_height_px,
            height_px=height_px,
            ppu=None if ppu is None else ppu / scale,
        )
        return VerticalLayout(
            bottom=inner.bottom / scale,
            top=inner.top / scale,
            ppu=inner.ppu * scale,
            levels=tuple(y / scale for y in inner.levels),
            tip=inner.tip / scale,
            label_pad=inner.label_pad / scale,
            step_px=inner.step_px,
            data_px=inner.data_px,
        )
    full_step = level_step_px(
        label_px=label_px, pad_px=pad_px, tip_px=tip_px, line_px=line_px
    )
    has_centers = len(centers) > 0
    c_hi = max(centers) if has_centers else data_max
    c_lo = min(centers) if has_centers else data_min
    reach = center_half_px + pad_px if has_centers else 0.0

    def ceiling(ppu: float) -> float:
        """Top of the data region: the data, their padding, centrality labels."""
        value = data_max + DATA_PAD * r
        return max(value, c_hi + reach / ppu) if has_centers else value

    def floor(ppu: float) -> float:
        value = data_min - DATA_PAD * r
        return min(value, c_lo - reach / ppu) if has_centers else value

    def y0(ppu: float) -> float:
        # The legs of the first bracket (with their caps) end at the ceiling.
        return ceiling(ppu) + (tip_px + line_px / 2) / ppu

    def top(ppu: float, step: float) -> float:
        if n_levels > 0:
            value = y0(ppu) + ((n_levels - 1) * step + 2 * pad_px + label_px) / ppu
        else:
            value = ceiling(ppu)
        return value + top_px / ppu

    def bottom(ppu: float) -> float:
        return floor(ppu) - bottom_px / ppu

    def height(ppu: float, step: float = full_step) -> float:
        return (top(ppu, step) - bottom(ppu)) * ppu

    def data_px(ppu: float) -> float:
        return (ceiling(ppu) - floor(ppu)) * ppu

    step = full_step
    if ppu is not None:
        pass
    elif height_px is not None:
        available = height_px - top_px - bottom_px
        floor_px = max(min(data_height_px, MIN_DATA_SHARE * available), 1.0)
        found = _solve(height, height_px)
        if found is None or data_px(found) < floor_px * (1 - 1e-9):
            # The stack at full spacing leaves the data too little room: give the
            # data their share and push the levels closer together; check()
            # reports the labels that then cross a line, or do not fit.
            share = _solve(data_px, floor_px)
            ppu = share if share is not None else MIN_DATA_SHARE * height_px / r
            spare = height(ppu, 0.0)
            per_step = height(ppu, 1.0) - spare
            step = 0.0
            if per_step > 0 and height_px > spare:
                step = min(full_step, (height_px - spare) / per_step)
        else:
            ppu = found
    else:
        if min_height_px is None:
            raise ValueError("give min_height_px or height_px")
        candidates = [
            p
            for p in (_solve(data_px, data_height_px), _solve(height, min_height_px))
            if p is not None
        ]
        ppu = max(candidates) if candidates else data_height_px / (1.2 * r)
    levels = tuple(y0(ppu) + n * step / ppu for n in range(n_levels))
    return VerticalLayout(
        bottom=bottom(ppu),
        top=top(ppu, step),
        ppu=ppu,
        levels=levels,
        tip=tip_px / ppu,
        label_pad=pad_px / ppu,
        step_px=step,
        data_px=data_px(ppu),
    )
