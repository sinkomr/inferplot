# NumPy's stubs leave the overloads of unique partially unknown in strict mode.
# pyright: reportUnknownMemberType=false
"""Numbers as written: the decimal a stored double stands for.

A double cannot hold most decimals: the 0.3 of a data file is stored as
0.299999999999999988898, and quantities that are equal in decimal arithmetic come
out unequal in binary (0.2 + 0.4 and 2 x 0.3 differ in their last place). Where a
method makes an exact decision on such a quantity (is this bootstrap resample 0 / 0
or infinite, is this T* at least |t|, is t exactly 0), the decision on the doubles
answers a question about the rounding, not about the data, and the result then
depends on whether the same data were written in whole numbers or in tenths
(DECISIONS.md D74: numbers that are the same as written are the same).

``written_value`` reads every double as the number it was written as: the shortest
decimal within two units in its last place, provided that decimal is coarse next to
the rounding (its last digit worth at least 2^20 units in the last place of the
value); otherwise the double itself. Exact decisions are then made in rational
arithmetic on these readings, so they are the decisions the same data give in
whole numbers, whatever decimal unit or shift they were written in.

* Two units in the last place covers a decimal converted from text (at most half a
  unit) and one computed from such decimals by one or two operations (``k * 0.3``
  is at most 0.8 units off 3k/10, ``k * 1.1`` and ``(k + 0.1) * 2.54`` at most
  1.2 units, measured over k = 1 to 20,000).
* A decimal whose last digit is worth at least 2^20 units in the last place has at
  most about 9 significant digits next to a double's 16 (whole numbers up to 2^33
  in the same way: ``rounding_tolerance``'s rule). A value measured or computed to
  many digits lies within two units of such a decimal by chance at most once in
  about 2^18 values, and is then read at most two units away; otherwise it is read
  as the double itself, so a genuine difference in such data, however small, stays
  a difference.
* A double whose significand ends in 20 zero bits (a half, a quarter:
  ``rounding_tolerance`` 0) is exact and is read as itself, and so is every whole
  number an integer column can hold (below 2^64: 5e15 + 1 is not 5e15).

Readings are symmetric (the reading of -v is minus that of v) and scale with
decimal units: v / 10, v + 0.1 and v x 2.54 are read as the decimals they stand
for, so the decisions do not change. They do not scale with powers of two beyond
what keeps the decimal short (0.3 / 2^30 has 24 significant digits and is read as
its double).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from inferplot.stats.descriptive import FloatArray, rounding_tolerance

__all__ = [
    "READ_ULPS",
    "all_equal_as_written",
    "as_written",
    "equal_as_written",
    "written_trim",
    "written_value",
    "written_values",
]

READ_ULPS: Final = 2
"""How many units in its last place a double may lie from the decimal it reads as."""

# The decimal's last digit must be worth at least 2^20 units in the last place of
# the value (the same 20 bits as ``rounding_tolerance``'s rule for exact values).
_MIN_UNIT_ULPS: Final = 2.0**20


# Whole numbers below this are read as themselves: everything an integer column
# can hold (64 bits).
_WHOLE_BELOW: Final = 2.0**64


def _whole(values: FloatArray) -> NDArray[np.bool_]:
    """Which values are whole numbers an integer column can hold (below 2^64),
    which are read as themselves. From 2^51 on, neighbouring whole numbers lie
    within two units in the last place of a round one (5e15 + 1 of 5e15; a
    nanosecond timestamp of the next one), and they are different numbers
    (counts, identifiers, timestamps). Beyond 2^64 a whole-valued double is a
    decimal in scientific notation (1.7e30) and is read as that decimal."""
    return (values == np.rint(values)) & (np.abs(values) < _WHOLE_BELOW)


def _exact_check(value: float, k: int, q: int) -> Fraction | None:
    """k x 10^q when it lies within READ_ULPS units in the last place of
    ``value``, decided exactly; else None."""
    candidate = Fraction(k) * Fraction(10) ** q
    if abs(Fraction(value) - candidate) <= READ_ULPS * Fraction(math.ulp(value)):
        return candidate
    return None


def written_values(values: FloatArray) -> list[Fraction]:
    """``written_value`` of every element of ``values`` (finite doubles), in their
    order. Each distinct value is read once; a fast floating-point screen leaves
    the exact check to the values that lie near a decimal."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    unique, inverse = np.unique(arr, return_inverse=True)
    readings: list[Fraction] = [Fraction(v) for v in unique.tolist()]
    inexact = ((rounding_tolerance(unique) > 0.0) & ~_whole(unique)).nonzero()[0]
    if inexact.size:
        magnitude = np.abs(unique[inexact])
        ulps: FloatArray = np.spacing(magnitude)
        finest = _MIN_UNIT_ULPS * ulps  # the smallest admissible decimal unit
        # Decimal units from 1e-290 to 1e308, which 10.0 ** q and 10.0 ** -q hold;
        # a value below about 1e-280 is read as its double.
        top = min(math.floor(math.log10(float(magnitude.max()))) + 1, 308)
        bottom = max(math.floor(math.log10(float(finest.min()))), -290)
        pending: NDArray[np.bool_] = np.ones(inexact.size, dtype=bool)
        for q in range(top, bottom - 1, -1):
            unit = 10.0**q
            active = pending & (unit >= finest)
            if not bool(active.any()):
                continue
            v = magnitude[active]
            # Near the largest double k x unit can overflow: an infinite residual
            # is simply not near.
            with np.errstate(over="ignore", invalid="ignore"):
                if q >= 0:
                    k = np.rint(v / unit)
                    residual = np.abs(v - k * unit)
                else:
                    factor = 10.0 ** (-q)
                    k = np.rint(v * factor)
                    residual = np.abs(v - k / factor)
            near = residual <= (READ_ULPS + 1) * ulps[active]
            if not bool(near.any()):
                continue
            positions = active.nonzero()[0][near]
            for position, k_value in zip(
                positions.tolist(), k[near].tolist(), strict=True
            ):
                index = int(inexact[position])
                value = float(unique[index])
                reading = _exact_check(abs(value), int(k_value), q)
                if reading is None:
                    continue
                readings[index] = reading if value > 0 else -reading
                pending[position] = False
    return [readings[i] for i in inverse.reshape(-1).tolist()]


def written_value(value: float) -> Fraction:
    """The number the double ``value`` stands for as written: the shortest decimal
    within ``READ_ULPS`` units in its last place whose last digit is worth at least
    2^20 of those units, or the double itself (module docstring)."""
    return written_values(np.array([float(value)]))[0]


def written_trim(values: Sequence[Fraction], g: int) -> tuple[Fraction, Fraction]:
    """The trimmed mean and the winsorized sum of squares (about the winsorized
    mean) of exact values, g trimmed and winsorized at each end, exactly."""
    ordered = sorted(values)
    m = len(ordered)
    middle = ordered[g : m - g]
    tmean = sum(middle, Fraction(0)) / len(middle)
    winsorized = [ordered[g]] * g + middle + [ordered[m - g - 1]] * g
    centre = sum(winsorized, Fraction(0)) / m
    return tmean, sum(((v - centre) ** 2 for v in winsorized), Fraction(0))


def equal_as_written(values: FloatArray, value: float) -> NDArray[np.bool_]:
    """Which of ``values`` equal ``value`` as written (``written_values``): 0.1 +
    0.2 equals 0.3, as 3 x 0.3 equals 0.9."""
    target = written_value(value)
    return np.array([r == target for r in written_values(values)], dtype=bool)


def as_written(values: FloatArray) -> FloatArray:
    """Each value as the double nearest the decimal it stands for
    (``written_values``): 0.1 + 0.2 becomes the double 0.3, so that two values
    that are the same number as written are the same double.

    The rank tests rank these (DECISIONS.md D126): a tie, an order and a zero
    difference are then what the data say, not what the rounding of a computed
    column says. A value converted from text of up to 15 significant digits is
    already that double and is returned unchanged, as is every whole number
    below 2^64 and every value without a short decimal reading; a changed value
    moves by at most ``READ_ULPS`` units in its last place (text of 16 or 17
    digits that spells out
    such a neighbour, "0.30000000000000004", is read as the short decimal too).
    Applying it twice changes nothing more.

    Values that cannot change are recognized without rational arithmetic
    (``_surely_unchanged``), so a million two-decimal values cost milliseconds;
    only the others are read one by one.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = arr.copy()
    todo = ~_surely_unchanged(arr)
    if bool(todo.any()):
        out[todo] = [float(r) for r in written_values(arr[todo])]
    return out


# Decimals of at most this many significant digits are recognized by
# _surely_unchanged: their integer significand stays below 1e15 < 2^53, and
# v x 10^m, off by at most two parts in 2^53, rounds to it (error under 0.23).
_SCREEN_DIGITS: Final = 14
# 10^m is an exact double for |m| <= 22.
_EXACT_POWER: Final = 22


def _surely_unchanged(arr: FloatArray) -> NDArray[np.bool_]:
    """Which values ``as_written`` returns as they are, decided in floating point:
    whole numbers and other exact values (``written_values`` reads them as
    themselves), and every double that is the one nearest a decimal of at most
    ``_SCREEN_DIGITS`` significant digits.

    The last holds because such a double v is read either as itself or as a
    decimal within two units in its last place, and no other decimal that short
    lies that near (two of them differ by at least 1e-14 |v|, 45 units), so a
    decimal reading is the decimal v is nearest to, whose double is v. The test
    is exact: with m chosen so that v x 10^m has ``_SCREEN_DIGITS`` digits before
    the point, k = rint(v x 10^m) is the decimal's significand whenever v is
    nearest such a decimal, and k / 10^m (or k x 10^-m) is one correctly rounded
    operation on two exact doubles, the double nearest that decimal. Where 10^m
    is not exact (values below about 1e-9 or above 1e35) nothing is claimed.
    """
    magnitude = np.abs(arr)
    sure: NDArray[np.bool_] = _whole(arr) | (rounding_tolerance(arr) == 0.0)
    rest = ~sure & np.isfinite(arr)
    if not bool(rest.any()):
        return sure
    v = magnitude[rest]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (_SCREEN_DIGITS - 1) - np.floor(np.log10(v))
    usable = np.isfinite(m) & (np.abs(m) <= _EXACT_POWER)
    m = np.where(usable, m, 0.0)
    scale = 10.0 ** np.abs(m)
    down = m >= 0  # v x 10^m, undone by a division
    k = np.rint(np.where(down, v * scale, v / scale))
    back = np.where(down, k / scale, k * scale)
    nearest = usable & (back == v) & (k < 10.0 ** (_SCREEN_DIGITS + 1))
    sure[rest.nonzero()[0][nearest]] = True
    return sure


def all_equal_as_written(values: FloatArray, value: float) -> bool:
    """Whether every one of ``values`` equals ``value`` as written."""
    return bool(equal_as_written(values, value).all())
