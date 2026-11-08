"""Descriptive statistics: the correctly rounded mean and the sample variance.

Centrality labels print the mean to ``digits`` decimals, so a mean that sits on a
decimal tie (12.225) prints differently when it is one unit in the last place off
("12.23" instead of "12.22"). ``mean`` therefore returns the arithmetic mean
correctly rounded: the exact rational sum of the doubles divided by n, rounded once
to the nearest double (ties to even). R's ``mean`` (a long-double sum refined by a
second pass) agrees with it on every fixture label; NumPy's pairwise sum and
``scipy.stats.trim_mean`` differ in the last bit on 26 of 100 fixture labels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
SampleLike: TypeAlias = NDArray[Any] | Sequence[float]
"""A sample: a one-dimensional NumPy array or a sequence of floats."""

__all__ = ["SampleLike", "as_sample", "mean", "variance"]

# Each double is m * 2^(e - 53) with an integer |m| < 2^53 (numpy.frexp). m is split
# into a high part (multiple of 2^27, as an integer < 2^27 in magnitude) and a low part
# in [0, 2^27): float64 sums of up to 2^26 such integers are exact (< 2^53).
_SPLIT: Final = 2.0**27
_CHUNK: Final = 1 << 26


def as_sample(x: SampleLike, *, name: str = "x", min_size: int = 1) -> FloatArray:
    """``x`` as a one-dimensional float64 array of finite values (ValueError if not)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {arr.shape}")
    if arr.size < min_size:
        raise ValueError(f"{name} needs at least {min_size} value(s); got {arr.size}")
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _exact_sum_scaled(x: FloatArray) -> tuple[int, int]:
    """The exact sum of ``x`` as ``(N, s)`` with sum = N * 2^s (Python integers)."""
    mantissa, exponent = np.frexp(x)
    if x.size == 0:
        return 0, 0
    e_min = int(exponent.min())
    total = 0
    for start in range(0, x.size, _CHUNK):
        m = mantissa[start : start + _CHUNK] * 2.0**53  # exact integers
        hi = np.floor(m / _SPLIT)  # exact: division by a power of two, then floor
        lo = m - hi * _SPLIT  # exact, in [0, 2^27)
        index = (exponent[start : start + _CHUNK] - e_min).astype(np.intp)
        sum_hi = np.bincount(index, weights=hi)
        sum_lo = np.bincount(index, weights=lo)
        for shift in ((sum_hi != 0.0) | (sum_lo != 0.0)).nonzero()[0].tolist():
            exact = (int(sum_hi[shift]) << 27) + int(sum_lo[shift])
            total += exact << shift
    return total, e_min - 53


def mean(x: SampleLike) -> float:
    """Arithmetic mean, correctly rounded (exact sum / n, one rounding).

    The value R's ``base::mean`` aims at (a long-double sum with a correction pass);
    computed exactly here: every double is an integer times a power of two, the
    integers are summed exactly per exponent, and Python's integer true division
    rounds the quotient once (correctly rounded, ties to even).

    Raises:
        ValueError: ``x`` is empty, not one-dimensional or not finite.
    """
    arr = as_sample(x)
    numerator, scale = _exact_sum_scaled(arr)
    n = arr.size
    if scale >= 0:
        return float((numerator << scale) / n)
    return float(numerator / (n << -scale))


def variance(x: SampleLike) -> float:
    """Sample variance with denominator n - 1, R's ``stats::var``.

    Corrected two-pass algorithm (Chan, Golub & LeVeque 1983, "Algorithms for
    computing the sample variance", The American Statistician 37: 242-247, eq. 1.7):
    sum((x - m)^2) minus the rounding correction (sum(x - m))^2 / n, with m the
    correctly rounded mean.

    Raises:
        ValueError: fewer than two values, not one-dimensional or not finite.
    """
    arr = as_sample(x, min_size=2)
    centred = arr - mean(arr)
    n = arr.size
    correction = math.fsum(centred.tolist()) ** 2 / n
    squares = math.fsum((centred * centred).tolist())
    return max(0.0, (squares - correction) / (n - 1))
