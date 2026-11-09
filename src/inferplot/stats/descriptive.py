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
from fractions import Fraction
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
SampleLike: TypeAlias = NDArray[Any] | Sequence[float]
"""A sample: a one-dimensional NumPy array or a sequence of floats."""

__all__ = [
    "SampleLike",
    "as_sample",
    "common_scale",
    "deviations_from_weighted_mean",
    "exact_mean",
    "exact_variance",
    "mean",
    "scaled_variance",
    "variance",
]

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


def exact_mean(x: FloatArray) -> Fraction:
    """The arithmetic mean of ``x`` (a non-empty, finite float64 array) as an exact
    rational: the exact sum of the doubles over n, not rounded.

    Differences of means are formed from these and rounded once,
    ``float(exact_mean(a) - exact_mean(b))``: the correctly rounded difference of
    the true means. Subtracting the two rounded means instead loses up to half a
    unit in the last place of each, an error of about ulp(mean) / |difference|
    relative, which is large when the means differ by about the rounding of
    their common level (DECISIONS.md, the exact-difference rule).
    """
    numerator, scale = _exact_sum_scaled(x)
    if scale >= 0:
        return Fraction(numerator << scale, x.size)
    return Fraction(numerator, x.size << -scale)


def deviations_from_weighted_mean(
    weights: Sequence[float], means: Sequence[Fraction]
) -> list[float]:
    """m_i - c for every i, c = sum w_i m_i / sum w_i, each correctly rounded.

    ``weights`` are doubles (> 0) and ``means`` exact rationals (``exact_mean``);
    the weighted centre and every deviation from it are formed exactly and rounded
    once, so a deviation keeps its relative accuracy however close the means lie
    next to their common level.
    """
    exact_weights = [Fraction(w) for w in weights]
    centre = sum(
        (w * m for w, m in zip(exact_weights, means, strict=True)), Fraction(0)
    ) / sum(exact_weights, Fraction(0))
    return [float(m - centre) for m in means]


def exact_variance(values: Sequence[Fraction]) -> float:
    """The sample variance (denominator n - 1) of exact rationals, computed exactly
    and rounded once (at least two values)."""
    n = len(values)
    centre = sum(values, Fraction(0)) / n
    return float(sum(((v - centre) ** 2 for v in values), Fraction(0)) / (n - 1))


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


def common_scale(*samples: FloatArray) -> float:
    """The power of two 2^-e that brings the largest |value| of ``samples`` into
    [1/2, 1) (1.0 when every value is 0).

    Multiplying a double by a power of two is exact (unless the product leaves the
    normal range), and so are the correctly rounded mean and the variance of the
    scaled values, scaled back: computing in this common space gives results that do
    not depend on the scale of the data, bit for bit for any data times 2^k that stay
    normal doubles, and squares of deviations that neither overflow nor underflow.
    Values below 2^-1021 times the largest lose bits; they are below the rounding
    of anything computed from the largest. Subnormal data are scaled by at most
    2^1023.
    """
    peak = max((float(np.max(np.abs(s))) for s in samples if s.size), default=0.0)
    if peak == 0.0:
        return 1.0
    exponent = math.frexp(peak)[1]
    return math.ldexp(1.0, min(-exponent, 1023))


def scaled_variance(x: FloatArray) -> float:
    """The sample variance of ``x`` without rescaling (see ``variance``).

    For values already brought near 1 by ``common_scale``; on raw data the squares
    can overflow (|x| above 1e154) or underflow (below 1e-154).

    The corrected two-pass form sums the squared deviations about the rounded mean
    m and subtracts (sum(x - m))^2 / n, which is algebraically the sum of squared
    deviations about the exact mean; with exact sums (``math.fsum``) it stays
    accurate also when the values differ only in their last bits: against exact
    rational arithmetic on such data, timestamp-like offsets (1.7e9 + [0, 1)) and
    wide-ranging values, within 3.1e-16 relative in 3,000 random samples
    (tests/stats/test_exact_differences.py).
    """
    centred = x - mean(x)
    n = x.size
    correction = math.fsum(centred.tolist()) ** 2 / n
    squares = math.fsum((centred * centred).tolist())
    return max(0.0, (squares - correction) / (n - 1))


def variance(x: SampleLike) -> float:
    """Sample variance with denominator n - 1, R's ``stats::var``.

    Corrected two-pass algorithm (Chan, Golub & LeVeque 1983, "Algorithms for
    computing the sample variance", The American Statistician 37: 242-247, eq. 1.7):
    sum((x - m)^2) minus the rounding correction (sum(x - m))^2 / n, with m the
    correctly rounded mean. Computed on the values times ``common_scale(x)`` and
    scaled back, so the squares neither overflow nor underflow: the result is
    accurate whenever the variance itself is a double (a standard deviation up to
    about 1.3e154) and equal, bit for bit, to the unscaled computation wherever
    that one neither overflows nor underflows.

    Raises:
        ValueError: fewer than two values, not one-dimensional or not finite, or a
            variance above the largest double.
    """
    arr = as_sample(x, min_size=2)
    scale = common_scale(arr)
    # Two divisions: scale * scale can underflow (scale is 2^-1024 at the most).
    value = scaled_variance(arr * scale) / scale / scale
    if math.isinf(value):
        raise ValueError(
            "the variance of x exceeds the largest double (about 1.8e308): its "
            "standard deviation is above about 1.3e154; rescale x, for example by "
            "changing its units"
        )
    return value
