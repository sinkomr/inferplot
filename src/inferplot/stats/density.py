# NumPy's stubs type several overloads (sort, where, clip, searchsorted, argmax, dot)
# as partially Unknown, which strict mode reports on every call; the arguments are
# all float64 or int64 arrays.
# pyright: reportUnknownMemberType=false
"""MAP estimate of a sample: the peak of a Sheather-Jones kernel density on a grid.

ggstatsplot labels each group with its MAP ("maximum a posteriori") estimate for
``type = "bayes"``: ``bayestestR::map_estimate(x)``, which by its help page and
``estimate_density()``'s evaluates ``stats::density(x, bw = "SJ", n = 2^10)`` on the
data range (``extend = FALSE``; the fixtures' MAP values all lie on that grid) and
returns the grid point where the density is largest. Three published pieces:

1. **Bandwidth**: Sheather & Jones (1991, JRSS B 53: 683-690), "solve-the-equation"
   rule, with the pairwise sums of the derivative functionals taken over binned
   distances (``bw.SJ(nb = 1000)``, Scott & Sheather 1985) and the equation solved by
   Brent's (1973) zero finder to ``uniroot``'s loose tolerance (``tol = 0.1 *
   lower``, "slightly more than one digit of accuracy", ``?bw.SJ``).
2. **Density**: Gaussian kernel, linear binning of the data onto a regular grid of
   1,024 points that extends ``ext = 4`` bandwidths beyond the data, discrete
   convolution with the kernel by FFT, and linear interpolation onto the output grid
   (Silverman 1982, Applied Statistics 31: 93-99, algorithm AS 176; ``?density``).
3. **MAP**: the first output grid point with the largest density.

The bandwidth's binning and constants are not all on R's help pages; the choices
below were settled against ggstatsplot's 30 MAP labels in the bayes fixtures, which
are all reproduced at the same grid point. Each alternative tried moves 1 to 14 of
them (tests/tolerances.py ``MAP_CENTRALITY`` lists them): bins anchored at min(x),
a bin width of range / 1000, pilot constants 1.241 and 1.230, netlib's Fortran
zeroin, another hmax constant, another FFT grid spacing. Every fixture value is
positive, so whether bins of negative values truncate toward zero (as here) or round
down is not yet pinned.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from inferplot.stats.descriptive import SampleLike, as_sample, common_scale, variance

FloatArray = NDArray[np.float64]

__all__ = ["MapEstimate", "bw_sj", "map_estimate"]

GRID_POINTS: Final = 1024
"""bayestestR's ``precision = 2^10``: points of the output grid."""
_N_BINS: Final = 1000  # bw.SJ's nb
_BIN_WIDEN: Final = 1.01  # bin width = 1.01 x range / nb (fixtures: 30 of 30)
_EXT: Final = 4.0  # density's ext: FFT grid from min - 4 bw to max + 4 bw
# Terms with (distance / h)^2 at or above this are dropped from the functional
# sums: exp(-500) = 7e-218 against the diagonal term 3 / sqrt(2 pi).
_DELMAX: Final = 1000.0
_TIE: Final = 1e-12  # densities this close (relative) to the maximum tie with it
_MAX_TRIES: Final = 99  # bracket enlargements before giving up
_MAX_ITER: Final = 1000  # uniroot's maxiter
_EPS: Final = float(np.finfo(np.float64).eps)
_SQRT_2PI: Final = math.sqrt(2.0 * math.pi)
# Bin indices are computed as doubles; beyond 2^53 they are no longer integers.
_MAX_BIN: Final = 2.0**53


@dataclass(frozen=True)
class MapEstimate:
    """MAP estimate of a sample and how it was read off the grid.

    Attributes:
        estimate: the grid point with the largest density.
        index: its position on the grid, 0 to 1023.
        grid_step: the grid spacing, (max(x) - min(x)) / 1023.
        bandwidth: the Sheather-Jones bandwidth of the kernel density.
    """

    estimate: float
    index: int
    grid_step: float
    bandwidth: float


def _quantile7(sorted_x: FloatArray, p: float) -> float:
    """Hyndman & Fan (1996) type 7 quantile (R's default) of sorted values."""
    h = (sorted_x.size - 1) * p
    j = math.floor(h)
    g = h - j
    lo = float(sorted_x[j])
    if g == 0.0:
        return lo
    hi = float(sorted_x[j + 1])
    return (1.0 - g) * lo + g * hi


def _pair_counts(x: FloatArray) -> tuple[FloatArray, float]:
    """Binned pairwise distances: ``(counts, width)``.

    The values fall into bins of width ``width`` = 1.01 (max - min) / 1000 anchored
    at zero (bin trunc(x / width)); ``counts[k]`` is the number of pairs i < j whose
    bins are k apart, so a pair's distance is taken as k x width. Bins anchored at
    zero make the bandwidth change slightly when the data are shifted or reflected,
    as ``?bw.SJ`` warns. O(n) for any n: the pairs are counted from the bin counts.

    Raises:
        ValueError: the values have no spread, or lie so far from zero relative to
            it (beyond 2^53 bin widths) that their bins are not resolved.
    """
    lo, hi = float(x.min()), float(x.max())
    width = _BIN_WIDEN * (hi - lo) / _N_BINS
    if not (math.isfinite(width) and width > 0.0):
        raise ValueError(
            "no Sheather-Jones bandwidth: the values have no spread or their range "
            "exceeds the largest double"
        )
    scaled = x / width
    if float(np.max(np.abs(scaled))) >= _MAX_BIN:
        raise ValueError(
            "no Sheather-Jones bandwidth: the values lie more than 2^53 bin widths "
            "(1.01 x range / 1000) from zero; subtract a constant near their center"
        )
    bins = np.trunc(scaled)
    index = (bins - bins.min()).astype(np.int64)
    per_bin = np.bincount(index)
    counts = np.correlate(per_bin, per_bin, mode="full")[per_bin.size - 1 :]
    counts[0] = (int(np.dot(per_bin, per_bin)) - x.size) // 2
    return counts.astype(np.float64), width


def _phi4(u: FloatArray) -> FloatArray:
    """Fourth derivative of the standard normal density."""
    u2 = u * u
    return (u2 * u2 - 6.0 * u2 + 3.0) * np.exp(-0.5 * u2) / _SQRT_2PI


def _phi6(u: FloatArray) -> FloatArray:
    """Sixth derivative of the standard normal density."""
    u2 = u * u
    return (
        (u2 * u2 * u2 - 15.0 * u2 * u2 + 45.0 * u2 - 15.0)
        * np.exp(-0.5 * u2)
        / _SQRT_2PI
    )


def _functional(
    order: int, counts: FloatArray, width: float, n: int, h: float
) -> float:
    """Sheather & Jones' (1991, section 2) estimate of the integrated squared
    derivative functional of order ``order`` (4 or 6) at bandwidth ``h``:

        (n (n - 1))^-1 h^-(order + 1) sum_i sum_j phi^(order)((X_i - X_j) / h),

    the double sum over all i and j, the diagonal included, with each distance
    replaced by its binned value.
    """
    u = np.arange(counts.size, dtype=np.float64) * width / h
    keep = u * u < _DELMAX
    kernel = _phi4 if order == 4 else _phi6
    zero = kernel(np.zeros(1))[0]
    total = 2.0 * float(np.sum(counts[keep] * kernel(u[keep]))) + n * float(zero)
    return total / (n * (n - 1.0) * h ** (order + 1))


def _zeroin(
    f: Callable[[float], float], a: float, b: float, fa: float, fb: float, tol: float
) -> float:
    """Brent's (1973, Algorithms for Minimization without Derivatives, ch. 4) zero
    finder, the routine ``?uniroot`` names (netlib's ``zeroin``): bisection
    safeguarding inverse quadratic and linear interpolation, converged when half the
    bracket is at most 2 eps |b| + tol / 2 or f(b) is 0. Stated as netlib's C
    translation does (the last step b - a recomputed every iteration), which
    reproduces the fixtures; the Fortran original moves 2 of their 30 MAP labels."""
    c, fc = a, fa
    for _ in range(_MAX_ITER + 1):
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2.0 * _EPS * abs(b) + tol / 2.0
        new_step = (c - b) / 2.0
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:  # linear interpolation
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:  # inverse quadratic interpolation
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0.0:
                q = -q
            else:
                p = -p
            if p < 0.75 * cb * q - abs(tol_act * q) / 2.0 and p < abs(
                prev_step * q / 2.0
            ):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0.0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if math.isnan(fb):
            raise ValueError("no Sheather-Jones bandwidth: the equation has no value")
        if (fb > 0.0 and fc > 0.0) or (fb < 0.0 and fc < 0.0):
            c, fc = a, fa
    return b  # pragma: no cover - uniroot warns and returns the last iterate


def _bw_sj_scaled(x: FloatArray) -> float:
    n = x.size
    ordered = np.sort(x)
    iqr = _quantile7(ordered, 0.75) - _quantile7(ordered, 0.25)
    scale = min(math.sqrt(variance(x)), iqr / 1.349)
    if not scale > 0.0:
        raise ValueError(
            "no Sheather-Jones bandwidth, so no MAP estimate: the values' standard "
            "deviation or interquartile range is 0 (all values tied, or more than "
            "half of them equal)"
        )
    counts, width = _pair_counts(x)
    # Pilot bandwidths a and b: Sheather & Jones' 0.920 and 0.912 times the IQR,
    # written as 1.24 and 1.23 times the scale min(SD, IQR / 1.349).
    a = 1.24 * scale * n ** (-1.0 / 7.0)
    b = 1.23 * scale * n ** (-1.0 / 9.0)
    td = -_functional(6, counts, width, n, b)
    sd_a = _functional(4, counts, width, n, a)
    if not (math.isfinite(td) and td > 0.0 and sd_a > 0.0):
        raise ValueError(
            "no Sheather-Jones bandwidth, so no MAP estimate: the sample is too "
            "sparse to estimate the density's curvature"
        )
    alpha2 = 1.357 * (sd_a / td) ** (1.0 / 7.0)
    c1 = 1.0 / (2.0 * math.sqrt(math.pi) * n)

    def equation(h: float) -> float:
        s = _functional(4, counts, width, n, alpha2 * h ** (5.0 / 7.0))
        if not s > 0.0:
            return math.nan
        return (c1 / s) ** 0.2 - h

    # Terrell's (1990) oversmoothed bandwidth bounds the search from above.
    hmax = 1.144 * scale * n ** (-0.2)
    lower, upper = 0.1 * hmax, hmax
    f_lower, f_upper = equation(lower), equation(upper)
    tries = 1
    while f_lower * f_upper > 0.0 or math.isnan(f_lower * f_upper):
        if tries > _MAX_TRIES or math.isnan(f_lower) or math.isnan(f_upper):
            raise ValueError(
                "no Sheather-Jones bandwidth, so no MAP estimate: no solution in "
                "the range of bandwidths searched"
            )
        # ?bw.SJ: the default interval is enlarged until it brackets the root.
        # Here alternately the upper end times 1.2 and the lower end divided by
        # 1.2, and the tolerance follows the final lower end. In the fixtures, 6 of
        # the 30 groups need enlarging: 5 once (upper end), mtcars am = 0 three
        # times (upper, lower, upper), and all 6 match; a tolerance kept at 0.1 x
        # the original lower end gives the same grid point there, so which one R
        # uses is not pinned yet.
        if tries % 2:
            upper *= 1.2
            f_upper = equation(upper)
        else:
            lower /= 1.2
            f_lower = equation(lower)
        tries += 1
    return _zeroin(equation, lower, upper, f_lower, f_upper, 0.1 * lower)


def bw_sj(x: SampleLike) -> float:
    """Sheather-Jones "solve-the-equation" bandwidth, R's ``stats::bw.SJ(x)``.

    Sheather & Jones (1991, JRSS B 53: 683-690, section 3): h solves
    h = (R(K) / (n S_D(alpha_2(h))))^(1/5) with the Gaussian kernel's R(K) =
    1 / (2 sqrt(pi)), alpha_2(h) = 1.357 (S_D(a) / T_D(b))^(1/7) h^(5/7) and the
    functionals S_D, T_D of ``_functional`` over binned distances (``nb = 1000``).
    The root is found in [0.1 hmax, hmax], hmax = 1.144 min(SD, IQR / 1.349)
    n^(-1/5) (enlarged until it brackets the root), to ``bw.SJ``'s tolerance 0.1 x
    the lower end, so the value is the zero finder's iterate, accurate to about
    1%, as R's is. Computed on the values times a power of two that brings them
    near 1 (exact), so the result scales exactly with the data.

    Raises:
        ValueError: fewer than two values, values that are not finite, no spread
            (standard deviation or interquartile range 0), or no root.
    """
    arr = as_sample(x, min_size=2)
    scale = common_scale(arr)
    return _bw_sj_scaled(arr * scale) / scale


def _density_on_grid(x: FloatArray, bw: float) -> FloatArray:
    """Kernel density of ``x`` at the 1,024 points from min(x) to max(x).

    ``stats::density(x, bw, n = 1024, from = min(x), to = max(x))``: the data's
    mass (1/n each) is split linearly between the two nearest points of a regular
    grid of 1,024 points from min - 4 bw to max + 4 bw, convolved with the
    Gaussian kernel sampled at the grid spacing (circular convolution by FFT over
    2,048 points, so no mass wraps around), negative rounding clipped to 0, and
    interpolated linearly at the output points (Silverman 1982, AS 176).
    """
    n = GRID_POINTS
    lo_x, hi_x = float(x.min()), float(x.max())
    lo = lo_x - _EXT * bw
    up = hi_x + _EXT * bw
    delta = (up - lo) / (n - 1)
    position = (x - lo) / delta
    cell = np.floor(position)
    frac = position - cell
    left = cell.astype(np.intp)
    weight = 1.0 / x.size
    mass = np.bincount(left, weights=weight * (1.0 - frac), minlength=2 * n)
    mass += np.bincount(left + 1, weights=weight * frac, minlength=2 * n)
    k = np.arange(2 * n, dtype=np.float64)
    distance = np.where(k <= n, k, 2 * n - k) * delta
    kernel = np.exp(-0.5 * (distance / bw) ** 2) / (bw * _SQRT_2PI)
    smoothed = np.fft.ifft(np.fft.fft(mass) * np.conj(np.fft.fft(kernel))).real
    nodes_y = np.maximum(smoothed[:n], 0.0)
    nodes_x = lo + np.arange(n, dtype=np.float64) * delta
    out_x = _output_grid(lo_x, hi_x)
    j = np.clip(np.searchsorted(nodes_x, out_x, side="right") - 1, 0, n - 2)
    t = (out_x - nodes_x[j]) / (nodes_x[j + 1] - nodes_x[j])
    return nodes_y[j] + (nodes_y[j + 1] - nodes_y[j]) * t


def _output_grid(lo: float, hi: float) -> FloatArray:
    """1,024 equally spaced points from ``lo`` to ``hi``, both ends exact."""
    step = (hi - lo) / (GRID_POINTS - 1)
    grid = lo + np.arange(GRID_POINTS, dtype=np.float64) * step
    grid[-1] = hi
    return grid


def map_estimate(x: SampleLike) -> MapEstimate:
    """MAP estimate of a sample, R's ``bayestestR::map_estimate(x)``.

    The kernel density of ``x`` with the Sheather-Jones bandwidth (``bw_sj``),
    evaluated at 1,024 equally spaced points from min(x) to max(x)
    (``estimate_density(precision = 2^10, extend = FALSE, bw = "SJ")``), and the
    first point where it is largest (R's ``which.max``). Grid points whose densities
    differ by less than 1e-12 relative count as tied, so the first of them is taken,
    as exact arithmetic would: the FFT leaves exactly tied points (a symmetric
    sample) unequal in the last bits, and which one R's own FFT rounding favours is
    not reproducible. Every MAP label in the bayes fixtures is reproduced at the same
    grid point (tests/tolerances.py ``MAP_CENTRALITY``).

    Computed on ``x`` times a power of two that brings it near 1 (exact), so the
    estimate scales exactly with the data.

    Raises:
        ValueError: fewer than two values, values that are not finite, or no
            Sheather-Jones bandwidth (no spread, or too sparse; see ``bw_sj``).
    """
    arr = as_sample(x, name="x", min_size=1)
    if arr.size < 2:
        raise ValueError(
            "a MAP estimate needs at least 2 values to estimate a density; got 1"
        )
    scale = common_scale(arr)
    scaled = arr * scale
    bw = _bw_sj_scaled(scaled)
    density = _density_on_grid(scaled, bw)
    # The first point within FFT rounding of the maximum: exactly tied points (a
    # symmetric sample) come out of the FFT unequal in the last bits, either way.
    peak = float(np.max(density))
    index = int(np.argmax(density >= peak * (1.0 - _TIE)))
    lo, hi = float(scaled.min()), float(scaled.max())
    grid = _output_grid(lo, hi)
    return MapEstimate(
        estimate=float(grid[index]) / scale,
        index=index,
        grid_step=(hi - lo) / (GRID_POINTS - 1) / scale,
        bandwidth=bw / scale,
    )
