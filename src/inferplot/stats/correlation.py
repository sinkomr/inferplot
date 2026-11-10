# NumPy's stubs leave overloads of sort and the ufunc methods partially unknown,
# which strict mode reports on every call; the arrays passed are all typed.
# pyright: reportUnknownMemberType=false
"""Correlation of two variables (``scatterstats``): Pearson's r, Spearman's rho,
the winsorized correlation, and the Bayes factor of Pearson's r.

The R reference is ``correlation::correlation()`` as statsExpressions 2.2.1's
``corr_test()`` calls it for ggstatsplot 1.1.4's ggscatterstats, on the complete
pairs (SPEC.md section 4, "scatterstats"); underneath, ``stats::cor.test`` (Pearson;
Spearman with ``exact = FALSE``), the winsorized values of ``datawizard::winsorize``
(``WRS2::wincor``'s r) tested as a Pearson r, and ``BayesFactor::correlationBF``.

Sums of squares and cross products are exact. Each variable is taken times its own
``common_scale`` (a power of two, exact) and centred on its correctly rounded mean,
the deviations formed exactly as two doubles each (Knuth's TwoSum), their products
exactly as two doubles each (Dekker's TwoProduct), and all of them summed exactly
(``exact_sum``), with the corrected two-pass term for the rounding of the mean; so
S_xx, S_yy and S_xy, and 1 - r^2 = (S_xx S_yy - S_xy^2) / (S_xx S_yy), are exact
rationals, rounded once. r, t and the Fisher-z interval therefore keep their
accuracy at any scale and for nearly collinear data, where R's ``cor`` (rounded
deviations, a rounded r and 1 - r^2 formed from it) does not: exactly collinear
data give r = +/-1, t = +/-inf and p = 0 (R: 1 - 2^-53 or so, t = 1.34e8 and
p = 1.06e-62 on edge_corr_perfect). Exact at any range of magnitudes: where the
deviations span more than about 2^480, the products of their parts would leave the
doubles' range, and the sums are formed from the values as integers instead.

Pairs that lie on one straight line as written (Fahrenheit against one-decimal
Celsius) are collinear too, although their doubles are not: ``pair_moments``
decides it exactly on the values as written (DECISIONS.md D74, D85).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special

from inferplot.stats.descriptive import (
    FloatArray,
    SampleLike,
    as_sample,
    common_scale,
    differences_same_within_rounding,
    exact_mean,
    exact_sum,
)
from inferplot.stats.noncentral import check_conf_level
from inferplot.stats.parametric import t_p_value
from inferplot.stats.robust import check_trim, trim_count
from inferplot.stats.srange import stirlerr
from inferplot.stats.written import written_values

__all__ = [
    "Correlation",
    "CorrelationMethod",
    "correlation_bf10",
    "correlation_log_bf10",
    "pair_moments",
    "pearson",
    "spearman",
    "winsorized_correlation",
]

CorrelationMethod = Literal["pearson", "spearman", "winsorized"]

# Fieller, Hartley & Pearson (1957): the variance of atanh(rho) for Spearman's rho
# is about 1.06 / (n - 3).
_SPEARMAN_VARIANCE: Final = 1.06
# Dekker's splitting constant 2^27 + 1.
_SPLIT: Final = 134217729.0


@dataclass(frozen=True, slots=True)
class Correlation:
    """A correlation coefficient with its test and Fisher-z confidence interval.

    ``estimate`` is r (Pearson, winsorized) or rho (Spearman). ``statistic`` is
    what ggstatsplot reports: Student's t on ``df`` = n - 2 for Pearson's and the
    winsorized r, Spearman's S = (n^3 - n)(1 - rho) / 6 for rho; ``t`` is the t
    the two-sided ``p_value`` comes from, t = r sqrt((n - 2) / (1 - r^2)), for all
    three. r = +/-1 gives t = +/-inf, p = 0 and the interval [r, r].
    """

    method: CorrelationMethod
    estimate: float
    statistic: float
    t: float
    df: float
    p_value: float
    conf_low: float
    conf_high: float
    conf_level: float
    n: int


# ------------------------------------------------------------------ exact sums


def _two_sum(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Knuth's TwoSum (TAOCP vol. 2, 4.2.2, Theorem B): a + b = s + e exactly."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _split(a: FloatArray) -> tuple[FloatArray, FloatArray]:
    c = _SPLIT * a
    high = c - (c - a)
    return high, a - high


def _two_product(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Dekker's (1971) TwoProduct: a b = p + e exactly (|a|, |b| < 2^996 and no
    underflow of the parts)."""
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def _deviations(x: FloatArray) -> tuple[FloatArray, FloatArray, Fraction, float]:
    """x times its common scale k minus its rounded mean, exactly as s + e, the
    exact sum of the deviations (the rounding of the mean times -n), and k."""
    scale = common_scale(x)
    scaled = x * scale
    centre = float(exact_mean(scaled))
    s, e = _two_sum(scaled, np.full_like(scaled, -centre))
    return s, e, exact_sum(np.concatenate([s, e])), scale


def _exact_dot(
    a: tuple[FloatArray, FloatArray], b: tuple[FloatArray, FloatArray]
) -> Fraction:
    """sum_i (a_s + a_e)(b_s + b_e) exactly."""
    parts: list[FloatArray] = []
    for u in a:
        for v in b:
            p, e = _two_product(u, v)
            parts.append(p)
            parts.append(e)
    return exact_sum(np.concatenate(parts))


@dataclass(frozen=True, slots=True)
class CrossMoments:
    """The sums of squares and cross products of the pairs about their exact
    means, exact: ``sxx`` = S_xx k_x^2, ``syy`` = S_yy k_y^2 and ``sxy`` =
    S_xy k_x k_y with k_x and k_y each variable's ``common_scale`` (``x_scale``,
    ``y_scale``; powers of two), so that they neither overflow nor underflow.
    Ratios free of units (r, t, 1 - r^2) are the same in either scale."""

    n: int
    sxx: Fraction
    syy: Fraction
    sxy: Fraction
    x_scale: float
    y_scale: float

    @property
    def r2(self) -> Fraction:
        return self.sxy * self.sxy / (self.sxx * self.syy)

    @property
    def one_minus_r2(self) -> Fraction:
        """1 - r^2 >= 0 exactly (Cauchy-Schwarz)."""
        return (self.sxx * self.syy - self.sxy * self.sxy) / (self.sxx * self.syy)

    @property
    def r(self) -> float:
        """r with |r| <= 1 (the square root of the correctly rounded r^2)."""
        return math.copysign(math.sqrt(float(self.r2)), float(self.sxy)) + 0.0


# Dekker's TwoProduct is exact while a product of two parts is at least about
# 2^-969 (its rounding error is then a double); below this, products of the
# deviations' parts could lose bits, and the sums are formed from integers.
_SMALLEST_EXACT_PRODUCT: Final = 2.0**-960


def _smallest_part(*parts: FloatArray) -> float:
    magnitudes = np.abs(np.concatenate(parts))
    nonzero = magnitudes[magnitudes > 0.0]
    return float(nonzero.min()) if nonzero.size else 1.0


def _as_integers(x: FloatArray) -> tuple[list[int], int]:
    """The doubles x as integers X_i times 2^e (one e for all), exactly."""
    mantissa, exponent = np.frexp(x)
    whole = (mantissa * 2.0**53).astype(np.int64).tolist()
    shifts = (exponent - 53).tolist()
    low = min(shifts)
    return [int(m) << (k - low) for m, k in zip(whole, shifts, strict=True)], low


def _integer_moments(x: FloatArray, y: FloatArray) -> tuple[Fraction, ...]:
    """S_xx, S_yy and S_xy of the doubles x and y exactly, from integers: n sum
    X^2 - (sum X)^2 over n, at any range of magnitudes."""
    xi, ex = _as_integers(x)
    yi, ey = _as_integers(y)
    n = len(xi)
    sx, sy = sum(xi), sum(yi)
    sxx = n * sum(v * v for v in xi) - sx * sx
    syy = n * sum(v * v for v in yi) - sy * sy
    sxy = n * sum(a * b for a, b in zip(xi, yi, strict=True)) - sx * sy
    two = Fraction(2)
    return (
        Fraction(sxx, n) * two ** (2 * ex),
        Fraction(syy, n) * two ** (2 * ey),
        Fraction(sxy, n) * two ** (ex + ey),
    )


def cross_moments(x: FloatArray, y: FloatArray) -> CrossMoments:
    """``CrossMoments`` of two float64 arrays of the same length (at least one
    value each, finite), from the exact deviations and products (module
    docstring). Exact at any range of magnitudes: where the deviations span so
    many orders of magnitude (about 2^480 between the largest and the smallest
    part) that a product of their parts would leave the doubles' range, the
    sums are formed from the values as integers instead (Phase 4 review, Z6:
    from a range of 2^537 Cauchy-Schwarz failed and log raised)."""
    sx, ex, tx, kx = _deviations(x)
    sy, ey, ty, ky = _deviations(y)
    n = x.size
    small_x, small_y = _smallest_part(sx, ex), _smallest_part(sy, ey)
    if min(small_x * small_x, small_y * small_y, small_x * small_y) < (
        _SMALLEST_EXACT_PRODUCT
    ):
        sxx, syy, sxy = _integer_moments(x * kx, y * ky)
        return CrossMoments(n=n, sxx=sxx, syy=syy, sxy=sxy, x_scale=kx, y_scale=ky)
    return CrossMoments(
        n=n,
        sxx=_exact_dot((sx, ex), (sx, ex)) - tx * tx / n,
        syy=_exact_dot((sy, ey), (sy, ey)) - ty * ty / n,
        sxy=_exact_dot((sx, ex), (sy, ey)) - tx * ty / n,
        x_scale=kx,
        y_scale=ky,
    )


# Pairs whose exact 1 - r^2 is at most this may lie on one line as written and are
# checked exactly (``pair_moments``). A reading lies within 2 units in the last
# place of its double and its decimal unit is at least 2^20 such units, while two
# different readings differ by at least that unit; so pairs collinear as written
# leave at most about 2 n 2^-38 (7e-12 n) of 1 - r^2 in the doubles. Checking from
# 1e-4 on covers that up to n = 1e7 and costs one exact pass, only for |r| above
# 0.99995.
_AS_WRITTEN_CHECK: Final = 1e-4


def _written_moments(x: FloatArray, y: FloatArray) -> tuple[Fraction, ...]:
    """S_xx, S_yy and S_xy of the values as written (``written_values``), exact."""
    wx, wy = written_values(x), written_values(y)
    n = len(wx)
    mx = sum(wx, Fraction(0)) / n
    my = sum(wy, Fraction(0)) / n
    dx = [v - mx for v in wx]
    dy = [v - my for v in wy]
    return (
        sum((d * d for d in dx), Fraction(0)),
        sum((d * d for d in dy), Fraction(0)),
        sum((a * b for a, b in zip(dx, dy, strict=True)), Fraction(0)),
    )


def pair_moments(x: FloatArray, y: FloatArray) -> CrossMoments:
    """``cross_moments`` of the pairs, except that pairs which lie on one straight
    line as written get the moments of the values as written (DECISIONS.md D74,
    D85): S_xx S_yy = S_xy^2 exactly, so r = +/-1, t = +/-inf, p = 0, the interval
    [r, r] and the Bayes factor's limit, as for data collinear in binary.

    Decimal data that are collinear as written (Fahrenheit and one-decimal
    Celsius, inches and centimetres, y = x / 10, y = 100 - x) are not collinear
    in binary: each stored value is off by up to half a unit in its last place,
    and r computed exactly from the doubles is 1 - 1e-30 or so, which made t
    about 1e15 and BF10 about 1e228 (Phase 4 review, Z2). Each value is read as
    the decimal it stands for (``written_values``: within 2 units in its last
    place, the decimal at least 2^20 units coarse, else the double itself) and
    collinearity is decided exactly on the readings. Nearly collinear measured
    data are not swallowed: values measured to many digits are read as their
    doubles, and decimals with real noise are not collinear as written, however
    small 1 - r^2 is (1e-12 from noise in the 7th digit stays 1e-12).
    """
    moments = cross_moments(x, y)
    om = moments.one_minus_r2
    if om == 0 or float(om) > _AS_WRITTEN_CHECK:
        return moments
    sxx, syy, sxy = _written_moments(x, y)
    if sxx == 0 or syy == 0 or sxx * syy != sxy * sxy:
        return moments
    kx, ky = Fraction(moments.x_scale), Fraction(moments.y_scale)
    return CrossMoments(
        n=moments.n,
        sxx=sxx * kx * kx,
        syy=syy * ky * ky,
        sxy=sxy * kx * ky,
        x_scale=moments.x_scale,
        y_scale=moments.y_scale,
    )


def _pairs(x: SampleLike, y: SampleLike) -> tuple[FloatArray, FloatArray]:
    a = as_sample(x, name="x", min_size=3)
    b = as_sample(y, name="y", min_size=3)
    if a.size != b.size:
        raise ValueError(
            f"x and y need the same number of values (pairs); got {a.size} and {b.size}"
        )
    return a, b


def _constant_error(name: str, what: str) -> ValueError:
    return ValueError(
        f"every value of {name} is the same{what} (exactly, or to within the "
        "rounding of the values in double precision), so its standard deviation is "
        "0 and the correlation is undefined"
    )


def _moments(x: FloatArray, y: FloatArray, what: str = "") -> CrossMoments:
    for name, values in (("x", x), ("y", y)):
        if differences_same_within_rounding(values, np.zeros_like(values)):
            raise _constant_error(name, what)
    return pair_moments(x, y)


# ------------------------------------------------------------------ the tests


def _t(m: CrossMoments) -> float:
    """t = r sqrt((n - 2) / (1 - r^2)), from the exact moments: t^2 = (n - 2)
    S_xy^2 / (S_xx S_yy - S_xy^2), rounded once; +/-inf when that is 0."""
    df = m.n - 2
    if m.sxy == 0:
        return 0.0
    numerator = m.sxy * m.sxy * df
    denominator = m.sxx * m.syy - m.sxy * m.sxy
    sign = 1.0 if m.sxy > 0 else -1.0
    if denominator == 0:
        return sign * math.inf
    ratio = numerator / denominator
    log_t = 0.5 * (math.log(ratio.numerator) - math.log(ratio.denominator))
    if log_t > 709.0:
        raise ValueError(  # pragma: no cover - needs 1 - r^2 below about 1e-616
            "the correlation's t statistic exceeds the largest double"
        )
    return sign * math.sqrt(float(ratio)) if log_t < 300.0 else sign * math.exp(log_t)


def _fisher_interval(
    m: CrossMoments, se: float, conf_level: float
) -> tuple[float, float]:
    """tanh(atanh(r) -/+ q se), q the standard normal quantile at (1 + level) / 2;
    atanh(r) = sign(r) log((1 + |r|) / sqrt(1 - r^2)) with 1 - r^2 exact, so it
    keeps its accuracy as |r| approaches 1; r = +/-1 gives [r, r]."""
    q = -float(special.ndtri((1.0 - conf_level) / 2.0))
    r = m.r
    om = float(m.one_minus_r2)
    if om == 0.0:
        return r, r
    z = math.copysign(math.log1p(abs(r)) - 0.5 * math.log(om), r)
    return math.tanh(z - q * se), math.tanh(z + q * se)


def _fisher_se(n: int, variance: float, method: str) -> float:
    if n <= 3:
        raise ValueError(
            f"the Fisher-z confidence interval of {method} needs at least 4 pairs "
            f"(its standard error is sqrt({variance:g} / (n - 3))); got {n}"
        )
    return math.sqrt(variance / (n - 3))


def _pearson_from(
    m: CrossMoments, method: CorrelationMethod, level: float, name: str
) -> Correlation:
    t = _t(m)
    df = float(m.n - 2)
    low, high = _fisher_interval(m, _fisher_se(m.n, 1.0, name), level)
    return Correlation(
        method=method,
        estimate=m.r,
        statistic=t,
        t=t,
        df=df,
        p_value=t_p_value(t, df, "two-sided"),
        conf_low=low,
        conf_high=high,
        conf_level=level,
        n=m.n,
    )


def pearson(x: SampleLike, y: SampleLike, *, conf_level: float = 0.95) -> Correlation:
    """Pearson's (1895) product-moment correlation r of the pairs (x_i, y_i),
    tested with Student's t = r sqrt((n - 2) / (1 - r^2)) on n - 2 df, two-sided,
    with Fisher's (1915) interval tanh(atanh(r) -/+ q / sqrt(n - 3)). Matches R's
    ``stats::cor.test(x, y)`` and ``correlation::correlation(method = "pearson")``
    as statsExpressions 2.2.1 calls it for ggscatterstats. Computed from exact sums
    (module docstring); r, t and p do not depend on the units of either variable.

    Raises:
        ValueError: fewer than 4 pairs (the interval needs n > 3; R's cor.test
            gives no interval at n = 3), unequal lengths, a non-finite value, a
            constant variable (R: every number NA, "the standard deviation is
            zero"; also when its values differ only within the rounding of the
            values, D74), or conf_level not strictly between 0 and 1.
    """
    level = check_conf_level(conf_level)
    a, b = _pairs(x, y)
    return _pearson_from(_moments(a, b), "pearson", level, "Pearson's r")


def _mid_ranks(x: FloatArray) -> FloatArray:
    """Mid-ranks 1..n (ties get the mean of their ranks); exact halves."""
    order = np.argsort(x, kind="stable")
    ordered = x[order]
    starts = np.concatenate([[True], ordered[1:] != ordered[:-1]])
    group = np.cumsum(starts) - 1
    first = np.flatnonzero(starts)
    last = np.append(first[1:], x.size) - 1
    mid = (first + last) / 2.0 + 1.0
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = mid[group]
    return ranks


def spearman(x: SampleLike, y: SampleLike, *, conf_level: float = 0.95) -> Correlation:
    """Spearman's (1904) rank correlation rho: Pearson's r of the mid-ranks.

    The reported statistic is S = (n^3 - n)(1 - rho) / 6 (with no ties, the sum of
    squared rank differences). p is two-sided from t = rho sqrt((n - 2) / (1 -
    rho^2)) on n - 2 df, always, also for small n without ties (R's
    ``cor.test(method = "spearman", exact = FALSE)``, which correlation::correlation
    runs: edge_corr_small prints 0.037 where the exact permutation p is 0.083).
    Interval tanh(atanh(rho) -/+ q sqrt(1.06 / (n - 3))) (Fieller, Hartley &
    Pearson 1957, Biometrika 44: 470-481); Fisher's 1 / sqrt(n - 3) misses every
    fixture. Matches ``correlation::correlation(method = "spearman")`` as
    statsExpressions 2.2.1 calls it for ggscatterstats. Ranks are exact halves, so
    rho, S, t and p are exact up to their last rounding, and do not depend on any
    increasing transformation of either variable.

    Raises:
        ValueError: as ``pearson`` (a constant variable has constant ranks).
    """
    level = check_conf_level(conf_level)
    a, b = _pairs(x, y)
    for name, values in (("x", a), ("y", b)):
        if differences_same_within_rounding(values, np.zeros_like(values)):
            raise _constant_error(name, "")
    m = _moments(_mid_ranks(a), _mid_ranks(b))
    t = _t(m)
    df = float(m.n - 2)
    rho = m.r
    # 1 - rho without cancellation near rho = 1: (1 - rho^2) / (1 + rho).
    one_minus = float(m.one_minus_r2 / (1 + Fraction(rho))) if rho > 0.0 else 1.0 - rho
    n = m.n
    low, high = _fisher_interval(
        m, _fisher_se(n, _SPEARMAN_VARIANCE, "Spearman's rho"), level
    )
    return Correlation(
        method="spearman",
        estimate=rho,
        statistic=(n * n * n - n) * one_minus / 6.0,
        t=t,
        df=df,
        p_value=t_p_value(t, df, "two-sided"),
        conf_low=low,
        conf_high=high,
        conf_level=level,
        n=n,
    )


def _winsorize(x: FloatArray, g: int) -> FloatArray:
    """Each value clipped to the (g + 1)-th smallest and (g + 1)-th largest."""
    ordered = np.sort(x)
    return np.minimum(np.maximum(x, ordered[g]), ordered[x.size - g - 1])


def winsorized_correlation(
    x: SampleLike, y: SampleLike, *, tr: float = 0.2, conf_level: float = 0.95
) -> Correlation:
    """The winsorized correlation (Wilcox 2017, Introduction to Robust Estimation
    and Hypothesis Testing, 4th ed., section 9.3.2): each variable winsorized at
    its own g = floor(tr n) (the g smallest values set to the (g + 1)-th smallest,
    the g largest to the (g + 1)-th largest, the pairs kept), then Pearson's r of
    the winsorized values (the r of ``WRS2::wincor``), tested as a Pearson r: t on
    n - 2 df (not wincor's own test on h - 2 df, whose p differs) and Fisher's
    interval at 1 / sqrt(n - 3). Matches ``correlation::correlation(winsorize =
    tr)`` as statsExpressions 2.2.1 calls it for ggscatterstats (probe-corr: the
    winsorized values and r equal datawizard's and wincor's; mtcars r = -0.864 at
    tr 0.2 and -0.897 at 0.1). g is R's floor(tr * n) in floating point (tr 0.29,
    n 100 trims 28).

    Raises:
        ValueError: as ``pearson``, a variable that is constant after winsorizing,
            or ``tr`` outside [0, 0.5).
    """
    t_trim = check_trim(tr)
    level = check_conf_level(conf_level)
    a, b = _pairs(x, y)
    g = trim_count(a.size, t_trim)
    m = _moments(_winsorize(a, g), _winsorize(b, g), " after winsorizing")
    return _pearson_from(m, "winsorized", level, "the winsorized correlation")


# ------------------------------------------------------------------ Bayes factor

# The trapezoid sums in u = logit(t) extend until the log integrand is this far
# below its peak (e^-60 = 9e-27).
_DROP: Final = 60.0
_BLOCK: Final = 256
_CONVERGED: Final = 1e-14
_MAX_HALVINGS: Final = 14
_STIRLING_FROM: Final = 15.0


def _log_gamma_ratio(x: float, d: float) -> float:
    """log Gamma(x + d) - log Gamma(x) for x, d > 0, without the cancellation of
    two large log-gamma values: from Stirling's formula with its remainder,
    d log x + (x + d - 1/2) log1p(d / x) - d + stirlerr(x + d) - stirlerr(x), for
    x >= 15, and below by the recurrence Gamma(x + 1) = x Gamma(x)."""
    steps = max(0, math.ceil(_STIRLING_FROM - x))
    terms: list[float] = []
    for k in range(steps):
        terms.append(-math.log1p(d / (x + k)))
    y = x + steps
    terms += [
        d * math.log(y),
        (y + d - 0.5) * math.log1p(d / y),
        -d,
        stirlerr(y + d),
        -stirlerr(y),
    ]
    return math.fsum(terms)


def _log_trapezoid(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]], lo: float, hi: float
) -> float:
    """log of the integral over the real line of exp(f(u)), f smooth, unimodal
    and decaying at least exponentially at both ends: the trapezoidal rule,
    which converges geometrically for such integrands (Trefethen & Weideman 2014,
    SIAM Review 56: 385-458), summed in log space; the peak is found by a scan of
    [lo, hi], the step starts at min(0.25, 0.2 / sqrt(curvature)) and is halved
    until two sums agree to 1e-14 in the log."""

    def f1(u: float) -> float:
        return float(f(np.array([u]))[0])

    coarse = np.arange(lo, hi + 0.25, 0.25)
    values = f(coarse)
    best = float(coarse[int(np.argmax(values))])
    found = optimize.minimize_scalar(
        lambda u: -f1(u),
        bounds=(best - 0.25, best + 0.25),
        method="bounded",
        options={"xatol": 1e-10},
    )
    centre = float(found.x)
    peak = f1(centre)
    step = 1e-3
    curvature = -(f1(centre + step) - 2.0 * peak + f1(centre - step)) / step**2
    h = min(0.25, 0.2 / math.sqrt(curvature)) if curvature > 0.0 else 0.25
    left, right = -_BLOCK, _BLOCK
    while f1(centre + h * left) > peak - _DROP:
        left -= _BLOCK
    while f1(centre + h * right) > peak - _DROP:
        right += _BLOCK
    start = centre + h * left
    count = right - left + 1
    terms = f(start + h * np.arange(count, dtype=np.float64))

    def total(t: NDArray[np.float64], width: float) -> float:
        top = float(np.max(t))
        return top + math.log(float(np.sum(np.exp(t - top)))) + math.log(width)

    estimate = total(terms, h)
    for _ in range(_MAX_HALVINGS):
        mids = f(start + h * (np.arange(count - 1, dtype=np.float64) + 0.5))
        terms = np.concatenate([terms, mids])
        count = 2 * count - 1
        h *= 0.5
        refined = total(terms, h)
        if abs(refined - estimate) <= _CONVERGED * max(1.0, abs(refined)):
            return refined
        estimate = refined
    raise ArithmeticError(  # pragma: no cover - the trapezoid sums converge fast
        "the correlation Bayes factor's quadrature did not converge"
    )


def _log_bf10(log_om: float, n: int, prior_scale: float) -> float:
    """log BF10 of a Pearson r with log(1 - r^2) = ``log_om`` (see
    ``correlation_log_bf10``; -inf for r = +/-1). The log integrand is a times a
    function of log(1 - r^2), so an absolute error e in ``log_om`` moves log BF10
    by up to a e: it is passed as a log, accurate to its own rounding."""
    alpha = 1.0 / prior_scale
    a = 0.5 * (n - 1)
    # log B(a, alpha) = log Gamma(alpha) - (log Gamma(a + alpha) - log Gamma(a)).
    log_b = math.lgamma(alpha) - _log_gamma_ratio(a, alpha)
    if log_om == 0.0:
        # r = 0: the integral is B(a, alpha + 1/2).
        return math.lgamma(alpha + 0.5) - _log_gamma_ratio(a, alpha + 0.5) - log_b
    if log_om == -math.inf:
        # r = +/-1: Gauss's 2F1(a, a; c; 1) = B(a, alpha + 1/2 - a) / B(a, alpha) when
        # alpha + 1/2 > a, else the integral diverges.
        rest = alpha + 0.5 - a
        if rest <= 0.0:
            return math.inf
        return math.lgamma(rest) - _log_gamma_ratio(a, rest) - log_b
    power = alpha + 0.5

    def f(u: NDArray[np.float64]) -> NDArray[np.float64]:
        # t^(a - 1) (1 - t)^(alpha - 1/2) (1 - r^2 t)^(-a) dt with t = 1 / (1 + e^-u),
        # dt = t (1 - t) du and 1 - r^2 t = t (e^-u + 1 - r^2): the log integrand is
        # (alpha + 1/2) log(1 - t) - a log(e^-u + 1 - r^2), with log(1 - t) =
        # -log(1 + e^u); no 1 - r^2 t is formed, so 1 - r^2 keeps its accuracy.
        return -power * np.logaddexp(0.0, u) - a * np.logaddexp(-u, log_om)

    reach = -log_om + 60.0 / power + math.log(a) + 40.0
    log_n = _log_trapezoid(f, -60.0 / a - 40.0, reach)
    return log_n - log_b


def correlation_log_bf10(r: float, n: int, *, prior_scale: float) -> float:
    """Natural log of the Bayes factor BF10 for a Pearson correlation r of n pairs
    against rho = 0, with a stretched beta(1/kappa, 1/kappa) prior on rho over
    (-1, 1), kappa = ``prior_scale`` (Ly, Verhagen & Wagenmakers 2016, Journal of
    Mathematical Psychology 72: 19-32, eq. 13, Jeffreys's exact test with the
    prior width kappa):

        BF10 = 2^((kappa - 2) / kappa) sqrt(pi) / B(1/kappa, 1/kappa)
               Gamma((n + 2/kappa - 1) / 2) / Gamma((n + 2/kappa) / 2)
               2F1((n - 1) / 2, (n - 1) / 2; (n + 2/kappa) / 2; r^2),

    which by Legendre's duplication formula and Euler's integral for 2F1 is
    int_0^1 t^(a - 1) (1 - t)^(alpha - 1/2) (1 - r^2 t)^(-a) dt / B(a, alpha) with
    a = (n - 1) / 2 and alpha = 1/kappa. The integral is taken over u = logit t by
    the trapezoidal rule in log space (``_log_trapezoid``), the beta function from
    log-gamma ratios that keep their accuracy for large n. Two-sided.

    Matches R's ``BayesFactor::correlationBF(y, x, rscale = prior_scale)``, whose
    documentation names this prior (a shifted, scaled beta(1/rscale, 1/rscale));
    ggstatsplot passes ``bf.prior`` as rscale and prints its prior parameter as
    1 / bf.prior (1.41 for 0.707). probe-corr's 28 finite values agree within
    1.25e-12 relative at R's own r (iris, n 150, BF10 1.2e44; the others within
    5.3e-14), BayesFactor's own floating point; 40-digit mpmath evaluations of the
    hypergeometric form agree within 1e-14 + 1e-15 |log BF10| in the log on a grid
    of n 3 to 2,000, |r| up to 1 - 1e-7 and r = +/-1, at kappa 0.707, 0.8 and 1/3
    (tests/stats/test_correlation.py). log(1 - r^2) is taken as log1p(-|r|) +
    log1p(|r|): log BF10 moves by about (n - 1) / 2 times an error in it.

    r = +/-1: the limit of BF10 as |r| -> 1, which is +inf when (n - 1) / 2 >=
    1/kappa + 1/2 (n >= 5 at kappa 0.707) and finite below (Gauss's 2F1 at 1).
    BayesFactor returns NaN there and ggstatsplot prints an empty log BF01.

    Raises:
        ValueError: n below 3, |r| above 1 or not finite, or prior_scale not a
            finite positive number.
    """
    value = float(r)
    if not (math.isfinite(value) and abs(value) <= 1.0):
        raise ValueError(f"r must be a number in [-1, 1]; got {r!r}")
    size = abs(value)
    # log(1 - r^2) = log1p(-|r|) + log1p(|r|), each to its own rounding.
    log_om = -math.inf if size == 1.0 else math.log1p(-size) + math.log1p(size)
    return _checked_log_bf10(log_om, n, prior_scale)


def _log_fraction(value: Fraction) -> float:
    """log of a positive exact rational to the rounding of the result: the log of
    its double plus the double's relative error (log(1 + d) = d to below the
    rounding, |d| <= 2^-53)."""
    high = float(value)
    if high == 0.0:  # pragma: no cover - 1 - r^2 is never below 1e-308 here
        return math.log(value.numerator) - math.log(value.denominator)
    return math.log(high) + float((value - Fraction(high)) / Fraction(high))


def _checked_log_bf10(log_om: float, n: int, prior_scale: float) -> float:
    if isinstance(n, bool) or int(n) != n or n < 3:
        raise ValueError(f"the correlation Bayes factor needs n >= 3 pairs; got {n!r}")
    scale = float(prior_scale)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(
            f"prior_scale must be a finite number > 0; got {prior_scale!r}"
        )
    return _log_bf10(log_om, int(n), scale)


def correlation_bf10(
    x: SampleLike, y: SampleLike, *, prior_scale: float
) -> tuple[float, float]:
    """Bayes factor for the Pearson correlation of the pairs (x_i, y_i):
    ``(bf10, log_bf01)``, ``correlation_log_bf10`` at the exact 1 - r^2 of the
    data (module docstring). Matches ``correlation::correlation(bayesian = TRUE,
    bayesian_prior = prior_scale)`` (``BayesFactor::correlationBF``) as
    statsExpressions 2.2.1 calls it for ggscatterstats with ``bf.prior``. ``bf10``
    is ``inf`` when it overflows a double (or for r = +/-1 with n >= 5 at
    prior_scale 0.707); ``log_bf01`` = -log BF10, finite where BF10 is.

    Raises:
        ValueError: as ``pearson`` (but n = 3 is allowed), or prior_scale not a
            finite positive number.
    """
    a, b = _pairs(x, y)
    m = _moments(a, b)
    om = m.one_minus_r2
    log_om = -math.inf if om == 0 else _log_fraction(om)
    log_bf10 = _checked_log_bf10(log_om, m.n, prior_scale)
    try:
        bf10 = math.exp(log_bf10)
    except OverflowError:
        bf10 = math.inf
    return bf10, -log_bf10
