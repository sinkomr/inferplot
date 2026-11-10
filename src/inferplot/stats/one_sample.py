# NumPy's stubs leave overloads of sort, lexsort and the ufunc methods partially
# unknown, which strict mode reports on every call; the arrays passed are all typed.
# pyright: reportUnknownMemberType=false
"""One sample against a test value mu (``histostats``): the one-sample t-test with
Hedges' g, Wilcoxon's signed-rank test with the rank-biserial r, the bootstrap-t
test of the trimmed mean, and the one-sample JZS Bayes factor.

The R references are what statsExpressions 2.2.1 calls for ggstatsplot 1.1.4's
gghistostats (SPEC.md section 4, "histostats"): ``stats::t.test(x, mu)`` with
``effectsize::hedges_g(x, mu)``, ``stats::wilcox.test(x, mu, exact = FALSE)`` with
``effectsize::rank_biserial(x, mu)``, ``WRS2::trimcibt(x, nv = mu)`` and
``BayesFactor::ttestBF(x, mu)``. A one-sample analysis is a paired one on the
differences x - mu, and the functions here share the paired ones' methods where
the method is the same (the noncentral-t interval, the signed ranks, the JZS
integral); where it is not, the docstring says so (the rank-biserial interval's
standard error counts only the nonzero differences).

Deterministic statistics are computed on x times its ``common_scale`` (a power of
two, exact), with the difference of the exact mean (or trimmed mean) and mu formed
exactly and rounded once, so they do not depend on the units of x and mu and keep
their relative accuracy when the mean lies within the rounding of mu.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import special

from inferplot.stats.bayes import jzs_log_bf10
from inferplot.stats.descriptive import (
    FloatArray,
    SampleLike,
    as_sample,
    common_scale,
    differences_same_within_rounding,
    exact_mean,
    scaled_variance,
)
from inferplot.stats.effect_sizes import EffectSizeCI, hedges_j
from inferplot.stats.noncentral import check_conf_level, t_ncp_interval
from inferplot.stats.nonparametric import (
    BOOTSTRAP_RESAMPLES,
    SeedLike,
    check_bootstrap_seed,
)
from inferplot.stats.parametric import TINY, Alternative, check_alternative, t_p_value
from inferplot.stats.robust import (
    bootstrap_replicates,
    check_trim,
    exact_trimmed_mean,
    exact_zero_sum,
    rows_trimmed,
    split_replicates,
    trim_count,
    winsorize_sorted,
)
from inferplot.stats.within_nonparametric import (
    SignedRank,
    _signed_ranks,  # pyright: ignore[reportPrivateUsage]
    signed_rank,
)

__all__ = [
    "BootstrapTrimmedMeanTest",
    "OneSampleT",
    "bootstrap_t_critical_value",
    "jzs_bf10_one_sample",
    "one_sample_hedges_g",
    "one_sample_rank_biserial",
    "one_sample_signed_rank",
    "one_sample_t",
    "trimmed_mean_bootstrap_t",
]


@dataclass(frozen=True, slots=True)
class OneSampleT:
    """The one-sample t-test of x against ``mu`` (R: ``stats::t.test(x, mu =
    mu)``): ``t`` = (mean - mu) / (sd / sqrt(n)), signed, on ``df`` = n - 1."""

    t: float
    df: float
    p_value: float
    n: int
    mu: float
    alternative: Alternative


@dataclass(frozen=True, slots=True)
class BootstrapTrimmedMeanTest:
    """The bootstrap-t test of a trimmed mean against ``mu`` (R:
    ``WRS2::trimcibt``).

    ``statistic`` is t = (trimmed mean - mu) / ``se``, signed, exact (it does not
    depend on the resamples); ``estimate`` the trimmed mean and ``conf_low`` and
    ``conf_high`` the interval estimate -/+ ``critical_value`` x ``se``;
    ``p_value`` the share of the bootstrap |T*| at least |t|. ``critical_value``,
    the bounds and the p-value are random: they come from ``resamples`` resamples
    drawn with ``seed``. ``undefined_resamples`` counts the resamples where T* is
    0 / 0 (left out: the critical value and p come from the others);
    ``infinite_resamples`` those where it is infinite (kept).
    """

    statistic: float
    p_value: float
    estimate: float
    se: float
    critical_value: float
    conf_low: float
    conf_high: float
    conf_level: float
    n: int
    mu: float
    tr: float
    seed: int
    resamples: int
    undefined_resamples: int = 0
    infinite_resamples: int = 0


# ------------------------------------------------------------------ helpers


def _sample(x: SampleLike, mu: float, min_size: int) -> tuple[FloatArray, float]:
    arr = as_sample(x, name="x", min_size=min_size)
    value = float(mu)
    if not math.isfinite(value):
        raise ValueError(f"mu (the test value) must be a finite number; got {mu!r}")
    return arr, value


def _constant(arr: FloatArray) -> bool:
    """Whether every value of ``arr`` is the same number, exactly or to within the
    rounding of the values in double precision (DECISIONS.md D74): 0.1 + 0.2 and
    0.3 are two doubles but one number as written."""
    return differences_same_within_rounding(arr, np.zeros_like(arr))


def _no_spread_error(method: str) -> ValueError:
    return ValueError(
        "every value of x is the same (exactly, or to within the rounding of the "
        f"values in double precision), so {method} has no standard deviation"
    )


def _all_mu_error(n: int, mu: float) -> ValueError:
    return ValueError(
        f"every one of the {n} values of x equals the test value {mu!r}, so the "
        "signed-rank test has nothing to rank (its statistic and the rank-biserial "
        "r are 0/0)"
    )


def _one_sample_t_value(arr: FloatArray, mu: float) -> float:
    """t = (mean - mu) / (sd / sqrt(n)) on x times its common scale: the exact mean
    minus mu rounded once, the corrected two-pass variance (``scaled_variance``)."""
    method = "the one-sample t-test"
    if _constant(arr):
        raise _no_spread_error(method)
    scale = common_scale(arr)
    n = arr.size
    term = scaled_variance(arr * scale) / n
    if not term >= TINY:
        raise ValueError(
            "the values of x vary by less than double precision can resolve next to "
            "the largest |value| (a standard deviation below roughly 1e-154 times "
            f"that value), so {method} has no accurate standard error. Check the "
            "data for entry or unit errors; rescaling does not help, since it keeps "
            "their ratio"
        )
    # scale is a power of two, so the product is exact before the one rounding.
    difference = float((exact_mean(arr) - Fraction(mu)) * Fraction(scale))
    t = difference / math.sqrt(term)
    if not math.isfinite(t):
        raise ValueError(
            "the one-sample t statistic exceeds the largest double (about 1.8e308): "
            "the test value lies too far from the values of x next to their spread. "
            "Check the data and the test value for entry or unit errors"
        )
    return t


# ------------------------------------------------------------------ parametric


def one_sample_t(
    x: SampleLike, *, mu: float = 0.0, alternative: Alternative = "two-sided"
) -> OneSampleT:
    """Student's (1908) one-sample t-test of the mean of x against ``mu``.

    t = (mean(x) - mu) / (sd(x) / sqrt(n)), df = n - 1, p from the t distribution
    following ``alternative`` (upper tails directly, never 1 - cdf). Matches R's
    ``stats::t.test(x, mu = mu, alternative = ...)`` as statsExpressions 2.2.1
    calls it for gghistostats. The mean minus mu is formed exactly and rounded
    once, so t keeps its relative accuracy when the mean lies within the rounding
    of mu (R subtracts the rounded mean); t and p do not depend on the units of x
    and mu (bit for bit under powers of two).

    Raises:
        ValueError: fewer than two values, a non-finite value or mu, every value
            the same (exactly or to within the rounding of the values, D74: R's
            ``t.test`` stops with "data are essentially constant"), a standard
            deviation below roughly 1e-154 times the largest |value|, a t above
            the largest double, or an unknown ``alternative``.
    """
    side = check_alternative(alternative)
    arr, value = _sample(x, mu, 2)
    t = _one_sample_t_value(arr, value)
    df = float(arr.size - 1)
    return OneSampleT(
        t=t,
        df=df,
        p_value=t_p_value(t, df, side),
        n=arr.size,
        mu=value,
        alternative=side,
    )


def one_sample_hedges_g(
    x: SampleLike,
    *,
    mu: float = 0.0,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Hedges' g of one sample against ``mu``: d = (mean - mu) / sd = t / sqrt(n)
    times Hedges' (1981) exact correction J(n - 1) (``hedges_j``).

    CI: the noncentral-t interval for the one-sample t at n - 1 df
    (``t_ncp_interval``; Steiger & Fouladi 1997), each bound divided by sqrt(n) and
    multiplied by J(n - 1); a one-sided ``alternative`` gives a one-sided interval
    whose far bound is infinite (ToothGrowth-greater: [-0.11, Inf]). Matches R's
    ``effectsize::hedges_g(x, mu = mu, ci = conf_level, alternative = ...)`` as
    statsExpressions 2.2.1 calls it for gghistostats; the same construction as the
    paired ``paired_hedges_g`` on the differences x - mu. With 2 values J(1) = 0,
    so g and both finite bounds are 0 by construction.

    R's interval is off at large |t|: effectsize inverts R's noncentral ``pt``,
    which loses accuracy at a large noncentrality (ToothGrowth against 100, t =
    -82.2 on 59 df: R's bounds -0.30% and -0.55% off). inferplot's noncentral t is
    accurate there (DECISIONS.md D81; tests/stats/test_one_sample.py checks the
    bounds against 40-digit mpmath inversions for |t| up to 1,000 and n up to
    2,000).

    Raises:
        ValueError: as ``one_sample_t``, |t| above 1e300, or conf_level not
            strictly between 0 and 1.
    """
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    arr, value = _sample(x, mu, 2)
    t = _one_sample_t_value(arr, value)
    n = arr.size
    df = float(n - 1)
    scale = hedges_j(df) / math.sqrt(n)
    low, high = t_ncp_interval(t, df, conf_level=level, alternative=side)
    return EffectSizeCI(
        # + 0.0 turns a zero product with a negative factor (J = 0) into +0.0.
        estimate=t * scale + 0.0,
        conf_low=low * scale + 0.0 if math.isfinite(low) else low,
        conf_high=high * scale + 0.0 if math.isfinite(high) else high,
        conf_level=level,
        sidedness=side,
    )


def jzs_bf10_one_sample(
    x: SampleLike, *, mu: float = 0.0, prior_scale: float
) -> tuple[float, float]:
    """JZS Bayes factor of one sample against ``mu``: ``(bf10, log_bf01)``.

    Rouder et al. (2009, eq. 1) with N = n and nu = n - 1 at the one-sample t
    (``one_sample_t``): ``jzs_log_bf10``, the integral the paired Bayes factor uses
    (``jzs_bf10_paired`` is this function on the differences); two-sided whatever
    the frequentist ``alternative`` is. Matches R's ``BayesFactor::ttestBF(x, mu =
    mu, rscale = prior_scale)`` as statsExpressions 2.2.1 calls it for gghistostats
    with ``bf.prior`` (0.707 by default, not sqrt(2)/2). BayesFactor's own
    quadrature misses the accurate value at large |t| (probe-onesamplebf: within
    5.2e-9 up to |t| 8, 2.6e-6 at 12, +68% at n 5 and |t| 1,000; ToothGrowth
    against 100, +6.1e-4); this function computes the accurate one (DECISIONS.md
    D61, D81; within 1e-13 of 40-digit references on the probe's 304 rows).
    ``bf10`` is ``inf`` when it overflows a double; ``log_bf01`` = -log(BF10)
    stays finite.

    Raises:
        ValueError: as ``one_sample_t``, or prior_scale is not a finite positive
            number.
    """
    arr, value = _sample(x, mu, 2)
    t = _one_sample_t_value(arr, value)
    n = float(arr.size)
    log_bf10 = jzs_log_bf10(t, n, n - 1.0, prior_scale=prior_scale)
    try:
        bf10 = math.exp(log_bf10)
    except OverflowError:
        bf10 = math.inf
    return bf10, -log_bf10


# ------------------------------------------------------------------ nonparametric


def one_sample_signed_rank(
    x: SampleLike, *, mu: float = 0.0, alternative: Alternative = "two-sided"
) -> SignedRank:
    """Wilcoxon's (1945) signed-rank test of x against ``mu``, normal
    approximation: ``signed_rank`` of x and mu, i.e. of the differences x - mu.

    The differences are R's: x - mu rounded to the nearest double, as
    ``stats::wilcox.test(x, mu = mu)`` forms them; those that are 0 (x equal to
    mu) are dropped, the others ranked by |x - mu| with mid-ranks, and V is the
    sum of the ranks of the positive ones. Ties are ties of these rounded
    differences (D64's rule for the paired test): two values equally far from mu
    as written can get different ranks (iris Sepal.Length against 5.8: |5.7 - 5.8|
    is 0.09999999999999964 and |5.9 - 5.8| is 0.10000000000000053; six such pairs
    in iris, one in ToothGrowth against 18), as in R, which the fixtures follow:
    ranking the differences as written gives V = 5288 instead of R's 5358.5 on
    iris-sepal and 1031 instead of 1030.5 on the ToothGrowth cases. Always the
    normal approximation with the tie and continuity corrections, never the
    exact distribution (R: ``exact = FALSE, correct = TRUE``, as
    statsExpressions 2.2.1 calls it for gghistostats).
    ``n`` counts every value, ``n_nonzero`` those ranked.

    Raises:
        ValueError: x is empty or has a non-finite value, mu is not finite, every
            value equals mu (R: V = 0 and p = NaN), or ``alternative`` is unknown.
    """
    arr, value = _sample(x, mu, 1)
    if bool((arr == value).all()):
        raise _all_mu_error(arr.size, value)
    return signed_rank(arr, np.full(arr.size, value), alternative=alternative)


def one_sample_rank_biserial(
    x: SampleLike,
    *,
    mu: float = 0.0,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Rank-biserial correlation of one sample against ``mu`` with a Fisher-z CI.

    r = (W+ - W-) / (W+ + W-), W+ and W- the rank sums of the positive and the
    negative differences x - mu among the nonzero ones (Kerby 2014, Comprehensive
    Psychology 3: 11.IT.3.1), ranked as ``one_sample_signed_rank`` ranks them.

    CI: tanh(atanh(r) -/+ q SE), SE = sqrt(m (m + 1)(2m + 1) / 6) / (m (m + 1) /
    2), with m the number of NONZERO differences, and q the standard normal
    quantile at (1 + conf_level) / 2; a one-sided ``alternative`` gives a
    one-sided interval at ``conf_level``: "greater" [lower, 1], "less" [-1,
    upper]. This differs from the paired r (``paired_rank_biserial``), whose SE
    counts every pair, zeros included: on the fixtures edge_one_ties (6 values
    equal to mu) and edge_one_outlier (1) every bound misses with every value
    counted. With |r| = 1 the bound on r's side is r itself. Matches R's
    ``effectsize::rank_biserial(x, mu = mu, ci = conf_level, alternative = ...)``
    as statsExpressions 2.2.1 calls it for gghistostats.

    Raises:
        ValueError: as ``one_sample_signed_rank``, or conf_level not strictly
            between 0 and 1.
    """
    side = check_alternative(alternative)
    level = check_conf_level(conf_level)
    arr, value = _sample(x, mu, 1)
    if bool((arr == value).all()):
        raise _all_mu_error(arr.size, value)
    ranks = _signed_ranks(arr, np.full(arr.size, value))
    r = float(Fraction(ranks.plus2 - ranks.minus2, ranks.plus2 + ranks.minus2))
    m = ranks.n_nonzero
    se = math.sqrt(m * (m + 1) * (2 * m + 1) / 6.0) / (m * (m + 1) / 2.0)
    tail = (1.0 - level) / 2.0 if side == "two-sided" else 1.0 - level
    q = -float(special.ndtri(tail))
    with np.errstate(divide="ignore"):
        z = float(np.arctanh(r))
    low = -1.0 if side == "less" else math.tanh(z - q * se)
    high = 1.0 if side == "greater" else math.tanh(z + q * se)
    return EffectSizeCI(
        estimate=r, conf_low=low, conf_high=high, conf_level=level, sidedness=side
    )


# ------------------------------------------------------------------ robust

# The trimcibt rule never takes the most extreme resample: from this order on the
# critical value would be the largest |T*| at any level.
_TOO_EXTREME_MESSAGE: Final = (
    "conf_level={level!r} is too close to {end} for the bootstrap-t interval from "
    "{count:,} bootstrap resamples: its critical value would be the {which} "
    "resampled |T*| whatever the level. conf_level must lie between about "
    "{low:g} and {high:g}"
)


def bootstrap_t_critical_value(
    ordered: NDArray[np.float64], conf_level: float
) -> float:
    """trimcibt's critical value: with alpha = 1 - conf_level computed in floating
    point and B sorted values |T*|, the round((1 - alpha) B)-th smallest (R's
    rounding, half to even on the binary value; Python's ``round`` does the same).
    probe-bootrules, R's own sorted |T*| at B = 39, 40, 99, 100, 199 and 200 and
    95% and 90%: 12 of 12.

    Raises:
        ValueError: the order would be 0 (conf_level below about 1 / (2B)) or B
            (the largest |T*|: conf_level above about 1 - 1 / (2B)), where the
            critical value would no longer change with the level.
    """
    count = ordered.size
    alpha = 1.0 - conf_level
    k = round((1.0 - alpha) * count)
    if not 1 <= k < count:
        near_one = k >= count
        raise ValueError(
            _TOO_EXTREME_MESSAGE.format(
                level=conf_level,
                end="1" if near_one else "0",
                count=count,
                which="largest" if near_one else "smallest",
                low=1.5 / max(count, 1),
                high=1.0 - 1.5 / max(count, 1),
            )
        )
    return float(ordered[k - 1])


def trimmed_mean_bootstrap_t(
    x: SampleLike,
    *,
    mu: float = 0.0,
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: SeedLike = 0,
) -> BootstrapTrimmedMeanTest:
    """The symmetric bootstrap-t test and interval of a trimmed mean (Wilcox 2017,
    Introduction to Robust Estimation and Hypothesis Testing, 4th ed., section
    4.4.3, the symmetric two-sided version; R: ``WRS2::trimcibt(x, nv = mu, tr,
    alpha = 1 - conf_level)``).

    With g = floor(tr n) values trimmed at each end and s_w the winsorized
    standard deviation: se = s_w / ((1 - 2 tr) sqrt(n)) and t = (trimmed mean -
    mu) / se, exact (the trimmed mean minus mu formed exactly and rounded once).
    Bootstrap: BOOTSTRAP_RESAMPLES resamples of n values drawn with replacement;
    T*_b = |trimmed mean*_b - trimmed mean| / se*_b. The interval is trimmed mean
    -/+ c se with c the round((1 - alpha) B)-th smallest T*
    (``bootstrap_t_critical_value``, alpha = 1 - conf_level computed in floating
    point), and p is the share of T* at least |t|. Read from the oracle: the
    replay of trimcibt reproduces ggstatsplot's 200-resample values on all 17
    robust gghistostats references, and probe-bootrules pins the order statistic
    on R's own sorted T*. ggstatsplot runs 200 resamples; inferplot runs
    ``BOOTSTRAP_RESAMPLES`` from its own stream, so the bounds and p differ from
    ggstatsplot's by Monte Carlo noise (compared with the fixtures' 20,000-resample
    references). ``alternative`` is not an argument: it has no effect in
    ggstatsplot (ToothGrowth-greater-robust equals the two-sided case).

    The resamples are drawn from the deviations of x from its rounded trimmed
    mean, in the order of their absolute value (then their value), so the row
    order of x does not matter, the scale of x does not either, and x and mu
    shifted together, or both negated, give the same T* (a negated sample draws
    the mirror images, except where two values lie exactly as far above as below
    the trimmed mean).

    A resample whose winsorized values are all one number has s_w* = 0 exactly:
    T* is 0 / 0 when that number is the sample's trimmed mean and infinite
    otherwise, decided exactly (``exact_zero_sum``, D74's rule). An infinite T* is
    kept (it counts toward p, and c is infinite when it falls among them); a 0 / 0
    one is left out and c and p come from the others (D65's rule), counted in
    ``undefined_resamples``.

    Raises:
        ValueError: fewer than two values left after trimming (n - 2 floor(tr n)
            < 2), a non-finite value or mu, winsorized values that are all the
            same (exactly or to within the rounding of the values, D74) or vary by
            less than double precision resolves next to the largest |value|, a
            conf_level so close to 0 or 1 that c would be the most extreme T*, so
            many 0 / 0 resamples that that happens, or an invalid ``tr``,
            ``conf_level`` or ``seed``.
    """
    t_trim = check_trim(tr)
    level = check_conf_level(conf_level)
    rng_seed = check_bootstrap_seed(seed)
    arr, value = _sample(x, mu, 2)
    method = "the bootstrap-t test of the trimmed mean"
    n = arr.size
    g = trim_count(n, t_trim)
    h = n - 2 * g
    if h < 2:
        raise ValueError(
            f"trimming {g} of {n} values at each end leaves {h}; {method} needs at "
            "least 2 values after trimming (use more values or a smaller tr)"
        )
    scale = common_scale(arr)
    ordered = np.sort(arr * scale)
    winsorized = winsorize_sorted(ordered, g)
    if _constant(winsorized / scale):
        raise ValueError(
            f"x has a winsorized variance of 0 (its values from the {g + 1}-th "
            f"smallest to the {g + 1}-th largest are all the same, exactly or to "
            "within the rounding of the values in double precision), so "
            f"{method} is undefined"
        )
    wvar = scaled_variance(winsorized)
    shrink = (1.0 - 2.0 * t_trim) * math.sqrt(n)
    if not wvar / n >= TINY:
        raise ValueError(
            "the values of x vary by less than double precision can resolve next to "
            "the largest |value| (a winsorized standard deviation below roughly "
            f"1e-154 times that value), so {method} has no accurate standard error. "
            "Check the data for entry or unit errors"
        )
    se = math.sqrt(wvar) / shrink
    tmean = exact_trimmed_mean(ordered, t_trim)
    t = float(tmean - Fraction(value) * Fraction(scale)) / se
    if not math.isfinite(t):
        raise ValueError(
            "the bootstrap-t statistic exceeds the largest double (about 1.8e308): "
            "the test value lies too far from the values of x next to their spread"
        )
    centre = float(tmean)
    deviations = ordered - centre
    deviations = deviations[np.lexsort((deviations, np.abs(deviations)))]
    middle = np.sort(deviations)[g : n - g]
    middle_list = middle.tolist()
    offset = float(exact_mean(middle))  # trimmed mean of the deviations

    def t_stars(draws: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        tmeans, wvars = rows_trimmed(draws[0], t_trim)
        with np.errstate(divide="ignore", invalid="ignore"):
            out: NDArray[np.float64] = np.abs(tmeans - offset) * shrink / np.sqrt(wvars)
        # A winsorized variance of 0: every winsorized value of the resample is one
        # deviation v (the row's trimmed mean, exactly); T* is 0 / 0 when v is the
        # sample's trimmed mean of the deviations, else infinite, decided exactly.
        for row in (wvars == 0.0).nonzero()[0].tolist():
            sign = exact_zero_sum([float(tmeans[row])] * h, middle_list)
            out[row] = math.nan if sign == 0.0 else math.inf
        return out

    rng = np.random.Generator(np.random.PCG64(rng_seed))
    replicates = split_replicates(bootstrap_replicates(rng, [deviations], [n], t_stars))
    defined = replicates.defined
    if defined.size == 0:
        raise ValueError(
            f"T* is undefined (0 / 0) in every one of the {BOOTSTRAP_RESAMPLES:,} "
            f"bootstrap resamples of {method} (the data have too few distinct values "
            "for this bootstrap)"
        )
    try:
        critical = bootstrap_t_critical_value(defined, level)
    except ValueError as error:
        if not replicates.undefined:
            raise
        raise ValueError(
            f"T* is undefined (0 / 0) in {replicates.undefined:,} of "
            f"{BOOTSTRAP_RESAMPLES:,} bootstrap resamples, because a resample's "
            "winsorized values are all its trimmed mean; that leaves "
            f"{defined.size:,}, too few for the bootstrap-t interval at "
            f"conf_level={level!r}: {error}"
        ) from None
    p_value = float(np.count_nonzero(defined >= abs(t))) / defined.size
    half = critical * se
    return BootstrapTrimmedMeanTest(
        statistic=t,
        p_value=p_value,
        estimate=centre / scale,
        se=se / scale,
        critical_value=critical,
        conf_low=(centre - half) / scale,
        conf_high=(centre + half) / scale,
        conf_level=level,
        n=n,
        mu=value,
        tr=t_trim,
        seed=rng_seed,
        resamples=BOOTSTRAP_RESAMPLES,
        undefined_resamples=replicates.undefined,
        infinite_resamples=replicates.infinite,
    )
