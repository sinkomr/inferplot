"""Paired t-test, Hedges' g of paired differences and the one-way repeated-measures
ANOVA with the Greenhouse-Geisser correction (``withinstats``, type parametric).

Every statistic here is computed on the conditions times one power of two that
brings the largest |value| of all of them into [1/2, 1) (``common_scale``), which is
exact, so the results do not depend on the units of the data (bit for bit under
powers of two), and no square overflows or underflows.

Paired differences are formed without rounding error: Knuth's TwoSum gives each
difference a_i - b_i exactly as s_i + e_i (s_i = fl(a_i - b_i)); their mean is the
exact difference of the exact means, rounded once (``exact_mean``, DECISIONS.md
D49); their deviations from it are (s_i - m) + e_i, and the sum of squares is the
corrected two-pass sum (Chan, Golub & LeVeque 1983, eq. 1.7) of those deviations
with exact sums (``math.fsum``). So t keeps its relative accuracy when the
differences vary only in their last bits or sit on a large common offset, where
R's ``t.test`` (``mean(x - y)`` and ``sd(x - y)`` of the rounded differences) loses
it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import special

from inferplot.stats.descriptive import (
    FloatArray,
    SampleLike,
    as_sample,
    common_scale,
    deviations_from_weighted_mean,
    differences_same_within_rounding,
    exact_mean,
    mean,
)
from inferplot.stats.effect_sizes import EffectSizeCI, hedges_j
from inferplot.stats.noncentral import check_conf_level, f_ncp_bound, t_ncp_interval
from inferplot.stats.padjust import PAdjustMethod, p_adjust
from inferplot.stats.parametric import TINY, Alternative, check_alternative, t_p_value

__all__ = [
    "PairedT",
    "PairedTPair",
    "RepeatedMeasuresAnova",
    "paired_differences_constant",
    "paired_hedges_g",
    "paired_t",
    "paired_t_pairs",
    "rm_anova",
    "rm_partial_omega_squared",
]


@dataclass(frozen=True, slots=True)
class PairedT:
    """The paired t-test of a - b (R: ``stats::t.test(a, b, paired = TRUE)``)."""

    t: float
    df: float
    p_value: float
    n: int
    alternative: Alternative


@dataclass(frozen=True, slots=True)
class PairedTPair:
    """The paired t-test of conditions ``i`` < ``j`` (input order): t of
    condition i minus condition j, its df, the raw two-sided p and the adjusted p."""

    i: int
    j: int
    t: float
    df: float
    p_value: float
    p_adjusted: float


@dataclass(frozen=True, slots=True)
class RepeatedMeasuresAnova:
    """The one-way repeated-measures ANOVA with the Greenhouse-Geisser correction.

    ``df1`` = k - 1 and ``df2`` = (k - 1)(n - 1) are the uncorrected df;
    ``df1_corrected`` and ``df2_corrected`` are them times ``epsilon``, and
    ``p_value`` is the upper tail of F at the corrected df (what ggstatsplot
    reports); ``p_uncorrected`` is the tail at the uncorrected df.
    """

    f: float
    df1: float
    df2: float
    epsilon: float
    df1_corrected: float
    df2_corrected: float
    p_value: float
    p_uncorrected: float
    n: int
    k: int


# --------------------------------------------------------------------------- #
# paired differences
# --------------------------------------------------------------------------- #


def _two_sum(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Knuth's TwoSum (TAOCP vol. 2, 4.2.2, Theorem B): s = fl(a + b) and the
    rounding error e, so that a + b = s + e exactly (no overflow here: every value
    lies in [-1, 1) in the common scale)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _paired_samples(a: SampleLike, b: SampleLike) -> tuple[FloatArray, FloatArray]:
    x = as_sample(a, name="a", min_size=2)
    y = as_sample(b, name="b", min_size=2)
    if x.size != y.size:
        raise ValueError(
            f"paired samples need the same number of values; got {x.size} and {y.size}"
        )
    return x, y


def _exact_differences(
    x: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Both samples times their common scale and the exact differences of the
    scaled values as (s, e) with x_i - y_i = s_i + e_i."""
    scale = common_scale(x, y)
    xs, ys = x * scale, y * scale
    s, e = _two_sum(xs, -ys)
    return xs, ys, s, e


def paired_differences_constant(a: SampleLike, b: SampleLike) -> bool:
    """Whether every difference a_i - b_i is the same number (all zero included),
    exactly or to within the rounding of the values in double precision
    (``differences_same_within_rounding``): the paired t-test then has no standard
    error. Decimal data in which every subject changes by 0.1 are constant in this
    sense, although the differences of the stored doubles vary in their 16th digit.

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, or
            the samples differ in length.
    """
    x, y = _paired_samples(a, b)
    return differences_same_within_rounding(x, y)


def _paired_t_value(x: FloatArray, y: FloatArray) -> float:
    """t = mean(d) / (sd(d) / sqrt(n)) of the exact differences d = x - y."""
    if differences_same_within_rounding(x, y):
        raise ValueError(
            "every difference between the paired values is the same (exactly, or to "
            "within the rounding of the values in double precision), so the paired "
            "t-test has no standard error (the standard deviation of the differences "
            "is 0)"
        )
    xs, ys, s, e = _exact_differences(x, y)
    n = x.size
    # The exact mean difference, rounded once.
    m = float(exact_mean(xs) - exact_mean(ys))
    deviations = (s - m) + e
    squares = math.fsum((deviations * deviations).tolist())
    correction = math.fsum(deviations.tolist()) ** 2 / n
    term = max(0.0, (squares - correction) / (n - 1)) / n
    if not term >= TINY:
        raise ValueError(
            "the differences between the paired values vary by less than double "
            "precision can resolve next to the largest |value| (a standard deviation "
            "of the differences below roughly 1e-154 times that value), so the paired "
            "t-test has no accurate standard error. Check the data for entry or unit "
            "errors; rescaling does not help, since it keeps their ratio"
        )
    return m / math.sqrt(term)


def paired_t(
    a: SampleLike, b: SampleLike, *, alternative: Alternative = "two-sided"
) -> PairedT:
    """Paired t-test of a - b (Student 1908): the one-sample t of the differences.

    t = mean(d) / (sd(d) / sqrt(n)) with d_i = a_i - b_i, df = n - 1, p from the
    t distribution following ``alternative`` (upper tails directly, never 1 - cdf).
    Matches R's ``stats::t.test(a, b, paired = TRUE, alternative = ...)`` as
    statsExpressions 2.2.1 calls it for ggwithinstats (t signed as the first level
    minus the second). Computed from the exact differences (module docstring), so
    t, df and p are the same, bit for bit, for both samples times any power of two
    that keeps them normal doubles.

    Limits: the scaled squared standard error var(d) / n must be at least
    ``TINY`` = 2^-1022 (a standard deviation of the differences above roughly
    1e-154 times the largest |value|), so |t| < 2^512.

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, the
            samples differ in length, every difference is the same (also all
            zero), the differences vary too little next to the values (above), or
            ``alternative`` is not one of "two-sided", "less", "greater".
    """
    side = check_alternative(alternative)
    x, y = _paired_samples(a, b)
    t = _paired_t_value(x, y)
    df = float(x.size - 1)
    return PairedT(
        t=t, df=df, p_value=t_p_value(t, df, side), n=x.size, alternative=side
    )


def paired_hedges_g(
    a: SampleLike,
    b: SampleLike,
    *,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Hedges' g of paired samples: the standardized mean difference of the
    differences, d_z = mean(d) / sd(d) = t / sqrt(n), times Hedges' (1981) exact
    correction J(n - 1) (``hedges_j``).

    CI: the noncentral-t interval for the paired t at n - 1 df
    (``t_ncp_interval``; Steiger & Fouladi 1997), each bound divided by sqrt(n)
    and multiplied by J(n - 1); a one-sided ``alternative`` gives a one-sided
    interval whose far bound is infinite. Matches R's ``effectsize::hedges_g(a,
    b, paired = TRUE, ci = conf_level, alternative = ...)`` as statsExpressions
    2.2.1 calls it for ggwithinstats: not d_av (the mean difference over the
    average SD of the conditions), which misses every fixture. With 2 subjects
    J(1) = 0, so g and both finite bounds are 0 (+0.0) by construction.

    Raises:
        ValueError: as ``paired_t``, or conf_level is not strictly between 0 and
            1.
    """
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    x, y = _paired_samples(a, b)
    t = _paired_t_value(x, y)
    n = x.size
    df = float(n - 1)
    j = hedges_j(df)
    scale = j / math.sqrt(n)
    low, high = t_ncp_interval(t, df, conf_level=level, alternative=side)
    return EffectSizeCI(
        # + 0.0 turns a zero product with a negative factor (J = 0) into +0.0.
        estimate=t * scale + 0.0,
        conf_low=low * scale + 0.0 if math.isfinite(low) else low,
        conf_high=high * scale + 0.0 if math.isfinite(high) else high,
        conf_level=level,
        sidedness=side,
    )


def paired_t_pairs(
    conditions: Sequence[SampleLike], *, p_adjust_method: PAdjustMethod = "holm"
) -> tuple[PairedTPair, ...]:
    """Paired t-tests of every pair of conditions i < j (input order), then
    ``p_adjust`` over all k(k - 1)/2 raw p-values.

    Each pair is ``paired_t`` of its own two conditions (t of condition i minus
    condition j, df = n - 1, two-sided p): each pair's own standard deviation of
    differences, not a pooled one. Matches R's ``stats::pairwise.t.test(y, x,
    paired = TRUE, p.adjust.method = "none")`` followed by ``stats::p.adjust``, as
    statsExpressions 2.2.1's ``pairwise_comparisons()`` reports them for
    ggwithinstats.

    Raises:
        ValueError: fewer than two conditions, conditions of different lengths, a
            pair as ``paired_t`` refuses it, or an unknown method.
    """
    if len(conditions) < 2:
        raise ValueError(
            f"pairwise tests need at least two conditions; got {len(conditions)}"
        )
    p_adjust([], p_adjust_method)  # an unknown method fails before any work
    rows: list[tuple[int, int, PairedT]] = []
    k = len(conditions)
    for i in range(k):
        for j in range(i + 1, k):
            rows.append((i, j, paired_t(conditions[i], conditions[j])))
    adjusted = p_adjust([row[2].p_value for row in rows], p_adjust_method)
    return tuple(
        PairedTPair(
            i=i, j=j, t=test.t, df=test.df, p_value=test.p_value, p_adjusted=adj
        )
        for (i, j, test), adj in zip(rows, adjusted, strict=True)
    )


# --------------------------------------------------------------------------- #
# repeated-measures ANOVA
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Decomposition:
    """Sums of squares of the subjects-by-conditions table in the common scale,
    and the cross-product matrix of its residuals (for epsilon)."""

    n: int
    k: int
    ss_conditions: float
    ss_subjects: float
    ss_error: float
    cross: FloatArray
    additive: bool
    """Every residual is exactly 0: each subject's values are the condition means
    plus a constant of its own (for such data the differences to the first
    condition are column constants, so every step below is exact)."""


def _matrix(conditions: Sequence[SampleLike]) -> FloatArray:
    """The conditions as the columns of an n x k matrix (complete subjects)."""
    if len(conditions) < 2:
        raise ValueError(
            f"a repeated-measures ANOVA needs at least two conditions; got "
            f"{len(conditions)}"
        )
    columns = [
        as_sample(c, name=f"condition {i + 1}", min_size=2)
        for i, c in enumerate(conditions)
    ]
    sizes = {c.size for c in columns}
    if len(sizes) != 1:
        raise ValueError(
            "every condition needs one value per subject (the same number of values); "
            f"got {[c.size for c in columns]}"
        )
    out = np.empty((columns[0].size, len(columns)), dtype=np.float64)
    for j, column in enumerate(columns):
        out[:, j] = column
    return out


def _centre_columns(values: FloatArray) -> FloatArray:
    """Each column minus its correctly rounded mean (each difference rounded once,
    relative to itself)."""
    return values - np.array([mean(values[:, j]) for j in range(values.shape[1])])


def _decompose(conditions: Sequence[SampleLike]) -> _Decomposition:
    """SS of conditions, subjects and error of y_ij = mu + s_i + c_j + r_ij.

    * Conditions: n sum_j (mean_j - grand mean)^2, the deviations formed exactly
      from the exact column means and rounded once (``deviations_from_weighted_
      mean``).
    * Subjects: k sum_i (rowmean_i - grand mean)^2, from the values minus their
      column means (each difference rounded once); the rounding of the column
      means shifts every row mean alike and is removed by the corrected two-pass
      sum.
    * Error: sum r_ij^2 of the doubly centred table. The subject effects are
      removed first and exactly, from the differences to the first condition
      (TwoSum: z_ij = y_ij - y_i1 = hi + lo); then the columns of hi are centred on
      their rounded means and lo added back, which leaves values of the size of the
      residuals (the rounding of a mean is a column constant); centring the rows
      and then the columns of that gives r. So each residual is accurate to a few
      units in its last place, also when subject or condition effects dwarf the
      error (R's ``aov`` projects the raw values).
    """
    raw = _matrix(conditions)
    n, k = raw.shape
    y = raw * common_scale(*(raw[:, j] for j in range(k)))
    columns = [np.ascontiguousarray(y[:, j]) for j in range(k)]
    means: list[Fraction] = [exact_mean(c) for c in columns]
    deviations = deviations_from_weighted_mean([1.0] * k, means)
    ss_conditions = n * math.fsum(d * d for d in deviations)

    centred = _centre_columns(y)
    row_means = np.array([math.fsum(row) / k for row in centred.tolist()])
    ss_subjects = k * max(
        0.0,
        math.fsum((row_means * row_means).tolist())
        - math.fsum(row_means.tolist()) ** 2 / n,
    )

    hi, lo = _two_sum(y, -y[:, :1])
    w = _centre_columns(hi) + lo
    w = w - (w.sum(axis=1) / k)[:, np.newaxis]
    residuals = w - w.mean(axis=0)
    ss_error = math.fsum((residuals * residuals).ravel().tolist())
    return _Decomposition(
        n=n,
        k=k,
        ss_conditions=ss_conditions,
        ss_subjects=ss_subjects,
        ss_error=ss_error,
        cross=residuals.T @ residuals,
        # No residual is exactly nonzero, or every condition differs from the
        # first by one amount for every subject to within the rounding of the
        # values: then the residuals are rounding, not data.
        additive=not bool((residuals != 0.0).any())
        or all(
            differences_same_within_rounding(
                np.ascontiguousarray(raw[:, j]), np.ascontiguousarray(raw[:, 0])
            )
            for j in range(1, k)
        ),
    )


def _checked_decomposition(conditions: Sequence[SampleLike]) -> _Decomposition:
    parts = _decompose(conditions)
    df_error = (parts.k - 1) * (parts.n - 1)
    if parts.additive:
        raise ValueError(
            "the conditions differ by the same amounts for every subject (each "
            "subject's values are the condition means plus a constant, exactly or to "
            "within the rounding of the values in double precision), so the "
            "repeated-measures ANOVA has no error variance"
        )
    if not parts.ss_error / df_error >= TINY:
        raise ValueError(
            "the residuals of the repeated-measures ANOVA (what is left after the "
            "subject and condition effects) are smaller than double precision can "
            "resolve next to the largest |value| (below roughly 1e-154 times it), so "
            "the error variance is not accurate. Check the data for entry or unit "
            "errors; rescaling does not help, since it keeps their ratio"
        )
    return parts


def _greenhouse_geisser(parts: _Decomposition) -> float:
    """Greenhouse & Geisser's (1959) epsilon, Box's (1954) estimator with the
    sample covariance matrix S of the conditions: tr(S*)^2 / ((k - 1) tr(S*^2)),
    S* = C S C with C the centring matrix, which is the covariance matrix of the
    residuals: S* = R'R / (n - 1) (the n - 1 cancels). It lies in [1 / (k - 1), 1]
    by the Cauchy-Schwarz inequality on the k - 1 nonzero eigenvalues; a rounding
    beyond either end is clamped back (it is 1 / (k - 1) exactly when S* has rank
    1, as with two subjects)."""
    cross = parts.cross
    trace = float(cross.trace())
    frobenius = math.fsum((cross * cross).ravel().tolist())
    epsilon = trace * trace / ((parts.k - 1) * frobenius)
    return min(1.0, max(1.0 / (parts.k - 1), epsilon))


def rm_anova(conditions: Sequence[SampleLike]) -> RepeatedMeasuresAnova:
    """One-way repeated-measures ANOVA with the Greenhouse-Geisser correction.

    ``conditions`` are the columns of the subjects-by-conditions table (complete
    subjects, entry i of every condition from subject i). F = MS_conditions /
    MS_error with MS = SS / df, df1 = k - 1 and df2 = (k - 1)(n - 1), from the
    decomposition into subject, condition and error sums of squares (``_decompose``;
    e.g. Maxwell & Delaney 2004, Designing Experiments and Analyzing Data, ch. 11).
    The Greenhouse & Geisser (1959, Psychometrika 24: 95-112) epsilon (Box 1954)
    multiplies both df, and p is the F upper tail at the corrected df (``fdtrc``,
    not 1 - cdf). Matches R's ``afex::aov_ez(..., anova_table = list(correction =
    "GG"))`` (``car::Anova``'s univariate tests with ``Pr(>F[GG])``) as
    statsExpressions 2.2.1 calls it for ggwithinstats: always Greenhouse-Geisser,
    never Huynh-Feldt (probe-rmanova has afex's three corrections, the epsilons
    and the error strata).

    Computed on the table times its ``common_scale``, so F, the df, epsilon and p
    do not depend on the units of the data (bit for bit under powers of two).

    Raises:
        ValueError: fewer than two conditions, conditions of unequal length or with
            fewer than two values or a non-finite value, an error sum of squares
            of 0 (every subject's profile is the condition means plus a constant)
            or below double precision next to the largest |value|, or an F above
            the largest double.
    """
    parts = _checked_decomposition(conditions)
    df1 = float(parts.k - 1)
    df2 = float((parts.k - 1) * (parts.n - 1))
    f = (parts.ss_conditions / df1) / (parts.ss_error / df2)
    if math.isinf(f):
        raise ValueError(
            "the F statistic exceeds the largest double (about 1.8e308), so it cannot "
            "be reported. Check the data for entry or unit errors"
        )
    epsilon = _greenhouse_geisser(parts)
    df1_gg, df2_gg = df1 * epsilon, df2 * epsilon
    return RepeatedMeasuresAnova(
        f=f,
        df1=df1,
        df2=df2,
        epsilon=epsilon,
        df1_corrected=df1_gg,
        df2_corrected=df2_gg,
        p_value=float(special.fdtrc(df1_gg, df2_gg, f)),
        p_uncorrected=float(special.fdtrc(df1, df2, f)),
        n=parts.n,
        k=parts.k,
    )


def rm_partial_omega_squared(
    conditions: Sequence[SampleLike], *, conf_level: float = 0.95
) -> EffectSizeCI:
    """Partial omega squared of the condition effect of a one-way
    repeated-measures design, with a one-sided noncentral-F CI.

    Estimate from the error strata (Olejnik & Algina 2003, Psychological Methods 8:
    434-447, Table 2, one within-subjects factor): omega = df_c (MS_c - MS_e) /
    (SS_c + SS_e + SS_s + MS_s), with condition (c), error (e) and subject (s) sums
    of squares and mean squares (MS_s = SS_s / (n - 1)), floored at 0: what R's
    ``effectsize::omega_squared(partial = TRUE)`` reports for the ``afex`` model
    ggwithinstats fits (probe-rmanova: 0.0783 on bugs_long, where the F-based
    ``F_to_omega2`` gives 0.181).

    CI, effectsize's construction (as for betweenstats, DECISIONS.md D33), at the
    uncorrected df df1 = k - 1, df2 = (k - 1)(n - 1), although the subtitle reports
    the Greenhouse-Geisser df: the estimate read as a partial eta squared gives
    F_w = omega df2 / ((1 - omega) df1); lambda solves P(F'(df1, df2, lambda) <=
    F_w) = conf_level (``f_ncp_bound``; Steiger 2004); the lower bound is lambda /
    (lambda + df2) and the upper bound 1. F_w is computed as df2 / df1 times
    df_c (MS_c - MS_e) / (SS_e + SS_s + MS_s + df_c MS_e), which is omega / (1 -
    omega) without the subtraction from 1. A zero estimate gives [0, 1].

    Raises:
        ValueError: as ``rm_anova``, or conf_level is not strictly between 0 and 1.
    """
    level = check_conf_level(conf_level)
    parts = _checked_decomposition(conditions)
    df_c = float(parts.k - 1)
    df_e = float((parts.k - 1) * (parts.n - 1))
    ms_c = parts.ss_conditions / df_c
    ms_e = parts.ss_error / df_e
    ms_s = parts.ss_subjects / (parts.n - 1)
    numerator = df_c * (ms_c - ms_e)
    if numerator > 0.0:
        estimate = numerator / (
            parts.ss_conditions + parts.ss_error + parts.ss_subjects + ms_s
        )
        rest = parts.ss_error + parts.ss_subjects + ms_s + df_c * ms_e
        f_w = df_e / df_c * (numerator / rest)
    else:
        estimate, f_w = 0.0, 0.0
    ncp = f_ncp_bound(f_w, df_c, df_e, prob=level)
    return EffectSizeCI(
        estimate=estimate,
        conf_low=ncp / (ncp + df_e) if math.isfinite(ncp) else 1.0,
        conf_high=1.0,
        conf_level=level,
        sidedness="greater",
    )
