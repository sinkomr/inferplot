"""Robust methods for dependent groups on trimmed means: Yuen's test for paired
samples, the trimmed-means repeated-measures ANOVA, rmmcp's pairwise comparisons of
the differences, and the AKP effect size of paired differences with its average
over all pairs of conditions (``withinstats``, type robust).

The R references are WRS2's ``yuend``, ``rmanova``, ``rmmcp``, ``dep.effect`` and
``wmcpAKP`` as statsExpressions 2.2.1 calls them for ggstatsplot 1.1.4 (SPEC.md
section 4, "withinstats"). The methods are implemented from Wilcox (2017,
Introduction to Robust Estimation and Hypothesis Testing, 4th ed., Academic Press:
dependent groups), Huynh & Feldt (1976) and Algina, Keselman & Penfield (2005); the
details WRS2 adds were read off the oracle's probe-wrs2within (yuend's 84 rows,
rmanova's 84 rows with rmmcp's tables, dep.effect's and wmcpAKP's outputs) and the
ggwithinstats fixtures' 20,000-resample references (DECISIONS.md log). WRS2 is
GPL; its source was not read.

Every statistic is computed on the conditions times their ``common_scale`` (a power
of two, exact), so the results do not depend on the units of the data. Paired
differences are formed exactly (Knuth's TwoSum: a_i - b_i = s_i + e_i) and ordered
by their exact values, trimmed means of them and of the conditions are exact
rationals rounded once, and variances are corrected two-pass sums of deviations
formed from those exact values, so a difference that varies only in its last bits,
or sits on a large common offset, keeps its relative accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from inferplot.stats.descriptive import (
    FloatArray,
    SampleLike,
    as_sample,
    common_scale,
    deviations_from_weighted_mean,
    differences_same_within_rounding,
    exact_mean,
)
from inferplot.stats.noncentral import check_conf_level
from inferplot.stats.nonparametric import (
    BOOTSTRAP_RESAMPLES,
    BootstrapEffectSizeCI,
    SeedLike,
    check_bootstrap_seed,
)
from inferplot.stats.padjust import PAdjustMethod, p_adjust
from inferplot.stats.parametric import TINY
from inferplot.stats.robust import (
    check_trim,
    exact_trimmed_mean,
    f_upper_p,
    replicate_bounds,
    rounded_percentile_bounds,
    rows_trimmed,
    split_replicates,
    trim_count,
    two_sided_t_p,
    winsorize_sorted,
    winsorized_normal_variance,
)
from inferplot.stats.smm import smm_quantile
from inferplot.stats.written import written_values

__all__ = [
    "RmmcpPair",
    "TrimmedRmAnova",
    "YuenDependentT",
    "average_dependent_akp",
    "dependent_akp",
    "paired_differences_winsorize_to_constant",
    "rmmcp",
    "trimmed_rm_anova",
    "yuen_dependent_t",
]

# With two subjects the Huynh-Feldt estimate of the trimmed-means repeated-measures
# ANOVA is 0 / 0 in exact arithmetic for any data.
_MIN_SUBJECTS_HUYNH_FELDT: Final = 3

# Upper bound on the values one bootstrap chunk draws (2^22 indices, 32 MiB).
_CHUNK_VALUES: Final = 1 << 22


@dataclass(frozen=True, slots=True)
class YuenDependentT:
    """Yuen's test of the trimmed means of two dependent samples (R:
    ``WRS2::yuend``): ``t`` signed as trimmed mean(a) - trimmed mean(b) (that
    ``difference``), ``se`` its standard error, df = h - 1."""

    t: float
    df: float
    p_value: float
    difference: float
    se: float
    n: int


@dataclass(frozen=True, slots=True)
class TrimmedRmAnova:
    """The trimmed-means repeated-measures ANOVA (R: ``WRS2::rmanova``).

    ``df1`` = (J - 1) epsilon and ``df2`` = (J - 1)(h - 1) epsilon, with
    ``epsilon`` the Huynh-Feldt estimate from the winsorized values capped at 1;
    ``epsilon_gg`` is the Greenhouse-Geisser estimate it starts from.
    """

    f: float
    df1: float
    df2: float
    p_value: float
    epsilon: float
    epsilon_gg: float
    n: int
    k: int


@dataclass(frozen=True, slots=True)
class RmmcpPair:
    """rmmcp's comparison of conditions ``i`` < ``j`` (input order) on their
    differences d = condition i - condition j: ``estimate`` the trimmed mean of d,
    ``t`` = estimate / se on ``df`` = h - 1, the raw two-sided p and the adjusted p;
    the CI is estimate -/+ ``critical_value`` x se."""

    i: int
    j: int
    estimate: float
    se: float
    df: float
    t: float
    critical_value: float
    conf_low: float
    conf_high: float
    p_value: float
    p_adjusted: float


# ------------------------------------------------------------------ helpers


def _sorted(x: FloatArray) -> FloatArray:
    out = x.copy()
    out.sort()
    return out


def _joined(first: FloatArray, second: FloatArray) -> FloatArray:
    out = np.empty(first.size + second.size, dtype=np.float64)
    out[: first.size] = first
    out[first.size :] = second
    return out


def _lexical_order(s: FloatArray, e: FloatArray) -> NDArray[np.intp]:
    """The permutation that sorts by s, ties broken by e (stable sorts)."""
    order = e.argsort(kind="stable")
    return order[s[order].argsort(kind="stable")]


def _two_sum(a: FloatArray, b: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Knuth's TwoSum (TAOCP vol. 2, 4.2.2, Theorem B): s = fl(a + b) and its
    rounding error e, a + b = s + e exactly (no overflow: |a|, |b| < 1 here)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def _matrix(conditions: Sequence[SampleLike], method: str) -> tuple[FloatArray, float]:
    """The conditions as the columns of an n x k matrix times their common scale,
    and that scale."""
    columns = [
        as_sample(c, name=f"condition {i + 1}", min_size=2)
        for i, c in enumerate(conditions)
    ]
    if len({c.size for c in columns}) != 1:
        raise ValueError(
            f"{method} needs one value per subject in every condition (the same "
            f"number of values); got {[c.size for c in columns]}"
        )
    scale = common_scale(*columns)
    out = np.empty((columns[0].size, len(columns)), dtype=np.float64)
    for j, column in enumerate(columns):
        out[:, j] = column * scale
    return out, scale


def _trimmed_size(n: int, tr: float, method: str) -> tuple[int, int]:
    """g = floor(tr n) and h = n - 2 g; at least 2 values must be left."""
    g = trim_count(n, tr)
    h = n - 2 * g
    if h < 2:
        raise ValueError(
            f"trimming {g} of {n} subjects at each end leaves {h}; {method} needs at "
            "least 2 subjects after trimming (use more subjects or a smaller tr)"
        )
    return g, h


def _variance_about(s: FloatArray, e: FloatArray, centre: Fraction) -> float:
    """The sample variance of the exact values s_i + e_i whose exact mean is
    ``centre``: the corrected two-pass sum (Chan, Golub & LeVeque 1983, eq. 1.7) of
    the deviations (s_i - m) + e_i about m = ``centre`` rounded once."""
    m = float(centre)
    deviations = (s - m) + e
    n = s.size
    squares = math.fsum((deviations * deviations).tolist())
    correction = math.fsum(deviations.tolist()) ** 2 / n
    return max(0.0, (squares - correction) / (n - 1))


@dataclass(frozen=True, slots=True)
class _Differences:
    """The exact differences s_i + e_i of two scaled conditions, in their exact
    order (rounding is monotone, so the order of s with ties broken by e is the
    order of the exact values), with their trimmed mean and winsorized variance."""

    s: FloatArray
    e: FloatArray
    tmean: Fraction
    wvar: float
    constant: bool  # every winsorized difference is the same exact number


def _snapped_differences(
    x: FloatArray, y: FloatArray, scale: float
) -> tuple[FloatArray, FloatArray]:
    """The exact differences x_i - y_i of the scaled values as (s, e), s_i + e_i
    exactly, in subject order, with the differences that are equal as written
    made the same number, and those that are 0 as written made 0.

    "As written" is ``written_values`` of the unscaled values (DECISIONS.md D74):
    two subjects whose readings differ by the same amount get one difference, the
    smallest of their exact differences, and a subject whose readings are equal
    gets 0. For decimal data this restores the ties the data have (37.3 - 37.2
    and 36.6 - 36.5 are different doubles, 0.09999999999999432 and
    0.10000000000000142): without it a resample in which every winsorized
    difference is "0.1" would have a winsorized variance of about 1e-29 instead
    of 0, and an AKP of about 1e13 instead of an infinite one. Values measured to
    many digits are read as their doubles, so their differences are kept as they
    are; so are differences that no other subject shares.
    """
    s, e = _two_sum(x, -y)
    read_x = written_values(x / scale)
    read_y = written_values(y / scale)
    groups: dict[Fraction, list[int]] = {}
    for i, (a, b) in enumerate(zip(read_x, read_y, strict=True)):
        groups.setdefault(a - b, []).append(i)
    out_s, out_e = s.copy(), e.copy()
    for difference, members in groups.items():
        if difference == 0:
            out_s[members], out_e[members] = 0.0, 0.0
        elif len(members) > 1:
            index = np.array(members)
            first = int(index[_lexical_order(s[index], e[index])[0]])
            out_s[index], out_e[index] = s[first], e[first]
    return out_s, out_e


def _differences(s: FloatArray, e: FloatArray, g: int) -> _Differences:
    """``_Differences`` of the differences s_i + e_i (``_snapped_differences``)."""
    order = _lexical_order(s, e)
    s, e = s[order], e[order]
    n = s.size
    middle = _joined(s[g : n - g], e[g : n - g])
    tmean = exact_mean(middle) * 2  # (sum s + sum e) / h, exactly
    ws, we = winsorize_sorted(s, g), winsorize_sorted(e, g)
    constant = bool(s[g] == s[n - g - 1]) and bool(e[g] == e[n - g - 1])
    wmean = exact_mean(_joined(ws, we)) * 2
    return _Differences(
        s=s,
        e=e,
        tmean=tmean,
        wvar=0.0 if constant else _variance_about(ws, we, wmean),
        constant=constant,
    )


def _winsorize_column(x: FloatArray, g: int) -> FloatArray:
    """``x`` winsorized at its own order statistics, in its own row order (the
    pairing with the other conditions is kept)."""
    ordered = _sorted(x)
    return np.minimum(np.maximum(x, ordered[g]), ordered[x.size - g - 1])


def _paired(a: SampleLike, b: SampleLike) -> tuple[FloatArray, FloatArray, float]:
    x = as_sample(a, name="a", min_size=2)
    y = as_sample(b, name="b", min_size=2)
    if x.size != y.size:
        raise ValueError(
            f"paired samples need the same number of values; got {x.size} and {y.size}"
        )
    scale = common_scale(x, y)
    return x * scale, y * scale, scale


def _no_spread(what: str, method: str) -> ValueError:
    return ValueError(
        f"{what} vary by less than double precision can resolve next to the largest "
        "|value| (a winsorized standard deviation below roughly 1e-154 times that "
        f"value), so {method} has no accurate standard error. Check the data for "
        "entry or unit errors."
    )


def paired_differences_winsorize_to_constant(
    a: SampleLike, b: SampleLike, *, tr: float = 0.2
) -> bool:
    """Whether winsorizing the paired differences a_i - b_i leaves one number
    (exactly, or to within the rounding of the values in double precision:
    ``_snapped_differences``), so that their winsorized standard deviation is 0 and
    the AKP effect size and rmmcp's standard error are undefined.

    Raises:
        ValueError: as ``dependent_akp`` for unusable samples or ``tr``.
    """
    t_trim = check_trim(tr)
    x, y, scale = _paired(a, b)
    g, _ = _trimmed_size(x.size, t_trim, "the AKP effect size of paired differences")
    return _differences(*_snapped_differences(x, y, scale), g).constant


# ------------------------------------------------------------------ yuend


def yuen_dependent_t(
    a: SampleLike, b: SampleLike, *, tr: float = 0.2
) -> YuenDependentT:
    """Yuen's test for the trimmed means of two dependent samples (Wilcox 2017,
    Introduction to Robust Estimation and Hypothesis Testing, 4th ed., section
    5.9.5; R: ``WRS2::yuend``).

    With n pairs, g = floor(tr n) and h = n - 2 g: each sample is winsorized at its
    own order statistics, keeping the pairing (W_a, W_b), and
    t = (trimmed mean(a) - trimmed mean(b)) / sqrt((n - 1) s^2(W_a - W_b) /
    (h (h - 1))), which is Wilcox's d_a + d_b - 2 d_ab with the winsorized
    variances and covariance; df = h - 1. ``t`` keeps the sign of a - b (WRS2's
    ``yuend`` and ggstatsplot report it signed, unlike ``yuen``). The 84 yuend rows
    of probe-wrs2within agree to 1.2e-14 relative.

    The p-value is the two-sided upper tail 2 P(T_df > |t|), computed directly;
    WRS2 computes 2 (1 - pt), rounded to multiples of 2^-52 (all 84 probe rows equal
    it), and inferplot reports the accurate tail (DECISIONS.md D43; ``wrs2_p``).

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, the
            samples differ in length, trimming leaves fewer than two pairs, the
            winsorized samples differ by the same amount in every pair (no
            standard error) or by too little to resolve, or ``tr`` is outside
            [0, 0.5).
    """
    t_trim = check_trim(tr)
    x, y, scale = _paired(a, b)
    n = x.size
    method = "Yuen's test for dependent samples"
    g, h = _trimmed_size(n, t_trim, method)
    wx, wy = _winsorize_column(x, g), _winsorize_column(y, g)
    if differences_same_within_rounding(wx / scale, wy / scale):
        raise ValueError(
            "after winsorizing each sample, every pair differs by the same amount "
            "(exactly, or to within the rounding of the values in double precision), "
            f"so {method} has no standard error"
        )
    s, e = _two_sum(wx, -wy)
    variance = _variance_about(s, e, exact_mean(wx) - exact_mean(wy))
    squared_se = (n - 1) * variance / (h * (h - 1))
    if not squared_se >= TINY:
        raise _no_spread("the winsorized pairs", method)
    se = math.sqrt(squared_se)
    difference = float(
        exact_trimmed_mean(_sorted(x), t_trim) - exact_trimmed_mean(_sorted(y), t_trim)
    )
    t = difference / se
    df = float(h - 1)
    return YuenDependentT(
        t=t,
        df=df,
        p_value=two_sided_t_p(abs(t), df),
        difference=difference / scale,
        se=se / scale,
        n=n,
    )


# ------------------------------------------------------------------ rmanova


def trimmed_rm_anova(
    conditions: Sequence[SampleLike], *, tr: float = 0.2
) -> TrimmedRmAnova:
    """The repeated-measures ANOVA on trimmed means of J dependent conditions
    (Wilcox 2017, section 8.1; R: ``WRS2::rmanova``).

    With n subjects, g = floor(tr n), h = n - 2 g, the trimmed means Xt_j and the
    winsorized table W (each condition winsorized at its own order statistics,
    subjects kept in rows): Q_c = h sum_j (Xt_j - mean(Xt))^2 and Q_e the sum of
    squares of W doubly centred (W_ij - W_.j - W_i. + W_..);
    F = (Q_c / (J - 1)) / (Q_e / ((h - 1)(J - 1))). Both df are multiplied by the
    Huynh & Feldt (1976, Journal of Educational Statistics 1: 69-82) estimate
    epsilon~ = (n (J - 1) epsilon^ - 2) / ((J - 1)(n - 1 - (J - 1) epsilon^)),
    capped at 1, where epsilon^ is the Greenhouse-Geisser (Box 1954) estimate from
    the covariance matrix of W: df1 = (J - 1) epsilon~, df2 = (J - 1)(h - 1)
    epsilon~ (read off probe-wrs2within's 84 rows: F to 1.5e-15, the df exactly
    (J - 1) and (J - 1)(h - 1) where epsilon~ exceeds 1, and the fixtures' fractional
    df where it does not). A denominator n - 1 - (J - 1) epsilon^ at or below 0,
    possible only with n <= J, means an estimate beyond 1 and is capped at 1 like
    any other. With two subjects epsilon^ is exactly 1 / (J - 1) and epsilon~ is
    0 / 0 for any data: refused (what WRS2 returns there is not known).

    The p-value is the upper tail P(F_{df1, df2} > F), computed directly; WRS2
    computes 1 - pf (all 84 probe rows equal it); DECISIONS.md D43, ``wrs2_p``.

    The deviations of the exact trimmed means from their mean are formed exactly
    and rounded once, and the residuals of W are formed as in ``rm_anova``
    (subject effects removed exactly by TwoSum), so F is accurate also when the
    conditions differ by about the rounding of their level.

    Raises:
        ValueError: fewer than two conditions, conditions of unequal length or
            with fewer than two values or a non-finite value, trimming leaving
            fewer than two subjects, a winsorized table with no error variance
            (each subject's winsorized values the condition means plus a constant)
            or one too small to resolve, fewer subjects than the correction needs
            (epsilon~ not positive), an F above the largest double, or ``tr``
            outside [0, 0.5).
    """
    t_trim = check_trim(tr)
    method = "the trimmed-means repeated-measures ANOVA"
    if len(conditions) < 2:
        raise ValueError(
            f"{method} needs at least two conditions; got {len(conditions)}"
        )
    y, scale = _matrix(conditions, method)
    n, k = y.shape
    g, h = _trimmed_size(n, t_trim, method)
    tmeans = [exact_trimmed_mean(_sorted(y[:, j]), t_trim) for j in range(k)]
    deviations = deviations_from_weighted_mean([1.0] * k, tmeans)
    q_c = h * math.fsum(d * d for d in deviations)
    w = np.empty_like(y)
    for j in range(k):
        w[:, j] = _winsorize_column(y[:, j], g)
    residuals = _double_centred(w)
    q_e = math.fsum((residuals * residuals).ravel().tolist())
    df_e = (h - 1) * (k - 1)
    first = np.ascontiguousarray(w[:, 0]) / scale
    if not bool((residuals != 0.0).any()) or all(
        differences_same_within_rounding(np.ascontiguousarray(w[:, j]) / scale, first)
        for j in range(1, k)
    ):
        raise ValueError(
            "after winsorizing, every subject's values are the condition means plus a "
            "constant of its own (exactly, or to within the rounding of the values in "
            f"double precision), so {method} has no error variance"
        )
    if not q_e / df_e >= TINY:
        raise _no_spread("the winsorized residuals", method)
    f = (q_c / (k - 1)) / (q_e / df_e)
    if math.isinf(f):
        raise ValueError(
            "the trimmed-means F exceeds the largest double (about 1.8e308), so it "
            "cannot be reported. Check the data for entry or unit errors."
        )
    # Exact sums: the result does not depend on the order of the subjects.
    cross = [
        [math.fsum((residuals[:, i] * residuals[:, j]).tolist()) for j in range(k)]
        for i in range(k)
    ]
    trace = math.fsum(cross[i][i] for i in range(k))
    frobenius = math.fsum(v * v for row in cross for v in row)
    epsilon_gg = min(1.0, max(1.0 / (k - 1), trace * trace / ((k - 1) * frobenius)))
    if n < _MIN_SUBJECTS_HUYNH_FELDT:
        # epsilon_gg is then exactly 1 / (k - 1), and the Huynh-Feldt estimate
        # (n (k - 1) eps - 2) / ((k - 1) (n - 1 - (k - 1) eps)) is 0 / 0.
        raise ValueError(
            f"{method} corrects its degrees of freedom by the Huynh-Feldt epsilon, "
            f"which is undefined (0 / 0) with {n} subjects; it needs at least "
            f"{_MIN_SUBJECTS_HUYNH_FELDT}"
        )
    denominator = (k - 1) * (n - 1 - (k - 1) * epsilon_gg)
    numerator = n * (k - 1) * epsilon_gg - 2.0
    # A non-positive denominator (few subjects next to the conditions) means an
    # estimate beyond 1, which is capped at 1 like any other.
    epsilon = 1.0 if denominator <= 0.0 else min(1.0, numerator / denominator)
    if not epsilon > 0.0:
        raise ValueError(
            f"{method} corrects its degrees of freedom by the Huynh-Feldt epsilon, "
            f"which is not positive with {n} subjects and {k} conditions; it needs "
            "more subjects"
        )
    df1 = (k - 1) * epsilon
    df2 = df_e * epsilon
    return TrimmedRmAnova(
        f=f,
        df1=df1,
        df2=df2,
        p_value=f_upper_p(f, df1, df2),
        epsilon=epsilon,
        epsilon_gg=epsilon_gg,
        n=n,
        k=k,
    )


def _double_centred(w: FloatArray) -> FloatArray:
    """w_ij - w_.j - w_i. + w_.., accurate to a few units in the last place of
    each residual (``within_parametric``'s ``_decompose``): the subject effects
    are removed exactly through the differences to the first condition (TwoSum),
    then the columns of the rounded part are centred and the rounding errors added
    back, and the rows and then the columns of that are centred."""
    hi, lo = _two_sum(w, -w[:, :1])
    centred = hi - _column_means(hi) + lo
    centred = centred - (centred.sum(axis=1) / w.shape[1])[:, np.newaxis]
    return centred - _column_means(centred)


def _column_means(x: FloatArray) -> FloatArray:
    """Each column's mean from its exact sum (``math.fsum``), so it does not
    depend on the order of the rows."""
    n = x.shape[0]
    return np.array([math.fsum(x[:, j].tolist()) / n for j in range(x.shape[1])])


# ------------------------------------------------------------------ rmmcp


def rmmcp(
    conditions: Sequence[SampleLike],
    *,
    tr: float = 0.2,
    conf_level: float = 0.95,
    p_adjust_method: PAdjustMethod = "holm",
) -> tuple[RmmcpPair, ...]:
    """Pairwise comparisons of dependent conditions on their differences, WRS2's
    ``rmmcp`` with ``dif = TRUE`` as ggstatsplot uses it (Wilcox 2017, section 8.2).

    For every pair i < j in input order, with d = condition i - condition j (exact
    differences), g = floor(tr n) and h = n - 2 g: estimate = the trimmed mean of
    d, se = s_w(d) / ((1 - 2 tr) sqrt(n)) with s_w(d) the winsorized standard
    deviation of d, t = estimate / se on h - 1 df, the raw p = 2 P(T > |t|)
    computed directly (WRS2: 2 (1 - pt); DECISIONS.md D43, ``wrs2_p``), and the CI
    estimate -/+ c se with c the upper alpha / (2C) quantile of t on h - 1 df
    (Bonferroni over the C = J (J - 1) / 2 pairs), alpha = 1 - conf_level. The raw
    p-values are then adjusted together with ``p_adjust_method``; rmmcp's own
    critical p-values are not used, as ggstatsplot does not use them.
    probe-wrs2within's 252 rmmcp rows agree: estimates and bounds to 3.1e-16 of
    the largest |value|, p to 3.6e-16 absolute.

    Raises:
        ValueError: fewer than two conditions, conditions of unequal length or
            with fewer than two values or a non-finite value, trimming leaving
            fewer than two subjects, a pair whose winsorized differences are all
            equal (or too close to resolve), an unknown ``p_adjust_method``, or an
            invalid ``tr`` or ``conf_level``.
    """
    t_trim = check_trim(tr)
    level = check_conf_level(conf_level)
    p_adjust([], p_adjust_method)  # an unknown method fails before any work
    method = "rmmcp's comparison"
    if len(conditions) < 2:
        raise ValueError(
            f"{method} needs at least two conditions; got {len(conditions)}"
        )
    y, scale = _matrix(conditions, method)
    n, k = y.shape
    g, h = _trimmed_size(n, t_trim, method)
    n_pairs = k * (k - 1) // 2
    df = float(h - 1)
    # P(|T| > c) = alpha / C, so P(T > c) = alpha / (2C).
    critical = smm_quantile(1, df, conf_level=1.0 - (1.0 - level) / n_pairs)
    rows: list[tuple[int, int, float, float, float, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            diffs = _differences(*_snapped_differences(y[:, i], y[:, j], scale), g)
            if diffs.constant:
                raise ValueError(
                    f"conditions {i + 1} and {j + 1}: after winsorizing, every "
                    f"subject's difference is the same, so {method} has no "
                    "standard error"
                )
            squared_se = diffs.wvar / ((1.0 - 2.0 * t_trim) ** 2 * n)
            if not squared_se >= TINY:
                raise _no_spread(
                    f"conditions {i + 1} and {j + 1}: the differences", method
                )
            se = math.sqrt(squared_se)
            estimate = float(diffs.tmean)
            t = estimate / se
            rows.append((i, j, estimate, se, t, two_sided_t_p(abs(t), df)))
    adjusted = p_adjust([row[5] for row in rows], p_adjust_method)
    return tuple(
        RmmcpPair(
            i=i,
            j=j,
            estimate=estimate / scale,
            se=se / scale,
            df=df,
            t=t,
            critical_value=critical,
            conf_low=(estimate - critical * se) / scale,
            conf_high=(estimate + critical * se) / scale,
            p_value=p,
            p_adjusted=adj,
        )
        for (i, j, estimate, se, t, p), adj in zip(rows, adjusted, strict=True)
    )


# ------------------------------------------------------------------ AKP


def _chunks(total_rows: int, values_per_row: int) -> list[int]:
    size = max(1, min(total_rows, _CHUNK_VALUES // max(1, values_per_row)))
    full, rest = divmod(total_rows, size)
    return [size] * full + ([rest] if rest else [])


def dep_effect_percentile_bounds(
    ordered: NDArray[np.float64], conf_level: float
) -> tuple[float, float]:
    """The percentile interval of WRS2's ``dep.effect`` and ``wmcpAKP``: the
    rounded rule (``rounded_percentile_bounds``), l = round(alpha / 2 B) and the
    (l + 1)-th and (B - l)-th smallest of the B values, with alpha = 1 -
    ``conf_level`` as written in decimal (exactly 0.05 at 0.95: those functions
    take a fixed alpha of 0.05, not 1 - 0.95 computed in floating point, which is
    0.050000000000000044). probe-bootrules, R's own sorted replicates at B = 39,
    40, 99, 100, 199 and 200: 6 of 6 for dep.effect and 6 of 6 for wmcpAKP's pair
    bounds (B = 100: the 3rd and 98th, where t1way's alpha gives the 4th and 97th
    and akp.effect's floor rule the 3rd and 97th). At 2,000 defined resamples it
    picks the same order statistics as the floor rule; with undefined (0 / 0)
    resamples left out (D65) B is the number of the others, and the two rules can
    differ by one order statistic.
    """
    alpha = float(Decimal(1) - Decimal(repr(float(conf_level))))
    return rounded_percentile_bounds(ordered, conf_level, alpha)


@dataclass(frozen=True, slots=True)
class _PairAkp:
    """One pair's AKP estimate and the centred differences its bootstrap draws:
    AKP* = c (centre + trimmed mean(deviations*)) / s_w(deviations*)."""

    estimate: float
    centre: float
    deviations: FloatArray  # in the canonical subject order
    signs: FloatArray  # of each subject's exact difference: -1.0, 0.0 or 1.0


def _pair_akp(
    x: FloatArray,
    y: FloatArray,
    scale: float,
    g: int,
    constant: float,
    names: str,
) -> _PairAkp:
    s, e = _snapped_differences(x, y, scale)
    diffs = _differences(s, e, g)
    method = "the AKP effect size of the paired differences"
    if diffs.constant:
        raise ValueError(
            f"{names}: after winsorizing, every subject's difference is the same "
            f"(a winsorized standard deviation of 0), so {method} is undefined"
        )
    if not diffs.wvar >= TINY:
        raise _no_spread(f"{names}: the differences", method)
    centre = float(diffs.tmean)
    # s is the exact difference rounded, so it has the difference's sign.
    return _PairAkp(
        estimate=constant * centre / math.sqrt(diffs.wvar),
        centre=centre,
        deviations=(s - centre) + e,
        signs=np.sign(s),
    )


def _canonical_rows(y: FloatArray) -> FloatArray:
    """The subjects (rows) in lexicographic order of their values, so that the
    bootstrap draws the same subjects whatever the row order of the data."""
    order: NDArray[np.intp] = np.arange(y.shape[0], dtype=np.intp)
    for j in range(y.shape[1] - 1, -1, -1):  # last key first, stable sorts
        column: FloatArray = y[order, j]
        order = order[column.argsort(kind="stable")]
    return y[order]


def _akp_replicates(
    rng: np.random.Generator, pairs: Sequence[_PairAkp], n: int, tr: float, c: float
) -> list[NDArray[np.float64]]:
    """BOOTSTRAP_RESAMPLES resamples of the subjects, each n draws with
    replacement; every pair's AKP is computed on the same resampled subjects."""
    out = [np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64) for _ in pairs]
    start = 0
    for rows in _chunks(BOOTSTRAP_RESAMPLES, n):
        index = rng.integers(0, n, size=(rows, n))
        for values, pair in zip(out, pairs, strict=True):
            tmeans, wvars = rows_trimmed(pair.deviations[index], tr)
            with np.errstate(divide="ignore", invalid="ignore"):
                chunk: NDArray[np.float64] = c * (pair.centre + tmeans) / np.sqrt(wvars)
            # A winsorized variance of 0: every winsorized difference of the
            # resample is one subject's difference d, and AKP* is +/-inf with the
            # sign of d, or 0 / 0 when d is 0. Decided from the exact difference
            # of a subject whose deviation is the row's trimmed mean (the
            # deviations are centred on a rounded mean, so centre + deviation
            # does not say whether d is 0).
            for row in (wvars == 0.0).nonzero()[0].tolist():
                subject = int((pair.deviations == tmeans[row]).nonzero()[0][0])
                sign = float(pair.signs[subject])
                chunk[row] = math.nan if sign == 0.0 else sign * math.inf
            values[start : start + rows] = chunk
        start += rows
    return out


def _akp_bootstrap(
    conditions: Sequence[SampleLike],
    pairs: Sequence[tuple[int, int]],
    *,
    tr: float,
    conf_level: float,
    seed: SeedLike,
    method: str,
) -> tuple[
    list[float], list[tuple[float, float]], tuple[int, ...], tuple[int, ...], int
]:
    """Each pair's AKP estimate, percentile bounds, and counts of undefined and
    infinite resamples, from one bootstrap of the subjects."""
    t_trim = check_trim(tr)
    level = check_conf_level(conf_level)
    rng_seed = check_bootstrap_seed(seed)
    y, scale = _matrix(conditions, method)
    y = _canonical_rows(y)
    n = y.shape[0]
    g, _ = _trimmed_size(n, t_trim, method)
    constant = math.sqrt(winsorized_normal_variance(t_trim))
    fitted = [
        _pair_akp(
            y[:, i],
            y[:, j],
            scale,
            g,
            constant,
            f"conditions {i + 1} and {j + 1}",
        )
        for i, j in pairs
    ]
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    bounds: list[tuple[float, float]] = []
    undefined: list[int] = []
    infinite: list[int] = []
    for values in _akp_replicates(rng, fitted, n, t_trim, constant):
        replicates = split_replicates(values)
        bounds.append(
            replicate_bounds(replicates, level, dep_effect_percentile_bounds, method)
        )
        undefined.append(replicates.undefined)
        infinite.append(replicates.infinite)
    estimates = [p.estimate for p in fitted]
    return estimates, bounds, tuple(undefined), tuple(infinite), rng_seed


def dependent_akp(
    a: SampleLike,
    b: SampleLike,
    *,
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: SeedLike = 0,
) -> BootstrapEffectSizeCI:
    """The AKP robust standardized difference of paired samples with a percentile
    bootstrap CI, the AKP row of R's ``WRS2::dep.effect`` as statsExpressions 2.2.1
    calls it for ggwithinstats.

    delta = s_N trimmed mean(d) / s_w(d) with d = a - b, s_w the winsorized
    standard deviation and s_N = sqrt(``winsorized_normal_variance(tr)``), which
    makes delta d_z under normality (Algina, Keselman & Penfield 2005,
    Psychological Methods 10: 317-328, applied to the differences). probe-
    wrs2within's 8 dep.effect estimates (tr 0.1 and 0.2) are this value times
    WRS2's own s_N, the one ``akp.effect`` uses (1.2e-9 relative above the exact
    value at tr 0.2, 4.6e-9 at 0.1; DECISIONS.md D42, ``akp_constant``).

    CI: BOOTSTRAP_RESAMPLES resamples of the pairs (subjects drawn with
    replacement, their two values together), the (l + 1)-th and (B - l)-th
    smallest delta, l = round(alpha / 2 B) with alpha = 1 - conf_level as written
    (``dep_effect_percentile_bounds``; probe-bootrules: R's own replicates, 6 of
    6; at B = 2,000 and 95% it coincides with ``akp_effect``'s floor rule). The
    scheme was read off the fixtures' 20,000-resample references: over 40 seeds
    resampling pairs puts every bound's mean within 1.9 standard errors of the
    mean of R's 50 replicate runs on the 11 distinct two-condition references;
    resampling the two samples apart misses each reference by 8 to 350 standard
    errors on at least one bound (DECISIONS.md log). A resample whose winsorized
    differences are constant gives +/-inf (kept, counted in
    ``infinite_resamples``; edge_within2_small's upper bound is Inf in R too) or
    0 / 0 (left out, counted in ``undefined_resamples``; ``akp_effect``'s rule).
    Subjects are drawn in an order fixed by their values, so the row order of the
    data does not matter.

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, the
            samples differ in length, trimming leaves fewer than two pairs, the
            winsorized differences are all equal (or too close to resolve), too
            many 0 / 0 resamples for the percentile interval, or an invalid
            ``tr``, ``conf_level`` or ``seed``.
    """
    method = "the AKP effect size of the paired differences"
    estimates, bounds, undefined, infinite, rng_seed = _akp_bootstrap(
        [a, b], [(0, 1)], tr=tr, conf_level=conf_level, seed=seed, method=method
    )
    (low, high) = bounds[0]
    return BootstrapEffectSizeCI(
        estimate=estimates[0],
        conf_low=low,
        conf_high=high,
        conf_level=check_conf_level(conf_level),
        sidedness="two-sided",
        seed=rng_seed,
        resamples=BOOTSTRAP_RESAMPLES,
        undefined_resamples=undefined[0],
        infinite_resamples=infinite[0],
    )


def average_dependent_akp(
    conditions: Sequence[SampleLike],
    *,
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: SeedLike = 0,
) -> BootstrapEffectSizeCI:
    """The average over all pairs of conditions of the AKP effect size of their
    differences, with the average of their percentile bootstrap bounds, R's
    ``WRS2::wmcpAKP`` as statsExpressions 2.2.1 calls it for ggwithinstats.

    For every pair i < j (input order), ``dependent_akp`` of condition i minus
    condition j; the estimate is the mean of the C = J (J - 1) / 2 estimates
    (probe-wrs2within: all 8 wmcpAKP estimates to 3.3e-16 relative with WRS2's
    s_N), and each bound the mean of the pairs' bounds. The bootstrap resamples
    the subjects once (B = 2,000) and computes every pair's AKP on the same
    resamples. Read off the fixtures' references (100 seeds): averaging per-pair
    intervals puts the bounds' means within 1.8 standard errors of the means of
    R's 50 replicate runs on the 12 distinct references with three or more
    conditions, while the percentile interval of the resampled average misses each
    by 73 to 1,000 on at least one bound; drawing the subjects once for all pairs
    matches the spread of R's replicate runs (mean log SD ratio +0.02, 0.9
    standard errors) better than a fresh draw per pair (-0.06, -2.4) (DECISIONS.md
    log). A pair's infinite bound makes the average infinite, as in R
    (edge_within_small: -Inf at 100 resamples). ``pair_undefined`` and
    ``pair_infinite`` count each pair's undefined and infinite values (of 2,000);
    ``undefined_resamples`` and ``infinite_resamples`` are the largest of them.

    Raises:
        ValueError: fewer than two conditions, conditions of unequal length or
            with fewer than two values or a non-finite value, trimming leaving
            fewer than two subjects, a pair whose winsorized differences are all
            equal (or too close to resolve), too many 0 / 0 resamples in a pair,
            bounds of opposite infinite sign in one average, or an invalid
            ``tr``, ``conf_level`` or ``seed``.
    """
    method = "the average AKP effect size"
    if len(conditions) < 2:
        raise ValueError(
            f"{method} needs at least two conditions; got {len(conditions)}"
        )
    k = len(conditions)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    estimates, bounds, undefined, infinite, rng_seed = _akp_bootstrap(
        conditions, pairs, tr=tr, conf_level=conf_level, seed=seed, method=method
    )
    low = _average([b[0] for b in bounds])
    high = _average([b[1] for b in bounds])
    if math.isnan(low) or math.isnan(high):
        raise ValueError(
            f"{method}'s bootstrap bounds are +Inf for one pair and -Inf for another, "
            "so their average is undefined"
        )
    return BootstrapEffectSizeCI(
        estimate=_average(estimates),
        conf_low=low,
        conf_high=high,
        conf_level=check_conf_level(conf_level),
        sidedness="two-sided",
        seed=rng_seed,
        resamples=BOOTSTRAP_RESAMPLES,
        undefined_resamples=max(undefined),
        infinite_resamples=max(infinite),
        pair_undefined=undefined,
        pair_infinite=infinite,
    )


def _average(values: Sequence[float]) -> float:
    if any(math.isinf(v) for v in values):
        return float(sum(values))  # +inf, -inf or nan
    return math.fsum(values) / len(values)
