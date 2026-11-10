"""Robust methods on trimmed means: Yuen's test, the trimmed-means ANOVA, lincon's
pairwise comparisons, and the AKP and explanatory effect sizes with bootstrap CIs.

The R references are WRS2's ``yuen``, ``t1way``, ``lincon`` and ``akp.effect`` as
statsExpressions 2.2.1 calls them for ggstatsplot 1.1.4 (SPEC.md section 4). The
methods are implemented from Yuen (1974), Wilcox (2017, Introduction to Robust
Estimation and Hypothesis Testing, 4th ed.), Algina, Keselman & Penfield (2005) and
Wilcox & Tian (2011); the details WRS2 adds (which constant, which resampling
order, which order statistics bound a percentile interval) were read off the
oracle fixtures, whose 100-resample values each bootstrap reproduces exactly when
R's random-number stream is replayed (tests/stats/test_robust_parity.py, DECISIONS.md
log).

Deterministic statistics are computed on the groups times their ``common_scale``
(a power of two, exact), so they do not depend on the units of y, bit for bit for
any power-of-two rescaling that keeps the data normal doubles. Trimmed means are
kept exact (``exact_mean``), and every difference of them, and every deviation from
t1way's weighted grand trimmed mean, is formed exactly and rounded once, so it
keeps its relative accuracy when the trimmed means differ by about the rounding of
their common level (DECISIONS.md, exact differences); WRS2 subtracts rounded
means. Bootstrap resamples are drawn from the data shifted by one of its values
(``_centred``), so resampled trimmed means are rounded relative to the spread.
Bootstraps resample each group's sorted values (so the row order of the data does
not matter) with ``numpy.random.Generator(PCG64(seed))`` and
:data:`BOOTSTRAP_RESAMPLES` resamples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import special

from inferplot.stats._smm_table import SMM_ROWS, SMM_TABLE_05
from inferplot.stats.descriptive import (
    FloatArray,
    SampleLike,
    as_sample,
    common_scale,
    deviations_from_weighted_mean,
    exact_mean,
    exact_variance,
    mean,
    scaled_variance,
    variance,
)
from inferplot.stats.noncentral import check_conf_level
from inferplot.stats.nonparametric import (
    BOOTSTRAP_RESAMPLES,
    BootstrapEffectSizeCI,
    SeedLike,
    check_bootstrap_seed,
)
from inferplot.stats.padjust import PAdjustMethod, p_adjust
from inferplot.stats.parametric import TINY, satterthwaite_df
from inferplot.stats.smm import smm_quantile

__all__ = [
    "SMM_MAX_TABLED",
    "SMM_ROWS",
    "LinconPair",
    "TrimmedMeansAnova",
    "YuenT",
    "akp_effect",
    "explanatory_effect",
    "lincon",
    "smm_critical_value",
    "trimmed_mean",
    "trimmed_means_anova",
    "winsorized_normal_variance",
    "winsorized_variance",
    "yuen_t",
]

# Upper bound on the values one bootstrap chunk draws (2^22 doubles, 32 MiB).
_CHUNK_VALUES: Final = 1 << 22

_SMM_CONF_LEVEL: Final = 0.95
# WRS2's table stops at 28 comparisons (8 groups); above, lincon simulates.
SMM_MAX_TABLED: Final = 28


@dataclass(frozen=True, slots=True)
class YuenT:
    """Yuen's two-sample trimmed-means test (R: ``WRS2::yuen``).

    ``t`` is |t| as ggstatsplot reports it; ``difference`` keeps the sign of
    trimmed mean(a) - trimmed mean(b), and ``se`` is its standard error.
    """

    t: float
    df: float
    p_value: float
    difference: float
    se: float


@dataclass(frozen=True, slots=True)
class TrimmedMeansAnova:
    """Heteroscedastic one-way ANOVA on trimmed means (R: ``WRS2::t1way``)."""

    f: float
    df1: float
    df2: float
    p_value: float


@dataclass(frozen=True, slots=True)
class LinconPair:
    """One of lincon's pairwise comparisons of groups ``i`` < ``j`` (input order).

    ``estimate`` = trimmed mean(i) - trimmed mean(j) and ``t`` = estimate / se;
    ``conf_low`` and ``conf_high`` are estimate -/+ ``critical_value`` x se.
    """

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


def check_trim(tr: float) -> float:
    value = float(tr)
    if not 0.0 <= value < 0.5:
        raise ValueError(f"tr must lie in [0, 0.5); got {tr!r}")
    return value


def _sorted(x: FloatArray) -> FloatArray:
    """A sorted copy of ``x`` (along its last axis)."""
    out = x.copy()
    out.sort()
    return out


def _joined(parts: Sequence[FloatArray]) -> FloatArray:
    """``parts`` side by side along the last axis, in order."""
    width = sum(part.shape[-1] for part in parts)
    out = np.empty((*parts[0].shape[:-1], width), dtype=np.float64)
    start = 0
    for part in parts:
        out[..., start : start + part.shape[-1]] = part
        start += part.shape[-1]
    return out


def trim_count(n: int, tr: float) -> int:
    """g = floor(tr n), the values trimmed (or winsorized) at each end, computed
    in floating point as R's ``floor(tr * n)``."""
    return math.floor(tr * n)


def winsorized_normal_variance(tr: float = 0.2) -> float:
    """The variance of a standard normal variable winsorized at tr in each tail.

    With c = Phi^-1(1 - tr): (1 - 2 tr) - 2 c phi(c) + 2 tr c^2, the integral of
    z^2 phi(z) over [-c, c] plus the two tails' mass tr at +/-c (Wilcox 2017,
    section 2.3; Algina, Keselman & Penfield 2005). 0.41208672670619 at tr = 0.2,
    whose square root, 0.64193981548600, is Algina et al.'s constant 0.642. WRS2
    uses approximations of it (DECISIONS.md D42): 0.4120867 in t1way's
    explanatory effect size and, in akp.effect, a square root 1.2e-9 relative
    above this one. inferplot uses the exact value (DECISIONS.md D42).
    """
    t = check_trim(tr)
    if t == 0.0:
        return 1.0
    c = -float(special.ndtri(t))
    phi = math.exp(-0.5 * c * c) / math.sqrt(2.0 * math.pi)
    return (1.0 - 2.0 * t) - 2.0 * c * phi + 2.0 * t * c * c


def trimmed_mean(x: SampleLike, *, tr: float = 0.2) -> float:
    """The tr-trimmed mean: the mean of the values left after removing the
    g = floor(tr n) smallest and g largest (Wilcox 2017, section 2.2.4), R's
    ``mean(x, trim = tr)``, as ggstatsplot's trimmed-mean labels and WRS2 use it.
    Correctly rounded (``mean``).

    Raises:
        ValueError: ``x`` is empty, not one-dimensional or not finite, or ``tr`` is
            outside [0, 0.5).
    """
    t = check_trim(tr)
    arr = _sorted(as_sample(x))
    g = trim_count(arr.size, t)
    return mean(arr[g : arr.size - g])


def exact_trimmed_mean(sorted_x: FloatArray, tr: float) -> Fraction:
    """The tr-trimmed mean of sorted values as an exact rational (``exact_mean``)."""
    g = trim_count(sorted_x.size, tr)
    return exact_mean(sorted_x[g : sorted_x.size - g])


def winsorize_sorted(sorted_x: FloatArray, g: int) -> FloatArray:
    out = sorted_x.copy()
    n = out.size
    if g:
        out[:g] = sorted_x[g]
        out[n - g :] = sorted_x[n - g - 1]
    return out


def winsorized_variance(x: SampleLike, *, tr: float = 0.2) -> float:
    """The tr-winsorized sample variance (Wilcox 2017, section 3.3.2): the g =
    floor(tr n) smallest values set to the (g + 1)-th smallest, the g largest to
    the (g + 1)-th largest, then the sample variance with denominator n - 1
    (``variance``). WRS2's ``winvar``.

    Raises:
        ValueError: fewer than two values, not one-dimensional or not finite, or
            ``tr`` outside [0, 0.5).
    """
    t = check_trim(tr)
    arr = _sorted(as_sample(x, min_size=2))
    return variance(winsorize_sorted(arr, trim_count(arr.size, t)))


@dataclass(frozen=True, slots=True)
class _Kernel:
    """One group's trimmed-mean summaries in the common scale."""

    n: int
    h: int
    tmean: Fraction  # exact (``exact_mean``): differences are rounded once
    wvar: float
    d: float  # squared standard error of the trimmed mean (Yuen 1974)


def _kernel(scaled: FloatArray, tr: float, name: str, method: str) -> _Kernel:
    """Trimmed mean, winsorized variance and Yuen's d = (n - 1) s_w^2 / (h (h - 1))
    of one scaled group, h = n - 2 g the values left after trimming."""
    ordered = _sorted(scaled)
    n = ordered.size
    g = trim_count(n, tr)
    h = n - 2 * g
    if h < 2:
        raise ValueError(
            f"{name}: trimming {g} of {n} values at each end leaves {h}; {method} "
            "needs at least 2 values after trimming (use more values or a smaller tr)"
        )
    wvar = scaled_variance(winsorize_sorted(ordered, g))
    return _Kernel(
        n=n,
        h=h,
        tmean=exact_trimmed_mean(ordered, tr),
        wvar=wvar,
        d=(n - 1) * wvar / (h * (h - 1)),
    )


def _zero_spread_message(name: str, raw: FloatArray, tr: float, method: str) -> str:
    ordered = _sorted(raw)
    g = trim_count(ordered.size, tr)
    if float(ordered[ordered.size - g - 1] - ordered[g]) == 0.0:
        return (
            f"{name} has a winsorized variance of 0 (its values from the "
            f"{g + 1}-th smallest to the {g + 1}-th largest are all equal), so "
            f"{method} is undefined"
        )
    return (
        f"{name} varies, but by less than double precision can represent next to "
        "the largest |value| of all groups (a winsorized standard deviation below "
        f"roughly 1e-154 times that value), so {method} has no accurate standard "
        "error. Check the data for entry or unit errors."
    )


def two_sided_t_p(t_abs: float, df: float) -> float:
    """2 P(T_df > |t|), the upper tail computed directly (SciPy's ``stdtr`` at
    -|t|). WRS2's yuen and lincon compute 2 (1 - pt(|t|, df)) instead, which
    rounds the one-sided tail to the spacing of doubles just below 1 (2^-53) and
    gives 0 below about 1.1e-16, where ggstatsplot would print 2.23e-308.
    inferplot reports the accurate tail (DECISIONS.md D43); against the oracle the
    two agree within that rounding (tolerance ``wrs2_p``)."""
    return min(1.0, 2.0 * float(special.stdtr(df, -t_abs)))


def f_upper_p(f: float, df1: float, df2: float) -> float:
    """P(F_{df1, df2} > F), the upper tail computed directly (SciPy's ``fdtrc``).
    WRS2's t1way computes 1 - pf(F, df1, df2) instead; inferplot reports the
    accurate tail (DECISIONS.md D43; see ``two_sided_t_p``)."""
    return float(special.fdtrc(df1, df2, f))


# ------------------------------------------------------------------ Yuen


def yuen_t(a: SampleLike, b: SampleLike, *, tr: float = 0.2) -> YuenT:
    """Yuen's (1974, Biometrika 61: 165-170) trimmed-means test of a against b.

    With g = floor(tr n) values trimmed at each end, h = n - 2g and s_w^2 the
    winsorized variance, each group has d = (n - 1) s_w^2 / (h (h - 1));
    t = (trimmed mean(a) - trimmed mean(b)) / sqrt(d_a + d_b) and the Welch-type
    df = (d_a + d_b)^2 / (d_a^2 / (h_a - 1) + d_b^2 / (h_b - 1)) (Wilcox 2017,
    section 5.3; computed from the shares d / (d_a + d_b)). R reference:
    ``WRS2::yuen``, as statsExpressions 2.2.1 calls it: ggstatsplot reports |t|,
    so ``t`` here is unsigned and ``difference`` carries the sign.

    The p-value is the two-sided upper tail 2 P(T_df > |t|), computed directly.
    R's value is 2 (1 - pt(|t|, df)), rounded to multiples of 2^-52 (the oracle's
    p-values all lie on that grid) and 0 below about 2.2e-16; inferplot reports
    the accurate tail (DECISIONS.md D43), within that rounding of R's
    (``wrs2_p``).

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value,
            trimming leaves fewer than two values, both winsorized variances are 0
            (or too small to resolve next to the largest |value|), or ``tr`` is
            outside [0, 0.5).
    """
    t_trim = check_trim(tr)
    raw_a = as_sample(a, name="a", min_size=2)
    raw_b = as_sample(b, name="b", min_size=2)
    scale = common_scale(raw_a, raw_b)
    ka = _kernel(raw_a * scale, t_trim, "sample a", "Yuen's test")
    kb = _kernel(raw_b * scale, t_trim, "sample b", "Yuen's test")
    total = ka.d + kb.d
    if not total >= TINY:
        if ka.wvar == 0.0 and kb.wvar == 0.0:
            raise ValueError(
                "both samples have a winsorized variance of 0, so Yuen's test is "
                "undefined"
            )
        which, values = ("sample a", raw_a) if ka.d < kb.d else ("sample b", raw_b)
        raise ValueError(_zero_spread_message(which, values, t_trim, "Yuen's test"))
    se = math.sqrt(total)
    difference = float(ka.tmean - kb.tmean)  # exact, rounded once
    t = difference / se
    df = satterthwaite_df(ka.d, ka.h, kb.d, kb.h)
    return YuenT(
        t=abs(t),
        df=df,
        p_value=two_sided_t_p(abs(t), df),
        difference=difference / scale,
        se=se / scale,
    )


# ------------------------------------------------------------------ t1way


def _group_kernels(
    groups: Sequence[SampleLike], tr: float, method: str, *, min_groups: int
) -> tuple[list[FloatArray], float, list[_Kernel]]:
    if len(groups) < min_groups:
        raise ValueError(
            f"{method} needs at least {min_groups} groups; got {len(groups)}"
        )
    raw = [
        as_sample(g, name=f"group {i + 1}", min_size=2) for i, g in enumerate(groups)
    ]
    scale = common_scale(*raw)
    kernels = [
        _kernel(r * scale, tr, f"group {i + 1}", method) for i, r in enumerate(raw)
    ]
    return raw, scale, kernels


def trimmed_means_anova(
    groups: Sequence[SampleLike], *, tr: float = 0.2
) -> TrimmedMeansAnova:
    """Wilcox's heteroscedastic one-way ANOVA on trimmed means, a Welch (1951)
    type test (Wilcox 2017, section 7.4.1; R: ``WRS2::t1way``).

    With each group's Yuen d_j (``yuen_t``), weights w_j = 1 / d_j, U = sum w_j,
    the weighted grand trimmed mean X = sum w_j Xt_j / U, A = sum w_j (Xt_j - X)^2
    / (J - 1) and lambda = sum (1 - w_j / U)^2 / (h_j - 1):
    F = A / (1 + 2 (J - 2) lambda / (J^2 - 1)), df1 = J - 1,
    df2 = (J^2 - 1) / (3 lambda). The p-value is the upper tail P(F > F_obs),
    computed directly; R's 1 - pf(F, df1, df2) is rounded to 2^-53 and 0 below
    about 1.1e-16 (DECISIONS.md D43; ``wrs2_p``, see ``yuen_t``).

    Limits as ``welch_anova``: every scaled d_j must be at least ``TINY``; the
    weights are then brought below 2^(1015 - bit_length(J)) by one power of two so
    no sum overflows, and F is scaled back.

    Raises:
        ValueError: fewer than two groups, a group with fewer than two values, a
            non-finite value, trimming leaving fewer than two values, a group with
            a winsorized variance of 0 (or too small to resolve next to the
            largest |value|), an F above the largest double, or ``tr`` outside
            [0, 0.5).
    """
    t_trim = check_trim(tr)
    method = "the trimmed-means ANOVA"
    raw, _, kernels = _group_kernels(groups, t_trim, method, min_groups=2)
    for index, kernel in enumerate(kernels):
        if not kernel.d >= TINY:
            raise ValueError(
                _zero_spread_message(f"group {index + 1}", raw[index], t_trim, method)
            )
    k = len(kernels)
    weights = [1.0 / kern.d for kern in kernels]
    top = 1015 - k.bit_length()
    shift = max(0, math.frexp(max(weights))[1] - top)
    if shift:
        weights = [math.ldexp(w, -shift) for w in weights]
    total = math.fsum(weights)
    # Deviations from the weighted grand trimmed mean, formed exactly from the
    # exact trimmed means and rounded once.
    deviations = deviations_from_weighted_mean(
        weights, [kern.tmean for kern in kernels]
    )
    a = math.fsum(w * d * d for w, d in zip(weights, deviations, strict=True)) / (k - 1)
    lam = math.fsum(
        (1.0 - w / total) ** 2 / (kern.h - 1)
        for w, kern in zip(weights, kernels, strict=True)
    )
    f = a / (1.0 + 2.0 * (k - 2) * lam / (k * k - 1))
    if shift:
        f = math.ldexp(f, shift)
    if math.isinf(f):
        raise ValueError(
            "the trimmed-means F exceeds the largest double (about 1.8e308), so it "
            "cannot be reported. Check the data for entry or unit errors."
        )
    df1 = float(k - 1)
    df2 = (k * k - 1) / (3.0 * lam)
    return TrimmedMeansAnova(f=f, df1=df1, df2=df2, p_value=f_upper_p(f, df1, df2))


# ------------------------------------------------------------------ lincon


def smm_critical_value(
    n_comparisons: int, df: float, *, conf_level: float = 0.95
) -> float:
    """lincon's critical value for ``n_comparisons`` pairs at a pair's Yuen ``df``.

    WRS2's ``lincon`` (alpha = 0.05) takes it from three sources (probe-smm):

    * C = 1: Student's t quantile, t_{0.975, df}.
    * C = 2..28: a two-decimal table of the upper 5% point of the Studentized
      maximum modulus distribution (max of C independent |Z| over a common
      sqrt(chi2_df / df); Stoline & Ury 1979, Technometrics 21: 87-93), with rows
      at df 2..12, 14, 16, 18, 20, 24, 30, 40, 60 and a last row, labelled 200, that
      holds the infinite-df values, interpolated linearly in 1/df between the rows
      on either side of ``df`` and read off the last row from df 200 on (SPEC.md
      section 4). The entries are WRS2's published ones (DECISIONS.md D51), 24 of
      which are not the rounded quantile (C = 3 at df 5 is 3.39 where the quantile
      is 3.3992), except C = 9 at df 2, a transcription error in WRS2 (8.83 for the
      quantile 8.38), kept at 8.38 (``inferplot.stats._smm_table``).
    * C above 28: WRS2 estimates the value by a simulation seeded by itself; here
      the exact Studentized maximum modulus quantile at the pair's df
      (``inferplot.stats.smm.smm_quantile``; DECISIONS.md D52). WRS2's simulated
      values are not that quantile: 3.4% to 6.6% below it at 36 and 45
      comparisons (probe-smm) and -0.35% to +4.7% off it on edge_k9, so these
      intervals differ from ggstatsplot's.

    Args:
        n_comparisons: C, the number of pairwise comparisons (k (k - 1) / 2 for k
            groups), at least 1.
        df: the pair's Yuen df: at least 2 for C up to 28 (WRS2's table and its
            t quantile stop there), at least 1 above.
        conf_level: only 0.95 (alpha = 0.05) is supported.

    Raises:
        NotImplementedError: ``conf_level`` other than 0.95, or df below 2 with C
            up to 28, where WRS2 stops with an error.
        ValueError: C below 1, or df not a number of at least 1.
    """
    level = check_conf_level(conf_level)
    if level != _SMM_CONF_LEVEL:
        raise NotImplementedError(
            "lincon's pairwise confidence intervals are implemented only for "
            f"conf_level = 0.95 (WRS2's alpha = 0.05 table); got {level!r}"
        )
    if isinstance(n_comparisons, bool) or n_comparisons < 1:
        raise ValueError(f"n_comparisons must be at least 1; got {n_comparisons!r}")
    value = float(df)
    if not value >= 1.0:
        raise ValueError(f"df must be a number of at least 1; got {df!r}")
    if n_comparisons > SMM_MAX_TABLED:
        return smm_quantile(n_comparisons, value, conf_level=level)
    if value < SMM_ROWS[0]:
        raise NotImplementedError(
            "lincon's critical value below 2 degrees of freedom is not known "
            f"(WRS2's table starts at df 2); got df {value!r}"
        )
    if n_comparisons == 1:
        # |T_df| is the maximum modulus of one comparison; smm_quantile computes it
        # to 1e-14 at any SciPy (SciPy 1.11's stdtrit is off by up to about 1e-9).
        return smm_quantile(1, value, conf_level=level)
    entries = SMM_TABLE_05[n_comparisons]
    if value >= SMM_ROWS[-1]:
        return entries[-1]
    upper = next(i for i, row in enumerate(SMM_ROWS) if row >= value)
    if SMM_ROWS[upper] == value:
        return entries[upper]
    lower = upper - 1
    r1, r2 = SMM_ROWS[lower], SMM_ROWS[upper]
    c1, c2 = entries[lower], entries[upper]
    weight = (1.0 / r1 - 1.0 / value) / (1.0 / r1 - 1.0 / r2)
    return c1 + (c2 - c1) * weight


def lincon(
    groups: Sequence[SampleLike],
    *,
    tr: float = 0.2,
    conf_level: float = 0.95,
    p_adjust_method: PAdjustMethod = "holm",
) -> tuple[LinconPair, ...]:
    """Pairwise comparisons of trimmed means, WRS2's ``lincon`` as ggstatsplot uses
    it (Wilcox 2017, section 7.4.1; Wilcox 1986).

    For every pair i < j in input order: estimate = trimmed mean(i) - trimmed
    mean(j), Yuen's standard error sqrt(d_i + d_j) and df (``yuen_t``),
    t = estimate / se, the raw p = 2 P(T_df > |t|) computed directly (R's
    2 (1 - pt) is rounded to 2^-52; DECISIONS.md D43, ``wrs2_p``), and the CI
    estimate -/+ c x se with c the Studentized maximum modulus critical value for
    C = k (k - 1) / 2 comparisons at the pair's df (``smm_critical_value``). The
    raw p-values are then adjusted together with ``p_adjust_method`` (ggstatsplot
    does not use lincon's own Hochberg adjustment; SPEC.md section 4).

    Raises:
        ValueError: fewer than two groups, a group with fewer than two values or a
            non-finite value, trimming leaving fewer than two values, a pair whose
            groups both have a winsorized variance of 0 (or too small to
            resolve), an unknown ``p_adjust_method``, or ``tr`` outside [0, 0.5).
        NotImplementedError: ``conf_level`` other than 0.95, or a pair with fewer
            than 2 df with up to 8 groups (``smm_critical_value``).
    """
    t_trim = check_trim(tr)
    check_conf_level(conf_level)
    p_adjust([], p_adjust_method)  # an unknown method fails before any work
    raw, scale, kernels = _group_kernels(groups, t_trim, "lincon", min_groups=2)
    k = len(kernels)
    n_comparisons = k * (k - 1) // 2
    smm_critical_value(n_comparisons, math.inf, conf_level=conf_level)
    rows: list[tuple[int, int, float, float, float, float, float, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            ki, kj = kernels[i], kernels[j]
            total = ki.d + kj.d
            if not total >= TINY:
                if ki.wvar == 0.0 and kj.wvar == 0.0:
                    raise ValueError(
                        f"groups {i + 1} and {j + 1} both have a winsorized variance "
                        "of 0, so their Yuen comparison is undefined"
                    )
                which = i if ki.d < kj.d else j
                raise ValueError(
                    _zero_spread_message(
                        f"group {which + 1}", raw[which], t_trim, "lincon"
                    )
                )
            se = math.sqrt(total)
            estimate = float(ki.tmean - kj.tmean)  # exact, rounded once
            t = estimate / se
            df = satterthwaite_df(ki.d, ki.h, kj.d, kj.h)
            critical = smm_critical_value(n_comparisons, df, conf_level=conf_level)
            p = two_sided_t_p(abs(t), df)
            rows.append((i, j, estimate, se, df, t, critical, p))
    adjusted = p_adjust([row[7] for row in rows], p_adjust_method)
    out: list[LinconPair] = []
    for (i, j, estimate, se, df, t, critical, p), adj in zip(
        rows, adjusted, strict=True
    ):
        half = critical * se
        out.append(
            LinconPair(
                i=i,
                j=j,
                estimate=estimate / scale,
                se=se / scale,
                df=df,
                t=t,
                critical_value=critical,
                conf_low=(estimate - half) / scale,
                conf_high=(estimate + half) / scale,
                p_value=p,
                p_adjusted=adj,
            )
        )
    return tuple(out)


# ------------------------------------------------------------------ bootstrap


def rows_trimmed(
    draws: NDArray[np.float64], tr: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trimmed mean and winsorized variance of every row of ``draws`` (which is
    reordered in place: afterwards columns g to m - g - 1 of a row hold the values
    its trimmed mean averages).

    A row whose winsorized values are all the same number (its (g + 1)-th smallest
    equals its (g + 1)-th largest) gets a variance of exactly 0 and that number as
    its trimmed mean: computed in floating point, the mean of eleven copies of 0.9
    is not 0.9 and their variance is about 1e-33, not 0, so a resample that is
    0 / 0 or infinite by the definition would come out as an arbitrary finite
    number. The callers decide such rows exactly (``exact_zero_sum``).
    """
    m = draws.shape[1]
    g = trim_count(m, tr)
    draws.partition([g, m - g - 1], axis=1)
    part = draws
    low: NDArray[np.float64] = part[:, g].copy()
    flat: NDArray[np.bool_] = low == part[:, m - g - 1]
    tmeans: NDArray[np.float64] = part[:, g : m - g].mean(axis=1)
    if g:
        part[:, :g] = part[:, g : g + 1]
        part[:, m - g :] = part[:, m - g - 1 : m - g]
    wvars: NDArray[np.float64] = part.var(axis=1, ddof=1)
    tmeans[flat] = low[flat]
    wvars[flat] = 0.0
    return tmeans, wvars


def exact_zero_sum(positive: Sequence[float], negative: Sequence[float]) -> float:
    """The sign of sum(positive) - sum(negative), exactly: 0.0, 1.0 or -1.0.

    ``math.fsum`` returns the correctly rounded sum of the doubles, which is 0
    only when the exact sum is 0 and has its sign otherwise. Used to decide
    whether a resample with a winsorized variance of 0 is 0 / 0 (undefined) or
    infinite, which a rounded difference of rounded means cannot decide.
    """
    total = math.fsum([*positive, *(-value for value in negative)])
    return 0.0 if total == 0.0 else math.copysign(1.0, total)


def _chunks(total_rows: int, values_per_row: int) -> list[int]:
    size = max(1, min(total_rows, _CHUNK_VALUES // max(1, values_per_row)))
    full, rest = divmod(total_rows, size)
    return [size] * full + ([rest] if rest else [])


def bootstrap_indices(
    rng: np.random.Generator,
    counts: Sequence[int],
    sizes: Sequence[int],
    statistic: Callable[[list[NDArray[np.int64]]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """``statistic`` on BOOTSTRAP_RESAMPLES resamples given as positions: in each
    chunk of replicates, group by group, ``sizes[j]`` positions drawn with
    replacement from the ``counts[j]`` values of group j (rows x sizes[j])."""
    out = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    start = 0
    for rows in _chunks(BOOTSTRAP_RESAMPLES, sum(sizes)):
        positions = [
            rng.integers(0, count, size=(rows, size))
            for count, size in zip(counts, sizes, strict=True)
        ]
        out[start : start + rows] = statistic(positions)
        start += rows
    return out


def bootstrap_replicates(
    rng: np.random.Generator,
    sorted_groups: Sequence[FloatArray],
    sizes: Sequence[int],
    statistic: Callable[[list[NDArray[np.float64]]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """``statistic`` on BOOTSTRAP_RESAMPLES resamples: in each chunk of
    replicates, group by group, ``sizes[j]`` draws with replacement from group j
    (``bootstrap_indices``, the same stream)."""
    return bootstrap_indices(
        rng,
        [values.size for values in sorted_groups],
        sizes,
        lambda positions: statistic(
            [
                values[index]
                for values, index in zip(sorted_groups, positions, strict=True)
            ]
        ),
    )


def _centred(ordered: Sequence[FloatArray]) -> list[FloatArray]:
    """The sorted groups minus one common value, the middle of the pooled values.

    The bootstrap statistics are differences of trimmed means over winsorized
    standard deviations, unchanged by a common shift. Resampled trimmed means are
    rounded at the level of the values they average, so on data far from 0 next to
    their spread (timestamps such as 1.7e9 + fractions, or groups that differ in
    their last bits) the rounding would swamp the differences; after the shift it
    is relative to the spread. The shift is a data value, so values within a factor
    of 2 of it are shifted exactly (Sterbenz), the order is kept, and the shifted
    data scale exactly with powers of two.
    """
    pooled = _sorted(_joined(list(ordered)))
    middle = float(pooled[pooled.size // 2])
    return [o - middle for o in ordered]


@dataclass(frozen=True, slots=True)
class Replicates:
    """The bootstrap values split as the percentile interval uses them.

    ``defined`` holds the sorted values that are not 0 / 0 (NaN); ``undefined``
    counts the 0 / 0 resamples, which are left out (DECISIONS.md D54, extended to
    the robust effect sizes); ``infinite`` counts the +/-inf resamples (a
    winsorized variance of 0 under a numerator that is not), which are kept, as
    WRS2's arithmetic gives the same infinities, so a bound can be infinite.
    """

    defined: NDArray[np.float64]
    undefined: int
    infinite: int


def split_replicates(values: NDArray[np.float64]) -> Replicates:
    defined = _sorted(values[~np.isnan(values)])
    return Replicates(
        defined=defined,
        undefined=values.size - defined.size,
        infinite=int(np.isinf(defined).sum()),
    )


def replicate_bounds(
    replicates: Replicates,
    conf_level: float,
    rule: Callable[[NDArray[np.float64], float], tuple[float, float]],
    what: str,
) -> tuple[float, float]:
    """The percentile ``rule`` over the defined resamples, its order statistics
    counted over them; a plain error when the 0 / 0 resamples leave too few."""
    if replicates.defined.size:
        try:
            return rule(replicates.defined, conf_level)
        except ValueError:
            if not replicates.undefined:
                raise
    total = replicates.undefined + replicates.defined.size
    raise ValueError(
        f"{what} is undefined (0 / 0) in {replicates.undefined:,} of {total:,} "
        "bootstrap resamples, because a resample's winsorized variance is 0 "
        "together with its numerator; that leaves "
        f"{replicates.defined.size:,}, too few for a percentile interval at "
        f"conf_level={conf_level!r} (the data have too few distinct values for "
        "this bootstrap CI)"
    )


def floor_percentile_bounds(
    ordered: NDArray[np.float64], conf_level: float
) -> tuple[float, float]:
    """akp.effect's percentile interval: with alpha = 1 - conf_level and B
    resamples, the (floor(alpha / 2 B) + 1)-th and floor((1 - alpha / 2) B)-th
    smallest values (oracle replay: the 3rd and 97th of 100, the 501st and 19,500th
    of 20,000)."""
    count = ordered.size
    alpha = 1.0 - conf_level
    low = math.floor(alpha / 2.0 * count)
    high = math.floor((1.0 - alpha / 2.0) * count)
    if low < 1:
        raise _level_too_extreme(conf_level, count)
    return float(ordered[low]), float(ordered[high - 1])


def _level_too_extreme(conf_level: float, count: int) -> ValueError:
    """The percentile rule would return the most extreme resamples at any level
    from here up, so the interval would stop widening with ``conf_level``."""
    return ValueError(
        f"conf_level={conf_level!r} is too close to 1 for a percentile interval "
        f"from {count:,} bootstrap resamples: its bounds would be the most extreme "
        "resampled values whatever the level. conf_level must leave at least one "
        f"resample beyond each bound (up to about {1.0 - 2.0 / count:g})"
    )


def rounded_percentile_bounds(
    ordered: NDArray[np.float64], conf_level: float, alpha: float
) -> tuple[float, float]:
    """The rounded percentile rule (Wilcox 2017, section 3.1): with
    l = round(alpha / 2 B), rounded half to even on the binary value as R's
    ``round`` does (Python's ``round`` does the same), the (l + 1)-th and
    (B - l)-th smallest of the B sorted values. ``conf_level`` names the level in
    the message when l would be 0 (the bounds would be the most extreme values at
    any level)."""
    count = ordered.size
    low = round(alpha / 2.0 * count)
    if low < 1:
        raise _level_too_extreme(conf_level, count)
    return float(ordered[low]), float(ordered[count - low - 1])


def _rounded_percentile_bounds(
    ordered: NDArray[np.float64], conf_level: float
) -> tuple[float, float]:
    """t1way's percentile interval (``rounded_percentile_bounds``) with
    alpha = 1 - conf_level computed in floating point (oracle replay: the 4th and
    97th of 100, because alpha / 2 B is 2.5000000000000022 there; the 501st and
    19,500th of 20,000; probe-bootrules: 12 of 12 at B = 39 to 200)."""
    return rounded_percentile_bounds(ordered, conf_level, 1.0 - conf_level)


def akp_effect(
    a: SampleLike,
    b: SampleLike,
    *,
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: SeedLike = 0,
) -> BootstrapEffectSizeCI:
    """AKP robust standardized difference with a percentile bootstrap CI.

    Algina, Keselman & Penfield (2005, Psychological Methods 10: 317-328):
    delta = s_N (trimmed mean(a) - trimmed mean(b)) / s_w(a), with s_w(a) the
    winsorized standard deviation of the first sample alone (their heteroscedastic
    variant, R's ``WRS2::akp.effect(EQVAR = FALSE)`` as statsExpressions 2.2.1
    calls it) and s_N = sqrt(``winsorized_normal_variance(tr)``), 0.642 at tr = 0.2,
    which makes delta Cohen's d under normality. The sign follows a - b.

    CI: BOOTSTRAP_RESAMPLES resamples, each drawing n_a values of a and then n_b of
    b with replacement; the bounds are the (floor(alpha / 2 B) + 1)-th and
    floor((1 - alpha / 2) B)-th smallest of the resampled deltas, alpha =
    1 - conf_level (WRS2's order statistics, read off the oracle's replay). A
    resample whose first sample has a winsorized variance of 0 while the trimmed
    means differ gives delta = +/-inf; it is kept (WRS2's formula gives the same,
    and edge_heavy_ties2's reference bound is -Inf), so a bound can be infinite,
    and counted in ``infinite_resamples``. A resample where the trimmed means are
    also equal gives 0 / 0: it is left out, B is the number of the others, and it
    is counted in ``undefined_resamples`` (DECISIONS.md D54's rule; the fixtures
    have too few such resamples to tell it from WRS2's alternatives, S3).

    WRS2 evaluates s_N numerically, off the exact value used here by an amount
    that varies with tr: 1.2e-9 relative above it at tr = 0.2, 4.6e-9 at 0.1, up
    to 4.2e-7 at 0.49 (probe-akp; tolerance ``akp_constant``; DECISIONS.md D42).

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value,
            trimming leaves fewer than two values, the first sample's winsorized
            variance is 0 (or too small to resolve next to the largest |value|),
            so many resamples are 0 / 0 that the others are too few for the
            percentile interval, or an invalid ``tr``, ``conf_level`` or ``seed``.
    """
    level = check_conf_level(conf_level)
    rng_seed = check_bootstrap_seed(seed)
    estimate, values = _akp_replicates(a, b, tr=tr, seed=rng_seed)
    replicates = split_replicates(values)
    low, high = replicate_bounds(
        replicates, level, floor_percentile_bounds, "the AKP effect size"
    )
    return BootstrapEffectSizeCI(
        estimate=estimate,
        conf_low=low,
        conf_high=high,
        conf_level=level,
        sidedness="two-sided",
        seed=rng_seed,
        resamples=BOOTSTRAP_RESAMPLES,
        undefined_resamples=replicates.undefined,
        infinite_resamples=replicates.infinite,
    )


def _akp_replicates(
    a: SampleLike, b: SampleLike, *, tr: float, seed: int
) -> tuple[float, NDArray[np.float64]]:
    """``akp_effect``'s estimate and its BOOTSTRAP_RESAMPLES resampled values in
    the order drawn, NaN for a 0 / 0 resample (tests read them to apply other
    percentile rules)."""
    t_trim = check_trim(tr)
    raw_a = as_sample(a, name="a", min_size=2)
    raw_b = as_sample(b, name="b", min_size=2)
    scale = common_scale(raw_a, raw_b)
    x, y = _sorted(raw_a * scale), _sorted(raw_b * scale)
    method = "the AKP effect size"
    ka = _kernel(x, t_trim, "sample a", method)
    kb = _kernel(y, t_trim, "sample b", method)
    if not ka.wvar >= TINY:
        raise ValueError(_zero_spread_message("sample a", raw_a, t_trim, method))
    constant = math.sqrt(winsorized_normal_variance(t_trim))
    estimate = constant * float(ka.tmean - kb.tmean) / math.sqrt(ka.wvar)

    g_b = trim_count(y.size, t_trim)
    h_b = y.size - 2 * g_b

    def deltas(draws: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        tm_a, wv_a = rows_trimmed(draws[0], t_trim)
        tm_b, _ = rows_trimmed(draws[1], t_trim)
        with np.errstate(divide="ignore", invalid="ignore"):
            out: NDArray[np.float64] = constant * (tm_a - tm_b) / np.sqrt(wv_a)
        # A winsorized variance of 0: the resample is +/-inf, or 0 / 0 when the
        # trimmed means are equal, decided exactly: tm_a is then one value of a,
        # and h_b tm_a - sum(the values b's trimmed mean averages) has the sign of
        # the difference of the exact trimmed means.
        for row in (wv_a == 0.0).nonzero()[0].tolist():
            middle = draws[1][row, g_b : y.size - g_b].tolist()
            sign = exact_zero_sum([float(tm_a[row])] * h_b, middle)
            out[row] = math.nan if sign == 0.0 else sign * math.inf
        return out

    rng = np.random.Generator(np.random.PCG64(seed))
    values = bootstrap_replicates(rng, _centred((x, y)), (x.size, y.size), deltas)
    return estimate, values


def explanatory_effect(
    groups: Sequence[SampleLike],
    *,
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: SeedLike = 0,
) -> BootstrapEffectSizeCI:
    """Wilcox & Tian's (2011, Journal of Applied Statistics 38: 1359-1368)
    explanatory measure of effect size xi with a percentile bootstrap CI, as
    ``WRS2::t1way`` reports it (``effsize``, ``effsize_ci``).

    xi^2 = sigma^2(Yhat) / sigma^2(Y): sigma^2(Yhat) is the sample variance (J - 1
    denominator) of the J trimmed means and sigma^2(Y) the winsorized variance of
    the pooled values divided by ``winsorized_normal_variance(tr)``, so that it
    estimates the variance under normality; xi = sqrt(xi^2) (Mair & Wilcox 2020,
    eqs. 7 and 10). xi can exceed 1.

    Bootstrap: with N the smallest group size, BOOTSTRAP_RESAMPLES resamples each
    draw N values with replacement from every group in turn (all groups the same
    size, as the pooled variance needs) and compute xi; the CI bounds are the
    (l + 1)-th and (B - l)-th smallest, l = round(alpha / 2 B) with alpha =
    1 - conf_level in floating point (WRS2's rule; at B = 100 alpha / 2 B is
    2.5000000000000022, so l = 3, which ggstatsplot's own values need). A resample
    whose pooled winsorized variance is 0 while its trimmed means differ gives
    xi = inf; it is kept and counted in ``infinite_resamples``. One whose trimmed
    means are also equal gives 0 / 0: it is left out, B is the number of the
    others, and it is counted in ``undefined_resamples`` (as ``akp_effect``).
    The groups draw from the stream in an order fixed by their contents (size,
    then sorted values), so the result does not depend on the order of the
    groups.

    Estimate: with equal group sizes, xi of the data (inf when the pooled
    winsorized variance is 0 and the trimmed means differ). With unequal sizes the
    median of the (defined) bootstrap values, as WRS2 reports it (oracle replay;
    the 100-resample values reproduce exactly), so it depends on ``seed``.

    WRS2 divides by 0.4120867 instead of the exact winsorized normal variance
    0.41208672670619, so its xi is 3.2e-8 relative below the value here (tolerance
    ``wrs2_xi_constant``; DECISIONS.md D42).

    Raises:
        ValueError: fewer than two groups, a group with fewer than two values or a
            non-finite value, a pooled winsorized variance of 0 together with equal
            trimmed means in the data (0 / 0), so many 0 / 0 resamples that the
            others are too few for the percentile interval, or an invalid ``tr``,
            ``conf_level`` or ``seed``.
    """
    t_trim = check_trim(tr)
    level = check_conf_level(conf_level)
    rng_seed = check_bootstrap_seed(seed)
    method = "the explanatory effect size"
    if len(groups) < 2:
        raise ValueError(f"{method} needs at least two groups; got {len(groups)}")
    raw = [
        as_sample(g, name=f"group {i + 1}", min_size=2) for i, g in enumerate(groups)
    ]
    scale = common_scale(*raw)
    ordered = [_sorted(r * scale) for r in raw]
    size = min(o.size for o in ordered)
    normal = winsorized_normal_variance(t_trim)
    k = len(ordered)

    g_draw = trim_count(size, t_trim)

    def xis(draws: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        _, pooled = rows_trimmed(_joined(draws), t_trim)
        tmeans = _joined([rows_trimmed(d, t_trim)[0][:, None] for d in draws])
        top = tmeans.var(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out: NDArray[np.float64] = np.sqrt(top / (pooled / normal))
        # A pooled winsorized variance of 0: the resample is infinite, or 0 / 0
        # when the trimmed means of all groups are equal, decided exactly (every
        # group draws the same number of values, so equal trimmed means are equal
        # sums of the values they average).
        for row in (pooled == 0.0).nonzero()[0].tolist():
            first = draws[0][row, g_draw : size - g_draw].tolist()
            equal = all(
                exact_zero_sum(d[row, g_draw : size - g_draw].tolist(), first) == 0.0
                for d in draws[1:]
            )
            out[row] = math.nan if equal else math.inf
        return out

    estimate: float | None = None
    if all(o.size == size for o in ordered):
        # The variance of the exact trimmed means, rounded once.
        top = exact_variance([exact_trimmed_mean(o, t_trim) for o in ordered])
        pooled = _sorted(_joined(ordered))
        wvar = scaled_variance(
            winsorize_sorted(pooled, trim_count(pooled.size, t_trim))
        )
        if wvar == 0.0:
            if top == 0.0:
                raise ValueError(
                    f"{method} is undefined (0 / 0): the trimmed means are equal and "
                    "the pooled winsorized variance is 0"
                )
            estimate = math.inf
        else:
            estimate = math.sqrt(top / (wvar / normal))
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    # The groups draw from the stream in an order fixed by their contents (size,
    # then sorted values), so the result does not depend on the order of the
    # groups (``rank_epsilon_squared`` does the same).
    canonical = sorted(_centred(ordered), key=lambda o: (o.size, o.tolist()))
    replicates = split_replicates(bootstrap_replicates(rng, canonical, [size] * k, xis))
    low, high = replicate_bounds(replicates, level, _rounded_percentile_bounds, method)
    if estimate is None:
        estimate = float(np.median(replicates.defined))
    return BootstrapEffectSizeCI(
        estimate=estimate,
        conf_low=low,
        conf_high=high,
        conf_level=level,
        sidedness="two-sided",
        seed=rng_seed,
        resamples=BOOTSTRAP_RESAMPLES,
        undefined_resamples=replicates.undefined,
        infinite_resamples=replicates.infinite,
    )
