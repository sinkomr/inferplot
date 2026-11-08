"""Standardized effect sizes with noncentrality-parameter confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from inferplot.stats.descriptive import SampleLike
from inferplot.stats.noncentral import check_conf_level, f_ncp_bound, t_ncp_interval
from inferplot.stats.parametric import Alternative, check_alternative, welch_t_scaled
from inferplot.stats.srange import stirlerr

__all__ = [
    "EffectSizeCI",
    "Sidedness",
    "hedges_g",
    "hedges_j",
    "partial_omega_squared",
]

Sidedness = Literal["two-sided", "greater", "less"]
"""Which bounds of an interval are estimated: "greater" is [low, max], "less"
[min, high]."""

# Below this a = df / 2 the log-gamma values are small, so their difference is exact
# to rounding; above it Stirling's remainder avoids the cancellation of two large
# log-gamma values.
_STIRLING_FROM: Final = 15.5


@dataclass(frozen=True, slots=True)
class EffectSizeCI:
    """An effect size with its confidence interval (plain Python floats)."""

    estimate: float
    conf_low: float
    conf_high: float
    conf_level: float
    sidedness: Sidedness


def hedges_j(df: float) -> float:
    """Hedges' exact small-sample correction J(df) = Gamma(df/2) /
    (sqrt(df/2) Gamma((df - 1)/2)) (Hedges 1981, Journal of Educational Statistics
    6: 107-128, eq. 6e).

    For a = df / 2 >= 15.5 it is evaluated as
    log J = -(a - 1) log1p(-1 / (2a)) - 1/2 + stirlerr(a) - stirlerr(a - 1/2),
    which follows from Stirling's formula with its remainder and keeps full relative
    accuracy for large df. J(1) = 0 (Gamma(0) is infinite), and so is J for df <= 1,
    where the ratio is not defined; R's ``exp(lgamma(.) - ...)`` also gives 0 at
    df = 1.
    """
    df = float(df)
    if math.isnan(df) or df <= 0.0:
        raise ValueError(f"df must be > 0; got {df!r}")
    if math.isinf(df):
        return 1.0
    if df <= 1.0:
        return 0.0
    a = 0.5 * df
    if a < _STIRLING_FROM:
        return math.exp(math.lgamma(a) - 0.5 * math.log(a) - math.lgamma(a - 0.5))
    log_j = -(a - 1.0) * math.log1p(-0.5 / a) - 0.5 + stirlerr(a) - stirlerr(a - 0.5)
    return math.exp(log_j)


def hedges_g(
    a: SampleLike,
    b: SampleLike,
    *,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Hedges' g of mean(a) - mean(b) with the unpooled standardizer and Welch's df.

    d = (mean(a) - mean(b)) / s with s = sqrt((v_a + v_b) / 2) (the standardizer for
    unequal variances, Delacre, Lakens, Ley, Liu & Leys 2021, International Review
    of Social Psychology 34: 1-21), g = d J(df_W) with Hedges' exact correction at
    the Welch df (Hedges 1981). CI: the noncentral-t interval for the Welch t at
    df_W (``t_ncp_interval``; Steiger & Fouladi 1997), each bound times
    (Welch SE / s) J(df_W). A one-sided ``alternative`` gives a one-sided interval
    whose far bound is infinite. At df_W = 1 (one constant sample, the other of two
    values) J = 0, so g and both bounds are 0 by construction; they are +0.0, never
    -0.0. Computed on both samples times their ``common_scale``: g and its bounds
    are the same, bit for bit, for the data times any power of two that keeps them
    normal doubles. Matches R's ``effectsize::hedges_g(pooled_sd =
    FALSE, ci = conf_level, alternative = ...)`` as statsExpressions 2.2.1 calls it
    (SPEC.md section 4).

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, both
            are constant, conf_level is not strictly between 0 and 1, or alternative
            is not one of "two-sided", "less", "greater".
    """
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    welch = welch_t_scaled(a, b)
    # Difference, SE and SD in the samples' common scale: g does not depend on it.
    sd = math.sqrt(0.5 * (welch.v_a + welch.v_b))
    j = hedges_j(welch.df)
    # + 0.0 turns a zero product with a negative factor (J = 0) into +0.0.
    estimate = welch.difference / sd * j + 0.0
    low, high = t_ncp_interval(welch.t, welch.df, conf_level=level, alternative=side)
    scale = welch.se / sd * j
    return EffectSizeCI(
        estimate=estimate,
        # The far bound of a one-sided interval stays infinite (also when J = 0).
        conf_low=low * scale + 0.0 if math.isfinite(low) else low,
        conf_high=high * scale + 0.0 if math.isfinite(high) else high,
        conf_level=level,
        sidedness=side,
    )


def partial_omega_squared(
    f: float, df1: float, df2: float, *, conf_level: float = 0.95
) -> EffectSizeCI:
    """Partial omega squared from an F statistic with a one-sided noncentral-F CI.

    Estimate (Olejnik & Algina 2003, Psychological Methods 8: 434-447, written in
    terms of F): omega = max(0, (F - 1) df1 / (F df1 + df2 + 1)), with Welch's F and
    its fractional df2 for Welch's ANOVA.

    CI, the construction of R's ``effectsize::omega_squared(ci = conf_level)`` with
    its default ``alternative = "greater"`` as ggstatsplot 1.1.4 calls it (found by
    black-box probes, DECISIONS.md D33): treat the estimate as a partial eta squared
    and convert it back to an F value, F_w = omega df2 / ((1 - omega) df1); invert the
    noncentral F there, lambda = ``f_ncp_bound(F_w, df1, df2, prob=conf_level)``
    (Steiger 2004); the lower bound is lambda / (lambda + df2) and the upper bound 1.
    F <= 1 gives omega 0 and the interval [0, 1]. F_w is computed through the
    identity F_w = (F - 1) df2 / (df1 + df2 + 1) (1 - omega = (df1 + df2 + 1) /
    (F df1 + df2 + 1)), which avoids dividing by 1 - omega, 0 once omega rounds to 1
    (F near 1e17); the two agree within 3e-15 relative where both are defined. A
    bound beyond the largest double gives a lower bound of 1.

    Run time: a few milliseconds per call at any F up to the largest double (the
    subsampled Poisson sum and the large-lambda bracket of ``f_ncp_bound``; before
    them the cost grew like sqrt(F), 125 s at F = 1e12). Against the full Poisson
    sum inverted from 0 (the method before), for F = 10 to 1e10 at six (df1, df2,
    level) settings the lower bounds agree within 1e-12 absolute (measured:
    see tests/stats/test_effect_sizes.py).

    The bound is the accurate root, not effectsize's: on the oracle probe
    probe-omega (577 rows with alternative "greater") effectsize agrees with an
    accurate uniroot bound within ``noncentral_ci`` in 574 rows, where inferplot
    matches it too; in 3 rows its root search stops early (at worst 2.1e-3 low,
    F = 10, df = (4, 5.5), 90%) and inferplot matches the accurate root
    (DECISIONS.md D33; tests/stats/test_probes.py). The estimate equals effectsize's
    bit for bit in all 577 rows.

    Raises:
        ValueError: F is negative or not finite, a df is not positive, or
            conf_level is not strictly between 0 and 1.
    """
    f, df1, df2 = float(f), float(df1), float(df2)
    for name, value in (("F", f), ("df1", df1), ("df2", df2)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
    if f < 0.0:
        raise ValueError(f"F must be >= 0; got {f!r}")
    if df1 <= 0.0 or df2 <= 0.0:
        raise ValueError(f"df1 and df2 must be > 0; got {df1!r} and {df2!r}")
    level = check_conf_level(conf_level)
    if math.isfinite(f * df1 + df2 + 1.0):
        estimate = max(0.0, (f - 1.0) * df1 / (f * df1 + df2 + 1.0))
    else:  # 1 - omega = (df1 + df2 + 1) / (F df1 + df2 + 1) rounds to 0
        estimate = 1.0
    # F_w = omega df2 / ((1 - omega) df1) = (F - 1) df2 / (df1 + df2 + 1) for F > 1,
    # without the division by 1 - omega, which rounds to 0 for F near 1e17.
    f_w = (f - 1.0) * (df2 / (df1 + df2 + 1.0)) if f > 1.0 else 0.0
    ncp = f_ncp_bound(f_w, df1, df2, prob=level)
    return EffectSizeCI(
        estimate=estimate,
        # A root beyond the largest double: the bound rounds to 1.
        conf_low=ncp / (ncp + df2) if math.isfinite(ncp) else 1.0,
        conf_high=1.0,
        conf_level=level,
        sidedness="greater",
    )
