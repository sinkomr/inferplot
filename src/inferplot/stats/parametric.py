"""Welch's t-test, Student's pooled t statistic and Welch's one-way ANOVA."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

from scipy import special

from inferplot.stats.descriptive import SampleLike, as_sample, mean, variance

__all__ = [
    "ALTERNATIVES",
    "Alternative",
    "StudentT",
    "WelchAnova",
    "WelchT",
    "check_alternative",
    "student_t",
    "t_p_value",
    "welch_anova",
    "welch_t",
]

Alternative = Literal["two-sided", "less", "greater"]
"""Direction of the alternative (R's ``alternative``, with "two.sided" respelled)."""

ALTERNATIVES: Final[tuple[Alternative, ...]] = get_args(Alternative)


def check_alternative(alternative: str) -> Alternative:
    """``alternative`` if it is one of :data:`ALTERNATIVES`, else ValueError."""
    for allowed in ALTERNATIVES:
        if alternative == allowed:
            return allowed
    raise ValueError(
        f"alternative must be one of {', '.join(ALTERNATIVES)}; got {alternative!r}"
    )


@dataclass(frozen=True, slots=True)
class WelchT:
    """Welch's two-sample t-test (R: ``stats::t.test(var.equal = FALSE)``)."""

    t: float
    df: float
    p_value: float
    mean_difference: float
    se: float
    alternative: Alternative


@dataclass(frozen=True, slots=True)
class StudentT:
    """Student's two-sample t statistic with the pooled variance."""

    t: float
    df: float


@dataclass(frozen=True, slots=True)
class WelchAnova:
    """Welch's heteroscedastic one-way ANOVA (R: ``stats::oneway.test``)."""

    f: float
    df1: float
    df2: float
    p_value: float


def _t_sf(t: float, df: float) -> float:
    """P(T_df > t), the Student t upper tail (SciPy's ``stdtr`` on -t)."""
    return float(special.stdtr(df, -t))


def t_p_value(t: float, df: float, alternative: Alternative) -> float:
    """p-value of a t statistic: two-sided 2 P(T > |t|), "greater" P(T > t),
    "less" P(T < t); upper tails computed directly, never as 1 - cdf."""
    if alternative == "two-sided":
        return min(1.0, 2.0 * _t_sf(abs(t), df))
    if alternative == "greater":
        return _t_sf(t, df)
    return _t_sf(-t, df)


def _two_samples(a: SampleLike, b: SampleLike) -> tuple[int, int, float, float]:
    """Sizes and variances of two samples, each with at least two values."""
    x = as_sample(a, name="a", min_size=2)
    y = as_sample(b, name="b", min_size=2)
    return x.size, y.size, variance(x), variance(y)


def welch_t(
    a: SampleLike, b: SampleLike, *, alternative: Alternative = "two-sided"
) -> WelchT:
    """Welch's t-test of mean(a) - mean(b) (Welch 1947, Biometrika 34: 28-35).

    t = (mean(a) - mean(b)) / sqrt(v_a / n_a + v_b / n_b) with the Welch-Satterthwaite
    df = (w_a + w_b)^2 / (w_a^2 / (n_a - 1) + w_b^2 / (n_b - 1)), w = v / n
    (Satterthwaite 1946). Matches R's ``stats::t.test(a, b, var.equal = FALSE,
    alternative = ...)``; R's "two.sided" is spelled "two-sided" here.

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, both
            samples are constant (the standard error is 0), or ``alternative`` is
            not one of :data:`ALTERNATIVES`.
    """
    side = check_alternative(alternative)
    n_a, n_b, v_a, v_b = _two_samples(a, b)
    w_a, w_b = v_a / n_a, v_b / n_b
    if w_a + w_b == 0.0:
        raise ValueError("both samples are constant: Welch's t is undefined")
    se = math.sqrt(w_a + w_b)
    difference = mean(a) - mean(b)
    t = difference / se
    df = (w_a + w_b) ** 2 / (w_a**2 / (n_a - 1) + w_b**2 / (n_b - 1))
    return WelchT(
        t=t,
        df=df,
        p_value=t_p_value(t, df, side),
        mean_difference=difference,
        se=se,
        alternative=side,
    )


def student_t(a: SampleLike, b: SampleLike) -> StudentT:
    """Student's two-sample t with the pooled variance (Student 1908; Fisher 1925).

    t = (mean(a) - mean(b)) / (s_p sqrt(1/n_a + 1/n_b)),
    s_p^2 = ((n_a - 1) v_a + (n_b - 1) v_b) / (n_a + n_b - 2), df = n_a + n_b - 2:
    the statistic of R's ``stats::t.test(var.equal = TRUE)``, which
    ``BayesFactor::ttestBF`` uses for two independent samples.

    Raises:
        ValueError: a sample has fewer than two values, a non-finite value, or
            both samples are constant (the pooled variance is 0).
    """
    n_a, n_b, v_a, v_b = _two_samples(a, b)
    df = n_a + n_b - 2
    pooled = ((n_a - 1) * v_a + (n_b - 1) * v_b) / df
    if pooled == 0.0:
        raise ValueError("both samples are constant: Student's t is undefined")
    t = (mean(a) - mean(b)) / math.sqrt(pooled * (1.0 / n_a + 1.0 / n_b))
    return StudentT(t=t, df=float(df))


def welch_anova(groups: Sequence[SampleLike]) -> WelchAnova:
    """Welch's heteroscedastic one-way ANOVA (Welch 1951, Biometrika 38: 330-336).

    With weights w_i = n_i / v_i, W = sum w_i, the weighted mean m = sum w_i mean_i / W
    and lambda = sum (1 - w_i / W)^2 / (n_i - 1):
    F = [sum w_i (mean_i - m)^2 / (k - 1)] / [1 + 2 (k - 2) lambda / (k^2 - 1)],
    df1 = k - 1, df2 = (k^2 - 1) / (3 lambda); p = P(F(df1, df2) > F). Matches R's
    ``stats::oneway.test(y ~ x, var.equal = FALSE)``.

    Raises:
        ValueError: fewer than two groups, a group with fewer than two values or a
            non-finite value, or a group with variance 0 (its weight is infinite).
    """
    if len(groups) < 2:
        raise ValueError(f"Welch's ANOVA needs at least two groups; got {len(groups)}")
    samples = [
        as_sample(g, name=f"group {i + 1}", min_size=2) for i, g in enumerate(groups)
    ]
    k = len(samples)
    n = [float(s.size) for s in samples]
    means = [mean(s) for s in samples]
    variances = [variance(s) for s in samples]
    for i, v in enumerate(variances):
        if v == 0.0:
            raise ValueError(
                f"group {i + 1} is constant: Welch's ANOVA weights need a variance > 0"
            )
    weights = [ni / vi for ni, vi in zip(n, variances, strict=True)]
    total = math.fsum(weights)
    centre = math.fsum(w * m for w, m in zip(weights, means, strict=True)) / total
    between = math.fsum(
        w * (m - centre) ** 2 for w, m in zip(weights, means, strict=True)
    ) / (k - 1)
    lam = math.fsum(
        (1.0 - w / total) ** 2 / (ni - 1.0) for w, ni in zip(weights, n, strict=True)
    )
    f = between / (1.0 + 2.0 * (k - 2) * lam / (k * k - 1))
    df1 = float(k - 1)
    df2 = (k * k - 1) / (3.0 * lam)
    p = float(special.fdtrc(df1, df2, f))
    return WelchAnova(f=f, df1=df1, df2=df2, p_value=p)
