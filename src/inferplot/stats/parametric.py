"""Welch's t-test, Student's pooled t statistic and Welch's one-way ANOVA."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
from scipy import special

from inferplot.stats.descriptive import (
    FloatArray,
    SampleLike,
    as_sample,
    common_scale,
    mean,
    scaled_variance,
)

__all__ = [
    "ALTERNATIVES",
    "TINY",
    "Alternative",
    "ScaledWelch",
    "StudentT",
    "WelchAnova",
    "WelchT",
    "check_alternative",
    "check_variance_term",
    "checked_scaled_variances",
    "constant_group_message",
    "satterthwaite_df",
    "student_t",
    "t_p_value",
    "welch_anova",
    "welch_t",
    "welch_t_scaled",
]

Alternative = Literal["two-sided", "less", "greater"]
"""Direction of the alternative (R's ``alternative``, with "two.sided" respelled)."""

ALTERNATIVES: Final[tuple[Alternative, ...]] = get_args(Alternative)

TINY: Final = 2.0**-1022
"""The smallest normal double. In the common scale (largest |value| in [1/2, 1))
a variance term v / n at least this large is accurate to about 2^-53 relative: the
subnormal squares and values below it carry absolute rounding errors of at most
2^-1075 each. Below it the term cannot be trusted, and a statistic that divides by
it is refused (``check_variance_term``, ``constant_group_message``)."""


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


def _two_samples(
    a: SampleLike, b: SampleLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float]:
    """Both samples (at least two values each), the same times their
    ``common_scale``, and that scale."""
    x = as_sample(a, name="a", min_size=2)
    y = as_sample(b, name="b", min_size=2)
    scale = common_scale(x, y)
    return x, y, x * scale, y * scale, scale


def check_variance_term(term: float, a: FloatArray, b: FloatArray, method: str) -> None:
    """Raise ValueError unless ``term``, the scaled variance term that two-sample
    ``method`` divides by, is at least :data:`TINY`.

    Two constant samples get "both samples are constant"; otherwise the samples
    that vary (``a`` and ``b`` are the raw values) are named: they vary by less than
    double precision resolves next to the largest |value| of both samples. That
    happens only when the sample with the largest |value| is constant, since a
    varying one has a scaled variance of at least about 2^-108 / n.
    """
    if term >= TINY:
        return
    varying = [name for name, s in (("a", a), ("b", b)) if float(np.ptp(s)) != 0.0]
    if not varying:
        raise ValueError(f"both samples are constant: {method} is undefined")
    if len(varying) == 1:
        other = "b" if varying[0] == "a" else "a"
        which = f"sample {other} is constant and sample {varying[0]} varies"
    else:  # not reached: the sample with the largest |value| would have to vary
        which = "samples a and b vary"
    raise ValueError(
        f"{which} by less than double precision can resolve next to the largest "
        "|value| of both samples (a standard deviation below roughly 1e-154 times "
        f"that value), so {method} has no accurate standard error. Check the data "
        "for entry or unit errors; rescaling both samples does not help, since it "
        "keeps their ratio."
    )


def satterthwaite_df(w_a: float, n_a: int, w_b: float, n_b: int) -> float:
    """Welch-Satterthwaite df of two variance terms w = v / n, from their shares.

    (w_a + w_b)^2 / (w_a^2 / (n_a - 1) + w_b^2 / (n_b - 1)) written as
    1 / (r_a^2 / (n_a - 1) + r_b^2 / (n_b - 1)) with r = w / (w_a + w_b), so no
    square of w can overflow or underflow (Satterthwaite 1946).
    """
    total = w_a + w_b
    r_a, r_b = w_a / total, w_b / total
    return 1.0 / (r_a * r_a / (n_a - 1) + r_b * r_b / (n_b - 1))


@dataclass(frozen=True, slots=True)
class ScaledWelch:
    """Welch's t in the samples' common scale: ``difference`` and ``se`` are the
    mean difference and its standard error times ``scale``."""

    t: float
    df: float
    difference: float
    se: float
    scale: float
    v_a: float
    v_b: float


def welch_t_scaled(a: SampleLike, b: SampleLike) -> ScaledWelch:
    """Welch's t of mean(a) - mean(b) on both samples times their ``common_scale``
    (``welch_t`` for the method); the mean difference, standard error and
    variances stay in that scale, where none of them overflows.

    Limits: the scaled squared standard error w_a + w_b must be at least
    :data:`TINY` (``check_variance_term``), so it is accurate to about 2^-53
    relative and |t| < 2 / sqrt(TINY) = 2^512 (about 1.3e154).

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, both
            samples are constant, or one is constant and the other varies by less
            than double precision resolves next to the constant one (a standard
            deviation below roughly 1e-154 times its |value|).
    """
    raw_a, raw_b, x, y, scale = _two_samples(a, b)
    n_a, n_b = x.size, y.size
    v_a, v_b = scaled_variance(x), scaled_variance(y)
    w_a, w_b = v_a / n_a, v_b / n_b
    check_variance_term(w_a + w_b, raw_a, raw_b, "Welch's t")
    se = math.sqrt(w_a + w_b)
    difference = mean(x) - mean(y)
    return ScaledWelch(
        t=difference / se,
        df=satterthwaite_df(w_a, n_a, w_b, n_b),
        difference=difference,
        se=se,
        scale=scale,
        v_a=v_a,
        v_b=v_b,
    )


def welch_t(
    a: SampleLike, b: SampleLike, *, alternative: Alternative = "two-sided"
) -> WelchT:
    """Welch's t-test of mean(a) - mean(b) (Welch 1947, Biometrika 34: 28-35).

    t = (mean(a) - mean(b)) / sqrt(v_a / n_a + v_b / n_b) with the Welch-Satterthwaite
    df = (w_a + w_b)^2 / (w_a^2 / (n_a - 1) + w_b^2 / (n_b - 1)), w = v / n
    (Satterthwaite 1946), computed from the shares w / (w_a + w_b)
    (``satterthwaite_df``). Matches R's ``stats::t.test(a, b, var.equal = FALSE,
    alternative = ...)``; R's "two.sided" is spelled "two-sided" here.

    Scale: everything is computed on both samples times their ``common_scale`` (a
    power of two, exact), so t, df and p are the same, bit for bit, for the data
    times any power of two that keeps them normal doubles, and ``mean_difference``
    and ``se`` scale exactly with the data.

    Limits: |t| stays below 2^512 (about 1.3e154): when one sample is constant and
    the other varies by less than about 1e-154 times the constant |value| the
    standard error cannot be computed accurately and ValueError is raised
    (``welch_t_scaled``).

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, both
            samples are constant (the standard error is 0), one is constant and the
            other varies too little next to it (above), the mean difference or
            the standard error exceeds the largest double, or ``alternative`` is
            not one of :data:`ALTERNATIVES`.
    """
    side = check_alternative(alternative)
    core = welch_t_scaled(a, b)
    difference = core.difference / core.scale
    se = core.se / core.scale
    if math.isinf(difference) or math.isinf(se):
        raise ValueError(
            "the mean difference or its standard error exceeds the largest double "
            "(about 1.8e308); rescale the samples, for example by changing units"
        )
    return WelchT(
        t=core.t,
        df=core.df,
        p_value=t_p_value(core.t, core.df, side),
        mean_difference=difference,
        se=se,
        alternative=side,
    )


def student_t(a: SampleLike, b: SampleLike) -> StudentT:
    """Student's two-sample t with the pooled variance (Student 1908; Fisher 1925).

    t = (mean(a) - mean(b)) / (s_p sqrt(1/n_a + 1/n_b)),
    s_p^2 = ((n_a - 1) v_a + (n_b - 1) v_b) / (n_a + n_b - 2), df = n_a + n_b - 2:
    the statistic of R's ``stats::t.test(var.equal = TRUE)``, which
    ``BayesFactor::ttestBF`` uses for two independent samples. Computed on both
    samples times their ``common_scale``, so t does not depend on the scale of the
    data (bit for bit under powers of two).

    Limits: the scaled squared standard error s_p^2 (1/n_a + 1/n_b) must be at
    least :data:`TINY` (``check_variance_term``), so it is accurate to about 2^-53
    relative and |t| < 2^512 (about 1.3e154).

    Raises:
        ValueError: a sample has fewer than two values, a non-finite value, both
            samples are constant (the pooled variance is 0), or one is constant
            and the other varies by less than double precision resolves next to it.
    """
    raw_a, raw_b, x, y, _ = _two_samples(a, b)
    n_a, n_b = x.size, y.size
    df = n_a + n_b - 2
    pooled = ((n_a - 1) * scaled_variance(x) + (n_b - 1) * scaled_variance(y)) / df
    term = pooled * (1.0 / n_a + 1.0 / n_b)
    check_variance_term(term, raw_a, raw_b, "Student's t")
    t = (mean(x) - mean(y)) / math.sqrt(term)
    return StudentT(t=t, df=float(df))


def constant_group_message(index: int, values: FloatArray, method: str) -> str:
    """Why group ``index`` (0-based) has no usable variance, in plain words: it is
    constant (``values``, the raw group, has a range of 0), or its scaled variance
    term v / n is below :data:`TINY`."""
    if float(np.ptp(values)) == 0.0:
        return f"group {index + 1} is constant: {method} needs a variance > 0"
    return (
        f"group {index + 1} varies, but by less than double precision can represent "
        "next to the largest value of all groups (a standard deviation below "
        f"roughly 1e-154 times that value), so {method} cannot weight it by its "
        "variance accurately. Check the data for entry or unit errors, or analyze "
        "groups of similar magnitude separately; rescaling all groups does not help."
    )


def checked_scaled_variances(
    raw: Sequence[FloatArray], samples: Sequence[FloatArray], method: str
) -> list[float]:
    """The variances v_i of the scaled groups ``samples``, each with a variance
    term v_i / n_i of at least :data:`TINY` (ValueError naming the first group
    below it, from its raw values ``raw``)."""
    variances: list[float] = []
    for index, sample in enumerate(samples):
        v = scaled_variance(sample)
        if not v / sample.size >= TINY:
            raise ValueError(constant_group_message(index, raw[index], method))
        variances.append(v)
    return variances


def welch_anova(groups: Sequence[SampleLike]) -> WelchAnova:
    """Welch's heteroscedastic one-way ANOVA (Welch 1951, Biometrika 38: 330-336).

    With weights w_i = n_i / v_i, W = sum w_i, the weighted mean m = sum w_i mean_i / W
    and lambda = sum (1 - w_i / W)^2 / (n_i - 1):
    F = [sum w_i (mean_i - m)^2 / (k - 1)] / [1 + 2 (k - 2) lambda / (k^2 - 1)],
    df1 = k - 1, df2 = (k^2 - 1) / (3 lambda); p = P(F(df1, df2) > F). Matches R's
    ``stats::oneway.test(y ~ x, var.equal = FALSE)``. Computed on the groups times
    their ``common_scale``, so F, df and p do not depend on the scale of the data
    (bit for bit under powers of two).

    Limits: every group's scaled variance term v_i / n_i must be at least
    :data:`TINY` (accurate to about 2^-53 relative; a standard deviation above
    roughly 1e-154 times the largest |value| of all groups). The weights are then
    finite but can reach 2^1022; when the largest reaches
    2^(1015 - bit_length(k)), all weights are multiplied by one power of
    two that brings it below that bound, so no sum overflows (the exactness of
    power-of-two scaling keeps F, df and p bit-identical wherever the unscaled sums
    did not overflow), and F is multiplied back at the end.

    Raises:
        ValueError: fewer than two groups, a group with fewer than two values or a
            non-finite value, a constant group or one whose variance term is below
            TINY (it varies by less than double precision can represent next to
            the largest value of all groups), or an F above the largest double.
    """
    if len(groups) < 2:
        raise ValueError(f"Welch's ANOVA needs at least two groups; got {len(groups)}")
    raw = [
        as_sample(g, name=f"group {i + 1}", min_size=2) for i, g in enumerate(groups)
    ]
    scale = common_scale(*raw)
    samples = [s * scale for s in raw]
    k = len(samples)
    n = [float(s.size) for s in samples]
    means = [mean(s) for s in samples]
    variances = checked_scaled_variances(raw, samples, "Welch's ANOVA")
    weights = [ni / vi for ni, vi in zip(n, variances, strict=True)]
    # Weights n_i / v_i are at most 1 / TINY = 2^1022 and at least 1/2 (v <= 2 for
    # values in [-1, 1)). One power of two brings the largest below 2^top, so the
    # sums (at most 4 k 2^top: |m| < 1, |m - c| < 2) cannot overflow and the
    # smallest weight stays a normal double; exact, so in-range results do not
    # change. F is scaled back below.
    top = 1015 - k.bit_length()
    shift = max(0, math.frexp(max(weights))[1] - top)
    if shift:
        weights = [math.ldexp(w, -shift) for w in weights]
    total = math.fsum(weights)
    centre = math.fsum(w * m for w, m in zip(weights, means, strict=True)) / total
    between = math.fsum(
        w * (m - centre) ** 2 for w, m in zip(weights, means, strict=True)
    ) / (k - 1)
    lam = math.fsum(
        (1.0 - w / total) ** 2 / (ni - 1.0) for w, ni in zip(weights, n, strict=True)
    )
    f = between / (1.0 + 2.0 * (k - 2) * lam / (k * k - 1))
    # Defensive: under the TINY condition a group with a weight near 2^1022 has a
    # mean within about 2^-458 of 0 in the common scale, so F has stayed far below
    # the largest double in every case tried.
    if shift:
        f = math.ldexp(f, shift)
    if math.isinf(f):
        raise ValueError(
            "Welch's F exceeds the largest double (about 1.8e308), so it cannot be "
            "reported. Check the data for entry or unit errors."
        )
    df1 = float(k - 1)
    df2 = (k * k - 1) / (3.0 * lam)
    p = float(special.fdtrc(df1, df2, f))
    return WelchAnova(f=f, df1=df1, df2=df2, p_value=p)
