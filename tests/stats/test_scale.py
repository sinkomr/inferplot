"""Scale robustness: the statistics do not depend on the units of the data.

Every method runs on the samples times their ``common_scale``, a power of two that
brings the largest |value| into [1/2, 1), so

* the data times 2^k give bit-for-bit the same t, df, p, F, g, CI bounds, omega,
  q and Bayes factors for every k that keeps the data normal doubles, and the
  mean difference and its standard error scale exactly;
* the data times 10^k (rounded once per value) agree within ``closed_form``
  (``noncentral_ci`` for the noncentral CI bounds), for k from -300 to 300.

Before, the Welch-Satterthwaite df squared v / n and the variance squared the
deviations: data times 1e-80 printed a wrong df (18.88 for 18.82), 1e-90 and 1e80
raised ZeroDivisionError and OverflowError (review findings STAT-1 and API-2).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from inferplot.stats import (
    games_howell,
    hedges_g,
    jzs_bf10_two_sample,
    partial_omega_squared,
    student_t,
    variance,
    welch_anova,
    welch_t,
)
from tests.tolerances import assert_close

RNG = np.random.Generator(np.random.PCG64(20261108))
# Values with magnitudes in [0.1, 10]: times 2^k they stay normal doubles for
# -1018 <= k <= 1020.
A = RNG.normal(0.0, 1.0, 10).clip(-10, 10)
B = RNG.normal(1.0, 2.0, 15).clip(-10, 10)
C = RNG.normal(-0.5, 0.5, 8).clip(-10, 10)
for _sample in (A, B, C):
    _sample[np.abs(_sample) < 0.1] = 0.1
GROUPS = (A, B, C)

ALL_POWERS_OF_TWO = range(-1000, 1001)
# The noncentral CIs, Bayes factors and studentized range tails take a few
# milliseconds each, so they are checked on every 25th power.
SOME_POWERS_OF_TWO = [*range(-1000, 1001, 25), -999, 999]
ALL_POWERS_OF_TEN = range(-300, 301)
SOME_POWERS_OF_TEN = [*range(-300, 301, 10), -299, 299]


def _cheap(a: np.ndarray, b: np.ndarray, groups: tuple[np.ndarray, ...]) -> list[float]:
    welch = welch_t(a, b)
    anova = welch_anova(groups)
    return [
        welch.t,
        welch.df,
        welch.p_value,
        student_t(a, b).t,
        anova.f,
        anova.df2,
        anova.p_value,
    ]


def _costly(
    a: np.ndarray, b: np.ndarray, groups: tuple[np.ndarray, ...]
) -> list[float]:
    g = hedges_g(a, b)
    anova = welch_anova(groups)
    omega = partial_omega_squared(anova.f, anova.df1, anova.df2)
    bf10, log_bf01 = jzs_bf10_two_sample(a, b, prior_scale=0.707)
    out = [g.estimate, g.conf_low, g.conf_high, omega.estimate, omega.conf_low]
    out += [bf10, log_bf01]
    for pair in games_howell(groups):
        assert pair.df is not None
        out += [pair.statistic, pair.df, pair.p_value, pair.p_adjusted]
    return out


def _scaled(factor: Callable[[np.ndarray], np.ndarray]) -> Any:
    return factor(A), factor(B), tuple(factor(g) for g in GROUPS)


def test_the_data_stay_normal_doubles() -> None:
    values = np.concatenate(GROUPS)
    tiny = np.finfo(np.float64).tiny
    assert np.abs(values).min() * 2.0**-1000 >= tiny
    assert np.abs(values).max() * 2.0**1000 < np.finfo(np.float64).max
    assert np.abs(values).min() * 1e-300 >= tiny


def test_powers_of_two_give_identical_results() -> None:
    base = _cheap(A, B, GROUPS)
    welch = welch_t(A, B)
    for k in ALL_POWERS_OF_TWO:
        a, b, groups = _scaled(lambda x, k=k: np.ldexp(x, k))
        assert _cheap(a, b, groups) == base, f"2^{k}"
        scaled = welch_t(a, b)
        assert scaled.mean_difference == math.ldexp(welch.mean_difference, k)
        assert scaled.se == math.ldexp(welch.se, k)
        # The variance scales by 4^k while it stays a normal double.
        exponent = math.frexp(variance(A))[1] + 2 * k
        if -1020 <= exponent <= 1024:
            assert variance(a) == math.ldexp(variance(A), 2 * k)
        elif exponent > 1024:
            with pytest.raises(ValueError, match="exceeds the largest double"):
                variance(a)


def test_powers_of_two_give_identical_effect_sizes_and_bayes_factors() -> None:
    base = _costly(A, B, GROUPS)
    for k in SOME_POWERS_OF_TWO:
        a, b, groups = _scaled(lambda x, k=k: np.ldexp(x, k))
        assert _costly(a, b, groups) == base, f"2^{k}"


def test_powers_of_ten_agree_within_closed_form() -> None:
    base = _cheap(A, B, GROUPS)
    for k in ALL_POWERS_OF_TEN:
        a, b, groups = _scaled(lambda x, k=k: x * 10.0**k)
        for i, (ours, expected) in enumerate(
            zip(_cheap(a, b, groups), base, strict=True)
        ):
            assert_close("closed_form", ours, expected, what=f"10^{k} value {i}")


def test_powers_of_ten_effect_sizes_and_bayes_factors() -> None:
    base = _costly(A, B, GROUPS)
    ci_bounds = {1, 2, 4}  # g's bounds and omega's lower bound
    for k in SOME_POWERS_OF_TEN:
        a, b, groups = _scaled(lambda x, k=k: x * 10.0**k)
        for i, (ours, expected) in enumerate(
            zip(_costly(a, b, groups), base, strict=True)
        ):
            kind = "noncentral_ci" if i in ci_bounds else "closed_form"
            assert_close(kind, ours, expected, what=f"10^{k} value {i}")


@pytest.mark.parametrize("scale", [1e-80, 1e-90, 1e80, 1e160, 1e-300, 1e300])
def test_review_scales(scale: float) -> None:
    # STAT-1: the review's data; df 18.816512766113657 at scale 1.
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 10)
    b = rng.normal(1.0, 2.0, 15)
    base = welch_t(a, b)
    welch = welch_t(a * scale, b * scale)
    assert base.df == pytest.approx(18.816512766113657, rel=1e-15)
    assert_close("closed_form", welch.df, base.df, what="df")
    assert_close("closed_form", welch.t, base.t, what="t")


def test_variance_of_huge_and_tiny_values() -> None:
    # Standard deviations of 1e150 and 1e-150: the squares of the deviations
    # overflow and underflow, the variances (1e300, 1e-300) do not.
    x = np.array([1.0, 2.0, 4.0, 7.0])
    expected = variance(x)
    for scale in (1e150, 1e-150):
        assert_close("closed_form", variance(x * scale), expected * scale**2, what="v")
    with pytest.raises(ValueError, match="exceeds the largest double"):
        variance(x * 1e160)


def test_welch_t_raises_when_the_mean_difference_overflows() -> None:
    a = np.array([-1.5e308, -1.2e308, -1.0e308])
    b = np.array([1.0e308, 1.2e308, 1.5e308])
    with pytest.raises(ValueError, match="largest double"):
        welch_t(a, b)
    # The scale-free statistics are still available.
    assert hedges_g(a, b).estimate < -5.0
    assert student_t(a, b).t < -10.0


def test_a_group_too_small_for_the_others_is_named() -> None:
    big = np.array([1e300, 2e300, 3e300])
    small = np.array([1e-10, 2e-10, 3e-10])
    with pytest.raises(ValueError, match="group 2 varies, but by less than double"):
        welch_anova([big, small, big * 2])
    with pytest.raises(ValueError, match="group 2 varies, but by less than double"):
        games_howell([big, small, big * 2])
    with pytest.raises(ValueError, match="group 2 is constant"):
        welch_anova([big, np.ones(3), big * 2])


def test_hedges_g_at_one_degree_of_freedom_is_positive_zero() -> None:
    # STAT-3: one constant group and one group of two values give Welch df = 1,
    # J(1) = 0, and g = 0 with the interval [0, 0]: +0.0, never -0.0.
    for a, b in (([5.0, 5.0, 5.0], [1.0, 3.0]), ([1.0, 3.0], [5.0, 5.0, 5.0])):
        g = hedges_g(np.array(a), np.array(b))
        assert welch_t(np.array(a), np.array(b)).df == 1.0
        for value in (g.estimate, g.conf_low, g.conf_high):
            assert value == 0.0
            assert math.copysign(1.0, value) == 1.0


@pytest.mark.slow
def test_every_power_of_two_gives_identical_effect_sizes_and_bayes_factors() -> None:
    # The fast test above takes every 25th power; this one all 2,001 (about 10 s).
    base = _costly(A, B, GROUPS)
    for k in ALL_POWERS_OF_TWO:
        a, b, groups = _scaled(lambda x, k=k: np.ldexp(x, k))
        assert _costly(a, b, groups) == base, f"2^{k}"
