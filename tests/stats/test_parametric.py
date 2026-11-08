"""Welch's t-test, Student's t and Welch's ANOVA."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from tools.reference import stats_reference as ref

from inferplot.stats import student_t, welch_anova, welch_t
from tests.tolerances import assert_close

RNG = np.random.Generator(np.random.PCG64(11))
A = RNG.normal(0.0, 1.0, 17)
B = RNG.normal(0.6, 2.0, 25)


@pytest.mark.parametrize("side", ["two-sided", "greater", "less"])
def test_welch_t_matches_scipy(side: str) -> None:
    result = welch_t(A, B, alternative=side)  # type: ignore[arg-type]
    expected = stats.ttest_ind(A, B, equal_var=False, alternative=side)  # type: ignore[arg-type]
    assert result.alternative == side
    assert_close("closed_form", result.t, float(expected.statistic), what="t")
    assert_close("closed_form", result.df, float(expected.df), what="df")
    assert_close("closed_form", result.p_value, float(expected.pvalue), what="p")
    assert_close("closed_form", result.mean_difference, A.mean() - B.mean(), what="d")
    assert result.se == pytest.approx(result.mean_difference / result.t, rel=1e-15)


def test_one_sided_p_values_add_up() -> None:
    greater = welch_t(A, B, alternative="greater").p_value
    less = welch_t(A, B, alternative="less").p_value
    two = welch_t(A, B).p_value
    assert greater + less == pytest.approx(1.0, rel=1e-15)
    assert two == pytest.approx(2.0 * min(greater, less), rel=1e-15)


def test_welch_t_small_and_degenerate_samples() -> None:
    # n = 2 per group
    result = welch_t([1.0, 2.0], [3.0, 5.0])
    expected = stats.ttest_ind([1.0, 2.0], [3.0, 5.0], equal_var=False)
    assert_close("closed_form", result.t, float(expected.statistic), what="t")
    # One constant group is allowed: df is the other group's n - 1.
    result = welch_t([4.0, 4.0, 4.0], [1.0, 2.0, 4.0, 7.0])
    assert result.df == pytest.approx(3.0, rel=1e-15)
    with pytest.raises(ValueError, match="both samples are constant"):
        welch_t([1.0, 1.0], [2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="at least 2"):
        welch_t([1.0], [2.0, 3.0])


def test_welch_t_huge_t_keeps_a_tiny_positive_p() -> None:
    a = np.linspace(0.0, 1.0, 50)
    result = welch_t(a, a + 100.0)
    assert result.t < -500.0
    assert 0.0 <= result.p_value < 1e-60
    tiny = welch_t(a, a + 3.0)
    assert 0.0 < tiny.p_value < 1e-40  # upper tails, never 1 - cdf
    assert welch_t(a, a + 100.0, alternative="greater").p_value == 1.0


def test_alternative_spelling() -> None:
    with pytest.raises(ValueError, match="two-sided"):
        welch_t(A, B, alternative="two.sided")  # type: ignore[arg-type]


def test_student_t_matches_scipy() -> None:
    result = student_t(A, B)
    expected = stats.ttest_ind(A, B, equal_var=True)
    assert_close("closed_form", result.t, float(expected.statistic), what="t")
    assert result.df == A.size + B.size - 2
    with pytest.raises(ValueError, match="constant"):
        student_t([1.0, 1.0], [1.0, 1.0])


def test_welch_anova_matches_reference() -> None:
    groups = [RNG.normal(m, s, n) for m, s, n in [(0, 1, 9), (1, 3, 14), (0.4, 0.5, 6)]]
    result = welch_anova(groups)
    f, df1, df2, p = ref.welch_anova(groups)
    assert_close("closed_form", result.f, f, what="F")
    assert result.df1 == df1 == 2.0
    assert_close("closed_form", result.df2, df2, what="df2")
    assert_close("closed_form", result.p_value, p, what="p")


def test_welch_anova_two_groups_is_welch_t_squared() -> None:
    result = welch_anova([A, B])
    t = welch_t(A, B)
    assert result.f == pytest.approx(t.t**2, rel=1e-13)
    assert result.df2 == pytest.approx(t.df, rel=1e-13)
    assert result.p_value == pytest.approx(t.p_value, rel=1e-12)


def test_welch_anova_huge_f_and_validation() -> None:
    groups = [np.linspace(0, 1, 20) + shift for shift in (0.0, 50.0, 100.0)]
    result = welch_anova(groups)
    assert result.f > 1e4
    assert 0.0 <= result.p_value < 1e-30
    with pytest.raises(ValueError, match="constant"):
        welch_anova([[1.0, 1.0], [1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="at least two groups"):
        welch_anova([A])
    with pytest.raises(ValueError, match="at least 2"):
        welch_anova([A, [1.0]])


def test_results_are_plain_python_numbers() -> None:
    for result in (welch_t(A, B), student_t(A, B), welch_anova([A, B, A + 1.0])):
        for name in result.__slots__:  # type: ignore[attr-defined]
            value = getattr(result, name)
            assert type(value) in (float, str), (name, type(value))
    assert math.isfinite(welch_t(A, B).p_value)
