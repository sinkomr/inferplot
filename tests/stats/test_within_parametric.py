"""Paired t, Hedges' g of paired samples, the repeated-measures ANOVA with the
Greenhouse-Geisser correction, partial omega squared and the paired t pairs."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats
from tools.reference import stats_reference as ref

from inferplot.stats import (
    f_ncp_bound,
    hedges_j,
    paired_differences_constant,
    paired_hedges_g,
    paired_t,
    paired_t_pairs,
    rm_anova,
    rm_partial_omega_squared,
    t_ncp_interval,
)
from tests.tolerances import assert_close

RNG = np.random.Generator(np.random.PCG64(31))
N = 14
SUBJECT = RNG.normal(0.0, 2.0, N)
A = 10.0 + SUBJECT + RNG.normal(0.0, 1.0, N)
B = 10.6 + SUBJECT + RNG.normal(0.0, 1.0, N)
C = 9.8 + SUBJECT + RNG.normal(0.0, 1.5, N)
D = 11.0 + SUBJECT + RNG.normal(0.0, 0.7, N)
TABLE = [A, B, C, D]


def _rm_reference(data: np.ndarray) -> tuple[float, float, float, float, float]:
    """(F, SS_c, SS_s, SS_e, epsilon) by the textbook formulas in NumPy."""
    n, k = data.shape
    grand = data.mean()
    ss_c = n * float(((data.mean(axis=0) - grand) ** 2).sum())
    ss_s = k * float(((data.mean(axis=1) - grand) ** 2).sum())
    ss_e = float(((data - grand) ** 2).sum()) - ss_c - ss_s
    f = (ss_c / (k - 1)) / (ss_e / ((k - 1) * (n - 1)))
    return f, ss_c, ss_s, ss_e, ref.rm_anova_gg(data)["epsilon"]


# --------------------------------------------------------------------------- #
# paired t
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side", ["two-sided", "greater", "less"])
def test_paired_t_matches_scipy(side: str) -> None:
    result = paired_t(A, B, alternative=side)  # type: ignore[arg-type]
    expected = stats.ttest_rel(A, B, alternative=side)  # type: ignore[arg-type]
    assert result.alternative == side
    assert result.df == N - 1
    assert result.n == N
    assert_close("closed_form", result.t, float(expected.statistic), what="t")
    assert_close("closed_form", result.p_value, float(expected.pvalue), what="p")


def test_paired_t_is_the_one_sample_t_of_the_differences() -> None:
    result = paired_t(A, B)
    one = stats.ttest_1samp(A - B, 0.0)
    assert_close("closed_form", result.t, float(one.statistic), what="t")


def test_sign_order_and_subject_order() -> None:
    forward, backward = paired_t(A, B), paired_t(B, A)
    assert backward.t == -forward.t
    assert backward.p_value == forward.p_value
    order = RNG.permutation(N)
    shuffled = paired_t(A[order], B[order])
    assert shuffled.t == pytest.approx(forward.t, rel=1e-15)
    # Pairing matters: shuffling one sample only changes the result.
    assert paired_t(A[order], B).t != pytest.approx(forward.t, rel=1e-3)


def test_one_sided_p_values_add_up() -> None:
    greater = paired_t(A, B, alternative="greater").p_value
    less = paired_t(A, B, alternative="less").p_value
    assert greater + less == pytest.approx(1.0, rel=1e-15)
    assert paired_t(A, B).p_value == pytest.approx(2 * min(greater, less), rel=1e-15)


@pytest.mark.parametrize("power", [-1000, -500, -60, 0, 60, 500, 900])
def test_powers_of_two_give_identical_bits(power: int) -> None:
    factor = 2.0**power
    base = paired_t(A, B)
    scaled = paired_t(A * factor, B * factor)
    assert (scaled.t, scaled.p_value) == (base.t, base.p_value)
    g = paired_hedges_g(A, B)
    assert paired_hedges_g(A * factor, B * factor) == g


@pytest.mark.parametrize("factor", [1e-150, 1e150, 1e-300, 1e300])
def test_extreme_scales(factor: float) -> None:
    base = paired_t(A, B)
    scaled = paired_t(A * factor, B * factor)
    assert scaled.t == pytest.approx(base.t, rel=1e-13)
    assert rm_anova([c * factor for c in TABLE]).f == pytest.approx(
        rm_anova(TABLE).f, rel=1e-12
    )


def test_exact_differences_on_a_large_offset() -> None:
    # Values near a timestamp, differing in their last bits: the paired t from
    # exact rational arithmetic, against R's route of rounded differences.
    offset = 1.7e9
    a = offset + np.array([0.25, 0.5, 0.75, 1.0, 1.25, 0.0])
    b = a + np.array([1, 2, 1, 3, 2, 2]) * math.ulp(offset)
    exact_d = [
        Fraction(float(x)) - Fraction(float(y)) for x, y in zip(a, b, strict=True)
    ]
    n = len(exact_d)
    mean = sum(exact_d, Fraction(0)) / n
    var = sum(((d - mean) ** 2 for d in exact_d), Fraction(0)) / (n - 1)
    exact_t = float(mean) / math.sqrt(float(var) / n)
    assert paired_t(a, b).t == pytest.approx(exact_t, rel=1e-15)


def test_two_and_three_subjects_and_ties() -> None:
    two = paired_t([1.0, 2.0], [3.0, 5.0])
    expected = stats.ttest_rel([1.0, 2.0], [3.0, 5.0])
    assert two.df == 1.0
    assert_close("closed_form", two.t, float(expected.statistic), what="t")
    three = paired_t([1.0, 2.0, 2.0], [2.0, 2.0, 4.0])
    expected = stats.ttest_rel([1.0, 2.0, 2.0], [2.0, 2.0, 4.0])
    assert_close("closed_form", three.t, float(expected.statistic), what="t")
    # Hedges' J(1) is 0: g and both finite bounds are +0.0 with two subjects.
    g = paired_hedges_g([1.0, 2.0], [3.0, 5.0])
    assert (g.estimate, g.conf_low, g.conf_high) == (0.0, 0.0, 0.0)
    assert math.copysign(1.0, g.estimate) == 1.0


def test_constant_and_zero_differences_raise() -> None:
    a = np.array([1.0, 2.0, 3.5, 7.0])
    for b in (a, a + 0.5, a - 3.0):
        assert paired_differences_constant(a, b)
        with pytest.raises(ValueError, match=r"every difference .* is the same"):
            paired_t(a, b)
        with pytest.raises(ValueError, match=r"every difference .* is the same"):
            paired_hedges_g(a, b)
    assert not paired_differences_constant(a, a + np.array([0.0, 0.0, 0.0, 1e-12]))
    # 0.1 + 0.2 - 0.3 is not 0.1 - 0.0 exactly: differences that round alike but
    # differ exactly are not constant.
    x = np.array([0.1 + 0.2, 0.1, 5.0])
    y = np.array([0.3, 0.0, 4.9])
    assert not paired_differences_constant(x, y)


def test_differences_below_double_precision_raise() -> None:
    # One condition constant at 1, the other varying by 1e-300 near 0: the exact
    # differences are not constant, but their spread is below what doubles
    # resolve next to 1.
    a = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0, 1e-300])
    assert not paired_differences_constant(a, b)
    with pytest.raises(ValueError, match="vary by less than double precision"):
        paired_t(a, b)


def test_input_checks() -> None:
    with pytest.raises(ValueError, match="same number of values"):
        paired_t([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least 2"):
        paired_t([1.0], [2.0])
    with pytest.raises(ValueError, match="finite"):
        paired_t([1.0, math.nan], [2.0, 3.0])
    with pytest.raises(ValueError, match="two-sided"):
        paired_t(A, B, alternative="two.sided")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Hedges' g of paired samples
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("level", [0.9, 0.95, 0.99])
def test_paired_hedges_g_definition(side: str, level: float) -> None:
    d = A - B
    j = ref.hedges_j(N - 1.0)
    g = paired_hedges_g(A, B, conf_level=level, alternative=side)  # type: ignore[arg-type]
    assert_close("closed_form", g.estimate, d.mean() / d.std(ddof=1) * j, what="g")
    t = paired_t(A, B).t
    low, high = t_ncp_interval(t, N - 1.0, conf_level=level, alternative=side)  # type: ignore[arg-type]
    scale = hedges_j(N - 1.0) / math.sqrt(N)
    assert g.conf_low == (low * scale if math.isfinite(low) else low)
    assert g.conf_high == (high * scale if math.isfinite(high) else high)
    assert g.sidedness == side
    assert g.conf_level == level
    assert g.conf_low < g.estimate < g.conf_high


def test_paired_hedges_g_is_antisymmetric() -> None:
    forward, backward = paired_hedges_g(A, B), paired_hedges_g(B, A)
    assert backward.estimate == -forward.estimate
    assert backward.conf_low == pytest.approx(-forward.conf_high, rel=1e-12)


# --------------------------------------------------------------------------- #
# repeated-measures ANOVA
# --------------------------------------------------------------------------- #


def test_rm_anova_matches_the_textbook_decomposition() -> None:
    data = np.column_stack(TABLE)
    f, _, _, _, epsilon = _rm_reference(data)
    result = rm_anova(TABLE)
    n, k = data.shape
    assert (result.n, result.k) == (n, k)
    assert (result.df1, result.df2) == (k - 1.0, (k - 1.0) * (n - 1.0))
    assert_close("closed_form", result.f, f, what="F")
    assert_close("closed_form", result.epsilon, epsilon, what="epsilon")
    assert result.df1_corrected == result.df1 * result.epsilon
    assert result.df2_corrected == result.df2 * result.epsilon
    expected_p = float(stats.f.sf(f, result.df1_corrected, result.df2_corrected))
    assert_close("closed_form", result.p_value, expected_p, what="GG p")
    plain = float(stats.f.sf(f, result.df1, result.df2))
    assert_close("closed_form", result.p_uncorrected, plain, what="p")


def test_rm_anova_of_two_conditions_is_the_paired_t_squared() -> None:
    result = rm_anova([A, B])
    t = paired_t(A, B)
    assert result.epsilon == 1.0  # one contrast: sphericity holds trivially
    assert result.f == pytest.approx(t.t**2, rel=1e-14)
    assert result.p_value == pytest.approx(t.p_value, rel=1e-12)


def test_rm_anova_invariances() -> None:
    base = rm_anova(TABLE)
    order = RNG.permutation(N)
    shuffled = rm_anova([c[order] for c in TABLE])
    assert shuffled.f == pytest.approx(base.f, rel=1e-14)
    assert shuffled.epsilon == pytest.approx(base.epsilon, rel=1e-14)
    swapped = rm_anova([TABLE[2], TABLE[0], TABLE[3], TABLE[1]])
    assert swapped.f == pytest.approx(base.f, rel=1e-14)
    assert swapped.epsilon == pytest.approx(base.epsilon, rel=1e-14)
    # A constant added to a subject (row) or a condition (column) changes nothing
    # but the condition effect.
    shifted = rm_anova([c + 100.0 * SUBJECT for c in TABLE])
    assert shifted.f == pytest.approx(base.f, rel=1e-12)
    assert shifted.epsilon == pytest.approx(base.epsilon, rel=1e-12)


def test_residuals_stay_accurate_under_large_subject_effects() -> None:
    # Subject effects 1e9 times the residuals: the error SS is computed from the
    # differences between conditions, so F keeps its accuracy; the same data
    # without the subject effects give the same F.
    big = [c - SUBJECT + 1e9 * SUBJECT for c in TABLE]
    plain = [c - SUBJECT for c in TABLE]
    assert rm_anova(big).f == pytest.approx(rm_anova(plain).f, rel=1e-6)
    # In exact arithmetic the decomposition is the same: check against Fractions.
    exact = [[Fraction(float(v)) for v in c] for c in big]
    n, k = N, len(exact)
    row = [sum((exact[j][i] for j in range(k)), Fraction(0)) / k for i in range(n)]
    col = [sum(exact[j], Fraction(0)) / n for j in range(k)]
    grand = sum(col, Fraction(0)) / k
    ss_e = sum(
        (
            (exact[j][i] - row[i] - col[j] + grand) ** 2
            for j in range(k)
            for i in range(n)
        ),
        Fraction(0),
    )
    ss_c = n * sum(((m - grand) ** 2 for m in col), Fraction(0))
    f = float((ss_c / (k - 1)) / (ss_e / ((k - 1) * (n - 1))))
    assert rm_anova(big).f == pytest.approx(f, rel=1e-13)


def test_epsilon_bounds_and_singular_covariance() -> None:
    # Two subjects and four conditions: the residual covariance has rank 1, so
    # epsilon is at its lower bound 1 / (k - 1).
    two = rm_anova(
        [
            np.array([1.0, 2.0]),
            np.array([3.0, 1.0]),
            np.array([2.0, 5.0]),
            np.array([0.0, 0.5]),
        ]
    )
    assert two.epsilon == pytest.approx(1.0 / 3.0, rel=1e-14)
    assert 1.0 / 3.0 <= two.epsilon <= 1.0
    for seed in range(20):
        rng = np.random.Generator(np.random.PCG64(seed))
        n, k = int(rng.integers(3, 12)), int(rng.integers(3, 7))
        result = rm_anova([rng.normal(0.0, 1.0, n) for _ in range(k)])
        assert 1.0 / (k - 1) <= result.epsilon <= 1.0


def test_rm_anova_refuses_what_has_no_error_variance() -> None:
    base = np.array([1.0, 4.0, 2.5, 7.0])
    additive = [base, base + 1.0, base - 2.0]  # each subject: means plus a constant
    with pytest.raises(ValueError, match="differ by the same amounts"):
        rm_anova(additive)
    with pytest.raises(ValueError, match="differ by the same amounts"):
        rm_partial_omega_squared(additive)
    ones = np.ones(4)
    tiny = [ones, 2.0 * ones, np.array([0.0, 0.0, 0.0, 1e-300])]
    with pytest.raises(ValueError, match="smaller than double precision"):
        rm_anova(tiny)
    with pytest.raises(ValueError, match="one value per subject"):
        rm_anova([base, base[:3]])
    with pytest.raises(ValueError, match="at least two conditions"):
        rm_anova([base])


def test_no_condition_effect_gives_f_zero() -> None:
    rng = np.random.Generator(np.random.PCG64(5))
    raw = rng.normal(0.0, 1.0, (8, 3))
    centred = raw - raw.mean(axis=0)  # equal condition means (up to rounding)
    result = rm_anova([centred[:, j] for j in range(3)])
    assert result.f < 1e-25
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# partial omega squared
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("level", [0.9, 0.95])
def test_partial_omega_squared_definition(level: float) -> None:
    data = np.column_stack(TABLE)
    n, k = data.shape
    _, ss_c, ss_s, ss_e, _ = _rm_reference(data)
    df_c, df_e = k - 1.0, (k - 1.0) * (n - 1.0)
    ms_c, ms_e, ms_s = ss_c / df_c, ss_e / df_e, ss_s / (n - 1.0)
    omega = df_c * (ms_c - ms_e) / (ss_c + ss_e + ss_s + ms_s)
    result = rm_partial_omega_squared(TABLE, conf_level=level)
    assert omega > 0
    assert_close("closed_form", result.estimate, omega, what="omega")
    f_w = omega * df_e / ((1.0 - omega) * df_c)
    ncp = f_ncp_bound(f_w, df_c, df_e, prob=level)
    assert_close("closed_form", result.conf_low, ncp / (ncp + df_e), what="low")
    assert (result.conf_high, result.sidedness, result.conf_level) == (
        1.0,
        "greater",
        level,
    )


def test_partial_omega_squared_is_floored_at_zero() -> None:
    rng = np.random.Generator(np.random.PCG64(8))
    raw = rng.normal(0.0, 1.0, (10, 3))
    centred = raw - raw.mean(axis=0)
    result = rm_partial_omega_squared([centred[:, j] for j in range(3)])
    assert (result.estimate, result.conf_low, result.conf_high) == (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# pairs
# --------------------------------------------------------------------------- #


def test_paired_t_pairs_match_scipy_and_p_adjust() -> None:
    pairs = paired_t_pairs(TABLE, p_adjust_method="bonferroni")
    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert [(p.i, p.j) for p in pairs] == expected_pairs
    for pair in pairs:
        expected = stats.ttest_rel(TABLE[pair.i], TABLE[pair.j])
        assert_close("closed_form", pair.t, float(expected.statistic), what="t")
        assert_close("closed_form", pair.p_value, float(expected.pvalue), what="p")
        assert pair.df == N - 1.0
        assert pair.p_adjusted == min(1.0, 6 * pair.p_value)
    with pytest.raises(ValueError, match="method"):
        paired_t_pairs(TABLE, p_adjust_method="Holm")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two"):
        paired_t_pairs([A])
