"""Unit tests of inferplot.stats.within_nonparametric: definitions, sign
conventions, invariances, degenerate inputs and an independent SciPy reference."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import stats as scipy_stats

from inferplot.stats import (
    durbin_conover,
    friedman,
    p_adjust,
    paired_rank_biserial,
    signed_rank,
)
from tests.tolerances import assert_close

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

RNG = np.random.default_rng(20261110)
A = np.round(RNG.normal(10.0, 2.0, 25), 1)
B = np.round(A + RNG.normal(0.4, 1.0, 25), 1)
SIDES = ("two-sided", "less", "greater")
# Twelve subjects, four conditions, rounded to whole numbers: ties within subjects.
TABLE = np.round(RNG.normal([5.0, 5.5, 6.5, 6.0], 1.2, (12, 4)))
COLUMNS = [TABLE[:, j].copy() for j in range(4)]


# --------------------------------------------------------------------------- #
# signed-rank test
# --------------------------------------------------------------------------- #


def test_signed_rank_by_hand() -> None:
    # d = a - b = [1, -2, 3, 0, 3]: zero dropped; |d| ranks 1, 2, 3.5, 3.5.
    a = np.array([2.0, 1.0, 5.0, 4.0, 6.0])
    b = np.array([1.0, 3.0, 2.0, 4.0, 3.0])
    result = signed_rank(a, b)
    assert result.statistic == 1.0 + 3.5 + 3.5
    assert (result.n, result.n_nonzero) == (5, 4)
    # m = 8 - 4 * 5 / 4 = 3; sigma^2 = 4 * 5 * 9 / 24 - (8 - 2) / 48 = 7.375.
    z = (3.0 - 0.5) / math.sqrt(7.375)
    assert result.z == pytest.approx(z, rel=1e-15)
    assert result.p_value == pytest.approx(2.0 * scipy_stats.norm.sf(z), rel=1e-14)


@pytest.mark.parametrize("side", SIDES)
def test_signed_rank_matches_scipy(side: Any) -> None:
    ours = signed_rank(A, B, alternative=side)
    theirs = scipy_stats.wilcoxon(
        A, B, zero_method="wilcox", correction=True, alternative=side, method="approx"
    )
    # SciPy reports min(V, n'(n' + 1)/2 - V) when two-sided; the p is the same.
    assert_close("closed_form", ours.p_value, float(theirs.pvalue), what="p")
    if side != "two-sided":
        assert ours.statistic == float(theirs.statistic)


def test_continuity_shift_follows_the_alternative() -> None:
    # One nonzero difference of -1 among four pairs (probe-signrank one_nonzero):
    # V = 0, m = -0.5, sigma = 0.5.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, 4.0, 4.0])
    assert signed_rank(a, b).p_value == 1.0  # (m - (-0.5)) / sigma = 0
    assert signed_rank(a, b, alternative="less").p_value == 0.5
    greater = signed_rank(a, b, alternative="greater")
    assert greater.z == -2.0
    assert greater.p_value == pytest.approx(scipy_stats.norm.sf(-2.0), rel=1e-15)


def test_swapping_the_samples_mirrors_the_test() -> None:
    forward = signed_rank(A, B)
    backward = signed_rank(B, A)
    n1 = forward.n_nonzero
    assert forward.statistic + backward.statistic == n1 * (n1 + 1) / 2
    assert forward.z == -backward.z
    assert forward.p_value == backward.p_value
    assert (
        signed_rank(A, B, alternative="less").p_value
        == signed_rank(B, A, alternative="greater").p_value
    )


def test_subject_order_does_not_matter() -> None:
    order = RNG.permutation(A.size)
    for side in SIDES:
        assert signed_rank(A[order], B[order], alternative=side) == signed_rank(
            A, B, alternative=side
        )


@pytest.mark.parametrize("factor", [2.0**-500, 2.0**600, -1.0])
def test_scale_invariance(factor: float) -> None:
    # Powers of two are exact, so V, z and p are the same bit for bit; -1 mirrors
    # V.
    base = signed_rank(A, B)
    scaled = signed_rank(A * factor, B * factor)
    if factor > 0:
        assert scaled == base
    else:
        assert scaled.p_value == base.p_value
        assert scaled.z == -base.z


@pytest.mark.parametrize("factor", [1e-150, 1e150, 1e-300, 1e300])
@pytest.mark.parametrize("side", SIDES)
def test_extreme_scales(factor: float, side: Any) -> None:
    # Multiplying by 1e+-150 rounds every value, which can break ties among the
    # one-decimal differences (R does the same with these data): the test is the
    # one of the scaled data, as SciPy computes it, and no square under- or
    # overflows.
    a, b = A * factor, B * factor
    ours = signed_rank(a, b, alternative=side)
    theirs = scipy_stats.wilcoxon(
        a, b, zero_method="wilcox", correction=True, alternative=side, method="approx"
    )
    assert_close("closed_form", ours.p_value, float(theirs.pvalue), what="p")
    es = paired_rank_biserial(a, b, alternative=side)
    assert math.isfinite(es.estimate)
    assert -1.0 <= es.conf_low <= es.estimate <= es.conf_high <= 1.0


def test_overflowing_differences() -> None:
    big = np.array([1.5e308, -1.6e308, 1.7e308, 1e308])
    small = np.array([-1.0e308, 1.2e308, -1.1e308, 1e308])
    result = signed_rank(big, small)
    halved = signed_rank(big / 2, small / 2)
    assert result == halved
    # |d| = 2.5e308 (rank 1, positive), 2.8e308 twice (ranks 2.5, one positive).
    assert result.statistic == 1.0 + 2.5
    assert result.n_nonzero == 3


def test_ties_reduce_the_variance() -> None:
    # Every |d| tied: sigma^2 = n(n + 1)(2n + 1)/24 - (n^3 - n)/48.
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    result = signed_rank(a, b)
    n = 6
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - (n**3 - n) / 48)
    assert result.statistic == 4 * 3.5
    expected = (4 * 3.5 - n * (n + 1) / 4 - 0.5) / sigma
    assert result.z == pytest.approx(expected, rel=1e-15)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_small_samples(n: int) -> None:
    a = np.arange(1.0, n + 1)
    result = signed_rank(a + 1.0, a)
    assert result.statistic == n * (n + 1) / 2
    assert 0.0 < result.p_value <= 1.0


def test_all_zero_differences_raise() -> None:
    a = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="every one of the 3 paired differences is 0"):
        signed_rank(a, a.copy())
    with pytest.raises(ValueError, match="every one of the 3 paired differences is 0"):
        paired_rank_biserial(a, a.copy())


def test_signed_rank_input_errors() -> None:
    with pytest.raises(ValueError, match="same number of values"):
        signed_rank([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        signed_rank([1.0, math.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match="alternative"):
        signed_rank([1.0, 2.0], [2.0, 1.0], alternative="two.sided")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# matched-pairs rank-biserial r
# --------------------------------------------------------------------------- #


def test_rank_biserial_by_hand() -> None:
    a = np.array([2.0, 1.0, 5.0, 4.0, 6.0])
    b = np.array([1.0, 3.0, 2.0, 4.0, 3.0])
    es = paired_rank_biserial(a, b)
    # W+ = 8, W- = 2 over the nonzero differences.
    assert es.estimate == (8 - 2) / 10
    # SE with n = 5 pairs, the zero difference included.
    se = math.sqrt(5 * 6 * 11 / 6) / (5 * 6 / 2)
    q = scipy_stats.norm.ppf(0.975)
    assert es.conf_low == pytest.approx(math.tanh(math.atanh(0.6) - q * se), rel=1e-14)
    assert es.conf_high == pytest.approx(math.tanh(math.atanh(0.6) + q * se), rel=1e-14)
    assert (es.conf_level, es.sidedness) == (0.95, "two-sided")


def test_rank_biserial_is_v_rescaled() -> None:
    result = signed_rank(A, B)
    n1 = result.n_nonzero
    total = n1 * (n1 + 1) / 2
    es = paired_rank_biserial(A, B)
    assert es.estimate == pytest.approx(2 * result.statistic / total - 1, rel=1e-15)
    assert paired_rank_biserial(B, A).estimate == -es.estimate


def test_rank_biserial_one_sided() -> None:
    two = paired_rank_biserial(A, B, conf_level=0.9)
    greater = paired_rank_biserial(A, B, alternative="greater", conf_level=0.95)
    less = paired_rank_biserial(A, B, alternative="less", conf_level=0.95)
    # A one-sided 95% bound is the matching bound of the two-sided 90% interval.
    # (1 - 0.9) / 2 and 1 - 0.95 differ in their last bit.
    assert greater.conf_low == pytest.approx(two.conf_low, rel=1e-15)
    assert greater.conf_high == 1.0
    assert less.conf_high == pytest.approx(two.conf_high, rel=1e-15)
    assert less.conf_low == -1.0


@pytest.mark.parametrize("side", SIDES)
def test_rank_biserial_at_its_bound(side: Any) -> None:
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 1.0, 1.0, 5.0])  # one zero, three positive differences
    es = paired_rank_biserial(a, b, alternative=side)
    assert es.estimate == 1.0
    expected = {"two-sided": (1.0, 1.0), "greater": (1.0, 1.0), "less": (-1.0, 1.0)}
    assert (es.conf_low, es.conf_high) == expected[side]
    mirrored = paired_rank_biserial(b, a, alternative=side)
    assert mirrored.estimate == -1.0


def test_rank_biserial_conf_level_errors() -> None:
    with pytest.raises(ValueError, match="conf_level"):
        paired_rank_biserial(A, B, conf_level=1.0)


@pytest.mark.parametrize("factor", [2.0**-700, 2.0**700])
def test_rank_biserial_scale_invariance(factor: float) -> None:
    assert paired_rank_biserial(A * factor, B * factor) == paired_rank_biserial(A, B)


# --------------------------------------------------------------------------- #
# Friedman's test and Durbin-Conover pairs
# --------------------------------------------------------------------------- #


def _within_ranks(columns: list[FloatArray]) -> FloatArray:
    return np.apply_along_axis(scipy_stats.rankdata, 1, np.column_stack(columns))


def test_friedman_by_hand() -> None:
    # Three subjects, three conditions; subject 3 ties conditions 2 and 3.
    columns = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 3.0, 5.0]),
        np.array([3.0, 1.0, 5.0]),
    ]
    # Ranks: (1, 2, 3), (2, 3, 1), (1, 2.5, 2.5); R = (4, 7.5, 6.5).
    # chi^2 = 12 sum (R_j - 6)^2 / (3 * 3 * 4 - 6 / 2) = 12 * 6.5 / 33.
    result = friedman(columns)
    assert result.statistic == pytest.approx(12 * 6.5 / 33, rel=1e-15)
    assert (result.df, result.n, result.k) == (2.0, 3, 3)
    assert result.p_value == pytest.approx(
        scipy_stats.chi2.sf(12 * 6.5 / 33, 2), rel=1e-14
    )


def test_friedman_matches_scipy() -> None:
    ours = friedman(COLUMNS)
    theirs = scipy_stats.friedmanchisquare(*COLUMNS)
    assert_close("closed_form", ours.statistic, float(theirs.statistic), what="chi2")
    assert_close("closed_form", ours.p_value, float(theirs.pvalue), what="p")


def test_friedman_invariances() -> None:
    base = friedman(COLUMNS)
    order = RNG.permutation(TABLE.shape[0])
    assert friedman([c[order] for c in COLUMNS]) == base
    assert friedman(COLUMNS[::-1]) == base
    # Ranks within subjects only: any increasing transform of each subject's
    # values, and any scale, leaves the test unchanged.
    assert friedman([np.exp(c) for c in COLUMNS]) == base
    for factor in (1e-300, 1e-150, 1e150, 1e300):
        assert friedman([c * factor for c in COLUMNS]) == base


def test_tied_subjects_add_nothing() -> None:
    tied = [np.append(c, 7.0) for c in COLUMNS]
    base, more = friedman(COLUMNS), friedman(tied)
    # A fully tied subject adds (k + 1) / 2 to every rank sum and to their null
    # mean, and k^3 - k to both terms of the denominator: chi^2 is unchanged.
    assert more.statistic == base.statistic
    assert more.n == base.n + 1


def test_friedman_all_tied_raises() -> None:
    same = [np.array([1.0, 2.0, 3.0])] * 3
    with pytest.raises(ValueError, match="same value in all 3 conditions"):
        friedman(same)
    with pytest.raises(ValueError, match="same value in all 3 conditions"):
        durbin_conover(same)


def test_friedman_input_errors() -> None:
    with pytest.raises(ValueError, match="at least two conditions"):
        friedman([np.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="same number of values"):
        friedman([np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])])
    with pytest.raises(ValueError, match="at least 2 value"):
        friedman([np.array([1.0]), np.array([2.0])])


def _durbin_reference(columns: list[FloatArray]) -> list[tuple[float, float]]:
    """Conover's (1999, section 5.8) form with the Friedman statistic T1."""
    ranks = _within_ranks(columns)
    b, k = ranks.shape
    sums = ranks.sum(axis=0)
    a = float((ranks**2).sum())
    c = b * k * (k + 1) ** 2 / 4
    t1 = (k - 1) * float(((sums - b * (k + 1) / 2) ** 2).sum()) / (a - c)
    df = (b - 1) * (k - 1)
    se = math.sqrt(2 * b * (a - c) / df * (1 - t1 / (b * (k - 1))))
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            t = abs(float(sums[i] - sums[j])) / se
            out.append((t, float(2 * scipy_stats.t.sf(t, df))))
    return out


def test_durbin_conover_matches_conover() -> None:
    pairs = durbin_conover(COLUMNS, p_adjust_method="none")
    reference = _durbin_reference(COLUMNS)
    assert [(p.i, p.j) for p in pairs] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    for pair, (t, p) in zip(pairs, reference, strict=True):
        assert pair.df == 33.0
        assert_close("closed_form", pair.statistic, t, what="t")
        assert_close("closed_form", pair.p_value, p, what="p")
        assert pair.p_adjusted == pair.p_value


@pytest.mark.parametrize("method", ["holm", "BH", "bonferroni", "hommel", "BY"])
def test_durbin_conover_adjusts_over_all_pairs(method: Any) -> None:
    pairs = durbin_conover(COLUMNS, p_adjust_method=method)
    raw = [p.p_value for p in pairs]
    assert [p.p_adjusted for p in pairs] == list(p_adjust(raw, method))


def test_durbin_conover_is_unsigned_and_order_free() -> None:
    forward = durbin_conover(COLUMNS)
    backward = durbin_conover(COLUMNS[::-1])
    k = len(COLUMNS)
    by_pair = {(p.i, p.j): p for p in backward}
    for pair in forward:
        mirrored = by_pair[(k - 1 - pair.j, k - 1 - pair.i)]
        assert mirrored.statistic == pair.statistic >= 0.0
        assert mirrored.p_value == pair.p_value
    order = RNG.permutation(TABLE.shape[0])
    assert durbin_conover([c[order] for c in COLUMNS]) == forward


def test_identical_rankings_give_infinite_t() -> None:
    # Every subject ranks the conditions a < b < c (edge_within_small): no error
    # variance, t = inf and p = 0 as R reports them.
    columns = [
        np.array([1.0, 5.0, 2.0]),
        np.array([2.0, 6.0, 3.5]),
        np.array([3.0, 7.0, 9.0]),
    ]
    pairs = durbin_conover(columns)
    assert [(p.statistic, p.p_value, p.p_adjusted) for p in pairs] == [
        (math.inf, 0.0, 0.0)
    ] * 3
    assert friedman(columns).statistic == 6.0  # n (k - 1) W with W = 1


def test_identical_rankings_with_a_tied_pair_raise() -> None:
    # Every subject ranks a < b = c: the pair (b, c) has t = 0/0.
    columns = [np.array([1.0, 5.0]), np.array([2.0, 6.0]), np.array([2.0, 6.0])]
    with pytest.raises(ValueError, match="conditions 2 and 3 share their rank"):
        durbin_conover(columns)


def test_two_subjects() -> None:
    columns = [np.array([1.0, 2.0]), np.array([3.0, 1.0]), np.array([2.0, 3.0])]
    pairs = durbin_conover(columns)
    assert {p.df for p in pairs} == {2.0}
    assert all(math.isfinite(p.statistic) for p in pairs)
