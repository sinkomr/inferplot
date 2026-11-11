"""Count tables whose statistic is exactly 0 (``inferplot.stats.contingency``).

R's order of floating-point operations (DECISIONS D107) leaves a rounding remainder
where the exact value is 0: [[0, 1], [5, 4]] printed "V = 3.95e-09" although its
bias-corrected phi^2 is 1/9 - 1/9 = 0, and counts that fit ``ratio`` exactly printed
"chi2 = 1.13e-31". Whether a value is exactly 0 is decided on the integers and on
``ratio`` as written, in rational arithmetic, and such a value is reported as 0 with
everything derived from it. Every expected value here comes from exact rational
arithmetic (``fractions.Fraction``), independently of the code under test.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest

from inferplot.stats import (
    chi2_ncp_interval,
    chisq_independence,
    cramers_v,
    goodness_of_fit,
    pairwise_fisher,
    pearsons_c,
    proportion_tests,
)


def _exact_chi2(table: list[list[int]]) -> Fraction:
    rows = [sum(r) for r in table]
    cols = [sum(c) for c in zip(*table, strict=True)]
    n = sum(rows)
    return sum(
        (
            (v - Fraction(rows[i] * cols[j], n)) ** 2 / Fraction(rows[i] * cols[j], n)
            for i, row in enumerate(table)
            for j, v in enumerate(row)
        ),
        Fraction(0),
    )


def _exact_phi2(table: list[list[int]]) -> Fraction:
    """The bias-corrected phi^2 before its clip at 0: chi^2 / n - df / (n - 1)."""
    n = sum(map(sum, table))
    df = (len(table) - 1) * (len(table[0]) - 1)
    return _exact_chi2(table) / n - Fraction(df, n - 1)


def _two_by_two(max_n: int) -> list[list[list[int]]]:
    """Every 2 x 2 table with 4 <= n <= max_n and no empty row or column."""
    out: list[list[list[int]]] = []
    for a in range(max_n + 1):
        for b in range(max_n + 1 - a):
            for c in range(max_n + 1 - a - b):
                for d in range(max_n + 1 - a - b - c):
                    if a + b + c + d >= 4 and 0 not in (a + b, c + d, a + c, b + d):
                        out.append([[a, b], [c, d]])
    return out


def _phi2_is_zero(t: list[list[int]]) -> bool:
    """For 2 x 2: chi^2 = n (ad - bc)^2 / (R1 R2 C1 C2), so phi^2 = 0 exactly when
    (ad - bc)^2 (n - 1) = R1 R2 C1 C2 (integers)."""
    (a, b), (c, d) = t
    n = a + b + c + d
    return (a * d - b * c) ** 2 * (n - 1) == (a + b) * (c + d) * (a + c) * (b + d)


# --------------------------------------------------------------------------- #
# Cramer's V: a corrected phi^2 of exactly 0 (or below)
# --------------------------------------------------------------------------- #


def test_corrected_phi2_of_exactly_zero_gives_v_zero() -> None:
    # Every 2 x 2 table with 4 <= n <= 40 whose corrected phi^2 is exactly 0: 208
    # tables, 32 of which used to report V of about 3e-9 (rounding noise).
    zero = [t for t in _two_by_two(40) if _phi2_is_zero(t)]
    assert len(zero) == 208
    for table in zero:
        assert _exact_phi2(table) == 0
        for writing in (table, [list(c) for c in zip(*table, strict=True)]):
            v = cramers_v(writing)
            assert v.estimate == 0.0, writing


def test_corrected_phi2_zero_keeps_its_interval() -> None:
    # [[0, 1], [5, 4]]: chi^2 = 10/9 on 1 df and n = 10, so phi^2 = 1/9 - 1/9 = 0.
    # The estimate is 0; the interval still inverts chi^2 = 10/9 (not 0).
    table = [[0, 1], [5, 4]]
    assert _exact_phi2(table) == 0
    v = cramers_v(table)
    assert v.estimate == 0.0
    chi2 = chisq_independence(table).statistic
    assert chi2 == pytest.approx(10 / 9, rel=1e-15)
    low, high = chi2_ncp_interval(chi2, 1, conf_level=0.95, alternative="two-sided")
    assert low == 0.0
    tilde = 2 - 1 / 9 - 1
    assert v.conf_low == 0.0
    assert v.conf_high == pytest.approx(
        math.sqrt((high / 10 - 1 / 9) / tilde), rel=1e-12
    )
    assert v.conf_high > 0.9  # printed [0.00, 0.95]


def test_positive_corrected_phi2_is_unchanged() -> None:
    # A corrected phi^2 just above 0 keeps R's order of operations: V > 0 and
    # equal to the formula on the floating-point chi-square.
    checked = 0
    for table in _two_by_two(24):
        phi2 = _exact_phi2(table)
        if not 0 < phi2 < Fraction(1, 100):
            continue
        v = cramers_v(table).estimate
        n = sum(map(sum, table))
        chi2 = chisq_independence(table).statistic
        tilde = 2 - 1 / (n - 1) - 1
        assert v == math.sqrt(max(0.0, chi2 / n - 1 / (n - 1)) / tilde), table
        assert v > 0.0
        checked += 1
    assert checked > 100


def test_pair_v_uses_the_exact_zero() -> None:
    # The pairwise V goes through cramers_v: rows 0 and 2 form [[0, 1], [5, 4]].
    pairs = pairwise_fisher(np.array([[0, 1], [9, 1], [5, 4]]))
    by_rows = {(p.row1, p.row2): p for p in pairs}
    effect = by_rows[(0, 2)].effect
    assert effect is not None
    assert effect.estimate == 0.0


# --------------------------------------------------------------------------- #
# Chi-square of independence: every observed count equals its expected count
# --------------------------------------------------------------------------- #


def test_exactly_independent_tables_give_chi_square_zero() -> None:
    # Outer products a b^T are exactly independent (n x_ij = R_i C_j); in R's
    # order of operations their chi-square came out as 1e-20 or so at counts
    # from about 1e10 (127 of 400 such tables in the review).
    rng = random.Random(7)
    noisy_before = 0
    for _ in range(200):
        scale = 10 ** rng.choice([1, 3, 5, 6, 7])
        a = [rng.randint(1, scale) for _ in range(rng.choice([2, 3, 4]))]
        b = [rng.randint(1, scale) for _ in range(rng.choice([2, 3, 4]))]
        table = [[x * y for y in b] for x in a]
        assert _exact_chi2(table) == 0
        test = chisq_independence(table)
        assert (test.statistic, test.p_value) == (0.0, 1.0), table
        v = cramers_v(table)
        assert (v.estimate, v.conf_low, v.conf_high) == (0.0, 0.0, 0.0), table
        unadjusted = cramers_v(table, adjust=False)
        assert unadjusted.estimate == 0.0
        x = np.array(table, dtype=np.float64)
        e = np.outer(x.sum(axis=1), x.sum(axis=0)) / x.sum()
        noisy_before += math.fsum(((x - e) ** 2 / e).ravel().tolist()) != 0.0
    assert noisy_before > 20  # R's order of operations leaves noise in these


def test_near_independence_keeps_rs_order() -> None:
    # One count off independence: chi-square > 0, as R computes it.
    table = [[30452 * 984788 + 1, 30452 * 519897], [553260 * 984788, 553260 * 519897]]
    exact = _exact_chi2(table)
    assert exact > 0
    test = chisq_independence(table)
    assert test.statistic > 0.0
    assert test.statistic == pytest.approx(float(exact), rel=1e-6)


# --------------------------------------------------------------------------- #
# Goodness of fit: counts that fit ratio exactly, ratio as written
# --------------------------------------------------------------------------- #


def _assert_exact_fit(counts: list[int], ratio: list[float] | None) -> None:
    test = goodness_of_fit(counts, ratio)
    assert (test.statistic, test.p_value) == (0.0, 1.0), (counts, ratio)
    c = pearsons_c(counts, ratio)
    assert (c.estimate, c.conf_low, c.conf_high) == (0.0, 0.0, 0.0), (counts, ratio)


def test_exact_fit_to_decimal_shares() -> None:
    # 8, 2, 7, 8 against 0.32, 0.08, 0.28, 0.32 as written: 25 x 0.32 = 8 and so
    # on. In R's order the doubles left chi2 = 1.13e-31 and C = 6.7e-17.
    _assert_exact_fit([8, 2, 7, 8], [0.32, 0.08, 0.28, 0.32])
    _assert_exact_fit([10, 20, 70], [0.1, 0.2, 0.7])
    _assert_exact_fit([999999, 1], [0.999999, 0.000001])


def test_exact_fits_to_shares_computed_as_fractions() -> None:
    # parts / sum(parts) as Python computes them: the review found 283 of 4,000
    # such exact fits with a reported statistic other than 0.
    rng = random.Random(2)
    noisy_before = 0
    for _ in range(1500):
        k = rng.choice([2, 3, 4, 5, 8, 12])
        parts = [rng.randint(1, 12) for _ in range(k)]
        s = sum(parts)
        m = rng.choice([1, 2, 3, 7, 10, 1000])
        ratio = [p / s for p in parts]
        if abs(math.fsum(ratio) - 1.0) > 1e-8:
            continue
        counts = [p * m for p in parts]
        test = goodness_of_fit(counts, ratio)
        assert (test.statistic, test.p_value) == (0.0, 1.0), (counts, ratio)
        x = np.array(counts, dtype=np.float64)
        e = x.sum() * np.array(ratio)
        noisy_before += math.fsum(((x - e) ** 2 / e).tolist()) != 0.0
    assert noisy_before > 50


@pytest.mark.parametrize("k", [3, 6, 7, 11, 13, 49, 97])
def test_equal_counts_against_equal_shares(k: int) -> None:
    # ratio=None means exactly 1 / k each, and so do k equal shares written as
    # floats (1 / 3 is not a third in binary).
    for m in (1, 7, 12345, 10**9 + 7):
        _assert_exact_fit([m] * k, None)
        _assert_exact_fit([m] * k, [1 / k] * k)


def test_shares_as_written_that_do_not_fit() -> None:
    # 0.333333333 as written is not a third: the counts 1, 1, 1 do not fit
    # 0.333333333, 0.333333333, 0.333333334 exactly, and the small statistic
    # (exactly 6e-18 / 0.999999999... as rationals) is reported, not 0.
    ratio = [0.333333333, 0.333333333, 0.333333334]
    shares = [Fraction("0.333333333"), Fraction("0.333333333"), Fraction("0.333333334")]
    exact = sum(
        ((x - 3 * p) ** 2 / (3 * p) for x, p in zip([1, 1, 1], shares, strict=True)),
        Fraction(0),
    )
    test = goodness_of_fit([1, 1, 1], ratio)
    assert test.statistic > 0.0
    assert test.statistic == pytest.approx(float(exact), rel=1e-6)
    # One count off an exact fit is not a fit either.
    assert goodness_of_fit([8, 2, 7, 9], [0.32, 0.08, 0.28, 0.32]).statistic > 0.0


def test_bar_tests_use_the_exact_fit() -> None:
    # The bar u fits 0.32, 0.08, 0.28, 0.32 exactly; the bar v does not.
    table = np.array([[8, 2], [2, 7], [7, 8], [8, 8]])
    bars = proportion_tests(table, [0.32, 0.08, 0.28, 0.32])
    u, v = bars
    assert u.test is not None
    assert u.effect is not None
    assert (u.test.statistic, u.test.p_value) == (0.0, 1.0)
    assert (u.effect.estimate, u.effect.conf_low, u.effect.conf_high) == (0.0,) * 3
    assert v.test is not None
    assert v.test.statistic > 0.0
    equal = proportion_tests(np.array([[5, 1], [5, 2], [5, 3]]))
    assert equal[0].test is not None
    assert equal[0].test.statistic == 0.0
