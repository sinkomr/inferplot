"""Count tables (``inferplot.stats.contingency``): definitions against exact rational
arithmetic and brute-force enumeration, invariances, degenerate tables and every
``ValueError``, large counts, and the cost of the exact Fisher test.

The oracle parity of every fixture and of probe-contingency is in
tests/stats/test_contingency_parity.py; the noncentral chi-square behind the
intervals in tests/stats/test_noncentral_chi2.py.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from fractions import Fraction
from typing import Any, Final

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import special, stats

from inferplot.stats import (
    chi2_ncp_interval,
    chisq_independence,
    cohens_g,
    column_percentages,
    cramers_v,
    fisher_exact,
    goodness_of_fit,
    mcnemar,
    p_adjust,
    pairwise_fisher,
    pearsons_c,
    proportion_tests,
)
from tests.test_reference_vs_oracle import (
    _fisher_two_by_c,  # pyright: ignore[reportPrivateUsage]
)

IntArray = NDArray[np.int64]
RNG: Final = np.random.default_rng(20261110)


def _random_table(rows: int, cols: int, scale: int) -> IntArray:
    while True:
        table = RNG.integers(0, scale, size=(rows, cols))
        if (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
            return table.astype(np.int64)


def _exact_chi2(table: IntArray) -> Fraction:
    t = [[int(v) for v in row] for row in table.tolist()]
    rows = [sum(r) for r in t]
    cols = [sum(c) for c in zip(*t, strict=True)]
    n = sum(rows)
    total = Fraction(0)
    for i, row in enumerate(t):
        for j, v in enumerate(row):
            e = Fraction(rows[i] * cols[j], n)
            total += (v - e) ** 2 / e
    return total


# --------------------------------------------------------------------------- #
# Chi-square of independence and Cramer's V
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", [(2, 2), (2, 5), (3, 3), (4, 6)])
def test_chi_square_is_pearsons_statistic(shape: tuple[int, int]) -> None:
    for _ in range(20):
        table = _random_table(*shape, scale=40)
        test = chisq_independence(table)
        exact = _exact_chi2(table)
        assert math.isclose(test.statistic, float(exact), rel_tol=1e-13, abs_tol=1e-15)
        assert test.df == (shape[0] - 1) * (shape[1] - 1)
        assert test.p_value == pytest.approx(
            float(special.gammaincc(test.df / 2, test.statistic / 2)), rel=1e-15
        )
        rows, cols = table.sum(axis=1), table.sum(axis=0)
        np.testing.assert_allclose(
            test.expected, np.outer(rows, cols) / table.sum(), rtol=1e-15
        )
        scipy_test = stats.chi2_contingency(table, correction=False)
        assert test.statistic == pytest.approx(float(scipy_test[0]), rel=1e-12)


def test_chi_square_invariances() -> None:
    table = _random_table(3, 4, scale=30)
    base = chisq_independence(table)
    v = cramers_v(table)
    for perm_rows in itertools.permutations(range(3)):
        for perm_cols in ((0, 1, 2, 3), (3, 1, 0, 2)):
            moved = table[list(perm_rows)][:, list(perm_cols)]
            assert chisq_independence(moved).statistic == pytest.approx(
                base.statistic, rel=1e-13
            )
            assert cramers_v(moved).estimate == pytest.approx(v.estimate, rel=1e-13)
    transposed = chisq_independence(table.T)
    assert transposed.statistic == pytest.approx(base.statistic, rel=1e-13)
    assert transposed.df == base.df
    vt = cramers_v(table.T)
    for a, b in ((vt.estimate, v.estimate), (vt.conf_low, v.conf_low)):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-15)
    assert vt.conf_high == pytest.approx(v.conf_high, rel=1e-12)
    # k times every count: chi-square times k, df and p's argument scale alike;
    # the unadjusted V does not change.
    for k in (2, 7, 1000):
        scaled = chisq_independence(table * k)
        assert scaled.statistic == pytest.approx(k * base.statistic, rel=1e-13)
        assert cramers_v(table * k, adjust=False).estimate == pytest.approx(
            cramers_v(table, adjust=False).estimate, rel=1e-13
        )


def test_cramers_v_definition() -> None:
    table = np.array([[12, 5, 7], [3, 9, 14]])
    n, r, c = 50, 2, 3
    chi2 = float(_exact_chi2(table))
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    tilde = min(r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)) - 1
    v = cramers_v(table, conf_level=0.9)
    assert v.estimate == pytest.approx(math.sqrt(phi2 / tilde), rel=1e-13)
    low, high = chi2_ncp_interval(chi2, 2, conf_level=0.9, alternative="two-sided")

    def convert(ncp: float) -> float:
        return math.sqrt(max(0.0, ncp / n - 2 / (n - 1)) / tilde)

    assert v.conf_low == pytest.approx(convert(low), rel=1e-12, abs=1e-15)
    assert v.conf_high == pytest.approx(convert(high), rel=1e-12)
    assert (v.conf_level, v.sidedness) == (0.9, "two-sided")
    greater = cramers_v(table, alternative="greater")
    less = cramers_v(table, alternative="less")
    assert greater.conf_high == 1.0
    assert less.conf_low == 0.0
    unadjusted = cramers_v(table, adjust=False)
    assert unadjusted.estimate == pytest.approx(math.sqrt(chi2 / n), rel=1e-13)


def test_cramers_v_limits() -> None:
    # Perfect association: V = 1 (capped); exact independence: 0 with p = 1.
    perfect = cramers_v(np.array([[8, 0, 0], [0, 6, 0], [0, 0, 7]]))
    assert perfect.estimate == 1.0
    assert perfect.conf_high == 1.0
    independent = np.array([[4, 8, 12], [6, 12, 18]])
    test = chisq_independence(independent)
    assert test.statistic == 0.0
    assert test.p_value == 1.0
    v = cramers_v(independent)
    assert (v.estimate, v.conf_low) == (0.0, 0.0)
    # The adjusted upper bound is 0 where phi^2 at the upper lambda is below the
    # correction (edge_bar_rounding: V 0, CI [0, 0]).
    small = cramers_v(np.array([[1, 1, 1, 1], [5, 1, 2, 3], [10, 1, 4, 4]]))
    assert (small.estimate, small.conf_low, small.conf_high) == (0.0, 0.0, 0.0)


def test_independence_refusals() -> None:
    with pytest.raises(ValueError, match="at least 2 rows and 2 columns"):
        chisq_independence(np.array([[3, 4, 5]]))
    with pytest.raises(ValueError, match="at least 2 rows and 2 columns"):
        chisq_independence(np.array([[3], [4]]))
    with pytest.raises(ValueError, match=r"empty rows \[1\], empty columns \[\]"):
        chisq_independence(np.array([[3, 4], [0, 0]]))
    with pytest.raises(ValueError, match=r"empty columns \[2\]"):
        cramers_v(np.array([[3, 4, 0], [1, 5, 0]]))
    # n no larger than the rows or columns: Bergsma's corrected table has no spread.
    with pytest.raises(ValueError, match="more counts than rows"):
        cramers_v(np.array([[1, 0], [0, 1]]))
    assert cramers_v(np.array([[1, 0], [0, 1]]), adjust=False).estimate == 1.0


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (np.array([[1, -1], [2, 3]]), "negative"),
        (np.array([[1.5, 1.0], [2.0, 3.0]]), "whole"),
        (np.array([[np.nan, 1.0], [2.0, 3.0]]), "whole"),
        (np.array([[True, False], [True, True]]), "whole numbers"),
        (np.array([["a", "b"], ["c", "d"]]), "whole numbers"),
        (np.zeros((2, 2, 2), dtype=np.int64), "rows x columns"),
        (np.zeros((0, 2), dtype=np.int64), "empty"),
        (np.array([[2**52, 2**52], [1, 1]], dtype=np.int64), "below 2\\^53"),
        (np.array([[1e300, 1.0], [1.0, 1.0]]), "below 2\\^53"),
    ],
)
def test_invalid_counts(table: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chisq_independence(table)


def test_whole_floats_and_lists_are_counts() -> None:
    as_int = chisq_independence(np.array([[3, 4], [5, 9]]))
    assert chisq_independence(np.array([[3.0, 4.0], [5.0, 9.0]])) == as_int
    assert chisq_independence([[3, 4], [5, 9]]) == as_int


# --------------------------------------------------------------------------- #
# McNemar, Bowker, Cohen's g
# --------------------------------------------------------------------------- #


def test_mcnemar_and_bowker_definitions() -> None:
    for r in (2, 3, 5):
        for _ in range(10):
            table = RNG.integers(1, 30, size=(r, r))
            test = mcnemar(table)
            exact = sum(
                Fraction((int(table[i, j]) - int(table[j, i])) ** 2)
                / (int(table[i, j]) + int(table[j, i]))
                for i in range(r)
                for j in range(i + 1, r)
            )
            # Each term rounded once, the terms summed exactly rounded: within an
            # ulp or two of the exact rational.
            assert test.statistic == pytest.approx(float(exact), rel=4.5e-16)
            assert test.df == r * (r - 1) // 2
            assert test.p_value == pytest.approx(
                float(special.gammaincc(test.df / 2, test.statistic / 2)), rel=1e-15
            )
            assert test.n_pairs == int(table.sum())
            # Symmetric in the two variables (transposition), and the diagonal
            # does not matter.
            assert mcnemar(table.T).statistic == test.statistic
            bumped = table + np.eye(r, dtype=np.int64) * 17
            assert mcnemar(bumped).statistic == test.statistic


def test_mcnemar_refusals() -> None:
    with pytest.raises(ValueError, match="discordant"):
        mcnemar(np.array([[9, 0], [0, 11]]))
    # R: NaN for a pair of empty mirrored cells (0 / 0), df still 3.
    with pytest.raises(ValueError, match=r"\[\(0, 1\), \(0, 2\)\]"):
        mcnemar(np.array([[5, 0, 0], [0, 5, 2], [0, 1, 5]]))
    # A declared level empty in both measurements (an empty row and column) leaves
    # its pairs of cells empty: refused, not dropped (review C5; R: NaN).
    with pytest.raises(ValueError, match=r"\[\(0, 2\), \(1, 2\)\]"):
        mcnemar(np.array([[5, 2, 0], [3, 6, 0], [0, 0, 0]]))
    with pytest.raises(ValueError, match=r"\[\(0, 1\), \(1, 2\)\]"):
        mcnemar(np.array([[5, 0, 2], [0, 0, 0], [3, 0, 6]]))
    with pytest.raises(ValueError, match="square"):
        mcnemar(np.array([[3, 6, 0], [2, 5, 7]]))
    with pytest.raises(ValueError, match="square"):
        cohens_g(np.array([[3]]))


def _wilson_by_quadratic(k: int, n: int, z: float) -> tuple[float, float]:
    """The roots in pi of (k / n - pi)^2 = z^2 pi (1 - pi) / n (the score test's
    acceptance limits), a different route to Wilson's interval."""
    p = k / n
    a = 1 + z * z / n
    b = -(2 * p + z * z / n)
    c = p * p
    disc = math.sqrt(b * b - 4 * a * c)
    return max(0.0, (-b - disc) / (2 * a)), min(1.0, (-b + disc) / (2 * a))


def test_cohens_g_definition() -> None:
    table = np.array([[8, 1, 3], [2, 6, 5], [6, 1, 7]])  # edge_bar_paired3
    g = cohens_g(table)
    assert g.estimate == pytest.approx(13 / 18 - 0.5, rel=1e-15)
    low, high = _wilson_by_quadratic(13, 18, float(stats.norm.isf(0.025)))
    assert g.conf_low == pytest.approx(low - 0.5, rel=1e-12)
    assert g.conf_high == pytest.approx(high - 0.5, rel=1e-12)
    one = cohens_g(np.array([[5, 0], [1, 4]]))  # one discordant pair
    assert one.estimate == 0.5
    assert one.conf_low == pytest.approx(-0.2934, abs=1e-4)
    assert one.conf_high == 0.5
    less = cohens_g(table, alternative="less", conf_level=0.9)
    greater = cohens_g(table, alternative="greater", conf_level=0.9)
    assert less.conf_low == -0.5
    assert greater.conf_high == 0.5
    z = float(stats.norm.isf(0.1))
    assert less.conf_high == pytest.approx(_wilson_by_quadratic(13, 18, z)[1] - 0.5)
    assert greater.conf_low == pytest.approx(_wilson_by_quadratic(13, 18, z)[0] - 0.5)
    # The side does not matter: the transposed table has the same g.
    assert cohens_g(table.T) == g
    with pytest.raises(ValueError, match="discordant"):
        cohens_g(np.array([[4, 0], [0, 4]]))


# --------------------------------------------------------------------------- #
# Goodness of fit, Pearson's C, per-bar tests, shares
# --------------------------------------------------------------------------- #


def test_goodness_of_fit_definition() -> None:
    for k in (2, 3, 6):
        for _ in range(10):
            x = RNG.integers(0, 40, size=k)
            if x.sum() == 0:
                continue
            ratio = RNG.dirichlet(np.ones(k))
            ratio[-1] = 1.0 - math.fsum(ratio[:-1].tolist())
            test = goodness_of_fit(x, ratio)
            ref = stats.chisquare(x, ratio * x.sum())
            assert test.statistic == pytest.approx(float(ref.statistic), rel=1e-12)
            assert test.p_value == pytest.approx(float(ref.pvalue), rel=1e-10)
            assert test.df == k - 1
            equal = goodness_of_fit(x)
            assert equal.ratio == tuple([1 / k] * k)


def test_goodness_of_fit_on_a_decimal_tie_is_rs_double() -> None:
    # edge_bar_rounding, bar g16: counts 1, 5, 10 against thirds. The exact value
    # for the doubles 1/3 rounds to 7.625; R's order of operations gives 7.625 +
    # 1 ulp, which R prints 7.63 (and the fixture's label shows), and so does this.
    test = goodness_of_fit([1, 5, 10])
    assert test.statistic == math.nextafter(7.625, 8.0)


def test_ratio_validation() -> None:
    x = [11, 7, 14]
    goodness_of_fit(x, [0.5, 0.25, 0.25 + 1e-9])  # within sqrt(2^-52) of 1
    with pytest.raises(ValueError, match="sum to 1"):
        goodness_of_fit(x, [0.5, 0.25, 0.25 + 1e-7])
    with pytest.raises(ValueError, match="sum to 1"):
        goodness_of_fit(x, [2.0, 1.0, 1.0])  # never rescaled
    with pytest.raises(ValueError, match="one probability per category"):
        goodness_of_fit(x, [0.5, 0.5])
    with pytest.raises(ValueError, match="positive"):
        goodness_of_fit(x, [0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="positive"):
        goodness_of_fit(x, [1.2, 0.2, -0.4])
    with pytest.raises(ValueError, match="positive"):
        goodness_of_fit(x, [math.nan, 0.5, 0.5])
    with pytest.raises(ValueError, match="at least 2 categories"):
        goodness_of_fit([12])
    with pytest.raises(ValueError, match="at least one count"):
        goodness_of_fit([0, 0, 0])
    with pytest.raises(ValueError, match="vector"):
        goodness_of_fit([[1, 2], [3, 4]])  # type: ignore[arg-type]


@pytest.mark.parametrize("tiny", [1e-6, 1e-12, 1e-30, 1e-100, 1e-200, 1e-305])
def test_tiny_shares_are_computed_at_once(tiny: float) -> None:
    # Review C7: from a share of about 1e-30 on, the interval's search ran for
    # about 5 seconds and raised ArithmeticError. chi^2 = sum x^2 / (n p) - n.
    ratio = [tiny, 1.0 - tiny]
    start = time.perf_counter()
    test = goodness_of_fit([5, 5], ratio)
    c = pearsons_c([5, 5], ratio)
    (bar,) = proportion_tests(np.array([[5], [5]]), ratio)
    assert time.perf_counter() - start < 1.0
    exact = sum(
        (Fraction(25) / (10 * Fraction(p)) for p in ratio), Fraction(0)
    ) - Fraction(10)
    assert test.statistic == pytest.approx(float(exact), rel=1e-13)
    assert test.p_value == 0.0
    low, high = chi2_ncp_interval(
        test.statistic, 1, conf_level=0.95, alternative="two-sided"
    )
    # From about 1e32 on, x -/+ 2 z sqrt(x) rounds to x itself.
    assert 0.0 < low <= test.statistic <= high < math.inf
    assert low < high or test.statistic > 1e32
    assert c.conf_low == pytest.approx(math.sqrt(low / (low + 10)), rel=1e-15)
    assert c.estimate == pytest.approx(
        math.sqrt(test.statistic / (test.statistic + 10)), rel=1e-15
    )
    assert c.conf_low <= c.estimate <= c.conf_high <= 1.0
    assert bar.test == test
    assert bar.effect == c


@pytest.mark.parametrize(
    ("counts", "ratio"),
    [
        ([5, 5], [1e-310, 1.0 - 1e-310]),
        ([5, 5], [5e-324, 1.0]),
        # each term finite (1.4e308), their sum beyond the largest double
        ([5, 5, 5], [1.2e-308, 1.2e-308, 1.0 - 2.4e-308]),
    ],
)
def test_a_statistic_beyond_the_doubles_names_ratio(
    counts: list[int], ratio: list[float]
) -> None:
    # Review C7: "x must be finite; got inf" from the interval, naming no argument.
    start = time.perf_counter()
    with pytest.raises(ValueError, match=r"^ratio gives category 0 a share of "):
        goodness_of_fit(counts, ratio)
    with pytest.raises(ValueError, match=r"^ratio gives category 0 a share of "):
        proportion_tests(np.array([counts]).T, ratio)
    assert time.perf_counter() - start < 1.0


def test_pearsons_c_definition() -> None:
    x = [11, 7, 14]
    test = goodness_of_fit(x)
    c = pearsons_c(x, conf_level=0.9)
    assert c.estimate == pytest.approx(
        math.sqrt(test.statistic / (test.statistic + 32)), rel=1e-15
    )
    low, high = chi2_ncp_interval(
        test.statistic, 2, conf_level=0.9, alternative="two-sided"
    )
    assert c.conf_low == pytest.approx(math.sqrt(low / (low + 32)), abs=1e-15)
    assert c.conf_high == pytest.approx(math.sqrt(high / (high + 32)), rel=1e-14)
    assert pearsons_c(x, alternative="greater").conf_high == 1.0
    assert pearsons_c(x, alternative="less").conf_low == 0.0
    # A declared empty category is a count of 0 and stays (zero_cell-oneway).
    assert goodness_of_fit([15, 0, 10]).statistic == pytest.approx(14.0, rel=1e-15)


def test_proportion_tests() -> None:
    table = np.array([[3, 0, 8, 0], [4, 0, 3, 0], [12, 0, 2, 1]])
    bars = proportion_tests(table, [0.25, 0.25, 0.5])
    assert [b.column for b in bars] == [0, 2, 3]  # the empty column has no bar
    for bar in bars:
        column = table[:, bar.column]
        assert bar.n == int(column.sum())
        assert bar.test == goodness_of_fit(column, [0.25, 0.25, 0.5])
        assert bar.effect == pearsons_c(column, [0.25, 0.25, 0.5])
        assert bar.effect is not None
        assert (bar.effect.conf_level, bar.effect.sidedness) == (0.95, "two-sided")
        assert bar.reason is None
    one_row = proportion_tests(np.array([[7, 5]]))
    assert [(b.column, b.n, b.test, b.effect) for b in one_row] == [
        (0, 7, None, None),
        (1, 5, None, None),
    ]
    assert all(b.reason for b in one_row)
    with pytest.raises(ValueError, match="one probability per category"):
        proportion_tests(table, [0.5, 0.5])


def _exact_gof(counts: list[int], shares: list[float]) -> Fraction:
    """sum (x - n p)^2 / (n p) in rational arithmetic on the doubles p."""
    n = sum(counts)
    return sum(
        (
            (x - n * Fraction(p)) ** 2 / (n * Fraction(p))
            for x, p in zip(counts, shares, strict=True)
        ),
        Fraction(0),
    )


def _check_gof(test: Any, counts: list[int], shares: list[float]) -> None:
    k = len(counts)
    exact = _exact_gof(counts, shares)
    assert test.df == k - 1
    assert test.n == sum(counts)
    assert test.statistic == pytest.approx(float(exact), rel=1e-13)
    assert test.p_value == pytest.approx(
        float(stats.chi2.sf(float(exact), k - 1)), rel=1e-12
    )


@pytest.mark.parametrize("k", [8, 9, 10, 11, 12, 50])
def test_goodness_of_fit_with_many_categories(k: int) -> None:
    # Review C5: the degrees of freedom are k - 1 for every k (a df wrong from 7
    # categories up went unnoticed), checked against exact rational arithmetic,
    # in the one-way test and in every bar's test.
    rng = np.random.default_rng(k)
    weights = rng.integers(1, 20, size=k)
    given = (weights / weights.sum()).tolist()
    for shares in (None, given):
        p = [1 / k] * k if shares is None else shares
        for _ in range(5):
            counts = rng.integers(0, 40, size=k).tolist()
            if sum(counts) == 0:
                continue
            _check_gof(goodness_of_fit(counts, shares), counts, p)
        table = rng.integers(0, 30, size=(k, 3))
        table[0, :] += 1  # every column holds counts
        bars = proportion_tests(table, shares)
        assert [b.column for b in bars] == [0, 1, 2]
        for bar in bars:
            column = [int(v) for v in table[:, bar.column].tolist()]
            _check_gof(bar.test, column, p)
            assert bar.effect is not None
            assert bar.effect == pearsons_c(column, shares)


def test_column_percentages() -> None:
    shares = column_percentages(np.array([[3, 0, 1], [1, 0, 1], [0, 0, 1]]))
    assert shares[0] == (75.0, None, (1 / 3) * 100)
    assert shares[1] == (25.0, None, (1 / 3) * 100)
    assert shares[2] == (0.0, None, (1 / 3) * 100)


# --------------------------------------------------------------------------- #
# Fisher's exact test
# --------------------------------------------------------------------------- #


def _fisher_by_enumeration(table: IntArray) -> Fraction:
    """The two-sided Fisher p of a 2 x c table by enumerating every first row with
    the margins, in exact rational arithmetic, with R's P <= P_obs (1 + 1e-7)."""
    first = [int(v) for v in table[0]]
    cols = [int(v) for v in table.sum(axis=0)]
    r1, n = sum(first), sum(cols)
    total = math.comb(n, r1)

    def prob(row: tuple[int, ...]) -> Fraction:
        return Fraction(
            math.prod(math.comb(c, a) for c, a in zip(cols, row, strict=True)), total
        )

    observed = prob(tuple(first)) * (1 + Fraction(1, 10**7))
    p = Fraction(0)
    for row in itertools.product(*(range(c + 1) for c in cols[:-1])):
        last = r1 - sum(row)
        if not 0 <= last <= cols[-1]:
            continue
        q = prob((*row, last))
        if q <= observed:
            p += q
    return p


@pytest.mark.parametrize("cols", [2, 3, 4, 5])
def test_fisher_against_exact_enumeration(cols: int) -> None:
    scale = {2: 25, 3: 12, 4: 7, 5: 5}[cols]
    worst = 0.0
    for _ in range(40):
        table = RNG.integers(0, scale, size=(2, cols)).astype(np.int64)
        exact = _fisher_by_enumeration(table)
        ours = fisher_exact(table).p_value
        worst = max(worst, abs(ours - float(exact)) / float(exact))
    # Measured: at most a few units of 1e-15.
    assert worst <= 1e-13


def test_fisher_ties_and_the_relative_allowance() -> None:
    # Mirror images have exactly the observed probability and always count; the
    # balanced table's p is 1.
    for table in ([[3, 5], [5, 3]], [[2, 7, 4], [4, 7, 2]], [[5, 5], [5, 5]]):
        exact = _fisher_by_enumeration(np.array(table))
        assert fisher_exact(np.array(table)).p_value == pytest.approx(
            float(exact), rel=1e-14
        )
    assert fisher_exact(np.array([[5, 5], [5, 5]])).p_value == 1.0


def test_fisher_two_by_two_against_scipy() -> None:
    worst = 0.0
    for _ in range(300):
        table = RNG.integers(0, 40, size=(2, 2))
        for side in ("two-sided", "less", "greater"):
            ours = fisher_exact(table, alternative=side).p_value
            ref = float(stats.fisher_exact(table, alternative=side)[1])
            worst = max(worst, abs(ours - ref) / ref)
    # Measured 8.7e-15 (SciPy sums the hypergeometric pmf in double precision).
    assert worst <= 1e-12


def test_fisher_one_sided_is_the_hypergeometric_tail() -> None:
    table = np.array([[1, 9], [8, 4]])
    a, r1, c0, n = 1, 10, 9, 22
    pmf = [
        Fraction(math.comb(c0, k) * math.comb(n - c0, r1 - k), math.comb(n, r1))
        for k in range(min(r1, c0) + 1)
    ]
    less = float(sum(pmf[: a + 1]))
    greater = float(sum(pmf[a:]))
    assert fisher_exact(table, alternative="less").p_value == pytest.approx(less)
    assert fisher_exact(table, alternative="greater").p_value == pytest.approx(greater)


def test_fisher_degenerate_tables() -> None:
    # One empty row (R: [[0, 0], [3, 4]] gives 1) or one non-empty column: a single
    # table has the margins, p = 1; empty columns are left out.
    assert fisher_exact(np.array([[0, 0], [3, 4]])).p_value == 1.0
    assert fisher_exact(np.array([[3, 0], [4, 0]])).p_value == 1.0
    with_empty = fisher_exact(np.array([[1, 0, 9], [8, 0, 4]]))
    assert with_empty.p_value == fisher_exact(np.array([[1, 9], [8, 4]])).p_value
    assert with_empty.columns == 2
    with pytest.raises(ValueError, match="2 x c"):
        fisher_exact(np.array([[1, 2], [3, 4], [5, 6]]))
    with pytest.raises(ValueError, match="2 x c"):
        fisher_exact(np.array([[1], [3]]))
    with pytest.raises(ValueError, match="one-sided"):
        fisher_exact(np.array([[1, 2, 3], [3, 4, 5]]), alternative="greater")


def test_fisher_at_large_counts() -> None:
    # 1e9 per cell (n = 4e9, any total below 2^53 works for 2 x 2): the exact p
    # approaches the chi-square p as n grows (here within 1e-3 relative), and the
    # mirror image of the table has the same p.
    table = np.array([[10**9, 10**9 + 50_000], [10**9 + 30_000, 10**9]])
    p = fisher_exact(table).p_value
    assert p == pytest.approx(chisq_independence(table).p_value, rel=1e-3)
    assert fisher_exact(table[::-1]).p_value == pytest.approx(p, rel=1e-12)
    # A table far from independence at n = 2e6: p = exp(-~1e3) underflows only
    # below the doubles; far enough, it is 0.
    far = np.array([[600_000, 400_000], [400_000, 600_000]])
    assert fisher_exact(far).p_value == 0.0
    near = np.array([[500_400, 499_600], [499_600, 500_400]])
    assert fisher_exact(near).p_value == pytest.approx(
        chisq_independence(near).p_value, rel=1e-2
    )


def _seconds(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def _table_2xc(n: int, cols: int, tilt: float) -> IntArray:
    """A 2 x cols table of total n with equal column totals; the first row's share
    of column j moves by tilt from 1/2 across the columns."""
    per = n // cols
    shares = 0.5 + tilt * (np.arange(cols) / (cols - 1) - 0.5)
    first = np.rint(per * shares).astype(np.int64)
    return np.vstack([first, per - first])


# The tables the brief asks to be timed, near independence and far from it. Best of
# three on the build machine (Apple M-series): 2 x 2 n = 1e6 0.017 s; 2 x 3 n =
# 2,000 0.004 s (p 0.18) and 0.015 s (p 3e-27); n = 10,000 0.033 s (p 0.05) and
# 0.29 s (p 9e-133); 2 x 5 n = 500 0.003 s (p 0.68) and 0.043 s (p 5e-13); n =
# 2,000 0.010 s (p 0.65) and 0.094 s (p 1.6e-4). The bound is the documented two
# seconds.
TIMED: Final = [
    (np.array([[500_000, 500_000], [499_000, 501_000]]), "2 x 2, n = 1e6, p 0.16"),
    (_table_2xc(2_000, 3, 0.05), "2 x 3, n = 2,000, p 0.18"),
    (_table_2xc(2_000, 3, 0.3), "2 x 3, n = 2,000, p 3e-27"),
    (_table_2xc(10_000, 3, 0.03), "2 x 3, n = 10,000, p 0.05"),
    (_table_2xc(10_000, 3, 0.3), "2 x 3, n = 10,000, p 9e-133"),
    (_table_2xc(500, 5, 0.1), "2 x 5, n = 500, p 0.68"),
    (_table_2xc(500, 5, 0.5), "2 x 5, n = 500, p 5e-13"),
    (_table_2xc(2_000, 5, 0.05), "2 x 5, n = 2,000, p 0.65"),
    (_table_2xc(2_000, 5, 0.15), "2 x 5, n = 2,000, p 1.6e-4"),
]


@pytest.mark.parametrize(("table", "what"), TIMED, ids=[w for _, w in TIMED])
def test_fisher_cost(table: IntArray, what: str) -> None:
    # These tables are within the search limit, so they are answered. The limit is
    # on the work (array elements), which does not depend on the machine; a wall
    # clock is not asserted (a loaded CI runner is several times slower than the
    # timings above). A generous bound still catches a search gone quadratic.
    seconds = _seconds(lambda: fisher_exact(table))
    assert seconds <= 60.0, f"{what}: {seconds:.3f} s"


def test_fisher_refuses_what_would_take_too_long() -> None:
    # 2 x 5, n = 2,000, far from independence (p ~ 1e-55): the search would need
    # about 1.2e8 array elements (3.2 s); it stops before the stage that would
    # pass 6e7 and says so. The stages before it are spent (measured 1.7 s), so
    # the refusal itself takes up to about the budget.
    # No wall clock is asserted: the refusal depends on the work, not the machine.
    table = _table_2xc(2_000, 5, 1.0)
    with pytest.raises(ValueError, match="more than about two seconds"):
        fisher_exact(table)
    with pytest.raises(ValueError, match="merge sparse levels"):
        fisher_exact(np.array([[2_000_000, 2_000_000, 2_000_000], [1, 1, 1]]))


@pytest.mark.parametrize(
    "table",
    [_table_2xc(500, 5, 0.1), _table_2xc(500, 5, 0.5), _table_2xc(400, 4, 0.8)],
    ids=["2x5 n500 p0.68", "2x5 n500 p5e-13", "2x4 n400"],
)
def test_fisher_two_by_c_against_meet_in_the_middle(table: IntArray) -> None:
    # tests/test_reference_vs_oracle.py's independent enumeration (both halves of
    # the columns enumerated, joined by their sums; log-gamma probabilities).
    assert fisher_exact(table).p_value == pytest.approx(
        _fisher_two_by_c(table), rel=1e-10
    )


def test_fisher_two_by_three_at_n_2000_against_a_full_sum() -> None:
    # Every table of the margins, in log space with log-gamma (independent of the
    # search): a 2 x 3 table has (c0 + 1)(c1 + 1) first rows at most.
    table = _table_2xc(2_000, 3, 0.1)
    first = table[0]
    cols = table.sum(axis=0)
    r1 = int(first.sum())
    a0 = np.arange(cols[0] + 1)[:, None]
    a1 = np.arange(cols[1] + 1)[None, :]
    a2 = r1 - a0 - a1
    valid = (a2 >= 0) & (a2 <= cols[2])
    a2 = np.where(valid, a2, 0)

    def log_choose(m: int, k: Any) -> Any:
        return (
            special.gammaln(m + 1) - special.gammaln(k + 1) - special.gammaln(m - k + 1)
        )

    logs = log_choose(cols[0], a0) + log_choose(cols[1], a1) + log_choose(cols[2], a2)
    logs = np.where(valid, logs, -np.inf)
    observed = float(logs[first[0], first[1]])
    counted = logs <= observed + math.log1p(1e-7)
    total = special.logsumexp(logs[valid])
    expected = math.exp(float(special.logsumexp(logs[counted & valid])) - total)
    assert fisher_exact(table).p_value == pytest.approx(expected, rel=1e-10)


# --------------------------------------------------------------------------- #
# Pairwise
# --------------------------------------------------------------------------- #


def test_pairwise_fisher() -> None:
    table = np.array([[3, 8], [4, 3], [0, 0], [12, 2]])
    pairs = pairwise_fisher(table, p_adjust_method="BH", conf_level=0.9)
    assert [(p.row1, p.row2) for p in pairs] == [(0, 1), (0, 3), (1, 3)]
    raw = [fisher_exact(table[[p.row1, p.row2]]).p_value for p in pairs]
    assert [p.p_value for p in pairs] == raw
    assert [p.p_adjusted for p in pairs] == list(p_adjust(raw, "BH"))
    for pair in pairs:
        assert pair.alternative == "two-sided"
        assert pair.effect == cramers_v(table[[pair.row1, pair.row2]], conf_level=0.9)
        assert pair.columns == (0, 1)
        assert pair.n == int(table[[pair.row1, pair.row2]].sum())


def test_pairwise_alternative_reaches_two_by_two_pairs_only() -> None:
    table = np.array([[3, 8, 0], [4, 3, 0], [12, 2, 5]])
    pairs = pairwise_fisher(table, alternative="greater")
    assert [(p.row1, p.row2, p.columns, p.alternative) for p in pairs] == [
        (0, 1, (0, 1), "greater"),
        (0, 2, (0, 1, 2), "two-sided"),
        (1, 2, (0, 1, 2), "two-sided"),
    ]
    assert (
        pairs[0].p_value == fisher_exact(table[:2, :2], alternative="greater").p_value
    )
    for pair in pairs:  # every pair's V interval follows alternative
        assert pair.effect is not None
        assert pair.effect.sidedness == "greater"


def test_pairwise_marks_what_is_not_defined() -> None:
    # Rows 0 and 1 hold counts in one column only: no test (R loses the whole
    # table); rows 0 and 2 form [[1, 0], [0, 1]]: a test, but no adjusted V.
    table = np.array([[1, 0], [3, 0], [0, 1]])
    pairs = pairwise_fisher(table)
    first, second, third = pairs
    assert (first.p_value, first.p_adjusted, first.alternative) == (None, None, None)
    assert first.reason is not None
    assert first.columns == (0,)
    assert second.p_value == 1.0
    assert second.effect is None
    assert second.effect_reason is not None
    assert third.effect is not None
    # The adjustment runs over the pairs with a test.
    assert [p.p_adjusted for p in pairs[1:]] == list(
        p_adjust([second.p_value, third.p_value], "holm")  # type: ignore[list-item]
    )


# --------------------------------------------------------------------------- #
# Large counts
# --------------------------------------------------------------------------- #


def test_large_counts_keep_their_accuracy() -> None:
    # Counts near 1e9 per cell. The chi-square statistics follow R's order of
    # operations (expected counts rounded once): relative error 1.2e-11 at chi^2
    # 1.6 here, absolute 2e-17 next to independence, against exact rationals.
    # McNemar's terms are exact integer ratios rounded once. Nothing overflows.
    big = 10**9
    table = np.array([[big, big + 50_000], [big + 30_000, big]])
    test = chisq_independence(table)
    assert test.statistic == pytest.approx(float(_exact_chi2(table)), rel=1e-10)
    near = np.array([[big, big + 7], [big + 3, big]])
    assert chisq_independence(near).statistic == pytest.approx(
        float(_exact_chi2(near)), abs=1e-15
    )
    mixed = np.array([[123456789, 987654321, 555555555], [999999999, 111111111, 7]])
    assert chisq_independence(mixed).statistic == pytest.approx(
        float(_exact_chi2(mixed)), rel=1e-13
    )
    paired = np.array([[big, big + 12_345], [big - 6_789, 3 * big]])
    b, c = big + 12_345, big - 6_789
    assert mcnemar(paired).statistic == float(Fraction((b - c) ** 2, b + c))
    gof = goodness_of_fit([big, big + 40_000, big - 40_000])
    n = 3 * big
    exact = sum(
        Fraction((x - Fraction(n, 3)) ** 2) / Fraction(n, 3)
        for x in (big, big + 40_000, big - 40_000)
    )
    assert gof.statistic == pytest.approx(float(exact), rel=1e-9)
    v = cramers_v(table)
    assert v.conf_low == 0.0  # chi^2 1.6 on 1 df: p = 0.21
    assert 0.0 < v.estimate < v.conf_high < 1e-4
