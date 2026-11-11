# NumPy's stubs leave overloads of sort, argsort and the ufunc methods partially
# unknown, which strict mode reports on every call; the arrays passed are all typed.
# pyright: reportUnknownMemberType=false
"""Count tables (``barstats``): Pearson's chi-square test of independence with
Cramer's V, McNemar's and Bowker's tests with Cohen's g, the chi-square goodness of
fit with Pearson's C, the per-bar goodness-of-fit tests, Fisher's exact test of a
2 x c table and the pairwise Fisher tests between the rows, and each cell's share
of its column.

The R references are what statsExpressions 2.2.1's ``contingency_table()`` and
ggstatsplot 1.1.4's ggbarstats call (SPEC.md section 4, "barstats"):
``stats::chisq.test(correct = FALSE)`` with ``effectsize::cramers_v(adjust =
TRUE)``, ``stats::mcnemar.test(correct = FALSE)`` with ``effectsize::cohens_g``,
``stats::chisq.test(x, p = ratio)`` with ``effectsize::pearsons_c``, and
``stats::fisher.test`` then ``stats::p.adjust`` for the pairs.

Every function takes a table of whole, non-negative counts (a NumPy array or nested
sequences; rows are the levels of x, columns the levels of y) and returns frozen
dataclasses of plain Python floats and ints. Which rows and columns a table holds is
the caller's business (the analysis drops empty levels where ggstatsplot does);
these functions test the table they are given and raise ``ValueError`` where the
statistic is not defined on it.

Large counts. Counts are held as Python integers wherever they are multiplied
(McNemar's numerators, the Fisher margins), so nothing overflows; a table total
must stay below 2^53 so that every margin is an exact double. The chi-square
statistics follow R's order of floating-point operations (expected counts as
outer(row sums, column sums) / n or n p, each term (x - E)^2 / E, the terms summed
exactly rounded, as R's long-double sum does): their absolute error is at most
about 2^-51 sqrt(chi^2 n) + 2^-50 chi^2 (the rounding of the expected counts times
sum |x - E| <= sqrt(chi^2 n)), so the relative error grows with the counts near
independence: worst over 200 random tables near independence, 5.5e-14 at 1e3
counts per cell, 1.4e-11 at 1e6, 1.3e-10 at 1e9, 3.4e-9 at 1e12, 9.4e-8 at 5e14
(at most 0.39 of the bound), far below the printed digits except at a decimal
tie. R's order is kept so that a statistic on a decimal tie prints as R prints it
(edge_bar_rounding, bar g16: 7.625 + 1 ulp, printed 7.63; an exactly rounded
statistic would print 7.62 there and change no other printed number in the
fixtures, DECISIONS D107). Where the exact value is 0, decided in
rational arithmetic on the counts and on ``ratio`` as written, it is 0: an exactly
independent table, an exact fit, a bias-corrected phi^2 of 0 or below (R's order
leaves noise there: "V = 3.95e-09" for [[0, 1], [5, 4]]).
Fisher's test works with log probabilities
in Loader's (2000) saddle-point form, whose absolute error does not grow with the
counts (fisher_exact).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import special

from inferplot.stats.effect_sizes import EffectSizeCI
from inferplot.stats.noncentral import (
    _bd0,  # pyright: ignore[reportPrivateUsage]
    _stirlerr,  # pyright: ignore[reportPrivateUsage]
    check_conf_level,
    chi2_ncp_interval,
)
from inferplot.stats.padjust import PAdjustMethod, p_adjust
from inferplot.stats.parametric import Alternative, check_alternative
from inferplot.stats.written import READ_ULPS, written_values

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CountTable: TypeAlias = NDArray[Any] | Sequence[Sequence[int]]
"""A table of counts: a two-dimensional NumPy array or nested sequences (rows x
columns)."""
CountVector: TypeAlias = NDArray[Any] | Sequence[int]
"""Counts of the categories of one variable."""
Probabilities: TypeAlias = NDArray[Any] | Sequence[float]
"""Expected probabilities, one per category."""

__all__ = [
    "BarTest",
    "CountTable",
    "CountVector",
    "FisherExact",
    "GoodnessOfFit",
    "IndependenceTest",
    "PairComparison",
    "Probabilities",
    "SymmetryTest",
    "chisq_independence",
    "cohens_g",
    "column_percentages",
    "cramers_v",
    "fisher_exact",
    "goodness_of_fit",
    "mcnemar",
    "pairwise_fisher",
    "pearsons_c",
    "proportion_tests",
]

# A table total must stay below this, so that every margin is an exact double.
_MAX_TOTAL: Final = 2**53
# R's chisq.test refuses probabilities whose sum is further than this from 1
# (sqrt(.Machine$double.eps)).
_SUM_TOLERANCE: Final = math.sqrt(2.0**-52)
# A positive corrected phi^2 within this many times (sqrt(chi^2 n) + chi^2 + 2 df)
# / n of 0 is checked in exact arithmetic (about 500 times its floating-point error
# bound; _corrected_phi2_vanishes).
_PHI2_SCREEN: Final = 2.0**-40
# A share of ratio without a decimal reading (D105) is read as the fraction with
# the smallest denominator up to this within READ_ULPS units in its last place
# (1 / 3, 2 / 7); two such fractions are at least 1e-12 apart, so it is unique.
_MAX_SHARE_DENOMINATOR: Final = 10**6
# The per-bar tests are always two-sided 95% intervals (ggstatsplot's
# onesample_data() passes neither alternative nor conf.level).
_BAR_CONF_LEVEL: Final = 0.95
# fisher.test counts a table as "at most as probable" as the observed one when its
# probability is at most the observed one's times 1 + 1e-7 (relErr).
_RELATIVE_TIE: Final = 1e-7
_LOG_TIE: Final = math.log1p(_RELATIVE_TIE)
_LOG_2PI: Final = math.log(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IndependenceTest:
    """Pearson's chi-square test of independence without continuity correction
    (R: ``chisq.test(x, correct = FALSE)``).

    ``statistic`` = sum (x - E)^2 / E on ``df`` = (r - 1)(c - 1), ``p_value`` its
    upper chi-square tail; ``expected`` the expected counts E = row sum x column sum
    / n (row by row), so that a caller can report small ones (R warns below 5).
    """

    statistic: float
    df: int
    p_value: float
    n: int
    expected: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class SymmetryTest:
    """McNemar's test (2 x 2) or Bowker's test of symmetry (r x r, r > 2) without
    continuity correction (R: ``mcnemar.test(x, correct = FALSE)``).

    ``statistic`` = sum over i < j of (n_ij - n_ji)^2 / (n_ij + n_ji) on ``df`` =
    r (r - 1) / 2; ``n_pairs`` the table total (subjects), ``discordant`` the
    number of pairs off the diagonal.
    """

    statistic: float
    df: int
    p_value: float
    n_pairs: int
    discordant: int


@dataclass(frozen=True, slots=True)
class GoodnessOfFit:
    """The chi-square goodness-of-fit test of counts against probabilities ``ratio``
    (R: ``chisq.test(x, p = ratio)``): ``statistic`` = sum (x - n p)^2 / (n p) on
    ``df`` = k - 1; ``expected`` the counts n p."""

    statistic: float
    df: int
    p_value: float
    n: int
    expected: tuple[float, ...]
    ratio: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BarTest:
    """The goodness-of-fit test within one column (bar) of a table.

    ``column`` is the column's index, ``n`` its total. ``test`` and ``effect``
    (Pearson's C, two-sided 95%) are None where the test is not defined, with
    ``reason`` saying why (ggstatsplot prints NA there).
    """

    column: int
    n: int
    test: GoodnessOfFit | None
    effect: EffectSizeCI | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class FisherExact:
    """Fisher's exact test of a 2 x c table (R: ``fisher.test``).

    ``p_value`` follows ``alternative`` (one-sided only for a 2 x 2 table); ``n`` is
    the table total and ``columns`` the number of columns that hold counts.
    """

    p_value: float
    alternative: Alternative
    n: int
    columns: int


@dataclass(frozen=True, slots=True)
class PairComparison:
    """One pair of rows (levels of x) of a table, ``row1`` < ``row2`` by index.

    ``columns`` are the indices of the columns the pair holds (its own 2 x c'
    table). ``p_value`` is Fisher's exact p of that table (``alternative`` as used:
    one-sided only where the pair's table is 2 x 2), ``p_adjusted`` the adjusted p
    over the family (the pairs with a test; one too large to compute counts with
    p = 1); ``effect`` the pair's Cramer's V (adjusted)
    with its interval. Where a number is not defined it is None and ``reason`` (for
    the test) or ``effect_reason`` (for V) says why.
    """

    row1: int
    row2: int
    columns: tuple[int, ...]
    n: int
    p_value: float | None
    p_adjusted: float | None
    alternative: Alternative | None
    effect: EffectSizeCI | None
    reason: str | None
    effect_reason: str | None


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #


def _as_counts(table: CountTable | CountVector, *, ndim: int = 2) -> IntArray:
    """``table`` as an int64 array of whole, non-negative counts with ``ndim``
    dimensions (ValueError otherwise); its total must stay below 2^53."""
    raw = np.asarray(table)
    if raw.ndim != ndim:
        what = "a table (rows x columns)" if ndim == 2 else "a vector of counts"
        raise ValueError(f"expected {what}; got an array of shape {raw.shape}")
    if raw.size == 0:
        raise ValueError("the table is empty")
    if raw.dtype == np.bool_ or not (
        np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.floating)
    ):
        raise ValueError(f"counts must be whole numbers; got dtype {raw.dtype}")
    if np.issubdtype(raw.dtype, np.floating):
        values = raw.astype(np.float64)
        if not bool(np.isfinite(values).all()) or bool(
            (values != np.floor(values)).any()
        ):
            raise ValueError("counts must be whole, finite numbers")
        if bool((np.abs(values) >= _MAX_TOTAL).any()):
            raise ValueError("counts must be below 2^53")
        counts = values.astype(np.int64)
    else:
        counts = raw.astype(np.int64)
    if bool((counts < 0).any()):
        raise ValueError("counts must not be negative")
    total = sum(int(v) for v in counts.ravel().tolist())
    if total >= _MAX_TOTAL:
        raise ValueError(
            f"the table total {total} must be below 2^53, so that every margin is "
            "an exact double"
        )
    return counts


def _chisq_sf(statistic: float, df: int) -> float:
    """Upper tail of the chi-square distribution, P(X > statistic), by the
    regularized upper incomplete gamma function (never 1 - cdf)."""
    return float(special.gammaincc(0.5 * df, 0.5 * statistic))


def _chisq_sum(terms: FloatArray) -> float:
    """The terms summed exactly rounded (R sums in long double)."""
    return math.fsum(terms.ravel().tolist())


# --------------------------------------------------------------------------- #
# Independence and Cramer's V
# --------------------------------------------------------------------------- #


def _check_margins(counts: IntArray) -> None:
    r, c = counts.shape
    if r < 2 or c < 2:
        raise ValueError(
            f"a test of independence needs at least 2 rows and 2 columns; the table "
            f"is {r} x {c}"
        )
    rows = [int(v) for v in counts.sum(axis=1).tolist()]
    cols = [int(v) for v in counts.sum(axis=0).tolist()]
    empty_rows = [i for i, v in enumerate(rows) if v == 0]
    empty_cols = [j for j, v in enumerate(cols) if v == 0]
    if empty_rows or empty_cols:
        raise ValueError(
            "a test of independence needs counts in every row and column; empty "
            f"rows {empty_rows}, empty columns {empty_cols}"
        )


def _margins(counts: IntArray) -> tuple[list[list[int]], list[int], list[int], int]:
    """The table, its row and column sums and its total as Python integers."""
    table = [[int(v) for v in row] for row in counts.tolist()]
    rows = [sum(row) for row in table]
    cols = [sum(col) for col in zip(*table, strict=True)]
    return table, rows, cols, sum(rows)


def _exactly_independent(counts: IntArray) -> bool:
    """Whether every count equals its expected count exactly (n x_ij = R_i C_j in
    integers), so that Pearson's statistic is exactly 0."""
    table, rows, cols, n = _margins(counts)
    return all(
        n * v == rows[i] * cols[j]
        for i, row in enumerate(table)
        for j, v in enumerate(row)
    )


def _independence(counts: IntArray) -> tuple[float, FloatArray]:
    """Pearson's statistic in R's order of operations, and the expected counts;
    exactly 0 where every count equals its expected count (``_exactly_independent``),
    where R's order leaves a rounding remainder (1.9e-20 at counts of 1e10)."""
    x = counts.astype(np.float64)
    rows = x.sum(axis=1)
    cols = x.sum(axis=0)
    n = float(x.sum())
    expected = np.outer(rows, cols) / n
    if _exactly_independent(counts):
        return 0.0, expected
    return _chisq_sum((x - expected) ** 2 / expected), expected


def chisq_independence(table: CountTable) -> IndependenceTest:
    """Pearson's (1900) chi-square test of independence on an r x c table, without
    Yates' continuity correction, also for 2 x 2 tables (R: ``chisq.test(x,
    correct = FALSE)``, as statsExpressions calls it; never a simulated p).

    chi^2 = sum over cells of (x - E)^2 / E with E = (row sum)(column sum) / n, on
    (r - 1)(c - 1) df; p is the upper tail P(X > chi^2) (regularized upper
    incomplete gamma). Expected counts below 5 only make R warn (ggstatsplot mutes
    it); they are returned for the caller to report.

    Raises:
        ValueError: the table is not r x c with r, c >= 2, a row or column is
            empty, or a count is negative, not whole or too large.
    """
    counts = _as_counts(table)
    _check_margins(counts)
    statistic, expected = _independence(counts)
    r, c = counts.shape
    df = (r - 1) * (c - 1)
    return IndependenceTest(
        statistic=statistic,
        df=df,
        p_value=_chisq_sf(statistic, df),
        n=int(counts.sum()),
        expected=tuple(tuple(float(v) for v in row) for row in expected.tolist()),
    )


def _cramers_v_from(chisq: float, n: int, r: int, c: int, *, adjust: bool) -> float:
    """Cramer's V of a chi-square (or a noncentrality in its place), capped at 1.

    Adjusted (Bergsma 2013): phi^2 = max(0, chi^2 / n - (r - 1)(c - 1) / (n - 1)),
    over min(r~, c~) - 1 with r~ = r - (r - 1)^2 / (n - 1); unadjusted (Cramer
    1946): chi^2 / n over min(r, c) - 1.
    """
    if math.isinf(chisq):
        return 1.0
    if adjust:
        phi2 = max(0.0, chisq / n - (r - 1) * (c - 1) / (n - 1))
        rows = r - (r - 1) ** 2 / (n - 1)
        cols = c - (c - 1) ** 2 / (n - 1)
        return min(1.0, math.sqrt(phi2 / (min(rows, cols) - 1.0)))
    return min(1.0, math.sqrt(chisq / n / (min(r, c) - 1)))


def _corrected_phi2_vanishes(counts: IntArray, chisq: float) -> bool:
    """Whether a positive floating-point corrected phi^2, chi^2 / n - df / (n - 1),
    is rounding noise: its exact value on the counts is 0 or below.

    chi^2 = n (S - 1) with S = sum x_ij^2 / (R_i C_j), so phi^2 <= 0 exactly when
    (S - 1)(n - 1) <= df, decided in rational arithmetic. Only values within
    ``_PHI2_SCREEN`` (sqrt(chi^2 n) + chi^2 + 2 df) / n of 0 are checked: the
    floating-point phi^2 is off by at most about 2^-49 times that (the expected
    counts' rounding times sum |x - E| <= sqrt(chi^2 n), and the roundings of the
    terms), so every value beyond is positive in exact arithmetic too.
    """
    table, rows, cols, n = _margins(counts)
    df = (len(rows) - 1) * (len(cols) - 1)
    phi2 = chisq / n - df / (n - 1)
    screen = _PHI2_SCREEN * (math.sqrt(chisq * n) + chisq + 2.0 * df) / n
    if not 0.0 < phi2 <= screen:
        return False
    common = math.lcm(*cols)
    s = Fraction(0)  # S times common
    for row, r in zip(table, rows, strict=True):
        scaled = (v * v * (common // c) for v, c in zip(row, cols, strict=True))
        s += Fraction(sum(scaled), r)
    return (s / common - 1) * (n - 1) <= df


def cramers_v(
    table: CountTable,
    *,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
    adjust: bool = True,
) -> EffectSizeCI:
    """Cramer's V of an r x c table with Bergsma's (2013, Journal of the Korean
    Statistical Society 42: 323-328) bias correction and its noncentral chi-square
    interval (R: ``effectsize::cramers_v(x, adjust = TRUE, ci = conf_level,
    alternative)``, as statsExpressions calls it with ggbarstats' ``alternative``,
    two-sided by default; "Cramer's V (adj.)").

    With chi^2 Pearson's statistic (``chisq_independence``) on n counts:
    phi^2 = max(0, chi^2 / n - (r - 1)(c - 1) / (n - 1)), r~ = r - (r - 1)^2 /
    (n - 1), c~ likewise, V = sqrt(phi^2 / (min(r~, c~) - 1)), capped at 1. The
    interval (Smithson 2003, ch. 5) inverts the noncentral chi-square in its
    noncentrality lambda at chi^2 (``chi2_ncp_interval``: two-sided bounds solve
    P(X' <= chi^2) = 1 - alpha / 2 and alpha / 2; "greater" gives [low, 1], "less"
    [0, high]; a bound is 0 where no lambda > 0 solves) and puts each bound through
    the same formula with lambda in place of chi^2, capped at 1. The adjusted upper
    bound can be 0 where phi^2 at the upper lambda is below the correction. V is 0
    where phi^2 is exactly 0 or below on the counts (``_corrected_phi2_vanishes``:
    [[0, 1], [5, 4]] has phi^2 = 1/9 - 1/9, where R's order of operations leaves
    V = 3.95e-09); its interval still inverts chi^2. The roots are the accurate
    ones: effectsize's search stops short on some bounds (DECISIONS D99; up to
    0.00096 in a fixture bound). ``adjust=False`` gives
    Cramer's (1946) V = sqrt(chi^2 / (n (min(r, c) - 1))).

    Raises:
        ValueError: as ``chisq_independence``; or n is at most the number of rows
            or columns (with the correction, min(r~, c~) - 1 <= 0: R gives NaN);
            or conf_level or alternative is invalid.
    """
    counts = _as_counts(table)
    _check_margins(counts)
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    r, c = counts.shape
    n = int(counts.sum())
    if adjust and n <= max(r, c):
        raise ValueError(
            f"the bias-corrected Cramer's V needs more counts than rows and columns; "
            f"the {r} x {c} table holds {n} (R gives NaN)"
        )
    statistic, _ = _independence(counts)
    df = (r - 1) * (c - 1)
    low, high = chi2_ncp_interval(statistic, df, conf_level=level, alternative=side)
    estimate = _cramers_v_from(statistic, n, r, c, adjust=adjust)
    if adjust and _corrected_phi2_vanishes(counts, statistic):
        estimate = 0.0
    return EffectSizeCI(
        estimate=estimate,
        conf_low=_cramers_v_from(low, n, r, c, adjust=adjust),
        conf_high=_cramers_v_from(high, n, r, c, adjust=adjust),
        conf_level=level,
        sidedness=side,
    )


# --------------------------------------------------------------------------- #
# McNemar, Bowker and Cohen's g
# --------------------------------------------------------------------------- #


def _square(counts: IntArray) -> int:
    r, c = counts.shape
    if r != c or r < 2:
        raise ValueError(
            f"a paired table must be square with at least 2 levels (both variables "
            f"on the same levels); the table is {r} x {c}"
        )
    return r


def _off_diagonal_pairs(counts: IntArray) -> list[tuple[int, int, int, int]]:
    """(i, j, n_ij, n_ji) for i < j, as Python integers."""
    r = counts.shape[0]
    rows = counts.tolist()
    return [
        (i, j, int(rows[i][j]), int(rows[j][i]))
        for i in range(r)
        for j in range(i + 1, r)
    ]


def mcnemar(table: CountTable) -> SymmetryTest:
    """McNemar's (1947) test on a 2 x 2 table of paired outcomes, Bowker's (1948)
    test of symmetry on a larger square table, without continuity correction (R:
    ``mcnemar.test(x, correct = FALSE)``, as statsExpressions calls it).

    statistic = sum over i < j of (n_ij - n_ji)^2 / (n_ij + n_ji) on r (r - 1) / 2
    df; for 2 x 2 (b - c)^2 / (b + c) on 1 df. Each term is an exact ratio of
    integers rounded once, the terms summed exactly rounded. p is the upper
    chi-square tail.

    A pair of cells with n_ij + n_ji = 0 makes its term 0 / 0: R returns NaN for
    the statistic and p then (probe-contingency: [[9, 0], [0, 11]] and [[5, 0, 0],
    [0, 5, 2], [0, 1, 5]], df still 1 and 3), and ggstatsplot shows no result. That
    is refused here, naming the empty pairs, rather than dropping them; with no
    discordant pair at all the test is not defined either (ggstatsplot drops the
    subtitle silently: edge_bar_paired_agree).

    Raises:
        ValueError: the table is not square with at least 2 levels, a pair of
            off-diagonal cells is empty (including the case of no discordant pair),
            or a count is negative, not whole or too large.
    """
    counts = _as_counts(table)
    r = _square(counts)
    pairs = _off_diagonal_pairs(counts)
    discordant = sum(a + b for _, _, a, b in pairs)
    if discordant == 0:
        raise ValueError(
            "McNemar's test needs at least one discordant pair (a count off the "
            "diagonal); every pair agrees"
        )
    empty = [(i, j) for i, j, a, b in pairs if a + b == 0]
    if empty:
        raise ValueError(
            "Bowker's test of symmetry is not defined when a pair of off-diagonal "
            f"cells is empty (0 / 0; R returns NaN): cells (row, column) {empty} and "
            "their mirror images hold no counts"
        )
    statistic = math.fsum(((a - b) * (a - b)) / (a + b) for _, _, a, b in pairs)
    df = r * (r - 1) // 2
    return SymmetryTest(
        statistic=statistic,
        df=df,
        p_value=_chisq_sf(statistic, df),
        n_pairs=int(counts.sum()),
        discordant=discordant,
    )


def _wilson(
    successes: int, trials: int, level: float, side: Alternative
) -> tuple[float, float]:
    """Wilson's (1927) score interval of a proportion without continuity correction,
    clipped to [0, 1] (R: ``prop.test(successes, trials, correct = FALSE)$conf.int``;
    Newcombe 1998, Statistics in Medicine 17: 857-872, method 3). Two-sided at
    ``level``, or one-sided ("greater": [low, 1], "less": [0, high]) with z at
    ``level``."""
    p = successes / trials
    alpha = 1.0 - level
    z = -float(special.ndtri(alpha / 2.0 if side == "two-sided" else alpha))
    z2 = z * z
    centre = p + z2 / (2.0 * trials)
    spread = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
    scale = 1.0 + z2 / trials
    low = 0.0 if side == "less" else max(0.0, (centre - spread) / scale)
    high = 1.0 if side == "greater" else min(1.0, (centre + spread) / scale)
    return low, high


def cohens_g(
    table: CountTable,
    *,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Cohen's (1988, Statistical Power Analysis, ch. 5) g of a square table of
    paired outcomes and its interval (R: ``effectsize::cohens_g(x, ci =
    conf_level, alternative)``).

    g = P - 0.5 with P the share of discordant pairs on the more frequent side:
    max(b, c) / (b + c) for a 2 x 2 table, and beyond 2 x 2 the sum over i < j of
    max(n_ij, n_ji) over all discordant pairs (edge_bar_paired3: 13 / 18). The
    interval is Wilson's score interval of P without continuity correction (as
    ``prop.test(correct = FALSE)``; the corrected default misses the oracle) minus
    0.5, so its lower bound can be negative; "greater" gives [low, 0.5], "less"
    [-0.5, high].

    Raises:
        ValueError: the table is not square with at least 2 levels, it holds no
            discordant pair (R: "not enough data"), a count is invalid, or
            conf_level or alternative is invalid.
    """
    counts = _as_counts(table)
    _square(counts)
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    pairs = _off_diagonal_pairs(counts)
    discordant = sum(a + b for _, _, a, b in pairs)
    if discordant == 0:
        raise ValueError(
            "Cohen's g needs at least one discordant pair (a count off the "
            "diagonal); every pair agrees"
        )
    more = sum(max(a, b) for _, _, a, b in pairs)
    low, high = _wilson(more, discordant, level, side)
    return EffectSizeCI(
        estimate=more / discordant - 0.5,
        conf_low=low - 0.5,
        conf_high=high - 0.5,
        conf_level=level,
        sidedness=side,
    )


# --------------------------------------------------------------------------- #
# Goodness of fit, Pearson's C, per-bar tests
# --------------------------------------------------------------------------- #


def _check_ratio(ratio: Probabilities | None, k: int) -> FloatArray:
    """The probabilities as R's ``chisq.test`` takes them: equal shares 1 / k (the
    doubles R computes) by default; otherwise as given, one per category, every
    one positive, summing to 1 within sqrt(2^-52) (never rescaled)."""
    if k < 2:
        raise ValueError(
            f"a goodness-of-fit test needs at least 2 categories; got {k} (R: "
            "\"'x' must at least have 2 elements\")"
        )
    if ratio is None:
        return np.full(k, 1.0 / k)
    p = np.asarray(ratio, dtype=np.float64)
    if p.ndim != 1 or p.size != k:
        raise ValueError(
            f"ratio must give one probability per category: {k} expected, got "
            f"{p.size} (R: \"'x' and 'p' must have the same number of elements\")"
        )
    if not bool(np.isfinite(p).all()) or bool((p <= 0.0).any()):
        raise ValueError(
            f"every probability in ratio must be positive and finite; got {p.tolist()}"
        )
    total = math.fsum(p.tolist())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(
            f"the probabilities in ratio must sum to 1; they sum to {total!r} (they "
            'are not rescaled; R: "probabilities must sum to 1.")'
        )
    return p


def _shares_as_written(ratio: Probabilities | None, p: FloatArray) -> list[Fraction]:
    """The shares of ``ratio`` as written, exactly, for deciding an exact fit.

    ``None`` is exactly 1 / k each. A share given as a double is read as the
    decimal it stands for (``written_values``, DECISIONS D105: 0.32 is 8/25);
    one without such a reading, as the fraction with the smallest denominator up
    to ``_MAX_SHARE_DENOMINATOR`` within ``READ_ULPS`` units in its last place
    (1 / 3 computed in binary is a third, so equal shares written as 1 / k are
    1 / k); otherwise as the double itself.
    """
    if ratio is None:
        return [Fraction(1, p.size)] * p.size
    out: list[Fraction] = []
    for value, reading in zip(p.tolist(), written_values(p), strict=True):
        if reading == Fraction(value):
            simple = Fraction(value).limit_denominator(_MAX_SHARE_DENOMINATOR)
            if abs(simple - reading) <= READ_ULPS * Fraction(math.ulp(value)):
                reading = simple
        # A positive share is never read as 0 (a subnormal within 2 units of it).
        out.append(reading if reading > 0 else Fraction(value))
    return out


def _goodness_of_fit(
    counts: IntArray, p: FloatArray, shares: Sequence[Fraction]
) -> GoodnessOfFit:
    """The test in R's order of operations; exactly 0 where every count equals n
    times its share as written (``_shares_as_written``), where R's order leaves a
    rounding remainder (1.13e-31 for 8, 2, 7, 8 against 0.32, 0.08, 0.28, 0.32)."""
    x = counts.astype(np.float64)
    n = float(x.sum())
    expected = n * p
    total = int(counts.sum())
    exact_fit = all(
        total * share == int(v)
        for v, share in zip(counts.tolist(), shares, strict=True)
    )
    statistic = 0.0
    if not exact_fit:
        with np.errstate(over="ignore"):
            terms = (x - expected) ** 2 / expected
        try:
            statistic = _chisq_sum(terms)
        except OverflowError:  # finite terms whose sum exceeds the doubles
            statistic = math.inf
        if math.isinf(statistic):
            i = int(np.argmax(terms))  # the largest term (the first infinite one)
            raise ValueError(
                f"ratio gives category {i} a share of {float(p[i])!r}, so small next "
                f"to its {int(counts[i])} observations (of {total}) that the "
                "goodness-of-fit statistic exceeds the largest floating-point number "
                "(R reports Inf). A share this small says the category cannot "
                "occur; if so, leave it out of the data and of ratio."
            )
    df = counts.size - 1
    return GoodnessOfFit(
        statistic=statistic,
        df=df,
        p_value=_chisq_sf(statistic, df),
        n=int(counts.sum()),
        expected=tuple(float(v) for v in expected.tolist()),
        ratio=tuple(float(v) for v in p.tolist()),
    )


def goodness_of_fit(
    counts: CountVector, ratio: Probabilities | None = None
) -> GoodnessOfFit:
    """Pearson's (1900) chi-square goodness-of-fit test of k counts against the
    probabilities ``ratio`` (R: ``chisq.test(x, p = ratio)``; no continuity
    correction applies to it).

    chi^2 = sum (x - n p)^2 / (n p) on k - 1 df, p the upper tail. ``ratio`` is in
    the order of the counts, by default equal shares (R's ``rep(1 / k, k)``, the
    doubles nearest 1 / k). It is validated as R validates it and never rescaled: a
    sum further than sqrt(2^-52) from 1 and a wrong length raise, as in R; a zero
    probability raises too (R accepts it and returns an infinite or NaN statistic).
    Every category is kept, a zero count included. Counts that fit the shares as
    written exactly (x_i = n p_i for every i, decided in rational arithmetic on
    ``ratio`` read as ``_shares_as_written`` says) give chi^2 = 0 and p = 1, where
    R's order of operations can leave a rounding remainder.

    Raises:
        ValueError: fewer than 2 categories, no counts at all, an invalid ratio,
            an invalid count, or a share of ratio so small next to its count that
            the statistic exceeds the largest double (a share below about 1e-307;
            R reports Inf). Every finite statistic is computed, with its interval
            (``chi2_ncp_bound``).
    """
    x = _as_counts(counts, ndim=1)
    p = _check_ratio(ratio, x.size)
    if int(x.sum()) == 0:
        raise ValueError("a goodness-of-fit test needs at least one count")
    return _goodness_of_fit(x, p, _shares_as_written(ratio, p))


def _pearsons_c_from(chisq: float, n: int) -> float:
    if math.isinf(chisq):
        return 1.0
    return math.sqrt(chisq / (chisq + n))


def _pearsons_c(test: GoodnessOfFit, level: float, side: Alternative) -> EffectSizeCI:
    low, high = chi2_ncp_interval(
        test.statistic, test.df, conf_level=level, alternative=side
    )
    return EffectSizeCI(
        estimate=_pearsons_c_from(test.statistic, test.n),
        conf_low=_pearsons_c_from(low, test.n),
        conf_high=_pearsons_c_from(high, test.n),
        conf_level=level,
        sidedness=side,
    )


def pearsons_c(
    counts: CountVector,
    ratio: Probabilities | None = None,
    *,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Pearson's (1904) contingency coefficient C of a goodness-of-fit test and its
    interval (R: ``effectsize::pearsons_c(x, p = ratio, ci = conf_level,
    alternative)``, two-sided as statsExpressions calls it by default).

    C = sqrt(chi^2 / (chi^2 + n)) with chi^2 the goodness-of-fit statistic
    (``goodness_of_fit``). The interval inverts the noncentral chi-square at chi^2
    on k - 1 df (``chi2_ncp_interval``) and puts each noncentrality bound lambda
    through sqrt(lambda / (lambda + n)); "greater" gives [low, 1], "less" [0, high].
    The roots are the accurate ones (D99: effectsize's stop short, 0.4987 for 0.4984
    on mtcars cyl).

    Raises:
        ValueError: as ``goodness_of_fit``, or conf_level or alternative is invalid.
    """
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    return _pearsons_c(goodness_of_fit(counts, ratio), level, side)


def proportion_tests(
    table: CountTable, ratio: Probabilities | None = None
) -> tuple[BarTest, ...]:
    """The goodness-of-fit test of each column (bar) of an r x c table against
    ``ratio``, with Pearson's C (ggstatsplot's ``proportion.test``: its
    ``onesample_data()``, which calls ``chisq.test`` and ``pearsons_c`` on the
    counts of x within each level of y).

    One entry per column that holds counts, in column order (ggstatsplot lists them
    in the reverse order of y's levels; the caller orders them). Every row is kept,
    zero counts included. The intervals are always two-sided at 95% and the p-values
    are never adjusted: ggstatsplot passes neither ``alternative``, ``conf.level``
    nor ``p.adjust.method`` to them. With one row the test is not defined
    (chisq.test: "'x' must at least have 2 elements"; ggstatsplot prints a row of NA):
    every entry then has ``test`` and ``effect`` None and a ``reason``.

    Raises:
        ValueError: an invalid count, or (with 2 or more rows) an invalid ratio,
            or a bar's statistic beyond the doubles (``goodness_of_fit``).
    """
    counts = _as_counts(table)
    k, c = counts.shape
    p = None if k < 2 else _check_ratio(ratio, k)
    shares = [] if p is None else _shares_as_written(ratio, p)
    out: list[BarTest] = []
    for j in range(c):
        column = counts[:, j]
        n = int(column.sum())
        if n == 0:
            continue
        if p is None:
            out.append(
                BarTest(
                    column=j,
                    n=n,
                    test=None,
                    effect=None,
                    reason=(
                        "a goodness-of-fit test needs at least 2 categories; x has "
                        "one level"
                    ),
                )
            )
            continue
        test = _goodness_of_fit(column, p, shares)
        out.append(
            BarTest(
                column=j,
                n=n,
                test=test,
                effect=_pearsons_c(test, _BAR_CONF_LEVEL, "two-sided"),
                reason=None,
            )
        )
    return tuple(out)


def column_percentages(table: CountTable) -> tuple[tuple[float | None, ...], ...]:
    """Each cell's share of its column (bar) in percent, as ggbarstats computes it
    for its labels: (count / column total) * 100 in double precision, in that order
    (probe-contingency's 927 percent labels: bit for bit; 100 * count / total and
    the exact share differ from R's double in 81 of them). None for a column with
    no counts. Row by row, like the table.

    The label is this double rounded by R's ``round(x, digits.perc)`` (12.5 gives
    12, 6.25 gives 6.2 at one digit, 0.05 gives 0; not Python's ``round()``, see
    ``inferplot.text.r_round``) and printed without trailing zeros; a zero cell
    gets no label. That belongs to the text layer.
    """
    counts = _as_counts(table)
    totals = counts.sum(axis=0)
    out: list[tuple[float | None, ...]] = []
    for row in counts.tolist():
        out.append(
            tuple(
                None if int(total) == 0 else (float(v) / float(total)) * 100.0
                for v, total in zip(row, totals.tolist(), strict=True)
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# Fisher's exact test of a 2 x c table
# --------------------------------------------------------------------------- #

# The cost of an exact test is counted in work units: elements of the arrays the
# computation processes, each weighted by the passes made over it (the weights
# below), so that a unit is about a nanosecond on the build machine (an Apple
# M-series laptop) and the count does not depend on the machine or its load.
_UNIT_GRID: Final = 35.0  # one child of one node in a column's grid
_UNIT_STEP: Final = 12.0  # one bisection step of one partial table
_UNIT_CHILD: Final = 40.0  # one partial table carried to the next column
_UNIT_LIST: Final = 6.0  # one placement of one completion of the list
_UNIT_BOUND: Final = 10.0  # one element of a column's bound arrays
_UNIT_VERTEX: Final = 3_000.0  # one vertex of the box (smallest completion)
_UNIT_PMF: Final = 40.0  # one hypergeometric log probability (2 x 2)
_UNIT_QUERY: Final = 40.0  # one partial table joined with the list
_UNIT_PIECE: Final = 800.0  # one block of a row of the list (a pass of Python)
_UNIT_CALL: Final = 20_000.0  # one pass of Python over a chunk
# One test may spend this much (about a second here) ...
_FISHER_WORK_LIMIT: Final = 1.0e9
# ... and the pairwise tests of one table this much in all (about two seconds).
_PAIRWISE_WORK_LIMIT: Final = 1.6e9
# A pairwise test whose smaller work estimate (_fisher_work) is above this is not
# attempted: the estimate exceeds the work spent by at most 1.1e4 times (largest
# answered estimate 2.5e12) on the tables of tests/data and the grids measured,
# so the attempt would only spend the budget the other pairs need.
_PAIRWISE_ESTIMATE_LIMIT: Final = 1e5 * _FISHER_WORK_LIMIT
_SPAN_SD: Final = 8.0
# Memory: the arrays of one test, counted in bytes before any is allocated, stay
# below this (every array the search allocates, temporaries included).
_FISHER_MEMORY_LIMIT: Final = 160 * 2**20
# The list side of the search may take this many bytes of it.
_FISHER_LIST_MEMORY: Final = 112 * 2**20
# The grids of the network side are built in chunks of at most this many cells
# (a node's row is never split, so a chunk holds at least one row).
_FISHER_GRID_CHUNK: Final = 1 << 16
# A column whose children per node outnumber this is searched without a grid.
_WIDE_ROW: Final = 4096
# ... whose rows keep the children with masses within 2^-1100 of the largest.
_WINDOW_DROP: Final = 1100.0 * math.log(2.0)
# The partial tables carried from one column to the next are handed on in chunks
# whose total over the columns of the search is at most this many.
_FISHER_PATH_TOTAL: Final = 1 << 17
# Bytes per grid cell (the grid's integer and float arrays and the row sums,
# temporaries included), per carried partial table and per list completion.
_BYTES_GRID: Final = 120
_BYTES_PATH: Final = 200
_BYTES_LIST: Final = 16
# Rows of the list up to this length are sorted and summed together (padded);
# longer ones one at a time. A join groups its partial tables by row when the
# rows average at least this many of them.
_SHORT_ROW: Final = 32
_QUERIES_PER_ROW: Final = 16
# Lists with at most this many rows (totals), and rows of at least this length on
# average, keep each row as its own array.
_ROWWISE_ROWS: Final = 4096
_ROW_LENGTH: Final = 64
# Pruning margin of the network side, relative to 1 + |log threshold| (the log
# probabilities are accurate to about 1e-13 relative): a partial table is summed
# whole or dropped only when its bound clears the threshold by this much; the
# others are carried to the exact comparison at the end.
_FISHER_MARGIN: Final = 1e-9
# Vertex enumeration of the lower bounds runs over 2^(m - 1) m vertices; from this
# many suffix columns on a constant bound is used instead.
_MAX_VERTEX_COLUMNS: Final = 10
# A row sum (in units of the row's largest term) below this may have lost terms
# to underflow and is summed again in log space.
_TINY: Final = 2.0**-900
# Tail sums of the 2 x 2 test: stop when the geometric bound of the rest is below
# this share of the sum; blocks grow from the first size to the second.
_TAIL_EPS: Final = 1e-17
_TAIL_BLOCK: Final = 256
_TAIL_BLOCK_MAX: Final = 1 << 15
# ... and their terms are computed in Loader's form every this many terms, by the
# exact ratio of consecutive terms in between.
_ANCHOR: Final = 256


class _TooLarge(Exception):
    """The exact computation would pass its work or memory limit."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind  # "work" or "memory"


class _Budget:
    """Work units one computation may spend, what it has spent, and the largest
    single memory request (bytes) it has made. ``charge`` is called before the
    step it pays for, so a step that would pass the limit is never started."""

    __slots__ = ("largest", "limit", "spent")

    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.spent = 0.0
        self.largest = 0

    def charge(self, units: float) -> None:
        if self.spent + units > self.limit:
            raise _TooLarge("work")
        self.spent += units

    def request(self, nbytes: float) -> None:
        if nbytes > _FISHER_MEMORY_LIMIT:
            raise _TooLarge("memory")
        self.largest = max(self.largest, int(nbytes))


def _log_ratio(num: int, den: int) -> float:
    """log(num / den) for 0 < num < den, accurate to a few units of 2^-53 relative
    (through log1p when num / den is near 1)."""
    if 2 * num >= den:
        return math.log1p(-(den - num) / den)
    return math.log(num / den)


def _split_mean(size: int, num: int, den: int) -> tuple[float, float]:
    """size * num / den as the nearest double and the rest (exact value minus the
    double), from exact rationals."""
    exact = Fraction(size * num, den)
    hi = float(exact)
    return hi, float(exact - Fraction(hi))


def _log_dbinom(x: FloatArray, size: int, num: int, den: int) -> FloatArray:
    """log P(X = x) for X ~ Binomial(size, p), p = num / den (0 < num < den), in
    Loader's (2000, "Fast and accurate computation of binomial probabilities")
    saddle-point form: stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n p) -
    bd0(n - x, n q) - log(2 pi x (1 - x / n)) / 2, whose absolute error stays near
    the rounding of its largest term whatever the size (no difference of large
    log-factorials). The means n p and n q are taken as exact rationals: bd0(x, m +
    d) = bd0(x, m) + d (m - x) / m to first order, with m the nearest double and d
    the rest; without it the rounding of n p alone gives an error of order 2^-53
    |x - n p|, which grows with the square root of n (1e-12 relative at n = 1e9)."""
    out = np.empty(x.shape, dtype=np.float64)
    n = float(size)
    zero = x == 0.0
    full = x == n
    if bool(zero.any()):
        out[zero] = n * _log_ratio(den - num, den)
    if bool(full.any()):
        out[full] = n * _log_ratio(num, den)
    mid = ~(zero | full)
    if bool(mid.any()):
        xm = x[mid]
        rest = n - xm
        m1, d1 = _split_mean(size, num, den)
        m2, d2 = _split_mean(size, den - num, den)
        lc = (
            _stirlerr(np.full(xm.shape, n))
            - _stirlerr(xm)
            - _stirlerr(rest)
            - (_bd0(xm, np.full(xm.shape, m1)) + d1 * ((m1 - xm) / m1))
            - (_bd0(rest, np.full(xm.shape, m2)) + d2 * ((m2 - rest) / m2))
        )
        lf = _LOG_2PI + np.log(xm) + np.log1p(-xm / n)
        out[mid] = lc - 0.5 * lf
    return out


# --- 2 x 2 ------------------------------------------------------------------ #


def _tail_log_sum(f: _LogPmf, start: int, stop: int, budget: _Budget) -> float:
    """log of sum exp(f(a)) for a from ``start`` to ``stop`` inclusive (either
    direction), where the terms fall away from ``start`` (a tail of a log-concave
    sequence beyond its mode). Summed in blocks of growing size (at most
    ``_TAIL_BLOCK_MAX`` terms, so memory stays small); the rest after a block is at
    most last term x r / (1 - r), r the ratio of the last two terms (the ratios
    fall away from the mode), and the sum stops once that is below 1e-17 of the
    sum. Each block is charged to ``budget`` before it is computed (the terms of
    a block come from _LogPmf.run)."""
    step = 1 if stop >= start else -1
    ref = float(f(np.array([float(start)]))[0])
    total = 0.0
    pos = start
    block = _TAIL_BLOCK
    while True:
        end = pos + step * (block - 1)
        if (end - stop) * step > 0:
            end = stop
        size = abs(end - pos) + 1
        budget.charge(size * _UNIT_PMF + _UNIT_CALL / 20.0)
        values = f.run(pos, size, step) - ref
        total += float(np.sum(np.exp(values)))
        if end == stop:
            break
        if values.size >= 2:
            ratio = math.exp(float(values[-1] - values[-2]))
            if ratio < 1.0:
                rest = math.exp(float(values[-1])) * ratio / (1.0 - ratio)
                if rest <= _TAIL_EPS * total:
                    break
        pos = end + step
        block = min(2 * block, _TAIL_BLOCK_MAX)
    return ref + math.log(total)


class _LogPmf:
    """log P(A = a) of the first cell of a 2 x 2 table with fixed margins (the
    hypergeometric distribution), from Loader's binomial form: P(a) = dbinom(a;
    c0, p) dbinom(r1 - a; c1, p) / dbinom(r1; n, p) for any p, here p = r1 / n."""

    def __init__(self, r1: int, c0: int, c1: int) -> None:
        n = c0 + c1
        self.r1, self.c0, self.c1, self.n = r1, c0, c1, n
        self.log_total = float(_log_dbinom(np.array([float(r1)]), n, r1, n)[0])

    def __call__(self, a: FloatArray) -> FloatArray:
        return (
            _log_dbinom(a, self.c0, self.r1, self.n)
            + _log_dbinom(float(self.r1) - a, self.c1, self.r1, self.n)
            - self.log_total
        )

    def run(self, start: int, count: int, step: int) -> FloatArray:
        """log P(A = a) at a = start, start + step, ... (``count`` terms, step 1
        or -1): Loader's form at every ``_ANCHOR``-th term, and from each such
        anchor on the exact ratio of consecutive terms, P(a + 1) / P(a) = (c0 -
        a)(r1 - a) / ((a + 1)(c1 - r1 + a + 1)), whose factors are integers below
        2^53: each log ratio is within 3 + |log ratio| units of 2^-53 of the true
        one, so a term is within about 256 x 4 x 2^-53 (2e-13) of its log."""
        a = start + step * np.arange(count, dtype=np.float64)
        out = np.empty(count)
        out[::_ANCHOR] = self(a[::_ANCHOR])
        if count == 1:
            return out
        c0, c1, r1 = float(self.c0), float(self.c1), float(self.r1)
        here = a[:-1]
        if step > 0:
            ratio = ((c0 - here) * (r1 - here)) / (
                (here + 1.0) * (c1 - r1 + here + 1.0)
            )
        else:
            ratio = (here * (c1 - r1 + here)) / ((c0 - here + 1.0) * (r1 - here + 1.0))
        steps = np.log(ratio)
        segments = -(-count // _ANCHOR)
        grid = np.zeros(segments * _ANCHOR)
        # grid[s * A + i] (i >= 1) holds the step into term s * A + i.
        grid[1:count] = steps
        grid = grid.reshape(segments, _ANCHOR)
        grid[:, 0] = 0.0
        np.cumsum(grid, axis=1, out=grid)
        rest = np.arange(count) % _ANCHOR != 0
        anchors = np.repeat(out[::_ANCHOR], _ANCHOR)[:count]
        out[rest] = anchors[rest] + grid.reshape(-1)[:count][rest]
        return out


def _first_true(lo: int, hi: int, pred: Callable[[int], bool]) -> int:
    """Smallest a in [lo, hi] with pred(a) true for a monotone predicate (false then
    true); hi + 1 if none."""
    while lo <= hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid - 1
        else:
            lo = mid + 1
    return lo


def _fisher_2x2(
    a: int, r1: int, c0: int, c1: int, side: Alternative, budget: _Budget
) -> float:
    """Fisher's exact p of a 2 x 2 table with first cell a, first row total r1 and
    column totals c0, c1 (all > 0 except possibly a)."""
    n = c0 + c1
    lo, hi = max(0, r1 - c1), min(r1, c0)
    if lo == hi:
        return 1.0
    f = _LogPmf(r1, c0, c1)

    def at(v: int) -> float:
        return float(f(np.array([float(v)]))[0])

    mode = min(max((r1 + 1) * (c0 + 1) // (n + 2), lo), hi)
    if side == "greater":  # P(A >= a)
        if a > mode:
            return min(1.0, math.exp(_tail_log_sum(f, a, hi, budget)))
        if a == lo:
            return 1.0
        return min(1.0, max(0.0, 1.0 - math.exp(_tail_log_sum(f, a - 1, lo, budget))))
    if side == "less":  # P(A <= a)
        if a < mode:
            return min(1.0, math.exp(_tail_log_sum(f, a, lo, budget)))
        if a == hi:
            return 1.0
        return min(1.0, max(0.0, 1.0 - math.exp(_tail_log_sum(f, a + 1, hi, budget))))
    threshold = at(a) + _LOG_TIE

    # f rises on [lo, mode] and falls on [mode + 1, hi]: the tables at most as
    # probable as the observed one are [lo, left] and [right, hi].
    def rarer(v: int) -> bool:
        return at(v) <= threshold

    left = _first_true(lo, mode, lambda v: not rarer(v)) - 1
    right = _first_true(mode + 1, hi, rarer)
    if left >= mode:
        # The most probable table counts, so every table does: p is 1 exactly.
        return 1.0
    parts: list[float] = []
    if left >= lo:
        parts.append(_tail_log_sum(f, left, lo, budget))
    if right <= hi:
        parts.append(_tail_log_sum(f, right, hi, budget))
    return min(1.0, math.fsum(math.exp(v) for v in parts))


def _sd_2x2(r1: int, c0: int, c1: int) -> float:
    """The standard deviation of the first cell of a 2 x 2 table with these margins
    (hypergeometric)."""
    n = c0 + c1
    if n < 2:
        return 0.0
    return math.sqrt(r1 * (n - r1) * c0 * c1 / (n * n * (n - 1.0)))


def _estimate_2x2(r1: int, c0: int, c1: int) -> float:
    """Work units a 2 x 2 test takes at most when its tails stop where the terms
    have fallen below 1e-17 of their sum: about 40 standard deviations of terms,
    and the blocks' doubling at most doubles that."""
    terms = 2.0 * (40.0 * _sd_2x2(r1, c0, c1) + 4.0 * _TAIL_BLOCK)
    return min(terms, float(min(r1, c0) + 1) * 2.0) * _UNIT_PMF


# --- 2 x c ------------------------------------------------------------------ #


def _prefix_rows(e: FloatArray) -> FloatArray:
    """Row-wise sums of the first i entries, i = 0..width (a leading column of
    zeros), of a non-negative array, in blocks of about sqrt(width) entries: the
    relative error of every sum is at most (2 sqrt(width) + 3) 2^-53 (Higham 2002,
    section 4.2, for the blocks and for the sums of their totals)."""
    rows, width = e.shape
    out = np.zeros((rows, width + 1))
    if width == 0:
        return out
    size = max(8, math.isqrt(width) + 1)
    blocks = -(-width // size)
    pad = np.zeros((rows, blocks * size))
    pad[:, :width] = e
    cube = pad.reshape(rows, blocks, size)
    np.cumsum(cube, axis=2, out=cube)
    if blocks > 1:
        carry = np.cumsum(cube[:, :, -1], axis=1)
        cube[:, 1:, :] += carry[:, :-1, None]
    out[:, 1:] = pad[:, :width]
    return out


def _suffix_max(logs: list[FloatArray]) -> FloatArray:
    """max of sum_j logs[j][a_j] over a_j in [0, len - 1] with sum a_j = s, for
    every s: each logs[j] is concave, so the greedy choice of the s largest
    increments is optimal; the sums are evaluated at the chosen counts."""
    increments = np.concatenate([np.diff(v) for v in logs])
    owner = np.concatenate(
        [np.full(v.size - 1, j, dtype=np.int64) for j, v in enumerate(logs)]
    )
    order = np.argsort(-increments, kind="stable")
    chosen = owner[order]
    out = np.zeros(increments.size + 1)
    for j, v in enumerate(logs):
        counts = np.concatenate(([0], np.cumsum(chosen == j)))
        out += v[counts]
    return out


def _suffix_min(logs: list[FloatArray]) -> FloatArray:
    """A lower bound of sum_j logs[j][a_j] with sum a_j = s for every s: the exact
    minimum over the vertices of the box (every a_j at 0 or its maximum but one,
    where a concave function attains its minimum) for up to
    ``_MAX_VERTEX_COLUMNS`` columns, else sum_j min logs[j]."""
    sizes = [v.size - 1 for v in logs]
    total = sum(sizes)
    m = len(logs)
    if m > _MAX_VERTEX_COLUMNS:
        return np.full(total + 1, sum(float(v.min()) for v in logs))
    out = np.full(total + 1, math.inf)
    for free in range(m):
        others = [j for j in range(m) if j != free]
        for mask in range(1 << len(others)):
            base = 0
            value = 0.0
            for bit, j in enumerate(others):
                if mask >> bit & 1:
                    base += sizes[j]
                    value += float(logs[j][-1])
                else:
                    value += float(logs[j][0])
            stop = base + sizes[free] + 1
            out[base:stop] = np.minimum(out[base:stop], value + logs[free])
    return out


def _bsearch(
    grid: FloatArray,
    rows: IntArray,
    lo: IntArray,
    hi: IntArray,
    threshold: FloatArray,
    *,
    above: bool,
) -> IntArray:
    """Per element, the first column index in [lo, hi) of ``grid[rows]`` where the
    value is > threshold (``above``) or <= threshold; hi if none (monotone
    predicate along the searched range)."""
    lo = lo.copy()
    hi = hi.copy()
    active = lo < hi
    while bool(active.any()):
        mid = (lo + hi) // 2
        safe = np.where(active, mid, 0)
        values = grid[rows, safe]
        hit = values > threshold if above else values <= threshold
        hi = np.where(active & hit, mid, hi)
        lo = np.where(active & ~hit, mid + 1, lo)
        active = lo < hi
    return lo


def _order_of(keys: IntArray) -> IntArray:
    """The stable sorting order of small non-negative integers (a radix sort when
    they fit 16 bits)."""
    if keys.size and int(keys.max()) < 2**15:
        return np.argsort(keys.astype(np.int16), kind="stable")
    return np.argsort(keys, kind="stable")


def _log_sum(values: FloatArray) -> float:
    """log sum exp(values) of the finite entries (-inf if none)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -math.inf
    top = float(finite.max())
    return top + math.log(float(np.sum(np.exp(finite - top))))


def _bits(size: float) -> float:
    """Bisection steps over ``size`` entries."""
    return math.ceil(math.log2(size + 1.0))


def _bound_work(sizes: list[int], m: int) -> float:
    """Work units of the bounds of network columns 0..m - 2 (column m - 1 reads
    its bounds off the list): the greedy maximum and the masses over the later
    columns' totals, and the vertices of the smallest."""
    work = 0.0
    for j in range(m - 1):
        later = sizes[j + 1 :]
        rest = float(sum(s - 1 for s in later) + 1)
        work += (len(later) + 2) * rest * _UNIT_BOUND
        if len(later) <= _MAX_VERTEX_COLUMNS:
            vertices = len(later) * 2.0 ** (len(later) - 1)
            work += vertices * (_UNIT_VERTEX + max(later) * _UNIT_BOUND)
    return work


@dataclass(frozen=True, slots=True)
class _Plan:
    """How the search of a 2 x c table is laid out, from its margins alone.

    ``order`` lists the columns (indices into the reduced table) as the search
    takes them: the network side first, then the ``list_columns`` of the list
    side. ``estimate`` is the work the search would take if nothing were pruned
    (an upper bound, used to order the pairwise tests), ``certain`` the work it
    takes whatever is pruned (the list and the bounds), ``memory`` the bytes of
    every array it may allocate at once (a bound, temporaries included)."""

    order: tuple[int, ...]
    list_columns: int
    list_size: int
    list_longest: int
    estimate: float
    certain: float
    memory: float


def _fits(plan: _Plan) -> bool:
    """Whether a plan's certain work and memory fit the limits of one test."""
    return plan.certain <= _FISHER_WORK_LIMIT and plan.memory <= _FISHER_MEMORY_LIMIT


def _list_memory(sizes: list[int]) -> tuple[int, int, float]:
    """The list of the completions over columns with these numbers of values: its
    size, its longest row (one row per total), and the bytes it takes at most
    while it is built (the list, its running sums, the last column's placements,
    and the temporaries of sorting and summing the longest row). A list beyond
    the memory limit by its size alone is not looked at further (its longest row
    is then given as its size)."""
    size = math.prod(sizes)
    rows = sum(sizes) - len(sizes) + 1
    if _rowwise(sizes):
        # row by row: the values and running sums, an array object per row
        nbytes = size * _BYTES_LIST + rows * 4 * 256.0
    else:
        before = size // min(sizes) if len(sizes) > 1 else 0
        nbytes = size * _BYTES_LIST + before * 48.0 + (1 << 18) * 40.0
    if nbytes > _FISHER_MEMORY_LIMIT:
        return size, size, float(nbytes + size * 64.0)
    if sum(sizes) <= 1 << 16:
        counts = np.ones(1)
        for s in sizes:
            counts = np.convolve(counts, np.ones(s))  # exact: counts below 2^53
        longest = int(counts.max())
    else:
        # A row holds at most one first row per choice of the other columns.
        longest = size // max(sizes)
    return size, longest, float(nbytes + longest * 64.0)


def _fisher_plan(widths: list[int], list_columns: int | None = None) -> _Plan:
    """The layout of the search over first rows b_j in [0, widths[j]] (see
    _fisher_2xc): the columns sorted by width (narrowest first); the list side
    takes the narrowest columns, as many as keep the work estimate lowest within
    the list's memory; the network side takes the others, narrowest first.
    ``list_columns`` forces the number of list columns."""
    k = len(widths)
    order = sorted(range(k), key=lambda j: widths[j])
    sizes = [widths[j] + 1 for j in order]
    best: _Plan | None = None
    choices = range(1, k + 1) if list_columns is None else [list_columns]
    for q in choices:
        list_sizes = sizes[:q]
        list_size, longest, list_bytes = _list_memory(list_sizes)
        if list_columns is None and q > 1 and list_bytes > _FISHER_LIST_MEMORY:
            break
        net = sizes[q:]
        m = len(net)
        total = float(sum(s - 1 for s in sizes))
        # Bounds of every network column's completions: three arrays over the
        # totals of the later columns (the greedy maximum passes over each of
        # those columns once).
        bound_cells = 0.0
        remaining = total
        for j in range(m):
            remaining -= net[j] - 1
            bound_cells += remaining + 1.0
        rows = sum(list_sizes) - q + 1
        list_work = _list_work(list_sizes, list_size, longest, rows)
        certain = list_work + _bound_work(net + list_sizes, m) + 2 * _UNIT_CALL
        if m:
            certain += net[0] * _UNIT_GRID
        work = certain
        paths = 1.0
        remaining = total
        for j in range(m):
            nodes = min(paths, remaining + 1.0)
            work += nodes * net[j] * _UNIT_GRID
            work += paths * 2.0 * _bits(net[j]) * _UNIT_STEP + _UNIT_CALL
            remaining -= net[j] - 1
            paths *= net[j]
            work += paths * _UNIT_CHILD
        if m:
            work += paths * (_UNIT_QUERY + 4.0 * _bits(longest) + _UNIT_CHILD)
        memory = (
            list_bytes
            + 8.0 * sum(sizes)  # the columns' log probabilities
            + 24.0 * bound_cells  # the bounds
            + 64.0 * (total + 1.0)  # the greedy maximum's temporaries
            + 4 * max(_FISHER_GRID_CHUNK, max(net, default=1)) * _BYTES_GRID
            + float(_FISHER_PATH_TOTAL * _BYTES_PATH)
        )
        plan = _Plan(
            order=tuple(order[q:] + order[:q]),
            list_columns=q,
            list_size=list_size,
            list_longest=longest,
            estimate=work,
            certain=certain,
            memory=memory,
        )
        # The lowest estimate among the plans whose certain work and memory fit
        # a test (if none fits, the lowest estimate: it is refused).
        if best is None or (_fits(plan), -plan.estimate) > (
            _fits(best),
            -best.estimate,
        ):
            best = plan
    assert best is not None
    return best


class _CompletionList:
    """The list side of the search: every first row of the last columns, grouped
    by its total t (a row of the list per total) and sorted by log probability
    within each row, with the running sums of exp(value - the row's largest,
    ``best``) along each row.

    Built a column at a time, widest first: each value x of the next column
    shifts every row t to row t + x, so a new row is a run of sorted blocks that
    a merging sort finishes (the narrowest column, placed last, leaves the
    fewest runs). Up to ``_ROWWISE_ROWS`` rows, each row is its own array (no
    array longer than the longest row, so the memory a test touches is the
    memory it holds; one large array per list and test would leave the
    allocator's caches of freed blocks to grow); with more rows (a wide column
    on the list side, short rows), one flat array with the offsets ``off``."""

    def __init__(self, logs: list[FloatArray], budget: _Budget) -> None:
        sizes = [v.size for v in logs]
        size, longest, nbytes = _list_memory(sizes)
        budget.request(nbytes)
        count = sum(sizes) - len(sizes) + 1
        budget.charge(_list_work(sizes, size, longest, count))
        logs = sorted(logs, key=lambda v: -v.size)
        self.longest = longest
        self.rowwise = _rowwise(sizes)
        if self.rowwise:
            rows = _build_rows(logs)
            self.rows: list[FloatArray] = rows
            self.cums: list[FloatArray] = [
                _prefix_rows(np.exp(r - r[-1])[None, :])[0, 1:] for r in rows
            ]
            self.best = np.array([float(r[-1]) for r in rows])
            self.worst = np.array([float(r[0]) for r in rows])
            totals = np.array([float(c[-1]) for c in self.cums])
            self.mass = self.best + np.log(totals)
            return
        values = logs[0].copy()
        counts = np.ones(values.size, dtype=np.int64)
        for v in logs[1:]:
            values, counts = _add_column(values, counts, v)
            _sort_rows(values, counts)
        off = np.concatenate(([0], np.cumsum(counts)))
        self.values = values
        self.off = off
        self.best = values[off[1:] - 1]
        self.worst = values[off[:-1]].copy()
        self.cum = _row_sums(values, counts, self.best)
        self.mass = self.best + np.log(self.cum[off[1:] - 1])


def _blocks(sizes: list[int]) -> float:
    """The blocks the list of columns with these numbers of values is built from
    row by row (widest column first): one per row and value of each next column."""
    order = sorted(sizes, reverse=True)
    blocks = 0.0
    totals = order[0]
    for s in order[1:]:
        blocks += (totals + s - 1) * s
        totals += s - 1
    return blocks


def _rowwise(sizes: list[int]) -> bool:
    """Whether the list of the columns with these numbers of values is kept row by
    row: few rows, long on average, built from long blocks (short rows make a
    small flat array, and short blocks cost a pass of Python each)."""
    size = math.prod(sizes)
    rows = sum(sizes) - len(sizes) + 1
    return (
        rows <= _ROWWISE_ROWS
        and size >= _ROW_LENGTH * rows
        and size >= _ROW_LENGTH * _blocks(sizes)
    )


def _build_rows(logs: list[FloatArray]) -> list[FloatArray]:
    """The list's rows (one sorted array per total), a column at a time."""
    rows = [np.array([v]) for v in logs[0].tolist()]
    for column in logs[1:]:
        values = column.tolist()
        new: list[FloatArray] = []
        for t in range(len(rows) + len(values) - 1):
            lo = max(0, t - len(rows) + 1)
            hi = min(len(values) - 1, t)
            row = np.concatenate([rows[t - x] + values[x] for x in range(lo, hi + 1)])
            row.sort(kind="stable")
            new.append(row)
        rows = new
    return rows


def _add_column(
    values: FloatArray, counts: IntArray, column: FloatArray
) -> tuple[FloatArray, IntArray]:
    """The list of the columns so far (rows by total, each sorted) extended by
    one more column with log probabilities ``column``: for each value x, old row
    t goes to new row t + x, after the blocks of old rows t + x, ..., t + 1."""
    rows = counts.size
    new_counts = np.convolve(counts, np.ones(column.size, dtype=np.int64))
    new_off = np.concatenate(([0], np.cumsum(new_counts)))
    old_off = np.concatenate(([0], np.cumsum(counts)))
    # inclusive running count of the old rows, flat beyond the last
    running = np.cumsum(np.concatenate((counts, np.zeros(column.size, dtype=np.int64))))
    row_of = np.repeat(np.arange(rows, dtype=np.int64), counts)
    base = np.arange(values.size, dtype=np.int64) - old_off[row_of]
    base -= running[row_of]
    out = np.empty(int(new_off[-1]))
    for x, value in enumerate(column.tolist()):
        target = row_of + x
        dest = new_off[target] + running[target] + base
        out[dest] = values + value
    return out, new_counts


def _short_row_batches(counts: IntArray) -> list[IntArray]:
    """The rows of at most ``_SHORT_ROW`` entries, in batches small enough that a
    padded block of them stays below 2^18 cells."""
    short = np.flatnonzero(counts <= _SHORT_ROW)
    step = max(1, (1 << 18) // _SHORT_ROW)
    return [short[i : i + step] for i in range(0, short.size, step)]


def _sort_rows(values: FloatArray, counts: IntArray) -> None:
    """Sort each row of the flat list in place: short rows together in padded
    blocks, long rows one at a time (a merging sort, which finishes sorted runs
    in about n log(runs) comparisons)."""
    off = np.concatenate(([0], np.cumsum(counts)))
    for batch in _short_row_batches(counts):
        width = int(counts[batch].max())
        if width < 2:
            continue
        index = off[batch][:, None] + np.arange(width)[None, :]
        valid = np.arange(width)[None, :] < counts[batch][:, None]
        index = np.where(valid, index, off[batch][:, None])
        block = np.where(valid, values[index], math.inf)
        block.sort(axis=1)
        values[index[valid]] = block[valid]
    for t in np.flatnonzero(counts > _SHORT_ROW).tolist():
        values[off[t] : off[t + 1]].sort(kind="stable")


def _row_sums(values: FloatArray, counts: IntArray, best: FloatArray) -> FloatArray:
    """Within each row, the running sums of exp(value - the row's largest), each
    with relative error at most (2 sqrt(row length) + 3) 2^-53 (_prefix_rows)."""
    off = np.concatenate(([0], np.cumsum(counts)))
    cum = np.empty(values.size)
    for batch in _short_row_batches(counts):
        width = int(counts[batch].max())
        index = off[batch][:, None] + np.arange(width)[None, :]
        valid = np.arange(width)[None, :] < counts[batch][:, None]
        index = np.where(valid, index, off[batch][:, None])
        e = np.where(valid, np.exp(values[index] - best[batch][:, None]), 0.0)
        sums = np.cumsum(e, axis=1)
        cum[index[valid]] = sums[valid]
    for t in np.flatnonzero(counts > _SHORT_ROW).tolist():
        row = values[off[t] : off[t + 1]]
        e = np.exp(row - best[t])
        cum[off[t] : off[t + 1]] = _prefix_rows(e[None, :])[0, 1:]
    return cum


def _list_work(sizes: list[int], size: int, longest: int, rows: int) -> float:
    """Work units of building the list: the placements of every column after the
    first (widest first) with the merging sorts, the running sums, and a pass of
    Python per block of a row (row by row) or per long row (flat)."""
    del longest
    order = sorted(sizes, reverse=True)
    placements = 0.0
    built = float(order[0])
    for s in order[1:]:
        built *= s
        placements += built * _bits(s)
    blocks = _blocks(sizes)
    python = (
        blocks * _UNIT_PIECE + rows * _UNIT_CALL / 4.0
        if _rowwise(sizes)
        else min(rows, size / _SHORT_ROW + 1.0) * len(sizes) * _UNIT_CALL
    )
    return (placements + size) * _UNIT_LIST + python


def _concave_peak(
    at: Callable[[IntArray, IntArray], FloatArray], ids: IntArray, length: IntArray
) -> IntArray:
    """Per id, the first x in [0, length) where the concave sequence at(id, .)
    stops rising (at(x + 1) <= at(x)), length - 1 if it never does; by
    bisection on the sign of the differences."""
    lo = np.zeros_like(length)
    hi = length - 1
    active = lo < hi
    while bool(active.any()):
        mid = (lo + hi) // 2
        hit = at(ids, mid + 1) <= at(ids, mid)
        hi = np.where(active & hit, mid, hi)
        lo = np.where(active & ~hit, mid + 1, lo)
        active = lo < hi
    return lo


def _first_where(
    at: Callable[[IntArray, IntArray], FloatArray],
    ids: IntArray,
    lo: IntArray,
    hi: IntArray,
    threshold: FloatArray,
    *,
    above: bool,
) -> IntArray:
    """_bsearch with the values read by ``at`` instead of from a grid."""
    lo = lo.copy()
    hi = hi.copy()
    active = lo < hi
    while bool(active.any()):
        mid = (lo + hi) // 2
        values = at(ids, mid)
        hit = values > threshold if above else values <= threshold
        hi = np.where(active & hit, mid, hi)
        lo = np.where(active & ~hit, mid + 1, lo)
        active = lo < hi
    return lo


class _Network:
    """The search of a 2 x c table (see _fisher_2xc). Columns 0..m - 1 of ``logs``
    form the network side, taken one at a time; the rest the list side."""

    def __init__(
        self,
        logs: list[FloatArray],
        lows: list[int],
        cols: list[int],
        num: int,
        den: int,
        threshold: float,
        list_columns: int,
        budget: _Budget,
    ) -> None:
        k = len(logs)
        m = k - list_columns
        self.logs = logs
        self.m = m
        self.threshold = threshold
        # Per column: the low ends of the ranges and the totals of the columns
        # from it on (the closed-form mass of the completions from column j).
        self.lows = [sum(lows[j:]) for j in range(k)]
        self.cols = [sum(cols[j:]) for j in range(k)]
        self.num = num
        self.den = den
        self.margin = _FISHER_MARGIN * (1.0 + abs(threshold))
        self.budget = budget
        widths = [v.size - 1 for v in logs]
        self.rest = [sum(widths[j:]) for j in range(k + 1)]
        self.parts: list[float] = []
        budget.charge(_bound_work([v.size for v in logs], m))
        self.list = _CompletionList(logs[m:], budget)
        # Per network column j: the largest and smallest log mass of one
        # completion over the later columns, and the log mass of all of them
        # (a sum of binomials with one p is binomial), by the total still to place.
        self.bounds: list[tuple[FloatArray, FloatArray, FloatArray]] = []
        for j in range(m):
            if j + 1 == m:
                self.bounds.append((self.list.best, self.list.worst, self.list.mass))
                continue
            later = logs[j + 1 :]
            total = self.rest[j + 1]
            low = sum(lows[j + 1 :])
            mass = _log_dbinom(
                np.arange(total + 1, dtype=np.float64) + low,
                sum(cols[j + 1 :]),
                num,
                den,
            )
            self.bounds.append((_suffix_max(later), _suffix_min(later), mass))

    def run(self, start: int) -> float:
        rem = np.array([start], dtype=np.int64)
        vals = np.zeros(1)
        if self.m == 0:
            self._final(rem, vals)
        else:
            self._descend(0, rem, vals)
        return _log_sum(np.array(self.parts))

    def _path_chunk(self) -> int:
        return max(1024, _FISHER_PATH_TOTAL // max(1, self.m))

    def _descend(self, j: int, rem: IntArray, vals: FloatArray) -> None:
        """Column j for the partial tables (remaining total, log mass so far),
        sorted by remaining total."""
        here = self.logs[j]
        width = here.size - 1
        last = j + 1 == self.m
        nodes, node_of = np.unique(rem, return_inverse=True)
        node_of = node_of.reshape(-1)
        a_lo = np.maximum(0, nodes - self.rest[j + 1])
        a_hi = np.minimum(width, nodes)
        length = a_hi - a_lo + 1
        longest = int(length.max())
        wide = longest > _WIDE_ROW
        if wide:
            start, window = self._windows(j, nodes, a_lo, length)
        else:
            start, window = np.zeros_like(length), length
        per_chunk = max(1, _FISHER_GRID_CHUNK // int(window.max()))
        cut = self.threshold - self.margin
        keep_below = self.threshold + self.margin
        # With one list column the bound of the last network column is each
        # child's own table: exact.
        exact = last and self.list.longest == 1
        seg_path: list[IntArray] = []
        seg_start: list[IntArray] = []
        seg_len: list[IntArray] = []
        for first in range(0, nodes.size, per_chunk):
            stop = min(nodes.size, first + per_chunk)
            lo_path, hi_path = (int(v) for v in np.searchsorted(node_of, [first, stop]))
            chunk_nodes = nodes[first:stop]
            chunk_lo = a_lo[first:stop]
            chunk_len = length[first:stop]
            row = node_of[lo_path:hi_path] - first
            v = vals[lo_path:hi_path]
            x1, x2, bands, counted = self._grid_tails(
                j,
                chunk_nodes,
                chunk_lo,
                chunk_len,
                (start[first:stop], window[first:stop], wide),
                row,
                v,
                (cut - v, keep_below - v if exact else None),
            )
            self.parts.append(_log_sum(counted))
            # The children between the two counted ends are carried on (those
            # whose every completion is more probable are dropped): x1..x2, or
            # with one list column the two bands x1..y1 - 1 and y2 + 1..x2 within
            # the margin of the threshold (``bands``: y1, y2; the children
            # between are exactly more probable).
            y1, y2 = bands
            paths = np.arange(lo_path, hi_path)
            seg_path.append(np.concatenate((paths, paths)))
            seg_start.append(np.concatenate((x1, y2 + 1)))
            seg_len.append(np.maximum(np.concatenate((y1 - x1, x2 - y2)), 0))
        self._carry(
            j,
            rem,
            vals,
            a_lo[node_of],
            (
                np.concatenate(seg_path),
                np.concatenate(seg_start),
                np.concatenate(seg_len),
            ),
        )

    def _carry(
        self,
        j: int,
        rem: IntArray,
        vals: FloatArray,
        low: IntArray,
        segments: tuple[IntArray, IntArray, IntArray],
    ) -> None:
        """The children of column j's partial tables (``segments``: for each run
        of children, its partial table, first child and length; ``low`` the
        first child's choice per partial table), handed to the next column (or
        the list) in bands of remaining total of at most about
        ``_path_chunk()`` children, each band sorted by remaining total, so that
        a node of the next column is processed once (one band) per call."""
        here = self.logs[j]
        _, worst, _ = self.bounds[j]
        last = j + 1 == self.m
        keep_below = self.threshold + self.margin
        seg_path, seg_start, seg_len = segments
        live = seg_len > 0
        seg_path, seg_start, seg_len = seg_path[live], seg_start[live], seg_len[live]
        if seg_path.size == 0:
            return
        base = rem[seg_path] - low[seg_path]  # remaining total after child 0
        top_rem = base - seg_start  # the largest remaining total of the run
        bottom_rem = top_rem - seg_len + 1
        span = self.rest[j + 1] + 1
        self.budget.charge((seg_path.size * 4.0 + span) * _UNIT_STEP + _UNIT_CALL)
        diff = np.zeros(span + 1, dtype=np.int64)
        np.add.at(diff, bottom_rem, 1)
        np.add.at(diff, top_rem + 1, -1)
        counts = np.cumsum(diff[:-1])
        before = np.cumsum(counts) - counts
        band_of = before // self._path_chunk()
        edges = np.flatnonzero(np.diff(band_of)) + 1
        lows = np.concatenate(([0], edges))
        highs = np.concatenate((edges, [span])) - 1
        for r_lo, r_hi in zip(lows.tolist(), highs.tolist(), strict=True):
            # Within the band: child x of a run has remaining total base - x.
            x_lo = np.maximum(seg_start, base - r_hi)
            x_hi = np.minimum(seg_start + seg_len - 1, base - r_lo)
            size = np.maximum(x_hi - x_lo + 1, 0)
            total = int(size.sum())
            if total == 0:
                continue
            ends = np.cumsum(size)
            starts = ends - size
            piece = self._path_chunk()
            for c0 in range(0, total, piece):
                c1 = min(total, c0 + piece)
                self.budget.request((c1 - c0) * _BYTES_PATH)
                self.budget.charge((c1 - c0) * _UNIT_CHILD + _UNIT_CALL)
                index = np.arange(c0, c1, dtype=np.int64)
                seg = np.searchsorted(ends, index, side="right")
                parent = seg_path[seg]
                choice = low[parent] + x_lo[seg] + (index - starts[seg])
                child_rem = rem[parent] - choice
                child_val = vals[parent] + here[choice]
                keep = child_val + worst[child_rem] <= keep_below
                child_rem = child_rem[keep]
                child_val = child_val[keep]
                del index, seg, parent, choice, keep
                if child_rem.size == 0:
                    continue
                if last:
                    self._final(child_rem, child_val)
                else:
                    order = _order_of(child_rem)
                    self._descend(j + 1, child_rem[order], child_val[order])

    def _windows(
        self, j: int, nodes: IntArray, a_lo: IntArray, length: IntArray
    ) -> tuple[IntArray, IntArray]:
        """In a wide column (more than ``_WIDE_ROW`` children per node), each
        node's window: the children whose mass (log-concave in the child) is
        within 2^-1100 of the node's largest, found by bisection (its first child
        and its length). The mass left out is below 2^-1100 of the node's, so
        under 1e-320 of the total in all."""
        here = self.logs[j]
        _, _, mass = self.bounds[j]
        self.budget.charge(nodes.size * 8.0 * _bits(float(length.max())) * _UNIT_STEP)

        def weight_at(ids: IntArray, x: IntArray) -> FloatArray:
            a = a_lo[ids] + np.minimum(np.maximum(x, 0), length[ids] - 1)
            return here[a] + mass[nodes[ids] - a]

        every = np.arange(nodes.size)
        mode = _concave_peak(weight_at, every, length)
        low = weight_at(every, mode) - _WINDOW_DROP
        start = _first_where(
            weight_at, every, np.zeros_like(mode), mode, low, above=True
        )
        stop = _first_where(weight_at, every, mode, length, low, above=False)
        return start, stop - start

    def _grid_tails(
        self,
        j: int,
        chunk_nodes: IntArray,
        chunk_lo: IntArray,
        chunk_len: IntArray,
        windows: tuple[IntArray, IntArray, bool],
        row: IntArray,
        v: FloatArray,
        limits: tuple[FloatArray, FloatArray | None],
    ) -> tuple[IntArray, IntArray, tuple[IntArray, IntArray], FloatArray]:
        """For the partial tables (node ``row``, log mass ``v``) at a chunk of
        nodes of column j: the children x1..x2 still open, whether none is
        (``closed``), and ``v`` plus the log mass counted at once (the children
        whose every completion counts: both ends of the concave bound, found by
        bisection). The masses of each node's children are summed along a row of
        a grid, over the node's window (``windows``: its first child and length,
        and whether the column is wide; see _windows); in a wide column the bound
        is read at the children the bisections need. ``limits``: the bound at or
        below which every completion counts, and, where the bound is exact (one
        list column), the bound above which none does; ``bands`` (y1, y2) cut the
        open children x1..x2 to those at or below the second (y1 = x2 + 1, y2 =
        x2 without it, or when none is above it)."""
        limit, keep = limits
        here = self.logs[j]
        best, _, mass = self.bounds[j]
        paths = row.size
        longest = int(chunk_len.max())
        start, width, wide = windows

        def bound_at(ids: IntArray, x: IntArray) -> FloatArray:
            a = chunk_lo[ids] + np.minimum(np.maximum(x, 0), chunk_len[ids] - 1)
            return here[a] + best[chunk_nodes[ids] - a]

        span = int(width.max())
        self.budget.request(chunk_nodes.size * span * _BYTES_GRID)
        self.budget.charge(
            chunk_nodes.size * span * _UNIT_GRID
            + paths * 2.0 * _bits(longest) * _UNIT_STEP
            + _UNIT_CALL
        )
        offsets = np.arange(span)
        valid = offsets[None, :] < width[:, None]
        a = chunk_lo[:, None] + start[:, None] + np.where(valid, offsets[None, :], 0)
        rest = chunk_nodes[:, None] - a
        weight = np.where(valid, here[a] + mass[rest], -math.inf)
        bound = None if wide else np.where(valid, here[a] + best[rest], -math.inf)
        del rest, a, valid, offsets
        top = weight.max(axis=1)
        e = np.exp(weight - top[:, None])
        prefix = _prefix_rows(e)
        suffix = _prefix_rows(e[:, ::-1])
        del e
        size = chunk_len[row]
        if bound is None:
            peak = _concave_peak(bound_at, row, size)

            def first(
                lo: IntArray, hi: IntArray, at: FloatArray, above: bool
            ) -> IntArray:
                return _first_where(bound_at, row, lo, hi, at, above=above)

        else:
            peak = np.argmax(bound, axis=1)[row]
            grid = bound

            def first(
                lo: IntArray, hi: IntArray, at: FloatArray, above: bool
            ) -> IntArray:
                return _bsearch(grid, row, lo, hi, at, above=above)

        zeros = np.zeros_like(peak)
        x1 = first(zeros, peak + 1, limit, True)
        x2 = first(peak, size, limit, False) - 1
        closed = x1 > peak
        y1 = np.where(closed, x1, x2 + 1)
        y2 = x2.copy()
        if keep is not None:
            self.budget.charge(paths * 2.0 * _bits(longest) * _UNIT_STEP)
            above = first(np.minimum(x1, peak + 1), peak + 1, keep, True)
            below = first(peak, np.maximum(x2 + 1, peak), keep, False) - 1
            split = ~closed & (above <= peak)
            y1 = np.where(split, above, y1)
            y2 = np.where(split, below, y2)
        # The counted ends in the window's coordinates.
        lo = start[row]
        room = width[row]
        left_end = np.where(closed, room, np.clip(x1 - lo, 0, room))
        right_from = np.where(closed, room, np.clip(x2 + 1 - lo, 0, room))
        left = prefix[row, left_end]
        right = suffix[row, np.clip(span - right_from, 0, span)]
        right = np.where(right_from < room, right, 0.0)
        tail = left + right
        with np.errstate(divide="ignore"):
            counted = v + top[row] + np.log(tail)
        some = (left_end > 0) | (right_from < room)
        for i in np.flatnonzero((tail < _TINY) & some).tolist():
            # The terms of this tail may have underflowed: sum it in log space.
            w = weight[int(row[i])]
            pieces = np.concatenate(
                (w[: int(left_end[i])], w[int(right_from[i]) : int(room[i])])
            )
            counted[i] = float(v[i]) + _log_sum(pieces)
        return x1, x2, (y1, y2), counted

    def _final(self, rem: IntArray, vals: FloatArray) -> None:
        """The exact comparison: for each partial table, the completions in the
        list whose log mass is at most threshold - its own (a prefix of the
        sorted row of its remaining total, found by bisection), and their mass
        from the row's running sums."""
        lst = self.list
        size = rem.size
        order = _order_of(rem)
        rem = rem[order]
        vals = vals[order]
        first = np.flatnonzero(np.concatenate(([True], rem[1:] != rem[:-1])))
        rows = rem[first]
        if lst.rowwise:
            self._final_rows(rem, vals, first, rows)
            return
        lo = lst.off[rem]
        if rows.size * _QUERIES_PER_ROW > size:
            limit = self.threshold - vals
            self.budget.charge(
                size * (_bits(lst.longest) * _UNIT_STEP + _UNIT_QUERY) + _UNIT_CALL
            )
            hi = lst.off[rem + 1]
            grid = lst.values[None, :]
            index = _bsearch(grid, np.zeros_like(rem), lo, hi, limit, above=True)
            count = index - lo
        else:
            self.budget.charge(
                size * (_UNIT_QUERY + 4.0 * _bits(lst.longest))
                + rows.size * _UNIT_CALL / 4.0
            )
            count = np.empty(size, dtype=np.int64)
            bounds = np.concatenate((first, [size])).tolist()
            for t, a, b in zip(rows.tolist(), bounds[:-1], bounds[1:], strict=True):
                # Within a row only the sum matters: by decreasing value, the
                # limits come in increasing order, and a sorted search reuses
                # its last place.
                vals[a:b] = np.sort(vals[a:b])[::-1]
                row = lst.values[lst.off[t] : lst.off[t + 1]]
                count[a:b] = np.searchsorted(row, self.threshold - vals[a:b], "right")
        tail = np.where(count > 0, lst.cum[np.maximum(lo + count - 1, 0)], 0.0)
        with np.errstate(divide="ignore"):
            counted = vals + lst.best[rem] + np.log(tail)
        for i in np.flatnonzero((tail < _TINY) & (count > 0)).tolist():
            start = int(lo[i])
            pieces = lst.values[start : start + int(count[i])]
            counted[i] = float(vals[i]) + _log_sum(pieces)
        self.parts.append(_log_sum(counted))

    def _final_rows(
        self, rem: IntArray, vals: FloatArray, first: IntArray, rows: IntArray
    ) -> None:
        """_final with the list kept row by row: the partial tables of each row
        (``rem`` sorted, ``first`` where each row's run starts)."""
        lst = self.list
        size = rem.size
        self.budget.charge(
            size * (_UNIT_QUERY + 4.0 * _bits(lst.longest))
            + rows.size * _UNIT_CALL / 4.0
        )
        counted = np.empty(size)
        bounds = np.concatenate((first, [size])).tolist()
        for t, a, b in zip(rows.tolist(), bounds[:-1], bounds[1:], strict=True):
            # Within a row only the sum matters: by decreasing value, the limits
            # come in increasing order, and a sorted search reuses its last place.
            v = np.sort(vals[a:b])[::-1]
            row: FloatArray = lst.rows[int(t)]
            count = np.searchsorted(row, self.threshold - v, "right")
            cum: FloatArray = lst.cums[int(t)]
            tail = np.where(count > 0, cum[np.maximum(count - 1, 0)], 0.0)
            with np.errstate(divide="ignore"):
                part = v + float(lst.best[t]) + np.log(tail)
            for i in np.flatnonzero((tail < _TINY) & (count > 0)).tolist():
                part[i] = float(v[i]) + _log_sum(row[: int(count[i])])
            counted[a:b] = part
        self.parts.append(_log_sum(counted))


def _fisher_2xc(
    first: list[int], cols: list[int], budget: _Budget, list_columns: int | None = None
) -> float:
    """Two-sided Fisher exact p of a 2 x c table, c >= 3, every column non-empty,
    both rows non-empty (fisher_exact describes the search). ``list_columns``
    overrides the plan's split (tests check that every split gives the same p)."""
    n = sum(cols)
    r1 = sum(first)
    r2 = n - r1
    lows = [max(0, c - r2) for c in cols]
    widths = [min(c, r1) - lo for c, lo in zip(cols, lows, strict=True)]
    plan = _fisher_plan(widths, list_columns)
    budget.request(plan.memory)
    if plan.certain > budget.limit - budget.spent:
        raise _TooLarge("work")
    order = list(plan.order)
    logs = [
        _log_dbinom(
            np.arange(widths[j] + 1, dtype=np.float64) + lows[j], cols[j], r1, n
        )
        for j in order
    ]
    obs = [first[j] - lows[j] for j in order]
    log_total = float(_log_dbinom(np.array([float(r1)]), n, r1, n)[0])
    threshold = math.fsum(float(v[b]) for v, b in zip(logs, obs, strict=True))
    threshold += _LOG_TIE
    start = r1 - sum(lows)
    if threshold >= float(_suffix_max(logs)[start]):
        # The most probable table counts, so every table does: p is 1 exactly.
        return 1.0
    log_tables = sum(math.log(w + 1.0) for w in widths)
    if log_tables + threshold - log_total < -746.0:
        # p <= (number of tables) P_obs (1 + 1e-7) < 2^-1075: it rounds to 0.
        return 0.0
    search = _Network(
        logs,
        [lows[j] for j in order],
        [cols[j] for j in order],
        r1,
        n,
        threshold,
        plan.list_columns,
        budget,
    )
    log_p = search.run(start) - log_total
    return min(1.0, math.exp(log_p))


# --- the test ---------------------------------------------------------------- #


def _canonical(first: list[int], second: list[int]) -> tuple[list[int], list[int]]:
    """The table with its columns sorted and its rows in a fixed order (the
    two-sided p does not depend on either), so that a table and its permutations
    give the same p to the last bit and the same work."""
    one = sorted(zip(first, second, strict=True))
    two = sorted(zip(second, first, strict=True))
    chosen = min(one, two)
    return [a for a, _ in chosen], [b for _, b in chosen]


def _fisher_work(table: IntArray, side: Alternative) -> tuple[float, float, float]:
    """Work units of fisher_exact on a 2 x c table, from the margins and the
    observed table: two estimates of the work after pruning, the plan's estimate
    with each column's range cut to 8 + sqrt(X^2) standard deviations either side
    of its mean (X^2 Pearson's statistic: where the tables the search cannot
    settle in bulk lie; for the order of the pairwise tests) and to 8 standard
    deviations (smaller; for the cutoff), and the work spent whatever is pruned."""
    rows = table.tolist()
    first = [int(v) for v in rows[0]]
    cols = [int(a) + int(b) for a, b in zip(rows[0], rows[1], strict=True)]
    kept = [j for j, c in enumerate(cols) if c > 0]
    n = sum(cols)
    r1 = sum(first)
    if len(kept) < 2 or r1 in (0, n):
        return 0.0, 0.0, 0.0
    if len(kept) == 2:
        estimate = _estimate_2x2(r1, cols[kept[0]], cols[kept[1]])
        return estimate, estimate, 0.0
    r2 = n - r1
    widths = [min(cols[j], r1) - max(0, cols[j] - r2) for j in kept]
    chi2 = sum(
        (first[j] - cols[j] * r1 / n) ** 2 / (cols[j] * r1 * r2 / n**2) for j in kept
    )
    estimates: list[float] = []
    for reach in (_SPAN_SD + math.sqrt(chi2), _SPAN_SD):
        spans: list[int] = []
        for j, w in zip(kept, widths, strict=True):
            c = cols[j]
            sd = math.sqrt(c * (r1 / n) * (r2 / n) * (n - c) / max(n - 1, 1))
            spans.append(min(w, 2 * math.ceil(reach * sd)))
        estimates.append(_fisher_plan(spans).estimate)
    return estimates[0], estimates[1], _fisher_plan(widths).certain


def _fisher(table: IntArray, side: Alternative, budget: _Budget) -> FisherExact:
    """fisher_exact on a validated 2 x c table, within ``budget`` (raises
    _TooLarge)."""
    rows = table.tolist()
    first = [int(v) for v in rows[0]]
    second = [int(v) for v in rows[1]]
    cols = [a + b for a, b in zip(first, second, strict=True)]
    n = sum(cols)
    kept = [j for j, c in enumerate(cols) if c > 0]
    r1 = sum(first)
    if len(kept) < 2 or r1 == 0 or r1 == n:
        return FisherExact(p_value=1.0, alternative=side, n=n, columns=len(kept))
    first = [first[j] for j in kept]
    second = [second[j] for j in kept]
    if side == "two-sided":
        first, second = _canonical(first, second)
    cols = [a + b for a, b in zip(first, second, strict=True)]
    r1 = sum(first)
    if len(kept) == 2:
        # A 2 x 2 table (as given, or with empty columns left out: then two-sided).
        p_value = _fisher_2x2(first[0], r1, cols[0], cols[1], side, budget)
        return FisherExact(p_value=p_value, alternative=side, n=n, columns=2)
    p_value = _fisher_2xc(first, cols, budget)
    return FisherExact(p_value=p_value, alternative=side, n=n, columns=len(kept))


def _check_fisher_table(counts: IntArray, side: Alternative) -> None:
    if counts.shape[0] != 2 or counts.shape[1] < 2:
        raise ValueError(
            f"Fisher's exact test here takes a 2 x c table with c >= 2; the table "
            f"is {counts.shape[0]} x {counts.shape[1]}"
        )
    if side != "two-sided" and counts.shape != (2, 2):
        raise ValueError(
            "a one-sided Fisher test is defined for a 2 x 2 table only; this table "
            f"is 2 x {counts.shape[1]} (R ignores alternative there): use "
            "alternative='two-sided'"
        )


def fisher_exact(
    table: CountTable, *, alternative: Alternative = "two-sided"
) -> FisherExact:
    """Fisher's (1935) exact test of a 2 x c table, exact (no Monte Carlo) (R:
    ``fisher.test(x, alternative)``).

    Given both margins, a table's probability is prod_j C(c_j, a_j) / C(n, r_1)
    (a_j the first row, c_j the column totals). The two-sided p sums the
    probabilities of every table at most as probable as the observed one, where,
    as in R, "at most" allows a relative 1e-7: P <= P_obs (1 + 1e-7), so tables
    that tie with the observed one (mirror images) always count. For a 2 x 2 table
    ``alternative`` "greater" gives P(A >= a), "less" P(A <= a) for the first cell
    a; a larger table takes only "two-sided" (R ignores ``alternative`` there).
    The two-sided p does not depend on the order of the rows or the columns, and
    the table is put in one fixed order first, so permuted tables give the same p
    to the last bit.

    Accuracy. Probabilities are compared in log space: each log P is a sum of
    Loader's (2000) saddle-point log binomial probabilities (with the binomial
    means as exact rationals), accurate to a few units of 2^-53 times the largest
    term, so a table is classified differently from the exact rational comparison
    only if its probability lies within about 1e-13 relative of P_obs (1 + 1e-7),
    far inside R's own allowance of 1e-7. The masses are summed with a relative
    error of at most about 2 sqrt(m) 2^-53 for a sum of m terms (blocked sums of
    terms scaled by their largest; terms that would underflow are summed in log
    space), so the p-value's relative error is of order 1e-13 (measured against
    exact rational enumeration: a few units of 1e-15 on small tables; against
    40-digit sums of the 2 x 2 tails, see tests/stats/test_fisher_cost.py).

    2 x 2: the probabilities are hypergeometric and log-concave in a, so the
    tables at most as probable form both ends of the range; their bounds are found
    by bisection and each end is summed outward in blocks until a geometric bound
    of the rest falls below 1e-17 of the sum: about 10 to 40 standard deviations of
    terms, each from the exact ratio of consecutive terms between Loader anchors
    every 256 terms (_LogPmf.run). With balanced margins the two-sided test is
    answered up to about n = 1e13 within the work limit below (n = 1e12: 0.2 s;
    1e13: 0.5 s), a one-sided one up to about 1e14; unbalanced margins go
    further. Beyond, the test is refused having spent at most the limit (about
    0.9 s at n = 1e15). Accuracy against 40-digit sums: 9e-15 relative or better
    from n = 10 to 1e10, two-sided and one-sided.

    2 x c, c >= 3: a table is its first row b (each b_j in the range its column
    allows, with a fixed total), and its log probability is a sum over the columns
    of concave functions of b_j. The columns are split in two sides, as the work
    estimate from the margins is lowest within the limits. The list side (the
    narrowest columns) lists every first row of its columns, grouped by total and
    sorted by log probability, with running sums of the masses. The network side
    takes the other columns one at a time in the manner of Mehta & Patel's (1983,
    Journal of the American Statistical Association 78: 427-434) network
    algorithm: for each partial first row the largest and smallest log
    probability of any completion are known (the largest by the greedy choice of
    increments, the log probabilities being concave in each count; the smallest
    at a vertex of the box, or read off the list), so the children of a partial
    row whose every completion counts are summed at once (the mass of all
    completions is binomial; such children form both ends of a concave sequence,
    found by bisection), children whose every completion is more probable are
    dropped, and only the others are carried on. A column with more than 4096
    children per node keeps, per node, the children whose mass is within 2^-1100
    of the largest (the rest is below 1e-320 of the total). After the last
    network column every partial row left is joined with the list (meet in the
    middle, Horowitz & Sahni 1974): the completions that count are a prefix of
    the sorted row of its remaining total, found by bisection, and their mass is
    that prefix's running sum. Only this last comparison decides a table that
    lies near the threshold, and it compares the table's own log probability.
    Two shortcuts are exact: p is 1 when the most probable table counts, and 0
    when the number of tables times P_obs (1 + 1e-7) is below 2^-1075.

    Cost. The work is counted in units of about a nanosecond on the build machine
    (array elements times the passes over them; machine-independent), and the
    memory in bytes of every array the search may hold at once, both before the
    step that spends them. A test may take 1e9 units (about a second) and 160 MB.
    The bytes, and the work that does not depend on pruning (the list and the
    bounds), follow from the margins and are checked before anything is
    allocated; the rest is counted as the search goes, so a test that would pass
    the limit raises ValueError having spent at most the limit. No array of the
    list is longer than its longest row when it is kept row by row (one large
    array per test would leave the allocator's caches of freed blocks to grow).
    The work grows with the number of columns, with n and with how far the table
    lies from independence: 2 x 8 at n = 300 takes about 0.1 s, 2 x 10 at n = 200
    0.1 to 0.4 s, 2 x 4 at n = 100,000 0.1 s; 2 x 10 at n = 1,000, 2 x 6 at n =
    10,000 far from independence and 2 x 5 at n = 100,000 are refused
    (tests/stats/test_fisher_cost.py).

    Columns without counts do not change the test and are left out; a table with
    one non-empty column or one empty row has a single table with its margins, p 1
    (R: [[0, 0], [3, 4]] gives 1).

    Raises:
        ValueError: the table is not 2 x c with c >= 2, a count is invalid,
            ``alternative`` is one-sided for a table that is not 2 x 2, or the
            exact computation is beyond the limits above.
    """
    counts = _as_counts(table)
    side = check_alternative(alternative)
    _check_fisher_table(counts, side)
    budget = _Budget(_FISHER_WORK_LIMIT)
    try:
        return _fisher(counts, side, budget)
    except _TooLarge as error:
        raise ValueError(_too_large_message(counts, error.kind)) from None


def _too_large_message(counts: IntArray, kind: str) -> str:
    cols = counts.sum(axis=0)
    c = int((cols > 0).sum())
    n = int(cols.sum())
    what = (
        "more memory than inferplot allows one test (160 MB)"
        if kind == "memory"
        else "more work than inferplot allows one test (about a second)"
    )
    if c == 2 or n >= 50 * c * c:
        advice = (
            "at this many observations the chi-square test of the table answers "
            "the same question"
        )
    else:
        advice = "merge sparse columns (levels with few counts), or compare fewer"
    return (
        f"Fisher's exact test of this 2 x {c} table (n = {n:,}) would take {what} "
        f"to compute exactly, and inferplot does not approximate it: {advice}"
    )


# --------------------------------------------------------------------------- #
# Pairwise Fisher tests
# --------------------------------------------------------------------------- #

_TOO_LARGE_REASON: Final = "too large to compute exactly"


def _pair_table(counts: IntArray, i: int, j: int) -> tuple[tuple[int, ...], IntArray]:
    pair = counts[[i, j]]
    columns = tuple(int(c) for c in np.flatnonzero(pair.sum(axis=0) > 0))
    return columns, pair[:, list(columns)]


def _pair_effect(
    sub: IntArray, level: float, side: Alternative
) -> tuple[EffectSizeCI | None, str | None]:
    if int(sub.sum()) <= max(sub.shape):
        return None, (
            "the bias-corrected Cramer's V needs more counts than columns (R: NaN)"
        )
    return cramers_v(sub, conf_level=level, alternative=side), None


def _pair_key(sub: IntArray, side: Alternative) -> tuple[object, ...]:
    """Pairs with equal keys have the same p (the two-sided p does not depend on
    the order of rows and columns; a one-sided 2 x 2 test does)."""
    first = [int(v) for v in sub[0]]
    second = [int(v) for v in sub[1]]
    if side == "two-sided":
        first, second = _canonical(first, second)
    return (side, tuple(first), tuple(second))


def _pairwise(
    counts: IntArray,
    level: float,
    side: Alternative,
    budget: _Budget,
) -> list[PairComparison]:
    """The pairs of rows with their p (or the reason there is none) and V, before
    adjustment; the Fisher tests share ``budget``."""
    present = [i for i in range(counts.shape[0]) if int(counts[i].sum()) > 0]
    pairs: list[tuple[int, int, tuple[int, ...], IntArray]] = []
    for index, i in enumerate(present):
        for j in present[index + 1 :]:
            columns, sub = _pair_table(counts, i, j)
            pairs.append((i, j, columns, sub))
    # The tests, cheapest first by their estimated work, each distinct table once;
    # the order depends on the tables only, never on the order of the rows.
    tests: dict[
        tuple[object, ...], tuple[float, float, float, IntArray, Alternative]
    ] = {}
    for _, _, columns, sub in pairs:
        if len(columns) < 2:
            continue
        used: Alternative = side if sub.shape == (2, 2) else "two-sided"
        key = _pair_key(sub, used)
        if key not in tests:
            estimate, least, certain = _fisher_work(sub, used)
            tests[key] = (estimate, least, certain, sub, used)
    results: dict[tuple[object, ...], float | None] = {}
    for key in sorted(tests, key=lambda k: (tests[k][0], repr(k))):
        _, least, certain, sub, used = tests[key]
        remaining = budget.limit - budget.spent
        if certain > remaining or least > _PAIRWISE_ESTIMATE_LIMIT:
            results[key] = None
            continue
        one = _Budget(min(_FISHER_WORK_LIMIT, remaining))
        try:
            results[key] = _fisher(sub, used, one).p_value
        except _TooLarge:
            results[key] = None
        budget.spent += one.spent
        budget.largest = max(budget.largest, one.largest)
    out: list[PairComparison] = []
    for i, j, columns, sub in pairs:
        n = int(sub.sum())
        if len(columns) < 2:
            out.append(
                PairComparison(
                    row1=i,
                    row2=j,
                    columns=columns,
                    n=n,
                    p_value=None,
                    p_adjusted=None,
                    alternative=None,
                    effect=None,
                    reason=(
                        "the two levels occur in one column only, so their table has "
                        "a single column and no test (R: fisher.test stops)"
                    ),
                    effect_reason=None,
                )
            )
            continue
        used = side if sub.shape == (2, 2) else "two-sided"
        p_value = results[_pair_key(sub, used)]
        effect, effect_reason = _pair_effect(sub, level, side)
        out.append(
            PairComparison(
                row1=i,
                row2=j,
                columns=columns,
                n=n,
                p_value=p_value,
                p_adjusted=None,
                alternative=used if p_value is not None else None,
                effect=effect,
                reason=None if p_value is not None else _TOO_LARGE_REASON,
                effect_reason=effect_reason,
            )
        )
    return out


def _pairwise_fisher(
    table: CountTable,
    p_adjust_method: PAdjustMethod,
    conf_level: float,
    alternative: Alternative,
    budget: _Budget,
) -> tuple[PairComparison, ...]:
    counts = _as_counts(table)
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    pairs = _pairwise(counts, level, side, budget)
    raw = [pair.p_value for pair in pairs if pair.p_value is not None]
    # A pair too large to compute is still a comparison of the family: it enters
    # the adjustment as p = 1, so that leaving it out never makes a computed pair
    # look more significant (pairwise_fisher's docstring).
    uncomputed = sum(1 for pair in pairs if pair.reason == _TOO_LARGE_REASON)
    adjusted = iter(
        p_adjust([*raw, *[1.0] * uncomputed], p_adjust_method)[: len(raw)]
        if raw
        else ()
    )
    out = [
        pair
        if pair.p_value is None
        else dataclasses.replace(pair, p_adjusted=next(adjusted))
        for pair in pairs
    ]
    return tuple(out)


def pairwise_fisher(
    table: CountTable,
    *,
    p_adjust_method: PAdjustMethod = "holm",
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> tuple[PairComparison, ...]:
    """Fisher's exact test between every two rows (levels of x) that hold counts,
    with adjusted p-values and each pair's Cramer's V (ggstatsplot 1.1.4's pairwise
    table for ggbarstats: ``fisher.test`` on each pair, ``p.adjust`` over the pairs,
    ``effectsize::cramers_v``).

    Pairs come in row order (i < j over the rows with counts). Each pair is tested
    on its own 2 x c' table: the two rows, and the columns where they hold counts.
    ``alternative`` reaches the test only where that table is 2 x 2 (as
    ``fisher.test``; the pair records the alternative used) and reaches every
    pair's V interval, at ``conf_level``. ``p_adjust_method`` adjusts the p-values
    of the pairs that have a test (``p_adjust``); a pair whose test is too large
    to compute (below) counts in that family with p = 1 (as ``p.adjust(p, n =)``
    allows for comparisons whose p is not at hand).

    Cost. The tests of one table share a budget of 1.6e9 work units (about two
    seconds; fisher_exact describes the units), and each test is also held to
    fisher_exact's own limit. They are computed cheapest first by their estimated
    work (from the margins and the observed table: _fisher_work), each distinct
    table once (a table and its permutations have the same p). A test is not
    computed when its certain work exceeds what is left, when its estimate is
    above 1e14 units (1e5 times a test's limit; the largest estimate of a test
    answered in the tables measured is 2.5e12), or when it would pass what is
    left. So the call returns after at
    most about two seconds of tests on any table, and which pairs are tested
    depends on the tables only, never on the order of the rows.

    A pair whose table keeps one column has no test (fisher.test stops; ggstatsplot
    loses the whole pairwise table silently, edge_bar_degenerate-single-x), and a
    pair whose test does not fit the budget has none either (``reason`` "too large
    to compute exactly"): both are returned with ``p_value`` None and a
    ``reason``. A single-column pair is no comparison and is left out of the
    adjustment. A pair too large to compute is one: the adjustment counts it with
    p = 1, the cautious choice, because which pairs fit the budget depends on the
    data (for Holm, Hochberg, Bonferroni and the false-discovery methods no
    computed pair then has a smaller adjusted p than it would have with every pair
    computed). A pair whose V is not
    defined (no more counts than columns) has ``effect`` None and an
    ``effect_reason``.

    Raises:
        ValueError: an invalid count, ``conf_level``, ``alternative`` or
            ``p_adjust_method``.
    """
    return _pairwise_fisher(
        table,
        p_adjust_method,
        conf_level,
        alternative,
        _Budget(_PAIRWISE_WORK_LIMIT),
    )
