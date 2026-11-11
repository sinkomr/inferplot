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
# Fisher's test of a 2 x c table (c >= 3): the largest number of array elements the
# search may touch (node grids, expanded tables, binary-search steps) before it
# gives up. Measured on an Apple M-series laptop: about 3e7 elements per second,
# so this is about two seconds (see fisher_exact).
_FISHER_WORK_LIMIT: Final = 6.0e7
# ... and the largest n (c - 2) for which the per-column arrays are built: the
# bounds and masses of every suffix of columns take about 24 n (c - 2) bytes, so
# about 100 MB here (2 x 3 up to n = 4e6; 2 x 5 up to n = 1.3e6).
_FISHER_SIZE_LIMIT: Final = 4_000_000
# The search builds its per-node grids in chunks of at most this many elements.
_FISHER_CHUNK: Final = 1 << 18
# Pruning margin of the search, relative to 1 + |log threshold| (the log
# probabilities are accurate to about 1e-14 relative).
_FISHER_MARGIN: Final = 1e-9
# Vertex enumeration of the lower bounds runs over 2^(m - 1) m vertices; from this
# many suffix columns on a constant bound is used instead.
_MAX_VERTEX_COLUMNS: Final = 10
# Tail sums of the 2 x 2 test: stop when the geometric bound of the rest is below
# this share of the sum.
_TAIL_EPS: Final = 1e-17
_TAIL_BLOCK: Final = 256


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
    over the pairs that have a test; ``effect`` the pair's Cramer's V (adjusted)
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


def _log_dbinom(x: FloatArray, size: int, p: float, q: float) -> FloatArray:
    """log P(X = x) for X ~ Binomial(size, p), q = 1 - p given separately, in
    Loader's (2000, "Fast and accurate computation of binomial probabilities")
    saddle-point form: stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n p) -
    bd0(n - x, n q) - log(2 pi x (1 - x / n)) / 2, whose absolute error stays near
    the rounding of its largest term whatever the size (no difference of large
    log-factorials)."""
    out = np.empty(x.shape, dtype=np.float64)
    n = float(size)
    zero = x == 0.0
    full = x == n
    out[zero] = n * math.log(q)
    out[full] = n * math.log(p)
    mid = ~(zero | full)
    if bool(mid.any()):
        xm = x[mid]
        rest = n - xm
        nn = np.full(xm.shape, n)
        lc = (
            _stirlerr(nn)
            - _stirlerr(xm)
            - _stirlerr(rest)
            - _bd0(xm, np.full(xm.shape, n * p))
            - _bd0(rest, np.full(xm.shape, n * q))
        )
        lf = _LOG_2PI + np.log(xm) + np.log1p(-xm / n)
        out[mid] = lc - 0.5 * lf
    return out


class _TooLarge(Exception):
    """The exact search would exceed its work limit."""


def _tail_log_sum(f: _LogPmf, start: int, stop: int) -> float:
    """log of sum exp(f(a)) for a from ``start`` to ``stop`` inclusive (either
    direction), where the terms fall away from ``start`` (a tail of a log-concave
    sequence beyond its mode). Summed in blocks of growing size; the rest after a
    block is at most last term x r / (1 - r), r the ratio of the last two terms
    (the ratios fall away from the mode), and the sum stops once that is below
    1e-17 of the sum."""
    step = 1 if stop >= start else -1
    ref = float(f(np.array([float(start)]))[0])
    total = 0.0
    pos = start
    block = _TAIL_BLOCK
    while True:
        end = pos + step * (block - 1)
        if (end - stop) * step > 0:
            end = stop
        idx = np.arange(pos, end + step, step, dtype=np.float64)
        values = f(idx) - ref
        total += float(np.sum(np.exp(values)))
        if end == stop:
            break
        if idx.size >= 2:
            ratio = math.exp(float(values[-1] - values[-2]))
            if ratio < 1.0:
                rest = math.exp(float(values[-1])) * ratio / (1.0 - ratio)
                if rest <= _TAIL_EPS * total:
                    break
        pos = end + step
        block *= 2
    return ref + math.log(total)


class _LogPmf:
    """log P(A = a) of the first cell of a 2 x 2 table with fixed margins (the
    hypergeometric distribution), from Loader's binomial form: P(a) = dbinom(a;
    c0, p) dbinom(r1 - a; c1, p) / dbinom(r1; n, p) for any p, here p = r1 / n."""

    def __init__(self, r1: int, c0: int, c1: int) -> None:
        n = c0 + c1
        self.r1, self.c0, self.c1 = r1, c0, c1
        self.p, self.q = r1 / n, (n - r1) / n
        self.log_total = float(_log_dbinom(np.array([float(r1)]), n, self.p, self.q)[0])

    def __call__(self, a: FloatArray) -> FloatArray:
        return (
            _log_dbinom(a, self.c0, self.p, self.q)
            + _log_dbinom(float(self.r1) - a, self.c1, self.p, self.q)
            - self.log_total
        )


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


def _fisher_2x2(a: int, r1: int, c0: int, c1: int, side: Alternative) -> float:
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
            return min(1.0, math.exp(_tail_log_sum(f, a, hi)))
        if a == lo:
            return 1.0
        return min(1.0, max(0.0, 1.0 - math.exp(_tail_log_sum(f, a - 1, lo))))
    if side == "less":  # P(A <= a)
        if a < mode:
            return min(1.0, math.exp(_tail_log_sum(f, a, lo)))
        if a == hi:
            return 1.0
        return min(1.0, max(0.0, 1.0 - math.exp(_tail_log_sum(f, a + 1, hi))))
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
        parts.append(_tail_log_sum(f, left, lo))
    if right <= hi:
        parts.append(_tail_log_sum(f, right, hi))
    return min(1.0, math.fsum(math.exp(v) for v in parts))


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


def _fisher_2xc(first: list[int], cols: list[int]) -> float:
    """Two-sided Fisher exact p of a 2 x c table, c >= 3, every column non-empty,
    both rows non-empty (fisher_exact describes the search)."""
    order = sorted(range(len(cols)), key=lambda j: cols[j])
    cols = [cols[j] for j in order]
    first = [first[j] for j in order]
    k = len(cols)
    n = sum(cols)
    r1 = sum(first)
    p, q = r1 / n, (n - r1) / n
    logs = [_log_dbinom(np.arange(c + 1, dtype=np.float64), c, p, q) for c in cols]
    log_total = float(_log_dbinom(np.array([float(r1)]), n, p, q)[0])
    threshold = math.fsum(float(logs[j][first[j]]) for j in range(k)) + _LOG_TIE
    if threshold >= float(_suffix_max(logs)[r1]):
        # The most probable table counts, so every table does: p is 1 exactly.
        return 1.0
    margin = _FISHER_MARGIN * (1.0 + abs(threshold))
    # Suffix j + 1 .. k - 1: the largest and smallest log mass of a completion and
    # the log mass of all completions (a sum of binomials with one p is binomial),
    # as functions of the count still to place in the first row.
    remaining = [sum(cols[j:]) for j in range(k + 1)]
    best: list[FloatArray] = [np.zeros(0)] * k
    worst: list[FloatArray] = [np.zeros(0)] * k
    mass: list[FloatArray] = [np.zeros(0)] * k
    for j in range(1, k):
        if j == k - 1:
            best[j] = worst[j] = mass[j] = logs[k - 1]
        else:
            best[j] = _suffix_max(logs[j:])
            worst[j] = _suffix_min(logs[j:])
            mass[j] = _log_dbinom(
                np.arange(remaining[j] + 1, dtype=np.float64), remaining[j], p, q
            )
    work = 0.0
    sums = np.zeros(1, dtype=np.int64)
    values = np.zeros(1)
    counted: list[FloatArray] = []
    for j in range(k - 1):
        final = j == k - 2
        rem = r1 - sums
        nodes, node_of = np.unique(rem, return_inverse=True)
        node_of = node_of.reshape(-1)
        a_lo = np.maximum(0, nodes - remaining[j + 1])
        a_hi = np.minimum(cols[j], nodes)
        width = int((a_hi - a_lo).max()) + 1
        work += float(nodes.size) * width + float(sums.size) * math.log2(width + 1)
        if work > _FISHER_WORK_LIMIT:
            raise _TooLarge
        # The tables so far grouped by node, the nodes in chunks of at most
        # _FISHER_CHUNK grid elements, so that memory stays bounded.
        order = np.argsort(node_of, kind="stable")
        sums, values, node_of = sums[order], values[order], node_of[order]
        per_chunk = max(1, _FISHER_CHUNK // width)
        next_sums: list[IntArray] = []
        next_values: list[FloatArray] = []
        for first_node in range(0, nodes.size, per_chunk):
            last_node = min(nodes.size, first_node + per_chunk)
            lo_state, hi_state = np.searchsorted(node_of, [first_node, last_node])
            tails, children = _fisher_stage(
                logs[j],
                best[j + 1],
                mass[j + 1],
                nodes[first_node:last_node],
                a_lo[first_node:last_node],
                a_hi[first_node:last_node],
                node_of[lo_state:hi_state] - first_node,
                values[lo_state:hi_state],
                threshold if final else threshold - margin,
                expand=not final,
            )
            counted.append(values[lo_state:hi_state] + tails)
            if children is None:
                continue
            parent, child_a = children
            work += float(child_a.size)
            if work > _FISHER_WORK_LIMIT:
                raise _TooLarge
            child_sums = sums[lo_state:hi_state][parent] + child_a
            child_values = values[lo_state:hi_state][parent] + logs[j][child_a]
            keep = child_values + worst[j + 1][r1 - child_sums] <= threshold + margin
            next_sums.append(child_sums[keep])
            next_values.append(child_values[keep])
        if final:
            break
        sums = np.concatenate(next_sums)
        values = np.concatenate(next_values)
        if sums.size == 0:
            break
    every = np.concatenate(counted)
    finite = every[np.isfinite(every)]
    if finite.size == 0:  # pragma: no cover - the observed table always counts
        return 0.0
    top = float(finite.max())
    log_p = top + math.log(float(np.sum(np.exp(finite - top)))) - log_total
    return min(1.0, math.exp(log_p))


def _fisher_stage(
    here_logs: FloatArray,
    best: FloatArray,
    mass: FloatArray,
    nodes: IntArray,
    a_lo: IntArray,
    a_hi: IntArray,
    node_of: IntArray,
    values: FloatArray,
    cut: float,
    *,
    expand: bool,
) -> tuple[FloatArray, tuple[IntArray, IntArray] | None]:
    """One column of the search for a chunk of nodes (counts still to place in the
    first row) and the partial tables at them (``node_of``, ``values``).

    Per table, the children a of this column with values + here_logs[a] +
    best[rest] <= ``cut`` have every completion counted: they form both ends of
    the concave bound, found by bisection, and their mass (``mass``: all
    completions) is summed from cumulative sums along the node's row. Returns the
    log of that counted mass per table and, with ``expand``, the children in
    between (parent index, a) for the next column.
    """
    width = int((a_hi - a_lo).max()) + 1
    a_grid = a_lo[:, None] + np.arange(width)[None, :]
    valid = a_grid <= a_hi[:, None]
    a_safe = np.where(valid, a_grid, a_lo[:, None])
    rest = nodes[:, None] - a_safe
    here = here_logs[a_safe]
    bound = np.where(valid, here + best[rest], -math.inf)
    weight = np.where(valid, here + mass[rest], -math.inf)
    with np.errstate(invalid="ignore"):
        prefix = np.logaddexp.accumulate(weight, axis=1)
        suffix = np.logaddexp.accumulate(weight[:, ::-1], axis=1)[:, ::-1]
    peak = np.argmax(bound, axis=1)[node_of]
    length = (a_hi - a_lo + 1)[node_of]
    limit = cut - values
    x1 = _bsearch(bound, node_of, np.zeros_like(peak), peak + 1, limit, above=True)
    x2 = _bsearch(bound, node_of, peak, length, limit, above=False) - 1
    none_open = x1 > peak
    open_ = ~none_open
    tails = np.where(none_open, prefix[node_of, length - 1], -math.inf)
    left = np.where(open_ & (x1 > 0), prefix[node_of, np.maximum(x1 - 1, 0)], -math.inf)
    right = np.where(
        open_ & (x2 + 1 < length),
        suffix[node_of, np.minimum(x2 + 1, length - 1)],
        -math.inf,
    )
    tails = np.where(open_, np.logaddexp(left, right), tails)
    if not expand:
        return tails, None
    span = np.where(open_, x2 - x1 + 1, 0)
    total = int(span.sum())
    parent = np.repeat(np.arange(values.size), span)
    offset = np.arange(total) - np.repeat(np.cumsum(span) - span, span)
    child_a = a_lo[node_of[parent]] + x1[parent] + offset
    return tails, (parent, child_a)


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

    Probabilities are compared in log space: each log P is a sum of Loader's
    (2000) saddle-point log binomial probabilities, accurate to about 1e-15 times
    the largest term, so a table is classified differently from the exact rational
    comparison only if its probability lies within about 1e-14 relative of P_obs
    (1 + 1e-7); no table of the fixtures or the probe comes near that (R's own
    double-precision sums are no closer).

    2 x 2: the probabilities are hypergeometric and log-concave in a, so the
    tables at most as probable form both ends of the range; their bounds are found
    by bisection and each end is summed outward in blocks until a geometric bound
    of the rest falls below 1e-17 of the sum (n = 1e6: about 1 ms; any total below
    2^53).

    2 x c, c >= 3: a search over the columns in the manner of Mehta & Patel's
    (1983, Journal of the American Statistical Association 78: 427-434) network
    algorithm. Columns are taken smallest first; for each partial first row the
    largest and smallest probability of any completion are known (the largest by
    the greedy choice of increments, the log probabilities being concave in each
    count; the smallest at a vertex of the box), so a partial row whose every
    completion is counted is summed at once (the mass of all completions is
    binomial) and one whose every completion is more probable is dropped; only the
    others are extended. The last two columns are summed from cumulative sums along
    the second-to-last column. The work grows with n, with c and with how far the
    observed table lies from independence; measured (tests/stats/test_contingency.py
    times them): 2 x 3 at n = 2,000 0.004 s (p 0.18) to 0.015 s (p 3e-27), at
    n = 10,000 0.03 s (p 0.05) to 0.29 s (p 9e-133); 2 x 5 at n = 500 0.003 s to
    0.04 s (p 5e-13), at n = 2,000 0.01 s (p 0.65) to 0.09 s (p 1.6e-4), while 2 x 5
    at n = 2,000 with p near 1e-55 would need about 1.2e8 array elements (3.2 s).
    When the search would need more than about two seconds (6e7 array elements,
    about 3e7 per second), or its arrays more than about 100 MB (n (c - 2) above
    4e6), it raises instead of approximating; the refusal comes before the stage
    that would cross the limit, after at most about two seconds.

    Columns without counts do not change the test and are left out; a table with
    one non-empty column or one empty row has a single table with its margins, p 1
    (R: [[0, 0], [3, 4]] gives 1).

    Raises:
        ValueError: the table is not 2 x c with c >= 2, a count is invalid,
            ``alternative`` is one-sided for a table that is not 2 x 2, or the
            exact computation is too large.
    """
    counts = _as_counts(table)
    side = check_alternative(alternative)
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
    rows = counts.tolist()
    first = [int(v) for v in rows[0]]
    cols = [int(a) + int(b) for a, b in zip(rows[0], rows[1], strict=True)]
    n = sum(cols)
    kept = [j for j, c in enumerate(cols) if c > 0]
    r1 = sum(first)
    if len(kept) < 2 or r1 == 0 or r1 == n:
        return FisherExact(p_value=1.0, alternative=side, n=n, columns=len(kept))
    if len(kept) == 2:
        # A 2 x 2 table (as given, or with empty columns left out: then two-sided).
        j0, j1 = kept
        p_value = _fisher_2x2(first[j0], r1, cols[j0], cols[j1], side)
        return FisherExact(p_value=p_value, alternative=side, n=n, columns=2)
    if n * (len(kept) - 2) > _FISHER_SIZE_LIMIT:
        raise ValueError(_too_large_message(len(kept), n))
    try:
        p_value = _fisher_2xc([first[j] for j in kept], [cols[j] for j in kept])
    except _TooLarge:
        raise ValueError(_too_large_message(len(kept), n)) from None
    return FisherExact(p_value=p_value, alternative=side, n=n, columns=len(kept))


def _too_large_message(c: int, n: int) -> str:
    return (
        f"Fisher's exact test of this 2 x {c} table (n = {n}) would take more than "
        "about two seconds to compute exactly, and inferplot does not approximate it: "
        "merge sparse levels of y, or compare fewer rows"
    )


# --------------------------------------------------------------------------- #
# Pairwise Fisher tests
# --------------------------------------------------------------------------- #


def _pair(
    counts: IntArray, i: int, j: int, level: float, side: Alternative
) -> PairComparison:
    """Rows i and j on the columns they hold: Fisher's p and Cramer's V (the
    adjusted p is filled in by pairwise_fisher)."""
    pair = counts[[i, j]]
    columns = tuple(int(c) for c in np.flatnonzero(pair.sum(axis=0) > 0))
    sub = pair[:, list(columns)]
    n = int(sub.sum())
    if len(columns) < 2:
        return PairComparison(
            row1=i,
            row2=j,
            columns=columns,
            n=n,
            p_value=None,
            p_adjusted=None,
            alternative=None,
            effect=None,
            reason=(
                "the two levels occur in one column only, so their table has a "
                "single column and no test (R: fisher.test stops)"
            ),
            effect_reason=None,
        )
    used: Alternative = side if sub.shape == (2, 2) else "two-sided"
    p_value = fisher_exact(sub, alternative=used).p_value
    effect: EffectSizeCI | None = None
    effect_reason: str | None = None
    if n <= max(sub.shape):
        effect_reason = (
            "the bias-corrected Cramer's V needs more counts than columns (R: NaN)"
        )
    else:
        effect = cramers_v(sub, conf_level=level, alternative=side)
    return PairComparison(
        row1=i,
        row2=j,
        columns=columns,
        n=n,
        p_value=p_value,
        p_adjusted=None,
        alternative=used,
        effect=effect,
        reason=None,
        effect_reason=effect_reason,
    )


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
    of the pairs that have a test (``p_adjust``).

    A pair whose table keeps one column has no test (fisher.test stops; ggstatsplot
    loses the whole pairwise table silently, edge_bar_degenerate-single-x): it is
    returned with ``p_value`` None and a ``reason``. A pair whose V is not defined
    (no more counts than columns) has ``effect`` None and an ``effect_reason``.

    Raises:
        ValueError: an invalid count, ``conf_level``, ``alternative`` or
            ``p_adjust_method``, or a pair's exact test is too large
            (``fisher_exact``).
    """
    counts = _as_counts(table)
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    present = [i for i in range(counts.shape[0]) if int(counts[i].sum()) > 0]
    pairs: list[PairComparison] = []
    for index, i in enumerate(present):
        for j in present[index + 1 :]:
            pairs.append(_pair(counts, i, j, level, side))
    raw = [pair.p_value for pair in pairs if pair.p_value is not None]
    adjusted = iter(p_adjust(raw, p_adjust_method) if raw else ())
    out = [
        pair
        if pair.p_value is None
        else dataclasses.replace(pair, p_adjusted=next(adjusted))
        for pair in pairs
    ]
    return tuple(out)
