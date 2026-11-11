"""Fisher's exact test (``inferplot.stats.contingency``): the bounded-cost search of
2 x c tables and its limits, the 2 x 2 tails at large totals, and the shared
budget of the pairwise tests (DECISIONS.md D107).

No wall clock is asserted: the limits are on work units and bytes, which the code
counts before it spends them (``_Budget``), and the tests assert on those counts.
Exactness is checked against brute-force enumeration in exact rationals (every
split of the columns between the two sides of the search), against the previous
implementation on 616 tables it answered (tests/data/fisher_regression.json,
tools/reference/fisher_regression.py), and against 40-digit sums of the 2 x 2
tails (tests/data/fisher_2x2_reference.json, tools/reference/fisher_2x2_mpmath.py).
The oracle parity of every fixture pair and of probe-contingency is in
tests/stats/test_contingency_parity.py.
"""

from __future__ import annotations

import itertools
import json
import math
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest
from numpy.typing import NDArray

from inferplot.stats import contingency as C
from inferplot.stats import fisher_exact, pairwise_fisher
from inferplot.stats.padjust import p_adjust
from tests.timing import allowed
from tests.tolerances import is_close

IntArray = NDArray[np.int64]
DATA: Final = Path(__file__).resolve().parents[1] / "data"
RNG: Final = np.random.default_rng(20261111)


def _enumerate(table: IntArray) -> Fraction:
    """The two-sided p by enumerating every first row with the margins, in exact
    rationals, with R's P <= P_obs (1 + 1e-7)."""
    first = [int(v) for v in table[0]]
    cols = [int(v) for v in table.sum(axis=0)]
    r1, n = sum(first), sum(cols)
    total = math.comb(n, r1)

    def weight(row: tuple[int, ...]) -> int:
        return math.prod(math.comb(c, a) for c, a in zip(cols, row, strict=True))

    observed = Fraction(weight(tuple(first))) * (1 + Fraction(1, 10**7))
    counted = 0
    for row in itertools.product(*(range(c + 1) for c in cols[:-1])):
        last = r1 - sum(row)
        if 0 <= last <= cols[-1]:
            w = weight((*row, last))
            if w <= observed:
                counted += w
    return Fraction(counted, total)


def _split_p(table: IntArray, list_columns: int) -> float:
    """The two-sided p of a 2 x c table (every column non-empty) with the search
    split as given: ``list_columns`` narrowest columns on the list side."""
    first, second = C._canonical(  # pyright: ignore[reportPrivateUsage]
        [int(v) for v in table[0]], [int(v) for v in table[1]]
    )
    cols = [a + b for a, b in zip(first, second, strict=True)]
    budget = C._Budget(1e12)  # pyright: ignore[reportPrivateUsage]
    return C._fisher_2xc(  # pyright: ignore[reportPrivateUsage]
        first, cols, budget, list_columns
    )


def _small_tables() -> list[IntArray]:
    tables = [
        # mirror images and equal columns: ties with the observed table
        np.array([[1, 2, 3, 3, 2, 1], [3, 2, 1, 1, 2, 3]]),
        np.array([[2, 7, 4], [4, 7, 2]]),
        np.array([[0, 1, 2, 3], [3, 2, 1, 0]]),
        np.array([[2, 2, 2, 2], [2, 2, 2, 2]]),
        np.array([[3, 0, 3, 0, 3], [0, 3, 0, 3, 0]]),
        np.array([[1, 1, 1, 1, 1, 4], [1, 1, 1, 1, 1, 0]]),
        # one sparse row, and a wide column next to narrow ones
        np.array([[1, 0, 0, 1, 0], [4, 5, 3, 2, 6]]),
        np.array([[9, 1, 0, 2], [3, 1, 2, 0]]),
    ]
    for cols, scale in ((3, 9), (4, 6), (5, 4), (6, 3)):
        for _ in range(8):
            while True:
                t = RNG.integers(0, scale, size=(2, cols))
                if (t.sum(axis=0) > 0).all() and (t.sum(axis=1) > 0).all():
                    break
            tables.append(t.astype(np.int64))
    return tables


SMALL: Final = _small_tables()


@pytest.mark.parametrize("table", SMALL, ids=[str(t.tolist()) for t in SMALL])
def test_every_split_matches_exact_enumeration(table: IntArray) -> None:
    # The list side takes 1 to c columns (c: the whole table is listed, no network
    # side): every split gives the exact p. Measured: at most 4e-15 relative.
    exact = _enumerate(table)
    for q in range(1, table.shape[1] + 1):
        p = _split_p(table, q)
        assert abs(p - float(exact)) <= 1e-13 * float(exact), (q, p, float(exact))
    assert fisher_exact(table).p_value == pytest.approx(float(exact), rel=1e-13)


REGRESSION: Final = json.loads((DATA / "fisher_regression.json").read_text())


def test_the_regression_table_holds_616_entries() -> None:
    # 403 tables of 2 x 3 to 2 x 9 and 71 of 2 x 2 with all three alternatives.
    rows = REGRESSION["tables"]
    wide = {(r["batch"], r["index"]) for r in rows if len(r["table"][0]) > 2}
    assert len(rows) == 616
    assert len(wide) == 403


def test_the_regression_table_reaches_every_part_of_the_search() -> None:
    # Tables with no network side (all columns listed), with one, and with a
    # wide column (more than _WIDE_ROW children per node: the windowed rows).
    kinds: set[str] = set()
    for row in REGRESSION["tables"]:
        table = np.array(row["table"], dtype=np.int64)
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 3 or row["alternative"] != "two-sided":
            continue
        first, second = C._canonical(  # pyright: ignore[reportPrivateUsage]
            [int(v) for v in table[0]], [int(v) for v in table[1]]
        )
        cols = [a + b for a, b in zip(first, second, strict=True)]
        r1, n = sum(first), sum(cols)
        if r1 in (0, n):
            continue
        widths = [min(c, r1) - max(0, c - (n - r1)) for c in cols]
        plan = C._fisher_plan(widths)  # pyright: ignore[reportPrivateUsage]
        network = [widths[j] for j in plan.order[: len(widths) - plan.list_columns]]
        if not network:
            kinds.add("list only")
        elif max(network) + 1 > C._WIDE_ROW:  # pyright: ignore[reportPrivateUsage]
            kinds.add("wide")
        else:
            kinds.add("network")
    assert kinds == {"list only", "network", "wide"}


@pytest.mark.parametrize("batch", range(8))
def test_the_previous_implementation_on_616_tables(batch: int) -> None:
    # Every table the previous search answered, its p unchanged within
    # closed_form; measured: at most 5.7e-14 relative (a 2 x 2 "less" tail of
    # 1e-147). None is refused.
    rows = [r for r in REGRESSION["tables"] if r["batch"] == batch]
    worst = 0.0
    for row in rows:
        table = np.array(row["table"], dtype=np.int64)
        expected = float.fromhex(row["p_value_hex"])
        p = fisher_exact(table, alternative=row["alternative"]).p_value
        assert is_close("closed_form", p, expected), (row, p)
        if expected > 0:
            worst = max(worst, abs(p - expected) / expected)
    assert worst < 1e-12


def test_permuted_tables_give_the_same_p_and_work() -> None:
    # The two-sided p depends on neither the order of the rows nor that of the
    # columns: the table is put in one order first, so the p is the same to the
    # last bit and so is the work (which pairs a budget admits must not depend
    # on the order of the levels).
    for _ in range(6):
        table = RNG.integers(1, 25, size=(2, 7)).astype(np.int64)
        budget = C._Budget(C._FISHER_WORK_LIMIT)  # pyright: ignore[reportPrivateUsage]
        p = C._fisher(table, "two-sided", budget).p_value  # pyright: ignore[reportPrivateUsage]
        for _ in range(3):
            permuted = table[:, RNG.permutation(7)][RNG.permutation(2)]
            other = C._Budget(C._FISHER_WORK_LIMIT)  # pyright: ignore[reportPrivateUsage]
            q = C._fisher(permuted, "two-sided", other).p_value  # pyright: ignore[reportPrivateUsage]
            assert q == p
            assert other.spent == budget.spent


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #

# A 2 x 20 table of 840 observations far from independence: the search would
# need more than the work limit.
FAR_2X20: Final = np.array(
    [[2 * j + 2 for j in range(20)], [40 - 2 * j for j in range(20)]]
)
# 2 x 3 with 1.3 million per cell: a column's grid row alone is beyond the memory
# limit, known from the margins.
DENSE_2X3: Final = np.full((2, 3), 1_300_000, dtype=np.int64)


def _reviewer_2xc(n: int, c: int, tilt: float) -> IntArray:
    rng = np.random.default_rng(0)
    cols = np.full(c, n // c)
    first = rng.binomial(cols, np.clip(0.5 + tilt * np.linspace(-1, 1, c), 0, 1))
    return np.array([first, cols - first], dtype=np.int64)


# Tables that are answered within the limits: the reviewers' 2 x 8 of 304 and
# 2 x 10 of 200 (which the previous search refused after 4.6 s and 2.3 GB), a
# 2 x 10 far from independence, 2 x 5 of 2,000 and 2 x 3 of 100,000 near
# independence, and a sparse 2 x 12.
ANSWERED: Final = [
    _reviewer_2xc(304, 8, 0.0),
    _reviewer_2xc(200, 10, 0.0),
    _reviewer_2xc(200, 10, 0.3),
    _reviewer_2xc(2_000, 5, 0.0),
    _reviewer_2xc(100_000, 3, 0.0),
    _reviewer_2xc(96, 12, 0.0),
]


@pytest.mark.parametrize("table", ANSWERED, ids=lambda t: f"2x{t.shape[1]}-n{t.sum()}")
def test_answered_within_the_limits_and_the_memory_bound(table: IntArray) -> None:
    # The work spent stays within the limit, and the bytes Python's allocator saw
    # at the peak (NumPy's arrays are traced) stay within the bound the plan
    # computed from the margins before allocating anything, itself within the
    # memory limit.
    first, second = C._canonical(  # pyright: ignore[reportPrivateUsage]
        [int(v) for v in table[0]], [int(v) for v in table[1]]
    )
    cols = [a + b for a, b in zip(first, second, strict=True)]
    r1, n = sum(first), sum(cols)
    widths = [min(c, r1) - max(0, c - (n - r1)) for c in cols]
    plan = C._fisher_plan(widths)  # pyright: ignore[reportPrivateUsage]
    assert plan.memory <= C._FISHER_MEMORY_LIMIT  # pyright: ignore[reportPrivateUsage]
    budget = C._Budget(C._FISHER_WORK_LIMIT)  # pyright: ignore[reportPrivateUsage]
    tracemalloc.start()
    try:
        p = C._fisher(table, "two-sided", budget).p_value  # pyright: ignore[reportPrivateUsage]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert 0.0 <= p <= 1.0
    assert budget.spent <= C._FISHER_WORK_LIMIT  # pyright: ignore[reportPrivateUsage]
    assert budget.largest <= plan.memory
    assert peak <= plan.memory, (peak / 2**20, plan.memory / 2**20)


def test_beyond_the_work_limit_is_refused_having_spent_at_most_the_limit() -> None:
    budget = C._Budget(C._FISHER_WORK_LIMIT)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(C._TooLarge) as info:  # pyright: ignore[reportPrivateUsage]
        C._fisher(FAR_2X20, "two-sided", budget)  # pyright: ignore[reportPrivateUsage]
    assert info.value.kind == "work"
    assert budget.spent <= C._FISHER_WORK_LIMIT  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="more work than inferplot allows") as error:
        fisher_exact(FAR_2X20)
    assert str(error.value) == (
        "Fisher's exact test of this 2 x 20 table (n = 840) would take more work "
        "than inferplot allows one test (about a second) to compute exactly, and "
        "inferplot does not approximate it: merge sparse columns (levels with few "
        "counts), or compare fewer"
    )


def test_beyond_the_memory_limit_is_refused_before_allocating() -> None:
    # The plan's bytes, from the margins alone, exceed the limit: nothing is
    # spent and nothing large is allocated.
    budget = C._Budget(C._FISHER_WORK_LIMIT)  # pyright: ignore[reportPrivateUsage]
    tracemalloc.start()
    try:
        with pytest.raises(C._TooLarge) as info:  # pyright: ignore[reportPrivateUsage]
            C._fisher(DENSE_2X3, "two-sided", budget)  # pyright: ignore[reportPrivateUsage]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert info.value.kind == "memory"
    assert budget.spent == 0.0
    assert budget.largest == 0
    assert peak < 2**20
    with pytest.raises(ValueError, match="more memory than inferplot allows") as error:
        fisher_exact(DENSE_2X3)
    assert "the chi-square test of the table answers the same question" in str(
        error.value
    )


def test_work_known_from_the_margins_is_refused_before_it_is_spent() -> None:
    # The list and the bounds cost what the margins say whatever is pruned: a
    # budget smaller than that refuses at once, having spent nothing.
    table = _reviewer_2xc(304, 8, 0.0)
    first, second = C._oriented(table, "two-sided")  # pyright: ignore[reportPrivateUsage]
    cols = [a + b for a, b in zip(first, second, strict=True)]
    _, widths = C._ranges(first, cols)  # pyright: ignore[reportPrivateUsage]
    certain = C._fisher_plan(widths).certain  # pyright: ignore[reportPrivateUsage]
    budget = C._Budget(certain / 2)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(C._TooLarge):  # pyright: ignore[reportPrivateUsage]
        C._fisher(table, "two-sided", budget)  # pyright: ignore[reportPrivateUsage]
    assert budget.spent == 0.0


def test_p_underflows_to_zero_without_a_search() -> None:
    # 2 x 3 at n = 3,000, far from independence: the number of tables times
    # P_obs (1 + 1e-7) is below 2^-1075, so p rounds to 0 (checked here by a full
    # sum in log space), and the search is not run.
    table = np.array([[1_000, 0, 0], [0, 1_000, 1_000]], dtype=np.int64)
    cols = table.sum(axis=0)
    r1 = int(table[0].sum())
    a0 = np.arange(cols[0] + 1)[:, None]
    a1 = np.arange(cols[1] + 1)[None, :]
    a2 = r1 - a0 - a1
    valid = (a2 >= 0) & (a2 <= cols[2])
    a2 = np.where(valid, a2, 0)

    def log_choose(m: int, k: Any) -> Any:
        return math.lgamma(m + 1) - _lgamma(k + 1) - _lgamma(m - k + 1)

    logs = log_choose(int(cols[0]), a0) + log_choose(int(cols[1]), a1)
    logs = logs + log_choose(int(cols[2]), a2)
    observed = float(logs[1_000, 0])
    counted = valid & (logs <= observed + math.log1p(1e-7))
    log_p = _logsumexp(logs[counted]) - _logsumexp(logs[valid])
    assert log_p < -1075 * math.log(2.0)
    budget = C._Budget(C._FISHER_WORK_LIMIT)  # pyright: ignore[reportPrivateUsage]
    assert C._fisher(table, "two-sided", budget).p_value == 0.0  # pyright: ignore[reportPrivateUsage]
    assert budget.spent == 0.0


def _lgamma(x: Any) -> Any:
    from scipy import special

    return special.gammaln(x)


def _logsumexp(x: Any) -> float:
    top = float(np.max(x))
    return top + math.log(float(np.sum(np.exp(x - top))))


# --------------------------------------------------------------------------- #
# 2 x 2 at large totals
# --------------------------------------------------------------------------- #


def test_two_by_two_against_40_digit_sums() -> None:
    # Totals from 10 to 1e10, five tables each (balanced 0.3 and 3 standard
    # deviations out, unequal margins, an exact mirror-image tie, a far tail),
    # three alternatives. Measured: at most 8.9e-15 relative (two-sided and
    # "greater"), 2.0e-15 ("less").
    rows = json.loads((DATA / "fisher_2x2_reference.json").read_text())["rows"]
    assert len(rows) == 50
    for row in rows:
        for side, value in row["p"].items():
            expected = float(value)
            p = fisher_exact(row["table"], alternative=side).p_value
            assert abs(p - expected) <= 1e-13 * expected, (row, side, p)


def test_two_by_two_beyond_the_work_limit() -> None:
    # Balanced margins at n near 1e15: the tails hold about 1e8 terms above the
    # double's resolution, beyond the work limit; refused having spent at most
    # the limit. Unequal margins at the same total have fewer and are answered.
    n = 10**15
    q = n // 4
    table = np.array([[q + 30_000_000, q], [q, q]], dtype=np.int64)
    budget = C._Budget(C._FISHER_WORK_LIMIT)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(C._TooLarge):  # pyright: ignore[reportPrivateUsage]
        C._fisher(table, "two-sided", budget)  # pyright: ignore[reportPrivateUsage]
    assert budget.spent <= C._FISHER_WORK_LIMIT  # pyright: ignore[reportPrivateUsage]
    small = np.array([[1_000, 10**12], [1_050, 10**12]], dtype=np.int64)
    assert 0.1 < fisher_exact(small).p_value < 0.5


# --------------------------------------------------------------------------- #
# Pairwise: one budget per table
# --------------------------------------------------------------------------- #


def _survey(rows: int, cols: int, n: int, seed: int) -> IntArray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, rows, n)
    y = np.clip(
        np.round(rng.normal(cols / 2 + (x - rows / 2) * 0.6, cols / 4)), 0, cols - 1
    ).astype(np.int64)
    table = np.zeros((rows, cols), dtype=np.int64)
    np.add.at(table, (x, y), 1)
    return table


@pytest.mark.parametrize("limit", [3e7, 1e8, C._PAIRWISE_WORK_LIMIT])  # pyright: ignore[reportPrivateUsage]
def test_pairwise_tests_share_one_budget(limit: float) -> None:
    # A 10 x 7 table (45 pairs, about 3e8 units in all): whatever the budget, the
    # work spent on all the pairs stays within it; each tested pair has exactly
    # fisher_exact's p, and the others say why they have none. The adjustment
    # counts every pair: one too large to compute enters with p = 1 (lead's
    # review). The default budget tests them all.
    table = _survey(10, 7, 600, seed=3)
    budget = C._Budget(limit)  # pyright: ignore[reportPrivateUsage]
    pairs = C._pairwise_fisher(  # pyright: ignore[reportPrivateUsage]
        table, "holm", 0.95, "two-sided", budget
    )
    assert len(pairs) == 45
    assert budget.spent <= limit
    tested = [p for p in pairs if p.p_value is not None]
    for pair in pairs:
        if pair.p_value is None:
            assert pair.reason == "too large to compute exactly"
            assert pair.p_adjusted is None
            assert pair.effect is not None
        else:
            sub = table[[pair.row1, pair.row2]][:, list(pair.columns)]
            assert pair.p_value == fisher_exact(sub).p_value
    expected = {3e7: (1, 44), 1e8: (5, 44), C._PAIRWISE_WORK_LIMIT: (45, 45)}  # pyright: ignore[reportPrivateUsage]
    low, high = expected[limit]
    assert low <= len(tested) <= high
    raw = [p.p_value for p in tested if p.p_value is not None]
    family = [*raw, *[1.0] * (len(pairs) - len(tested))]
    assert [p.p_adjusted for p in tested] == list(
        C.p_adjust(family, "holm")[: len(raw)]
    )


def test_pairwise_result_does_not_depend_on_the_order_of_the_rows() -> None:
    # A tight budget admits only some pairs; which ones, and their p-values, are
    # the same whatever the order of the rows (levels of x).
    table = _survey(8, 10, 900, seed=5)
    limit = 1.5e8

    def run(order: list[int]) -> dict[frozenset[int], float | None]:
        budget = C._Budget(limit)  # pyright: ignore[reportPrivateUsage]
        pairs = C._pairwise_fisher(  # pyright: ignore[reportPrivateUsage]
            table[order], "none", 0.95, "two-sided", budget
        )
        return {frozenset((order[p.row1], order[p.row2])): p.p_value for p in pairs}

    base = run(list(range(8)))
    assert 0 < sum(v is not None for v in base.values()) < 28
    for _ in range(3):
        order = [int(i) for i in RNG.permutation(8)]
        assert run(order) == base


def test_pairwise_default_budget_on_the_reviewers_tables() -> None:
    # The public function with its own budget: a 3 x 8 table of 377 observations
    # (the previous search took 19 s and 3.2 GB) has every pair tested.
    rng = np.random.default_rng(7)
    table = rng.integers(1, 30, size=(3, 8))
    pairs = pairwise_fisher(table)
    assert all(p.p_value is not None for p in pairs)


def test_uncomputed_pairs_count_in_the_adjustment() -> None:
    # Lead's review: which pairs fit the budget depends on the data, so a pair too
    # large to compute stays in the family with p = 1. The computed pairs are
    # adjusted as p.adjust(p, n = <pairs with a test, computed or not>) adjusts
    # them, never as if the uncomputed comparisons had not been made.
    table = [
        [3, 9, 1, 0, 2, 6, 1, 4],
        [8, 2, 5, 1, 0, 3, 2, 1],
        [1, 4, 6, 3, 2, 0, 5, 2],
        [4, 4, 0, 6, 3, 1, 2, 7],
    ]
    full = C.pairwise_fisher(table)
    assert all(pair.p_value is not None for pair in full)
    raw = [pair.p_value for pair in full]
    for method in ("holm", "hochberg", "bonferroni", "BH", "BY", "hommel", "none"):
        complete = C.pairwise_fisher(table, p_adjust_method=method)
        # A budget that pays for the three cheapest tests only.
        spent = sorted(_work_of_each_pair(table))
        budget = C._Budget(sum(spent[:3]) + 1)
        partial = C._pairwise_fisher(table, method, 0.95, "two-sided", budget)
        computed = [
            (pair, pair.p_value, pair.p_adjusted)
            for pair in partial
            if pair.p_value is not None and pair.p_adjusted is not None
        ]
        skipped = [pair for pair in partial if pair.p_value is None]
        assert computed
        assert skipped
        assert len(computed) + len(skipped) == len(partial)
        assert all(pair.reason == C._TOO_LARGE_REASON for pair in skipped)
        assert all(pair.p_adjusted is None for pair in skipped)
        want = p_adjust([*[p for _, p, _ in computed], *[1.0] * len(skipped)], method)
        assert [adjusted for _, _, adjusted in computed] == list(want[: len(computed)])
        by_rows = {(pair.row1, pair.row2): pair.p_adjusted for pair in complete}
        for pair, p_value, adjusted in computed:
            every = by_rows[(pair.row1, pair.row2)]
            assert every is not None
            # Never smaller than with every pair computed.
            assert adjusted >= every, method
            assert adjusted >= p_value
    assert len(raw) == 6


def _work_of_each_pair(table: list[list[int]]) -> list[float]:
    work = []
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            budget = C._Budget(C._FISHER_WORK_LIMIT)
            C._pairwise_fisher([table[i], table[j]], "none", 0.95, "two-sided", budget)
            work.append(budget.spent)
    return work


# --------------------------------------------------------------------------- #
# Everything the pairs take is counted (review P5B, A2)
# --------------------------------------------------------------------------- #

# The least work a 2 x 2 test, the planning of a pair and one Cramer's V take, in
# units of about a nanosecond (measured: 25 us, 115 us and 0.8 ms or more).
LEAST_2X2: Final = 25_000.0
LEAST_PLAN: Final = 100_000.0
LEAST_EFFECT: Final = 1.0e6


def _random_table(rows: int, cols: int, n: int, seed: int) -> IntArray:
    rng = np.random.default_rng(seed)
    table = np.zeros((rows, cols), dtype=np.int64)
    np.add.at(table, (rng.integers(0, rows, n), rng.integers(0, cols, n)), 1)
    return table


def test_two_by_two_tests_are_charged_their_work() -> None:
    # The reviewer's 100 x 2 table of 100,000 observations: 4,950 2 x 2 pairs.
    # Each test was charged about 21,000 units while it took 9 to 18 ms (88 s
    # in all). Now a budget of 2e7 units pays for at most 2e7 / 25,000 tests.
    table = _random_table(100, 2, 100_000, seed=1)
    budget = C._Budget(2e7)
    pairs = C._pairwise_fisher(table, "holm", 0.95, "two-sided", budget)
    tested = [p for p in pairs if p.p_value is not None]
    assert budget.spent <= 2e7
    assert 0 < len(tested) <= 2e7 / LEAST_2X2
    for pair in tested[:20]:
        sub = table[[pair.row1, pair.row2]]
        assert pair.p_value == fisher_exact(sub).p_value


def test_every_two_by_two_pair_within_the_budget() -> None:
    # With the default budget all 4,950 are tested, together, in about a second.
    table = _random_table(100, 2, 100_000, seed=1)
    start = time.perf_counter()
    pairs = pairwise_fisher(table)
    assert time.perf_counter() - start < allowed(4.0)
    assert all(p.p_value is not None for p in pairs)
    rng = np.random.default_rng(3)
    for k in rng.choice(len(pairs), 25, replace=False).tolist():
        pair = pairs[k]
        assert pair.p_value == fisher_exact(table[[pair.row1, pair.row2]]).p_value


def test_pairs_too_wide_for_any_search_are_not_planned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reviewer's 3 x 2000 table of 20,000: planning its three 2 x ~2000
    # tables took 53 s (O(c^2) per plan). No search of 669 or more columns fits
    # one test (_certain_floor), so they are not planned at all.
    plans: list[int] = []
    original = C._fisher_plan

    def counted(widths: list[int], list_columns: int | None = None) -> C._Plan:
        plans.append(len(widths))
        return original(widths, list_columns)

    monkeypatch.setattr(C, "_fisher_plan", counted)
    table = _random_table(3, 2000, 20_000, seed=1)
    start = time.perf_counter()
    pairs = pairwise_fisher(table)
    assert time.perf_counter() - start < allowed(2.0)
    assert plans == []
    assert all(p.reason == C._TOO_LARGE_REASON for p in pairs)
    assert C._certain_floor(668) <= C._FISHER_WORK_LIMIT < C._certain_floor(669)


def test_planning_is_charged_within_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # 435 pairs of 2 x 6 tables: planning every one was free (three plans each).
    # Now it is charged, up to half the budget, smallest tables first.
    plans: list[int] = []
    original = C._fisher_plan

    def counted(widths: list[int], list_columns: int | None = None) -> C._Plan:
        plans.append(len(widths))
        return original(widths, list_columns)

    monkeypatch.setattr(C, "_fisher_plan", counted)
    table = _random_table(30, 6, 3_000, seed=2)
    budget = C._Budget(2e6)
    pairs = C._pairwise_fisher(table, "holm", 0.95, "two-sided", budget)
    assert len(pairs) == 435
    assert budget.spent <= 2e6
    assert len(plans) <= 2e6 / LEAST_PLAN


def test_planning_cost_is_linear_in_the_columns() -> None:
    # A plan of 2,000 columns takes a few milliseconds (it took seconds).
    rng = np.random.default_rng(4)
    widths = [int(v) for v in rng.integers(1, 6, 2000)]
    start = time.perf_counter()
    C._fisher_plan(widths)
    assert time.perf_counter() - start < allowed(0.1)


def test_effect_intervals_have_their_own_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each pair's V interval took about a millisecond, uncounted: 666 of them on
    # a 37 x 37 table took 10 s. They draw on their own allowance, smallest
    # tables first (a group of equal size wholly or not at all); the others say
    # why they have no V.
    calls: list[int] = []
    original = C.cramers_v

    def counted(table: Any, **kw: Any) -> Any:
        calls.append(1)
        return original(table, **kw)

    monkeypatch.setattr(C, "cramers_v", counted)
    table = _random_table(37, 37, 5_000, seed=1)
    effects = C._Budget(5e7)
    pairs = C._pairwise_fisher(
        table, "holm", 0.95, "two-sided", C._Budget(1e6), effects
    )
    assert effects.spent <= effects.limit
    assert 0 < len(calls) <= effects.limit / LEAST_EFFECT
    reached = [p for p in pairs if p.effect is not None]
    missed = [p for p in pairs if p.effect_reason == C._EFFECT_NOT_REACHED]
    assert len(reached) == len(calls)
    assert len(reached) + len(missed) == len(pairs)
    assert max((len(p.columns), p.n) for p in reached) < min(
        (len(p.columns), p.n) for p in missed
    )
    for pair in reached[:5]:
        sub = table[np.ix_((pair.row1, pair.row2), pair.columns)]
        assert pair.effect == original(sub)


def test_the_public_pairs_count_their_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 70 x 37 sparse: 2,415 pairs. Every interval used to be computed.
    calls: list[int] = []
    original = C.cramers_v

    def counted(table: Any, **kw: Any) -> Any:
        calls.append(1)
        return original(table, **kw)

    monkeypatch.setattr(C, "cramers_v", counted)
    table = _random_table(70, 37, 3_000, seed=5)
    start = time.perf_counter()
    pairs = pairwise_fisher(table)
    assert time.perf_counter() - start < allowed(4.0)
    assert len(pairs) == 2415
    most = (C._PAIRWISE_WORK_LIMIT + C._PAIRWISE_EFFECT_LIMIT) / LEAST_EFFECT
    assert 0 < len(calls) <= most


def _limited(
    table: IntArray, order: list[int], limit: float, effect_limit: float
) -> dict[frozenset[int], tuple[object, ...]]:
    pairs = C._pairwise_fisher(
        table[order],
        "holm",
        0.95,
        "two-sided",
        C._Budget(limit),
        C._Budget(effect_limit),
    )
    return {
        frozenset((order[p.row1], order[p.row2])): (
            p.p_value,
            p.p_adjusted,
            p.reason,
            p.effect,
            p.effect_reason,
        )
        for p in pairs
    }


def test_limited_pairs_do_not_depend_on_the_order_of_the_rows() -> None:
    # Budgets that admit some 2 x 2 tests, some larger ones and some V
    # intervals: which ones, and every number, are the same in any row order.
    tables = [_random_table(40, 2, 4_000, seed=6), _random_table(12, 5, 900, seed=7)]
    for table, limit, effect_limit in ((tables[0], 3e7, 3e7), (tables[1], 4e7, 2e7)):
        base = _limited(table, list(range(table.shape[0])), limit, effect_limit)
        values = list(base.values())
        assert any(v[0] is None for v in values)
        assert any(v[0] is not None for v in values)
        assert any(v[3] is None for v in values)
        for _ in range(3):
            order = [int(i) for i in RNG.permutation(table.shape[0])]
            assert _limited(table, order, limit, effect_limit) == base


def test_two_by_two_tests_together_give_each_its_own_p() -> None:
    # A batch of 2 x 2 tests (both sides, all margins) gives every table the p it
    # gets alone, to the last bit, and spends at most each table's bound.
    rng = np.random.default_rng(8)
    items: list[tuple[int, int, int, int, Any]] = []
    for _ in range(300):
        scale = int(10 ** rng.uniform(0.5, 7))
        t = rng.integers(0, scale, size=(2, 2)) + rng.integers(0, 2, size=(2, 2))
        if (t.sum(axis=0) == 0).any() or (t.sum(axis=1) == 0).any():
            continue
        side = ["two-sided", "greater", "less"][int(rng.integers(0, 3))]
        first = [int(v) for v in t[0]]
        second = [int(v) for v in t[1]]
        if side == "two-sided":
            first, second = C._canonical(first, second)
        cols = [a + b for a, b in zip(first, second, strict=True)]
        items.append((first[0], sum(first), cols[0], cols[1], side))
    bounds = [C._estimate_2x2(r1, c0, c1) for _, r1, c0, c1, _ in items]
    rounds = max(C._rounds_2x2(r1, c0, c1) for _, r1, c0, c1, _ in items)
    together, _ = C._fisher_2x2_many(
        items, bounds, sum(bounds) + rounds * C._UNIT_ROUND
    )
    for item, p_value in zip(items, together, strict=True):
        a, r1, c0, c1, side = item
        alone = C._fisher_2x2(a, r1, c0, c1, side, C._Budget(C._FISHER_WORK_LIMIT))
        assert p_value == alone, item


def test_the_planners_closed_forms_equal_the_sums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _plan_bound_work equals _bound_work over the rotated sizes; the running
    # sums equal the convolution; the long network side through NumPy equals
    # the loop; _certain_floor is below every plan's certain work.
    rng = np.random.default_rng(9)
    for _ in range(60):
        k = int(rng.integers(2, 90))
        sizes = sorted(int(v) for v in rng.integers(2, 40, k))
        prefix = [0]
        for s in sizes:
            prefix.append(prefix[-1] + s - 1)
        sum_p = [0]
        sum_ip = [0]
        for i, v in enumerate(prefix):
            sum_p.append(sum_p[-1] + v)
            sum_ip.append(sum_ip[-1] + i * v)
        for q in (1, k // 2, k):
            rotated = sizes[q:] + sizes[:q]
            assert C._plan_bound_work(sizes, q, prefix, sum_p, sum_ip) == (
                C._bound_work(rotated, k - q)
            )
        counts = np.ones(1)
        for s in sizes[:4]:
            assert np.array_equal(
                C._window_sums(counts, s), np.convolve(counts, np.ones(s))
            )
            counts = np.convolve(counts, np.ones(s))
        widths = [s - 1 for s in sizes]
        plan = C._fisher_plan(widths)
        assert C._certain_floor(k) <= plan.certain
        monkeypatch.setattr(C, "_LONG_NETWORK", 0)
        assert C._fisher_plan(widths) == plan
        monkeypatch.setattr(C, "_LONG_NETWORK", 64)


# --------------------------------------------------------------------------- #
# p = 0 only below the smallest double
# --------------------------------------------------------------------------- #

# 2 x 3 tables whose exact p lies between 1e-320 and 1e-300 (subnormal or near),
# and whose shortcut bound as the search computes it, (number of first rows in
# the box of the columns' ranges) x P_obs (1 + 1e-7), is below 1e-304: a
# shortcut that rounded p to 0 from about 1e-304 instead of 2^-1075 (the
# reviewer's planted bug) would report 0 for each.
TINY: Final = [
    [[575, 6, 0], [0, 500, 5]],
    [[550, 3, 0], [0, 500, 12]],
    [[525, 3, 0], [0, 500, 40]],
    [[575, 1, 0], [0, 500, 2]],
    [[550, 3, 0], [0, 500, 40]],
]


def _exact_p_2x3(table: list[list[int]]) -> tuple[int, int, int]:
    """The exact two-sided p as counted / total (integers; a table counts when
    its weight is at most the observed one times 1 + 1e-7, R's rule), and the
    observed weight times the number of first rows in the box of the columns'
    ranges (the shortcut's bound, over total)."""
    first = table[0]
    cols = [a + b for a, b in zip(table[0], table[1], strict=True)]
    r1 = sum(first)
    weights = [[math.comb(c, a) for a in range(c + 1)] for c in cols]
    observed = math.prod(weights[j][first[j]] for j in range(3))
    counted = 0
    for a0 in range(cols[0] + 1):
        for a1 in range(max(0, r1 - a0 - cols[2]), min(cols[1], r1 - a0) + 1):
            w = weights[0][a0] * weights[1][a1] * weights[2][r1 - a0 - a1]
            if w * 10**7 <= observed * (10**7 + 1):
                counted += w
    r2 = sum(cols) - r1
    box = math.prod(min(c, r1) - max(0, c - r2) + 1 for c in cols)
    return counted, math.comb(sum(cols), r1), box * observed


@pytest.mark.parametrize("table", TINY, ids=[str(t) for t in TINY])
def test_a_p_value_between_1e_320_and_1e_300_is_not_rounded_to_0(
    table: list[list[int]],
) -> None:
    counted, total, bound = _exact_p_2x3(table)
    exact = counted / total  # correctly rounded, subnormals included
    assert 1e-320 < exact < 1e-300
    log_bound = math.log(bound) + math.log1p(1e-7) - math.log(total)
    assert -1075 * math.log(2.0) < log_bound < math.log(1e-304)
    p = fisher_exact(table).p_value
    assert p > 0.0
    # Relative 1e-12 for a normal p; a subnormal holds fewer digits, so an
    # absolute two units of the smallest subnormal as well.
    assert abs(p - exact) <= 1e-12 * exact + 2 * 5e-324, (p, exact)
