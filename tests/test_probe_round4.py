"""What R does where review round 4 could only reason about it.

Two probes (oracle/generate.R, "Probes of review round 4"). Each test states an
answer as a fact about R 4.5.3 with ggstatsplot 1.1.4 and re-computes it
independently, the Fisher p-values in exact integer arithmetic.

1. Fisher's exact test of a table larger than 2 x 2 (probe-fisherties). The
   two-sided p sums the probabilities of every table with the observed margins
   that is "at most as probable" as the observed one. For 2 x 2 tables
   ``fisher.test`` allows a relative 1e-7 (documented, and confirmed by
   probe-contingency). For larger tables it allows more: on 194 tables (105 2 x 3,
   81 2 x 4 and 8 3 x 3, n up to 300), each with one other table whose probability
   lies just above or just below the observed one's, every p R gives is the exact
   p of the rule "P <= P_obs (1 + t)" for each t with 3.44838e-7 <= t <
   3.45432e-7 and for no t outside (the ends: two 2 x 3 tables whose other table
   lies at 1 + 3.44838e-7, counted, and at 1 + 3.45432e-7, left out). That is an
   allowance of about 3.45e-7, the same on the log scale. R's p agrees with the
   exact p of that rule within 3.8e-12 relative; raising ``workspace`` (with
   ``hybrid = FALSE`` stated) moves it by at most 3.7e-15. ggbarstats' pairwise
   Fisher p is ``fisher.test``'s, bit for bit. inferplot's ``fisher_exact`` allows
   1e-7 for every shape, so on the 45 2 x c tables whose other table lies between
   1 + 1e-7 and 1 + 3.45e-7 it leaves that table out and its p is lower than R's
   by 0.38% to 7.9% (``FISHER_PINNED``; the lead decides). On the other 141 2 x c
   tables it agrees with R within 3.8e-12.
2. R's ``round(x, d)`` far from percent labels (probe-roundbig). R returns x
   unchanged when d + log10(2) (e + 1/2) > 15, e the binary exponent of |x| (2^e <=
   |x| < 2^(e + 1)), and otherwise rounds as ``?Round`` describes. That reproduces
   all 482 values from 1e6 to 1e17 at d = 1 to 8 (352 of them change when
   rounded); no rule d + log10|x| > 15 + delta does, for any delta.
   ``inferplot.text.r_round`` keeps x when d + log10|x| > 15 and differs from R on
   43 of them, all where the two cutoffs part (|x| >= 10^(14.85 - d)). Percent
   labels (|x| <= 100) never get there for d <= 13.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

import pytest

from inferplot import text as tx
from inferplot.stats import fisher_exact
from tests.oracle_fixtures import Row, probe, r_double


def _hex(text: object) -> float:
    value = r_double(str(text))
    assert value is not None
    return value


# 1. Fisher's exact test: which tables count as "at most as probable" ---

Table = tuple[tuple[int, ...], ...]
UNBOUNDED = Fraction(10**9)  # above every ratio of these tables


def _table(row: Row) -> Table:
    return tuple(tuple(int(v) for v in r) for r in row["table"])


def _fillings(rows: Sequence[int], cols: Sequence[int]) -> Iterator[Table]:
    """Every table of non-negative integers with these row and column sums."""
    if len(rows) == 1:
        yield (tuple(cols),)
        return

    def first_row(j: int, left: int, acc: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if j == len(cols) - 1:
            if left <= cols[j]:
                yield (*acc, left)
            return
        for v in range(min(left, cols[j]) + 1):
            yield from first_row(j + 1, left - v, (*acc, v))

    for head in first_row(0, rows[0], ()):
        rest = [c - v for c, v in zip(cols, head, strict=True)]
        for tail in _fillings(rows[1:], rest):
            yield (head, *tail)


@cache
def _weights(table: Table) -> tuple[list[int], int]:
    """Integers proportional to the probability, given both margins, of every
    table with the margins of ``table`` (n! / prod n_ij!), and the observed one's.
    For two rows the same up to a constant: prod_j C(c_j, a_j), a the first row."""
    cols = [sum(c) for c in zip(*table, strict=True)]
    if len(table) == 2:
        r1 = sum(table[0])
        combs = [[math.comb(c, a) for a in range(c + 1)] for c in cols]
        out: list[int] = []
        for head in itertools.product(*(range(c + 1) for c in cols[:-1])):
            rest = r1 - sum(head)
            if 0 <= rest <= cols[-1]:
                w = combs[-1][rest]
                for comb, a in zip(combs, head, strict=False):
                    w *= comb[a]
                out.append(w)
        observed = math.prod(
            math.comb(c, a) for c, a in zip(cols, table[0], strict=True)
        )
        return out, observed
    n = sum(cols)
    fact = [math.factorial(k) for k in range(n + 1)]

    def weight(t: Table) -> int:
        return fact[n] // math.prod(fact[v] for r in t for v in r)

    rows = [sum(r) for r in table]
    return [weight(t) for t in _fillings(rows, cols)], weight(table)


@dataclass(frozen=True)
class _Steps:
    """The two-sided p under the rule "count a table when its probability is at
    most the observed one's times 1 + t", as a step function of t, in integers:
    the distinct weights w (increasing), the weight of every table up to each
    (cums), the observed table's weight and the total. Rules whose 1 + t lies in
    [w_k / observed, w_(k+1) / observed) give p = cums[k] / total."""

    weights: list[int]
    cums: list[int]
    observed: int
    total: int

    def ratio(self, k: int) -> Fraction:
        return Fraction(self.weights[k], self.observed)

    def p(self, k: int) -> float:
        return self.cums[k] / self.total  # correctly rounded


@cache
def _steps(table: Table) -> _Steps:
    weights, observed = _weights(table)
    distinct: list[int] = []
    cums: list[int] = []
    cum = 0
    for w, group in itertools.groupby(sorted(weights)):
        cum += w * len(list(group))
        distinct.append(w)
        cums.append(cum)
    return _Steps(distinct, cums, observed, cum)


def _p_at(table: Table, t: Fraction) -> Fraction:
    """The exact two-sided p under the rule ratio <= 1 + t."""
    s = _steps(table)
    # The tables with w t.den <= observed (t.den + t.num).
    limit = s.observed * (t.denominator + t.numerator)
    k = bisect.bisect_right(s.weights, limit // t.denominator) - 1
    while k >= 0 and s.weights[k] * t.denominator > limit:
        k -= 1
    return Fraction(s.cums[k], s.total) if k >= 0 else Fraction(0)


def _near(table: Table, rel: Fraction) -> list[Fraction]:
    """The ratios w / observed of the other tables within ``rel`` of the observed
    probability, exact ties (ratio 1) left out."""
    s = _steps(table)
    low = bisect.bisect_left(s.weights, math.floor(s.observed * (1 - rel)))
    high = bisect.bisect_right(s.weights, math.ceil(s.observed * (1 + rel)))
    return [
        s.ratio(k)
        for k in range(low, high)
        if s.weights[k] != s.observed and abs(s.ratio(k) - 1) <= rel
    ]


def _nearest(table: Table) -> Fraction:
    """The ratio to the observed probability of the nearest other table that is
    not an exact tie (ratio 1)."""
    s = _steps(table)
    k = bisect.bisect_left(s.weights, s.observed)
    neighbours = [j for j in (k - 1, k, k + 1) if 0 <= j < len(s.weights)]
    ratios = [s.ratio(j) for j in neighbours if s.weights[j] != s.observed]
    return min(ratios, key=lambda r: abs(r - 1))


def _rule_interval(table: Table, p: float, rel_tol: float) -> tuple[Fraction, Fraction]:
    """The rules ratio <= 1 + t, t in [low, high), whose exact p equals ``p``
    within ``rel_tol``; exactly one step of the step function must match."""
    s = _steps(table)
    k = bisect.bisect_left(s.cums, math.floor(p * s.total))
    matches = [
        j
        for j in range(max(k - 2, 0), min(k + 3, len(s.cums)))
        if abs(s.p(j) - p) <= rel_tol * p
    ]
    assert len(matches) == 1, (table, p, matches)
    k = matches[0]
    assert k == 0 or abs(s.p(k - 1) - p) > rel_tol * p
    assert k + 1 == len(s.cums) or abs(s.p(k + 1) - p) > rel_tol * p
    high = s.ratio(k + 1) - 1 if k + 1 < len(s.cums) else UNBOUNDED
    return s.ratio(k) - 1, high


R_REL_TOL = 4e-12  # R's p against the exact p of its rule: measured 3.74e-12

# The ends of the interval of t that every probe table allows: the near ties of two
# 2 x 3 tables, the first counted, the second left out.
R_RULE_LOW = (((93, 11, 11), (100, 5, 42)), 3.44838453e-7)
R_RULE_HIGH = (((20, 48, 28), (47, 33, 57)), 3.45432457e-7)


def _fisher_rows() -> list[Row]:
    return probe("probe-fisherties").rows("tables")


def test_fisher_ties_probe_covers_its_design() -> None:
    rows = _fisher_rows()
    tables = [_table(row) for row in rows]
    assert len(set(tables)) == len(tables) == 194
    shapes = [(len(t), len(t[0])) for t in tables]
    assert {s: shapes.count(s) for s in set(shapes)} == {
        (2, 3): 105,
        (2, 4): 81,
        (3, 3): 8,
    }
    assert max(sum(map(sum, t)) for t in tables) <= 300
    assert tables[:3] == [  # the reviewer's
        ((13, 18, 60), (15, 13, 14)),
        ((35, 25, 45), (1, 29, 12)),
        ((34, 54, 17), (23, 29, 71)),
    ]
    for row in rows:
        for call in ("default", "exact_workspace"):
            result = row[call]
            assert result["error"] is None
            assert result["warnings"] == []
            assert result["method"] == "Fisher's Exact Test for Count Data"
            assert float(result["p_value_17"]) == _hex(result["p_value_hex"])
        # Raising the workspace (hybrid = FALSE is the default) changes only the
        # last bits.
        default = _hex(row["default"]["p_value_hex"])
        raised = _hex(row["exact_workspace"]["p_value_hex"])
        assert math.isclose(default, raised, rel_tol=4e-15)


def test_every_table_has_one_near_tie() -> None:
    # The design: each table has exactly one other table within 2e-5 relative of
    # its probability (exact ties aside), above or below, from 1.8e-10 to 1e-5 away.
    above: list[float] = []
    below: list[float] = []
    for row in _fisher_rows():
        table = _table(row)
        near = _near(table, Fraction(2, 10**5))
        assert near == [_nearest(table)], table
        rel = float(near[0] - 1)
        (above if rel > 0 else below).append(rel)
    assert (len(above), len(below)) == (ABOVE, BELOW)
    assert min(above) == pytest.approx(1.7776e-10, rel=1e-4)
    assert max(below) == pytest.approx(-1.7776e-10, rel=1e-4)
    assert max(above) <= 1e-5
    assert min(below) >= -1e-5
    # Tables near the edge of R's allowance, on either side of it.
    assert sum(3.4e-7 < r <= 3.45e-7 for r in above) == EDGE_BELOW
    assert sum(3.45e-7 < r <= 3.5e-7 for r in above) == EDGE_ABOVE


ABOVE, BELOW = 99, 95  # the reviewer's three tables among the 99
EDGE_BELOW, EDGE_ABOVE = 6, 5


def test_fisher_rule_for_tables_larger_than_2x2() -> None:
    low, high = Fraction(-1), UNBOUNDED
    by_shape: dict[tuple[int, int], tuple[Fraction, Fraction]] = {}
    for row in _fisher_rows():
        table = _table(row)
        shape = (len(table), len(table[0]))
        for call in ("default", "exact_workspace"):
            p = _hex(row[call]["p_value_hex"])
            t_low, t_high = _rule_interval(table, p, R_REL_TOL)
            low, high = max(low, t_low), min(high, t_high)
            s_low, s_high = by_shape.get(shape, (Fraction(-1), UNBOUNDED))
            by_shape[shape] = (max(s_low, t_low), min(s_high, t_high))
    # One relative allowance t fits every table, and the probe pins it to
    # [3.44838e-7, 3.45432e-7): the ratios of a table counted and a table left out.
    assert low < high
    for (table, rel), end in ((R_RULE_LOW, low), (R_RULE_HIGH, high)):
        assert end == _nearest(table) - 1
        assert float(end) == pytest.approx(rel, rel=1e-8)
    # Not the 2 x 2 rule (1e-7), nor "at most" without an allowance; 3.45e-7 is
    # (relative, or on the log scale: exp(3.45e-7) - 1).
    assert not low <= Fraction(1, 10**7) < high
    assert not low <= 0 < high
    assert low <= Fraction(345, 10**9) < high
    assert low <= Fraction(math.expm1(3.45e-7)) < high
    # Each shape alone is consistent with the same t; the tight ends come from the
    # 2 x 3 tables.
    assert by_shape[2, 3] == (low, high)
    for shape in ((2, 4), (3, 3)):
        s_low, s_high = by_shape[shape]
        assert s_low <= low
        assert high <= s_high
    assert float(by_shape[2, 4][0]) == pytest.approx(3.4411e-7, rel=1e-4)
    assert float(by_shape[3, 3][0]) == pytest.approx(2.5660e-7, rel=1e-4)


def test_tables_just_below_the_observed_one_always_count() -> None:
    # The controls: a table slightly less probable (ratio 1 - 1.8e-10 to 1 - 1e-5)
    # counts under every rule t >= 0, and R counts it.
    controls = 0
    for row in _fisher_rows():
        table = _table(row)
        if _nearest(table) > 1:
            continue
        controls += 1
        expected = float(_p_at(table, Fraction(0)))
        assert math.isclose(
            _hex(row["default"]["p_value_hex"]), expected, rel_tol=R_REL_TOL
        )
    assert controls == BELOW


def test_reviewers_tables_count_the_near_tie() -> None:
    rows = {_table(row): row for row in _fisher_rows()}
    for table, rel, without, with_tie in (
        (((13, 18, 60), (15, 13, 14)), 3.3759e-7, 0.0012610025, 0.0013059799),
        (((35, 25, 45), (1, 29, 12)), 1.2391e-7, 8.8225e-08, 9.1995e-08),
        (((34, 54, 17), (23, 29, 71)), 3.3293e-7, 3.3141e-10, 3.3702e-10),
    ):
        assert float(_nearest(table) - 1) == pytest.approx(rel, rel=1e-4)
        assert float(_p_at(table, Fraction(1, 10**7))) == pytest.approx(
            without, rel=1e-4
        )
        assert float(_p_at(table, Fraction(345, 10**9))) == pytest.approx(
            with_tie, rel=1e-4
        )
        assert _hex(rows[table]["default"]["p_value_hex"]) == pytest.approx(
            with_tie, rel=1e-4
        )


def test_ggbarstats_pairwise_fisher_p_is_fisher_test() -> None:
    # ggstatsplot passes nothing that changes the rule: the pair a-b, whose table is
    # the probe's 2 x 3 table, gets fisher.test's default p bit for bit, and the
    # pairs with the third level c are exact Fisher p-values too.
    rows = {_table(row): row for row in _fisher_rows()}
    pairs = probe("probe-fisherties").rows("pairs")
    assert [_table(pair) for pair in pairs] == [
        ((13, 18, 60), (15, 13, 14)),
        ((29, 21, 60), (39, 16, 96)),
    ]
    for pair in pairs:
        table = _table(pair)
        third = tuple(int(v) for v in pair["third"])
        assert pair["error"] is None
        assert pair["warnings"] == []
        by_pair = {(r["group1"], r["group2"]): r for r in pair["pairwise"]}
        assert list(by_pair) == [("a", "b"), ("a", "c"), ("b", "c")]
        assert {r["test"] for r in pair["pairwise"]} == {"Fisher's exact test"}
        assert by_pair["a", "b"]["p_value_hex"] == rows[table]["default"]["p_value_hex"]
        for key, sub in (
            (("a", "c"), (table[0], third)),
            (("b", "c"), (table[1], third)),
        ):
            p = _hex(by_pair[key]["p_value_hex"])
            t_low, t_high = _rule_interval(sub, p, R_REL_TOL)
            assert t_low <= Fraction(345, 10**9) < t_high


# inferplot's fisher_exact counts a table when P <= P_obs (1 + 1e-7), for every
# shape (stats/contingency.py, _RELATIVE_TIE); R allows about 3.45e-7 for tables
# larger than 2 x 2. On these 2 x c tables the other table lies between 1 + 1e-7
# and 1 + 3.45e-7: R counts it, inferplot does not. table -> (R's p, inferplot's
# p). A deliberate difference until the lead decides (to follow R: an allowance t
# with 3.44838e-7 <= t < 3.45432e-7 for c >= 3, 1e-7 for 2 x 2).
FISHER_PINNED: dict[Table, tuple[float, float]] = {
    ((13, 18, 60), (15, 13, 14)): (0.0013059798779535934, 0.0012610025028378833),
    ((35, 25, 45), (1, 29, 12)): (9.199547847338761e-08, 8.822549057332907e-08),
    ((34, 54, 17), (23, 29, 71)): (3.370240050130255e-10, 3.314120547340896e-10),
    ((2, 49, 36), (15, 15, 120)): (4.402389372704729e-14, 4.139158885760152e-14),
    ((6, 31, 0), (15, 93, 31)): (0.0010241302679134972, 0.0009666691358365084),
    ((12, 22, 22), (41, 58, 0)): (1.910620700524422e-11, 1.8104910782317604e-11),
    ((29, 55, 0), (37, 165, 9)): (0.0016572842013058272, 0.001572993876219911),
    ((27, 49, 5), (56, 14, 13)): (2.3534548587945683e-08, 2.239863514031993e-08),
    ((6, 86, 7), (25, 83, 3)): (0.0013670941151442979, 0.0013057442509399218),
    ((19, 7, 21), (103, 3, 22)): (1.2394459541000818e-06, 1.1476576737892165e-06),
    ((19, 27, 13), (23, 113, 2)): (3.740555119381527e-08, 3.491371752770361e-08),
    ((6, 17, 24), (3, 81, 98)): (0.005853496038989664, 0.005524518967692021),
    ((20, 15, 10), (40, 0, 27)): (5.755038740030897e-07, 5.432413354508208e-07),
    ((1, 20, 6), (10, 40, 28)): (0.13531681960680528, 0.12824554631222046),
    ((28, 7, 20), (71, 69, 4)): (2.9153104230195943e-11, 2.7658950846947964e-11),
    ((13, 2, 26), (112, 23, 11)): (1.364300157048733e-12, 1.2566363819238064e-12),
    ((27, 7, 39), (23, 1, 122)): (2.2733707589970015e-06, 2.108571792976295e-06),
    ((3, 59, 3), (5, 85, 24)): (0.006511152807351097, 0.006073754818602875),
    ((13, 8, 30), (1, 9, 56)): (0.00014907171333116985, 0.00013977303390457624),
    ((14, 8, 30), (2, 9, 56)): (0.00035210945755288894, 0.00033233413100868225),
    ((16, 29, 5), (9, 23, 56)): (8.060246300714873e-10, 7.710507328894335e-10),
    ((44, 8, 8), (115, 95, 9)): (8.038800690152423e-06, 7.722927755150175e-06),
    ((93, 11, 11), (100, 5, 42)): (0.00014807037253135137, 0.00014399437040190854),
    ((34, 9, 67), (80, 17, 43)): (8.91977310043489e-06, 8.740464771624777e-06),
    ((82, 35, 7), (72, 11, 52)): (6.896141988316556e-12, 6.76325680220148e-12),
    ((29, 21, 60), (39, 16, 96)): (0.1341095152604861, 0.13191230847146454),
    ((0, 15, 7, 5), (33, 42, 15, 1)): (5.866698719773279e-06, 5.795765799052731e-06),
    ((15, 20, 5, 1), (4, 62, 2, 27)): (1.1911090485249129e-08, 1.1777453213234553e-08),
    ((15, 0, 12, 0), (7, 12, 42, 31)): (1.5155524550037544e-08, 1.501339822288599e-08),
    ((2, 8, 38, 6), (47, 2, 29, 7)): (5.951148429180323e-11, 5.902712696969411e-11),
    ((9, 15, 11, 0), (1, 42, 11, 31)): (8.727665264472667e-09, 8.659953727904987e-09),
    ((25, 4, 6, 5), (53, 4, 10, 30)): (0.0748498776953722, 0.07436329842396797),
    ((5, 0, 26, 0), (0, 17, 42, 25)): (1.8123080483358923e-08, 1.7715299723263268e-08),
    ((2, 10, 8, 7), (62, 27, 8, 2)): (1.0277108551228088e-08, 1.0137727804354938e-08),
    ((28, 14, 4, 2), (24, 8, 1, 16)): (0.001070841349475393, 0.0010626534512635552),
    ((16, 4, 2, 2), (23, 16, 28, 9)): (0.0075916751674047446, 0.007536879751747009),
    ((11, 11, 2, 6), (62, 11, 16, 17)): (0.0059061813467699426, 0.005873963963149396),
    ((22, 11, 11, 0), (11, 40, 3, 15)): (1.340737985426193e-08, 1.3338976346611462e-08),
    ((18, 10, 30, 1), (0, 3, 56, 16)): (6.435899347785298e-11, 6.328072196726265e-11),
    ((17, 30, 1, 9), (3, 10, 4, 56)): (3.1797943249989924e-13, 3.136526544586171e-13),
    ((9, 5, 0, 12), (29, 56, 8, 10)): (5.53456449202222e-05, 5.484217712039365e-05),
    ((10, 6, 10, 2), (29, 56, 2, 16)): (1.7497641770066544e-06, 1.7356987583235606e-06),
    ((8, 5, 7, 29), (9, 12, 1, 56)): (0.020214914129227858, 0.020091607587326605),
    ((24, 12, 12, 2), (30, 29, 23, 4)): (0.4917727543335654, 0.48898016683259093),
    ((13, 7, 15, 5), (3, 17, 23, 53)): (7.910266746517538e-08, 7.879883405796543e-08),
}


def test_inferplot_fisher_differs_from_r_as_pinned() -> None:
    seen: set[Table] = set()
    agree = 0
    for row in _fisher_rows():
        table = _table(row)
        if len(table) != 2:
            continue  # inferplot tests 2 x c tables only
        r_p = _hex(row["default"]["p_value_hex"])
        ours = fisher_exact([list(r) for r in table]).p_value
        # inferplot's p is the exact p of its own rule.
        assert math.isclose(
            ours, float(_p_at(table, Fraction(1, 10**7))), rel_tol=R_REL_TOL
        )
        if table in FISHER_PINNED:
            seen.add(table)
            pinned_r, pinned_ours = FISHER_PINNED[table]
            assert r_p == pinned_r
            assert math.isclose(ours, pinned_ours, rel_tol=1e-13)
            assert Fraction(1, 10**7) < _nearest(table) - 1 < Fraction(345, 10**9)
            assert 3.8e-3 < (r_p - ours) / r_p < 7.9e-2
        else:
            agree += 1
            assert math.isclose(ours, r_p, rel_tol=R_REL_TOL), table
    assert seen == set(FISHER_PINNED)
    assert len(seen) == 45
    assert agree == 141


# 2. R's round() far from percent labels ---


def _documented_round(x: float, digits: int) -> float:
    """round(x, digits) for digits >= 1 and x > 0 as ``?Round`` describes it: the
    nearer of x rounded down and up to that decimal digit, in double arithmetic,
    the even one on a tie of distances (as in tests/test_probe_round3.py)."""
    scale = 10.0**digits
    down = math.floor(x * scale)
    low, high = down / scale, math.ceil(x * scale) / scale
    up_distance, down_distance = high - x, x - low
    if up_distance < down_distance or (up_distance == down_distance and down % 2 == 1):
        return high
    return low


def _binary_exponent(x: float) -> int:
    """e with 2^e <= |x| < 2^(e + 1), exactly (math.log2 rounds up just below a
    power of two)."""
    return math.frexp(x)[1] - 1


def _r_keeps(x: float, digits: int) -> bool:
    """R returns x itself (the rule the probe shows)."""
    return digits + math.log10(2) * (_binary_exponent(x) + 0.5) > 15


def _decimal_keeps(x: float, digits: int) -> bool:
    """inferplot's r_round returns x itself."""
    return digits + math.log10(abs(x)) > 15


def _round_rows() -> list[tuple[float, int, float]]:
    rows = probe("probe-roundbig").rows("values")
    return [(_hex(r["x_hex"]), int(r["digits"]), _hex(r["rounded_hex"])) for r in rows]


def test_round_keeps_x_by_its_binary_exponent() -> None:
    rows = _round_rows()
    assert len(rows) == 482
    assert min(x for x, _, _ in rows) > 1e6
    assert max(x for x, _, _ in rows) < 1.2e17
    assert {d for _, d, _ in rows} == set(range(1, 9))
    informative = 0
    c_low, c_high = -math.inf, math.inf
    delta_low, delta_high = -math.inf, math.inf
    for x, d, r_value in rows:
        rounded = _documented_round(x, d)
        # R keeps x when d + log10(2) (e + 1/2) > 15, and rounds otherwise.
        assert r_value == (x if _r_keeps(x, d) else rounded), (x, d)
        if rounded == x:
            continue
        informative += 1
        # Which c (keep when d + log10(2) (e + c) > 15) and which delta (keep when
        # d + log10 x > 15 + delta) fit this value.
        c_edge = (15 - d) / math.log10(2) - _binary_exponent(x)
        delta_edge = d + math.log10(x) - 15
        if r_value == x:
            c_low, delta_high = max(c_low, c_edge), min(delta_high, delta_edge)
        else:
            c_high, delta_low = min(c_high, c_edge), max(delta_low, delta_edge)
    assert informative == 352
    # The probe pins c to (0.2535, 0.5070]; no decimal cutoff fits at all.
    assert c_low < 0.5 <= c_high
    assert c_low == pytest.approx(0.25350, abs=1e-4)
    assert c_high == pytest.approx(0.50699, abs=1e-4)
    assert delta_low > delta_high


def test_reviewers_round_examples() -> None:
    rows = {(x, d): r for x, d, r in _round_rows()}
    assert rows[1000000999.9999999, 6] == 1000001000.0
    assert rows[100000099999.99998, 4] == 100000100000.0
    # inferplot keeps both as they are (pinned below).
    assert tx.r_round(1000000999.9999999, 6) == 1000000999.9999999
    assert tx.r_round(100000099999.99998, 4) == 100000099999.99998


# inferplot.text.r_round keeps x when d + log10|x| > 15; R decides by the binary
# exponent (above). They part only where log10|x| and log10(2) (e + 1/2) lie on
# either side of 15 - d, never for percent labels. A deliberate difference until
# the lead decides; the count and two examples, one in each direction.
ROUND_DIFFERENCES = 43


def test_r_round_differs_from_r_far_from_percent_labels_as_pinned() -> None:
    differ: list[tuple[float, int, float, float]] = []
    for x, d, r_value in _round_rows():
        ours = tx.r_round(x, d)
        assert ours == (x if _decimal_keeps(x, d) else _documented_round(x, d))
        if ours != r_value:
            differ.append((x, d, r_value, ours))
            assert _decimal_keeps(x, d) != _r_keeps(x, d)
            assert math.log10(x) >= 14.85 - d
    assert len(differ) == ROUND_DIFFERENCES
    # R rounds where inferplot keeps x, and keeps x where inferplot rounds.
    assert (1073741823.9999999, 6, 1073741824.0, 1073741823.9999999) in differ
    assert (10000000000000.004, 2, 10000000000000.004, 10000000000000.0) in differ
    # Percent labels: for 0 < |x| <= 100 both cutoffs round, up to 13 digits. Both
    # grow with |x| within a binade, so the largest value of each binade decides.
    for d in range(14):
        for e in range(-1074, 7):
            top = min(math.nextafter(math.ldexp(1.0, e + 1), 0.0), 100.0)
            assert not _decimal_keeps(top, d)
            assert not _r_keeps(top, d)
