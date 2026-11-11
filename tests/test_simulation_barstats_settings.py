"""Fast checks of the barstats simulation settings (SPEC.md section 11).

The slow tests in ``tests/test_simulation_barstats.py`` go through
``inferplot.stats`` (except one setting) and centre most bands on rates computed
exactly by enumeration in ``tests/simulation_settings.py``. These tests check:

* that the enumerations count inferplot's own p-values and intervals, table by
  table, on small designs (the exact rates use closed forms and their own
  noncentral chi-square CDF, not inferplot);
* that the shortcuts the slow tests take (``two_way_table``, ``paired_table``,
  ``one_way_counts``: which table barstats tests, or whether it refuses; the
  per-bar and pairwise p-values without the effect sizes; Cochran's rule) give
  what ``ip.analyze.barstats`` gives, refusals included, on tables drawn to hit
  every refusal;
* the noncentral chi-square CDF against the 40-digit mpmath values in
  ``tests/data/ncx2_reference.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import inferplot as ip
from inferplot import check
from inferplot import stats as st
from tests import simulation_settings as ss
from tests.tolerances import assert_close

NCX2_REFERENCE = Path(__file__).resolve().parent / "data" / "ncx2_reference.json"


def _table(row: object, shape: tuple[int, int]) -> ss.IntTable:
    return np.asarray(row, dtype=np.int64).reshape(shape)


def test_compositions_are_the_multinomial_support() -> None:
    for n, m in ((0, 3), (1, 1), (5, 2), (7, 4), (12, 3)):
        rows = ss.compositions(n, m)
        assert rows.shape == (math.comb(n + m - 1, m - 1), m)
        assert bool((rows.sum(axis=1) == n).all())
        assert bool((rows >= 0).all())
        assert len(np.unique(rows, axis=0)) == len(rows)
    weights = ss.multinomial_pmf(ss.compositions(9, 4), (0.1, 0.2, 0.3, 0.4))
    assert math.fsum(weights.tolist()) == pytest.approx(1.0, abs=1e-13)


def test_ncx2_cdf_matches_the_mpmath_reference() -> None:
    # The series sums at most about 100 positive terms, each with the relative
    # error of SciPy's regularized incomplete gamma (about 1e-14): 1e-10 relative
    # leaves a hundredfold headroom. Only the small noncentralities the settings
    # use (up to 30) and lower tails above 1e-12.
    rows = json.loads(NCX2_REFERENCE.read_text(encoding="utf-8"))["cdf"]
    checked = 0
    for row in rows:
        ncp, df, x = float(row["ncp"]), float(row["df"]), float(row["x"])
        lower = 10.0 ** float(row["log10_lower"])
        if ncp > 30.0 or df > 10.0 or lower < 1e-12:
            continue
        value = float(ss.ncx2_cdf(np.array([x]), df, ncp)[0])
        assert value == pytest.approx(lower, rel=1e-10), row
        checked += 1
    assert checked >= 15


def test_exact_two_by_two_counts_inferplots_tests_and_intervals() -> None:
    n, p, q = 11, (0.3, 0.7), (0.45, 0.55)
    exact = ss.exact_two_by_two(n, p, q)
    tables = ss.compositions(n, 4)
    weights = ss.multinomial_pmf(tables, ss.outer(p, q))
    sums: dict[str, float] = dict.fromkeys(
        ("p", "v0", "v0_greater", "refused", "flagged"), 0.0
    )
    for row, weight in zip(tables, weights.tolist(), strict=True):
        kept = ss.two_way_table(_table(row, (2, 2)))
        if kept is None:
            sums["refused"] += weight
            continue
        test = st.chisq_independence(kept)
        two = st.cramers_v(kept)
        greater = st.cramers_v(kept, alternative="greater")
        sums["p"] += weight * (test.p_value < ss.ALPHA)
        sums["v0"] += weight * ss.covers(two.conf_low, two.conf_high, 0.0)
        sums["v0_greater"] += weight * ss.covers(
            greater.conf_low, greater.conf_high, 0.0
        )
        sums["flagged"] += weight * ss.cochran_fails(np.asarray(test.expected))
    for key, value in sums.items():
        assert exact[key] == pytest.approx(value, abs=1e-12), key
    assert exact["p"] == pytest.approx(
        exact["p_flagged"] + exact["p_unflagged"], abs=1e-15
    )


@pytest.mark.parametrize(
    ("probs", "ratio"),
    [
        ((1 / 3, 1 / 3, 1 / 3), None),
        ((0.5, 0.3, 0.2), None),
        ((0.5, 0.3, 0.2), (0.5, 0.3, 0.2)),
        ((0.25, 0.25, 0.5), (0.4, 0.3, 0.3)),
    ],
)
def test_exact_one_way_counts_inferplots_tests_and_intervals(
    probs: tuple[float, ...], ratio: tuple[float, ...] | None
) -> None:
    n = 9
    exact = ss.exact_one_way(n, probs, ratio)
    p0 = np.full(3, 1 / 3) if ratio is None else np.asarray(ratio)
    phi2 = float(((np.asarray(probs) - p0) ** 2 / p0).sum())
    true_c = math.sqrt(phi2 / (1.0 + phi2))
    outcomes = ss.compositions(n, 3)
    weights = ss.multinomial_pmf(outcomes, probs)
    sums: dict[str, float] = dict.fromkeys(
        ("p", "dropped", "refused", "c_two", "c_greater", "c_less"), 0.0
    )
    for row, weight in zip(outcomes, weights.tolist(), strict=True):
        read = ss.one_way_counts(np.asarray(row, dtype=np.int64), ratio)
        if read is None:
            sums["refused"] += weight
            continue
        counts, dropped = read
        sums["dropped"] += weight * dropped
        sums["p"] += weight * (st.goodness_of_fit(counts, ratio).p_value < ss.ALPHA)
        for side, alternative in ss.SIDES.items():
            es = st.pearsons_c(counts, ratio, alternative=alternative)
            sums[f"c_{side}"] += weight * ss.covers(es.conf_low, es.conf_high, true_c)
    for key, value in sums.items():
        assert exact[key] == pytest.approx(value, abs=1e-12), key


@pytest.mark.parametrize(("cells", "n"), [(ss.MC_FEW, 30), (ss.G_CELLS, 40)])
def test_exact_paired_counts_inferplots_tests_and_intervals(
    cells: tuple[tuple[float, ...], ...], n: int
) -> None:
    # Every table of n pairs (5,456 and 12,341), through paired_table and
    # inferplot, at two of the settings' designs; the exact rates leave out the
    # refusals for a level that one measurement lacks, whose probability is
    # checked to be below 1e-8 here.
    exact = ss.exact_paired(n, cells)
    true_g = ss.cohens_g_of(cells)
    tables = ss.compositions(n, 4)
    weights = ss.multinomial_pmf(tables, ss.cells_flat(cells))
    sums: dict[str, float] = dict.fromkeys(("p", "g_two", "g_greater", "g_less"), 0.0)
    no_discordant = lacking = 0.0
    for row, weight in zip(tables, weights.tolist(), strict=True):
        table = _table(row, (2, 2))
        kept = ss.paired_table(table)
        if kept is None:
            if table[0, 1] + table[1, 0] == 0:
                no_discordant += weight
            else:
                lacking += weight
            continue
        sums["p"] += weight * (st.mcnemar(kept).p_value < ss.ALPHA)
        for side, alternative in ss.SIDES.items():
            es = st.cohens_g(kept, alternative=alternative)
            sums[f"g_{side}"] += weight * ss.covers(es.conf_low, es.conf_high, true_g)
    assert lacking < 1e-8
    assert exact["refused"] == pytest.approx(no_discordant, abs=1e-12)
    for key, value in sums.items():
        assert exact[key] == pytest.approx(value, abs=1e-8), key


def test_pair_refusal_probability_by_enumeration() -> None:
    n = 7
    cells = ((0.3, 0.05, 0.1), (0.05, 0.2, 0.05), (0.1, 0.05, 0.1))
    tables = ss.compositions(n, 9).reshape(-1, 3, 3)
    weights = ss.multinomial_pmf(tables.reshape(-1, 9), ss.cells_flat(cells))
    exchanged = tables + np.transpose(tables, (0, 2, 1))
    upper = exchanged[:, [0, 0, 1], [1, 2, 2]]
    empty = (upper == 0).any(axis=1)
    assert ss.refusal_some_pair_empty(n, cells) == pytest.approx(
        float(weights[empty].sum()), abs=1e-13
    )


def test_bar_rate_counts_inferplots_proportion_tests() -> None:
    outcomes = ss.compositions(ss.BAR_SIZE, ss.BAR_LEVELS)
    weights = ss.multinomial_pmf(outcomes, (1 / 3,) * 3)
    total = 0.0
    for row, weight in zip(outcomes, weights.tolist(), strict=True):
        # One bar beside a second bar that holds every level.
        table = np.column_stack([row, np.ones(3, dtype=np.int64)])
        bar = st.proportion_tests(table)[0]
        assert bar.test is not None
        total += weight * (bar.test.p_value < ss.ALPHA)
    assert total == pytest.approx(ss.BAR_RATE, abs=1e-12)


# --------------------------------------------------------------- the shortcuts


def _two_way_tables() -> list[ss.IntTable]:
    """Tables of the simulated two-way designs and sparse ones that leave rows or
    columns empty, a single level, or no more observations than levels."""
    rng = np.random.Generator(np.random.PCG64(20261690))
    out: list[ss.IntTable] = []
    designs = [
        (ss.TWO_N, (2, 2), ss.outer(ss.TWO_P, ss.TWO_Q), 4),
        (ss.SPARSE_N, (2, 2), ss.outer(ss.SPARSE_P, ss.SPARSE_Q), 6),
        (ss.NULL34_N, (3, 4), ss.outer(ss.NULL34_P, ss.NULL34_Q), 3),
        (ss.FISHER4_N, (4, 3), ss.outer(ss.FISHER4_P, ss.FISHER4_Q), 3),
        (ss.FISHER3_N, (3, 2), ss.outer(ss.FISHER3_P, ss.FISHER3_Q), 3),
        (8, (3, 4), ss.outer((0.1, 0.3, 0.6), (0.05, 0.15, 0.3, 0.5)), 8),
    ]
    for n, shape, probs, count in designs:
        for row in rng.multinomial(n, probs, size=count):
            out.append(_table(row, shape))
    out += [
        _table([[3, 0], [0, 0]], (2, 2)),
        _table([[1, 1], [0, 0]], (2, 2)),
        _table([[1, 0, 1], [0, 1, 0]], (2, 3)),
        _table([[2, 0, 0], [0, 0, 0], [1, 0, 0]], (3, 3)),
        _table([[5, 0, 0], [0, 4, 0], [0, 0, 6]], (3, 3)),
        _table([[4, 3, 0], [2, 5, 0], [0, 0, 0]], (3, 3)),
    ]
    return out


@pytest.mark.parametrize("index", range(33))
def test_two_way_shortcut_is_barstats(index: int) -> None:
    counts = _two_way_tables()[index]
    kept = ss.two_way_table(counts)
    frame = ss.table_frame(counts)
    if kept is None:
        with pytest.raises(ValueError, match=r"V needs more|needs observations in two"):
            ip.analyze.barstats(frame, "x", "y", counts="n", bf_message=False)
        return
    result = ip.analyze.barstats(frame, "x", "y", counts="n", bf_message=False)
    assert result.test is not None
    assert result.effect_size is not None
    test = st.chisq_independence(kept)
    assert_close("closed_form", result.test.p_value, test.p_value, what="p")
    v = st.cramers_v(kept)
    es = result.effect_size
    assert (es.estimate, es.conf_low, es.conf_high) == (
        v.estimate,
        v.conf_low,
        v.conf_high,
    )
    # Cochran's rule as check() applies it to the main test.
    expected = np.asarray(check.expected_counts(result), dtype=np.float64)
    assert ss.cochran_fails(np.asarray(test.expected).ravel()) == ss.cochran_fails(
        expected
    )
    issues = {issue.code: issue for issue in result.check()}
    main_flagged = "small_expected" in issues and "the chi-square test of" in (
        issues["small_expected"].message
    )
    assert main_flagged == ss.cochran_fails(expected)
    # The bars' p-values without Pearson's C, and the pairs' without V.
    bars = [b for b in result.bars if b.test is not None]
    assert len(bars) == kept.shape[1]
    for j, bar in enumerate(bars):
        assert bar.test is not None
        assert_close(
            "closed_form",
            bar.test.p_value,
            st.goodness_of_fit(kept[:, j]).p_value,
            what=f"bar {j}",
        )
    if kept.shape[0] < 3:
        assert result.pairwise == ()
        return
    raw = ss.fisher_pair_p_values(kept)
    pairs = st.pairwise_fisher(kept)
    assert [p.p_value for p in pairs] == raw
    tested = [p for p in raw if p is not None]
    adjusted = iter(st.p_adjust(tested, "holm") if tested else ())
    assert len(result.pairwise) == len(raw)
    for row, p in zip(result.pairwise, raw, strict=True):
        assert row.p_value == p
        assert row.p_adjusted == (None if p is None else next(adjusted))


def _paired_tables() -> list[ss.IntTable]:
    rng = np.random.Generator(np.random.PCG64(20261691))
    out: list[ss.IntTable] = []
    designs = [
        (30, (2, 2), ss.MC_HOMOGENEOUS, 3),
        (30, (2, 2), ss.MC_FEW, 6),
        (40, (2, 2), ss.G_CELLS, 3),
        (30, (3, 3), ss.BOWKER, 6),
        (6, (3, 3), ss.BOWKER, 4),
    ]
    for n, shape, cells, count in designs:
        for row in rng.multinomial(n, ss.cells_flat(cells), size=count):
            out.append(_table(row, shape))
    out += [
        _table([[5, 3], [0, 0]], (2, 2)),  # x lacks a1: not square
        _table([[5, 0], [3, 0]], (2, 2)),  # y lacks a1
        _table([[5, 0], [0, 0]], (2, 2)),  # one level
        _table([[5, 0], [0, 4]], (2, 2)),  # no discordant pair
        _table([[5, 2, 0], [1, 4, 0], [0, 0, 0]], (3, 3)),  # a2 in neither: 2 x 2
        _table([[5, 2, 0], [1, 4, 0], [0, 0, 3]], (3, 3)),  # a2 exchanges none
    ]
    return out


@pytest.mark.parametrize("index", range(28))
def test_paired_shortcut_is_barstats(index: int) -> None:
    counts = _paired_tables()[index]
    kept = ss.paired_table(counts)
    frame = ss.table_frame(counts, paired=True)
    if kept is None:
        with pytest.raises(ValueError, match=r"same categories|single level|0/0"):
            ip.analyze.barstats(frame, "x", "y", counts="n", paired=True)
        return
    result = ip.analyze.barstats(frame, "x", "y", counts="n", paired=True)
    assert result.test is not None
    assert result.effect_size is not None
    assert_close("closed_form", result.test.p_value, st.mcnemar(kept).p_value, what="p")
    g = st.cohens_g(kept)
    es = result.effect_size
    assert (es.estimate, es.conf_low, es.conf_high) == (
        g.estimate,
        g.conf_low,
        g.conf_high,
    )


def _one_way_cases() -> list[tuple[ss.IntTable, tuple[float, ...] | None]]:
    rng = np.random.Generator(np.random.PCG64(20261692))
    out: list[tuple[ss.IntTable, tuple[float, ...] | None]] = []
    for n, probs, ratio, count in (
        (ss.GOF_SMALL_N, (1 / 3,) * 3, None, 3),
        (ss.GOF_SIX_N, (1 / 6,) * 6, None, 3),
        (ss.GOF_RATIO_N, ss.GOF_RATIO, ss.GOF_RATIO, 3),
        (6, (1 / 6,) * 6, None, 3),
        (6, ss.GOF_RATIO, ss.GOF_RATIO, 3),
    ):
        for row in rng.multinomial(n, probs, size=count):
            out.append((np.asarray(row, dtype=np.int64), ratio))
    out += [
        (np.array([4, 0, 0], dtype=np.int64), None),
        (np.array([4, 0, 2], dtype=np.int64), None),
        (np.array([4, 0, 2], dtype=np.int64), ss.GOF_RATIO),
    ]
    return out


@pytest.mark.parametrize("index", range(18))
def test_one_way_shortcut_is_barstats(index: int) -> None:
    counts, ratio = _one_way_cases()[index]
    read = ss.one_way_counts(counts, ratio)
    import pandas as pd

    frame = pd.DataFrame(
        {"x": [f"a{i}" for i in range(counts.size)], "n": counts.tolist()}
    )
    if read is None:
        with pytest.raises(ValueError, match=r"single category|ratio has"):
            ip.analyze.barstats(frame, "x", counts="n", ratio=ratio)
        return
    kept, dropped = read
    result = ip.analyze.barstats(frame, "x", counts="n", ratio=ratio)
    assert result.test is not None
    assert result.effect_size is not None
    assert len(result.levels) == kept.size
    assert dropped == (kept.size < counts.size)
    assert_close(
        "closed_form",
        result.test.p_value,
        st.goodness_of_fit(kept, ratio).p_value,
        what="p",
    )
    c = st.pearsons_c(kept, ratio)
    es = result.effect_size
    assert (es.estimate, es.conf_low, es.conf_high) == (
        c.estimate,
        c.conf_low,
        c.conf_high,
    )


def test_case_counts() -> None:
    assert len(_two_way_tables()) == 33
    assert len(_paired_tables()) == 28
    assert len(_one_way_cases()) == 18


def test_population_values() -> None:
    # Independent cells give V = 0; a 2 x 2 table's V is |phi|.
    assert ss.cramers_v_of(np.outer((0.3, 0.7), (0.2, 0.5, 0.3)).tolist()) == (
        pytest.approx(0.0, abs=1e-15)
    )
    a, b, c, d = 0.3, 0.1, 0.2, 0.4
    phi = (a * d - b * c) / math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    assert ss.cramers_v_of(((a, b), (c, d))) == pytest.approx(abs(phi), rel=1e-14)
    assert ss.pearsons_c_of((0.25, 0.25, 0.5), (0.25, 0.25, 0.5)) == 0.0
    assert ss.cohens_g_of(ss.G_CELLS) == pytest.approx(0.2, abs=1e-15)
    assert ss.cohens_g_of(ss.BOWKER) == pytest.approx(0.0, abs=1e-15)
