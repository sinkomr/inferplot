"""``ip.analyze.barstats`` against every ggbarstats and ggpiestats fixture.

Each completed fixture is replayed through ``ip.analyze.barstats`` on its CSV read
as pandas, polars and pyarrow with the fixture's own arguments
(``tests/bar_cases.py``, which translates them with ``tests/oracle_args.py``), and
every number the fixture holds is compared under its ``tests/tolerances.py`` kind:
the table, its levels, n and the rows dropped; each cell's count and share of its
bar; the subtitle row (statistic, df, p, method, n, the effect size with its
interval, level and sidedness); each bar's proportion test (statistic, df, p,
Pearson's C with its interval); each pair's Fisher p, adjusted p and Cramér's V
with its interval.

Interval bounds where effectsize's root search stops short (DECISIONS.md D99) are
those of ``CHISQ_ROOT_MISSES``: R's value is checked against the recorded one,
inferplot's against the recorded accurate root, and the two are shown to differ
by more than ``noncentral_ci`` (the statistics' own tests check the accurate roots
against 40-digit references). One of them prints differently at 3 digits
(mtcars-cyl-digits3: 0.499 in R, 0.498 accurate); printing is the texts' to test.

Fixtures inferplot answers differently from ggstatsplot, by name:

* the 22 refused by design (``bar_cases.REFUSALS``): type robust (the replay with
  type parametric equals ggstatsplot's numbers, which are the parametric ones),
  type bayes (v0.2), the degenerate tables, paired variables with different levels,
  paired data where no unit changes level, and a ``ratio`` of the wrong length or
  sum (the rescaled ratio gives mtcars-cyl-ratio's numbers);
* edge_bar_degenerate-single-x: R loses the whole pairwise table (fisher.test
  stops on the pair a-b, whose table has one column); inferplot keeps the pair
  without a test and computes the other two, checked against SciPy;
* edge_bar_paired_agree and edge_bar_paired_unequal: R draws the bars without a
  subtitle; inferplot raises, and draws the same bars with
  ``results_subtitle=False`` (paired_agree) or once both columns are declared on
  the same levels (the -declared fixture, answered).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import inferplot as ip
from inferplot import stats
from inferplot.check import expected_counts
from inferplot.result import StatsResult
from tests import bar_cases as bc
from tests.oracle_fixtures import (
    CHISQ_ROOT_MISSES,
    Backend,
    OracleCase,
    number,
    read_dataset,
)
from tests.tolerances import assert_close, is_close

analysis: Any = bc.bars

# The fixture whose numbers ggstatsplot gives for each replay: robust runs the
# parametric test; the rescaled ratio of ratio-sum4 is mtcars-cyl-ratio's.
REPLAYED: Final[dict[str, str]] = {
    name: (
        "ggbarstats-mtcars-cyl-ratio-parametric"
        if name == "ggbarstats-mtcars-cyl-ratio-sum4-parametric"
        else name.replace("-robust", "-parametric")
    )
    for name, refusal in bc.REFUSALS.items()
    if refusal.replay is not None
}
NAMES: Final = sorted([c.name for c in bc.ANSWERED] + list(REPLAYED))
PARAMS: Final = [
    pytest.param(name, backend, id=f"{name}-{backend}")
    for name in NAMES
    for backend in bc.BACKENDS
]
SINGLE_X: Final = "ggbarstats-edge_bar_degenerate-single-x-parametric"


@functools.cache
def result_of(name: str, backend: Backend = "pandas") -> StatsResult:
    """The analysis of a fixture (its replay where it is refused), cached."""
    refusal = bc.REFUSALS.get(name)
    changes = None if refusal is None else refusal.replay
    return bc.assembled(bc.BY_NAME[name], backend, changes)


def numbers_of(name: str) -> OracleCase:
    """The fixture whose numbers the analysis of ``name`` must give."""
    return bc.BY_NAME[REPLAYED.get(name, name)]


def _design(case: OracleCase) -> str:
    if case.contingency_data["y"] is None:
        return "one-way"
    return "paired" if case.args.get("paired") is True else "two-way"


def _side(case: OracleCase) -> str:
    return {"two.sided": "two-sided"}.get(
        str(case.args.get("alternative", "two.sided")),
        str(case.args.get("alternative", "two.sided")),
    )


def _bound_probability(level: float, sidedness: str, side: str) -> float | None:
    """The probability P(X <= chi2) a noncentral bound solves for, None for the
    fixed end of a one-sided interval."""
    alpha = 1 - level
    tail = alpha / 2 if sidedness == "two-sided" else alpha
    if side == "low":
        return None if sidedness == "less" else 1 - tail
    return None if sidedness == "greater" else tail


Pins = set[tuple[str, str, int, int, int, int, str]]


def check_bounds(
    effect: str,
    low: float | None,
    high: float | None,
    level: float,
    sidedness: str,
    row: Mapping[str, Any],
    key: tuple[float, int, int, int, int],
    what: str,
) -> Pins:
    """Both noncentral chi-square bounds against R's within ``noncentral_ci``, or,
    for a bound in CHISQ_ROOT_MISSES, each side against its recorded value; returns
    the keys of the pinned bounds. ``key`` is (chi2, df, n, rows, columns)."""
    statistic, df, n, rows, columns = key
    pinned: Pins = set()
    for side, mine in (("low", low), ("high", high)):
        r_value = number(row[f"conf.{side}"])
        assert mine is not None, what
        probability = _bound_probability(level, sidedness, side)
        pin = (
            None
            if probability is None
            else (
                effect,
                f"{statistic:.10g}",
                df,
                n,
                rows,
                columns,
                f"{probability:.6g}",
            )
        )
        miss = None if pin is None else CHISQ_ROOT_MISSES.get(pin)
        if pin is None or miss is None:
            assert_close("noncentral_ci", mine, r_value, what=f"{what} {side}")
            continue
        r_pinned, accurate = miss
        assert math.isclose(r_value, r_pinned, rel_tol=1e-9), (what, side, r_value)
        assert math.isclose(mine, accurate, rel_tol=1e-9), (what, side, mine)
        assert not is_close("noncentral_ci", mine, r_value), (what, side, "stale")
        pinned.add(pin)
    return pinned


# --------------------------------------------------------------------------- #
# the fixtures
# --------------------------------------------------------------------------- #


def test_fixture_counts() -> None:
    assert len(bc.CASES) == 89
    assert len(bc.ANSWERED) == 67
    assert len(bc.REFUSALS) == 22
    # 7 robust replays and the rescaled ratio.
    assert len(REPLAYED) == 8
    assert len(PARAMS) == 75 * 3
    assert not [c.name for c in bc.CASES if c.error is not None]


@pytest.mark.parametrize(("name", "backend"), PARAMS)
def test_table_cells_and_bars(name: str, backend: Backend) -> None:
    case, result = numbers_of(name), result_of(name, backend)
    data = case.contingency_data
    assert (result.analysis, result.design) == ("barstats", "contingency")
    assert result.type == case.type  # a replay's twin is the parametric case
    assert list(result.levels) == data["x_levels"]
    assert list(result.y_levels) == (data["y_levels"] or [])
    assert [list(row) for row in result.counts] == case.counts_table()
    assert (result.n_obs, result.n_dropped) == (data["n"], data["rows_dropped"])
    assert result.settings.seed == 0 == result.provenance.seed
    assert result.provenance.citations
    # Each cell's count and share of its bar, against the descriptive table.
    y = data["y"]
    cells = {(cell.bar, cell.level): cell for cell in result.cells}
    rows = case.descriptive
    assert len(rows) == len(cells)
    for row in rows:
        key = (None if y is None else str(row[y]), str(row[data["x"]]))
        cell = cells[key]
        assert cell.count == int(row["counts"]), (name, key)
        assert_close("closed_form", cell.percent, number(row["perc"]), what=str(key))
    shares = stats.column_percentages(result.counts)
    for cell in result.cells:
        i = result.levels.index(cell.level)
        j = 0 if cell.bar is None else result.y_levels.index(cell.bar)
        assert cell.percent == shares[i][j]
    # Each bar's n, against the per-bar table (with y) or the table total.
    if y is None:
        assert [(bar.level, bar.n) for bar in result.bars] == [(None, data["n"])]
        assert result.bars[0].test is None
    else:
        drawn = {str(row[y]): int(row["counts"]) for row in case.one_sample}
        assert {bar.level: bar.n for bar in result.bars} == drawn


@pytest.mark.parametrize(("name", "backend"), PARAMS)
def test_main_test(name: str, backend: Backend) -> None:
    check_main(name, backend)


def check_main(name: str, backend: Backend) -> Pins:
    case, result = numbers_of(name), result_of(name, backend)
    test, es = result.test, result.effect_size
    if case.args.get("results.subtitle") is False:
        assert (test, es, result.pairwise) == (None, None, ())
        assert not case.subtitle_rows
        return set()
    assert test is not None
    assert es is not None
    row = case.subtitle_data
    design = _design(case)
    name_, label, method = bc.TESTS[design]
    assert (test.name, test.label, test.method) == (name_, label, method)
    assert test.method == row["method"]
    assert (test.statistic_name, test.df, test.alternative) == (
        "chi2",
        None,
        "two-sided",
    )
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="chi2")
    assert test.df_error == number(row["df"])
    assert_close("closed_form", test.p_value, number(row["p.value"]), what="p")
    assert int(number(row["n.obs"])) == result.n_obs
    es_name, es_label, ci_method = bc.EFFECTS[design]
    assert (es.name, es.label, es.ci_method) == (es_name, es_label, ci_method)
    assert row["effectsize"] == bc.R_EFFECTS[design]
    assert es.conf_level == number(row["conf.level"]) == result.settings.conf_level
    assert es.sidedness == _side(case) == result.settings.alternative
    assert_close("closed_form", es.estimate, number(row["estimate"]), what="estimate")
    if design == "paired":
        assert_close("closed_form", es.conf_low, number(row["conf.low"]), what="low")
        assert_close("closed_form", es.conf_high, number(row["conf.high"]), what="high")
        return set()
    counts = np.array(result.counts)
    kept = counts[counts.sum(axis=1) > 0][:, counts.sum(axis=0) > 0]
    return check_bounds(
        es_name,
        es.conf_low,
        es.conf_high,
        es.conf_level,
        es.sidedness,
        row,
        (test.statistic, int(test.df_error or 0), result.n_obs, *kept.shape),
        f"{name} {es_name}",
    )


@pytest.mark.parametrize(("name", "backend"), PARAMS)
def test_bar_tests(name: str, backend: Backend) -> None:
    check_bars(name, backend)


def check_bars(name: str, backend: Backend) -> Pins:
    case, result = numbers_of(name), result_of(name, backend)
    y = case.contingency_data["y"]
    if y is None:
        assert not case.one_sample
        return set()
    pinned: Pins = set()
    rows = {str(row[y]): row for row in case.one_sample}
    # ggstatsplot lists the bars in the reverse order of y's levels.
    assert [str(row[y]) for row in case.one_sample] == [
        bar.level for bar in result.bars
    ][::-1]
    for bar in result.bars:
        assert bar.level is not None
        row = rows[bar.level]
        what = f"{name} bar {bar.level}"
        test, es = bar.test, bar.effect_size
        assert test is not None, what
        assert es is not None, what
        assert bar.no_test_reason is None
        assert (test.name, test.label, test.method) == bc.TESTS["one-way"]
        assert test.method == row["method"]
        assert (test.statistic_name, test.alternative) == ("chi2", "two-sided")
        assert_close("closed_form", test.statistic, number(row["statistic"]), what=what)
        assert test.df_error == number(row["df"])
        assert_close("closed_form", test.p_value, number(row["p.value"]), what=what)
        assert (es.name, es.label, es.ci_method) == bc.EFFECTS["one-way"]
        assert row["effectsize"] == "Pearson's C"
        assert (es.conf_level, es.sidedness) == (0.95, "two-sided")
        assert number(row["conf.level"]) == 0.95
        assert_close("closed_form", es.estimate, number(row["estimate"]), what=what)
        pinned |= check_bounds(
            "pearsons_c",
            es.conf_low,
            es.conf_high,
            0.95,
            "two-sided",
            row,
            (test.statistic, int(test.df_error or 0), bar.n, len(result.levels), 1),
            what,
        )
    return pinned


P_ADJUST_LABELS: Final = {
    "holm": "Holm",
    "hochberg": "Hochberg",
    "hommel": "Hommel",
    "bonferroni": "Bonferroni",
    "BH": "FDR",
    "BY": "BY",
    "fdr": "FDR",
    "none": "None",
}


@pytest.mark.parametrize(("name", "backend"), PARAMS)
def test_pairwise(name: str, backend: Backend) -> None:
    check_pairs(name, backend)


def check_pairs(name: str, backend: Backend) -> Pins:
    case, result = numbers_of(name), result_of(name, backend)
    if name == SINGLE_X:
        return set()  # R loses the table: test_single_x_keeps_the_pairs_r_loses
    pinned: Pins = set()
    rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    assert [(p.group1, p.group2) for p in result.pairwise] == sorted(
        rows, key=lambda k: (result.levels.index(k[0]), result.levels.index(k[1]))
    )
    method = result.settings.p_adjust_method
    counts = np.array(result.counts)
    for pair in result.pairwise:
        row = rows[(pair.group1, pair.group2)]
        what = f"{name} {pair.group1}-{pair.group2}"
        assert (pair.test, pair.test_label) == ("fisher_exact", "Fisher's exact test")
        assert row["test"] == "Fisher's exact test"
        assert (pair.statistic_name, pair.statistic, pair.df) == ("", None, None)
        assert pair.adjust_method == method
        assert row["p.adjust.method"] == P_ADJUST_LABELS[method]
        assert pair.no_test_reason is None
        assert pair.displayed is False
        assert_close("closed_form", pair.p_value, number(row["p.value"]), what=what)
        assert_close(
            "closed_form", pair.p_adjusted, number(row["p.value.adj"]), what=what
        )
        assert pair.p_adjusted is not None
        assert pair.significant == (pair.p_adjusted < 0.05)
        assert row["effectsize"] == "Cramer's V (adj.)"
        assert (
            pair.conf_level == number(row["conf.level"]) == result.settings.conf_level
        )
        assert_close("closed_form", pair.estimate, number(row["estimate"]), what=what)
        sub = counts[
            [result.levels.index(pair.group1), result.levels.index(pair.group2)]
        ]
        sub = sub[:, sub.sum(axis=0) > 0]
        chi2 = stats.chisq_independence(sub)
        pinned |= check_bounds(
            "cramers_v",
            pair.conf_low,
            pair.conf_high,
            result.settings.conf_level,
            result.settings.alternative,
            row,
            (chi2.statistic, chi2.df, chi2.n, *sub.shape),
            what,
        )
    return pinned


def test_every_pinned_fixture_bound_is_met_through_the_api() -> None:
    # D99: the 9 distinct fixture bounds where effectsize's root search stops
    # short (the subtitle of mtcars-cyl-both and -digits3; the bars of
    # paired-cl90, paired_small and mtcars-am-cyl; the pairs of edge_bar_large,
    # mtcars-cyl-am-cl90 and edge_bar_rounding-both) are all reached by the
    # analysis, which reports the accurate root for each.
    met: Pins = set()
    for name in NAMES:
        met |= check_main(name, "pandas")
        met |= check_bars(name, "pandas")
        met |= check_pairs(name, "pandas")
    assert len(met) == 9
    assert met <= set(CHISQ_ROOT_MISSES)


@pytest.mark.parametrize("name", NAMES)
def test_results_do_not_depend_on_the_backend(name: str) -> None:
    first = _unlibrary(result_of(name, "pandas"))
    for backend in ("polars", "pyarrow"):
        assert _unlibrary(result_of(name, backend)).to_json() == first.to_json()


def _unlibrary(result: StatsResult) -> StatsResult:
    """The result with its recipes cut: they are written in each frame's own
    library (review B14)."""
    warnings = tuple(
        re.sub(
            r"(leave (?:it|them) out): .*",
            r"\1: ...",
            re.sub(r"(in their names), .*", r"\1, ...", w),
        )
        for w in result.warnings
    )
    return dataclasses.replace(result, warnings=warnings)


# --------------------------------------------------------------------------- #
# the fixtures' numbers (bar_cases.parts) and the analysis agree field by field
# --------------------------------------------------------------------------- #


def _r_pinned(value: float) -> float | None:
    """The accurate root behind an R bound that stops short (D99), else None."""
    for r_value, accurate in CHISQ_ROOT_MISSES.values():
        if math.isclose(value, r_value, rel_tol=1e-9):
            return accurate
    return None


def _same(kind: str, mine: object, theirs: object, what: str) -> None:
    """Strings, ints, bools and None exactly; floats under ``kind``, or a bound
    R stopped short on against the accurate root."""
    if isinstance(theirs, float) and isinstance(mine, float):
        accurate = _r_pinned(theirs) if kind == "noncentral_ci" else None
        if accurate is not None and not is_close(kind, mine, theirs):
            assert math.isclose(mine, accurate, rel_tol=1e-9), what
            return
        assert_close(kind, mine, theirs, what=what)
        return
    assert mine == theirs, (what, mine, theirs)


_BOUNDS: Final = frozenset({"conf_low", "conf_high"})


def _same_fields(mine: Any, theirs: Any, what: str, *, wilson: bool) -> None:
    assert type(mine) is type(theirs), what
    for field in dataclasses.fields(theirs):
        name = field.name
        kind = "noncentral_ci" if name in _BOUNDS and not wilson else "closed_form"
        _same(kind, getattr(mine, name), getattr(theirs, name), f"{what}.{name}")


@pytest.mark.parametrize("name", NAMES)
def test_fixture_numbers_and_the_analysis_agree(name: str) -> None:
    refusal = bc.REFUSALS.get(name)
    changes = None if refusal is None else refusal.replay
    table, args, _ = bc.prepared(bc.BY_NAME[name], "pandas", changes)
    numbers = numbers_of(name)
    theirs = bc.parts(numbers, table, args)
    mine = analysis._ANALYSES[args.design](table, args)
    wilson = args.design == "paired"
    for field in ("test", "effect_size"):
        a, b = getattr(mine, field), getattr(theirs, field)
        if b is None:
            assert a is None
            continue
        _same_fields(a, b, f"{name}.{field}", wilson=wilson and field == "effect_size")
    assert len(mine.bar_tests) == len(theirs.bar_tests)
    for i, (a, b) in enumerate(zip(mine.bar_tests, theirs.bar_tests, strict=True)):
        assert a.no_test_reason == b.no_test_reason
        _same_fields(a.test, b.test, f"{name} bar {i}", wilson=False)
        _same_fields(a.effect_size, b.effect_size, f"{name} bar {i} C", wilson=False)
    if name == SINGLE_X:
        assert not theirs.pairwise  # R has none
        assert len(mine.pairwise) == 3
        return
    assert len(mine.pairwise) == len(theirs.pairwise)
    for a, b in zip(mine.pairwise, theirs.pairwise, strict=True):
        _same_fields(a, b, f"{name} {b.group1}-{b.group2}", wilson=False)


# --------------------------------------------------------------------------- #
# what inferplot does where ggstatsplot loses or refuses something
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", bc.BACKENDS)
def test_single_x_keeps_the_pairs_r_loses(backend: Backend) -> None:
    case = bc.BY_NAME[SINGLE_X]
    # R's side: fisher.test stops on the pair a-b, and the whole table is lost.
    assert case.pairwise == []
    assert case.expected_hidden_errors == ("pairwise_comparisons",)
    errors = {s["component"]: s["error"] for s in case.suppressed}
    assert "'x' must have at least 2 rows and columns" in errors["pairwise_comparisons"]
    assert case.counts_table() == [[3, 0], [4, 0], [0, 5]]
    # inferplot's: a-b without a test, a-c and b-c tested and adjusted together.
    result = result_of(SINGLE_X, backend)
    ab, ac, bc_ = result.pairwise
    assert (ab.group1, ab.group2, ab.p_value, ab.p_adjusted, ab.significant) == (
        "a",
        "b",
        None,
        None,
        None,
    )
    assert ab.no_test_reason == (
        "both levels occur only in 'p' of 'y2', so their table has a single "
        "column and no test; R's fisher.test stops there"
    )
    assert (ab.estimate, ab.conf_low, ab.conf_high) == (None, None, None)
    p_ac = float(scipy_stats.fisher_exact([[3, 0], [0, 5]])[1])
    p_bc = float(scipy_stats.fisher_exact([[4, 0], [0, 5]])[1])
    assert_close("closed_form", ac.p_value, p_ac, what="a-c")
    assert_close("closed_form", bc_.p_value, p_bc, what="b-c")
    holm = stats.p_adjust([p_ac, p_bc], "holm")
    assert_close("closed_form", ac.p_adjusted, holm[0], what="a-c holm")
    assert_close("closed_form", bc_.p_adjusted, holm[1], what="b-c holm")
    assert result.warnings[:2] == (
        "The pair 'a' and 'b' of 'x3' has no Fisher test: both levels occur only in "
        "'p' of 'y2', so their table has a single column, and their shares cannot "
        "differ across the levels of 'y2'. Nothing needs fixing unless you need "
        "that comparison.",
        "1 of the 3 pairwise Fisher tests has no p-value (above); it stays in "
        "result.pairwise without one, and p_adjust_method='holm' adjusts the other 2 "
        "among themselves. ggstatsplot loses the whole pairwise table here, "
        "silently.",
    )
    (issue,) = [i for i in result.check() if i.code == "pairs_without_test"]
    assert issue.severity == "info"
    assert issue.message == (
        "1 of the 3 pairwise comparisons of 'x3' has no p-value: 'a' and 'b' (both "
        "levels occur only in 'p' of 'y2', so their table has a single column and "
        "no test; R's fisher.test stops there). The other 2 are adjusted among "
        "themselves."
    )
    # The pair stays in the table and the frame, without numbers.
    assert "a       b       NA    NA" in str(result)
    frame = result.to_frame()
    untested = frame[(frame["group1"] == "a") & (frame["group2"] == "b")]
    assert len(untested) == 1
    assert untested["p_value"].isna().all()
    assert StatsResult.from_json(result.to_json()) == result


def test_paired_agree_is_refused_and_its_bars_drawn() -> None:
    case = bc.BY_NAME["ggbarstats-edge_bar_paired_agree-parametric"]
    # R's side: Cohen's g stops ("not enough data") and the subtitle is dropped;
    # the bars' tests are computed.
    assert case.expected_hidden_errors == ("subtitle",)
    errors = {s["component"]: s["error"] for s in case.suppressed}
    assert errors["subtitle"] == "not enough data"
    assert case.counts_table() == [[11, 0], [0, 9]]
    frame = read_dataset(case)
    with pytest.raises(ValueError, match="none does") as info:
        ip.analyze.barstats(frame, "before", "after", paired=True)
    assert str(info.value) == (
        "McNemar's test compares the pairs whose level changes from 'before' to "
        "'after', and none does: each of the 20 pairs has the same level in both "
        "('no': 11, 'yes': 9), so the test is not defined (R's statistic is 0/0, "
        "and ggstatsplot leaves out its subtitle silently). Report the agreement "
        "as it is, or pass results_subtitle=False to draw the bars without a test."
    )
    # The remedy works: the bars, with R's proportion tests.
    result = ip.analyze.barstats(
        frame, "before", "after", paired=True, results_subtitle=False
    )
    assert (result.test, result.effect_size) == (None, None)
    rows = {str(row["after"]): row for row in case.one_sample}
    for bar in result.bars:
        assert bar.level is not None
        assert bar.test is not None
        row = rows[bar.level]
        assert_close(
            "closed_form", bar.test.statistic, number(row["statistic"]), what=""
        )
        assert_close("closed_form", bar.test.p_value, number(row["p.value"]), what="")


def test_paired_unequal_is_refused_and_the_declared_levels_answer() -> None:
    case = bc.BY_NAME["ggbarstats-edge_bar_paired_unequal-parametric"]
    assert case.expected_hidden_errors == ("subtitle",)
    with pytest.raises(ValueError, match=r"\('maybe' only in 'after'\)"):
        ip.analyze.barstats(read_dataset(case), "before", "after", paired=True)
    declared = result_of("ggbarstats-edge_bar_paired_unequal-declared-parametric")
    assert declared.test is not None
    assert declared.test.df_error == 3.0  # Bowker's test on the 3 x 3 table


@pytest.mark.parametrize("name", sorted(bc.REFUSALS))
@pytest.mark.parametrize("backend", bc.BACKENDS)
def test_refusals(name: str, backend: Backend) -> None:
    case = bc.BY_NAME[name]
    refusal = bc.REFUSALS[name]
    args, keywords = bc.call_arguments(case)
    with pytest.raises(refusal.error) as info:
        ip.analyze.barstats(read_dataset(case, backend), *args, **keywords)
    assert refusal.match in str(info.value)
    if case.type == "bayes":
        return  # v0.2
    if refusal.replay is None:
        # ggstatsplot drops the subtitle silently on every one of them.
        assert "subtitle" in case.expected_hidden_errors, name
        assert not case.subtitle_rows
        return
    twin = bc.BY_NAME[REPLAYED[name]]
    if case.type == "robust":
        # ggstatsplot's robust numbers are its parametric ones, table for table.
        for table in (
            "subtitle_data",
            "pairwise_comparisons_data",
            "descriptive_data",
            "one_sample_data",
        ):
            assert case.table(table) == twin.table(table), (name, table)
    replay = result_of(name, backend)
    assert replay.to_json() == result_of(twin.name, backend).to_json()


# --------------------------------------------------------------------------- #
# small_expected against R's own warnings
# --------------------------------------------------------------------------- #

# Fixtures where the main test's small_expected differs from R's warning, and why.
SMALL_EXPECTED_DIFFERS: Final = {
    # R's chisq.test warns when any expected count is below 5; Cochran's rule
    # allows 20% of them below 5 (one of six here, 4.0) and none below 1.
    "ggbarstats-edge_bar_independent-parametric": "R only",
    # R's mcnemar.test checks nothing; half the changing pairs of each two levels
    # (1.5, 4.5, 3.0 and 0.5 to 4.0) fail Cochran's rule.
    "ggbarstats-edge_bar_paired3-parametric": "inferplot only",
    "ggbarstats-edge_bar_paired_unequal-declared-parametric": "inferplot only",
}
_R_WARNING: Final = "Chi-squared approximation may be incorrect"
_BAR_PART: Final = re.compile(
    r"[Tt]he proportion tests? of the bars? ((?:'[^']*'(?:, )?)+)"
)


def _r_warns(case: OracleCase, component: str, level: str | None = None) -> bool:
    return any(
        _R_WARNING in w
        for s in case.suppressed
        if s["component"] == component and (level is None or s.get("level") == level)
        for w in s["warnings"]
    )


@pytest.mark.parametrize("name", NAMES)
def test_small_expected_follows_r_warnings(name: str) -> None:
    case, result = numbers_of(name), result_of(name)
    issues = [i for i in result.check() if i.code == "small_expected"]
    message = issues[0].message if issues else ""
    main = " expected counts of " in message
    r_main = _r_warns(case, "subtitle")
    differs = SMALL_EXPECTED_DIFFERS.get(name)
    if differs is None:
        assert main == r_main, (name, message)
    else:
        assert (main, r_main) == (differs == "inferplot only", differs == "R only")
    if _design(case) == "paired" and main:
        # McNemar's p: R gives no warning of its own (edge_bar_paired_small's
        # comes from prop.test inside Cohen's g interval: one changing pair).
        assert "R's mcnemar.test does not check this" in message
    # The bars: flagged exactly where R warns, drawn or not: the APA sentence
    # reports the bars' tests whenever y is given (D108; review B17).
    match = _BAR_PART.search(message)
    flagged = set() if match is None else set(re.findall(r"'([^']*)'", match.group(1)))
    r_bars = {
        bar.level
        for bar in result.bars
        if bar.level is not None and _r_warns(case, "proportion_test", bar.level)
    }
    assert flagged == r_bars, (name, flagged, r_bars)


def test_where_small_expected_and_r_differ() -> None:
    # edge_bar_independent: one expected count of six below 5 (R warns; within
    # Cochran's 20%). paired3 and paired_unequal-declared: half the changing
    # pairs of each two levels are all below 5 (R's mcnemar.test checks nothing).
    independent = expected_counts(
        result_of("ggbarstats-edge_bar_independent-parametric")
    )
    assert sorted(independent)[:2] == [4.0, 6.0]
    for name in (
        "ggbarstats-edge_bar_paired3-parametric",
        "ggbarstats-edge_bar_paired_unequal-declared-parametric",
    ):
        assert max(expected_counts(result_of(name))) < 5
    assert len(SMALL_EXPECTED_DIFFERS) == 3


# --------------------------------------------------------------------------- #
# round trips, frames, printing and check() on every fixture
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
def test_round_trips_frames_and_printing(name: str) -> None:
    result = result_of(name)
    assert StatsResult.from_json(result.to_json()) == result
    assert StatsResult.from_json(result.to_json(include_values=False)) == result
    assert StatsResult.from_dict(result.to_dict()) == result
    rows = result.frame_rows()
    expected = (
        (result.test is not None)
        + sum(bar.test is not None for bar in result.bars)
        + len(result.pairwise)
    )
    assert len(rows) == expected
    assert len(result.to_frame()) == expected
    assert result.to_frame("polars").height == expected
    text = str(result)
    assert text.startswith(result.text.apa)
    if result.pairwise:
        assert "Pairwise comparisons (Fisher's exact test):" in text
    for issue in result.check():
        assert "group" not in issue.message, (name, issue)
        assert "subject" not in issue.message, (name, issue)


def test_checks_on_the_fixtures() -> None:
    codes = {name: sorted({i.code for i in result_of(name).check()}) for name in NAMES}
    allowed = {
        "small_expected",
        "rows_dropped",
        "pairs_without_test",
        "empty_level_tested",
    }
    assert set().union(*codes.values()) <= allowed
    assert [n for n, c in codes.items() if "pairs_without_test" in c] == [SINGLE_X]
    # A declared level of x without observations, counted in the bars' drawn
    # tests (review B18).
    assert sorted(n for n, c in codes.items() if "empty_level_tested" in c) == [
        "ggbarstats-edge_bar_paired_unequal-declared-parametric",
        "ggbarstats-edge_bar_zero_cell-declared-parametric",
    ]
    assert sorted(n for n, c in codes.items() if "rows_dropped" in c) == [
        "ggbarstats-edge_bar_missing-counts-parametric",
        "ggbarstats-edge_bar_missing-oneway-parametric",
        "ggbarstats-edge_bar_missing-parametric",
    ]


# --------------------------------------------------------------------------- #
# plain data, independent of R
# --------------------------------------------------------------------------- #


def _frame(table: Sequence[Sequence[int]]) -> pd.DataFrame:
    """A table of counts as a data frame with a counts column."""
    rows = [
        {"x": f"x{i}", "y": f"y{j}", "n": int(count)}
        for i, row in enumerate(table)
        for j, count in enumerate(row)
    ]
    frame = pd.DataFrame(rows)
    return frame


def test_random_tables_match_scipy() -> None:
    rng = np.random.default_rng(20261110)
    checked = 0
    for _ in range(40):
        r, c = (int(v) for v in rng.integers(2, 5, size=2))
        table = rng.integers(1, 30, size=(r, c))
        result = ip.analyze.barstats(_frame(table.tolist()), "x", "y", counts="n")
        test = result.test
        assert test is not None
        chi2, p, df, _ = scipy_stats.chi2_contingency(table, correction=False)
        assert_close("closed_form", test.statistic, float(chi2), what="chi2")
        assert_close("closed_form", test.p_value, float(p), what="p")
        assert test.df_error == float(df)
        # Each bar: the goodness of fit of its column against equal shares.
        for bar, column in zip(result.bars, table.T, strict=True):
            assert bar.test is not None
            gof = scipy_stats.chisquare(column)
            assert_close("closed_form", bar.test.statistic, float(gof[0]), what="gof")
            assert_close("closed_form", bar.test.p_value, float(gof[1]), what="gof p")
        # The pairs of a 2-column table: SciPy's 2 x 2 Fisher test.
        if c == 2 and r >= 3:
            for pair in result.pairwise:
                i, j = int(pair.group1[1:]), int(pair.group2[1:])
                expected = scipy_stats.fisher_exact(table[[i, j]])[1]
                assert_close(
                    "closed_form", pair.p_value, float(expected), what="fisher"
                )
                checked += 1
        # One way: x's counts against equal shares.
        totals = table.sum(axis=1)
        one_way = ip.analyze.barstats(
            _frame([[int(v)] for v in totals]).drop(columns="y"), "x", counts="n"
        )
        assert one_way.test is not None
        gof = scipy_stats.chisquare(totals)
        assert_close("closed_form", one_way.test.statistic, float(gof[0]), what="gof")
        assert_close("closed_form", one_way.test.p_value, float(gof[1]), what="gof p")
    assert checked > 10


# --------------------------------------------------------------------------- #
# the cases no fixture holds
# --------------------------------------------------------------------------- #


def test_a_fisher_pair_beyond_the_cost_limit_keeps_the_others() -> None:
    # x0 and x1 are 840 observations over 20 levels of y, far apart: the exact
    # test of their 2 x 20 table is beyond the work limit (DECISIONS.md D107) and
    # is not computed; x0-x2 and x1-x2 (2 x 20 tables whose second row holds 5
    # observations) are cheap and tested.
    table = [
        [2 * j + 2 for j in range(20)],
        [40 - 2 * j for j in range(20)],
        [3, 2] + [0] * 18,
    ]
    result = ip.analyze.barstats(_frame(table), "x", "y", counts="n")
    ab, ac, bc_ = result.pairwise
    assert (ab.p_value, ab.p_adjusted, ab.significant) == (None, None, None)
    assert ab.no_test_reason == (
        "too large to compute exactly within inferplot's limits (a 2 x 20 table of "
        "840), not approximated"
    )
    assert ab.estimate is not None  # Cramér's V is still reported
    assert ac.p_value is not None
    assert bc_.p_value is not None
    for pair, rows in ((ac, [0, 2]), (bc_, [1, 2])):
        sub = np.array(table)[rows]
        expected = stats.fisher_exact(sub[:, sub.sum(axis=0) > 0]).p_value
        assert pair.p_value == expected
    # The pair too large to compute still counts in the adjustment, with p = 1.
    holm = stats.p_adjust([ac.p_value, bc_.p_value, 1.0], "holm")
    assert (ac.p_adjusted, bc_.p_adjusted) == tuple(holm[:2])
    assert holm[0] >= stats.p_adjust([ac.p_value, bc_.p_value], "holm")[0]
    first = next(
        i for i, w in enumerate(result.warnings) if w.startswith("Fisher's exact")
    )
    assert result.warnings[first] == (
        "Fisher's exact test was computed for 2 of the 3 pairs of levels of 'x'; "
        "the other 1 ('x0' and 'x1') is too large to compute exactly within "
        "inferplot's limits (about a second of work for one test and two seconds "
        "for all the pairs of one table, the cheapest computed first), and "
        "inferplot does not approximate them. Their Cramér's V is reported. To "
        "test them, merge sparse levels of 'y' (levels with few observations), or "
        "compare two levels of 'x' in a call of their own."
    )
    assert result.warnings[first + 1] == (
        "1 of the 3 pairwise Fisher tests has no p-value (above); it stays in "
        "result.pairwise without one, and p_adjust_method='holm' adjusts the other "
        "2 as 3 comparisons: a pair too large to compute counts as a test with "
        "p = 1, so leaving it out makes no computed pair look more significant."
    )
    (issue,) = [i for i in result.check() if i.code == "pairs_without_test"]
    assert "The other 2 are adjusted as 3 comparisons" in issue.message
    assert issue.message.startswith(
        "1 of the 3 pairwise comparisons of 'x' has no p-value: 'x0' and 'x1' (too "
        "large to compute exactly"
    )


def test_bowker_with_two_levels_that_never_exchange_pairs() -> None:
    # probe-contingency's [[5, 0, 0], [0, 5, 2], [0, 1, 5]]: R returns NaN.
    table = [[5, 0, 0], [0, 5, 2], [0, 1, 5]]
    rows = [
        {"before": f"l{i}", "after": f"l{j}", "n": count}
        for i, row in enumerate(table)
        for j, count in enumerate(row)
    ]
    frame = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="Bowker") as info:
        ip.analyze.barstats(frame, "before", "after", paired=True, counts="n")
    assert str(info.value) == (
        "Bowker's test of symmetry (McNemar's test beyond two levels) needs pairs "
        "that change between every two levels, and no pair changes between 'l0' "
        "and 'l1' or between 'l0' and 'l2', in either direction: those terms are "
        "0/0, so the test is not defined (R's statistic is NaN, and ggstatsplot "
        "leaves out its subtitle silently). Merge levels so that every two "
        "exchange some pairs, or compare two levels that do exchange pairs on "
        "their own (keep the units whose both measurements are among those two: a "
        "2 x 2 McNemar test)."
    )
    # The remedy: the two levels that exchange pairs, alone.
    kept = frame[frame["before"].isin(["l1", "l2"]) & frame["after"].isin(["l1", "l2"])]
    result = ip.analyze.barstats(kept, "before", "after", paired=True, counts="n")
    assert result.test is not None
    assert_close("closed_form", result.test.statistic, 1 / 3, what="(2 - 1)^2 / 3")


def test_a_ratio_with_a_zero_share_is_refused() -> None:
    frame = pd.DataFrame({"x": list("aabbc")})
    with pytest.raises(ValueError, match="positive numbers") as info:
        ip.analyze.barstats(frame, "x", ratio=[0.5, 0.5, 0.0])
    assert str(info.value).endswith(
        "got 0.0. A share of 0 expects no observations in its level: the "
        "goodness-of-fit statistic divides by it (R returns Inf or NaN). If the "
        "level cannot occur, leave it out of the data (do not declare it) and give "
        "the shares of the others."
    )


def test_too_few_observations_for_cramers_v() -> None:
    frame = pd.DataFrame({"x": ["a", "b"], "y": ["p", "q"]})
    with pytest.raises(ValueError, match="holds 2 \\(R gives NaN for it"):
        ip.analyze.barstats(frame, "x", "y")
    # The bars alone are computed.
    assert ip.analyze.barstats(frame, "x", "y", results_subtitle=False).test is None


# --------------------------------------------------------------------------- #
# the arguments without effect, on real results
# --------------------------------------------------------------------------- #


def _warnings(name: str, **changes: Any) -> tuple[str, ...]:
    return bc.assembled(bc.BY_NAME[name], changes=changes).warnings


def test_seed_has_no_effect() -> None:
    warnings = _warnings("ggbarstats-mtcars-cyl-am-parametric", seed=7)
    assert (
        "seed=7 has no effect: no test of counts draws random numbers (the "
        "chi-square, McNemar and exact Fisher tests use the counts as they are); it "
        "is recorded in the result."
    ) in warnings
    plain = result_of("ggbarstats-mtcars-cyl-am-parametric")
    seeded = bc.assembled(
        bc.BY_NAME["ggbarstats-mtcars-cyl-am-parametric"], changes={"seed": 7}
    )
    assert (seeded.test, seeded.pairwise) == (plain.test, plain.pairwise)
    assert seeded.settings.seed == seeded.provenance.seed == 7


def test_p_adjust_method_with_a_single_pair() -> None:
    # edge_bar_zero_cell-declared: x declares three levels, two hold data.
    warnings = _warnings(
        "ggbarstats-edge_bar_zero_cell-declared-parametric", p_adjust_method="BH"
    )
    assert (
        "p_adjust_method='BH' has no effect: it adjusts the pairwise Fisher tests "
        "between the levels of x, and only one pair has a test ('b' and 'a'), and a "
        "single p-value is its own adjusted p-value under every method."
    ) in warnings
    assert not any(
        "p_adjust_method" in w
        for w in _warnings("ggbarstats-mtcars-cyl-am-parametric", p_adjust_method="BH")
    )


def test_alternative_names_what_it_changes() -> None:
    # edge_bar_rounding: every pair's table is 2 x 3 or 2 x 4, so no Fisher test
    # follows alternative.
    rounding = _warnings("ggbarstats-edge_bar_rounding-parametric", alternative="less")
    assert (
        "alternative='less' makes the Cramér's V interval one-sided, [0, high], also "
        "for each pair; the pairs' Fisher tests stay two-sided, as no pair's table "
        "is 2 x 2 (fisher.test ignores alternative beyond 2 x 2); the chi-square "
        "test has no direction, so its p-value is the same, as are the proportion "
        "tests of the bars (as in ggstatsplot)."
    ) in rounding
    one_way = _warnings("ggbarstats-mtcars-cyl-parametric", alternative="greater")
    assert (
        "alternative='greater' makes the Pearson's C interval one-sided, [low, 1]; "
        "the goodness-of-fit test has no direction, so its p-value is the same (as "
        "in ggstatsplot)."
    ) in one_way
    paired = _warnings("ggbarstats-edge_bar_paired-parametric", alternative="less")
    assert (
        "alternative='less' makes the Cohen's g interval one-sided, [-0.5, high]; "
        "McNemar's test has no direction, so its p-value is the same, as are the "
        "proportion tests of the bars (as in ggstatsplot)."
    ) in paired
    two_by_two = _warnings("ggbarstats-mtcars-am-vs-parametric", alternative="less")
    assert any(
        w.startswith(
            "alternative='less' makes the Cramér's V interval one-sided, [0, high]; "
            "the chi-square test"
        )
        for w in two_by_two
    )


def test_conf_level_without_pairs() -> None:
    assert (
        "The proportion test of each bar reports Pearson's C with a 95% interval "
        "whatever conf_level says (as in ggstatsplot): conf_level=0.9 applies to the "
        "main effect size only."
    ) in _warnings("ggbarstats-mtcars-am-vs-parametric", conf_level=0.9)


def test_ratio_with_y_reaches_the_bars_only() -> None:
    case = "ggbarstats-mtcars-cyl-am-ratio-parametric"
    with_ratio = result_of(case)
    without = bc.assembled(bc.BY_NAME[case], changes={"ratio": None})
    assert (with_ratio.test, with_ratio.pairwise) == (without.test, without.pairwise)
    assert with_ratio.bars != without.bars
    assert (
        "ratio= sets the expected shares of the proportion test of each bar; the "
        "chi-square test of independence does not use it (as in ggstatsplot)."
    ) in with_ratio.warnings


def test_results_subtitle_false_computes_only_the_bars() -> None:
    quiet = result_of("ggbarstats-mtcars-cyl-am-nosubtitle-parametric")
    full = result_of("ggbarstats-mtcars-cyl-am-parametric")
    assert (quiet.test, quiet.effect_size, quiet.pairwise) == (None, None, ())
    assert quiet.bars == full.bars
    assert quiet.cells == full.cells
    assert all(bar.effect_size is not None for bar in quiet.bars)
