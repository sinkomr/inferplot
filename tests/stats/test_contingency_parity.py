"""inferplot.stats' count-table methods against every barstats fixture and the probe.

Each function runs on the table the fixture analyzed (``contingency_data``: rows the
levels of x, columns the levels of y, declared empty levels included) with the
fixture's arguments, so a failure here is a method difference, not a data-reading
one. Every number the fixture holds for the statistics is compared: the subtitle row
(test statistic, df, p, n, the effect size and its interval and level, and each of
them as the subtitle prints it), the per-bar tests, the pairwise table, and each
cell's share of its bar; then probe-contingency's grids.

Interval bounds where effectsize's root search stops short (DECISIONS.md D99) are
those of ``CHISQ_ROOT_MISSES`` (tests/oracle_fixtures.py): there R's value is checked
against the recorded one, inferplot's against the recorded accurate root, and the
two are shown to differ by more than ``noncentral_ci``. The accurate roots are
checked against 40-digit mpmath inversions in tests/stats/test_noncentral_chi2.py.
"""

from __future__ import annotations

import math
import re
from typing import Any, Final

import numpy as np
import pytest
from numpy.typing import NDArray

from inferplot.stats import (
    EffectSizeCI,
    PAdjustMethod,
    chisq_independence,
    cohens_g,
    column_percentages,
    cramers_v,
    fisher_exact,
    goodness_of_fit,
    mcnemar,
    pairwise_fisher,
    pearsons_c,
    proportion_tests,
)
from tests.oracle_fixtures import (
    CHISQ_ROOT_MISSES,
    OracleCase,
    Row,
    cases_for,
    number,
    probe,
    r_double,
    require_cases,
)
from tests.stats.fixture_data import alternative, conf_level, p_adjust_method, value
from tests.tolerances import assert_close, is_close

IntArray = NDArray[np.int64]

CASES: Final = require_cases(
    [*cases_for("ggbarstats"), *cases_for("ggpiestats")], "barstats fixtures"
)
FAILED: Final = cases_for("ggbarstats", failed=True)

# ggstatsplot's own failure (no table at all): a weight that is not whole. The data
# layer refuses it before any statistic runs; the statistics refuse such counts too
# (test_cases_ggstatsplot_fails_on).
FAILED_IN_R: Final = frozenset({"ggbarstats-edge_bar_counts-frac-parametric"})

# Cases where ggstatsplot drops a result silently (expected_hidden_errors), with what
# inferplot's functions do on the same table (test_cases_ggstatsplot_drops).
DROPPED: Final = {
    "ggbarstats-edge_bar_degenerate-one-cell-parametric": "1 x 1",
    "ggbarstats-edge_bar_degenerate-one-column-parametric": "r x 1",
    "ggbarstats-edge_bar_degenerate-one-row-parametric": "1 x c",
    "ggbarstats-edge_bar_degenerate-oneway-one-parametric": "one category",
    "ggbarstats-edge_bar_degenerate-single-x-parametric": "pair in one column",
    "ggbarstats-edge_bar_paired_agree-parametric": "no discordant pair",
    "ggbarstats-edge_bar_paired_unequal-parametric": "not square",
    "ggbarstats-mtcars-cyl-ratio-short-parametric": "ratio length",
    "ggbarstats-mtcars-cyl-ratio-sum4-parametric": "ratio sum",
}


def _design(case: OracleCase) -> str:
    if case.contingency_data["y"] is None:
        return "one-way"
    return "paired" if case.args.get("paired", False) is True else "independence"


def _table(case: OracleCase) -> IntArray:
    return np.array(case.counts_table(), dtype=np.int64)


def _without_empty(table: IntArray) -> IntArray:
    """The table without empty rows and columns (statsExpressions drops unused
    levels before chisq.test; the analysis does the same)."""
    return table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]


def _ratio(case: OracleCase) -> list[float] | None:
    given = case.args.get("ratio")
    return None if given is None else [float(v) for v in given]


def _digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def _params(design: str, *, needs: str = "subtitle") -> list[Any]:
    chosen = [
        c
        for c in CASES
        # The Bayes subtitle is v0.2; its per-bar and pairwise tables are the
        # frequentist ones of every type.
        if (c.type != "bayes" or needs != "subtitle")
        and _design(c) == design
        and (
            (needs == "subtitle" and bool(c.subtitle_rows))
            or (needs == "one_sample" and bool(c.one_sample))
            or (needs == "pairwise" and bool(c.pairwise))
        )
    ]
    return [pytest.param(c, id=c.name) for c in require_cases(chosen, design)]


def _format_value(x: float, digits: int) -> str:
    """insight::format_value as probe-format pins it (SPEC.md section 7): fixed
    digits, scientific below 10^-digits and from 1e5 on."""
    if x != 0 and (math.log10(abs(x)) < -digits or abs(x) >= 1e5):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


_QUOTED = re.compile(r'== "([^"]*)"')
_CI = re.compile(r'"\[" \* "([^"]*)", "([^"]*)" \* "\]"')


def _printed(expression: str) -> tuple[list[str], tuple[str, str]]:
    """The quoted numbers of a subtitle expression (statistic, p, estimate, n) and
    its interval's two bounds."""
    match = _CI.search(expression)
    assert match is not None, expression
    return _QUOTED.findall(expression), (match.group(1), match.group(2))


def _bound_probabilities(level: float, side: str) -> tuple[float, float]:
    alpha = 1 - level
    tail = alpha / 2 if side == "two-sided" else alpha
    low = math.nan if side == "less" else 1 - tail
    high = math.nan if side == "greater" else tail
    return low, high


def _check_bounds(
    effect: str,
    ours: EffectSizeCI,
    r_bounds: tuple[float, float],
    statistic: float,
    df: int,
    n: int,
    shape: tuple[int, int],
    what: str,
) -> set[str]:
    """Both noncentral chi-square bounds against R's within ``noncentral_ci``, or,
    for a bound in CHISQ_ROOT_MISSES, each side against its recorded value; returns
    the sides whose bound is pinned."""
    probabilities = _bound_probabilities(ours.conf_level, ours.sidedness)
    pinned_sides: set[str] = set()
    for side, prob, mine, r_value in zip(
        ("low", "high"),
        probabilities,
        (ours.conf_low, ours.conf_high),
        r_bounds,
        strict=True,
    ):
        key = (effect, f"{statistic:.10g}", df, n, *shape, f"{prob:.6g}")
        pinned = None if math.isnan(prob) else CHISQ_ROOT_MISSES.get(key)
        if pinned is None:
            assert_close("noncentral_ci", mine, r_value, what=f"{what} {side}")
            continue
        r_pinned, accurate = pinned
        assert math.isclose(r_value, r_pinned, rel_tol=1e-9), (what, side, r_value)
        assert math.isclose(mine, accurate, rel_tol=1e-9), (what, side, mine)
        assert not is_close("noncentral_ci", mine, r_value), (what, side, "stale")
        pinned_sides.add(side)
    return pinned_sides


def _check_effect_row(
    row: Row,
    ours: EffectSizeCI,
    effect: str,
    statistic: float,
    df: int,
    n: int,
    shape: tuple[int, int],
    what: str,
) -> set[str]:
    assert_close("closed_form", ours.estimate, value(row, "estimate"), what=what)
    assert ours.conf_level == value(row, "conf.level")
    if effect == "cohens_g":
        assert_close("closed_form", ours.conf_low, value(row, "conf.low"), what=what)
        assert_close("closed_form", ours.conf_high, value(row, "conf.high"), what=what)
        return set()
    return _check_bounds(
        effect,
        ours,
        (value(row, "conf.low"), value(row, "conf.high")),
        statistic,
        df,
        n,
        shape,
        what,
    )


# Bounds where R's stopped-short root (CHISQ_ROOT_MISSES) and the accurate one print
# differently: (case, side) -> (R's text, inferplot's text). At 2 digits every
# pinned fixture bound prints alike (D99); at 3 digits mtcars cyl's does not.
PRINTED_DIFFERENTLY: Final = {
    ("ggbarstats-mtcars-cyl-digits3-parametric", "high"): ("0.499", "0.498"),
}


def _check_printed(
    case: OracleCase,
    row: Row,
    statistic: float,
    p: float,
    ours: EffectSizeCI,
    n: int,
    pinned: set[str],
    digits: int | None = None,
) -> None:
    """Every number of the subtitle as ggstatsplot prints it, from inferplot's
    values: the statistic, p, the estimate and both bounds to ``digits``, n. A
    pinned bound (D99) prints as inferplot's accurate value, which differs from R's
    text only in PRINTED_DIFFERENTLY. ``digits``: the case's unless given."""
    digits = _digits(case) if digits is None else digits
    quoted, bounds = _printed(str(row["expression"]))
    assert quoted[0] == _format_value(statistic, digits), (quoted, statistic)
    assert quoted[1] == _format_value(p, digits), (quoted, p)
    assert quoted[2] == _format_value(ours.estimate, digits), (quoted, ours)
    for side, text, mine in zip(
        ("low", "high"), bounds, (ours.conf_low, ours.conf_high), strict=True
    ):
        expected = PRINTED_DIFFERENTLY.get((case.name, side))
        if expected is None:
            assert text == _format_value(mine, digits), (case.name, side, text, mine)
        else:
            assert side in pinned
            assert (text, _format_value(mine, digits)) == expected
    assert quoted[-1] == (f"{n:,}" if n >= 1000 else str(n))


# --------------------------------------------------------------------------- #
# Subtitles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _params("independence"))
def test_chi_square_of_independence_and_cramers_v(case: OracleCase) -> None:
    row = case.subtitle_data
    table = _without_empty(_table(case))
    test = chisq_independence(table)
    assert row["method"] == "Pearson's Chi-squared test"
    assert_close("closed_form", test.statistic, value(row, "statistic"), what="chi2")
    assert test.df == value(row, "df")
    assert_close("closed_form", test.p_value, value(row, "p.value"), what="p")
    assert test.n == int(value(row, "n.obs")) == case.contingency_data["n"]
    effect = cramers_v(
        table, conf_level=conf_level(case), alternative=alternative(case)
    )
    assert row["effectsize"] == "Cramer's V (adj.)"
    pinned = _check_effect_row(
        row,
        effect,
        "cramers_v",
        test.statistic,
        test.df,
        test.n,
        (table.shape[0], table.shape[1]),
        case.name,
    )
    _check_printed(case, row, test.statistic, test.p_value, effect, test.n, pinned)


@pytest.mark.parametrize("case", _params("paired"))
def test_mcnemar_and_cohens_g(case: OracleCase) -> None:
    row = case.subtitle_data
    table = _table(case)  # every declared level, none dropped
    test = mcnemar(table)
    assert row["method"] == "McNemar's Chi-squared test"
    assert_close("closed_form", test.statistic, value(row, "statistic"), what="chi2")
    assert test.df == value(row, "df")
    assert_close("closed_form", test.p_value, value(row, "p.value"), what="p")
    assert test.n_pairs == int(value(row, "n.obs"))
    effect = cohens_g(table, conf_level=conf_level(case), alternative=alternative(case))
    assert row["effectsize"] == "Cohen's g"
    _check_effect_row(row, effect, "cohens_g", 0.0, 0, 0, (0, 0), case.name)
    _check_printed(case, row, test.statistic, test.p_value, effect, test.n_pairs, set())


@pytest.mark.parametrize("case", _params("one-way"))
def test_goodness_of_fit_and_pearsons_c(case: OracleCase) -> None:
    row = case.subtitle_data
    counts = _table(case)[:, 0]  # every level, a declared empty one as a zero
    test = goodness_of_fit(counts, _ratio(case))
    assert row["method"] == "Chi-squared test for given probabilities"
    assert_close("closed_form", test.statistic, value(row, "statistic"), what="chi2")
    assert test.df == value(row, "df")
    assert_close("closed_form", test.p_value, value(row, "p.value"), what="p")
    assert test.n == int(value(row, "n.obs"))
    effect = pearsons_c(
        counts,
        _ratio(case),
        conf_level=conf_level(case),
        alternative=alternative(case),
    )
    assert row["effectsize"] == "Pearson's C"
    pinned = _check_effect_row(
        row,
        effect,
        "pearsons_c",
        test.statistic,
        test.df,
        test.n,
        (counts.size, 1),
        case.name,
    )
    _check_printed(case, row, test.statistic, test.p_value, effect, test.n, pinned)


# --------------------------------------------------------------------------- #
# Per-bar tests, pairwise tests, shares
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", _params("independence", needs="one_sample"))
def test_per_bar_tests(case: OracleCase) -> None:
    _check_per_bar(case)


@pytest.mark.parametrize("case", _params("paired", needs="one_sample"))
def test_per_bar_tests_of_paired_designs(case: OracleCase) -> None:
    _check_per_bar(case)


def _check_per_bar(case: OracleCase) -> None:
    table = _table(case)
    data = case.contingency_data
    y_levels = [str(v) for v in data["y_levels"]]
    bars = proportion_tests(table, _ratio(case))
    by_column = {bar.column: bar for bar in bars}
    rows = case.one_sample
    # ggstatsplot lists the bars that hold counts in the reverse order of y's levels.
    assert [str(r[data["y"]]) for r in rows] == [y_levels[b.column] for b in bars][::-1]
    for row in rows:
        bar = by_column[y_levels.index(str(row[data["y"]]))]
        what = f"{case.name} bar {row[data['y']]}"
        assert bar.n == int(row["counts"])
        if math.isnan(number(row["statistic"])):
            assert bar.test is None, what
            assert bar.effect is None, what
            assert bar.reason, what
            assert "proportion_test" in case.expected_hidden_errors
            continue
        assert bar.test is not None, what
        assert bar.effect is not None, what
        assert_close(
            "closed_form", bar.test.statistic, value(row, "statistic"), what=what
        )
        assert bar.test.df == value(row, "df")
        assert_close("closed_form", bar.test.p_value, value(row, "p.value"), what=what)
        pinned = _check_effect_row(
            row,
            bar.effect,
            "pearsons_c",
            bar.test.statistic,
            bar.test.df,
            bar.n,
            (table.shape[0], 1),
            what,
        )
        assert (bar.effect.conf_level, bar.effect.sidedness) == (0.95, "two-sided")
        # The row's expression (not drawn) is always printed to 2 digits; the
        # label drawn above the bar follows `digits`.
        _check_printed(
            case,
            row,
            bar.test.statistic,
            bar.test.p_value,
            bar.effect,
            bar.n,
            pinned,
            digits=2,
        )
        digits = _digits(case)
        p_text = _format_value(bar.test.p_value, digits)
        assert row[".p.label"] == f"list(~italic(p)=='{p_text}')"
        assert row[".label"] == (
            f"list(~chi['gof']^2~({bar.test.df})=="
            f"{_format_value(bar.test.statistic, digits)}, ~italic(p)=='{p_text}', "
            f"~italic(n)=='{bar.n:,}')"
        )


P_ADJUST_LABELS: Final[dict[str, str]] = {
    "holm": "Holm",
    "hochberg": "Hochberg",
    "hommel": "Hommel",
    "bonferroni": "Bonferroni",
    "BH": "FDR",
    "BY": "BY",
    "fdr": "FDR",
    "none": "None",
}


@pytest.mark.parametrize("case", _params("independence", needs="pairwise"))
def test_pairwise_fisher(case: OracleCase) -> None:
    table = _table(case)
    x_levels = [str(v) for v in case.contingency_data["x_levels"]]
    method: PAdjustMethod = p_adjust_method(case)
    pairs = pairwise_fisher(
        table,
        p_adjust_method=method,
        conf_level=conf_level(case),
        alternative=alternative(case),
    )
    rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    assert len(rows) == len(pairs)
    for pair in pairs:
        key = (x_levels[pair.row1], x_levels[pair.row2])
        row = rows[key]
        what = f"{case.name} {key[0]}-{key[1]}"
        assert pair.p_value is not None, what
        assert pair.p_adjusted is not None, what
        assert_close("closed_form", pair.p_value, value(row, "p.value"), what=what)
        assert_close(
            "closed_form", pair.p_adjusted, value(row, "p.value.adj"), what=what
        )
        assert row["p.adjust.method"] == P_ADJUST_LABELS[method]
        assert row["test"] == "Fisher's exact test"
        assert pair.effect is not None, what
        sub = table[[pair.row1, pair.row2]][:, list(pair.columns)]
        chi2 = chisq_independence(sub)
        _check_effect_row(
            row,
            pair.effect,
            "cramers_v",
            chi2.statistic,
            chi2.df,
            chi2.n,
            (2, len(pair.columns)),
            what,
        )


@pytest.mark.parametrize("case", [pytest.param(c, id=c.name) for c in CASES])
def test_cell_shares(case: OracleCase) -> None:
    # Each cell's share of its bar, as ggbarstats' descriptive table holds it (15
    # significant digits).
    table = _table(case)
    shares = column_percentages(table)
    data = case.contingency_data
    x_levels = [str(v) for v in data["x_levels"]]
    y_levels = [str(v) for v in data["y_levels"]]
    rows = case.descriptive
    assert rows
    for row in rows:
        i = x_levels.index(str(row[data["x"]]))
        j = 0 if data["y"] is None else y_levels.index(str(row[data["y"]]))
        share = shares[i][j]
        assert share is not None
        assert int(row["counts"]) == int(table[i, j])
        assert_close("closed_form", share, value(row, "perc"), what=case.name)


# --------------------------------------------------------------------------- #
# Cases ggstatsplot fails on or drops silently
# --------------------------------------------------------------------------- #


def test_cases_ggstatsplot_fails_on() -> None:
    assert {c.name for c in FAILED} == FAILED_IN_R
    # A count that is not whole: refused by every function.
    with pytest.raises(ValueError, match="whole"):
        chisq_independence(np.array([[1.5, 2.0], [3.0, 4.0]]))


def test_cases_ggstatsplot_drops() -> None:
    dropped = {c.name for c in CASES if c.expected_hidden_errors and c.type != "bayes"}
    assert dropped == set(DROPPED)
    by_name = {c.name: c for c in CASES}

    def table(name: str) -> IntArray:
        return _table(by_name[name])

    # One cell, one column, one row: no test of independence (R turns the table into
    # a goodness-of-fit test and cramers_v refuses; ggstatsplot drops the subtitle).
    for name in (
        "ggbarstats-edge_bar_degenerate-one-cell-parametric",
        "ggbarstats-edge_bar_degenerate-one-column-parametric",
        "ggbarstats-edge_bar_degenerate-one-row-parametric",
    ):
        with pytest.raises(ValueError, match="at least 2 rows and 2 columns"):
            chisq_independence(table(name))
        with pytest.raises(ValueError, match="at least 2 rows and 2 columns"):
            cramers_v(table(name))
    # The per-bar tests of a single row: a marked entry per bar (R: a row of NA).
    bars = proportion_tests(table("ggbarstats-edge_bar_degenerate-one-row-parametric"))
    assert [b.column for b in bars] == [0, 1]
    assert all(b.test is None and b.reason for b in bars)
    bars = proportion_tests(table("ggbarstats-edge_bar_degenerate-one-cell-parametric"))
    assert len(bars) == 1
    assert bars[0].test is None
    # One category in a one-way design.
    with pytest.raises(ValueError, match="at least 2 categories"):
        goodness_of_fit(
            table("ggbarstats-edge_bar_degenerate-oneway-one-parametric")[:, 0]
        )
    # A pair of rows in one column: no Fisher test for that pair (R loses the whole
    # pairwise table).
    pairs = pairwise_fisher(table("ggbarstats-edge_bar_degenerate-single-x-parametric"))
    assert [(p.row1, p.row2, p.p_value is None) for p in pairs] == [
        (0, 1, True),
        (0, 2, False),
        (1, 2, False),
    ]
    assert pairs[0].reason is not None
    assert pairs[0].p_adjusted is None
    # Paired: no discordant pair; levels that differ (not square).
    agree = table("ggbarstats-edge_bar_paired_agree-parametric")
    with pytest.raises(ValueError, match="discordant"):
        mcnemar(agree)
    with pytest.raises(ValueError, match="discordant"):
        cohens_g(agree)
    with pytest.raises(ValueError, match="square"):
        mcnemar(table("ggbarstats-edge_bar_paired_unequal-parametric"))
    # ratio of the wrong length, or not summing to 1 (never rescaled).
    counts = table("ggbarstats-mtcars-cyl-ratio-short-parametric")[:, 0]
    with pytest.raises(ValueError, match="one probability per category"):
        goodness_of_fit(counts, [0.5, 0.5])
    with pytest.raises(ValueError, match="sum to 1"):
        goodness_of_fit(counts, [2.0, 1.0, 1.0])


def test_every_fixture_is_covered() -> None:
    # Every completed case has its subtitle, per-bar tests or shares checked above;
    # the Bayes subtitles are v0.2.
    designs = {_design(c) for c in CASES}
    assert designs == {"one-way", "paired", "independence"}
    assert len(CASES) == 89
    assert len(FAILED) == 1


# --------------------------------------------------------------------------- #
# probe-contingency
# --------------------------------------------------------------------------- #

PROBE: Final = probe("probe-contingency")


def _hex(text: str) -> float:
    number_ = r_double(text)
    assert number_ is not None
    return number_


def _side(r_alternative: str) -> Any:
    return {"two.sided": "two-sided"}.get(r_alternative, r_alternative)


def test_probe_chi_square_of_independence() -> None:
    checked = 0
    for entry in PROBE.rows("chisq"):
        m = np.array(entry["table"], dtype=np.int64)
        result = entry["uncorrected"]
        if (m.sum(axis=0) == 0).any() or (m.sum(axis=1) == 0).any():
            # R: NaN with a warning; inferplot refuses the empty row.
            assert math.isnan(_hex(result["statistic_hex"]))
            with pytest.raises(ValueError, match="empty rows"):
                chisq_independence(m)
            continue
        test = chisq_independence(m)
        what = str(m.tolist())
        assert_close(
            "closed_form", test.statistic, _hex(result["statistic_hex"]), what=what
        )
        assert test.df == result["parameter"]
        assert_close(
            "closed_form", test.p_value, _hex(result["p_value_hex"]), what=what
        )
        # R's expected counts are column-major (a matrix read down its columns).
        expected = np.array([_hex(h) for h in result["expected_hex"]])
        ours = np.array(test.expected).ravel(order="F")
        np.testing.assert_allclose(ours, expected, rtol=1e-15, atol=0)
        checked += 1
    assert checked == 21


def test_probe_goodness_of_fit() -> None:
    messages = {
        "'x' must at least have 2 elements": "at least 2 categories",
        "'x' and 'p' must have the same number of elements": "one probability",
        "probabilities must sum to 1.": "sum to 1",
    }
    checked = 0
    for entry in PROBE.rows("goodness_of_fit"):
        x = np.array(entry["x"], dtype=np.int64)
        ratio = [_hex(h) for h in entry["p_hex"]] if entry["p_given"] else None
        result = entry["chisq"]
        if result["error"] is not None:
            with pytest.raises(ValueError, match=messages[result["error"]]):
                goodness_of_fit(x, ratio if ratio is not None else None)
            continue
        test = goodness_of_fit(x, ratio)
        what = str(x.tolist())
        assert_close(
            "closed_form", test.statistic, _hex(result["statistic_hex"]), what=what
        )
        assert test.df == result["parameter"]
        assert_close(
            "closed_form", test.p_value, _hex(result["p_value_hex"]), what=what
        )
        expected = [_hex(h) for h in result["expected_hex"]]
        np.testing.assert_allclose(test.expected, expected, rtol=1e-15, atol=0)
        for row in entry["pearsons_c"]:
            effect = pearsons_c(
                x, ratio, conf_level=row["ci"], alternative=_side(row["alternative"])
            )
            assert_close(
                "closed_form", effect.estimate, _hex(row["estimate_hex"]), what=what
            )
            _check_bounds(
                "pearsons_c",
                effect,
                (_hex(row["ci_low_hex"]), _hex(row["ci_high_hex"])),
                test.statistic,
                test.df,
                test.n,
                (x.size, 1),
                f"{what} {row['alternative']} {row['ci']}",
            )
            checked += 1
    # 10 designs that R answers x 6 rows (3 alternatives x 2 levels).
    assert checked == 60


def test_probe_cramers_v() -> None:
    checked = 0
    for entry in PROBE.rows("cramers_v"):
        m = np.array(entry["table"], dtype=np.int64)
        for row in entry["rows"]:
            adjust = bool(row["adjust"])
            side = _side(row["alternative"])
            if row["error"] is not None:
                # An empty row: R refuses it as inferplot does.
                with pytest.raises(ValueError, match="empty rows"):
                    cramers_v(m, conf_level=row["ci"], alternative=side, adjust=adjust)
                continue
            if adjust and m.sum() <= max(m.shape):
                # n = 2 in a 2 x 2 table: R gives NaN, inferplot refuses.
                assert math.isnan(_hex(row["estimate_hex"]))
                with pytest.raises(ValueError, match="more counts than rows"):
                    cramers_v(m, conf_level=row["ci"], alternative=side)
                continue
            effect = cramers_v(m, conf_level=row["ci"], alternative=side, adjust=adjust)
            test = chisq_independence(m)
            what = f"{m.tolist()} adjust={adjust} {side} {row['ci']}"
            assert_close(
                "closed_form", effect.estimate, _hex(row["estimate_hex"]), what=what
            )
            _check_bounds(
                "cramers_v" if adjust else "cramers_v_unadjusted",
                effect,
                (_hex(row["ci_low_hex"]), _hex(row["ci_high_hex"])),
                test.statistic,
                test.df,
                test.n,
                (m.shape[0], m.shape[1]),
                what,
            )
            checked += 1
    # 20 tables x 12 rows, and the 6 unadjusted rows of n = 2 (the table with an
    # empty row left out).
    assert checked == 246


def test_probe_mcnemar_and_cohens_g() -> None:
    checked = 0
    for entry in PROBE.rows("paired"):
        m = np.array(entry["table"], dtype=np.int64)
        result = entry["uncorrected"]
        if m.shape[0] != m.shape[1]:
            assert result["error"].startswith("'x' must be square")
            with pytest.raises(ValueError, match="square"):
                mcnemar(m)
            with pytest.raises(ValueError, match="square"):
                cohens_g(m)
            continue
        if math.isnan(_hex(result["statistic_hex"])):
            # A pair of empty off-diagonal cells: 0 / 0, R returns NaN on the same df.
            with pytest.raises(ValueError, match=r"discordant|empty"):
                mcnemar(m)
        else:
            test = mcnemar(m)
            what = str(m.tolist())
            assert_close(
                "closed_form", test.statistic, _hex(result["statistic_hex"]), what=what
            )
            assert test.df == result["parameter"]
            assert_close(
                "closed_form", test.p_value, _hex(result["p_value_hex"]), what=what
            )
        for row in entry["cohens_g"]:
            side = _side(row["alternative"])
            if row["error"] is not None:
                assert row["error"] == "not enough data"
                with pytest.raises(ValueError, match="discordant"):
                    cohens_g(m, conf_level=row["ci"], alternative=side)
                continue
            effect = cohens_g(m, conf_level=row["ci"], alternative=side)
            what = f"{m.tolist()} {side} {row['ci']}"
            for ours, key in (
                (effect.estimate, "estimate_hex"),
                (effect.conf_low, "ci_low_hex"),
                (effect.conf_high, "ci_high_hex"),
            ):
                assert_close("closed_form", ours, _hex(row[key]), what=f"{what} {key}")
            checked += 1
    # 10 square tables with a discordant pair x 6 rows.
    assert checked == 60


def test_probe_fisher() -> None:
    checked = 0
    for entry in PROBE.rows("fisher"):
        m = np.array(entry["table"], dtype=np.int64)
        for test in entry["tests"]:
            side = _side(test["alternative"])
            r_p = _hex(test["p_value_hex"])
            what = f"{m.tolist()} {side}"
            if m.shape[0] != 2:
                # fisher.test's r x c tables are outside barstats (pairs of rows).
                with pytest.raises(ValueError, match="2 x c"):
                    fisher_exact(m)
                continue
            if m.shape != (2, 2) and side != "two-sided":
                # R ignores alternative beyond 2 x 2 (the same p); inferplot refuses.
                assert r_p == _hex(entry["tests"][0]["p_value_hex"])
                with pytest.raises(ValueError, match="one-sided"):
                    fisher_exact(m, alternative=side)
                continue
            p = fisher_exact(m, alternative=side).p_value
            assert_close("closed_form", p, r_p, what=what)
            checked += 1
    # 14 2 x 2 tables x 3 alternatives, 4 2 x c tables two-sided.
    assert checked == 46


def test_probe_percent_labels() -> None:
    # ggbarstats' share of a bar, bit for bit, and R's round() as Python's.
    rows = PROBE.rows("percent_labels")
    for row in rows:
        k, n = int(row["k"]), int(row["n"])
        share = column_percentages([[k], [n - k]])[0][0]
        assert share is not None
        assert share == _hex(row["perc_hex"]), row
        assert round(share, int(row["digits"])) == _hex(row["rounded_hex"]), row
    assert len(rows) == 927
