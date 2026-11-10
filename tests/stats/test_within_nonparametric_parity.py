"""inferplot.stats' signed-rank test and matched-pairs rank-biserial r (two
conditions), Friedman's test and Durbin-Conover's pairs (three or more) against
every nonparametric ggwithinstats fixture, against probe-signrank
(``stats::wilcox.test(paired = TRUE)`` by exact, correct and alternative,
``effectsize::rank_biserial(paired = TRUE)`` by ci and alternative) and against
probe-friedman (``stats::friedman.test`` and the unadjusted
``PMCMRplus::durbinAllPairsTest`` on 11 designs).

The statistics run on the subjects-by-conditions table each fixture records
(``within_data``), not through the data layer, so a failure here is a method
difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inferplot.stats import (
    durbin_conover,
    friedman,
    paired_rank_biserial,
    signed_rank,
)
from tests.oracle_fixtures import (
    OracleCase,
    cases_for,
    number,
    probe,
    r_double,
    require_cases,
)
from tests.tolerances import assert_close

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

CASES = require_cases(
    [
        c
        for c in cases_for("ggwithinstats")
        if c.type == "nonparametric" and c.subtitle_rows
    ],
    "nonparametric ggwithinstats fixtures with a subtitle",
)
TWO = [pytest.param(c, id=c.name) for c in CASES if len(c.analysis_levels) == 2]
MORE = [pytest.param(c, id=c.name) for c in CASES if len(c.analysis_levels) >= 3]
PAIRS = [
    pytest.param(c, id=c.name)
    for c in cases_for("ggwithinstats")
    if c.type == "nonparametric" and c.pairwise
]
SIGNRANK = probe("probe-signrank").rows("pairs")
DESIGNS = probe("probe-friedman").rows("designs")


def _columns(case: OracleCase) -> list[FloatArray]:
    return [np.array(v, dtype=np.float64) for v in case.within_columns().values()]


def _side(value: object) -> Any:
    return str(value).replace(".", "-")


@pytest.mark.parametrize("case", TWO)
def test_signed_rank_and_rank_biserial(case: OracleCase) -> None:
    a, b = _columns(case)
    row = case.subtitle_data
    side = _side(case.args.get("alternative", "two.sided"))
    level = float(case.args.get("conf.level", 0.95))
    result = signed_rank(a, b, alternative=side)
    assert row["method"] == "Wilcoxon signed rank test"
    assert_close("closed_form", result.statistic, number(row["statistic"]), what="V")
    assert_close("closed_form", result.p_value, number(row["p.value"]), what="p")
    r = paired_rank_biserial(a, b, conf_level=level, alternative=side)
    assert row["effectsize"] == "r (rank biserial)"
    assert row["conf.method"] == "normal"
    assert_close("closed_form", r.estimate, number(row["estimate"]), what="r")
    assert_close("closed_form", r.conf_low, number(row["conf.low"]), what="low")
    assert_close("closed_form", r.conf_high, number(row["conf.high"]), what="high")


# --------------------------------------------------------------------------- #
# probe-signrank
# --------------------------------------------------------------------------- #


def _pair(entry: Any) -> tuple[FloatArray, FloatArray]:
    a = [r_double(v) for v in entry["a_hex"]]
    b = [r_double(v) for v in entry["b_hex"]]
    return np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)


def test_probe_covers_the_designs() -> None:
    names = {p["name"] for p in SIGNRANK}
    assert len(names) == len(SIGNRANK) == 12
    assert {"all_zero", "one_nonzero", "n3", "untied_n49", "untied_n50"} <= names
    sizes = {p["name"]: len(p["a_hex"]) for p in SIGNRANK}
    assert (sizes["n3"], sizes["untied_n49"], sizes["untied_n50"]) == (3, 49, 50)


@pytest.mark.parametrize("entry", SIGNRANK, ids=[p["name"] for p in SIGNRANK])
def test_probe_signed_rank(entry: Any) -> None:
    """ggstatsplot's call, exact = FALSE with correct = TRUE, for each alternative.
    The other rows show what that choice changes: without the continuity
    correction every p differs, and R's default exact = NULL computes the exact
    distribution where there are no ties, zeros and fewer than 50 pairs."""
    a, b = _pair(entry)
    for row in entry["wilcox"]:
        assert row["error"] is None
        side = _side(row["alternative"])
        expected_p = r_double(row["p_value_hex"])
        assert expected_p is not None
        if entry["name"] == "all_zero":
            # R: V = 0 with p NaN (or 1 from an infinite z): undefined, refused.
            with pytest.raises(ValueError, match="paired differences is 0"):
                signed_rank(a, b, alternative=side)
            continue
        result = signed_rank(a, b, alternative=side)
        what = f"{entry['name']} {row['exact']} {row['correct']} {side}"
        assert result.statistic == r_double(row["statistic_hex"]), what
        if row["exact"] is False and row["correct"] is True:
            assert (
                row["method"] == "Wilcoxon signed rank test with continuity correction"
            )
            assert_close("closed_form", result.p_value, expected_p, what=what)
        elif row["correct"] is False and row["exact"] is False:
            # Negative control: the uncorrected normal p differs unless z = 0.
            assert result.p_value != expected_p or result.z == 0.0, what


@pytest.mark.parametrize("entry", SIGNRANK, ids=[p["name"] for p in SIGNRANK])
def test_probe_rank_biserial(entry: Any) -> None:
    a, b = _pair(entry)
    for row in entry["rank_biserial"]:
        assert row["error"] is None
        side = _side(row["alternative"])
        level = float(row["ci"])
        what = f"{entry['name']} {level} {side}"
        if entry["name"] == "all_zero":
            # effectsize reports r = 1 for 0/0; inferplot refuses (no difference).
            assert r_double(row["estimate_hex"]) == 1.0
            with pytest.raises(ValueError, match="paired differences is 0"):
                paired_rank_biserial(a, b, conf_level=level, alternative=side)
            continue
        r = paired_rank_biserial(a, b, conf_level=level, alternative=side)
        for got, column in (
            (r.estimate, "estimate_hex"),
            (r.conf_low, "conf_low_hex"),
            (r.conf_high, "conf_high_hex"),
        ):
            expected = r_double(row[column])
            assert expected is not None
            assert_close("closed_form", got, expected, what=f"{what} {column}")


def test_probe_rank_biserial_se_counts_zero_differences() -> None:
    """Negative control: the SE from the nonzero pairs only misses
    edge_within2_ties' interval (by 0.037 at the lower bound)."""
    (entry,) = [p for p in SIGNRANK if p["name"] == "edge_within2_ties"]
    a, b = _pair(entry)
    nonzero = a != b
    assert not nonzero.all()
    (row,) = [
        r
        for r in entry["rank_biserial"]
        if r["ci"] == 0.95 and r["alternative"] == "two.sided"
    ]
    wrong = paired_rank_biserial(a[nonzero], b[nonzero])
    expected = r_double(row["conf_low_hex"])
    assert expected is not None
    assert abs(wrong.conf_low - expected) > 0.03


# --------------------------------------------------------------------------- #
# three or more conditions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", MORE)
def test_friedman(case: OracleCase) -> None:
    row = case.subtitle_data
    result = friedman(_columns(case))
    assert row["method"] == "Friedman rank sum test"
    assert_close("closed_form", result.statistic, number(row["statistic"]), what="chi")
    assert_close("closed_form", result.df, number(row["df.error"]), what="df")
    assert_close("closed_form", result.p_value, number(row["p.value"]), what="p")


@pytest.mark.parametrize("case", PAIRS)
def test_durbin_conover_pairs(case: OracleCase) -> None:
    method = str(case.args.get("p.adjust.method", "holm"))
    levels = case.analysis_levels
    rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    pairs = durbin_conover(_columns(case), p_adjust_method=method)  # type: ignore[arg-type]
    assert len(pairs) == len(rows)
    for pair in pairs:
        row = rows[(levels[pair.i], levels[pair.j])]
        what = f"{case.name} {levels[pair.i]}-{levels[pair.j]}"
        assert row["test"] == "Durbin-Conover"
        assert row["distribution"] == "t"
        t, p = number(row["statistic"]), number(row["p.value"])
        assert_close("closed_form", pair.statistic, t, what=what)
        assert_close("closed_form", pair.p_adjusted, p, what=what)


def _design(design: Any) -> list[FloatArray]:
    out = []
    for level in design["levels"]:
        values = [r_double(v) for v in design["values_hex"][level]]
        assert all(v is not None for v in values)
        out.append(np.array(values, dtype=np.float64))
    return out


def test_friedman_probe_covers_the_designs() -> None:
    names = {d["name"] for d in DESIGNS}
    assert len(names) == len(DESIGNS) == 11
    assert {"bugs_long", "edge_within_small", "tied_rows", "untied"} <= names


@pytest.mark.parametrize("design", DESIGNS, ids=[d["name"] for d in DESIGNS])
def test_probe_friedman(design: Any) -> None:
    columns = _design(design)
    assert design["friedman_error"] is None
    result = friedman(columns)
    for got, column in ((result.statistic, "statistic"), (result.p_value, "p_value")):
        expected = r_double(design[f"friedman_{column}_hex"])
        assert expected is not None
        assert_close("closed_form", got, expected, what=f"{design['name']} {column}")
    assert result.df == number(design["friedman_parameter"])


@pytest.mark.parametrize("design", DESIGNS, ids=[d["name"] for d in DESIGNS])
def test_probe_durbin_conover(design: Any) -> None:
    """The unadjusted pairs; R's statistics are never negative (|t|)."""
    columns = _design(design)
    levels = design["levels"]
    assert design["durbin_error"] is None
    assert design["durbin_dist"] == "t"
    rows = {(r["group1"], r["group2"]): r for r in design["durbin_pairs"]}
    pairs = durbin_conover(columns, p_adjust_method="none")
    assert len(pairs) == len(rows)
    for pair in pairs:
        row = rows[(levels[pair.i], levels[pair.j])]
        what = f"{design['name']} {levels[pair.i]}-{levels[pair.j]}"
        statistic = r_double(row["statistic_hex"])
        p = r_double(row["p_value_hex"])
        assert statistic is not None
        assert p is not None
        assert statistic >= 0.0
        assert_close("closed_form", pair.statistic, statistic, what=f"{what} t")
        assert_close("closed_form", pair.p_value, p, what=f"{what} p")
        assert pair.p_adjusted == pair.p_value
