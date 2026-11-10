"""inferplot.stats' Yuen test for dependent samples, the AKP effect size of paired
differences and its average over pairs, the trimmed-means repeated-measures ANOVA
and rmmcp's comparisons against every robust ggwithinstats fixture and against
probe-wrs2within (WRS2's ``yuend``, ``rmanova``, ``rmmcp``, ``dep.effect`` and
``wmcpAKP`` on a grid of inputs, with 1 - cdf and the exact tail next to each
p-value).

The statistics run on the subjects-by-conditions table each fixture records
(``within_data``), not through the data layer, so a failure here is a method
difference. Bootstrap bounds are compared with each fixture's 20,000-resample R
reference within 4 Monte Carlo SDs of R's 50 replicate runs at 2,000 resamples
(``assert_within_monte_carlo``); an infinite reference (every one of R's
replicate runs infinite too) needs the same infinity.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest
from scipy import stats as scipy_stats

from inferplot.stats import (
    BOOTSTRAP_RESAMPLES,
    average_dependent_akp,
    dependent_akp,
    rmmcp,
    trimmed_mean,
    trimmed_rm_anova,
    winsorized_normal_variance,
    yuen_dependent_t,
)
from tests.oracle_fixtures import (
    OracleCase,
    cases_for,
    number,
    probe,
    r_double,
    require_cases,
    wrs2_akp_constant,
)
from tests.tolerances import assert_close, assert_within_monte_carlo

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

ROBUST = require_cases(
    [c for c in cases_for("ggwithinstats") if c.type == "robust"],
    "robust ggwithinstats fixtures",
)
WITH_SUBTITLE = [c for c in ROBUST if c.subtitle_rows]
TWO = [pytest.param(c, id=c.name) for c in WITH_SUBTITLE if len(c.analysis_levels) == 2]
MORE = [pytest.param(c, id=c.name) for c in WITH_SUBTITLE if len(c.analysis_levels) > 2]
PAIRS = [pytest.param(c, id=c.name) for c in ROBUST if c.pairwise]
PROBE = probe("probe-wrs2within")
# ggstatsplot's dep.effect and wmcpAKP intervals are 95% whatever conf.level says
# (sleep-cl90-robust, bugs_long-cl90-robust).
EFFECT_CONF_LEVEL = 0.95


def _columns(case: OracleCase) -> list[FloatArray]:
    return [np.array(v, dtype=np.float64) for v in case.within_columns().values()]


def _tr(case: OracleCase) -> float:
    return float(case.args.get("tr", 0.2))


def _as_wrs2(estimate: float, tr: float) -> float:
    """An AKP value times WRS2's own scaling constant over the exact one (probe-akp;
    dep.effect and wmcpAKP use akp.effect's constant, probe-wrs2within)."""
    return estimate * wrs2_akp_constant(tr) / math.sqrt(winsorized_normal_variance(tr))


def test_fixture_counts() -> None:
    # 41 robust fixtures: 15 with two conditions, 26 with three or more; two of
    # them (sleep and bugs_long with results.subtitle = FALSE) have no subtitle.
    two = [c for c in ROBUST if len(c.analysis_levels) == 2]
    assert (len(two), len(ROBUST) - len(two)) == (15, 26)
    assert len(WITH_SUBTITLE) == 39
    assert not [
        c for c in cases_for("ggwithinstats", failed=True) if c.type == "robust"
    ]


def _check_bounds(case: OracleCase, low: float, high: float) -> None:
    reference = case.bootstrap_reference
    assert reference is not None
    assert number(reference["conf_level"]) == EFFECT_CONF_LEVEL
    replicates = reference["replicates"]
    assert replicates["resamples"] == BOOTSTRAP_RESAMPLES
    for bound, name in ((low, "conf_low"), (high, "conf_high")):
        expected = number(reference[name])
        mean = number(replicates[f"{name}_mean"])
        if math.isinf(expected):
            # Every one of R's 50 runs at 2,000 resamples has this bound infinite.
            assert mean == expected, case.name
        assert_within_monte_carlo(
            bound,
            expected,
            number(replicates[f"{name}_sd"]) if math.isfinite(expected) else 0.0,
            what=f"{case.name} {name}",
        )


# --------------------------------------------------------------------------- #
# two conditions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", TWO)
def test_yuend(case: OracleCase) -> None:
    a, b = _columns(case)
    row = case.subtitle_data
    assert row["method"] == "Yuen's test on trimmed means for dependent samples"
    test = yuen_dependent_t(a, b, tr=_tr(case))
    assert_close("closed_form", test.t, number(row["statistic"]), what="t")
    assert test.df == number(row["df.error"])
    assert row.get("df") is None
    # R's p is 2 (1 - pt), rounded to 2^-52 (D43).
    assert_close(
        "wrs2_p", test.p_value, number(row["p.value"]), what="p", atol_scale=2.0
    )
    assert test.n == number(row["n.obs"]) == a.size


@pytest.mark.parametrize("case", TWO)
def test_dependent_akp(case: OracleCase) -> None:
    a, b = _columns(case)
    row = case.subtitle_data
    trim = _tr(case)
    reference = case.bootstrap_reference
    assert reference is not None
    assert reference["call"].startswith("WRS2::dep.effect(")
    assert f"tr = {trim}" in reference["call"]
    assert (
        row["effectsize"] == "Algina-Keselman-Penfield robust standardized difference"
    )
    # The interval is 95% in ggstatsplot whatever conf.level asks for.
    assert number(row["conf.level"]) == EFFECT_CONF_LEVEL
    assert reference["requested_conf_level"] == float(case.args.get("conf.level", 0.95))
    result = dependent_akp(a, b, tr=trim)
    assert (result.resamples, result.seed, result.conf_level) == (2000, 0, 0.95)
    assert reference["replicates"]["estimate_sd"] == 0
    for expected in (number(row["estimate"]), number(reference["estimate"])):
        assert_close(
            "closed_form", _as_wrs2(result.estimate, trim), expected, what="AKP"
        )
    _check_bounds(case, result.conf_low, result.conf_high)


# --------------------------------------------------------------------------- #
# three or more conditions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", MORE)
def test_trimmed_rm_anova(case: OracleCase) -> None:
    row = case.subtitle_data
    assert row["method"] == (
        "A heteroscedastic one-way repeated measures ANOVA for trimmed means"
    )
    anova = trimmed_rm_anova(_columns(case), tr=_tr(case))
    assert_close("closed_form", anova.f, number(row["statistic"]), what="F")
    assert_close("closed_form", anova.df1, number(row["df"]), what="df1")
    assert_close("closed_form", anova.df2, number(row["df.error"]), what="df2")
    # R's p is 1 - pf, rounded to 2^-53 (D43).
    assert_close("wrs2_p", anova.p_value, number(row["p.value"]), what="p")


@pytest.mark.parametrize("case", MORE)
def test_average_dependent_akp(case: OracleCase) -> None:
    row = case.subtitle_data
    trim = _tr(case)
    reference = case.bootstrap_reference
    assert reference is not None
    assert reference["call"].startswith("WRS2::wmcpAKP(")
    assert row["effectsize"] == (
        "Algina-Keselman-Penfield robust standardized difference average"
    )
    assert number(row["conf.level"]) == EFFECT_CONF_LEVEL
    result = average_dependent_akp(_columns(case), tr=trim)
    assert (result.resamples, result.seed, result.conf_level) == (2000, 0, 0.95)
    for expected in (number(row["estimate"]), number(reference["estimate"])):
        assert_close(
            "closed_form", _as_wrs2(result.estimate, trim), expected, what="AKP"
        )
    _check_bounds(case, result.conf_low, result.conf_high)


@pytest.mark.parametrize("case", PAIRS)
def test_rmmcp(case: OracleCase) -> None:
    columns = _columns(case)
    levels = case.analysis_levels
    method = str(case.args.get("p.adjust.method", "holm"))
    pairs = {
        (levels[p.i], levels[p.j]): p
        for p in rmmcp(columns, tr=_tr(case), p_adjust_method=method)  # type: ignore[arg-type]
    }
    assert len(pairs) == len(case.pairwise)
    scale = max(float(np.max(np.abs(c))) for c in columns)
    for row in case.pairwise:
        pair = pairs[(str(row["group1"]), str(row["group2"]))]
        what = f"{case.name} {row['group1']}-{row['group2']}"
        assert row["test"] == "Yuen's trimmed means"
        # rmmcp's intervals are 95% whatever conf.level says (bugs_long-cl90).
        assert number(row["conf.level"]) == EFFECT_CONF_LEVEL
        for got, column in (
            (pair.estimate, "estimate"),
            (pair.conf_low, "conf.low"),
            (pair.conf_high, "conf.high"),
        ):
            # closed_form's 1e-10 relative to the data: an estimate can be near 0.
            assert abs(got - number(row[column])) <= 1e-10 * scale, f"{what} {column}"
        assert_close(
            "wrs2_p",
            pair.p_adjusted,
            number(row["p.value"]),
            what=f"{what} adjusted p",
            atol_scale=2.0 * len(pairs),
        )


# --------------------------------------------------------------------------- #
# probe-wrs2within
# --------------------------------------------------------------------------- #


def _hex(values: list[str]) -> FloatArray:
    out = [r_double(v) for v in values]
    assert all(v is not None for v in out)
    return np.array(out, dtype=np.float64)


def _value(text: str) -> float:
    value = r_double(text)
    assert value is not None
    return value


@pytest.mark.parametrize(
    "row",
    PROBE.rows("yuend"),
    ids=[f"n{r['n']}-shift{r['shift']}" for r in PROBE.rows("yuend")],
)
def test_probe_yuend(row: Any) -> None:
    x, y = _hex(row["x_hex"]), _hex(row["y_hex"])
    test = yuen_dependent_t(x, y)
    assert_close("closed_form", test.t, _value(row["test_hex"]), what="t")
    assert test.df == _value(row["df_hex"])
    assert_close(
        "closed_form", test.difference, _value(row["diff_hex"]), what="difference"
    )
    # WRS2's p is 2 (1 - pt) bit for bit; inferplot's is the exact tail.
    one_minus_cdf = _value(row["one_minus_cdf_hex"])
    assert _value(row["p_value_hex"]) == min(1.0, 2.0 * one_minus_cdf)
    assert_close(
        "closed_form", test.p_value, 2.0 * _value(row["upper_tail_hex"]), what="tail"
    )
    assert_close(
        "wrs2_p", test.p_value, _value(row["p_value_hex"]), what="p", atol_scale=2.0
    )
    # yuend's interval: difference -/+ t_{0.975, df} se.
    low, high = (_value(v) for v in row["conf_int_hex"])
    critical = float(scipy_stats.t.ppf(0.975, test.df))
    assert_close("closed_form", (high - low) / 2.0, critical * test.se, what="half")
    assert_close("closed_form", (high + low) / 2.0, test.difference, what="centre")


@pytest.mark.parametrize(
    "row",
    PROBE.rows("rmanova"),
    ids=[f"n{r['n']}-shift{r['shift']}" for r in PROBE.rows("rmanova")],
)
def test_probe_rmanova_and_rmmcp(row: Any) -> None:
    columns = [_hex(c) for c in row["values_hex"]]
    anova = trimmed_rm_anova(columns)
    assert_close("closed_form", anova.f, _value(row["test_hex"]), what="F")
    assert_close("closed_form", anova.df1, _value(row["df1_hex"]), what="df1")
    assert_close("closed_form", anova.df2, _value(row["df2_hex"]), what="df2")
    assert _value(row["p_value_hex"]) == _value(row["one_minus_cdf_hex"])
    assert_close("closed_form", anova.p_value, _value(row["upper_tail_hex"]), what="p")
    assert_close("wrs2_p", anova.p_value, _value(row["p_value_hex"]), what="R p")
    # rmmcp's raw p-values (p.adjust.method "none"), estimates and intervals.
    pairs = {(p.i, p.j): p for p in rmmcp(columns, p_adjust_method="none")}
    scale = max(float(np.max(np.abs(c))) for c in columns)
    for entry in row["rmmcp"]:
        pair = pairs[(int(entry["Group"]) - 1, int(entry["Group.1"]) - 1)]
        for got, column in (
            (pair.estimate, "psihat"),
            (pair.conf_low, "ci.lower"),
            (pair.conf_high, "ci.upper"),
        ):
            assert abs(got - _value(entry[f"{column}_hex"])) <= 1e-14 * scale, column
        assert_close(
            "wrs2_p",
            pair.p_value,
            _value(entry["p.value_hex"]),
            what="rmmcp p",
            atol_scale=2.0,
        )
        assert pair.p_value == pair.p_adjusted


@pytest.mark.parametrize(
    "entry",
    PROBE.rows("dep_effect"),
    ids=[f"{e['name']}-tr{e['tr']}" for e in PROBE.rows("dep_effect")],
)
def test_probe_dep_effect_estimate(entry: Any) -> None:
    a, b = (_hex(entry["values_hex"][level]) for level in entry["levels"])
    trim = float(entry["tr"])
    assert entry["error"] is None
    (tidy,) = entry["tidy"]
    assert tidy["term"] == "AKP"
    result = dependent_akp(a, b, tr=trim)
    assert_close(
        "closed_form",
        _as_wrs2(result.estimate, trim),
        _value(tidy["estimate_hex"]),
        what="dep.effect AKP",
    )
    # The other rows of dep.effect's table are other effect sizes (quantile shift,
    # sign); the AKP row is the trimmed mean of the differences over their
    # winsorized SD, not a two-sample AKP of the conditions.
    two_sample = (trimmed_mean(a, tr=trim) - trimmed_mean(b, tr=trim)) / math.sqrt(
        np.var(a, ddof=1)
    )
    assert not math.isclose(result.estimate, two_sample, rel_tol=1e-3)


@pytest.mark.parametrize(
    "entry",
    PROBE.rows("wmcp_akp"),
    ids=[f"{e['name']}-tr{e['tr']}" for e in PROBE.rows("wmcp_akp")],
)
def test_probe_wmcp_akp_estimate(entry: Any) -> None:
    columns = [_hex(entry["values_hex"][level]) for level in entry["levels"]]
    trim = float(entry["tr"])
    assert entry["error"] is None
    result = average_dependent_akp(columns, tr=trim)
    expected = _value(entry["value_hex"]["Effect.Size"][0])
    assert_close(
        "closed_form", _as_wrs2(result.estimate, trim), expected, what="wmcpAKP"
    )
    # It is the mean of the pairs' dep.effect AKP, i < j in level order.
    pairs = [
        dependent_akp(columns[i], columns[j], tr=trim).estimate
        for i, j in itertools.combinations(range(len(columns)), 2)
    ]
    assert_close("closed_form", result.estimate, math.fsum(pairs) / len(pairs), what="")
