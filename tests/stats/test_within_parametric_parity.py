"""inferplot.stats' paired t, Hedges' g, repeated-measures ANOVA, partial omega
squared and paired t pairs against every parametric ggwithinstats fixture and
against probe-rmanova (``afex::aov_ez`` with no, GG and HF corrections, the
sphericity epsilons, the error strata and ``effectsize::omega_squared``).

The statistics run on the subjects-by-conditions table each fixture records
(``within_data``), not through the data layer, so a failure here is a method
difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inferplot.stats import (
    paired_hedges_g,
    paired_t,
    paired_t_pairs,
    rm_anova,
    rm_partial_omega_squared,
)
from tests.oracle_fixtures import (
    OracleCase,
    cases_for,
    number,
    probe,
    r_double,
    require_cases,
)
from tests.tolerances import assert_close, is_close

CASES = require_cases(
    [
        c
        for c in cases_for("ggwithinstats")
        if c.type == "parametric" and c.subtitle_rows
    ],
    "parametric ggwithinstats fixtures with a subtitle",
)
TWO = [pytest.param(c, id=c.name) for c in CASES if len(c.analysis_levels) == 2]
MORE = [pytest.param(c, id=c.name) for c in CASES if len(c.analysis_levels) >= 3]
PAIRS = [
    pytest.param(c, id=c.name)
    for c in cases_for("ggwithinstats")
    if c.type == "parametric" and c.pairwise
]
DESIGNS = probe("probe-rmanova").rows("designs")


def _columns(case: OracleCase) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    return [np.array(v, dtype=np.float64) for v in case.within_columns().values()]


def _side(case: OracleCase) -> Any:
    return str(case.args.get("alternative", "two.sided")).replace(".", "-")


@pytest.mark.parametrize("case", TWO)
def test_paired_t_and_hedges_g(case: OracleCase) -> None:
    a, b = _columns(case)
    row = case.subtitle_data
    side = _side(case)
    level = float(case.args.get("conf.level", 0.95))
    result = paired_t(a, b, alternative=side)
    assert row["method"] == "Paired t-test"
    assert_close("closed_form", result.t, number(row["statistic"]), what="t")
    assert_close("closed_form", result.df, number(row["df.error"]), what="df")
    assert_close("closed_form", result.p_value, number(row["p.value"]), what="p")
    g = paired_hedges_g(a, b, conf_level=level, alternative=side)
    assert row["effectsize"] == "Hedges' g"
    assert_close("closed_form", g.estimate, number(row["estimate"]), what="g")
    assert_close("noncentral_ci", g.conf_low, number(row["conf.low"]), what="low")
    assert_close("noncentral_ci", g.conf_high, number(row["conf.high"]), what="high")


@pytest.mark.parametrize("case", MORE)
def test_rm_anova_and_omega(case: OracleCase) -> None:
    columns = _columns(case)
    row = case.subtitle_data
    level = float(case.args.get("conf.level", 0.95))
    anova = rm_anova(columns)
    assert_close("closed_form", anova.f, number(row["statistic"]), what="F")
    assert_close("closed_form", anova.df1_corrected, number(row["df"]), what="df1")
    assert_close(
        "closed_form", anova.df2_corrected, number(row["df.error"]), what="df2"
    )
    assert_close("closed_form", anova.p_value, number(row["p.value"]), what="p")
    omega = rm_partial_omega_squared(columns, conf_level=level)
    assert row["effectsize"] == "Omega2 (partial)"
    assert number(row["conf.level"]) == level
    assert_close("closed_form", omega.estimate, number(row["estimate"]), what="omega")
    assert_close("noncentral_ci", omega.conf_low, number(row["conf.low"]), what="low")
    assert omega.conf_high == number(row["conf.high"]) == 1.0


@pytest.mark.parametrize("case", PAIRS)
def test_paired_t_pairs(case: OracleCase) -> None:
    method = str(case.args.get("p.adjust.method", "holm"))
    levels = case.analysis_levels
    rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    pairs = paired_t_pairs(_columns(case), p_adjust_method=method)  # type: ignore[arg-type]
    assert len(pairs) == len(rows)
    for pair in pairs:
        row = rows[(levels[pair.i], levels[pair.j])]
        assert_close(
            "closed_form", pair.p_adjusted, number(row["p.value"]), what=f"{pair}"
        )


# --------------------------------------------------------------------------- #
# probe-rmanova
# --------------------------------------------------------------------------- #


def _design(design: Any) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    out = []
    for level in design["levels"]:
        values = [r_double(v) for v in design["values_hex"][level]]
        assert all(v is not None for v in values)
        out.append(np.array(values, dtype=np.float64))
    return out


def _table(design: Any, correction: str) -> Any:
    (entry,) = [a for a in design["anova"] if a["correction"] == correction]
    assert entry["error"] is None
    (row,) = entry["table"]
    return row


def test_probe_covers_the_designs() -> None:
    names = {d["name"] for d in DESIGNS}
    assert len(names) == len(DESIGNS) == 9
    assert {"bugs_long", "edge_within_small", "edge_within_nonspherical"} <= names


@pytest.mark.parametrize("design", DESIGNS, ids=[d["name"] for d in DESIGNS])
def test_probe_rmanova(design: Any) -> None:
    columns = _design(design)
    anova = rm_anova(columns)
    plain, gg = _table(design, "none"), _table(design, "GG")
    for got, row, column in (
        (anova.f, plain, "F"),
        (anova.df1, plain, "num Df"),
        (anova.df2, plain, "den Df"),
        (anova.p_uncorrected, plain, "Pr(>F)"),
        (anova.f, gg, "F"),
        (anova.df1_corrected, gg, "num Df"),
        (anova.df2_corrected, gg, "den Df"),
        (anova.p_value, gg, "Pr(>F)"),
    ):
        expected = r_double(row[f"{column}_hex"])
        assert expected is not None
        assert_close("closed_form", got, expected, what=f"{design['name']} {column}")
    (adjustment,) = design["pval_adjustments"]
    assert_close(
        "closed_form", anova.epsilon, number(adjustment["GG eps"]), what="GG eps"
    )
    # Huynh-Feldt is never used: its df differ from the Greenhouse-Geisser ones
    # wherever its epsilon is below 1 (edge_within_missing and edge_within_ties
    # have HF eps > 1, which afex treats as 1, the uncorrected df).
    hf = _table(design, "HF")
    if number(adjustment["HF eps"]) < 1.0:
        assert not is_close("closed_form", anova.df1_corrected, number(hf["num Df"]))


@pytest.mark.parametrize("design", DESIGNS, ids=[d["name"] for d in DESIGNS])
def test_probe_rmanova_strata_and_omega(design: Any) -> None:
    columns = _design(design)
    strata = design["strata"]
    (subjects,) = strata["Error: .rowid"]
    condition, residuals = strata["Error: .rowid:condition"]
    # The error strata as afex's aov reports them: F is MS_c / MS_e uncorrected.
    f = number(condition["Mean Sq"]) / number(residuals["Mean Sq"])
    assert_close("closed_form", rm_anova(columns).f, f, what="strata F")
    ss = (
        number(condition["Sum Sq"]),
        number(residuals["Sum Sq"]),
        number(subjects["Sum Sq"]),
    )
    df_c, df_e, df_s = (
        number(condition["Df"]),
        number(residuals["Df"]),
        number(subjects["Df"]),
    )
    omega_strata = max(
        0.0, df_c * (ss[0] / df_c - ss[1] / df_e) / (sum(ss) + ss[2] / df_s)
    )
    for entry in design["omega_squared"]:
        assert entry["error"] is None
        (row,) = entry["value"]
        level = number(entry["ci"])
        omega = rm_partial_omega_squared(columns, conf_level=level)
        assert_close(
            "closed_form", omega.estimate, number(row["Omega2_partial"]), what="omega"
        )
        assert_close("closed_form", omega_strata, omega.estimate, what="strata omega")
        assert_close("noncentral_ci", omega.conf_low, number(row["CI_low"]), what="low")
        assert omega.conf_high == number(row["CI_high"]) == 1.0
    # Negative control: effectsize's F_to_omega2 on the uncorrected F is another
    # quantity (0.181 against 0.078 on bugs_long).
    for entry in design["f_to_omega2_uncorrected"]:
        (row,) = entry["value"]
        other = number(row["Omega2_partial"])
        estimate = rm_partial_omega_squared(columns).estimate
        assert other == estimate == 0.0 or not is_close("closed_form", estimate, other)
