"""Independent cross-checks of the oracle fixtures against tools/reference and SciPy.

These tests do not exercise inferplot. They confirm that the fixture pipeline (CSV,
R, JSON, loader) carries numbers that an independent implementation of the
published method reproduces, and they pin down which variant R ran.

Two groups:
    parametric     Welch t (SciPy)
    nonparametric  Mann-Whitney U (SciPy), variant chosen by R's default rule
    robust         Yuen (reference)
    bayes          JZS Bayes factor (reference)

Three or more groups:
    parametric     Welch ANOVA, partial omega squared, Games-Howell then Holm
                   (reference)
    nonparametric  Kruskal-Wallis (SciPy)
    robust         trimmed-means ANOVA (reference t1way)
    bayes          none: Bayes ANOVA arrives in v0.2
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats
from tools.reference import stats_reference as ref

from tests.oracle_fixtures import (
    OracleCase,
    cases_for,
    number,
    r_level_label,
    read_dataset,
)
from tests.tolerances import assert_close

FloatArray = NDArray[np.float64]


def _params(type_: str, *, two_groups: bool) -> list[Any]:
    return [
        pytest.param(case, id=case.name)
        for case in cases_for("ggbetweenstats")
        if case.type == type_ and (len(case.analysis_levels) == 2) == two_groups
    ]


def _groups(case: OracleCase) -> list[FloatArray]:
    frame = read_dataset(case)
    x, y = case.args["x"], case.args["y"]
    used = frame[[x, y]].dropna()
    labels = used[x].map(r_level_label)
    return [
        used.loc[labels == level, y].to_numpy(dtype=np.float64)
        for level in case.analysis_levels
    ]


def _value(row: Any, column: str) -> float:
    assert column in row, f"no {column!r} column; columns are {sorted(row)}"
    return number(row[column])


@pytest.mark.parametrize("case", _params("parametric", two_groups=True))
def test_welch_t(case: OracleCase) -> None:
    a, b = _groups(case)
    result = stats.ttest_ind(a, b, equal_var=False)
    row = case.subtitle_data
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="Welch t"
    )
    assert_close(
        "closed_form", float(result.df), _value(row, "df.error"), what="Welch df"
    )
    assert_close(
        "closed_form", float(result.pvalue), _value(row, "p.value"), what="Welch p"
    )


@pytest.mark.parametrize("case", _params("parametric", two_groups=False))
def test_welch_anova_omega_and_games_howell(case: OracleCase) -> None:
    groups = _groups(case)
    f_stat, df1, df2, p = ref.welch_anova(groups)
    row = case.subtitle_data
    assert_close("closed_form", f_stat, _value(row, "statistic"), what="Welch F")
    assert_close("closed_form", df1, _value(row, "df"), what="Welch df1")
    assert_close("closed_form", df2, _value(row, "df.error"), what="Welch df2")
    assert_close("closed_form", p, _value(row, "p.value"), what="Welch p")
    omega = ref.omega2_partial_from_f(f_stat, df1, df2)
    assert_close(
        "closed_form", omega, _value(row, "estimate"), what="partial omega squared"
    )

    # ggstatsplot 1.1.4 reports q with the sign of mean(group2) - mean(group1);
    # the reference returns |q|.
    levels = case.analysis_levels
    pairs = ref.games_howell(groups)
    adjusted = ref.holm([p_raw for *_, p_raw in pairs])
    expected = {
        (levels[i], levels[j]): (
            math.copysign(q, float(groups[j].mean() - groups[i].mean())),
            p_adj,
        )
        for (i, j, q, _), p_adj in zip(pairs, adjusted, strict=True)
    }
    assert len(case.pairwise) == len(expected)
    for pair in case.pairwise:
        q, p_adj = expected[(str(pair["group1"]), str(pair["group2"]))]
        what = f"Games-Howell {pair['group1']}-{pair['group2']}"
        assert pair["test"] == "Games-Howell"
        assert pair["p.adjust.method"] == "Holm"
        assert_close("closed_form", q, number(pair["statistic"]), what=f"{what} q")
        assert_close(
            "games_howell_p", p_adj, number(pair["p.value"]), what=f"{what} Holm p"
        )


@pytest.mark.parametrize("case", _params("nonparametric", two_groups=True))
def test_mann_whitney(case: OracleCase) -> None:
    a, b = _groups(case)
    row = case.subtitle_data
    # R's wilcox.test defaults: exact p when both samples have fewer than 50 values
    # and there are no ties, otherwise the normal approximation with continuity
    # correction. The method string in the fixture does not say which was used.
    ties = np.unique(np.concatenate([a, b])).size < a.size + b.size
    if a.size < 50 and b.size < 50 and not ties:
        result = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact")
    else:
        result = stats.mannwhitneyu(
            a, b, alternative="two-sided", method="asymptotic", use_continuity=True
        )
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="W"
    )
    assert_close("closed_form", float(result.pvalue), _value(row, "p.value"), what="p")


@pytest.mark.parametrize("case", _params("nonparametric", two_groups=False))
def test_kruskal_wallis(case: OracleCase) -> None:
    groups = _groups(case)
    result = stats.kruskal(*groups)
    row = case.subtitle_data
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="H"
    )
    assert_close("closed_form", float(result.pvalue), _value(row, "p.value"), what="p")


@pytest.mark.parametrize("case", _params("robust", two_groups=True))
def test_yuen(case: OracleCase) -> None:
    a, b = _groups(case)
    t, df, p = ref.yuen(a, b, tr=0.2)
    row = case.subtitle_data
    assert_close("closed_form", t, _value(row, "statistic"), what="Yuen t")
    assert_close("closed_form", df, _value(row, "df.error"), what="Yuen df")
    assert_close("closed_form", p, _value(row, "p.value"), what="Yuen p")


@pytest.mark.parametrize("case", _params("robust", two_groups=False))
def test_t1way(case: OracleCase) -> None:
    f_stat, df1, df2, p = ref.t1way(_groups(case), tr=0.2)
    row = case.subtitle_data
    assert_close("closed_form", f_stat, _value(row, "statistic"), what="t1way F")
    assert_close("closed_form", df1, _value(row, "df"), what="t1way df1")
    assert_close("closed_form", df2, _value(row, "df.error"), what="t1way df2")
    assert_close("closed_form", p, _value(row, "p.value"), what="t1way p")


@pytest.mark.parametrize("case", _params("bayes", two_groups=True))
def test_jzs_bayes_factor(case: OracleCase) -> None:
    a, b = _groups(case)
    t = float(stats.ttest_ind(a, b, equal_var=True).statistic)
    n_eff = a.size * b.size / (a.size + b.size)
    bf10 = ref.jzs_bf10_t(t, n_eff, float(a.size + b.size - 2), r=0.707)
    row = case.subtitle_data
    assert_close("bayes_factor", bf10, _value(row, "bf10"), what="BF10")
    assert math.isfinite(bf10)
