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
    nonparametric  Kruskal-Wallis (SciPy); Dunn then Holm (inline, Dunn 1964)
    robust         trimmed-means ANOVA (reference t1way); lincon pairwise Yuen tests
                   then Holm, with the Studentized maximum modulus CI (inline)
    bayes          none: Bayes ANOVA arrives in v0.2
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, optimize, stats
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


@pytest.mark.parametrize("case", _params("nonparametric", two_groups=False))
def test_dunn_holm(case: OracleCase) -> None:
    # Dunn (1964): rank all N values together (mid-ranks for ties); for groups i < j,
    # z = (Rbar_i - Rbar_j) / sqrt(S2 * (1/n_i + 1/n_j)), with the tie correction
    # S2 = N(N + 1)/12 - sum(t^3 - t) / (12(N - 1)) over sets of t tied values.
    # ggstatsplot 1.1.4 (PMCMRplus::kwAllPairsDunnTest) reports |z|, not a signed z,
    # and Holm-adjusts two-sided normal p-values across all k(k - 1)/2 pairs.
    groups = _groups(case)
    pooled = np.concatenate(groups)
    n = pooled.size
    sizes = [g.size for g in groups]
    ranks = stats.rankdata(pooled)
    mean_ranks = [float(r.mean()) for r in np.split(ranks, np.cumsum(sizes)[:-1])]
    ties = np.unique(pooled, return_counts=True)[1].astype(np.float64)
    s2 = n * (n + 1) / 12 - float((ties**3 - ties).sum()) / (12 * (n - 1))
    pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    z_values = [
        abs(mean_ranks[i] - mean_ranks[j])
        / math.sqrt(s2 * (1 / sizes[i] + 1 / sizes[j]))
        for i, j in pairs
    ]
    adjusted = ref.holm([float(2 * stats.norm.sf(z)) for z in z_values])
    levels = case.analysis_levels
    expected = {
        (levels[i], levels[j]): (z, p_adj)
        for (i, j), z, p_adj in zip(pairs, z_values, adjusted, strict=True)
    }
    assert len(case.pairwise) == len(expected)
    for pair in case.pairwise:
        z, p_adj = expected[(str(pair["group1"]), str(pair["group2"]))]
        what = f"Dunn {pair['group1']}-{pair['group2']}"
        assert pair["test"] == "Dunn"
        assert pair["distribution"] == "z"
        assert pair["p.adjust.method"] == "Holm"
        assert_close("closed_form", z, number(pair["statistic"]), what=f"{what} z")
        assert_close(
            "closed_form", p_adj, number(pair["p.value"]), what=f"{what} Holm p"
        )


def _smm_cdf(c: float, n_comparisons: int, df: float) -> float:
    """CDF at ``c`` of the Studentized maximum modulus with ``n_comparisons`` terms.

    The statistic is max_i |Z_i| / S with Z_1..Z_C iid N(0, 1) and, independently,
    S**2 ~ chi2(df) / df, so P(max <= c) = E[erf(c S / sqrt(2)) ** C]. Integrated
    over the density of S = chi(df) / sqrt(df).
    """
    log_norm = (1 - df / 2) * math.log(2) - math.lgamma(df / 2) + math.log(df) / 2

    def integrand(s: float) -> float:
        if s <= 0.0:
            return 0.0
        x = s * math.sqrt(df)
        density = math.exp(log_norm + (df - 1) * math.log(x) - x * x / 2)
        return math.erf(c * s / math.sqrt(2)) ** n_comparisons * density

    upper = float(stats.chi.isf(1e-15, df)) / math.sqrt(df)
    value, _ = integrate.quad(integrand, 0.0, upper, limit=200)
    return value


def _smm_table_entry(alpha: float, n_comparisons: int, df: int) -> float:
    """Upper ``alpha`` point of the Studentized maximum modulus, to two decimals."""
    exact = float(
        optimize.brentq(
            lambda c: _smm_cdf(c, n_comparisons, df) - (1 - alpha),
            1.0,
            50.0,
            xtol=1e-13,
        )
    )
    assert abs(exact * 100 % 1 - 0.5) > 1e-6, f"{exact} is at a rounding boundary"
    return round(exact, 2)


def _lincon_critical_value(alpha: float, n_comparisons: int, df: float) -> float:
    """Critical value of WRS2::lincon's pairwise CIs, as the fixtures show it.

    Two-decimal Studentized maximum modulus table entries at the integer df on either
    side of the Yuen df, interpolated linearly in 1 / df. The exact quantile at the
    Yuen df is up to 1.5e-3 (relative) from the critical value the fixtures imply,
    and Bonferroni t up to 2.1e-2.
    """
    lo, hi = math.floor(df), math.ceil(df)
    at_lo = _smm_table_entry(alpha, n_comparisons, lo)
    if hi == lo:
        return at_lo
    at_hi = _smm_table_entry(alpha, n_comparisons, hi)
    return at_lo + (1 / lo - 1 / df) / (1 / lo - 1 / hi) * (at_hi - at_lo)


@pytest.mark.parametrize("case", _params("robust", two_groups=False))
def test_lincon_holm(case: OracleCase) -> None:
    # ggstatsplot 1.1.4's robust pairwise rows (WRS2::lincon, 20% trimming):
    #   estimate  trimmed mean of group1 minus trimmed mean of group2
    #   p.value   Holm over the per-pair Yuen p-values; Hochberg (lincon's own
    #             default) does not match the PlantGrowth fixture
    #   conf.*    estimate -/+ critical value x Yuen standard error, with the
    #             critical value at the pair's Yuen df and C = number of pairs.
    #             Both fixtures have 3 groups, so 3 pairs: they cannot tell C =
    #             pairs from C = groups.
    levels = case.analysis_levels
    groups = dict(zip(levels, _groups(case), strict=True))
    pairs = [
        (levels[i], levels[j])
        for i in range(len(levels))
        for j in range(i + 1, len(levels))
    ]
    yuen = {(g1, g2): ref.yuen(groups[g1], groups[g2], tr=0.2) for g1, g2 in pairs}
    adjusted = dict(
        zip(pairs, ref.holm([yuen[pair][2] for pair in pairs]), strict=True)
    )
    assert len(case.pairwise) == len(pairs)
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        assert (g1, g2) in yuen, f"unexpected pair {g1}-{g2}"
        what = f"lincon {g1}-{g2}"
        assert row["test"] == "Yuen's trimmed means"
        assert row["p.adjust.method"] == "Holm"
        mean1, d1, _ = ref._trimmed_kernel(groups[g1], 0.2)
        mean2, d2, _ = ref._trimmed_kernel(groups[g2], 0.2)
        estimate = mean1 - mean2
        _, df, _ = yuen[(g1, g2)]
        alpha = 1 - _value(row, "conf.level")
        half_width = _lincon_critical_value(alpha, len(pairs), df) * math.sqrt(d1 + d2)
        assert_close(
            "closed_form", estimate, _value(row, "estimate"), what=f"{what} estimate"
        )
        assert_close(
            "closed_form",
            adjusted[(g1, g2)],
            _value(row, "p.value"),
            what=f"{what} Holm p",
        )
        assert_close(
            "closed_form",
            estimate - half_width,
            _value(row, "conf.low"),
            what=f"{what} conf.low",
        )
        assert_close(
            "closed_form",
            estimate + half_width,
            _value(row, "conf.high"),
            what=f"{what} conf.high",
        )
