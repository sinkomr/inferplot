"""Statistics-level parity of the robust layer with ggstatsplot 1.1.4.

Every robust ggbetweenstats fixture is recomputed with ``inferplot.stats`` from the
fixture's own dataset and groups: Yuen's test and the AKP effect size for two groups;
the trimmed-means ANOVA (WRS2's t1way), the explanatory measure of effect size and
lincon's pairwise rows for three or more. Bootstrap CIs are compared with each
fixture's 20,000-resample R reference (SPEC.md section 10), never with
ggstatsplot's 100-resample values.
"""

from __future__ import annotations

import math

import pytest

from inferplot.stats import (
    BOOTSTRAP_RESAMPLES,
    LinconPair,
    akp_effect,
    explanatory_effect,
    lincon,
    smm_critical_value,
    trimmed_mean,
    trimmed_means_anova,
    yuen_t,
)
from inferplot.stats.robust import SMM_ROWS
from tests.oracle_fixtures import OracleCase, number, r_level_label
from tests.stats.fixture_data import (
    between_cases,
    conf_level,
    groups,
    p_adjust_method,
    params,
    value,
)
from tests.tolerances import assert_close, assert_within_monte_carlo

ROBUST = between_cases("robust")
TWO = between_cases("robust", two_groups=True)
MANY = between_cases("robust", two_groups=False)


def _equal_sizes(case: OracleCase) -> bool:
    return len(set(case.n_per_level.values())) == 1


def test_every_robust_case_has_the_tables_checked_here() -> None:
    assert len(ROBUST) == len(TWO) + len(MANY) >= 10
    for case in ROBUST:
        assert case.subtitle_rows, case.name
        assert case.bootstrap_reference is not None, case.name
        if len(case.analysis_levels) > 2:
            assert case.pairwise, case.name


# ------------------------------------------------------------------ two groups


@pytest.mark.parametrize("case", params(TWO, "two-group robust subtitles"))
def test_yuen(case: OracleCase) -> None:
    a, b = groups(case)
    row = case.subtitle_data
    assert row["method"] == "Yuen's test on trimmed means for independent samples"
    result = yuen_t(a, b)
    # ggstatsplot reports |t|, whatever the sign of the trimmed-mean difference.
    assert_close("closed_form", result.t, value(row, "statistic"), what="Yuen t")
    assert result.t >= 0.0
    assert_close("closed_form", result.df, value(row, "df.error"), what="Yuen df")
    # R rounds 2 (1 - pt(|t|, df)) to 2^-52; inferplot reports the accurate tail
    # (D43), within one such step of it.
    assert_close(
        "wrs2_p", result.p_value, value(row, "p.value"), what="Yuen p", atol_scale=2.0
    )


@pytest.mark.parametrize("case", params(TWO, "two-group robust effect sizes"))
def test_akp_effect(case: OracleCase) -> None:
    a, b = groups(case)
    level = conf_level(case)
    row = case.subtitle_data
    assert row["effectsize"] == (
        "Algina-Keselman-Penfield robust standardized difference"
    )
    reference = case.bootstrap_reference
    assert reference is not None
    assert "EQVAR = FALSE" in reference["call"]
    assert reference["conf_level"] == level
    result = akp_effect(a, b, conf_level=level)
    assert result.resamples == BOOTSTRAP_RESAMPLES
    assert result.seed == 0
    # The estimate does not depend on the resamples (replicate SD 0); it differs
    # from WRS2's only by WRS2's scaling constant (tolerances.py).
    assert reference["replicates"]["estimate_sd"] == 0
    assert_close(
        "wrs2_akp_constant", result.estimate, value(row, "estimate"), what="AKP"
    )
    assert_close(
        "wrs2_akp_constant",
        result.estimate,
        number(reference["estimate"]),
        what="AKP (reference)",
    )
    _check_bootstrap_bounds(case, result.conf_low, result.conf_high)


def _check_bootstrap_bounds(case: OracleCase, low: float, high: float) -> None:
    reference = case.bootstrap_reference
    assert reference is not None
    replicates = reference["replicates"]
    assert replicates["resamples"] == BOOTSTRAP_RESAMPLES
    assert_within_monte_carlo(
        low,
        number(reference["conf_low"]),
        number(replicates["conf_low_sd"]),
        what=f"{case.name} lower bound",
    )
    assert_within_monte_carlo(
        high,
        number(reference["conf_high"]),
        number(replicates["conf_high_sd"]),
        what=f"{case.name} upper bound",
    )


# ------------------------------------------------------------------ k >= 3


@pytest.mark.parametrize("case", params(MANY, "robust subtitles, k >= 3"))
def test_trimmed_means_anova(case: OracleCase) -> None:
    row = case.subtitle_data
    assert row["method"] == "A heteroscedastic one-way ANOVA for trimmed means"
    result = trimmed_means_anova(groups(case))
    assert_close("closed_form", result.f, value(row, "statistic"), what="F")
    assert result.df1 == value(row, "df")
    assert_close("closed_form", result.df2, value(row, "df.error"), what="df2")
    # R rounds 1 - pf(F, df1, df2) to 2^-53; inferplot reports the accurate tail
    # (D43), within one such step of it.
    assert_close("wrs2_p", result.p_value, value(row, "p.value"), what="t1way p")


@pytest.mark.parametrize("case", params(MANY, "robust effect sizes, k >= 3"))
def test_explanatory_effect(case: OracleCase) -> None:
    level = conf_level(case)
    row = case.subtitle_data
    assert row["effectsize"] == "Explanatory measure of effect size"
    reference = case.bootstrap_reference
    assert reference is not None
    assert reference["conf_level"] == level
    result = explanatory_effect(groups(case), conf_level=level)
    assert result.resamples == BOOTSTRAP_RESAMPLES
    replicates = reference["replicates"]
    if _equal_sizes(case):
        # Equal group sizes: the estimate does not depend on the resamples and
        # differs from WRS2's only by WRS2's constant 0.4120867 (tolerances.py).
        assert replicates["estimate_sd"] == 0
        assert_close(
            "wrs2_xi_constant", result.estimate, value(row, "estimate"), what="xi"
        )
        assert_close(
            "wrs2_xi_constant",
            result.estimate,
            number(reference["estimate"]),
            what="xi (reference)",
        )
    else:
        # Unequal sizes: the estimate is the median of the bootstrap values, so it
        # is random and is compared like a bound.
        assert number(replicates["estimate_sd"]) > 0
        assert_within_monte_carlo(
            result.estimate,
            number(reference["estimate"]),
            number(replicates["estimate_sd"]),
            what=f"{case.name} xi",
        )
    _check_bootstrap_bounds(case, result.conf_low, result.conf_high)


def _pairs_by_name(case: OracleCase) -> dict[tuple[str, str], LinconPair]:
    levels = case.analysis_levels
    pairs = lincon(groups(case), p_adjust_method=p_adjust_method(case))
    assert len(pairs) == len(levels) * (len(levels) - 1) // 2 == len(case.pairwise)
    return {(levels[p.i], levels[p.j]): p for p in pairs}


@pytest.mark.parametrize("case", params(MANY, "robust pairwise rows, k >= 3"))
def test_lincon(case: OracleCase) -> None:
    by_name = _pairs_by_name(case)
    m = len(by_name)
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        what = f"lincon {g1}-{g2}"
        assert row["test"] == "Yuen's trimmed means"
        assert number(row["conf.level"]) == conf_level(case)
        pair = by_name[(g1, g2)]  # rows keep lincon's order: group1 is the earlier
        assert_close(
            "closed_form", pair.estimate, number(row["estimate"]), what=f"{what} est"
        )
        assert_close(
            "closed_form", pair.conf_low, number(row["conf.low"]), what=f"{what} low"
        )
        assert_close(
            "closed_form", pair.conf_high, number(row["conf.high"]), what=f"{what} hi"
        )
        # R's raw p is 2 (1 - pt), rounded to 2^-52 (inferplot: the accurate tail,
        # D43); the adjustment multiplies that step by at most m.
        assert_close(
            "wrs2_p",
            pair.p_adjusted,
            number(row["p.value"]),
            what=f"{what} adjusted p",
            atol_scale=2.0 * m,
        )


@pytest.mark.parametrize("case", params(MANY, "lincon critical values, k >= 3"))
def test_smm_entries_pinned_by_the_fixtures(case: OracleCase) -> None:
    # Each row's CI half-width over its Yuen SE is the critical value R used; the
    # table interpolation must reproduce it.
    by_name = _pairs_by_name(case)
    n_comparisons = len(by_name)
    for row in case.pairwise:
        pair = by_name[(str(row["group1"]), str(row["group2"]))]
        implied = (number(row["conf.high"]) - number(row["conf.low"])) / (2 * pair.se)
        critical = smm_critical_value(n_comparisons, pair.df)
        assert critical == pair.critical_value
        assert_close("closed_form", critical, implied, what=f"{case.name} c")
        assert 2.0 <= pair.df <= SMM_ROWS[-1]


@pytest.mark.parametrize("case", params(ROBUST, "robust centrality labels"))
def test_centrality_trimmed_means(case: OracleCase) -> None:
    layers = case.layer_data("GeomLabelRepel")
    assert layers, case.name
    by_level = dict(zip(case.analysis_levels, groups(case), strict=True))
    y = case.args["y"]
    for row in layers[0]:
        level = r_level_label(row[case.args["x"]])
        values = by_level[level]
        assert int(row["n.obs"]) == values.size
        assert_close(
            "closed_form",
            trimmed_mean(values),
            number(row[y]),
            what=f"{case.name} {level} trimmed mean",
        )
        assert "trimmed" in str(row["expression"])
        assert math.isfinite(number(row[y]))
