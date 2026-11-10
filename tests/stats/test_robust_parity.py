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
from inferplot.stats.robust import SMM_MAX_TABLED, winsorized_normal_variance
from tests.oracle_fixtures import (
    SIMULATED_LINCON_CRITICAL_VALUES,
    OracleCase,
    Row,
    cases_for,
    check_simulated_lincon_pair,
    number,
    r_level_label,
    require_cases,
    wrs2_akp_constant,
)
from tests.stats.fixture_data import (
    between_cases,
    conf_level,
    groups,
    p_adjust_method,
    params,
    tr,
    value,
)
from tests.tolerances import assert_close, assert_within_monte_carlo

ROBUST = between_cases("robust")
TWO = between_cases("robust", two_groups=True)
MANY = between_cases("robust", two_groups=False)
# Cases ggstatsplot fails on after it has computed the subtitle (edge_n2: lincon
# stops below 2 df). The subtitle it made first is compared like any other.
FAILED_WITH_SUBTITLE = [
    case
    for case in cases_for("ggbetweenstats", failed=True)
    if case.type == "robust" and "subtitle" in case.partial_components
]
# lincon's intervals are 95% whatever conf.level says: ggstatsplot does not pass it
# (PlantGrowth-cl90-robust; DECISIONS.md D53).
LINCON_CONF_LEVEL = 0.95
# The trimming levels at which WRS2's xi constant has been measured (tolerance
# wrs2_xi_constant): an equal-size fixture at another tr needs a new measurement.
XI_MEASURED_TR = (0.1, 0.2)


def _equal_sizes(case: OracleCase) -> bool:
    return len(set(case.n_per_level.values())) == 1


def _has_subtitle(case: OracleCase) -> bool:
    return bool(case.subtitle_rows)


def _has_pairs(case: OracleCase) -> bool:
    return bool(case.pairwise)


def test_every_robust_case_has_the_tables_checked_here() -> None:
    # Every table is there unless the call turns it off: results.subtitle = FALSE
    # leaves out the subtitle (and with it the bootstrap), pairwise.display =
    # "none" the pairwise rows.
    assert len(ROBUST) == len(TWO) + len(MANY) >= 10
    for case in ROBUST:
        with_subtitle = case.args.get("results.subtitle", True) is not False
        assert bool(case.subtitle_rows) == with_subtitle, case.name
        assert (case.bootstrap_reference is not None) == with_subtitle, case.name
        if len(case.analysis_levels) > 2:
            with_pairs = case.args.get("pairwise.display") != "none"
            assert bool(case.pairwise) == with_pairs, case.name


# ------------------------------------------------------------------ two groups


@pytest.mark.parametrize(
    "case", params(TWO, "two-group robust subtitles", where=_has_subtitle)
)
def test_yuen(case: OracleCase) -> None:
    a, b = groups(case)
    row = case.subtitle_data
    assert row["method"] == "Yuen's test on trimmed means for independent samples"
    result = yuen_t(a, b, tr=tr(case))
    # ggstatsplot reports |t|, whatever the sign of the trimmed-mean difference.
    assert_close("closed_form", result.t, value(row, "statistic"), what="Yuen t")
    assert result.t >= 0.0
    assert_close("closed_form", result.df, value(row, "df.error"), what="Yuen df")
    # R rounds 2 (1 - pt(|t|, df)) to 2^-52; inferplot reports the accurate tail
    # (D43), within one such step of it.
    assert_close(
        "wrs2_p", result.p_value, value(row, "p.value"), what="Yuen p", atol_scale=2.0
    )


@pytest.mark.parametrize(
    "case", params(TWO, "two-group robust effect sizes", where=_has_subtitle)
)
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
    trim = tr(case)
    assert f"tr = {trim}" in reference["call"]
    result = akp_effect(a, b, tr=trim, conf_level=level)
    assert result.resamples == BOOTSTRAP_RESAMPLES
    assert result.seed == 0
    # The estimate does not depend on the resamples (replicate SD 0); it differs
    # from WRS2's only by WRS2's scaling constant at this tr, which probe-akp
    # measures (oracle_fixtures.wrs2_akp_constant).
    assert reference["replicates"]["estimate_sd"] == 0
    as_wrs2 = (
        result.estimate
        * wrs2_akp_constant(trim)
        / math.sqrt(winsorized_normal_variance(trim))
    )
    assert_close("closed_form", as_wrs2, value(row, "estimate"), what="AKP")
    assert_close(
        "closed_form", as_wrs2, number(reference["estimate"]), what="AKP (reference)"
    )
    _check_bootstrap_bounds(
        case,
        result.conf_low,
        result.conf_high,
        fixed_kind="closed_form",
        fixed_scale=as_wrs2 / result.estimate if result.estimate else None,
    )


def _check_bootstrap_bounds(
    case: OracleCase,
    low: float,
    high: float,
    *,
    fixed_kind: str | None = None,
    fixed_scale: float | None = None,
) -> None:
    """The bounds against the 20,000-resample reference (SPEC.md section 10).

    ``fixed_kind``: the tolerance kind for a bound that R's 50 replicate runs all
    agree on (replicate SD 0). Such a bound does not depend on the resamples; it
    is one value of the statistic, and differs from R's the way the estimate does
    (edge_n2's upper bound of xi: the largest value three groups of two can give,
    1 - 3.2404e-8 times inferplot's, WRS2's constant at tr 0.2). ``fixed_scale``
    multiplies such a bound first: WRS2's AKP constant over the exact one (D42), as
    for the estimate (edge_akp_undefined2's lower bound, the same resample value in
    every run).
    """
    reference = case.bootstrap_reference
    assert reference is not None
    replicates = reference["replicates"]
    assert replicates["resamples"] == BOOTSTRAP_RESAMPLES
    for bound, name in ((low, "conf_low"), (high, "conf_high")):
        expected = number(reference[name])
        mc_sd = number(replicates[f"{name}_sd"])
        what = f"{case.name} {name}"
        if fixed_kind is not None and mc_sd == 0 and math.isfinite(expected):
            assert number(replicates[f"{name}_mean"]) == expected, what
            scaled = bound if fixed_scale is None else bound * fixed_scale
            assert_close(fixed_kind, scaled, expected, what=what)
        else:
            assert_within_monte_carlo(bound, expected, mc_sd, what=what)


# ------------------------------------------------------------------ k >= 3


@pytest.mark.parametrize(
    "case", params(MANY, "robust subtitles, k >= 3", where=_has_subtitle)
)
def test_trimmed_means_anova(case: OracleCase) -> None:
    _check_trimmed_means_anova(case, case.subtitle_data)


def _check_trimmed_means_anova(case: OracleCase, row: Row) -> None:
    assert row["method"] == "A heteroscedastic one-way ANOVA for trimmed means"
    result = trimmed_means_anova(groups(case), tr=tr(case))
    assert_close("closed_form", result.f, value(row, "statistic"), what="F")
    assert result.df1 == value(row, "df")
    assert_close("closed_form", result.df2, value(row, "df.error"), what="df2")
    # R rounds 1 - pf(F, df1, df2) to 2^-53; inferplot reports the accurate tail
    # (D43), within one such step of it.
    assert_close("wrs2_p", result.p_value, value(row, "p.value"), what="t1way p")


@pytest.mark.parametrize(
    "case", params(MANY, "robust effect sizes, k >= 3", where=_has_subtitle)
)
def test_explanatory_effect(case: OracleCase) -> None:
    _check_explanatory_effect(case, case.subtitle_data)


def _check_explanatory_effect(case: OracleCase, row: Row) -> None:
    level = conf_level(case)
    assert row["effectsize"] == "Explanatory measure of effect size"
    reference = case.bootstrap_reference
    assert reference is not None
    assert reference["conf_level"] == level
    trim = tr(case)
    assert f"tr = {trim}," in reference["call"]
    result = explanatory_effect(groups(case), tr=trim, conf_level=level)
    assert result.resamples == BOOTSTRAP_RESAMPLES
    replicates = reference["replicates"]
    if _equal_sizes(case):
        # Equal group sizes: the estimate does not depend on the resamples and
        # differs from WRS2's only by WRS2's constant at this tr (tolerances.py).
        assert trim in XI_MEASURED_TR, case.name
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
    _check_bootstrap_bounds(
        case,
        result.conf_low,
        result.conf_high,
        fixed_kind="wrs2_xi_constant" if _equal_sizes(case) else None,
    )


def test_subtitles_of_cases_ggstatsplot_fails_on() -> None:
    """ggstatsplot computed these subtitles before it failed at the pairwise step,
    so the omnibus test and its effect size have R values although the case as a
    whole has none (edge_n2: two values per group, Yuen df below 2)."""
    cases = require_cases(FAILED_WITH_SUBTITLE, "failed robust cases with a subtitle")
    for case in cases:
        assert len(case.analysis_levels) > 2, case.name
        (row,) = case.partial("subtitle")
        _check_trimmed_means_anova(case, row)
        _check_explanatory_effect(case, row)


def _pairs_by_name(case: OracleCase) -> dict[tuple[str, str], LinconPair]:
    levels = case.analysis_levels
    pairs = lincon(
        groups(case),
        tr=tr(case),
        conf_level=LINCON_CONF_LEVEL,
        p_adjust_method=p_adjust_method(case),
    )
    assert len(pairs) == len(levels) * (len(levels) - 1) // 2 == len(case.pairwise)
    return {(levels[p.i], levels[p.j]): p for p in pairs}


@pytest.mark.parametrize(
    "case", params(MANY, "robust pairwise rows, k >= 3", where=_has_pairs)
)
def test_lincon(case: OracleCase) -> None:
    by_name = _pairs_by_name(case)
    m = len(by_name)
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        what = f"lincon {g1}-{g2}"
        assert row["test"] == "Yuen's trimmed means"
        assert number(row["conf.level"]) == LINCON_CONF_LEVEL
        pair = by_name[(g1, g2)]  # rows keep lincon's order: group1 is the earlier
        assert_close(
            "closed_form", pair.estimate, number(row["estimate"]), what=f"{what} est"
        )
        if m > SMM_MAX_TABLED:
            # WRS2 simulates the critical value; inferplot's is the exact quantile
            # (D52). The pair is pinned instead (test_smm_entries_...).
            _check_simulated_pair(case, pair, row)
        else:
            assert_close(
                "closed_form",
                pair.conf_low,
                number(row["conf.low"]),
                what=f"{what} low",
            )
            assert_close(
                "closed_form",
                pair.conf_high,
                number(row["conf.high"]),
                what=f"{what} hi",
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


def _implied_critical_value(row: object, pair: LinconPair) -> float:
    """R's critical value: its CI half-width over the pair's Yuen SE."""
    return (number(row["conf.high"]) - number(row["conf.low"])) / (2 * pair.se)  # type: ignore[index]


def _check_simulated_pair(case: OracleCase, pair: LinconPair, row: object) -> None:
    g1, g2 = str(row["group1"]), str(row["group2"])  # type: ignore[index]
    check_simulated_lincon_pair(
        case.name,
        g1,
        g2,
        r_critical=_implied_critical_value(row, pair),
        critical=pair.critical_value,
    )
    half = pair.critical_value * pair.se
    assert_close("closed_form", pair.estimate - half, pair.conf_low, what="low")
    assert_close("closed_form", pair.estimate + half, pair.conf_high, what="high")


@pytest.mark.parametrize(
    "case", params(MANY, "lincon critical values, k >= 3", where=_has_pairs)
)
def test_smm_entries_pinned_by_the_fixtures(case: OracleCase) -> None:
    # Each row's CI half-width over its Yuen SE is the critical value R used; up to
    # 28 comparisons the table interpolation must reproduce it (from df 200 on, the
    # last row: edge_large_n's df are about 1,490).
    by_name = _pairs_by_name(case)
    n_comparisons = len(by_name)
    for row in case.pairwise:
        pair = by_name[(str(row["group1"]), str(row["group2"]))]
        critical = smm_critical_value(n_comparisons, pair.df)
        assert critical == pair.critical_value
        assert pair.df >= 2.0
        if n_comparisons > SMM_MAX_TABLED:
            _check_simulated_pair(case, pair, row)
            continue
        implied = _implied_critical_value(row, pair)
        assert_close("closed_form", critical, implied, what=f"{case.name} c")


def test_simulated_critical_values_are_all_pinned() -> None:
    # Every pair of every case past the table is in the pinned list, and nothing
    # else is: today edge_k9's 36 pairs.
    expected = {
        (case.name, str(row["group1"]), str(row["group2"]))
        for case in MANY
        if len(case.pairwise) > SMM_MAX_TABLED
        for row in case.pairwise
    }
    assert set(SIMULATED_LINCON_CRITICAL_VALUES) == expected
    assert len(expected) == 36


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
            trimmed_mean(values, tr=tr(case)),
            number(row[y]),
            what=f"{case.name} {level} trimmed mean",
        )
        assert "trimmed" in str(row["expression"])
        assert math.isfinite(number(row[y]))
