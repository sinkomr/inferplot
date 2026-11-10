"""``ip.analyze.betweenstats`` against every nonparametric, robust and bayes
ggbetweenstats fixture.

Each fixture of these types is replayed as the inferplot call
``tests/oracle_args.translate`` builds from its R arguments, on the fixture's CSV read
as pandas, polars and pyarrow, so a new fixture is tested without new code. Numbers
are compared under their ``tests/tolerances.py`` kinds; texts as the plotmath ASTs
ggstatsplot drew (``tests/plotmath.py``), rendered with inferplot's own renderers.

Bootstrap-derived numbers follow SPEC.md section 10's two-part rule: inferplot's
value is within the bootstrap tolerance of the fixture's 20,000-resample reference,
and inferplot's formatter applied to ggstatsplot's own (100-resample) value
reproduces ggstatsplot's string. Every other number in a text is inferplot's own,
compared as the exact string.

Where inferplot refuses a call ggstatsplot answers, by design, the refusal is
checked and the replay SPEC.md prescribes is compared instead
(``tests/betweenstats_cases.py``).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from scipy import stats as scipy_stats
from tools.reference import stats_reference as ref

import inferplot as ip
from inferplot import stats
from inferplot import text as tx
from inferplot.result import EffectSize, StatsResult
from inferplot.stats.density import GRID_POINTS
from tests.betweenstats_cases import (
    BY_NAME,
    CASES,
    FAILED,
    REFUSED,
    call_arguments,
    refusal,
    replay_arguments,
)
from tests.oracle_fixtures import (
    SIMULATED_LINCON_CRITICAL_VALUES,
    Backend,
    OracleCase,
    check_simulated_lincon_pair,
    expected_bf10,
    number,
    r_level_label,
    read_dataset,
    wrs2_akp_constant,
)
from tests.plotmath import parse
from tests.tolerances import (
    assert_close,
    assert_within_grid,
    assert_within_monte_carlo,
    is_close,
)

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
_METHODS = {
    "mann_whitney": "Wilcoxon rank sum test",
    "kruskal_wallis": "Kruskal-Wallis rank sum test",
    "yuen_t": "Yuen's test on trimmed means for independent samples",
    "trimmed_means_anova": "A heteroscedastic one-way ANOVA for trimmed means",
}
_EFFECT_SIZES = {
    "rank_biserial": "r (rank biserial)",
    "rank_epsilon_squared": "Epsilon2 (rank)",
    "akp_delta": "Algina-Keselman-Penfield robust standardized difference",
    "explanatory_xi": "Explanatory measure of effect size",
}
_CENTRALITY = {
    "nonparametric": "median",
    "robust": "trimmed_mean",
    "bayes": "map",
}
_PAIR_TESTS = {"nonparametric": "Dunn", "robust": "Yuen's trimmed means"}
# WRS2's xi constant is measured at these trimming levels (tolerance
# wrs2_xi_constant); an equal-size fixture at another tr needs a new measurement.
_XI_MEASURED_TR = (0.1, 0.2)

# Subtitles whose p differs from ggstatsplot's because WRS2 computes it as 1 - cdf
# and inferplot reports the accurate tail (DECISIONS.md D43): (case, ggstatsplot's
# p, inferplot's p) as printed. iris Petal.Length robust: t1way's 1 - pf is exactly
# 0, which statsExpressions prints as the smallest double; the accurate tail is
# 1.36e-42. iris without setosa, Yuen: R's 2 (1 - pt) is 5.77e-15 and 4.44e-16
# (multiples of 2^-53), the accurate tails 5.93e-15 and 5.43e-16.
DIFFERING_SUBTITLES: tuple[tuple[str, str, str], ...] = (
    ("ggbetweenstats-iris-petal-robust", "2.23e-308", "1.36e-42"),
    ("ggbetweenstats-iris_no_setosa-petal-robust", "5.77e-15", "5.93e-15"),
    ("ggbetweenstats-iris_no_setosa-petalwidth-robust", "4.44e-16", "5.43e-16"),
)

# Bracket labels that differ from ggstatsplot's: (case, group1, group2,
# ggstatsplot's number, inferplot's number). iris Petal.Length robust: lincon's
# Holm-adjusted p after WRS2's 1 - pt, exactly 0 for the two setosa pairs and
# 5.77e-15 for the third (D43). iris Petal.Length bayes: BayesFactor misses the
# accurate BF10 by 1.16e-3 relative for setosa-versicolor (D50,
# oracle_fixtures.ACCURATE_BAYES_FACTORS), which moves log BF01 across a rounding
# boundary (ggstatsplot -133.6872, accurate -133.6862).
DIFFERING_LABELS: tuple[tuple[str, str, str, str, str], ...] = (
    ("ggbetweenstats-iris-petal-bayes", "setosa", "versicolor", "-133.69", "-133.68"),
    ("ggbetweenstats-iris-petal-robust", "setosa", "versicolor", "0.00", "7.28e-30"),
    ("ggbetweenstats-iris-petal-robust", "setosa", "virginica", "0.00", "9.12e-32"),
    (
        "ggbetweenstats-iris-petal-robust",
        "versicolor",
        "virginica",
        "5.77e-15",
        "5.93e-15",
    ),
)


# --------------------------------------------------------------------------- #
# the call and the refusals
# --------------------------------------------------------------------------- #


@functools.cache
def analyze(name: str, backend: Backend) -> StatsResult:
    """The fixture's call, or its replay when inferplot refuses the call."""
    case = BY_NAME[name]
    args, kwargs = replay_arguments(case)
    return ip.analyze.betweenstats(read_dataset(case, backend), *args, **kwargs)


def _params(cases: Any) -> list[Any]:
    return [
        pytest.param(case.name, backend, id=f"{case.name}-{backend}")
        for case in cases
        for backend in BACKENDS
    ]


def _names(cases: Any) -> list[Any]:
    return [pytest.param(case.name, id=case.name) for case in cases]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def _conf_level(case: OracleCase) -> float:
    return float(case.args.get("conf.level", 0.95))


def _groups(case: OracleCase) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    frame = read_dataset(case)
    x, y = str(case.args["x"]), str(case.args["y"])
    used = frame[[x, y]].dropna()
    labels = used[x].astype(object).map(r_level_label)
    return [
        used.loc[labels == level, y].to_numpy(dtype=np.float64)
        for level in case.analysis_levels
    ]


def _label_rows(case: OracleCase, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    x = str(case.args["x"])
    return {r_level_label(row[x]): row for row in rows}


def _centrality_layers(case: OracleCase) -> list[dict[str, Any]]:
    layers = [
        _label_rows(case, rows)
        for geom in ("GeomPoint", "GeomLabelRepel")
        for rows in case.layer_data(geom)
    ]
    assert len(layers) == 2, f"{case.name}: expected two centrality layers"
    return layers


def _drawn(case: OracleCase) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for params in case.layer_params("GeomSignif"):
        out.extend((str(a), str(b)) for a, b in params["stat"]["comparisons"])
    return out


def _drawn_annotations(case: OracleCase) -> list[str]:
    out: list[str] = []
    for params in case.layer_params("GeomSignif"):
        annotations = params["stat"]["annotations"]
        if isinstance(annotations, str):  # R's JSON writer unboxes length one
            annotations = [annotations]
        out.extend(str(a) for a in annotations)
    return out


def _csv_dropped(case: OracleCase) -> int:
    frame = read_dataset(case, "pandas")
    used = frame[[str(case.args["x"]), str(case.args["y"])]]
    return int(used.isna().any(axis=1).sum())


def _random_estimate(case: OracleCase) -> bool:
    """Whether the effect-size estimate itself depends on the resamples (t1way's
    xi with unequal group sizes is the median of the bootstrap values)."""
    reference = case.bootstrap_reference
    return reference is not None and number(reference["replicates"]["estimate_sd"]) > 0


def _with_r_bootstrap(
    case: OracleCase, es: EffectSize, row: Mapping[str, Any]
) -> EffectSize:
    """``es`` with ggstatsplot's own values (``row``, its subtitle table) of the
    bootstrap-derived numbers."""
    if case.bootstrap_reference is None:
        return es
    changes: dict[str, float] = {
        "conf_low": number(row["conf.low"]),
        "conf_high": number(row["conf.high"]),
    }
    if _random_estimate(case):
        changes["estimate"] = number(row["estimate"])
    return dataclasses.replace(es, **changes)


def _pair_rows(case: OracleCase, result: StatsResult) -> list[Mapping[str, Any]]:
    """ggstatsplot's pairwise rows, in inferplot's order (they must agree)."""
    rows = case.pairwise
    assert [(p.group1, p.group2) for p in result.pairwise] == [
        (str(r["group1"]), str(r["group2"])) for r in rows
    ]
    return rows


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_refused_calls_raise() -> None:
    # One test over all refused fixtures (none yet among the Phase 0 cases), so an
    # empty selection is not an empty parametrization.
    for case in REFUSED:
        found = refusal(case)
        assert found is not None
        args, kwargs = call_arguments(case)
        with pytest.raises(found.error, match=found.match):
            ip.analyze.betweenstats(read_dataset(case), *args, **kwargs)


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_levels_and_counts(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    assert result.analysis == "betweenstats"
    assert result.type == case.type
    assert result.levels == case.analysis_levels
    assert {d.level: d.n for d in result.descriptives} == case.n_per_level
    assert [len(v) for v in result.values] == [d.n for d in result.descriptives]
    assert result.n_obs == sum(case.n_per_level.values())
    assert result.n_dropped == _csv_dropped(case)


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_centrality(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    y = str(case.args["y"])
    groups = dict(zip(case.analysis_levels, _groups(case), strict=True))
    for layer in _centrality_layers(case):
        assert list(layer) == list(result.levels)
        for d in result.descriptives:
            row = layer[d.level]
            what = f"{name} {d.centrality} {d.level}"
            assert d.centrality == _CENTRALITY[case.type]
            assert int(row["n.obs"]) == d.n
            expected = number(row[y])
            assert d.value is not None
            if d.centrality == "map":
                values = groups[d.level]
                low, high = float(values.min()), float(values.max())
                step = (high - low) / (GRID_POINTS - 1)
                assert_within_grid(d.value, expected, origin=low, step=step, what=what)
            else:
                assert_close("closed_form", d.value, expected, what=what)


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_main_test(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    test = result.test
    if not case.subtitle_rows:  # results.subtitle = FALSE: R reports nothing
        assert case.args.get("results.subtitle") is False
        assert result.text.subtitle is None
        return
    if case.type == "bayes":
        assert test is None
        return
    assert test is not None
    row = case.subtitle_data
    assert test.method == row["method"] == _METHODS[test.name]
    alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
    assert test.alternative == alternative
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="stat")
    if test.name in ("yuen_t", "trimmed_means_anova"):
        # R's p is 1 - cdf, rounded to 2^-53 per tail; inferplot's is the accurate
        # tail (D43), within one rounding step (twice that for Yuen's two tails).
        scale = 2.0 if test.name == "yuen_t" else 1.0
        assert_close(
            "wrs2_p", test.p_value, number(row["p.value"]), what="p", atol_scale=scale
        )
    else:
        assert_close("closed_form", test.p_value, number(row["p.value"]), what="p")
    for field, column in (("df", "df"), ("df_error", "df.error")):
        value = getattr(test, field)
        if column in row and row[column] is not None:
            assert value is not None
            assert_close("closed_form", value, number(row[column]), what=column)
        else:
            assert value is None, field


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_effect_size(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    es = result.effect_size
    if not case.subtitle_rows or case.type == "bayes":
        return
    assert es is not None
    row = case.subtitle_data
    assert row["effectsize"] == _EFFECT_SIZES[es.name]
    assert es.conf_level == number(row["conf.level"]) == _conf_level(case)
    reference = case.bootstrap_reference
    if reference is None:  # closed form: rank-biserial r
        assert es.bootstrap_resamples is None
        assert es.bootstrap_seed is None
        for field, column in (
            ("estimate", "estimate"),
            ("conf_low", "conf.low"),
            ("conf_high", "conf.high"),
        ):
            expected = number(row[column])
            assert_close("closed_form", getattr(es, field), expected, what=column)
        alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
        assert es.sidedness == alternative
        return
    assert es.ci_method == "percentile bootstrap"
    assert es.bootstrap_resamples == stats.BOOTSTRAP_RESAMPLES
    assert es.bootstrap_seed == result.settings.seed == result.provenance.seed
    assert number(reference["conf_level"]) == es.conf_level
    replicates = reference["replicates"]
    if _random_estimate(case):
        assert_within_monte_carlo(
            es.estimate,
            number(reference["estimate"]),
            number(replicates["estimate_sd"]),
            what=f"{name} estimate",
        )
    elif es.name == "akp_delta":
        # WRS2's AKP constant at this tr, measured by probe-akp, in place of the
        # exact one inferplot uses (D42); the rest agrees to rounding.
        trim = result.settings.tr
        as_wrs2 = (
            es.estimate
            * wrs2_akp_constant(trim)
            / math.sqrt(stats.winsorized_normal_variance(trim))
        )
        assert_close("closed_form", as_wrs2, number(row["estimate"]), what="AKP")
    else:
        kind = "closed_form"
        if es.name == "explanatory_xi":
            assert result.settings.tr in _XI_MEASURED_TR, name
            kind = "wrs2_xi_constant"
        assert_close(kind, es.estimate, number(row["estimate"]), what="estimate")
    for bound in ("conf_low", "conf_high"):
        expected = number(reference[bound])
        mc_sd = number(replicates[f"{bound}_sd"])
        if es.name == "akp_delta" and mc_sd == 0 and math.isfinite(expected):
            # A bound every replicate run agrees on is one resample's AKP, which
            # differs from R's by WRS2's constant as the estimate does (D42;
            # edge_akp_undefined2's lower bound).
            trim = result.settings.tr
            scale = wrs2_akp_constant(trim) / math.sqrt(
                stats.winsorized_normal_variance(trim)
            )
            assert number(replicates[f"{bound}_mean"]) == expected
            assert_close(
                "closed_form", getattr(es, bound) * scale, expected, what=bound
            )
            continue
        assert_within_monte_carlo(
            getattr(es, bound), expected, mc_sd, what=f"{name} {bound}"
        )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_bayes_factor(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    if case.type != "bayes" or len(result.levels) >= 3 or not case.subtitle_rows:
        assert result.bayes is None
        return
    row = case.subtitle_data
    assert result.bayes is not None
    assert result.bayes.prior == row["prior.distribution"] == "cauchy"
    assert result.bayes.prior_scale == number(row["prior.scale"])
    assert_close(
        "ttest_bayes_factor", result.bayes.bf10, number(row["bf10"]), what="bf10"
    )
    assert_close(
        "ttest_bayes_factor",
        result.bayes.log_bf01,
        -number(row["log_e_bf10"]),
        what="log_bf01",
    )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_pairwise_rows(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    k = len(result.levels)
    if k == 2:
        assert result.pairwise == ()
        return
    assert len(result.pairwise) == k * (k - 1) // 2
    if case.args.get("pairwise.display") == "none":
        # ggstatsplot skips the comparisons; inferplot computes them, draws none.
        assert case.pairwise == []
        assert not any(pair.displayed for pair in result.pairwise)
        return
    rows = _pair_rows(case, result)
    m = len(rows)
    order = {level: i for i, level in enumerate(result.levels)}
    for pair, row in zip(result.pairwise, rows, strict=True):
        what = f"{name} {pair.group1}-{pair.group2}"
        assert order[pair.group1] < order[pair.group2]
        assert pair.test_label == row["test"]
        if case.type == "bayes":
            assert row["test"] == "Student's t"
            assert pair.bayes is not None
            assert pair.p_value is None
            assert pair.p_adjusted is None
            assert pair.significant is None
            assert pair.bayes.prior_scale == number(row["prior.scale"]) == 0.707
            expected = expected_bf10(
                name, pair.group1, pair.group2, number(row["bf10"])
            )
            assert_close("ttest_bayes_factor", pair.bayes.bf10, expected, what=what)
            continue
        method = str(case.args.get("p.adjust.method", "holm"))
        assert pair.adjust_method == method
        assert row["p.adjust.method"] == tx.p_adjust_label(method)
        assert pair.p_adjusted is not None
        assert pair.significant == (pair.p_adjusted < 0.05)
        if case.type == "nonparametric":
            assert row["distribution"] == "z"
            assert pair.statistic_name == "z"
            assert_close(
                "closed_form", pair.statistic, number(row["statistic"]), what=what
            )
            assert_close(
                "closed_form", pair.p_adjusted, number(row["p.value"]), what=what
            )
        else:
            # lincon's intervals are 95% whatever conf_level says (D53).
            assert number(row["conf.level"]) == pair.conf_level == 0.95
            assert pair.estimate is not None
            assert_close(
                "closed_form", pair.estimate, number(row["estimate"]), what=what
            )
            if (name, pair.group1, pair.group2) in SIMULATED_LINCON_CRITICAL_VALUES:
                # Past lincon's table WRS2 simulates the critical value; inferplot
                # computes the exact quantile (D52). Both are pinned.
                _check_simulated_interval(name, pair, row)
            else:
                for field, column in (
                    ("conf_low", "conf.low"),
                    ("conf_high", "conf.high"),
                ):
                    value = getattr(pair, field)
                    assert value is not None
                    expected = number(row[column])
                    assert_close(
                        "closed_form", value, expected, what=f"{what} {column}"
                    )
            # R's raw p is 2 (1 - pt), rounded to 2^-52 (D43); the adjustment
            # multiplies that step by at most m.
            assert_close(
                "wrs2_p",
                pair.p_adjusted,
                number(row["p.value"]),
                what=what,
                atol_scale=2.0 * m,
            )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_pairwise_raw_p_and_df(name: str, backend: Backend) -> None:
    """The pairwise fields ggstatsplot's rows do not carry, by independent routes.

    ggstatsplot reports only the adjusted p of a pair, so ``p_value`` (unadjusted)
    and lincon's ``statistic`` and ``df`` have no column to be compared with. Dunn:
    the unadjusted p is the two-sided normal tail of R's own z. lincon: a pair is
    Yuen's test on its two groups, which SciPy implements on its own
    (``ttest_ind(trim=)``). Over the fixtures these agree to 3e-13 relative.
    """
    case, result = BY_NAME[name], analyze(name, backend)
    if case.type == "bayes" or not result.pairwise:
        return
    groups = dict(zip(case.analysis_levels, _groups(case), strict=True))
    r_rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    unadjusted = result.settings.p_adjust_method == "none"
    for pair in result.pairwise:
        what = f"{name} {pair.group1}-{pair.group2}"
        assert pair.p_value is not None
        assert pair.p_adjusted is not None
        assert pair.p_value <= pair.p_adjusted
        if unadjusted:
            assert pair.p_value == pair.p_adjusted
        if case.type == "nonparametric":
            assert pair.df is None
            if not r_rows:  # pairwise.display = "none": R computed no pairs
                continue
            z = number(r_rows[(pair.group1, pair.group2)]["statistic"])
            expected = 2.0 * float(scipy_stats.norm.sf(abs(z)))
            assert_close("closed_form", pair.p_value, expected, what=f"{what} raw p")
        elif case.type == "robust":
            yuen = scipy_stats.ttest_ind(
                groups[pair.group1],
                groups[pair.group2],
                equal_var=False,
                trim=result.settings.tr,
            )
            assert pair.df is not None
            assert_close(
                "closed_form", pair.statistic, float(yuen.statistic), what=f"{what} t"
            )
            assert_close("closed_form", pair.df, float(yuen.df), what=f"{what} df")
            assert_close(
                "closed_form", pair.p_value, float(yuen.pvalue), what=f"{what} raw p"
            )


def _check_simulated_interval(name: str, pair: Any, row: Mapping[str, Any]) -> None:
    """A pinned pair: R's and inferplot's critical values as pinned, each recovered
    from its own interval with the pair's Yuen SE (|estimate| / |t|)."""
    assert pair.statistic != 0.0
    se = abs(pair.estimate / pair.statistic)
    check_simulated_lincon_pair(
        name,
        pair.group1,
        pair.group2,
        r_critical=(number(row["conf.high"]) - number(row["conf.low"])) / (2 * se),
        critical=(pair.conf_high - pair.conf_low) / (2 * se),
    )
    assert_close(
        "closed_form",
        pair.conf_low + pair.conf_high,
        2 * pair.estimate,
        what=f"{name} {pair.group1}-{pair.group2} centre",
    )


# The warnings these fixtures add beyond the general ones (empty levels, arguments
# without effect, the v0.2 posterior): D53's 95% pairwise intervals and D54's tied
# resamples, each in exactly the case that needs it.
EXPECTED_METHOD_WARNINGS: Mapping[str, str] = {
    "ggbetweenstats-PlantGrowth-cl90-robust": (
        "The pairwise confidence intervals are 95% intervals, not conf_level=0.9, "
        "which applies to the effect size: ggstatsplot does not pass conf.level to "
        "the pairwise comparisons (WRS2's lincon), and inferplot follows it. Each "
        "pair records its conf_level."
    ),
    "ggbetweenstats-edge_ties_small-nonparametric": (
        "8 of 2,000 bootstrap resamples have every value tied, where rank epsilon "
        "squared is undefined; its confidence interval comes from the other 1,992, "
        "as in ggstatsplot."
    ),
}


def test_method_warnings() -> None:
    markers = ("The pairwise confidence intervals are", "bootstrap resamples have")
    found = {
        case.name: warning
        for case in CASES
        for warning in analyze(case.name, "pandas").warnings
        if any(marker in warning for marker in markers)
    }
    assert found == EXPECTED_METHOD_WARNINGS
    tied = analyze("ggbetweenstats-edge_ties_small-nonparametric", "pandas")
    assert tied.effect_size is not None
    assert tied.effect_size.bootstrap_resamples == stats.BOOTSTRAP_RESAMPLES


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_drawn_pairs(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == _drawn(case)


# --------------------------------------------------------------------------- #
# texts
# --------------------------------------------------------------------------- #


def _expected_subtitle(case: OracleCase, result: StatsResult) -> tx.Node | None:
    """inferplot's subtitle AST, which must equal ggstatsplot's once ggstatsplot's
    own bootstrap values are put in (checked separately in ``test_subtitle``)."""
    digits = _digits(case)
    if not result.settings.results_subtitle:
        return None
    if result.type == "bayes":
        if result.bayes is None:
            return None
        return tx.bayes_factor_expression(
            log_bf01=result.bayes.log_bf01,
            prior=result.bayes.prior,
            prior_scale=result.bayes.prior_scale,
            digits=digits,
        )
    if result.test is None or result.effect_size is None:
        return None
    return tx.subtitle_expression(
        result.test, result.effect_size, n_obs=result.n_obs, digits=digits
    )


def _v01(node: tx.Node) -> tx.Node:
    """The first and last items of a Bayes expression (D32, D41)."""
    assert isinstance(node, tx.Call)
    assert node.fn == tx.Sym("list")
    return tx.Call(tx.Sym("list"), (node.args[0], node.args[-1]))


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_subtitle(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    figure = tx.figure_texts(result)
    ours = _expected_subtitle(case, result)
    if ours is None:
        assert result.text.subtitle is None
        assert figure.subtitle is None
        assert case.subtitle is None or (
            case.type == "bayes" and len(result.levels) >= 3
        )
        return
    assert result.text.subtitle == tx.to_plain(ours)
    assert figure.subtitle == tx.to_mathtext(ours)
    assert case.subtitle is not None
    theirs = parse(case.subtitle)
    if case.type == "bayes":
        assert ours == _v01(theirs)
        return
    # Every string from inferplot's own numbers, except the bootstrap-derived ones,
    # which are inferplot's formatter applied to ggstatsplot's values; a pinned p
    # (DIFFERING_SUBTITLES) is checked by _subtitle_difference.
    if _subtitle_difference(case, result) is None:
        assert _mixed_subtitle(case, result, case.subtitle_data) == theirs


def _subtitle_difference(
    case: OracleCase, result: StatsResult
) -> tuple[str, str, str] | None:
    """(case, ggstatsplot's p, inferplot's p) where the subtitle differs only in p,
    inferplot's p being within ``wrs2_p`` of ggstatsplot's; None where the
    subtitle is ggstatsplot's."""
    if result.test is None or not case.subtitle_rows or case.type == "bayes":
        return None
    row = case.subtitle_data
    theirs = parse(str(case.subtitle))
    if _mixed_subtitle(case, result, row) == theirs:
        return None
    r_p = number(row["p.value"])
    with_r_p = dataclasses.replace(
        result, test=dataclasses.replace(result.test, p_value=r_p)
    )
    assert _mixed_subtitle(case, with_r_p, row) == theirs, case.name
    scale = 2.0 if result.test.name == "yuen_t" else 1.0
    assert is_close("wrs2_p", result.test.p_value, r_p, atol_scale=scale), case.name
    digits = _digits(case)
    return (
        case.name,
        tx.format_value(r_p, digits),
        tx.format_value(result.test.p_value, digits),
    )


def test_differing_subtitles_are_pinned() -> None:
    differing = [
        found
        for case in CASES
        if (found := _subtitle_difference(case, analyze(case.name, "pandas")))
        is not None
    ]
    assert tuple(differing) == DIFFERING_SUBTITLES


def _mixed_subtitle(
    case: OracleCase, result: StatsResult, row: Mapping[str, Any]
) -> tx.Node:
    assert result.test is not None
    assert result.effect_size is not None
    return tx.subtitle_expression(
        result.test,
        _with_r_bootstrap(case, result.effect_size, row),
        n_obs=result.n_obs,
        digits=_digits(case),
    )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_caption(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    figure = tx.figure_texts(result)
    assert case.caption is None  # ggstatsplot draws no Bayes caption for these types
    line = tx.pairwise_caption_line(
        result.pairwise,
        display=result.settings.pairwise_display,
        adjust_method=result.settings.p_adjust_method,
        alpha=result.settings.pairwise_alpha,
    )
    expected = () if line is None else (line,)
    assert figure.caption_lines == expected
    assert result.text.caption == (line if line is not None else None)
    if line is None:
        return
    if case.type == "bayes":
        assert line == (
            "Pairwise: Bayes factors (Student's t, JZS prior 0.707), unadjusted; "
            f"showing all {len(result.pairwise)} pairs."
        )
    else:
        assert line.startswith(f"Pairwise: {_PAIR_TESTS[case.type]}, ")


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_centrality_and_tick_labels(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    figure = tx.figure_texts(result)
    for layer in _centrality_layers(case):
        expected = [parse(str(layer[level]["expression"])) for level in result.levels]
        assert list(result.text.centrality_labels) == [tx.to_plain(n) for n in expected]
        assert list(figure.centrality_labels) == [tx.to_mathtext(n) for n in expected]
        ticks = [str(layer[level]["n.expression"]) for level in result.levels]
        assert list(result.text.tick_labels) == ticks
        assert list(figure.tick_labels) == ticks


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_bracket_labels(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    if not case.pairwise:
        return
    _label_check(case, result)


def _label_check(
    case: OracleCase, result: StatsResult
) -> list[tuple[str, str, str, str, str]]:
    """Compare every bracket label with ggstatsplot's; return the differing ones as
    (case, group1, group2, ggstatsplot's number, inferplot's number), each checked
    to be inferplot's formatter on a value within tolerance of ggstatsplot's."""
    figure = tx.figure_texts(result)
    rows = _pair_rows(case, result)
    digits = _digits(case)
    differing: list[tuple[str, str, str, str, str]] = []
    for i, (pair, row) in enumerate(zip(result.pairwise, rows, strict=True)):
        theirs = parse(str(row["expression"]))
        ours = tx.to_plain(theirs)
        if result.text.pairwise_labels[i] == ours:
            assert figure.pairwise_labels[i] == tx.to_mathtext(theirs)
            continue
        if pair.bayes is not None:
            # inferplot's formatter on ggstatsplot's BF reproduces its label; the
            # value inferplot prints is the accurate one (D50).
            r_bf10 = number(row["bf10"])
            assert theirs == tx.pairwise_bf_expression(
                log_bf01=-math.log(r_bf10), digits=digits
            )
            accurate = expected_bf10(case.name, pair.group1, pair.group2, r_bf10)
            assert accurate != r_bf10
            assert_close("ttest_bayes_factor", pair.bayes.bf10, accurate, what="bf10")
            r_text = tx.format_value(-math.log(r_bf10), digits)
            our_text = tx.format_value(pair.bayes.log_bf01, digits)
        else:
            assert pair.p_adjusted is not None
            r_p = number(row["p.value"])
            assert theirs == tx.pairwise_p_expression(
                p_adjusted=r_p, adjust_method=pair.adjust_method, digits=digits
            )
            # Within wrs2_p of R's 1 - pt route (D43), as test_pairwise_rows checks.
            assert case.type == "robust"
            assert is_close("wrs2_p", pair.p_adjusted, r_p, atol_scale=2.0 * len(rows))
            r_text = tx.format_value(r_p, digits)
            our_text = tx.format_value(pair.p_adjusted, digits)
        assert r_text != our_text
        differing.append((case.name, pair.group1, pair.group2, r_text, our_text))
    drawn = [i for i, pair in enumerate(result.pairwise) if pair.displayed]
    annotations = _drawn_annotations(case)
    assert len(annotations) == len(drawn)
    for i, source in zip(drawn, annotations, strict=True):
        assert parse(source) == parse(str(rows[i]["expression"]))
    return differing


def test_differing_labels_are_pinned() -> None:
    differing = [
        item
        for case in CASES
        if case.pairwise
        for item in _label_check(case, analyze(case.name, "pandas"))
    ]
    assert tuple(differing) == DIFFERING_LABELS


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_plain_texts(name: str, backend: Backend) -> None:
    result = analyze(name, backend)
    text = result.text
    assert text.apa.isascii()
    assert text.apa.endswith(".")
    assert len(text.alt_text) <= tx.ALT_TEXT_MAX_CHARS
    word = {"median": "median", "trimmed_mean": "trimmed mean", "map": "MAP"}
    assert word[result.descriptives[0].centrality] in text.alt_text


# --------------------------------------------------------------------------- #
# backends, JSON, row order
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", _names(CASES))
def test_backends_give_identical_results(name: str) -> None:
    first = analyze(name, "pandas").to_json()
    for backend in BACKENDS[1:]:
        assert analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", _names(CASES))
def test_json_round_trip(name: str) -> None:
    result = analyze(name, "pandas")
    assert StatsResult.from_json(result.to_json()) == result
    frame = result.to_frame()
    assert len(frame) == len(result.frame_rows())


# --------------------------------------------------------------------------- #
# cases ggstatsplot fails on
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class _Failed:
    """What inferplot does where ggstatsplot fails: the refusal (a fragment of its
    message, which must give the reason and what to change) and, where another
    call computes what ggstatsplot had made before it failed, that call's extra
    keywords."""

    message: str
    replay: Mapping[str, Any] | None


_FAILED_CASES: dict[str, _Failed] = {
    # Two values per group: WRS2's lincon stops below 2 df, and so does inferplot.
    # R's subtitle (made first) is compared in tests/stats/test_robust_parity.py.
    "ggbetweenstats-edge_n2-robust": _Failed(
        "has fewer than 2 degrees of freedom, where the critical values of WRS2's "
        "lincon table stop",
        None,
    ),
    # R fails while building the table of pairwise Bayes factors. inferplot
    # refuses the Bayesian ANOVA subtitle (v0.2) like any type "bayes" call with
    # three or more groups; without the subtitle it computes the estimates R made
    # and the pairwise Bayes factors R could not.
    "ggbetweenstats-edge_n2-bayes": _Failed(
        "the subtitle ggstatsplot draws here is a Bayesian ANOVA, which arrives in "
        "inferplot 0.2",
        {"results_subtitle": False},
    ),
}


def test_failed_cases() -> None:
    """ggstatsplot fails on these. inferplot refuses each with its own reason
    (listed per case, so a new failed fixture needs a decision here), and where a
    replay computes, its centrality labels equal the partial outputs ggstatsplot
    made first and its pairwise Bayes factors equal the independent reference."""
    assert {case.name for case in FAILED} == set(_FAILED_CASES)
    for case in FAILED:
        _check_failed_case(case)


def _check_failed_case(case: OracleCase) -> None:
    expected = _FAILED_CASES[case.name]
    assert case.expected_error is not None
    assert case.expected_error["component"] == "pairwise_comparisons"
    assert len(case.analysis_levels) >= 3
    args, kwargs = call_arguments(case)
    with pytest.raises(NotImplementedError) as refused:
        ip.analyze.betweenstats(read_dataset(case), *args, **kwargs)
    assert expected.message in str(refused.value)
    if expected.replay is None:
        return
    result = ip.analyze.betweenstats(
        read_dataset(case), *args, **{**kwargs, **expected.replay}
    )
    assert result.text.subtitle is None
    labels = _label_rows(case, case.partial("centrality_description"))
    for i, d in enumerate(result.descriptives):
        label = parse(str(labels[d.level]["expression"]))
        assert result.text.centrality_labels[i] == tx.to_plain(label)
    assert case.type == "bayes"
    groups = dict(zip(case.analysis_levels, _groups(case), strict=True))
    k = len(case.analysis_levels)
    assert len(result.pairwise) == k * (k - 1) // 2
    for pair in result.pairwise:
        a, b = groups[pair.group1], groups[pair.group2]
        student = float(scipy_stats.ttest_ind(a, b, equal_var=True).statistic)
        reference = ref.jzs_bf10_t(
            student, a.size * b.size / (a.size + b.size), float(a.size + b.size - 2)
        )
        assert pair.bayes is not None
        assert_close(
            "ttest_bayes_factor",
            pair.bayes.bf10,
            reference,
            what=f"{case.name} {pair.group1}-{pair.group2} BF10",
        )
