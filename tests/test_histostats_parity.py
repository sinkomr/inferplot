"""``ip.analyze.histostats`` against every gghistostats fixture, all four types.

Each fixture is replayed through ``ip.analyze.histostats`` on its CSV read as
pandas, polars and pyarrow (``tests/no_level_cases.py``) with the fixture's own
arguments, and every number the fixture holds is compared under its
``tests/tolerances.py`` kind: the test statistic, df, p and method, the effect size
with its interval and level, the Bayes factor and its prior (subtitle of type
bayes, caption of type parametric), the centrality value, n and the rows dropped.
The bootstrap-t bounds and p are compared with the fixture's 20,000-resample
reference within 4 Monte Carlo SDs (``assert_within_monte_carlo``).

Fixtures inferplot answers differently from ggstatsplot, by name:

* ToothGrowth-greater-robust and ToothGrowth-greater-bayes: a one-sided
  ``alternative`` with a type whose test is always two-sided raises (SPEC.md
  section 2); ggstatsplot ignores it, and the call without it gives ggstatsplot's
  output, which the replay is compared with;
* ToothGrowth-tv100 (t = -82.2): R's Hedges' g interval and BayesFactor's Bayes
  factor are measurably off (D81); R's misses are pinned as the fixture holds
  them (``ONE_SAMPLE_R_MISSES``), and inferplot reports the statistics layer's
  accurate values, which tests/stats/test_one_sample.py checks against 40-digit
  references.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from tests import no_level_cases as nl
from tests.oracle_fixtures import OracleCase, number
from tests.test_reference_vs_oracle import ONE_SAMPLE_R_MISSES
from tests.tolerances import (
    assert_close,
    assert_within_grid,
    assert_within_monte_carlo,
    is_close,
)

CASES = nl.HISTO
REFUSED = sorted(n for n in nl.REFUSALS if n.startswith("gghistostats-"))
_METHODS = {
    "one_sample_t": "One Sample t-test",
    "wilcoxon_signed_rank": "Wilcoxon signed rank test",
    "trimmed_mean_bootstrap_t": "Bootstrap-t method for one-sample test",
}
_EFFECT_SIZES = {
    "hedges_g": "Hedges' g",
    "rank_biserial": "r (rank biserial)",
    "trimmed_mean": "Trimmed mean",
}
_CENTRALITY = {
    "parametric": "mean",
    "nonparametric": "median",
    "robust": "trimmed_mean",
    "bayes": "map",
}
# Resamples of 2,000 (seed 0) whose bootstrap t is infinite, by case (kept, as in
# R: edge_one_small's interval is [-Inf, Inf] in R's reference too).
INFINITE_RESAMPLES = {
    "gghistostats-edge_one_small-robust": 169,
    "gghistostats-edge_one_ties-robust": 34,
}


def _result(name: str, backend: nl.Backend) -> StatsResult:
    result = nl.analyzed(name, backend)
    assert result is not None
    return result


def _mu(case: OracleCase) -> float:
    return float(case.args.get("test.value", 0.0))


def _r_side(row: Any) -> str:
    return str(row.get("alternative", "two.sided")).replace(".", "-")


def _close_or_r_miss(
    kind: str, mine: float, oracle: float, case: OracleCase, what: str
) -> None:
    """``assert_close``, or for R's pinned miss (D81) R's relative miss as the
    fixture holds it, within 1% of the recorded one."""
    miss = ONE_SAMPLE_R_MISSES.get((case.name, what))
    if miss is None:
        assert_close(kind, mine, oracle, what=f"{case.name} {what}")
        return
    assert not is_close(kind, mine, oracle), (case.name, what)
    measured = oracle / mine - 1
    assert abs(measured - miss) <= 0.01 * abs(miss), (case.name, what, measured)


def test_fixture_counts() -> None:
    assert len(CASES) == 76
    counts = {
        t: sum(c.type == t for c in CASES)
        for t in ("parametric", "nonparametric", "robust", "bayes")
    }
    assert counts == {"parametric": 22, "nonparametric": 18, "robust": 18, "bayes": 18}
    assert REFUSED == [
        "gghistostats-ToothGrowth-greater-bayes",
        "gghistostats-ToothGrowth-greater-robust",
    ]
    assert not [c.name for c in CASES if c.error is not None]
    # Every answered case on three backends: 74 + 2 replays, times 3.
    assert len(nl.params(CASES)) == 76 * 3


@pytest.mark.parametrize("name", REFUSED)
@pytest.mark.parametrize("backend", nl.BACKENDS)
def test_one_sided_robust_and_bayes_are_refused(name: str, backend: nl.Backend) -> None:
    case = nl.BY_NAME[name]
    refusal = nl.REFUSALS[name]
    with pytest.raises(refusal.error, match=refusal.match):
        nl.analyze(case, nl.frame_of(case, backend))
    # ggstatsplot ignored the argument: its output equals the two-sided case's.
    two_sided = nl.BY_NAME[name.replace("-greater-", "-")]
    assert case.subtitle_data == two_sided.subtitle_data
    assert _result(name, backend).to_json() == _result(two_sided.name, backend).to_json()


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_values_counts_and_design(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    data = case.analyzed_data
    assert (result.analysis, result.design, result.type) == (
        "histostats",
        "one-sample",
        case.type,
    )
    assert (result.x, result.y, result.levels, result.pairwise) == (
        case.args["x"],
        None,
        (),
        (),
    )
    assert result.values == (tuple(case.analyzed_values("x")),)
    assert result.n_obs == data["n"] == len(data["rows"])
    assert result.n_dropped == data["rows_dropped"]
    assert result.n_obs + result.n_dropped == case.raw["dataset"]["rows"]
    assert result.settings.test_value == _mu(case)
    assert result.settings.seed == 0 == result.provenance.seed
    assert result.provenance.citations


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_centrality(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    (descriptive,) = result.descriptives
    assert descriptive.level == case.args["x"]
    assert descriptive.n == result.n_obs
    assert descriptive.centrality == _CENTRALITY[case.type]
    assert descriptive.value is not None
    x = np.array(case.analyzed_values("x"), dtype=np.float64)
    lines = case.centrality
    if lines is None:  # centrality.plotting = FALSE: R computes none
        assert case.args.get("centrality.plotting") is False
        assert_close("closed_form", descriptive.value, stats.mean(x), what="mean")
        return
    (row,) = lines
    assert int(number(row["n.obs"])) == result.n_obs
    expected = number(row["var"])
    if case.type == "bayes":
        found = stats.map_estimate(x)
        assert descriptive.value == found.estimate
        assert_within_grid(
            descriptive.value,
            expected,
            origin=float(x.min()),
            step=found.grid_step,
            what=f"{name} MAP",
        )
        return
    assert_close("closed_form", descriptive.value, expected, what=f"{name} centrality")


def _subtitle_row(case: OracleCase) -> Any | None:
    """The case's subtitle row, or that of the same call with the subtitle (the
    numbers do not depend on it: ToothGrowth-nosubtitle-* against ToothGrowth-*)."""
    if case.subtitle_rows:
        return case.subtitle_data
    assert case.args.get("results.subtitle") is False
    same = nl.BY_NAME[case.name.replace("-nosubtitle-", "-")]
    return same.subtitle_data


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_main_test(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    test = result.test
    if case.type == "bayes":
        assert test is None  # D45
        return
    assert test is not None  # computed also without a subtitle
    row = _subtitle_row(case)
    assert row is not None
    assert test.method == row["method"] == _METHODS[test.name]
    assert int(number(row["n.obs"])) == result.n_obs
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="stat")
    if case.type == "robust":
        assert (test.statistic_name, test.df, test.df_error) == ("t", None, None)
        assert test.alternative == "two-sided"
        reference = case.bootstrap_reference
        if reference is None:  # no subtitle: the subtitle case's reference
            reference = nl.BY_NAME[case.name.replace("-nosubtitle-", "-")]
            reference = reference.bootstrap_reference
        assert reference is not None
        assert_within_monte_carlo(
            test.p_value,
            number(reference["p_value"]),
            number(reference["replicates"]["p_value_sd"]),
            what=f"{name} p",
        )
        return
    assert test.alternative == _r_side(row)
    assert_close("closed_form", test.p_value, number(row["p.value"]), what="p")
    if case.type == "parametric":
        assert (test.statistic_name, test.df) == ("t", None)
        assert number(row["mu"]) == _mu(case)
        assert test.df_error is not None
        assert_close("closed_form", test.df_error, number(row["df.error"]), what="df")
    else:
        assert (test.statistic_name, test.df, test.df_error) == ("V", None, None)


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_effect_size(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    es = result.effect_size
    if case.type == "bayes":
        assert es is None  # D45; the posterior estimate arrives in v0.2
        return
    assert es is not None
    row = _subtitle_row(case)
    assert row is not None
    level = float(case.args.get("conf.level", 0.95))
    assert row["effectsize"] == _EFFECT_SIZES[es.name]
    assert es.conf_level == number(row["conf.level"]) == level
    if case.type == "robust":
        assert (es.ci_method, es.sidedness) == ("bootstrap-t", "two-sided")
        assert (es.bootstrap_resamples, es.bootstrap_seed) == (2000, 0)
        assert es.undefined_resamples == 0
        assert es.infinite_resamples == INFINITE_RESAMPLES.get(
            name.replace("-nosubtitle-", "-"), 0
        )
        assert_close("closed_form", es.estimate, number(row["estimate"]), what="tm")
        reference = case.bootstrap_reference or nl.BY_NAME[
            case.name.replace("-nosubtitle-", "-")
        ].bootstrap_reference
        assert reference is not None
        assert reference["conf_level"] == level
        for value, key in ((es.conf_low, "conf_low"), (es.conf_high, "conf_high")):
            assert_within_monte_carlo(
                value,
                number(reference[key]),
                number(reference["replicates"][f"{key}_sd"]),
                what=f"{name} {key}",
            )
        return
    assert es.bootstrap_resamples is None
    assert es.sidedness == _r_side(row)
    assert_close("closed_form", es.estimate, number(row["estimate"]), what="estimate")
    kind = "noncentral_ci" if case.type == "parametric" else "closed_form"
    for value, column, what in (
        (es.conf_low, "conf.low", "g CI low"),
        (es.conf_high, "conf.high", "g CI high"),
    ):
        _close_or_r_miss(kind, value, number(row[column]), case, what)


@pytest.mark.parametrize(("name", "backend"), nl.params(CASES))
def test_bayes_factor(name: str, backend: nl.Backend) -> None:
    case, result = nl.BY_NAME[name], _result(name, backend)
    bayes = result.bayes
    if case.type == "bayes" and not case.subtitle_rows:
        assert bayes is None  # no subtitle: nothing is computed (as withinstats)
        return
    row = case.subtitle_data if case.type == "bayes" else case.caption_data
    if row is None:
        # No caption: the type has none, or bf.message = FALSE, or no subtitle.
        assert bayes is None
        return
    assert bayes is not None and row is not None
    prior = float(case.args.get("bf.prior", 0.707))
    assert (bayes.prior, bayes.prior_scale, bayes.test) == (
        "cauchy",
        prior,
        "jzs_one_sample",
    )
    assert number(row["prior.scale"]) == prior
    assert row["prior.distribution"] == "cauchy"
    assert math.isclose(bayes.log_bf01, -math.log(bayes.bf10), rel_tol=1e-15)
    what = "BF10" if case.type == "bayes" else "caption BF10"
    _close_or_r_miss("ttest_bayes_factor", bayes.bf10, number(row["bf10"]), case, what)
    # The analysis reports what the statistics layer computes, accurate also
    # where BayesFactor misses (ToothGrowth-tv100, D81).
    x = np.array(case.analyzed_values("x"), dtype=np.float64)
    expected = stats.jzs_bf10_one_sample(x, mu=_mu(case), prior_scale=prior)
    assert (bayes.bf10, bayes.log_bf01) == expected


@pytest.mark.parametrize(
    "name",
    ["gghistostats-ToothGrowth-tv100-parametric", "gghistostats-ToothGrowth-tv100-bayes"],
)
def test_tv100_reports_the_accurate_values(name: str) -> None:
    # D81: the interval and the Bayes factor are the statistics layer's accurate
    # values; R's are off by the pinned misses (tests/test_reference_vs_oracle.py
    # measures them against tools/reference).
    case, result = nl.BY_NAME[name], _result(name, "pandas")
    x = np.array(case.analyzed_values("x"), dtype=np.float64)
    if case.type == "parametric":
        es = result.effect_size
        assert es is not None
        g = stats.one_sample_hedges_g(x, mu=100.0)
        assert (es.estimate, es.conf_low, es.conf_high) == (
            g.estimate,
            g.conf_low,
            g.conf_high,
        )
        row = case.subtitle_data
        for value, column, what in (
            (es.conf_low, "conf.low", "g CI low"),
            (es.conf_high, "conf.high", "g CI high"),
        ):
            miss = ONE_SAMPLE_R_MISSES[(name, what)]
            assert number(row[column]) / value - 1 == pytest.approx(miss, rel=0.01)
    bayes = result.bayes
    assert bayes is not None
    row = case.subtitle_data if case.type == "bayes" else case.caption_data
    assert row is not None
    what = "BF10" if case.type == "bayes" else "caption BF10"
    miss = ONE_SAMPLE_R_MISSES[(name, what)]
    assert number(row["bf10"]) / bayes.bf10 - 1 == pytest.approx(miss, rel=0.01)


def test_results_do_not_depend_on_the_backend() -> None:
    for case in CASES:
        results = {
            backend: nl.analyzed(case.name, backend) for backend in nl.backends(case)
        }
        first = results["pandas"]
        assert first is not None
        for result in results.values():
            assert result is not None
            assert result.to_json() == first.to_json()


def test_infinite_bootstrap_resamples_are_reported() -> None:
    result = _result("gghistostats-edge_one_small-robust", "pandas")
    es = result.effect_size
    assert es is not None
    assert (es.conf_low, es.conf_high) == (-math.inf, math.inf)  # as R's reference
    assert result.warnings == (
        "169 of 2,000 bootstrap resamples give an infinite bootstrap t, because "
        "every winsorized value of the resample is the same number (a standard "
        "error of 0). They are kept, as in WRS2, so the trimmed mean's interval "
        "[-Inf, Inf] can be unbounded. Collect more values, or use "
        "type='nonparametric'.",
    )
    (issue,) = [i for i in result.check() if i.code == "infinite_effect_size"]
    assert "169 of 2,000 bootstrap resamples" in issue.message
    back = StatsResult.from_json(result.to_json())
    assert back == result
    assert back.check() == result.check()


def test_seed_reaches_the_bootstrap_only() -> None:
    case = nl.BY_NAME["gghistostats-ToothGrowth-robust"]
    frame = nl.frame_of(case)
    first = nl.analyze(case, frame)
    again = nl.analyze(case, frame)
    other = nl.analyze(case, frame, seed=7)
    assert first == again
    assert first.effect_size is not None and other.effect_size is not None
    assert first.effect_size.conf_low != other.effect_size.conf_low
    assert other.effect_size.bootstrap_seed == other.settings.seed == 7
    assert first.test is not None and other.test is not None
    assert first.test.statistic == other.test.statistic  # exact, not resampled


def test_every_fixture_is_answered_or_refused_by_name() -> None:
    # The two one-sided refusals are answered by their replay; nothing is skipped.
    answered = {p.values[0] for p in nl.params(CASES)}
    assert answered == {c.name for c in CASES}
    with pytest.raises(ValueError, match="alternative='greater' applies to"):
        ip.analyze.histostats(
            nl.frame_of(nl.BY_NAME[REFUSED[0]]),
            "len",
            type="bayes",
            test_value=18,
            alternative="greater",
        )
