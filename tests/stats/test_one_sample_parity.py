"""inferplot.stats' one-sample methods against every gghistostats fixture.

Each statistic runs on the values the fixture analyzed (``analyzed_data``, rows with
x present in file order) with the fixture's arguments, so a failure here is a method
difference, not a data-reading one. Every number the fixture holds for the
statistics is compared: the test statistic, df, p, the effect size and its interval
and level, the Bayes factor and its prior, n, and the centrality value the line and
label show (also where ggstatsplot draws no subtitle).

R's numbers that inferplot computes more accurately (DECISIONS.md D81) are those of
``ONE_SAMPLE_R_MISSES`` (tests/test_reference_vs_oracle.py): there R's relative
miss is checked against the recorded one, and inferplot's value against an
independent high-precision reference (tests/stats/test_one_sample.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from inferplot.stats import (
    jzs_bf10_one_sample,
    map_estimate,
    mean,
    median,
    one_sample_hedges_g,
    one_sample_rank_biserial,
    one_sample_signed_rank,
    one_sample_t,
    trimmed_mean,
    trimmed_mean_bootstrap_t,
)
from tests.oracle_fixtures import OracleCase, cases_for, number, require_cases
from tests.stats.fixture_data import alternative, bf_prior, conf_level, tr, value
from tests.test_reference_vs_oracle import ONE_SAMPLE_R_MISSES
from tests.tolerances import (
    assert_close,
    assert_within_grid,
    assert_within_monte_carlo,
    is_close,
)

FloatArray = NDArray[np.float64]

CASES = require_cases(cases_for("gghistostats"), "gghistostats fixtures")


# Resamples of 2,000 (seed 0) whose T* is infinite, by case.
INFINITE_RESAMPLES = {
    "gghistostats-edge_one_small-robust": 169,
    "gghistostats-edge_one_ties-robust": 34,
}


def _params(type_: str, *, needs: str = "subtitle") -> list[Any]:
    chosen = [
        c
        for c in CASES
        if c.type == type_
        and (c.subtitle_rows if needs == "subtitle" else c.caption_data is not None)
    ]
    return [pytest.param(c, id=c.name) for c in require_cases(chosen, type_)]


def _x(case: OracleCase) -> FloatArray:
    return np.array(case.analyzed_values("x"), dtype=np.float64)


def _mu(case: OracleCase) -> float:
    return float(case.args.get("test.value", 0.0))


def _close_or_r_miss(
    kind: str, mine: float, oracle: float, case: OracleCase, what: str
) -> None:
    """``assert_close``, or for a pinned R miss (D81) R's recorded relative miss."""
    miss = ONE_SAMPLE_R_MISSES.get((case.name, what))
    if miss is None:
        assert_close(kind, mine, oracle, what=f"{case.name} {what}")
        return
    assert not is_close(kind, mine, oracle)
    measured = oracle / mine - 1
    assert abs(measured - miss) <= 0.01 * abs(miss), (case.name, what, measured)


def test_case_counts() -> None:
    counts = {
        t: sum(c.type == t for c in CASES)
        for t in ("parametric", "nonparametric", "robust", "bayes")
    }
    assert counts == {"parametric": 22, "nonparametric": 18, "robust": 18, "bayes": 18}


@pytest.mark.parametrize("case", _params("parametric"))
def test_one_sample_t_and_hedges_g(case: OracleCase) -> None:
    x, mu = _x(case), _mu(case)
    row = case.subtitle_data
    side = alternative(case)
    level = conf_level(case)
    assert row["method"] == "One Sample t-test"
    assert value(row, "mu") == mu
    assert int(value(row, "n.obs")) == x.size
    result = one_sample_t(x, mu=mu, alternative=side)
    assert_close("closed_form", result.t, value(row, "statistic"), what="t")
    assert_close("closed_form", result.df, value(row, "df.error"), what="df")
    assert_close("closed_form", result.p_value, value(row, "p.value"), what="p")
    g = one_sample_hedges_g(x, mu=mu, conf_level=level, alternative=side)
    assert row["effectsize"] == "Hedges' g"
    assert g.conf_level == value(row, "conf.level")
    assert_close("closed_form", g.estimate, value(row, "estimate"), what="g")
    _close_or_r_miss(
        "noncentral_ci", g.conf_low, value(row, "conf.low"), case, "g CI low"
    )
    _close_or_r_miss(
        "noncentral_ci", g.conf_high, value(row, "conf.high"), case, "g CI high"
    )


def _check_bayes_factor(case: OracleCase, row: Any, what: str) -> None:
    x, mu = _x(case), _mu(case)
    prior = bf_prior(case)
    assert value(row, "prior.scale") == prior
    assert int(value(row, "n.obs")) == x.size
    bf10, log_bf01 = jzs_bf10_one_sample(x, mu=mu, prior_scale=prior)
    assert math.isclose(log_bf01, -math.log(bf10), rel_tol=1e-15, abs_tol=1e-15)
    _close_or_r_miss("ttest_bayes_factor", bf10, value(row, "bf10"), case, what)


@pytest.mark.parametrize("case", _params("parametric", needs="caption"))
def test_one_sample_bayes_factor_caption(case: OracleCase) -> None:
    caption = case.caption_data
    assert caption is not None
    _check_bayes_factor(case, caption, "caption BF10")


@pytest.mark.parametrize("case", _params("bayes"))
def test_one_sample_bayes_factor(case: OracleCase) -> None:
    _check_bayes_factor(case, case.subtitle_data, "BF10")


@pytest.mark.parametrize("case", _params("nonparametric"))
def test_one_sample_signed_rank_and_rank_biserial(case: OracleCase) -> None:
    x, mu = _x(case), _mu(case)
    row = case.subtitle_data
    side = alternative(case)
    assert row["method"] == "Wilcoxon signed rank test"
    assert int(value(row, "n.obs")) == x.size
    test = one_sample_signed_rank(x, mu=mu, alternative=side)
    assert test.n == x.size
    assert_close("closed_form", test.statistic, value(row, "statistic"), what="V")
    assert_close("closed_form", test.p_value, value(row, "p.value"), what="p")
    r = one_sample_rank_biserial(
        x, mu=mu, conf_level=conf_level(case), alternative=side
    )
    assert row["effectsize"] == "r (rank biserial)"
    assert r.conf_level == value(row, "conf.level")
    assert_close("closed_form", r.estimate, value(row, "estimate"), what="r")
    assert_close("closed_form", r.conf_low, value(row, "conf.low"), what="r low")
    assert_close("closed_form", r.conf_high, value(row, "conf.high"), what="r high")


@pytest.mark.parametrize("case", _params("robust"))
def test_trimmed_mean_bootstrap_t(case: OracleCase) -> None:
    # t and the trimmed mean are exact; the bounds and p are bootstrap quantities,
    # compared with the 20,000-resample reference within 4 Monte Carlo SDs.
    x, mu = _x(case), _mu(case)
    row = case.subtitle_data
    level = conf_level(case)
    trim = tr(case)
    assert row["method"] == "Bootstrap-t method for one-sample test"
    assert int(value(row, "n.obs")) == x.size
    result = trimmed_mean_bootstrap_t(x, mu=mu, tr=trim, conf_level=level)
    # No resample is 0/0 on any fixture; T* is infinite (a resample whose
    # winsorized values are all one number other than the trimmed mean) on two,
    # kept as in R: edge_one_small's interval is [-Inf, Inf] in R's reference too.
    assert result.undefined_resamples == 0
    assert result.infinite_resamples == INFINITE_RESAMPLES.get(case.name, 0)
    assert_close("closed_form", result.statistic, value(row, "statistic"), what="t")
    assert_close("closed_form", result.estimate, value(row, "estimate"), what="tmean")
    assert result.conf_level == value(row, "conf.level")
    reference = case.bootstrap_reference
    assert reference is not None
    assert reference["conf_level"] == level
    assert f"tr = {trim}," in reference["call"]
    assert f"nv = {mu:g}," in reference["call"]
    assert_close(
        "closed_form", result.statistic, number(reference["raw"]["test.stat"]), what="t"
    )
    replicates = reference["replicates"]
    for mine, name in (
        (result.conf_low, "conf_low"),
        (result.conf_high, "conf_high"),
        (result.p_value, "p_value"),
    ):
        assert_within_monte_carlo(
            mine,
            number(reference[name]),
            number(replicates[f"{name}_sd"]),
            what=f"{case.name} {name}",
        )


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c.name) for c in CASES if c.centrality is not None],
)
def test_centrality(case: OracleCase) -> None:
    x = _x(case)
    centrality = case.centrality
    assert centrality is not None
    (row,) = centrality
    assert int(number(row["n.obs"])) == x.size
    expected = number(row["var"])
    if case.type == "bayes":
        found = map_estimate(x)
        assert_within_grid(
            found.estimate,
            expected,
            origin=float(x.min()),
            step=found.grid_step,
            what=f"{case.name} MAP",
        )
        return
    got = {
        "parametric": lambda: mean(x),
        "nonparametric": lambda: median(x),
        "robust": lambda: trimmed_mean(x, tr=tr(case)),
    }[case.type]()
    assert_close("closed_form", got, expected, what=f"{case.name} centrality")


def test_every_case_is_compared() -> None:
    # Each case has its subtitle compared above or, without one, its centrality.
    without = [c.name for c in CASES if not c.subtitle_rows]
    assert sorted(without) == [
        f"gghistostats-ToothGrowth-nosubtitle-{t}"
        for t in ("bayes", "nonparametric", "parametric", "robust")
    ]
    assert [c.name for c in CASES if c.centrality is None] == [
        "gghistostats-ToothGrowth-nocentrality-parametric"
    ]
