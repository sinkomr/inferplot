"""Statistics-level parity with ggstatsplot 1.1.4 on the oracle fixtures.

Every parametric ggbetweenstats fixture is recomputed with ``inferplot.stats`` from
the fixture's own dataset, groups and arguments (alternative, conf.level,
p.adjust.method, bf.prior; ggstatsplot's defaults when absent), and so is every
two-sample ``BayesFactor::ttestBF`` value in the fixtures: the Bayes caption of the
two-group parametric fixtures, the subtitle of the two-group bayes fixtures and the
pairwise rows of the bayes fixtures with three or more groups. The analysis layer's
own parity tests (package E) check the same numbers through ``betweenstats``.
"""

from __future__ import annotations

import math

import pytest

from inferplot.stats import (
    games_howell,
    hedges_g,
    jzs_bf10_two_sample,
    mean,
    partial_omega_squared,
    variance,
    welch_anova,
    welch_t,
)
from tests.oracle_fixtures import OracleCase, number, r_level_label
from tests.stats.fixture_data import (
    PAIRWISE_BF_PRIOR,
    alternative,
    between_cases,
    bf_prior,
    conf_level,
    groups,
    p_adjust_method,
    params,
    value,
)
from tests.tolerances import assert_close

PARAMETRIC = between_cases("parametric")
TWO = between_cases("parametric", two_groups=True)
MANY = between_cases("parametric", two_groups=False)
R_SPELLING = {"two-sided": "two.sided", "greater": "greater", "less": "less"}


def _has_subtitle(case: OracleCase) -> bool:
    return bool(case.subtitle_rows)


def _has_caption(case: OracleCase) -> bool:
    return case.caption_data is not None


def _has_pairs(case: OracleCase) -> bool:
    return bool(case.pairwise)


def test_every_parametric_case_is_checked_or_has_no_table() -> None:
    # A case leaves the parity tests below only when ggstatsplot computed nothing
    # to compare, and only for the documented reason.
    for case in PARAMETRIC:
        if not _has_subtitle(case):
            assert case.args.get("results.subtitle") is False, case.name
        if len(case.analysis_levels) == 2 and not _has_caption(case):
            assert (
                case.args.get("bf.message") is False
                or case.args.get("results.subtitle") is False
            ), case.name
        if len(case.analysis_levels) > 2 and not _has_pairs(case):
            assert case.args.get("pairwise.display") == "none", case.name


@pytest.mark.parametrize(
    "case", params(TWO, "two-group parametric subtitles", where=_has_subtitle)
)
def test_welch_t_and_hedges_g(case: OracleCase) -> None:
    a, b = groups(case)
    side = alternative(case)
    level = conf_level(case)
    row = case.subtitle_data
    assert row["alternative"] == R_SPELLING[side]
    assert value(row, "conf.level") == level
    result = welch_t(a, b, alternative=side)
    assert_close("closed_form", result.t, value(row, "statistic"), what="Welch t")
    assert_close("closed_form", result.df, value(row, "df.error"), what="Welch df")
    assert_close("closed_form", result.p_value, value(row, "p.value"), what="p")
    assert_close("closed_form", mean(a), value(row, "mean.parameter1"), what="mean 1")
    assert_close("closed_form", mean(b), value(row, "mean.parameter2"), what="mean 2")
    assert row["effectsize"] == "Hedges' g"
    g = hedges_g(a, b, conf_level=level, alternative=side)
    assert g.sidedness == side
    assert_close("closed_form", g.estimate, value(row, "estimate"), what="g")
    assert_close("noncentral_ci", g.conf_low, value(row, "conf.low"), what="g low")
    assert_close("noncentral_ci", g.conf_high, value(row, "conf.high"), what="g high")


@pytest.mark.parametrize(
    "case", params(MANY, "parametric subtitles, k >= 3", where=_has_subtitle)
)
def test_welch_anova_and_partial_omega_squared(case: OracleCase) -> None:
    level = conf_level(case)
    row = case.subtitle_data
    result = welch_anova(groups(case))
    assert_close("closed_form", result.f, value(row, "statistic"), what="Welch F")
    assert_close("closed_form", result.df1, value(row, "df"), what="df1")
    assert_close("closed_form", result.df2, value(row, "df.error"), what="df2")
    assert_close("closed_form", result.p_value, value(row, "p.value"), what="p")
    assert row["effectsize"] == "Omega2"
    assert value(row, "conf.level") == level
    omega = partial_omega_squared(result.f, result.df1, result.df2, conf_level=level)
    assert_close("closed_form", omega.estimate, value(row, "estimate"), what="omega")
    assert_close("noncentral_ci", omega.conf_low, value(row, "conf.low"), what="low")
    assert_close("noncentral_ci", omega.conf_high, value(row, "conf.high"), what="hi")


@pytest.mark.parametrize(
    "case", params(MANY, "parametric pairwise rows, k >= 3", where=_has_pairs)
)
def test_games_howell(case: OracleCase) -> None:
    levels = case.analysis_levels
    method = p_adjust_method(case)
    pairs = games_howell(groups(case), p_adjust_method=method)
    by_name = {(levels[p.i], levels[p.j]): p for p in pairs}
    assert len(case.pairwise) == len(pairs) == len(levels) * (len(levels) - 1) // 2
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        what = f"Games-Howell {g1}-{g2}"
        assert row["test"] == "Games-Howell"
        if method == "holm":  # the only label the fixtures pin so far (text: C)
            assert row["p.adjust.method"] == "Holm"
        if (g1, g2) in by_name:
            pair, sign = by_name[(g1, g2)], 1.0
        else:
            pair, sign = by_name[(g2, g1)], -1.0
        # The fixture's q is mean(group2) - mean(group1) over its standard error.
        q = sign * pair.statistic
        assert_close("closed_form", q, number(row["statistic"]), what=f"{what} q")
        means = dict(zip(levels, (mean(g) for g in groups(case)), strict=True))
        assert math.copysign(1.0, q) == math.copysign(1.0, means[g2] - means[g1])
        assert_close(
            "games_howell_p", pair.p_adjusted, number(row["p.value"]), what=f"{what} p"
        )


@pytest.mark.parametrize("case", params(PARAMETRIC, "parametric centrality labels"))
def test_centrality_means(case: OracleCase) -> None:
    layers = case.layer_data("GeomLabelRepel")
    if not layers:
        assert case.args.get("centrality.plotting") is False, case.name
        return
    by_level = dict(zip(case.analysis_levels, groups(case), strict=True))
    y = case.args["y"]
    for row in layers[0]:
        level = r_level_label(row[case.args["x"]])
        values = by_level[level]
        what = f"{case.name} {level}"
        assert int(row["n.obs"]) == values.size
        assert_close("closed_form", mean(values), number(row[y]), what=f"{what} mean")
        assert_close(
            "closed_form",
            math.sqrt(variance(values)),
            number(row["std.dev"]),
            what=f"{what} SD",
        )


# ------------------------------------------------------------------ Bayes factors


def _check_bf(row: object, a: object, b: object, prior: float, *, what: str) -> None:
    assert isinstance(row, dict)
    assert number(row["prior.scale"]) == prior, what
    bf10, log_bf01 = jzs_bf10_two_sample(a, b, prior_scale=prior)  # type: ignore[arg-type]
    assert_close("ttest_bayes_factor", bf10, value(row, "bf10"), what=f"{what} BF10")
    assert log_bf01 == pytest.approx(-math.log(bf10), rel=1e-15, abs=1e-15)


@pytest.mark.parametrize(
    "case", params(TWO, "two-group parametric Bayes captions", where=_has_caption)
)
def test_bayes_caption_two_groups(case: OracleCase) -> None:
    # ggstatsplot's caption is ttestBF with rscale = bf.prior, two-sided whatever
    # `alternative` says.
    a, b = groups(case)
    caption = case.caption_data
    _check_bf(dict(caption or {}), a, b, bf_prior(case), what=f"{case.name} caption")


@pytest.mark.parametrize(
    "case",
    params(
        between_cases("bayes", two_groups=True),
        "two-group bayes subtitles",
        where=_has_subtitle,
    ),
)
def test_bayes_subtitle_two_groups(case: OracleCase) -> None:
    a, b = groups(case)
    _check_bf(dict(case.subtitle_data), a, b, bf_prior(case), what=case.name)


@pytest.mark.parametrize(
    "case",
    params(
        between_cases("bayes", two_groups=False),
        "bayes pairwise rows, k >= 3",
        where=_has_pairs,
    ),
)
def test_bayes_pairwise(case: OracleCase) -> None:
    by_level = dict(zip(case.analysis_levels, groups(case), strict=True))
    k = len(by_level)
    assert len(case.pairwise) == k * (k - 1) // 2
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        _check_bf(
            dict(row),
            by_level[g1],
            by_level[g2],
            PAIRWISE_BF_PRIOR,
            what=f"{case.name} {g1}-{g2}",
        )
