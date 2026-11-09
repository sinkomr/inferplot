"""Statistics-level parity of the nonparametric methods with ggstatsplot 1.1.4.

Every nonparametric ggbetweenstats fixture that ran is recomputed with
``inferplot.stats`` from the fixture's own dataset, groups and arguments
(alternative, conf.level, p.adjust.method; ggstatsplot's defaults when absent), so
new fixtures are checked without new code:

- two groups: Mann-Whitney W and p (``wilcox.test(exact = FALSE)``), the
  rank-biserial r and its CI, all within ``closed_form``;
- three or more groups: Kruskal-Wallis chi-square, df and p and the rank epsilon
  squared estimate within ``closed_form``; its bootstrap CI within the bootstrap
  tolerance of the fixture's 20,000-resample R reference (SPEC.md section 10); and
  Dunn's |z| and adjusted p (``p_adjust`` of inferplot's raw p) within
  ``closed_form``.
"""

from __future__ import annotations

import pytest

from inferplot.stats import (
    dunn,
    kruskal_wallis,
    mann_whitney,
    p_adjust,
    rank_biserial,
    rank_epsilon_squared,
)
from tests.oracle_fixtures import OracleCase, number
from tests.stats.fixture_data import (
    alternative,
    between_cases,
    conf_level,
    groups,
    p_adjust_method,
    params,
    value,
)
from tests.tolerances import assert_close, assert_within_monte_carlo

NONPARAMETRIC = between_cases("nonparametric")
TWO = between_cases("nonparametric", two_groups=True)
MANY = between_cases("nonparametric", two_groups=False)
R_SPELLING = {"two-sided": "two.sided", "greater": "greater", "less": "less"}
# statsExpressions' label for each p.adjust method the fixtures pin (text: C).
ADJUST_LABEL = {"holm": "Holm", "none": "None", "bonferroni": "Bonferroni"}


def _has_subtitle(case: OracleCase) -> bool:
    return bool(case.subtitle_rows)


def _has_pairs(case: OracleCase) -> bool:
    return bool(case.pairwise)


def test_every_nonparametric_case_is_checked_or_has_no_table() -> None:
    for case in NONPARAMETRIC:
        if not _has_subtitle(case):
            assert case.args.get("results.subtitle") is False, case.name
        if len(case.analysis_levels) > 2 and not _has_pairs(case):
            assert case.args.get("pairwise.display") == "none", case.name


@pytest.mark.parametrize(
    "case", params(TWO, "two-group nonparametric subtitles", where=_has_subtitle)
)
def test_mann_whitney_and_rank_biserial(case: OracleCase) -> None:
    a, b = groups(case)
    side = alternative(case)
    level = conf_level(case)
    row = case.subtitle_data
    assert row["method"] == "Wilcoxon rank sum test"
    assert row["alternative"] == R_SPELLING[side]
    result = mann_whitney(a, b, alternative=side)
    assert result.alternative == side
    assert_close("closed_form", result.statistic, value(row, "statistic"), what="W")
    assert_close("closed_form", result.p_value, value(row, "p.value"), what="p")
    assert row["effectsize"] == "r (rank biserial)"
    assert value(row, "conf.level") == level
    r = rank_biserial(a, b, conf_level=level, alternative=side)
    assert r.sidedness == side
    assert_close("closed_form", r.estimate, value(row, "estimate"), what="r")
    assert_close("closed_form", r.conf_low, value(row, "conf.low"), what="r low")
    assert_close("closed_form", r.conf_high, value(row, "conf.high"), what="r high")


@pytest.mark.parametrize(
    "case", params(MANY, "nonparametric subtitles, k >= 3", where=_has_subtitle)
)
def test_kruskal_wallis_and_rank_epsilon_squared(case: OracleCase) -> None:
    data = groups(case)
    level = conf_level(case)
    row = case.subtitle_data
    assert row["method"] == "Kruskal-Wallis rank sum test"
    result = kruskal_wallis(data)
    assert_close("closed_form", result.statistic, value(row, "statistic"), what="H")
    assert_close("closed_form", result.df, value(row, "df.error"), what="df")
    assert_close("closed_form", result.p_value, value(row, "p.value"), what="p")
    assert row["effectsize"] == "Epsilon2 (rank)"
    assert row["conf.method"] == "percentile bootstrap"
    assert value(row, "conf.level") == level
    epsilon = rank_epsilon_squared(data, conf_level=level)
    assert epsilon.sidedness == "greater"
    assert epsilon.conf_level == level
    assert epsilon.seed == 0
    assert epsilon.resamples == 2000
    # The estimate does not depend on the resamples: ggstatsplot's own value.
    assert_close("closed_form", epsilon.estimate, value(row, "estimate"), what="eps")
    assert value(row, "conf.high") == 1.0
    assert epsilon.conf_high == 1.0
    # The bound is random: the 20,000-resample R reference (SPEC.md section 10).
    reference = case.bootstrap_reference
    assert reference is not None, f"{case.name}: no bootstrap_reference"
    assert "rank_epsilon_squared" in str(reference["call"])
    assert 'alternative = "greater"' in str(reference["call"])
    assert number(reference["conf_level"]) == level
    spread = reference["replicates"]
    assert int(spread["resamples"]) == epsilon.resamples
    for got, key, sd_key in (
        (epsilon.estimate, "estimate", "estimate_sd"),
        (epsilon.conf_low, "conf_low", "conf_low_sd"),
        (epsilon.conf_high, "conf_high", "conf_high_sd"),
    ):
        assert_within_monte_carlo(
            got,
            number(reference[key]),
            number(spread[sd_key]),
            what=f"{case.name} rank epsilon squared {key}",
        )


@pytest.mark.parametrize(
    "case", params(MANY, "nonparametric pairwise rows, k >= 3", where=_has_pairs)
)
def test_dunn(case: OracleCase) -> None:
    levels = case.analysis_levels
    method = p_adjust_method(case)
    pairs = dunn(groups(case))
    assert [(p.i, p.j) for p in pairs] == [
        (i, j) for i in range(len(levels)) for j in range(i + 1, len(levels))
    ]
    adjusted = p_adjust([p.p_value for p in pairs], method)
    by_name = {
        (levels[p.i], levels[p.j]): (p, adj)
        for p, adj in zip(pairs, adjusted, strict=True)
    }
    assert len(case.pairwise) == len(pairs)
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        what = f"{case.name} Dunn {g1}-{g2}"
        assert row["test"] == "Dunn"
        assert row["distribution"] == "z"
        assert row["alternative"] == "two.sided"
        if method in ADJUST_LABEL:
            assert row["p.adjust.method"] == ADJUST_LABEL[method]
        # group1 is the earlier level in the analysis order (SPEC.md section 4).
        pair, p_adj = by_name[(g1, g2)]
        assert_close("closed_form", pair.statistic, number(row["statistic"]), what=what)
        assert_close("closed_form", p_adj, number(row["p.value"]), what=f"{what} p")
