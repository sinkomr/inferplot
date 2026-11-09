"""``ip.analyze.betweenstats`` with type nonparametric, robust and bayes: argument
rules, plain errors, warnings, seeds, texts and invariances.

Numbers against the R oracle are in ``tests/test_betweenstats_types_parity.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from tests.oracle_fixtures import DATASET_DIR

betweenstats: Any = ip.analyze.betweenstats


@pytest.fixture(scope="module")
def plants() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "PlantGrowth.csv")


@pytest.fixture(scope="module")
def teeth() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "ToothGrowth.csv")


def frame(groups: dict[str, list[float]]) -> pd.DataFrame:
    rows = [(g, v) for g, values in groups.items() for v in values]
    return pd.DataFrame(rows, columns=["g", "y"])


def _codes(result: StatsResult) -> set[str]:
    return {issue.code for issue in result.check()}


# --------------------------------------------------------------------------- #
# type nonparametric
# --------------------------------------------------------------------------- #


def test_nonparametric_alias(plants: pd.DataFrame) -> None:
    full = betweenstats(plants, "group", "weight", type="nonparametric")
    assert betweenstats(plants, "group", "weight", type="np") == full
    assert full.type == "nonparametric"
    assert full.settings.tr == 0.2


def test_nonparametric_two_groups(teeth: pd.DataFrame) -> None:
    result = betweenstats(teeth, "supp", "len", type="np")
    assert result.test.name == "mann_whitney"
    assert result.test.statistic_name == "W"
    assert (result.test.df, result.test.df_error) == (None, None)
    assert result.effect_size.name == "rank_biserial"
    assert result.effect_size.bootstrap_resamples is None
    assert result.effect_size.bootstrap_seed is None
    assert result.bayes is None
    assert result.pairwise == ()
    assert result.text.caption is None
    assert result.text.apa == (
        "Mann-Whitney U test: W = 575.50, p = .064, rank-biserial r = .28, "
        "95% CI [-.01, .52], n = 60."
    )
    for d, (_, group) in zip(result.descriptives, teeth.groupby("supp"), strict=True):
        assert d.centrality == "median"
        assert d.value == float(np.median(group["len"]))
    assert "median 22.70" in result.text.alt_text


def test_nonparametric_one_sided(teeth: pd.DataFrame) -> None:
    result = betweenstats(teeth, "supp", "len", type="np", alternative="greater")
    assert result.test.alternative == "greater"
    assert result.effect_size.sidedness == "greater"
    assert result.effect_size.conf_high == 1.0
    assert result.text.apa.startswith("Mann-Whitney U test (one-sided, greater): ")


def test_nonparametric_three_groups(plants: pd.DataFrame) -> None:
    result = betweenstats(plants, "group", "weight", type="np", seed=0)
    assert result.test.name == "kruskal_wallis"
    assert result.test.statistic_name == "chi2"
    assert (result.test.df, result.test.df_error) == (None, 2.0)
    es = result.effect_size
    assert es.name == "rank_epsilon_squared"
    assert es.ci_method == "percentile bootstrap"
    assert es.sidedness == "greater"
    assert es.conf_high == 1.0
    assert es.bootstrap_resamples == stats.BOOTSTRAP_RESAMPLES == 2000
    assert es.bootstrap_seed == 0 == result.provenance.seed
    assert [p.test_label for p in result.pairwise] == ["Dunn"] * 3
    assert all(p.statistic_name == "z" and p.statistic >= 0 for p in result.pairwise)
    assert result.text.caption == (
        "Pairwise: Dunn, Holm adjusted; showing 1 of 3 pairs (significant only)."
    )
    assert result.text.apa.startswith(
        "Kruskal-Wallis test: chi2(2) = 7.99, p = .018, rank epsilon squared = .28, "
        "95% CI ["
    )
    assert "Dunn, Holm adjusted, alpha = 0.05" in result.text.alt_text
    table = str(result)
    assert "Pairwise comparisons (Dunn):" in table
    frame_rows = result.to_frame()
    assert list(frame_rows["test"]) == ["kruskal_wallis", "dunn", "dunn", "dunn"]


def test_seed_moves_only_the_bootstrap_bound(plants: pd.DataFrame) -> None:
    first = betweenstats(plants, "group", "weight", type="np", seed=0)
    again = betweenstats(plants, "group", "weight", type="np", seed=0)
    other = betweenstats(plants, "group", "weight", type="np", seed=7)
    assert first == again
    assert other.effect_size.bootstrap_seed == 7 == other.settings.seed
    assert other.effect_size.estimate == first.effect_size.estimate
    assert other.effect_size.conf_low != first.effect_size.conf_low
    assert other.test == first.test
    assert other.pairwise == first.pairwise


def test_nonparametric_argument_rules(
    plants: pd.DataFrame, teeth: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="Kruskal-Wallis test"):
        betweenstats(plants, "group", "weight", type="np", alternative="less")
    with pytest.raises(ValueError, match="trims nothing"):
        betweenstats(teeth, "supp", "len", type="np", tr=0.1)
    with pytest.raises(ValueError, match="only for type='parametric'"):
        betweenstats(teeth, "supp", "len", type="np", bf_message=True)
    quiet = betweenstats(teeth, "supp", "len", type="np", bf_message=False)
    assert quiet.warnings == ()
    warned = betweenstats(teeth, "supp", "len", type="np", bf_prior=0.8)
    assert warned.warnings == (
        "bf_prior=0.8 has no effect: no Bayes factor is computed "
        "(type='nonparametric' has no Bayes factor).",
    )


def test_nonparametric_constant_data_raises() -> None:
    data = frame({"a": [1.0, 1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]})
    with pytest.raises(ValueError, match=r"Every value of y is the same.*x='g'"):
        betweenstats(data, "g", "y", type="np")


def test_nonparametric_single_values_are_allowed() -> None:
    # The rank tests need no variance: a group of one value is fine (as in R).
    data = frame({"a": [1.0], "b": [2.0, 3.0, 4.0]})
    result = betweenstats(data, "g", "y", type="np")
    assert result.test.name == "mann_whitney"
    assert "small_n" in _codes(result)


def test_heavy_ties_check() -> None:
    tied = frame({"a": [1.0, 1, 1, 2, 2, 3], "b": [2.0, 2, 3, 3, 3, 4]})
    result = betweenstats(tied, "g", "y", type="np")
    (issue,) = [i for i in result.check() if i.code == "heavy_ties"]
    assert issue.severity == "warning"
    assert "4 distinct values among 12, so 8 of them (67%) are ties" in issue.message
    # Ties are the values beyond the first of each distinct value: here 4 of 10
    # values sit in a tie, but only 2 (20%) repeat one, which is not heavy.
    pairs = frame({"a": [1.0, 1, 2, 2, 3], "b": [4.0, 5, 6, 7, 8]})
    assert "heavy_ties" not in _codes(betweenstats(pairs, "g", "y", type="np"))
    # Only rank-based tests get it.
    assert "heavy_ties" not in _codes(betweenstats(tied, "g", "y"))
    distinct = frame({"a": [1.0, 2, 3, 4, 5], "b": [6.0, 7, 8, 9, 9]})
    assert "heavy_ties" not in _codes(betweenstats(distinct, "g", "y", type="np"))


def test_nonparametric_row_order_and_relabelling(plants: pd.DataFrame) -> None:
    base = betweenstats(plants, "group", "weight", type="np")
    shuffled = plants.sample(frac=1.0, random_state=3).reset_index(drop=True)
    again = betweenstats(shuffled, "group", "weight", type="np")
    # Every number and text; values are in data order, so they differ.
    assert again.test == base.test
    assert again.effect_size == base.effect_size
    assert again.pairwise == base.pairwise
    assert again.descriptives == base.descriptives
    assert again.text == base.text
    renamed = plants.assign(
        group=plants["group"].replace({"ctrl": "x", "trt1": "y", "trt2": "z"})
    )
    relabelled = betweenstats(renamed, "group", "weight", type="np")
    assert relabelled.test == base.test
    assert relabelled.effect_size == base.effect_size
    assert [p.p_adjusted for p in relabelled.pairwise] == [
        p.p_adjusted for p in base.pairwise
    ]
    reordered = betweenstats(
        plants, "group", "weight", type="np", order=["trt2", "trt1", "ctrl"]
    )
    assert reordered.test == base.test
    assert reordered.effect_size == base.effect_size


# --------------------------------------------------------------------------- #
# type robust
# --------------------------------------------------------------------------- #


def test_robust_alias(plants: pd.DataFrame) -> None:
    full = betweenstats(plants, "group", "weight", type="robust")
    assert betweenstats(plants, "group", "weight", type="r") == full
    assert full.type == "robust"


def test_robust_two_groups(teeth: pd.DataFrame) -> None:
    result = betweenstats(teeth, "supp", "len", type="robust")
    assert result.test.name == "yuen_t"
    assert result.test.label == "Yuen's t-test (20% trimmed means)"
    assert result.test.statistic >= 0  # |t|, as ggstatsplot reports it
    es = result.effect_size
    assert es.name == "akp_delta"
    assert es.ci_method == "percentile bootstrap"
    assert es.sidedness == "two-sided"
    assert (es.bootstrap_resamples, es.bootstrap_seed) == (2000, 0)
    assert result.pairwise == ()
    assert result.text.caption is None
    for d, (_, group) in zip(result.descriptives, teeth.groupby("supp"), strict=True):
        assert d.centrality == "trimmed_mean"
        assert d.value == stats.trimmed_mean(group["len"].to_numpy(), tr=0.2)
    assert "20% trimmed mean 21.71" in result.text.alt_text
    assert result.text.apa.startswith(
        "Yuen's t-test (20% trimmed means): t(33.49) = 2.29, p = .029, AKP robust "
        "standardized difference = 0.68, 95% CI ["
    )


def test_robust_three_groups(plants: pd.DataFrame) -> None:
    result = betweenstats(plants, "group", "weight", type="robust")
    assert result.test.name == "trimmed_means_anova"
    assert (result.test.df, result.test.statistic_name) == (2.0, "F")
    es = result.effect_size
    assert es.name == "explanatory_xi"
    assert (es.bootstrap_resamples, es.bootstrap_seed) == (2000, 0)
    assert result.text.caption == (
        "Pairwise: Yuen's trimmed means, Holm adjusted; showing 1 of 3 pairs "
        "(significant only)."
    )
    for pair in result.pairwise:
        assert pair.test_label == "Yuen's trimmed means"
        assert pair.statistic_name == "t"
        assert pair.estimate is not None
        assert pair.conf_low is not None
        assert pair.conf_high is not None
        assert pair.conf_level == 0.95
        assert pair.conf_low < pair.estimate < pair.conf_high
        # t carries the sign of the difference group1 - group2
        assert math.copysign(1.0, pair.statistic) == math.copysign(1.0, pair.estimate)
    frame_rows = result.to_frame()
    assert frame_rows["estimate"].iloc[1:].notna().all()
    assert "Pairwise comparisons (Yuen's trimmed means):" in str(result)


def test_robust_tr(plants: pd.DataFrame, teeth: pd.DataFrame) -> None:
    result = betweenstats(teeth, "supp", "len", type="robust", tr=0.1)
    assert result.settings.tr == 0.1
    assert result.test.label == "Yuen's t-test (10% trimmed means)"
    assert "10% trimmed mean" in result.text.alt_text
    values = teeth.loc[teeth["supp"] == "OJ", "len"].to_numpy()
    assert result.descriptives[0].value == stats.trimmed_mean(values, tr=0.1)
    untrimmed = betweenstats(plants, "group", "weight", type="robust", tr=0.0)
    assert [d.value for d in untrimmed.descriptives] == [
        stats.mean(g["weight"].to_numpy()) for _, g in plants.groupby("group")
    ]


def test_robust_argument_rules(plants: pd.DataFrame, teeth: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="always two-sided"):
        betweenstats(teeth, "supp", "len", type="robust", alternative="greater")
    with pytest.raises(ValueError, match="only for type='parametric'"):
        betweenstats(teeth, "supp", "len", type="robust", bf_message=True)
    with pytest.raises(NotImplementedError, match=r"conf_level=0\.9 .*Pass conf_level"):
        betweenstats(plants, "group", "weight", type="robust", conf_level=0.9)
    # Two groups take any level (no lincon table involved).
    two = betweenstats(teeth, "supp", "len", type="robust", conf_level=0.9)
    assert two.effect_size.conf_level == 0.9
    warned = betweenstats(teeth, "supp", "len", type="robust", bf_prior=0.5)
    assert warned.warnings == (
        "bf_prior=0.5 has no effect: no Bayes factor is computed "
        "(type='robust' has no Bayes factor).",
    )


def test_robust_plain_errors() -> None:
    few = frame({"a": [1.0, 2.0, 3.0], "b": [4.0], "c": [1.0, 5.0, 2.0]})
    with pytest.raises(ValueError, match=r"at least 2 observations.*'b' \(n = 1\)"):
        betweenstats(few, "g", "y", type="robust")
    with pytest.raises(ValueError, match=r"tr=0\.45\) leaves fewer than 2 values"):
        betweenstats(few.iloc[[0, 1, 2, 4, 5]], "g", "y", type="robust", tr=0.45)
    flat = frame({"a": [1.0, 2, 2, 2, 2, 3], "b": [1.0, 4, 5, 6, 7], "c": [3.0, 1, 2]})
    with pytest.raises(ValueError, match=r"winsorized variance.*'a' of x='g'"):
        betweenstats(flat, "g", "y", type="robust")
    two_flat = frame({"a": [1.0, 2, 2, 2, 2, 3], "b": [1.0, 4, 5, 6, 7]})
    with pytest.raises(ValueError, match=r"first group, 'a'.*order="):
        betweenstats(two_flat, "g", "y", type="robust")
    # The other order works: AKP divides by the first group's spread only.
    swapped = betweenstats(two_flat, "g", "y", type="robust", order=["b", "a"])
    assert swapped.levels == ("b", "a")
    tiny = frame({"a": [1.0, 2.0], "b": [5.0, 7.0], "c": [3.0, 9.0]})
    with pytest.raises(NotImplementedError, match="fewer than 2 degrees of freedom"):
        betweenstats(tiny, "g", "y", type="robust", tr=0.0)


def test_robust_bootstrap_error_names_x() -> None:
    # Most values of 'a' are 5, as are all of 'b': 759 of 2,000 AKP resamples
    # have a winsorized variance of 0 and a difference of 0.
    data = frame({"a": [5.0] * 7 + [6.0, 7.0, 8.0], "b": [5.0] * 10})
    with pytest.raises(
        ValueError, match=r"0 / 0.*\(the groups of x='g'\)\. Pass a smaller tr"
    ):
        betweenstats(data, "g", "y", type="robust")


def test_robust_seed_and_order(plants: pd.DataFrame) -> None:
    base = betweenstats(plants, "group", "weight", type="robust")
    other = betweenstats(plants, "group", "weight", type="robust", seed=3)
    assert other.effect_size.bootstrap_seed == 3
    assert other.effect_size.conf_low != base.effect_size.conf_low
    assert other.test == base.test
    shuffled = plants.sample(frac=1.0, random_state=5).reset_index(drop=True)
    again = betweenstats(shuffled, "group", "weight", type="robust")
    assert again.effect_size == base.effect_size
    assert again.pairwise == base.pairwise
    assert again.text == base.text


# --------------------------------------------------------------------------- #
# type bayes
# --------------------------------------------------------------------------- #

POSTERIOR_WARNING = (
    "The posterior difference and its credible interval, which ggstatsplot adds to "
    "the Bayes factor subtitle, arrive in inferplot 0.2."
)


def test_bayes_two_groups(teeth: pd.DataFrame) -> None:
    result = betweenstats(teeth, "supp", "len", type="bayes")
    assert betweenstats(teeth, "supp", "len", type="bf") == result
    assert result.type == "bayes"
    assert result.test is None
    assert result.effect_size is None
    assert result.pairwise == ()
    oj = teeth.loc[teeth["supp"] == "OJ", "len"].to_numpy(dtype=float)
    vc = teeth.loc[teeth["supp"] == "VC", "len"].to_numpy(dtype=float)
    bf10, log_bf01 = stats.jzs_bf10_two_sample(oj, vc, prior_scale=0.707)
    assert result.bayes is not None
    assert (result.bayes.bf10, result.bayes.log_bf01) == (bf10, log_bf01)
    assert result.bayes.prior_scale == 0.707
    assert [d.centrality for d in result.descriptives] == ["map", "map"]
    assert result.descriptives[0].value == stats.map_estimate(oj).estimate
    assert result.text.subtitle == "log_e(BF_01) = -0.18, r_Cauchy^JZS = 0.71"
    assert result.text.caption is None
    assert result.text.apa == (
        "Bayesian t-test (JZS prior, r = 0.707): BF10 = 1.20, log(BF01) = -0.18, "
        "n = 60."
    )
    assert "MAP estimate 24.91" in result.text.alt_text
    assert result.warnings == (POSTERIOR_WARNING,)
    frame_rows = result.to_frame()
    assert list(frame_rows["test"]) == ["jzs_two_sample"]
    assert frame_rows.loc[0, "bf10"] == bf10
    assert StatsResult.from_json(result.to_json()) == result


def test_bayes_prior_and_no_subtitle(teeth: pd.DataFrame) -> None:
    wide = betweenstats(teeth, "supp", "len", type="bayes", bf_prior=0.8)
    assert wide.bayes is not None
    assert wide.bayes.prior_scale == 0.8
    assert wide.text.subtitle is not None
    assert wide.text.subtitle.endswith("r_Cauchy^JZS = 0.80")
    bare = betweenstats(teeth, "supp", "len", type="bayes", results_subtitle=False)
    assert bare.bayes is None
    assert bare.text.subtitle is None
    assert bare.warnings == ()
    assert bare.text.apa.startswith("No test was computed")
    unused = betweenstats(
        teeth, "supp", "len", type="bayes", results_subtitle=False, bf_prior=0.8
    )
    assert unused.warnings == (
        "bf_prior=0.8 has no effect: no Bayes factor is computed "
        "(results_subtitle=False leaves the Bayes factor out).",
    )


def test_bayes_three_groups_needs_results_subtitle_false(plants: pd.DataFrame) -> None:
    with pytest.raises(NotImplementedError, match=r"Bayesian ANOVA.*0\.2") as info:
        betweenstats(plants, "group", "weight", type="bayes")
    assert "results_subtitle=False" in str(info.value)
    result = betweenstats(
        plants, "group", "weight", type="bayes", results_subtitle=False
    )
    assert result.test is None
    assert result.effect_size is None
    assert result.bayes is None
    assert result.warnings == ()
    groups = {g: v["weight"].to_numpy(dtype=float) for g, v in plants.groupby("group")}
    assert len(result.pairwise) == 3
    for pair in result.pairwise:
        assert pair.bayes is not None
        assert pair.p_value is None
        assert pair.p_adjusted is None
        assert pair.significant is None
        assert pair.displayed
        assert pair.adjust_method == "none"
        assert pair.test_label == "Student's t"
        a, b = groups[pair.group1], groups[pair.group2]
        bf10, _ = stats.jzs_bf10_two_sample(a, b, prior_scale=0.707)
        assert pair.bayes.bf10 == bf10
        student = stats.student_t(a, b)
        assert (pair.statistic, pair.df) == (student.t, student.df)
    assert result.text.caption == (
        "Pairwise: Bayes factors (Student's t, JZS prior 0.707), unadjusted; "
        "showing all 3 pairs."
    )
    assert result.text.pairwise_labels[0] == "log_e(BF_01) = 0.43"
    assert result.text.apa.startswith(
        "Bayesian t-tests of each pair (JZS prior, r = 0.707): ctrl vs trt1, "
        "BF10 = 0.65; "
    )
    assert result.check() == ()
    table = str(result)
    assert "Pairwise Bayes factors (Student's t, JZS prior 0.707):" in table
    assert list(result.to_frame()["test"]) == ["jzs_two_sample"] * 3
    assert StatsResult.from_json(result.to_json()) == result
    hidden = betweenstats(
        plants,
        "group",
        "weight",
        type="bayes",
        results_subtitle=False,
        pairwise_display="none",
    )
    assert not any(pair.displayed for pair in hidden.pairwise)
    assert hidden.text.caption is None


def test_bayes_argument_rules(plants: pd.DataFrame, teeth: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Bayes factor is always two-sided"):
        betweenstats(teeth, "supp", "len", type="bayes", alternative="less")
    with pytest.raises(ValueError, match="trims nothing"):
        betweenstats(teeth, "supp", "len", type="bayes", tr=0.1)
    with pytest.raises(ValueError, match="reports the Bayes factor in the subtitle"):
        betweenstats(teeth, "supp", "len", type="bayes", bf_message=True)
    two = betweenstats(teeth, "supp", "len", type="bayes", conf_level=0.9)
    assert two.warnings == (
        POSTERIOR_WARNING,
        "conf_level=0.9 has no effect: type='bayes' shows no interval in this "
        "version (the posterior credible interval arrives in inferplot 0.2).",
    )
    many = betweenstats(
        plants,
        "group",
        "weight",
        type="bayes",
        results_subtitle=False,
        p_adjust_method="BH",
        pairwise_alpha=0.1,
        pairwise_display="ns",
        bf_prior=0.5,
    )
    assert many.warnings == (
        "p_adjust_method='BH' has no effect: Bayes factors are not adjusted for "
        "multiple comparisons (as in ggstatsplot).",
        "pairwise_alpha=0.1 has no effect: Bayes factors are not compared with a "
        "significance level, and every pair is drawn.",
        "pairwise_display='ns' has no effect: Bayes factors have no significance, "
        "so every pair is drawn (as in ggstatsplot); pass pairwise_display='none' "
        "to draw none.",
        "bf_prior=0.5 has no effect: no Bayes factor is computed (the pairwise "
        "Bayes factors always use the prior scale 0.707, as in ggstatsplot, and "
        "the Bayesian ANOVA arrives in inferplot 0.2).",
    )
    assert all(pair.displayed for pair in many.pairwise)
    assert all(pair.bayes and pair.bayes.prior_scale == 0.707 for pair in many.pairwise)


def test_bayes_plain_errors() -> None:
    one = frame({"a": [1.0, 2.0, 4.0], "b": [3.0]})
    with pytest.raises(ValueError, match=r"at least 2 observations.*'b' \(n = 1\)"):
        betweenstats(one, "g", "y", type="bayes")
    flat = frame({"a": [1.0, 2.0, 4.0, 7.0], "b": [3.0, 3.0, 3.0]})
    with pytest.raises(ValueError, match=r"MAP estimate.*group 'b' of x='g'"):
        betweenstats(flat, "g", "y", type="bayes")
    with pytest.raises(ValueError, match="type='nonparametric'"):
        betweenstats(flat, "g", "y", type="bayes")


def test_bayes_row_order(plants: pd.DataFrame) -> None:
    base = betweenstats(plants, "group", "weight", type="bayes", results_subtitle=False)
    shuffled = plants.sample(frac=1.0, random_state=11).reset_index(drop=True)
    again = betweenstats(
        shuffled, "group", "weight", type="bayes", results_subtitle=False
    )
    assert again.pairwise == base.pairwise
    assert again.text == base.text
    for a, b in zip(again.descriptives, base.descriptives, strict=True):
        assert a.value == pytest.approx(b.value, rel=1e-12)
