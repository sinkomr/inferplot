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
    assert "11 of 12 values" in issue.message
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
