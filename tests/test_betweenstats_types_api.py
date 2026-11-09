"""``ip.analyze.betweenstats`` with type nonparametric, robust and bayes: argument
rules, plain errors, warnings, seeds, texts and invariances.

Numbers against the R oracle are in ``tests/test_betweenstats_types_parity.py``.
"""

from __future__ import annotations

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
