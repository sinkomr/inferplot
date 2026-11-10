"""``ip.analyze.betweenstats`` with type nonparametric, robust and bayes: argument
rules, plain errors, warnings, seeds, texts and invariances.

Numbers against the R oracle are in ``tests/test_betweenstats_types_parity.py``.
"""

from __future__ import annotations

import math
from decimal import ROUND_FLOOR, Decimal, localcontext
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot import stats
from inferplot.analyses.betweenstats import (
    _named_groups,  # pyright: ignore[reportPrivateUsage]
)
from inferplot.result import StatsResult
from inferplot.text import figure_texts
from tests.oracle_fixtures import DATASET_DIR
from tests.render._helpers import numbered_texts

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
    assert result.text.apa.startswith(
        "Mann-Whitney U test (one-sided: OJ greater than VC): "
    )


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


def test_undefined_resamples_are_in_the_result() -> None:
    # Three groups of three on two values: some resamples tie every value, where
    # rank epsilon squared is undefined; they are left out of the interval (D54).
    data = frame({"a": [1.0, 1, 2], "b": [1.0, 2, 2], "c": [1.0, 1, 2]})
    result = betweenstats(data, "g", "y", type="np")
    es = result.effect_size
    assert es.bootstrap_resamples == stats.BOOTSTRAP_RESAMPLES
    assert es.undefined_resamples is not None
    assert 0 < es.undefined_resamples < es.bootstrap_resamples
    used = es.bootstrap_resamples - es.undefined_resamples
    (note,) = [w for w in result.warnings if "resamples" in w]
    assert f"{es.undefined_resamples:,} of {es.bootstrap_resamples:,}" in note
    assert f"{used:,}" in note
    assert StatsResult.from_json(result.to_json()).effect_size == es
    # Every bootstrap interval records the count; analytic ones do not.
    a = [1.3, 2.1, 3.7, 4.2, 5.9, 6.4, 2.8]
    plain = frame({"a": a, "b": [2.2, 4.8, 6.1, 8.5, 10.3, 11.1, 7.7]})
    counts = {
        kind: betweenstats(plain, "g", "y", type=kind).effect_size.undefined_resamples
        for kind in ("r", "np", "p")
    }
    assert counts == {"r": 0, "np": None, "p": None}


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
    # Three or more groups: the effect size follows conf_level, the pairwise
    # intervals stay at 95% as in ggstatsplot (D53), and the result says so.
    many = betweenstats(plants, "group", "weight", type="robust", conf_level=0.9)
    assert many.effect_size.conf_level == many.settings.conf_level == 0.9
    assert [pair.conf_level for pair in many.pairwise] == [0.95] * 3
    at_95 = betweenstats(plants, "group", "weight", type="robust")
    assert many.pairwise == at_95.pairwise
    assert at_95.warnings == ()
    (warning,) = many.warnings
    assert warning.startswith("The pairwise confidence intervals are 95% intervals")
    assert "conf_level=0.9," in warning
    assert [i.code for i in many.check()] == ["pairs_hidden", "pairwise_conf_level"]
    assert "pairwise_conf_level" not in [i.code for i in at_95.check()]
    shown = betweenstats(
        plants, "group", "weight", type="robust", conf_level=0.9, pairwise_display="all"
    )
    (issue,) = shown.check()
    assert issue.code == "pairwise_conf_level"
    assert issue.severity == "info"
    assert "95% intervals" in issue.message
    assert "conf_level=0.9 " in issue.message
    # Two groups take any level (no pairwise comparisons).
    two = betweenstats(teeth, "supp", "len", type="robust", conf_level=0.9)
    assert two.effect_size.conf_level == 0.9
    assert two.warnings == ()
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


def test_robust_conf_level_too_close_to_one_raises(
    plants: pd.DataFrame, teeth: pd.DataFrame
) -> None:
    # Above about 0.999 the bootstrap bounds would be the most extreme resamples
    # at any level. The message names the argument and a remedy, and is not the
    # statistics layer's message with words replaced.
    for data, x, y in ((teeth, "supp", "len"), (plants, "group", "weight")):
        with pytest.raises(ValueError, match="conf_level") as refused:
            ip.analyze.betweenstats(data, x, y, type="robust", conf_level=0.9999)
        message = str(refused.value)
        assert message.startswith("conf_level=0.9999 is too close to 1 for the boot")
        assert message.endswith(
            "Pass conf_level=0.999 or lower, or use type='parametric', whose "
            "intervals are not resampled."
        )
        result = ip.analyze.betweenstats(data, x, y, type="robust", conf_level=0.999)
        assert result.effect_size is not None
        assert result.effect_size.conf_level == 0.999


@pytest.mark.parametrize("scale", [1e-300, 1e-165, 1e156, 1e300])
def test_robust_does_not_depend_on_the_unit_of_y(
    plants: pd.DataFrame, teeth: pd.DataFrame, scale: float
) -> None:
    # The squares of the values underflow or overflow at these units; the checks
    # before the analysis must not mistake that for constant groups.
    for data, x, y in ((teeth, "supp", "len"), (plants, "group", "weight")):
        base = ip.analyze.betweenstats(data, x, y, type="robust")
        scaled = ip.analyze.betweenstats(
            data.assign(**{y: data[y] * scale}), x, y, type="robust"
        )
        assert base.test is not None
        assert scaled.test is not None
        assert base.effect_size is not None
        assert scaled.effect_size is not None
        assert scaled.test.statistic == pytest.approx(base.test.statistic, rel=1e-12)
        assert scaled.test.p_value == pytest.approx(base.test.p_value, rel=1e-10)
        assert scaled.effect_size.estimate == pytest.approx(
            base.effect_size.estimate, rel=1e-12
        )
        assert scaled.effect_size.conf_low == pytest.approx(
            base.effect_size.conf_low, rel=1e-12
        )


def test_statistics_messages_keep_words_that_contain_sample_b() -> None:
    # "sample b" is replaced by the group's name only as whole words: "resample
    # beyond" must stay, and a backslash in a level name must not be read as an
    # escape.
    out = _named_groups(
        "sample b has one resample beyond the bound", ["a\\1", "b\\g<0>"], "g"
    )
    assert out == ("The group 'b\\\\g<0>' of x='g' has one resample beyond the bound")
    # A level name that is put in is not read again as a group or a sample.
    out = _named_groups("group 1 differs from sample b", ["sample b", "group 1"], "g")
    assert out == (
        "The group 'sample b' of x='g' differs from the group 'group 1' of x='g'"
    )


def test_bayes_pair_errors_with_levels_named_like_group_numbers() -> None:
    # The pair's message already names its levels; it is not renamed a second
    # time, which read "group 7" as the seventh of three groups (IndexError).
    a, b, c = [1e160] * 4, [1.0, 2, 3, 4], [2.0, 3, 4, 6]
    data = frame({"group 7": a, "group 8": b, "group 9": c})
    with pytest.raises(ValueError, match="cannot be computed") as info:
        betweenstats(data, "g", "y", type="bayes", results_subtitle=False)
    assert str(info.value).startswith(
        "The Bayes factor of 'group 7' against 'group 8' cannot be computed: the "
        "group 'group 7' of x='g' is constant and the group 'group 8' of x='g' varies"
    )


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
        "bf_prior=0.5 has no effect: the pairwise Bayes factors always use the "
        "prior scale 0.707, as in ggstatsplot; the Bayesian ANOVA arrives in "
        "inferplot 0.2.",
    )
    assert all(pair.displayed for pair in many.pairwise)
    assert all(pair.bayes and pair.bayes.prior_scale == 0.707 for pair in many.pairwise)


def test_bayes_plain_errors() -> None:
    one = frame({"a": [1.0, 2.0, 4.0], "b": [3.0]})
    with pytest.raises(ValueError, match=r"at least 2 observations.*'b' \(n = 1\)"):
        betweenstats(one, "g", "y", type="bayes")


def test_bayes_map_of_constant_and_zero_iqr_groups() -> None:
    # D55, from probe-map. A constant group's MAP is its value (bayestestR's), with
    # no warning.
    flat = frame({"a": [1.0, 2.0, 4.0, 7.0], "b": [3.0, 3.0, 3.0]})
    result = betweenstats(flat, "g", "y", type="bayes")
    assert result.descriptives[1].value == 3.0
    assert result.text.centrality_labels[1] == "μ_MAP = 3.00"
    assert not [w for w in result.warnings if "MAP" in w]
    # A zero interquartile range with spread: R's bw.SJ fails, the MAP is NA and
    # ggstatsplot prints an empty label. inferplot: value None, a warning that says
    # why, the same empty label in the result's texts, nothing drawn for it.
    sparse = frame({"a": [1.0, 2.0, 4.0, 7.0], "b": [1.0] * 7 + [5.0]})
    result = betweenstats(sparse, "g", "y", type="bayes")
    assert result.descriptives[0].value == stats.map_estimate([1.0, 2, 4, 7]).estimate
    assert result.descriptives[1].value is None
    (warning,) = [w for w in result.warnings if "MAP" in w]
    assert warning.startswith(
        "The group 'b' of x='g' (n = 8) has no MAP estimate, so no centrality label "
        "is drawn for it: no Sheather-Jones bandwidth"
    )
    assert result.text.centrality_labels[1] == "μ_MAP = "
    assert "b (n = 8, no MAP estimate)" in result.text.alt_text
    assert "NA" not in result.text.alt_text
    assert StatsResult.from_json(result.to_json()) == result
    fig = ip.render.betweenstats(result)
    assert numbered_texts(fig, "centrality-label") == [
        figure_texts(result).centrality_labels[0]
    ]
    assert not [i for i in fig.check() if i.code in ("label_overlap", "text_clipped")]


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


# --------------------------------------------------------------------------- #
# end-of-phase-2 review findings
# --------------------------------------------------------------------------- #


def _bf10_from_log(log_bf01: float) -> str:
    """BF10 = exp(-log_bf01) as "m.mme+NNN", from 50-digit decimal arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 50
        log10 = Decimal(-log_bf01) / Decimal(10).ln()
        exponent = int(log10.to_integral_value(rounding=ROUND_FLOOR))
        mantissa = (Decimal(10) ** (log10 - exponent)).quantize(Decimal("0.01"))
        if mantissa >= 10:
            mantissa = (mantissa / 10).quantize(Decimal("0.01"))
            exponent += 1
    return f"{mantissa}e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"


@pytest.fixture(scope="module")
def separated() -> pd.DataFrame:
    # Bayes factors beyond the largest double: log(BF01) -761 (a-b), -991 (a-c).
    rng = np.random.default_rng(2)
    values = [rng.normal(m, 1, 3000) for m in (0.0, 1.0, 1.2)]
    return pd.DataFrame(
        {"g": np.repeat(["a", "b", "c"], 3000), "y": np.concatenate(values)}
    )


def test_bayes_factor_beyond_a_double_prints_from_its_log(
    separated: pd.DataFrame,
) -> None:
    two = betweenstats(separated[separated["g"] != "c"], "g", "y", type="bayes")
    assert two.bayes.bf10 == math.inf  # the stored double; log_bf01 has the value
    log_bf01 = two.bayes.log_bf01
    assert log_bf01 == pytest.approx(-761.29, abs=0.005)
    text = _bf10_from_log(log_bf01)
    assert text == "4.19e+330"
    assert two.text.apa == (
        f"Bayesian t-test (JZS prior, r = 0.707): BF10 = {text}, "
        "log(BF01) = -761.29, n = 6,000."
    )
    assert "Inf" not in two.text.alt_text
    assert StatsResult.from_json(two.to_json()) == two
    three = betweenstats(separated, "g", "y", type="bayes", results_subtitle=False)
    by_pair = {(p.group1, p.group2): p.bayes for p in three.pairwise}
    assert by_pair["a", "c"].bf10 == by_pair["a", "b"].bf10 == math.inf
    strongest = _bf10_from_log(by_pair["a", "c"].log_bf01)
    assert strongest == "2.02e+430"
    # The strongest pair is a-c (the largest -log BF01), not the first infinite one.
    alt = three.text.alt_text
    assert f"BF10 from 7.04e+09 (b vs c) to {strongest} (a vs c);" in alt
    assert f"a vs c, BF10 = {strongest};" in three.text.apa
    table = str(three)
    assert "Inf" not in table
    assert "2.02e+430" in table
    assert "4.19e+330" in table


def test_bayes_alt_text_without_brackets(plants: pd.DataFrame) -> None:
    drawn = betweenstats(
        plants, "group", "weight", type="bayes", results_subtitle=False
    )
    assert drawn.text.alt_text.endswith("brackets show log(BF01) for every pair.")
    hidden = betweenstats(
        plants,
        "group",
        "weight",
        type="bayes",
        results_subtitle=False,
        pairwise_display="none",
    )
    assert hidden.text.alt_text.endswith(
        "; no brackets are drawn (pairwise_display='none')."
    )
    # pairwise_display="none" was asked for: nothing to report (U11).
    assert "pairs_hidden" not in _codes(hidden)


def test_bayes_alt_text_shortens_long_levels() -> None:
    names = [
        f"Treatment with the experimental compound at a very high dose level number {i}"
        for i in (1, 2, 3)
    ]
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "g": np.repeat(names, 10),
            "y": np.concatenate([rng.normal(m, 1, 10) for m in (0.0, 1.0, 2.0)]),
        }
    )
    bayes = betweenstats(data, "g", "y", type="bayes", results_subtitle=False)
    robust = betweenstats(data, "g", "y", type="robust")
    for result in (bayes, robust):
        alt = result.text.alt_text
        assert alt.startswith("Plot comparing y across the 3 groups of g ("), alt
        assert len(alt) <= 400, alt
    short = "Treatment with the …"
    assert bayes.text.alt_text.endswith(
        f"BF10 from 0.67 ({short} vs {short}) to 659.96 ({short} vs {short}); "
        "brackets show log(BF01) for every pair."
    )


def test_robust_low_df_raises_for_any_number_of_pairs() -> None:
    # 9 groups of 2 values: 36 pairs, above WRS2's table (28), each with a Yuen df
    # below 2. SPEC sections 2 and 4: NotImplementedError whatever the count.
    nine = frame({f"g{i}": [float(i), float(i) + 1.0 + 0.1 * i] for i in range(9)})
    with pytest.raises(NotImplementedError, match="fewer than 2 degrees of freedom"):
        betweenstats(nine, "g", "y", type="robust")


def test_bayes_pair_errors_name_the_pair() -> None:
    a, b, c = [1.0, 2, 3, 4, 5], [2.0, 3, 4, 5, 6], [1e200] * 5
    with pytest.raises(ValueError, match="cannot be computed") as info:
        betweenstats(
            frame({"a": a, "b": b, "c": c}),
            "g",
            "y",
            type="bayes",
            results_subtitle=False,
        )
    message = str(info.value)
    assert message.startswith(
        "The Bayes factor of 'a' against 'c' cannot be computed: the group 'c' of "
        "x='g' is constant and the group 'a' of x='g' varies"
    )
    assert "sample a" not in message
    assert "sample b" not in message
    assert ".." not in message
    assert message.count("Check the data") == 1
    with pytest.raises(ValueError, match="is constant") as info:
        betweenstats(frame({"a": a, "c": c}), "g", "y", type="bayes")
    message = str(info.value)
    assert message.startswith("The group 'c' of x='g' is constant")
    assert ".." not in message
    assert message.count("Check the data") == 1
    assert message.endswith("since it keeps their ratio.")


def test_robust_infinite_akp_resamples_are_counted(teeth: pd.DataFrame) -> None:
    data = frame({"a": [0.0, 0, 0, 0, 0, 0, 1, 2], "b": [3.0, 4, 5, 6, 7, 8]})
    result = betweenstats(data, "g", "y", type="robust")
    es = result.effect_size
    assert es.infinite_resamples == 721
    assert es.conf_low == -math.inf  # kept, as WRS2's arithmetic gives it
    assert math.isfinite(es.estimate)
    assert "95% CI [-Inf, -2.45]" in result.text.apa
    assert result.warnings == (
        "721 of 2,000 bootstrap resamples give an infinite AKP robust standardized "
        "difference, because in them the first group's winsorized variance is 0 "
        "while the trimmed means differ. They are kept, so "
        "the confidence interval [-Inf, -2.45] can be unbounded. Pass a smaller "
        "tr, or use type='nonparametric'.",
    )
    (issue,) = [i for i in result.check() if i.code == "infinite_effect_size"]
    assert issue.severity == "warning"
    assert "721 of 2,000" in issue.message
    assert StatsResult.from_json(result.to_json()) == result
    clean = betweenstats(teeth, "supp", "len", type="robust")
    assert clean.effect_size.infinite_resamples == 0
    assert "infinite_effect_size" not in _codes(clean)


def test_robust_infinite_xi_resamples() -> None:
    data = frame(
        {
            "a": [1.0, 0, 1, 1, 1, 1, 1, 1, 0],
            "b": [0.0, 0, -1, -1, -1],
            "c": [0.0, -1, -1, 0, 0, 0, 0],
        }
    )
    result = betweenstats(data, "g", "y", type="robust")
    es = result.effect_size
    assert es.infinite_resamples == 162
    assert es.conf_high == math.inf
    assert math.isfinite(es.estimate)  # the median of the resamples (unequal n)
    assert [w for w in result.warnings if "infinite" in w] == [
        "162 of 2,000 bootstrap resamples give an infinite explanatory effect size "
        "xi, because in them the pooled winsorized variance is 0 while the trimmed "
        "means differ. They are kept, so the confidence "
        "interval [0.65, Inf] can be unbounded. Pass a smaller tr, or use "
        "type='nonparametric'."
    ]
    assert "infinite_effect_size" in _codes(result)


def test_robust_infinite_xi_estimate() -> None:
    # Equal sizes: every group varies after winsorizing, the pooled values do not
    # (40 of 200 at -1 and 40 at 1, winsorized to 0), and the trimmed means differ:
    # xi = sqrt(var / 0) = inf, as in WRS2.
    low = [-1.0] * 20 + [0.0] * 30
    high = [0.0] * 30 + [1.0] * 20
    result = betweenstats(
        frame({"a": low, "b": high, "c": low, "d": high}), "g", "y", type="robust"
    )
    es = result.effect_size
    assert es.estimate == math.inf
    assert es.infinite_resamples == 593
    assert "xi = Inf, 95% CI [0.43, Inf]" in result.text.apa
    assert result.warnings[0] == (
        "The explanatory effect size xi is infinite: the pooled winsorized variance "
        "is 0 while the trimmed means differ in the data."
    )
    (issue,) = [i for i in result.check() if i.code == "infinite_effect_size"]
    assert issue.message.startswith("The explanatory effect size xi is infinite.")


# --------------------------------------------------------------------------- #
# an estimate outside its own bootstrap interval (U6)
# --------------------------------------------------------------------------- #


def _noise_groups() -> pd.DataFrame:
    names = [f"grp{i:02d}" for i in range(12)]
    y = np.random.default_rng(2).normal(size=72)
    return pd.DataFrame({"g": np.repeat(names, 6), "y": y})


@pytest.mark.parametrize(
    ("kind", "words"),
    [
        (
            "np",
            "The rank epsilon squared, 0.074, lies outside its own 95% CI "
            "[0.107, 1.000].",
        ),
        (
            "r",
            "The explanatory effect size xi, 0.325, lies outside its own 95% CI "
            "[0.345, 0.786].",
        ),
    ],
)
def test_estimate_outside_its_interval(kind: str, words: str) -> None:
    # Twelve groups of six values of pure noise: the effect is small next to the
    # bootstrap's own bias, and the percentile interval misses the estimate.
    result = betweenstats(_noise_groups(), "g", "y", type=kind)
    es = result.effect_size
    assert not es.conf_low <= es.estimate <= es.conf_high
    (issue,) = [i for i in result.check() if i.code == "estimate_outside_interval"]
    assert issue.severity == "warning"
    assert issue.message.startswith(words)
    assert "percentile bootstrap" in issue.message
    assert "small next to the bootstrap's own bias" in issue.message
    assert "rough range" in issue.fix
    assert "p-value" in issue.fix
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert embedded.check() == result.check()


def test_estimate_inside_its_interval(plants: pd.DataFrame) -> None:
    for kind in ("p", "np", "r"):
        result = betweenstats(plants, "group", "weight", type=kind)
        es = result.effect_size
        assert es.conf_low <= es.estimate <= es.conf_high
        assert "estimate_outside_interval" not in _codes(result)
