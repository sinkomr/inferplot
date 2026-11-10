"""``ip.analyze.betweenstats``: signature, argument checks, rules and result API.

Numbers against the R oracle are in ``tests/test_betweenstats_parity.py``; this file
checks the behaviour around them: every argument reaches its validator, the
two-group and three-group rules (alternative, Bayes caption, pairwise display), the
errors for data the parametric methods cannot use, warnings, settings and the
result's export methods on real results.
"""

from __future__ import annotations

import inspect
import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from tests.oracle_fixtures import DATASET_DIR

betweenstats: Any = ip.analyze.betweenstats

K3_BAYES_WARNING = (
    "The Bayes factor caption for three or more groups arrives in inferplot 0.2."
)


@pytest.fixture(scope="module")
def plants() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "PlantGrowth.csv")


@pytest.fixture(scope="module")
def teeth() -> pd.DataFrame:
    return pd.read_csv(DATASET_DIR / "ToothGrowth.csv")


@pytest.fixture(scope="module")
def anova(plants: pd.DataFrame) -> StatsResult:
    return betweenstats(plants, "group", "weight")


@pytest.fixture(scope="module")
def two(teeth: pd.DataFrame) -> StatsResult:
    return betweenstats(teeth, "supp", "len")


def frame(groups: dict[str, list[float]]) -> pd.DataFrame:
    rows = [(g, v) for g, values in groups.items() for v in values]
    return pd.DataFrame(rows, columns=["g", "y"])


# --------------------------------------------------------------------------- #
# signature and keyword names
# --------------------------------------------------------------------------- #

DEFAULTS = {
    "type": "parametric",
    "pairwise_display": "significant",
    "p_adjust_method": "holm",
    "pairwise_alpha": 0.05,
    "conf_level": 0.95,
    "digits": 2,
    "tr": 0.2,
    "bf_prior": 0.707,
    "bf_message": None,
    "results_subtitle": True,
    "alternative": "two-sided",
    "seed": 0,
    "order": None,
}


def test_signature() -> None:
    signature = inspect.signature(ip.analyze.betweenstats)
    params = list(signature.parameters.values())
    kind = inspect.Parameter
    assert [p.name for p in params[:3]] == ["data", "x", "y"]
    assert all(p.kind is kind.POSITIONAL_OR_KEYWORD for p in params[:3])
    assert all(p.default is kind.empty for p in params[:3])
    rest = params[3:]
    assert all(p.kind is kind.KEYWORD_ONLY for p in rest)
    assert {p.name: p.default for p in rest} == DEFAULTS
    assert [p.name for p in rest] == list(DEFAULTS)
    assert ip.analyze.betweenstats.__name__ == "betweenstats"
    assert ip.analyze.betweenstats.__doc__


def test_bf_prior_default_is_the_number_0_707() -> None:
    default = inspect.signature(ip.analyze.betweenstats).parameters["bf_prior"].default
    assert default == 0.707
    assert default != math.sqrt(2) / 2


def test_unknown_keyword_suggests_closest(plants: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match=r"Did you mean conf_level=0\.9\?"):
        betweenstats(plants, "group", "weight", conf_levle=0.9)
    with pytest.raises(TypeError, match="Its arguments are"):
        betweenstats(plants, "group", "weight", zzzz=1)


def test_dotted_keyword_names_snake_case(plants: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match=r"Write conf_level=0\.9"):
        betweenstats(plants, "group", "weight", **{"conf.level": 0.9})


@pytest.mark.parametrize(
    ("keyword", "error", "message"),
    [
        ("var.equal", TypeError, "removed in ggstatsplot 1.0.0"),
        ("nboot", TypeError, "2,000 seeded"),
        ("effsize.type", TypeError, "Hedges' g and partial omega squared"),
        ("k", TypeError, "renamed digits"),
        ("pairwise.comparisons", TypeError, "pairwise_display='none'"),
        ("output", TypeError, "ip.analyze.betweenstats"),
        ("messages", TypeError, "never prints"),
        ("hue", TypeError, "seaborn"),
        ("ci", TypeError, "conf_level"),
        ("subject_id", TypeError, "independent groups"),
        ("grouping_var", NotImplementedError, "v0.2"),
        ("grouping.var", NotImplementedError, "v0.2"),
    ],
)
def test_removed_and_borrowed_keywords(
    plants: pd.DataFrame, keyword: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        betweenstats(plants, "group", "weight", **{keyword: True})


def test_extra_positional_arguments(plants: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match=r"type='np'"):
        betweenstats(plants, "group", "weight", "np")


# --------------------------------------------------------------------------- #
# value validators
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("conf_level", 1.5, ValueError, "strictly between 0 and 1"),
        ("conf_level", 95, ValueError, "not a percentage"),
        ("conf_level", True, TypeError, "must be a number"),
        ("digits", -1, ValueError, ">= 0"),
        ("digits", 2.0, TypeError, "whole number"),
        ("alternative", "two.sided", ValueError, "two-sided"),
        ("alternative", "bigger", ValueError, "use one of"),
        ("p_adjust_method", "Holm", ValueError, "holm"),
        ("p_adjust_method", "tukey", ValueError, "use one of"),
        ("pairwise_display", "sig", ValueError, "use one of"),
        ("pairwise_display", None, TypeError, "pairwise_display='none'"),
        ("pairwise_alpha", 0.0, ValueError, "strictly between 0 and 1"),
        ("pairwise_alpha", "0.05", TypeError, "must be a number"),
        ("bf_prior", 0.0, ValueError, "greater than 0"),
        ("bf_prior", math.inf, ValueError, "finite"),
        ("bf_message", "yes", TypeError, "None, True or False"),
        ("bf_message", 1, TypeError, "None, True or False"),
        ("results_subtitle", None, TypeError, "True or False"),
        ("seed", -1, ValueError, ">= 0"),
        ("seed", 1.5, TypeError, "whole number"),
        ("tr", 0.5, ValueError, "0 <= tr < 0.5"),
        ("type", "anova", ValueError, "use one of"),
        ("type", None, TypeError, "must be a string"),
        ("order", "ctrl", TypeError, "list of level names"),
        ("order", ["ctrl", "trt1"], ValueError, "missing from order"),
        ("order", ["ctrl", "trt1", "trt2", "x"], ValueError, "not levels of x"),
        ("order", ["ctrl", "ctrl", "trt1", "trt2"], ValueError, "more than once"),
    ],
)
def test_value_validators(
    plants: pd.DataFrame,
    keyword: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        betweenstats(plants, "group", "weight", **{keyword: value})


def test_type_aliases(plants: pd.DataFrame, anova: StatsResult) -> None:
    result = betweenstats(plants, "group", "weight", type="p")
    assert result.type == "parametric"
    assert result == anova


@pytest.mark.parametrize(
    ("value", "canonical"),
    [
        ("nonparametric", "nonparametric"),
        ("np", "nonparametric"),
        ("robust", "robust"),
        ("r", "robust"),
        ("bayes", "bayes"),
        ("bf", "bayes"),
    ],
)
def test_every_type_and_alias_is_implemented(
    teeth: pd.DataFrame, value: str, canonical: str
) -> None:
    # tests/test_betweenstats_types_api.py checks each type in detail.
    assert betweenstats(teeth, "supp", "len", type=value).type == canonical


@pytest.mark.parametrize("tr", [0.0, 0.1, 0.25, 0.49])
def test_tr_other_than_the_default_raises_for_parametric(
    plants: pd.DataFrame, tr: float
) -> None:
    # ggstatsplot ignores tr outside type "robust"; inferplot never drops an argument.
    with pytest.raises(ValueError, match="trims nothing") as info:
        betweenstats(plants, "group", "weight", tr=tr)
    assert f"tr={tr!r}" in str(info.value)
    assert betweenstats(plants, "group", "weight", tr=0.2).settings.tr == 0.2


def test_type_checked_before_the_data() -> None:
    with pytest.raises(ValueError, match="use one of"):
        betweenstats(pd.DataFrame({"a": [1]}), "nope", "nope", type="bootstrap")
    with pytest.raises(ValueError, match="trims nothing"):
        betweenstats(pd.DataFrame({"a": [1]}), "nope", "nope", type="bf", tr=0.1)


# --------------------------------------------------------------------------- #
# data errors and validity
# --------------------------------------------------------------------------- #


def test_data_errors(plants: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="long format"):
        betweenstats([1, 2, 3], "group", "weight")
    with pytest.raises(KeyError, match="weight"):
        betweenstats(plants, "group", "wieght")
    with pytest.raises(ValueError, match="at least 2 groups"):
        betweenstats(frame({"a": [1.0, 2.0, 3.0]}), "g", "y")
    with pytest.raises(ValueError, match="infinite"):
        betweenstats(frame({"a": [1.0, math.inf], "b": [2.0, 3.0]}), "g", "y")
    with pytest.raises(TypeError, match="must be numeric"):
        betweenstats(pd.DataFrame({"g": ["a", "b"], "y": ["1", "2"]}), "g", "y")


@pytest.mark.parametrize("k", [2, 3])
def test_groups_need_two_observations(k: int) -> None:
    groups = {"a": [1.0, 2.0, 3.0], "b": [4.0], "c": [5.0, 7.0]}
    data = frame(dict(list(groups.items())[:k]))
    with pytest.raises(ValueError, match=r"at least 2 observations.*'b' \(n = 1\)"):
        betweenstats(data, "g", "y")


def test_two_constant_groups_raise() -> None:
    data = frame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0]})
    with pytest.raises(ValueError, match=r"Both groups.*'a' and 'b'.*standard error"):
        betweenstats(data, "g", "y")


def test_one_constant_group_of_two_is_fine() -> None:
    data = frame({"a": [1.0, 1.0, 1.0], "b": [2.0, 3.0, 5.0]})
    result = betweenstats(data, "g", "y")
    expected = stats.welch_t([1.0, 1.0, 1.0], [2.0, 3.0, 5.0])
    assert result.test.statistic == expected.t
    assert result.bayes is not None


@pytest.mark.parametrize("constant", [["b"], ["a", "c"]])
def test_constant_group_of_three_raises(constant: list[str]) -> None:
    groups = {"a": [1.0, 2.0, 4.0], "b": [3.0, 5.0], "c": [6.0, 8.0, 9.0]}
    for level in constant:
        groups[level] = [7.0] * len(groups[level])
    with pytest.raises(ValueError, match="variance") as info:
        betweenstats(frame(groups), "g", "y")
    for level in constant:
        assert repr(level) in str(info.value)


def test_empty_declared_level_is_named_in_warnings() -> None:
    data = frame({"a": [1.0, 2.0, 4.0], "b": [3.0, 5.0, 4.0]})
    data["g"] = pd.Categorical(data["g"], categories=["a", "zero", "b"])
    result = betweenstats(data, "g", "y")
    assert result.levels == ("a", "b")
    assert any("'zero'" in w and "left out" in w for w in result.warnings)


_VANISHED = (
    "The level 'c' of 'g' was left out of the analysis: every row of it has a "
    "missing 'y'."
)


@pytest.mark.parametrize("kind", ["pandas text", "polars", "pyarrow", "integer"])
def test_a_level_whose_y_is_all_missing_is_named_in_warnings(kind: str) -> None:
    # Every row of 'c' misses y: Welch's ANOVA becomes a t-test, and the result
    # says why, whatever the column type (a declared level says it already).
    g = ["a"] * 3 + ["b"] * 3 + ["c"] * 2
    y = [1.0, 2.0, 4.0, 3.0, 5.0, 4.0, None, None]
    data: Any = pd.DataFrame({"g": g, "y": y})
    if kind == "polars":
        data = pl.DataFrame({"g": g, "y": y})
    elif kind == "pyarrow":
        import pyarrow as pa

        data = pa.table({"g": pa.array(g).dictionary_encode(), "y": y})
    elif kind == "integer":
        data = pd.DataFrame({"g": [{"a": 1, "b": 2, "c": 3}[v] for v in g], "y": y})
    result = betweenstats(data, "g", "y")
    expected = _VANISHED if kind != "integer" else _VANISHED.replace("'c'", "'3'")
    assert result.test.method == "Welch Two Sample t-test"
    assert result.warnings == (expected,)
    two_left = betweenstats(data, "g", "y", order=list(result.levels)[::-1])
    assert two_left.warnings == (expected,)
    gone = 3 if kind == "integer" else "c"
    with pytest.raises(ValueError, match="have no rows with both x and y present"):
        betweenstats(data, "g", "y", order=[*result.levels, gone])


def test_levels_left_out_name_both_and_one_level_left_raises() -> None:
    data = pd.DataFrame(
        {"g": ["a", "a", "b", "c", "c"], "y": [1.0, 2.0, None, None, None]}
    )
    with pytest.raises(ValueError, match=r"needs at least 2 groups with data; found 1"):
        betweenstats(data, "g", "y")
    data = pd.DataFrame(
        {
            "g": ["a"] * 4 + ["b"] * 4 + ["c", "d", "d"],
            "y": [1.0, 2.0, 4.0, 3.0, 3.0, 5.0, 4.0, 6.0, None, None, None],
        }
    )
    assert betweenstats(data, "g", "y").warnings == (
        "The levels 'c' and 'd' of 'g' were left out of the analysis: every row of "
        "them has a missing 'y'.",
    )


def test_unordered_dictionary_order_is_named_in_warnings() -> None:
    import pyarrow as pa

    g = ["treatment"] * 3 + ["control"] * 3
    table = pa.table(
        {"g": pa.array(g).dictionary_encode(), "y": [3.0, 4.0, 6.0, 1.0, 2.0, 4.0]}
    )
    result = betweenstats(table, "g", "y")
    assert result.levels == ("control", "treatment")
    assert result.warnings == (
        "The levels of 'g' are in sorted order ('control', 'treatment'), not in the "
        "order of its pyarrow dictionary ('treatment', 'control'): a dictionary that "
        "is not marked ordered, as dictionary_encode(), pyarrow.csv and polars' "
        ".to_arrow() make, lists values in the order they first appear in the rows, "
        "so it does not declare an order. To choose the order, pass order= (a list "
        "of the levels) or mark the dictionary ordered.",
    )
    assert betweenstats(table, "g", "y", order=["treatment", "control"]).warnings == ()
    # Many levels are shortened.
    names = [f"L{i:02d}" for i in range(12)][::-1]
    many = pa.table(
        {
            "g": pa.array(np.repeat(names, 3)).dictionary_encode(),
            "y": np.arange(36.0) % 7,
        }
    )
    (note,) = betweenstats(many, "g", "y", type="nonparametric").warnings
    assert "('L00', 'L01', 'L02', 'L03', 'L04', ... (12 levels))" in note
    assert "('L11', 'L10', 'L09', 'L08', 'L07', ... (12 levels))" in note


def test_levels_that_differ_only_in_whitespace_are_kept_and_named() -> None:
    data = pd.DataFrame(
        {"g": ["a", "a ", " a", "b"] * 3, "y": [1.0, 2.0, 4.0, 3.0, 5.0, 4.0] * 2}
    )
    result = betweenstats(data, "g", "y", type="nonparametric")
    assert result.levels == (" a", "a", "a ", "b")  # never merged
    assert result.warnings == (
        "The levels ' a', 'a' and 'a ' of 'g' differ only in spaces at their start "
        "or end, so they are analyzed as separate groups whose labels look alike. "
        "If they are the same group, remove the spaces first, for example with "
        ".str.strip() in pandas or .str.strip_chars() in polars.",
    )


def test_rows_with_missing_values_are_counted() -> None:
    data = pd.DataFrame(
        {"g": ["a", "a", "a", None, "b", "b", "b"], "y": [1, 2, 3, 4, 5, None, 9.0]}
    )
    result = betweenstats(data, "g", "y")
    assert (result.n_obs, result.n_dropped) == (5, 2)
    assert result.values == ((1.0, 2.0, 3.0), (5.0, 9.0))
    assert "rows_dropped" in {issue.code for issue in result.check()}


# --------------------------------------------------------------------------- #
# two groups
# --------------------------------------------------------------------------- #


def test_two_groups(two: StatsResult, teeth: pd.DataFrame) -> None:
    oj = teeth.loc[teeth["supp"] == "OJ", "len"].to_numpy(dtype=float)
    vc = teeth.loc[teeth["supp"] == "VC", "len"].to_numpy(dtype=float)
    assert two.levels == ("OJ", "VC")
    assert two.test is not None
    assert two.effect_size is not None
    assert two.values == (tuple(oj.tolist()), tuple(vc.tolist()))
    welch = stats.welch_t(oj, vc)
    assert (two.test.name, two.test.label) == ("welch_t", "Welch's t-test")
    assert two.test.method == "Welch Two Sample t-test"
    assert (two.test.statistic, two.test.df_error) == (welch.t, welch.df)
    assert two.test.df is None
    g = stats.hedges_g(oj, vc)
    es = two.effect_size
    assert (es.name, es.label, es.ci_method) == (
        "hedges_g",
        "Hedges' g",
        "noncentral t",
    )
    assert (es.estimate, es.conf_low, es.conf_high) == (
        g.estimate,
        g.conf_low,
        g.conf_high,
    )
    assert es.sidedness == "two-sided"
    bf10, log_bf01 = stats.jzs_bf10_two_sample(oj, vc, prior_scale=0.707)
    assert two.bayes is not None
    assert (two.bayes.bf10, two.bayes.log_bf01) == (bf10, log_bf01)
    assert (two.bayes.prior, two.bayes.prior_scale) == ("cauchy", 0.707)
    assert two.bayes.test == "jzs_two_sample"
    assert two.pairwise == ()
    assert two.warnings == ()
    assert two.text.caption is not None
    assert "BF_01" in two.text.caption
    assert [d.value for d in two.descriptives] == [stats.mean(oj), stats.mean(vc)]


@pytest.mark.parametrize("side", ["less", "greater"])
def test_one_sided_two_groups(teeth: pd.DataFrame, side: str) -> None:
    result = betweenstats(teeth, "supp", "len", alternative=side)
    assert result.test.alternative == side
    assert result.effect_size.sidedness == side
    far = (
        result.effect_size.conf_high
        if side == "greater"
        else result.effect_size.conf_low
    )
    assert math.isinf(far)
    two_sided = betweenstats(teeth, "supp", "len")
    assert result.bayes == two_sided.bayes  # the Bayes factor stays two-sided
    (warning,) = result.warnings
    assert "two-sided" in warning
    assert f"alternative={side!r}" in warning
    assert StatsResult.from_json(result.to_json()) == result


def test_bf_prior_reaches_the_bayes_factor(teeth: pd.DataFrame) -> None:
    result = betweenstats(teeth, "supp", "len", bf_prior=0.8)
    assert result.bayes is not None
    assert result.bayes.prior_scale == 0.8
    assert result.settings.bf_prior == 0.8
    assert result.text.caption is not None
    assert result.text.caption.endswith("0.80")


@pytest.mark.parametrize(
    ("bf_message", "results_subtitle", "computed"),
    [(None, True, True), (True, True, True), (False, True, False),
     (None, False, False), (False, False, False)],
)  # fmt: skip
def test_two_group_bayes_rules(
    teeth: pd.DataFrame, bf_message: bool | None, results_subtitle: bool, computed: bool
) -> None:
    result = betweenstats(
        teeth,
        "supp",
        "len",
        bf_message=bf_message,
        results_subtitle=results_subtitle,
    )
    assert (result.bayes is not None) == computed
    assert result.warnings == ()
    if computed:
        assert result.text.caption is not None
    else:
        assert result.text.caption is None
    assert (result.text.subtitle is not None) == results_subtitle
    citations = " ".join(result.provenance.citations)
    assert ("Rouder" in citations) == computed


@pytest.mark.parametrize("k", [2, 3])
def test_bf_message_true_without_subtitle_raises(
    teeth: pd.DataFrame, plants: pd.DataFrame, k: int
) -> None:
    data, x, y = (teeth, "supp", "len") if k == 2 else (plants, "group", "weight")
    with pytest.raises(ValueError, match="results_subtitle"):
        betweenstats(data, x, y, bf_message=True, results_subtitle=False)


# --------------------------------------------------------------------------- #
# three or more groups
# --------------------------------------------------------------------------- #


def test_three_groups(anova: StatsResult, plants: pd.DataFrame) -> None:
    groups = [
        plants.loc[plants["group"] == g, "weight"].to_numpy(dtype=float)
        for g in ("ctrl", "trt1", "trt2")
    ]
    welch = stats.welch_anova(groups)
    test = anova.test
    assert test is not None
    assert anova.effect_size is not None
    assert (test.name, test.label, test.statistic_name) == (
        "welch_anova",
        "Welch's ANOVA",
        "F",
    )
    assert test.method == "One-way analysis of means (not assuming equal variances)"
    assert (test.statistic, test.df, test.df_error, test.p_value) == (
        welch.f,
        welch.df1,
        welch.df2,
        welch.p_value,
    )
    omega = stats.partial_omega_squared(welch.f, welch.df1, welch.df2)
    es = anova.effect_size
    assert (es.name, es.label, es.ci_method, es.sidedness) == (
        "partial_omega_squared",
        "partial omega squared",
        "noncentral F",
        "greater",
    )
    assert (es.estimate, es.conf_low, es.conf_high) == (
        omega.estimate,
        omega.conf_low,
        omega.conf_high,
    )
    assert anova.bayes is None
    assert anova.warnings == (K3_BAYES_WARNING,)
    pairs = stats.games_howell(groups)
    assert [(p.group1, p.group2) for p in anova.pairwise] == [
        ("ctrl", "trt1"),
        ("ctrl", "trt2"),
        ("trt1", "trt2"),
    ]
    for pair, expected in zip(anova.pairwise, pairs, strict=True):
        assert (pair.statistic, pair.df) == (expected.statistic, expected.df)
        assert (pair.p_value, pair.p_adjusted) == (
            expected.p_value,
            expected.p_adjusted,
        )
        assert (pair.statistic_name, pair.test, pair.test_label) == (
            "q",
            "games_howell",
            "Games-Howell",
        )
        assert pair.adjust_method == "holm"


@pytest.mark.parametrize("side", ["less", "greater"])
def test_one_sided_with_three_groups_raises(plants: pd.DataFrame, side: str) -> None:
    with pytest.raises(ValueError, match="only to two groups"):
        betweenstats(plants, "group", "weight", alternative=side)


@pytest.mark.parametrize(
    ("bf_message", "results_subtitle", "warned"),
    [(None, True, True), (False, True, False), (None, False, False),
     (False, False, False)],
)  # fmt: skip
def test_three_group_bayes_rules(
    plants: pd.DataFrame, bf_message: bool | None, results_subtitle: bool, warned: bool
) -> None:
    result = betweenstats(
        plants,
        "group",
        "weight",
        bf_message=bf_message,
        results_subtitle=results_subtitle,
    )
    assert result.bayes is None
    assert result.warnings == ((K3_BAYES_WARNING,) if warned else ())
    assert result.text.caption is not None
    assert "BF" not in result.text.caption


def test_bf_message_true_with_three_groups_is_v02(plants: pd.DataFrame) -> None:
    with pytest.raises(NotImplementedError, match=r"inferplot 0\.2"):
        betweenstats(plants, "group", "weight", bf_message=True)


@pytest.mark.parametrize(
    ("display", "canonical"),
    [
        ("significant", "significant"),
        ("s", "significant"),
        ("non-significant", "non-significant"),
        ("ns", "non-significant"),
        ("all", "all"),
        ("none", "none"),
    ],
)
def test_pairwise_display(plants: pd.DataFrame, display: str, canonical: str) -> None:
    result = betweenstats(plants, "group", "weight", pairwise_display=display)
    assert result.settings.pairwise_display == canonical
    assert len(result.pairwise) == 3
    for pair in result.pairwise:
        expected = {
            "significant": pair.significant,
            "non-significant": not pair.significant,
            "all": True,
            "none": False,
        }[canonical]
        assert pair.displayed is expected
    codes = {issue.code for issue in result.check()}
    # An explicit "none" is what was asked for (U11); the others report hidden pairs.
    hidden = not all(p.displayed for p in result.pairwise)
    assert ("pairs_hidden" in codes) == (hidden and canonical != "none")


def test_pairwise_alpha_is_strict(plants: pd.DataFrame, anova: StatsResult) -> None:
    holm = [pair.p_adjusted for pair in anova.pairwise if pair.p_adjusted]
    assert len(holm) == len(anova.pairwise)
    assert [pair.significant for pair in anova.pairwise] == [False, False, False]
    # alpha equal to a pair's adjusted p: not significant (strictly below)
    at = betweenstats(plants, "group", "weight", pairwise_alpha=min(holm))
    assert not any(pair.significant for pair in at.pairwise)
    above = betweenstats(plants, "group", "weight", pairwise_alpha=0.1)
    assert [pair.significant for pair in above.pairwise] == [p < 0.1 for p in holm]
    assert [pair.displayed for pair in above.pairwise] == [p < 0.1 for p in holm]
    assert above.settings.pairwise_alpha == 0.1


@pytest.mark.parametrize("method", list(stats.P_ADJUST_METHODS))
def test_p_adjust_method(plants: pd.DataFrame, method: str) -> None:
    result = betweenstats(plants, "group", "weight", p_adjust_method=method)
    raw = [pair.p_value for pair in result.pairwise]
    expected = stats.p_adjust(raw, method)  # type: ignore[arg-type]
    assert [pair.p_adjusted for pair in result.pairwise] == list(expected)
    assert {pair.adjust_method for pair in result.pairwise} == {method}
    assert result.settings.p_adjust_method == method
    citations = " ".join(result.provenance.citations)
    assert "Games, P. A. & Howell" in citations
    names = {
        "holm": "Holm, S.",
        "hochberg": "Hochberg, Y.",
        "hommel": "Hommel, G.",
        "bonferroni": "Bonferroni",
        "BH": "Benjamini, Y. & Hochberg",
        "fdr": "Benjamini, Y. & Hochberg",
        "BY": "Benjamini, Y. & Yekutieli",
    }
    if method in names:
        assert names[method] in citations
    else:
        assert "adjustment" not in citations


def test_level_order_sets_orientation_and_rows_sort_as_text(
    plants: pd.DataFrame,
) -> None:
    result = betweenstats(plants, "group", "weight", order=["trt2", "ctrl", "trt1"])
    assert result.levels == ("trt2", "ctrl", "trt1")
    # ggstatsplot's rows: group1 earlier in the level order, sorted by group1 then
    # group2 as text (fixture PlantGrowth-declared)
    assert [(p.group1, p.group2) for p in result.pairwise] == [
        ("ctrl", "trt1"),
        ("trt2", "ctrl"),
        ("trt2", "trt1"),
    ]
    means = {d.level: d.value for d in result.descriptives}
    for pair in result.pairwise:
        assert math.copysign(1.0, pair.statistic) == math.copysign(
            1.0, means[pair.group2] - means[pair.group1]
        )


def test_rows_sort_by_code_point() -> None:
    data = frame(
        {
            "b": [1.0, 2.0, 4.0],
            "B": [2.0, 3.0, 7.0],
            "a": [3.0, 5.0, 6.0],
            "10": [1.0, 4.0, 4.5],
            "9": [2.0, 2.5, 3.0],
        }
    )
    result = betweenstats(data, "g", "y")
    assert result.levels == ("10", "9", "B", "a", "b")
    pairs = [(p.group1, p.group2) for p in result.pairwise]
    assert pairs == sorted(pairs)
    assert len(pairs) == 10


# --------------------------------------------------------------------------- #
# settings, provenance and the result API
# --------------------------------------------------------------------------- #


def test_settings_record_canonical_values(plants: pd.DataFrame) -> None:
    result = betweenstats(
        plants,
        "group",
        "weight",
        type="p",
        pairwise_display="ns",
        p_adjust_method="BY",
        pairwise_alpha=0.1,
        conf_level=0.9,
        digits=3,
        tr=0.2,
        bf_prior=0.5,
        bf_message=False,
        seed=7,
    )
    settings = result.settings
    assert result.type == "parametric"
    assert settings.pairwise_display == "non-significant"
    assert settings.p_adjust_method == "BY"
    assert (settings.pairwise_alpha, settings.conf_level) == (0.1, 0.9)
    assert (settings.digits, settings.tr, settings.bf_prior) == (3, 0.2, 0.5)
    assert (settings.bf_message, settings.results_subtitle) == (False, True)
    assert (settings.seed, result.provenance.seed) == (7, 7)
    assert settings.alternative == "two-sided"
    assert result.effect_size.conf_level == 0.9
    assert result.text.subtitle is not None
    assert "CI_90%" in result.text.subtitle


def test_float32_arguments_read_as_they_print(teeth: pd.DataFrame) -> None:
    # A float32 0.9 is 0.8999999761581421 as a double; it is read as the 0.9 it
    # prints as, so the settings, warnings and texts show 0.9 (STATUS U12).
    result = betweenstats(
        teeth,
        "supp",
        "len",
        conf_level=np.float32(0.9),
        pairwise_alpha=np.float32(0.01),
        bf_prior=np.float32(0.5),
        bf_message=False,
    )
    settings = result.settings
    assert (settings.conf_level, settings.pairwise_alpha, settings.bf_prior) == (
        0.9,
        0.01,
        0.5,
    )
    assert result.effect_size is not None
    assert result.effect_size.conf_level == 0.9
    assert result.text.subtitle is not None
    assert "CI_90%" in result.text.subtitle
    assert any("pairwise_alpha=0.01" in w for w in result.warnings)
    assert any("bf_prior=0.5" in w for w in result.warnings)
    dumped = result.to_json()
    assert "0.8999999" not in dumped
    assert "0.0099999" not in dumped
    robust = betweenstats(teeth, "supp", "len", type="robust", tr=np.float32(0.1))
    assert robust.settings.tr == 0.1
    assert "0.1000000" not in robust.to_json()


def test_result_fields(anova: StatsResult) -> None:
    assert (anova.analysis, anova.design, anova.type) == (
        "betweenstats",
        "between",
        "parametric",
    )
    assert (anova.x, anova.y, anova.subject_id) == ("group", "weight", None)
    assert (anova.n_obs, anova.n_dropped, anova.dropped_subjects) == (30, 0, 0)
    assert [(d.level, d.n, d.centrality) for d in anova.descriptives] == [
        ("ctrl", 10, "mean"),
        ("trt1", 10, "mean"),
        ("trt2", 10, "mean"),
    ]
    provenance = anova.provenance
    assert provenance.parity_target == "ggstatsplot 1.1.4"
    assert provenance.inferplot == ip.__version__
    citations = " ".join(provenance.citations)
    for author in ("Welch, B. L. (1951)", "Olejnik", "Steiger", "Games", "Holm"):
        assert author in citations
    assert "Rouder" not in citations
    assert "Hedges" not in citations


def test_two_group_citations(two: StatsResult) -> None:
    citations = " ".join(two.provenance.citations)
    for author in ("Welch, B. L. (1947)", "Hedges", "Delacre", "Steiger", "Rouder"):
        assert author in citations
    assert "Games" not in citations


@pytest.mark.parametrize("name", ["anova", "two"])
def test_json_round_trip(name: str, request: pytest.FixtureRequest) -> None:
    result: StatsResult = request.getfixturevalue(name)
    text = result.to_json()
    json.loads(text)  # strict JSON
    assert StatsResult.from_json(text) == result
    assert StatsResult.from_dict(result.to_dict()) == result


def test_to_frame(anova: StatsResult, two: StatsResult) -> None:
    pandas_frame = anova.to_frame("pandas")
    assert isinstance(pandas_frame, pd.DataFrame)
    assert len(pandas_frame) == 4
    assert list(pandas_frame["test"]) == ["welch_anova"] + ["games_howell"] * 3
    assert anova.test is not None
    assert pandas_frame.loc[0, "statistic"] == anova.test.statistic
    polars_frame = anova.to_frame("polars")
    assert isinstance(polars_frame, pl.DataFrame)
    assert polars_frame.height == 4
    assert polars_frame["p_adjusted"].to_list()[1:] == [
        pair.p_adjusted for pair in anova.pairwise
    ]
    two_frame = two.to_frame("pandas")
    assert len(two_frame) == 1
    assert two.bayes is not None
    assert two_frame.loc[0, "bf10"] == two.bayes.bf10


def test_str(anova: StatsResult, two: StatsResult) -> None:
    assert str(two) == two.text.apa
    assert str(two).startswith("Welch's t-test: t(55.31) = 1.92, p = .061")
    text = str(anova)
    assert text.startswith(anova.text.apa)
    assert "Pairwise comparisons (Games-Howell):" in text
    assert "ctrl" in text
    assert "trt2" in text


def test_numpy_and_integer_inputs() -> None:
    data = pd.DataFrame(
        {"g": np.array([1, 1, 1, 2, 2, 2, 3, 3]), "y": np.arange(8, dtype=np.int64)}
    )
    data.loc[7, "y"] = 20
    result = betweenstats(data, "g", "y")
    assert result.levels == ("1", "2", "3")
    assert all(type(v) is float for vs in result.values for v in vs)
    assert type(result.test.statistic) is float
    assert StatsResult.from_json(result.to_json()) == result


# --------------------------------------------------------------------------- #
# extreme data: well-separated groups, huge and tiny scales


@pytest.mark.parametrize("sd", [1e-2, 1e-4, 1e-6, 1e-8, 1e-12])
def test_well_separated_low_noise_groups(sd: float) -> None:
    # API-1: groups at 10, 20 and 30 with little noise give Welch F of 1e7 to 1e27;
    # before, sd = 1e-4 ran for minutes and sd = 1e-8 raised ZeroDivisionError.
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "g": np.repeat(["a", "b", "c"], 5),
            "y": np.repeat([10.0, 20.0, 30.0], 5) + rng.normal(0.0, sd, 15),
        }
    )
    start = time.perf_counter()
    result = betweenstats(data, "g", "y")
    assert time.perf_counter() - start < 5.0
    assert result.test.statistic > 1e6
    es = result.effect_size
    assert 0.99 < es.conf_low <= es.estimate <= 1.0
    assert es.conf_high == 1.0
    assert "CI_95% [1.00, 1.00]" in result.text.subtitle


@pytest.mark.parametrize("base", [1e-300, 1e-160, 1e200, 1e300])
def test_extreme_magnitudes_give_the_unit_scale_result(base: float) -> None:
    # API-2: 1e-300 was called constant (its variance underflowed), 1e-160 raised
    # ZeroDivisionError, 1e200 OverflowError and 1e300 "df must be > 0; got nan".
    unit = betweenstats(frame({"a": [1.0, 2.0, 3.0], "b": [4.0, 6.0, 5.0]}), "g", "y")
    data = frame({"a": [1.0, 2.0, 3.0], "b": [4.0, 6.0, 5.0]})
    data["y"] = data["y"] * base
    result = betweenstats(data, "g", "y")
    assert result.text.subtitle == unit.text.subtitle
    assert result.test.df_error == pytest.approx(unit.test.df_error, rel=1e-14)
    assert result.effect_size.estimate == pytest.approx(
        unit.effect_size.estimate, rel=1e-13
    )


@pytest.mark.parametrize("scale", [1e-80, 1e-90, 1e80, 1e160])
def test_review_scales_end_to_end(scale: float) -> None:
    # STAT-1: data times 1e-80 printed t_Welch(18.88); 1e-90 and 1e80 raised.
    rng = np.random.default_rng(1)
    y = np.r_[rng.normal(0.0, 1.0, 10), rng.normal(1.0, 2.0, 15)]
    data = pd.DataFrame({"g": ["a"] * 10 + ["b"] * 15, "y": y * scale})
    result = betweenstats(data, "g", "y")
    assert result.text.subtitle.startswith("t_Welch(18.82) = -0.93, p = 0.37")


def test_a_group_wider_than_the_largest_double_raises() -> None:
    # NEW-6: the range overflows, not the standard deviation (1.5e308 here).
    data = frame({"a": [-1.5e308, 1.5e308, 0.0], "b": [1.0, 2.0, 4.0]})
    with pytest.raises(
        ValueError,
        match=r"'a' of x='g' range more widely than a double can represent: their "
        r"largest minus their smallest value exceeds the largest double",
    ):
        betweenstats(data, "g", "y")


def test_a_mean_difference_beyond_the_largest_double_is_analyzed() -> None:
    # NEW-7: the difference of the means (-2.7e308) overflows in y units, but the
    # analysis reports t, df and p only, which do not depend on the units: the same
    # numbers as for the data divided by 2^100.
    a, b = [-1.5e308, -1.2e308, -1.0e308], [1.0e308, 1.2e308, 1.5e308]
    result = betweenstats(frame({"a": a, "b": b}), "g", "y")
    small = [[math.ldexp(v, -100) for v in g] for g in (a, b)]
    unit = betweenstats(frame({"a": small[0], "b": small[1]}), "g", "y")
    for field in ("statistic", "df_error", "p_value"):
        assert getattr(result.test, field) == getattr(unit.test, field)
    assert result.test.statistic < -10.0
    assert result.effect_size.estimate == unit.effect_size.estimate


def test_a_group_too_small_next_to_the_others_raises() -> None:
    groups = {"a": [1e300, 2e300, 3e300], "b": [1e-10, 2e-10, 4e-10]}
    groups["c"] = [2e300, 5e300, 3e300]
    with pytest.raises(ValueError, match=r"'b' of x='g' vary by less than double"):
        betweenstats(frame(groups), "g", "y")


def test_values_that_differ_are_not_constant() -> None:
    # Constant means all values equal: a group at 1e-300 varies.
    groups = {"a": [1e-300, 2e-300, 3e-300], "b": [5.0, 5.0], "c": [1.0, 2.0, 4.0]}
    with pytest.raises(ValueError, match=r"variance, which is 0 for the group 'b'"):
        betweenstats(frame(groups), "g", "y")


def test_hedges_g_at_one_degree_of_freedom_warns() -> None:
    # STAT-3: one constant group and one group of two values: Welch df = 1, J = 0.
    result = betweenstats(frame({"a": [5.0, 5.0, 5.0], "b": [1.0, 3.0]}), "g", "y")
    assert result.test.df_error == 1.0
    es = result.effect_size
    for value in (es.estimate, es.conf_low, es.conf_high):
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0
    assert "CI_95% [0.00, 0.00]" in result.text.subtitle
    assert "-0.00" not in result.text.subtitle
    assert "95% CI [0.00, 0.00]" in result.text.apa
    assert result.warnings == (HEDGES_J_ZERO,)
    assert '"-0.0' not in result.to_json()


def test_hedges_g_at_one_degree_of_freedom_with_a_varying_group() -> None:
    # The warning names the condition tested (Welch's df is 1), which also holds
    # when the other group varies, but negligibly next to the two-value group.
    a, b = [5.0 - 1e-6, 5.0, 5.0 + 1e-6], [1e3, 3e3]
    result = betweenstats(frame({"a": a, "b": b}), "g", "y")
    assert result.test.df_error == 1.0
    assert result.effect_size.estimate == 0.0
    assert result.warnings == (HEDGES_J_ZERO,)
    assert "no variability" not in result.warnings[0]


def test_hedges_g_below_two_degrees_of_freedom_warns() -> None:
    # Two values against fifty: Welch's df is just above 1, where Hedges' correction
    # shrinks a standardized difference of 1.2 to 0.03 (effectsize does the same).
    rng = np.random.default_rng(5)
    a, b = [0.0, 2.0], rng.normal(-0.64, 0.5, 50).tolist()
    result = betweenstats(frame({"a": a, "b": b}), "g", "y")
    df = result.test.df_error
    assert df is not None
    assert 1.0 < df < 1.1
    (warning,) = result.warnings
    assert warning.startswith("Hedges' g is the standardized difference times a")
    assert f"Welch's df is {df:.2f}, below 2" in warning
    d = (np.mean(a) - np.mean(b)) / math.sqrt(
        (np.var(a, ddof=1) + np.var(b, ddof=1)) / 2
    )
    assert abs(d) > 1.0
    assert abs(result.effect_size.estimate) < 0.1 * abs(d)
    # From 2 df on (two values in each group, equal variances) there is no warning.
    even = betweenstats(frame({"a": [0.0, 2.0], "b": [5.0, 7.0]}), "g", "y")
    assert even.test.df_error == 2.0
    assert even.warnings == ()


HEDGES_J_ZERO = (
    "Hedges' g is 0 by construction at 1 degree of freedom. Welch's df is 1 here "
    "because one group has two values and the other group's variance is 0 or "
    "negligible next to it, so g and its confidence interval carry no information "
    "about the size of the difference."
)


# --------------------------------------------------------------------------- #
# arguments that have no effect on these data: a warnings entry each (STAT-2)


def test_defaults_add_no_warning(two: StatsResult, anova: StatsResult) -> None:
    assert two.warnings == ()
    assert anova.warnings == (K3_BAYES_WARNING,)


def test_default_aliases_add_no_warning(teeth: pd.DataFrame) -> None:
    result = betweenstats(
        teeth,
        "supp",
        "len",
        pairwise_display="s",
        p_adjust_method="holm",
        pairwise_alpha=0.05,
        bf_prior=0.707,
    )
    assert result.warnings == ()


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        (
            "p_adjust_method",
            "bonferroni",
            "p_adjust_method='bonferroni' has no effect: x='supp' has 2 groups, so "
            "there are no pairwise comparisons to adjust.",
        ),
        (
            "pairwise_display",
            "all",
            "pairwise_display='all' has no effect: x='supp' has 2 groups, so there "
            "are no pairwise comparisons to display.",
        ),
        (
            "pairwise_display",
            "ns",
            "pairwise_display='ns' has no effect: x='supp' has 2 groups, so there "
            "are no pairwise comparisons to display.",
        ),
        (
            "pairwise_alpha",
            0.01,
            "pairwise_alpha=0.01 has no effect: x='supp' has 2 groups, so there are "
            "no pairwise comparisons to call significant.",
        ),
        (
            "pairwise_alpha",
            np.float64(0.01),
            "pairwise_alpha=0.01 has no effect: x='supp' has 2 groups, so there are "
            "no pairwise comparisons to call significant.",
        ),
    ],
)
def test_pairwise_arguments_with_two_groups_warn(
    teeth: pd.DataFrame, argument: str, value: object, message: str
) -> None:
    result = betweenstats(teeth, "supp", "len", **{argument: value})
    assert result.warnings == (message,)
    assert result.pairwise == ()


def test_every_pairwise_argument_warns_once(teeth: pd.DataFrame) -> None:
    result = betweenstats(
        teeth,
        "supp",
        "len",
        p_adjust_method="BH",
        pairwise_display="none",
        pairwise_alpha=0.1,
    )
    assert [w.split(" has no effect")[0] for w in result.warnings] == [
        "p_adjust_method='BH'",
        "pairwise_display='none'",
        "pairwise_alpha=0.1",
    ]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"bf_message": False}, "bf_message=False leaves the Bayes factor out"),
        (
            {"results_subtitle": False},
            "results_subtitle=False leaves the Bayes factor out",
        ),
    ],
)
def test_bf_prior_without_a_bayes_factor_warns(
    teeth: pd.DataFrame, kwargs: dict[str, Any], reason: str
) -> None:
    result = betweenstats(teeth, "supp", "len", bf_prior=2.0, **kwargs)
    assert result.bayes is None
    assert result.warnings == (
        f"bf_prior=2.0 has no effect: no Bayes factor is computed ({reason}).",
    )


def test_bf_prior_with_three_groups_warns(plants: pd.DataFrame) -> None:
    result = betweenstats(plants, "group", "weight", bf_prior=1.0)
    assert result.warnings == (
        K3_BAYES_WARNING,
        "bf_prior=1.0 has no effect: no Bayes factor is computed (x='group' has 3 "
        "groups, and the Bayes factor for three or more groups arrives in "
        "inferplot 0.2).",
    )


def test_bf_prior_with_a_bayes_factor_does_not_warn(teeth: pd.DataFrame) -> None:
    result = betweenstats(teeth, "supp", "len", bf_prior=1.0)
    assert result.bayes is not None
    assert result.warnings == ()


def test_pairwise_arguments_with_three_groups_do_not_warn(
    plants: pd.DataFrame,
) -> None:
    result = betweenstats(
        plants,
        "group",
        "weight",
        p_adjust_method="BH",
        pairwise_display="all",
        pairwise_alpha=0.1,
    )
    assert result.warnings == (K3_BAYES_WARNING,)
