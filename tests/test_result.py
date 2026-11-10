"""StatsResult: plain types, JSON, frames, check() and printing."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import inspect
import json
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot import text as tx
from inferplot.check import SMALL_N, Issue, result_issues
from inferplot.result import (
    PARITY_TARGET,
    SCHEMA_VERSION,
    Descriptive,
    EffectSize,
    PairwiseComparison,
    StatsResult,
    TestResult,
    current_provenance,
    tied_count,
)
from tests.oracle_fixtures import DATASET_DIR, OracleCase, load_cases
from tests.test_text_expressions import result_from_fixture


def _case(name: str) -> OracleCase:
    (case,) = [c for c in load_cases() if c.name == f"ggbetweenstats-{name}"]
    return case


def _no_nan(result: StatsResult) -> StatsResult:
    """The fixture's pairs lack a raw p; give them one so equality is meaningful."""
    pairs = tuple(
        dataclasses.replace(p, p_value=(p.p_adjusted or 0.0) / 3)
        for p in result.pairwise
    )
    return dataclasses.replace(result, pairwise=pairs)


@pytest.fixture
def anova() -> StatsResult:
    return _no_nan(result_from_fixture(_case("PlantGrowth-parametric")))


@pytest.fixture
def one_sided() -> StatsResult:
    return result_from_fixture(_case("ToothGrowth-greater-parametric"))


# --------------------------------------------------------------------------- #
# plain types
# --------------------------------------------------------------------------- #


def test_numpy_scalars_and_lists_become_plain() -> None:
    n: Any = np.int64(3)
    d = Descriptive(level=np.str_("a"), n=n, centrality="mean", value=np.float64(1.5))
    assert type(d.level) is str
    assert type(d.n) is int
    assert type(d.value) is float
    assert repr(d) == "Descriptive(level='a', n=3, centrality='mean', value=1.5)"
    yes: Any = np.bool_(True)
    no: Any = np.bool_(False)
    pair = PairwiseComparison(
        "a",
        "b",
        "q",
        1.0,
        None,
        0.1,
        0.2,
        "holm",
        "games_howell",
        "Games-Howell",
        yes,
        no,
    )
    assert pair.significant is True
    assert pair.displayed is False
    test = TestResult("welch_t", "W", "m", "t", 1.0, None, 3.0, 0.5, "two-sided")
    assert test.df is None
    with pytest.raises(TypeError, match="plain Python values"):
        Descriptive(level="a", n=1, centrality="mean", value=object())  # type: ignore[arg-type]


def test_lists_become_tuples(anova: StatsResult) -> None:
    result = dataclasses.replace(anova, levels=list(anova.levels), warnings=["w"])  # type: ignore[arg-type]
    assert result.levels == anova.levels
    assert result.warnings == ("w",)


def test_numpy_arrays_become_tuples_of_floats(anova: StatsResult) -> None:
    arrays: Any = tuple(np.asarray(v, dtype=np.float64) for v in anova.values)
    result = dataclasses.replace(anova, values=arrays)
    assert result.values == anova.values
    assert all(type(v) is tuple for v in result.values)
    assert all(type(x) is float for v in result.values for x in v)


def test_values_hold_each_level_and_survive_json(anova: StatsResult) -> None:
    assert len(anova.values) == len(anova.levels)
    assert [len(v) for v in anova.values] == [d.n for d in anova.descriptives]
    assert StatsResult.from_json(anova.to_json()).values == anova.values


def test_subjects_survive_json_but_not_the_embedded_copy(anova: StatsResult) -> None:
    # A between design has none; labels are plain strings, and the copy save()
    # embeds leaves them out with the values (D36).
    assert anova.subjects == ()
    assert json.loads(anova.to_json())["subjects"] == []
    labels: Any = (np.str_("s1"), "s2")
    within = dataclasses.replace(
        anova, design="within", subject_id="id", subjects=labels
    )
    assert within.subjects == ("s1", "s2")
    assert all(type(s) is str for s in within.subjects)
    assert StatsResult.from_json(within.to_json()) == within
    embedded = StatsResult.from_json(within.to_json(include_values=False))
    assert (embedded.values, embedded.subjects) == ((), ())
    assert embedded.subject_id == "id"


@pytest.mark.parametrize(
    "values",
    [
        ((1, 2, 3), (4, 5)),
        [[1, 2, 3], [4, 5]],
        (np.array([1, 2, 3], dtype=np.int64), np.array([4, 5], dtype=np.int32)),
        (np.array([1.0, 2.0, 3.0]), (np.int64(4), np.float32(5.0))),
    ],
    ids=["int tuples", "int lists", "int arrays", "mixed scalars"],
)
def test_values_are_floats_whatever_the_input(anova: StatsResult, values: Any) -> None:
    """Integer data still gives floats, so the JSON writes 1.0 and reads back equal
    and of the same types."""
    result = dataclasses.replace(anova, values=values)
    assert result.values == ((1.0, 2.0, 3.0), (4.0, 5.0))
    assert all(type(x) is float for v in result.values for x in v)
    assert json.loads(result.to_json())["values"] == [[1.0, 2.0, 3.0], [4.0, 5.0]]
    assert '"values": [\n    [\n      1.0,' in result.to_json()
    back = StatsResult.from_json(result.to_json())
    assert back == result
    assert all(type(x) is float for v in back.values for x in v)
    assert result.to_dict()["values"] == [[1.0, 2.0, 3.0], [4.0, 5.0]]


def test_float_fields_become_floats_and_reject_bools(anova: StatsResult) -> None:
    d = Descriptive(level="a", n=3, centrality="mean", value=cast_any(2))
    assert type(d.value) is float
    pair = anova.pairwise[0]
    assert type(dataclasses.replace(pair, df=cast_any(7)).df) is float
    assert dataclasses.replace(pair, df=None).df is None
    with pytest.raises(TypeError, match=r"Descriptive\.value must be a number"):
        Descriptive(level="a", n=3, centrality="mean", value=cast_any(True))
    with pytest.raises(TypeError, match=r"StatsResult\.values must be a number"):
        dataclasses.replace(anova, values=((1.0, cast_any(False)),))


def cast_any(value: object) -> Any:
    return value


def test_results_are_frozen(anova: StatsResult) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        anova.n_obs = 3  # type: ignore[misc]


def test_testresult_is_not_collected_by_pytest() -> None:
    assert TestResult.__test__ is False


def test_provenance() -> None:
    prov = current_provenance(seed=7, citations=("Welch 1951",))
    assert prov.seed == 7
    assert prov.parity_target == PARITY_TARGET == "ggstatsplot 1.1.4"
    assert prov.numpy == importlib.metadata.version("numpy")
    assert prov.matplotlib == importlib.metadata.version("matplotlib")
    assert prov.narwhals == importlib.metadata.version("narwhals")
    assert prov.citations == ("Welch 1951",)


# --------------------------------------------------------------------------- #
# dict and JSON
# --------------------------------------------------------------------------- #

_NON_STRICT = re.compile(r"\b(NaN|Infinity|-Infinity)\b")


def _strict_loads(text: str) -> Any:
    def reject(name: str) -> Any:
        raise ValueError(f"non-standard JSON literal {name}")

    return json.loads(text, parse_constant=reject)


def test_to_dict_keeps_floats(one_sided: StatsResult) -> None:
    d = one_sided.to_dict()
    assert d["effect_size"]["conf_high"] == math.inf
    assert isinstance(d["levels"], list)
    assert isinstance(d["descriptives"][0], dict)
    assert d["test"]["df"] is None
    assert type(d["settings"]["conf_level"]) is float
    assert one_sided.test is not None
    with_int = dataclasses.replace(
        one_sided, test=dataclasses.replace(one_sided.test, df_error=55)
    )
    assert type(with_int.to_dict()["test"]["df_error"]) is float


def test_to_json_is_strict_and_round_trips_inf(one_sided: StatsResult) -> None:
    text = one_sided.to_json()
    assert not _NON_STRICT.search(text)
    assert text.isascii()
    data = _strict_loads(text)
    assert data["effect_size"]["conf_high"] == "Inf"
    assert StatsResult.from_json(text) == one_sided
    less = result_from_fixture(_case("ToothGrowth-less-parametric"))
    assert _strict_loads(less.to_json())["effect_size"]["conf_low"] == "-Inf"
    assert StatsResult.from_json(less.to_json()) == less


def test_to_json_round_trip_with_pairs(anova: StatsResult) -> None:
    text = anova.to_json(indent=None)
    assert "\n" not in text
    assert StatsResult.from_json(text) == anova
    assert StatsResult.from_dict(anova.to_dict()) == anova


def test_nan_becomes_null() -> None:
    result = result_from_fixture(_case("PlantGrowth-parametric"))  # raw p is NaN
    data = _strict_loads(result.to_json())
    assert data["pairwise"][0]["p_value"] is None
    back = StatsResult.from_json(result.to_json())
    # A pair's p-value can be None (Bayes factor pairs have none), so its JSON
    # null reads back as None: NaN and None both mean "missing" there.
    assert back.pairwise[0].p_value is None
    assert back.to_json() == result.to_json()
    # In a field that is always a number, null reads back as NaN.
    assert result.effect_size is not None
    nan_es = dataclasses.replace(result.effect_size, estimate=math.nan)
    with_nan = dataclasses.replace(result, effect_size=nan_es)
    assert _strict_loads(with_nan.to_json())["effect_size"]["estimate"] is None
    again = StatsResult.from_json(with_nan.to_json())
    assert again.effect_size is not None
    assert math.isnan(again.effect_size.estimate)


def test_from_dict_rejects_bad_input(anova: StatsResult) -> None:
    data = anova.to_dict()
    data["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        StatsResult.from_dict(data)
    data = anova.to_dict()
    data["n_obs"] = "30"
    with pytest.raises(TypeError, match="n_obs"):
        StatsResult.from_dict(data)


def test_schema_version_is_the_first_key(anova: StatsResult) -> None:
    # 3 since histostats and scatterstats (point_labels, label_var, test_value).
    assert SCHEMA_VERSION == 3
    for text in (anova.to_json(), anova.to_json(indent=None, include_values=False)):
        data = _strict_loads(text)
        assert next(iter(data)) == "schema_version"
        assert data["schema_version"] == 3
    assert next(iter(anova.to_dict())) == "schema_version"
    assert anova.to_dict()["schema_version"] == 3
    # Not a field of the result: it describes the JSON, not the analysis.
    assert "schema_version" not in {f.name for f in dataclasses.fields(anova)}


def test_from_json_refuses_another_schema_version(anova: StatsResult) -> None:
    rerun = "compute the result again with this version of inferplot"
    # A result written before the version existed (schema version 1).
    data = anova.to_dict()
    del data["schema_version"]
    with pytest.raises(ValueError, match="no schema_version") as info:
        StatsResult.from_dict(data)
    assert type(info.value) is ValueError
    message = str(info.value)
    assert "schema version 1" in message
    assert "schema version 3 only" in message
    assert rerun in message
    # JSON from a newer (or older, versioned) inferplot: version 2 is the field
    # set before histostats and scatterstats.
    for other in (1, 2, 4, "3", 3.0, None):
        data = json.loads(anova.to_json())
        data["schema_version"] = other
        with pytest.raises(ValueError, match="has schema_version") as info:
            StatsResult.from_json(json.dumps(data))
        assert type(info.value) is ValueError
        assert f"schema_version {other!r}" in str(info.value)
        assert "schema version 3 only" in str(info.value)
        assert rerun in str(info.value)


def test_effect_size_resample_counts_go_together() -> None:
    base = {
        "name": "rank_epsilon_squared",
        "label": "rank epsilon squared",
        "estimate": 0.2,
        "conf_level": 0.95,
        "conf_low": 0.1,
        "conf_high": 1.0,
        "ci_method": "percentile bootstrap",
        "sidedness": "greater",
    }
    full = EffectSize(
        **base,
        bootstrap_resamples=2000,
        bootstrap_seed=0,
        undefined_resamples=13,
        infinite_resamples=0,
    )
    assert full.undefined_resamples == 13
    assert EffectSize(**base).undefined_resamples is None
    with pytest.raises(ValueError, match="must record undefined_resamples"):
        EffectSize(**base, bootstrap_resamples=2000, bootstrap_seed=0)
    with pytest.raises(ValueError, match="give bootstrap_resamples too"):
        EffectSize(**base, undefined_resamples=0)
    # JSON key order follows SPEC section 6.
    keys = _field_names(full)
    assert keys.index("undefined_resamples") == keys.index("infinite_resamples") - 1


def _field_names(value: object) -> list[str]:
    return [f.name for f in dataclasses.fields(value)]  # type: ignore[arg-type]


def test_json_holds_every_text(anova: StatsResult) -> None:
    data = _strict_loads(anova.to_json())
    assert data["text"]["apa"] == anova.text.apa
    assert data["text"]["tick_labels"] == list(anova.text.tick_labels)
    assert data["provenance"]["parity_target"] == "ggstatsplot 1.1.4"


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #


def test_to_frame_pandas(anova: StatsResult) -> None:
    frame = anova.to_frame("pandas")
    assert len(frame) == 1 + len(anova.pairwise)
    assert list(frame["test"]) == [
        "welch_anova",
        "games_howell",
        "games_howell",
        "games_howell",
    ]
    main = frame.iloc[0]
    assert anova.test is not None
    assert main["statistic"] == anova.test.statistic
    assert main["df"] == 2.0
    assert main["effect_size"] == "partial_omega_squared"
    assert math.isnan(main["p_adjusted"])
    assert main["group1"] is None
    assert frame["significant"].dtype == "boolean"
    assert frame["significant"].isna().iloc[0]
    assert list(frame["group1"].iloc[1:]) == [p.group1 for p in anova.pairwise]
    assert list(frame["p_adjusted"].iloc[1:]) == [p.p_adjusted for p in anova.pairwise]
    assert str(frame["statistic"].dtype) == "float64"


def test_to_frame_polars(anova: StatsResult) -> None:
    import polars as pl

    frame = anova.to_frame("polars")
    assert frame.height == 1 + len(anova.pairwise)
    assert frame.schema["significant"] == pl.Boolean
    assert frame.schema["df"] == pl.Float64
    assert frame["test"].to_list()[0] == "welch_anova"
    assert frame["p_adjusted"].to_list()[0] is None
    assert frame["displayed"].to_list()[1:] == [p.displayed for p in anova.pairwise]


def test_to_frame_backends_agree(one_sided: StatsResult) -> None:
    pd_frame = one_sided.to_frame()
    pl_frame = one_sided.to_frame("polars")
    assert list(pd_frame.columns) == pl_frame.columns
    assert pd_frame["conf_high"].iloc[0] == pl_frame["conf_high"][0] == math.inf
    assert pd_frame["bf10"].iloc[0] == pl_frame["bf10"][0] == one_sided.bayes.bf10  # type: ignore[union-attr]


def test_to_frame_says_which_intervals_are_one_sided(
    anova: StatsResult, one_sided: StatsResult
) -> None:
    # Welch's ANOVA is a two-sided test whose omega squared interval is [low, 1].
    frame = anova.to_frame()
    assert list(frame.columns).index("ci_sidedness") == (
        list(frame.columns).index("conf_high") + 1
    )
    assert frame["alternative"].iloc[0] == "two-sided"
    assert frame["ci_sidedness"].iloc[0] == "greater"
    assert frame["conf_high"].iloc[0] == 1.0
    # Games-Howell pairs have no interval.
    assert frame["ci_sidedness"].iloc[1:].isna().all()
    assert one_sided.to_frame("polars")["ci_sidedness"].to_list() == ["greater"]
    two_sided = result_from_fixture(_case("ToothGrowth-parametric"))
    assert two_sided.to_frame()["ci_sidedness"].to_list() == ["two-sided"]


def test_tied_count() -> None:
    assert tied_count(()) == 0
    assert tied_count(((1.0, 2.0), (3.0,))) == 0
    # Values beyond the first of each distinct value, every level pooled.
    assert tied_count(((1.0, 1.0, 2.0), (2.0, 2.0, 3.0))) == 3
    assert tied_count(((0.0,), (-0.0,))) == 1


def test_to_frame_rejects_unknown_backend(anova: StatsResult) -> None:
    with pytest.raises(ValueError, match="backend"):
        anova.to_frame("pyarrow")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# check()
# --------------------------------------------------------------------------- #


def _codes(issues: tuple[Issue, ...]) -> list[str]:
    return [issue.code for issue in issues]


def test_check_clean(one_sided: StatsResult) -> None:
    assert one_sided.check() == ()
    assert result_issues(one_sided) == ()


def test_check_small_n(one_sided: StatsResult) -> None:
    small = (
        dataclasses.replace(one_sided.descriptives[0], n=SMALL_N - 1),
        one_sided.descriptives[1],
    )
    issues = dataclasses.replace(one_sided, descriptives=small).check()
    assert _codes(issues) == ["small_n"]
    assert issues[0].severity == "warning"
    assert "'OJ' (n = 4)" in issues[0].message
    edge = result_from_fixture(_case("edge_unbalanced-parametric"))
    assert "small_n" in _codes(edge.check())


def test_check_rows_dropped(one_sided: StatsResult) -> None:
    issues = dataclasses.replace(one_sided, n_dropped=3).check()
    assert _codes(issues) == ["rows_dropped"]
    assert issues[0].severity == "info"
    assert "3 rows were left out" in issues[0].message
    issues = dataclasses.replace(one_sided, dropped_subjects=1).check()
    assert _codes(issues) == ["rows_dropped"]
    assert "1 subject was left out" in issues[0].message


def test_check_pairs_hidden(anova: StatsResult) -> None:
    issues = anova.check()
    assert _codes(issues) == ["pairs_hidden"]
    assert issues[0].severity == "info"
    assert "3 of 3 pairwise comparisons are not drawn" in issues[0].message
    assert "pairwise_display='all'" in issues[0].fix
    shown = tuple(dataclasses.replace(p, displayed=True) for p in anova.pairwise)
    assert dataclasses.replace(anova, pairwise=shown).check() == ()


def test_check_pairs_hidden_not_for_an_explicit_none(anova: StatsResult) -> None:
    # pairwise_display="none" is never a default (the default is "significant"),
    # so the setting itself says the caller asked for no brackets (U11).
    settings = dataclasses.replace(anova.settings, pairwise_display="none")
    asked = dataclasses.replace(anova, settings=settings)
    assert all(not p.displayed for p in asked.pairwise)
    assert asked.check() == ()
    assert StatsResult.from_json(asked.to_json(include_values=False)).check() == ()


def _check_cases() -> list[Any]:
    rng = np.random.default_rng(11)
    likert = pd.DataFrame(
        {"g": np.repeat(["a", "b", "c"], 30), "y": rng.integers(1, 6, 90) * 1.0}
    )
    sparse = pd.DataFrame(
        {"g": np.repeat(["a", "b"], 4), "y": [1.0, 2, 2, np.nan, 3, 3, 3, 4]}
    )
    plants = pd.read_csv(DATASET_DIR / "PlantGrowth.csv")
    bugs = pd.read_csv(DATASET_DIR / "bugs_long.csv")
    between = ip.analyze.betweenstats
    within = ip.analyze.withinstats
    return [
        pytest.param(lambda: between(likert, "g", "y", type="np"), id="likert-np"),
        pytest.param(
            lambda: between(likert[likert.g != "c"], "g", "y", type="np"),
            id="likert2-np",
        ),
        pytest.param(lambda: between(likert, "g", "y", type="r"), id="likert-robust"),
        pytest.param(lambda: between(sparse, "g", "y", type="np"), id="small-ties-np"),
        pytest.param(lambda: between(plants, "group", "weight"), id="plants"),
        pytest.param(
            lambda: between(plants, "group", "weight", pairwise_display="none"),
            id="plants-none",
        ),
        pytest.param(
            lambda: within(bugs, "condition", "desire", subject_id="subject"),
            id="bugs-within",
        ),
    ]


@pytest.mark.parametrize("make", _check_cases())
def test_check_is_the_same_without_values(make: Any) -> None:
    """The copy save() embeds (no observations) reports the same issues (U11)."""
    result: StatsResult = make()
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert embedded.values == ()
    assert embedded.check() == result.check()


def test_heavy_ties_reads_the_stored_count() -> None:
    likert = pd.DataFrame(
        {"g": np.repeat(["a", "b"], 30), "y": np.tile([1.0, 2, 3, 4, 5], 12)}
    )
    result = ip.analyze.betweenstats(likert, "g", "y", type="np")
    assert result.n_tied == 60 - 5 == tied_count(result.values)
    (issue,) = [i for i in result.check() if i.code == "heavy_ties"]
    assert "only 5 distinct values among 60, so 55 of them (92%) are ties" in (
        issue.message
    )
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert embedded.n_tied == 55
    assert "heavy_ties" in _codes(embedded.check())
    # The count, not the values, decides.
    untied = dataclasses.replace(result, n_tied=0)
    assert "heavy_ties" not in _codes(untied.check())


def test_no_check_reads_the_observations() -> None:
    import inferplot.check

    source = inspect.getsource(inferplot.check)
    assert "result.values" not in source
    assert "result.subjects" not in source


def test_sign_and_df_conventions_as_documented() -> None:
    """U8: the conventions TestResult and PairwiseComparison document. Levels a <
    b < c by construction, so a sign shows which difference is taken."""
    rng = np.random.default_rng(1)
    shift = np.repeat([0.0, 1.0, 2.0], 12)
    three = pd.DataFrame(
        {"g": np.repeat(["a", "b", "c"], 12), "y": rng.normal(size=36) + shift}
    )
    two = three[three.g != "b"]
    between: Any = ip.analyze.betweenstats

    welch = between(two, "g", "y")
    assert welch.test.statistic < 0
    assert welch.effect_size.estimate < 0
    assert (welch.test.df, welch.test.df_error is not None) == (None, True)
    mann = between(two, "g", "y", type="np")
    assert mann.test.statistic >= 0
    assert mann.effect_size.estimate < 0
    assert (mann.test.df, mann.test.df_error) == (None, None)
    yuen = between(two, "g", "y", type="r")
    assert yuen.test.statistic > 0
    assert yuen.effect_size.estimate < 0
    assert yuen.test.df is None

    anova = between(three, "g", "y")
    assert (anova.test.df, anova.test.df_error is not None) == (2.0, True)
    assert all(p.statistic > 0 for p in anova.pairwise)  # q: group2 - group1
    kruskal = between(three, "g", "y", type="np")
    assert (kruskal.test.df, kruskal.test.df_error) == (None, 2.0)
    assert all(p.statistic >= 0 and p.df is None for p in kruskal.pairwise)
    lincon = between(three, "g", "y", type="r")
    assert all(p.statistic < 0 and (p.estimate or 0) < 0 for p in lincon.pairwise)
    bayes = between(three, "g", "y", type="bayes", results_subtitle=False)
    assert all(p.statistic < 0 and p.df == 22 for p in bayes.pairwise)

    within: Any = ip.analyze.withinstats
    paired = three.assign(s=np.tile(np.arange(12), 3))
    t = within(paired[paired.g != "b"], "g", "y", subject_id="s")
    assert t.test.statistic < 0
    assert (t.test.df, t.test.df_error) == (None, 11.0)
    rm = within(paired, "g", "y", subject_id="s")
    assert rm.test.df is not None
    assert rm.test.df_error is not None
    assert all(p.statistic < 0 and p.df == 11 for p in rm.pairwise)


def test_issue_is_frozen_and_prints() -> None:
    issue = Issue("small_n", "warning", "Too few.", "Collect more.")
    assert str(issue) == "[warning] small_n: Too few. Collect more."
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.code = "x"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# __str__
# --------------------------------------------------------------------------- #


def test_str_two_groups(one_sided: StatsResult) -> None:
    assert str(one_sided) == one_sided.text.apa


def test_str_with_pairs(anova: StatsResult) -> None:
    lines = str(anova).splitlines()
    assert lines[0] == anova.text.apa
    assert lines[1] == ""
    assert lines[2] == "Pairwise comparisons (Games-Howell):"
    assert lines[3].split() == [
        "group1",
        "group2",
        "q",
        "df",
        "p",
        "p",
        "(Holm)",
        "significant",
        "shown",
    ]
    assert lines[4].split() == [
        "ctrl",
        "trt1",
        "-1.68",
        "NA",
        ".158",
        ".475",
        "no",
        "no",
    ]
    assert len(lines) == 4 + len(anova.pairwise)


@pytest.mark.parametrize(
    ("method", "heading"),
    [
        ("holm", "p (Holm)"),
        ("hochberg", "p (Hochberg)"),
        ("hommel", "p (Hommel)"),
        ("bonferroni", "p (Bonferroni)"),
        ("BH", "p (Benjamini-Hochberg)"),
        ("fdr", "p (Benjamini-Hochberg)"),
        ("BY", "p (Benjamini-Yekutieli)"),
        ("none", "p (unadj.)"),
    ],
)
def test_str_names_the_adjustment(
    anova: StatsResult, method: str, heading: str
) -> None:
    settings = dataclasses.replace(anova.settings, p_adjust_method=method)
    header = str(dataclasses.replace(anova, settings=settings)).splitlines()[3]
    assert f"  {heading}  " in header


def test_texts_match_builders(anova: StatsResult) -> None:
    rebuilt = tx.build_texts(
        x=anova.x,
        y=anova.y,
        descriptives=anova.descriptives,
        test=anova.test,
        effect_size=anova.effect_size,
        bayes=anova.bayes,
        pairwise=anova.pairwise,
        n_obs=anova.n_obs,
        settings=anova.settings,
    )
    assert rebuilt == anova.text


# --------------------------------------------------------------------------- #
# designs without levels: one-sample (histostats) and correlation (scatterstats)
# --------------------------------------------------------------------------- #


def _no_level_case(function: str, name: str) -> OracleCase:
    (case,) = [c for c in load_cases() if c.name == f"{function}-{name}"]
    return case


@pytest.fixture
def one_sample() -> StatsResult:
    from tests.no_level_cases import analyzed

    result = analyzed(
        _no_level_case("gghistostats", "edge_one_missing-parametric").name
    )
    assert result is not None
    return result


@pytest.fixture
def correlation() -> StatsResult:
    from tests.no_level_cases import analyzed

    result = analyzed(_no_level_case("ggscatterstats", "mtcars-labels-parametric").name)
    assert result is not None
    return result


def _bayes_factor() -> Any:
    from inferplot.result import BayesFactor

    return BayesFactor(
        bf10=3.0, log_bf01=-math.log(3.0), prior="cauchy", prior_scale=0.707, test="x"
    )


def test_settings_test_value_defaults_to_none(anova: StatsResult) -> None:
    assert anova.settings.test_value is None
    assert anova.point_labels == ()
    assert anova.label_var is None
    assert "test_value" in json.loads(anova.to_json())["settings"]


def test_new_designs_round_trip_with_special_values(
    one_sample: StatsResult, correlation: StatsResult
) -> None:
    for result in (one_sample, correlation):
        assert result.test is not None
        assert result.effect_size is not None
        special = dataclasses.replace(
            result,
            test=dataclasses.replace(result.test, statistic=-math.inf),
            effect_size=dataclasses.replace(
                result.effect_size, conf_low=-math.inf, conf_high=math.inf
            ),
        )
        assert StatsResult.from_json(special.to_json()) == special
        assert StatsResult.from_dict(special.to_dict()) == special
        data = _strict_loads(special.to_json())
        assert data["test"]["statistic"] == "-Inf"
    assert one_sample.y is None
    assert json.loads(one_sample.to_json())["y"] is None
    assert one_sample.settings.test_value == 10.0
    assert StatsResult.from_json(one_sample.to_json()).settings.test_value == 10.0


def test_point_labels_are_left_out_of_the_embedded_copy(
    correlation: StatsResult,
) -> None:
    assert correlation.label_var == "model"
    assert len(correlation.point_labels) == correlation.n_obs == 32
    data = json.loads(correlation.to_json(include_values=False))
    assert (data["values"], data["subjects"], data["point_labels"]) == ([], [], [])
    embedded = StatsResult.from_json(correlation.to_json(include_values=False))
    assert (embedded.values, embedded.point_labels) == ((), ())
    assert embedded.label_var == "model"
    assert embedded.check() == correlation.check()


def test_to_frame_has_the_main_row_only(
    one_sample: StatsResult, correlation: StatsResult
) -> None:
    for result in (one_sample, correlation):
        frame = result.to_frame()
        assert len(frame) == 1
        assert frame["group1"].isna().all()
        assert frame["group2"].isna().all()
        assert result.test is not None
        assert frame["statistic"][0] == result.test.statistic
        assert len(result.to_frame(backend="polars")) == 1
    # A Bayes factor without a test names its design.
    for result, label in (
        (correlation, "Bayesian Pearson correlation"),
        (one_sample, "Bayesian one-sample t-test"),
    ):
        bayes = dataclasses.replace(
            result, type="bayes", test=None, effect_size=None, bayes=_bayes_factor()
        )
        assert bayes.to_frame()["label"][0] == label


@pytest.mark.parametrize(
    ("design", "changes", "problem"),
    [
        ("one-sample", {"y": "len"}, "y is None in design 'one-sample'"),
        ("one-sample", {"levels": ("a",)}, "levels must be empty"),
        ("one-sample", {"subject_id": "s"}, "subject_id must be None"),
        ("one-sample", {"values": ((1.0,), (2.0,))}, "values must hold 1 tuple(s)"),
        ("one-sample", {"values": ((1.0, 2.0),)}, "must hold n_obs = 10 values"),
        ("one-sample", {"descriptives": ()}, "descriptives must hold one entry"),
        ("one-sample", {"analysis": "scatterstats"}, "comes from histostats"),
        ("one-sample", {"analysis": "piestats"}, "unknown analysis 'piestats'"),
        ("one-sample", {"label_var": "m"}, "belong to design 'correlation' only"),
        ("one-sample", {"dropped_subjects": 1}, "dropped_subjects and"),
        ("correlation", {"y": None}, "y is None in design 'one-sample'"),
        ("correlation", {"values": ((1.0,),)}, "values must hold 2 tuple(s)"),
        ("correlation", {"point_labels": ("a", "b")}, "one text per pair"),
        ("correlation", {"label_var": None}, "point_labels need label_var"),
        ("correlation", {"descriptives": "first"}, "descriptives must be empty"),
        ("correlation", {"pairwise": "any"}, "pairwise must be empty"),
        ("correlation", {"subjects": ("s1",)}, "subjects must be empty"),
        ("correlation", {"design": "pairs"}, "unknown design"),
    ],
)
def test_wrong_shapes_are_refused(
    one_sample: StatsResult,
    correlation: StatsResult,
    anova: StatsResult,
    design: str,
    changes: dict[str, Any],
    problem: str,
) -> None:
    result = one_sample if design == "one-sample" else correlation
    if changes.get("descriptives") == "first":
        changes = {"descriptives": anova.descriptives[:1]}
    if changes.get("pairwise") == "any":
        changes = {"pairwise": anova.pairwise}
    with pytest.raises(ValueError, match=re.escape(problem)):
        dataclasses.replace(result, **changes)


def test_settings_must_fit_the_design(
    one_sample: StatsResult, correlation: StatsResult, anova: StatsResult
) -> None:
    settings = one_sample.settings
    with pytest.raises(ValueError, match="test_value is required"):
        dataclasses.replace(
            one_sample, settings=dataclasses.replace(settings, test_value=None)
        )
    with pytest.raises(ValueError, match="test_value must be finite"):
        dataclasses.replace(
            one_sample, settings=dataclasses.replace(settings, test_value=math.inf)
        )
    with pytest.raises(ValueError, match="must be None in every other design"):
        dataclasses.replace(
            anova, settings=dataclasses.replace(anova.settings, test_value=1.0)
        )
    with pytest.raises(ValueError, match="correlations are tested two-sided"):
        dataclasses.replace(
            correlation,
            settings=dataclasses.replace(correlation.settings, alternative="less"),
        )
    # The comparison designs keep their y and refuse the new fields.
    with pytest.raises(ValueError, match="y is None in design 'one-sample'"):
        dataclasses.replace(anova, y=None)
    with pytest.raises(ValueError, match="belong to design 'correlation' only"):
        dataclasses.replace(anova, point_labels=("a",))
    # JSON that breaks a rule is refused when read back.
    data = json.loads(one_sample.to_json())
    data["levels"] = ["a"]
    with pytest.raises(ValueError, match="levels must be empty"):
        StatsResult.from_dict(data)


def test_the_copy_without_values_is_valid(
    one_sample: StatsResult, correlation: StatsResult
) -> None:
    for result in (one_sample, correlation):
        assert dataclasses.replace(result, values=(), point_labels=()).values == ()


def test_new_tied_counts() -> None:
    from inferplot.result import correlation_tied_count, one_sample_tied_count

    # |x - 3|: 0 (dropped), 1, 1, 2: 4 values, 2 distinct nonzero sizes.
    assert one_sample_tied_count([3.0, 2.0, 4.0, 5.0], 3.0) == 2
    assert one_sample_tied_count([1.0, 2.0, 4.0], 0.0) == 0
    assert correlation_tied_count([1.0, 1.0, 2.0], [5.0, 6.0, 7.0]) == 1
    assert correlation_tied_count([1.0, 1.0, 1.0], [5.0, 5.0, 7.0]) == 3
    with pytest.raises(ValueError, match="must pair up"):
        correlation_tied_count([1.0], [1.0, 2.0])


def test_checks_of_the_new_designs_speak_of_values_and_pairs(
    one_sample: StatsResult, correlation: StatsResult
) -> None:
    # edge_one_missing: 2 rows without x.
    (issue,) = one_sample.check()
    assert issue.code == "rows_dropped"
    assert issue.message == "2 rows were left out because 'value' is missing."
    assert correlation.check() == ()
    small = dataclasses.replace(
        one_sample,
        n_obs=4,
        values=(),
        descriptives=(dataclasses.replace(one_sample.descriptives[0], n=4),),
        n_dropped=0,
    )
    (issue,) = small.check()
    assert issue.message == (
        "Only 4 values of 'value' were analyzed, fewer than 5. The test and its "
        "interval are unreliable with so few values."
    )
    pairs = dataclasses.replace(
        correlation, n_obs=3, values=(), point_labels=(), n_dropped=2
    )
    codes = {i.code: i for i in pairs.check()}
    assert codes["small_n"].message == (
        "Only 3 complete pairs of 'wt' and 'mpg' were analyzed, fewer than 5. The "
        "test and its interval are unreliable with so few pairs."
    )
    assert codes["rows_dropped"].message == (
        "2 rows were left out because 'wt' or 'mpg' is missing."
    )


def test_heavy_ties_of_the_new_designs(
    one_sample: StatsResult, correlation: StatsResult
) -> None:
    ranked = dataclasses.replace(one_sample, type="nonparametric", n_tied=4)
    (issue,) = [i for i in ranked.check() if i.code == "heavy_ties"]
    assert issue.message.startswith(
        "4 of the 10 values of 'value' (40%) equal 10 or lie as far from it as "
        "another value, so the signed-rank test drops the equal ones and ranks only "
        "6 distinct distances."
    )
    fewer = dataclasses.replace(ranked, n_tied=2)
    assert "heavy_ties" not in [i.code for i in fewer.check()]
    spearman = dataclasses.replace(correlation, type="nonparametric", n_tied=17)
    (issue,) = [i for i in spearman.check() if i.code == "heavy_ties"]
    assert issue.message.startswith(
        "17 of the 64 values of 'wt' and 'mpg' (27%) tie with another value of the "
        "same column"
    )
    # Only the rank methods.
    parametric = dataclasses.replace(correlation, n_tied=60)
    assert "heavy_ties" not in [i.code for i in parametric.check()]


def test_texts_of_the_new_designs(
    one_sample: StatsResult, correlation: StatsResult
) -> None:
    # ggstatsplot's subtitle and caption, one centrality label for histostats, no
    # tick or pairwise labels; tests/test_no_level_texts.py checks every fixture.
    for result in (one_sample, correlation):
        texts = result.text
        assert texts.tick_labels == texts.pairwise_labels == ()
        assert texts.alt_text.endswith(texts.apa)
        assert str(result) == texts.apa
    texts = one_sample.text
    assert texts.subtitle == (
        "t_Student(9) = 2.22, p = 0.05, g_Hedges = 0.64, CI_95% [-9.17e-03, 1.27], "
        "n_obs = 10"
    )
    assert texts.caption == "log_e(BF_01) = -0.52, r_Cauchy^JZS = 0.71"
    assert texts.centrality_labels == ("\u03bc_mean = 11.02",)
    assert texts.apa.startswith("One-sample t-test against 10: t(9) = ")
    assert texts.alt_text.startswith(
        "Histogram of value (10 values, from 8.90 to 13.20). A vertical line marks "
        "the mean, 11.02. One-sample t-test"
    )
    texts = correlation.text
    assert texts.subtitle == (
        "t_Student(30) = -9.56, p = 1.29e-10, r_Pearson = -0.87, CI_95% [-0.93, "
        "-0.74], n_pairs = 32"
    )
    assert texts.caption == "log_e(BF_01) = -17.84, r_beta^JZS = 1.41"
    assert texts.centrality_labels == ()
    assert texts.apa == (
        "Pearson correlation: t(30) = -9.56, p < .001, Pearson's r = -.87, "
        "95% CI [-.93, -.74], n = 32 pairs."
    )
    assert texts.alt_text == (
        "Scatter plot of mpg against wt (32 pairs): mpg tends to fall as wt rises. "
        "It also shows a least-squares line with its 95% confidence band, marginal "
        "histograms of wt and mpg, and labels from model on every point. " + texts.apa
    )


def _rebuilt(result: StatsResult, **changes: Any) -> Any:
    """The texts of ``result`` with ``changes`` to its parts."""
    parts = {
        "x": result.x,
        "y": result.y,
        "descriptives": result.descriptives,
        "test": result.test,
        "effect_size": result.effect_size,
        "bayes": result.bayes,
        "pairwise": (),
        "n_obs": result.n_obs,
        "settings": result.settings,
        "design": result.design,
    }
    return tx.build_texts(**{**parts, **changes})


def test_sentences_name_direction_and_bayes(
    one_sample: StatsResult, correlation: StatsResult
) -> None:
    assert one_sample.test is not None
    less = dataclasses.replace(one_sample.test, alternative="less")
    assert _rebuilt(one_sample, test=less).apa.startswith(
        "One-sample t-test (one-sided: value less than 10): t(9) = "
    )
    bayes = _bayes_factor()
    one = _rebuilt(one_sample, test=None, effect_size=None, bayes=bayes)
    assert one.apa == (
        "Bayesian one-sample t-test against 10 (JZS prior, r = 0.707): BF10 = 3.00, "
        "log(BF01) = -1.10, n = 10."
    )
    pair = _rebuilt(correlation, test=None, effect_size=None, bayes=bayes)
    assert pair.apa == (
        "Bayesian Pearson correlation (stretched beta prior, r_beta = 0.71): BF10 = "
        "3.00, log(BF01) = -1.10, n = 32 pairs."
    )
    none = _rebuilt(correlation, test=None, effect_size=None, bayes=None)
    assert none.apa == "No test was computed (results_subtitle=False), n = 32 pairs."
    # Long column names are shortened in the description, never in the sentence.
    long = _rebuilt(correlation, x="w" * 300, y="m" * 300)
    assert len(long.alt_text) <= 400
    assert long.alt_text.endswith(correlation.text.apa)
    with pytest.raises(ValueError, match="has no pairwise comparisons"):
        _rebuilt(correlation, pairwise=_anova_pairs())


def _anova_pairs() -> Any:
    return _no_nan(result_from_fixture(_case("PlantGrowth-parametric"))).pairwise
