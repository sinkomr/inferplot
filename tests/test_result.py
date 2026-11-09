"""StatsResult: plain types, JSON, frames, check() and printing."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import math
import re
from typing import Any

import numpy as np
import pytest

from inferplot import text as tx
from inferplot.check import SMALL_N, Issue, result_issues
from inferplot.result import (
    PARITY_TARGET,
    Descriptive,
    PairwiseComparison,
    StatsResult,
    TestResult,
    current_provenance,
)
from tests.oracle_fixtures import OracleCase, load_cases
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
