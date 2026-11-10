"""``ip.analyze.withinstats(type="nonparametric")`` against every nonparametric
ggwithinstats fixture.

Each fixture is replayed through ``ip.analyze.withinstats`` on its CSV read as
pandas, polars and pyarrow (``tests/within_np_cases.py``), and every number the
fixture holds is compared under its ``tests/tolerances.py`` kind: the signed-rank
V and its p with the rank-biserial r and its interval (two conditions), the
centrality medians and n. Texts are not tested here (the withinstats text layer is
adapted separately).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot.result import StatsResult
from tests import within_np_cases as nc
from tests import within_pb_cases as wc
from tests.oracle_fixtures import number
from tests.tolerances import assert_close

_METHODS = {"wilcoxon_signed_rank": "Wilcoxon signed rank test"}
_EFFECT_SIZES = {"rank_biserial": "r (rank biserial)"}


def test_fixture_counts() -> None:
    # The brief's inventory: 15 cases with two conditions, 25 with three or more.
    assert (len(nc.TWO), len(nc.MORE)) == (15, 25)
    hidden = {c.name for c in nc.CASES if c.expected_hidden_errors}
    assert hidden == set(nc.HIDDEN)


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.TWO))
def test_levels_counts_and_values(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    assert (result.analysis, result.design, result.type) == (
        "withinstats",
        "within",
        "nonparametric",
    )
    data = case.within_data
    n = len(data["subjects"])
    assert result.levels == case.analysis_levels
    assert result.subjects == tuple(str(s) for s in data["subjects"])
    assert result.n_obs == n
    assert {d.level: d.n for d in result.descriptives} == dict.fromkeys(
        result.levels, n
    )
    expected = [list(column) for column in wc.conditions(case)]
    assert [list(v) for v in result.values] == expected
    assert result.dropped_subjects == len(data["dropped_subjects"])


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.TWO))
def test_centrality(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    y = str(case.args["y"])
    for layer in wc.centrality_layers(case):
        assert list(layer) == list(result.levels)
        for d in result.descriptives:
            row = layer[d.level]
            assert d.centrality == "median"
            assert int(row["n.obs"]) == d.n
            assert d.value is not None
            assert_close("closed_form", d.value, number(row[y]), what=f"{name} median")


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.TWO))
def test_main_test(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    test = result.test
    assert test is not None  # computed also without a subtitle
    if not case.subtitle_rows:  # results.subtitle = FALSE: R reports nothing
        assert case.args.get("results.subtitle") is False
        assert result.text.subtitle is None
        return
    row = case.subtitle_data
    assert test.method == row["method"] == _METHODS[test.name]
    alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
    assert test.alternative == alternative
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="stat")
    assert_close("closed_form", test.p_value, number(row["p.value"]), what="p")
    assert number(row["n.obs"]) == result.n_obs
    assert "df" not in row or row["df"] is None
    assert "df.error" not in row or row["df.error"] is None
    assert (test.df, test.df_error) == (None, None)


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.TWO))
def test_effect_size(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    es = result.effect_size
    assert es is not None
    if not case.subtitle_rows:
        return
    row = case.subtitle_data
    assert row["effectsize"] == _EFFECT_SIZES[es.name]
    level = float(case.args.get("conf.level", 0.95))
    assert es.conf_level == number(row["conf.level"]) == level
    assert es.bootstrap_resamples is None
    assert es.ci_method == "normal approximation (Fisher z)"
    alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
    assert es.sidedness == alternative
    for field, column in (
        ("estimate", "estimate"),
        ("conf_low", "conf.low"),
        ("conf_high", "conf.high"),
    ):
        value, expected = getattr(es, field), number(row[column])
        assert_close("closed_form", value, expected, what=f"{name} {column}")


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.TWO))
def test_no_bayes_factor_and_no_pairs(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    assert case.caption_data is None
    assert result.bayes is None
    assert result.pairwise == ()
    assert case.pairwise == []


# Warnings these fixtures add beyond the data notes: the collapsed rank-biserial
# interval where every nonzero difference has the same sign.
@pytest.mark.parametrize("name", nc.names(nc.TWO))
def test_warnings(name: str) -> None:
    result = nc.analyze(name, "pandas")
    method_warnings = [
        w for w in result.warnings if "left out of the analysis" not in w
    ]
    es = result.effect_size
    assert es is not None
    collapsed = abs(es.estimate) == 1.0 and es.conf_low == es.conf_high
    assert len(method_warnings) == int(collapsed)
    if collapsed:
        (entry,) = method_warnings
        assert entry.startswith("Every nonzero difference between ")
        assert "has the same sign, so the rank-biserial r is at its bound" in entry


@pytest.mark.parametrize("name", nc.names(nc.TWO))
def test_backends_give_identical_results(name: str) -> None:
    first = nc.analyze(name, "pandas").to_json()
    for backend in wc.BACKENDS[1:]:
        assert nc.analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", nc.names(nc.TWO))
def test_json_round_trip(name: str) -> None:
    result = nc.analyze(name, "pandas")
    assert StatsResult.from_json(result.to_json()) == result
    assert len(result.to_frame()) == len(result.frame_rows())


# --------------------------------------------------------------------------- #
# argument rules and data errors
# --------------------------------------------------------------------------- #


def _long(columns: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {"subject": i, "condition": level, "score": value}
        for level, values in columns.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def _within(frame: pd.DataFrame, **kwargs: Any) -> StatsResult:
    return ip.analyze.withinstats(
        frame, "condition", "score", subject_id="subject", type="np", **kwargs
    )


TWO_FRAME = _long({"pre": [1.0, 2.0, 4.0, 3.5, 6.0], "post": [2.0, 2.5, 5.5, 3.0, 8.0]})


def test_two_conditions_arguments_without_effect_warn() -> None:
    result = _within(
        TWO_FRAME,
        p_adjust_method="BH",
        pairwise_display="all",
        pairwise_alpha=0.1,
        bf_prior=0.8,
    )
    what = "x='condition' has 2 conditions, so there are no pairwise comparisons to"
    assert list(result.warnings) == [
        f"p_adjust_method='BH' has no effect: {what} adjust.",
        f"pairwise_display='all' has no effect: {what} display.",
        f"pairwise_alpha=0.1 has no effect: {what} call significant.",
        "bf_prior=0.8 has no effect: no Bayes factor is computed "
        "(type='nonparametric' has no Bayes factor).",
    ]
    assert _within(TWO_FRAME, pairwise_display="s", bf_prior=0.707).warnings == ()


def test_arguments_that_cannot_apply_raise() -> None:
    with pytest.raises(ValueError, match=r"tr=0\.1 sets the trimming"):
        _within(TWO_FRAME, tr=0.1)
    with pytest.raises(ValueError, match="bf_message=True asks for a Bayes factor"):
        _within(TWO_FRAME, bf_message=True)


def test_alternative_and_conf_level_reach_the_test_and_interval() -> None:
    both = _within(TWO_FRAME)
    greater = _within(TWO_FRAME, alternative="greater")
    less = _within(TWO_FRAME, alternative="less", conf_level=0.9)
    assert both.test is not None
    assert greater.test is not None
    assert less.test is not None
    # Levels in sorted order, so the differences are post - pre: mostly positive.
    assert both.levels == ("post", "pre")
    assert greater.test.p_value < both.test.p_value < less.test.p_value
    es_g, es_l = greater.effect_size, less.effect_size
    assert es_g is not None
    assert es_l is not None
    assert (es_g.sidedness, es_g.conf_high) == ("greater", 1.0)
    assert (es_l.sidedness, es_l.conf_low, es_l.conf_level) == ("less", -1.0, 0.9)


def test_all_zero_differences_raise_naming_the_levels() -> None:
    same = _long({"pre": [1.0, 2.0, 4.0], "post": [1.0, 2.0, 4.0]})
    with pytest.raises(ValueError, match="every paired difference is 0") as error:
        _within(same)
    assert str(error.value) == (
        "Every subject has the same value at 'post' as at 'pre' (x='condition'), so "
        "every paired difference is 0 and the signed-rank test has nothing to rank: "
        "its statistic and the rank-biserial r are 0/0. There is no difference to "
        "test; check the data for entry errors."
    )


def test_constant_nonzero_differences_are_tested() -> None:
    # The paired t-test refuses these (no variance); the signed-rank test does not.
    shifted = _long({"pre": [1.0, 2.0, 4.0, 5.0], "post": [2.0, 3.0, 5.0, 6.0]})
    result = _within(shifted)
    assert result.test is not None
    assert result.effect_size is not None
    # post - pre = +1 for every subject: V = 10, r = 1, interval [1, 1].
    assert result.test.statistic == 10.0
    assert result.effect_size.estimate == 1.0
    assert (result.effect_size.conf_low, result.effect_size.conf_high) == (1.0, 1.0)
    (entry,) = result.warnings
    assert entry.startswith(
        "Every nonzero difference between 'post' and 'pre' of x='condition' has the "
        "same sign, so the rank-biserial r is at its bound (1)"
    )
