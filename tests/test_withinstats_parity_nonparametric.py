"""``ip.analyze.withinstats(type="nonparametric")`` against every nonparametric
ggwithinstats fixture.

Each fixture is replayed through ``ip.analyze.withinstats`` on its CSV read as
pandas, polars and pyarrow (``tests/within_np_cases.py``), and every number the
fixture holds is compared under its ``tests/tolerances.py`` kind: the signed-rank
V and its p with the rank-biserial r and its interval (two conditions); Friedman's
chi-square, df and p, Kendall's W and its bootstrap bound (against the fixture's
20,000-resample reference within Monte Carlo error), the adjusted Durbin-Conover
p-values and |t| (three or more); the drawn pairs, the centrality medians and n.
The fields R's pairwise rows do not carry (each pair's unadjusted p and df) are
compared with an independent computation of Conover's formula. Texts are not
tested here (the withinstats text layer is adapted separately).

edge_within_small is the one case where ggstatsplot swallows an error: every
subject ranks the conditions alike, effectsize's bootstrap of Kendall's W fails,
and ggstatsplot leaves out the subtitle. inferplot reports the test and W = 1 with
the interval [1, 1] and a warnings entry (``test_hidden_errors``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from tests import within_np_cases as nc
from tests import within_pb_cases as wc
from tests.oracle_fixtures import number, probe, r_double
from tests.tolerances import assert_close, assert_within_monte_carlo

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

_METHODS = {
    "wilcoxon_signed_rank": "Wilcoxon signed rank test",
    "friedman": "Friedman rank sum test",
}
_EFFECT_SIZES = {"rank_biserial": "r (rank biserial)", "kendalls_w": "Kendall's W"}


def test_fixture_counts() -> None:
    # The brief's inventory: 15 cases with two conditions, 25 with three or more,
    # and edge_rm_some_tied (review T7); ggstatsplot fails on none of these (it fails
    # on edge_rm_all_tied, a failed case), and swallows an error in the HIDDEN ones.
    assert (len(nc.TWO), len(nc.MORE)) == (15, 26)
    hidden = {c.name: c.expected_hidden_errors for c in nc.CASES}
    assert {name: e for name, e in hidden.items() if e} == {
        name: (component,) for name, component in nc.HIDDEN.items()
    }


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.CASES))
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


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.CASES))
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


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.CASES))
def test_main_test(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    test = result.test
    assert test is not None  # computed also without a subtitle
    if not case.subtitle_rows:
        # results.subtitle = FALSE, or the subtitle ggstatsplot drops (HIDDEN).
        assert case.args.get("results.subtitle") is False or name in nc.HIDDEN
        assert (result.text.subtitle is None) == (name not in nc.HIDDEN)
        return
    row = case.subtitle_data
    assert test.method == row["method"] == _METHODS[test.name]
    alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
    assert test.alternative == alternative
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="stat")
    assert_close("closed_form", test.p_value, number(row["p.value"]), what="p")
    assert number(row["n.obs"]) == result.n_obs
    assert row.get("df") is None
    assert test.df is None
    if test.name == "friedman":
        assert test.df_error is not None
        assert_close("closed_form", test.df_error, number(row["df.error"]), what="df")
    else:
        assert row.get("df.error") is None
        assert test.df_error is None


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.CASES))
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
    reference = case.bootstrap_reference
    if reference is None:  # closed form: the rank-biserial r
        assert es.name == "rank_biserial"
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
        return
    assert es.name == "kendalls_w"
    assert row["conf.method"] == es.ci_method == "percentile bootstrap"
    assert es.bootstrap_resamples == stats.BOOTSTRAP_RESAMPLES
    assert es.bootstrap_seed == result.settings.seed == result.provenance.seed
    assert number(reference["conf_level"]) == es.conf_level
    assert es.sidedness == "greater"
    assert_close("closed_form", es.estimate, number(row["estimate"]), what="W")
    assert_close("closed_form", es.estimate, number(reference["estimate"]), what="W")
    replicates = reference["replicates"]
    for bound in ("conf_low", "conf_high"):
        assert_within_monte_carlo(
            getattr(es, bound),
            number(reference[bound]),
            number(replicates[f"{bound}_sd"]),
            what=f"{name} {bound}",
        )


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.CASES))
def test_no_bayes_factor(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    assert case.caption_data is None
    assert result.bayes is None


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.CASES))
def test_pairwise_rows(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    k = len(result.levels)
    if k == 2:
        assert result.pairwise == ()
        assert case.pairwise == []
        return
    assert len(result.pairwise) == k * (k - 1) // 2
    if case.args.get("pairwise.display") == "none":
        # ggstatsplot skips the comparisons; inferplot computes them, draws none.
        assert case.pairwise == []
        assert not any(pair.displayed for pair in result.pairwise)
        return
    rows = wc.pair_rows(case, result)
    method = str(case.args.get("p.adjust.method", "holm"))
    for pair, row in zip(result.pairwise, rows, strict=True):
        what = f"{name} {pair.group1}-{pair.group2}"
        assert pair.test_label == row["test"] == "Durbin-Conover"
        assert pair.test == "durbin_conover"
        assert pair.statistic_name == "t"
        assert row["distribution"] == "t"
        assert pair.adjust_method == method
        assert pair.p_adjusted is not None
        assert pair.significant == (pair.p_adjusted < 0.05)
        assert_close("closed_form", pair.statistic, number(row["statistic"]), what=what)
        assert_close("closed_form", pair.p_adjusted, number(row["p.value"]), what=what)


def _conover(columns: Sequence[FloatArray]) -> dict[tuple[int, int], tuple[float, ...]]:
    """|t|, df and the two-sided p of every pair by Conover's (1999, section 5.8)
    formula with the Friedman statistic, from SciPy's within-row ranks."""
    ranks = np.apply_along_axis(scipy_stats.rankdata, 1, np.column_stack(columns))
    b, k = ranks.shape
    sums = ranks.sum(axis=0)
    a = float((ranks**2).sum())
    c = b * k * (k + 1) ** 2 / 4
    t1 = (k - 1) * float(((sums - b * (k + 1) / 2) ** 2).sum()) / (a - c)
    df = float((b - 1) * (k - 1))
    variance = 2 * b * (a - c) / df * (1 - t1 / (b * (k - 1)))
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(float(sums[i] - sums[j]))
            t = diff / math.sqrt(variance) if variance > 1e-9 else math.inf
            out[(i, j)] = (t, df, float(2 * scipy_stats.t.sf(t, df)))
    return out


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.MORE))
def test_pairwise_raw_p_statistic_and_df(name: str, backend: wc.Backend) -> None:
    """The pairwise fields ggstatsplot's rows do not carry, by an independent
    route (``_conover``)."""
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    levels = list(case.analysis_levels)
    reference = _conover(wc.conditions(case))
    unadjusted = result.settings.p_adjust_method == "none"
    for pair in result.pairwise:
        what = f"{name} {pair.group1}-{pair.group2}"
        t, df, p = reference[(levels.index(pair.group1), levels.index(pair.group2))]
        assert pair.df == df
        assert pair.p_value is not None
        assert pair.p_adjusted is not None
        assert pair.p_value <= pair.p_adjusted
        if unadjusted:
            assert pair.p_value == pair.p_adjusted
        assert_close("closed_form", pair.statistic, t, what=f"{what} t")
        assert_close("closed_form", pair.p_value, p, what=f"{what} p")


@pytest.mark.parametrize(("name", "backend"), nc.params(nc.CASES))
def test_drawn_pairs(name: str, backend: wc.Backend) -> None:
    case, result = nc.BY_NAME[name], nc.analyze(name, backend)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == wc.drawn(case)


def test_hidden_errors() -> None:
    """edge_within_small: every subject ranks a < b < c. ggstatsplot computes the
    pairwise rows (t = Inf, p = 0) and the medians, then effectsize's bootstrap of
    Kendall's W fails and the subtitle is dropped without a word. inferplot reports
    what R computes outside ggstatsplot (probe-friedman: Friedman's chi-square 8
    and p, W = 1), the interval [1, 1] and a warnings entry saying why."""
    (name,) = nc.HIDDEN
    case = nc.BY_NAME[name]
    (suppressed,) = [s for s in case.raw["suppressed"] if s["component"] == "subtitle"]
    assert suppressed["error"] == "arguments imply differing number of rows: 1, 0"
    assert not case.subtitle_rows
    (design,) = [
        d
        for d in probe("probe-friedman").rows("designs")
        if d["name"] == "edge_within_small"
    ]
    assert design["kendalls_w_ci_error"] == suppressed["error"]
    result = nc.analyze(name, "pandas")
    test, es = result.test, result.effect_size
    assert test is not None
    assert es is not None
    for got, column in ((test.statistic, "statistic"), (test.p_value, "p_value")):
        expected = r_double(design[f"friedman_{column}_hex"])
        assert expected is not None
        assert_close("closed_form", got, expected, what=column)
    assert r_double(design["kendalls_w_hex"]) == es.estimate == 1.0
    assert (es.conf_low, es.conf_high) == (1.0, 1.0)
    assert result.text.subtitle is not None
    assert all(p.statistic == math.inf and p.p_value == 0.0 for p in result.pairwise)
    (entry,) = [w for w in result.warnings if "left out" not in w]
    assert entry.startswith(
        "All 4 subjects rank the levels of x='condition' in the same order, so "
        "Kendall's W is 1 and so is every bootstrap resample"
    )


# Warnings these fixtures add beyond the data notes: the collapsed rank-biserial
# interval where every nonzero difference has the same sign, the identical
# rankings of edge_within_small (test_hidden_errors), and Kendall's W resamples of
# constant subjects only, left out of the interval (D54, D64; edge_rm_some_tied).
@pytest.mark.parametrize("name", nc.names(nc.CASES))
def test_warnings(name: str) -> None:
    result = nc.analyze(name, "pandas")
    method_warnings = [
        w for w in result.warnings if "left out of the analysis" not in w
    ]
    es = result.effect_size
    assert es is not None
    collapsed = abs(es.estimate) == 1.0 and es.conf_low == es.conf_high
    undefined = (es.undefined_resamples or 0) > 0
    undefined_notes = [w for w in method_warnings if "bootstrap resamples" in w]
    assert len(undefined_notes) == int(undefined)
    method_warnings = [w for w in method_warnings if w not in undefined_notes]
    assert len(method_warnings) == int(collapsed)
    if collapsed and es.name == "rank_biserial":
        (entry,) = method_warnings
        assert entry.startswith("Every nonzero difference between ")
        assert "has the same sign, so the rank-biserial r is at its bound" in entry
    elif collapsed:
        assert name in nc.HIDDEN


@pytest.mark.parametrize("name", nc.names(nc.CASES))
def test_backends_give_identical_results(name: str) -> None:
    first = nc.analyze(name, "pandas").to_json()
    for backend in wc.BACKENDS[1:]:
        assert nc.analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", nc.names(nc.CASES))
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


THREE_FRAME = _long(
    {
        "a": [1.0, 2.0, 4.0, 3.5, 6.0, 2.0],
        "b": [2.0, 2.5, 5.5, 3.0, 8.0, 2.5],
        "c": [0.5, 2.5, 3.0, 4.5, 6.5, 3.0],
    }
)


def test_three_conditions_arguments() -> None:
    result = _within(THREE_FRAME, bf_prior=0.8, p_adjust_method="BH")
    assert list(result.warnings) == [
        "bf_prior=0.8 has no effect: no Bayes factor is computed "
        "(type='nonparametric' has no Bayes factor)."
    ]
    assert {p.adjust_method for p in result.pairwise} == {"BH"}
    with pytest.raises(ValueError, match="alternative='less' applies only to two"):
        _within(THREE_FRAME, alternative="less")
    none = _within(THREE_FRAME, pairwise_display="none")
    assert len(none.pairwise) == 3
    assert not any(p.displayed for p in none.pairwise)
    every = _within(THREE_FRAME, pairwise_display="all")
    assert all(p.displayed for p in every.pairwise)


def test_seed_and_conf_level_reach_kendalls_w() -> None:
    default = _within(THREE_FRAME).effect_size
    other = _within(THREE_FRAME, seed=7).effect_size
    wider = _within(THREE_FRAME, conf_level=0.99).effect_size
    assert default is not None
    assert other is not None
    assert wider is not None
    assert (default.bootstrap_seed, other.bootstrap_seed) == (0, 7)
    assert default.estimate == other.estimate == wider.estimate
    assert wider.conf_level == 0.99
    assert wider.conf_low <= default.conf_low


def test_order_of_subjects_and_levels_does_not_change_the_interval() -> None:
    base = _within(THREE_FRAME)
    shuffled = THREE_FRAME.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert _within(shuffled).to_json() == base.to_json()
    reordered = _within(THREE_FRAME, order=["c", "a", "b"])
    assert reordered.effect_size == base.effect_size


def test_identical_rankings_report_w_one() -> None:
    frame = _long({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.5], "c": [7.0, 9.0, 8.0]})
    result = _within(frame)
    es = result.effect_size
    assert es is not None
    assert (es.estimate, es.conf_low, es.conf_high) == (1.0, 1.0, 1.0)
    assert [(p.statistic, p.p_value, p.p_adjusted) for p in result.pairwise] == [
        (math.inf, 0.0, 0.0)
    ] * 3
    (entry,) = result.warnings
    assert entry.startswith("All 3 subjects rank the levels of x='condition'")
    assert '"Inf"' in result.to_json()
    assert StatsResult.from_json(result.to_json()) == result


def test_every_subject_tied_raises() -> None:
    frame = _long({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0], "c": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="nothing to rank within") as error:
        _within(frame)
    assert str(error.value) == (
        "Every subject has the same value at all 3 levels of x='condition', so "
        "Friedman's test has nothing to rank within the subjects: its statistic "
        "and Kendall's W are 0/0. There is no difference to test; check the data "
        "for entry errors."
    )


def test_identical_rankings_with_a_tied_pair_raise() -> None:
    frame = _long({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match="'b' and 'c' share their rank in every"):
        _within(frame)


def test_resamples_of_tied_subjects_are_left_out() -> None:
    # Two of three subjects have the same value in every condition: about
    # (2/3)^3 of the resamples draw only them, where W is 0/0 (D54).
    frame = _long({"a": [1.0, 5.0, 2.0], "b": [1.0, 5.0, 3.0], "c": [1.0, 5.0, 1.0]})
    result = _within(frame)
    es = result.effect_size
    assert es is not None
    (entry,) = result.warnings
    tied = int(entry.split(" of ")[0].replace(",", ""))
    assert 450 < tied < 750
    assert entry == (
        f"{tied:,} of 2,000 bootstrap resamples draw only subjects with the same "
        "value in every condition, where Kendall's W is undefined; its confidence "
        f"interval comes from the other {2000 - tied:,}."
    )
    assert es.estimate == 1.0 / 3.0


def test_extreme_conf_level_raises_plainly() -> None:
    with pytest.raises(ValueError, match="The bootstrap interval of Kendall's W"):
        _within(THREE_FRAME, conf_level=0.9999)
