"""``ip.analyze.withinstats(type="parametric")`` against every parametric
ggwithinstats fixture.

Each fixture is replayed through ``ip.analyze.withinstats`` on its CSV read as
pandas, polars and pyarrow (``tests/within_pb_cases.py``), and every number the
fixture holds is compared under its ``tests/tolerances.py`` kind: the paired t or
the repeated-measures F with the Greenhouse-Geisser df, p, Hedges' g or partial
omega squared with its interval, the caption's paired Bayes factor, the adjusted
pairwise p-values, the drawn pairs, the centrality means and n. The fields R's
rows do not carry (each pair's t, df and unadjusted p) are compared with SciPy's
``ttest_rel``. Texts are not tested here (the withinstats text layer is adapted
separately).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from scipy import stats as scipy_stats

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from tests import within_pb_cases as wc
from tests.oracle_fixtures import DATASET_DIR, FIXTURE_DIR, cases_for, number
from tests.tolerances import assert_close, is_close

CASES = wc.cases_of("parametric")
TWO = [c for c in CASES if len(c.analysis_levels) == 2]
MORE = [c for c in CASES if len(c.analysis_levels) >= 3]
_METHODS = {
    "paired_t": "Paired t-test",
    "rm_anova": "ANOVA estimation for factorial designs using 'afex'",
}
_EFFECT_SIZES = {"hedges_g": "Hedges' g", "partial_omega_squared": "Omega2 (partial)"}


def test_fixture_counts() -> None:
    # The brief's inventory: 18 cases with two conditions, 26 with three or more,
    # and no case where ggstatsplot failed.
    assert (len(TWO), len(MORE)) == (18, 26)
    failed = cases_for("ggwithinstats", failed=True)
    assert not [c for c in failed if c.type in ("parametric", "bayes")]


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_levels_counts_and_values(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    assert (result.analysis, result.design, result.type) == (
        "withinstats",
        "within",
        "parametric",
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


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_centrality(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    y = str(case.args["y"])
    for layer in wc.centrality_layers(case):
        assert list(layer) == list(result.levels)
        for d in result.descriptives:
            row = layer[d.level]
            assert d.centrality == "mean"
            assert int(row["n.obs"]) == d.n
            assert d.value is not None
            assert_close("closed_form", d.value, number(row[y]), what=f"{name} mean")


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_main_test(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
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
    for field, column in (("df", "df"), ("df_error", "df.error")):
        value = getattr(test, field)
        if column in row and row[column] is not None:
            assert value is not None
            assert_close("closed_form", value, number(row[column]), what=column)
        else:
            assert value is None, field


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_effect_size(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    es = result.effect_size
    assert es is not None
    if not case.subtitle_rows:
        return
    row = case.subtitle_data
    assert row["effectsize"] == _EFFECT_SIZES[es.name]
    level = float(case.args.get("conf.level", 0.95))
    assert es.conf_level == number(row["conf.level"]) == level
    assert es.bootstrap_resamples is None
    assert_close("closed_form", es.estimate, number(row["estimate"]), what="estimate")
    for field, column in (("conf_low", "conf.low"), ("conf_high", "conf.high")):
        value, expected = getattr(es, field), number(row[column])
        assert_close("noncentral_ci", value, expected, what=f"{name} {column}")
    if es.name == "hedges_g":
        alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
        assert es.sidedness == alternative
    else:
        assert es.sidedness == "greater"
        assert es.conf_high == 1.0


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_caption_bayes_factor(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    caption = case.caption_data
    if len(result.levels) >= 3:
        # ggstatsplot's caption is a Bayesian repeated-measures ANOVA (v0.2).
        assert result.bayes is None
        has_caption = caption is not None
        assert has_caption == (
            case.args.get("results.subtitle") is not False
            and case.args.get("bf.message") is not False
        )
        warned = "The Bayes factor caption for three or more conditions arrives in "
        assert any(w.startswith(warned) for w in result.warnings) == has_caption
        return
    if caption is None:
        assert result.bayes is None
        assert (
            case.args.get("results.subtitle") is False
            or case.args.get("bf.message") is False
        )
        return
    bayes = result.bayes
    assert bayes is not None
    assert bayes.prior == caption["prior.distribution"] == "cauchy"
    assert bayes.prior_scale == number(caption["prior.scale"])
    assert bayes.prior_scale == float(case.args.get("bf.prior", 0.707))
    assert bayes.test == "jzs_paired"
    assert_close("ttest_bayes_factor", bayes.bf10, number(caption["bf10"]), what="bf")
    assert_close(
        "ttest_bayes_factor",
        bayes.log_bf01,
        -number(caption["log_e_bf10"]),
        what="log_bf01",
    )


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_pairwise_rows(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    k = len(result.levels)
    if k == 2:
        assert result.pairwise == ()
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
        assert pair.test_label == row["test"] == "Student's t"
        assert pair.test == "paired_t"
        assert pair.adjust_method == method
        assert pair.p_adjusted is not None
        assert pair.significant == (pair.p_adjusted < 0.05)
        assert_close("closed_form", pair.p_adjusted, number(row["p.value"]), what=what)


@pytest.mark.parametrize(("name", "backend"), wc.params(MORE))
def test_pairwise_raw_p_statistic_and_df(name: str, backend: wc.Backend) -> None:
    """The pairwise fields ggstatsplot's rows do not carry, by an independent
    route: each pair is SciPy's ``ttest_rel`` of its two conditions (group1 minus
    group2), its own differences and n - 1 df, not a pooled error term."""
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    columns = dict(zip(case.analysis_levels, wc.conditions(case), strict=True))
    unadjusted = result.settings.p_adjust_method == "none"
    for pair in result.pairwise:
        what = f"{name} {pair.group1}-{pair.group2}"
        a, b = columns[pair.group1], columns[pair.group2]
        reference = scipy_stats.ttest_rel(a, b)
        assert pair.df == a.size - 1.0
        assert pair.p_value is not None
        assert pair.p_adjusted is not None
        assert pair.p_value <= pair.p_adjusted
        if unadjusted:
            assert pair.p_value == pair.p_adjusted
        assert_close(
            "closed_form", pair.statistic, float(reference.statistic), what=f"{what} t"
        )
        assert_close(
            "closed_form", pair.p_value, float(reference.pvalue), what=f"{what} p"
        )


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_drawn_pairs(name: str, backend: wc.Backend) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, backend)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == wc.drawn(case)


@pytest.mark.parametrize("name", wc.names(MORE))
def test_omega_interval_uses_the_uncorrected_df(name: str) -> None:
    """Negative control: the omega squared bound at the Greenhouse-Geisser df
    misses the fixtures where the bound is above 0 (bugs_long)."""
    case, result = wc.BY_NAME[name], wc.analyze(name, "pandas")
    es, test = result.effect_size, result.test
    assert es is not None
    assert test is not None
    if not case.subtitle_rows or es.conf_low == 0.0:
        return
    omega = es.estimate
    df1, df2 = test.df, test.df_error
    assert df1 is not None
    assert df2 is not None
    ncp = stats.f_ncp_bound(
        omega * df2 / ((1 - omega) * df1), df1, df2, prob=es.conf_level
    )
    assert not is_close(
        "noncentral_ci", ncp / (ncp + df2), number(case.subtitle_data["conf.low"])
    )


# Warnings these fixtures add beyond the data notes: the v0.2 caption for three or
# more conditions (checked in test_caption_bayes_factor) and none other.
@pytest.mark.parametrize("name", wc.names(CASES))
def test_warnings(name: str) -> None:
    case, result = wc.BY_NAME[name], wc.analyze(name, "pandas")
    k = len(result.levels)
    method_warnings = [
        w
        for w in result.warnings
        if "left out of the analysis" not in w
        and not w.startswith("The Bayes factor caption for three or more")
    ]
    expected: list[str] = []
    side = str(case.args.get("alternative", "two.sided"))
    if k == 2 and side != "two.sided" and result.bayes is not None:
        expected.append(
            "The Bayes factor in the caption is two-sided: it weighs any difference "
            f"between the conditions, while alternative={side!r} applies to the "
            "t-test and the Hedges' g interval only (as in ggstatsplot)."
        )
    assert method_warnings == expected


@pytest.mark.parametrize("name", wc.names(CASES))
def test_backends_give_identical_results(name: str) -> None:
    first = wc.analyze(name, "pandas").to_json()
    for backend in wc.BACKENDS[1:]:
        assert wc.analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", wc.names(CASES))
def test_json_round_trip(name: str) -> None:
    result = wc.analyze(name, "pandas")
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
        frame, "condition", "score", subject_id="subject", **kwargs
    )


TWO_FRAME = _long({"pre": [1.0, 2.0, 4.0, 3.5, 6.0], "post": [2.0, 2.5, 5.5, 3.0, 8.0]})
THREE_FRAME = _long(
    {
        "a": [1.0, 2.0, 4.0, 3.5, 6.0],
        "b": [2.0, 2.5, 5.5, 3.0, 8.0],
        "c": [0.5, 2.5, 3.0, 4.5, 6.5],
    }
)


def test_arguments_without_effect_warn() -> None:
    result = _within(
        TWO_FRAME, p_adjust_method="BH", pairwise_display="all", pairwise_alpha=0.1
    )
    what = "x='condition' has 2 conditions, so there are no pairwise comparisons to"
    assert [w for w in result.warnings if what in w] == [
        f"p_adjust_method='BH' has no effect: {what} adjust.",
        f"pairwise_display='all' has no effect: {what} display.",
        f"pairwise_alpha=0.1 has no effect: {what} call significant.",
    ]
    for kwargs, why in (
        ({"bf_message": False}, "bf_message=False leaves the Bayes factor out"),
        ({"results_subtitle": False}, "results_subtitle=False leaves the Bayes"),
    ):
        result = _within(TWO_FRAME, bf_prior=0.8, **kwargs)
        assert result.bayes is None
        (entry,) = [w for w in result.warnings if w.startswith("bf_prior=")]
        assert entry.startswith("bf_prior=0.8 has no effect: no Bayes factor is ")
        assert why in entry
    result = _within(THREE_FRAME, bf_prior=0.8)
    (entry,) = [w for w in result.warnings if w.startswith("bf_prior=")]
    assert "x='condition' has 3 conditions" in entry
    # The defaults and their aliases add nothing.
    plain = _within(TWO_FRAME, pairwise_display="s", bf_prior=0.707)
    assert plain.warnings == ()
    assert plain.bayes is not None
    assert plain.bayes.prior_scale == 0.707


def test_bf_prior_reaches_the_caption() -> None:
    a = _within(TWO_FRAME).bayes
    b = _within(TWO_FRAME, bf_prior=0.8).bayes
    assert a is not None
    assert b is not None
    assert (a.prior_scale, b.prior_scale) == (0.707, 0.8)
    assert a.bf10 != b.bf10


def test_two_subjects_warn_that_g_is_zero() -> None:
    result = _within(_long({"pre": [1.0, 2.0], "post": [3.0, 5.0]}))
    es = result.effect_size
    assert es is not None
    assert (es.estimate, es.conf_low, es.conf_high) == (0.0, 0.0, 0.0)
    assert any(w.startswith("Hedges' g is 0 by construction") for w in result.warnings)


def test_constant_differences_raise_naming_the_levels() -> None:
    same = _long({"pre": [1.0, 2.0, 4.0], "post": [1.0, 2.0, 4.0]})
    with pytest.raises(
        ValueError, match="Every subject has the same value at 'post' as at 'pre'"
    ):
        _within(same)
    shifted = _long({"pre": [1.0, 2.0, 4.0], "post": [2.0, 3.0, 5.0]})
    with pytest.raises(ValueError, match="by the same amount") as error:
        _within(shifted)
    # Levels in sorted order: 'post' comes first.
    assert str(error.value) == (
        "Every subject's value at 'post' differs from its value at 'pre' by the same "
        "amount (exactly, or to within the rounding of y in double precision) "
        "(x='condition'), so the paired differences have no variability and "
        "the paired t-test has no standard error. Check the data for entry errors, "
        "or use type='nonparametric'."
    )
    # Decimal data in which every subject is 0.1 lower: the stored differences are
    # 0.10000000000000009 and 0.09999999999999964, whose standard deviation is
    # rounding, not data. Refused like exactly constant differences (R's t.test
    # stops with "data are essentially constant"), where an exact computation on
    # the doubles reported t = 1,007,035,491,024,313.
    pre = [1.1, 2.3, 3.1, 4.7, 5.2, 6.3]
    decimal = _long({"pre": pre, "post": [v - 0.1 for v in pre]})
    assert len({a - b for a, b in zip(pre, [v - 0.1 for v in pre], strict=True)}) > 1
    for kind in ("parametric", "bayes"):
        with pytest.raises(ValueError, match="by the same amount"):
            _within(decimal, type=kind)
    # Differences that vary by more than the rounding of the values are computed.
    drops = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
    varied = _long(
        {"pre": pre, "post": [v - d for v, d in zip(pre, drops, strict=True)]}
    )
    assert _within(varied).test is not None
    tiny = _long({"pre": [1.0, 1.0, 1.0], "post": [0.0, 0.0, 1e-300]})
    with pytest.raises(ValueError, match="vary by less than double precision"):
        _within(tiny)


def test_rm_anova_data_errors() -> None:
    additive = _long({"a": [1.0, 2.0, 4.0], "b": [2.0, 3.0, 5.0], "c": [0.0, 1.0, 3.0]})
    with pytest.raises(ValueError, match="has no error variance"):
        _within(additive)
    # The same in one decimal: each subject's level plus a condition effect. The
    # residuals are rounding (an exact computation on the doubles gave F = 3.9e30).
    level = [1.1, 2.3, 3.1, 4.7, 5.2, 6.3]
    decimal = _long(
        {"a": level, "b": [v + 0.3 for v in level], "c": [v - 0.1 for v in level]}
    )
    with pytest.raises(ValueError, match=r"no error variance|by the same amount"):
        _within(decimal)
    # One pair with constant differences: the omnibus test exists, the pair does
    # not; nothing is returned in part.
    one_pair = _long(
        {
            "a": [1.0, 2.0, 4.0, 3.0],
            "b": [2.0, 3.0, 5.0, 4.0],
            "c": [0.0, 4.0, 1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="value at 'a' differs from its value at 'b'"):
        _within(one_pair)


def test_fewer_subjects_than_conditions_keeps_the_correction() -> None:
    # Review case T7 (D62), held out of the fixture list in probe-heldout: afex
    # cannot compute a sphericity correction with 3 subjects and 4 conditions, and
    # ggstatsplot prints the uncorrected F(3, 6). inferplot keeps the
    # Greenhouse-Geisser correction and says what ggstatsplot prints. Both sides
    # are pinned: R's F, uncorrected df, p and effect size from the probe.
    held_out = json.loads(
        (FIXTURE_DIR / "probe-heldout.json").read_text(encoding="utf-8")
    )
    (held,) = [
        case["fixture"]
        for case in held_out["cases"]
        if case["name"] == "ggwithinstats-edge_rm_n3k4-parametric"
    ]
    (row,) = held["stats"]["subtitle_data"]
    data = pd.read_csv(DATASET_DIR / "edge_rm_n3k4.csv")
    result = ip.analyze.withinstats(data, "condition", "value", subject_id="subject")
    test, es = result.test, result.effect_size
    assert test is not None
    assert es is not None
    assert (number(row["df"]), number(row["df.error"])) == (3.0, 6.0)
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="F")
    assert_close("closed_form", es.estimate, number(row["estimate"]), what="omega")
    assert (es.conf_low, es.conf_high) == (row["conf.low"], row["conf.high"])
    wide = data.pivot(index="subject", columns="condition", values="value")
    anova = stats.rm_anova([wide[c].to_numpy(dtype=float) for c in wide.columns])
    assert (anova.n, anova.k, anova.df1, anova.df2) == (3, 4, 3.0, 6.0)
    assert_close(
        "closed_form", anova.p_uncorrected, number(row["p.value"]), what="R's p"
    )
    # inferplot's own numbers: epsilon 0.350, F(1.05, 2.10), p = 0.104.
    assert anova.epsilon == pytest.approx(0.34973459244281, rel=1e-12)
    assert test.df == pytest.approx(3 * anova.epsilon)
    assert test.df_error == pytest.approx(6 * anova.epsilon)
    assert test.p_value == pytest.approx(0.10382469558640772, rel=1e-10)
    (note,) = [w for w in result.warnings if "fewer subjects" in w]
    assert note == (
        "With fewer subjects (3) than conditions (4) ggstatsplot reports the "
        "repeated-measures ANOVA without a sphericity correction, because afex "
        "cannot compute one there: F(3, 6), p = 0.0176. inferplot applies the "
        "Greenhouse-Geisser correction as with more subjects (epsilon = 0.35), so "
        "its degrees of freedom and p differ from ggstatsplot's; F is the same."
    )
