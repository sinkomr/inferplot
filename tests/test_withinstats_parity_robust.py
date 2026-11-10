"""``ip.analyze.withinstats(type="robust")`` against every robust ggwithinstats
fixture.

Each fixture is replayed through ``ip.analyze.withinstats`` on its CSV read as
pandas, polars and pyarrow, and every number the fixture holds is compared under
its ``tests/tolerances.py`` kind: Yuen's t for dependent samples or the
trimmed-means repeated-measures F with its Huynh-Feldt df, p (WRS2's 1 - cdf,
``wrs2_p``), the AKP effect size of the paired differences or its average over
the pairs (WRS2's scaling constant applied, as in the betweenstats robust parity),
its bootstrap interval against the 20,000-resample reference, rmmcp's pairwise
estimates, intervals and adjusted p-values, the drawn pairs, the centrality
trimmed means and n. The fields R's rows do not carry (each pair's t, df and
unadjusted p) are compared with an independent computation from SciPy's
``trim_mean`` and ``mstats.winsorize``. Texts are not tested here.

ggstatsplot answers one call that inferplot refuses by design: a one-sided
``alternative`` with type robust (sleep-greater-robust) raises ``ValueError``
(ggstatsplot ignores it); the replay without it must give ggstatsplot's output.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from scipy.stats import mstats

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from tests import within_pb_cases as wc
from tests.oracle_fixtures import (
    Backend,
    OracleCase,
    cases_for,
    number,
    read_dataset,
    wrs2_akp_constant,
)
from tests.tolerances import assert_close, assert_within_monte_carlo

CASES = [c for c in cases_for("ggwithinstats") if c.type == "robust"]
BY_NAME = {c.name: c for c in CASES}
TWO = [c for c in CASES if len(c.analysis_levels) == 2]
MORE = [c for c in CASES if len(c.analysis_levels) >= 3]
EFFECT_CONF_LEVEL = 0.95
_METHODS = {
    "yuen_dependent_t": "Yuen's test on trimmed means for dependent samples",
    "trimmed_rm_anova": (
        "A heteroscedastic one-way repeated measures ANOVA for trimmed means"
    ),
}
_EFFECT_SIZES = {
    "akp_delta": "Algina-Keselman-Penfield robust standardized difference",
    "akp_delta_avg": "Algina-Keselman-Penfield robust standardized difference average",
}


def _refused(case: OracleCase) -> bool:
    """A one-sided alternative: inferplot raises, the replay drops it."""
    _, kwargs = wc.call_arguments(case)
    return kwargs.get("alternative", "two-sided") != "two-sided"


@functools.cache
def analyze(name: str, backend: Backend) -> StatsResult:
    case = BY_NAME[name]
    args, kwargs = wc.call_arguments(case)
    if _refused(case):
        kwargs.pop("alternative")
    return ip.analyze.withinstats(read_dataset(case, backend), *args, **kwargs)


def _tr(case: OracleCase) -> float:
    return float(case.args.get("tr", 0.2))


def _as_wrs2(estimate: float, tr: float) -> float:
    return (
        estimate
        * wrs2_akp_constant(tr)
        / math.sqrt(stats.winsorized_normal_variance(tr))
    )


def test_fixture_counts() -> None:
    # The brief's inventory: 15 cases with two conditions, 26 with three or more,
    # and no case where ggstatsplot failed.
    assert (len(TWO), len(MORE)) == (15, 26)
    failed = cases_for("ggwithinstats", failed=True)
    assert not [c for c in failed if c.type == "robust"]
    assert [c.name for c in CASES if _refused(c)] == [
        "ggwithinstats-sleep-greater-robust"
    ]


@pytest.mark.parametrize("name", wc.names([c for c in CASES if _refused(c)]))
def test_one_sided_alternative_is_refused(name: str) -> None:
    case = BY_NAME[name]
    args, kwargs = wc.call_arguments(case)
    with pytest.raises(ValueError, match="alternative='greater' applies to"):
        ip.analyze.withinstats(read_dataset(case), *args, **kwargs)
    # The replay without it is ggstatsplot's output (the two-sided case).
    plain = BY_NAME["ggwithinstats-sleep-robust"]
    assert case.subtitle_data["statistic"] == plain.subtitle_data["statistic"]
    assert analyze(name, "pandas").to_json() == analyze(plain.name, "pandas").to_json()


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_levels_counts_and_values(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    assert (result.analysis, result.design, result.type) == (
        "withinstats",
        "within",
        "robust",
    )
    data = case.within_data
    n = len(data["subjects"])
    assert result.levels == case.analysis_levels
    assert result.subjects == tuple(str(s) for s in data["subjects"])
    assert result.n_obs == n
    assert {d.level: d.n for d in result.descriptives} == dict.fromkeys(
        result.levels, n
    )
    assert [list(v) for v in result.values] == [
        list(column) for column in wc.conditions(case)
    ]
    assert result.dropped_subjects == len(data["dropped_subjects"])
    assert result.settings.tr == _tr(case)


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_centrality(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    y = str(case.args["y"])
    for layer in wc.centrality_layers(case):
        assert list(layer) == list(result.levels)
        for d in result.descriptives:
            row = layer[d.level]
            assert d.centrality == "trimmed_mean"
            assert int(row["n.obs"]) == d.n
            assert d.value is not None
            assert_close("closed_form", d.value, number(row[y]), what=f"{name} tmean")


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_main_test(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    test = result.test
    assert test is not None  # computed also without a subtitle
    assert test.alternative == "two-sided"
    if not case.subtitle_rows:  # results.subtitle = FALSE: R reports nothing
        assert case.args.get("results.subtitle") is False
        assert result.text.subtitle is None
        return
    row = case.subtitle_data
    assert test.method == row["method"] == _METHODS[test.name]
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="stat")
    # R's p is 1 - cdf, rounded to 2^-53 (two-sided: 2 x that step; D43).
    two = len(result.levels) == 2
    assert_close(
        "wrs2_p",
        test.p_value,
        number(row["p.value"]),
        what="p",
        atol_scale=2.0 if two else 1.0,
    )
    for field, column in (("df", "df"), ("df_error", "df.error")):
        value = getattr(test, field)
        if column in row and row[column] is not None:
            assert value is not None
            assert_close("closed_form", value, number(row[column]), what=column)
        else:
            assert value is None, field


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_effect_size(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    es = result.effect_size
    assert es is not None
    # Always a 95% percentile bootstrap interval from 2,000 seeded resamples.
    assert es.conf_level == EFFECT_CONF_LEVEL
    assert (es.ci_method, es.sidedness) == ("percentile bootstrap", "two-sided")
    assert (es.bootstrap_resamples, es.bootstrap_seed) == (2000, 0)
    reference = case.bootstrap_reference
    if not case.subtitle_rows:
        assert reference is None
        return
    row = case.subtitle_data
    assert row["effectsize"] == _EFFECT_SIZES[es.name]
    assert number(row["conf.level"]) == EFFECT_CONF_LEVEL
    assert_close(
        "closed_form",
        _as_wrs2(es.estimate, _tr(case)),
        number(row["estimate"]),
        what="AKP",
    )
    assert reference is not None
    assert reference["requested_conf_level"] == float(case.args.get("conf.level", 0.95))
    assert number(reference["conf_level"]) == EFFECT_CONF_LEVEL
    replicates = reference["replicates"]
    for bound, column in ((es.conf_low, "conf_low"), (es.conf_high, "conf_high")):
        expected = number(reference[column])
        if math.isinf(expected):
            # All of R's 50 runs at 2,000 resamples are infinite there too.
            assert number(replicates[f"{column}_mean"]) == expected
            assert bound == expected, f"{name} {column}"
            continue
        assert_within_monte_carlo(
            bound, expected, number(replicates[f"{column}_sd"]), what=f"{name} {column}"
        )


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_pairwise_rows(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
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
    scale = max(abs(v) for column in result.values for v in column)
    for pair, row in zip(result.pairwise, rows, strict=True):
        what = f"{name} {pair.group1}-{pair.group2}"
        assert pair.test_label == row["test"] == "Yuen's trimmed means"
        assert pair.test == "rmmcp"
        assert pair.adjust_method == method
        assert pair.conf_level == number(row["conf.level"]) == EFFECT_CONF_LEVEL
        assert pair.p_adjusted is not None
        assert pair.significant == (pair.p_adjusted < 0.05)
        assert_close(
            "wrs2_p",
            pair.p_adjusted,
            number(row["p.value"]),
            what=what,
            atol_scale=2.0 * len(rows),
        )
        for got, column in (
            (pair.estimate, "estimate"),
            (pair.conf_low, "conf.low"),
            (pair.conf_high, "conf.high"),
        ):
            assert got is not None
            # closed_form's 1e-10 relative to the data (an estimate can be ~0).
            assert abs(got - number(row[column])) <= 1e-10 * scale, f"{what} {column}"


@pytest.mark.parametrize(("name", "backend"), wc.params(MORE))
def test_pairwise_raw_p_statistic_and_df(name: str, backend: Backend) -> None:
    """The pairwise fields ggstatsplot's rows do not carry, by an independent
    route: the one-sample trimmed t of each pair's differences (group1 minus
    group2) from SciPy's ``trim_mean`` and ``mstats.winsorize``, on h - 1 df."""
    case, result = BY_NAME[name], analyze(name, backend)
    columns = dict(zip(case.analysis_levels, wc.conditions(case), strict=True))
    tr = _tr(case)
    unadjusted = result.settings.p_adjust_method == "none"
    for pair in result.pairwise:
        what = f"{name} {pair.group1}-{pair.group2}"
        d = columns[pair.group1] - columns[pair.group2]
        n = d.size
        df = n - 2 * math.floor(tr * n) - 1
        winsorized = np.asarray(mstats.winsorize(d, limits=(tr, tr)), dtype=float)
        se = math.sqrt(float(np.var(winsorized, ddof=1))) / (
            (1 - 2 * tr) * math.sqrt(n)
        )
        t = float(scipy_stats.trim_mean(d, tr)) / se
        assert pair.df == df
        assert pair.p_value is not None
        assert pair.p_adjusted is not None
        assert pair.p_value <= pair.p_adjusted
        if unadjusted:
            assert pair.p_value == pair.p_adjusted
        assert_close("closed_form", pair.statistic, t, what=f"{what} t")
        p = 2.0 * float(scipy_stats.t.sf(abs(t), df))
        assert_close("closed_form", pair.p_value, p, what=f"{what} p")


@pytest.mark.parametrize(("name", "backend"), wc.params(CASES))
def test_drawn_pairs(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == wc.drawn(case)


def _expected_warnings(case: OracleCase, result: StatsResult) -> list[str]:
    level = float(case.args.get("conf.level", 0.95))
    es = result.effect_size
    assert es is not None
    out: list[str] = []
    if es.infinite_resamples:
        if len(result.levels) == 2:
            out.append(f"{es.infinite_resamples:,} of the ")
        else:
            # The entry totals the pairs' values; the result keeps the most of
            # one pair, in resamples (review of phase 3, finding 6).
            out.append("")
    if level != EFFECT_CONF_LEVEL:
        out.append(f"conf_level={level!r} has no effect: ")
    return out


@pytest.mark.parametrize("name", wc.names(CASES))
def test_warnings(name: str) -> None:
    case, result = BY_NAME[name], analyze(name, "pandas")
    method_warnings = [
        w for w in result.warnings if "left out of the analysis" not in w
    ]
    expected = _expected_warnings(case, result)
    assert len(method_warnings) == len(expected)
    for warning, start in zip(method_warnings, expected, strict=True):
        assert warning.startswith(start)
    es = result.effect_size
    assert es is not None
    if es.infinite_resamples and len(result.levels) > 2:
        (entry,) = [w for w in method_warnings if "give an infinite AKP" in w]
        total = int(entry.split(" of the ", 1)[0].replace(",", ""))
        pairs = len(result.pairwise)
        assert es.infinite_resamples <= total <= pairs * es.infinite_resamples
        assert "bootstrap values" in entry
        assert f"(at most {es.infinite_resamples:,} of one pair's 2,000)" in entry
    if es.infinite_resamples:
        codes = [issue.code for issue in result.check()]
        assert "infinite_effect_size" in codes


@pytest.mark.parametrize("name", wc.names(CASES))
def test_backends_give_identical_results(name: str) -> None:
    first = analyze(name, "pandas").to_json()
    for backend in wc.BACKENDS[1:]:
        assert analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", wc.names(CASES))
def test_json_round_trip(name: str) -> None:
    result = analyze(name, "pandas")
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
        frame, "condition", "score", subject_id="subject", type="robust", **kwargs
    )


_RNG = np.random.Generator(np.random.PCG64(42))
_BASE = np.round(_RNG.normal(10.0, 2.0, 15), 2)
TWO_FRAME = _long(
    {
        "pre": _BASE.tolist(),
        "post": np.round(_BASE + _RNG.normal(0.8, 1.0, 15), 2).tolist(),
    }
)
THREE_FRAME = _long(
    {
        "a": _BASE.tolist(),
        "b": np.round(_BASE + _RNG.normal(0.8, 1.0, 15), 2).tolist(),
        "c": np.round(_BASE + _RNG.normal(0.3, 1.5, 15), 2).tolist(),
    }
)


def test_arguments_without_effect_warn() -> None:
    result = _within(
        TWO_FRAME,
        p_adjust_method="BH",
        pairwise_display="all",
        pairwise_alpha=0.1,
        bf_prior=0.8,
        conf_level=0.9,
    )
    es = result.effect_size
    assert es is not None
    assert es.conf_level == 0.95
    what = "x='condition' has 2 conditions, so there are no pairwise comparisons to"
    assert result.warnings == (
        "conf_level=0.9 has no effect: the AKP effect size's bootstrap interval is "
        "always a 95% interval, as in ggstatsplot, which does not pass conf.level "
        "to WRS2 here; the effect size records conf_level 0.95.",
        f"p_adjust_method='BH' has no effect: {what} adjust.",
        f"pairwise_display='all' has no effect: {what} display.",
        f"pairwise_alpha=0.1 has no effect: {what} call significant.",
        "bf_prior=0.8 has no effect: no Bayes factor is computed (type='robust' has "
        "no Bayes factor).",
    )
    three = _within(THREE_FRAME, conf_level=0.99)
    assert all(p.conf_level == 0.95 for p in three.pairwise)
    assert three.warnings[0].startswith(
        "conf_level=0.99 has no effect: the average AKP effect size's bootstrap "
        "interval and the pairwise intervals are always 95% intervals"
    )
    assert "pairwise_conf_level" in [issue.code for issue in three.check()]
    assert _within(TWO_FRAME).warnings == ()


def test_arguments_that_cannot_apply_raise() -> None:
    with pytest.raises(ValueError, match="alternative='less' applies to"):
        _within(TWO_FRAME, alternative="less")
    with pytest.raises(ValueError, match="bf_message=True asks for a Bayes factor"):
        _within(TWO_FRAME, bf_message=True)


def test_tr_reaches_every_statistic() -> None:
    plain, light = _within(THREE_FRAME), _within(THREE_FRAME, tr=0.1)
    assert plain.test is not None
    assert light.test is not None
    assert plain.test.statistic != light.test.statistic
    assert plain.descriptives != light.descriptives
    assert (
        light.test.label == "Repeated-measures trimmed-means ANOVA (10% trimmed means)"
    )


def test_two_subjects_and_three_conditions_are_refused() -> None:
    # The Huynh-Feldt correction of the trimmed-means ANOVA is 0 / 0 with two
    # subjects for any data; it was silently taken as 1 (Phase 3 review).
    frame = _long({"a": [1.0, 2.5], "b": [2.0, 2.0], "c": [4.0, 1.0]})
    with pytest.raises(ValueError, match="needs at least 3") as error:
        _within(frame)
    assert "Collect more subjects" in str(error.value)
    assert "smaller tr" not in str(error.value)


def test_seed_reaches_the_bootstrap() -> None:
    first, again, other = (_within(TWO_FRAME, seed=s) for s in (3, 3, 4))
    assert first.effect_size == again.effect_size
    assert first.effect_size is not None
    assert other.effect_size is not None
    assert first.effect_size.conf_low != other.effect_size.conf_low
    assert first.effect_size.bootstrap_seed == 3
    assert first.provenance.seed == 3


def test_data_errors_name_the_levels() -> None:
    same = _long({"pre": [1.0, 2.0, 4.0, 3.0, 5.0], "post": [2.0, 3.0, 5.0, 4.0, 6.0]})
    with pytest.raises(ValueError, match="every subject's difference") as error:
        _within(same)
    assert str(error.value) == (
        "After winsorizing 20% at each end, every subject's difference between the "
        "levels 'post' and 'pre' of x='condition' is the same (exactly, or to "
        "within the rounding of y in double precision), so the AKP effect size of "
        "the paired differences (their trimmed mean over their winsorized standard "
        "deviation) is undefined. Check the data for entry errors, or use "
        "type='nonparametric'."
    )  # every difference is 1 before winsorizing too: a smaller tr cannot help
    # Every subject 0.1 lower, in decimals: the stored differences vary in their
    # 16th digit only. Refused like the exactly constant case, where an exact
    # computation on the doubles reported t = 6.4e14 and an AKP of 2.6e14.
    pre = [1.1, 2.3, 3.1, 4.7, 5.2, 6.3]
    decimal = _long({"pre": pre, "post": [v - 0.1 for v in pre]})
    with pytest.raises(ValueError, match="is the same \\(exactly, or to within"):
        _within(decimal)
    one_pair = _long(
        {
            "a": [1.0, 2.0, 4.0, 3.0, 9.0, 4.0],
            "b": [2.0, 3.0, 5.0, 4.0, 10.0, 5.0],
            "c": [0.0, 4.0, 1.0, 2.0, 3.0, 8.0],
        }
    )
    with pytest.raises(ValueError, match="levels 'a' and 'b' of x='condition'"):
        _within(one_pair)
    tiny = _long({"pre": [1.0, 2.0], "post": [3.0, 5.0]})
    assert _within(tiny).test is not None  # g = 0: nothing trimmed
    with pytest.raises(ValueError, match="leaves 1 at the levels of x='condition'"):
        _within(_long({"pre": [1.0, 2.0, 4.0], "post": [3.0, 5.0, 4.0]}), tr=0.4)
