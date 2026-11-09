"""``ip.analyze.betweenstats`` against every nonparametric, robust and bayes
ggbetweenstats fixture.

Each fixture of these types is replayed as the inferplot call
``tests/oracle_args.translate`` builds from its R arguments, on the fixture's CSV read
as pandas, polars and pyarrow, so a new fixture is tested without new code. Numbers
are compared under their ``tests/tolerances.py`` kinds; texts as the plotmath ASTs
ggstatsplot drew (``tests/plotmath.py``), rendered with inferplot's own renderers.

Bootstrap-derived numbers follow SPEC.md section 10's two-part rule: inferplot's
value is within the bootstrap tolerance of the fixture's 20,000-resample reference,
and inferplot's formatter applied to ggstatsplot's own (100-resample) value
reproduces ggstatsplot's string. Every other number in a text is inferplot's own,
compared as the exact string.

Where inferplot refuses a call ggstatsplot answers, by design, the refusal is
checked and the replay SPEC.md prescribes is compared instead
(``tests/betweenstats_cases.py``).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import inferplot as ip
from inferplot import stats
from inferplot import text as tx
from inferplot.result import EffectSize, StatsResult
from inferplot.stats.density import GRID_POINTS
from tests.betweenstats_cases import (
    BY_NAME,
    CASES,
    FAILED,
    REFUSED,
    call_arguments,
    refusal,
    replay_arguments,
)
from tests.oracle_fixtures import (
    Backend,
    OracleCase,
    number,
    r_level_label,
    read_dataset,
)
from tests.plotmath import parse
from tests.tolerances import (
    assert_close,
    assert_within_grid,
    assert_within_monte_carlo,
)

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
_METHODS = {
    "mann_whitney": "Wilcoxon rank sum test",
    "kruskal_wallis": "Kruskal-Wallis rank sum test",
    "yuen_t": "Yuen's test on trimmed means for independent samples",
    "trimmed_means_anova": "A heteroscedastic one-way ANOVA for trimmed means",
}
_EFFECT_SIZES = {
    "rank_biserial": "r (rank biserial)",
    "rank_epsilon_squared": "Epsilon2 (rank)",
    "akp_delta": "Algina-Keselman-Penfield robust standardized difference",
    "explanatory_xi": "Explanatory measure of effect size",
}
_CENTRALITY = {
    "nonparametric": "median",
    "robust": "trimmed_mean",
    "bayes": "map",
}
_PAIR_TESTS = {"nonparametric": "Dunn", "robust": "Yuen's trimmed means"}


# --------------------------------------------------------------------------- #
# the call and the refusals
# --------------------------------------------------------------------------- #


@functools.cache
def analyze(name: str, backend: Backend) -> StatsResult:
    """The fixture's call, or its replay when inferplot refuses the call."""
    case = BY_NAME[name]
    args, kwargs = replay_arguments(case)
    return ip.analyze.betweenstats(read_dataset(case, backend), *args, **kwargs)


def _params(cases: Any) -> list[Any]:
    return [
        pytest.param(case.name, backend, id=f"{case.name}-{backend}")
        for case in cases
        for backend in BACKENDS
    ]


def _names(cases: Any) -> list[Any]:
    return [pytest.param(case.name, id=case.name) for case in cases]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def _conf_level(case: OracleCase) -> float:
    return float(case.args.get("conf.level", 0.95))


def _groups(case: OracleCase) -> list[np.ndarray[Any, np.dtype[np.float64]]]:
    frame = read_dataset(case)
    x, y = str(case.args["x"]), str(case.args["y"])
    used = frame[[x, y]].dropna()
    labels = used[x].astype(object).map(r_level_label)
    return [
        used.loc[labels == level, y].to_numpy(dtype=np.float64)
        for level in case.analysis_levels
    ]


def _label_rows(case: OracleCase, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    x = str(case.args["x"])
    return {r_level_label(row[x]): row for row in rows}


def _centrality_layers(case: OracleCase) -> list[dict[str, Any]]:
    layers = [
        _label_rows(case, rows)
        for geom in ("GeomPoint", "GeomLabelRepel")
        for rows in case.layer_data(geom)
    ]
    assert len(layers) == 2, f"{case.name}: expected two centrality layers"
    return layers


def _drawn(case: OracleCase) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for params in case.layer_params("GeomSignif"):
        out.extend((str(a), str(b)) for a, b in params["stat"]["comparisons"])
    return out


def _drawn_annotations(case: OracleCase) -> list[str]:
    out: list[str] = []
    for params in case.layer_params("GeomSignif"):
        annotations = params["stat"]["annotations"]
        if isinstance(annotations, str):  # R's JSON writer unboxes length one
            annotations = [annotations]
        out.extend(str(a) for a in annotations)
    return out


def _csv_dropped(case: OracleCase) -> int:
    frame = read_dataset(case, "pandas")
    used = frame[[str(case.args["x"]), str(case.args["y"])]]
    return int(used.isna().any(axis=1).sum())


def _random_estimate(case: OracleCase) -> bool:
    """Whether the effect-size estimate itself depends on the resamples (t1way's
    xi with unequal group sizes is the median of the bootstrap values)."""
    reference = case.bootstrap_reference
    return reference is not None and number(reference["replicates"]["estimate_sd"]) > 0


def _with_r_bootstrap(
    case: OracleCase, es: EffectSize, row: Mapping[str, Any]
) -> EffectSize:
    """``es`` with ggstatsplot's own values (``row``, its subtitle table) of the
    bootstrap-derived numbers."""
    if case.bootstrap_reference is None:
        return es
    changes: dict[str, float] = {
        "conf_low": number(row["conf.low"]),
        "conf_high": number(row["conf.high"]),
    }
    if _random_estimate(case):
        changes["estimate"] = number(row["estimate"])
    return dataclasses.replace(es, **changes)


def _pair_rows(case: OracleCase, result: StatsResult) -> list[Mapping[str, Any]]:
    """ggstatsplot's pairwise rows, in inferplot's order (they must agree)."""
    rows = case.pairwise
    assert [(p.group1, p.group2) for p in result.pairwise] == [
        (str(r["group1"]), str(r["group2"])) for r in rows
    ]
    return rows


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #


def test_refused_calls_raise() -> None:
    # One test over all refused fixtures (none yet among the Phase 0 cases), so an
    # empty selection is not an empty parametrization.
    for case in REFUSED:
        found = refusal(case)
        assert found is not None
        args, kwargs = call_arguments(case)
        with pytest.raises(found.error, match=found.match):
            ip.analyze.betweenstats(read_dataset(case), *args, **kwargs)


# --------------------------------------------------------------------------- #
# numbers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_levels_and_counts(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    assert result.analysis == "betweenstats"
    assert result.type == case.type
    assert result.levels == case.analysis_levels
    assert {d.level: d.n for d in result.descriptives} == case.n_per_level
    assert [len(v) for v in result.values] == [d.n for d in result.descriptives]
    assert result.n_obs == sum(case.n_per_level.values())
    assert result.n_dropped == _csv_dropped(case)


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_centrality(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    y = str(case.args["y"])
    groups = dict(zip(case.analysis_levels, _groups(case), strict=True))
    for layer in _centrality_layers(case):
        assert list(layer) == list(result.levels)
        for d in result.descriptives:
            row = layer[d.level]
            what = f"{name} {d.centrality} {d.level}"
            assert d.centrality == _CENTRALITY[case.type]
            assert int(row["n.obs"]) == d.n
            expected = number(row[y])
            if d.centrality == "map":
                values = groups[d.level]
                low, high = float(values.min()), float(values.max())
                step = (high - low) / (GRID_POINTS - 1)
                assert_within_grid(d.value, expected, origin=low, step=step, what=what)
            else:
                assert_close("closed_form", d.value, expected, what=what)


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_main_test(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    test = result.test
    if not case.subtitle_rows:  # results.subtitle = FALSE: R reports nothing
        assert case.args.get("results.subtitle") is False
        assert result.text.subtitle is None
        return
    if case.type == "bayes":
        assert test is None
        return
    assert test is not None
    row = case.subtitle_data
    assert test.method == row["method"] == _METHODS[test.name]
    alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
    assert test.alternative == alternative
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="stat")
    if test.name in ("yuen_t", "trimmed_means_anova"):
        # R's p is 1 - cdf, rounded to 2^-53 per tail; inferplot's is the accurate
        # tail (D43), within one rounding step (twice that for Yuen's two tails).
        scale = 2.0 if test.name == "yuen_t" else 1.0
        assert_close(
            "wrs2_p", test.p_value, number(row["p.value"]), what="p", atol_scale=scale
        )
    else:
        assert_close("closed_form", test.p_value, number(row["p.value"]), what="p")
    for field, column in (("df", "df"), ("df_error", "df.error")):
        value = getattr(test, field)
        if column in row and row[column] is not None:
            assert value is not None
            assert_close("closed_form", value, number(row[column]), what=column)
        else:
            assert value is None, field


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_effect_size(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    es = result.effect_size
    if not case.subtitle_rows or case.type == "bayes":
        return
    assert es is not None
    row = case.subtitle_data
    assert row["effectsize"] == _EFFECT_SIZES[es.name]
    assert es.conf_level == number(row["conf.level"]) == _conf_level(case)
    reference = case.bootstrap_reference
    if reference is None:  # closed form: rank-biserial r
        assert es.bootstrap_resamples is None
        assert es.bootstrap_seed is None
        for field, column in (
            ("estimate", "estimate"),
            ("conf_low", "conf.low"),
            ("conf_high", "conf.high"),
        ):
            expected = number(row[column])
            assert_close("closed_form", getattr(es, field), expected, what=column)
        alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
        assert es.sidedness == alternative
        return
    assert es.ci_method == "percentile bootstrap"
    assert es.bootstrap_resamples == stats.BOOTSTRAP_RESAMPLES
    assert es.bootstrap_seed == result.settings.seed == result.provenance.seed
    assert number(reference["conf_level"]) == es.conf_level
    replicates = reference["replicates"]
    if _random_estimate(case):
        assert_within_monte_carlo(
            es.estimate,
            number(reference["estimate"]),
            number(replicates["estimate_sd"]),
            what=f"{name} estimate",
        )
    else:
        kind = {
            "akp_delta": "wrs2_akp_constant",
            "explanatory_xi": "wrs2_xi_constant",
        }.get(es.name, "closed_form")
        assert_close(kind, es.estimate, number(row["estimate"]), what="estimate")
    for bound in ("conf_low", "conf_high"):
        assert_within_monte_carlo(
            getattr(es, bound),
            number(reference[bound]),
            number(replicates[f"{bound}_sd"]),
            what=f"{name} {bound}",
        )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_bayes_factor(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    if case.type != "bayes" or len(result.levels) >= 3 or not case.subtitle_rows:
        assert result.bayes is None
        return
    row = case.subtitle_data
    assert result.bayes is not None
    assert result.bayes.prior == row["prior.distribution"] == "cauchy"
    assert result.bayes.prior_scale == number(row["prior.scale"])
    assert_close(
        "ttest_bayes_factor", result.bayes.bf10, number(row["bf10"]), what="bf10"
    )
    assert_close(
        "ttest_bayes_factor",
        result.bayes.log_bf01,
        -number(row["log_e_bf10"]),
        what="log_bf01",
    )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
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
    rows = _pair_rows(case, result)
    m = len(rows)
    order = {level: i for i, level in enumerate(result.levels)}
    for pair, row in zip(result.pairwise, rows, strict=True):
        what = f"{name} {pair.group1}-{pair.group2}"
        assert order[pair.group1] < order[pair.group2]
        assert pair.test_label == row["test"]
        if case.type == "bayes":
            assert row["test"] == "Student's t"
            assert pair.bayes is not None
            assert pair.p_value is None
            assert pair.p_adjusted is None
            assert pair.significant is None
            assert pair.bayes.prior_scale == number(row["prior.scale"]) == 0.707
            assert_close(
                "ttest_bayes_factor", pair.bayes.bf10, number(row["bf10"]), what=what
            )
            continue
        method = str(case.args.get("p.adjust.method", "holm"))
        assert pair.adjust_method == method
        assert row["p.adjust.method"] == tx.p_adjust_label(method)
        assert pair.p_adjusted is not None
        assert pair.significant == (pair.p_adjusted < 0.05)
        if case.type == "nonparametric":
            assert row["distribution"] == "z"
            assert pair.statistic_name == "z"
            assert_close(
                "closed_form", pair.statistic, number(row["statistic"]), what=what
            )
            assert_close(
                "closed_form", pair.p_adjusted, number(row["p.value"]), what=what
            )
        else:
            assert number(row["conf.level"]) == pair.conf_level == 0.95
            for field, column in (
                ("estimate", "estimate"),
                ("conf_low", "conf.low"),
                ("conf_high", "conf.high"),
            ):
                value = getattr(pair, field)
                assert value is not None
                expected = number(row[column])
                assert_close("closed_form", value, expected, what=f"{what} {column}")
            # R's raw p is 2 (1 - pt), rounded to 2^-52 (D43); the adjustment
            # multiplies that step by at most m.
            assert_close(
                "wrs2_p",
                pair.p_adjusted,
                number(row["p.value"]),
                what=what,
                atol_scale=2.0 * m,
            )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_drawn_pairs(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == _drawn(case)


# --------------------------------------------------------------------------- #
# texts
# --------------------------------------------------------------------------- #


def _expected_subtitle(case: OracleCase, result: StatsResult) -> tx.Node | None:
    """inferplot's subtitle AST, which must equal ggstatsplot's once ggstatsplot's
    own bootstrap values are put in (checked separately in ``test_subtitle``)."""
    digits = _digits(case)
    if result.type == "bayes":
        if result.bayes is None:
            return None
        return tx.bayes_factor_expression(
            log_bf01=result.bayes.log_bf01,
            prior=result.bayes.prior,
            prior_scale=result.bayes.prior_scale,
            digits=digits,
        )
    if result.test is None or result.effect_size is None:
        return None
    return tx.subtitle_expression(
        result.test, result.effect_size, n_obs=result.n_obs, digits=digits
    )


def _v01(node: tx.Node) -> tx.Node:
    """The first and last items of a Bayes expression (D32, D41)."""
    assert isinstance(node, tx.Call)
    assert node.fn == tx.Sym("list")
    return tx.Call(tx.Sym("list"), (node.args[0], node.args[-1]))


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_subtitle(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    figure = tx.figure_texts(result)
    ours = _expected_subtitle(case, result)
    if ours is None:
        assert result.text.subtitle is None
        assert figure.subtitle is None
        assert case.subtitle is None or (
            case.type == "bayes" and len(result.levels) >= 3
        )
        return
    assert result.text.subtitle == tx.to_plain(ours)
    assert figure.subtitle == tx.to_mathtext(ours)
    assert case.subtitle is not None
    theirs = parse(case.subtitle)
    if case.type == "bayes":
        assert ours == _v01(theirs)
        return
    # Every string from inferplot's own numbers, except the bootstrap-derived ones,
    # which are inferplot's formatter applied to ggstatsplot's values.
    assert _mixed_subtitle(case, result, case.subtitle_data) == theirs


def _mixed_subtitle(
    case: OracleCase, result: StatsResult, row: Mapping[str, Any]
) -> tx.Node:
    assert result.test is not None
    assert result.effect_size is not None
    return tx.subtitle_expression(
        result.test,
        _with_r_bootstrap(case, result.effect_size, row),
        n_obs=result.n_obs,
        digits=_digits(case),
    )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_caption(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    figure = tx.figure_texts(result)
    assert case.caption is None  # ggstatsplot draws no Bayes caption for these types
    line = tx.pairwise_caption_line(
        result.pairwise,
        display=result.settings.pairwise_display,
        adjust_method=result.settings.p_adjust_method,
    )
    expected = () if line is None else (line,)
    assert figure.caption_lines == expected
    assert result.text.caption == (line if line is not None else None)
    if line is None:
        return
    if case.type == "bayes":
        assert line == (
            "Pairwise: Bayes factors (Student's t, JZS prior 0.707), unadjusted; "
            f"showing all {len(result.pairwise)} pairs."
        )
    else:
        assert line.startswith(f"Pairwise: {_PAIR_TESTS[case.type]}, ")


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_centrality_and_tick_labels(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    figure = tx.figure_texts(result)
    for layer in _centrality_layers(case):
        expected = [parse(str(layer[level]["expression"])) for level in result.levels]
        assert list(result.text.centrality_labels) == [tx.to_plain(n) for n in expected]
        assert list(figure.centrality_labels) == [tx.to_mathtext(n) for n in expected]
        ticks = [str(layer[level]["n.expression"]) for level in result.levels]
        assert list(result.text.tick_labels) == ticks
        assert list(figure.tick_labels) == ticks


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_bracket_labels(name: str, backend: Backend) -> None:
    case, result = BY_NAME[name], analyze(name, backend)
    if not case.pairwise:
        return
    figure = tx.figure_texts(result)
    rows = _pair_rows(case, result)
    for i, row in enumerate(rows):
        theirs = parse(str(row["expression"]))
        assert result.text.pairwise_labels[i] == tx.to_plain(theirs)
        assert figure.pairwise_labels[i] == tx.to_mathtext(theirs)
    drawn = [i for i, pair in enumerate(result.pairwise) if pair.displayed]
    annotations = _drawn_annotations(case)
    assert len(annotations) == len(drawn)
    for i, source in zip(drawn, annotations, strict=True):
        assert parse(source) == parse(str(rows[i]["expression"]))


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_plain_texts(name: str, backend: Backend) -> None:
    result = analyze(name, backend)
    text = result.text
    assert text.apa.isascii()
    assert text.apa.endswith(".")
    assert len(text.alt_text) <= tx.ALT_TEXT_MAX_CHARS
    word = {"median": "median", "trimmed_mean": "trimmed mean", "map": "MAP"}
    assert word[result.descriptives[0].centrality] in text.alt_text


# --------------------------------------------------------------------------- #
# backends, JSON, row order
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", _names(CASES))
def test_backends_give_identical_results(name: str) -> None:
    first = analyze(name, "pandas").to_json()
    for backend in BACKENDS[1:]:
        assert analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", _names(CASES))
def test_json_round_trip(name: str) -> None:
    result = analyze(name, "pandas")
    assert StatsResult.from_json(result.to_json()) == result
    frame = result.to_frame()
    assert len(frame) == len(result.frame_rows())


# --------------------------------------------------------------------------- #
# cases ggstatsplot fails on
# --------------------------------------------------------------------------- #


def test_failed_cases() -> None:
    """ggstatsplot fails on these (none yet among the Phase 0 cases); inferplot
    either refuses with a plain error at the same component (pairwise comparisons)
    or computes, and then its subtitle and centrality labels equal the partial
    outputs ggstatsplot made first."""
    for case in FAILED:
        _check_failed_case(case)


def _check_failed_case(case: OracleCase) -> None:
    assert case.expected_error is not None
    args, kwargs = call_arguments(case)
    result: StatsResult | None = None
    refused: str | None = None
    try:
        result = ip.analyze.betweenstats(read_dataset(case), *args, **kwargs)
    except (ValueError, NotImplementedError) as error:
        refused = str(error)
    if refused is not None:
        # Refused where ggstatsplot fails too, with a message naming what to change.
        assert case.expected_error["component"] == "pairwise_comparisons"
        assert len(case.analysis_levels) >= 3
        assert any(word in refused for word in ("x=", "conf_level", "type="))
        return
    assert result is not None
    rows = case.partial("subtitle")
    if rows and result.type != "bayes":
        (row,) = rows
        assert _mixed_subtitle(case, result, row) == parse(str(row["expression"]))
    labels = _label_rows(case, case.partial("centrality_description"))
    for i, d in enumerate(result.descriptives):
        label = parse(str(labels[d.level]["expression"]))
        assert result.text.centrality_labels[i] == tx.to_plain(label)
