"""``ip.analyze.betweenstats`` against every parametric ggbetweenstats fixture.

Each completed parametric fixture is replayed as an inferplot call built by
``tests/oracle_args.translate`` (it raises on an R argument without a mapping) on the
fixture's CSV read as pandas, polars and pyarrow. Every number is compared under its
``tests/tolerances.py`` kind, and every text as the plotmath AST ggstatsplot drew,
parsed by ``tests/plotmath.py`` and rendered with inferplot's own renderers: plain
text against ``result.text`` and mathtext against ``inferplot.text.figure_texts``.

Games-Howell bracket labels follow SPEC.md section 10: inferplot computes the raw p
from an accurate studentized range tail (DECISIONS.md D31), so its label may differ
from ggstatsplot's only where the two adjusted p-values format differently and are
within ``games_howell_p``; inferplot's formatter applied to ggstatsplot's own value
must reproduce ggstatsplot's label. The pairs whose label differs are pinned below.

ggstatsplot fails on ``edge_n2`` (every Welch df below 2, where R's ptukey returns
NaN); inferplot computes it, and its texts must equal the outputs ggstatsplot made
before failing.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from typing import Any

import pytest

import inferplot as ip
from inferplot import stats
from inferplot import text as tx
from inferplot.result import StatsResult
from tests.oracle_args import translate
from tests.oracle_fixtures import (
    Backend,
    OracleCase,
    cases_for,
    number,
    probe,
    r_level_label,
    read_dataset,
    require_cases,
)
from tests.plotmath import parse
from tests.tolerances import assert_close, is_close

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")

CASES = require_cases(
    [c for c in cases_for("ggbetweenstats") if c.type == "parametric"],
    "parametric ggbetweenstats",
)
FAILED = require_cases(
    [c for c in cases_for("ggbetweenstats", failed=True) if c.type == "parametric"],
    "parametric ggbetweenstats that ggstatsplot fails on",
)
_BY_NAME = {case.name: case for case in (*CASES, *FAILED)}

# Pairs whose bracket label differs from ggstatsplot's, because R's ptukey is off in
# the far upper tail (SPEC.md section 5 item 6, DECISIONS.md D31):
# (case, group1, group2, ggstatsplot's label number, inferplot's label number).
# The raw p-values behind them: R's ptukey 4.24921864361494e-09, 4.71159555814893e-10
# and 1.41319738133916e-08 (probe-ptukey) against inferplot's 4.2831e-09, 7.4517e-15
# and 1.3686e-08; each difference is within R's measured far-tail error.
DIFFERING_LABELS: tuple[tuple[str, str, str, str, str], ...] = (
    ("ggbetweenstats-edge_negative-parametric", "b", "c", "1.27e-08", "1.28e-08"),
    ("ggbetweenstats-iris-parametric", "setosa", "versicolor", "1.41e-09", "2.24e-14"),
    ("ggbetweenstats-iris-parametric", "setosa", "virginica", "2.83e-08", "2.74e-08"),
)


def _params(cases: Any) -> list[Any]:
    return [
        pytest.param(case.name, backend, id=f"{case.name}-{backend}")
        for case in cases
        for backend in BACKENDS
    ]


@functools.cache
def analyze(name: str, backend: Backend) -> StatsResult:
    """The fixture's call, run by inferplot on the fixture's CSV."""
    case = _BY_NAME[name]
    call = translate(case.function, case.args)
    assert call.function == "betweenstats"
    function = getattr(ip.analyze, call.function)
    return function(read_dataset(case, backend), *call.args, **call.kwargs)


def _digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def _label_rows(case: OracleCase, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    x = str(case.args["x"])
    return {r_level_label(row[x]): row for row in rows}


def _centrality_layers(case: OracleCase) -> list[dict[str, Any]]:
    """The two centrality layers (point and label), each by level."""
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
        # R's JSON writer unboxes a vector of length one into a scalar
        if isinstance(annotations, str):
            annotations = [annotations]
        out.extend(str(a) for a in annotations)
    return out


def _probe_pairs(name: str) -> dict[frozenset[str], Mapping[str, Any]]:
    """probe-ptukey's unadjusted gamesHowellTest rows of one case, by pair."""
    for entry in probe("probe-ptukey").rows("games_howell"):
        if entry["case"] == name:
            return {frozenset((p["group1"], p["group2"])): p for p in entry["pairs"]}
    raise KeyError(f"probe-ptukey has no Games-Howell rows for {name}")


def _v01_caption(source: str) -> tx.Node:
    """inferplot's v0.1 Bayes caption: the first and last items of ggstatsplot's."""
    node = parse(source)
    assert isinstance(node, tx.Call)
    assert node.fn == tx.Sym("list")
    return tx.Call(tx.Sym("list"), (node.args[0], node.args[-1]))


def _csv_dropped(case: OracleCase) -> int:
    frame = read_dataset(case, "pandas")
    used = frame[[str(case.args["x"]), str(case.args["y"])]]
    return int(used.isna().any(axis=1).sum())


# --------------------------------------------------------------------------- #
# completed cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_levels_and_counts(name: str, backend: Backend) -> None:
    case, result = _BY_NAME[name], analyze(name, backend)
    assert result.analysis == "betweenstats"
    assert result.type == "parametric"
    assert result.levels == case.analysis_levels
    counts = {d.level: d.n for d in result.descriptives}
    assert counts == case.n_per_level
    assert [len(v) for v in result.values] == [d.n for d in result.descriptives]
    assert result.n_obs == sum(case.n_per_level.values())
    assert result.n_dropped == _csv_dropped(case)
    if case.subtitle_rows:
        assert result.n_obs == int(case.subtitle_data["n.obs"])


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_means(name: str, backend: Backend) -> None:
    case, result = _BY_NAME[name], analyze(name, backend)
    y = str(case.args["y"])
    for layer in _centrality_layers(case):
        assert list(layer) == list(result.levels)
        for d in result.descriptives:
            row = layer[d.level]
            assert d.centrality == "mean"
            assert int(row["n.obs"]) == d.n
            assert_close(
                "closed_form", d.value, number(row[y]), what=f"{name} mean {d.level}"
            )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_main_test_and_effect_size(name: str, backend: Backend) -> None:
    case, result = _BY_NAME[name], analyze(name, backend)
    test, es = result.test, result.effect_size
    assert test is not None
    assert es is not None
    if not case.subtitle_rows:  # results.subtitle = FALSE: R reports nothing
        assert case.args.get("results.subtitle") is False
        return
    row = case.subtitle_data
    two = len(result.levels) == 2
    assert test.method == row["method"]
    assert test.name == ("welch_t" if two else "welch_anova")
    assert test.statistic_name == ("t" if two else "F")
    alternative = str(row.get("alternative", "two.sided")).replace(".", "-")
    assert test.alternative == alternative
    assert_close("closed_form", test.statistic, number(row["statistic"]), what="stat")
    assert_close("closed_form", test.p_value, number(row["p.value"]), what="p")
    assert test.df_error is not None
    assert_close("closed_form", test.df_error, number(row["df.error"]), what="df")
    if two:
        assert test.df is None
        assert es.name == "hedges_g"
        assert row["effectsize"] == "Hedges' g"
        assert es.ci_method == "noncentral t"
        assert es.sidedness == alternative
    else:
        assert test.df is not None
        assert_close("closed_form", test.df, number(row["df"]), what="df1")
        assert es.name == "partial_omega_squared"
        assert row["effectsize"] == "Omega2"
        assert es.ci_method == "noncentral F"
        assert es.sidedness == "greater"
    assert row["conf.method"] == "ncp"
    assert es.conf_level == number(row["conf.level"])
    assert_close("closed_form", es.estimate, number(row["estimate"]), what="estimate")
    for bound in ("conf_low", "conf_high"):
        expected = number(row[bound.replace("_", ".")])
        assert_close("noncentral_ci", getattr(es, bound), expected, what=bound)


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_bayes_factor(name: str, backend: Backend) -> None:
    case, result = _BY_NAME[name], analyze(name, backend)
    two = len(result.levels) == 2
    if not two or case.caption is None:
        assert result.bayes is None
        if not two and case.caption is not None:
            # ggstatsplot drew a Bayes ANOVA caption; inferplot says it is v0.2.
            assert any("inferplot 0.2" in w for w in result.warnings)
        return
    row = case.caption_data
    assert row is not None
    assert result.bayes is not None
    assert result.bayes.prior == row["prior.distribution"] == "cauchy"
    assert result.bayes.prior_scale == number(row["prior.scale"])
    assert result.bayes.test == "jzs_two_sample"
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
    case, result = _BY_NAME[name], analyze(name, backend)
    if len(result.levels) == 2:
        assert result.pairwise == ()
        return
    k = len(result.levels)
    assert len(result.pairwise) == k * (k - 1) // 2
    rows = case.pairwise
    if case.args.get("pairwise.display") == "none":
        # ggstatsplot skips the comparisons; inferplot computes them, draws none.
        assert rows == []
        assert not any(pair.displayed for pair in result.pairwise)
    else:
        assert [(p.group1, p.group2) for p in result.pairwise] == [
            (str(r["group1"]), str(r["group2"])) for r in rows
        ]
    order = {level: i for i, level in enumerate(result.levels)}
    probe_rows = _probe_pairs(name)
    method = str(case.args.get("p.adjust.method", "holm"))
    for i, pair in enumerate(result.pairwise):
        what = f"{name} {pair.group1}-{pair.group2}"
        assert order[pair.group1] < order[pair.group2]
        assert pair.adjust_method == method
        assert pair.test_label == "Games-Howell"
        assert pair.p_value is not None
        assert pair.p_adjusted is not None
        assert pair.significant == (pair.p_adjusted < 0.05)
        reference = probe_rows[frozenset((pair.group1, pair.group2))]
        sign = 1.0 if reference["group1"] == pair.group1 else -1.0
        assert_close(
            "closed_form",
            pair.statistic,
            sign * number(reference["statistic"]),
            what=f"{what} q (probe)",
        )
        assert pair.df is not None
        assert_close(
            "closed_form", pair.df, number(reference["welch_df"]), what=f"{what} df"
        )
        assert_close(
            "games_howell_p",
            pair.p_value,
            number(reference["p_value"]),
            what=f"{what} raw p",
        )
        if not rows:
            continue
        row = rows[i]
        assert row["test"] == "Games-Howell"
        assert row["distribution"] == "q"
        assert row["p.adjust.method"] == tx.p_adjust_label(method)
        assert_close(
            "closed_form", pair.statistic, number(row["statistic"]), what=f"{what} q"
        )
        assert_close(
            "games_howell_p",
            pair.p_adjusted,
            number(row["p.value"]),
            what=f"{what} adjusted p",
        )


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_drawn_pairs(name: str, backend: Backend) -> None:
    case, result = _BY_NAME[name], analyze(name, backend)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == _drawn(case)


def _label_check(
    case: OracleCase, result: StatsResult
) -> list[tuple[str, str, str, str, str]]:
    """Check every bracket label under SPEC section 10's rule; return the pairs
    whose label differs from ggstatsplot's."""
    digits = _digits(case)
    figure = tx.figure_texts(result)
    differing: list[tuple[str, str, str, str, str]] = []
    rows = case.pairwise
    labels = result.text.pairwise_labels
    assert len(labels) == len(figure.pairwise_labels) == len(result.pairwise)
    for i, (pair, row) in enumerate(zip(result.pairwise, rows, strict=True)):
        assert pair.p_adjusted is not None
        ours = tx.pairwise_p_expression(
            p_adjusted=pair.p_adjusted, adjust_method=pair.adjust_method, digits=digits
        )
        assert labels[i] == tx.to_plain(ours)
        assert figure.pairwise_labels[i] == tx.to_mathtext(ours)
        theirs = parse(str(row["expression"]))
        r_p = number(row["p.value"])
        # inferplot's formatter on ggstatsplot's own value reproduces its label
        assert theirs == tx.pairwise_p_expression(
            p_adjusted=r_p, adjust_method=pair.adjust_method, digits=digits
        )
        r_text = tx.format_value(r_p, digits)
        our_text = tx.format_value(pair.p_adjusted, digits)
        if r_text == our_text:
            assert labels[i] == tx.to_plain(theirs)
            assert figure.pairwise_labels[i] == tx.to_mathtext(theirs)
        else:
            assert is_close("games_howell_p", pair.p_adjusted, r_p)
            differing.append((case.name, pair.group1, pair.group2, r_text, our_text))
    # the drawn annotations are the drawn pairs' labels
    drawn = [i for i, pair in enumerate(result.pairwise) if pair.displayed]
    annotations = _drawn_annotations(case)
    assert len(annotations) == len(drawn)
    for i, source in zip(drawn, annotations, strict=True):
        assert parse(source) == parse(str(rows[i]["expression"]))
    return differing


@pytest.mark.parametrize(("name", "backend"), _params(CASES))
def test_texts(name: str, backend: Backend) -> None:
    case, result = _BY_NAME[name], analyze(name, backend)
    figure = tx.figure_texts(result)
    # subtitle
    if case.subtitle is None:
        assert result.text.subtitle is None
        assert figure.subtitle is None
    else:
        node = parse(case.subtitle)
        assert result.text.subtitle == tx.to_plain(node)
        assert figure.subtitle == tx.to_mathtext(node)
    # the v0.1 Bayes caption line, then the pairwise line
    bayes_lines: tuple[str, ...] = ()
    bayes_plain: tuple[str, ...] = ()
    if len(result.levels) == 2 and case.caption is not None:
        caption = _v01_caption(case.caption)
        bayes_lines = (tx.to_mathtext(caption),)
        bayes_plain = (tx.to_plain(caption),)
    pairwise_line = tx.pairwise_caption_line(
        result.pairwise,
        display=result.settings.pairwise_display,
        adjust_method=result.settings.p_adjust_method,
    )
    extra = () if pairwise_line is None else (pairwise_line,)
    assert figure.caption_lines == bayes_lines + extra
    plain = bayes_plain + extra
    assert result.text.caption == ("\n".join(plain) if plain else None)
    # centrality labels (both layers) and tick labels
    for layer in _centrality_layers(case):
        expected = [parse(str(layer[level]["expression"])) for level in result.levels]
        assert list(result.text.centrality_labels) == [tx.to_plain(n) for n in expected]
        assert list(figure.centrality_labels) == [tx.to_mathtext(n) for n in expected]
        ticks = [str(layer[level]["n.expression"]) for level in result.levels]
        assert list(result.text.tick_labels) == ticks
        assert list(figure.tick_labels) == ticks
    # bracket labels
    if case.pairwise:
        _label_check(case, result)


def test_differing_labels_are_pinned() -> None:
    differing = [
        item
        for case in CASES
        if case.pairwise
        for item in _label_check(case, analyze(case.name, "pandas"))
    ]
    assert tuple(differing) == DIFFERING_LABELS


@pytest.mark.parametrize("name", [case.name for case in CASES])
def test_backends_give_identical_results(name: str) -> None:
    first = analyze(name, "pandas").to_json()
    for backend in BACKENDS[1:]:
        assert analyze(name, backend).to_json() == first


@pytest.mark.parametrize("name", [case.name for case in CASES])
def test_json_round_trip(name: str) -> None:
    result = analyze(name, "pandas")
    assert StatsResult.from_json(result.to_json()) == result


# --------------------------------------------------------------------------- #
# the case ggstatsplot fails on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "backend"), _params(FAILED))
def test_failed_case_computes(name: str, backend: Backend) -> None:
    case, result = _BY_NAME[name], analyze(name, backend)
    assert case.expected_error is not None
    assert case.expected_error["component"] == "pairwise_comparisons"
    assert result.levels == case.analysis_levels
    assert {d.level: d.n for d in result.descriptives} == case.n_per_level
    figure = tx.figure_texts(result)
    # subtitle: the partial output ggstatsplot computed before failing
    (row,) = case.partial("subtitle")
    node = parse(str(row["expression"]))
    assert result.text.subtitle == tx.to_plain(node)
    assert figure.subtitle == tx.to_mathtext(node)
    assert result.test is not None
    assert result.effect_size is not None
    assert_close(
        "closed_form", result.test.statistic, number(row["statistic"]), what="F"
    )
    assert result.test.df_error is not None
    assert_close(
        "closed_form", result.test.df_error, number(row["df.error"]), what="df"
    )
    assert_close("closed_form", result.test.p_value, number(row["p.value"]), what="p")
    assert_close(
        "closed_form", result.effect_size.estimate, number(row["estimate"]), what="es"
    )
    for bound in ("conf_low", "conf_high"):
        expected = number(row[bound.replace("_", ".")])
        assert_close(
            "noncentral_ci", getattr(result.effect_size, bound), expected, what=bound
        )
    # Bayes caption: ggstatsplot's is a Bayes ANOVA (three groups), a v0.2 design
    assert case.partial("caption")
    assert len(result.levels) >= 3
    assert result.bayes is None
    assert all("BF" not in line for line in figure.caption_lines)
    assert any("inferplot 0.2" in w for w in result.warnings)
    # centrality and tick labels
    rows = _label_rows(case, case.partial("centrality_description"))
    y = str(case.args["y"])
    for i, d in enumerate(result.descriptives):
        row = rows[d.level]
        label = parse(str(row["expression"]))
        assert result.text.centrality_labels[i] == tx.to_plain(label)
        assert figure.centrality_labels[i] == tx.to_mathtext(label)
        assert (
            result.text.tick_labels[i] == figure.tick_labels[i] == row["n.expression"]
        )
        key = y if y in row else "value"
        assert_close("closed_form", d.value, number(row[key]), what=f"mean {d.level}")
    # pairwise: finite p from the accurate tail, where R's ptukey gives NaN
    k = len(result.levels)
    probe_rows = _probe_pairs(name)
    for pair in result.pairwise:
        assert pair.df is not None
        assert pair.df < 2.0
        assert pair.p_value is not None
        assert pair.p_adjusted is not None
        assert math.isfinite(pair.p_value)
        assert math.isfinite(pair.p_adjusted)
        assert pair.p_value == stats.srange_sf(abs(pair.statistic), k, pair.df)
        reference = probe_rows[frozenset((pair.group1, pair.group2))]
        assert reference["p_value"] is None  # R: NaN
        sign = 1.0 if reference["group1"] == pair.group1 else -1.0
        assert_close(
            "closed_form",
            pair.statistic,
            sign * number(reference["statistic"]),
            what="q",
        )
        assert_close("closed_form", pair.df, number(reference["welch_df"]), what="df")
    assert "small_n" in {issue.code for issue in result.check()}
    assert StatsResult.from_json(result.to_json()) == result
