"""The texts of ``ip.analyze.withinstats`` against every ggwithinstats fixture.

Each of the 159 fixtures, of every type, is replayed as the inferplot call
``tests/oracle_args.translate`` builds from its R arguments (or the replay SPEC.md
prescribes where inferplot refuses the call: ``tests/within_pb_cases.refusal``), and
every text ggstatsplot drew is compared with inferplot's as a parsed plotmath
expression (``tests/plotmath.py``), rendered with inferplot's own renderers: the
subtitle (``n["pairs"]``, the corrected df with ``digits`` decimals, each within
test's and effect size's symbols, the robust intervals' fixed 95%), the caption,
the centrality and tick labels (complete subjects only) and every bracket label,
and which pairs get a bracket (the plot's ``GeomSignif`` layers).

Numbers in a text are inferplot's own, compared as the exact string, except the
bootstrap-derived ones (Kendall's W's bound, the AKP bounds), which follow SPEC.md
section 10's rule: inferplot's formatter applied to ggstatsplot's own 100-resample
values must reproduce ggstatsplot's string (the values themselves are compared with
the 20,000-resample references in the parity tests). A string that differs is
allowed only as a pinned, explained entry of :data:`DIFFERING_SUBTITLES`,
:data:`DIFFERING_CAPTIONS` or :data:`DIFFERING_LABELS`, each with both numbers.

The texts are compared on the pandas reading; the parity tests show that the three
backends give byte-identical results (``to_json()``), texts included.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import pytest

from inferplot import text as tx
from inferplot.result import EffectSize, StatsResult
from tests import within_pb_cases as wc
from tests.oracle_fixtures import OracleCase, number, r_level_label
from tests.plotmath import parse
from tests.tolerances import is_close

CASES = wc.ALL_CASES

# R's `test` column of the pairwise rows, by type: the caption names it.
_PAIR_TESTS = {
    "parametric": "Student's t",
    "nonparametric": "Durbin-Conover",
    "robust": "Yuen's trimmed means",
    "bayes": "Student's t",
}
_ADJUST_WORDS = {
    "holm": "Holm adjusted",
    "hochberg": "Hochberg adjusted",
    "hommel": "Hommel adjusted",
    "bonferroni": "Bonferroni adjusted",
    "BH": "Benjamini-Hochberg adjusted",
    "fdr": "Benjamini-Hochberg adjusted",
    "BY": "Benjamini-Yekutieli adjusted",
    "none": "unadjusted",
}
_DISPLAY_NOTES = {
    "significant": "significant only",
    "non-significant": "non-significant only",
}

# Subtitles that differ from ggstatsplot's: (case, ggstatsplot's text, inferplot's
# text, why). edge_within_small nonparametric: every subject ranks the three
# conditions alike, Kendall's W is 1 in every resample, effectsize's bootstrap fails
# and ggstatsplot drops the subtitle without a word; inferplot reports Friedman's
# test with W = 1 and [1, 1] and a warnings entry (D64).
DIFFERING_SUBTITLES: tuple[tuple[str, str, str], ...] = (
    (
        "ggwithinstats-edge_within_small-nonparametric",
        "(none)",
        "\u03c7_Friedman^2(2) = 8.00, p = 0.02, W_Kendall = 1.00, "
        "CI_95% [1.00, 1.00], n_pairs = 4",
    ),
)

# Captions that differ from ggstatsplot's beyond SPEC.md section 5 item 5 (the
# Bayes ANOVA caption of three or more conditions, v0.2, replaced by a warnings
# entry, which test_caption checks on every such fixture): none.
DIFFERING_CAPTIONS: tuple[tuple[str, str, str], ...] = ()

# Bracket labels that differ from ggstatsplot's: (case, group1, group2,
# ggstatsplot's number, inferplot's number): none. rmmcp's p is WRS2's 2 (1 - pt)
# and inferplot's the accurate tail (D43), and BayesFactor's paired Bayes factors
# miss the accurate ones at large |t| (D61), but on these fixtures neither moves a
# printed digit.
DIFFERING_LABELS: tuple[tuple[str, str, str, str, str], ...] = ()


def analyze(name: str) -> StatsResult:
    """The fixture's call, or its replay when inferplot refuses the call."""
    return wc.analyze(name, "pandas")


def _params() -> list[Any]:
    return [pytest.param(case.name, id=case.name) for case in CASES]


def _digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def _random_estimate(case: OracleCase) -> bool:
    reference = case.bootstrap_reference
    return reference is not None and number(reference["replicates"]["estimate_sd"]) > 0


def _with_r_bootstrap(
    case: OracleCase, es: EffectSize, row: Mapping[str, Any]
) -> EffectSize:
    """``es`` with ggstatsplot's own values of the bootstrap-derived numbers."""
    if case.bootstrap_reference is None:
        return es
    changes: dict[str, float] = {
        "conf_low": number(row["conf.low"]),
        "conf_high": number(row["conf.high"]),
    }
    if _random_estimate(case):
        changes["estimate"] = number(row["estimate"])
    return dataclasses.replace(es, **changes)


def _v01(node: tx.Node) -> tx.Node:
    """The first and last items of a Bayes expression (D32, D41)."""
    assert isinstance(node, tx.Call)
    assert node.fn == tx.Sym("list")
    return tx.Call(tx.Sym("list"), (node.args[0], node.args[-1]))


def _drawn(case: OracleCase) -> list[tuple[str, str]]:
    return wc.drawn(case)


def _drawn_annotations(case: OracleCase) -> list[str]:
    out: list[str] = []
    for params in case.layer_params("GeomSignif"):
        annotations = params["stat"]["annotations"]
        if isinstance(annotations, str):  # R's JSON writer unboxes length one
            annotations = [annotations]
        out.extend(str(a) for a in annotations)
    return out


def test_every_within_fixture_is_replayed() -> None:
    # 159 Phase 3 cases and edge_rm_some_tied (review T7).
    assert len(CASES) == 160
    assert {c.type for c in CASES} == set(wc.TYPES)


# --------------------------------------------------------------------------- #
# subtitle
# --------------------------------------------------------------------------- #


def _mixed_subtitle(case: OracleCase, result: StatsResult) -> tx.Node:
    """inferplot's subtitle with ggstatsplot's bootstrap values put in."""
    assert result.test is not None
    assert result.effect_size is not None
    return tx.subtitle_expression(
        result.test,
        _with_r_bootstrap(case, result.effect_size, case.subtitle_data),
        n_obs=result.n_obs,
        digits=_digits(case),
        paired=True,
    )


def _subtitle_difference(
    case: OracleCase, result: StatsResult
) -> tuple[str, str, str] | None:
    """(case, ggstatsplot's subtitle, inferplot's) where they differ; None where
    inferplot's is ggstatsplot's (or neither draws one)."""
    ours = result.text.subtitle
    if case.subtitle is None:
        if ours is None:
            return None
        return (case.name, "(none)", ours)
    if ours is None:
        # The replay of a refused call (the Bayesian repeated-measures ANOVA,
        # v0.2) draws no subtitle; test_subtitle checks that.
        refused = wc.refusal(case)
        assert refused is not None, case.name
        assert refused.replay.get("results_subtitle") is False, case.name
        return None
    theirs = parse(case.subtitle)
    if case.type == "bayes":
        mine = tx.bayes_factor_expression(
            log_bf01=_bayes_log_bf01(result),
            prior="cauchy",
            prior_scale=result.settings.bf_prior,
            digits=_digits(case),
        )
        return None if mine == _v01(theirs) else (case.name, case.subtitle, ours)
    if _mixed_subtitle(case, result) == theirs:
        return None
    return (case.name, tx.to_plain(theirs), ours)


def _bayes_log_bf01(result: StatsResult) -> float:
    assert result.bayes is not None
    return result.bayes.log_bf01


@pytest.mark.parametrize("name", _params())
def test_subtitle(name: str) -> None:
    case, result = wc.BY_NAME[name], analyze(name)
    figure = tx.figure_texts(result)
    refused = wc.refusal(case)
    if result.text.subtitle is None:
        assert figure.subtitle is None
        # No subtitle where the call asks for none, or where the replay turns off
        # the Bayesian repeated-measures ANOVA (v0.2).
        if refused is not None and refused.replay.get("results_subtitle") is False:
            assert case.subtitle is not None
            assert len(result.levels) >= 3
        else:
            assert case.subtitle is None
            assert not result.settings.results_subtitle
        return
    # The drawn and the stored subtitle are one expression, from the result's
    # numbers; it equals ggstatsplot's or is a pinned difference.
    node = _own_subtitle(case, result)
    assert figure.subtitle == tx.to_mathtext(node)
    assert result.text.subtitle == tx.to_plain(node)
    difference = _subtitle_difference(case, result)
    assert difference is None or difference in DIFFERING_SUBTITLES
    if case.type != "bayes":
        assert "n_pairs = " in result.text.subtitle
    if difference is None and case.bootstrap_reference is None:
        assert case.subtitle is not None
        theirs = parse(case.subtitle)
        assert node == (_v01(theirs) if case.type == "bayes" else theirs)


def _own_subtitle(case: OracleCase, result: StatsResult) -> tx.Node:
    """The subtitle AST inferplot builds from the result's own numbers."""
    if result.test is None:
        assert result.bayes is not None
        return tx.bayes_factor_expression(
            log_bf01=result.bayes.log_bf01,
            prior=result.bayes.prior,
            prior_scale=result.bayes.prior_scale,
            digits=_digits(case),
        )
    assert result.effect_size is not None
    return tx.subtitle_expression(
        result.test,
        result.effect_size,
        n_obs=result.n_obs,
        digits=_digits(case),
        paired=True,
    )


def test_differing_subtitles_are_pinned() -> None:
    differing = [
        found
        for case in CASES
        if (found := _subtitle_difference(case, analyze(case.name))) is not None
    ]
    assert tuple(differing) == DIFFERING_SUBTITLES


def test_hidden_subtitle_is_reported_with_a_warning() -> None:
    """The one subtitle ggstatsplot drops (D64) is drawn, with its reason."""
    name = DIFFERING_SUBTITLES[0][0]
    result = analyze(name)
    assert result.effect_size is not None
    assert result.effect_size.estimate == 1.0
    assert any("Kendall's W is 1" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# caption
# --------------------------------------------------------------------------- #


def _expected_pairwise_line(case: OracleCase, result: StatsResult) -> str | None:
    """The caption's pairwise line written out from the case's arguments and the
    brackets ggstatsplot drew (SPEC.md section 7)."""
    m = len(result.pairwise)
    display = result.settings.pairwise_display
    if m == 0 or display == "none":
        return None
    test = _PAIR_TESTS[case.type]
    if case.type == "bayes":
        return (
            f"Pairwise: Bayes factors ({test}, JZS prior 0.707), unadjusted; "
            f"showing all {m} pairs."
        )
    method = f"{test}, {_ADJUST_WORDS[result.settings.p_adjust_method]}"
    if display == "all":
        return f"Pairwise: {method}; showing all {m} pairs."
    assert result.settings.pairwise_alpha == 0.05
    drawn = len(_drawn(case))
    note = _DISPLAY_NOTES[display]
    return f"Pairwise: {method}; showing {drawn} of {m} pairs ({note})."


@pytest.mark.parametrize("name", _params())
def test_caption(name: str) -> None:
    case, result = wc.BY_NAME[name], analyze(name)
    figure = tx.figure_texts(result)
    k = len(result.levels)
    lines: list[str] = []
    plain: list[str] = []
    if case.caption is not None and k == 2:
        # The two-condition parametric Bayes caption: ggstatsplot's first and
        # last items (D32), from inferplot's paired Bayes factor (D61).
        assert case.type == "parametric"
        v01 = _v01(parse(case.caption))
        assert result.bayes is not None
        ours = tx.bayes_factor_expression(
            log_bf01=result.bayes.log_bf01,
            prior=result.bayes.prior,
            prior_scale=result.bayes.prior_scale,
            digits=_digits(case),
        )
        assert ours == v01
        lines.append(tx.to_mathtext(v01))
        plain.append(tx.to_plain(v01))
    elif case.caption is not None:
        # The Bayes ANOVA caption (v0.2): not drawn, and warnings says so.
        assert case.type == "parametric"
        assert result.bayes is None
        assert any(
            "caption for three or more conditions arrives in inferplot 0.2" in w
            for w in result.warnings
        )
    else:
        assert result.bayes is None or case.type == "bayes"
    line = _expected_pairwise_line(case, result)
    if line is not None:
        lines.append(line)
        plain.append(line)
    assert figure.caption_lines == tuple(lines)
    assert result.text.caption == ("\n".join(plain) if plain else None)


def test_pairwise_test_column_names_the_caption_test() -> None:
    for case in CASES:
        for row in case.pairwise:
            assert str(row["test"]) == _PAIR_TESTS[case.type], case.name


def test_differing_captions_are_pinned() -> None:
    assert DIFFERING_CAPTIONS == ()


# --------------------------------------------------------------------------- #
# centrality and tick labels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", _params())
def test_centrality_and_tick_labels(name: str) -> None:
    """Both centrality label layers (the points and the labels) describe the
    complete subjects only: their n is the number of complete subjects."""
    case, result = wc.BY_NAME[name], analyze(name)
    figure = tx.figure_texts(result)
    for layer in wc.centrality_layers(case):
        assert list(layer) == list(result.levels)
        expected = [parse(str(layer[level]["expression"])) for level in result.levels]
        assert list(result.text.centrality_labels) == [tx.to_plain(n) for n in expected]
        assert list(figure.centrality_labels) == [tx.to_mathtext(n) for n in expected]
        ticks = [str(layer[level]["n.expression"]) for level in result.levels]
        assert list(result.text.tick_labels) == ticks
        assert list(figure.tick_labels) == ticks
        assert {int(layer[level]["n.obs"]) for level in result.levels} == {result.n_obs}
    x = str(case.args["x"])
    assert [r_level_label(level) for level in case.within_data["levels"]] == list(
        result.levels
    ), x


# --------------------------------------------------------------------------- #
# bracket labels and the pairs drawn
# --------------------------------------------------------------------------- #


def _label_check(
    case: OracleCase, result: StatsResult
) -> list[tuple[str, str, str, str, str]]:
    """Compare every bracket label with ggstatsplot's; return the differing ones,
    each checked to be inferplot's formatter on a value within tolerance of
    ggstatsplot's."""
    figure = tx.figure_texts(result)
    rows = wc.pair_rows(case, result)
    digits = _digits(case)
    differing: list[tuple[str, str, str, str, str]] = []
    for i, (pair, row) in enumerate(zip(result.pairwise, rows, strict=True)):
        theirs = parse(str(row["expression"]))
        if result.text.pairwise_labels[i] == tx.to_plain(theirs):
            assert figure.pairwise_labels[i] == tx.to_mathtext(theirs)
            continue
        if pair.bayes is not None:
            r_bf10 = number(row["bf10"])
            assert theirs == tx.pairwise_bf_expression(
                log_bf01=-math.log(r_bf10), digits=digits
            )
            assert is_close("ttest_bayes_factor", pair.bayes.bf10, r_bf10)
            r_text = tx.format_value(-math.log(r_bf10), digits)
            our_text = tx.format_value(pair.bayes.log_bf01, digits)
        else:
            assert pair.p_adjusted is not None
            r_p = number(row["p.value"])
            assert theirs == tx.pairwise_p_expression(
                p_adjusted=r_p, adjust_method=pair.adjust_method, digits=digits
            )
            assert case.type == "robust"
            assert is_close("wrs2_p", pair.p_adjusted, r_p, atol_scale=2.0 * len(rows))
            r_text = tx.format_value(r_p, digits)
            our_text = tx.format_value(pair.p_adjusted, digits)
        assert r_text != our_text
        differing.append((case.name, pair.group1, pair.group2, r_text, our_text))
    drawn = [i for i, pair in enumerate(result.pairwise) if pair.displayed]
    annotations = _drawn_annotations(case)
    assert len(annotations) == len(drawn)
    for i, source in zip(drawn, annotations, strict=True):
        assert parse(source) == parse(str(rows[i]["expression"]))
    return differing


@pytest.mark.parametrize("name", _params())
def test_bracket_labels_and_drawn_pairs(name: str) -> None:
    case, result = wc.BY_NAME[name], analyze(name)
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == _drawn(case)
    if case.pairwise:
        _label_check(case, result)
    else:
        # ggstatsplot computes no pairs with two conditions or display "none".
        assert len(result.levels) == 2 or result.settings.pairwise_display == "none"
        assert not drawn


def test_differing_labels_are_pinned() -> None:
    differing = [
        item
        for case in CASES
        if case.pairwise
        for item in _label_check(case, analyze(case.name))
    ]
    assert tuple(differing) == DIFFERING_LABELS
