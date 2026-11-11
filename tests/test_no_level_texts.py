"""The texts of ``ip.analyze.histostats`` and ``ip.analyze.scatterstats`` against
every gghistostats and ggscatterstats fixture.

Two layers of checks, as for the earlier functions:

* the builders: every expression a fixture holds (each subtitle and caption table
  row, the drawn subtitle and caption, gghistostats' centrality label) is rebuilt
  from that fixture's own numbers with ``inferplot.text``'s builders and must equal
  ggstatsplot's, parsed (``tests/plotmath.py``), as an AST: every symbol, subscript,
  df and number string. This covers all 165 fixtures, also the 13 that inferplot
  refuses (constant x and 3 pairs: ggstatsplot prints "NA"; constant one-sample
  data: ggstatsplot prints no subtitle, or t = Inf with an NA interval) and the
  exactly collinear ones (an empty log Bayes factor). One stored number loses the
  side of a rounding tie R's double lay on (:data:`STORED_TIES`);
* the results: each fixture inferplot answers (152; the two one-sided robust and
  bayes histostats calls by their two-sided replay, ``tests/no_level_cases.py``)
  is replayed through ``ip.analyze`` and its subtitle, caption and centrality
  label, built from inferplot's own numbers, must equal ggstatsplot's or be a
  pinned, explained difference (:data:`DIFFERING_SUBTITLES`,
  :data:`DIFFERING_CAPTIONS`). The bootstrap-derived numbers of the robust
  histostats subtitle (the bootstrap-t p and interval) follow SPEC.md section
  10: inferplot's formatter applied to ggstatsplot's own 200-resample values must
  reproduce its strings (the values themselves are compared with the
  20,000-resample references in tests/test_histostats_parity.py). Type bayes and
  the Bayes caption are the first and last items of ggstatsplot's expression
  (log_e BF01 and the prior; the posterior estimate arrives in v0.2, D32, D41).

Every expression is drawn with matplotlib's mathtext (the oldest supported
matplotlib, 3.8, included), and its glyphs must be those plotmath draws. The APA
sentences and alt texts of every fixture are checked for form; exact strings are
pinned for standard cases of each function and type; ``figure_alt_text``'s
keywords are checked in every combination.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot import text as tx
from inferplot.result import BayesFactor, StatsResult
from tests import no_level_cases as nl
from tests.oracle_fixtures import OracleCase, number, read_csv
from tests.plotmath import expected_glyphs, parse
from tests.test_text_expressions import effect_size_key, prior_key, statistic_key
from tests.test_text_render import _normalized, render_glyphs

Row = Mapping[str, Any]

CASES: tuple[OracleCase, ...] = (*nl.HISTO, *nl.SCATTER)

# Subtitles that differ from ggstatsplot's: case -> (ggstatsplot's text,
# inferplot's text, why), both as plain text.
_COLLINEAR = (
    "exactly collinear data (D85, SPEC section 5 item 20): inferplot reports the "
    "exact limits (t = +/-Inf, S = 0, p = 0, printed 2.23e-308 as ggstatsplot "
    "prints a p of 0) where ggstatsplot prints the rounding of R's arithmetic"
)
_COLLINEAR_BF = (
    "exactly collinear data (D86): inferplot's Bayes factor is its limit, BF10 = "
    "Inf (log_e BF01 = -Inf); BayesFactor returns NaN and ggstatsplot prints an "
    "empty value"
)
DIFFERING_SUBTITLES: Mapping[str, tuple[str, str, str]] = {
    "gghistostats-ToothGrowth-tv100-parametric": (
        "t_Student(59) = -82.21, p = 1.46e-62, g_Hedges = -10.48, "
        "CI_95% [-12.34, -8.53], n_obs = 60",
        "t_Student(59) = -82.21, p = 1.46e-62, g_Hedges = -10.48, "
        "CI_95% [-12.38, -8.57], n_obs = 60",
        "t = -82.2 (D81, SPEC section 5 item 19): R's noncentral-t interval is "
        "-0.30% and -0.55% off; inferplot reports the accurate bounds, checked "
        "against 40-digit references in tests/stats",
    ),
    "ggscatterstats-edge_corr_perfect-bayes": (
        "log_e(BF_01) = , \u03c1_Pearson^posterior = 1.00, CI_95%^HDI [1.00, 1.00], "
        "r_beta^JZS = 1.41",
        "log_e(BF_01) = -Inf, r_beta^JZS = 1.41",
        _COLLINEAR_BF,
    ),
    "ggscatterstats-edge_corr_perfect-neg-bayes": (
        "log_e(BF_01) = , \u03c1_Pearson^posterior = -1.00, CI_95%^HDI [-1.00, "
        "-1.00], r_beta^JZS = 1.41",
        "log_e(BF_01) = -Inf, r_beta^JZS = 1.41",
        _COLLINEAR_BF,
    ),
    "ggscatterstats-edge_corr_perfect-neg-nonparametric": (
        "S = 330.00, p = 1.70e-61, \u03c1_Spearman = -1.00, CI_95% [-1.00, -1.00], "
        "n_pairs = 10",
        "S = 330.00, p = 2.23e-308, \u03c1_Spearman = -1.00, CI_95% [-1.00, -1.00], "
        "n_pairs = 10",
        _COLLINEAR,
    ),
    "ggscatterstats-edge_corr_perfect-neg-robust": (
        "t_Student(8) = -1.90e+08, p = 6.65e-64, r_Winsorized = -1.00, "
        "CI_95% [-1.00, -1.00], n_pairs = 10",
        "t_Student(8) = -Inf, p = 2.23e-308, r_Winsorized = -1.00, "
        "CI_95% [-1.00, -1.00], n_pairs = 10",
        _COLLINEAR,
    ),
    "ggscatterstats-edge_corr_perfect-nonparametric": (
        "S = 3.66e-14, p = 1.06e-62, \u03c1_Spearman = 1.00, CI_95% [1.00, 1.00], "
        "n_pairs = 10",
        "S = 0.00, p = 2.23e-308, \u03c1_Spearman = 1.00, CI_95% [1.00, 1.00], "
        "n_pairs = 10",
        _COLLINEAR,
    ),
    "ggscatterstats-edge_corr_perfect-parametric": (
        "t_Student(8) = 1.34e+08, p = 1.06e-62, r_Pearson = 1.00, "
        "CI_95% [1.00, 1.00], n_pairs = 10",
        "t_Student(8) = Inf, p = 2.23e-308, r_Pearson = 1.00, "
        "CI_95% [1.00, 1.00], n_pairs = 10",
        _COLLINEAR,
    ),
    "ggscatterstats-edge_corr_perfect-robust": (
        "t_Student(8) = 1.34e+08, p = 1.06e-62, r_Winsorized = 1.00, "
        "CI_95% [1.00, 1.00], n_pairs = 10",
        "t_Student(8) = Inf, p = 2.23e-308, r_Winsorized = 1.00, "
        "CI_95% [1.00, 1.00], n_pairs = 10",
        _COLLINEAR,
    ),
    # n = 2 (barstats oracle run: 3.1 and 4.7 against 3): t is a rounding tie for
    # the data as written. (The robust case, t = 0.675 also a tie, now prints
    # R's 0.68: its t is formed from exact quantities and rounded once.)
    "gghistostats-edge_one_n2-parametric": (
        "t_Student(1) = 1.13, p = 0.46, g_Hedges = 0.00, CI_95% [0.00, 0.00], "
        "n_obs = 2",
        "t_Student(1) = 1.12, p = 0.46, g_Hedges = 0.00, CI_95% [0.00, 0.00], "
        "n_obs = 2",
        "t = 0.9 / 0.8 = 1.125 exactly for the decimal data, a tie at 2 digits. "
        "inferplot's t is the correctly rounded double, 1.125, which C's sprintf "
        "prints 1.12 (ties to even); R's arithmetic gives 1.1250000000000004, "
        "printed 1.13. The exact t of the stored doubles is 1.125 + 1.04e-16, so "
        "1.13 is its correct decimal rounding",
    ),
}

# Expressions whose number the fixture stores to 15 significant digits, which
# loses the side of a rounding tie R's double lay on: case -> (R's string, the
# string the formatter gives the stored number). edge_one_n2-parametric: R's t is
# 1.1250000000000004 (printed 1.13), stored as 1.125 (printed 1.12).
STORED_TIES: Mapping[str, tuple[str, str]] = {
    "gghistostats-edge_one_n2-parametric": ("1.13", "1.12"),
}

# Bayes captions that differ from ggstatsplot's first and last items.
DIFFERING_CAPTIONS: Mapping[str, tuple[str, str, str]] = {
    f"ggscatterstats-edge_corr_perfect{sign}-parametric": (
        "log_e(BF_01) = , r_beta^JZS = 1.41",
        "log_e(BF_01) = -Inf, r_beta^JZS = 1.41",
        _COLLINEAR_BF,
    )
    for sign in ("", "-neg")
}

# The subtitle of edge_corr_perfect-neg-parametric is ggstatsplot's character for
# character: R's own t is -Inf there and its p 0 (2.23e-308).
PERFECT_BUT_EQUAL = "ggscatterstats-edge_corr_perfect-neg-parametric"


@functools.cache
def analyzed(name: str) -> StatsResult | None:
    return nl.analyzed(name, "pandas")


def _answered() -> list[OracleCase]:
    return [
        c
        for c in CASES
        if nl.REFUSALS.get(c.name, None) is None
        or nl.REFUSALS[c.name].replay is not None
    ]


def _params(cases: list[OracleCase] | tuple[OracleCase, ...]) -> list[Any]:
    return [pytest.param(c.name, id=c.name) for c in cases]


def _digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def _v01(node: tx.Node) -> tx.Node:
    """The first and last items of a Bayes expression (D32, D41)."""
    assert isinstance(node, tx.Call)
    assert node.fn == tx.Sym("list")
    return tx.Call(tx.Sym("list"), (node.args[0], node.args[-1]))


def test_every_fixture_is_covered() -> None:
    # 76 and 64, and 16 and 9 from the barstats oracle run (edge_one_n2,
    # edge_one_n3, edge_one_constant, edge_one_constant-binwidth1; edge_corr_n3,
    # edge_corr_n4, edge_corr_label_missing); edge_one_near_mu-literal from review
    # round 3.
    assert (len(nl.HISTO), len(nl.SCATTER)) == (93, 73)
    refused = sorted(c.name for c in CASES if analyzed(c.name) is None)
    assert refused == sorted(
        [
            f"gghistostats-edge_one_constant{width}-{t}"
            for width in ("", "-binwidth1")
            for t in ("bayes", "parametric", "robust")
        ]
        + [
            f"ggscatterstats-edge_corr_constant_x-{t}"
            for t in ("bayes", "nonparametric", "parametric", "robust")
        ]
        + [
            f"ggscatterstats-edge_corr_n3-{t}"
            for t in ("nonparametric", "parametric", "robust")
        ]
    )
    assert len(_answered()) == 153


# --------------------------------------------------------------------------- #
# the builders, on every fixture's own numbers
# --------------------------------------------------------------------------- #


def _rebuild_row(case: OracleCase, row: Row) -> tx.Node:
    """A subtitle or caption row's expression, from the row's numbers."""
    digits = _digits(case)
    if "bf10" in row:
        return tx.bayes_expression(
            log_bf01=-math.log(number(row["bf10"])),
            effect_size=effect_size_key(str(row["effectsize"])),
            estimate=number(row["estimate"]),
            conf_level=number(row["conf.level"]),
            conf_method=str(row["conf.method"]),
            conf_low=number(row["conf.low"]),
            conf_high=number(row["conf.high"]),
            prior=prior_key(str(row["method"])),
            prior_scale=number(row["prior.scale"]),
            digits=digits,
        )
    # Neither design has a fractional df: t-tests and correlations print n - 1
    # and n - 2 unquoted, the signed-rank and Spearman tests none.
    assert "df" not in row
    return tx.frequentist_expression(
        statistic=statistic_key(str(row["method"])),
        statistic_value=number(row["statistic"]),
        p_value=number(row["p.value"]),
        effect_size=effect_size_key(str(row["effectsize"])),
        estimate=number(row["estimate"]),
        conf_level=number(row["conf.level"]),
        conf_low=number(row["conf.low"]),
        conf_high=number(row["conf.high"]),
        n_obs=int(row["n.obs"]),
        df_error=number(row["df.error"]) if "df.error" in row else None,
        digits=digits,
        paired=case.function == "ggscatterstats",
    )


def fixture_expressions(case: OracleCase) -> list[tuple[str, str, tx.Node]]:
    """(where, ggstatsplot's deparsed expression, the expression rebuilt from the
    fixture's numbers) for every expression of one fixture."""
    out: list[tuple[str, str, tx.Node]] = []
    rows = case.subtitle_rows
    for i, row in enumerate(rows):
        out.append(
            (f"subtitle_data[{i}]", str(row["expression"]), _rebuild_row(case, row))
        )
    if case.subtitle is not None:
        out.append(("subtitle", case.subtitle, _rebuild_row(case, rows[0])))
    captions = case.table("caption_data")
    for i, row in enumerate(captions):
        out.append(
            (f"caption_data[{i}]", str(row["expression"]), _rebuild_row(case, row))
        )
    if case.caption is not None:
        out.append(("caption", case.caption, _rebuild_row(case, captions[0])))
    if case.function == "gghistostats" and case.centrality is not None:
        centrality = {
            "parametric": "mean",
            "nonparametric": "median",
            "robust": "trimmed_mean",
            "bayes": "map",
        }[case.type]
        for i, row in enumerate(case.centrality):
            out.append(
                (
                    f"centrality[{i}]",
                    str(row["expression"]),
                    tx.centrality_expression(
                        centrality=centrality,
                        value=number(row["var"]),
                        digits=_digits(case),
                    ),
                )
            )
    return out


@pytest.mark.parametrize("name", _params(CASES))
def test_fixture_expressions_rebuild_from_their_numbers(name: str) -> None:
    case = nl.BY_NAME[name]
    found = fixture_expressions(case)
    where = {w.split("[")[0] for w, _, _ in found}
    results = bool(case.args.get("results.subtitle", True))
    hidden = case.expected_hidden_errors
    # A subtitle exactly where the call asks for one, but where ggstatsplot drops
    # it (edge_corr_constant_x-bayes: correlationBF stops on a constant column).
    assert ("subtitle" in where) == (results and "subtitle" not in hidden)
    bayes_caption = (
        case.type == "parametric"
        and results
        and case.args.get("bf.message") is not False
        and "caption" not in hidden
    )
    assert ("caption" in where) == bayes_caption
    has_line = case.function == "gghistostats" and case.args.get(
        "centrality.plotting", True
    )
    assert ("centrality" in where) == bool(has_line)
    tie = STORED_TIES.get(name)
    for location, source, rebuilt in found:
        if tie is not None and location.startswith("subtitle"):
            # The one difference is the tie's string (STORED_TIES).
            theirs, stored = (f'"{s}"' for s in tie)
            assert source.count(theirs) == 1
            assert rebuilt != parse(source)
            assert rebuilt == parse(source.replace(theirs, stored)), source
            continue
        assert rebuilt == parse(source), f"{location}: {source}"


def test_fixture_rows_cover_every_template() -> None:
    """Every statistic, effect size and prior of both designs occurs."""
    seen: set[str] = set()
    for case in CASES:
        for _, source, _ in fixture_expressions(case):
            seen.update(re.findall(r'\["(\w+)"\]', source))
    assert {
        "Student",
        "Wilcoxon",
        "bootstrapped",
        "Hedges",
        "biserial",
        "trimmed",
        "Pearson",
        "Spearman",
        "Winsorized",
        "Cauchy",
        "beta",
        "obs",
        "pairs",
    } <= seen


# --------------------------------------------------------------------------- #
# inferplot's own texts against ggstatsplot's
# --------------------------------------------------------------------------- #


def _own_subtitle(result: StatsResult) -> tx.Node:
    """The subtitle AST inferplot builds from the result's own numbers."""
    digits = result.settings.digits
    if result.test is None:
        assert result.bayes is not None
        return tx.bayes_factor_expression(
            log_bf01=result.bayes.log_bf01,
            prior=result.bayes.prior,
            prior_scale=result.bayes.prior_scale,
            digits=digits,
        )
    assert result.effect_size is not None
    return tx.subtitle_expression(
        result.test,
        result.effect_size,
        n_obs=result.n_obs,
        digits=digits,
        paired=result.design == "correlation",
    )


def _comparable_subtitle(case: OracleCase, result: StatsResult) -> tx.Node:
    """inferplot's subtitle as compared with ggstatsplot's: with ggstatsplot's own
    values of the bootstrap-derived numbers (the robust histostats p and interval;
    SPEC section 10), and for type bayes the expression whose first and last items
    ggstatsplot's must share."""
    if result.test is None or case.bootstrap_reference is None:
        return _own_subtitle(result)
    assert result.effect_size is not None
    assert result.test.name == "trimmed_mean_bootstrap_t"
    row = case.subtitle_data
    test = dataclasses.replace(result.test, p_value=number(row["p.value"]))
    es = dataclasses.replace(
        result.effect_size,
        conf_low=number(row["conf.low"]),
        conf_high=number(row["conf.high"]),
    )
    return tx.subtitle_expression(
        test, es, n_obs=result.n_obs, digits=result.settings.digits, paired=False
    )


def _subtitle_difference(case: OracleCase) -> tuple[str, str] | None:
    """(ggstatsplot's subtitle, inferplot's), as plain text, where they differ."""
    result = analyzed(case.name)
    assert result is not None
    if case.subtitle is None:
        assert result.text.subtitle is None
        return None
    theirs = parse(case.subtitle)
    mine = _comparable_subtitle(case, result)
    expected = _v01(theirs) if result.test is None else theirs
    if mine == expected:
        return None
    assert result.text.subtitle is not None
    return (tx.to_plain(theirs), result.text.subtitle)


@pytest.mark.parametrize("name", _params(_answered()))
def test_subtitle(name: str) -> None:
    case, result = nl.BY_NAME[name], analyzed(name)
    assert result is not None
    figure = tx.figure_texts(result)
    if result.text.subtitle is None:
        assert figure.subtitle is None
        assert case.subtitle is None
        assert not result.settings.results_subtitle
        return
    node = _own_subtitle(result)
    # The drawn and the stored subtitle are one expression.
    assert figure.subtitle == tx.to_mathtext(node)
    assert result.text.subtitle == tx.to_plain(node)
    n = "n_pairs = " if result.design == "correlation" else "n_obs = "
    if result.test is not None:
        assert n + tx.format_n(result.n_obs) in result.text.subtitle
    difference = _subtitle_difference(case)
    if difference is None:
        if case.bootstrap_reference is None:
            assert case.subtitle is not None
            theirs = parse(case.subtitle)
            assert node == (_v01(theirs) if result.test is None else theirs)
    else:
        assert name in DIFFERING_SUBTITLES
        assert difference == DIFFERING_SUBTITLES[name][:2]


def test_differing_subtitles_are_pinned() -> None:
    differing = {
        case.name: difference
        for case in _answered()
        if (difference := _subtitle_difference(case)) is not None
    }
    assert differing == {k: v[:2] for k, v in DIFFERING_SUBTITLES.items()}
    assert PERFECT_BUT_EQUAL not in differing
    assert nl.BY_NAME[PERFECT_BUT_EQUAL].subtitle is not None


def test_robust_bootstrap_strings_follow_the_formatter() -> None:
    """The robust histostats subtitles: inferplot's statistic and estimate
    strings are ggstatsplot's; its bootstrap p and bounds come from 2,000
    resamples where ggstatsplot uses 200, and the formatter turns ggstatsplot's
    own values into its strings (including a bootstrap p of 0 as 2.23e-308)."""
    robust = [c for c in nl.HISTO if c.bootstrap_reference is not None]
    # 17, and 4 from the barstats oracle run: edge_one_n2, edge_one_n3,
    # edge_one_constant and edge_one_constant-binwidth1 (the last two refused).
    assert len(robust) == 21
    zero_p = 0
    for case in robust:
        result = analyzed(case.name)
        if result is None:  # constant data (tests/test_histostats_parity.py)
            assert nl.REFUSALS[case.name].replay is None
            continue
        assert result.test is not None
        if case.subtitle is None:
            continue
        row = case.subtitle_data
        theirs = parse(case.subtitle)
        assert isinstance(theirs, tx.Call)
        mine = _own_subtitle(result)
        assert isinstance(mine, tx.Call)
        # the bootstrap t, the trimmed mean and n are exact (edge_one_n2: t is a
        # rounding tie, DIFFERING_SUBTITLES)
        assert (mine.args[0] == theirs.args[0]) == (
            case.name not in DIFFERING_SUBTITLES
        )
        assert mine.args[2] == theirs.args[2]
        assert mine.args[-1] == theirs.args[-1]
        if number(row["p.value"]) < 1e-300:  # stored as .Machine$double.xmin
            zero_p += 1
            assert tx.format_p_subtitle(0.0) in case.subtitle
            # inferplot's 2,000 resamples give 0 too, printed the same way
            assert result.test.p_value == 0.0
            assert theirs.args[1] == mine.args[1]
    assert zero_p == 3


def _caption_difference(case: OracleCase) -> tuple[str, str] | None:
    result = analyzed(case.name)
    assert result is not None
    if case.caption is None:
        assert result.text.caption is None
        return None
    theirs = _v01(parse(case.caption))
    if result.text.caption == tx.to_plain(theirs):
        return None
    assert result.text.caption is not None
    return (tx.to_plain(theirs), result.text.caption)


@pytest.mark.parametrize("name", _params(_answered()))
def test_caption(name: str) -> None:
    case, result = nl.BY_NAME[name], analyzed(name)
    assert result is not None
    figure = tx.figure_texts(result)
    if case.caption is None:
        assert result.text.caption is None
        assert figure.caption_lines == ()
        drawn = (
            result.type == "parametric"
            and result.settings.results_subtitle
            and result.settings.bf_message is not False
        )
        assert not drawn
        return
    # The Bayes caption of type parametric: ggstatsplot's first and last items
    # (D32), built from inferplot's Bayes factor and the prior it prints.
    assert result.bayes is not None
    node = tx.bayes_factor_expression(
        log_bf01=result.bayes.log_bf01,
        prior=result.bayes.prior,
        prior_scale=result.bayes.prior_scale,
        digits=result.settings.digits,
    )
    assert figure.caption_lines == (tx.to_mathtext(node),)
    caption = result.text.caption
    assert caption is not None
    assert caption == tx.to_plain(node)
    scale = "r_Cauchy^JZS = " if result.design == "one-sample" else "r_beta^JZS = "
    assert scale in caption
    difference = _caption_difference(case)
    if difference is None:
        assert node == _v01(parse(case.caption))
    else:
        assert difference == DIFFERING_CAPTIONS[name][:2]


def test_differing_captions_are_pinned() -> None:
    differing = {
        case.name: difference
        for case in _answered()
        if (difference := _caption_difference(case)) is not None
    }
    assert differing == {k: v[:2] for k, v in DIFFERING_CAPTIONS.items()}


@pytest.mark.parametrize("name", _params(_answered()))
def test_centrality_and_other_labels(name: str) -> None:
    """histostats: one label, the centrality line's, equal to ggstatsplot's on
    every fixture that draws one (the nocentrality fixture computes none; inferplot's
    figure leaves it out, as the renderer's ``centrality_plotting``); scatterstats:
    none. Neither design has tick or pairwise labels."""
    case, result = nl.BY_NAME[name], analyzed(name)
    assert result is not None
    figure = tx.figure_texts(result)
    assert result.text.tick_labels == figure.tick_labels == ()
    assert result.text.pairwise_labels == figure.pairwise_labels == ()
    if result.design == "correlation":
        assert result.text.centrality_labels == figure.centrality_labels == ()
        return
    (d,) = result.descriptives
    node = tx.centrality_expression(
        centrality=d.centrality, value=d.value, digits=result.settings.digits
    )
    assert figure.centrality_labels == (tx.to_mathtext(node),)
    assert result.text.centrality_labels == (tx.to_plain(node),)
    if case.centrality is None:
        assert case.args.get("centrality.plotting") is False
        return
    (row,) = case.centrality
    assert node == parse(str(row["expression"]))


# --------------------------------------------------------------------------- #
# mathtext
# --------------------------------------------------------------------------- #


@functools.cache
def _render_items() -> tuple[
    list[tuple[str, str, bool]], dict[str, str], dict[str, str]
]:
    """Every distinct expression of the new fixtures (ggstatsplot's, parsed) and
    every one inferplot draws for them (subtitle, Bayes caption, centrality)."""
    sources: dict[str, tx.Node] = {}
    for case in CASES:
        for _, source, _ in fixture_expressions(case):
            sources.setdefault(source, parse(source))
        result = analyzed(case.name)
        if result is None:
            continue
        ex = tx._expressions(  # pyright: ignore[reportPrivateUsage]
            descriptives=result.descriptives,
            test=result.test,
            effect_size=result.effect_size,
            bayes=result.bayes,
            pairwise=(),
            n_obs=result.n_obs,
            settings=result.settings,
            paired=result.design == "correlation",
        )
        for node in (ex.subtitle, ex.bayes_caption, *ex.centrality):
            if node is not None:
                sources.setdefault(f"[inferplot] {tx.to_plain(node)}", node)
    items: list[tuple[str, str, bool]] = []
    expected: dict[str, str] = {}
    where: dict[str, str] = {}
    for k, (source, node) in enumerate(sorted(sources.items())):
        gid = f"n{k}"
        items.append((gid, tx.to_mathtext(node), True))
        expected[gid] = expected_glyphs(node)
        where[gid] = source
    return items, expected, where


_PAGE = 50


def test_every_new_expression_is_rendered() -> None:
    items, _, where = _render_items()
    assert len(items) > 200
    assert any(w.startswith("[inferplot]") for w in where.values())


@pytest.mark.parametrize("start", range(0, 400, _PAGE))
def test_mathtext_renders_with_plotmath_glyphs(start: int) -> None:
    items, expected, where = _render_items()
    page = items[start : start + _PAGE]
    if not page:
        return
    glyphs = render_glyphs(page)
    failures = []
    for gid, _, _ in page:
        got, wanted = _normalized(glyphs[gid]), _normalized(expected[gid])
        if got != wanted:
            failures.append(
                f"{where[gid]}: missing {wanted - got}, extra {got - wanted}"
            )
    assert not failures


def test_render_pages_reach_every_item() -> None:
    items, _, _ = _render_items()
    assert len(items) <= 400


# --------------------------------------------------------------------------- #
# the APA sentence and the alt text on every fixture
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_GROUP_WORDS = re.compile(r"\b(groups?|levels?|conditions?|subjects?)\b", re.I)
# R's and Python's spellings of special values that inferplot never prints
# ("NA", "NaN", "nan", "inf", "Infinity", "None"); "Inf" and "-Inf" are R's
# spelling of an infinite number, the one inferplot uses.
_BAD_SPECIAL = re.compile(r"\b(NA|NaN|nan|inf|Infinity|infinity|None)\b")


def _sentences(text: str) -> int:
    return len(_SENTENCE_END.findall(text))


def _apa(x: float, bounded: bool = False) -> str:
    return tx._apa_number(x, bounded=bounded)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("name", _params(_answered()))
def test_apa_sentence_and_alt_text_form(name: str) -> None:
    result = analyzed(name)
    assert result is not None
    apa, alt = result.text.apa, result.text.alt_text
    one_sample = result.design == "one-sample"
    n = f"n = {tx.format_n(result.n_obs)}" + ("" if one_sample else " pairs")
    for text in (apa, alt):
        assert not _BAD_SPECIAL.search(text), text
        assert not _GROUP_WORDS.search(text), text
        assert text.isascii()  # the fixtures' column names are ASCII
        assert not one_sample or "pairs" not in text
    assert apa.endswith(f"{n}.")
    assert _sentences(apa) == 1
    assert str(result) == apa
    # numbers equal the result's
    test, es, bayes = result.test, result.effect_size, result.bayes
    if test is not None and es is not None:
        assert f"{test.statistic_name}" in apa
        assert f" = {_apa(test.statistic)}, " in apa
        bounded = es.name in (
            "pearson_r",
            "spearman_rho",
            "winsorized_r",
            "rank_biserial",
        )
        level = tx.format_conf_level(es.conf_level)
        if es.name == "trimmed_mean":
            # In the data's units: the subtitle's number rule with digits
            # (review item Y4), "19.24" at digits 2, "19.244" at digits 3.
            digits = result.settings.digits
            shown = [
                tx.format_value(x, digits)
                for x in (es.estimate, es.conf_low, es.conf_high)
            ]
        else:
            shown = [_apa(x, bounded) for x in (es.estimate, es.conf_low, es.conf_high)]
        assert f"{es.label} = {shown[0]}, " in apa
        assert f"{level} CI [{shown[1]}, {shown[2]}]" in apa
        if one_sample:
            against = tx.r_number_text(result.settings.test_value or 0.0)
            assert (f" against {against}: " in apa) == (test.alternative == "two-sided")
    elif bayes is not None:
        assert f"BF10 = {tx.format_bf10(bayes.bf10, bayes.log_bf01)}, " in apa
        assert f"log(BF01) = {_apa(bayes.log_bf01)}, " in apa
    else:
        assert apa == f"No test was computed (results_subtitle=False), {n}."
    # the alt text: 2 to 4 sentences, at most 400 characters, the figure first
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
    assert 2 <= _sentences(alt) <= 4, alt
    assert result.x in alt
    if one_sample:
        assert alt.startswith(f"Histogram of {result.x} (")
        (d,) = result.descriptives
        if d.value is not None:
            assert f", {tx.format_value(d.value, result.settings.digits)}." in alt
    else:
        assert result.y is not None
        assert alt.startswith(f"Scatter plot of {result.y} against {result.x} (")
        assert "least-squares line" in alt
        assert ("labels from " in alt) == any(result.point_labels)
    if result.settings.results_subtitle or apa.startswith("No test was computed"):
        assert alt.endswith(apa)
    else:
        assert alt.endswith(f"No test result is shown (results_subtitle=False), {n}.")
    assert alt == tx.figure_alt_text(result)


def test_ascii_except_the_users_names() -> None:
    """The sentences are ASCII except column names, printed as given, and the
    ellipsis that shortens a long one."""
    frame = pd.DataFrame({"Länge ä": [1.0, 2.0, 3.0, 4.0, 6.0]})
    result = ip.analyze.histostats(
        frame, "Länge ä", test_value=2, alternative="greater"
    )
    assert result.text.apa.startswith(
        "One-sample t-test (one-sided: Länge ä greater than 2): t(4) = "
    )
    long = "Länge" * 20
    frame = pd.DataFrame({long: [1.0, 2.0, 3.0, 4.0, 6.0]})
    result = ip.analyze.histostats(frame, long, test_value=2, alternative="less")
    assert result.text.apa.startswith(
        f"One-sample t-test (one-sided: {long[:39]}… less than 2): "
    )
    assert len(result.text.alt_text) <= tx.ALT_TEXT_MAX_CHARS


# --------------------------------------------------------------------------- #
# exact strings
# --------------------------------------------------------------------------- #

_TOOTH = "Histogram of len (60 values, from 4.20 to 33.90)."
_MTCARS = "Scatter plot of mpg against wt (32 pairs): mpg tends to fall as wt rises."
_LAYERS = (
    "It also shows a least-squares line with its 95% confidence band, and marginal "
    "histograms of wt and mpg."
)

EXACT: Mapping[str, tuple[str, str]] = {
    "gghistostats-ToothGrowth-parametric": (
        "One-sample t-test against 18: t(59) = 0.82, p = .413, Hedges' g = 0.10, "
        "95% CI [-0.15, 0.36], n = 60.",
        f"{_TOOTH} A vertical line marks the mean, 18.81. {{apa}}",
    ),
    "gghistostats-ToothGrowth-nonparametric": (
        "Wilcoxon signed-rank test against 18: V = 1030.50, p = .397, rank-biserial "
        "r = .13, 95% CI [-.16, .40], n = 60.",
        f"{_TOOTH} A vertical line marks the median, 19.25. {{apa}}",
    ),
    "gghistostats-ToothGrowth-bayes": (
        "Bayesian one-sample t-test against 18 (JZS prior, r = 0.707): BF10 = 0.20, "
        "log(BF01) = 1.63, n = 60.",
        f"{_TOOTH} A vertical line marks the MAP estimate, 24.12. {{apa}}",
    ),
    "gghistostats-ToothGrowth-prior08-bayes": (
        "Bayesian one-sample t-test against 18 (JZS prior, r = 0.8): BF10 = 0.17, "
        "log(BF01) = 1.75, n = 60.",
        f"{_TOOTH} A vertical line marks the MAP estimate, 24.12. {{apa}}",
    ),
    "gghistostats-ToothGrowth-greater-parametric": (
        "One-sample t-test (one-sided: len greater than 18): t(59) = 0.82, p = .207, "
        "Hedges' g = 0.10, 95% CI [-0.11, Inf], n = 60.",
        f"{_TOOTH} A vertical line marks the mean, 18.81. {{apa}}",
    ),
    "gghistostats-ToothGrowth-less-nonparametric": (
        "Wilcoxon signed-rank test (one-sided: len less than 18): V = 1030.50, "
        "p = .803, rank-biserial r = .13, 95% CI [-1.00, .35], n = 60.",
        f"{_TOOTH} A vertical line marks the median, 19.25. {{apa}}",
    ),
    "gghistostats-ToothGrowth-nosubtitle-parametric": (
        "One-sample t-test against 18: t(59) = 0.82, p = .413, Hedges' g = 0.10, "
        "95% CI [-0.15, 0.36], n = 60.",
        f"{_TOOTH} A vertical line marks the mean, 18.81. No test result is shown "
        "(results_subtitle=False), n = 60.",
    ),
    "gghistostats-ToothGrowth-nosubtitle-bayes": (
        "No test was computed (results_subtitle=False), n = 60.",
        f"{_TOOTH} A vertical line marks the MAP estimate, 24.12. {{apa}}",
    ),
    "gghistostats-edge_one_large_n-parametric": (
        "One-sample t-test against 100: t(1999) = 0.04, p = .969, Hedges' g = 0.00, "
        "95% CI [-0.04, 0.04], n = 2,000.",
        "Histogram of value (2,000 values, from 80.01 to 119.97). A vertical line "
        "marks the mean, 100.01. {apa}",
    ),
    "ggscatterstats-mtcars-parametric": (
        "Pearson correlation: t(30) = -9.56, p < .001, Pearson's r = -.87, 95% CI "
        "[-.93, -.74], n = 32 pairs.",
        f"{_MTCARS} {_LAYERS} {{apa}}",
    ),
    "ggscatterstats-mtcars-nonparametric": (
        "Spearman correlation: S = 10292.32, p < .001, Spearman's rho = -.89, 95% CI "
        "[-.94, -.77], n = 32 pairs.",
        f"{_MTCARS} {_LAYERS} {{apa}}",
    ),
    "ggscatterstats-mtcars-robust": (
        "Winsorized Pearson correlation (20% winsorized): t(30) = -9.41, p < .001, "
        "winsorized r = -.86, 95% CI [-.93, -.74], n = 32 pairs.",
        f"{_MTCARS} {_LAYERS} {{apa}}",
    ),
    "ggscatterstats-mtcars-tr01-robust": (
        "Winsorized Pearson correlation (10% winsorized): t(30) = -11.10, p < .001, "
        "winsorized r = -.90, 95% CI [-.95, -.80], n = 32 pairs.",
        f"{_MTCARS} {_LAYERS} {{apa}}",
    ),
    "ggscatterstats-mtcars-bayes": (
        "Bayesian Pearson correlation (stretched beta prior, r_beta = 1.41): BF10 = "
        "5.62e+07, log(BF01) = -17.84, n = 32 pairs.",
        f"Scatter plot of mpg against wt (32 pairs). {_LAYERS} {{apa}}",
    ),
    "ggscatterstats-mtcars-prior08-bayes": (
        "Bayesian Pearson correlation (stretched beta prior, r_beta = 1.25): BF10 = "
        "6.42e+07, log(BF01) = -17.98, n = 32 pairs.",
        f"Scatter plot of mpg against wt (32 pairs). {_LAYERS} {{apa}}",
    ),
    "ggscatterstats-mtcars-nosubtitle-bayes": (
        "No test was computed (results_subtitle=False), n = 32 pairs.",
        f"Scatter plot of mpg against wt (32 pairs). {_LAYERS} {{apa}}",
    ),
    "ggscatterstats-mtcars-nosubtitle-parametric": (
        "Pearson correlation: t(30) = -9.56, p < .001, Pearson's r = -.87, 95% CI "
        "[-.93, -.74], n = 32 pairs.",
        f"Scatter plot of mpg against wt (32 pairs). {_LAYERS} No test result is "
        "shown (results_subtitle=False), n = 32 pairs.",
    ),
    "ggscatterstats-mtcars-labelexpr-parametric": (
        "Pearson correlation: t(30) = -9.56, p < .001, Pearson's r = -.87, 95% CI "
        "[-.93, -.74], n = 32 pairs.",
        f"{_MTCARS} It also shows a least-squares line with its 95% confidence band, "
        "marginal histograms of wt and mpg, and labels from model on 9 of the 32 "
        "points. {apa}",
    ),
    "ggscatterstats-edge_corr_null-parametric": (
        "Pearson correlation: t(14) = 0.41, p = .685, Pearson's r = .11, 95% CI "
        "[-.41, .57], n = 16 pairs.",
        "Scatter plot of y against x (16 pairs). It also shows a least-squares line "
        "with its 95% confidence band, and marginal histograms of x and y. {apa}",
    ),
    # exactly collinear data: the limits (D85, D86)
    "ggscatterstats-edge_corr_perfect-parametric": (
        "Pearson correlation: t(8) = Inf, p < .001, Pearson's r = 1.00, 95% CI "
        "[1.00, 1.00], n = 10 pairs.",
        "Scatter plot of y against x (10 pairs): every point lies on one rising "
        "straight line. It also shows a least-squares line with its 95% confidence "
        "band, and marginal histograms of x and y. {apa}",
    ),
    "ggscatterstats-edge_corr_perfect-neg-robust": (
        "Winsorized Pearson correlation (20% winsorized): t(8) = -Inf, p < .001, "
        "winsorized r = -1.00, 95% CI [-1.00, -1.00], n = 10 pairs.",
        "Scatter plot of y_neg against x (10 pairs): y_neg tends to fall as x rises. "
        "It also shows a least-squares line with its 95% confidence band, and "
        "marginal histograms of x and y_neg. {apa}",
    ),
    "ggscatterstats-edge_corr_perfect-nonparametric": (
        "Spearman correlation: S = 0.00, p < .001, Spearman's rho = 1.00, 95% CI "
        "[1.00, 1.00], n = 10 pairs.",
        "Scatter plot of y against x (10 pairs): y tends to rise as x rises. It also "
        "shows a least-squares line with its 95% confidence band, and marginal "
        "histograms of x and y. {apa}",
    ),
    "ggscatterstats-edge_corr_perfect-bayes": (
        "Bayesian Pearson correlation (stretched beta prior, r_beta = 1.41): BF10 = "
        "Inf, log(BF01) = -Inf, n = 10 pairs.",
        "Scatter plot of y against x (10 pairs). It also shows a least-squares line "
        "with its 95% confidence band, and marginal histograms of x and y. {apa}",
    ),
}


@pytest.mark.parametrize("name", sorted(EXACT))
def test_exact_strings(name: str) -> None:
    result = analyzed(name)
    assert result is not None
    apa, alt = EXACT[name]
    assert result.text.apa == apa
    assert result.text.alt_text == alt.replace("{apa}", apa)


def _robust_sentence(result: StatsResult) -> str:
    """The robust histostats sentence, its bootstrap numbers from the result (the
    resamples, not the format, depend on NumPy's generator)."""
    test, es = result.test, result.effect_size
    assert test is not None
    assert es is not None
    against = tx.r_number_text(result.settings.test_value or 0.0)
    p = tx._apa_p(test.p_value)  # pyright: ignore[reportPrivateUsage]
    return (
        f"Bootstrap-t test of the 20% trimmed mean against {against}: t = "
        f"{_apa(test.statistic)}, {p}, trimmed mean = {_apa(es.estimate)}, 95% CI "
        f"[{_apa(es.conf_low)}, {_apa(es.conf_high)}], n = {result.n_obs}."
    )


@pytest.mark.parametrize(
    ("name", "t", "estimate", "p", "interval"),
    [
        ("gghistostats-ToothGrowth-robust", "1.00", "19.24", None, None),
        # bootstrap p of 0 (no resample as extreme): "p < .001"
        ("gghistostats-ToothGrowth-tv100-robust", "-64.60", "19.24", "p < .001", None),
        ("gghistostats-mtcars-robust", "18.09", "19.22", "p < .001", None),
        # an unbounded bootstrap-t interval (edge_one_small, as WRS2)
        ("gghistostats-edge_one_small-robust", "1.72", "4.07", None, "[-Inf, Inf]"),
    ],
)
def test_robust_one_sample_sentences(
    name: str, t: str, estimate: str, p: str | None, interval: str | None
) -> None:
    result = analyzed(name)
    assert result is not None
    apa = result.text.apa
    assert apa == _robust_sentence(result)
    assert f": t = {t}, " in apa
    assert f"trimmed mean = {estimate}, " in apa
    if p is not None:
        assert f", {p}, " in apa
    if interval is not None:
        assert f"95% CI {interval}, " in apa
    head = (
        f"Histogram of {result.x} ({result.n_obs} values, from "
        f"{tx.format_value(min(result.values[0]))} to "
        f"{tx.format_value(max(result.values[0]))}). A vertical line marks the 20% "
        f"trimmed mean, {estimate}. "
    )
    assert result.text.alt_text == head + apa


# --------------------------------------------------------------------------- #
# figure_alt_text: what the figure draws
# --------------------------------------------------------------------------- #


def test_figure_alt_text_histostats_every_combination() -> None:
    result = analyzed("gghistostats-ToothGrowth-parametric")
    assert result is not None
    apa = result.text.apa
    assert tx.figure_alt_text(result) == result.text.alt_text
    for centrality, line, bins in itertools.product(
        (True, False), (None, False, True), (None, 1, 12)
    ):
        alt = tx.figure_alt_text(
            result, centrality=centrality, test_value_line=line, bins=bins
        )
        detail = {None: "", 1: ", in 1 bin", 12: ", in 12 bins"}[bins]
        head = f"Histogram of len (60 values, from 4.20 to 33.90{detail})."
        marks = [
            m
            for m, on in (("the mean, 18.81", centrality), ("the test value, 18", line))
            if on
        ]
        if len(marks) == 2:
            middle = " Vertical lines mark the mean, 18.81, and the test value, 18."
        elif marks:
            middle = f" A vertical line marks {marks[0]}."
        else:
            middle = ""
        assert alt == f"{head}{middle} {apa}"
        assert 2 <= _sentences(alt) <= 3


def test_figure_alt_text_histostats_without_a_map_estimate() -> None:
    result = analyzed("gghistostats-ToothGrowth-bayes")
    assert result is not None
    (d,) = result.descriptives
    missing = dataclasses.replace(
        result, descriptives=(dataclasses.replace(d, value=None),)
    )
    alt = tx.figure_alt_text(missing, test_value_line=True)
    assert (
        "A vertical line marks the test value, 18. No MAP estimate could be "
        "computed, so no line marks it. "
    ) in alt
    assert 2 <= _sentences(alt) <= 4


def test_figure_alt_text_scatterstats_every_combination() -> None:
    result = analyzed("ggscatterstats-mtcars-labels-parametric")
    assert result is not None
    apa = result.text.apa
    assert tx.figure_alt_text(result) == result.text.alt_text
    band = "a least-squares line with its 95% confidence band"
    histograms = "marginal histograms of wt and mpg"
    for smooth, marginal, labelled, shown in itertools.product(
        (None, True, False), (None, True, False), (None, 0, 1, 9, 32), (None, 10, 32)
    ):
        alt = tx.figure_alt_text(
            result,
            smooth=smooth,
            marginal=marginal,
            labelled_points=labelled,
            points_shown=shown,
        )
        sample = (
            "" if shown in (None, 32) else "; a random sample of 10 is drawn as points"
        )
        head = (
            f"Scatter plot of mpg against wt (32 pairs{sample}): mpg tends to fall "
            "as wt rises."
        )
        labels = {
            None: "labels from model on every point",
            0: None,
            1: "labels from model on 1 of the 32 points",
            9: "labels from model on 9 of the 32 points",
            32: "labels from model on every point",
        }[labelled]
        parts = [
            p
            for p, on in (
                (band, smooth is not False),
                (histograms, marginal is not False),
                (labels, labels is not None),
            )
            if on and p is not None
        ]
        if not parts:
            middle = ""
        elif len(parts) == 1:
            middle = f" It also shows {parts[0]}."
        else:
            middle = f" It also shows {', '.join(parts[:-1])}, and {parts[-1]}."
        assert alt == f"{head}{middle} {apa}", (smooth, marginal, labelled, shown)
        assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
        assert 2 <= _sentences(alt) <= 3


def test_figure_alt_text_conf_level_names_the_band() -> None:
    result = analyzed("ggscatterstats-mtcars-cl90-parametric")
    assert result is not None
    assert "a least-squares line with its 90% confidence band" in result.text.alt_text


@pytest.mark.parametrize(
    ("name", "keywords", "message"),
    [
        ("gghistostats-ToothGrowth-parametric", {"smooth": True}, "smooth does not"),
        ("gghistostats-ToothGrowth-parametric", {"subject_paths": 3}, "subject_paths"),
        (
            "ggscatterstats-mtcars-parametric",
            {"test_value_line": True, "bins": 3},
            "test_value_line, bins do not describe a scatterstats figure",
        ),
        (
            "ggscatterstats-mtcars-parametric",
            {"centrality": False},
            "centrality does not",
        ),
        (
            "ggscatterstats-mtcars-parametric",
            {"points_shown": -1},
            "a count of 0 or more",
        ),
        ("gghistostats-ToothGrowth-parametric", {"bins": True}, "a count of 0 or more"),
    ],
)
def test_figure_alt_text_refuses_other_designs_keywords(
    name: str, keywords: dict[str, Any], message: str
) -> None:
    result = analyzed(name)
    assert result is not None
    with pytest.raises(ValueError, match=message):
        tx.figure_alt_text(result, **keywords)


def test_figure_alt_text_grouped_designs_refuse_the_new_keywords() -> None:
    plants = read_csv("PlantGrowth.csv")
    between: Any = ip.analyze.betweenstats
    result = between(plants, "group", "weight")
    for keyword in ("bins", "smooth", "labelled_points", "subject_paths"):
        keywords: dict[str, Any] = {keyword: 1}
        with pytest.raises(ValueError, match=f"{keyword} does not describe"):
            tx.figure_alt_text(result, **keywords)


def test_embedded_copy_without_values_names_no_range() -> None:
    result = analyzed("gghistostats-ToothGrowth-parametric")
    assert result is not None
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert embedded.text == result.text  # the stored texts travel unchanged
    assert tx.figure_alt_text(embedded).startswith("Histogram of len (60 values). ")


# --------------------------------------------------------------------------- #
# shortening
# --------------------------------------------------------------------------- #


def test_long_names_are_shortened_never_the_result() -> None:
    frame = read_csv("mtcars.csv")
    long_x, long_y, long_label = "w" * 300, "m" * 300, "l" * 300
    frame = frame.rename(columns={"wt": long_x, "mpg": long_y, "model": long_label})
    scatter: Any = ip.analyze.scatterstats
    result = scatter(frame, long_x, long_y, label_var=long_label)
    alt = result.text.alt_text
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
    assert alt.endswith(result.text.apa)
    assert 2 <= _sentences(alt) <= 3
    histo: Any = ip.analyze.histostats
    one = histo(frame, long_y, test_value=20, alternative="greater")
    assert len(one.text.alt_text) <= tx.ALT_TEXT_MAX_CHARS
    assert one.text.apa.startswith(f"One-sample t-test (one-sided: {'m' * 39}… ")
    assert one.text.alt_text == (
        f"Histogram of {'m' * 39}… (32 values, from 10.40 to 33.90). A vertical "
        f"line marks the mean, 20.09. {one.text.apa}"
    )


def test_too_long_a_result_keeps_the_head_and_the_whole_result() -> None:
    result = analyzed("ggscatterstats-mtcars-parametric")
    assert result is not None
    assert result.test is not None
    huge = dataclasses.replace(result.test, statistic=-1e300)
    texts = tx.build_texts(
        x="w" * 500,
        y="m" * 500,
        descriptives=(),
        test=huge,
        effect_size=result.effect_size,
        bayes=result.bayes,
        pairwise=(),
        n_obs=result.n_obs,
        settings=result.settings,
        design="correlation",
    )
    head = f"Scatter plot of {'m' * 11}… against {'w' * 11}… (32 pairs). "
    assert texts.alt_text == head + texts.apa
    assert len(texts.alt_text) > tx.ALT_TEXT_MAX_CHARS


# --------------------------------------------------------------------------- #
# str(), to_frame() and the messages that name a text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("gghistostats-ToothGrowth-parametric", "One-sample t-test"),
        ("gghistostats-ToothGrowth-nonparametric", "Wilcoxon signed-rank test"),
        (
            "gghistostats-ToothGrowth-robust",
            "Bootstrap-t test of the 20% trimmed mean",
        ),
        ("gghistostats-ToothGrowth-bayes", "Bayesian one-sample t-test"),
        ("ggscatterstats-mtcars-parametric", "Pearson correlation"),
        ("ggscatterstats-mtcars-nonparametric", "Spearman correlation"),
        (
            "ggscatterstats-mtcars-robust",
            "Winsorized Pearson correlation (20% winsorized)",
        ),
        ("ggscatterstats-mtcars-bayes", "Bayesian Pearson correlation"),
    ],
)
def test_str_and_frame_labels(name: str, label: str) -> None:
    result = analyzed(name)
    assert result is not None
    assert str(result) == result.text.apa
    assert result.text.apa.startswith(label)
    frame = result.to_frame()
    assert list(frame["label"]) == [label]


def test_every_template_names_a_test_of_the_new_designs() -> None:
    names = {
        r.test.name
        for c in _answered()
        if (r := analyzed(c.name)) is not None and r.test is not None
    }
    assert names == {
        "one_sample_t",
        "wilcoxon_signed_rank",
        "trimmed_mean_bootstrap_t",
        "pearson_correlation",
        "spearman_correlation",
        "winsorized_correlation",
    }
    for name in names:
        assert name in tx._TEST_TEMPLATES  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("name", _params(_answered()))
def test_warnings_name_only_texts_that_are_drawn(name: str) -> None:
    """A warnings entry about the caption or the Bayes factor subtitle appears only
    where that text is drawn."""
    result = analyzed(name)
    assert result is not None
    for warning in result.warnings:
        if "in the caption" in warning:
            assert result.text.caption is not None, warning
        if "adds to the Bayes factor subtitle" in warning:
            assert result.text.subtitle is not None, warning
            assert result.test is None
            assert result.bayes is not None


def test_the_special_sentences() -> None:
    """Every special case of the sentence, as one table (SPEC section 7)."""
    result = analyzed("ggscatterstats-mtcars-parametric")
    assert result is not None
    beyond = BayesFactor(
        bf10=math.inf,
        log_bf01=-800.0,
        prior="beta",
        prior_scale=1 / 0.707,
        test="pearson_correlation",
    )
    sentence = tx.correlation_sentence(None, None, beyond, n_obs=5000)
    # a BF10 beyond a double prints from log(BF01), as in the other designs
    assert sentence == (
        "Bayesian Pearson correlation (stretched beta prior, r_beta = 1.41): BF10 = "
        "2.73e+347, log(BF01) = -800.00, n = 5,000 pairs."
    )
    assert result.test is not None
    less = dataclasses.replace(result.test, alternative="less")
    with pytest.raises(ValueError, match="a correlation test is two-sided"):
        tx.correlation_sentence(less, result.effect_size, None, n_obs=32)
    one = analyzed("gghistostats-ToothGrowth-parametric")
    assert one is not None
    with pytest.raises(ValueError, match=r"needs settings\.test_value"):
        tx.one_sample_sentence(
            one.test,
            one.effect_size,
            None,
            x="len",
            n_obs=60,
            settings=dataclasses.replace(one.settings, test_value=None),
        )
