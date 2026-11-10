"""The texts of ``barstats`` against every ggbarstats and ggpiestats fixture.

Two layers of checks, as for the other functions (tests/test_no_level_texts.py):

* the builders, on every fixture's own numbers: each subtitle row and drawn
  subtitle (``contingency_subtitle_expression``), each bar's proportion-test row
  and its drawn p label (``bar_p_expression``), the n label under each bar
  (``n_label``), each cell's label (``cell_label``, from count / bar n x 100) and
  each pair's label (``pairwise_p_expression``) must equal ggstatsplot's, the
  expressions as parsed plotmath ASTs (``tests/plotmath.py``), the plain labels
  character for character. This covers all 89 fixtures with output, also the
  ones inferplot refuses (D101: the degenerate tables, whose bars ggstatsplot
  labels with an empty p, and the robust and bayes types);
* the results: each fixture inferplot answers (68, and the 7 robust ones by
  their parametric replay) is assembled from its own numbers
  (``tests/bar_cases.py``; the statistics package joins later), with the
  noncentral chi-square bounds R misses (D99, ``CHISQ_ROOT_MISSES``) replaced by
  the accurate ones inferplot computes, and every text of ``result.text`` and of
  ``figure_texts`` must equal ggstatsplot's or be a pinned, explained difference
  (:data:`DIFFERING_SUBTITLES`).

Every expression is drawn with matplotlib's mathtext and its glyphs must be those
plotmath draws. The percent formula and R's ``round()`` are checked on the 927
labels of the oracle probe ``probe-contingency``. The APA sentences and alt texts
of every answered fixture are checked for form and for numbers equal to the
result's; exact strings are pinned for standard cases of each design;
``figure_alt_text``'s keywords are checked in every combination.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot import text as tx
from inferplot.result import Bar, Cell, EffectSize, StatsResult, TestResult, Texts
from tests import bar_cases as bc
from tests.oracle_fixtures import (
    CHISQ_ROOT_MISSES,
    FIXTURE_DIR,
    OracleCase,
    number,
    read_csv,
)
from tests.plotmath import expected_glyphs, parse
from tests.test_text_render import _normalized, render_glyphs

Row = Mapping[str, Any]
module: Any = bc.bars

# Every case with ggstatsplot output (edge_bar_counts-frac fails in R).
CASES: tuple[OracleCase, ...] = bc.CASES


def _params(names: list[str] | tuple[str, ...]) -> list[Any]:
    return [pytest.param(n, id=n) for n in names]


# --------------------------------------------------------------------------- #
# reading a fixture
# --------------------------------------------------------------------------- #


def _design(case: OracleCase) -> str:
    data = case.contingency_data
    if data["y"] is None:
        return "one-way"
    return "paired" if case.args.get("paired") else "two-way"


def _digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def _label_kind(case: OracleCase) -> str:
    return str(case.args.get("label", "percentage"))


def _digits_perc(case: OracleCase) -> int:
    return int(case.args.get("digits.perc", 0))


def _y_key(case: OracleCase, row: Row) -> str | None:
    y = case.contingency_data["y"]
    return None if y is None else str(row[y])


def _text_layers(case: OracleCase) -> tuple[list[Any], list[Any], list[Any]]:
    """The drawn labels: (cell labels, p labels above the bars, n labels under
    them), each in ggplot2's built order (None for a label not drawn)."""
    cells: list[Any] = []
    p_labels: list[Any] = []
    n_labels: list[Any] = []
    for layer in case.plot_layers:
        if layer["geom"] not in ("GeomLabel", "GeomText"):
            continue
        built = [row["label"] for row in layer["built"] or []]
        if layer["geom"] == "GeomLabel":
            cells += built
        elif layer["geom"] == "GeomText":
            parsed = bool(layer["params"]["geom"].get("parse"))
            (p_labels if parsed else n_labels).extend(built)
    return cells, p_labels, n_labels


# --------------------------------------------------------------------------- #
# the builders, on every fixture's own numbers
# --------------------------------------------------------------------------- #


def _main_row_expression(case: OracleCase, row: Row) -> tx.Node:
    test = TestResult(
        name="placeholder",
        label="placeholder",
        method=str(row["method"]),
        statistic_name="chi2",
        statistic=number(row["statistic"]),
        df=number(row["df"]),
        df_error=None,
        p_value=number(row["p.value"]),
        alternative="two-sided",
    )
    design = _design(case)
    es = EffectSize(
        name=tx._CONTINGENCY_EFFECT_SIZES[design],  # pyright: ignore[reportPrivateUsage]
        label="placeholder",
        estimate=number(row["estimate"]),
        conf_level=number(row["conf.level"]),
        conf_low=number(row["conf.low"]),
        conf_high=number(row["conf.high"]),
        ci_method="placeholder",
        sidedness="two-sided",
    )
    return tx.contingency_subtitle_expression(
        test, es, design=design, n_obs=int(row["n.obs"]), digits=_digits(case)
    )


def _bar_row_expression(case: OracleCase, row: Row) -> tx.Node:
    """A bar's proportion test row as statsExpressions prints it (never drawn by
    ggbarstats, which draws only its p). ggbarstats' ``.chisq_test_safe()``
    calls ``contingency_table()`` without ``digits``, so the row's expression
    has 2 digits whatever ``digits`` is (mtcars-cyl-am-digits3), while the
    drawn p label has ``digits``."""
    return tx.frequentist_expression(
        statistic="gof_chi2",
        statistic_value=number(row["statistic"]),
        p_value=number(row["p.value"]),
        effect_size="pearsons_c",
        estimate=number(row["estimate"]),
        conf_level=number(row["conf.level"]),
        conf_low=number(row["conf.low"]),
        conf_high=number(row["conf.high"]),
        n_obs=int(row["n.obs"]),
        df=number(row["df"]),
        digits=2,
    )


def _bar_totals(case: OracleCase) -> dict[str | None, int]:
    totals: dict[str | None, int] = {}
    for row in case.descriptive:
        key = _y_key(case, row)
        totals[key] = totals.get(key, 0) + int(row["counts"])
    return totals


def _cell_labels_from(case: OracleCase) -> list[tuple[str, str]]:
    """(ggstatsplot's label or "", inferplot's cell_label from the row's count)
    for every row of descriptive_data."""
    totals = _bar_totals(case)
    out: list[tuple[str, str]] = []
    x = case.contingency_data["x"]
    for row in case.descriptive:
        count = int(row["counts"])
        percent = count / totals[_y_key(case, row)] * 100
        mine = tx.cell_label(
            count, percent, label=_label_kind(case), digits_perc=_digits_perc(case)
        )
        theirs = row[".label"]
        assert str(row[x])  # every row names its level of x
        out.append(("" if theirs is None else str(theirs), mine))
    return out


# Bar rows whose statistic the fixture stores to 15 significant digits, which
# loses the side of a rounding tie R's double lay on: (case, bar) -> (R's string,
# the string the formatter gives the stored number). edge_bar_rounding's bar g16
# (counts 1, 5, 10): chi-square is 366 / 48 = 7.625 exactly; R's arithmetic gives
# a double above it (printed 7.63), stored as 7.625 (printed 7.62, ties to even).
# Only the bar's full expression shows it, which ggbarstats never draws (its p
# label is 0.02 either way).
STORED_TIES: Mapping[tuple[str, str | None], tuple[str, str]] = {
    (f"ggbarstats-edge_bar_rounding{variant}-parametric", "g16"): ("7.63", "7.62")
    for variant in ("", "-both", "-counts", "-digitsperc1", "-digitsperc2")
}


@pytest.mark.parametrize("name", _params([c.name for c in CASES]))
def test_fixture_texts_rebuild_from_their_numbers(name: str) -> None:
    case = bc.BY_NAME[name]
    digits = _digits(case)
    # the subtitle: frequentist rows only (the Bayes subtitle is v0.2)
    if case.type != "bayes":
        rows = case.subtitle_rows
        assert len(rows) <= 1
        for row in rows:
            node = _main_row_expression(case, row)
            assert node == parse(str(row["expression"]))
            assert case.subtitle is not None
            assert node == parse(case.subtitle)
        if not rows:
            assert case.subtitle is None
    # each bar: its test's expression, its p label and its n label
    by_bar: dict[str | None, Row] = {}
    for row in case.one_sample:
        source = row.get("expression")
        if source is not None:
            tie = STORED_TIES.get((name, _y_key(case, row)))
            if tie is not None:
                theirs, stored = (f'"{s}"' for s in tie)
                assert str(source).count(theirs) == 1
                source = str(source).replace(theirs, stored)
            assert _bar_row_expression(case, row) == parse(str(source))
        assert tx.bar_p_expression(number(row["p.value"]), digits) == parse(
            str(row[".p.label"])
        )
        assert tx.n_label(int(row["counts"])) == row["N"]
        by_bar[_y_key(case, row)] = row
    # each cell's label
    for theirs, mine in _cell_labels_from(case):
        assert mine == theirs
    # each pair's label (computed, never drawn)
    for row in case.pairwise:
        method = str(case.args.get("p.adjust.method", "holm"))
        assert tx.pairwise_p_expression(
            p_adjusted=number(row["p.value.adj"]), adjust_method=method, digits=digits
        ) == parse(str(row["expression"]))
    # what the plot draws is what the tables hold
    cells, p_labels, n_labels = _text_layers(case)
    assert sorted(c or "" for c in cells) == sorted(
        theirs for theirs, _ in _cell_labels_from(case)
    )
    if p_labels:
        assert sorted(p_labels) == sorted(str(r[".p.label"]) for r in case.one_sample)
    if case.function == "ggbarstats":
        expected_n = (
            [str(r["N"]) for r in case.one_sample]
            if by_bar
            else [tx.n_label(int(case.contingency_data["n"]))]
        )
        assert sorted(n_labels) == sorted(expected_n)


def test_the_fixtures_cover_every_label() -> None:
    """Counts of what the builders reproduce, character for character."""
    subtitles = bar_p = n = cells = pairs = empty_p = 0
    for case in CASES:
        if case.type != "bayes" and case.subtitle is not None:
            subtitles += 1
        bar_p += len(case.one_sample)
        empty_p += sum(1 for r in case.one_sample if r["p.value"] is None)
        n += len(case.one_sample) or (case.function == "ggbarstats")
        cells += sum(1 for theirs, _ in _cell_labels_from(case) if theirs)
        pairs += len(case.pairwise)
    assert len(CASES) == 89
    assert (subtitles, bar_p, n, cells, pairs) == (71, 158, 175, 485, 152)
    # A bar whose test fails (one level of x: refused by inferplot, D101) has an
    # empty p in ggstatsplot (format_value(NA) is ""), not "NA".
    assert empty_p == 3


def test_percent_formula_and_rounding_on_the_probe() -> None:
    """probe-contingency's 927 labels (k of n at digits 0 to 2): the percent is
    (k / n) * 100 bit for bit, R's round() gives Python's round() on that double,
    and the label is the rounded double as R prints it. 100 * k / n would give
    another double in 81 of them and another label in 3, the ties 57.5 (23 of
    40, 115 of 200) and 54.5 (109 of 200), which R's double lies below or above."""
    probe = json.loads((FIXTURE_DIR / "probe-contingency.json").read_text())
    rows = probe["percent_labels"]
    assert len(rows) == 927
    other_double = other_label = 0
    for row in rows:
        k, n, digits = int(row["k"]), int(row["n"]), int(row["digits"])
        percent = k / n * 100
        assert percent == float.fromhex(row["perc_hex"])
        assert round(percent, digits) == float.fromhex(row["rounded_hex"])
        assert tx.percent_text(percent, digits) + "%" == row["label"]
        assert (
            tx.cell_label(k, percent, label="percentage", digits_perc=digits)
            == (row["label"])
        )
        other = 100 * k / n
        other_double += other != percent
        other_label += tx.percent_text(other, digits) + "%" != row["label"]
    assert (other_double, other_label) == (81, 3)


def test_cell_label_forms() -> None:
    assert tx.cell_label(0, 0.0, label="percentage", digits_perc=0) == ""
    assert tx.cell_label(1, 12.5, label="percentage", digits_perc=0) == "12%"
    assert tx.cell_label(1, 6.25, label="percentage", digits_perc=1) == "6.2%"
    assert tx.cell_label(1, 50.0, label="percentage", digits_perc=2) == "50%"
    assert tx.cell_label(1490, 67.7, label="counts", digits_perc=0) == "1,490"
    assert tx.cell_label(1490, 67.7, label="both", digits_perc=1) == "1,490\n(67.7%)"
    assert tx.n_label(1490) == "(n = 1,490)"
    with pytest.raises(ValueError, match="unknown cell label"):
        tx.cell_label(1, 1.0, label="count", digits_perc=0)
    # an exact p of 0 prints as ggstatsplot stores it (.Machine$double.xmin), a
    # missing one as ""
    assert tx.to_plain(tx.bar_p_expression(0.0)) == "p = 2.23e-308"
    assert tx.to_plain(tx.bar_p_expression(None)) == "p = "
    assert tx.to_mathtext(tx.bar_p_expression(0.0912, 3)) == (
        "$\\ \\mathit{p} = \\text{0.091}$"
    )


# --------------------------------------------------------------------------- #
# inferplot's results against ggstatsplot's texts
# --------------------------------------------------------------------------- #

_REPLAYED = tuple(
    c
    for c in CASES
    if c.name in bc.REFUSALS
    and bc.REFUSALS[c.name].replay is not None
    and c.type == "robust"
)
# Two fixtures bar_cases can assemble that the statistics refuse (D101), as
# ggstatsplot prints no subtitle for them: no discordant pair (McNemar's 0/0),
# and a pair of levels of x whose table keeps one column (Fisher's test stops and
# ggstatsplot loses every pair). Their labels are checked on their own numbers
# above.
REFUSED_BY_STATISTICS = (
    "ggbarstats-edge_bar_paired_agree-parametric",
    "ggbarstats-edge_bar_degenerate-single-x-parametric",
)
ANSWERED_NAMES: tuple[str, ...] = tuple(
    c.name for c in (*bc.ANSWERED, *_REPLAYED) if c.name not in REFUSED_BY_STATISTICS
)


def _accurate(es: EffectSize | None) -> EffectSize | None:
    """The interval inferplot reports (D99): a bound where effectsize's root
    search stops short (``CHISQ_ROOT_MISSES``) replaced by the accurate root."""
    if es is None:
        return None
    misses = dict(CHISQ_ROOT_MISSES.values())

    def fixed(bound: float) -> float:
        for r_bound, accurate in misses.items():
            if math.isclose(bound, r_bound, rel_tol=1e-13):
                return accurate
        return bound

    return dataclasses.replace(
        es, conf_low=fixed(es.conf_low), conf_high=fixed(es.conf_high)
    )


@functools.cache
def result_for(name: str) -> StatsResult:
    """The fixture's result from its own numbers (the robust ones by their
    parametric replay), with D99's accurate bounds, and its texts rebuilt."""
    case = bc.BY_NAME[name]
    refusal = bc.REFUSALS.get(name)
    changes = None if refusal is None else refusal.replay
    table, args, digits = bc.prepared(case, "pandas", changes)
    parts = bc.parts(case, table, args)
    parts = dataclasses.replace(parts, effect_size=_accurate(parts.effect_size))
    return module._assemble(table, args, parts, digits=digits)  # pyright: ignore[reportPrivateUsage]


_D99 = (
    "D99: R's upper bound of Pearson's C is 0.498657 where the accurate root of "
    "the noncentral chi-square equation is 0.498397 (CHISQ_ROOT_MISSES, "
    "mtcars-cyl C); at 3 digits ggstatsplot prints 0.499 and inferplot 0.498"
)
# case -> (ggstatsplot's subtitle, inferplot's, why), as plain text.
DIFFERING_SUBTITLES: Mapping[str, tuple[str, str, str]] = {
    "ggbarstats-mtcars-cyl-digits3-parametric": (
        "χ_gof^2(2) = 2.312, p = 0.315, C_Pearson = 0.260, "
        "CI_95% [0.000, 0.499], n_obs = 32",
        "χ_gof^2(2) = 2.312, p = 0.315, C_Pearson = 0.260, "
        "CI_95% [0.000, 0.498], n_obs = 32",
        _D99,
    ),
}


def _own_subtitle(result: StatsResult) -> tx.Node | None:
    if result.test is None or result.effect_size is None:
        return None
    return tx.contingency_subtitle_expression(
        result.test,
        result.effect_size,
        design=tx._contingency_design(result.y, result.settings),  # pyright: ignore[reportPrivateUsage]
        n_obs=result.n_obs,
        digits=result.settings.digits,
    )


def test_every_fixture_is_covered() -> None:
    assert len(bc.ANSWERED) == 68
    assert len(_REPLAYED) == 7
    assert len(ANSWERED_NAMES) == 73
    agree, single_x = (bc.BY_NAME[name] for name in REFUSED_BY_STATISTICS)
    assert agree.subtitle is None
    assert single_x.subtitle is not None
    assert single_x.pairwise == []


def _subtitle_difference(name: str) -> tuple[str, str] | None:
    case, result = bc.BY_NAME[name], result_for(name)
    node = _own_subtitle(result)
    if case.subtitle is None:
        assert node is None
        return None
    assert node is not None
    theirs = parse(case.subtitle)
    if node == theirs:
        return None
    return (tx.to_plain(theirs), tx.to_plain(node))


@pytest.mark.parametrize("name", _params(ANSWERED_NAMES))
def test_subtitle_and_caption(name: str) -> None:
    case, result = bc.BY_NAME[name], result_for(name)
    figure = tx.figure_texts(result)
    # no caption in v0.1 (ggstatsplot's Bayes caption arrives in 0.2)
    assert result.text.caption is None
    assert figure.caption_lines == ()
    node = _own_subtitle(result)
    if node is None:
        assert result.text.subtitle is None
        assert figure.subtitle is None
        assert case.subtitle is None
        assert not result.settings.results_subtitle
        return
    subtitle = result.text.subtitle
    assert subtitle is not None
    assert subtitle == tx.to_plain(node)
    assert figure.subtitle == tx.to_mathtext(node)
    n = "n_pairs = " if result.settings.paired else "n_obs = "
    assert n + tx.format_n(result.n_obs) in subtitle
    difference = _subtitle_difference(name)
    if difference is None:
        assert case.subtitle is not None
        assert node == parse(case.subtitle)
    else:
        assert difference == DIFFERING_SUBTITLES[name][:2]


def test_differing_subtitles_are_pinned() -> None:
    differing = {
        name: difference
        for name in ANSWERED_NAMES
        if (difference := _subtitle_difference(name)) is not None
    }
    assert differing == {k: v[:2] for k, v in DIFFERING_SUBTITLES.items()}


def test_how_many_texts_equal_ggstatsplots() -> None:
    """The counts the report gives, as (equal character for character, of): the
    subtitles, the p labels above the bars, the n labels under them (ggbarstats;
    ggpiestats draws none) and the labels inside the cells (zero cells, which
    have none, left out) of the answered fixtures."""
    subtitles = [0, 0]
    bar_labels = [0, 0]
    n_labels = [0, 0]
    cells = [0, 0]
    for name in ANSWERED_NAMES:
        case, result = bc.BY_NAME[name], result_for(name)
        texts = result.text
        if case.subtitle is not None:
            subtitles[1] += 1
            subtitles[0] += _subtitle_difference(name) is None
        rows = {_y_key(case, r): r for r in case.one_sample}
        for bar, label, n in zip(
            result.bars, texts.bar_labels, texts.n_labels, strict=True
        ):
            if bar.level in rows:
                row = rows[bar.level]
                bar_labels[1] += 1
                bar_labels[0] += label == tx.to_plain(parse(str(row[".p.label"])))
                n_labels[1] += 1
                n_labels[0] += n == row["N"]
            elif case.function == "ggbarstats":
                n_labels[1] += 1
                n_labels[0] += n == _text_layers(case)[2][0]
        x = case.contingency_data["x"]
        theirs = {
            (_y_key(case, r), str(r[x])): r[".label"] or "" for r in case.descriptive
        }
        for cell, label in zip(result.cells, texts.cell_labels, strict=True):
            if cell.count:
                cells[1] += 1
                cells[0] += label == theirs[(cell.bar, cell.level)]
    assert (subtitles, bar_labels, n_labels, cells) == (
        [69, 70],
        [136, 136],
        [149, 149],
        [428, 428],
    )


@pytest.mark.parametrize("name", _params(ANSWERED_NAMES))
def test_bar_n_cell_and_pair_labels(name: str) -> None:
    case, result = bc.BY_NAME[name], result_for(name)
    texts, figure = result.text, tx.figure_texts(result)
    digits = result.settings.digits
    rows = {_y_key(case, r): r for r in case.one_sample}
    assert len(texts.bar_labels) == len(texts.n_labels) == len(result.bars)
    for k, bar in enumerate(result.bars):
        if bar.level is None:  # the single bar without y: the main test is its test
            assert texts.bar_labels[k] == figure.bar_labels[k] == ""
            if case.function == "ggbarstats":
                assert texts.n_labels[k] == _text_layers(case)[2][0]
            continue
        row = rows[bar.level]
        theirs = parse(str(row[".p.label"]))
        assert bar.test is not None
        assert tx.bar_p_expression(bar.test.p_value, digits) == theirs
        assert texts.bar_labels[k] == tx.to_plain(theirs)
        assert figure.bar_labels[k] == tx.to_mathtext(theirs)
        assert texts.n_labels[k] == figure.n_labels[k] == row["N"]
    # the p labels are drawn exactly where ggstatsplot draws them
    drawn = bool(_text_layers(case)[1])
    assert bool(result.settings.proportion_test) == drawn
    # cells: one label per cell, ggstatsplot's ("" for a zero cell)
    theirs_cells = {
        (_y_key(case, r), str(r[case.contingency_data["x"]])): r[".label"] or ""
        for r in case.descriptive
    }
    assert texts.cell_labels == figure.cell_labels
    assert [theirs_cells[(c.bar, c.level)] for c in result.cells] == list(
        texts.cell_labels
    )
    # pairs: ggstatsplot's labels (never drawn)
    assert len(texts.pairwise_labels) == len(result.pairwise) == len(case.pairwise)
    for label, row in zip(texts.pairwise_labels, case.pairwise, strict=True):
        assert label == tx.to_plain(parse(str(row["expression"])))
    # titles and ticks
    plot = case.plot_texts
    assert figure.legend_title == plot["legend_title"] == result.x
    assert figure.x_title == plot["labels"].get("x") == result.y
    assert plot["labels"].get("y") is None
    assert figure.y_tick_labels == tuple(f"{p}%" for p in range(0, 101, 10))
    ticks = tuple("" if b.level is None else b.level for b in result.bars)
    assert texts.tick_labels == figure.tick_labels == ticks
    assert texts.centrality_labels == figure.centrality_labels == ()


# --------------------------------------------------------------------------- #
# mathtext
# --------------------------------------------------------------------------- #


@functools.cache
def _render_items() -> tuple[
    list[tuple[str, str, bool]], dict[str, str], dict[str, str]
]:
    """Every distinct barstats expression of the fixtures (ggstatsplot's,
    parsed: subtitles, bar p labels, pair labels) and every one inferplot draws."""
    sources: dict[str, tx.Node] = {}
    for case in CASES:
        if case.type != "bayes" and case.subtitle is not None:
            sources.setdefault(case.subtitle, parse(case.subtitle))
        for row in case.one_sample:
            source = str(row[".p.label"])
            sources.setdefault(source, parse(source))
        for row in case.pairwise:
            source = str(row["expression"])
            sources.setdefault(source, parse(source))
    for name in ANSWERED_NAMES:
        result = result_for(name)
        node = _own_subtitle(result)
        if node is not None:
            sources.setdefault(f"[inferplot] {tx.to_plain(node)}", node)
    items: list[tuple[str, str, bool]] = []
    expected: dict[str, str] = {}
    where: dict[str, str] = {}
    for k, (source, node) in enumerate(sorted(sources.items())):
        gid = f"b{k}"
        items.append((gid, tx.to_mathtext(node), True))
        expected[gid] = expected_glyphs(node)
        where[gid] = source
    return items, expected, where


_PAGE = 50


def test_every_expression_is_rendered() -> None:
    items, _, where = _render_items()
    assert 100 < len(items) <= 200
    assert any(w.startswith("[inferplot]") for w in where.values())
    assert any(w.startswith("list(~italic(p)") for w in where.values())


@pytest.mark.parametrize("start", range(0, 200, _PAGE))
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


# --------------------------------------------------------------------------- #
# the APA sentence and the alt text on every answered fixture
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_BAD_SPECIAL = re.compile(r"\b(NA|NaN|nan|inf|Infinity|infinity|None)\b")


def _sentences(text: str) -> int:
    return len(_SENTENCE_END.findall(text))


def _apa(x: float, bounded: bool = True) -> str:
    return tx._apa_number(x, bounded=bounded)  # pyright: ignore[reportPrivateUsage]


def _main_numbers(result: StatsResult) -> str:
    """The test and effect-size part every main sentence must hold."""
    test, es = result.test, result.effect_size
    assert test is not None
    assert es is not None
    df = test.df_error if test.df_error is not None else test.df
    assert df is not None
    sided = "" if es.sidedness == "two-sided" else "one-sided "
    return (
        f"{test.statistic_name}({int(df)}, N = {tx.format_n(result.n_obs)}) = "
        f"{_apa(test.statistic, False)}, "
        f"{tx._apa_p(test.p_value)}, {es.label} = {_apa(es.estimate)}, "  # pyright: ignore[reportPrivateUsage]
        f"{tx.format_conf_level(es.conf_level)} {sided}CI [{_apa(es.conf_low)}, "
        f"{_apa(es.conf_high)}]."
    )


@pytest.mark.parametrize("name", _params(ANSWERED_NAMES))
def test_apa_sentence_and_alt_text_form(name: str) -> None:
    result = result_for(name)
    apa, alt = result.text.apa, result.text.alt_text
    paired = bool(result.settings.paired)
    for text in (apa, alt):
        assert not _BAD_SPECIAL.search(text), text
        assert text.isascii()  # the fixtures' names are ASCII
    test = result.test
    if test is not None:
        main = f"{test.label} ("
        assert apa.startswith(main)
        assert _main_numbers(result) in apa
    else:
        n = f"n = {tx.format_n(result.n_obs)}{' pairs' if paired else ''}."
        assert apa.startswith(
            tx._CONTINGENCY_NO_TEST[  # pyright: ignore[reportPrivateUsage]
                tx._contingency_design(result.y, result.settings)  # pyright: ignore[reportPrivateUsage]
            ]
            + f" (results_subtitle=False), {n}"
        )
    # the second sentence: the bars' tests (with y) and the pairs
    tested = [bar for bar in result.bars if bar.test is not None]
    assert _sentences(apa) == 1 + bool(tested or result.pairwise)
    if tested:
        below = sum(1 for b in tested if b.test is not None and b.test.p_value < 0.05)
        assert f": p < .05 in {below} of {len(tested)} bar" in apa
    if result.pairwise:
        below = sum(1 for pair in result.pairwise if pair.significant)
        pairs = tx._plural(len(result.pairwise), "pair")  # pyright: ignore[reportPrivateUsage]
        assert f"p < .05 in {below} of {pairs}." in apa
        assert "Fisher's exact tests between the levels of " in apa
    assert str(result).startswith(apa)
    # the alt text: 2 or 3 sentences, at most 400 characters, the chart first,
    # the compact main sentence last
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
    assert 2 <= _sentences(alt) <= 3, alt
    assert alt.startswith("Stacked bar chart of ")
    assert result.x in alt
    compact = tx.contingency_sentence(
        result.test,
        result.effect_size,
        x=result.x,
        y=result.y,
        n_obs=result.n_obs,
        settings=result.settings,
        compact=True,
    )
    assert alt.endswith(" " + compact)
    for bar in result.bars:
        assert tx.n_label(bar.n) in alt
    assert alt == tx.figure_alt_text(result)


# --------------------------------------------------------------------------- #
# exact strings
# --------------------------------------------------------------------------- #

_CYL_AM = (
    "Stacked bar chart of cyl within each level of am, each bar summing to 100%: "
    "0 (n = 19): 4 16%, 6 21%, 8 63%; 1 (n = 13): 4 62%, 6 23%, 8 15%."
)
_CYL = (
    "Stacked bar chart of the shares of cyl in one bar (n = 32): 4 34%, 6 22%, 8 44%."
)
_PAIRED = (
    "Stacked bar chart of before within each level of after (40 units measured "
    "twice), each bar summing to 100%: no (n = 26): no 54%, yes 46%; yes (n = 14): "
    "no 29%, yes 71%."
)
_BARS_CYL_AM = (
    "Goodness-of-fit tests within each level of am (shares of cyl against equal "
    "shares, unadjusted): p < .05 in 1 of 2 bars; Fisher's exact tests between the "
    "levels of cyl (Holm adjusted): adjusted p < .05 in 1 of 3 pairs."
)

EXACT: Mapping[str, tuple[str, str]] = {
    "ggbarstats-mtcars-cyl-am-parametric": (
        "Pearson's chi-square test (shares of cyl across the levels of am): "
        f"chi2(2, N = 32) = 8.74, p = .013, Cramer's V = .46, 95% CI [.00, .82]. "
        f"{_BARS_CYL_AM}",
        f"{_CYL_AM} Labels show percentages; the p-value above each bar tests its "
        "shares against equal shares (0: p = 0.02, 1: p = 0.09); a legend names the "
        "3 levels of cyl. Pearson's chi-square test: chi2(2, N = 32) = 8.74, p = "
        ".013, Cramer's V = .46, 95% CI [.00, .82].",
    ),
    "ggbarstats-mtcars-cyl-am-greater-parametric": (
        "Pearson's chi-square test (shares of cyl across the levels of am): "
        "chi2(2, N = 32) = 8.74, p = .013, Cramer's V = .46, 95% one-sided CI [.00, "
        "1.00]. Goodness-of-fit tests within each level of am (shares of cyl against "
        "equal shares, unadjusted): p < .05 in 1 of 2 bars; Fisher's exact tests "
        "between the levels of cyl (Holm adjusted): adjusted p < .05 in 0 of 3 "
        "pairs.",
        f"{_CYL_AM} Labels show percentages; the p-value above each bar tests its "
        "shares against equal shares (0: p = 0.02, 1: p = 0.09). Pearson's "
        "chi-square test: chi2(2, N = 32) = 8.74, p = .013, Cramer's V = .46, 95% "
        "one-sided CI [.00, 1.00].",
    ),
    "ggbarstats-mtcars-cyl-am-nosubtitle-parametric": (
        "No test of independence was computed (results_subtitle=False), n = 32. "
        "Goodness-of-fit tests within each level of am (shares of cyl against equal "
        "shares, unadjusted): p < .05 in 1 of 2 bars.",
        f"{_CYL_AM} Labels show percentages; a legend names the 3 levels of cyl. No "
        "test of independence was computed (results_subtitle=False), n = 32.",
    ),
    "ggbarstats-mtcars-cyl-am-counts-parametric": (
        "Pearson's chi-square test (shares of cyl across the levels of am): "
        f"chi2(2, N = 32) = 8.74, p = .013, Cramer's V = .46, 95% CI [.00, .82]. "
        f"{_BARS_CYL_AM}",
        f"{_CYL_AM} Labels show counts; the p-value above each bar tests its shares "
        "against equal shares (0: p = 0.02, 1: p = 0.09); a legend names the 3 "
        "levels of cyl. Pearson's chi-square test: chi2(2, N = 32) = 8.74, p = "
        ".013, Cramer's V = .46, 95% CI [.00, .82].",
    ),
    "ggbarstats-mtcars-cyl-am-ratio-parametric": (
        "Pearson's chi-square test (shares of cyl across the levels of am): "
        "chi2(2, N = 32) = 8.74, p = .013, Cramer's V = .46, 95% CI [.00, .82]. "
        "Goodness-of-fit tests within each level of am (shares of cyl against "
        "expected shares 0.25, 0.25 and 0.5, unadjusted): p < .05 in 1 of 2 bars; "
        "Fisher's exact tests between the levels of cyl (Holm adjusted): adjusted "
        "p < .05 in 1 of 3 pairs.",
        f"{_CYL_AM} Labels show percentages; the p-value above each bar tests its "
        "shares against expected shares 0.25, 0.25 and 0.5 (0: p = 0.49, 1: p = "
        "6.48e-03). Pearson's chi-square test: chi2(2, N = 32) = 8.74, p = .013, "
        "Cramer's V = .46, 95% CI [.00, .82].",
    ),
    "ggbarstats-mtcars-cyl-parametric": (
        "Chi-square goodness-of-fit test (shares of cyl against equal shares): "
        "chi2(2, N = 32) = 2.31, p = .315, Pearson's C = .26, 95% CI [.00, .50].",
        f"{_CYL} Labels show percentages; a legend names the 3 levels of cyl. "
        "Chi-square goodness-of-fit test (against equal shares): chi2(2, N = 32) = "
        "2.31, p = .315, Pearson's C = .26, 95% CI [.00, .50].",
    ),
    "ggbarstats-mtcars-cyl-ratio-parametric": (
        "Chi-square goodness-of-fit test (shares of cyl against expected shares "
        "0.5, 0.25 and 0.25): chi2(2, N = 32) = 6.19, p = .045, Pearson's C = .40, "
        "95% CI [.00, .60].",
        f"{_CYL} Labels show percentages; a legend names the 3 levels of cyl. "
        "Chi-square goodness-of-fit test (against expected shares 0.5, 0.25 and "
        "0.25): chi2(2, N = 32) = 6.19, p = .045, Pearson's C = .40, 95% CI [.00, "
        ".60].",
    ),
    "ggbarstats-mtcars-cyl-nosubtitle-parametric": (
        "No goodness-of-fit test was computed (results_subtitle=False), n = 32.",
        f"{_CYL} Labels show percentages; a legend names the 3 levels of cyl. "
        "No goodness-of-fit test was computed (results_subtitle=False), n = 32.",
    ),
    "ggbarstats-edge_bar_paired-parametric": (
        "McNemar's test (before and after, the same units measured twice): "
        "chi2(1, N = 40) = 4.00, p = .046, Cohen's g = .25, 95% CI [.01, .40]. "
        "Goodness-of-fit tests within each level of after (shares of before "
        "against equal shares, unadjusted): p < .05 in 0 of 2 bars.",
        f"{_PAIRED} Labels show percentages; the p-value above each bar tests its "
        "shares against equal shares (no: p = 0.69, yes: p = 0.11). McNemar's "
        "test: chi2(1, N = 40) = 4.00, p = .046, Cohen's g = .25, 95% CI [.01, .40].",
    ),
    "ggbarstats-edge_bar_paired-less-parametric": (
        "McNemar's test (before and after, the same units measured twice): "
        "chi2(1, N = 40) = 4.00, p = .046, Cohen's g = .25, 95% one-sided CI [-.50, "
        ".38]. Goodness-of-fit tests within each level of after (shares of before "
        "against equal shares, unadjusted): p < .05 in 0 of 2 bars.",
        f"{_PAIRED} Labels show percentages; the p-value above each bar tests its "
        "shares against equal shares (no: p = 0.69, yes: p = 0.11). McNemar's "
        "test: chi2(1, N = 40) = 4.00, p = .046, Cohen's g = .25, 95% one-sided CI "
        "[-.50, .38].",
    ),
    "ggbarstats-Titanic_full-class-padj-none-parametric": (
        "Pearson's chi-square test (shares of Class across the levels of "
        "Survived): chi2(3, N = 2,201) = 190.40, p < .001, Cramer's V = .29, 95% CI "
        "[.25, .33]. Goodness-of-fit tests within each level of Survived (shares of "
        "Class against equal shares, unadjusted): p < .05 in 2 of 2 bars; Fisher's "
        "exact tests between the levels of Class (unadjusted): p < .05 in 5 of 6 "
        "pairs.",
        "Stacked bar chart of Class within each level of Survived, each bar summing "
        "to 100%: No (n = 1,490): 1st 8%, 2nd 11%, 3rd 35%, Crew 45%; Yes (n = "
        "711): 1st 29%, 2nd 17%, 3rd 25%, Crew 30%. Labels show percentages; the "
        "p-value above each bar tests its shares against equal shares. Pearson's "
        "chi-square test: chi2(3, N = 2,201) = 190.40, p < .001, Cramer's V = .29, "
        "95% CI [.25, .33].",
    ),
}


@pytest.mark.parametrize("name", sorted(EXACT))
def test_exact_strings(name: str) -> None:
    result = result_for(name)
    apa, alt = EXACT[name]
    assert result.text.apa == apa
    assert result.text.alt_text == alt


def test_shortening_keeps_the_largest_share() -> None:
    """Four bars of three shares do not fit with every share and the p list:
    each bar keeps its largest share (ties named together)."""
    result = result_for("ggbarstats-edge_bar_rounding-both-parametric")
    assert result.text.alt_text.startswith(
        "Stacked bar chart of x within each level of y, each bar summing to 100%: "
        "g16 (n = 16): largest c 62%; g3 (n = 3): largest a, b and c 33% each; g7 "
        "(n = 7): largest c 57%; g8 (n = 8): largest c 50%. Labels show counts and "
        "percentages; the p-value above each bar tests its shares against equal "
        "shares. "
    )


# --------------------------------------------------------------------------- #
# figure_alt_text: what the figure draws
# --------------------------------------------------------------------------- #


def test_figure_alt_text_every_combination() -> None:
    result = result_for("ggbarstats-mtcars-am-vs-parametric")
    assert tx.figure_alt_text(result) == result.text.alt_text
    head = (
        "Stacked bar chart of am within each level of vs, each bar summing to 100%: "
        "0 (n = 18): 0 67%, 1 33%; 1 (n = 14): 0 50%, 1 50%."
    )
    end = (
        "Pearson's chi-square test: chi2(1, N = 32) = 0.91, p = .341, Cramer's V = "
        ".00, 95% CI [.00, .49]."
    )
    for cells, bars, legend in itertools.product(
        (None, True, False), (None, True, False), (None, True, False)
    ):
        alt = tx.figure_alt_text(
            result, cell_labels=cells, bar_tests=bars, legend=legend
        )
        parts = [
            p
            for p, on in (
                ("labels show percentages", cells is not False),
                (
                    "the p-value above each bar tests its shares against equal shares "
                    "(0: p = 0.16, 1: p = 1.00)",
                    bars is not False,
                ),
                ("a legend names the 2 levels of am", legend is not False),
            )
            if on
        ]
        middle = " " + "; ".join(parts) + "." if parts else ""
        middle = middle[:2].upper() + middle[2:]
        expected = f"{head}{middle} {end}"
        if len(expected) > tx.ALT_TEXT_MAX_CHARS:
            # the legend clause is the first to go
            parts.pop()
            middle = " " + "; ".join(parts) + "."
            middle = middle[:2].upper() + middle[2:]
            expected = f"{head}{middle} {end}"
        assert alt == expected, (cells, bars, legend)
        assert 2 <= _sentences(alt) <= 3


def test_figure_alt_text_without_proportion_tests() -> None:
    result = result_for("ggbarstats-mtcars-cyl-am-noproptest-parametric")
    assert "p-value" not in result.text.alt_text
    drawn = tx.figure_alt_text(result, bar_tests=True, legend=False)
    assert "(0: p = 0.02, 1: p = 0.09). Pearson's" in drawn


@pytest.mark.parametrize(
    ("name", "keywords", "message"),
    [
        (
            "ggbarstats-mtcars-cyl-parametric",
            {"bar_tests": True},
            "without y the single bar has no test of its own",
        ),
        (
            "ggbarstats-mtcars-cyl-am-parametric",
            {"subject_paths": 3},
            "subject_paths does not describe a barstats figure",
        ),
        (
            "ggbarstats-mtcars-cyl-am-parametric",
            {"smooth": True, "bins": 3},
            "bins, smooth do not describe a barstats figure",
        ),
        (
            "ggbarstats-mtcars-cyl-am-parametric",
            {"centrality": False},
            "centrality does not describe",
        ),
    ],
)
def test_figure_alt_text_refuses_other_keywords(
    name: str, keywords: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tx.figure_alt_text(result_for(name), **keywords)


def test_other_designs_refuse_the_barstats_keywords() -> None:
    plants = read_csv("PlantGrowth.csv")
    between: Any = ip.analyze.betweenstats
    result = between(plants, "group", "weight")
    for keyword in ("cell_labels", "bar_tests", "legend"):
        keywords: dict[str, Any] = {keyword: True}
        with pytest.raises(ValueError, match=f"{keyword} does not describe"):
            tx.figure_alt_text(result, **keywords)


# --------------------------------------------------------------------------- #
# shortening
# --------------------------------------------------------------------------- #


def test_long_names_are_shortened_never_the_result() -> None:
    long_x, long_y = "x" * 300, "y" * 300
    levels = [f"{'level ' * 10}{k}" for k in range(4)]
    frame = pd.DataFrame(
        {
            long_x: [levels[k % 4] for k in range(40)],
            long_y: [
                ["first " * 9 + "a", "first " * 9 + "b"][k % 3 % 2] for k in range(40)
            ],
        }
    )
    result = _stand_in_result(frame, long_x, long_y)
    alt = result.text.alt_text
    assert len(alt) <= tx.ALT_TEXT_MAX_CHARS
    assert 2 <= _sentences(alt) <= 3
    assert "…" in alt  # names shortened with an ellipsis
    assert alt.endswith(
        tx.contingency_sentence(
            result.test,
            result.effect_size,
            x=long_x,
            y=long_y,
            n_obs=40,
            settings=result.settings,
            name_limit=12,
            compact=True,
        )
    )


def test_too_long_a_result_keeps_the_head_and_the_whole_result() -> None:
    result = result_for("ggbarstats-mtcars-cyl-am-parametric")
    assert result.test is not None
    huge = dataclasses.replace(result.test, label="L" * 500)
    alt = tx.contingency_alt_text(
        x=result.x,
        y=result.y,
        levels=result.levels,
        test=huge,
        effect_size=result.effect_size,
        bars=result.bars,
        cells=result.cells,
        n_obs=result.n_obs,
        settings=result.settings,
    )
    assert alt.startswith(
        "Stacked bar chart of cyl within each level of am, each bar summing to "
        f"100%. {'L' * 500}: chi2(2, N = 32)"
    )
    assert len(alt) > tx.ALT_TEXT_MAX_CHARS


def _stand_in_result(frame: pd.DataFrame, x: str, y: str) -> StatsResult:
    """A two-way result from stand-in numbers (the statistics join later)."""
    stand_in = bc.BY_NAME["ggbarstats-mtcars-cyl-am-nosubtitle-parametric"]
    table, args, digits = module._prepare(frame, x, y)  # pyright: ignore[reportPrivateUsage]
    parts = bc.parts(stand_in, table, args)
    return module._assemble(table, args, parts, digits=digits)  # pyright: ignore[reportPrivateUsage]


# --------------------------------------------------------------------------- #
# str(), to_frame(), the contract
# --------------------------------------------------------------------------- #


def test_str_and_frame_labels() -> None:
    result = result_for("ggbarstats-mtcars-cyl-am-parametric")
    printed = str(result)
    assert printed.startswith(result.text.apa + "\n\nPairwise comparisons (Fisher's ")
    frame = result.to_frame()
    assert list(frame["label"]) == [
        "Pearson's chi-square test",
        "Chi-square goodness-of-fit test",
        "Chi-square goodness-of-fit test",
        "Fisher's exact test",
        "Fisher's exact test",
        "Fisher's exact test",
    ]
    for name in ("pearson_chi_square", "mcnemar", "chi_square_gof"):
        assert name in tx._TEST_TEMPLATES  # pyright: ignore[reportPrivateUsage]
    test, es = result.test, result.effect_size
    assert test is not None
    assert es is not None
    # subtitle_expression (the other designs' route) agrees for the new names
    assert tx.subtitle_expression(test, es, n_obs=32, digits=2) == _own_subtitle(result)


def test_texts_travel_with_the_result() -> None:
    result = result_for("ggbarstats-edge_bar_rounding-both-parametric")
    copy = StatsResult.from_json(result.to_json(include_values=False))
    assert copy.text == result.text
    assert copy.text.cell_labels[0] == "1\n(6%)"


def test_the_contract_refuses_texts_that_do_not_fit() -> None:
    result = result_for("ggbarstats-mtcars-cyl-am-parametric")
    short = dataclasses.replace(result.text, n_labels=result.text.n_labels[:1])
    with pytest.raises(ValueError, match="one label per bar"):
        dataclasses.replace(result, text=short)
    cells = dataclasses.replace(result.text, cell_labels=())
    with pytest.raises(ValueError, match="one label per cell"):
        dataclasses.replace(result, text=cells)
    one = result_for("ggbarstats-mtcars-cyl-parametric")
    plain = Texts(
        subtitle=None,
        caption=None,
        apa="x",
        alt_text="x",
        centrality_labels=(),
        tick_labels=(),
        pairwise_labels=(),
        n_labels=("(n = 1)",),
    )
    assert one.text.n_labels == ("(n = 32)",)
    between: Any = ip.analyze.betweenstats
    other = between(read_csv("PlantGrowth.csv"), "group", "weight")
    with pytest.raises(ValueError, match=r"text\.n_labels belong to design"):
        dataclasses.replace(other, text=plain)


def test_unknown_designs_and_effect_sizes_are_refused() -> None:
    result = result_for("ggbarstats-mtcars-cyl-am-parametric")
    assert result.test is not None
    assert result.effect_size is not None
    with pytest.raises(ValueError, match="unknown barstats design"):
        tx.contingency_subtitle_expression(
            result.test, result.effect_size, design="three-way", n_obs=32, digits=2
        )
    wrong = dataclasses.replace(result.effect_size, name="pearsons_c")
    with pytest.raises(ValueError, match="reports 'cramers_v'"):
        tx.contingency_subtitle_expression(
            result.test, wrong, design="two-way", n_obs=32, digits=2
        )


def test_types_of_the_parts() -> None:
    result = result_for("ggbarstats-mtcars-cyl-parametric")
    assert isinstance(result.bars[0], Bar)
    assert isinstance(result.cells[0], Cell)
    assert result.text.tick_labels == ("",)
    assert result.text.bar_labels == ("",)
