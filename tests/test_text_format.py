"""Number formatting reproduces every number string in every oracle fixture.

Each fixture's expressions hold the strings ggstatsplot printed; the fixture's
tables hold the numbers behind them (15 significant digits). Every pair is checked:
subtitle and caption rows, pairwise rows, bracket annotations and both centrality
label layers (value and the n tick label).

Centrality values that sit on a decimal tie (5.155) cannot be decided from 15
significant digits; they are recomputed from the dataset exactly as R computes them
(:func:`centrality_from_data`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from inferplot import text as tx
from tests.oracle_fixtures import (
    OracleCase,
    load_cases,
    number,
    probe,
    r_double,
    r_level_label,
    read_dataset,
    require_cases,
)
from tests.plotmath import parse
from tests.tolerances import assert_close

Row = Mapping[str, Any]

_Q = r'"([^"]*)"'
_CI_RE = re.compile(
    r"CI\[" + _Q + r"\](?:\^\w+)? ~ \"\[\" \* " + _Q + r", " + _Q + r' \* "\]"'
)
_EQ_RE = re.compile(r"== " + _Q)
_DF2_RE = re.compile(r"\]\((\S+?), (\S+?)\) ==")
_DF1_RE = re.compile(r'\* "\(" \* (\S+?) \* "\)" ==')


def all_case_params() -> list[Any]:
    """Every fixture, of every function and type, as a pytest param."""
    cases = require_cases(load_cases(), "any function")
    return [pytest.param(case, id=case.name) for case in cases]


# --------------------------------------------------------------------------- #
# R's centrality values, recomputed exactly
# --------------------------------------------------------------------------- #


def exact_mean(values: Sequence[float]) -> float:
    """The correctly rounded mean, as R's mean() gives it (long-double accumulation)."""
    return float(sum(Fraction(v) for v in values) / len(values))


def median(values: Sequence[float]) -> float:
    """R's median: the middle value, or the mean of the two middle values."""
    s = sorted(values)
    n = len(s)
    h = n // 2
    return s[h] if n % 2 else exact_mean([s[h - 1], s[h]])


def trimmed_mean(values: Sequence[float], tr: float) -> float:
    """R's mean(x, trim = tr): drop floor(n * tr) values from each end."""
    s = sorted(values)
    n = len(s)
    lo = math.floor(n * tr)
    return exact_mean(s[lo : n - lo])


def near_decimal_tie(x: float, digits: int) -> bool:
    """True when ``x`` is within 15-significant-digit storage error of a decimal tie."""
    if not math.isfinite(x) or x == 0:
        return False
    scaled = abs(x) * 10**digits
    return abs(scaled - math.floor(scaled) - 0.5) < 1e-12 * max(1.0, scaled)


def case_digits(case: OracleCase) -> int:
    return int(case.args.get("digits", 2))


def centrality_from_data(case: OracleCase, level: str) -> float:
    """The level's centrality value recomputed from the dataset as R computes it."""
    x, y = str(case.args["x"]), str(case.args["y"])
    frame = read_dataset(case, "pandas")[[x, y]].dropna()
    values = [
        float(v)
        for v, k in zip(frame[y], frame[x], strict=True)
        if r_level_label(k) == level
    ]
    if case.type == "parametric":
        return exact_mean(values)
    if case.type == "nonparametric":
        return median(values)
    if case.type == "robust":
        return trimmed_mean(values, float(case.args.get("tr", 0.2)))
    raise ValueError(
        f"{case.name}: {case.type} centrality cannot be recomputed exactly"
    )


def centrality_value(case: OracleCase, row: Row) -> float:
    """The centrality value behind a label: the fixture's, or recomputed at a tie."""
    value = number(row[str(case.args["y"])])
    if near_decimal_tie(value, case_digits(case)) and case.type != "bayes":
        return centrality_from_data(case, str(row[str(case.args["x"])]))
    return value


def subtitle_rows(case: OracleCase) -> list[Row]:
    """The subtitle table: ``subtitle_data``, or for a case where ggstatsplot failed
    what its hidden subtitle call computed before the failure."""
    if case.error is not None:
        return _partial_rows(case, "subtitle")
    return case.subtitle_rows


def caption_rows(case: OracleCase) -> list[Row]:
    """The caption table, from ``caption_data`` or a failed case's partial output."""
    if case.error is not None:
        return _partial_rows(case, "caption")
    return case.table("caption_data")


def _partial_rows(case: OracleCase, component: str) -> list[Row]:
    value = case.partial(component)
    return [] if value is None else list(value)


def centrality_rows(case: OracleCase) -> list[tuple[str, Row]]:
    """(geom, row) for every row of every centrality label layer.

    A case where ggstatsplot failed has no plot; its rows are what the hidden
    ``centrality_description()`` call returned (geom "centrality_description"),
    the data both label layers would have drawn.
    """
    if case.error is not None:
        return [
            ("centrality_description", row)
            for row in _partial_rows(case, "centrality_description")
        ]
    out: list[tuple[str, Row]] = []
    for layer in case.plot_layers:
        if layer["geom"] in ("GeomPoint", "GeomLabelRepel") and layer["data"]:
            out.extend((str(layer["geom"]), row) for row in layer["data"])
    return out


def df_digits(method: str, digits: int) -> tuple[int, int]:
    """(digits.df, digits.df.error) for a between-subjects test, as statsExpressions."""
    m = method.lower()
    fractional = (
        "welch" in m
        or "not assuming equal variances" in m
        or "yuen" in m
        or "trimmed means" in m
    )
    return 0, (digits if fractional else 0)


# --------------------------------------------------------------------------- #
# (printed string, recomputed string) pairs of one fixture
# --------------------------------------------------------------------------- #


class Collector:
    def __init__(self, case: OracleCase) -> None:
        self.case = case
        self.digits = case_digits(case)
        self.pairs: list[tuple[str, str, float, str, str]] = []
        self.problems: list[str] = []

    def add(self, where: str, field: str, raw: float, expected: str, got: str) -> None:
        self.pairs.append((where, field, raw, expected, got))

    def frequentist(self, where: str, row: Row) -> None:
        d = self.digits
        expr = str(row["expression"])
        eqs = _EQ_RE.findall(expr)
        ci = _CI_RE.search(expr)
        if len(eqs) != 4 or ci is None:
            self.problems.append(f"{where}: unexpected template {expr}")
            return
        stat_s, p_s, est_s, n_s = eqs
        level_s, low_s, high_s = ci.groups()
        d_df, d_dfe = df_digits(str(row["method"]), d)
        v = number
        self.add(
            where,
            "statistic",
            v(row["statistic"]),
            stat_s,
            tx.format_value(v(row["statistic"]), d),
        )
        self.add(
            where,
            "p",
            v(row["p.value"]),
            p_s,
            tx.format_p_subtitle(v(row["p.value"]), d),
        )
        self.add(
            where,
            "estimate",
            v(row["estimate"]),
            est_s,
            tx.format_value(v(row["estimate"]), d),
        )
        self.add(
            where,
            "conf_low",
            v(row["conf.low"]),
            low_s,
            tx.format_value(v(row["conf.low"]), d),
        )
        self.add(
            where,
            "conf_high",
            v(row["conf.high"]),
            high_s,
            tx.format_value(v(row["conf.high"]), d),
        )
        self.add(
            where,
            "conf_level",
            v(row["conf.level"]),
            level_s,
            tx.format_conf_level(v(row["conf.level"])),
        )
        self.add(where, "n", v(row["n.obs"]), n_s, tx.format_n(int(row["n.obs"])))
        m2 = _DF2_RE.search(expr)
        m1 = _DF1_RE.search(expr)
        if m2:
            self.add(
                where, "df", v(row["df"]), m2.group(1), tx.format_df(v(row["df"]), d_df)
            )
            self.add(
                where,
                "df_error",
                v(row["df.error"]),
                m2.group(2),
                tx.format_df(v(row["df.error"]), d_dfe),
            )
        elif m1:
            col = "df.error" if "df.error" in row else "df"
            self.add(
                where,
                "df_error",
                v(row[col]),
                m1.group(1),
                tx.format_df(v(row[col]), d_dfe),
            )
        elif "df" in row or "df.error" in row:
            self.problems.append(f"{where}: df column but no df in {expr}")

    def bayes(self, where: str, row: Row) -> None:
        d = self.digits
        expr = str(row["expression"])
        eqs = _EQ_RE.findall(expr)
        ci = _CI_RE.search(expr)
        if len(eqs) != 3 or ci is None:
            self.problems.append(f"{where}: unexpected template {expr}")
            return
        bf_s, est_s, scale_s = eqs
        level_s, low_s, high_s = ci.groups()
        bf10 = number(row["bf10"])
        self.add(where, "log_bf01", -math.log(bf10), bf_s, tx.format_bf01_log(bf10, d))
        for field, col, s in (
            ("estimate", "estimate", est_s),
            ("conf_low", "conf.low", low_s),
            ("conf_high", "conf.high", high_s),
            ("prior_scale", "prior.scale", scale_s),
        ):
            self.add(
                where, field, number(row[col]), s, tx.format_value(number(row[col]), d)
            )
        level = number(row["conf.level"])
        self.add(where, "conf_level", level, level_s, tx.format_conf_level(level))

    def row(self, where: str, row: Row) -> None:
        if "bf10" in row:
            self.bayes(where, row)
        else:
            self.frequentist(where, row)

    def pair_label(self, where: str, row: Row, printed: str) -> None:
        eqs = _EQ_RE.findall(printed)
        if len(eqs) != 1:
            self.problems.append(f"{where}: unexpected label {printed}")
            return
        if "bf10" in row:
            bf10 = number(row["bf10"])
            self.add(
                where,
                "log_bf01",
                -math.log(bf10),
                eqs[0],
                tx.format_bf01_log(bf10, self.digits),
            )
        else:
            p = number(row["p.value"])
            self.add(where, "p", p, eqs[0], tx.format_value(p, self.digits, missing=""))

    def centrality(self, rows: list[tuple[str, Row]]) -> None:
        x = str(self.case.args["x"])
        for geom, row in rows:
            where = f"{geom}[{row[x]}]"
            value = centrality_value(self.case, row)
            expected = _EQ_RE.findall(str(row["expression"]))
            if len(expected) != 1:
                self.problems.append(f"{where}: unexpected label {row['expression']}")
                continue
            self.add(
                where,
                "centrality",
                value,
                expected[0],
                tx.format_value(value, self.digits, missing=""),
            )
            n = int(row["n.obs"])
            self.add(
                where,
                "n_expression",
                float(n),
                str(row["n.expression"]),
                tx.tick_label(r_level_label(row[x]), n),
            )


def collect(case: OracleCase) -> Collector:
    """Every (printed, recomputed) number string of one fixture."""
    c = Collector(case)
    if case.error is not None and (
        case.raw["stats"] is not None or case.subtitle is not None
    ):
        c.problems.append("an erroring case should hold no results")
    for i, row in enumerate(subtitle_rows(case)):
        c.row(f"subtitle_data[{i}]", row)
    for i, row in enumerate(caption_rows(case)):
        c.row(f"caption_data[{i}]", row)
    c.centrality(centrality_rows(case))
    if case.error is not None:
        return c  # no pairwise table and no brackets: that call failed
    rows = case.pairwise
    by_pair = {(str(r["group1"]), str(r["group2"])): r for r in rows}
    for i, row in enumerate(rows):
        c.pair_label(f"pairwise[{i}]", row, str(row["expression"]))
    for params in case.layer_params("GeomSignif"):
        stat = params["stat"]
        annotations = stat["annotations"]
        annotations = (
            [annotations] if isinstance(annotations, str) else list(annotations)
        )
        comparisons = [tuple(map(str, pair)) for pair in stat["comparisons"]]
        if len(annotations) != len(comparisons):
            c.problems.append(
                "GeomSignif: annotations and comparisons differ in length"
            )
        for pair, ann in zip(comparisons, annotations, strict=False):
            row = by_pair.get((pair[0], pair[1]))
            if row is None:
                c.problems.append(f"GeomSignif: bracket {pair} has no pairwise row")
                continue
            c.pair_label(f"GeomSignif{pair}", row, ann)
    return c


@pytest.mark.parametrize("case", all_case_params())
def test_every_number_string(case: OracleCase) -> None:
    c = collect(case)
    assert not c.problems
    assert c.pairs, "no number strings found"
    mismatches = [p for p in c.pairs if p[3] != p[4]]
    assert not mismatches


def test_phase0_fixtures_hold_1604_number_strings() -> None:
    """The 45 Phase 0 fixtures: the count the prototype measured, as a guard."""
    phase0 = [c for c in load_cases() if c.name in _PHASE0]
    assert len(phase0) == 45
    assert sum(len(collect(c).pairs) for c in phase0) == 1604


def test_decimal_ties_are_recomputed_and_correct() -> None:
    """The labels whose value sits on a decimal tie, decided from the data.

    Ten labels per layer in the Phase 0 fixtures; the nonparametric medians
    PlantGrowth, edge_k5 and edge_unbalanced are exact ties of two middle values.
    """
    ties = 0
    for case in load_cases():
        if case.type == "bayes":
            continue
        y = str(case.args["y"])
        x = str(case.args["x"])
        for _, row in centrality_rows(case):
            stored = number(row[y])
            if not near_decimal_tie(stored, case_digits(case)):
                continue
            ties += 1
            value = centrality_from_data(case, str(row[x]))
            assert_close(
                "closed_form", value, stored, what=f"{case.name} {row[x]} centrality"
            )
            expected = _EQ_RE.findall(str(row["expression"]))[0]
            assert tx.format_value(value, case_digits(case)) == expected
    assert ties >= 20


_PHASE0 = frozenset(
    f"ggbetweenstats-{label}-{type_}"
    for label in (
        "ToothGrowth",
        "PlantGrowth",
        "mtcars",
        "mtcars-am",
        "mtcars-disp",
        "iris",
        "bugs_long",
        "edge_k2",
        "edge_k5",
        "edge_unbalanced",
    )
    for type_ in ("parametric", "nonparametric", "robust", "bayes")
) | {
    "ggbetweenstats-ToothGrowth-greater-parametric",
    "ggbetweenstats-ToothGrowth-less-parametric",
    "ggbetweenstats-ToothGrowth-greater-nonparametric",
    "ggbetweenstats-ToothGrowth-less-nonparametric",
    "ggbetweenstats-mtcars-am-drat-parametric",
}


# --------------------------------------------------------------------------- #
# documented rules (insight's reference manual; R's printing of doubles)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("x", "want"),
    [
        (1.20, "1.20"),
        (1.2012313, "1.20"),
        (0.0045, "4.50e-03"),
        (234.0, "234.00"),
        (-23.0, "-23.00"),
        (0.12, "0.12"),
        (0.34, "0.34"),
        (0.123, "0.12"),
        (0.345, "0.34"),
        (3.0, "3.00"),
        (0.0, "0.00"),
        (0.00709, "7.09e-03"),
        (1.5e-14, "1.50e-14"),
        (575.5, "575.50"),
        (math.inf, "Inf"),
        (-math.inf, "-Inf"),
    ],
)
def test_format_value_documented(x: float, want: str) -> None:
    assert tx.format_value(x) == want


def test_format_value_missing() -> None:
    assert tx.format_value(None) == "NA"
    assert tx.format_value(None, missing="") == ""
    assert tx.format_value(math.nan, missing="") == ""
    with pytest.raises(ValueError, match="digits"):
        tx.format_value(1.0, -1)


def test_pinned_rules() -> None:
    """The rules probe-format pinned, at the rows the comments in text.py cite."""
    assert tx.SCI_LARGE_THRESHOLD == 1e5
    # large switch on the unrounded value
    assert tx.format_value(99999.0) == "99999.00"
    assert tx.format_value(99999.996) == "100000.00"
    assert tx.format_value(99999.99999999997, 0) == "100000"
    assert tx.format_value(1e5) == "1.00e+05"
    assert tx.format_value(123456.0) == "1.23e+05"
    # small switch at log10(|x|) < -digits, on the unrounded value
    assert tx.format_value(0.000999, 0) == "1e-03"
    assert tx.format_value(0.000999, 3) == "9.990e-04"
    assert tx.format_value(0.000999, 4) == "0.0010"
    assert tx.format_value(0.0095, 2) == "9.50e-03"
    assert tx.format_value(0.0095, 3) == "0.009"
    assert tx.format_value(0.0099999, 2) == "1.00e-02"
    assert tx.format_value(0.0099999, 3) == "0.010"
    assert tx.format_value(0.0045, 3) == "0.004"
    assert tx.format_value(0.00045, 3) == "4.500e-04"
    # missing and infinite values
    assert tx.format_value(math.nan, 3, missing="") == ""
    assert tx.format_value(math.nan, 3, missing="NA") == "NA"
    assert tx.format_value(math.inf, 0) == "Inf"
    assert tx.format_value(-math.inf, 4) == "-Inf"
    # counts
    assert tx.N_BIG_MARK == ","
    assert tx.format_n(999) == "999"
    assert tx.format_n(1000) == "1,000"
    assert tx.format_n(1234567) == "1,234,567"


def test_negative_zero_stays_unpinned() -> None:
    """R handed the probe's -0 to format_value as +0, so no row pins it.

    Every "values" row reports ``negative_zero`` false, although the grid ends with
    a -0 (two sets of rows read "0x0p+0"), and the end-to-end subtitle, whose
    estimate is written -0, prints "0.00". inferplot keeps C sprintf's "-0.00".
    """
    rows = probe("probe-format").rows("values")
    assert not any(row["negative_zero"] for row in rows)
    zeros = [row for row in rows if row["hex"] == "0x0p+0"]
    assert len(zeros) == 10
    assert all(row["format_value"] == row["sprintf_f"] for row in zeros)
    assert tx.UNPINNED_KEEP_NEGATIVE_ZERO
    assert tx.format_value(-0.0) == "-0.00"


@pytest.mark.parametrize(
    ("x", "want"),
    [
        (2.0, "2"),
        (97.4, "97.4"),
        (17.13, "17.13"),
        (1e5, "1e+05"),
        (123456.0, "123456"),
        (1e-4, "1e-04"),
        (1e-3, "0.001"),
        (1 / 3, "0.333333333333333"),
        (0.1 + 0.2, "0.3"),
        (1e-300, "1e-300"),
        (-1.5e-5, "-1.5e-05"),
        (1234567.0, "1234567"),
        (100000.5, "100000.5"),
        (1e15, "1e+15"),
        (0.95 * 100, "95"),
        (0.9 * 100, "90"),
        (0.0, "0"),
        (math.nan, "NaN"),
    ],
)
def test_r_number_text(x: float, want: str) -> None:
    assert tx.r_number_text(x) == want


def test_df_conf_level_p_and_bf() -> None:
    assert tx.format_df(97.401587118856, 2) == "97.4"
    assert tx.format_df(2.0, 0) == "2"
    assert tx.format_df(27.1977973620292, 2) == "27.2"
    assert tx.format_df(9.999, 2) == "10"
    assert tx.format_df(math.nan, 2) == "NA"
    assert tx.format_conf_level(0.9) == "90%"
    assert tx.format_conf_level(0.999) == "99.9%"
    assert tx.format_p_subtitle(0.0) == "2.23e-308"
    assert tx.format_bf01_log(math.e) == "-1.00"
    assert tx.format_bf01_log(0.0) == "Inf"
    assert tx.format_bf01_log(math.inf) == "-Inf"
    assert tx.format_bf01_log(math.nan) == ""
    with pytest.raises(ValueError, match="negative"):
        tx.format_bf01_log(-1.0)
    with pytest.raises(ValueError, match="n must be"):
        tx.format_n(-1)


# --------------------------------------------------------------------------- #
# probe-format: R's own output on a grid (tests/fixtures/oracle/probe-format.json)
# --------------------------------------------------------------------------- #


def _value_rows() -> list[Row]:
    rows = probe("probe-format").rows("values")
    # 1,002 grid values (decimal ties and their one-ulp neighbours, subnormals, the
    # largest double, Inf, -Inf, -0, NA, NaN) at digits 0 to 4
    assert len(rows) == 5010
    assert {int(row["digits"]) for row in rows} == {0, 1, 2, 3, 4}
    return rows


def test_probe_format_value() -> None:
    """``insight::format_value(x, digits)`` (missing ""): every row."""
    bad = [
        (row["hex"], row["digits"], row["format_value"], got)
        for row in _value_rows()
        if (got := tx.format_value(r_double(row["hex"]), row["digits"], missing=""))
        != row["format_value"]
    ]
    assert not bad


def test_probe_to_char() -> None:
    """statsExpressions' ``.to_char(x, digits)`` (missing "NA"): every row."""
    bad = [
        (row["hex"], row["digits"], row["to_char"], got)
        for row in _value_rows()
        if (got := tx.format_value(r_double(row["hex"]), row["digits"], missing="NA"))
        != row["to_char"]
    ]
    assert not bad


def test_probe_df_route() -> None:
    """A df pasted unquoted, parsed and deparsed by R (``df_deparse``): every row.

    The route applies to every row: R parsed each ``.to_char`` text back (none of
    them errored), negative values and the constants NA, Inf and -Inf included.
    """
    bad: list[tuple[str, int, str, str]] = []
    for row in _value_rows():
        assert not str(row["df_deparse"]).startswith("<error")
        x = r_double(row["hex"])
        got = tx.format_df(math.nan if x is None else x, row["digits"])
        if got != row["df_deparse"]:
            bad.append((row["hex"], row["digits"], row["df_deparse"], got))
    assert not bad


def test_probe_counts() -> None:
    """``.prettyNum(n)`` for doubles and integers from 0 to 1e9."""
    rows = probe("probe-format").rows("counts")
    assert {row["type"] for row in rows} == {"double", "integer"}
    assert len(rows) == 34
    for row in rows:
        n = r_double(row["hex"])
        assert n is not None
        assert n.is_integer()
        assert tx.format_n(int(n)) == row["prettyNum"], row


def test_probe_conf_levels() -> None:
    """``paste0(conf.level * 100L, "%")`` for nine levels from 0.5 to 0.9999."""
    rows = probe("probe-format").rows("conf_levels")
    assert len(rows) == 9
    for row in rows:
        level = r_double(row["hex"])
        assert level is not None
        assert tx.format_conf_level(level) == row["label"], row


# The inputs oracle/generate.R's format_probe() gave statsExpressions (end_to_end).
_E2E_FREQUENTIST: dict[str, Any] = {
    "statistic_value": 123456.7,
    "p_value": 0.0,
    # The generator writes -0, but R handed format_value +0 (see
    # test_negative_zero_stays_unpinned), and "0.00" is what it printed.
    "estimate": 0.0,
    "conf_level": 0.9,
    "conf_low": -math.inf,
    "conf_high": math.nan,  # NA_real_
    "n_obs": 123456,
    "df": 1e5,
    "df_error": 99999.996,
}
_E2E_METHOD = "One-way analysis of means (not assuming equal variances)"


@pytest.mark.parametrize(("key", "digits"), [("subtitle", 2), ("subtitle_digits3", 3)])
def test_probe_end_to_end_subtitle(key: str, digits: int) -> None:
    """``add_expression_col(digits, digits.df = 0, digits.df.error = digits)``."""
    from tests.test_text_expressions import effect_size_key, statistic_key

    built = tx.frequentist_expression(
        statistic=statistic_key(_E2E_METHOD),
        effect_size=effect_size_key("Omega2"),
        digits=digits,
        digits_df=0,
        digits_df_error=digits,
        **_E2E_FREQUENTIST,
    )
    assert built == parse(probe("probe-format").section("end_to_end")[key])


def test_probe_end_to_end_pairwise() -> None:
    """``.pairwise_p_adjust_expr(p, "none", 2)``: an exact 0 p prints "0.00"."""
    labels = probe("probe-format").section("end_to_end")["pairwise"]
    p_values = (0.0, 1e-300, 0.004999, 0.00999, 1.0)
    assert len(labels) == len(p_values)
    for p, label in zip(p_values, labels, strict=True):
        built = tx.pairwise_p_expression(p_adjusted=p, adjust_method="none", digits=2)
        assert built == parse(label)


def test_probe_end_to_end_bayes() -> None:
    """The Bayesian template with a Bayes factor of exactly 0 (exp(-123456.7))."""
    from tests.test_text_expressions import effect_size_key, prior_key

    bf10 = math.exp(-123456.7)
    assert bf10 == 0.0
    assert tx.format_bf01_log(bf10) == "Inf"
    built = tx.bayes_expression(
        log_bf01=math.inf,  # -log(0), as R computes it
        effect_size=effect_size_key("Bayesian t-test"),
        estimate=0.1234,
        conf_level=0.95,
        conf_method="ETI",
        conf_low=-0.0049,
        conf_high=1e6,
        prior=prior_key("Bayesian t-test"),
        prior_scale=0.707,
        digits=2,
    )
    assert built == parse(probe("probe-format").section("end_to_end")["bayes"])


# format_probe()'s tie_data: two groups of eight whose centrality values sit on
# decimal ties.
_TIE_DATA = {
    "a": [10.1, 9.2, 9.7, 10.4, 9.5, 9.9, 9.8, 9.68],
    "b": [12.2, 12.39, 12.1, 12.3, 12.31, 12.29, 12.6, 11.9],
}


def _tie_centrality(kind: str, values: list[float]) -> float:
    if kind == "parametric":
        return exact_mean(values)
    if kind == "nonparametric":
        return median(values)
    return trimmed_mean(values, 0.2)


_TIE_SUBSCRIPTS = {
    "parametric": "mean",
    "nonparametric": "median",
    "robust": "trimmed_mean",
}


def test_probe_centrality() -> None:
    """R's centrality doubles on tie data, and the labels drawn from them.

    The values are recomputed exactly as R computes them (the mean also through
    ``inferplot.stats.mean``), bit for bit, and each label is rebuilt from R's own
    double.
    """
    from inferplot.stats import mean

    sections = probe("probe-format").rows("centrality")
    assert [s["type"] for s in sections] == list(_TIE_SUBSCRIPTS)
    for section in sections:
        kind = str(section["type"])
        assert len(section["values_hex"]) == len(_TIE_DATA)
        for hex_value, label, group in zip(
            section["values_hex"], section["labels"], _TIE_DATA.values(), strict=True
        ):
            value = r_double(hex_value)
            assert value is not None
            assert _tie_centrality(kind, group) == value
            if kind == "parametric":
                assert mean(np.asarray(group, dtype=np.float64)) == value
            built = tx.centrality_expression(
                centrality=_TIE_SUBSCRIPTS[kind], value=value, digits=2
            )
            assert built == parse(label)
