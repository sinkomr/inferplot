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

import pytest

from inferplot import text as tx
from tests.oracle_fixtures import (
    OracleCase,
    load_cases,
    number,
    r_level_label,
    read_dataset,
    require_cases,
)

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


def centrality_rows(case: OracleCase) -> list[tuple[str, Row]]:
    """(geom, row) for every row of every centrality label layer."""
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


def collect(case: OracleCase) -> Collector:
    """Every (printed, recomputed) number string of one fixture."""
    c = Collector(case)
    if case.error is not None:
        if case.raw["stats"] is not None or case.subtitle is not None:
            c.problems.append("an erroring case should hold no results")
        return c
    for i, row in enumerate(case.subtitle_rows):
        c.row(f"subtitle_data[{i}]", row)
    for i, row in enumerate(case.table("caption_data")):
        c.row(f"caption_data[{i}]", row)
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
    x = str(case.args["x"])
    for geom, row in centrality_rows(case):
        where = f"{geom}[{row[x]}]"
        value = centrality_value(case, row)
        expected = _EQ_RE.findall(str(row["expression"]))
        if len(expected) != 1:
            c.problems.append(f"{where}: unexpected label {row['expression']}")
            continue
        c.add(
            where,
            "centrality",
            value,
            expected[0],
            tx.format_value(value, c.digits, missing=""),
        )
        n = int(row["n.obs"])
        c.add(
            where,
            "n_expression",
            float(n),
            str(row["n.expression"]),
            tx.tick_label(r_level_label(row[x]), n),
        )
    return c


@pytest.mark.parametrize("case", all_case_params())
def test_every_number_string(case: OracleCase) -> None:
    c = collect(case)
    assert not c.problems
    if case.error is None:
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
        if case.error is not None or case.type == "bayes":
            continue
        y = str(case.args["y"])
        x = str(case.args["x"])
        for _, row in centrality_rows(case):
            stored = number(row[y])
            if not near_decimal_tie(stored, case_digits(case)):
                continue
            ties += 1
            value = centrality_from_data(case, str(row[x]))
            assert value == pytest.approx(stored, rel=1e-13)
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


def test_unpinned_hypotheses() -> None:
    """The working hypotheses the oracle probe will pin (constants in text.py)."""
    assert tx.UNPINNED_SCI_LARGE_THRESHOLD == 1e5
    assert tx.format_value(99999.0) == "99999.00"
    assert tx.format_value(123456.0) == "1.23e+05"
    assert tx.UNPINNED_SMALL_SCI_SCALES_WITH_DIGITS
    assert tx.format_value(0.0045, 3) == "0.004"  # log10 >= -3: fixed at digits 3
    assert tx.format_value(0.00045, 3) == "4.500e-04"
    assert tx.format_n(1234567) == "1,234,567"
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
    with pytest.raises(ValueError, match="n must be"):
        tx.format_n(-1)
