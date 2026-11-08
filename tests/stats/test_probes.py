"""inferplot.stats against the R oracle probes (``tests/fixtures/oracle/probe-*.json``).

The probes record what the R functions behind ggstatsplot 1.1.4 return on fixed
inputs, as exact doubles (R's ``sprintf("%a")``, read with ``r_double``):
``stats::p.adjust`` on fixed vectors (probe-padjust), ``stats::ptukey`` on a grid
and ``PMCMRplus::gamesHowellTest``'s unadjusted p per fixture pair (probe-ptukey),
``effectsize``'s omega squared CI next to an accurate root of the same equation
(probe-omega), and R's centrality doubles on decimal-tie data (probe-format).
"""

from __future__ import annotations

import functools
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inferplot.stats import (
    P_ADJUST_METHODS,
    PAdjustMethod,
    PairResult,
    games_howell,
    mean,
    p_adjust,
    partial_omega_squared,
    srange_sf,
)
from tests.oracle_fixtures import (
    ROOT,
    OracleCase,
    Row,
    load_cases,
    number,
    probe,
    r_double,
)
from tests.stats.fixture_data import groups, p_adjust_method
from tests.tolerances import TOLERANCES, assert_close, is_close

REFERENCE = (
    Path(__file__).resolve().parents[1] / "data" / "studentized_range_reference.json"
)


def _exact(text: str) -> float:
    """A probe double that must be present (not R's NA)."""
    value = r_double(text)
    assert value is not None, f"unexpected NA: {text!r}"
    return value


def _ulps(a: float, b: float) -> int:
    """Distance in units in the last place between two finite doubles of one sign."""
    bits = np.array([a, b], dtype=np.float64).view(np.int64)
    return abs(int(bits[0]) - int(bits[1]))


# ------------------------------------------------------------------ p.adjust


def _padjust_cases() -> list[Row]:
    cases = probe("probe-padjust").rows("cases")
    assert len(cases) == 20
    return cases


def _complete(case: Row) -> bool:
    return all(text != "NA" for text in case["p_hex"])


def test_p_adjust_methods_are_r_methods() -> None:
    assert tuple(probe("probe-padjust").section("methods")) == P_ADJUST_METHODS


def test_p_adjust_matches_r_on_every_complete_vector() -> None:
    """Every method on the 19 vectors without missing values.

    All within ``closed_form``, and bitwise equal to R except BY on the 1008-value
    vector: R sums the BY factor q = sum(1/k, k <= n) in long double (x86_64, 64-bit
    significand), which at n = 1008 rounds to the double 1 ulp above the correctly
    rounded sum inferplot uses (``math.fsum``), so all 1008 BY values come out 1 or
    2 ulp below R's. The next test shows that this is the whole difference.
    """
    cases = [c for c in _padjust_cases() if _complete(c)]
    assert len(cases) == 19
    differ: dict[tuple[str, str], Counter[int]] = {}
    below = 0
    for case in cases:
        p = [_exact(h) for h in case["p_hex"]]
        for method in P_ADJUST_METHODS:
            expected = [_exact(h) for h in case["adjusted"][method]["hex"]]
            got = p_adjust(p, method)
            assert len(got) == len(expected)
            for i, (g, e) in enumerate(zip(got, expected, strict=True)):
                what = f"{case['name']} {method} [{i}]"
                assert_close("closed_form", g, e, what=what)
                if g != e:
                    differ.setdefault((case["name"], method), Counter())[
                        _ulps(g, e)
                    ] += 1
                    below += g < e
    assert differ == {("by_n1008", "BY"): Counter({1: 928, 2: 80})}
    assert below == 1008


def test_by_difference_is_r_long_double_harmonic_sum() -> None:
    padjust = probe("probe-padjust")
    r_q1007, r_q1008 = (_exact(h) for h in padjust.section("harmonic_sum_hex"))
    assert padjust.section("longdouble_digits") == 64
    assert math.fsum(1.0 / k for k in range(1, 1008)) == r_q1007
    fsum_1008 = math.fsum(1.0 / k for k in range(1, 1009))
    assert _ulps(fsum_1008, r_q1008) == 1
    assert fsum_1008 < r_q1008
    # With R's q, BY = step-up of ((q n) / i) p_(i) (the order padjust.py documents)
    # reproduces every R value bit for bit.
    case = next(c for c in _padjust_cases() if c["name"] == "by_n1008")
    p = np.array([_exact(h) for h in case["p_hex"]])
    order = p.argsort(kind="stable")
    n = p.size
    scaled = ((r_q1008 * float(n)) / np.arange(1, n + 1, dtype=np.float64)) * p[order]
    stepped = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    adjusted = np.empty(n)
    adjusted[order] = stepped
    expected = np.array([_exact(h) for h in case["adjusted"]["BY"]["hex"]])
    assert np.array_equal(adjusted, expected)


@pytest.mark.parametrize("method", P_ADJUST_METHODS)
def test_p_adjust_refuses_missing_values(method: PAdjustMethod) -> None:
    """The vector with NA: inferplot raises where R skips the missing values.

    ``inferplot.stats`` takes clean arrays only (statistics rules: no missing-value
    handling below the analysis layer), and a Games-Howell table never has a missing
    p-value: inferplot computes every pair or the analysis stops first. R keeps NA in
    place and adjusts the rest with n = the number of non-missing values; the same
    numbers come from passing only those values, which is pinned here so a caller
    that ever needs R's behaviour knows how to get it.
    """
    case = next(c for c in _padjust_cases() if not _complete(c))
    assert case["name"] == "with_na"
    p = [r_double(h) for h in case["p_hex"]]
    with pytest.raises(ValueError, match=r"p-values must lie in \[0, 1\]"):
        p_adjust([math.nan if v is None else v for v in p], method)
    expected = [r_double(h) for h in case["adjusted"][method]["hex"]]
    assert [v is None for v in expected] == [v is None for v in p]
    present = [v for v in p if v is not None]
    got = p_adjust(present, method)
    assert list(got) == [v for v in expected if v is not None]


# ------------------------------------------------------------------ ptukey grid

# Largest |R's ptukey(q, k, df, lower.tail = FALSE) - srange_sf(q, k, df)| per df on
# probe-ptukey's "tails" grid (k = 2, 3, 4, 5, 6, 10, 20; q = 1 to 30), the numbers
# srange_sf's docstring states. srange_sf is accurate to about 1e-13 relative
# against the mpmath references (df up to 1000 and inf for k >= 3; srange_sf's
# docstring says why the df 25000 and 1e5 differences are R's too), so these are
# R's errors.
PTUKEY_MAX_ABS_ERROR = {
    2.0: 1.35e-3,
    5.0: 3.20e-5,
    10.0: 1.78e-7,
    20.0: 1.68e-8,
    50.0: 1.85e-8,
    94.6978: 1.55e-8,
    200.0: 8.28e-9,
    1000.0: 1.09e-8,
    25000.0: 5.28e-7,
    1e5: 1.88e-5,
    math.inf: 1.18e-8,
}


@functools.cache
def _ptukey_grid() -> tuple[tuple[float, int, float, float | None, float], ...]:
    """(q, k, df, R's upper tail or None for NaN, srange_sf) for every grid row."""
    out: list[tuple[float, int, float, float | None, float]] = []
    for row in probe("probe-ptukey").rows("tails"):
        q, df = _exact(row["q_hex"]), _exact(row["df_hex"])
        k = int(row["nmeans"])
        r = r_double(row["upper_tail_hex"])
        out.append(
            (q, k, df, None if r is None or math.isnan(r) else r, srange_sf(q, k, df))
        )
    return tuple(out)


def test_ptukey_error_by_df() -> None:
    grid = _ptukey_grid()
    assert len(grid) == 1365
    worst: dict[float, float] = {}
    nan_dfs: Counter[float] = Counter()
    for _, _, df, r, ours in grid:
        assert 0.0 <= ours <= 1.0
        if r is None:
            nan_dfs[df] += 1
            continue
        worst[df] = max(worst.get(df, 0.0), abs(ours - r))
    # R returns NaN ("NaNs produced") for every point at df 1 and 1.5.
    assert nan_dfs == Counter({1.0: 105, 1.5: 105})
    assert worst.keys() == PTUKEY_MAX_ABS_ERROR.keys()
    for df, measured in worst.items():
        assert measured == pytest.approx(PTUKEY_MAX_ABS_ERROR[df], rel=5e-3), df


def test_ptukey_uses_its_infinite_df_value_at_df_1e5() -> None:
    by_point = {(q, k, df): r for q, k, df, r, _ in _ptukey_grid()}
    at_1e5 = {(q, k): r for (q, k, df), r in by_point.items() if df == 1e5}
    assert len(at_1e5) == 105
    assert all(r == by_point[(q, k, math.inf)] for (q, k), r in at_1e5.items())


# ------------------------------------------------------------------ Games-Howell


def _cases() -> dict[str, OracleCase]:
    return {case.name: case for case in load_cases()}


def _gh_entries() -> list[Row]:
    entries = probe("probe-ptukey").rows("games_howell")
    assert len(entries) == 30
    return entries


def _reference_points(name: str) -> list[dict[str, Any]]:
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    return list(data["sets"][name]["points"])


def _check_pairs(entry: Row, case: OracleCase) -> list[PairResult]:
    """games_howell's raw pairs, checked against the probe's q and Welch df."""
    levels = case.analysis_levels
    assert tuple(entry["levels"]) == levels
    assert entry["p_adjust_method"] == "none"
    assert {str(k): int(v) for k, v in entry["n_per_level"].items()} == case.n_per_level
    pairs = games_howell(groups(case), p_adjust_method="none")
    assert len(entry["pairs"]) == len(pairs) == len(levels) * (len(levels) - 1) // 2
    for pair, row in zip(pairs, entry["pairs"], strict=True):
        assert (levels[pair.i], levels[pair.j]) == (row["group1"], row["group2"])
        what = f"{case.name} {row['group1']}-{row['group2']}"
        assert pair.df is not None
        # The probe's statistic is test$statistic[level_j, level_i]: the same sign
        # as inferplot's q, mean(group j) - mean(group i).
        assert_close(
            "closed_form",
            pair.statistic,
            _exact(row["statistic_hex"]),
            what=f"{what} q",
        )
        assert_close(
            "closed_form", pair.df, _exact(row["welch_df_hex"]), what=f"{what} df"
        )
    return list(pairs)


@pytest.mark.parametrize(
    "entry", [pytest.param(e, id=e["case"]) for e in _gh_entries()]
)
def test_games_howell_raw_against_games_howell_test(entry: Row) -> None:
    case = _cases()[entry["case"]]
    pairs = _check_pairs(entry, case)
    if case.error is not None:
        # edge_n2: every Welch df is below 2 and R's ptukey returns NaN, so
        # ggstatsplot fails; inferplot's p is checked against mpmath instead.
        assert case.name == "ggbetweenstats-edge_n2-parametric"
        assert entry["r_warnings"] == ["NaNs produced"]
        refs = {
            tuple(p["pair"]): p
            for p in _reference_points("failed_fixture_pairs")
            if p["case"] == case.name
        }
        assert len(refs) == len(pairs) == 3
        for pair, row in zip(pairs, entry["pairs"], strict=True):
            assert row["p_value_hex"] == "NaN"
            assert pair.df is not None
            assert pair.df < 2.0
            ref = refs[(row["group1"], row["group2"])]
            assert ref["k"] == 3
            assert_close("closed_form", abs(pair.statistic), ref["q"], what="ref q")
            assert_close("closed_form", pair.df, ref["df"], what="ref df")
            assert_close(
                "studentized_range_tail",
                pair.p_value,
                float(Decimal(ref["p"])),
                what=f"{case.name} {row['group1']}-{row['group2']} raw p",
            )
        return
    for pair, row in zip(pairs, entry["pairs"], strict=True):
        assert_close(
            "games_howell_p",
            pair.p_value,
            _exact(row["p_value_hex"]),
            what=f"{case.name} {row['group1']}-{row['group2']} raw p",
        )


def test_games_howell_entries_cover_every_parametric_case_with_pairs() -> None:
    expected = {
        case.name
        for case in load_cases()
        if case.function == "ggbetweenstats"
        and case.type == "parametric"
        and len(case.analysis_levels) >= 3
    }
    assert {e["case"] for e in _gh_entries()} == expected


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(e, id=e["case"])
        for e in _gh_entries()
        if _cases()[e["case"]].error is None
        and _cases()[e["case"]].table("pairwise_comparisons_data")
    ],
)
def test_ggstatsplot_adjusted_p_is_p_adjust_of_games_howell_test(
    entry: Row,
) -> None:
    """ggstatsplot's adjusted p is exactly p.adjust of gamesHowellTest's raw p.

    With the case's p.adjust.method over all of the case's pairs, R's raw p-values
    adjusted by inferplot's ``p_adjust`` give the fixture's adjusted p within
    ``closed_form`` (the fixture keeps 15 significant digits). So every difference
    between inferplot's adjusted Games-Howell p and ggstatsplot's comes from the raw
    tail (DECISIONS.md D31), multiplied by the adjustment.
    """
    case = _cases()[entry["case"]]
    method = p_adjust_method(case)
    raw = [_exact(row["p_value_hex"]) for row in entry["pairs"]]
    adjusted = p_adjust(raw, method)
    fixture = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    assert len(fixture) == len(raw)
    for row, adj in zip(entry["pairs"], adjusted, strict=True):
        target = fixture[(row["group1"], row["group2"])]
        assert_close(
            "closed_form",
            adj,
            number(target["p.value"]),
            what=f"{case.name} {row['group1']}-{row['group2']} {method}",
        )


def test_games_howell_p_margin_is_ptukey_error_times_holm() -> None:
    """The tightest fixture pair under ``games_howell_p``: InsectSprays B-E.

    R's raw p is 2.0e-10 below the accurate one there (ptukey's error at p 3.7e-6);
    Holm multiplies it by 14, so inferplot's adjusted p is 2.79e-9 above
    ggstatsplot's, 92% of the 3.05e-9 the tolerance allows at p 5.1e-5.
    """
    tol = TOLERANCES["games_howell_p"]
    usage: list[tuple[float, str, str, str]] = []
    entries = {e["case"]: e for e in _gh_entries()}
    for name, case in _cases().items():
        if name not in entries or case.error is not None or not case.pairwise:
            continue
        levels = case.analysis_levels
        pairs = games_howell(groups(case), p_adjust_method=p_adjust_method(case))
        by_name = {(levels[p.i], levels[p.j]): p for p in pairs}
        for row in case.pairwise:
            g1, g2 = str(row["group1"]), str(row["group2"])
            pair = by_name.get((g1, g2)) or by_name[(g2, g1)]
            expected = number(row["p.value"])
            allowed = tol.atol + tol.rtol * abs(expected)
            usage.append((abs(pair.p_adjusted - expected) / allowed, name, g1, g2))
    assert len(usage) == 106
    share, name, g1, g2 = max(usage)
    assert (name, g1, g2) == ("ggbetweenstats-InsectSprays-parametric", "B", "E")
    assert share == pytest.approx(0.916, abs=2e-3)
    entry = entries[name]
    case = _cases()[name]
    raw_r = [_exact(row["p_value_hex"]) for row in entry["pairs"]]
    pairs = games_howell(groups(case), p_adjust_method="none")
    index = next(
        i
        for i, row in enumerate(entry["pairs"])
        if (row["group1"], row["group2"]) == ("B", "E")
    )
    rank = sorted(range(len(raw_r)), key=raw_r.__getitem__).index(index)
    multiplier = len(raw_r) - rank
    assert multiplier == 14
    raw_error = pairs[index].p_value - raw_r[index]
    assert raw_error == pytest.approx(2.0e-10, rel=0.01)
    holm = games_howell(groups(case))[index].p_adjusted
    fixture = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    shift = holm - number(fixture[("B", "E")]["p.value"])
    assert shift == pytest.approx(multiplier * raw_error, rel=1e-6)


# ------------------------------------------------------------------ omega squared

# Rows where effectsize 1.0.3's lower bound misses the accurate root (uniroot, tol
# 1e-12) by more than noncentral_ci: its root search stopped early. (F, df1, df2,
# ci) as the probe's JSON numbers give them. DECISIONS.md D33.
OMEGA_SEARCH_MISSES = {
    (10.0, 4.0, 5.5, 0.9),
    (10.0, 2.0, 18.0318456017699, 0.95),
    (10.0, 2.0, 18.03, 0.95),
}


def test_partial_omega_squared_against_effectsize() -> None:
    """probe-omega's rows with alternative "greater", ggstatsplot's construction.

    inferplot implements only that CI (one-sided, upper bound 1: what ggstatsplot
    1.1.4 shows for every ANOVA), so the probe's "less" and "two.sided" rows,
    which record effectsize's other alternatives, have nothing to compare with.

    Estimate within ``closed_form`` (in fact bit for bit); upper bound exactly 1.
    Lower bound: where effectsize agrees with the accurate root within
    ``noncentral_ci``, inferplot matches effectsize within ``noncentral_ci``;
    elsewhere (effectsize's search stopped early) inferplot matches the accurate
    root, and those rows are pinned.
    """
    rows = probe("probe-omega").rows("rows")
    assert Counter(r["alternative"] for r in rows) == Counter(
        {"greater": 577, "less": 577, "two.sided": 577}
    )
    misses: dict[tuple[float, float, float, float], float] = {}
    bitwise_estimates = 0
    for row in rows:
        if row["alternative"] != "greater":
            continue
        key = (float(row["f"]), float(row["df1"]), float(row["df2"]), float(row["ci"]))
        what = f"omega F={key[0]} df=({key[1]}, {key[2]}) ci={key[3]}"
        assert row["omega2_error"] is None
        assert row["omega2_warnings"] == []
        f, df2, ci = _exact(row["f_hex"]), _exact(row["df2_hex"]), _exact(row["ci_hex"])
        ours = partial_omega_squared(f, float(row["df1"]), df2, conf_level=ci)
        estimate = _exact(row["omega2"]["hex"])
        assert_close("closed_form", ours.estimate, estimate, what=what)
        bitwise_estimates += ours.estimate == estimate
        assert _exact(row["omega2_high"]["hex"]) == 1.0
        assert ours.conf_high == 1.0
        assert ours.sidedness == "greater"
        effectsize = _exact(row["omega2_low"]["hex"])
        accurate_low = row["accurate_low"]
        assert accurate_low["error"] is None
        accurate = _exact(accurate_low["bound"]["hex"])
        if is_close("noncentral_ci", effectsize, accurate):
            assert_close("noncentral_ci", ours.conf_low, effectsize, what=f"{what} low")
        else:
            misses[key] = abs(effectsize - accurate)
            assert_close(
                "noncentral_ci", ours.conf_low, accurate, what=f"{what} accurate low"
            )
    assert bitwise_estimates == 577  # the same arithmetic as effectsize's
    assert misses.keys() == OMEGA_SEARCH_MISSES
    worst = max(misses, key=misses.__getitem__)
    assert worst == (10.0, 4.0, 5.5, 0.9)
    assert misses[worst] == pytest.approx(2.08e-3, rel=0.01)


# ------------------------------------------------------------------ centrality

# The tie data of probe-format's "centrality" section, as oracle/generate.R defines
# it (tie_data); the probe stores only R's results.
TIE_DATA = {
    "a": (10.1, 9.2, 9.7, 10.4, 9.5, 9.9, 9.8, 9.68),
    "b": (12.2, 12.39, 12.1, 12.3, 12.31, 12.29, 12.6, 11.9),
}


def test_tie_data_is_the_generator_data() -> None:
    source = (ROOT / "oracle" / "generate.R").read_text(encoding="utf-8")
    assert "10.1, 9.2, 9.7, 10.4, 9.5, 9.9, 9.8, 9.68," in source
    assert "12.2, 12.39, 12.1, 12.3, 12.31, 12.29, 12.6, 11.9" in source


def test_mean_is_r_mean_bitwise() -> None:
    entry = next(
        e for e in probe("probe-format").rows("centrality") if e["type"] == "parametric"
    )
    assert len(entry["values_hex"]) == len(TIE_DATA)
    for values, text in zip(TIE_DATA.values(), entry["values_hex"], strict=True):
        assert mean(np.array(values)) == _exact(text)
