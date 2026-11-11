"""What R does where the reviews of Phases 3 and 4 could only reason about it.

Review round 3 of the oracle (generate.R's ROUND3_CASES, the round-3 entries of
HELD_OUT_CASES, and four probes). Each test states an answer as a fact about R 4.5.3
with ggstatsplot 1.1.4 and its packages, and re-computes it independently where a
method exists. Where inferplot differs today the case is held out (probe-heldout),
not marked as failing; the tests here say nothing about inferplot except where it
agrees with a probe that no fixture covers (the paired signed-rank test).

1. Signed-rank test, a value equal to the test value only as written (W9, D88,
   D105; probe-nearzero). ``wilcox.test`` drops a difference only when it is
   exactly 0: 0.1 + 0.2 - 0.3 = 2^-54 is ranked, the smallest of all, and so is
   0.7 - 0.4 - 0.3 = -2^-54. ggstatsplot prints what ``wilcox.test`` and
   ``rank_biserial`` give, n counting every value.
2. The bootstrap-t test in two writings (Z1, D105; probe-trimcibt). On the
   reviewer's ratings ``trimcibt`` gives the same result in whole numbers, in tenths
   and in whole numbers times 0.1, resample for resample (a T* of exactly 0 in whole
   numbers is below 1e-15 in tenths, nothing else moves): about 11% of T* are 0/0,
   almost none infinite, and the interval is NA because ``sort()`` drops the 0/0
   values while the bound keeps the position round(0.95 B) and p divides by all B.
3. Cramer's V at 19 to 60 df (B5; probe-bartables, edge_bar_wide*). R prints the
   adjusted V above its own upper bound, as inferplot does.
4. Exact zeros (C3; probe-bartables, edge_bar_exact_fit, held-out edge_bar_zero_v).
   R prints its rounding noise (chi-squared 1.13e-31, C 6.71e-17, V 5.59e-09).
5. Percent labels (D108; probe-round, edge_bar_ties-*). R's ``round(x, d)`` for d
   >= 1 picks between the two decimal candidates by their distance to the double,
   the even one on a tie of distances; for 98 values k / 2000 at one digit that is
   not Python's ``round()``.
6. One-sided Fisher pairs (edge_bar_reduced-*). A pair whose own table keeps two
   levels of y is a 2 x 2 table and gets the one-sided p of ``fisher.test``, with
   group1 (the earlier level) as its first row; 2 x 3 pairs stay two-sided. The
   rows come sorted as text by group1, then group2, also when x is declared in
   another order (held out: inferplot lists them in level order).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

import inferplot as ip
from tests.oracle_fixtures import (
    DATASET_DIR,
    OracleCase,
    Row,
    cases_for,
    number,
    probe,
    r_double,
)

MU = 0.3
TINY = 2.0**-54  # 0.1 + 0.2 - 0.3 and -(0.7 - 0.4 - 0.3)
CASE_SEED = 20261107


def _hex(text: object) -> float:
    value = r_double(str(text))
    return math.nan if value is None else value


def _hexes(texts: Sequence[object]) -> list[float]:
    return [_hex(t) for t in texts]


# 1. Signed-rank test ---


def _signed_rank(
    d: Sequence[float], *, paired: bool = False
) -> tuple[float, float, int, float, float, float]:
    """wilcox.test(exact = FALSE, correct = TRUE) as documented: differences exactly 0
    dropped, the others ranked by |d| exactly (average ranks), the normal
    approximation with the tie and continuity corrections; and the rank-biserial r
    with its Fisher-z interval, whose standard error counts the m nonzero
    differences of one sample and every pair of paired data (SPEC.md section 4)."""
    arr = np.asarray(d, dtype=float)
    kept = arr[arr != 0]
    m = int(kept.size)
    counted = arr.size if paired else m
    ranks = stats.rankdata(np.abs(kept))
    v = float(ranks[kept > 0].sum())
    _, ties = np.unique(ranks, return_counts=True)
    z = v - m * (m + 1) / 4
    sigma = math.sqrt(
        m * (m + 1) * (2 * m + 1) / 24 - float((ties**3 - ties).sum()) / 48
    )
    z = (z - math.copysign(0.5, z)) / sigma
    p = 2 * min(stats.norm.cdf(z), stats.norm.sf(z))
    total = m * (m + 1) / 2
    r = (2 * v - total) / total
    c = counted
    se = math.sqrt(c * (c + 1) * (2 * c + 1) / 6) / (c * (c + 1) / 2)
    q = stats.norm.ppf(0.975)
    low, high = math.tanh(math.atanh(r) - q * se), math.tanh(math.atanh(r) + q * se)
    return v, float(p), m, r, low, high


def _signif7(x: float) -> float:
    return float(f"{x:.7g}")


def _v_digits7(d: Sequence[float]) -> float:
    """V with digits.rank = 7: the nonzero |d| rounded to 7 significant digits
    before ranking."""
    arr = np.asarray(d, dtype=float)
    kept = arr[arr != 0]
    ranks = stats.rankdata([_signif7(abs(x)) for x in kept])
    return float(ranks[kept > 0].sum())


def _check_signed_rank(row: Row, d: list[float], n: int, ggstatsplot: str) -> None:
    v, p, m, r, low, high = _signed_rank(d, paired=ggstatsplot == "ggwithinstats")
    wilcox = row["wilcox"]
    assert wilcox["error"] is None
    assert _hex(wilcox["statistic_hex"]) == v
    assert math.isclose(_hex(wilcox["p_value_hex"]), p, rel_tol=1e-13)
    assert _hex(row["wilcox_digits7"]["statistic_hex"]) == _v_digits7(d)
    biserial = row["rank_biserial"]
    assert math.isclose(_hex(biserial["estimate_hex"]), r, rel_tol=1e-14)
    assert math.isclose(_hex(biserial["ci_low_hex"]), low, rel_tol=1e-12)
    assert math.isclose(_hex(biserial["ci_high_hex"]), high, rel_tol=1e-12)
    # ggstatsplot prints exactly what the two functions return, with n every value.
    shown = row[ggstatsplot]
    assert shown["error"] is None
    assert shown["statistic_hex"] == wilcox["statistic_hex"]
    assert shown["p_value_hex"] == wilcox["p_value_hex"]
    assert shown["estimate_hex"] == biserial["estimate_hex"]
    assert shown["conf_low_hex"] == biserial["ci_low_hex"]
    assert shown["conf_high_hex"] == biserial["ci_high_hex"]
    assert shown["n_obs"] == n
    assert row["exact_zeros"] == n - m


def test_one_sample_signed_rank_ranks_differences_of_2_to_the_minus_54() -> None:
    nearzero = probe("probe-nearzero")
    assert _hex(nearzero.section("mu_hex")) == MU
    rows = {row["column"]: row for row in nearzero.rows("one_sample")}
    assert list(rows) == ["literal", "sum", "below", "both", "sum_ties"]
    csv = pd.read_csv(DATASET_DIR / "edge_one_near_mu.csv", dtype=str)
    found: dict[str, tuple[float, int]] = {}
    for column, row in rows.items():
        x = _hexes(row["x_hex"])
        assert x == [float(text) for text in csv[column]]  # R read the CSV exactly
        d = _hexes(row["differences_hex"])
        assert d == [value - MU for value in x]
        _check_signed_rank(row, d, len(x), "gghistostats")
        found[column] = (_hex(row["wilcox"]["statistic_hex"]), int(row["exact_zeros"]))
    # The literal 0.3 is an exact zero and is dropped; 0.1 + 0.2 and 0.7 - 0.4 are
    # +-2^-54 from 0.3 and R ranks them first (V 49 and 48 where dropping them as
    # inferplot does gives 41).
    assert _hexes(rows["sum"]["differences_hex"])[0] == TINY
    assert _hexes(rows["below"]["differences_hex"])[0] == -TINY
    assert found == {
        "literal": (41.0, 1),
        "sum": (49.0, 0),
        "below": (48.0, 0),
        "both": (39.0, 1),
        "sum_ties": (48.0, 0),
    }


def test_held_out_histostats_cases_print_what_the_probe_records() -> None:
    nearzero = {
        row["column"]: row for row in probe("probe-nearzero").rows("one_sample")
    }
    held = {e["name"]: e["fixture"] for e in probe("probe-heldout").rows("cases")}
    fixtures = {c.name: c.raw for c in cases_for("gghistostats")}
    for column, row in nearzero.items():
        name = f"gghistostats-edge_one_near_mu-{column}-nonparametric"
        fixture = fixtures[name] if column == "literal" else held[name]
        (subtitle,) = fixture["stats"]["subtitle_data"]
        assert number(subtitle["statistic"]) == _hex(row["wilcox"]["statistic_hex"])
        assert math.isclose(
            number(subtitle["p.value"]),
            _hex(row["wilcox"]["p_value_hex"]),
            rel_tol=1e-14,
        )
        assert fixture["subtitle"] == row["gghistostats"]["subtitle"]
    assert (
        '"V")["Wilcoxon"] == "49.00"'
        in held["gghistostats-edge_one_near_mu-sum-nonparametric"]["subtitle"]
    )


def test_paired_signed_rank_ranks_a_difference_of_2_to_the_minus_54() -> None:
    row = probe("probe-nearzero").section("paired")
    a, b = _hexes(row["a_hex"]), _hexes(row["b_hex"])
    d = _hexes(row["differences_hex"])
    assert d == [x - y for x, y in zip(a, b, strict=True)]
    assert d[0] == TINY
    assert d[1] == 0.0
    _check_signed_rank(row, d, len(a), "ggwithinstats")
    assert _hex(row["wilcox"]["statistic_hex"]) == 35.5


def test_withinstats_agrees_with_the_paired_probe() -> None:
    # inferplot ranks a paired difference unless it is exactly 0, as R does (D64).
    row = probe("probe-nearzero").section("paired")
    frame = pd.read_csv(
        DATASET_DIR / "edge_paired_near_zero.csv", dtype={"value": str}
    ).assign(value=lambda f: f["value"].map(float))
    result = ip.analyze.withinstats(
        frame, "condition", "value", subject_id="subject", type="nonparametric"
    )
    shown = row["ggwithinstats"]
    test = result.test
    assert test is not None
    assert test.statistic == _hex(shown["statistic_hex"])
    assert math.isclose(test.p_value, _hex(shown["p_value_hex"]), rel_tol=1e-13)
    effect = result.effect_size
    assert effect is not None
    assert math.isclose(effect.estimate, _hex(shown["estimate_hex"]), rel_tol=1e-13)
    assert effect.conf_low is not None
    assert effect.conf_high is not None
    assert math.isclose(effect.conf_low, _hex(shown["conf_low_hex"]), rel_tol=1e-12)
    assert math.isclose(effect.conf_high, _hex(shown["conf_high_hex"]), rel_tol=1e-12)


# 2. The bootstrap-t test in three writings ---

TRIMCIBT_B = (200, 2000, 20000)
WRITINGS = ("whole", "tenths", "times_tenth")
SCALE = {"whole": 1.0, "tenths": 10.0, "times_tenth": 10.0}


def _winsorized_se(x: Sequence[float], tr: float = 0.2) -> float:
    """trimse: the winsorized SD / ((1 - 2 tr) sqrt(n)) (Wilcox 2012, section 4.3)."""
    s = np.sort(np.asarray(x, dtype=float))
    n = s.size
    g = math.floor(tr * n)
    w = np.clip(s, s[g], s[n - g - 1])
    return float(np.std(w, ddof=1)) / ((1 - 2 * tr) * math.sqrt(n))


def _trimcibt_rows() -> list[Row]:
    rows = probe("probe-trimcibt").rows("rows")
    assert len(rows) == 2 * len(WRITINGS) * len(TRIMCIBT_B) * 3
    return rows


def test_trimcibt_bounds_and_p_from_its_own_replicates() -> None:
    # p is the share of all B whose T* reaches |t| (a 0/0 T* is NaN and counts as
    # not reaching it); the bound is the round(0.95 B)-th smallest of the T* that
    # sort() keeps, NA when fewer remain.
    for row in _trimcibt_rows():
        B = int(row["B"])
        assert row["error"] is None
        assert row["unbalanced_traces"] == 0
        assert row["sorted_calls"] == 1
        assert row["nan"] + row["infinite"] + row["finite"] == B
        estimate, low, high, p, t = _hexes(row["values_hex"])
        if row["t_star_hex"] is None:
            assert B == 20000
            continue
        t_star = np.asarray(_hexes(row["t_star_hex"]))
        assert t_star.size == B
        assert int(np.isnan(t_star).sum()) == row["nan"]
        assert int(np.isinf(t_star).sum()) == row["infinite"]
        kept = np.sort(t_star[~np.isnan(t_star)])
        assert p == float(np.sum(abs(t) <= kept)) / B
        icrit = round(0.95 * B)
        x = _hexes(row["x_hex"])
        if icrit > kept.size:
            assert math.isnan(low)
            assert math.isnan(high)
            continue
        se = _winsorized_se(x)
        assert math.isclose(low, estimate - kept[icrit - 1] * se, rel_tol=1e-13)
        assert math.isclose(high, estimate + kept[icrit - 1] * se, rel_tol=1e-13)


def test_trimcibt_gives_the_same_result_in_every_writing() -> None:
    # The same resamples (one seed) in whole numbers, tenths and whole numbers x 0.1:
    # the same 0/0 resamples, the same p and the same interval to the scale. R's
    # mean() and var() of equal values are exact, and the trimmed mean of these
    # ratings in tenths is the double of the value as written, so each 0/0 resample
    # (every winsorized value equal to the trimmed mean) is 0/0 in every writing.
    by_key: dict[tuple[str, int, int], dict[str, Row]] = {}
    for row in _trimcibt_rows():
        by_key.setdefault((row["set"], row["B"], row["seed"]), {})[row["writing"]] = row
    for (_, B, _), rows in by_key.items():
        assert list(rows) == list(WRITINGS)
        whole = rows["whole"]
        w_values = _hexes(whole["values_hex"])
        for writing in WRITINGS[1:]:
            other = rows[writing]
            assert (other["nan"], other["infinite"]) == (
                whole["nan"],
                whole["infinite"],
            )
            values = _hexes(other["values_hex"])
            assert values[3] == w_values[3]  # p
            for mine, theirs in zip(values[:3], w_values[:3], strict=True):
                if math.isnan(theirs):
                    assert math.isnan(mine)
                else:
                    assert math.isclose(mine * SCALE[writing], theirs, rel_tol=1e-14)
            assert math.isclose(values[4], w_values[4], rel_tol=1e-14, abs_tol=0)
            if B <= 2000:
                mine_t = np.asarray(_hexes(other["t_star_hex"]))
                theirs_t = np.asarray(_hexes(whole["t_star_hex"]))
                assert np.array_equal(np.isnan(mine_t), np.isnan(theirs_t))
                assert np.array_equal(np.isinf(mine_t), np.isinf(theirs_t))
                # A resample whose trimmed mean is the data's gives T* = 0 exactly in
                # whole numbers and rounding noise below 1e-15 in tenths; it reaches
                # |t| in neither writing unless t is 0, where both count it.
                zero = theirs_t == 0
                assert zero.sum() > 0
                assert np.all(np.abs(mine_t[zero]) < 1e-15)
                rest = np.isfinite(theirs_t) & ~zero
                assert np.allclose(mine_t[rest], theirs_t[rest], rtol=1e-12, atol=0)


def test_trimcibt_interval_is_na_on_the_reviewers_ratings() -> None:
    # About 11% of the resamples are 0/0, so fewer than round(0.95 B) T* remain and
    # the interval is NA at every B and seed; the centred set (2.5% 0/0, t = 0
    # exactly) has an interval and p = (B - #0/0) / B.
    for row in _trimcibt_rows():
        B = int(row["B"])
        estimate, low, high, p, t = _hexes(row["values_hex"])
        if row["set"] == "ratings":
            assert 0.10 < row["nan"] / B < 0.15
            assert math.isnan(low)
            assert math.isnan(high)
            assert estimate * SCALE[row["writing"]] == pytest.approx(3.0, rel=1e-15)
        else:
            assert t == 0.0
            assert 0.01 < row["nan"] / B < 0.03
            assert p == (B - row["nan"]) / B
            assert math.isfinite(low)
            assert math.isfinite(high)
    seed_rows = {
        (r["set"], r["writing"]): r
        for r in _trimcibt_rows()
        if r["B"] == 200 and r["seed"] == CASE_SEED
    }
    assert seed_rows["ratings", "whole"]["nan"] == 28
    assert _hexes(seed_rows["ratings", "whole"]["values_hex"])[3] == 0.12
    assert _hexes(seed_rows["centred", "whole"]["values_hex"])[3] == 0.975


def test_held_out_trimcibt_cases_print_the_probe_at_200_resamples() -> None:
    # ggstatsplot's subtitle is trimcibt at nboot 200 after the case's seed: p 0.12 and
    # the interval [NA, NA] in both writings; p 0.975 ("0.97") at t = 0.
    probe_rows = {
        (r["set"], r["writing"]): r
        for r in _trimcibt_rows()
        if r["B"] == 200 and r["seed"] == CASE_SEED
    }
    held = {e["name"]: e["fixture"] for e in probe("probe-heldout").rows("cases")}
    columns = {
        "whole": ("ratings", "whole"),
        "tenths": ("ratings", "tenths"),
        "centred_whole": ("centred", "whole"),
        "centred_tenths": ("centred", "tenths"),
    }
    for column, key in columns.items():
        fixture = held[f"gghistostats-edge_one_ratings-{column}-robust"]
        (row,) = fixture["stats"]["subtitle_data"]
        _estimate, low, high, p, t = _hexes(probe_rows[key]["values_hex"])
        assert number(row["p.value"]) == p
        assert math.isclose(number(row["statistic"]), t, rel_tol=1e-14, abs_tol=0)
        if math.isnan(low):
            assert row["conf.low"] is None
            assert row["conf.high"] is None
            assert fixture["expected_na_values"] == ["subtitle"]
        else:
            assert math.isclose(number(row["conf.low"]), low, rel_tol=1e-14)
            assert math.isclose(number(row["conf.high"]), high, rel_tol=1e-14)


# 3. Cramer's V at large df ---


def _chisq_table(table: Sequence[Sequence[float]]) -> tuple[float, int, float]:
    t = np.asarray(table, dtype=float)
    chi2, p, df, _ = stats.chi2_contingency(t, correction=False)
    return float(chi2), int(df), float(p)


def _adjusted_v(chi2: float, n: float, r: int, k: int) -> float:
    """Bergsma (2013): phi^2 = max(0, chi^2 / n - (r - 1)(k - 1) / (n - 1)) over
    min(r~, k~) - 1, r~ = r - (r - 1)^2 / (n - 1)."""
    phi2 = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    shrunk = min(r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)) - 1
    return math.sqrt(phi2 / shrunk)


def _upper_ncp(chi2: float, df: int) -> float:
    """The noncentrality whose chi-square cdf at chi2 is 0.025."""
    return float(
        optimize.brentq(
            lambda ncp: stats.ncx2.cdf(chi2, df, ncp) - 0.025,
            1e-9,
            10 * chi2 + 100,
            xtol=1e-12,
        )
    )


def test_cramers_v_lies_above_its_upper_bound_on_wide_tables() -> None:
    # The estimate puts chi^2 into the adjusted formula, each bound a noncentrality
    # lambda; at large df chi^2 - df is about lambda, so chi^2 exceeds the upper
    # lambda and V its upper bound (or the bound is 0 and V is not).
    wide = {t["name"]: t for t in probe("probe-bartables").rows("wide")}
    assert list(wide) == [
        "edge_bar_wide20",
        "edge_bar_wide40",
        "edge_bar_wide61",
        "edge_bar_wide61r",
    ]
    printed: dict[str, tuple[float, float, float]] = {}
    for name, t in wide.items():
        table = t["table"]
        n = sum(map(sum, table))
        r, k = len(table), len(table[0])
        assert _hex(t["n_hex"]) == n
        chi2, df, p = _chisq_table(table)
        assert math.isclose(_hex(t["chisq"]["statistic_hex"]), chi2, rel_tol=1e-13)
        assert t["chisq"]["parameter"] == df == (r - 1) * (k - 1)
        assert math.isclose(_hex(t["chisq"]["p_value_hex"]), p, rel_tol=1e-10)
        adjusted, plain = t["cramers_v"]
        assert adjusted["adjust"] is True
        assert plain["adjust"] is False
        estimate = _hex(adjusted["estimate_hex"])
        assert math.isclose(estimate, _adjusted_v(chi2, n, r, k), rel_tol=1e-12)
        high = _hex(adjusted["ci_high_hex"])
        assert _hex(adjusted["ci_low_hex"]) == 0.0
        assert estimate > high
        ncp = _upper_ncp(chi2, df)
        assert ncp < chi2
        expected_high = _adjusted_v(ncp, n, r, k)
        assert math.isclose(high, expected_high, rel_tol=1e-5, abs_tol=1e-7)
        printed[name] = (round(estimate, 2), 0.0, round(high, 2))
    assert printed == {
        "edge_bar_wide20": (0.07, 0.0, 0.04),
        "edge_bar_wide40": (0.14, 0.0, 0.0),
        "edge_bar_wide61": (0.15, 0.0, 0.0),
        "edge_bar_wide61r": (0.33, 0.0, 0.28),
    }


def test_wide_table_fixtures_print_v_above_its_interval() -> None:
    cases = {c.name: c for c in cases_for("ggbarstats")}
    wide = {t["name"]: t for t in probe("probe-bartables").rows("wide")}
    for name, t in wide.items():
        case = cases[f"ggbarstats-{name}-parametric"]
        row = case.subtitle_data
        adjusted = t["cramers_v"][0]
        assert math.isclose(
            number(row["estimate"]), _hex(adjusted["estimate_hex"]), rel_tol=1e-14
        )
        assert math.isclose(
            number(row["conf.high"]),
            _hex(adjusted["ci_high_hex"]),
            rel_tol=1e-14,
            abs_tol=0,
        )
        assert number(row["estimate"]) > number(row["conf.high"])
    reviewer = cases["ggbarstats-edge_bar_wide61r-parametric"]
    assert '== "0.33", CI["95%"] ~ "[" * "0.00", "0.28" * "]"' in str(reviewer.subtitle)


# 4. Exact zeros printed as rounding noise ---


def _zeros() -> dict[str, Row]:
    rows = {t["name"]: t for t in probe("probe-bartables").rows("zeros")}
    assert list(rows) == [
        "edge_bar_zero_v",
        "edge_bar_exact_fit",
        "edge_bar_equal49",
        "big_2x2",
        "big_3x3",
    ]
    return rows


def test_corrected_v_of_an_exact_zero_is_rounding_noise_in_r() -> None:
    # a (0, 1), b (5, 4): chi^2 = 10/9 = df n / (n - 1), so phi^2 is exactly 0; R's
    # doubles leave chi^2 / n - 1/9 = 2^-56 and effectsize's V is 5.59e-09, which
    # ggstatsplot prints (held-out case; inferplot's rounding gives 3.95e-09).
    zero_v = _zeros()["edge_bar_zero_v"]
    assert _hex(zero_v["chisq"]["statistic_hex"]) == 10 / 9
    assert _hex(zero_v["phi2_corrected_hex"]) == 2.0**-56
    adjusted = zero_v["cramers_v"][0]
    v = _hex(adjusted["estimate_hex"])
    assert v == pytest.approx(5.587935447692872e-09, rel=1e-15)
    # V^2 (min(r~, k~) - 1) = 2^-55: effectsize's phi^2 is twice the noise above.
    assert v * v * (2 - 1 / 9 - 1) == pytest.approx(2.0**-55, rel=1e-12)
    table = zero_v["contingency_table"]
    assert '"V"))["Cramer"] == "5.59e-09"' in table["expression"]
    held = {e["name"]: e["fixture"] for e in probe("probe-heldout").rows("cases")}
    assert (
        held["ggbarstats-edge_bar_zero_v-parametric"]["subtitle"] == table["expression"]
    )


def _gof_statistic(x: Sequence[float], p: Sequence[float]) -> float:
    """chisq.test's goodness of fit as R forms it: E = n p, sum((x - E)^2 / E)."""
    n = math.fsum(x)
    expected = [n * share for share in p]
    return math.fsum((o - e) ** 2 / e for o, e in zip(x, expected, strict=True))


def test_exact_fits_print_rounding_noise_in_r() -> None:
    rows = _zeros()
    printed = {}
    for name in ("edge_bar_exact_fit", "edge_bar_equal49"):
        row = rows[name]
        x = [float(v) for v in row["x"]]
        p = _hexes(row["p_hex"])
        chi2 = _hex(row["chisq"]["statistic_hex"])
        assert chi2 == _gof_statistic(x, p)
        c = _hex(row["pearsons_c"]["estimate_hex"])
        assert c == math.sqrt(chi2 / (chi2 + math.fsum(x)))
        assert _hex(row["pearsons_c"]["ci_low_hex"]) == 0.0
        assert _hex(row["pearsons_c"]["ci_high_hex"]) == 0.0
        printed[name] = (chi2, c, row["contingency_table"]["expression"])
    assert printed["edge_bar_exact_fit"][:2] == (
        1.126944150315731e-31,
        6.713997766802521e-17,
    )
    assert printed["edge_bar_equal49"][:2] == (1.2079432611196747e-30, 2.0**-53)
    assert '"(" * 3 * ")" == "1.13e-31"' in printed["edge_bar_exact_fit"][2]
    assert '"C"))["Pearson"] == "6.71e-17"' in printed["edge_bar_exact_fit"][2]
    assert '"(" * 48 * ")" == "1.21e-30"' in printed["edge_bar_equal49"][2]
    assert '"C"))["Pearson"] == "1.11e-16"' in printed["edge_bar_equal49"][2]
    # The fixture of the exact fit prints the same; ggbarstats draws no more than 24
    # levels, so for 49 the subtitle comes from contingency_table() alone.
    (case,) = [
        c
        for c in cases_for("ggbarstats")
        if c.name.startswith("ggbarstats-edge_bar_exact_fit")
    ]
    assert case.subtitle == printed["edge_bar_exact_fit"][2]
    stop = rows["edge_bar_equal49"]["ggbarstats"]["error"]
    assert "must be a whole number between 1 and 24, not the number 49" in stop


def test_counts_beyond_an_integer_stop_ggbarstats() -> None:
    # Cells of about 1e10 given through counts: tidyr::uncount() cannot repeat a row
    # more than .Machine$integer.max times, so ggbarstats stops before any test. The
    # tests on the table: an exactly independent 2 x 2 gives chi^2 0; the 3 x 3 (1e9
    # r_i c_j) gives 1.30e-21 from R's rounding of the expected counts; the adjusted
    # V is 0 for both (phi^2 below 0 is set to 0).
    rows = _zeros()
    for name in ("big_2x2", "big_3x3"):
        row = rows[name]
        error = row["ggbarstats"]["error"]
        assert "Can't convert from `weights` <double> to <integer>" in error
        assert _hex(row["cramers_v"][0]["estimate_hex"]) == 0.0
        assert _hex(row["phi2_corrected_hex"]) < 0
    assert _hex(rows["big_2x2"]["chisq"]["statistic_hex"]) == 0.0
    assert _hex(rows["big_3x3"]["chisq"]["statistic_hex"]) == pytest.approx(
        1.3008530279903701e-21, rel=1e-15
    )


# 5. R's round() on percent labels ---


def r_round(x: float, digits: int) -> float:
    """R >= 4.0.0's round(x, digits) as its documentation describes it: for digits
    >= 1 the two candidates, x rounded down and up to that decimal digit, and the
    one nearer to x in double arithmetic; on a tie of the two distances the one with
    an even last digit. digits = 0 rounds half to even (IEEE nearbyint)."""
    if digits == 0:
        return float(round(x))
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    scale = 10.0**digits
    down = math.floor(x * scale)
    low, high = down / scale, math.ceil(x * scale) / scale
    up_distance, down_distance = high - x, x - low
    if up_distance < down_distance or (up_distance == down_distance and down % 2 == 1):
        return sign * high
    return sign * low


def _label(value: float) -> str:
    return f"{format(value, '.15g')}%"


def test_r_round_at_decimal_ties() -> None:
    grid = probe("probe-round").rows("grid")
    assert len(grid) == 20 + 100 + 100 + 100 + 1000
    differ: dict[tuple[int, int], list[int]] = {}
    for row in grid:
        n, k = int(row["n"]), int(row["k"])
        perc = _hex(row["perc_hex"])
        assert perc == (k / n) * 100
        for d in range(3):
            rounded = _hex(row["rounded_hex"][d])
            assert rounded == r_round(perc, d), (n, k, d)
            assert row["labels"][d] == _label(rounded)
            if rounded != round(perc, d):
                differ.setdefault((n, d), []).append(k)
    # Python's round() decides on the exact binary value, R on the two candidates'
    # distances in doubles: they part on 98 of the 1,000 one-digit ties of k / 2000
    # (1/2000 is 0.05000000000000000277 and Python gives 0.1, R 0).
    assert list(differ) == [(2000, 1)]
    assert len(differ[2000, 1]) == 98
    assert differ[2000, 1][:6] == [1, 21, 39, 89, 91, 109]


def test_ggbarstats_cell_labels_follow_r_round() -> None:
    paths = probe("probe-round").rows("label_path")
    assert [(p["n"], p["digits_perc"]) for p in paths] == [
        (n, d) for n in (40, 200, 1000, 2000) for d in range(3)
    ]
    cells = 0
    for path in paths:
        assert path["error"] is None
        for cell in path["cells"]:
            perc = _hex(cell["perc_hex"])
            assert perc == (cell["counts"] / path["n"]) * 100
            assert cell["label"] == _label(r_round(perc, path["digits_perc"]))
            cells += 1
    assert cells == 3 * 2 * (10 + 50 + 50 + 550)


def test_held_out_tie_labels_are_r_rounds() -> None:
    held = {e["name"]: e["fixture"] for e in probe("probe-heldout").rows("cases")}
    fixture = held["ggbarstats-edge_bar_ties-digitsperc1-parametric"]
    labels = {
        (r["y"], r["x"]): r[".label"] for r in fixture["stats"]["descriptive_data"]
    }
    assert labels["n2000k001", "a"] == "0%"  # Python's round() gives 0.1
    assert labels["n2000k003", "b"] == "99.9%"  # 99.85: Python 99.8
    assert labels["n2000k089", "a"] == "4.4%"  # 4.45: Python 4.5
    assert labels["n2000k091", "a"] == "4.6%"  # 4.55: Python 4.5


# 6. One-sided Fisher pairs on reduced tables ---

REDUCED = {"a": (6, 1, 0), "b": (2, 7, 0), "c": (5, 0, 2), "d": (1, 0, 6)}


def _fisher_two_sided(table: list[list[int]]) -> float:
    """fisher.test's two-sided p of a 2 x c table: the probability of every table
    with the same margins at most as probable as the observed one (relative
    tolerance 1e-7), by enumeration."""
    rows = [sum(r) for r in table]
    cols = [table[0][j] + table[1][j] for j in range(len(table[0]))]
    n = sum(rows)

    def prob(first: Sequence[int]) -> float:
        num = math.prod(math.comb(c, x) for c, x in zip(cols, first, strict=True))
        return num / math.comb(n, rows[0])

    observed = prob(table[0])
    total = 0.0
    for first in itertools.product(*(range(c + 1) for c in cols)):
        if sum(first) == rows[0]:
            q = prob(first)
            if q <= observed * (1 + 1e-7):
                total += q
    return total


def _pair_p(group1: str, group2: str, alternative: str) -> tuple[float, int]:
    columns = [j for j in range(3) if REDUCED[group1][j] + REDUCED[group2][j] > 0]
    table = [[REDUCED[g][j] for j in columns] for g in (group1, group2)]
    if len(columns) == 2 and alternative != "two.sided":
        side = "greater" if alternative == "greater" else "less"
        return float(stats.fisher_exact(table, alternative=side).pvalue), 2
    return _fisher_two_sided(table), len(columns)


def _reduced_cases() -> dict[tuple[bool, str], OracleCase]:
    """The edge_bar_reduced cases: fixtures in text order, held out (probe-heldout)
    with x declared d, b, c, a."""
    heldout = probe("probe-heldout")
    held = [
        OracleCase(name=e["name"], path=heldout.path, raw=e["fixture"])
        for e in heldout.rows("cases")
    ]
    out: dict[tuple[bool, str], OracleCase] = {}
    for case in (*cases_for("ggbarstats"), *held):
        if case.name.startswith("ggbarstats-edge_bar_reduced-"):
            declared = "-declared-" in case.name
            assert declared == (case in held)
            out[declared, str(case.args.get("alternative"))] = case
    assert sorted(out) == sorted(
        (d, a) for d in (False, True) for a in ("two.sided", "greater", "less")
    )
    return out


def test_one_sided_fisher_pairs_on_reduced_tables() -> None:
    for (declared, alternative), case in _reduced_cases().items():
        levels = ["d", "b", "c", "a"] if declared else ["a", "b", "c", "d"]
        assert list(case.analysis_levels) == levels
        pairs = case.pairwise
        # group1 is the earlier level; the rows are sorted by group1, then group2,
        # as text.
        assert [(r["group1"], r["group2"]) for r in pairs] == sorted(
            (min(g, h, key=levels.index), max(g, h, key=levels.index))
            for g, h in itertools.combinations(levels, 2)
        )
        for row in pairs:
            expected, width = _pair_p(row["group1"], row["group2"], alternative)
            assert math.isclose(number(row["p.value"]), expected, rel_tol=1e-12), (
                case.name,
                row["group1"],
                row["group2"],
                width,
            )


def test_which_pair_row_is_greater() -> None:
    # The 2 x 2 pairs a-b (p, q) and c-d (p, r): "greater" is the odds ratio of
    # group1's first column over group2's, so declaring d, b, c, a turns a-b into
    # b-a and c-d into d-c, and their greater and less p swap.
    p = {
        (declared, alt, row["group1"], row["group2"]): number(row["p.value"])
        for (declared, alt), case in _reduced_cases().items()
        for row in case.pairwise
    }
    assert p[False, "greater", "a", "b"] == pytest.approx(0.0202797202797203, rel=1e-12)
    assert p[False, "less", "a", "b"] == pytest.approx(0.999300699300699, rel=1e-12)
    assert p[True, "greater", "b", "a"] == p[False, "less", "a", "b"]
    assert p[True, "less", "b", "a"] == p[False, "greater", "a", "b"]
    assert p[False, "greater", "c", "d"] == pytest.approx(0.0512820512820513, rel=1e-12)
    assert p[True, "greater", "d", "c"] == p[False, "less", "c", "d"]
    # The 2 x 3 pairs keep the two-sided p whatever alternative says.
    for declared in (False, True):
        for g1, g2 in (("a", "c"), ("b", "c"), ("a", "d"), ("b", "d")):
            if declared:
                g1, g2 = sorted((g1, g2), key=["d", "b", "c", "a"].index)
            values = {
                p[declared, alt, g1, g2] for alt in ("two.sided", "greater", "less")
            }
            assert len(values) == 1
