"""The noncentral chi-square distribution and the inversion behind the intervals of
Cramer's V and Pearson's C, against 40-digit mpmath (tests/data/ncx2_reference.json,
tools/reference/ncx2_mpmath.py)."""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from scipy import special, stats

from inferplot.stats import chi2_ncp_bound, chi2_ncp_interval, noncentral_chi2_logcdf
from tests.tolerances import assert_close

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "ncx2_reference.json"
# Tail probabilities checked as probabilities (both tails); below, as log P.
SMALLEST = 1e-20
# Points where inferplot's lower tail inherits the error of SciPy's gammainc at
# a = df / 2 + j near 5e5 (measured against mpmath: 3.0e-11 relative at P(505005,
# 500000) = 8.0e-13): x near 1e6, P(X' <= x) about 3e-7, 1.0e-9 and 8.5e-10
# relative; the upper tails there agree to 4e-16. Each is pinned at its measured
# error (with 10% headroom); every other point is within noncentral_cdf.
GAMMAINC_LIMITED = {
    ("1000000", 10, "1010000", "lower"): 1.1e-9,
    ("990000", 10, "1000000", "lower"): 0.95e-9,
}


@cache
def _reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def test_roots_match_the_mpmath_inversions() -> None:
    # 728 roots: x = 0.1 to 10,000, df = 1 to 50, every probability of a two- or
    # one-sided interval at levels 0.5, 0.9, 0.95 and 0.999. Measured: at most
    # 1.5e-13 relative (x = 50, df = 20, prob 0.9995, root 1.08).
    rows = _reference()["roots"]
    assert len(rows) == 728
    zeros = 0
    for row in rows:
        expected = float(row["ncp"])
        ours = chi2_ncp_bound(float(row["x"]), row["df"], prob=float(row["prob"]))
        if expected == 0.0:
            assert ours == 0.0, row
            zeros += 1
            continue
        assert math.isclose(ours, expected, rel_tol=2e-13, abs_tol=1e-14), (row, ours)
    assert 0 < zeros < len(rows)


def test_tails_match_mpmath() -> None:
    checked = 0
    for row in _reference()["cdf"]:
        args = (float(row["x"]), float(row["df"]), float(row["ncp"]))
        for side in ("lower", "upper"):
            expected_log = float(row[f"log10_{side}"]) * math.log(10.0)
            ours_log = noncentral_chi2_logcdf(*args, upper=side == "upper")
            key = (row["x"], row["df"], row["ncp"], side)
            if key in GAMMAINC_LIMITED:
                relative = abs(math.expm1(ours_log - expected_log))
                assert relative <= GAMMAINC_LIMITED[key], (key, relative)
                assert relative > 1e-10, (key, "no longer limited: tighten")
            elif expected_log >= math.log(SMALLEST):
                assert_close(
                    "noncentral_cdf",
                    math.exp(ours_log),
                    math.exp(expected_log),
                    what=f"{args} {side}",
                )
            else:
                assert_close("noncentral_cdf", ours_log, expected_log, what=f"{args}")
            checked += 1
    assert checked == 132


def test_central_case_is_scipys_chi_square() -> None:
    for x in (0.01, 0.7, 3.0, 40.0, 900.0):
        for df in (1.0, 2.0, 7.5, 60.0):
            lower = math.exp(noncentral_chi2_logcdf(x, df, 0.0))
            upper = math.exp(noncentral_chi2_logcdf(x, df, 0.0, upper=True))
            assert lower == pytest.approx(special.gammainc(df / 2, x / 2), rel=1e-14)
            assert upper == pytest.approx(special.gammaincc(df / 2, x / 2), rel=1e-13)


def test_moderate_values_agree_with_scipys_ncx2() -> None:
    # SciPy's ncx2 (cdflib) as a loose second opinion where it is reliable.
    for x, df, ncp in ((3.0, 2.0, 1.5), (12.0, 4.0, 6.0), (30.0, 9.0, 25.0)):
        ours = math.exp(noncentral_chi2_logcdf(x, df, ncp))
        assert ours == pytest.approx(float(stats.ncx2.cdf(x, df, ncp)), rel=1e-9)


def test_special_cases_and_validation() -> None:
    assert noncentral_chi2_logcdf(0.0, 3.0, 5.0) == -math.inf
    assert noncentral_chi2_logcdf(0.0, 3.0, 5.0, upper=True) == 0.0
    for bad in (
        (-1.0, 2.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 2.0, -0.5),
        (math.nan, 2.0, 0.0),
        (1.0, math.inf, 0.0),
    ):
        with pytest.raises(ValueError, match="must be"):
            noncentral_chi2_logcdf(*bad)
    with pytest.raises(ValueError, match="prob"):
        chi2_ncp_bound(2.0, 1.0, prob=0.0)
    with pytest.raises(ValueError, match="conf_level"):
        chi2_ncp_interval(2.0, 1.0, conf_level=1.0, alternative="two-sided")
    with pytest.raises(ValueError, match="alternative"):
        chi2_ncp_interval(2.0, 1.0, conf_level=0.95, alternative="two.sided")  # type: ignore[arg-type]


@pytest.mark.parametrize("x", [0.0, 0.3, 4.0, 85.0, 3000.0])
@pytest.mark.parametrize("df", [1.0, 4.0, 30.0])
def test_interval_sides(x: float, df: float) -> None:
    two = chi2_ncp_interval(x, df, conf_level=0.9, alternative="two-sided")
    greater = chi2_ncp_interval(x, df, conf_level=0.95, alternative="greater")
    less = chi2_ncp_interval(x, df, conf_level=0.95, alternative="less")
    # A two-sided 90% interval has the one-sided 95% bounds (up to the rounding of
    # 1 - 0.9 and 1 - 0.95, whose halves differ in the last bits).
    assert greater[1] == math.inf
    assert less[0] == 0.0
    assert greater[0] == pytest.approx(two[0], rel=1e-13, abs=1e-15)
    assert less[1] == pytest.approx(two[1], rel=1e-13)
    assert 0.0 <= two[0] <= two[1]
    if x == 0.0:
        assert two == (0.0, 0.0)
    # Each positive bound solves its equation.
    if two[0] > 0.0:
        upper = math.exp(noncentral_chi2_logcdf(x, df, two[0], upper=True))
        assert upper == pytest.approx(0.05, rel=1e-11)
    if two[1] > 0.0:
        lower = math.exp(noncentral_chi2_logcdf(x, df, two[1]))
        assert lower == pytest.approx(0.05, rel=1e-11)


def test_bounds_are_monotone_in_x_and_level() -> None:
    previous = (-1.0, -1.0)
    for x in (0.5, 2.0, 8.0, 32.0, 128.0, 512.0):
        low, high = chi2_ncp_interval(x, 3.0, conf_level=0.95, alternative="two-sided")
        assert low >= previous[0]
        assert high > previous[1]
        previous = (low, high)
    narrow = chi2_ncp_interval(40.0, 3.0, conf_level=0.5, alternative="two-sided")
    wide = chi2_ncp_interval(40.0, 3.0, conf_level=0.999, alternative="two-sided")
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


@pytest.mark.parametrize("x", [1e5, 1e7, 1e9, 1e12])
def test_huge_chi_square_follows_the_normal_limit(x: float) -> None:
    # Counts of 1e9 per cell give chi-squares this large. For large lambda, X' is
    # close to N(df + lambda, 2 (df + 2 lambda)): the bounds lie within a few
    # standard deviations of x - df, and each solves its equation.
    df = 3.0
    low, high = chi2_ncp_interval(x, df, conf_level=0.95, alternative="two-sided")
    sd = math.sqrt(2.0 * (df + 2.0 * x))
    assert x - df - 2.5 * sd < low < x - df - 1.5 * sd
    assert x - df + 1.5 * sd < high < x - df + 2.5 * sd
    upper = math.exp(noncentral_chi2_logcdf(x, df, low, upper=True))
    lower = math.exp(noncentral_chi2_logcdf(x, df, high))
    assert upper == pytest.approx(0.025, rel=1e-6)
    assert lower == pytest.approx(0.025, rel=1e-6)
