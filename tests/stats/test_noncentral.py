"""Noncentral t and F distribution functions and their noncentrality inversion."""

from __future__ import annotations

import itertools
import json
import math
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from scipy import special, stats

from inferplot.stats import f_ncp_bound, t_ncp_interval
from inferplot.stats.noncentral import noncentral_f_logcdf, noncentral_t_logcdf
from tests.tolerances import assert_close

REFERENCE = Path(__file__).resolve().parents[1] / "data" / "noncentral_reference.json"
# Tail probabilities checked against the references (both tails).
SMALLEST = 1e-20


@cache
def _reference() -> dict[str, Any]:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


def _check_tails(name: str, logcdf: Any, keys: tuple[str, ...]) -> int:
    checked = 0
    for point in _reference()["sets"][name]["points"]:
        args = [float(point[k]) for k in keys]
        for side in ("lower", "upper"):
            expected = float(Decimal(point[side]))
            if expected < SMALLEST:
                continue
            ours = math.exp(logcdf(*args, upper=side == "upper"))
            assert_close("noncentral_cdf", ours, expected, what=f"{name} {args} {side}")
            checked += 1
    return checked


def test_noncentral_t_matches_high_precision_reference() -> None:
    assert _check_tails("noncentral_t", noncentral_t_logcdf, ("t", "df", "ncp")) > 500


def test_noncentral_f_matches_high_precision_reference() -> None:
    keys = ("f", "df1", "df2", "ncp")
    assert _check_tails("noncentral_f", noncentral_f_logcdf, keys) > 300


def test_noncentral_t_special_cases() -> None:
    # ncp = 0 is Student's t; t = 0 gives Phi(-ncp); df = inf is N(ncp, 1).
    for t, df in [(-3.0, 4.0), (0.7, 1.0), (2.5, 37.5), (12.0, 300.0)]:
        expected = float(special.stdtr(df, t))
        assert math.exp(noncentral_t_logcdf(t, df, 0.0)) == pytest.approx(
            expected, rel=1e-13
        )
    assert noncentral_t_logcdf(0.0, 5.0, 1.3) == pytest.approx(
        float(special.log_ndtr(-1.3)), rel=1e-15
    )
    assert noncentral_t_logcdf(2.0, math.inf, 0.5, upper=True) == pytest.approx(
        float(special.log_ndtr(-1.5)), rel=1e-15
    )


def test_noncentral_f_special_cases() -> None:
    for f, d1, d2 in [(0.5, 1.0, 3.5), (2.0, 3.0, 17.13), (30.0, 2.0, 200.0)]:
        expected = float(special.fdtr(d1, d2, f))
        assert math.exp(noncentral_f_logcdf(f, d1, d2, 0.0)) == pytest.approx(
            expected, rel=1e-12
        )
    assert noncentral_f_logcdf(0.0, 2.0, 10.0, 3.0) == -math.inf
    assert noncentral_f_logcdf(0.0, 2.0, 10.0, 3.0, upper=True) == 0.0


# ------------------------------------------------------------------- t inversion


def _lower_tail(t: float, df: float, ncp: float) -> float:
    return math.exp(noncentral_t_logcdf(t, df, ncp))


@pytest.mark.parametrize(
    ("t", "df"),
    [(1.9, 55.3), (-3.7, 18.3), (0.0, 7.0), (0.02, 2.0), (25.0, 98.0), (-8.0, 1.0)],
)
@pytest.mark.parametrize("level", [0.5, 0.9, 0.95, 0.999])
def test_two_sided_bounds_hit_their_tail_probabilities(
    t: float, df: float, level: float
) -> None:
    low, high = t_ncp_interval(t, df, conf_level=level, alternative="two-sided")
    tail = (1.0 - level) / 2.0
    assert low < t * (1 - 1 / (4 * df)) < high or t == 0.0
    upper_at_low = math.exp(noncentral_t_logcdf(t, df, low, upper=True))
    assert upper_at_low == pytest.approx(tail, rel=1e-10)
    assert _lower_tail(t, df, high) == pytest.approx(tail, rel=1e-10)


def test_one_sided_bounds() -> None:
    t, df = 1.915, 55.31
    greater = t_ncp_interval(t, df, conf_level=0.95, alternative="greater")
    less = t_ncp_interval(t, df, conf_level=0.95, alternative="less")
    two_90 = t_ncp_interval(t, df, conf_level=0.90, alternative="two-sided")
    assert greater[1] == math.inf
    assert less[0] == -math.inf
    # A one-sided 95% bound is the matching bound of the two-sided 90% interval.
    assert greater[0] == pytest.approx(two_90[0], rel=1e-12)
    assert less[1] == pytest.approx(two_90[1], rel=1e-12)


def test_interval_is_antisymmetric_in_t_and_widens_with_level() -> None:
    low, high = t_ncp_interval(2.3, 14.2, conf_level=0.95, alternative="two-sided")
    neg = t_ncp_interval(-2.3, 14.2, conf_level=0.95, alternative="two-sided")
    assert neg == pytest.approx((-high, -low), rel=1e-12)
    widths = [
        t_ncp_interval(2.3, 14.2, conf_level=c, alternative="two-sided")
        for c in (0.5, 0.8, 0.9, 0.95, 0.99, 0.9999)
    ]
    assert all(a[0] > b[0] and a[1] < b[1] for a, b in itertools_pairwise(widths))


def itertools_pairwise(items: list[Any]) -> list[tuple[Any, Any]]:
    return list(itertools.pairwise(items))


def test_normal_limit() -> None:
    z = float(stats.norm.isf(0.025))
    assert t_ncp_interval(1.5, math.inf, conf_level=0.95, alternative="two-sided") == (
        pytest.approx(1.5 - z, rel=1e-12),
        pytest.approx(1.5 + z, rel=1e-12),
    )
    # Very large df approaches it.
    low, high = t_ncp_interval(1.5, 1e9, conf_level=0.95, alternative="two-sided")
    assert (low, high) == (
        pytest.approx(1.5 - z, rel=1e-8),
        pytest.approx(1.5 + z, rel=1e-8),
    )


@pytest.mark.parametrize("t", [-1e6, -1e4, -500.0, 500.0, 1e4, 1e6])
@pytest.mark.parametrize("df", [1.0, 2.0, 10.0, 1000.0, 1e6])
def test_extreme_t_is_finite_and_matches_the_large_ncp_limit(
    t: float, df: float
) -> None:
    # SciPy's nctdtr returns NaN far in the tails (1.18) or loses accuracy (1.11);
    # the inversion here uses its own distribution function and stays finite.
    low, high = t_ncp_interval(t, df, conf_level=0.95, alternative="two-sided")
    assert math.isfinite(low)
    assert math.isfinite(high)
    assert low < high
    # For |ncp| >> 1, T' ~ ncp / S with S = chi_df / sqrt(df), so P(T' <= t) = p
    # gives ncp = t * sqrt(chi2_df quantile / df) (up to O(1 / |ncp|)).
    s_lo = math.sqrt(float(stats.chi2.ppf(0.025, df)) / df)
    s_hi = math.sqrt(float(stats.chi2.isf(0.025, df)) / df)
    expected = sorted((t * s_lo, t * s_hi))
    assert low == pytest.approx(expected[0], rel=5e-3, abs=3.0)
    assert high == pytest.approx(expected[1], rel=5e-3, abs=3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"t": math.nan, "df": 5.0}, "finite"),
        ({"t": math.inf, "df": 5.0}, "finite"),
        ({"t": 1.0, "df": 0.0}, "df must"),
        ({"t": 1.0, "df": 5.0, "conf_level": 1.0}, "conf_level"),
        ({"t": 1.0, "df": 5.0, "conf_level": 0.0}, "conf_level"),
        ({"t": 1.0, "df": 5.0, "alternative": "two.sided"}, "alternative"),
    ],
)
def test_t_interval_validation(kwargs: dict[str, Any], message: str) -> None:
    args: dict[str, Any] = {"conf_level": 0.95, "alternative": "two-sided", **kwargs}
    with pytest.raises(ValueError, match=message):
        t_ncp_interval(**args)


# ------------------------------------------------------------------- F inversion


@pytest.mark.parametrize(
    ("f", "df1", "df2"),
    [(5.18, 2.0, 17.13), (31.6, 4.0, 22.3), (1000.0, 2.0, 97.4), (3.0, 1.0, 3.5)],
)
@pytest.mark.parametrize("prob", [0.05, 0.5, 0.95, 0.999])
def test_f_bound_hits_its_probability(
    f: float, df1: float, df2: float, prob: float
) -> None:
    ncp = f_ncp_bound(f, df1, df2, prob=prob)
    if ncp == 0.0:
        assert math.exp(noncentral_f_logcdf(f, df1, df2, 0.0)) <= prob
        return
    if prob > 0.5:
        reached = math.exp(noncentral_f_logcdf(f, df1, df2, ncp, upper=True))
        assert reached == pytest.approx(1.0 - prob, rel=1e-10)
    else:
        assert math.exp(noncentral_f_logcdf(f, df1, df2, ncp)) == pytest.approx(
            prob, rel=1e-10
        )


def test_f_bound_is_zero_when_lambda_zero_is_already_below() -> None:
    assert f_ncp_bound(0.0, 2.0, 10.0, prob=0.95) == 0.0
    assert f_ncp_bound(0.4, 2.0, 10.0, prob=0.95) == 0.0
    assert f_ncp_bound(1e5, 3.0, 50.0, prob=0.95) > 2e5  # huge F stays finite
    with pytest.raises(ValueError, match="prob"):
        f_ncp_bound(2.0, 2.0, 10.0, prob=1.0)
    with pytest.raises(ValueError, match="finite"):
        f_ncp_bound(math.inf, 2.0, 10.0, prob=0.5)
