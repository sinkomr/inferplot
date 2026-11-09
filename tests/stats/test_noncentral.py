"""Noncentral t and F distribution functions and their noncentrality inversion."""

from __future__ import annotations

import itertools
import json
import math
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy
from scipy import special, stats

from inferplot.stats import f_ncp_bound, t_ncp_interval
from inferplot.stats.noncentral import (
    _log_gamma_ratio,  # pyright: ignore[reportPrivateUsage]
    log_beta_factors,
    noncentral_f_logcdf,
    noncentral_t_logcdf,
)
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


def test_noncentral_f_at_large_ncp_matches_the_integral_reference() -> None:
    # NEW-2: Poisson means 1e5 to 1e12, both tails from 1e-3 to 1e-20
    # (references by the chi'^2 x chi^2 integral, which shares nothing with the
    # Poisson sum).
    keys = ("f", "df1", "df2", "ncp")
    assert _check_tails("noncentral_f_large", noncentral_f_logcdf, keys) >= 60


def test_noncentral_f_at_large_df2_matches_the_exact_sum() -> None:
    # NEW-5: df2 = 1e6 and 1e8, where SciPy's betainc with b = 5e7 is off by
    # about 1e-10 (exact negative-binomial sums for df1 = 2).
    keys = ("f", "df1", "df2", "ncp")
    assert _check_tails("noncentral_f_df1_two", noncentral_f_logcdf, keys) >= 40


def test_noncentral_f_far_in_its_tails_matches_the_integral_reference() -> None:
    # NEW-4: log P down to -2000, where SciPy's betainc underflows (it returned 0
    # below about 1e-280 and log P came out 12 too small at -677.7). Compared as
    # log probabilities (the probabilities are below the doubles).
    checked = 0
    for point in _reference()["sets"]["noncentral_f_deep"]["points"]:
        args = [float(point[k]) for k in ("f", "df1", "df2", "ncp")]
        for side in ("lower", "upper"):
            expected = float(Decimal(point[f"log_{side}"]))
            if expected > math.log(SMALLEST):
                continue  # within the range of the other reference tests
            ours = noncentral_f_logcdf(*args, upper=side == "upper")
            assert_close("noncentral_cdf", ours, expected, what=f"deep {args} {side}")
            checked += 1
    assert checked >= 8


@pytest.mark.parametrize(
    ("f", "df1", "ncp"),
    [
        (1e-100, 2.0, 2e5),  # NEW-3: was log P = 0 (a NaN taken as P = 1)
        (1e-300, 4.0, 1e3),
        (0.5, 2.0, 2e5),
        (1e-3, 7.0, 3e7),
        (2.0, 2.0, 40.0),
        (1e5, 3.0, 2e5),
        (3e5, 2.5, 1e11),
        (4.0, 9.0, 1e14),
    ],
)
def test_noncentral_f_with_two_denominator_df_is_closed_form(
    f: float, df1: float, ncp: float
) -> None:
    # df2 = 2 makes every beta factor I_x(a, 1) = x^a, so the Poisson mixture sums
    # to P(F' <= f) = x^(df1 / 2) exp(-ncp (1 - x) / 2), x = df1 f / (df1 f + 2):
    # an exact reference at any ncp and any depth.
    x = df1 * f / (df1 * f + 2.0)
    y = 2.0 / (df1 * f + 2.0)
    log_lower = 0.5 * df1 * math.log(x) - 0.5 * ncp * y
    log_upper = math.log(-math.expm1(log_lower))
    ours_lower = noncentral_f_logcdf(f, df1, 2.0, ncp)
    ours_upper = noncentral_f_logcdf(f, df1, 2.0, ncp, upper=True)
    assert_close("noncentral_cdf", ours_lower, log_lower, what="log lower")
    if log_upper < -1e-300:
        assert_close("noncentral_cdf", ours_upper, log_upper, what="log upper")
    else:
        assert ours_upper == pytest.approx(log_upper, abs=1e-15)


def test_a_nan_is_never_a_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    # NEW-3: min(0.0, nan) is 0.0, so a NaN beta factor read as P = 1.
    from inferplot.stats import noncentral

    def nan_factors(a: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.full_like(a, math.nan)

    monkeypatch.setattr(noncentral, "log_beta_factors", nan_factors)
    with pytest.raises(ArithmeticError, match="not a number"):
        noncentral_f_logcdf(2.0, 3.0, 10.0, 5.0)


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


@pytest.mark.parametrize("t", [1e150, 2.0**512, 1e200, 1e300, -1e300])
@pytest.mark.parametrize("df", [1.0, 2.0, 30.0, 1e6, math.inf])
@pytest.mark.parametrize("side", ["two-sided", "greater", "less"])
def test_huge_t_matches_the_large_ncp_limit(t: float, df: float, side: Any) -> None:
    # Was "t and ncp must be finite; got 1e+200 and -inf": the start value squared
    # t. For |ncp| >> 1, P(T' <= t) = p at ncp = t sqrt(q / df), q the chi-square
    # quantile (the normal shift t -+ z at df = inf).
    low, high = t_ncp_interval(t, df, conf_level=0.95, alternative=side)
    tail = 0.025 if side == "two-sided" else 0.05
    if math.isinf(df):
        z = float(stats.norm.isf(tail))
        expected = (t - z, t + z)
    else:
        s_lo = math.sqrt(float(stats.chi2.ppf(tail, df)) / df)
        s_hi = math.sqrt(float(stats.chi2.isf(tail, df)) / df)
        expected = tuple(sorted((t * s_lo, t * s_hi)))
    if side != "less":
        assert low == pytest.approx(expected[0], rel=4e-15)
    else:
        assert low == -math.inf
    if side != "greater":
        assert high == pytest.approx(expected[1], rel=4e-15)
    else:
        assert high == math.inf


@pytest.mark.parametrize("t", [1.0000000001e300, -1e301, 1.7e308])
def test_t_beyond_1e300_raises_plainly(t: float) -> None:
    with pytest.raises(ValueError, match=r"too large for a confidence interval") as e:
        t_ncp_interval(t, 4.0, conf_level=0.95, alternative="two-sided")
    assert "inf" not in str(e.value)


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


# SciPy before 1.12 computes betainc with older code whose error at these a
# (1e4 to 1e7) is noise of up to about 1e-10 relative per value; it does not cancel
# between the subsampled and the full sum (measured: 4.7e-10 on SciPy 1.11.1,
# 2.4e-14 on 1.12.0 and 1.18.1). The subsampling itself is checked at 1e-12.
OLD_BETAINC = tuple(int(p) for p in scipy.__version__.split(".")[:2]) < (1, 12)
SUBSAMPLE_TOL = 1e-9 if OLD_BETAINC else 1e-12


def _compensated_cumsum(terms: list[float]) -> list[float]:
    """Running sums with Neumaier's compensation."""
    out: list[float] = []
    total, carry = 0.0, 0.0
    for term in terms:
        t = total + term
        if abs(total) >= abs(term):
            carry += (total - t) + term
        else:
            carry += (term - t) + total
        total = t
        out.append(total + carry)
    return out


def _log_poisson_ratios(mu: float, lo: int, hi: int) -> np.ndarray:
    """log(w_j / w_m), m = floor(mu), j = lo..hi, summed outward from the mode:
    log(w_j / w_m) = -sum_{i=m+1}^{j} log1p((i - mu) / mu) above the mode and
    sum_{i=j+1}^{m} log1p((i - mu) / mu) below it. The terms are small near the
    mode and each is accurate, so the sums keep about 1e-14 absolute accuracy,
    where -mu + j log mu - log j! loses about 1e-10 at mu = 1e5."""
    m = math.floor(mu)
    up = _compensated_cumsum([-math.log1p((i - mu) / mu) for i in range(m + 1, hi + 1)])
    down = _compensated_cumsum([math.log1p((i - mu) / mu) for i in range(m, lo, -1)])
    return np.array([*reversed(down), 0.0, *up])


def _full_poisson_sum(
    f: float, df1: float, df2: float, ncp: float, upper: bool
) -> float:
    """log P(F' <= f) (or > f) summed over every integer j of the Poisson window,
    with independently computed weights (``_log_poisson_ratios``)."""
    mu = 0.5 * ncp
    span = math.ceil(15.0 * math.sqrt(mu) + 30.0)
    lo, hi = max(0, math.floor(mu) - span), math.floor(mu) + span
    j = np.arange(lo, hi + 1)
    rel = _log_poisson_ratios(mu, lo, hi)
    log_w = rel - special.logsumexp(rel)
    denom = df1 * f + df2
    # The same beta factors as noncentral_f_logcdf, at every integer j: the test
    # checks the subsampling and the weights, not SciPy's betainc (whose error,
    # up to 1e-10 relative at these a in SciPy 1.11, would otherwise not cancel).
    log_i = log_beta_factors(
        0.5 * df1 + j.astype(np.float64),
        0.5 * df2,
        df1 * f / denom,
        df2 / denom,
        df2 / (df1 * f),
        upper=upper,
    )
    return float(special.logsumexp(log_w + log_i))


@pytest.mark.parametrize("mu", [1.2e4, 1e5, 2.5e6])
@pytest.mark.parametrize(("df1", "df2"), [(2.0, 6.77), (5.0, 1.5), (3.0, 2300.0)])
@pytest.mark.parametrize("spread", [-2.0, 0.0, 1.5])
@pytest.mark.parametrize("upper", [False, True])
def test_subsampled_poisson_sum_matches_the_full_sum(
    mu: float, df1: float, df2: float, spread: float, upper: bool
) -> None:
    # From mu = 1e4 on every h-th term is summed (h = floor(sqrt(mu) / 8)): the
    # trapezoidal rule on an analytic summand. f is placed `spread` (roughly in
    # SDs of log F') from the median of F'.
    ncp = 2.0 * mu
    median = (df1 + ncp) / df1
    f = median * math.exp(spread * math.sqrt(2.0 / df2 + 4.0 / ncp))
    ours = noncentral_f_logcdf(f, df1, df2, ncp, upper=upper)
    full = _full_poisson_sum(f, df1, df2, ncp, upper)
    assert ours == pytest.approx(full, rel=SUBSAMPLE_TOL, abs=SUBSAMPLE_TOL)


# I_y(b, a) at y = r / (1 + r), r = z / a, from mpmath 1.3 at 50 digits
# (mpmath.betainc(b, a, 0, y, regularized=True), y the double given here):
# (b, a, z, value). a is 1e5 and 1e7 times max(b, 1), z is b and b +/- 2 sqrt(b).
LARGE_A_BETA = (
    (0.5, 100000.0, 0.001, "0.035670547082068689644"),
    (0.5, 100000.0, 0.5, "0.68268828228497562808"),
    (0.5, 100000.0, 1.9142135623730951, "0.94960841208301662138"),
    (0.5, 10000000.0, 0.001, "0.035670591283203496002"),
    (0.5, 10000000.0, 0.5, "0.6826894800385498352"),
    (0.5, 10000000.0, 1.9142135623730951, "0.94960978761147567718"),
    (1.0, 100000.0, 0.001, "0.00099950016163000594915"),
    (1.0, 100000.0, 1.0, "0.63211871943901596569"),
    (1.0, 100000.0, 3.0, "0.95021069120845871446"),
    (1.0, 10000000.0, 0.001, "0.00099950016657505828195"),
    (1.0, 10000000.0, 1.0, "0.63212054043458635506"),
    (1.0, 10000000.0, 3.0, "0.95021290922795471981"),
    (3.0, 300000.0, 0.001, "1.6654338082480179404e-10"),
    (3.0, 300000.0, 3.0, "0.57680879867003043088"),
    (3.0, 300000.0, 6.464101615137754, "0.95580807500988530192"),
    (3.0, 30000000.0, 0.001, "1.6654173329446447817e-10"),
    (3.0, 30000000.0, 3.0, "0.57680990767106663614"),
    (3.0, 30000000.0, 6.464101615137754, "0.95580962521702239868"),
    (30.0, 3000000.0, 19.045548849896676, "0.01220444960653581212"),
    (30.0, 3000000.0, 30.0, "0.52428265072314619014"),
    (30.0, 3000000.0, 40.954451150103324, "0.96836683701600711194"),
    (30.0, 300000000.0, 19.045548849896676, "0.012204203347717216484"),
    (30.0, 300000000.0, 30.0, "0.52428301026195377054"),
    (30.0, 300000000.0, 40.954451150103324, "0.96836769054221919059"),
    (1150.0, 115000000.0, 1082.1767001687474, "0.021135611755136874544"),
    (1150.0, 115000000.0, 1150.0, "0.50392134877192653638"),
    (1150.0, 115000000.0, 1217.8232998312526, "0.97568112902253744888"),
    (1150.0, 11500000000.0, 1082.1767001687474, "0.021135126758918721202"),
    (1150.0, 11500000000.0, 1150.0, "0.50392140699998370382"),
    (1150.0, 11500000000.0, 1217.8232998312526, "0.97568171380032822591"),
)


@pytest.mark.parametrize(("b", "a", "z", "expected"), LARGE_A_BETA)
def test_large_a_beta_factors(b: float, a: float, z: float, expected: str) -> None:
    # For a >= 1e4 max(b, 1) the gamma-ratio route replaces SciPy's betainc,
    # which is off by up to 2.8e-10 here (b = 3 and 30).
    log_u, log_l = _log_gamma_ratio(np.array([a]), np.array([b]), z / a)
    value = Decimal(expected)
    assert math.exp(log_u[0]) == pytest.approx(float(value), rel=1e-13)
    assert math.exp(log_l[0]) == pytest.approx(float(1 - value), rel=1e-13)


@pytest.mark.parametrize("f", [1e6, 1e12, 1e50, 1e200, 1e300])
@pytest.mark.parametrize("prob", [0.05, 0.5, 0.95, 0.999])
@pytest.mark.parametrize(("df1", "df2"), [(2.0, 60.0), (5.0, 1.5), (3.0, 2300.0)])
def test_f_bound_at_huge_f(f: float, df1: float, df2: float, prob: float) -> None:
    # The large-lambda bracket and both tails at any f; the root's probability is
    # the requested one.
    ncp = f_ncp_bound(f, df1, df2, prob=prob)
    assert 0.0 < ncp < math.inf
    if prob > 0.5:
        reached = math.exp(noncentral_f_logcdf(f, df1, df2, ncp, upper=True))
        assert reached == pytest.approx(1.0 - prob, rel=1e-9)
    else:
        reached = math.exp(noncentral_f_logcdf(f, df1, df2, ncp))
        assert reached == pytest.approx(prob, rel=1e-9)
    # The large-lambda start: lambda / (df1 f q / df2) tends to 1.
    q = float(special.chdtri(df2, prob))
    assert ncp / (df1 * f * q / df2) == pytest.approx(1.0, rel=0.05)


def test_f_bound_beyond_the_largest_double_is_inf() -> None:
    biggest = float(np.finfo(np.float64).max)
    assert f_ncp_bound(biggest, 2.0, 60.0, prob=0.95) == math.inf
