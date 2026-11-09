"""Hedges' g (unpooled standardizer, Welch df) and partial omega squared."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pytest
from tools.reference import stats_reference as ref

from inferplot.stats import (
    f_ncp_bound,
    hedges_g,
    hedges_j,
    partial_omega_squared,
    welch_anova,
)
from tests.stats.test_noncentral import OLD_BETAINC
from tests.tolerances import assert_close

RNG = np.random.Generator(np.random.PCG64(5))
A = RNG.normal(10.0, 2.0, 14)
B = RNG.normal(8.5, 3.5, 21)


@pytest.mark.parametrize("df", [1.5, 2.0, 3.7, 10.0, 30.9, 31.0, 31.1, 55.3, 1e3])
def test_hedges_j_matches_reference(df: float) -> None:
    # tools/reference subtracts two log-gamma values, which loses digits as df
    # grows (1.4e-8 relative at df = 1e7); the large-df behaviour is checked below.
    assert_close("closed_form", hedges_j(df), ref.hedges_j(df), what=f"J({df})")


@pytest.mark.parametrize("df", [1e4, 1e6, 1e7, 1e10])
def test_hedges_j_large_df(df: float) -> None:
    # 1 - J(df) = 3 / (4 df) + 7 / (32 df^2) + O(df^-3), so Hedges' (1981, eq. 6f)
    # 3 / (4 df - 1) is off by about 1 / (24 df) relative; 1 - J itself carries the
    # rounding of J, 1.1e-16 absolute (1.5e-6 relative at df = 1e10).
    ratio = (1.0 - hedges_j(df)) * (4.0 * df - 1.0) / 3.0
    assert ratio == pytest.approx(1.0, abs=1.0 / (12.0 * df) + 1e-5 * (df > 1e8))


def test_hedges_j_limits() -> None:
    assert hedges_j(math.inf) == 1.0
    assert hedges_j(1.0) == 0.0
    # Continuous across the switch to Stirling's remainder (df = 31).
    assert hedges_j(31.0 - 1e-9) == pytest.approx(hedges_j(31.0), rel=1e-12)
    # Hedges' approximation 1 - 3 / (4 df - 1) for large df.
    assert hedges_j(1e4) == pytest.approx(1.0 - 3.0 / (4e4 - 1.0), rel=1e-9)
    with pytest.raises(ValueError, match="df must"):
        hedges_j(0.0)


def test_hedges_g_definition() -> None:
    g = hedges_g(A, B)
    va, vb = A.var(ddof=1), B.var(ddof=1)
    wa, wb = va / A.size, vb / B.size
    df = (wa + wb) ** 2 / (wa**2 / (A.size - 1) + wb**2 / (B.size - 1))
    d = (A.mean() - B.mean()) / math.sqrt((va + vb) / 2.0)
    assert_close("closed_form", g.estimate, d * ref.hedges_j(df), what="g")
    assert g.conf_low < g.estimate < g.conf_high
    assert (g.conf_level, g.sidedness) == (0.95, "two-sided")


def test_hedges_g_invariances() -> None:
    g = hedges_g(A, B)
    shifted = hedges_g(3.0 * A + 100.0, 3.0 * B + 100.0)
    assert shifted.estimate == pytest.approx(g.estimate, rel=1e-12)
    assert shifted.conf_low == pytest.approx(g.conf_low, rel=1e-10)
    swapped = hedges_g(B, A)
    assert swapped.estimate == pytest.approx(-g.estimate, rel=1e-14)
    assert swapped.conf_low == pytest.approx(-g.conf_high, rel=1e-10)


def test_hedges_g_one_sided() -> None:
    greater = hedges_g(A, B, alternative="greater")
    less = hedges_g(A, B, alternative="less")
    two_90 = hedges_g(A, B, conf_level=0.90)
    assert greater.conf_high == math.inf
    assert greater.sidedness == "greater"
    assert less.conf_low == -math.inf
    assert less.sidedness == "less"
    assert greater.conf_low == pytest.approx(two_90.conf_low, rel=1e-10)
    assert less.conf_high == pytest.approx(two_90.conf_high, rel=1e-10)


def test_hedges_g_conf_levels_nest() -> None:
    intervals = [hedges_g(A, B, conf_level=c) for c in (0.5, 0.8, 0.9, 0.95, 0.99)]
    for narrow, wide in itertools.pairwise(intervals):
        assert wide.conf_low < narrow.conf_low < narrow.conf_high < wide.conf_high


def test_hedges_g_edge_samples() -> None:
    small = hedges_g([1.0, 2.0], [4.0, 6.0])  # n = 2 per group
    assert math.isfinite(small.estimate)
    assert small.conf_low < small.conf_high
    one_constant = hedges_g([5.0, 5.0, 5.0], [1.0, 2.0, 4.0, 7.0])
    assert math.isfinite(one_constant.conf_low)
    with pytest.raises(ValueError, match="constant"):
        hedges_g([1.0, 1.0], [2.0, 2.0])
    # Welch df = 1 (n = 2 against a constant group): J(1) = 0, as R's lgamma gives;
    # the far bound of a one-sided interval stays infinite.
    degenerate = hedges_g([1.0, 3.0], [5.0, 5.0], alternative="greater")
    assert degenerate.estimate == 0.0
    assert degenerate.conf_high == math.inf


def test_hedges_g_huge_effect() -> None:
    a = np.linspace(0.0, 1.0, 40)
    g = hedges_g(a, a + 30.0)
    assert g.estimate < -50.0
    assert math.isfinite(g.conf_low)
    assert g.conf_low < g.estimate < g.conf_high < 0.0


def test_partial_omega_squared_estimate_and_ci() -> None:
    groups = [
        RNG.normal(m, s, n) for m, s, n in [(0, 1, 12), (1.5, 2, 15), (0.5, 1, 9)]
    ]
    anova = welch_anova(groups)
    omega = partial_omega_squared(anova.f, anova.df1, anova.df2)
    expected = ref.omega2_partial_from_f(anova.f, anova.df1, anova.df2)
    assert_close("closed_form", omega.estimate, expected, what="omega")
    assert 0.0 <= omega.conf_low <= omega.estimate
    assert (omega.conf_high, omega.sidedness) == (1.0, "greater")


@pytest.mark.parametrize("f", [0.0, 0.3, 0.999, 1.0])
def test_f_at_most_one_gives_zero_and_the_unit_interval(f: float) -> None:
    omega = partial_omega_squared(f, 2.0, 15.7)
    assert (omega.estimate, omega.conf_low, omega.conf_high) == (0.0, 0.0, 1.0)


def test_partial_omega_squared_levels_and_huge_f() -> None:
    lows = [
        partial_omega_squared(12.0, 3.0, 40.0, conf_level=c).conf_low
        for c in (0.5, 0.8, 0.9, 0.95, 0.99)
    ]
    assert lows == sorted(lows, reverse=True)
    huge = partial_omega_squared(1e5, 2.0, 60.0)
    assert 0.99 < huge.conf_low < huge.estimate < 1.0
    with pytest.raises(ValueError, match="F must"):
        partial_omega_squared(-1.0, 2.0, 10.0)
    with pytest.raises(ValueError, match="conf_level"):
        partial_omega_squared(3.0, 2.0, 10.0, conf_level=1.5)


# Noncentrality bounds lambda = f_ncp_bound(F_w, df1, df2, prob=level) of the full
# Poisson sum inverted from 0, the method before the subsampled sum (inferplot at
# 9389d24, SciPy 1.18; up to 18.8 s per value at F = 1e10). F_w is
# (F - 1) df2 / (df1 + df2 + 1). (df1, df2, level, F, lambda).
FULL_SUM_BOUNDS = (
    (2.0, 60.0, 0.95, 1e3, 1353.1767795625597),
    (2.0, 60.0, 0.95, 1e5, 137086.45016475377),
    (2.0, 60.0, 0.95, 1e7, 13710444.818211379),
    (2.0, 60.0, 0.95, 1e9, 1371046281.943696),
    (2.0, 60.0, 0.95, 1e10, 13710462983.082462),
    (2.0, 6.77, 0.95, 1e3, 412.84528302553207),
    (2.0, 6.77, 0.95, 1e5, 41791.797665590835),
    (2.0, 6.77, 0.95, 1e7, 4179689.217149753),
    (2.0, 6.77, 0.95, 1e9, 417969431.1877129),
    (2.0, 6.77, 0.95, 1e10, 4179694358.192101),
    (4.0, 5.5, 0.9, 1e3, 718.3366131308259),
    (4.0, 5.5, 0.9, 1e5, 72458.75560663512),
    (4.0, 5.5, 0.9, 1e7, 7246501.474199522),
    (4.0, 5.5, 0.9, 1e9, 724650773.341813),
    (4.0, 5.5, 0.9, 1e10, 7246507790.320328),
    (5.0, 1.5, 0.99, 1e3, 0.0),
    (5.0, 1.5, 0.99, 1e5, 252.46363606601352),
    (5.0, 1.5, 0.99, 1e7, 25692.180388048055),
    (5.0, 1.5, 0.99, 1e9, 2569663.416017914),
    (5.0, 1.5, 0.99, 1e10, 25696674.648627266),
    (2.0, 2300.0, 0.95, 1e3, 1821.9917668007802),
    (2.0, 2300.0, 0.95, 1e5, 190039.36880806417),
    (2.0, 2300.0, 0.95, 1e7, 19014991.00478171),
    (2.0, 2300.0, 0.95, 1e9, 1901510218.7471213),
    (2.0, 2300.0, 0.95, 1e10, 19015103198.279842),
    (9.0, 40.0, 0.95, 1e3, 4746.468500299005),
    (9.0, 40.0, 0.95, 1e5, 477142.1957349044),
    (9.0, 40.0, 0.95, 1e7, 47716720.49170035),
    (9.0, 40.0, 0.95, 1e9, 4771674550.142294),
    (9.0, 40.0, 0.95, 1e10, 47716745728.784775),
)


@pytest.mark.parametrize(("df1", "df2", "level", "f", "ncp"), FULL_SUM_BOUNDS)
def test_partial_omega_squared_matches_the_full_poisson_sum(
    df1: float, df2: float, level: float, f: float, ncp: float
) -> None:
    omega = partial_omega_squared(f, df1, df2, conf_level=level)
    assert abs(omega.conf_low - ncp / (ncp + df2)) <= 1e-12
    f_w = (f - 1.0) * (df2 / (df1 + df2 + 1.0))
    ours = f_ncp_bound(f_w, df1, df2, prob=level)
    # Measured: within 3.4e-12 relative over 114 values from F = 10 to 1e10 with
    # SciPy 1.18; 7.8e-11 with SciPy 1.11.1, whose betainc carries up to 1e-10
    # relative error at a = 1e4 to 1e7 (tests/stats/test_noncentral.py).
    assert ours == pytest.approx(ncp, rel=2e-10 if OLD_BETAINC else 1e-11, abs=1e-9)


@pytest.mark.parametrize("f", [1.0001, 1.5, 3.0, 47.0, 1e3, 1e8, 1e14])
@pytest.mark.parametrize(("df1", "df2"), [(2.0, 6.77), (5.0, 1.5), (9.0, 2300.0)])
def test_f_w_identity(f: float, df1: float, df2: float) -> None:
    # omega df2 / ((1 - omega) df1) = (F - 1) df2 / (df1 + df2 + 1), since
    # 1 - omega = (df1 + df2 + 1) / (F df1 + df2 + 1).
    omega = (f - 1.0) * df1 / (f * df1 + df2 + 1.0)
    via_omega = omega * df2 / ((1.0 - omega) * df1)
    identity = (f - 1.0) * (df2 / (df1 + df2 + 1.0))
    # The rounding of 1 - omega grows like F eps.
    assert identity == pytest.approx(via_omega, rel=1e-15 * max(1.0, f) + 1e-14)


@pytest.mark.parametrize("f", [1e12, 1e17, 1e50, 1e300, float(np.finfo(float).max)])
@pytest.mark.parametrize(
    ("df1", "df2", "level"),
    [(2.0, 60.0, 0.95), (2.0, 6.77, 0.95), (5.0, 1.5, 0.99), (2.0, 2300.0, 0.9)],
)
def test_partial_omega_squared_is_fast_at_any_f(
    f: float, df1: float, df2: float, level: float
) -> None:
    # API-1 / STAT-6: 125 s at F = 1e12 and ZeroDivisionError from F about 1e17
    # before; measured now at most 0.04 s per call (F = 1e300).
    start = time.perf_counter()
    omega = partial_omega_squared(f, df1, df2, conf_level=level)
    seconds = time.perf_counter() - start
    assert seconds < 2.0, f"F={f!r} took {seconds:.2f} s"
    assert 0.999 < omega.conf_low <= omega.estimate <= 1.0
    assert (omega.conf_high, omega.sidedness) == (1.0, "greater")
    if f >= 1e50:
        assert omega.conf_low == omega.estimate == 1.0


def test_omega_bound_below_the_estimate_where_it_rounds_to_one() -> None:
    # At F = 1e17 the estimate rounds to 1 for df = (2, 6.77); the bound is the
    # largest double below 1 or 1 itself, never above the estimate.
    omega = partial_omega_squared(1e17, 2.0, 6.77)
    assert omega.estimate == 1.0
    assert 1.0 - 1e-15 < omega.conf_low <= 1.0
