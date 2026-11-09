"""Rank-based methods: SciPy cross-checks, properties, edge inputs and timing.

``tools/reference/stats_reference.py`` covers no rank-based method, so the
independent cross-checks are SciPy's ``mannwhitneyu`` and ``kruskal`` and an inline
Dunn statistic from ``scipy.stats.rankdata`` (Dunn 1964), on data with and without
ties. Oracle parity is in ``test_nonparametric_parity.py``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from scipy import stats

from inferplot.stats import (
    BOOTSTRAP_RESAMPLES,
    Alternative,
    dunn,
    kruskal_wallis,
    mann_whitney,
    rank_biserial,
    rank_epsilon_squared,
)
from inferplot.stats.nonparametric import _percentile_lower
from tests.tolerances import assert_close

FloatArray = NDArray[np.float64]
RNG = np.random.Generator(np.random.PCG64(20261108))
# Rounded to one decimal, so the pooled values have many ties.
TIED = [np.round(RNG.normal(m, 1.0, n), 1) for m, n in [(0, 12), (0.6, 9), (0.2, 15)]]
DISTINCT = [RNG.normal(m, 1.0, n) for m, n in [(0, 7), (1, 11), (0.4, 9), (2, 5)]]
SIDES: tuple[Alternative, ...] = ("two-sided", "greater", "less")


def _scipy_dunn(groups: list[FloatArray]) -> list[tuple[int, int, float, float]]:
    """Dunn (1964) from scipy.stats.rankdata, with the usual tie correction."""
    pooled = np.concatenate(groups)
    n = pooled.size
    ranks = np.split(stats.rankdata(pooled), np.cumsum([g.size for g in groups])[:-1])
    t = np.unique(pooled, return_counts=True)[1].astype(np.float64)
    s2 = n * (n + 1) / 12 - float((t**3 - t).sum()) / (12 * (n - 1))
    out = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            se = math.sqrt(s2 * (1 / groups[i].size + 1 / groups[j].size))
            z = abs(float(ranks[i].mean() - ranks[j].mean())) / se
            out.append((i, j, z, float(2 * stats.norm.sf(z))))
    return out


# ------------------------------------------------------------- SciPy cross-checks


@pytest.mark.parametrize("data", [TIED, DISTINCT], ids=["ties", "distinct"])
@pytest.mark.parametrize("side", SIDES)
def test_mann_whitney_matches_scipy(data: list[FloatArray], side: Alternative) -> None:
    a, b = data[0], data[1]
    got = mann_whitney(a, b, alternative=side)
    want = stats.mannwhitneyu(
        a, b, alternative=side, method="asymptotic", use_continuity=True
    )
    assert_close("closed_form", got.statistic, float(want.statistic), what="W")
    assert_close("closed_form", got.p_value, float(want.pvalue), what="p")


@pytest.mark.parametrize("data", [TIED, DISTINCT], ids=["ties", "distinct"])
def test_kruskal_wallis_and_dunn_match_scipy(data: list[FloatArray]) -> None:
    got = kruskal_wallis(data)
    want = stats.kruskal(*data)
    assert_close("closed_form", got.statistic, float(want.statistic), what="H")
    assert_close("closed_form", got.p_value, float(want.pvalue), what="p")
    assert got.df == len(data) - 1
    pairs = dunn(data)
    for pair, (i, j, z, p) in zip(pairs, _scipy_dunn(data), strict=True):
        assert (pair.i, pair.j) == (i, j)
        assert_close("closed_form", pair.statistic, z, what="Dunn z")
        assert_close("closed_form", pair.p_value, p, what="Dunn p")


def test_rank_epsilon_squared_estimate_is_h_over_n_minus_1() -> None:
    for data in (TIED, DISTINCT):
        h = kruskal_wallis(data).statistic
        n = sum(g.size for g in data)
        got = rank_epsilon_squared(data).estimate
        assert_close("closed_form", got, h / (n - 1), what="epsilon")


def test_small_h_is_accurate() -> None:
    # Nearly identical rank distributions: H is small, and the textbook form
    # 12 / (N (N + 1)) sum R^2 / n - 3 (N + 1) cancels about six digits here.
    a = np.arange(0.0, 2000.0, 2.0)
    b = np.arange(1.0, 2001.0, 2.0)
    n_a = n_b = 1000
    n = n_a + n_b
    # Exact: the rank sums are n^2 and n^2 + n for groups of size n (ranks 1..2n).
    ssb = (
        n_a * (n_a**2 / n_a - (n + 1) / 2) ** 2
        + n_b * ((n_a**2 + n_a) / n_b - (n + 1) / 2) ** 2
    )
    h_exact = 12 * ssb / (n * (n + 1))
    assert kruskal_wallis([a, b]).statistic == pytest.approx(h_exact, rel=1e-15)


# -------------------------------------------------------------------- definitions


def test_mann_whitney_symmetry() -> None:
    a, b = TIED[0], TIED[1]
    ab, ba = mann_whitney(a, b), mann_whitney(b, a)
    assert ab.statistic + ba.statistic == a.size * b.size
    assert ab.p_value == ba.p_value
    assert mann_whitney(a, b, alternative="greater").p_value == (
        mann_whitney(b, a, alternative="less").p_value
    )


def test_mann_whitney_at_the_centre_has_p_one() -> None:
    # W = n_a n_b / 2: no continuity shift, z = 0.
    result = mann_whitney([1.0, 4.0], [2.0, 3.0])
    assert result.statistic == 2.0
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_rank_biserial_symmetry_and_sides() -> None:
    a, b = TIED[0], TIED[2]
    two = rank_biserial(a, b)
    mirrored = rank_biserial(b, a)
    assert mirrored.estimate == -two.estimate
    assert mirrored.conf_low == pytest.approx(-two.conf_high, rel=1e-15)
    greater = rank_biserial(a, b, alternative="greater")
    less = rank_biserial(a, b, alternative="less")
    assert (greater.sidedness, greater.conf_high) == ("greater", 1.0)
    assert (less.sidedness, less.conf_low) == ("less", -1.0)
    # A one-sided 95% bound is the two-sided 90% bound.
    ninety = rank_biserial(a, b, conf_level=0.90)
    assert greater.conf_low == pytest.approx(ninety.conf_low, rel=1e-14)
    assert less.conf_high == pytest.approx(ninety.conf_high, rel=1e-14)


def test_dunn_pairs_in_input_order() -> None:
    pairs = dunn(DISTINCT)
    assert [(p.i, p.j) for p in pairs] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    assert all(p.statistic >= 0.0 for p in pairs)


def test_percentile_rule_of_boot_ci() -> None:
    # boot.ci(type = "perc"): order statistic (R + 1) alpha, interpolated on the
    # normal quantile scale (Davison & Hinkley 1997, eq. 5.8).
    values = np.arange(1.0, 100.0)  # R = 99: (R + 1) 0.05 = 5, an order statistic
    assert _percentile_lower(values[::-1].copy(), 0.05) == 5.0
    values = np.arange(1.0, 101.0)  # R = 100: 5.05, between the 5th and 6th
    z = stats.norm.ppf
    expected = 5.0 + (z(0.05) - z(5 / 101)) / (z(6 / 101) - z(5 / 101))
    assert _percentile_lower(values, 0.05) == pytest.approx(expected, rel=1e-15)
    assert 5.0 < expected < 6.0
    assert abs(expected - 5.05) > 1e-3  # not linear interpolation (5.0536)


# ------------------------------------------------------------ bootstrap randomness


def test_same_seed_same_interval_other_seed_other_interval() -> None:
    first = rank_epsilon_squared(TIED, seed=3)
    assert rank_epsilon_squared(TIED, seed=3) == first
    other = rank_epsilon_squared(TIED, seed=4)
    assert other.conf_low != first.conf_low
    assert other.estimate == first.estimate
    assert (first.seed, other.seed) == (3, 4)
    assert first.resamples == BOOTSTRAP_RESAMPLES == 2000


def test_default_seed_is_zero() -> None:
    assert rank_epsilon_squared(TIED) == rank_epsilon_squared(TIED, seed=0)


# --------------------------------------------------------------------- properties

_values = st.lists(st.integers(min_value=-6, max_value=6), min_size=1, max_size=12).map(
    lambda v: np.asarray(v, dtype=np.float64)
)
_groups = st.lists(_values, min_size=2, max_size=4)


def _all_tied(groups: list[FloatArray]) -> bool:
    return np.unique(np.concatenate(groups)).size == 1


def _tied_resample_possible(groups: list[FloatArray]) -> bool:
    """Whether one value occurs in every group (needed for a tied resample)."""
    common = set(groups[0].tolist())
    for g in groups[1:]:
        common &= set(g.tolist())
    return bool(common)


def _deterministic(groups: list[FloatArray]) -> tuple[object, ...]:
    kw = kruskal_wallis(groups)
    pairs = dunn(groups)
    return (kw, tuple((p.statistic, p.p_value) for p in pairs))


@settings(max_examples=60, deadline=None)
@given(_groups)
def test_invariances(groups: list[FloatArray]) -> None:
    if _all_tied(groups):
        with pytest.raises(ValueError, match="tied"):
            kruskal_wallis(groups)
        return
    base = _deterministic(groups)
    # A strictly increasing transform keeps every rank: identical numbers.
    transformed = [np.exp(g / 4.0) * 3.0 - 1.0 for g in groups]
    assert _deterministic(transformed) == base
    # Row order within groups.
    rng = np.random.Generator(np.random.PCG64(len(groups)))
    shuffled = [rng.permutation(g) for g in groups]
    assert _deterministic(shuffled) == base
    # Group relabelling: reversing the groups maps pair (i, j) to (k-1-j, k-1-i).
    k = len(groups)
    reversed_pairs = {
        (k - 1 - p.j, k - 1 - p.i): (p.statistic, p.p_value) for p in dunn(groups[::-1])
    }
    assert kruskal_wallis(groups[::-1]) == base[0]
    for p in dunn(groups):
        assert reversed_pairs[(p.i, p.j)] == (p.statistic, p.p_value)
    # Two groups: Mann-Whitney and the rank-biserial r behave the same way.
    a, b = groups[0], groups[1]
    if not _all_tied([a, b]):
        mw = mann_whitney(a, b)
        assert mann_whitney(transformed[0], transformed[1]) == mw
        assert mann_whitney(shuffled[0], shuffled[1]) == mw
    rb = rank_biserial(a, b)
    assert rank_biserial(transformed[0], transformed[1]) == rb
    assert rank_biserial(shuffled[0], shuffled[1]) == rb
    assert -1.0 <= rb.conf_low <= rb.estimate <= rb.conf_high <= 1.0


@settings(max_examples=15, deadline=None)
@given(
    st.lists(
        st.lists(st.integers(-20, 20), min_size=4, max_size=10).map(
            lambda v: np.asarray(v, dtype=np.float64)
        ),
        min_size=2,
        max_size=4,
    )
)
def test_bootstrap_invariances(groups: list[FloatArray]) -> None:
    if _all_tied(groups) or _tied_resample_possible(groups):
        # Undefined (every value tied) or possibly undefined (a resample with every
        # value tied): checked in test_all_values_tied and
        # test_tied_resamples_are_left_out.
        return
    base = rank_epsilon_squared(groups, seed=1)
    assert 0.0 <= base.conf_low <= 1.0
    assert 0.0 <= base.estimate <= 1.0
    rng = np.random.Generator(np.random.PCG64(9))
    assert rank_epsilon_squared([g * 2.0 + 5.0 for g in groups], seed=1) == base
    assert rank_epsilon_squared([np.exp(g / 10) for g in groups], seed=1) == base
    assert rank_epsilon_squared([rng.permutation(g) for g in groups], seed=1) == base
    assert rank_epsilon_squared(groups[::-1], seed=1) == base


# ------------------------------------------------------------------- edge inputs


def test_all_values_tied() -> None:
    same = [np.full(5, 2.5), np.full(4, 2.5), np.full(3, 2.5)]
    for fn in (kruskal_wallis, dunn, rank_epsilon_squared):
        with pytest.raises(ValueError, match="all 12 values are tied"):
            fn(same)  # type: ignore[operator]
    with pytest.raises(ValueError, match="all 9 values are tied"):
        mann_whitney(same[0], same[1])
    # The rank-biserial r is defined: no pair differs, so r = 0.
    r = rank_biserial(same[0], same[1])
    assert r.estimate == 0.0
    assert r.conf_low == -r.conf_high


def test_one_value_per_group() -> None:
    groups = [np.array([1.0]), np.array([3.0]), np.array([2.0])]
    kw = kruskal_wallis(groups)
    assert kw.statistic == pytest.approx(2.0, rel=1e-15)  # (N - 1) SSB / SST = 2
    assert [p.statistic for p in dunn(groups)] == pytest.approx(
        [2.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)], rel=1e-15
    )
    # Every resample is the data itself: the bound is the estimate.
    eps = rank_epsilon_squared(groups)
    assert eps.estimate == 1.0
    assert eps.conf_low == 1.0
    mw = mann_whitney([1.0], [3.0])
    assert mw.statistic == 0.0
    assert mw.p_value == 1.0  # |d| = 1/2 is all continuity correction
    r = rank_biserial([1.0], [3.0])
    assert (r.estimate, r.conf_low, r.conf_high) == (-1.0, -1.0, -1.0)
    r = rank_biserial([3.0], [1.0], alternative="less")
    assert (r.estimate, r.conf_low, r.conf_high) == (1.0, -1.0, 1.0)


def test_two_values_per_group() -> None:
    groups = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    assert kruskal_wallis(groups).statistic == pytest.approx(
        float(stats.kruskal(*groups).statistic), rel=1e-15
    )
    eps = rank_epsilon_squared(groups)
    assert 0.0 <= eps.conf_low <= eps.estimate <= 1.0
    mw = mann_whitney(groups[0], groups[1])
    want = stats.mannwhitneyu(*groups[:2], method="asymptotic", use_continuity=True)
    assert_close("closed_form", mw.p_value, float(want.pvalue), what="p")


def test_constant_group() -> None:
    groups = [np.full(6, 1.0), np.array([0.5, 2.0, 3.0, 4.0]), np.array([5.0, 6.0])]
    assert_close(
        "closed_form",
        kruskal_wallis(groups).statistic,
        float(stats.kruskal(*groups).statistic),
        what="H",
    )
    assert 0.0 < rank_epsilon_squared(groups).conf_low < 1.0


def test_tied_resamples_are_left_out() -> None:
    # The value 1 is in every group and the groups are tiny: (2/3)^6 = 8.8% of the
    # stratified resamples have every value tied, where epsilon is undefined. They
    # are left out of the percentile and counted (D54, R's rule per the oracle).
    groups = [np.array([1.0, 1.0, 2.0]), np.array([1.0, 1.0, 3.0])]
    result = rank_epsilon_squared(groups, seed=4)
    assert result.resamples == 2000
    assert 120 < result.undefined_resamples < 230
    again = rank_epsilon_squared(groups[::-1], seed=4)
    assert again == result
    assert 0.0 <= result.conf_low <= result.estimate
    # Too few resamples left for the order statistic the level needs: 1,821 or
    # fewer defined values cannot give the 0.05% quantile.
    with pytest.raises(ValueError, match=r"every value tied, which leaves \d+"):
        rank_epsilon_squared(groups, conf_level=0.9995, seed=4)


def test_no_tied_resample_is_counted_as_zero() -> None:
    clean = rank_epsilon_squared([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])])
    assert clean.undefined_resamples == 0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_raise(bad: float) -> None:
    a, b = np.array([1.0, bad, 3.0]), np.array([2.0, 4.0])
    calls: list[Callable[[], object]] = [
        lambda: mann_whitney(a, b),
        lambda: rank_biserial(b, a),
        lambda: kruskal_wallis([a, b]),
        lambda: dunn([b, a]),
        lambda: rank_epsilon_squared([a, b]),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="finite"):
            call()


def test_bad_arguments_raise() -> None:
    a, b = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    with pytest.raises(ValueError, match="at least two groups"):
        kruskal_wallis([a])
    with pytest.raises(ValueError, match="at least two groups"):
        dunn([a])
    with pytest.raises(ValueError, match="at least 1 value"):
        kruskal_wallis([a, np.array([])])
    with pytest.raises(ValueError, match="at least 1 value"):
        mann_whitney(a, [])
    with pytest.raises(ValueError, match="alternative"):
        mann_whitney(a, b, alternative="two.sided")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="conf_level"):
        rank_biserial(a, b, conf_level=1.0)
    with pytest.raises(ValueError, match="conf_level"):
        rank_epsilon_squared([a, b], conf_level=0.0)
    with pytest.raises(ValueError, match="order statistic"):
        rank_epsilon_squared(TIED, conf_level=0.9999)
    for seed in (-1, 1.5, True):
        with pytest.raises(ValueError, match="seed"):
            rank_epsilon_squared([a, b], seed=seed)  # type: ignore[arg-type]


# ------------------------------------------------------------------------- timing


def _best_of_three(fn: Callable[[], object]) -> float:
    fn()
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def test_closed_form_methods_at_n_1e5_take_under_a_second() -> None:
    rng = np.random.Generator(np.random.PCG64(5))
    groups = [np.round(rng.normal(m, 1.0, 25_000), 2) for m in (0.0, 0.1, 0.2, 0.3)]
    for name, fn in [
        ("mann_whitney", lambda: mann_whitney(groups[0], groups[1] + 0.0)),
        ("rank_biserial", lambda: rank_biserial(groups[0], groups[1])),
        ("kruskal_wallis", lambda: kruskal_wallis(groups)),
        ("dunn", lambda: dunn(groups)),
    ]:
        seconds = _best_of_three(fn)
        assert seconds < 1.0, f"{name} on 1e5 values took {seconds:.3f} s"
