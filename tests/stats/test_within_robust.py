"""The robust methods for dependent groups: definitions against plain NumPy
formulas, sign conventions, invariances, the bootstrap's seeding and its undefined
and infinite resamples, degenerate inputs and extreme scales (oracle parity is in
test_within_robust_parity.py)."""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pytest
from scipy import stats as scipy_stats

from inferplot.stats import (
    average_dependent_akp,
    dependent_akp,
    p_adjust,
    rmmcp,
    trimmed_rm_anova,
    winsorized_normal_variance,
    yuen_dependent_t,
)

FloatArray = np.ndarray[Any, np.dtype[np.float64]]
C = math.sqrt(winsorized_normal_variance(0.2))


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _table(n: int, k: int, seed: int) -> list[FloatArray]:
    rng = _rng(seed)
    subject = rng.normal(0.0, 2.0, n)
    return [subject + 0.4 * j + rng.standard_t(4, n) for j in range(k)]


def _winsorized(x: FloatArray, tr: float = 0.2) -> FloatArray:
    ordered = np.sort(x)
    g = math.floor(tr * x.size)
    return np.clip(x, ordered[g], ordered[x.size - g - 1])


def _tmean(x: FloatArray, tr: float = 0.2) -> float:
    ordered = np.sort(x)
    g = math.floor(tr * x.size)
    return float(ordered[g : x.size - g].mean())


# ------------------------------------------------------------------ yuend


@pytest.mark.parametrize("seed", range(5))
def test_yuend_definition(seed: int) -> None:
    a, b = _table(17, 2, seed)
    n, g = a.size, math.floor(0.2 * a.size)
    h = n - 2 * g
    wa, wb = _winsorized(a), _winsorized(b)
    var = np.var(wa, ddof=1) + np.var(wb, ddof=1) - 2 * np.cov(wa, wb)[0, 1]
    se = math.sqrt((n - 1) * var / (h * (h - 1)))
    t = (_tmean(a) - _tmean(b)) / se
    test = yuen_dependent_t(a, b)
    assert test.t == pytest.approx(t, rel=1e-12)
    assert test.df == h - 1
    assert test.n == n
    assert test.se == pytest.approx(se, rel=1e-12)
    assert test.p_value == pytest.approx(2 * scipy_stats.t.sf(abs(t), h - 1), rel=1e-12)
    # Signed as a - b: swapping the samples flips t and keeps df and p.
    swapped = yuen_dependent_t(b, a)
    assert swapped.t == -test.t
    assert (swapped.df, swapped.p_value, swapped.se) == (test.df, test.p_value, test.se)


def test_yuend_invariances() -> None:
    a, b = _table(23, 2, 7)
    base = yuen_dependent_t(a, b)
    order = _rng(1).permutation(a.size)
    assert yuen_dependent_t(a[order], b[order]) == base  # subject order
    for k in (-60, 3, 70):
        scaled = yuen_dependent_t(np.ldexp(a, k), np.ldexp(b, k))
        assert (scaled.t, scaled.df, scaled.p_value) == (base.t, base.df, base.p_value)
        assert scaled.difference == math.ldexp(base.difference, k)
    for factor in (1e-150, 1e150):
        scaled = yuen_dependent_t(a * factor, b * factor)
        assert scaled.t == pytest.approx(base.t, rel=1e-13)
        assert scaled.difference == pytest.approx(base.difference * factor, rel=1e-13)


def test_yuend_large_offset_keeps_accuracy() -> None:
    # Pairs on a common offset of 2^40: the differences are exact, so t is the
    # same as without the offset.
    a, b = _table(15, 2, 3)
    a, b = np.round(a * 1024) / 1024, np.round(b * 1024) / 1024  # exact + 2^40
    base = yuen_dependent_t(a, b)
    shifted = yuen_dependent_t(a + 2.0**40, b + 2.0**40)
    assert shifted.t == pytest.approx(base.t, rel=1e-9)


def test_yuend_degenerate_inputs() -> None:
    a = np.array([1.0, 2.0, 4.0, 3.5, 6.0])
    with pytest.raises(ValueError, match="every pair differs by the same amount"):
        yuen_dependent_t(a, a)  # all-zero differences
    with pytest.raises(ValueError, match="every pair differs by the same amount"):
        yuen_dependent_t(a, a - 1.5)
    with pytest.raises(ValueError, match="same number of values"):
        yuen_dependent_t(a, a[:4])
    with pytest.raises(ValueError, match="at least 2 subjects after trimming"):
        yuen_dependent_t(a[:3], a[:3] + np.array([0.0, 1.0, 3.0]), tr=0.4)
    # Two and three pairs: g = 0, h = n.
    two = yuen_dependent_t([1.0, 2.0], [3.0, 5.0])
    assert two.df == 1.0
    assert math.isfinite(two.t)
    three = yuen_dependent_t([1.0, 2.0, 2.0], [3.0, 5.0, 2.0])
    assert three.df == 2.0


# ------------------------------------------------------------------ dep.effect


@pytest.mark.parametrize("seed", range(4))
def test_dependent_akp_definition(seed: int) -> None:
    a, b = _table(19, 2, seed)
    d = a - b
    expected = C * _tmean(d) / math.sqrt(np.var(_winsorized(d), ddof=1))
    result = dependent_akp(a, b)
    assert result.estimate == pytest.approx(expected, rel=1e-12)
    assert result.conf_low <= result.estimate <= result.conf_high
    assert (result.resamples, result.conf_level, result.sidedness) == (
        2000,
        0.95,
        "two-sided",
    )
    assert dependent_akp(b, a).estimate == -result.estimate


def test_dependent_akp_bootstrap_is_seeded_and_ignores_row_order() -> None:
    a, b = _table(25, 2, 11)
    first = dependent_akp(a, b, seed=5)
    assert first.seed == 5
    assert type(first.seed) is int
    assert dependent_akp(a, b, seed=5) == first
    assert dependent_akp(a, b, seed=6) != first
    order = _rng(2).permutation(a.size)
    assert dependent_akp(a[order], b[order], seed=5) == first
    # Scale-free: every power of two gives the same estimate and bounds.
    scaled = dependent_akp(np.ldexp(a, 40), np.ldexp(b, 40), seed=5)
    assert scaled == first
    for factor in (1e-150, 1e150):
        other = dependent_akp(a * factor, b * factor, seed=5)
        assert other.estimate == pytest.approx(first.estimate, rel=1e-13)
        assert other.conf_low == pytest.approx(first.conf_low, rel=1e-12)
        assert other.conf_high == pytest.approx(first.conf_high, rel=1e-12)
    with pytest.raises(ValueError, match="seed"):
        dependent_akp(a, b, seed=-1)
    with pytest.raises(ValueError, match="conf_level"):
        dependent_akp(a, b, conf_level=1.0)


def test_dependent_akp_infinite_and_undefined_resamples() -> None:
    # Differences: most are 5; a resample whose winsorized differences are all 5
    # has AKP = 5 / 0 = +inf, kept and counted (so the upper bound is +inf).
    b = np.zeros(10)
    infinite = dependent_akp(np.array([5.0] * 5 + [1.0, 2.0, 3.0, 8.0, 9.0]), b)
    assert infinite.infinite_resamples > 50
    assert infinite.undefined_resamples == 0
    assert infinite.conf_high == math.inf
    assert math.isfinite(infinite.estimate)
    # Most differences are 0: a resample winsorized to all zeros is 0 / 0, left
    # out of the percentile interval and counted (S3, D54's rule).
    undefined = dependent_akp(np.array([0.0] * 6 + [1.0, 2.0, 3.0, 4.0]), b)
    assert undefined.undefined_resamples > 50
    assert math.isfinite(undefined.conf_low)
    assert math.isfinite(undefined.conf_high)


def test_dependent_akp_degenerate_inputs() -> None:
    a = np.array([1.0, 2.0, 4.0, 3.5, 6.0, 2.5])
    with pytest.raises(ValueError, match="every subject's difference is the same"):
        dependent_akp(a, a)
    with pytest.raises(ValueError, match="every subject's difference is the same"):
        dependent_akp(a, a + 3.0)
    # Differences [0, 0, 0, 0, 1, 7]: the 2nd to 5th smallest are not all equal.
    assert math.isfinite(dependent_akp(a, a - [0, 0, 0, 0, 1, 7]).estimate)
    with pytest.raises(ValueError, match="every subject's difference is the same"):
        dependent_akp(a, a - [0, 0, 0, 0, 0, 7])  # winsorized: all 0


# ------------------------------------------------------------------ rmanova


def _rmanova_reference(columns: list[FloatArray], tr: float = 0.2) -> tuple[float, ...]:
    x = np.column_stack(columns)
    n, k = x.shape
    g = math.floor(tr * n)
    h = n - 2 * g
    tm = np.array([_tmean(c, tr) for c in columns])
    w = np.column_stack([_winsorized(c, tr) for c in columns])
    qc = h * np.sum((tm - tm.mean()) ** 2)
    r = w - w.mean(axis=0) - w.mean(axis=1, keepdims=True) + w.mean()
    f = (qc / (k - 1)) / (np.sum(r**2) / ((h - 1) * (k - 1)))
    v = np.cov(w, rowvar=False)
    a = k * k * (np.mean(np.diag(v)) - v.mean()) ** 2 / (k - 1)
    b = np.sum(v * v) - 2 * k * np.sum(v.mean(axis=1) ** 2) + k * k * v.mean() ** 2
    e_hat = a / b
    e_tilde = min(
        1.0, (n * (k - 1) * e_hat - 2) / ((k - 1) * (n - 1 - (k - 1) * e_hat))
    )
    return f, (k - 1) * e_tilde, (k - 1) * (h - 1) * e_tilde


@pytest.mark.parametrize(("n", "k", "seed"), [(12, 3, 0), (30, 4, 1), (9, 5, 2)])
def test_trimmed_rm_anova_definition(n: int, k: int, seed: int) -> None:
    columns = _table(n, k, seed)
    f, df1, df2 = _rmanova_reference(columns)
    anova = trimmed_rm_anova(columns)
    assert anova.f == pytest.approx(f, rel=1e-12)
    assert anova.df1 == pytest.approx(df1, rel=1e-12)
    assert anova.df2 == pytest.approx(df2, rel=1e-12)
    assert anova.p_value == pytest.approx(
        scipy_stats.f.sf(anova.f, anova.df1, anova.df2), rel=1e-10
    )
    assert 0.0 < anova.epsilon <= 1.0
    assert 1.0 / (k - 1) <= anova.epsilon_gg <= 1.0
    assert (anova.n, anova.k) == (n, k)


def test_trimmed_rm_anova_invariances() -> None:
    columns = _table(14, 4, 9)
    base = trimmed_rm_anova(columns)
    for order in itertools.permutations(range(4)):
        other = trimmed_rm_anova([columns[i] for i in order])
        assert other.f == pytest.approx(base.f, rel=1e-14)
        assert other.df1 == pytest.approx(base.df1, rel=1e-14)
    rows = _rng(3).permutation(14)
    assert trimmed_rm_anova([c[rows] for c in columns]) == base
    assert trimmed_rm_anova([np.ldexp(c, -80) for c in columns]) == base
    for factor in (1e-150, 1e150):
        other = trimmed_rm_anova([c * factor for c in columns])
        assert other.f == pytest.approx(base.f, rel=1e-13)
        assert other.df2 == pytest.approx(base.df2, rel=1e-13)


def test_trimmed_rm_anova_caps_epsilon_and_refuses_no_error() -> None:
    # Independent normal conditions: the Huynh-Feldt estimate is above 1 and the
    # df are (k - 1) and (k - 1)(h - 1) exactly.
    rng = _rng(4)
    columns = [rng.normal(size=40) for _ in range(3)]
    anova = trimmed_rm_anova(columns)
    if anova.epsilon == 1.0:
        assert (anova.df1, anova.df2) == (2.0, 2.0 * (40 - 16 - 1))
    base = np.array([1.0, 4.0, 2.0, 8.0, 5.0, 7.0])
    with pytest.raises(ValueError, match="no error variance"):
        trimmed_rm_anova([base, base + 1.0, base - 2.0])
    with pytest.raises(ValueError, match="at least two conditions"):
        trimmed_rm_anova([base])
    with pytest.raises(ValueError, match="same number of values"):
        trimmed_rm_anova([base, base[:5], base])


# ------------------------------------------------------------------ rmmcp


def test_rmmcp_definition() -> None:
    columns = _table(21, 3, 5)
    n = 21
    g = math.floor(0.2 * n)
    pairs = rmmcp(columns, p_adjust_method="none")
    assert [(p.i, p.j) for p in pairs] == [(0, 1), (0, 2), (1, 2)]
    critical = scipy_stats.t.ppf(1 - 0.05 / (2 * 3), n - 2 * g - 1)
    for pair in pairs:
        d = columns[pair.i] - columns[pair.j]
        se = math.sqrt(np.var(_winsorized(d), ddof=1)) / (0.6 * math.sqrt(n))
        assert pair.estimate == pytest.approx(_tmean(d), rel=1e-12)
        assert pair.se == pytest.approx(se, rel=1e-12)
        assert pair.df == n - 2 * g - 1
        assert pair.t == pytest.approx(_tmean(d) / se, rel=1e-12)
        assert pair.critical_value == pytest.approx(critical, rel=1e-12)
        assert pair.conf_low == pytest.approx(pair.estimate - critical * se, rel=1e-12)
        assert pair.p_value == pair.p_adjusted
    holm = rmmcp(columns)
    assert tuple(p.p_adjusted for p in holm) == p_adjust([p.p_value for p in pairs], "holm")
    with pytest.raises(ValueError, match=r"p_adjust_method|unknown"):
        rmmcp(columns, p_adjust_method="nope")  # type: ignore[arg-type]
    flat = [columns[0], columns[0] + 1.0, columns[1]]
    with pytest.raises(ValueError, match="conditions 1 and 2"):
        rmmcp(flat)


# ------------------------------------------------------------------ wmcpAKP


def test_average_dependent_akp() -> None:
    columns = _table(16, 4, 8)
    result = average_dependent_akp(columns, seed=3)
    pairs = [
        dependent_akp(columns[i], columns[j]).estimate
        for i, j in itertools.combinations(range(4), 2)
    ]
    assert result.estimate == pytest.approx(math.fsum(pairs) / 6, rel=1e-14)
    assert result.conf_low < result.estimate < result.conf_high
    assert average_dependent_akp(columns, seed=3) == result
    rows = _rng(5).permutation(16)
    assert average_dependent_akp([c[rows] for c in columns], seed=3) == result
    # Reversing the conditions negates every pair's difference: the estimate is
    # exactly negated (the bounds only within Monte Carlo error, other subjects
    # are drawn).
    assert average_dependent_akp(columns[::-1]).estimate == -result.estimate
    with pytest.raises(ValueError, match="at least two conditions"):
        average_dependent_akp([columns[0]])
