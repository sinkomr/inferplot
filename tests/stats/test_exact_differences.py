"""Differences of means against exact rational arithmetic on the raw doubles.

Every statistic that subtracts means (Welch's and Student's t, Hedges' g,
Games-Howell's q, Welch's F, Yuen's t, the trimmed-means F, lincon, AKP and the
explanatory effect size) forms each difference from the exact means and rounds
once (DECISIONS.md, the exact-difference rule). Subtracting two rounded means
instead loses up to half a unit in the last place (ulp) of each, an error of about
ulp(mean) / |difference| relative: on groups that differ only in their last bits
the t-test reported t = -2.62, p = 0.04 where the exact t is -0.655, p = 0.537.

The references below use ``fractions.Fraction`` on the doubles as given, so they
are exact up to one final rounding (and a square root). The bound on our relative
error is ``_ULPS`` units of 2^-53: the statistics round each variance (measured at
most 3.4e-16 relative, about 3 units, ``test_variance_is_accurate_on_hard_data``),
a few products and one square root.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import special

from inferplot.stats import (
    akp_effect,
    explanatory_effect,
    games_howell,
    hedges_g,
    hedges_j,
    lincon,
    student_t,
    trimmed_means_anova,
    welch_anova,
    welch_t,
    winsorized_normal_variance,
    yuen_t,
)
from inferplot.stats.descriptive import (
    common_scale,
    deviations_from_weighted_mean,
    exact_mean,
    exact_variance,
    scaled_variance,
)

FloatArray = NDArray[np.float64]
Exact = Fraction

_U = 2.0**-53
# Measured worst over every case below: 9.2 units (Hedges' g, where J(df) adds
# its own rounding), at most 2.0 for every other statistic. Subtracting rounded
# means was off by up to 27 relative (2.4e17 units) on the same data.
_ULPS = 16.0

THIRD = 0.1 + 0.2  # 0.30000000000000004, one ulp above 0.3


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _cases() -> dict[str, list[FloatArray]]:
    rng = _rng(7)
    stamps = 1.7e9 + np.round(rng.random((3, 9)), 6)
    return {
        # Groups that differ only in the last bit (the review's example).
        "last bit": [
            np.array([THIRD, 0.3, 0.3, 0.3]),
            np.array([0.3, 0.3, THIRD, THIRD]),
            np.array([0.3, THIRD, 0.3, THIRD, 0.3]),
        ],
        # Values 1 + k 2^-52: every difference sits in the last bits.
        "last bits of 1": [
            1.0 + 2.0**-52 * rng.integers(0, 8, n).astype(np.float64)
            for n in (9, 11, 10)
        ],
        # Epoch seconds with microseconds.
        "timestamps": [stamps[0], stamps[1] + 0.05, stamps[2]],
        # A large common level, small multiples of 2^-2 on top (exact doubles).
        "offset 2^50": [
            2.0**50 + 0.25 * rng.integers(0, 9, n).astype(np.float64) for n in (7, 9, 8)
        ],
        # A negative level of 3e12 with steps of 1e-3.
        "offset -3e12": [
            -3e12 + 1e-3 * rng.integers(0, 50, n).astype(np.float64) + shift
            for n, shift in ((10, 0.0), (8, 0.004), (12, 0.002))
        ],
        # 1e6 with differences of 1e-6.
        "1e6 + 1e-6 k": [
            1e6 + 1e-6 * np.array([1.0, 3.0, 2.0, 5.0, 4.0, 2.0]),
            1e6 + 1e-6 * np.array([2.0, 4.0, 3.0, 7.0, 4.5, 6.0]),
            1e6 + 1e-6 * np.array([0.0, 2.0, 1.0, 3.0, 2.5, 1.5]),
        ],
    }


CASES = _cases()
NAMES = sorted(CASES)


# ------------------------------------------------------------------ references


def _fr(x: FloatArray) -> list[Exact]:
    return [Fraction(v) for v in x.tolist()]


def _mean(x: Sequence[Exact]) -> Exact:
    return sum(x, Fraction(0)) / len(x)


def _var(x: Sequence[Exact]) -> Exact:
    m = _mean(x)
    return sum(((v - m) ** 2 for v in x), Fraction(0)) / (len(x) - 1)


def _signed_sqrt_ratio(num: Exact, den_sq: Exact) -> float:
    """num / sqrt(den_sq), from the exact square (one rounding and a sqrt)."""
    return math.copysign(math.sqrt(float(num * num / den_sq)), float(num))


def _satterthwaite(w: Sequence[Exact], df: Sequence[int]) -> float:
    total = sum(w, Fraction(0))
    return float(
        total**2 / sum((wi**2 / d for wi, d in zip(w, df, strict=True)), Fraction(0))
    )


def _welch(a: FloatArray, b: FloatArray) -> tuple[float, float, Exact]:
    xa, xb = _fr(a), _fr(b)
    w = [_var(xa) / len(xa), _var(xb) / len(xb)]
    diff = _mean(xa) - _mean(xb)
    return (
        _signed_sqrt_ratio(diff, w[0] + w[1]),
        _satterthwaite(w, [len(xa) - 1, len(xb) - 1]),
        diff,
    )


def _sorted_fr(x: FloatArray) -> list[Exact]:
    return sorted(_fr(x))


def _trim(x: FloatArray, tr: float = 0.2) -> tuple[Exact, Exact, int, int]:
    """Exact trimmed mean, winsorized variance, n and h."""
    s = _sorted_fr(x)
    n = len(s)
    g = math.floor(tr * n)
    kept = s[g : n - g]
    wins = [kept[0]] * g + kept + [kept[-1]] * g
    return _mean(kept), _var(wins), n, n - 2 * g


def _yuen_d(x: FloatArray) -> tuple[Exact, Exact, int]:
    tm, wv, n, h = _trim(x)
    return tm, Fraction(n - 1) * wv / (h * (h - 1)), h


def _welch_type_f(
    centres: Sequence[Exact], weights: Sequence[Exact], df: Sequence[int]
) -> float:
    k = len(centres)
    total = sum(weights, Fraction(0))
    c = sum((w * m for w, m in zip(weights, centres, strict=True)), Fraction(0)) / total
    between = sum(
        (w * (m - c) ** 2 for w, m in zip(weights, centres, strict=True)), Fraction(0)
    )
    lam = sum(
        ((1 - w / total) ** 2 / d for w, d in zip(weights, df, strict=True)),
        Fraction(0),
    )
    return float(between / (k - 1) / (1 + Fraction(2 * (k - 2)) * lam / (k * k - 1)))


def _close(ours: float, exact: float, what: str) -> None:
    assert abs(ours - exact) <= _ULPS * _U * abs(exact), (
        f"{what}: {ours!r} against exact {exact!r} "
        f"({abs(ours - exact) / abs(exact) / _U:.3g} units of 2^-53)"
    )


# ------------------------------------------------------------------ the review


def test_the_review_example() -> None:
    a, b, c = CASES["last bit"]
    welch = welch_t(a, b)
    assert welch.t == pytest.approx(-0.6546536707079772, rel=1e-15)
    assert welch.p_value == pytest.approx(0.53744, abs=1e-5)  # was 0.04 (t = -2.62)
    assert welch.mean_difference == float(_welch(a, b)[2])
    anova = welch_anova([a, b, c])
    assert anova.f == pytest.approx(0.20289, abs=1e-5)  # was 5.44, p 0.041
    assert anova.p_value == pytest.approx(0.821, abs=1e-3)


# ------------------------------------------------------------------ parametric


@pytest.mark.parametrize("name", NAMES)
def test_welch_and_student_t(name: str) -> None:
    a, b, _ = CASES[name]
    t, df, diff = _welch(a, b)
    welch = welch_t(a, b)
    _close(welch.t, t, "Welch t")
    _close(welch.df, df, "Welch df")
    # The difference in y units is the exact one, correctly rounded.
    assert welch.mean_difference == float(diff)
    p = 2.0 * float(special.stdtr(df, -abs(t)))
    assert welch.p_value == pytest.approx(p, rel=1e-12)
    xa, xb = _fr(a), _fr(b)
    na, nb = len(xa), len(xb)
    pooled = ((na - 1) * _var(xa) + (nb - 1) * _var(xb)) / (na + nb - 2)
    student = _signed_sqrt_ratio(diff, pooled * Fraction(na + nb, na * nb))
    _close(student_t(a, b).t, student, "Student t")


@pytest.mark.parametrize("name", NAMES)
def test_hedges_g(name: str) -> None:
    a, b, _ = CASES[name]
    xa, xb = _fr(a), _fr(b)
    _, df, diff = _welch(a, b)
    d = _signed_sqrt_ratio(diff, (_var(xa) + _var(xb)) / 2)
    _close(hedges_g(a, b).estimate, d * hedges_j(df), "Hedges' g")


@pytest.mark.parametrize("name", NAMES)
def test_games_howell(name: str) -> None:
    groups = CASES[name]
    exact = [_fr(g) for g in groups]
    w = [_var(x) / len(x) for x in exact]
    for pair in games_howell(groups):
        i, j = pair.i, pair.j
        q = _signed_sqrt_ratio(_mean(exact[j]) - _mean(exact[i]), (w[i] + w[j]) / 2)
        _close(pair.statistic, q, f"Games-Howell q {i}-{j}")


@pytest.mark.parametrize("name", NAMES)
def test_welch_anova(name: str) -> None:
    groups = CASES[name]
    exact = [_fr(g) for g in groups]
    weights = [Fraction(len(x)) / _var(x) for x in exact]
    f = _welch_type_f([_mean(x) for x in exact], weights, [len(x) - 1 for x in exact])
    _close(welch_anova(groups).f, f, "Welch F")


# ------------------------------------------------------------------ robust


@pytest.mark.parametrize("name", NAMES)
def test_yuen_and_lincon(name: str) -> None:
    groups = CASES[name]
    kernels = [_yuen_d(g) for g in groups]
    for pair in lincon(groups):
        (mi, di, _), (mj, dj, _) = kernels[pair.i], kernels[pair.j]
        t = _signed_sqrt_ratio(mi - mj, di + dj)
        _close(pair.t, t, f"lincon t {pair.i}-{pair.j}")
        assert pair.estimate == float(mi - mj)
    (ma, da, ha), (mb, db, hb) = kernels[0], kernels[1]
    yuen = yuen_t(groups[0], groups[1])
    _close(yuen.t, abs(_signed_sqrt_ratio(ma - mb, da + db)), "Yuen t")
    _close(yuen.df, _satterthwaite([da, db], [ha - 1, hb - 1]), "Yuen df")
    assert yuen.difference == float(ma - mb)


@pytest.mark.parametrize("name", NAMES)
def test_trimmed_means_anova(name: str) -> None:
    groups = CASES[name]
    kernels = [_yuen_d(g) for g in groups]
    f = _welch_type_f(
        [m for m, _, _ in kernels],
        [1 / d for _, d, _ in kernels],
        [h - 1 for _, _, h in kernels],
    )
    _close(trimmed_means_anova(groups).f, f, "trimmed-means F")


# "last bit" has two distinct values per group: some resamples are constant and
# the bootstrap refuses them (0 / 0); "last bits of 1" covers the same ground.
@pytest.mark.parametrize("name", [n for n in NAMES if n != "last bit"])
def test_akp_estimate(name: str) -> None:
    a, b, _ = CASES[name]
    ma, wva, _, _ = _trim(a)
    mb, _, _, _ = _trim(b)
    constant = math.sqrt(winsorized_normal_variance(0.2))
    exact = constant * _signed_sqrt_ratio(ma - mb, wva)
    _close(akp_effect(a, b).estimate, exact, "AKP")


def test_explanatory_estimate() -> None:
    # Equal group sizes: the estimate is xi of the data, not a bootstrap median.
    rng = _rng(3)
    groups = [1.7e9 + np.round(rng.random(8), 6) + 0.01 * i for i in range(3)]
    exact = [_trim(g) for g in groups]
    top = _var([m for m, _, _, _ in exact])
    pooled = _trim(np.concatenate(groups))[1]
    xi2 = top / pooled * Fraction(winsorized_normal_variance(0.2))
    _close(explanatory_effect(groups).estimate, math.sqrt(float(xi2)), "xi")


def test_bootstrap_resamples_are_shifted_before_averaging() -> None:
    # Resampled trimmed means are averaged in floating point; on timestamps the
    # rounding at 1.7e9 (2.4e-7) moved the AKP bounds by 1e-6 relative before the
    # resamples were shifted to the middle of the data. Now the bounds do not
    # change when the data move by a whole number of seconds.
    rng = _rng(11)
    a, b = np.round(rng.random(15), 6), np.round(rng.random(12), 6) + 0.1
    base = akp_effect(a, b)
    moved = akp_effect(a + 1.7e9, b + 1.7e9)
    # The shifted data are a + 1.7e9 rounded once (ulp 2.4e-7), so the bounds agree
    # to about 2.4e-7 / spread, not exactly.
    assert moved.conf_low == pytest.approx(base.conf_low, rel=1e-5)
    assert moved.conf_high == pytest.approx(base.conf_high, rel=1e-5)
    x0, x1 = (
        explanatory_effect([a, b, b + 0.2]),
        explanatory_effect([a + 1.7e9, b + 1.7e9, b + 0.2 + 1.7e9]),
    )
    assert x1.conf_low == pytest.approx(x0.conf_low, rel=1e-5)
    assert x1.conf_high == pytest.approx(x0.conf_high, rel=1e-5)


# ------------------------------------------------------------------ helpers


def test_variance_is_accurate_on_hard_data() -> None:
    # The corrected two-pass variance centres on the exact mean in effect (the
    # correction term removes the rounding of the mean): measured worst 3.4e-16
    # relative over these 3,000 samples, so no change was needed there.
    rng = _rng(0)
    worst = 0.0
    for trial in range(3000):
        n = int(rng.integers(2, 40))
        kind = trial % 6
        if kind == 0:
            x = 1.7e9 + rng.random(n)
        elif kind == 1:
            x = 1.0 + rng.integers(-3, 4, n) * 2.0**-52
        elif kind == 2:
            x = 1e6 + 1e-6 * rng.integers(0, 9, n)
        elif kind == 3:
            x = np.where(rng.random(n) < 0.5, 0.3, THIRD)
        elif kind == 4:
            x = rng.normal(1e8, 1e-3, n)
        else:
            x = rng.normal(0.0, 1.0, n) * 10.0 ** rng.integers(-5, 5, n)
        x = np.asarray(x, dtype=np.float64)
        scaled: FloatArray = x * common_scale(x)
        exact = _var(_fr(scaled))
        if exact == 0:
            continue
        error = float(abs(Fraction(scaled_variance(scaled)) - exact) / exact)
        worst = max(worst, error)
    assert worst <= 4 * _U


def test_exact_helpers() -> None:
    x = np.array([THIRD, 0.3, 0.3])
    assert exact_mean(x) == _mean(_fr(x))
    assert exact_variance([Fraction(1), Fraction(2), Fraction(4)]) == float(
        Fraction(7, 3)
    )
    means = [Fraction(1, 3), Fraction(2, 3)]
    deviations = deviations_from_weighted_mean([1.0, 3.0], means)
    assert deviations == [float(Fraction(-1, 4)), float(Fraction(1, 12))]
