"""inferplot.stats' correlations (scatterstats): definitions, the Bayes factor
against high-precision references, exactness, invariances and degenerate inputs.

``tests/data/corrbf_reference.json`` (``tools/reference/corrbf_mpmath.py``) holds
40-digit mpmath values of the correlation Bayes factor's hypergeometric form on a
grid of n 3 to 2,000, r 0 to 1 - 1e-7 and r = 1, at kappa 0.707, 0.8 and 1/3.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from inferplot.stats import (
    correlation_bf10,
    correlation_log_bf10,
    pearson,
    spearman,
    winsorized_correlation,
)
from inferplot.stats.correlation import cross_moments

FloatArray = NDArray[np.float64]
REFERENCE: Final = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "corrbf_reference.json").read_text()
)["rows"]
RNG = np.random.default_rng(20261111)
X: Final = RNG.normal(5.0, 2.0, size=40)
Y: Final = 0.6 * X + RNG.normal(0.0, 1.5, size=40)


def _exact_r(x: FloatArray, y: FloatArray) -> tuple[Fraction, Fraction, Fraction]:
    fx = [Fraction(v) for v in x.tolist()]
    fy = [Fraction(v) for v in y.tolist()]
    mx = sum(fx, Fraction(0)) / len(fx)
    my = sum(fy, Fraction(0)) / len(fy)
    sxx = sum(((a - mx) ** 2 for a in fx), Fraction(0))
    syy = sum(((b - my) ** 2 for b in fy), Fraction(0))
    sxy = sum(((a - mx) * (b - my) for a, b in zip(fx, fy, strict=True)), Fraction(0))
    return sxx, syy, sxy


# ------------------------------------------------------------------ definitions


def test_pearson_definition() -> None:
    res = pearson(X, Y)
    ref = stats.pearsonr(X, Y)
    assert math.isclose(res.estimate, float(ref.statistic), rel_tol=1e-14)
    assert math.isclose(res.p_value, float(ref.pvalue), rel_tol=1e-11)
    n = X.size
    t = res.estimate * math.sqrt((n - 2) / (1 - res.estimate**2))
    assert math.isclose(res.t, t, rel_tol=1e-13)
    assert res.statistic == res.t
    assert res.df == n - 2
    q = float(stats.norm.isf(0.025))
    z = math.atanh(res.estimate)
    assert math.isclose(
        res.conf_low, math.tanh(z - q / math.sqrt(n - 3)), rel_tol=1e-14
    )
    assert math.isclose(
        res.conf_high, math.tanh(z + q / math.sqrt(n - 3)), rel_tol=1e-14
    )
    assert (res.method, res.n, res.conf_level) == ("pearson", n, 0.95)


def test_pearson_is_exact() -> None:
    # Against exact rational sums, on values that differ only in their last bits
    # on a large offset and on wide-ranging values.
    for x, y in (
        (1.7e9 + RNG.random(25), 0.25 * RNG.random(25) - 1e12),
        (
            RNG.normal(size=30) * 10.0 ** RNG.integers(-30, 30, size=30),
            RNG.normal(size=30),
        ),
    ):
        sxx, syy, sxy = _exact_r(x, y)
        r2 = sxy * sxy / (sxx * syy)
        res = pearson(x, y)
        assert math.isclose(res.estimate**2, float(r2), rel_tol=4e-16)
        t2 = (len(x) - 2) * sxy * sxy / (sxx * syy - sxy * sxy)
        assert math.isclose(res.t**2, float(t2), rel_tol=4e-16)
        moments = cross_moments(x, y)
        kx, ky = Fraction(moments.x_scale), Fraction(moments.y_scale)
        assert moments.sxx == sxx * kx * kx
        assert moments.sxy == sxy * kx * ky


def test_spearman_definition() -> None:
    xr = np.round(X)  # ties
    res = spearman(xr, Y)
    ref = stats.spearmanr(xr, Y)
    assert math.isclose(res.estimate, float(ref.statistic), rel_tol=1e-14)
    # p from the t approximation on n - 2 df, always.
    n = X.size
    t = res.estimate * math.sqrt((n - 2) / (1 - res.estimate**2))
    assert math.isclose(
        res.p_value, 2 * float(stats.t.sf(abs(t), n - 2)), rel_tol=1e-12
    )
    assert math.isclose(
        res.statistic, (n**3 - n) * (1 - res.estimate) / 6, rel_tol=1e-13
    )
    q = float(stats.norm.isf(0.025))
    se = math.sqrt(1.06 / (n - 3))
    z = math.atanh(res.estimate)
    assert math.isclose(res.conf_low, math.tanh(z - q * se), rel_tol=1e-14)
    assert math.isclose(res.conf_high, math.tanh(z + q * se), rel_tol=1e-14)
    # Without ties S is the sum of squared rank differences, exactly.
    plain = spearman(X, Y)
    d = stats.rankdata(X) - stats.rankdata(Y)
    assert plain.statistic == float((d * d).sum())


def test_spearman_small_n_uses_the_t_approximation() -> None:
    # edge_corr_small: R prints p 0.037 (t approximation), the exact p is 0.083.
    x = np.array([1.2, 2.9, 3.1, 4.8, 5.5])
    y = np.array([2.3, 2.1, 4.0, 4.6, 6.2])
    res = spearman(x, y)
    assert res.estimate == 0.9
    assert res.statistic == 2.0
    assert math.isclose(res.p_value, 0.0373860734684988, rel_tol=1e-12)
    exact = stats.spearmanr(x, y)  # SciPy's own p: the t approximation as well
    assert math.isclose(res.p_value, float(exact.pvalue), rel_tol=1e-12)


def test_winsorized_correlation_definition() -> None:
    for tr in (0.1, 0.2, 0.29):
        res = winsorized_correlation(X, Y, tr=tr)
        g = math.floor(tr * X.size)
        wx = np.clip(X, np.sort(X)[g], np.sort(X)[X.size - g - 1])
        wy = np.clip(Y, np.sort(Y)[g], np.sort(Y)[Y.size - g - 1])
        ref = pearson(wx, wy)
        assert res.estimate == ref.estimate
        assert (res.t, res.p_value, res.conf_low) == (ref.t, ref.p_value, ref.conf_low)
        assert res.method == "winsorized"
    # tr 0.29 at n 100 trims floor(28.999999999999996) = 28, as R's floor does.
    x = np.arange(100.0)
    res = winsorized_correlation(x, x[::-1] + RNG.normal(size=100), tr=0.29)
    assert res.n == 100
    assert math.floor(0.29 * 100) == 28


# ------------------------------------------------------------------ exactly collinear


def test_exactly_collinear_data() -> None:
    x = np.arange(1.0, 11.0)
    for y, sign in ((2 * x + 1, 1.0), (27 - 3 * x, -1.0)):
        for res in (pearson(x, y), winsorized_correlation(x, y)):
            assert res.estimate == sign
            assert res.t == sign * math.inf
            assert res.p_value == 0.0
            assert (res.conf_low, res.conf_high) == (sign, sign)
        rho = spearman(x, y)
        assert rho.estimate == sign
        assert rho.statistic == (0.0 if sign > 0 else 330.0)
        assert rho.p_value == 0.0
        bf10, log_bf01 = correlation_bf10(x, y, prior_scale=0.707)
        assert (bf10, log_bf01) == (math.inf, -math.inf)


def test_nearly_collinear_data_keep_their_t() -> None:
    # y = 2x + 1 plus a perturbation of one part in 1e12: t is large but finite
    # and exact to rounding, where 1 - r^2 from a rounded r would lose 4 digits.
    x = np.arange(1.0, 21.0)
    y = 2 * x + 1 + 1e-11 * np.sin(x)
    sxx, syy, sxy = _exact_r(x, y)
    t2 = 18 * sxy * sxy / (sxx * syy - sxy * sxy)
    res = pearson(x, y)
    assert math.isfinite(res.t)
    assert math.isclose(res.t, math.sqrt(float(t2)), rel_tol=1e-15)


# ------------------------------------------------------------------ Bayes factor


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(r, id=f"n{r['n']}-r{r['r']}-k{float.fromhex(r['kappa_hex']):.3f}")
        for r in REFERENCE
    ],
)
def test_bayes_factor_against_mpmath(row: dict[str, object]) -> None:
    kappa = float.fromhex(str(row["kappa_hex"]))
    n = int(str(row["n"]))
    r = float(str(row["r"]))
    expected = math.inf if row["log_bf10"] == "inf" else float(str(row["log_bf10"]))
    for sign in (1.0, -1.0):
        mine = correlation_log_bf10(sign * r, n, prior_scale=kappa)
        if math.isinf(expected):
            assert mine == expected
            continue
        # BF10 within 1e-14 relative, plus a few roundings of a large log BF10
        # (measured: 6.3e-16 relative in the log at n 2,000, log BF10 91; 1.2e-16
        # at 15,384).
        assert abs(mine - expected) <= 1e-14 + 1e-15 * abs(expected)


def test_bayes_factor_reference_grid() -> None:
    assert len(REFERENCE) == 8 * 10 * 3
    assert {r["n"] for r in REFERENCE} == {3, 4, 5, 10, 30, 100, 500, 2000}


def test_bayes_factor_at_r_one() -> None:
    # The limit as |r| -> 1: finite while (n - 1) / 2 < 1/kappa + 1/2 (Gauss's 2F1
    # at 1), infinite from there; continuous in r.
    for n, kappa in ((3, 0.707), (4, 0.707), (5, 1 / 3), (7, 1 / 3)):
        at_one = correlation_log_bf10(1.0, n, prior_scale=kappa)
        near = correlation_log_bf10(1 - 1e-12, n, prior_scale=kappa)
        assert math.isfinite(at_one)
        assert math.isclose(at_one, near, rel_tol=1e-4)
    assert correlation_log_bf10(-1.0, 5, prior_scale=0.707) == math.inf
    assert correlation_log_bf10(1.0, 8, prior_scale=1 / 3) == math.inf


def test_bayes_factor_data_and_prior() -> None:
    bf10, log_bf01 = correlation_bf10(X, Y, prior_scale=0.707)
    r = pearson(X, Y).estimate
    assert math.isclose(
        -log_bf01, correlation_log_bf10(r, X.size, prior_scale=0.707), rel_tol=1e-14
    )
    assert math.isclose(bf10, math.exp(-log_bf01), rel_tol=1e-15)
    # 0.707 is not sqrt(2)/2.
    other = correlation_log_bf10(r, X.size, prior_scale=math.sqrt(2) / 2)
    assert abs(other + log_bf01) > 1e-6
    # Overflow: BF10 is inf while log BF01 stays finite.
    big = np.arange(2000.0)
    bf, lb = correlation_bf10(
        big, big + RNG.normal(0, 30, size=2000), prior_scale=0.707
    )
    assert bf == math.inf
    assert math.isfinite(lb)
    assert lb < -709.0


# ------------------------------------------------------------------ invariances


def _numbers(x: FloatArray, y: FloatArray) -> list[float]:
    out: list[float] = []
    for res in (pearson(x, y), spearman(x, y), winsorized_correlation(x, y)):
        out += [res.estimate, res.statistic, res.p_value, res.conf_low, res.conf_high]
    out.append(correlation_bf10(x, y, prior_scale=0.707)[0])
    return out


def test_order_of_pairs_and_swapping_x_and_y() -> None:
    base = _numbers(X, Y)
    order = np.random.default_rng(3).permutation(X.size)
    assert _numbers(X[order], Y[order]) == base
    assert _numbers(Y, X) == base


@pytest.mark.parametrize(
    ("shift", "scale"),
    [
        (0.0, 2.0**-60),
        (0.0, 2.0**200),
        (1e6, 1.0),
        (0.0, 1e-150),
        (0.0, 1e150),
        (-3.0, 7.3),
    ],
)
def test_shift_and_positive_rescaling(shift: float, scale: float) -> None:
    base = _numbers(X, Y)
    for got in (_numbers(X * scale + shift, Y), _numbers(X, Y * scale + shift)):
        for a, b in zip(got, base, strict=True):
            assert a == b or math.isclose(a, b, rel_tol=1e-9), (shift, scale, a, b)
    # Spearman's rho depends on the ranks only: exactly the same.
    assert spearman(X * scale + shift, Y) == spearman(X, Y)


def test_negative_rescaling_flips_the_sign() -> None:
    for fn in (pearson, spearman, winsorized_correlation):
        base = fn(X, Y)
        flipped = fn(-2.0 * X, Y)
        assert math.isclose(flipped.estimate, -base.estimate, rel_tol=1e-14)
        assert math.isclose(flipped.p_value, base.p_value, rel_tol=1e-12)
        assert math.isclose(flipped.conf_low, -base.conf_high, rel_tol=1e-14)
    assert math.isclose(
        correlation_bf10(-X, Y, prior_scale=0.8)[0],
        correlation_bf10(X, Y, prior_scale=0.8)[0],
        rel_tol=1e-14,
    )


# ------------------------------------------------------------------ degenerate


def test_degenerate_inputs() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="every value of x is the same"):
        pearson(np.full(5, 5.0), x)
    with pytest.raises(ValueError, match="every value of y is the same"):
        spearman(x, np.full(5, 2.0))
    with pytest.raises(ValueError, match="every value of x is the same"):
        # 0.1 + 0.2 and 0.3 are one number as written (D74).
        pearson(np.array([0.1 + 0.2, 0.3, 0.3, 0.3]), np.arange(4.0))
    with pytest.raises(ValueError, match="after winsorizing"):
        winsorized_correlation(
            np.array([1.0, 5, 5, 5, 5, 5, 5, 9, 2, 5]), np.arange(10.0)
        )
    with pytest.raises(ValueError, match="at least 4 pairs"):
        pearson(x[:3], x[:3] ** 2)
    with pytest.raises(ValueError, match="at least 3"):
        pearson(x[:2], x[:2])
    with pytest.raises(ValueError, match="same number"):
        pearson(x, x[:4])
    with pytest.raises(ValueError, match="finite"):
        pearson(np.array([1.0, 2.0, math.inf, 4.0]), x[:4])
    with pytest.raises(ValueError, match="conf_level"):
        pearson(X, Y, conf_level=1.0)
    with pytest.raises(ValueError, match="tr must"):
        winsorized_correlation(X, Y, tr=0.5)
    with pytest.raises(ValueError, match="prior_scale"):
        correlation_bf10(X, Y, prior_scale=0.0)
    with pytest.raises(ValueError, match="n >= 3"):
        correlation_log_bf10(0.5, 2, prior_scale=0.707)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        correlation_log_bf10(1.5, 10, prior_scale=0.707)
    # n = 3: the Bayes factor exists (only the Fisher interval needs n > 3).
    bf10, _ = correlation_bf10(x[:3], np.array([1.0, 3.0, 2.0]), prior_scale=0.707)
    assert math.isfinite(bf10)


# ------------------------------------------------------------ collinear as written


def _celsius() -> NDArray[np.float64]:
    return np.round(np.random.default_rng(1).normal(20, 5, 20), 1)


@pytest.mark.parametrize(
    ("make_y", "sign"),
    [
        (lambda c: c * 1.8 + 32, 1.0),  # Fahrenheit
        (lambda c: c * 2.54, 1.0),  # inches to centimetres
        (lambda c: c / 10, 1.0),
        (lambda c: 100 - c, -1.0),
    ],
)
def test_collinear_as_written_is_collinear(
    make_y: Callable[[float], float], sign: float
) -> None:
    # Review Z2: decimal columns collinear as written printed t of order 1e15 and
    # BF10 of 1e228 without a warning; as in binary-exact data (D85) they now give
    # r = +/-1, t = +/-inf, p = 0, [r, r] and the Bayes factor's limit.
    x = _celsius()
    y = np.array([make_y(v) for v in x.tolist()])
    result = pearson(x, y)
    assert (result.estimate, result.t, result.p_value) == (sign, sign * math.inf, 0.0)
    assert (result.conf_low, result.conf_high) == (sign, sign)
    assert winsorized_correlation(x, y).t == sign * math.inf
    assert correlation_bf10(x, y, prior_scale=1 / 0.707)[0] == math.inf


def test_nearly_collinear_data_are_not_swallowed() -> None:
    # Measured data with real noise far below the printed digits: 1 - r^2 of about
    # 1e-12, finite t, as computed from the doubles.
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 50)
    y = 2 * x + 1 + 1e-6 * rng.normal(0, 1, 50)
    result = pearson(x, y)
    om = float(cross_moments(x, y).one_minus_r2)
    assert 1e-14 < om < 1e-10
    assert math.isfinite(result.t)
    assert result.t == pytest.approx(math.sqrt(48 / om), rel=1e-9)
    # Decimals with real noise in their last written digit: not collinear either.
    c = _celsius()
    f = np.round(c * 1.8 + 32 + rng.choice([-0.0001, 0.0, 0.0001], c.size), 4)
    assert math.isfinite(pearson(c, f).t)
    assert 0.0 < 1 - abs(pearson(c, f).estimate) < 1e-6
