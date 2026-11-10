"""inferplot.stats' one-sample methods (histostats): definitions, references,
invariances, degenerate inputs and seeds.

High-precision references (40-digit mpmath, committed under tests/data):
``onesample_ncp_reference.json`` (``tools/reference/onesample_ncp_mpmath.py``), the
noncentral-t inversions behind the one-sample Hedges' g interval at |t| from 0 to
1,000 and n from 2 to 2,000; ``onesamplebf_reference.json``
(``tools/reference/onesamplebf_mpmath.py``), the one-sample JZS Bayes factor on
probe-onesamplebf's 304 rows, where BayesFactor itself misses at large |t| (D61,
D81; R's side is pinned in tests/stats/test_probe_onesamplebf.py).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from inferplot.stats import (
    BOOTSTRAP_RESAMPLES,
    bootstrap_t_critical_value,
    hedges_j,
    jzs_bf10_one_sample,
    jzs_log_bf10,
    one_sample_hedges_g,
    one_sample_rank_biserial,
    one_sample_signed_rank,
    one_sample_t,
    paired_hedges_g,
    paired_rank_biserial,
    t_ncp_interval,
    trimmed_mean,
    trimmed_mean_bootstrap_t,
    winsorized_variance,
)
from tests.oracle_fixtures import probe, r_double

FloatArray = NDArray[np.float64]
DATA: Final = Path(__file__).resolve().parents[1] / "data"
NCP_ROWS = json.loads((DATA / "onesample_ncp_reference.json").read_text())["rows"]
BF_ROWS = json.loads((DATA / "onesamplebf_reference.json").read_text())["rows"]
PROBE_BF_ROWS = probe("probe-onesamplebf").rows("rows")

RNG = np.random.default_rng(20261110)
SAMPLE: Final = RNG.normal(10.0, 3.0, size=37)


# ------------------------------------------------------------------ parametric


@pytest.mark.parametrize("side", ["two-sided", "less", "greater"])
def test_one_sample_t_is_students(side: str) -> None:
    for mu in (0.0, 9.0, 10.5, -3.0):
        mine = one_sample_t(SAMPLE, mu=mu, alternative=side)  # type: ignore[arg-type]
        ref = stats.ttest_1samp(SAMPLE, mu, alternative=side)  # type: ignore[arg-type]
        assert math.isclose(mine.t, float(ref.statistic), rel_tol=1e-13)
        assert math.isclose(mine.p_value, float(ref.pvalue), rel_tol=1e-12)
        assert mine.df == SAMPLE.size - 1
        assert (mine.n, mine.mu, mine.alternative) == (SAMPLE.size, mu, side)


def test_one_sample_t_is_exact_next_to_mu() -> None:
    # The mean lies within the rounding of mu: R's mean(x) - mu loses the digits;
    # the exact difference is (sum x - n mu) / n.
    x = np.array([1e9 + 0.1, 1e9 + 0.2, 1e9 + 0.4, 1e9 + 0.3])
    mu = 1e9 + 0.25
    exact = sum(Fraction(v) for v in x.tolist()) / 4 - Fraction(mu)
    centre = sum(Fraction(v) for v in x.tolist()) / 4
    var = sum((Fraction(v) - centre) ** 2 for v in x.tolist()) / 3
    t = float(exact) / math.sqrt(float(var) / 4)
    assert math.isclose(one_sample_t(x, mu=mu).t, t, rel_tol=1e-14)


def test_hedges_g_is_the_paired_construction_on_x_minus_mu() -> None:
    for side in ("two-sided", "less", "greater"):
        for level in (0.9, 0.95):
            mine = one_sample_hedges_g(
                SAMPLE,
                mu=9.0,
                conf_level=level,
                alternative=side,  # type: ignore[arg-type]
            )
            paired = paired_hedges_g(
                SAMPLE,
                np.full(SAMPLE.size, 9.0),
                conf_level=level,
                alternative=side,  # type: ignore[arg-type]
            )
            for a, b in zip(
                (mine.estimate, mine.conf_low, mine.conf_high),
                (paired.estimate, paired.conf_low, paired.conf_high),
                strict=True,
            ):
                assert a == b or math.isclose(a, b, rel_tol=1e-13)
            assert mine.sidedness == side


def test_hedges_g_definition_and_two_values() -> None:
    n = SAMPLE.size
    d = (SAMPLE.mean() - 9.0) / SAMPLE.std(ddof=1)
    g = one_sample_hedges_g(SAMPLE, mu=9.0)
    assert math.isclose(g.estimate, d * hedges_j(n - 1.0), rel_tol=1e-13)
    # n = 2: J(1) = 0, so g and both bounds are +0.0.
    two = one_sample_hedges_g([1.0, 3.0], mu=0.5)
    assert (two.estimate, two.conf_low, two.conf_high) == (0.0, 0.0, 0.0)
    assert math.copysign(1.0, two.conf_low) == 1.0


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(r, id=f"t{r['t']}-n{r['n']}-{r['side']}{r['tail']}")
        for r in NCP_ROWS
    ],
)
def test_noncentral_bounds_against_mpmath(row: dict[str, object]) -> None:
    # The bounds behind the one-sample Hedges' g interval, |t| from 0 to 1,000 and n
    # from 2 to 2,000, against 40-digit inversions. The negative t mirrors them.
    t = float(str(row["t"]))
    df = int(str(row["n"])) - 1.0
    tail = float(str(row["tail"]))
    expected = float(str(row["ncp"]))
    side = "greater" if row["side"] == "lower" else "less"
    low, high = t_ncp_interval(t, df, conf_level=1.0 - tail, alternative=side)
    mine = low if row["side"] == "lower" else high
    assert math.isclose(mine, expected, rel_tol=2e-13, abs_tol=1e-14), (mine, expected)
    mirror = t_ncp_interval(
        -t,
        df,
        conf_level=1.0 - tail,
        alternative=("less" if side == "greater" else "greater"),
    )
    other = mirror[1] if row["side"] == "lower" else mirror[0]
    assert math.isclose(-other, expected, rel_tol=2e-13, abs_tol=1e-14)


def test_noncentral_reference_grid() -> None:
    assert len(NCP_ROWS) == 7 * 5 * 3
    assert {int(r["n"]) for r in NCP_ROWS} == {2, 5, 30, 200, 2000}
    assert max(float(r["t"]) for r in NCP_ROWS) == 1000.0


def test_hedges_g_at_large_t_follows_the_accurate_inversion() -> None:
    # ToothGrowth against 100 (t = -82.2 on 59 df): R's bounds are -0.30% and
    # -0.55% off (ONE_SAMPLE_R_MISSES); inferplot's are the accurate inversion
    # scaled by J(59) / sqrt(60), here checked at t = 1,000 against mpmath.
    (low_row,) = [
        r
        for r in NCP_ROWS
        if r["t"] == "1000"
        and r["n"] == 30
        and r["side"] == "lower"
        and r["tail"] == "0.025"
    ]
    (high_row,) = [
        r
        for r in NCP_ROWS
        if r["t"] == "1000" and r["n"] == 30 and r["side"] == "upper"
    ]
    x = np.linspace(-1.0, 1.0, 30)
    sd = float(x.std(ddof=1))
    mu = -1000.0 * sd / math.sqrt(30)  # t = 1,000 exactly up to rounding
    g = one_sample_hedges_g(x, mu=mu)
    t = one_sample_t(x, mu=mu).t
    assert math.isclose(t, 1000.0, rel_tol=1e-13)
    scale = hedges_j(29.0) / math.sqrt(30)
    for mine, row in ((g.conf_low, low_row), (g.conf_high, high_row)):
        # d(bound)/dt is about 1, so t's last-bit offset moves the bound by ~1e-13.
        assert math.isclose(mine, float(row["ncp"]) * scale, rel_tol=1e-12)


@pytest.mark.parametrize("index", range(len(BF_ROWS)))
def test_bayes_factor_against_mpmath(index: int) -> None:
    # At R's own t (isolating the integral) within 1e-13 in log BF10, and on the
    # probe's data (inferplot's t, within a few units of 2^-53 of the exact t).
    ref = BF_ROWS[index]
    row = PROBE_BF_ROWS[index]
    assert ref["row"] == index
    assert ref["t_hex"] == row["t_hex"]
    n = float(row["n"])
    r = float.fromhex(row["rscale_hex"])
    expected = float(ref["log_bf10"])
    t = float.fromhex(row["t_hex"])
    mine = jzs_log_bf10(t, n, n - 1.0, prior_scale=r)
    assert abs(mine - expected) <= 1e-13 * max(1.0, abs(expected))
    x = np.array([float.fromhex(v) for v in row["x_hex"]])
    bf10, log_bf01 = jzs_bf10_one_sample(x, mu=float(row["mu"]), prior_scale=r)
    t_exact = float(ref["t_exact"])
    # The data's t against the exact t, then the Bayes factor at the data's t.
    assert math.isclose(
        one_sample_t(x, mu=float(row["mu"])).t, t_exact, rel_tol=1e-14, abs_tol=1e-15
    )
    assert math.isclose(
        -log_bf01,
        jzs_log_bf10(one_sample_t(x, mu=float(row["mu"])).t, n, n - 1.0, prior_scale=r),
        rel_tol=0.0,
        abs_tol=0.0,
    )
    assert math.isfinite(bf10) or -log_bf01 > 709.0


def test_bayesfactor_misses_the_reference_where_d81_says() -> None:
    # R's log BF10 against the 40-digit reference: within ttest_bayes_factor up to
    # |t| 8 (5.2e-9), off beyond (up to +68% at n 5, |t| 1,000).
    worst_small = 0.0
    worst = 0.0
    for ref, row in zip(BF_ROWS, PROBE_BF_ROWS, strict=True):
        r_log = r_double(row["log_bf10_hex"])
        assert r_log is not None
        miss = math.expm1(r_log - float(ref["log_bf10"]))
        if row["t_target"] <= 8:
            worst_small = max(worst_small, abs(miss))
        worst = max(worst, abs(miss))
    assert worst_small < 6e-9
    assert 0.6 < worst < 0.75


# ------------------------------------------------------------------ nonparametric


def _signed_rank_reference(d: FloatArray, side: str) -> tuple[float, float]:
    nonzero = d[d != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    v = float(ranks[nonzero > 0].sum())
    m = nonzero.size
    _, counts = np.unique(np.abs(nonzero), return_counts=True)
    sigma = math.sqrt(
        m * (m + 1) * (2 * m + 1) / 24 - float((counts**3 - counts).sum()) / 48
    )
    centre = v - m * (m + 1) / 4
    if side == "two-sided":
        z = (centre - math.copysign(0.5, centre)) / sigma if centre else 0.0
        return v, 2 * float(stats.norm.sf(abs(z)))
    if side == "greater":
        return v, float(stats.norm.sf((centre - 0.5) / sigma))
    return v, float(stats.norm.cdf((centre + 0.5) / sigma))


@pytest.mark.parametrize("side", ["two-sided", "less", "greater"])
def test_signed_rank_definition(side: str) -> None:
    x = np.round(SAMPLE)  # ties, and values equal to mu
    for mu in (9.0, 10.0, 12.0):
        test = one_sample_signed_rank(x, mu=mu, alternative=side)  # type: ignore[arg-type]
        v, p = _signed_rank_reference(x - mu, side)
        assert test.statistic == v
        assert math.isclose(test.p_value, p, rel_tol=1e-12)
        assert test.n == x.size
        assert test.n_nonzero == int((x != mu).sum())


def test_signed_rank_ranks_the_rounded_differences_as_r() -> None:
    # |5.7 - 5.8| and |5.9 - 5.8| are 0.1 as written but two doubles
    # (0.09999999999999964 and 0.10000000000000053): R ranks them apart, and so
    # does inferplot (the fixtures need it: iris-sepal, ToothGrowth).
    x = np.array([5.7, 5.9, 6.4, 5.0])
    d = x - 5.8
    assert abs(d[0]) < abs(d[1])
    v, _ = _signed_rank_reference(d, "two-sided")
    assert v == 2.0 + 3.0
    assert one_sample_signed_rank(x, mu=5.8).statistic == v


def test_rank_biserial_counts_only_nonzero_differences() -> None:
    x = np.array([3.0, 3, 3, 4, 5, 2, 4, 6, 1, 5, 4, 3])
    mine = one_sample_rank_biserial(x, mu=3.0)
    d = x - 3.0
    nonzero = d[d != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    plus, minus = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    r = (plus - minus) / (plus + minus)
    m = nonzero.size
    se = math.sqrt(m * (m + 1) * (2 * m + 1) / 6) / (m * (m + 1) / 2)
    q = float(stats.norm.isf(0.025))
    assert math.isclose(mine.estimate, r, rel_tol=1e-15)
    assert math.isclose(mine.conf_low, math.tanh(math.atanh(r) - q * se), rel_tol=1e-14)
    assert math.isclose(
        mine.conf_high, math.tanh(math.atanh(r) + q * se), rel_tol=1e-14
    )
    # The paired r counts every pair (zeros too): a different interval.
    paired = paired_rank_biserial(x, np.full(x.size, 3.0))
    assert paired.estimate == mine.estimate
    assert paired.conf_low != mine.conf_low
    greater = one_sample_rank_biserial(x, mu=3.0, alternative="greater")
    assert greater.conf_high == 1.0
    assert math.isclose(
        greater.conf_low, math.tanh(math.atanh(r) - float(stats.norm.isf(0.05)) * se)
    )
    less = one_sample_rank_biserial(x, mu=3.0, alternative="less")
    assert less.conf_low == -1.0


def test_rank_biserial_of_one_sign() -> None:
    r = one_sample_rank_biserial([4.0, 5.0, 6.0, 4.0], mu=3.0)
    assert (r.estimate, r.conf_low, r.conf_high) == (1.0, 1.0, 1.0)
    # Constant x away from mu: the test works (all differences tie).
    test = one_sample_signed_rank([5.0, 5.0, 5.0], mu=3.0)
    assert test.statistic == 6.0


# ------------------------------------------------------------------ robust


def _trimmed_t(x: FloatArray, mu: float, tr: float) -> tuple[float, float]:
    n = x.size
    se = math.sqrt(winsorized_variance(x, tr=tr)) / ((1 - 2 * tr) * math.sqrt(n))
    return trimmed_mean(x, tr=tr), (trimmed_mean(x, tr=tr) - mu) / se


def test_bootstrap_t_definition() -> None:
    res = trimmed_mean_bootstrap_t(SAMPLE, mu=9.0, tr=0.2, seed=3)
    tm, t = _trimmed_t(SAMPLE, 9.0, 0.2)
    assert res.estimate == tm
    assert math.isclose(res.statistic, t, rel_tol=1e-13)
    assert math.isclose(res.conf_low, tm - res.critical_value * res.se, rel_tol=1e-15)
    assert math.isclose(res.conf_high, tm + res.critical_value * res.se, rel_tol=1e-15)
    assert 0.0 <= res.p_value <= 1.0
    assert res.resamples == BOOTSTRAP_RESAMPLES
    assert (res.seed, res.tr, res.n, res.mu) == (3, 0.2, SAMPLE.size, 9.0)
    # Replay: draw the same resamples by hand from the canonical order.
    rng = np.random.Generator(np.random.PCG64(3))
    centre = res.estimate
    deviations = np.sort(SAMPLE) - centre
    deviations = deviations[np.lexsort((deviations, np.abs(deviations)))]
    index = rng.integers(0, SAMPLE.size, size=(BOOTSTRAP_RESAMPLES, SAMPLE.size))
    draws = deviations[index]
    offset = trimmed_mean(deviations, tr=0.2)
    t_star = np.array(
        [
            abs(trimmed_mean(row, tr=0.2) - offset)
            / (
                math.sqrt(winsorized_variance(row, tr=0.2))
                / (0.6 * math.sqrt(SAMPLE.size))
            )
            for row in draws
        ]
    )
    t_star.sort()
    c = float(t_star[round(0.95 * BOOTSTRAP_RESAMPLES) - 1])
    assert math.isclose(res.critical_value, c, rel_tol=1e-12)
    assert res.p_value == float(np.mean(t_star >= abs(res.statistic)))


def test_bootstrap_t_critical_value_rule() -> None:
    ordered = np.arange(1.0, 201.0)
    assert bootstrap_t_critical_value(ordered, 0.95) == 190.0
    assert bootstrap_t_critical_value(ordered, 0.9) == 180.0
    with pytest.raises(ValueError, match="too close to 1"):
        bootstrap_t_critical_value(ordered, 0.999)
    with pytest.raises(ValueError, match="too close to 0"):
        bootstrap_t_critical_value(ordered, 0.001)


def test_bootstrap_t_infinite_resamples() -> None:
    # n = 5, three values the same: a resample whose middle three are one value
    # other than the trimmed mean has T* = inf (kept); none is 0/0 here.
    res = trimmed_mean_bootstrap_t([3.1, 4.7, 2.2, 5.9, 4.4], mu=3.0)
    assert res.infinite_resamples > 100
    assert res.undefined_resamples == 0
    assert (res.conf_low, res.conf_high) == (-math.inf, math.inf)
    # The trimmed mean is a data value: resamples winsorized to it are 0/0.
    zero = trimmed_mean_bootstrap_t([1.0, 2.0, 2.0, 2.0, 3.0, 2.5, 1.5, 2.0], mu=0.0)
    assert zero.undefined_resamples > 0
    assert zero.resamples == BOOTSTRAP_RESAMPLES


def test_bootstrap_t_seeds() -> None:
    first = trimmed_mean_bootstrap_t(SAMPLE, mu=9.0, seed=5)
    assert trimmed_mean_bootstrap_t(SAMPLE, mu=9.0, seed=5) == first
    assert trimmed_mean_bootstrap_t(SAMPLE, mu=9.0, seed=6) != first
    assert trimmed_mean_bootstrap_t(SAMPLE, mu=9.0, seed=np.int64(5)) == first
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match="seed"):
            trimmed_mean_bootstrap_t(SAMPLE, mu=9.0, seed=bad)  # type: ignore[arg-type]


# ------------------------------------------------------------------ invariances


def _all(x: FloatArray, mu: float, seed: int = 1) -> list[float]:
    t = one_sample_t(x, mu=mu)
    g = one_sample_hedges_g(x, mu=mu)
    w = one_sample_signed_rank(x, mu=mu)
    r = one_sample_rank_biserial(x, mu=mu)
    b = trimmed_mean_bootstrap_t(x, mu=mu, seed=seed)
    bf, _ = jzs_bf10_one_sample(x, mu=mu, prior_scale=0.707)
    return [
        t.t, t.p_value, g.estimate, g.conf_low, g.conf_high, w.statistic, w.p_value,
        r.estimate, r.conf_low, r.conf_high, b.statistic, b.p_value,
        b.critical_value, bf,
    ]  # fmt: skip


DYADIC: Final = np.array([2.5, 3.25, 7.0, 1.5, 4.0, 5.75, 3.0, 6.5, 2.0, 4.5, 9.25])


def test_order_does_not_matter() -> None:
    base = _all(SAMPLE, 9.5)
    order = np.random.default_rng(1).permutation(SAMPLE.size)
    assert _all(SAMPLE[order], 9.5) == base


def test_shift_of_x_and_mu_together() -> None:
    # On dyadic data, where the shift is exact: the same numbers, to the rounding
    # of the variances (computed in another power-of-two scale) and of the
    # resampled means.
    for a, b in zip(_all(DYADIC + 64.0, 68.0), _all(DYADIC, 4.0), strict=True):
        assert a == b or math.isclose(a, b, rel_tol=1e-14), (a, b)


@pytest.mark.parametrize("factor", [2.0**-40, 2.0**300, 1e-150, 1e150, 3.7])
def test_change_of_unit(factor: float) -> None:
    base = _all(SAMPLE, 9.5)
    scaled = _all(SAMPLE * factor, 9.5 * factor)
    for a, b in zip(scaled, base, strict=True):
        assert a == b or math.isclose(a, b, rel_tol=1e-12), (factor, a, b)


def test_sign_flip() -> None:
    base = _all(DYADIC, 4.0)
    flipped = _all(-DYADIC, -4.0)
    t, p, g, lo, hi, v, pw, r, rlo, rhi, bt, bp, c, bf = base
    m = int((DYADIC != 4.0).sum())
    assert flipped[0] == -t
    assert flipped[1] == p
    assert flipped[2] == -g
    assert math.isclose(flipped[3], -hi, rel_tol=1e-13)
    assert math.isclose(flipped[4], -lo, rel_tol=1e-13)
    assert flipped[5] == m * (m + 1) / 2 - v
    assert flipped[6] == pw
    assert flipped[7] == -r
    assert flipped[8] == -rhi
    assert flipped[9] == -rlo
    # The bootstrap draws the mirror images (no two values here are exactly as far
    # above as below the trimmed mean), so p is the same and T* and c are the same
    # up to the rounding of the resampled means (summed in mirrored order).
    assert flipped[10] == -bt
    assert flipped[11] == bp
    assert math.isclose(flipped[12], c, rel_tol=1e-14)
    assert flipped[13] == bf


# ------------------------------------------------------------------ degenerate


def test_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        one_sample_t([1.0], mu=0.0)
    with pytest.raises(ValueError, match="same"):
        one_sample_t([2.0, 2.0, 2.0], mu=0.0)
    with pytest.raises(ValueError, match="same"):
        # 0.1 + 0.2 and 0.3 are one number as written (D74).
        one_sample_hedges_g([0.1 + 0.2, 0.3, 0.3], mu=0.0)
    with pytest.raises(ValueError, match="same"):
        jzs_bf10_one_sample([4.0, 4.0], mu=0.0, prior_scale=0.707)
    with pytest.raises(ValueError, match="finite"):
        one_sample_t([1.0, math.nan], mu=0.0)
    with pytest.raises(ValueError, match="finite"):
        one_sample_t([1.0, 2.0], mu=math.inf)
    with pytest.raises(ValueError, match="test value"):
        one_sample_signed_rank([3.0, 3.0], mu=3.0)
    with pytest.raises(ValueError, match="test value"):
        one_sample_rank_biserial([3.0], mu=3.0)
    with pytest.raises(ValueError, match="at least 2 values after trimming"):
        trimmed_mean_bootstrap_t([1.0, 2.0, 3.0], mu=0.0, tr=0.4)
    with pytest.raises(ValueError, match="winsorized variance of 0"):
        trimmed_mean_bootstrap_t([1.0, 5.0, 5.0, 5.0, 9.0], mu=0.0)
    with pytest.raises(ValueError, match="tr must"):
        trimmed_mean_bootstrap_t(SAMPLE, mu=0.0, tr=0.5)
    with pytest.raises(ValueError, match="conf_level"):
        one_sample_hedges_g(SAMPLE, mu=0.0, conf_level=1.0)
    with pytest.raises(ValueError, match="alternative"):
        one_sample_t(SAMPLE, mu=0.0, alternative="two.sided")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="largest double"):
        one_sample_t([1.0, 1.0 + 2.0**-40], mu=-1e300)
