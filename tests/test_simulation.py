"""Simulations of error rates and interval coverage (SPEC.md section 11; ``slow``).

Every replication draws normal groups with unequal variances and unequal sizes, runs
the full analysis (``ip.analyze.betweenstats`` on a pandas frame), and records
whether the test rejects at alpha = 0.05 (equal means) or whether the 95% Hedges' g
interval covers the true standardized difference
delta = (mu_1 - mu_2) / sqrt((sigma_1^2 + sigma_2^2) / 2), at delta = 0 and at
nonzero delta. Each rate must lie in the central 99% band of the binomial
distribution of 2,000 replications at the nominal rate, so a correct method fails
one of these tests about once in 100 seeds; the seeds are fixed, so the result is
reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

import inferplot as ip

pytestmark = pytest.mark.slow

betweenstats: Any = ip.analyze.betweenstats

REPLICATIONS = 2_000
ALPHA = 0.05


def binomial_band(rate: float, n: int = REPLICATIONS) -> tuple[float, float]:
    """Central 99% band of the observed rate of a Binomial(n, rate) count."""
    low = float(sps.binom.ppf(0.005, n, rate)) / n
    high = float(sps.binom.ppf(0.995, n, rate)) / n
    return low, high


def simulate(
    sizes: Sequence[int],
    sds: Sequence[float],
    *,
    seed: int,
    means: Sequence[float] | None = None,
) -> list[pd.DataFrame]:
    """REPLICATIONS frames of normal groups with the given SDs and means (0)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    labels = np.repeat([f"g{i}" for i in range(len(sizes))], sizes)
    centres = [0.0] * len(sizes) if means is None else list(means)
    frames: list[pd.DataFrame] = []
    for _ in range(REPLICATIONS):
        y = np.concatenate(
            [
                rng.normal(mu, sd, size=n)
                for n, sd, mu in zip(sizes, sds, centres, strict=True)
            ]
        )
        frames.append(pd.DataFrame({"g": labels, "y": y}))
    return frames


def test_welch_t_type_one_error() -> None:
    frames = simulate((10, 25), (3.0, 1.0), seed=20261108)
    rejections = sum(
        betweenstats(f, "g", "y", bf_message=False).test.p_value < ALPHA for f in frames
    )
    rate = rejections / REPLICATIONS
    low, high = binomial_band(ALPHA)
    print(f"Welch t: Type I error {rate:.4f}, 99% band [{low:.4f}, {high:.4f}]")
    assert low <= rate <= high, f"Welch t Type I error {rate} outside [{low}, {high}]"


def test_welch_anova_type_one_error() -> None:
    frames = simulate((8, 15, 30), (1.0, 2.0, 4.0), seed=20261109)
    rejections = sum(
        betweenstats(f, "g", "y", pairwise_display="none").test.p_value < ALPHA
        for f in frames
    )
    rate = rejections / REPLICATIONS
    low, high = binomial_band(ALPHA)
    print(f"Welch ANOVA: Type I error {rate:.4f}, 99% band [{low:.4f}, {high:.4f}]")
    assert low <= rate <= high, (
        f"Welch ANOVA Type I error {rate} outside [{low}, {high}]"
    )


def _coverage(frames: list[pd.DataFrame], delta: float) -> float:
    covered = 0
    for frame in frames:
        es = betweenstats(frame, "g", "y", bf_message=False).effect_size
        assert es.conf_level == 0.95
        covered += es.conf_low <= delta <= es.conf_high
    return covered / REPLICATIONS


def test_hedges_g_interval_coverage_at_zero_is_the_type_one_error() -> None:
    """The Welch t-test's Type I error test seen through the interval.

    At delta = 0 the noncentral-t interval covers 0 exactly when the two-sided Welch
    p is at least 0.05 (both bounds are the same positive multiple of the ncp
    bounds, and the ncp interval covers 0 iff P(T > t) and P(T < t) are both at
    least alpha / 2), so this repeats the Type I error test; the nonzero-delta test
    below is the coverage test of the interval itself.
    """
    rate = _coverage(simulate((12, 30), (1.0, 2.5), seed=20261110), 0.0)
    low, high = binomial_band(0.95)
    print(f"Hedges' g at 0: CI coverage {rate:.4f}, 99% band [{low:.4f}, {high:.4f}]")
    assert low <= rate <= high, f"Hedges' g CI coverage {rate} outside [{low}, {high}]"


@pytest.mark.parametrize(
    ("difference", "seed"), [(0.5, 20261111), (1.5, 20261112), (-1.0, 20261113)]
)
def test_hedges_g_interval_coverage_of_a_nonzero_difference(
    difference: float, seed: int
) -> None:
    # Unequal SDs (1 and 2.5) and sizes (12 and 30): delta = difference / 1.904,
    # 0.263, 0.788 and -0.525; the first level is the minuend.
    sizes, sds = (12, 30), (1.0, 2.5)
    delta = difference / float(np.sqrt((sds[0] ** 2 + sds[1] ** 2) / 2.0))
    frames = simulate(sizes, sds, seed=seed, means=(difference, 0.0))
    rate = _coverage(frames, delta)
    low, high = binomial_band(0.95)
    print(
        f"Hedges' g at delta {delta:.3f}: CI coverage {rate:.4f}, "
        f"99% band [{low:.4f}, {high:.4f}]"
    )
    assert low <= rate <= high, (
        f"Hedges' g CI coverage of delta {delta} is {rate}, outside [{low}, {high}]"
    )
