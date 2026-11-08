"""Simulations under the null hypothesis (SPEC.md section 11; marker ``slow``).

Every replication draws normal groups with equal means but unequal variances and
unequal sizes, runs the full analysis (``ip.analyze.betweenstats`` on a pandas
frame), and records whether the test rejects at alpha = 0.05 or whether the 95%
Hedges' g interval covers the true standardized difference, 0. Each rate must lie in
the central 99% band of the binomial distribution of 2,000 replications at the
nominal rate, so a correct method fails one of these tests about once in 100 seeds;
the seeds are fixed, so the result is reproducible.
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
    sizes: Sequence[int], sds: Sequence[float], *, seed: int
) -> list[pd.DataFrame]:
    """REPLICATIONS frames of normal groups with mean 0 and the given SDs."""
    rng = np.random.Generator(np.random.PCG64(seed))
    labels = np.repeat([f"g{i}" for i in range(len(sizes))], sizes)
    frames: list[pd.DataFrame] = []
    for _ in range(REPLICATIONS):
        y = np.concatenate(
            [rng.normal(0.0, sd, size=n) for n, sd in zip(sizes, sds, strict=True)]
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


def test_hedges_g_interval_coverage() -> None:
    frames = simulate((12, 30), (1.0, 2.5), seed=20261110)
    covered = 0
    for frame in frames:
        es = betweenstats(frame, "g", "y", bf_message=False).effect_size
        assert es.conf_level == 0.95
        covered += es.conf_low <= 0.0 <= es.conf_high
    rate = covered / REPLICATIONS
    low, high = binomial_band(0.95)
    print(f"Hedges' g: CI coverage {rate:.4f}, 99% band [{low:.4f}, {high:.4f}]")
    assert low <= rate <= high, f"Hedges' g CI coverage {rate} outside [{low}, {high}]"
