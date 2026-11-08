"""Time budgets from the Phase 1 contract (best of three runs, after a warm-up)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import numpy as np

from inferplot.stats import mean


def _best_of_three(fn: Callable[[], object]) -> float:
    fn()
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def test_mean_of_a_million_values_takes_at_most_0_2_s() -> None:
    x = np.random.Generator(np.random.PCG64(1)).normal(5.0, 3.0, 1_000_000)
    seconds = _best_of_three(lambda: mean(x))
    assert seconds <= 0.2, f"mean of 1e6 values took {seconds:.3f} s"
