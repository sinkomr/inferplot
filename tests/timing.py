"""Wall-clock limits that hold on a busy or slow machine.

A few tests guard against a method becoming slow (a quadratic loop where a sort
would do). A fixed number of seconds fails for the wrong reason on a loaded
machine or a slow CI runner: the release's test run from the sdist failed once at
0.538 s against a limit of 0.5 s while other test runs shared the machine. Such a
limit is therefore written ``allowed(0.5)``: the seconds an idle machine is
allowed, multiplied by how much slower than that this machine is right now,
measured on a small fixed workload at the moment of the check.
"""

from __future__ import annotations

import math
import time

import numpy as np

# The calibration workload's time on a machine that gets the stated limit; a
# machine k times slower gets k times the limit. Measured on the development
# laptop while several test runs shared it (load average 88): 0.019 to 0.022 s,
# so the limits there are about doubled.
_REFERENCE_SECONDS = 0.01
_VALUES = np.random.Generator(np.random.PCG64(0)).random(120_000)


def _calibration_seconds() -> float:
    """The best of three runs of a fixed workload, half interpreted Python and
    half NumPy, as the code under test is."""
    best = math.inf
    for _ in range(3):
        start = time.perf_counter()
        total = 0.0
        for i in range(60_000):
            total += i * 0.5
        np.sort(_VALUES)
        best = min(best, time.perf_counter() - start)
    return best


def allowed(seconds: float) -> float:
    """``seconds`` on a machine at the reference speed; proportionally more on a
    slower or busier one, never less."""
    return seconds * max(1.0, _calibration_seconds() / _REFERENCE_SECONDS)
