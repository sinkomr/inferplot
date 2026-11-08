"""StatsResult objects for the rendering tests, computed by ip.analyze.betweenstats.

One result per completed parametric ggbetweenstats fixture, from the fixture's own
dataset and arguments (``tests/oracle_args.translate``), plus results from small
synthetic frames (many levels, many points, hand-picked values).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from inferplot import analyze
from inferplot.result import StatsResult
from tests.oracle_args import translate
from tests.oracle_fixtures import OracleCase, cases_for, read_dataset


def parametric_cases() -> tuple[OracleCase, ...]:
    """Every parametric ggbetweenstats fixture ggstatsplot completed."""
    return tuple(c for c in cases_for("ggbetweenstats") if c.type == "parametric")


def fixture_result(case: OracleCase) -> StatsResult:
    """The result ip.analyze.betweenstats computes for ``case``, with its arguments."""
    return _fixture_result(case.name)


@functools.cache
def _fixture_result(name: str) -> StatsResult:
    (case,) = [c for c in parametric_cases() if c.name == name]
    call = translate(case.function, case.args)
    return analyze.betweenstats(read_dataset(case), *call.args, **call.kwargs)


def frame_result(
    levels: Sequence[str],
    groups: Sequence[ArrayLike],
    *,
    x: str = "group",
    y: str = "value",
    **kwargs: Any,
) -> StatsResult:
    """ip.analyze.betweenstats on a long frame of ``groups``, in ``levels`` order."""
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    frame = pd.DataFrame(
        {
            x: np.repeat(np.asarray(levels, dtype=object), [a.size for a in arrays]),
            y: np.concatenate(arrays),
        }
    )
    return analyze.betweenstats(frame, x, y, order=list(levels), **kwargs)


def many_levels_result(k: int = 12, *, n: int = 15, seed: int = 1) -> StatsResult:
    """``k`` groups drawn from the same normal distribution, with long-ish names."""
    rng = np.random.default_rng(seed)
    levels = [f"site {i + 1:02d}" for i in range(k)]
    groups = [rng.normal(50.0, 10.0, size=n) for _ in range(k)]
    return frame_result(levels, groups, x="site", y="yield")


def many_points_result(n_per_group: int = 2_200, *, seed: int = 2) -> StatsResult:
    """Three groups with ``3 * n_per_group`` values in all (6,600 by default)."""
    rng = np.random.default_rng(seed)
    groups = [rng.normal(mu, 1.0, size=n_per_group) for mu in (0.0, 0.3, 0.5)]
    return frame_result(["a", "b", "c"], groups, x="arm", y="score")
