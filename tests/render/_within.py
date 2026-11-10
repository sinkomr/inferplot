"""withinstats results for the rendering tests, built without within statistics.

``ip.analyze.withinstats`` raises until its statistics land, so these results are
assembled with ``inferplot.analyses.withinstats._assemble`` from the data of a
subjects-by-levels matrix and parts borrowed from ``ip.analyze.betweenstats`` on the
same values (its test, effect size, Bayes factor, pairwise comparisons and means):
the numbers are not a within-subjects analysis, but every text, bracket and value
has the shape the renderer draws.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from inferplot import analyze
from inferplot._data import read_within
from inferplot.result import StatsResult

_module: Any = importlib.import_module("inferplot.analyses.withinstats")


def within_result(
    levels: Sequence[str],
    matrix: ArrayLike,
    *,
    x: str = "condition",
    y: str = "score",
    subject_id: str = "subject",
    pairwise_display: str = "significant",
    results_subtitle: bool = True,
    seed: int = 0,
) -> StatsResult:
    """A within result for ``matrix`` (subjects by levels, in ``levels`` order)."""
    values = np.asarray(matrix, dtype=np.float64)
    n, k = values.shape
    assert k == len(levels)
    subjects = [f"s{i + 1:04d}" for i in range(n)]
    frame = pd.DataFrame(
        {
            subject_id: np.repeat(subjects, k),
            x: np.tile(np.asarray(levels, dtype=object), n),
            y: values.reshape(-1),
        }
    )
    data = read_within(frame, x, y, subject_id, order=list(levels))
    between = analyze.betweenstats(
        frame,
        x,
        y,
        order=list(levels),
        pairwise_display=pairwise_display,
        results_subtitle=results_subtitle,
        seed=seed,
    )
    parts = _module._Parts(
        test=between.test,
        effect_size=between.effect_size,
        bayes=between.bayes,
        pairwise=between.pairwise,
        centrality="mean",
        centers=tuple(d.value for d in between.descriptives),
        warnings=(),
        citations=(),
    )
    args = _module._Args(
        x=x,
        subject_id=subject_id,
        type="parametric",
        conf_level=0.95,
        alternative="two-sided",
        p_adjust_method="holm",
        pairwise_alpha=0.05,
        pairwise_display=between.settings.pairwise_display,
        prior_scale=0.707,
        bf_message=None,
        results_subtitle=results_subtitle,
        tr=0.2,
        seed=seed,
    )
    return _module._assemble(data, args, parts, digits=2)


def normal_matrix(
    n: int,
    means: Sequence[float],
    *,
    sd: float = 1.0,
    subject_sd: float = 1.0,
    seed: int = 3,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """``n`` subjects at ``len(means)`` levels: a subject effect plus noise."""
    rng = np.random.default_rng(seed)
    subject = rng.normal(0.0, subject_sd, size=(n, 1))
    return (
        np.asarray(means, dtype=np.float64)[None, :]
        + subject
        + rng.normal(0.0, sd, size=(n, len(means)))
    )


@functools.cache
def two_conditions(n: int = 20, *, seed: int = 3) -> StatsResult:
    """Two conditions, ``n`` subjects, a clear rise from "before" to "after"."""
    return within_result(["before", "after"], normal_matrix(n, [10.0, 11.5], seed=seed))


@functools.cache
def three_conditions(pairwise_display: str = "all") -> StatsResult:
    """Three conditions of 24 subjects, rising then falling, every pair shown."""
    return within_result(
        ["low", "medium", "high"],
        normal_matrix(24, [4.0, 6.0, 5.0], sd=0.8, seed=5),
        x="dose",
        y="response",
        pairwise_display=pairwise_display,
    )


@functools.cache
def five_conditions(pairwise_display: str = "significant") -> StatsResult:
    """Five weekly conditions of 30 subjects."""
    return within_result(
        ["week 1", "week 2", "week 3", "week 4", "week 5"],
        normal_matrix(30, [5.0, 5.4, 6.4, 6.1, 7.2], sd=0.9, seed=7),
        x="week",
        y="score",
        pairwise_display=pairwise_display,
    )


@functools.cache
def many_subjects(n: int, k: int = 2, *, seed: int = 11) -> StatsResult:
    """``n`` subjects at ``k`` conditions (``n * k`` values)."""
    means = [10.0 + 0.4 * j for j in range(k)]
    levels = [f"t{j + 1}" for j in range(k)]
    return within_result(levels, normal_matrix(n, means, seed=seed))
