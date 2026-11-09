"""Pairwise comparisons after a one-way design: Games-Howell."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from inferplot.stats.descriptive import SampleLike, as_sample, common_scale, mean
from inferplot.stats.padjust import PAdjustMethod, p_adjust
from inferplot.stats.parametric import checked_scaled_variances, satterthwaite_df
from inferplot.stats.srange import srange_sf

__all__ = ["PairResult", "games_howell"]


@dataclass(frozen=True, slots=True)
class PairResult:
    """One pairwise comparison of groups ``i`` < ``j`` (indices in input order)."""

    i: int
    j: int
    statistic: float
    df: float | None
    p_value: float
    p_adjusted: float


def games_howell(
    groups: Sequence[SampleLike], *, p_adjust_method: PAdjustMethod = "holm"
) -> tuple[PairResult, ...]:
    """Games-Howell pairwise comparisons, then ``p_adjust`` over all pairs.

    Games & Howell (1976, Journal of Educational Statistics 1: 113-125): for every
    pair i < j in input order, q = (mean_j - mean_i) / sqrt((v_i/n_i + v_j/n_j) / 2),
    signed as mean(group j) - mean(group i) as ggstatsplot 1.1.4 reports it; df is
    the pair's Welch-Satterthwaite df; the raw p is the studentized range upper tail
    P(Q_{k,df} > |q|) with k the number of groups (``srange_sf``, accurate in the
    far tail, DECISIONS.md D31). The k(k-1)/2 raw p-values are then adjusted together
    with ``p_adjust_method``. Computed on the groups times their ``common_scale``
    (q and df do not depend on the scale of the data, bit for bit under powers of
    two), with the df from the variance shares (``satterthwaite_df``). R reference:
    ``PMCMRplus::gamesHowellTest`` followed by
    ``stats::p.adjust`` (Holm by default), as statsExpressions 2.2.1 calls them.

    Limits: every group's scaled variance term v_i / n_i must be at least
    ``TINY`` = 2^-1022 (``welch_anova``'s condition; a standard deviation above
    roughly 1e-154 times the largest |value| of all groups), so each term is
    accurate to about 2^-53 relative and |q| < 2^512.5 (about 1.9e154).

    Raises:
        ValueError: fewer than two groups, a group with fewer than two values or a
            non-finite value, a constant group or one whose variance term is below
            TINY (it varies by less than double precision can represent next to
            the largest value of all groups), or an unknown method.
    """
    if len(groups) < 2:
        raise ValueError(f"Games-Howell needs at least two groups; got {len(groups)}")
    p_adjust([], p_adjust_method)  # an unknown method fails before any work
    raw = [
        as_sample(g, name=f"group {i + 1}", min_size=2) for i, g in enumerate(groups)
    ]
    scale = common_scale(*raw)
    samples = [s * scale for s in raw]
    k = len(samples)
    n = [s.size for s in samples]
    means = [mean(s) for s in samples]
    variances = checked_scaled_variances(raw, samples, "Games-Howell")
    weights = [v / size for v, size in zip(variances, n, strict=True)]
    rows: list[tuple[int, int, float, float, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            w = weights[i] + weights[j]
            q = (means[j] - means[i]) / math.sqrt(0.5 * w)
            df = satterthwaite_df(weights[i], n[i], weights[j], n[j])
            rows.append((i, j, q, df, srange_sf(abs(q), k, df)))
    adjusted = p_adjust([row[4] for row in rows], p_adjust_method)
    return tuple(
        PairResult(i=i, j=j, statistic=q, df=df, p_value=p, p_adjusted=adj)
        for (i, j, q, df, p), adj in zip(rows, adjusted, strict=True)
    )
