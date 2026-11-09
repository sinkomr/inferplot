# NumPy's stubs leave overloads of sort, cumsum, unique, concatenate and the ufunc
# methods partially unknown, which strict mode reports on every call; the arrays
# passed are all typed.
# pyright: reportUnknownMemberType=false
"""Rank-based tests and effect sizes for independent groups.

Mann-Whitney U with the rank-biserial r for two groups; Kruskal-Wallis with rank
epsilon squared (bootstrap CI) and Dunn's pairwise z for any number of groups.

All of them rank the pooled values once, with mid-ranks for ties. With N values and
tied sets of sizes t, the total rank sum of squares about (N + 1) / 2 is
SST = (N^3 - N - sum(t^3 - t)) / 12, which every tie correction below is written in:
Kruskal-Wallis' divisor, Dunn's rank variance SST / (N - 1) and the Mann-Whitney
variance n1 n2 SST / (N (N - 1)). Twice a mid-rank is an integer, so rank sums,
SST and the between-group sum of squares are computed exactly (Python integers and
fractions) and each statistic is rounded once; R's ``kruskal.test`` subtracts
3 (N + 1) from a sum of squares, which loses digits when H is small, so where R and
inferplot differ there, inferplot's H is the accurate one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy import special

from inferplot.stats.descriptive import FloatArray, SampleLike, as_sample
from inferplot.stats.effect_sizes import EffectSizeCI
from inferplot.stats.noncentral import check_conf_level
from inferplot.stats.parametric import Alternative, check_alternative

__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BootstrapEffectSizeCI",
    "DunnPair",
    "KruskalWallis",
    "MannWhitney",
    "SeedLike",
    "check_bootstrap_seed",
    "dunn",
    "kruskal_wallis",
    "mann_whitney",
    "rank_biserial",
    "rank_epsilon_squared",
]

IntArray = NDArray[np.int64]
# A bootstrap seed: a non-negative Python or NumPy integer (never a bool).
SeedLike: TypeAlias = int | np.integer[Any]

BOOTSTRAP_RESAMPLES: Final = 2000
"""Resamples of every bootstrap in inferplot (SPEC.md section 5, item 2)."""

# Resamples ranked together in one block of the bootstrap: bounds the block's
# (resamples x values) work arrays to about this many elements.
_BLOCK_ELEMENTS: Final = 4_000_000


@dataclass(frozen=True, slots=True)
class MannWhitney:
    """Mann-Whitney U test, normal approximation (R: ``stats::wilcox.test``).

    ``statistic`` is R's W, the U of the first sample; ``z`` the continuity- and
    tie-corrected standard normal deviate whose tail gives ``p_value``.
    """

    statistic: float
    z: float
    p_value: float
    alternative: Alternative


@dataclass(frozen=True, slots=True)
class KruskalWallis:
    """Kruskal-Wallis rank sum test (R: ``stats::kruskal.test``)."""

    statistic: float
    df: float
    p_value: float


@dataclass(frozen=True, slots=True)
class DunnPair:
    """Dunn's test of groups ``i`` < ``j`` (indices in input order).

    ``statistic`` is |z| (unsigned, as ggstatsplot reports it), ``p_value`` the raw
    two-sided normal p, before any multiplicity adjustment.
    """

    i: int
    j: int
    statistic: float
    p_value: float


def check_bootstrap_seed(seed: SeedLike) -> int:
    """A bootstrap seed as a Python int: a non-negative ``int`` or NumPy integer,
    never a bool (the one check every bootstrap in ``inferplot.stats`` uses).

    Raises:
        ValueError: a bool, a negative number, or not an integer.
    """
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int | np.integer)  # pyright: ignore[reportUnnecessaryIsInstance]
        or seed < 0
    ):
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}")
    return int(seed)


@dataclass(frozen=True, slots=True)
class BootstrapEffectSizeCI:
    """An effect size with a percentile bootstrap CI and how it was drawn.

    ``undefined_resamples`` counts the resamples where the effect size is undefined
    (rank epsilon squared when every value of a resample is tied); the interval
    comes from the other ``resamples - undefined_resamples``.
    ``infinite_resamples`` counts the resamples where it is +/-inf (the AKP and
    explanatory effect sizes when a resample's winsorized variance is 0 and its
    numerator is not); they are kept, so a bound can be infinite.
    """

    estimate: float
    conf_low: float
    conf_high: float
    conf_level: float
    sidedness: Alternative
    seed: int
    resamples: int
    undefined_resamples: int = 0
    infinite_resamples: int = 0


@dataclass(frozen=True, slots=True)
class _Ranked:
    """Pooled mid-ranks of the groups, kept exact.

    ``rank_sums2[g]`` is twice the rank sum of group g (an integer), ``sizes[g]``
    its size, ``n`` the total, ``sst12`` 12 SST = N^3 - N - sum(t^3 - t).
    """

    rank_sums2: tuple[int, ...]
    sizes: tuple[int, ...]
    n: int
    sst12: int


def _samples(groups: Sequence[SampleLike], method: str) -> list[FloatArray]:
    if len(groups) < 2:
        raise ValueError(f"{method} needs at least two groups; got {len(groups)}")
    return [as_sample(g, name=f"group {i + 1}") for i, g in enumerate(groups)]


def _codes(samples: Sequence[FloatArray]) -> tuple[IntArray, IntArray]:
    """Each pooled value's index among the sorted distinct values, and how often
    each distinct value occurs."""
    _, inverse, counts = np.unique(
        np.concatenate(samples), return_inverse=True, return_counts=True
    )
    return inverse.astype(np.int64).ravel(), counts.astype(np.int64)


def _rank(samples: Sequence[FloatArray], method: str) -> _Ranked:
    """Mid-rank the pooled samples; ValueError when every value is tied."""
    codes, counts = _codes(samples)
    # Twice the mid-rank of each distinct value: 2 cumsum - t + 1, an integer.
    mid2 = 2 * np.cumsum(counts) - counts + 1
    ranks2 = mid2[codes]
    sizes = tuple(int(s.size) for s in samples)
    starts = np.cumsum((0, *sizes[:-1]))
    sums = np.add.reduceat(ranks2, starts) if ranks2.size else np.zeros(0, np.int64)
    n = int(codes.size)
    ties = sum(t * t * t - t for t in counts.tolist() if t > 1)
    sst12 = n * n * n - n - ties
    if sst12 == 0:
        raise ValueError(
            f"all {n} values are tied, so {method} is undefined (the ranks have no "
            "variance)"
        )
    return _Ranked(
        rank_sums2=tuple(int(v) for v in sums.tolist()),
        sizes=sizes,
        n=n,
        sst12=sst12,
    )


def _ssb(ranked: _Ranked) -> Fraction:
    """Between-group sum of squares of the ranks about (N + 1) / 2, exactly."""
    n = ranked.n
    total = Fraction(0)
    for s2, size in zip(ranked.rank_sums2, ranked.sizes, strict=True):
        # n_g (S_g / n_g - (N + 1) / 2)^2 = (2 S_g - n_g (N + 1))^2 / (4 n_g)
        d = s2 - size * (n + 1)
        total += Fraction(d * d, 4 * size)
    return total


def mann_whitney(
    a: SampleLike, b: SampleLike, *, alternative: Alternative = "two-sided"
) -> MannWhitney:
    """Mann-Whitney U test of a against b, normal approximation (Mann & Whitney
    1947, Annals of Mathematical Statistics 18: 50-60; Wilcoxon 1945).

    W = (sum of the pooled mid-ranks of a) - n_a (n_a + 1) / 2, the U of ``a``.
    With d = W - n_a n_b / 2 and the tie-corrected variance
    sigma^2 = n_a n_b / 12 (N + 1 - sum(t^3 - t) / (N (N - 1))):
    two-sided z = (d - sign(d) / 2) / sigma and p = 2 P(Z > |z|); "greater"
    z = (d - 1/2) / sigma, p = P(Z > z); "less" z = (d + 1/2) / sigma, p = P(Z < z).
    Always the normal approximation with continuity correction, never the exact
    distribution, also without ties and with small samples: the p of R's
    ``stats::wilcox.test(a, b, exact = FALSE, correct = TRUE, alternative = ...)``
    as statsExpressions 2.2.1 calls it for ggstatsplot (SPEC.md section 4).

    Raises:
        ValueError: a sample is empty or has a non-finite value, every value is
            tied (sigma = 0), or ``alternative`` is unknown.
    """
    side = check_alternative(alternative)
    x = as_sample(a, name="a")
    y = as_sample(b, name="b")
    ranked = _rank([x, y], "the Mann-Whitney test")
    n_a, n_b, n = x.size, y.size, ranked.n
    # 2 W = 2 R_a - n_a (n_a + 1); 2 d = 2 W - n_a n_b, both integers.
    w2 = ranked.rank_sums2[0] - n_a * (n_a + 1)
    d2 = w2 - n_a * n_b
    if side == "two-sided":
        shift = (d2 > 0) - (d2 < 0)
    else:
        shift = 1 if side == "greater" else -1
    # sigma^2 = n_a n_b SST / (N (N - 1)), SST = sst12 / 12.
    variance = Fraction(n_a * n_b * ranked.sst12, 12 * n * (n - 1))
    z = (d2 - shift) / 2 / math.sqrt(variance)
    if side == "two-sided":
        p = min(1.0, 2.0 * float(special.ndtr(-abs(z))))
    elif side == "greater":
        p = float(special.ndtr(-z))
    else:
        p = float(special.ndtr(z))
    return MannWhitney(statistic=w2 / 2, z=z, p_value=p, alternative=side)


def rank_biserial(
    a: SampleLike,
    b: SampleLike,
    *,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Rank-biserial correlation of a against b with a Fisher-z CI.

    r = 2 U / (n_a n_b) - 1 with U the Mann-Whitney U of ``a`` (Glass 1965,
    Educational and Psychological Measurement 25: 431-437; Kerby 2014,
    Comprehensive Psychology 3: 11.IT.3.1): the share of pairs in which a is
    larger minus the share in which b is, ties counting half. CI:
    tanh(atanh(r) -/+ q sqrt((n_a + n_b + 1) / (3 n_a n_b))), q the standard normal
    quantile at (1 + conf_level) / 2, ties ignored in the standard error. A
    one-sided ``alternative`` gives a one-sided interval at ``conf_level``:
    "greater" [lower, 1], "less" [-1, upper]. Matches R's
    ``effectsize::rank_biserial(ci = conf_level, alternative = ...)`` as
    statsExpressions 2.2.1 calls it (SPEC.md section 4). With |r| = 1 both bounds
    of the two-sided interval are r (atanh(r) is infinite).

    Raises:
        ValueError: a sample is empty or has a non-finite value, conf_level is not
            strictly between 0 and 1, or ``alternative`` is unknown.
    """
    side = check_alternative(alternative)
    level = check_conf_level(conf_level)
    x = as_sample(a, name="a")
    y = as_sample(b, name="b")
    n_a, n_b = x.size, y.size
    codes, counts = _codes([x, y])
    mid2 = 2 * np.cumsum(counts) - counts + 1
    w2 = int(mid2[codes[:n_a]].sum()) - n_a * (n_a + 1)
    # r = 2 U / (n_a n_b) - 1 = (2 U - n_a n_b) / (n_a n_b), rounded once.
    r = float(Fraction(w2 - n_a * n_b, n_a * n_b))
    se = math.sqrt((n_a + n_b + 1) / (3.0 * n_a * n_b))
    tail = (1.0 - level) / 2.0 if side == "two-sided" else 1.0 - level
    q = -float(special.ndtri(tail))
    with np.errstate(divide="ignore"):
        z = float(np.arctanh(r))
    low = -1.0 if side == "less" else math.tanh(z - q * se)
    high = 1.0 if side == "greater" else math.tanh(z + q * se)
    return EffectSizeCI(
        estimate=r, conf_low=low, conf_high=high, conf_level=level, sidedness=side
    )


def kruskal_wallis(groups: Sequence[SampleLike]) -> KruskalWallis:
    """Kruskal-Wallis rank sum test with the tie correction (Kruskal & Wallis
    1952, Journal of the American Statistical Association 47: 583-621).

    H = [12 / (N (N + 1)) sum n_g (Rbar_g - (N + 1) / 2)^2] / C with
    C = 1 - sum(t^3 - t) / (N^3 - N), which is (N - 1) SSB / SST of the mid-ranks;
    df = k - 1 and p = P(chi^2_df > H). Computed exactly and rounded once (module
    docstring). Matches R's ``stats::kruskal.test`` (SPEC.md section 4).

    Raises:
        ValueError: fewer than two groups, an empty group, a non-finite value, or
            every value tied (C = 0).
    """
    samples = _samples(groups, "Kruskal-Wallis")
    ranked = _rank(samples, "the Kruskal-Wallis test")
    h = float((ranked.n - 1) * 12 * _ssb(ranked) / ranked.sst12)
    df = float(len(samples) - 1)
    return KruskalWallis(statistic=h, df=df, p_value=float(special.chdtrc(df, h)))


def dunn(groups: Sequence[SampleLike]) -> tuple[DunnPair, ...]:
    """Dunn's all-pairs test after Kruskal-Wallis (Dunn 1964, Technometrics 6:
    241-252), raw p-values.

    For every pair i < j in input order, z = |Rbar_i - Rbar_j| / sqrt(S^2 (1/n_i +
    1/n_j)) with the mean pooled mid-ranks Rbar and the tie-corrected rank variance
    S^2 = N (N + 1) / 12 - sum(t^3 - t) / (12 (N - 1)) = SST / (N - 1); p = 2 P(Z > z).
    z is unsigned, as ggstatsplot 1.1.4 reports it. Adjust the raw p-values with
    ``p_adjust``. Matches R's ``PMCMRplus::kwAllPairsDunnTest`` as statsExpressions
    2.2.1 calls it (SPEC.md section 4).

    Raises:
        ValueError: fewer than two groups, an empty group, a non-finite value, or
            every value tied.
    """
    samples = _samples(groups, "Dunn's test")
    ranked = _rank(samples, "Dunn's test")
    n = ranked.n
    k = len(samples)
    out: list[DunnPair] = []
    for i in range(k):
        for j in range(i + 1, k):
            n_i, n_j = ranked.sizes[i], ranked.sizes[j]
            # 2 (Rbar_i - Rbar_j) and S^2 (1/n_i + 1/n_j) = SST (n_i + n_j) /
            # ((N - 1) n_i n_j), SST = sst12 / 12; both exact.
            diff2 = Fraction(ranked.rank_sums2[i], n_i) - Fraction(
                ranked.rank_sums2[j], n_j
            )
            var = Fraction(ranked.sst12 * (n_i + n_j), 12 * (n - 1) * n_i * n_j)
            # z^2 = diff^2 / var, rounded once.
            z = math.sqrt(float(diff2 * diff2 / (4 * var)))
            p = min(1.0, 2.0 * float(special.ndtr(-z)))
            out.append(DunnPair(i=i, j=j, statistic=z, p_value=p))
    return tuple(out)


def _percentile_lower(values: FloatArray, alpha: float) -> float:
    """The bootstrap alpha-quantile as R's ``boot::boot.ci(type = "perc")`` takes
    it: the order statistic (R + 1) alpha of the R sorted values, interpolated on
    the normal quantile scale between the neighbouring order statistics k and
    k + 1 when (R + 1) alpha is not an integer (Davison & Hinkley 1997, Bootstrap
    Methods and their Application, section 5.2 and eq. 5.8)."""
    t = np.sort(values)
    count = t.size
    rk = (count + 1) * alpha
    k = math.floor(rk)
    if k < 1 or k >= count:
        raise ValueError(
            f"a percentile interval at this conf_level needs order statistic "
            f"{rk:.4g} of {count} resamples; conf_level must leave at least one "
            "resample on either side"
        )
    if rk == k:
        return float(t[k - 1])
    z_k = float(special.ndtri(k / (count + 1)))
    z_k1 = float(special.ndtri((k + 1) / (count + 1)))
    z_a = float(special.ndtri(alpha))
    lo, hi = float(t[k - 1]), float(t[k])
    return lo + (z_a - z_k) / (z_k1 - z_k) * (hi - lo)


def _epsilon_block(
    drawn: IntArray, n_distinct: int, starts: IntArray, sizes: FloatArray
) -> FloatArray:
    """Rank epsilon squared, SSB / SST of the mid-ranks, of each resample.

    ``drawn`` (resamples x N) holds the distinct-value code of every value of each
    resample; the values of group g are at positions ``starts[g]`` onwards. NaN
    where a resample has every value tied.
    """
    count, n = drawn.shape
    flat = drawn + (np.arange(count, dtype=np.int64) * n_distinct)[:, None]
    ties = np.bincount(flat.ravel(), minlength=count * n_distinct)
    ties = ties.reshape(count, n_distinct)
    # Twice the mid-rank of each distinct value in each resample (integers).
    mid2 = 2 * np.cumsum(ties, axis=1) - ties + 1
    ranks2 = np.take_along_axis(mid2, drawn, axis=1)
    sums2 = np.add.reduceat(ranks2, starts, axis=1).astype(np.float64)
    d = sums2 - sizes * float(n + 1)
    ssb = np.sum(d * d / (4.0 * sizes), axis=1)
    t = ties.astype(np.float64)
    sst = (float(n) ** 3 - n - np.sum(t * t * t - t, axis=1)) / 12.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sst > 0.0, ssb / sst, np.nan)


def _canonical_groups(codes: IntArray, sizes: Sequence[int]) -> list[IntArray]:
    """Each group's distinct-value codes sorted, the groups in an order that
    depends only on their contents (size, then codes): the bootstrap then draws
    the same values whatever the row order within groups and the group order."""
    segments = np.split(codes, np.cumsum(sizes)[:-1].tolist())
    ordered = [np.sort(s) for s in segments]
    ordered.sort(key=lambda s: (s.size, s.tobytes()))
    return ordered


def rank_epsilon_squared(
    groups: Sequence[SampleLike], *, conf_level: float = 0.95, seed: SeedLike = 0
) -> BootstrapEffectSizeCI:
    """Rank epsilon squared with a one-sided percentile bootstrap CI [lower, 1].

    Estimate (Tomczak & Tomczak 2014, Trends in Sport Sciences 21: 19-25, eq. 9):
    E_R^2 = H / ((N^2 - 1) / (N + 1)) = H / (N - 1) with the tie-corrected
    Kruskal-Wallis H, which equals SSB / SST of the pooled mid-ranks, in [0, 1].

    CI: ``BOOTSTRAP_RESAMPLES`` (2,000) stratified resamples, each drawing n_g
    values with replacement within every group g (group sizes fixed, every value
    re-ranked); the lower bound is the bootstrap (1 - conf_level) quantile with
    ``boot.ci``'s percentile rule (``_percentile_lower``), the upper bound 1.
    A resample whose values are all tied (possible only when one value occurs in
    every group) has no epsilon squared; such resamples are left out and the
    quantile is taken over the others, which are counted in
    ``undefined_resamples`` (DECISIONS.md D54). That is R's rule as the oracle pins
    it (edge_ties_small, 0.65% of resamples tied): over 40 seeds the 20,000-resample
    lower bound is R's reference 0.008754 every time and the 2,000-resample bound
    averages 0.00739 over 400 seeds against R's 0.00730 (0.16 standard errors).
    Counting them as 0 instead gives 0.00204 and 0.00321 (6.7 standard errors).
    Matches R's ``effectsize::rank_epsilon_squared(ci = conf_level,
    alternative = "greater")`` as statsExpressions 2.2.1 calls it (with 100
    resamples there), within Monte Carlo error. The stratified scheme was found
    against the oracle's 20,000-resample references: over 50 seeds the mean
    lower bound agrees with R's 50 replicate runs within 1.3 standard errors in
    all 7 fixtures, while resampling whole rows (group sizes varying) puts the
    bound up to 220 replicate SDs too low (mtcars cyl x disp; DECISIONS.md).

    Randomness: ``numpy.random.Generator(numpy.random.PCG64(seed))``; resample b
    draws its indices in one call, ``integers(0, n_g)`` for each position of each
    group. The draws index each group's sorted values, and the groups are taken in
    an order fixed by their contents, so the interval depends only on the data and
    the seed: not on the row order within a group, the order of the groups, or any
    strictly increasing transform of the values (each is bit for bit the same).

    Raises:
        ValueError: fewer than two groups, an empty group, a non-finite value,
            every value tied, conf_level not strictly between 0 and 1 or too
            extreme for the resamples left (below 1/2001 or above 2000/2001 with
            none tied), or a negative seed.
    """
    level = check_conf_level(conf_level)
    rng_seed = check_bootstrap_seed(seed)
    samples = _samples(groups, "Rank epsilon squared")
    ranked = _rank(samples, "rank epsilon squared")
    estimate = float(12 * _ssb(ranked) / ranked.sst12)

    codes, counts = _codes(samples)
    canonical = _canonical_groups(codes, ranked.sizes)
    pooled = np.concatenate(canonical)
    n = ranked.n
    sizes = np.asarray([g.size for g in canonical], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    high = np.repeat(sizes, sizes)
    offset = np.repeat(starts, sizes)
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    block = max(1, min(BOOTSTRAP_RESAMPLES, _BLOCK_ELEMENTS // max(n, counts.size)))
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    sizes_f = sizes.astype(np.float64)
    for first in range(0, BOOTSTRAP_RESAMPLES, block):
        last = min(BOOTSTRAP_RESAMPLES, first + block)
        draws = np.empty((last - first, n), dtype=np.int64)
        for row in range(last - first):
            draws[row] = rng.integers(0, high, dtype=np.int64) + offset
        values[first:last] = _epsilon_block(pooled[draws], counts.size, starts, sizes_f)
    defined = values[~np.isnan(values)]
    undefined = BOOTSTRAP_RESAMPLES - defined.size
    try:
        low = _percentile_lower(defined, 1.0 - level)
    except ValueError:
        if not undefined:
            raise
        raise ValueError(
            f"{undefined} of {BOOTSTRAP_RESAMPLES} bootstrap resamples have every "
            f"value tied, which leaves {defined.size}, too few for a percentile "
            f"interval at conf_level={level!r} (one value occurs in every group, and "
            "the groups are small or dominated by it)"
        ) from None
    return BootstrapEffectSizeCI(
        estimate=estimate,
        conf_low=low,
        conf_high=1.0,
        conf_level=level,
        sidedness="greater",
        seed=rng_seed,
        resamples=BOOTSTRAP_RESAMPLES,
        undefined_resamples=undefined,
    )
