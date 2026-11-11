# NumPy's stubs leave overloads of sort, unique and the ufunc methods partially
# unknown, which strict mode reports on every call; the arrays passed are all typed.
# pyright: reportUnknownMemberType=false
"""Rank-based tests and effect sizes for repeated measures (``withinstats``, type
nonparametric).

Two conditions: Wilcoxon's signed-rank test of the paired differences with the
matched-pairs rank-biserial r. Three or more: Friedman's test with Kendall's W
(percentile bootstrap CI over subjects) and Durbin-Conover's pairwise t.

Every rank here is a mid-rank, and twice a mid-rank is an integer, so rank sums,
tie corrections and sums of squares of ranks are computed exactly (Python integers
and fractions) and each statistic is rounded once.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import special

from inferplot.stats.descriptive import FloatArray, SampleLike, as_sample
from inferplot.stats.effect_sizes import EffectSizeCI
from inferplot.stats.noncentral import check_conf_level
from inferplot.stats.nonparametric import (
    BOOTSTRAP_RESAMPLES,
    BootstrapEffectSizeCI,
    SeedLike,
    _percentile_lower,  # pyright: ignore[reportPrivateUsage]
    check_bootstrap_seed,
)
from inferplot.stats.padjust import PAdjustMethod, p_adjust
from inferplot.stats.parametric import Alternative, check_alternative
from inferplot.stats.written import as_written

__all__ = [
    "DurbinConoverPair",
    "Friedman",
    "SignedRank",
    "durbin_conover",
    "friedman",
    "kendalls_w",
    "paired_rank_biserial",
    "signed_rank",
]

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SignedRank:
    """Wilcoxon's signed-rank test of a - b (R: ``stats::wilcox.test(a, b,
    paired = TRUE, exact = FALSE, correct = TRUE)``).

    ``statistic`` is R's V, the sum of the ranks of the positive differences;
    ``z`` the tie- and continuity-corrected standard normal deviate whose tail
    gives ``p_value``; ``n`` counts every pair, ``n_nonzero`` the pairs whose
    difference is not 0 (the ones ranked).
    """

    statistic: float
    z: float
    p_value: float
    n: int
    n_nonzero: int
    alternative: Alternative


@dataclass(frozen=True, slots=True)
class Friedman:
    """Friedman's rank sum test of k conditions measured on n subjects (R:
    ``stats::friedman.test``)."""

    statistic: float
    df: float
    p_value: float
    n: int
    k: int


@dataclass(frozen=True, slots=True)
class DurbinConoverPair:
    """Durbin-Conover's test of conditions ``i`` < ``j`` (input order).

    ``statistic`` is |t| (unsigned, as ggstatsplot reports it) on ``df`` degrees
    of freedom, ``p_value`` the raw two-sided p and ``p_adjusted`` the p after the
    multiplicity adjustment over all pairs. When every subject ranks the
    conditions alike the error variance is 0: |t| is inf and both p are 0.
    """

    i: int
    j: int
    statistic: float
    df: float
    p_value: float
    p_adjusted: float


# --------------------------------------------------------------------------- #
# paired differences and their signed ranks
# --------------------------------------------------------------------------- #


def _paired_samples(a: SampleLike, b: SampleLike) -> tuple[FloatArray, FloatArray]:
    # As written (D126): a pair of 0.1 + 0.2 and 0.3 is a zero difference, as a
    # value equal to the test value as written is in the one-sample test (D105).
    x = as_written(as_sample(a, name="a"))
    y = as_written(as_sample(b, name="b"))
    if x.size != y.size:
        raise ValueError(
            f"paired samples need the same number of values; got {x.size} and {y.size}"
        )
    return x, y


def _differences(x: FloatArray, y: FloatArray) -> FloatArray:
    """The rounded differences x - y, as R's ``wilcox.test`` forms them, so that
    two differences tie exactly when they tie in R. Where a difference overflows
    (values beyond 8.9e307 of opposite sign) both samples are halved first, which
    is exact for normal doubles and keeps every tie and order."""
    with np.errstate(over="ignore", invalid="ignore"):
        d = x - y
    if bool(np.isfinite(d).all()):
        return d
    return x * 0.5 - y * 0.5


@dataclass(frozen=True, slots=True)
class _SignedRanks:
    """Twice the positive and the negative rank sums of the nonzero differences
    (integers), their count and the tie sum sum(t^3 - t) of their |d|."""

    plus2: int
    minus2: int
    n_nonzero: int
    ties: int


def _signed_ranks(x: FloatArray, y: FloatArray) -> _SignedRanks:
    d = _differences(x, y)
    nonzero = d[x != y]
    _, codes, counts = np.unique(
        np.abs(nonzero), return_inverse=True, return_counts=True
    )
    # Twice the mid-rank of each distinct |d|: 2 cumsum - t + 1, an integer.
    mid2 = 2 * np.cumsum(counts) - counts + 1
    ranks2 = mid2[codes.ravel()]
    plus2 = int(ranks2[nonzero > 0].sum())
    minus2 = int(ranks2[nonzero < 0].sum())
    ties = sum(t * t * t - t for t in counts.tolist() if t > 1)
    return _SignedRanks(
        plus2=plus2, minus2=minus2, n_nonzero=int(nonzero.size), ties=ties
    )


def _all_zero_error(n: int) -> ValueError:
    return ValueError(
        f"every one of the {n} paired differences is 0, so the signed-rank test has "
        "nothing to rank (its statistic and the rank-biserial r are 0/0)"
    )


def signed_rank(
    a: SampleLike, b: SampleLike, *, alternative: Alternative = "two-sided"
) -> SignedRank:
    """Wilcoxon's (1945, Biometrics Bulletin 1: 80-83) signed-rank test of the
    paired differences d = a - b, normal approximation.

    Zero differences are dropped; the n' others are ranked by |d| with mid-ranks
    for ties, and V is the sum of the ranks of the positive ones. With
    m = V - n'(n' + 1) / 4 and the tie-corrected variance
    sigma^2 = n'(n' + 1)(2n' + 1) / 24 - sum(t^3 - t) / 48 (Hollander, Wolfe &
    Chicken 2014, Nonparametric Statistical Methods, 3rd ed., section 3.1):
    two-sided z = (m - sign(m) / 2) / sigma and p = 2 P(Z > |z|); "greater"
    z = (m - 1/2) / sigma, p = P(Z > z); "less" z = (m + 1/2) / sigma,
    p = P(Z < z). Always the normal approximation with the continuity correction,
    never the exact distribution, also without ties and with few pairs: R's
    ``stats::wilcox.test(a, b, paired = TRUE, exact = FALSE, correct = TRUE,
    alternative = ...)`` as statsExpressions 2.2.1 calls it for ggwithinstats
    (probe-signrank: every combination of exact, correct and alternative on 12
    sets of pairs). sigma > 0 whenever a difference is nonzero: the ties remove at
    most n'^3 - n', which leaves 48 sigma^2 >= 3 n'(n' + 1)^2.

    The differences are the rounded a - b, as R forms them (``_differences``), so
    ties among |d| are R's; z is computed from exact integers and rounded once.

    Raises:
        ValueError: a sample is empty or has a non-finite value, the samples
            differ in length, every difference is 0 (R gives V = 0 and p = NaN),
            or ``alternative`` is unknown.
    """
    side = check_alternative(alternative)
    x, y = _paired_samples(a, b)
    ranks = _signed_ranks(x, y)
    n1 = ranks.n_nonzero
    if n1 == 0:
        raise _all_zero_error(x.size)
    # 4 m = 2 (2V) - n'(n' + 1), an integer; the continuity shift 1/2 is 2 in it.
    m4 = 2 * ranks.plus2 - n1 * (n1 + 1)
    if side == "two-sided":
        shift = 2 * ((m4 > 0) - (m4 < 0))
    else:
        shift = 2 if side == "greater" else -2
    num4 = m4 - shift
    # 48 sigma^2 = 2 n'(n' + 1)(2n' + 1) - sum(t^3 - t) > 0; z^2 = 3 num4^2 / that.
    var48 = 2 * n1 * (n1 + 1) * (2 * n1 + 1) - ranks.ties
    z = math.copysign(math.sqrt(float(Fraction(3 * num4 * num4, var48))), num4)
    if side == "two-sided":
        p = min(1.0, 2.0 * float(special.ndtr(-abs(z))))
    elif side == "greater":
        p = float(special.ndtr(-z))
    else:
        p = float(special.ndtr(z))
    return SignedRank(
        statistic=ranks.plus2 / 2,
        z=z,
        p_value=p,
        n=int(x.size),
        n_nonzero=n1,
        alternative=side,
    )


def paired_rank_biserial(
    a: SampleLike,
    b: SampleLike,
    *,
    conf_level: float = 0.95,
    alternative: Alternative = "two-sided",
) -> EffectSizeCI:
    """Matched-pairs rank-biserial correlation of a - b with a Fisher-z CI.

    r = (W+ - W-) / (W+ + W-), W+ and W- the rank sums of the positive and the
    negative differences among the nonzero ones (King & Minium 2003, Statistical
    Reasoning in Psychology and Education; Kerby 2014, Comprehensive Psychology 3:
    11.IT.3.1): the share of the rank total on the positive side minus the share on
    the negative side, in [-1, 1], positive when a tends to exceed b.

    CI: tanh(atanh(r) -/+ q SE) with SE = sqrt(n (n + 1)(2n + 1) / 6) /
    (n (n + 1) / 2), the null standard deviation of r, where n counts every pair,
    zero differences included (probe-signrank: counting only the nonzero ones
    misses edge_within2_ties by 0.037), and q the standard normal quantile at
    (1 + conf_level) / 2. A one-sided ``alternative`` gives a one-sided interval at
    ``conf_level``: "greater" [lower, 1], "less" [-1, upper]. With |r| = 1,
    atanh(r) is infinite and the bound on r's side is r itself (sleep: [-1, -1]).
    Matches R's ``effectsize::rank_biserial(a, b, paired = TRUE, ci = conf_level,
    alternative = ...)`` as statsExpressions 2.2.1 calls it for ggwithinstats.

    Raises:
        ValueError: as ``signed_rank``, or conf_level is not strictly between 0
            and 1.
    """
    side = check_alternative(alternative)
    level = check_conf_level(conf_level)
    x, y = _paired_samples(a, b)
    ranks = _signed_ranks(x, y)
    if ranks.n_nonzero == 0:
        raise _all_zero_error(x.size)
    r = float(Fraction(ranks.plus2 - ranks.minus2, ranks.plus2 + ranks.minus2))
    n = x.size
    se = math.sqrt(n * (n + 1) * (2 * n + 1) / 6.0) / (n * (n + 1) / 2.0)
    tail = (1.0 - level) / 2.0 if side == "two-sided" else 1.0 - level
    q = -float(special.ndtri(tail))
    with np.errstate(divide="ignore"):
        z = float(np.arctanh(r))
    low = -1.0 if side == "less" else math.tanh(z - q * se)
    high = 1.0 if side == "greater" else math.tanh(z + q * se)
    return EffectSizeCI(
        estimate=r, conf_low=low, conf_high=high, conf_level=level, sidedness=side
    )


# --------------------------------------------------------------------------- #
# ranks within subjects (three or more conditions)
# --------------------------------------------------------------------------- #

# Subjects ranked together in one block: bounds the (subjects x k x k) comparison
# arrays to about this many elements.
_RANK_BLOCK_ELEMENTS: Final = 4_000_000


@dataclass(frozen=True, slots=True)
class _Blocks:
    """The subjects-by-conditions table ranked within each subject.

    ``ranks2[i, j]`` is twice the mid-rank of condition j within subject i (an
    integer from 2 to 2k), ``ties[i]`` subject i's sum(t^3 - t) over its tied
    sets; ``rank_sums2`` the column sums of ``ranks2`` (twice the rank sums R_j),
    and ``spread`` = n k (k^2 - 1) - sum(t^3 - t), 12 times the total
    within-subject sum of squares of the ranks (0 exactly when every subject has
    the same value in every condition).
    """

    ranks2: IntArray
    ties: IntArray
    n: int
    k: int
    rank_sums2: tuple[int, ...]
    spread: int

    @property
    def ss2(self) -> int:
        """sum_j (2 R_j - n (k + 1))^2: four times the sum of squares of the rank
        sums about their null mean."""
        centre = self.n * (self.k + 1)
        return sum((r - centre) ** 2 for r in self.rank_sums2)


def _table(conditions: Sequence[SampleLike], method: str) -> FloatArray:
    """The conditions as the columns of an n x k matrix (complete subjects)."""
    if len(conditions) < 2:
        raise ValueError(
            f"{method} needs at least two conditions; got {len(conditions)}"
        )
    # Ranked within each subject as written (D126).
    columns = [
        as_written(as_sample(c, name=f"condition {i + 1}", min_size=2))
        for i, c in enumerate(conditions)
    ]
    sizes = {c.size for c in columns}
    if len(sizes) != 1:
        raise ValueError(
            "every condition needs one value per subject (the same number of values); "
            f"got {[c.size for c in columns]}"
        )
    return np.column_stack(columns)


def _rank_rows(values: FloatArray) -> tuple[IntArray, IntArray]:
    """Twice the within-row mid-ranks, 2 (#below) + #equal + 1, and each row's
    sum(t^3 - t), which is the sum over its entries of #equal^2 - 1."""
    n, k = values.shape
    ranks2 = np.empty((n, k), dtype=np.int64)
    ties = np.empty(n, dtype=np.int64)
    step = max(1, _RANK_BLOCK_ELEMENTS // (k * k))
    for first in range(0, n, step):
        v = values[first : first + step]
        # Entry [i, a, b] compares value b of row i with value a.
        below = (v[:, np.newaxis, :] < v[:, :, np.newaxis]).sum(axis=2)
        equal = (v[:, np.newaxis, :] == v[:, :, np.newaxis]).sum(axis=2)
        ranks2[first : first + step] = 2 * below + equal + 1
        ties[first : first + step] = (equal * equal - 1).sum(axis=1)
    return ranks2, ties


def _blocks(conditions: Sequence[SampleLike], method: str) -> _Blocks:
    """Rank the table within subjects; ValueError when every subject has the same
    value in every condition (no within-subject variation to rank)."""
    ranks2, ties = _rank_rows(_table(conditions, method))
    n, k = ranks2.shape
    spread = n * k * (k * k - 1) - int(ties.sum())
    if spread == 0:
        raise ValueError(
            f"every one of the {n} subjects has the same value in all {k} "
            f"conditions, so {method} has nothing to rank (its statistic is 0/0)"
        )
    return _Blocks(
        ranks2=ranks2,
        ties=ties,
        n=n,
        k=k,
        rank_sums2=tuple(int(v) for v in ranks2.sum(axis=0).tolist()),
        spread=spread,
    )


def friedman(conditions: Sequence[SampleLike]) -> Friedman:
    """Friedman's (1937, Journal of the American Statistical Association 32:
    675-701) rank sum test for k conditions measured on the same n subjects.

    The values are ranked within each subject (mid-ranks for ties) and R_j is the
    rank sum of condition j. With the tie correction (Conover 1999, Practical
    Nonparametric Statistics, 3rd ed., section 5.8; Hollander, Wolfe & Chicken
    2014, section 7.1): chi^2 = 12 sum_j (R_j - n (k + 1) / 2)^2 /
    (n k (k + 1) - sum(t^3 - t) / (k - 1)), the sum over every tied set of every
    subject, on k - 1 df; p = P(chi^2_(k-1) > chi^2), the upper tail directly.
    A subject whose values are all tied adds nothing to either sum. Computed from
    exact integers (twice the mid-ranks) and rounded once. Matches R's
    ``stats::friedman.test`` as statsExpressions 2.2.1 calls it for ggwithinstats
    (probe-friedman: 11 designs, among them tied rows and ties within subjects).

    Raises:
        ValueError: fewer than two conditions or two subjects, conditions of
            unequal length or with a non-finite value, or every subject has the
            same value in every condition (R: NaN).
    """
    blocks = _blocks(conditions, "Friedman's test")
    k = blocks.k
    # chi^2 = 3 (k - 1) sum (2 R_j - n (k + 1))^2 / (n k (k^2 - 1) - sum(t^3 - t)).
    statistic = float(Fraction(3 * (k - 1) * blocks.ss2, blocks.spread))
    df = float(k - 1)
    return Friedman(
        statistic=statistic,
        df=df,
        p_value=float(special.chdtrc(df, statistic)),
        n=blocks.n,
        k=k,
    )


def durbin_conover(
    conditions: Sequence[SampleLike], *, p_adjust_method: PAdjustMethod = "holm"
) -> tuple[DurbinConoverPair, ...]:
    """Durbin-Conover all-pairs comparisons after Friedman's test, then
    ``p_adjust`` over all k (k - 1) / 2 raw p-values.

    Conover (1999, Practical Nonparametric Statistics, 3rd ed., section 5.8,
    complete block designs; Durbin 1951, British Journal of Statistical
    Psychology 4: 85-90): with the within-subject mid-ranks r_ij, their sum of
    squares A = sum r_ij^2 and the condition rank sums R_j, conditions i and j
    differ by t = |R_i - R_j| / sqrt(2 (n A - sum_j R_j^2) / ((n - 1)(k - 1))) on
    (n - 1)(k - 1) df, two-sided p = 2 P(T > t). This is Durbin's form for a
    balanced incomplete block design, (A - C)(1 - T / (n (k - 1))) with Durbin's
    T, when every condition is in every block, as R's
    ``PMCMRplus::durbinAllPairsTest`` computes it for ggwithinstats (statsExpressions
    2.2.1; probe-friedman). t is unsigned, as ggstatsplot reports it, and computed
    from exact integers, rounded once.

    When every subject ranks the conditions alike (Kendall's W = 1),
    n A = sum_j R_j^2: the error variance is 0, so t = inf and p = 0 for every pair
    of conditions with different rank sums, as R reports them (edge_within_small).

    Raises:
        ValueError: as ``friedman``; or every subject ranks the conditions alike
            and two conditions share their rank in every subject (t = 0/0); or an
            unknown ``p_adjust_method``.
    """
    p_adjust([], p_adjust_method)  # an unknown method fails before any work
    blocks = _blocks(conditions, "the Durbin-Conover test")
    n, k = blocks.n, blocks.k
    sums = blocks.rank_sums2
    a4 = int(np.square(blocks.ranks2).sum())
    # 4 (n A - sum R_j^2), in twice-ranks.
    error4 = n * a4 - sum(r * r for r in sums)
    df_int = (n - 1) * (k - 1)
    df = float(df_int)
    rows: list[tuple[int, int, float, float]] = []
    for i in range(k):
        for j in range(i + 1, k):
            diff2 = sums[i] - sums[j]
            if error4 == 0:
                if diff2 == 0:
                    raise ValueError(
                        "every subject ranks the conditions alike and conditions "
                        f"{i + 1} and {j + 1} share their rank in every subject, so "
                        "the Durbin-Conover t of that pair is 0/0"
                    )
                rows.append((i, j, math.inf, 0.0))
                continue
            # t^2 = (2 R_i - 2 R_j)^2 (n - 1)(k - 1) / (2 * 4 (n A - sum R^2)).
            t = math.sqrt(float(Fraction(diff2 * diff2 * df_int, 2 * error4)))
            rows.append((i, j, t, min(1.0, 2.0 * float(special.stdtr(df, -t)))))
    adjusted = p_adjust([row[3] for row in rows], p_adjust_method)
    return tuple(
        DurbinConoverPair(i=i, j=j, statistic=t, df=df, p_value=p, p_adjusted=adj)
        for (i, j, t, p), adj in zip(rows, adjusted, strict=True)
    )


# --------------------------------------------------------------------------- #
# Kendall's W
# --------------------------------------------------------------------------- #

# Resamples drawn together in one block of the bootstrap: bounds the block's
# (resamples x subjects) count array to about this many elements.
_BOOT_BLOCK_ELEMENTS: Final = 4_000_000
# Arrangements of conditions that tie on their invariants tried for the canonical
# table (``_canonical_ranks``); beyond this many the input order breaks the tie.
_MAX_ARRANGEMENTS: Final = 720


def _canonical_ranks(ranks2: IntArray) -> IntArray:
    """The within-subject twice-ranks with the conditions and the subjects in an
    order that depends only on the table's contents.

    Conditions are ordered by their rank sum, then by their sorted ranks; the
    subjects (rows) are then sorted lexicographically. Conditions that tie on both
    keys are tried in every arrangement (up to ``_MAX_ARRANGEMENTS`` in all) and
    the one whose sorted table is smallest is kept, so the result is the same for
    any order of the subjects and of the conditions; beyond that many
    arrangements (seven or more conditions with identical rank sums and rank
    multisets) the input order of the tied conditions is kept."""
    n, k = ranks2.shape
    keys = [
        (int(ranks2[:, j].sum()), tuple(sorted(ranks2[:, j].tolist())))
        for j in range(k)
    ]
    order = sorted(range(k), key=lambda j: keys[j])
    classes: list[list[int]] = []
    for j in order:
        if classes and keys[classes[-1][0]] == keys[j]:
            classes[-1].append(j)
        else:
            classes.append([j])
    count = math.prod(math.factorial(len(c)) for c in classes)
    if count > _MAX_ARRANGEMENTS:
        arrangements = [order]
    else:
        arrangements = [
            [j for part in parts for j in part]
            for parts in itertools.product(
                *(itertools.permutations(c) for c in classes)
            )
        ]
    best: IntArray | None = None
    best_key = b""
    for arrangement in arrangements:
        table = ranks2[:, arrangement]
        table = table[np.lexsort(table.T[::-1])]
        key = table.tobytes()
        if best is None or key < best_key:
            best, best_key = table, key
    assert best is not None
    return np.ascontiguousarray(best).reshape(n, k)


def _w_resamples(ranks2: IntArray, seed: int, resamples: int) -> tuple[FloatArray, int]:
    """Kendall's W of ``resamples`` bootstrap resamples of the subjects (rows of
    ``ranks2``), NaN where every drawn subject is fully tied, and that count.

    Resample b draws n row indices in one call, ``integers(0, n, size=n)``; its
    rank sums are counts @ ranks2 and its tie sum counts @ ties, all exact
    integers in float64, so W is rounded once, as the estimate is."""
    n, k = ranks2.shape
    ties = np.zeros(n, dtype=np.int64)
    for j in range(k):
        equal = (ranks2 == ranks2[:, j : j + 1]).sum(axis=1)
        ties += equal * equal - 1
    r2 = ranks2.astype(np.float64)
    t = ties.astype(np.float64)
    centre = float(n * (k + 1))
    full = float(n * k * (k * k - 1))
    rng = np.random.Generator(np.random.PCG64(seed))
    block = max(1, min(resamples, _BOOT_BLOCK_ELEMENTS // n))
    values = np.empty(resamples, dtype=np.float64)
    for first in range(0, resamples, block):
        last = min(resamples, first + block)
        counts = np.empty((last - first, n), dtype=np.float64)
        for row in range(last - first):
            counts[row] = np.bincount(rng.integers(0, n, size=n), minlength=n)
        deviations = counts @ r2 - centre
        ss2 = (deviations * deviations).sum(axis=1)
        spread = full - counts @ t
        with np.errstate(invalid="ignore", divide="ignore"):
            values[first:last] = np.where(
                spread > 0.0, 3.0 * ss2 / (n * spread), np.nan
            )
    return values, int(np.isnan(values).sum())


def kendalls_w(
    conditions: Sequence[SampleLike], *, conf_level: float = 0.95, seed: SeedLike = 0
) -> BootstrapEffectSizeCI:
    """Kendall's W, the coefficient of concordance of the subjects' rankings of
    the conditions, with a one-sided percentile bootstrap CI [lower, 1].

    Estimate (Kendall & Babington Smith 1939, Annals of Mathematical Statistics
    10: 275-287; with ties, Kendall 1948, Rank Correlation Methods, ch. 6): W =
    12 sum_j (R_j - n (k + 1) / 2)^2 / (n^2 (k^3 - k) - n sum(t^3 - t)), which is
    the tie-corrected Friedman chi^2 / (n (k - 1)), in [0, 1]; 1 when every
    subject ranks the conditions alike. Matches R's ``effectsize::kendalls_w``
    as statsExpressions 2.2.1 calls it for ggwithinstats.

    CI: ``BOOTSTRAP_RESAMPLES`` (2,000) resamples of the subjects, each drawing n
    whole subjects (rows of the subjects-by-conditions table) with replacement
    and recomputing W from their within-subject ranks; the lower bound is the
    bootstrap (1 - conf_level) quantile with ``boot.ci``'s percentile rule
    (``_percentile_lower``), the upper bound 1: effectsize's interval
    (``alternative = "greater"``, 100 resamples there), matched within Monte
    Carlo error against the oracle's references: over 40 seeds the mean
    2,000-resample lower bound agrees with the mean of R's 50 replicate runs of
    2,000 within 1.6 standard errors on all 7 fixtures where the bound varies
    (bugs_long at 0.95 and 0.90, edge_within_k5, _large_n, _missing,
    _nonspherical, _ties), and equals R's on the 4 where it does not; the lower
    bound of a two-sided interval misses by 17 to 39 Monte Carlo SDs. A resample whose
    drawn subjects all have the same value in every condition has no W; such
    resamples are left out and the quantile is taken over the others, counted in
    ``undefined_resamples`` (DECISIONS.md D54). With W = 1 every resample has
    W = 1 and the interval is [1, 1] (R's bootstrap fails there).

    Randomness: ``numpy.random.Generator(numpy.random.PCG64(seed))``; resample b
    draws its n indices in one call. The draws index the rows of the ranked table
    in a canonical order (``_canonical_ranks``), so the interval depends only on
    the data and the seed: not on the order of the subjects, the order of the
    conditions or any strictly increasing transform of the values.

    Raises:
        ValueError: as ``friedman``; conf_level not strictly between 0 and 1 or
            too extreme for the resamples left (below 1/2001 or above 2000/2001
            with none undefined); too many undefined resamples; or a negative
            seed.
    """
    level = check_conf_level(conf_level)
    rng_seed = check_bootstrap_seed(seed)
    blocks = _blocks(conditions, "Kendall's W")
    estimate = float(Fraction(3 * blocks.ss2, blocks.n * blocks.spread))
    values, undefined = _w_resamples(
        _canonical_ranks(blocks.ranks2), rng_seed, BOOTSTRAP_RESAMPLES
    )
    defined = values[~np.isnan(values)]
    try:
        low = _percentile_lower(defined, 1.0 - level)
    except ValueError:
        if not undefined:
            raise
        raise ValueError(
            f"{undefined} of {BOOTSTRAP_RESAMPLES} bootstrap resamples draw only "
            f"subjects with the same value in every condition, which leaves "
            f"{defined.size}, too few for a percentile interval at "
            f"conf_level={level!r} (most subjects have the same value in every "
            "condition)"
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
