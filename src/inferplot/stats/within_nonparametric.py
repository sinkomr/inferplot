# NumPy's stubs leave overloads of sort, unique and the ufunc methods partially
# unknown, which strict mode reports on every call; the arrays passed are all typed.
# pyright: reportUnknownMemberType=false
"""Rank-based tests and effect sizes for repeated measures (``withinstats``, type
nonparametric).

Two conditions: Wilcoxon's signed-rank test of the paired differences with the
matched-pairs rank-biserial r.

Every rank here is a mid-rank, and twice a mid-rank is an integer, so rank sums,
tie corrections and sums of squares of ranks are computed exactly (Python integers
and fractions) and each statistic is rounded once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy import special

from inferplot.stats.descriptive import FloatArray, SampleLike, as_sample
from inferplot.stats.effect_sizes import EffectSizeCI
from inferplot.stats.noncentral import check_conf_level
from inferplot.stats.parametric import Alternative, check_alternative

__all__ = [
    "SignedRank",
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


# --------------------------------------------------------------------------- #
# paired differences and their signed ranks
# --------------------------------------------------------------------------- #


def _paired_samples(a: SampleLike, b: SampleLike) -> tuple[FloatArray, FloatArray]:
    x = as_sample(a, name="a")
    y = as_sample(b, name="b")
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
