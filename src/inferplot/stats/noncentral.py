"""Noncentral t and F distributions and the inversion of their noncentrality parameter.

Confidence intervals for standardized effect sizes invert a noncentral distribution
in its noncentrality parameter at the observed statistic (Steiger & Fouladi 1997,
"Noncentrality interval estimation and the evaluation of statistical models", in
Harlow, Mulaik & Steiger (eds.), What if there were no significance tests?;
Smithson 2003, Confidence Intervals, Sage; Steiger 2004, Psychological Methods 9:
164-182). Both distribution functions are computed here rather than with
``scipy.special.nctdtr`` / ``ncfdtr``: at the SciPy floor (1.11) those are off by up
to 1.5e-3 (t) and 0.1 or more (F) relative to min(P, 1 - P) at moderate
probabilities, and current SciPy's ``nctdtr`` returns NaN far in the tails.

Noncentral t (Johnson, Kotz & Balakrishnan 1995, Continuous Univariate Distributions,
vol. 2, ch. 31): T' = (Z + delta) / S with Z ~ N(0, 1) independent of
S = chi_nu / sqrt(nu), so

    P(T' <= t) = E[Phi(t S - delta)],  P(T' > t) = E[Phi(delta - t S)].

Both are integrated over u = log S (density from ``srange.log_scaled_chi_density``)
by composite 16-point Gauss-Legendre panels, summed in log space with
``scipy.special.log_ndtr``, so either tail keeps its relative accuracy. Panels are at
most min(0.5, 2 / sqrt(2 nu)) wide (the width of the density in u) and have extra
edges where the argument t e^u - delta crosses -40, -38, ..., 40, which resolves the
step of Phi when |delta| is large (its width in u is about 1 / |delta|). The range is
cut where the density is e^-80 below its peak, so the absolute error is below 1e-34
and the relative error stays near machine precision for probabilities above 1e-20.

Noncentral F (Johnson, Kotz & Balakrishnan 1995, ch. 30): the Poisson(lambda / 2)
mixture of beta distributions,

    P(F' <= f) = sum_j w_j I_x(df1 / 2 + j, df2 / 2),   x = df1 f / (df1 f + df2),

and the upper tail as sum_j w_j I_{1 - x}(df2 / 2, df1 / 2 + j) with 1 - x computed as
df2 / (df1 f + df2). The Poisson weights are built from their mode by cumulative
sums of log i and normalized over a window of +/- (15 sqrt(mu) + 30) terms, which
avoids the cancellation of -mu + j log mu - log j! for large mu.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special

from inferplot.stats.parametric import Alternative, check_alternative
from inferplot.stats.srange import log_scaled_chi_density

FloatArray = NDArray[np.float64]

__all__ = [
    "check_conf_level",
    "f_ncp_bound",
    "noncentral_f_logcdf",
    "noncentral_t_logcdf",
    "t_ncp_interval",
]

# The density of log S is followed until it is this many log units below its peak.
_DENSITY_DROP: Final = 80.0
# Breakpoints of Phi's argument t e^u - delta (in units of the standard normal).
_STEP_EDGES: Final = np.arange(-40.0, 41.0, 2.0)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_LOG_GL_WEIGHTS: Final = np.log(_GL_WEIGHTS)
# Poisson window half-width: 15 sqrt(mu) + 30 terms on each side of the mode.
_POISSON_SD: Final = 15.0
_POISSON_PAD: Final = 30.0
# Root finding: absolute tolerance on the noncentrality parameter.
_XTOL: Final = 1e-13
_MAX_EXPANSIONS: Final = 2000


def check_conf_level(conf_level: float) -> float:
    """``conf_level`` as a float strictly between 0 and 1 (ValueError otherwise)."""
    value = float(conf_level)
    if not 0.0 < value < 1.0:
        raise ValueError(f"conf_level must lie strictly between 0 and 1; got {value!r}")
    return value


def _logsumexp(a: FloatArray) -> float:
    peak = float(np.max(a))
    if not math.isfinite(peak):
        return peak
    return peak + math.log(float(np.sum(np.exp(a - peak))))


@functools.lru_cache(maxsize=64)
def _density_edges(nu: float) -> tuple[float, ...]:
    """Panel edges in u = log S covering the density down to e^-_DENSITY_DROP."""
    half = 0.5 * nu
    level = _DENSITY_DROP / half

    def excess(u: float) -> float:
        return math.expm1(2.0 * u) - 2.0 * u - level

    right = optimize.brentq(excess, 0.0, 0.5 * math.log1p(level) + 1.0, xtol=1e-12)
    left = optimize.brentq(excess, -0.5 * (level + 1.0) - 1.0, 0.0, xtol=1e-12)
    width = min(0.5, 2.0 / math.sqrt(2.0 * nu))
    count = max(1, math.ceil((right - left) / width))
    return tuple(np.linspace(left, right, count + 1).tolist())


def _check_t_args(t: float, df: float, ncp: float) -> None:
    if not (math.isfinite(t) and math.isfinite(ncp)):
        raise ValueError(f"t and ncp must be finite; got {t!r} and {ncp!r}")
    if math.isnan(df) or df <= 0.0:
        raise ValueError(f"df must be > 0 (inf for the normal limit); got {df!r}")


def noncentral_t_logcdf(
    t: float, df: float, ncp: float, *, upper: bool = False
) -> float:
    """log P(T' <= t), or log P(T' > t) with ``upper``, for T' ~ t'(df, ncp).

    The distribution R evaluates as ``pt(t, df, ncp)`` (module docstring for the
    method). ``df = inf`` gives the normal N(ncp, 1).
    """
    t, df, ncp = float(t), float(df), float(ncp)
    _check_t_args(t, df, ncp)
    sign = -1.0 if upper else 1.0
    if math.isinf(df):
        return float(special.log_ndtr(sign * (t - ncp)))
    if t == 0.0:
        return float(special.log_ndtr(-sign * ncp))
    edges = list(_density_edges(df))
    left, right = edges[0], edges[-1]
    ratios = (ncp + _STEP_EDGES) / t
    steps = np.log(ratios[ratios > 0.0])
    inside = steps[(steps > left) & (steps < right)]
    grid = np.array(sorted({*edges, *inside.tolist()}), dtype=np.float64)
    a, b = grid[:-1], grid[1:]
    half = 0.5 * (b - a)
    u = (0.5 * (a + b))[:, None] + half[:, None] * _GL_NODES[None, :]
    log_w = np.log(half)[:, None] + _LOG_GL_WEIGHTS[None, :]
    u = u.ravel()
    with np.errstate(over="ignore"):
        x = t * np.exp(u) - ncp
    terms = log_w.ravel() + log_scaled_chi_density(u, df) + special.log_ndtr(sign * x)
    return min(0.0, _logsumexp(terms))


def _check_f_args(f: float, df1: float, df2: float, ncp: float) -> None:
    for name, value in (("f", f), ("df1", df1), ("df2", df2), ("ncp", ncp)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
    if f < 0.0 or ncp < 0.0:
        raise ValueError(f"f and ncp must be >= 0; got {f!r} and {ncp!r}")
    if df1 <= 0.0 or df2 <= 0.0:
        raise ValueError(f"df1 and df2 must be > 0; got {df1!r} and {df2!r}")


def _log_poisson_window(mu: float) -> tuple[NDArray[np.int64], FloatArray]:
    """Indices j and normalized log Poisson(mu) weights in a window around the mode."""
    if mu == 0.0:
        return np.zeros(1, dtype=np.int64), np.zeros(1)
    mode = math.floor(mu)
    span = math.ceil(_POISSON_SD * math.sqrt(mu) + _POISSON_PAD)
    lo, hi = max(0, mode - span), mode + span
    j = np.arange(lo, hi + 1, dtype=np.int64)
    # log w_j - log w_mode = (j - mode) log mu - log(j! / mode!), the factorial ratio
    # as a cumulative sum of log i, so no large terms cancel.
    logs = np.log(np.arange(lo + 1, hi + 1, dtype=np.float64))  # log(lo+1)..log(hi)
    cumulative = np.zeros(logs.size + 1)  # log(j! / lo!)
    cumulative[1:] = logs.cumsum()
    at_mode = cumulative[mode - lo]
    rel = (j - mode).astype(np.float64) * math.log(mu) - (cumulative - at_mode)
    return j, rel - _logsumexp(rel)


def noncentral_f_logcdf(
    f: float, df1: float, df2: float, ncp: float, *, upper: bool = False
) -> float:
    """log P(F' <= f), or log P(F' > f) with ``upper``, for F' ~ F'(df1, df2, ncp).

    The distribution R evaluates as ``pf(f, df1, df2, ncp)`` (module docstring for
    the method); ``ncp = 0`` is the central F.
    """
    f, df1, df2, ncp = float(f), float(df1), float(df2), float(ncp)
    _check_f_args(f, df1, df2, ncp)
    if f == 0.0:
        return 0.0 if upper else -math.inf
    j, log_w = _log_poisson_window(0.5 * ncp)
    denom = df1 * f + df2
    a = 0.5 * df1 + j.astype(np.float64)
    with np.errstate(divide="ignore"):
        if upper:
            log_i = np.log(special.betainc(0.5 * df2, a, df2 / denom))
        else:
            log_i = np.log(special.betainc(a, 0.5 * df2, df1 * f / denom))
    return min(0.0, _logsumexp(log_w + log_i))


def _solve_increasing(
    g: Callable[[float], float], start: float, step: float, *, what: str
) -> float:
    """Root of an increasing function g, bracketed outward from ``start``."""
    lo, hi = start - step, start + step
    width = step
    for _ in range(_MAX_EXPANSIONS):
        if g(lo) < 0.0:
            break
        hi = lo
        width *= 2.0
        lo -= width
    else:  # pragma: no cover - g tends to -inf below the root
        raise ArithmeticError(f"could not bracket the {what} from below")
    width = step
    for _ in range(_MAX_EXPANSIONS):
        if g(hi) > 0.0:
            break
        lo = hi
        width *= 2.0
        hi += width
    else:  # pragma: no cover - g tends to +inf above the root
        raise ArithmeticError(f"could not bracket the {what} from above")
    root = optimize.brentq(g, lo, hi, xtol=_XTOL, rtol=4.0 * 2.0**-52, maxiter=500)
    return float(root)


def _clip(value: float) -> float:
    """Keep log probabilities finite for the root finder (both are monotone)."""
    return max(value, -1e300)


def _t_ncp(t: float, df: float, tail: float, *, upper: bool) -> float:
    """ncp with P(T' <= t) = tail (``upper`` False) or P(T' > t) = tail (True)."""
    log_tail = math.log(tail)
    scale = math.sqrt(1.0 + t * t / (2.0 * df)) if math.isfinite(df) else 1.0
    shrink = 1.0 - 1.0 / (4.0 * df) if df > 0.5 else 0.5
    # Normal approximation (Johnson, Kotz & Balakrishnan 1995, ch. 31) as the start.
    z = float(special.ndtri(tail))
    start = t * shrink + (z if upper else -z) * scale

    if upper:  # P(T' > t) increases with ncp

        def g(ncp: float) -> float:
            return _clip(noncentral_t_logcdf(t, df, ncp, upper=True)) - log_tail

    else:  # P(T' <= t) decreases with ncp

        def g(ncp: float) -> float:
            return log_tail - _clip(noncentral_t_logcdf(t, df, ncp))

    return _solve_increasing(g, start, scale, what="noncentrality parameter of t")


def t_ncp_interval(
    t: float,
    df: float,
    *,
    conf_level: float,
    alternative: Alternative,
) -> tuple[float, float]:
    """Confidence interval for the noncentrality parameter of an observed t.

    Test inversion (Steiger & Fouladi 1997; Cumming & Finch 2001, Educational and
    Psychological Measurement 61: 532-574): with alpha = 1 - conf_level, the
    two-sided bounds solve P(T'(df, ncp) <= t) = 1 - alpha / 2 (lower) and
    = alpha / 2 (upper); "greater" gives (the bound at 1 - alpha, +inf) and "less"
    (-inf, the bound at alpha). Probabilities above 1/2 are solved through the upper
    tail, P(T' > t) = alpha / 2 or alpha, so they keep their relative accuracy. The
    construction ``effectsize`` uses for ``hedges_g`` (R: ``pt(t, df, ncp)``
    inverted in ``ncp``).

    Raises:
        ValueError: t is not finite, df is not positive, conf_level is not strictly
            between 0 and 1, or alternative is not one of "two-sided", "less",
            "greater".
    """
    t, df = float(t), float(df)
    _check_t_args(t, df, 0.0)
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    alpha = 1.0 - level
    tail = alpha / 2.0 if side == "two-sided" else alpha
    low = -math.inf if side == "less" else _t_ncp(t, df, tail, upper=True)
    high = math.inf if side == "greater" else _t_ncp(t, df, tail, upper=False)
    return low, high


def f_ncp_bound(f: float, df1: float, df2: float, *, prob: float) -> float:
    """Noncentrality lambda >= 0 with P(F'(df1, df2, lambda) <= f) = ``prob``.

    The probability falls as lambda grows; when even lambda = 0 gives a probability
    at or below ``prob`` there is no positive root and the bound is 0.0, the edge of
    the parameter space (Steiger 2004, section 3). Probabilities above 1/2 are solved
    through the upper tail, P(F' > f) = 1 - prob. The inversion behind
    ``effectsize``'s noncentral-F intervals (R: ``pf(f, df1, df2, ncp)``).

    Raises:
        ValueError: an argument is not finite, f < 0, a df is not positive, or
            ``prob`` is not strictly between 0 and 1.
    """
    f, df1, df2, prob = float(f), float(df1), float(df2), float(prob)
    _check_f_args(f, df1, df2, 0.0)
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must lie strictly between 0 and 1; got {prob!r}")
    upper = prob > 0.5
    log_target = math.log1p(-prob) if upper else math.log(prob)
    if upper:
        at_zero = noncentral_f_logcdf(f, df1, df2, 0.0, upper=True)
        if at_zero >= log_target:
            return 0.0

        def g(ncp: float) -> float:
            return _clip(noncentral_f_logcdf(f, df1, df2, ncp, upper=True)) - log_target

    else:
        at_zero = noncentral_f_logcdf(f, df1, df2, 0.0)
        if at_zero <= log_target:
            return 0.0

        def g(ncp: float) -> float:
            return log_target - _clip(noncentral_f_logcdf(f, df1, df2, ncp))

    high = max(1.0, 2.0 * f * df1)
    for _ in range(_MAX_EXPANSIONS):
        if g(high) > 0.0:
            break
        high *= 2.0
    else:  # pragma: no cover - P(F' <= f) tends to 0 as lambda grows
        raise ArithmeticError(
            f"could not bracket the noncentrality parameter of F = {f!r}"
        )
    root = optimize.brentq(g, 0.0, high, xtol=_XTOL, rtol=4.0 * 2.0**-52, maxiter=500)
    return float(root)
