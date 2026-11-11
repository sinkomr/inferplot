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
df2 / (df1 f + df2); for x > 1/2 the lower tail's factors of at least 1/2 are taken
as 1 - I_{1-x}(df2 / 2, df1 / 2 + j), since x itself has lost the digits of 1 - x
(eps / (1 - x) relative: 4e-12 at 1 - x = 3e-5). The Poisson weights are normalized
over a window of +/- (15 sqrt(mu) + 30) terms around the mode, mu = lambda / 2.

* mu below 1e4: every term of the window is summed; the weights are built from the
  mode by cumulative sums of log i, which avoids the cancellation of
  -mu + j log mu - log j!.
* mu from 1e4 on: the sum is taken on every h-th term only, h = floor(sqrt(mu) / 8),
  weighted by h. The summand w_j I(j) extends to an analytic function of j whose
  scale is at least sqrt(mu) (the Poisson weights' standard deviation; the beta
  factor varies more slowly), so this is the trapezoidal rule with about 8 nodes per
  standard deviation, whose error falls like exp(-2 pi^2 (sqrt(mu) / h)^2)
  (Trefethen & Weideman 2014, SIAM Review 56: 385-458). The weights at the nodes
  come from Loader's saddle-point form log w_j = -stirlerr(j) - mu phi((j - mu) / mu)
  - log sqrt(2 pi j), phi(v) = (1 + v) log1p(v) - v (Loader 2000, "Fast and
  accurate computation of binomial probabilities"), which stays accurate where the
  cumulative sums of a million log i drift (measured: log-weight error 3.7e-9 for the
  cumulative sums at mu = 4.5e6, 3e-14 for Loader's form, against mpmath). 4, 8, 16
  and 64 nodes per SD give the same sums to 1e-15; against the full sum with
  independently computed weights (tests/stats/test_noncentral.py, mu up to 2.5e6,
  both tails) they agree within 1e-12 relative. The window holds about 250 nodes
  whatever mu is.
* Beta factors with one parameter at least 1e4 and at least 10 times the other
  (and 1): SciPy's
  ``betainc`` loses accuracy there (up to 2.8e-10 absolute at b = 3 and 30,
  a = 3e5 to 3e8, and 1e-10 relative at a = 10, b = 5e7, in SciPy 1.18; 1e-3 at
  a = 1e14 in SciPy 1.11, NaN beyond). With independent gamma variables, the
  factor with the large parameter a is I_y(b, a) = P(G_b <= r G_a), r = y / x,
  or its complement I_x(a, b) (``_log_gamma_ratio``; a large b swaps the roles,
  I_x(a, b) = P(G_a <= G_b x / y)). With z = r a, the side below its median is
  computed and the other as log(1 - exp(.)): I_y(b, a) = E[P(b, r G_a)] for
  z <= b and I_x(a, b) = E[Q(b, r G_a)] above (P, Q the regularized incomplete
  gamma functions), by 32-node Gauss-Hermite quadrature in log G_a, centred on
  the peak of the integrand and scaled by its curvature (the log-gamma density's
  skewness is a^-1/2, so the integrand is close to Gaussian there;
  ``_log_gamma_ratio_side``). Everything is in log
  space with Loader's (2000) saddle-point forms, so no factor underflows and
  nothing cancels at a = 1e14. Against mpmath (40 digits; I_x(a, b) by an
  independent integral over G_a, I_y(b, a) by its betainc) at b = 0.25 to 1150
  with a / b from 1e2 to 1e12 and at b = 1e3 with a / b = 10 and 30 (a >= 1e4),
  both sides from 1/2 down to 1e-300: log P within 6.9e-15 max(1, |log P|). At
  a = b = 1e4 the quadrature is not accurate (3e-2 at log P = -56), hence the
  factor 10.
* Far tails: SciPy's ``betainc`` values below 1e-250 (it underflows to 0 near
  1e-280) are recomputed in log space by the continued fraction DLMF 8.17.22
  (``_log_beta_cf``), which converges quickly there. Where the beta factors shift
  the dominant terms out of the window around mu (far in a tail: at f = 1e-100,
  df = (2, 2) and ncp = 2e5 the largest term is j = 0), the sum is taken over a
  window around the largest term instead (``_tilted_window``). Against the
  chi'^2 x chi^2 integral in mpmath (tests/data/noncentral_reference.json) at
  ncp up to 2e12 for tails from 1e-3 to 1e-20 the relative error is at most
  5.0e-14 (3.1e-13 against exact sums at df2 = 1e6 and 1e8, df1 = 2), and log P
  is within 2.3e-15 relative of the reference down to log P = -2000.

Noncentral chi-square (Johnson, Kotz & Balakrishnan 1995, ch. 29): the
Poisson(lambda / 2) mixture of central chi-squares,

    P(X' <= x) = sum_j w_j P(df / 2 + j, x / 2),

P (and Q for the upper tail) the regularized incomplete gamma functions, summed over
the same Poisson window as the noncentral F (every term below a Poisson mean of 1e4,
every h-th term with Loader's weights above it, the window moved to the dominant
terms far in a tail). The gamma factors come from SciPy's ``gammainc`` /
``gammaincc``, recomputed in log space by a series or a continued fraction where
they fall below 1e-250 (``_log_gamma_tail``), so either tail keeps its relative
accuracy far out. Against 40-digit mpmath (tests/data/ncx2_reference.json) see
tests/stats/test_noncentral.py.
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
from inferplot.stats.srange import (
    expm1_minus_x,
    log_scaled_chi_density,
    stirlerr_large,
)

FloatArray = NDArray[np.float64]

__all__ = [
    "check_conf_level",
    "chi2_ncp_bound",
    "chi2_ncp_interval",
    "f_ncp_bound",
    "log_beta_factors",
    "noncentral_chi2_logcdf",
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
# From this Poisson mean on, the mixture is summed on every h-th term (module
# docstring), h = floor(sqrt(mu) / _NODES_PER_SD).
_SUBSAMPLE_FROM: Final = 1e4
_NODES_PER_SD: Final = 8.0
# Beta factors with one parameter at least this many times the other (and 1) are
# computed as gamma ratios (``_log_gamma_ratio``), not with SciPy's betainc
# (measured accurate from a factor 10 on; at a = b = 1e4 it is not).
_LARGE_RATIO: Final = 10.0
# ... and the larger parameter at least this large (the quadrature in log G needs
# a near-Gaussian log-gamma density).
_LARGE_MIN: Final = 1e4
# SciPy's betainc values below this are recomputed in log space (it underflows to
# 0 near 1e-280 and loses digits among the subnormals).
_BETAINC_FLOOR: Final = 1e-250
# SciPy's gammaincc values below this are recomputed in log space.
_GAMMA_FLOOR: Final = 1e-250
# Series and continued fractions: relative size of the last term or step, and the
# iteration limit (far above any input's needs; reaching it raises).
_SERIES_EPS: Final = 1e-17
_LENTZ_EPS: Final = 2.0**-52
_LENTZ_TINY: Final = 1e-300
_MAX_ITERATIONS: Final = 1_000_000
# Gauss-Hermite rule for the large-parameter lower side, and the fixed-point steps
# that centre it.
_HERMITE_NODES: Final = 32
_HERMITE_T, _HERMITE_W = np.polynomial.hermite.hermgauss(_HERMITE_NODES)
_HERMITE_LOG_W: Final = np.log(_HERMITE_W) + _HERMITE_T * _HERMITE_T
_PEAK_STEPS: Final = 4
# A window end within e^-_EDGE_DROP of the largest term means terms are missing.
_EDGE_DROP: Final = 50.0
_LOG_SQRT_2PI: Final = 0.5 * math.log(2.0 * math.pi)
_LOG_PI: Final = math.log(math.pi)
_LOG2: Final = math.log(2.0)
# Bounds above this noncentrality are searched from the large-lambda start value
# (f_ncp_bound); below it from 0, as before.
_TIGHT_BRACKET_FROM: Final = 1e4
# t_ncp_interval accepts |t| up to this: a bound is at most about 25 |t| for df >= 0.1
# at any conf_level (t times the root of a chi-square quantile over df), so every
# bound and bracket stays a finite double.
_T_MAX: Final = 1e300
# Root finding: absolute tolerance on the noncentrality parameter.
_XTOL: Final = 1e-13
# From this chi-square on, chi2_ncp_bound solves the Cornish-Fisher expansion of
# the noncentral chi-square quantile to its lambda^-1/2 term instead of searching
# (_chi2_ncp_expansion): what it leaves out is below a unit in the last place of
# lambda here (measured: at most 0.5 units from 1.4e9 on).
_CHI2_EXPANSION_FROM: Final = 1e9
# For prob < 0.5 (the bound above x) the expansion takes over much earlier: there
# the search's lower tail (noncentral_chi2_logcdf with lambda > x) loses accuracy
# from about 2e6 on (relative error at prob 0.025: 1e-13 at 3e6, 6e-12 at 1e7,
# 5e-11 from 1e8; at prob 0.0005: 2e-11 at 3e6, 1e-9 at 1e7), while the
# expansion's own error falls as 1 / x^2 (4e-13 at 2e6 and 2e-14 at 1e7 at prob
# 0.0005; 1e-13 and 5e-15 at prob 0.025). They cross near 2e6. For prob > 0.5 the
# search stays accurate (below 3e-16) up to 1e9. Measured against 40-digit roots
# (tools/reference/ncx2_mid_mpmath.py; DECISIONS.md D119).
_CHI2_EXPANSION_FROM_ABOVE: Final = 2e6
# The expansion is used while df is at most this share of x (measured accurate to
# its error above up to df = 10,000 at x = 3e6 and at 1e9, a share of 0.0033).
_CHI2_EXPANSION_DF_SHARE: Final = 1e-3
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
    return _checked_log_probability(
        _logsumexp(terms), f"t'(df={df!r}, ncp={ncp!r}) at t={t!r}"
    )


def _check_f_args(f: float, df1: float, df2: float, ncp: float) -> None:
    for name, value in (("f", f), ("df1", df1), ("df2", df2), ("ncp", ncp)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
    if f < 0.0 or ncp < 0.0:
        raise ValueError(f"f and ncp must be >= 0; got {f!r} and {ncp!r}")
    if df1 <= 0.0 or df2 <= 0.0:
        raise ValueError(f"df1 and df2 must be > 0; got {df1!r} and {df2!r}")


def _phi(v: FloatArray) -> FloatArray:
    """(1 + v) log1p(v) - v without cancellation near v = 0."""
    out = (1.0 + v) * np.log1p(v) - v
    small = np.abs(v) < 0.1
    if bool(small.any()):
        vs = v[small]
        # sum_{n >= 2} (-v)^n / (n (n - 1)), Horner form; 28 terms give < 1e-27 v^2.
        acc = np.zeros_like(vs)
        for n in range(30, 1, -1):
            acc = acc * -vs + 1.0 / (n * (n - 1))
        out[small] = acc * vs * vs
    return out


def _log_poisson_window(mu: float) -> tuple[FloatArray, FloatArray]:
    """Nodes j and normalized log Poisson(mu) weights in a window around the mode.

    Below ``_SUBSAMPLE_FROM`` every integer of the window, with exact weights; from
    it on every h-th integer with Loader's weights times h (module docstring).
    """
    if mu == 0.0:
        return np.zeros(1), np.zeros(1)
    if mu >= _SUBSAMPLE_FROM:
        sd = math.sqrt(mu)
        # As floats: a Python int above 2^63 would make NumPy 1.26 build an
        # object array.
        step = float(math.floor(sd / _NODES_PER_SD))
        count = math.ceil((_POISSON_SD * sd + _POISSON_PAD) / step)
        # mu >= 1e4: the window, 15 sqrt(mu) + 30 wide, stays far above j = 15.
        nodes = float(math.floor(mu)) + step * np.arange(
            -count, count + 1, dtype=np.float64
        )
        log_w = (
            -stirlerr_large(nodes) - mu * _phi((nodes - mu) / mu) - 0.5 * np.log(nodes)
        )
        return nodes, log_w - _logsumexp(log_w)
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
    return j.astype(np.float64), rel - _logsumexp(rel)


def _broadcast(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """``x`` and ``y`` broadcast to their common shape (as new arrays)."""
    zeros = np.zeros(np.broadcast_shapes(x.shape, y.shape))
    return x + zeros, y + zeros


def _stirlerr(x: FloatArray) -> FloatArray:
    """log Gamma(x) - (x - 1/2) log x + x - log sqrt(2 pi), element-wise (x > 0)."""
    out = np.empty_like(x)
    large = x >= 15.0
    if bool(large.any()):
        out[large] = stirlerr_large(x[large])
    small = ~large
    if bool(small.any()):
        xs = x[small]
        out[small] = special.gammaln(xs) - (xs - 0.5) * np.log(xs) + xs - _LOG_SQRT_2PI
    return out


def _bd0(x: FloatArray, m: FloatArray) -> FloatArray:
    """x log(x / m) + m - x >= 0 for x, m > 0, without cancellation (Loader 2000,
    "Fast and accurate computation of binomial probabilities"): m phi((x - m) / m)
    near x = m, x log1p((x - m) / m) - (x - m) within a factor of 2, the direct
    form beyond."""
    x, m = _broadcast(x, m)
    out = np.empty(x.shape)
    near = np.abs(x - m) < 0.1 * m
    if bool(near.any()):
        mn = m[near]
        out[near] = mn * _phi((x[near] - mn) / mn)
    # Within a factor of 2 x - m is exact (Sterbenz), so x log1p((x - m) / m)
    # - (x - m) keeps its relative accuracy where the two terms nearly cancel.
    mid = ~near & (x >= 0.5 * m) & (x <= 2.0 * m)
    if bool(mid.any()):
        xm, mm = x[mid], m[mid]
        diff = xm - mm
        out[mid] = xm * np.log1p(diff / mm) - diff
    far = ~near & ~mid
    if bool(far.any()):
        xf, mf = x[far], m[far]
        out[far] = xf * (np.log(xf) - np.log(mf)) + mf - xf
    return out


def _log_beta_prefactor(p: FloatArray, q: FloatArray, x: float, y: float) -> FloatArray:
    """log(x^p y^q / (p B(p, q))), y = 1 - x given separately, in Loader's form
    -bd0(p, n x) - bd0(q, n y) + log sqrt(p q / (2 pi n)) + stirlerr(n)
    - stirlerr(p) - stirlerr(q) - log p, n = p + q (the terms n x - p and n y - q
    cancel because x + y = 1), so neither p log x nor log B(p, q) loses digits
    when p or q is large."""
    n = p + q
    return (
        -_bd0(p, n * x)
        - _bd0(q, n * y)
        + 0.5 * (np.log(p) + np.log(q) - np.log(n))
        - _LOG_SQRT_2PI
        + _stirlerr(n)
        - _stirlerr(p)
        - _stirlerr(q)
        - np.log(p)
    )


def _log1mexp(x: FloatArray) -> FloatArray:
    """log(1 - exp(x)) for x <= 0 (Maechler 2012, "Accurately computing
    log(1 - exp(-|a|))"), -inf at x = 0."""
    out = np.empty_like(x)
    near = x > -_LOG2
    with np.errstate(divide="ignore"):
        out[near] = np.log(-np.expm1(x[near]))
    out[~near] = np.log1p(-np.exp(x[~near]))
    return out


def _converged(done: bool, what: str) -> None:
    if not done:  # pragma: no cover - far above the iterations any input needs
        raise ArithmeticError(f"the {what} did not converge")


def _log_beta_cf(p: FloatArray, q: FloatArray, x: float, y: float) -> FloatArray:
    """log I_x(p, q) by the continued fraction (DLMF 8.17.22)
    I_x(p, q) = x^p y^q / (p B(p, q)) / (1 + d_1 / (1 + d_2 / (1 + ...))),
    d_(2m+1) = -(p + m)(p + q + m) x / ((p + 2m)(p + 2m + 1)),
    d_(2m) = m (q - m) x / ((p + 2m - 1)(p + 2m)), evaluated by the modified Lentz
    algorithm (Thompson & Barnett 1986, Journal of Computational Physics 64:
    490-509), the prefactor by ``_log_beta_prefactor``. Used far in a tail, where
    x is well below (p + 1) / (p + q + 2) and it converges in a few dozen terms.
    """
    p, q = _broadcast(p, q)
    n = p + q
    f = np.ones(p.shape)
    c = np.ones(p.shape)
    d = np.zeros(p.shape)
    done = False
    for i in range(1, _MAX_ITERATIONS):
        m = i // 2
        if i % 2:
            coef = -(p + m) * (n + m) * x / ((p + 2 * m) * (p + 2 * m + 1.0))
        else:
            coef = m * (q - m) * x / ((p + 2 * m - 1.0) * (p + 2 * m))
        d = 1.0 + coef * d
        d[d == 0.0] = _LENTZ_TINY
        c = 1.0 + coef / c
        c[c == 0.0] = _LENTZ_TINY
        d = 1.0 / d
        delta = c * d
        f = f * delta
        if i % 2 == 0 and bool((np.abs(delta - 1.0) <= _LENTZ_EPS).all()):
            done = True
            break
    _converged(done, "continued fraction of the incomplete beta function")
    return _log_beta_prefactor(p, q, x, y) - np.log(f)


def _log_gamma_density(b: FloatArray, w: FloatArray) -> FloatArray:
    """log of the gamma(b) density at w in Loader's form, -bd0(b, w) - log(w / b)
    - log sqrt(2 pi b) - stirlerr(b) (= (b - 1) log w - w - log Gamma(b))."""
    return -_bd0(b, w) - np.log(w / b) - 0.5 * np.log(b) - _LOG_SQRT_2PI - _stirlerr(b)


def _log_gamma_tail(
    b: FloatArray, w: FloatArray, *, upper: bool
) -> tuple[FloatArray, FloatArray]:
    """log T(b, w) and log(T(b, w) / p(w)), T the regularized upper incomplete
    gamma function Q (``upper``) or the lower one P, p the gamma(b) density.

    SciPy's ``gammaincc`` / ``gammainc`` where T is at least ``_GAMMA_FLOOR``.
    Below, where it underflows, the ratio T / p directly: for Q (far above the
    mean) w K with Legendre's continued fraction Gamma(b, w) = e^-w w^b K,
    K = 1 / (w + 1 - b - 1 (1 - b) / (w + 3 - b - 2 (2 - b) / (w + 5 - b - ...)))
    (DLMF 8.9.2, even part) by the modified Lentz algorithm; for P (far below the
    mean) (w / b) M with Kummer's series M = sum_k w^k / ((b + 1) ... (b + k))
    (DLMF 8.7.1); both converge quickly there. The ratio stays accurate where
    log T and log p are both huge (w = 1e274) and their difference is not.
    """
    b, w = _broadcast(b, w)
    with np.errstate(under="ignore"):
        if upper:
            tail = np.asarray(special.gammaincc(b, w), dtype=np.float64)
        else:
            tail = np.asarray(special.gammainc(b, w), dtype=np.float64)
    log_tail = np.empty(w.shape)
    log_ratio = np.empty(w.shape)
    direct = tail >= _GAMMA_FLOOR
    if bool(direct.any()):
        log_tail[direct] = np.log(tail[direct])
        log_ratio[direct] = log_tail[direct] - _log_gamma_density(b[direct], w[direct])
    deep = ~direct
    if bool(deep.any()):
        bd, wd = b[deep], w[deep]
        done = False
        if upper:
            coef = wd + 1.0 - bd
            f = coef.copy()
            c = coef.copy()
            d = np.zeros_like(wd)
            for n in range(1, _MAX_ITERATIONS):
                an = -n * (n - bd)
                coef = coef + 2.0
                d = coef + an * d
                d[d == 0.0] = _LENTZ_TINY
                c = coef + an / c
                c[c == 0.0] = _LENTZ_TINY
                d = 1.0 / d
                delta = c * d
                f = f * delta
                if bool((np.abs(delta - 1.0) <= _LENTZ_EPS).all()):
                    done = True
                    break
            ratio = np.log(wd) - np.log(f)
        else:
            term = np.ones(bd.shape)
            total = np.ones(bd.shape)
            for k in range(1, _MAX_ITERATIONS):
                term = term * wd / (bd + k)
                total = total + term
                if bool((term <= _SERIES_EPS * total).all()):
                    done = True
                    break
            ratio = np.log(wd / bd) + np.log(total)
        _converged(done, "incomplete gamma function's continued fraction or series")
        log_ratio[deep] = ratio
        log_tail[deep] = _log_gamma_density(bd, wd) + ratio
    return log_tail, log_ratio


def _log_gamma_ratio_side(
    big: FloatArray, small: FloatArray, z: FloatArray, *, upper: bool
) -> FloatArray:
    """log P(G_small > G_big z / big) (``upper``) or log P(G_small <= G_big z / big)
    for independent gamma variables, big large: the beta factors I_x(big, small)
    and I_y(small, big), z / big = y / x.

    The integral of Q(small, w) (or P(small, w)), w = z g / big, against the
    gamma(big) density, in u = log(g / big), by Gauss-Hermite quadrature
    (``_HERMITE_NODES`` nodes) centred on the peak of the integrand and scaled by
    its curvature. The integrand is close to Gaussian in u: the log-gamma density
    has skewness about big^-1/2, and the gamma(small) tail varies far more slowly
    in u (on a scale of about 1 / sqrt(small) in the bulk and 1 / w in the far
    upper tail, against the 1 / sqrt(big) of the density). The peak solves
    big (e^u - 1) = -/+ w p(w) / T(w) (T the tail, p the gamma(small) density),
    found by a few fixed-point steps from u = 0; the curvature by a second
    difference over about one standard deviation. The gamma(big) density times g
    is exp(-big (e^u - 1 - u)) / (sqrt(2 pi / big) e^stirlerr(big)) (Loader's
    form, evaluated at the nodes u, so big = 1e14 loses nothing to the rounding
    of g), and T is evaluated in log space (``_log_gamma_tail``),
    so either side keeps its relative accuracy however far in its tail.
    """

    def log_tail(b: FloatArray, w: FloatArray) -> FloatArray:
        return _log_gamma_tail(b, w, upper=upper)[0]

    centre = np.zeros(big.shape)
    w = z
    for _ in range(_PEAK_STEPS):
        w = z * np.exp(centre)
        # The tail's pull on the peak, w p(w) / T(w) over big, in logs (rr = z / big
        # reaches 1e300): e^u = 1 / (1 + rr p / Q) or 1 + w p / (P big).
        log_ratio = _log_gamma_tail(small, w, upper=upper)[1]
        if upper:
            centre = -np.logaddexp(0.0, np.log(z) - np.log(big) - log_ratio)
        else:
            centre = np.logaddexp(0.0, np.log(w) - np.log(big) - log_ratio)

    def log_integrand(u: FloatArray) -> FloatArray:
        return -big * expm1_minus_x(u) + log_tail(small, z * np.exp(u))

    delta = 1.0 / np.sqrt(big * np.exp(centre) + w + 1.0)
    curvature = -(
        log_integrand(centre + delta)
        - 2.0 * log_integrand(centre)
        + log_integrand(centre - delta)
    ) / (delta * delta)
    if not bool((curvature > 0.0).all()):  # pragma: no cover - log-concave
        raise ArithmeticError("the gamma-ratio integrand is not log-concave")
    u = centre[:, None] + np.sqrt(2.0 / curvature)[:, None] * _HERMITE_T
    terms = (
        _HERMITE_LOG_W
        - big[:, None] * expm1_minus_x(u)
        + log_tail(small[:, None], z[:, None] * np.exp(u))
    )
    peak = terms.max(axis=1)
    total = np.log(np.exp(terms - peak[:, None]).sum(axis=1))
    return (
        peak
        + total
        + 0.5 * (np.log(big) - np.log(curvature) - _LOG_PI)
        - _stirlerr(big)
    )


def _log_gamma_ratio(
    big: FloatArray, small: FloatArray, rr: float
) -> tuple[FloatArray, FloatArray]:
    """log U and log L, U = P(G_small <= rr G_big) = I_y(small, big) and
    L = 1 - U = I_x(big, small) with x = 1 / (1 + rr), y = rr / (1 + rr), for
    big >= ``_LARGE_RATIO`` max(small, 1): the side below its median by
    ``_log_gamma_ratio_side`` (U for z = rr big <= small, L above), the other as
    log(1 - exp(.))."""
    big, small = _broadcast(big, small)
    log_u = np.zeros(big.shape)
    log_l = np.zeros(big.shape)
    with np.errstate(over="ignore"):
        z = rr * big
    # The limits: z = 0 (x = 1) gives U = 0; z beyond the largest double gives
    # L = 0 to double precision (log L is about -z).
    log_u[z == 0.0] = -math.inf
    log_l[np.isinf(z)] = -math.inf
    low = (z <= small) & (z > 0.0)
    if bool(low.any()):
        log_u[low] = _log_gamma_ratio_side(big[low], small[low], z[low], upper=False)
        log_l[low] = _log1mexp(log_u[low])
    high = (z > small) & np.isfinite(z)
    if bool(high.any()):
        log_l[high] = _log_gamma_ratio_side(big[high], small[high], z[high], upper=True)
        log_u[high] = _log1mexp(log_l[high])
    return log_u, log_l


def log_beta_factors(
    a: FloatArray, b: float, x: float, y: float, r: float, *, upper: bool
) -> FloatArray:
    """log I_y(b, a) (``upper``) or log I_x(a, b), with y = 1 - x and r = y / x
    given separately (all accurate), per element of a: the beta factors of the
    noncentral F mixture (module docstring).

    * a >= ``_LARGE_RATIO`` max(b, 1): I_y(b, a) = P(G_b <= r G_a) and its
      complement, as gamma ratios (``_log_gamma_ratio``); b >= ``_LARGE_RATIO``
      max(a, 1): the same with the roles swapped, I_x(a, b) = P(G_a <= G_b x / y).
    * otherwise SciPy's ``betainc``; values below ``_BETAINC_FLOOR``, where it
      underflows, from the continued fraction in log space (``_log_beta_cf``).

    Equal values of a (far out, the Poisson nodes round to the same double) are
    computed once.
    """
    order = a.argsort(kind="stable")
    ordered = a[order]
    first = np.ones(a.shape, dtype=bool)
    first[1:] = ordered[1:] != ordered[:-1]
    if not bool(first.all()):
        values = log_beta_factors(ordered[first], b, x, y, r, upper=upper)
        out = np.empty_like(a)
        out[order] = values[first.cumsum() - 1]
        return out
    out = np.empty_like(a)
    small_b = np.full_like(a, b)
    large_a = (a >= _LARGE_RATIO * max(b, 1.0)) & (a >= _LARGE_MIN)
    large_b = ~large_a & (b / _LARGE_RATIO >= np.maximum(a, 1.0)) & (b >= _LARGE_MIN)
    if bool(large_a.any()):
        log_u, log_l = _log_gamma_ratio(a[large_a], small_b[large_a], r)
        out[large_a] = log_u if upper else log_l
    if bool(large_b.any()):
        with np.errstate(divide="ignore"):
            rr = x / y
        log_u, log_l = _log_gamma_ratio(small_b[large_b], a[large_b], rr)
        out[large_b] = log_l if upper else log_u
    rest = ~(large_a | large_b)
    if bool(rest.any()):
        small = a[rest]
        if upper:
            values = np.asarray(special.betainc(b, small, y), dtype=np.float64)
        elif x > 0.5:
            # I_x(a, b) = 1 - I_y(b, a) from y, which keeps the digits that x has
            # lost, where that complement is at least 1/2 (so 1 - c keeps its
            # relative accuracy); below 1/2, from x as before.
            values = 1.0 - np.asarray(special.betainc(b, small, y), dtype=np.float64)
            direct = values < 0.5
            if bool(direct.any()):
                values[direct] = special.betainc(small[direct], b, x)
        else:
            values = np.asarray(special.betainc(small, b, x), dtype=np.float64)
        with np.errstate(divide="ignore"):
            logs = np.log(values)
        deep = values < _BETAINC_FLOOR
        if bool(deep.any()):
            # The requested side is far in its tail, where the continued fraction
            # converges quickly.
            if upper:
                logs[deep] = _log_beta_cf(
                    np.full(int(deep.sum()), b), small[deep], y, x
                )
            else:
                logs[deep] = _log_beta_cf(
                    small[deep], np.full(int(deep.sum()), b), x, y
                )
        out[rest] = logs
    return out


def _checked_log_probability(value: float, what: str) -> float:
    """``value`` capped at 0 (log 1); ArithmeticError if it is NaN, which must
    never pass as a probability (min(0.0, nan) is 0.0)."""
    if math.isnan(value):
        raise ArithmeticError(f"the log probability of {what} is not a number")
    return min(0.0, value)


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
    mu = 0.5 * ncp
    b = 0.5 * df2
    denom = df1 * f + df2
    if math.isfinite(denom):
        y, x, r = df2 / denom, df1 * f / denom, df2 / (df1 * f)
    else:  # df1 f overflows: 1 - x = r / (1 + r) = r to double precision
        r = df2 / df1 / f
        y, x = r, 1.0
    what = f"F'(df1={df1!r}, df2={df2!r}, ncp={ncp!r}) at f={f!r}"

    def log_factors(j: FloatArray) -> FloatArray:
        return log_beta_factors(0.5 * df1 + j, b, x, y, r, upper=upper)

    j, log_w = _log_poisson_window(mu)
    terms = log_w + log_factors(j)
    if mu > 0.0 and _misses_terms(j, log_w, terms):
        # Far in a tail the beta factors shift the dominant terms out of the
        # window around mu: sum around them instead (``_tilted_window``).
        j, log_w = _tilted_window(mu, log_factors, upper=upper)
        terms = log_w + log_factors(j)
        if _misses_terms(j, log_w, terms):  # pragma: no cover - not reached
            raise ArithmeticError(f"the Poisson sum of {what} could not be bracketed")
    return _checked_log_probability(_logsumexp(terms), what)


def _misses_terms(j: FloatArray, log_w: FloatArray, terms: FloatArray) -> bool:
    """Whether an end of the window (other than j = 0) holds a term within
    e^-``_EDGE_DROP`` of the largest although its Poisson weight is not: the beta
    factors have then shifted the dominant terms towards that end, and the sum
    misses terms beyond it. (A window whose weights do not vary, as at mu = 1e50,
    where the nodes round to mu, needs no check.) The summand is log-concave in
    j, so terms beyond an end that is that far below the peak fall at least
    geometrically and add less than e^-50 / (slope)."""
    peak = float(np.max(terms))
    if not math.isfinite(peak):
        return False
    weight_peak = float(np.max(log_w))

    def open_end(i: int) -> bool:
        return (
            float(terms[i]) > peak - _EDGE_DROP
            and float(log_w[i]) < weight_peak - _EDGE_DROP
        )

    return (float(j[0]) > 0.0 and open_end(0)) or open_end(-1)


def _log_poisson_pmf(j: FloatArray, mu: float) -> FloatArray:
    """log P(J = j), J ~ Poisson(mu), for integers j >= 0 (as floats), in Loader's
    saddle-point form -stirlerr(j) - bd0(j, mu) - log sqrt(2 pi j) (Loader 2000)."""
    out = np.full(j.shape, -mu)
    positive = j > 0.0
    if bool(positive.any()):
        jp = j[positive]
        out[positive] = (
            -_stirlerr(jp) - _bd0(jp, np.full(jp.shape, mu)) - 0.5 * np.log(jp)
        ) - _LOG_SQRT_2PI
    return out


def _tilted_window(
    mu: float, log_factors: Callable[[FloatArray], FloatArray], *, upper: bool
) -> tuple[FloatArray, FloatArray]:
    """Nodes j and log Poisson(mu) weights around the dominant term of
    sum_j w_j I_j when it lies outside the window around mu.

    The log of the summand has slope log mu - digamma(j + 1) + d/dj log I_j in j,
    zero at the largest term (below mu for the lower tail, whose beta factors fall
    with j, above it for the upper). The slope is found by Brent's method, with
    differences over Delta = sqrt(j) terms (about one standard deviation of the
    summand), so that the rounding of log I_j (about 1e-16 |log I_j|) does not
    move the root; the digamma term is taken at the midpoint, which matches the
    difference of log j! over Delta to O(Delta^2 / j^3). The summand is
    log-concave with curvature at least that of the Poisson weights, 1 / j, so the
    window j* +/- (15 sqrt(j*) + 30) holds it as the window around mu holds the
    weights, subsampled the same way from j* = 1e4 on; it is centred once more on
    its largest term.
    """

    def slope(j: float) -> float:
        delta = max(1.0, math.floor(math.sqrt(j)))
        pair = log_factors(np.array([j, j + delta]))
        rise = float(pair[1] - pair[0]) / delta
        return math.log(mu) - float(special.digamma(j + 1.0 + 0.5 * delta)) + rise

    if upper:
        lo, hi = mu, 2.0 * mu + 1.0
        while slope(hi) > 0.0:
            lo, hi = hi, 2.0 * hi
        centre = optimize.brentq(slope, lo, hi, xtol=0.5)
    elif slope(0.0) <= 0.0:
        centre = 0.0
    else:
        centre = optimize.brentq(slope, 0.0, mu, xtol=0.5)
    j, log_w = _window_at(centre, mu)
    terms: FloatArray = log_w + log_factors(j)
    largest = float(j[int(terms.argmax())])
    return _window_at(largest, mu)


def _window_at(centre: float, mu: float) -> tuple[FloatArray, FloatArray]:
    """Nodes j +/- (15 sqrt(j) + 30) around ``centre`` (every h-th from 1e4 on,
    h = floor(sqrt(centre) / 8)) and their log Poisson(mu) weights times h."""
    spread = math.sqrt(max(centre, 1.0))
    half = _POISSON_SD * spread + _POISSON_PAD
    step = (
        float(math.floor(spread / _NODES_PER_SD)) if centre >= _SUBSAMPLE_FROM else 1.0
    )
    first = max(0.0, math.floor(centre) - step * math.ceil(half / step))
    count = math.ceil((math.floor(centre) + half - first) / step) + 1
    j = first + step * np.arange(count, dtype=np.float64)
    return j, _log_poisson_pmf(j, mu) + math.log(step)


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
    if not math.isfinite(df):
        scale = 1.0
    elif abs(t) < 1e150:
        scale = math.sqrt(1.0 + t * t / (2.0 * df))
    else:  # t * t would overflow; the 1 is far below the rounding of t^2 / (2 df)
        scale = abs(t) / math.sqrt(2.0 * df)
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

    Range: |t| up to 1e300 (beyond it a bound can exceed the largest double). For
    df >= 1 and |t| from 1e150 to 1e300 the bounds agree with the large-ncp limit
    t sqrt(q / df) (q the chi-square(df) quantile of the tail) within 2e-15
    relative; Welch's t of two samples stays below 2^512 (about 1.3e154,
    ``welch_t_scaled``). Below df = 1, which Welch's df never is, large |t| is
    not covered: at df = 1e-3 and |t| = 1e150 the integration takes minutes, and
    at df = 0.1, t = 1e100 and conf_level 1 - 2^-53 the root search raises
    RuntimeError.

    Raises:
        ValueError: t is not finite or |t| > 1e300, df is not positive, conf_level
            is not strictly between 0 and 1, or alternative is not one of
            "two-sided", "less", "greater".
    """
    t, df = float(t), float(df)
    _check_t_args(t, df, 0.0)
    if abs(t) > _T_MAX:
        raise ValueError(
            f"t = {t:.3g} is too large for a confidence interval: the bounds are "
            "computed for |t| up to 1e300, and beyond it they can exceed the largest "
            "double (about 1.8e308)"
        )
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    alpha = 1.0 - level
    tail = alpha / 2.0 if side == "two-sided" else alpha
    low = -math.inf if side == "less" else _t_ncp(t, df, tail, upper=True)
    high = math.inf if side == "greater" else _t_ncp(t, df, tail, upper=False)
    return low, high


_LARGEST: Final = float(np.finfo(np.float64).max)
_BRACKET_STEP: Final = 4.0


def _f_ncp_from(g: Callable[[float], float], start: float) -> float:
    """Root of an increasing g near ``start`` > 0 (math.inf beyond the doubles).

    The bracket is start +/- 4 sqrt(start), widened four-fold until it holds the
    root; g(0) < 0 is known, so the lower end stops at 0.
    """
    if not math.isfinite(start):
        return math.inf
    step = _BRACKET_STEP * math.sqrt(start)
    lo = start - step
    while lo > 0.0 and g(lo) >= 0.0:
        step *= _BRACKET_STEP
        lo = start - step
    lo = max(lo, 0.0)
    step = _BRACKET_STEP * math.sqrt(start)
    while True:
        hi = min(start + step, _LARGEST)
        if g(hi) > 0.0:
            break
        if hi == _LARGEST:
            return math.inf
        step *= _BRACKET_STEP
    root = optimize.brentq(g, lo, hi, xtol=_XTOL, rtol=4.0 * 2.0**-52, maxiter=500)
    return float(root)


def f_ncp_bound(f: float, df1: float, df2: float, *, prob: float) -> float:
    """Noncentrality lambda >= 0 with P(F'(df1, df2, lambda) <= f) = ``prob``.

    The probability falls as lambda grows; when even lambda = 0 gives a probability
    at or below ``prob`` there is no positive root and the bound is 0.0, the edge of
    the parameter space (Steiger 2004, section 3). Probabilities above 1/2 are solved
    through the upper tail, P(F' > f) = 1 - prob. The inversion behind
    ``effectsize``'s noncentral-F intervals (R: ``pf(f, df1, df2, ncp)``).

    Root search: Brent's method. When the large-lambda value
    lambda_0 = df1 (f q / df2 - 1) (q: the chi-square(df2) quantile with upper tail
    ``prob``, from F' ~ ((lambda + df1) / df1) / (chi2_df2 / df2) for large lambda)
    is at least 1e4, the bracket is lambda_0 +/- 4 sqrt(lambda_0), widened four-fold
    until it holds the root; otherwise it runs from 0 to 2 f df1, doubled until it
    holds the root. With the subsampled Poisson sum (module docstring) the search
    takes 1 ms to 0.25 s at any f. Returns ``math.inf`` when the root lies
    beyond the largest double.

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

    # Large lambda: F' is close to ((lambda + df1) / df1) / (chi2_df2 / df2), so the
    # root is close to df1 (f q / df2 - 1) with q the chi2_df2 quantile that has
    # upper tail prob.
    q = float(special.chdtri(df2, prob))
    start = df1 * (f / df2 * q - 1.0)
    if start >= _TIGHT_BRACKET_FROM:
        return _f_ncp_from(g, start)
    high = min(max(1.0, 2.0 * f * df1), _LARGEST)
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


def _check_chi2_args(x: float, df: float, ncp: float) -> None:
    for name, value in (("x", x), ("df", df), ("ncp", ncp)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; got {value!r}")
    if x < 0.0 or ncp < 0.0:
        raise ValueError(f"x and ncp must be >= 0; got {x!r} and {ncp!r}")
    if df <= 0.0:
        raise ValueError(f"df must be > 0; got {df!r}")


def noncentral_chi2_logcdf(
    x: float, df: float, ncp: float, *, upper: bool = False
) -> float:
    """log P(X' <= x), or log P(X' > x) with ``upper``, for X' ~ chi'^2(df, ncp).

    The distribution R evaluates as ``pchisq(x, df, ncp)`` (module docstring for the
    method: the Poisson mixture of regularized incomplete gamma functions); ``ncp =
    0`` is the central chi-square.

    Raises:
        ValueError: an argument is not finite, x or ncp is negative, or df is not
            positive.
    """
    x, df, ncp = float(x), float(df), float(ncp)
    _check_chi2_args(x, df, ncp)
    if x == 0.0:
        return 0.0 if upper else -math.inf
    mu = 0.5 * ncp
    half_df = 0.5 * df
    half_x = 0.5 * x
    what = f"chi'^2(df={df!r}, ncp={ncp!r}) at x={x!r}"

    def log_factors(j: FloatArray) -> FloatArray:
        return _log_gamma_tail(half_df + j, np.full(j.shape, half_x), upper=upper)[0]

    j, log_w = _log_poisson_window(mu)
    terms = log_w + log_factors(j)
    if mu > 0.0 and _misses_terms(j, log_w, terms):
        # As for the noncentral F: far in a tail the gamma factors shift the
        # dominant terms out of the window around mu.
        j, log_w = _tilted_window(mu, log_factors, upper=upper)
        terms = log_w + log_factors(j)
        if _misses_terms(j, log_w, terms):  # pragma: no cover - not reached
            raise ArithmeticError(f"the Poisson sum of {what} could not be bracketed")
    return _checked_log_probability(_logsumexp(terms), what)


def _chi2_ncp_start(x: float, df: float, prob: float) -> float:
    """The noncentrality with P(X' <= x) = prob under the normal approximation
    X' ~ N(df + lambda, 2 (df + 2 lambda)) (Johnson, Kotz & Balakrishnan 1995,
    ch. 29): the root of (x - df - lambda)^2 = z^2 (2 df + 4 lambda) on the side
    where x - df - lambda has the sign of z = Phi^-1(prob)."""
    z = float(special.ndtri(prob))
    d = x - df
    centre = d + 2.0 * z * z
    disc = centre * centre - d * d + 2.0 * z * z * df
    if disc < 0.0:  # pragma: no cover - disc >= 4 z^2 (x + z^2) >= 0
        return 0.0
    root = math.sqrt(disc)
    return centre - root if z > 0.0 else centre + root


def _chi2_ncp_expansion(x: float, df: float, prob: float) -> float:
    """The noncentrality lambda with P(chi'^2(df, lambda) <= x) = prob for a large
    x (at least ``_CHI2_EXPANSION_FROM``), from the Cornish-Fisher expansion of the
    prob-quantile (Cornish & Fisher 1937; Johnson, Kotz & Balakrishnan 1994, vol. 1,
    ch. 12) in the cumulants of chi'^2, k_r = 2^(r - 1) (r - 1)! (df + r lambda)
    (Johnson, Kotz & Balakrishnan 1995, vol. 2, ch. 29). With s = sqrt(k_2),
    g_1 = k_3 / s^3, g_2 = k_4 / s^4 and z = Phi^-1(prob):

        x = k_1 + s (z + g_1 (z^2 - 1) / 6
                     + g_2 (z^3 - 3 z) / 24 - g_1^2 (2 z^3 - 5 z) / 36)
            + O(z^4 / lambda),

    the last two terms of order lambda^-1/2. The solution in lambda, by fixed-point
    steps lambda <- x - (everything else), which contract by about |z| /
    sqrt(lambda), is the root to the rounding of x: against 50-digit roots for
    df = 1 and 3 (exact representations; tools/reference/ncx2_large_mpmath.py) at
    most 0.5 units in its last place from x = 1e9 to 1.7e308 (1.7 units at 1.4e8,
    316 at 1.4e7, where the search is used). The search it replaces is off there:
    from x of about 1e7 to 1e12 by up to 3e10 units in the last place (5e-11
    relative at prob 0.025, 3e-9 at 0.001; noncentral_chi2_logcdf's tails above
    x), and from about 1e31 on it did not converge.
    """
    z = float(special.ndtri(prob))
    z2 = z * z
    root = x
    for _ in range(8):
        # Every cumulant ratio in r = df / lambda, so that nothing overflows up to
        # the largest double: s = sqrt(2 lambda (r + 2)), s g_1 = 4 (r + 3) /
        # (r + 2), s g_2 = 24 (r + 4) / ((r + 2) s), s g_1^2 = 16 (r + 3)^2 /
        # ((r + 2)^2 s).
        r = df / root
        s = math.sqrt(2.0 * (r + 2.0)) * math.sqrt(root)
        skew = (2.0 / 3.0) * (z2 - 1.0) * (r + 3.0) / (r + 2.0)
        second = (
            (r + 4.0) / (r + 2.0) * (z2 - 3.0) * z
            - 4.0 / 9.0 * ((r + 3.0) / (r + 2.0)) ** 2 * (2.0 * z2 - 5.0) * z
        ) / s
        step = x - (df + z * s + skew + second)
        if step == root:
            break
        root = step
    return root


def chi2_ncp_bound(x: float, df: float, *, prob: float) -> float:
    """Noncentrality lambda >= 0 with P(chi'^2(df, lambda) <= x) = ``prob``.

    The probability falls as lambda grows; when even lambda = 0 gives a probability
    at or below ``prob`` there is no positive root and the bound is 0.0, the edge of
    the parameter space (Steiger 2004, section 3; Smithson 2003, ch. 5). The
    inversion behind ``effectsize``'s noncentral chi-square intervals of Cramer's V
    and Pearson's C (R: ``pchisq(x, df, ncp)`` inverted in ``ncp``). Probabilities
    above 1/2 are solved through the upper tail, P(X' > x) = 1 - prob, so both keep
    their relative accuracy.

    Root search: Brent's method to 1e-13 absolute and 4 ulp relative, which is the
    accurate root; effectsize's own search stops earlier on some bounds (D99).
    When the normal approximation of the root (``_chi2_ncp_start``) is at least 1e4
    the bracket is that value +/- 4 sqrt(value), widened four-fold until it holds the
    root; otherwise it runs from 0 to max(1, 2 x), doubled until it holds the root.
    Returns ``math.inf`` when the root lies beyond the largest double. From x =
    1e9 on (``_CHI2_EXPANSION_FROM``), and from 2e6 on for prob < 0.5
    (``_CHI2_EXPANSION_FROM_ABOVE``: the bound above x, where the search loses
    accuracy), with df at most a thousandth of x, the root is the solution of the
    Cornish-Fisher expansion (``_chi2_ncp_expansion``): accurate to the rounding
    of x from 1e9, and within 5e-13 relative at 2e6 (1e-13 at prob 0.025), so
    every finite x has its bounds at once (the search's incomplete gamma series
    stopped converging from about 1e31 on, after about 5 seconds).

    Raises:
        ValueError: an argument is not finite, x < 0, df is not positive, or
            ``prob`` is not strictly between 0 and 1.
    """
    x, df, prob = float(x), float(df), float(prob)
    _check_chi2_args(x, df, 0.0)
    if not 0.0 < prob < 1.0:
        raise ValueError(f"prob must lie strictly between 0 and 1; got {prob!r}")
    if df <= _CHI2_EXPANSION_DF_SHARE * x and (
        x >= _CHI2_EXPANSION_FROM or (prob < 0.5 and x >= _CHI2_EXPANSION_FROM_ABOVE)
    ):
        return _chi2_ncp_expansion(x, df, prob)
    upper = prob > 0.5
    log_target = math.log1p(-prob) if upper else math.log(prob)
    if upper:
        at_zero = noncentral_chi2_logcdf(x, df, 0.0, upper=True)
        if at_zero >= log_target:
            return 0.0

        def g(ncp: float) -> float:
            return _clip(noncentral_chi2_logcdf(x, df, ncp, upper=True)) - log_target

    else:
        at_zero = noncentral_chi2_logcdf(x, df, 0.0)
        if at_zero <= log_target:
            return 0.0

        def g(ncp: float) -> float:
            return log_target - _clip(noncentral_chi2_logcdf(x, df, ncp))

    start = _chi2_ncp_start(x, df, prob)
    if start >= _TIGHT_BRACKET_FROM:
        return _f_ncp_from(g, start)
    high = min(max(1.0, 2.0 * x), _LARGEST)
    for _ in range(_MAX_EXPANSIONS):
        if g(high) > 0.0:
            break
        high *= 2.0
    else:  # pragma: no cover - P(X' <= x) tends to 0 as lambda grows
        raise ArithmeticError(
            f"could not bracket the noncentrality parameter of chi-square = {x!r}"
        )
    root = optimize.brentq(g, 0.0, high, xtol=_XTOL, rtol=4.0 * 2.0**-52, maxiter=500)
    return float(root)


def chi2_ncp_interval(
    x: float,
    df: float,
    *,
    conf_level: float,
    alternative: Alternative,
) -> tuple[float, float]:
    """Confidence interval for the noncentrality parameter of an observed
    chi-square ``x`` on ``df`` degrees of freedom.

    Test inversion (Steiger & Fouladi 1997; Smithson 2003, ch. 5): with alpha = 1 -
    conf_level, the two-sided bounds solve P(chi'^2(df, lambda) <= x) = 1 - alpha / 2
    (lower) and = alpha / 2 (upper), each 0 where no lambda > 0 solves
    (``chi2_ncp_bound``); "greater" gives (the bound at 1 - alpha, +inf) and "less"
    (0, the bound at alpha). The construction ``effectsize`` uses for
    ``cramers_v`` and ``pearsons_c`` (``ci_method = "ncp"``).

    Raises:
        ValueError: x is not finite or negative, df is not positive, conf_level is
            not strictly between 0 and 1, or alternative is not one of
            "two-sided", "less", "greater".
    """
    x, df = float(x), float(df)
    _check_chi2_args(x, df, 0.0)
    level = check_conf_level(conf_level)
    side = check_alternative(alternative)
    alpha = 1.0 - level
    tail = alpha / 2.0 if side == "two-sided" else alpha
    low = 0.0 if side == "less" else chi2_ncp_bound(x, df, prob=1.0 - tail)
    high = math.inf if side == "greater" else chi2_ncp_bound(x, df, prob=tail)
    return low, high
