"""JZS Bayes factors for t statistics (Rouder, Speckman, Sun, Morey & Iverson 2009).

Rouder et al. (2009, Psychonomic Bulletin & Review 16: 225-237, eq. 1 with the
Cauchy prior of scale r written as a normal mixture over g with an
inverse-gamma(1/2, 1/2) mixing density) give, for a t statistic with df nu and
effective sample size N,

    BF10 = integral_0^inf (1 + N g r^2)^(-1/2)
           (1 + t^2 / ((1 + N g r^2) nu))^(-(nu + 1)/2)
           (2 pi)^(-1/2) g^(-3/2) e^(-1/(2g)) dg  /  (1 + t^2 / nu)^(-(nu + 1)/2).

Here the integral is taken over u = log g, where the integrand is smooth, positive
and decays at least exponentially at both ends (like e^-u on the right and like
exp(-e^-u / 2) on the left), so the trapezoidal rule on the whole line converges
geometrically (Trefethen & Weideman 2014, SIAM Review 56: 385-458). The ratio to the
null likelihood is taken inside the integral and every node is summed in log space,
so log BF10 stays finite when both likelihoods underflow or BF10 overflows. The step
starts at min(0.1, 0.2 / sqrt(curvature at the peak)) and is halved until two
successive sums agree to 1e-14 in the log.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from inferplot.stats.descriptive import SampleLike, as_sample
from inferplot.stats.parametric import student_t

FloatArray = NDArray[np.float64]

__all__ = ["jzs_bf10_two_sample", "jzs_log_bf10"]

_LOG_SQRT_2PI: Final = 0.5 * math.log(2.0 * math.pi)
# Nodes are added until the log integrand is this far below its peak (e^-60 = 9e-27).
_DROP: Final = 60.0
_BLOCK: Final = 256
_CONVERGED: Final = 1e-14
_MAX_HALVINGS: Final = 12
# Halvings done before convergence is checked (a test raises it to exercise the
# refinement).
_MIN_HALVINGS = 1


def _log_integrand(
    u: FloatArray, t: float, log_nr2: float, df: float, log_null: float
) -> FloatArray:
    """log of the BF10 integrand in u = log g (including dg = g du)."""
    log_scale = np.logaddexp(0.0, log_nr2 + u)  # log(1 + N r^2 g)
    if t == 0.0:
        log_alt = -0.5 * log_scale
    else:
        log_ratio = 2.0 * math.log(abs(t)) - math.log(df) - log_scale
        with np.errstate(over="ignore"):
            log1p_ratio = np.log1p(np.exp(log_ratio))
        # Where t^2 / ((1 + N r^2 g) df) overflows, log1p(ratio) is log(ratio) to
        # double precision (ratio > 1.8e308).
        overflowed = np.isinf(log1p_ratio)
        log1p_ratio[overflowed] = log_ratio[overflowed]
        log_alt = -0.5 * log_scale - 0.5 * (df + 1.0) * log1p_ratio
    with np.errstate(over="ignore"):
        log_prior = -_LOG_SQRT_2PI - 0.5 * u - 0.5 * np.exp(-u)
    return log_alt - log_null + log_prior


def _logsumexp(a: FloatArray) -> float:
    peak = float(np.max(a))
    return peak + math.log(float(np.sum(np.exp(a - peak))))


def _join(a: FloatArray, b: FloatArray) -> FloatArray:
    out = np.empty(a.size + b.size)
    out[: a.size] = a
    out[a.size :] = b
    return out


def _trapezoid(f: FloatArray, h: float) -> float:
    """log of h * sum(exp(f)) (both ends are negligible, so no end corrections)."""
    return _logsumexp(f) + math.log(h)


def _log1p_t2_over_df(t: float, df: float) -> float:
    """log(1 + t^2 / df), also where t^2 / df overflows (then 2 log|t| - log df,
    since df / t^2 is below the rounding of the logarithm)."""
    ratio = t * t / df
    if math.isinf(ratio):
        return 2.0 * math.log(abs(t)) - math.log(df)
    return math.log1p(ratio)


def jzs_log_bf10(t: float, n_eff: float, df: float, *, prior_scale: float) -> float:
    """Natural log of the JZS Bayes factor BF10 of a t statistic.

    Rouder et al. (2009), eq. 1, with a Cauchy prior of scale ``prior_scale`` on the
    standardized effect; two-sided. One sample or paired: n_eff = n, df = n - 1; two
    independent samples: n_eff = n_a n_b / (n_a + n_b), df = n_a + n_b - 2. The
    quantity ``BayesFactor::ttestBF(..., rscale = prior_scale)`` reports (R; its
    own quadrature error is about 1e-9 relative, tolerance ``ttest_bayes_factor``).

    Any finite t works: where t^2 / df overflows (|t| above about 1.3e154 sqrt(df))
    the logarithms are taken without forming it. For large |t| the result follows
    the closed-form limit log BF10 = (df - 1) / 2 log(t^2 / df) + log(2 / (df - 1))
    + log(n_eff r^2) / 2 - log(2 pi) / 2 (df > 1), within 1e-13 relative from
    |t| = 1e100 to the largest double (tests/stats/test_bayes.py).

    Raises:
        ValueError: t is not finite or n_eff, df or prior_scale is not a finite
            positive number.
    """
    t, n_eff, df, r = float(t), float(n_eff), float(df), float(prior_scale)
    if not math.isfinite(t):
        raise ValueError(f"t must be finite; got {t!r}")
    for name, value in (("n_eff", n_eff), ("df", df), ("prior_scale", r)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite number > 0; got {value!r}")
    log_nr2 = math.log(n_eff) + 2.0 * math.log(r)
    log_null = -0.5 * (df + 1.0) * _log1p_t2_over_df(t, df) if t != 0.0 else 0.0

    def f(u: FloatArray) -> FloatArray:
        return _log_integrand(u, t, log_nr2, df, log_null)

    def f1(u: float) -> float:
        return float(f(np.array([u]))[0])

    # Locate the peak: a coarse scan wide enough for any finite t (the peak lies
    # near u = log(t^2 / (df n_eff r^2))), then a bounded refinement.
    reach = 2.0 * math.log1p(abs(t)) + abs(log_nr2) + 40.0
    coarse = np.arange(-30.0, reach, 0.25)
    values = f(coarse)
    best = float(coarse[int(values.argmax())])
    found = optimize.minimize_scalar(
        lambda u: -f1(u),
        bounds=(best - 0.25, best + 0.25),
        method="bounded",
        options={"xatol": 1e-10},
    )
    centre = float(found.x)
    peak = f1(centre)
    step = 1e-3
    curvature = -(f1(centre + step) - 2.0 * peak + f1(centre - step)) / step**2
    h = min(0.1, 0.2 / math.sqrt(curvature)) if curvature > 0.0 else 0.1

    # Nodes centre + h * j, extended in blocks until both ends are _DROP below the peak.
    lo, hi = -_BLOCK, _BLOCK
    while f1(centre + h * lo) > peak - _DROP:
        lo -= _BLOCK
    while f1(centre + h * hi) > peak - _DROP:
        hi += _BLOCK
    start = centre + h * lo
    count = hi - lo + 1  # nodes start, start + h, ..., start + (count - 1) h
    terms = f(start + h * np.arange(count, dtype=np.float64))
    estimate = _trapezoid(terms, h)
    for halving in range(1, _MAX_HALVINGS + 1):
        # Halving the step adds the midpoints of the current grid; the old terms
        # are reused (the sum does not depend on their order).
        offsets = np.arange(count - 1, dtype=np.float64) + 0.5
        terms = _join(terms, f(start + h * offsets))
        count = 2 * count - 1
        h *= 0.5
        refined = _trapezoid(terms, h)
        converged = abs(refined - estimate) <= _CONVERGED * max(1.0, abs(refined))
        if converged and halving >= _MIN_HALVINGS:
            return refined
        estimate = refined
    raise ArithmeticError(  # pragma: no cover - the trapezoid sums converge fast
        f"JZS Bayes factor quadrature did not converge for t={t!r}, df={df!r}"
    )


def jzs_bf10_two_sample(
    a: SampleLike, b: SampleLike, *, prior_scale: float
) -> tuple[float, float]:
    """JZS Bayes factor of two independent samples: ``(bf10, log_bf01)``.

    Student's pooled t with n_eff = n_a n_b / (n_a + n_b) and df = n_a + n_b - 2 in
    ``jzs_log_bf10`` (Rouder et al. 2009); two-sided whatever the frequentist
    ``alternative`` is. Matches R's ``BayesFactor::ttestBF(formula = y ~ x,
    rscale = prior_scale)`` as ggstatsplot 1.1.4 calls it with ``bf.prior``
    (0.707 by default, not sqrt(2)/2). ``bf10`` is ``inf`` when it overflows a
    double; ``log_bf01`` = -log(BF10) stays finite.

    Raises:
        ValueError: a sample has fewer than two values or a non-finite value, both
            are constant, one is constant and the other varies by less than double
            precision resolves next to it (``student_t``), or prior_scale is not a
            finite positive number.
    """
    x = as_sample(a, name="a", min_size=2)
    y = as_sample(b, name="b", min_size=2)
    student = student_t(x, y)
    n_eff = x.size * y.size / (x.size + y.size)
    log_bf10 = jzs_log_bf10(student.t, n_eff, student.df, prior_scale=prior_scale)
    try:
        bf10 = math.exp(log_bf10)
    except OverflowError:
        bf10 = math.inf
    return bf10, -log_bf10
