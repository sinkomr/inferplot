"""The Studentized maximum modulus distribution: distribution function and quantile.

With Z_1..Z_C independent standard normal and, independently, S = sqrt(chi2_df / df),
the Studentized maximum modulus is M = max_i |Z_i| / S (Stoline & Ury 1979,
Technometrics 21: 87-93; Hochberg & Tamhane 1987, Multiple Comparison Procedures,
section 2.1), so

    P(M <= c) = E[(2 Phi(c S) - 1)^C] = int_0^inf erf(c s / sqrt 2)^C f_S(s) ds,
    f_S(s) = 2 (df / 2)^(df / 2) / Gamma(df / 2) s^(df - 1) exp(-df s^2 / 2),

and at df = infinity erf(c / sqrt 2)^C. WRS2's ``lincon`` reads its critical values
from a two-decimal table of this distribution for up to 28 comparisons
(``inferplot.stats.robust.smm_critical_value``); above 28 it estimates them by a
seeded simulation, where inferplot computes the exact quantile with
:func:`smm_quantile` (DECISIONS.md D52).

The integral is a composite Gauss-Legendre rule over panels whose edges are quantiles
of S (from 1e-20 to 1 - 1e-20; the mass outside them is below 2e-20), evaluated with
NumPy in one pass per c, and the quantile is found with Brent's method to 1e-14
relative. Accuracy against 40-digit mpmath references
(``tests/data/smm_reference.json``, ``tools/reference/smm_mpmath.py``): see the
``smm_quantile`` tolerance in ``tests/tolerances.py``.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, special

from inferplot.stats.noncentral import check_conf_level

__all__ = ["smm_cdf", "smm_quantile"]

# Tail probabilities of S at the panel edges: lower tails up to the median, then
# upper tails (each tail is inverted directly, so 1e-20 is not lost to rounding).
_LOWER_TAILS: Final = (1e-20, 1e-12, 1e-8, 1e-5, 1e-3, 0.02, 0.1, 0.3, 0.5)
_UPPER_TAILS: Final = (0.3, 0.1, 0.02, 1e-3, 1e-5, 1e-8, 1e-12, 1e-20)
_NODES: Final = 48
# Above this df the panel edges come from the normal limit of S, 1 + Z / sqrt(2 df),
# and the nodes are held as offsets from 1 (the edges only bound the panels; the
# density itself is exact).
_NORMAL_EDGES_DF: Final = 1e4


def _stirling_remainder(a: float) -> float:
    """lgamma(a) - ((a - 1/2) log a - a + log(2 pi) / 2), without cancellation for
    large a (Abramowitz & Stegun 6.1.41)."""
    if a < 10.0:
        return (
            float(special.gammaln(a))
            - ((a - 0.5) * math.log(a) - a)
            - (0.5 * math.log(2.0 * math.pi))
        )
    inv2 = 1.0 / (a * a)
    return (
        1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0))
    ) / a


def _log_u_minus_u_plus_1(d: NDArray[np.float64]) -> NDArray[np.float64]:
    """log u - u + 1 for u = (1 + d)^2, accurate also near d = 0."""
    x: NDArray[np.float64] = d * (2.0 + d)  # u - 1
    s: NDArray[np.float64] = 1.0 + d
    with np.errstate(divide="ignore"):
        direct: NDArray[np.float64] = 2.0 * np.log(s) - x
        near_one: NDArray[np.float64] = np.log1p(x) - x
    out = near_one.copy()
    far = s < 0.7
    out[far] = direct[far]
    small: NDArray[np.bool_] = np.abs(x) < 0.01
    if bool(small.any()):
        xs = x[small]
        series = np.zeros_like(xs)
        for k in range(12, 1, -1):  # -x^2/2 + x^3/3 - ..., |x| < 0.01: 12 terms
            series = xs * ((-1.0) ** (k + 1) / k + series)
        out[small] = xs * series
    return out


def _edge_offsets(df: float) -> NDArray[np.float64]:
    """Panel edges as offsets from 1 (edge = 1 + offset)."""
    if df > _NORMAL_EDGES_DF:
        z = np.asarray(
            [float(special.ndtri(p)) for p in _LOWER_TAILS]
            + [-float(special.ndtri(q)) for q in _UPPER_TAILS]
        )
        return z / math.sqrt(2.0 * df)
    a = 0.5 * df
    chi2 = [2.0 * float(special.gammaincinv(a, p)) for p in _LOWER_TAILS] + [
        2.0 * float(special.gammainccinv(a, q)) for q in _UPPER_TAILS
    ]
    return np.sqrt(np.asarray(chi2, dtype=np.float64) / df) - 1.0


@lru_cache(maxsize=256)
def _rule(df: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Nodes s and weights w (density times Gauss-Legendre weight) for E[g(S)].

    With a = df / 2 and u = s^2 the density of S is written as
    sqrt(df / pi) exp(-R(a)) u^(-1/2) exp(a (log u - u + 1)), R the Stirling
    remainder, which keeps it accurate at any df.
    """
    nodes, node_weights = np.polynomial.legendre.leggauss(_NODES)
    x = np.asarray(nodes, dtype=np.float64)
    w = np.asarray(node_weights, dtype=np.float64)
    edges: list[float] = _edge_offsets(df).tolist()
    d = np.empty((len(edges) - 1) * _NODES, dtype=np.float64)
    gauss = np.empty_like(d)
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        d[i * _NODES : (i + 1) * _NODES] = 0.5 * (hi + lo) + 0.5 * (hi - lo) * x
        gauss[i * _NODES : (i + 1) * _NODES] = 0.5 * (hi - lo) * w
    log_density: NDArray[np.float64] = (
        0.5 * math.log(df / math.pi)
        - _stirling_remainder(0.5 * df)
        - np.log1p(d)
        + 0.5 * df * _log_u_minus_u_plus_1(d)
    )
    weights: NDArray[np.float64] = np.exp(log_density) * gauss
    return 1.0 + d, weights


def _check(n_comparisons: int, df: float) -> float:
    if isinstance(n_comparisons, bool) or n_comparisons < 1:  # pyright: ignore[reportUnnecessaryComparison]
        raise ValueError(f"n_comparisons must be at least 1; got {n_comparisons!r}")
    value = float(df)
    if not value >= 1.0:
        raise ValueError(f"df must be at least 1 (or inf); got {df!r}")
    return value


def smm_cdf(c: float, n_comparisons: int, df: float) -> float:
    """P(M <= c) for the Studentized maximum modulus with ``n_comparisons`` terms and
    ``df`` degrees of freedom (``math.inf`` for the normal limit); see the module
    docstring.

    Raises:
        ValueError: ``n_comparisons`` below 1 or ``df`` below 1 or NaN.
    """
    value = _check(n_comparisons, df)
    if not c > 0.0:
        return 0.0
    if math.isinf(value):
        return float(special.erf(c / math.sqrt(2.0))) ** n_comparisons
    s, weights = _rule(value)
    inner: NDArray[np.float64] = np.asarray(
        special.erf(c * s / math.sqrt(2.0)), dtype=np.float64
    )
    return math.fsum((weights * inner**n_comparisons).tolist())


def smm_quantile(n_comparisons: int, df: float, *, conf_level: float = 0.95) -> float:
    """The ``conf_level`` quantile of the Studentized maximum modulus: the c with
    P(M <= c) = conf_level for ``n_comparisons`` terms at ``df`` degrees of freedom
    (any real df >= 1, or ``math.inf``).

    Used for lincon's critical values above 28 comparisons, where WRS2 (``lincon``,
    via its ``smmvalv2``) estimates them by a simulation seeded by itself. That
    simulation does not reproduce this quantile: on probe-smm's lincon runs with 36
    and 45 comparisons at alpha 0.05 its values are 3.4% to 6.6% below it (and with
    one comparison it comes within 1% of the one-sided t quantile), so inferplot's
    intervals there are wider than ggstatsplot's (DECISIONS.md D52; needs the
    owner's confirmation).

    Raises:
        ValueError: ``n_comparisons`` below 1, ``df`` below 1 or NaN, or
            ``conf_level`` not strictly between 0 and 1.
    """
    value = _check(n_comparisons, df)
    level = check_conf_level(conf_level)
    # The infinite-df quantile is a lower bound: the quantile falls as df grows.
    # (2 Phi(c) - 1)^C = level: Phi(-c) = (1 - level^(1/C)) / 2, without cancellation.
    tail = -0.5 * math.expm1(math.log(level) / n_comparisons)
    floor = -float(special.ndtri(tail))
    if math.isinf(value):
        return floor
    if smm_cdf(floor, n_comparisons, value) >= level:
        return floor  # df so large that the two agree to rounding
    upper = 2.0 * floor
    while smm_cdf(upper, n_comparisons, value) < level:
        upper *= 2.0
    return float(
        optimize.brentq(
            lambda c: smm_cdf(c, n_comparisons, value) - level,
            floor,
            upper,
            xtol=1e-15,
            rtol=1e-14,
        )
    )
