"""Multiplicity adjustment of p-values, matching R's ``stats::p.adjust``.

Clean-room implementation from the published definitions and R's ``?p.adjust`` help
page (documentation only; R's source was not read):

* Bonferroni: ``min(1, n p_i)``.
* Holm (1979), step-down: with ``p_(1) <= ... <= p_(n)``,
  ``adj_(i) = min(1, max_{j <= i} (n - j + 1) p_(j))``.
* Hochberg (1988), step-up: ``adj_(i) = min_{j >= i} min(1, (n - j + 1) p_(j))``.
* Hommel (1988): the closed testing procedure with Simes' (1986) test of every
  intersection hypothesis. The adjusted p-value of ``H_i`` is the largest Simes
  p-value ``min_k |I| p_(k:I) / k`` over all index sets ``I`` containing ``i``,
  computed with Wright's (1992, Biometrics 48, 1005-1013) algorithm: for a set size
  ``m`` the worst set containing ``i`` is ``i`` together with the ``m - 1`` largest
  other p-values.
* Benjamini & Hochberg (1995), "BH" and its alias "fdr", step-up:
  ``adj_(i) = min_{j >= i} min(1, n p_(j) / j)``.
* Benjamini & Yekutieli (2001, Theorem 1.3), "BY": BH multiplied by
  ``q = sum_{k=1}^{n} 1/k``.
* "none": the p-values unchanged.

``n`` is always the number of p-values passed (R's default ``n = length(p)``). Ties
need no special treatment: every definition gives tied p-values the same adjusted
value whichever order a sort puts them in.

Floating point. Every product ``c * p`` with an integer ``c`` (Bonferroni, Holm,
Hochberg, Hommel's ``m p_(i)``) is a single rounding, so its order of operations
cannot differ from R's. Simes' quotient is computed as ``(m * p) / k``: of four
orderings this is the only one that reproduces all 40 values of a published R
``p.adjust(p, "hommel")`` output at its printed 16 significant digits (statsmodels'
test_multi.py, test_hommel). BH multiplies by ``n / i`` first, ``(n / i) * p``:
for the largest p-value the factor is exactly 1, so the adjusted value is never
below the raw one (``(n * p) / n`` rounds below ``p`` in about 6% of cases). BY uses
``((q * n) / i) * p``, whose factor is never below BH's, so BY >= BH holds bit for
bit. The alternatives differ by at most 2 ulp (tests/stats/test_padjust.py,
operation-order check); the oracle probe probe-padjust (R 4.5.3, x86_64) pins the
choice: every method on its 19 vectors without missing values, from 1 to 1008
p-values, is bitwise equal to R except BY on 1008 values, where R's long-double
harmonic sum is the whole difference (``_by``; tests/stats/test_probes.py).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Final, Literal, get_args

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

PAdjustMethod = Literal[
    "holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"
]
"""R's ``p.adjust.methods``, spelled exactly as R spells them."""

P_ADJUST_METHODS: Final[tuple[PAdjustMethod, ...]] = get_args(PAdjustMethod)


def p_adjust(
    p_values: Sequence[float] | FloatArray, method: PAdjustMethod = "holm"
) -> tuple[float, ...]:
    """Adjust ``p_values`` for multiple comparisons as R's ``stats::p.adjust`` does.

    Must match ``stats::p.adjust(p, method = method)`` with the default
    ``n = length(p)``.

    Args:
        p_values: One-dimensional p-values, each finite and in [0, 1]. The order is
            kept: element ``i`` of the result belongs to element ``i`` here.
        method: One of :data:`P_ADJUST_METHODS`, exactly as R spells it ("BH" and
            "BY" upper case; "fdr" is BH). R's abbreviations are not accepted.

    Returns:
        The adjusted p-values as Python floats, in the input order.

    Raises:
        ValueError: ``method`` is not one of R's names, ``p_values`` is not
            one-dimensional, or a value is missing, infinite or outside [0, 1].
    """
    if method not in P_ADJUST_METHODS:
        raise ValueError(
            f"p_adjust_method must be one of {', '.join(P_ADJUST_METHODS)}; "
            f"got {method!r}"
        )
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p-values must be one-dimensional; got shape {p.shape}")
    valid = (p >= 0.0) & (p <= 1.0)  # false for NaN too
    if not bool(valid.all()):
        bad = p[~valid]
        raise ValueError(f"p-values must lie in [0, 1]; got {bad.tolist()}")
    n = p.size
    if n == 0:
        return ()
    if method == "none":
        return tuple(float(v) for v in p)
    if method == "bonferroni":
        return tuple(float(v) for v in np.minimum(1.0, n * p))

    order = p.argsort(kind="stable")
    sorted_p = p[order]
    adjusted_sorted = _ADJUST_SORTED[method](sorted_p)
    adjusted = np.empty(n, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return tuple(float(v) for v in adjusted)


def _holm(sorted_p: FloatArray) -> FloatArray:
    """Holm (1979): running maximum of ``(n - i + 1) p_(i)``, capped at 1."""
    n = sorted_p.size
    multiplier = np.arange(n, 0, -1, dtype=np.float64)
    return np.minimum(1.0, _running(multiplier * sorted_p, max))


def _step_up(scaled: FloatArray) -> FloatArray:
    """Running minimum from the largest p-value down, capped at 1."""
    return np.minimum(1.0, _running(scaled[::-1], min)[::-1])


def _running(values: FloatArray, pick: Callable[[float, float], float]) -> FloatArray:
    """Running ``max`` or ``min`` (exact, so identical to NumPy's ``accumulate``).

    Written with ``itertools.accumulate`` because NumPy's ufunc ``accumulate`` is
    partially unknown to pyright's strict mode.
    """
    as_floats: list[float] = values.tolist()
    return np.fromiter(
        itertools.accumulate(as_floats, pick), dtype=np.float64, count=values.size
    )


def _hochberg(sorted_p: FloatArray) -> FloatArray:
    """Hochberg (1988): step-up with the Holm multipliers ``n - i + 1``."""
    n = sorted_p.size
    return _step_up(np.arange(n, 0, -1, dtype=np.float64) * sorted_p)


def _bh(sorted_p: FloatArray) -> FloatArray:
    """Benjamini & Hochberg (1995): step-up of ``n p_(i) / i``, as ``(n / i) p``."""
    n = sorted_p.size
    rank = np.arange(1, n + 1, dtype=np.float64)
    return _step_up((float(n) / rank) * sorted_p)


def _by(sorted_p: FloatArray) -> FloatArray:
    """Benjamini & Yekutieli (2001): BH times ``q = sum_{k <= n} 1/k``.

    ``q`` is summed with ``math.fsum`` (correctly rounded). R's ``sum`` accumulates
    in long double where the platform has one (``?sum``); on x86_64, where the
    oracle runs, that is a 64-bit significand, which gives the same double as
    ``fsum`` for every n < 1008 and differs by 1 ulp at n = 1008, 1138, 1259, ...
    (tests/stats/test_padjust.py, harmonic-sum check). A plain double loop differs
    from n = 4 on. The oracle probe confirms it: R's sum equals ``fsum`` at
    n = 1007 and is 1 ulp above it at n = 1008, so there all 1008 BY values are
    1 or 2 ulp below R's (928 and 80 of them), and with R's sum substituted they
    are bitwise equal. inferplot keeps the correctly rounded sum.
    """
    n = sorted_p.size
    q = math.fsum(1.0 / k for k in range(1, n + 1))
    rank = np.arange(1, n + 1, dtype=np.float64)
    return _step_up(((q * float(n)) / rank) * sorted_p)


def _hommel(sorted_p: FloatArray) -> FloatArray:
    """Hommel (1988) by Wright's (1992) algorithm, in sorted order.

    For each set size m = n, n - 1, ..., 2, let ``c`` be the Simes p-value of the m
    largest p-values, ``min_{k=1..m} m p_(n-m+k) / k``. Each of those m hypotheses
    has a set of size m with Simes p-value ``c``; each smaller one, ``i <= n - m``,
    has ``{i}`` with the m - 1 largest, whose Simes p-value is ``min(c, m p_(i))``
    (the term ``m p_(n-m+1) / 1`` in ``c`` is at least ``m p_(i)``). The adjusted
    p-value is the maximum over m, starting from ``p_(i)`` itself (m = 1). It never
    exceeds ``p_(n)`` and is non-decreasing in ``p``, so it needs neither a cap nor
    a monotone pass.
    """
    n = sorted_p.size
    adjusted = sorted_p.copy()
    for m in range(n, 1, -1):
        rest = n - m
        top = sorted_p[rest:]
        k = np.arange(1, m + 1, dtype=np.float64)
        c_min = float(np.min((float(m) * top) / k))
        np.maximum(adjusted[rest:], c_min, out=adjusted[rest:])
        if rest:
            np.maximum(
                adjusted[:rest],
                np.minimum(c_min, float(m) * sorted_p[:rest]),
                out=adjusted[:rest],
            )
    return adjusted


_ADJUST_SORTED: Final = {
    "holm": _holm,
    "hochberg": _hochberg,
    "hommel": _hommel,
    "BH": _bh,
    "fdr": _bh,
    "BY": _by,
}
