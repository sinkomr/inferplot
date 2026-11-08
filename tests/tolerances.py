"""Tolerance registry for comparisons with the R oracle (SPEC.md section 10).

Every oracle comparison names one entry here. Change an entry only with a measured
reason written into it, and never loosen one to get a pass (CLAUDE.md rule 2): find
out why the numbers differ instead.

Closeness is ``|actual - expected| <= atol + rtol * |expected|``, with the oracle
value as ``expected``. Equal infinities and two NaNs count as equal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Tolerance:
    """Relative-plus-absolute tolerance for one kind of quantity."""

    rtol: float
    atol: float
    reason: str


@dataclass(frozen=True)
class MonteCarloTolerance:
    """Bootstrap quantities: within ``n_sd`` Monte Carlo standard deviations.

    The standard deviation is the spread of a 2,000-resample bound across the
    oracle's replicate R runs (fixture field ``bootstrap_reference.replicates``),
    and the center is the 20,000-resample R reference.
    """

    n_sd: float
    reason: str


@dataclass(frozen=True)
class GridTolerance:
    """Quantities read off a grid: within ``steps`` grid steps."""

    steps: float
    reason: str


TOLERANCES: Mapping[str, Tolerance] = MappingProxyType(
    {
        "closed_form": Tolerance(
            rtol=1e-10,
            atol=0.0,
            reason=(
                "Closed-form statistics, degrees of freedom, and p-values from the "
                "t, F, chi-square and normal distributions: R and SciPy are both "
                "exact to double precision."
            ),
        ),
        "games_howell_p": Tolerance(
            rtol=1e-6,
            atol=0.0,
            reason=(
                "R's ptukey and SciPy's studentized_range differ by up to 2.8e-7 "
                "relative (measured on the ggstatsplot 1.1.4 PlantGrowth and mtcars "
                "fixtures; 2.6e-7 on plotsalot's m3-between). Games-Howell p only."
            ),
        ),
        "noncentral_ci": Tolerance(
            rtol=1e-6,
            atol=0.0,
            reason=(
                "Starting value from SPEC.md section 10 for CIs found by inverting "
                "noncentral t, F or chi-square distributions (R root-finding "
                "precision). Replace with a measured value when the first such CI "
                "is implemented."
            ),
        ),
        "bayes_factor": Tolerance(
            rtol=1e-4,
            atol=0.0,
            reason=(
                "BayesFactor computes Bayes factors by numerical integration; its "
                "reported relative error is of this order."
            ),
        ),
    }
)

BOOTSTRAP = MonteCarloTolerance(
    n_sd=4.0,
    reason=(
        "Bootstrap CIs and p-values are random. Four Monte Carlo standard "
        "deviations of a 2,000-resample bound, measured from the oracle's "
        "replicate runs, around the 20,000-resample R reference."
    ),
)

MAP_CENTRALITY = GridTolerance(
    steps=1.0,
    reason="The MAP estimate is the peak of a kernel density evaluated on a grid.",
)

SUBTITLE_TEXT = (
    "Subtitle numbers are compared as exact strings: same rounding, same switch to "
    "scientific notation, same digits (SPEC.md section 7)."
)


def _equal_special(actual: float, expected: float) -> bool | None:
    """Decide NaN and infinity cases; ``None`` means both are finite."""
    if math.isnan(actual) or math.isnan(expected):
        return math.isnan(actual) and math.isnan(expected)
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    return None


def is_close(kind: str, actual: float, expected: float) -> bool:
    """Whether ``actual`` matches the oracle value ``expected`` under ``kind``."""
    tol = TOLERANCES[kind]
    special = _equal_special(actual, expected)
    if special is not None:
        return special
    return abs(actual - expected) <= tol.atol + tol.rtol * abs(expected)


def assert_close(kind: str, actual: float, expected: float, *, what: str) -> None:
    """Assert closeness under the registered tolerance ``kind``."""
    if is_close(kind, actual, expected):
        return
    tol = TOLERANCES[kind]
    diff = abs(actual - expected)
    rel = (
        diff / abs(expected)
        if expected not in (0.0,) and math.isfinite(diff)
        else math.inf
    )
    raise AssertionError(
        f"{what}: got {actual!r}, oracle {expected!r} "
        f"(abs diff {diff:.3g}, rel diff {rel:.3g}; "
        f"tolerance {kind}: rtol={tol.rtol:g}, atol={tol.atol:g})"
    )


def assert_within_monte_carlo(
    actual: float, reference: float, mc_sd: float, *, what: str
) -> None:
    """Assert a bootstrap quantity is within ``BOOTSTRAP.n_sd`` Monte Carlo SDs."""
    special = _equal_special(actual, reference)
    if special is not None:
        if special:
            return
        raise AssertionError(f"{what}: got {actual!r}, reference {reference!r}")
    limit = BOOTSTRAP.n_sd * mc_sd
    if abs(actual - reference) <= limit:
        return
    raise AssertionError(
        f"{what}: got {actual!r}, 20,000-resample reference {reference!r}; "
        f"difference {abs(actual - reference):.3g} exceeds {BOOTSTRAP.n_sd:g} x "
        f"Monte Carlo SD {mc_sd:.3g}"
    )


def assert_within_grid(
    actual: float, expected: float, step: float, *, what: str
) -> None:
    """Assert a grid-based quantity is within ``MAP_CENTRALITY.steps`` grid steps."""
    if abs(actual - expected) <= MAP_CENTRALITY.steps * step:
        return
    raise AssertionError(
        f"{what}: got {actual!r}, oracle {expected!r}; more than "
        f"{MAP_CENTRALITY.steps:g} grid step(s) of {step:.3g} apart"
    )


def assert_same_text(actual: str, expected: str, *, what: str) -> None:
    """Assert exact string equality for formatted numbers (subtitle parity)."""
    if actual != expected:
        raise AssertionError(f"{what}: got {actual!r}, oracle {expected!r}")
