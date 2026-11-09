"""Tolerance registry for comparisons with the R oracle (SPEC.md section 10).

Every oracle comparison names one entry here. Change an entry only with a measured
reason written into it, and never loosen one to get a pass (CLAUDE.md rule 2): find
out why the numbers differ instead.

Closeness is ``|actual - expected| <= atol * atol_scale + rtol * |expected|``, with
the oracle value as ``expected``; ``atol_scale`` is 1 unless an entry's reason says
how to scale its absolute part. Equal infinities and two NaNs count as equal.
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
    and the center is the 20,000-resample R reference. The band is never narrower
    than the ``closed_form`` tolerance: a quantity whose replicate SD is 0 does not
    depend on the resamples (AKP and rank epsilon squared point estimates, an upper
    bound fixed at 1), and the fixture stores it to 15 significant digits.
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
            atol=3e-9,
            reason=(
                "Games-Howell p against ggstatsplot. inferplot computes the "
                "studentized range tail accurately (DECISIONS.md D31); ggstatsplot's "
                "adjusted p is exactly p.adjust of gamesHowellTest's raw p (probe-"
                "ptukey; within closed_form in every completed case with a "
                "pairwise table), so every "
                "difference is R's ptukey error times the adjustment. Raw p over "
                "the 67 distinct pairs (|q|, Welch df, k) of the 29 completed cases "
                "in probe-ptukey (70 with edge_n2, where R's ptukey is NaN): at "
                "most 2.7e-10 relative for p >= 0.01 (edge_unbalanced a-c) and "
                "3.5e-7 for p >= 1e-4 (InsectSprays E-F), but up to 4.7e-10 "
                "absolute in the far tail (iris setosa-versicolor: R 4.7e-10 where "
                "the accurate tail is 7.5e-15, printed 1.41e-9 after Holm). "
                "Tightest of the 106 "
                "adjusted pairs in the 28 completed cases with a pairwise table: "
                "InsectSprays B-E, R's raw p 2.0e-10 low at 3.7e-6, times Holm's "
                "14 = 2.79e-9 against the 3.05e-9 allowed at p 5.1e-5 (92%); next "
                "edge_k5 a-e 2.47e-9 (82%). The same entry serves the SciPy "
                "cross-check, whose own far-tail error (order 1e-14) is far inside "
                "it."
            ),
        ),
        "wrs2_p": Tolerance(
            rtol=1e-10,
            atol=2.0**-53,
            reason=(
                "p-values WRS2 computes as 1 - pt or 1 - pf (t1way and lincon in the "
                "fixtures; yuen presumably too): the one-sided tail rounded to the "
                "spacing of doubles just below 1 (2**-53). Rounding SciPy's tail the "
                "same way, 1 - (1 - sf), matches every fixture to 2.4e-11 relative "
                "except iris t1way, where R's tail lands on the neighbouring grid "
                "point: one step, 1.1e-16 absolute (1.6e-6 relative). The absolute "
                "part is that one step of the one-sided tail; pass atol_scale = the "
                "factor applied to the tail afterwards (2 for a two-sided p, times "
                "the number of pairs for a Holm-adjusted p)."
            ),
        ),
        "noncentral_ci": Tolerance(
            rtol=1e-6,
            atol=4e-8,
            reason=(
                "CI bounds found by inverting a noncentral t or F distribution in "
                "its noncentrality parameter: R's root finding is accurate to about "
                "1e-8 absolute, whatever the size of the bound. Measured with "
                "inferplot.stats (own noncentral t and F functions, accurate to "
                "2.5e-12 relative; tests/stats/test_noncentral.py) on the parametric "
                "fixtures among the 72 ggstatsplot 1.1.4 cases that completed: "
                "Hedges' g, 18 finite bounds "
                "in 10 fixtures, max abs 1.9e-8 (mtcars-am-drat upper), max rel "
                "7.5e-7 (ToothGrowth lower, -0.0217, 26% of the allowance); partial "
                "omega squared lower bound, 13 nonzero bounds in 13 fixtures, max "
                "abs 1.5e-8 (edge_k5), max rel 1.0e-7 (edge_empty_level, 0.0503); "
                "the lower bound 0 of the other 15 fixtures, every upper bound of "
                "1 and the two infinite one-sided bounds are exact. On probe-omega's "
                "577 one-sided rows inferplot is within it of effectsize wherever "
                "effectsize is within it of an accurate root (574 rows; worst "
                "1.7e-7 at 0.331) and of the accurate root in the 3 rows where "
                "effectsize's search stopped early (DECISIONS.md D33). atol is about "
                "twice the largest absolute error in the fixtures, so a bound near "
                "0 passes on its absolute error, and every fixture bound passes on "
                "atol alone; rtol stays at SPEC.md section 10's 1e-6, which the "
                "probe's worst agreeing row needs (F 31.6, df 5 and 5.5, 99%: "
                "effectsize 1.65e-7 from the accurate root, inferplot 5.8e-9)."
            ),
        ),
        "studentized_range_tail": Tolerance(
            rtol=4e-13,
            atol=0.0,
            reason=(
                "inferplot.stats.srange_sf against the high-precision mpmath "
                "references in tests/data/studentized_range_reference.json (792 "
                "points: 384 on a grid k = 3..50, df = 1..1000; 72 at df = Inf; "
                "224 exact k = 2 values, df = 1..1e12 and Inf; the 31 Games-Howell "
                "pairs of the Phase 0 fixtures, max 2.9e-15; the 78 pairs (42 "
                "distinct points) of the later completed fixtures, max 5.2e-15; the "
                "3 pairs of edge_n2, Welch df 1.06 to 1.42, 1.2e-16). Measured max "
                "relative error 7.3e-15 for tails >= 1e-15 (522 points), 4.5e-14 "
                "down to 1e-100 (210) and 1.4e-13 at 1e-300 (60; k = 2, df = Inf), "
                "the same at SciPy 1.11.1 and 1.18.1: the rounding "
                "of exp(log P), about |log P| x 2.2e-16, at most 1.6e-13 for any "
                "normal double. 4e-13 leaves 2.8 times headroom."
            ),
        ),
        "noncentral_cdf": Tolerance(
            rtol=2e-11,
            atol=0.0,
            reason=(
                "inferplot's noncentral t and F distribution functions (either tail) "
                "against the mpmath references in tests/data/noncentral_reference.json "
                "(329 t points, t = -50..500, df = 1..1e4; 192 F points), for tail "
                "probabilities >= 1e-20. Measured max relative error: t 1.6e-14 at "
                "SciPy 1.11 and 1.18; F 1.8e-13 at SciPy 1.18 and 2.5e-12 at SciPy "
                "1.11, whose incomplete beta function is less accurate at df2 = 5000. "
                "2e-11 leaves 8 times headroom over the worst."
            ),
        ),
        "ttest_bayes_factor": Tolerance(
            rtol=1e-8,
            atol=0.0,
            reason=(
                "JZS Bayes factors of t tests (BayesFactor::ttestBF, one-dimensional "
                "quadrature), measured on two-sample values only: tools/reference "
                "reproduces all 35 distinct ones in the 45 ggstatsplot 1.1.4 "
                "fixtures to at most 1.5e-9 relative (iris setosa-virginica, BF10 "
                "2.2e6), and that residual is BayesFactor's own integration error. "
                "A prior scale of sqrt(2)/2 instead of 0.707 moves them by 3.7e-6 or "
                "more, and Welch's t instead of Student's moves every unbalanced "
                "pair by 1.5e-4 or more (bugs_long, 92 against 91), so both fail. "
                "Measure paired and one-sample values when their fixtures arrive; "
                "other Bayes factor families need their own entry."
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
    "scientific notation, same digits (SPEC.md section 7). Degrees of freedom are "
    "numbers, not strings, in ggstatsplot's expression: rounded to `digits` decimals "
    "when fractional (0 otherwise), then shown without trailing zeros (97.40 reads "
    "97.4)."
)


def _equal_special(actual: float, expected: float) -> bool | None:
    """Decide NaN and infinity cases; ``None`` means both are finite."""
    if math.isnan(actual) or math.isnan(expected):
        return math.isnan(actual) and math.isnan(expected)
    if math.isinf(actual) or math.isinf(expected):
        return actual == expected
    return None


def is_close(
    kind: str, actual: float, expected: float, *, atol_scale: float = 1.0
) -> bool:
    """Whether ``actual`` matches the oracle value ``expected`` under ``kind``."""
    if not atol_scale >= 1.0:
        raise ValueError(f"atol_scale must be at least 1, got {atol_scale!r}")
    tol = TOLERANCES[kind]
    special = _equal_special(actual, expected)
    if special is not None:
        return special
    return abs(actual - expected) <= tol.atol * atol_scale + tol.rtol * abs(expected)


def assert_close(
    kind: str,
    actual: float,
    expected: float,
    *,
    what: str,
    atol_scale: float = 1.0,
) -> None:
    """Assert closeness under the registered tolerance ``kind``."""
    if is_close(kind, actual, expected, atol_scale=atol_scale):
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
        f"tolerance {kind}: rtol={tol.rtol:g}, atol={tol.atol:g} x {atol_scale:g})"
    )


def monte_carlo_limit(reference: float, mc_sd: float) -> float:
    """Largest accepted difference from ``reference`` for a bootstrap quantity.

    ``BOOTSTRAP.n_sd`` Monte Carlo SDs, but never less than the ``closed_form``
    tolerance, which applies alone when the replicate SD is 0 (see
    ``MonteCarloTolerance``).
    """
    if not (math.isfinite(mc_sd) and mc_sd >= 0.0):
        raise ValueError(f"Monte Carlo SD must be finite and >= 0, got {mc_sd!r}")
    closed = TOLERANCES["closed_form"]
    return max(BOOTSTRAP.n_sd * mc_sd, closed.atol + closed.rtol * abs(reference))


def assert_within_monte_carlo(
    actual: float, reference: float, mc_sd: float, *, what: str
) -> None:
    """Assert a bootstrap quantity is within ``monte_carlo_limit`` of the reference."""
    special = _equal_special(actual, reference)
    if special is not None:
        if special:
            return
        raise AssertionError(f"{what}: got {actual!r}, reference {reference!r}")
    limit = monte_carlo_limit(reference, mc_sd)
    if abs(actual - reference) <= limit:
        return
    raise AssertionError(
        f"{what}: got {actual!r}, 20,000-resample reference {reference!r}; "
        f"difference {abs(actual - reference):.3g} exceeds the limit {limit:.3g} "
        f"({BOOTSTRAP.n_sd:g} x Monte Carlo SD {mc_sd:.3g}, at least closed_form)"
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
