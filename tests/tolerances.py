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
    """Quantities read off a grid: grid points at most ``steps`` apart.

    Both values are located on the grid ``origin + i * step``; the oracle's must lie
    on it to within ``on_grid`` steps (the fixtures store 15 significant digits),
    and the two grid indices may differ by at most ``steps``.
    """

    steps: int
    on_grid: float
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
                "p-values WRS2 computes as 1 - pt or 1 - pf (yuen, t1way and lincon: "
                "probe-wrs2p's 420 p-values over |t| from 0 to about 40 at df 4 to "
                "700 all equal R's 1 - cdf bit for bit, and only 27 equal the exact "
                "tail): R rounds the one-sided tail to the spacing of doubles just "
                "below 1 (2**-53), and all 10 robust omnibus p-values lie on that "
                "grid. "
                "inferplot reports the accurate tail (DECISIONS.md D43), which lies "
                "within one step of R's on every fixture. Measured on the 10 robust "
                "fixtures: Yuen 1.17e-16 absolute (0.53 of the two-sided step, "
                "mtcars-am); t1way 7.5e-17 (0.67 step, PlantGrowth), largest "
                "relative 8.0e-7 (iris, p 7.0e-11); lincon after Holm 4.4e-16 (0.67 "
                "of 2 x 3 steps, edge_unbalanced a-b), largest relative 3.2e-6 "
                "(iris setosa-versicolor, p 5.1e-11). Rounding SciPy's tail R's way, "
                "1 - (1 - sf), lands on R's grid point except iris t1way (one step). "
                "The absolute part is one step of the one-sided tail; pass "
                "atol_scale = the factor applied to the tail afterwards (2 for a "
                "two-sided p, times the number of pairs for a Holm-adjusted p)."
            ),
        ),
        "wrs2_xi_constant": Tolerance(
            rtol=3.3e-8,
            atol=0.0,
            reason=(
                "Explanatory effect size xi against WRS2::t1way with equal group "
                "sizes, where it does not depend on the resamples. WRS2 divides the "
                "pooled winsorized variance by its own approximation of the "
                "winsorized standard normal variance, inferplot by the exact value "
                "(DECISIONS.md D42), so the two differ by a constant ratio at each "
                "tr. tr 0.2: WRS2 divides by 0.4120867 for the exact "
                "0.41208672670619, so its xi is 1 - 3.2404e-8 times inferplot's; "
                "measured on the 3 equal-size robust fixtures with three or more "
                "groups and a subtitle, the same ratio to 1e-15 in each, and with "
                "0.4120867 the 20,000-resample references replay exactly. tr 0.1 "
                "(PlantGrowth-tr01, subtitle and reference): 1 + 3.0242e-8 times "
                "inferplot's (WRS2's divisor 0.6786545179 for the exact "
                "0.6786545590). 3.3e-8 leaves 2% headroom over the larger; it is "
                "measured at tr 0.1 and 0.2 only, and the parity tests refuse an "
                "equal-size fixture at another tr. With unequal sizes the estimate "
                "is random and compared under BOOTSTRAP."
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
                "2e-11 leaves 8 times headroom over the worst. The same bound holds "
                "for the larger arguments added with the beta factors in log space "
                "(SciPy 1.18; all within it at the lowest-direct SciPy too): 63 "
                "values of noncentral_f_large (Poisson means 1e5 to "
                "1e12, references by the chi'^2 x chi^2 integral), max 5.0e-14 "
                "(4.96e-10 before, from the large-a expansion); 45 values of "
                "noncentral_f_df1_two (df2 = 1e6 and 1e8, exact sums), max 3.1e-13 "
                "(9.2e-11 before, SciPy's betainc at b = 5e7). Below 1e-20 the rtol "
                "applies to log P: 15 values of noncentral_f_deep down to log P = "
                "-2000, max 2.3e-15 (before, betainc's underflow gave log P 12 too "
                "small at -677.7, and -inf further out)."
            ),
        ),
        "ttest_bayes_factor": Tolerance(
            rtol=1e-8,
            atol=0.0,
            reason=(
                "JZS Bayes factors of t tests (BayesFactor::ttestBF, one-dimensional "
                "quadrature), measured on two-sample values only: tools/reference "
                "reproduces 75 of the 77 distinct ones in the 158 completed "
                "ggstatsplot 1.1.4 fixtures to at most 2.3e-9 relative "
                "(edge_large_n a-c, BF10 5.1e5; iris setosa-virginica Sepal.Width, "
                "1.5e-9), and that residual is BayesFactor's own integration error. "
                "The other two, iris Petal.Length setosa-versicolor and "
                "setosa-virginica (t -39.5 and -50.0, BF10 1e58 and 3e67), BayesFactor "
                "misses by 1.2e-3 and 6.1e-5; they are compared with 50-digit values "
                "(tests/oracle_fixtures.py ACCURATE_BAYES_FACTORS, DECISIONS.md D31 "
                "pattern). "
                "A prior scale of sqrt(2)/2 instead of 0.707 moves them by 3.7e-6 or "
                "more, and Welch's t instead of Student's moves every unbalanced "
                "pair by 1.5e-4 or more (bugs_long, 92 against 91), so both fail. "
                "Paired: every paired Bayes factor in the ggwithinstats fixtures "
                "(sleep, bugs_long, edge_within*) is within 9.3e-10 of "
                "tools/reference (tests/test_reference_vs_oracle.py); BayesFactor "
                "itself misses large |t| (probe-pairedbf: within 2.6e-6 up to "
                "|t| 12, -0.78% at n 30 and t 50); those misses are pinned in "
                "tests/stats/test_bayes_paired.py and inferplot is within 1.1e-13 "
                "of 40-digit references there (D61). Measure one-sample values when "
                "their fixtures arrive; other Bayes factor families need their own "
                "entry."
            ),
        ),
        "correlation_bayes_factor": Tolerance(
            rtol=5e-12,
            atol=0.0,
            reason=(
                "Bayes factors of a Pearson correlation (BayesFactor::correlationBF, "
                "the stretched beta prior of Ly, Verhagen & Wagenmakers 2016, which "
                "it evaluates in closed form: its error estimate is 0). inferplot's "
                "correlation_bf10 is within 1.8e-12 of 40-digit mpmath values of "
                "the hypergeometric form on a grid of n 3 to 2,000 and |r| up to "
                "1 - 1e-7 (tests/data/corrbf_reference.json), which is the rounding "
                "of a log BF10 of 15,384 (1.2e-16 relative in the log). Measured "
                "against R: the 26 finite fixture values (9 distinct; Bayes "
                "subtitles and parametric captions, prior 0.707 and 0.8) at most "
                "3.07e-12 "
                "(edge_corr_large_n, n 2,000, BF10 1.37e172), then 1.19e-12 (iris, "
                "n 150), all others within 5.3e-14; probe-corr's 28 finite values "
                "at R's own r at most 1.25e-12 (iris), so the residual is "
                "BayesFactor's own floating point, growing with n. 5e-12 leaves "
                "1.6 times headroom over the largest; ttest_bayes_factor's 1e-8 "
                "would hide a wrong prior parameterization only above 1e-8, and "
                "kappa = 0.707 against sqrt(2)/2 moves these values by 1e-6 or more."
            ),
        ),
        "akp_constant": Tolerance(
            rtol=5e-7,
            atol=0.0,
            reason=(
                "WRS2::akp.effect's scaling constant against the exact SD of a "
                "standard normal winsorized at tr (DECISIONS.md D42): probe-akp "
                "measures it as AKP / ((trimmed mean 1 - trimmed mean 2) / "
                "winsorized SD of group 1), the same to 4 ulps on every dataset at "
                "a given tr, so WRS2 evaluates it numerically. Its relative offset "
                "from the exact value is irregular in tr: 1.2e-9 at 0.2 "
                "(ggstatsplot's default), 4.6e-9 at 0.1, at most 3.4e-8 for tr up "
                "to 0.25 and 9.7e-9 at 0.3, rising to 4.2e-7 at 0.49 (tr 0.01 to "
                "0.49 by 0.01). The bound is that maximum, rounded up. The AKP "
                "parity tests do not use this bound: they scale inferplot's "
                "estimate by the probe's constant at the case's tr over the exact "
                "one and compare under closed_form (every AKP estimate of the 4 "
                "two-group robust fixtures with a subtitle, at tr 0.1 and 0.2, and "
                "its 20,000-resample reference, agrees to 2.2e-16 relative)."
            ),
        ),
        "smm_quantile": Tolerance(
            rtol=3e-14,
            atol=0.0,
            reason=(
                "inferplot.stats.smm_quantile, the Studentized maximum modulus "
                "quantile, against the 30-digit mpmath references in "
                "tests/data/smm_reference.json (tools/reference/smm_mpmath.py; "
                "148 points: C = 1 to 300 comparisons, df 1 to 1e5 and infinity, "
                "levels 0.5 to 0.99). Measured max relative error 1.1e-14 at finite "
                "df (C 300, df 1) and 2.2e-16 at infinite df. 3e-14 leaves 2.7 "
                "times headroom."
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
    steps=0,
    on_grid=1e-6,
    reason=(
        "The MAP estimate (bayestestR::map_estimate) is the peak of a kernel density "
        "evaluated on 1,024 points from min(x) to max(x), so it is a grid point and "
        "parity means the same grid point. Measured: inferplot.stats.map_estimate "
        "puts all 30 MAP labels of the 10 ggbetweenstats bayes fixtures on the "
        "grid point ggstatsplot chose (0 steps; the fixture values sit within "
        "1.3e-11 steps of their grid points, the rounding of their 15 significant "
        "digits, hence on_grid 1e-6). Zero steps separates the method variants, "
        "each of which moves some of the 30 labels (count, largest move): bins "
        "anchored at min(x) instead of zero (3, 1 step), pilot constants 1.241 "
        "and 1.230 instead of 1.24 and 1.23 (3, 3), a bin width of range / 1000 "
        "instead of 1.01 x range / 1000 (1, 13), netlib's Fortran zeroin instead "
        "of its C translation (2, 2), hmax constant 1.06 instead of 1.144 (3, 2), "
        "an FFT grid spacing of 2 (up - lo) / 2047 (13, 2). SPEC.md section 10's "
        "starting value was one step, which the first two variants would pass."
    ),
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
    actual: float | None,
    expected: float,
    *,
    what: str,
    atol_scale: float = 1.0,
) -> None:
    """Assert closeness under the registered tolerance ``kind``. A missing
    ``actual`` (None, such as the statistic of a pair that has none) fails."""
    if actual is None:
        raise AssertionError(f"{what}: got None, oracle {expected!r}")
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
    actual: float, expected: float, *, origin: float, step: float, what: str
) -> None:
    """Assert two values on the grid ``origin + i * step`` are at most
    ``MAP_CENTRALITY.steps`` grid points apart (the oracle's must lie on the grid)."""
    if not (math.isfinite(step) and step > 0.0):
        raise ValueError(f"grid step must be finite and > 0, got {step!r}")
    tol = MAP_CENTRALITY
    at_expected = (expected - origin) / step
    if abs(at_expected - round(at_expected)) > tol.on_grid:
        raise AssertionError(
            f"{what}: oracle {expected!r} is not on the grid {origin!r} + i x "
            f"{step!r} (position {at_expected!r})"
        )
    at_actual = (actual - origin) / step
    if abs(at_actual - round(at_actual)) > tol.on_grid:
        raise AssertionError(
            f"{what}: {actual!r} is not on the grid {origin!r} + i x {step!r}"
        )
    apart = abs(round(at_actual) - round(at_expected))
    if apart <= tol.steps:
        return
    raise AssertionError(
        f"{what}: got {actual!r} (grid point {round(at_actual)}), oracle "
        f"{expected!r} (grid point {round(at_expected)}); {apart} grid step(s) "
        f"of {step:.3g} apart, more than {tol.steps}"
    )


def assert_same_text(actual: str, expected: str, *, what: str) -> None:
    """Assert exact string equality for formatted numbers (subtitle parity)."""
    if actual != expected:
        raise AssertionError(f"{what}: got {actual!r}, oracle {expected!r}")
