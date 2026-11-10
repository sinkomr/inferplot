"""Settings of the error-rate and coverage simulations (SPEC.md section 11).

Shared by the slow simulation tests (``tests/test_simulation*.py``) and by the
reference script ``tools/reference/simulation_rates.py``, which runs every setting
that has a quantity of kind 2 once, at ten times the replications or more, with
another seed, and writes ``tests/data/simulation_reference.json``.

A setting draws all its replications' data at once from
``numpy.random.Generator(numpy.random.PCG64(seed))``, then computes, for each
replication, whether each of its quantities happened: a test rejecting at alpha =
0.05, an interval covering the true value, a family of pairwise tests rejecting at
least once. Bootstrap methods get the replication's index as their seed, so
replications use different resamples and every run is reproducible.

Each quantity is one of two kinds (the brief of the simulation package):

1. exact, or asymptotically exact with an error known to be small at the simulated
   n: the observed rate must lie in the central 99% binomial band of the nominal
   rate;
2. approximate by construction (R's method, which inferplot reproduces, known or
   found not to hold its level in that setting): the rate must lie in the 99%
   binomial band of the rate established by the reference script (or computed
   exactly where the null distribution is known: the rank tests' and the Hedges' g
   intervals' rates), and its ``why`` says what the nominal rate is and why the
   actual one differs.

The kind is decided from the method's theory, never from a test's outcome.

True values. Every interval is checked against the population value its estimate
converges to (the probability limit), derived below with its citation:

* Hedges' g: delta = (mu - mu0) / sigma (one sample, paired differences).
* rank-biserial r: 2 P(X > Y) - 1 (two samples), 2 P(D1 + D2 > 0) - 1 (one sample
  and paired: the Walsh averages, Kerby 2014).
* rank epsilon squared: 12 sum_j lambda_j (m_j - 1/2)^2, with lambda_j the group's
  share and m_j = sum_l lambda_l P(X_l < X_j) its mean pooled-distribution value
  (the limit of H / N; Kruskal & Wallis 1952, Andrews 1954).
* Kendall's W: 12 sum_j (rho_j - (k + 1)/2)^2 / (k (k^2 - 1)), with rho_j = 1 +
  sum_{l != j} P(Y_j > Y_l) the expected within-subject rank (the limit of the
  Friedman chi-square over n (k - 1); Friedman 1937).
* AKP: (mu_t1 - mu_t2) / sigma_1 for normal groups (Algina, Keselman & Penfield
  2005: c(tr) cancels the winsorized SD of a normal variable); the paired AKP:
  mu_d / sigma_d; the average AKP: the mean over pairs of those.
* xi: sqrt(sum_j (mu_tj - mean mu_t)^2 / (J - 1) / (sigma_w^2(pooled) / v_w)),
  sigma_w^2(pooled) the 20% winsorized variance of the equal mixture of the
  groups, v_w that of a standard normal (Wilcox & Tian 2011; the limit of the
  estimate with equal group sizes).
* partial omega squared (between): sigma_A^2 / (sigma_A^2 + sigma^2) with
  sigma_A^2 = sum_j (mu_j - mean mu)^2 / k (Olejnik & Algina 2003), the quantity the
  label names; (within) the limit of the error-strata formula, (sum alpha_j^2 / k) /
  (sum alpha_j^2 / k + sigma_s^2 + sigma_e^2).
* Pearson's r: rho. Spearman's rho of a bivariate normal: (6 / pi) asin(rho / 2)
  (Pearson 1907; Kruskal 1958, JASA 53: 814-861, section 5.3). The 20% winsorized
  correlation of a bivariate normal: E[W(X) W(Y)] / E[W(X)^2], W the clipping at
  the 20% and 80% quantiles (Wilcox 2017, section 9.3), by one-dimensional
  quadrature of the closed-form conditional mean E[W(Y) | X = x] (Y | X = x is
  normal); ``winsorized_rho_of_normal``.
* trimmed mean of a standard lognormal: e^(1/2) (Phi(z_.8 - 1) - Phi(z_.2 - 1)) /
  0.6, from E[X; a < X < b] = e^(1/2) (Phi(ln b - 1) - Phi(ln a - 1)).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import integrate, optimize
from scipy import stats as sps

import inferplot as ip
from inferplot import stats as st

REFERENCE_FILE = Path(__file__).resolve().parent / "data" / "simulation_reference.json"
ALPHA = 0.05
Z975 = 1.959963984540054

Array = NDArray[np.float64]
Outcome = dict[str, bool]


@dataclass(frozen=True)
class Quantity:
    """One simulated rate: ``key`` names it in the outcome dict, ``what`` says what
    it is in plain words, ``nominal`` is the rate the method claims (None for the
    Bayes factor sanity rates, which have none), ``kind`` 1 or 2 (module
    docstring) and ``why`` the reason for the kind."""

    key: str
    what: str
    nominal: float | None
    kind: Literal[1, 2]
    why: str


@dataclass(frozen=True)
class Setting:
    """A simulation: ``draw(rng, reps)`` returns every replication's data as one
    array indexed by replication; ``outcome(data[i], i)`` the quantities of
    replication i. ``reps`` and ``seed`` are the test's; ``reference_reps`` and
    ``reference_seed`` the reference script's (0 replications: every quantity is of
    kind 1 and no reference is needed). ``params`` describes the setting; the
    reference file records it, so a changed setting with an old reference fails.
    ``level`` says whether the replications go through the public analysis
    function ("api") or through ``inferplot.stats`` ("stats")."""

    name: str
    function: str
    level: Literal["api", "stats"]
    params: str
    draw: Callable[[np.random.Generator, int], Array]
    outcome: Callable[[Array, int], Outcome]
    quantities: tuple[Quantity, ...]
    reps: int
    seed: int
    reference_reps: int = 0
    reference_seed: int = 0
    exact: dict[str, float] = field(default_factory=dict[str, float])


def binomial_band(rate: float, n: int) -> tuple[float, float]:
    """Central 99% band of the observed rate of a Binomial(n, rate) count."""
    low = float(sps.binom.ppf(0.005, n, rate)) / n
    high = float(sps.binom.ppf(0.995, n, rate)) / n
    return low, high


def run(setting: Setting, reps: int, seed: int, offset: int = 0) -> dict[str, int]:
    """Counts of each quantity over ``reps`` replications drawn with ``seed``;
    replication i gets bootstrap seed ``offset + i``."""
    rng = np.random.Generator(np.random.PCG64(seed))
    data = setting.draw(rng, reps)
    hits = {q.key: 0 for q in setting.quantities}
    for i in range(reps):
        out = setting.outcome(data[i], offset + i)
        for key in hits:
            hits[key] += bool(out[key])
    return hits


def load_reference() -> dict[str, Any]:
    if not REFERENCE_FILE.exists():
        return {}
    with REFERENCE_FILE.open(encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def expected_rate(setting: Setting, quantity: Quantity) -> float:
    """The rate the test's band is centred on: the nominal rate (kind 1) or the
    established one (kind 2: the exact rate where known, else the reference
    simulation's)."""
    if quantity.kind == 1:
        assert quantity.nominal is not None
        return quantity.nominal
    if quantity.key in setting.exact:
        return setting.exact[quantity.key]
    entry = load_reference()[setting.name]
    assert entry["params"] == setting.params, (
        f"{setting.name}: the reference was made for another setting; re-run "
        "tools/reference/simulation_rates.py"
    )
    return entry["hits"][quantity.key] / entry["reps"]


def assert_rates(setting: Setting) -> None:
    """Run the setting at its test replications and seed; every quantity's rate
    must lie in the 99% binomial band of its expected rate. Prints each rate, its
    band and its distance from the nearer edge in binomial standard deviations."""
    hits = run(setting, setting.reps, setting.seed)
    failures: list[str] = []
    for quantity in setting.quantities:
        expected = expected_rate(setting, quantity)
        rate = hits[quantity.key] / setting.reps
        low, high = binomial_band(expected, setting.reps)
        sd = math.sqrt(max(expected * (1.0 - expected), 1e-300) / setting.reps)
        margin = min(rate - low, high - rate) / sd
        nominal = "none" if quantity.nominal is None else f"{quantity.nominal:.3f}"
        line = (
            f"{setting.name}/{quantity.key} (kind {quantity.kind}): {quantity.what}: "
            f"{rate:.4f} of {setting.reps}, expected {expected:.4f} (nominal "
            f"{nominal}), 99% band [{low:.4f}, {high:.4f}], {margin:+.2f} SD inside"
        )
        print(line)
        if not low <= rate <= high:
            failures.append(line)
    assert not failures, "\n".join(failures)


# ------------------------------------------------------------------ true values


def phi(x: float) -> float:
    return float(sps.norm.cdf(x))


def winsorized_normal_variance(tr: float = 0.2) -> float:
    """E[W(Z)^2] for Z standard normal clipped at its tr and 1 - tr quantiles."""
    c = float(sps.norm.isf(tr))
    return 2.0 * c * c * tr + (1.0 - 2.0 * tr) - 2.0 * c * float(sps.norm.pdf(c))


def _clipped_normal_mean(m: float, s: float, c: float) -> float:
    """E[clip(Y, -c, c)] for Y ~ N(m, s^2)."""
    a, b = (-c - m) / s, (c - m) / s
    pa, pb = float(sps.norm.cdf(a)), float(sps.norm.cdf(b))
    da, db = float(sps.norm.pdf(a)), float(sps.norm.pdf(b))
    return -c * pa + c * (1.0 - pb) + m * (pb - pa) + s * (da - db)


def winsorized_rho_of_normal(rho: float, tr: float = 0.2) -> float:
    """The population tr-winsorized correlation of a standard bivariate normal."""
    if rho == 0.0:
        return 0.0
    c = float(sps.norm.isf(tr))
    s = math.sqrt(1.0 - rho * rho)

    def inner(x: float) -> float:
        w = min(max(x, -c), c)
        return w * _clipped_normal_mean(rho * x, s, c) * float(sps.norm.pdf(x))

    total = 0.0
    for lo, hi in ((-math.inf, -c), (-c, c), (c, math.inf)):
        value, _ = integrate.quad(inner, lo, hi, epsabs=1e-13, epsrel=1e-12)
        total += value
    return total / winsorized_normal_variance(tr)


def spearman_rho_of_normal(rho: float) -> float:
    return 6.0 / math.pi * math.asin(rho / 2.0)


def winsorized_variance_of_mixture(means: Sequence[float], tr: float = 0.2) -> float:
    """20% winsorized variance of the equal mixture of N(mu_j, 1)."""
    k = len(means)

    def cdf(x: float) -> float:
        return sum(phi(x - m) for m in means) / k

    def quantile(p: float) -> float:
        lo, hi = min(means) - 10.0, max(means) + 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    a, b = quantile(tr), quantile(1.0 - tr)

    def moment(power: int) -> float:
        def f(x: float) -> float:
            return x**power * sum(float(sps.norm.pdf(x - m)) for m in means) / k

        value, _ = integrate.quad(f, a, b, epsabs=1e-13, epsrel=1e-12)
        return value

    m1 = tr * a + moment(1) + tr * b
    m2 = tr * a * a + moment(2) + tr * b * b
    return m2 - m1 * m1


def xi_of_normal_groups(means: Sequence[float], tr: float = 0.2) -> float:
    centre = sum(means) / len(means)
    top = sum((m - centre) ** 2 for m in means) / (len(means) - 1)
    bottom = winsorized_variance_of_mixture(means, tr) / winsorized_normal_variance(tr)
    return math.sqrt(top / bottom)


def epsilon_squared_limit(means: Sequence[float], sizes: Sequence[int]) -> float:
    total = sum(sizes)
    shares = [s / total for s in sizes]
    value = 0.0
    for j, mj in enumerate(means):
        m = sum(
            lam * phi((mj - ml) / math.sqrt(2.0))
            for lam, ml in zip(shares, means, strict=True)
        )
        value += shares[j] * (m - 0.5) ** 2
    return 12.0 * value


def kendall_w_limit(effects: Sequence[float], sd_error: float) -> float:
    k = len(effects)
    total = 0.0
    for j, aj in enumerate(effects):
        rho_j = 1.0 + sum(
            phi((aj - al) / (sd_error * math.sqrt(2.0)))
            for l_, al in enumerate(effects)
            if l_ != j
        )
        total += (rho_j - (k + 1) / 2.0) ** 2
    return 12.0 * total / (k * (k * k - 1))


def lognormal_trimmed_mean(tr: float = 0.2) -> float:
    za, zb = float(sps.norm.ppf(tr)), float(sps.norm.ppf(1.0 - tr))
    return math.exp(0.5) * (phi(zb - 1.0) - phi(za - 1.0)) / (1.0 - 2.0 * tr)


def _hedges_j(df: float) -> float:
    return math.exp(
        math.lgamma(df / 2.0) - 0.5 * math.log(df / 2.0) - math.lgamma((df - 1.0) / 2.0)
    )


def hedges_g_coverage(
    n: int, delta: float, side: Literal["two-sided", "greater"]
) -> float:
    """Exact coverage of delta by effectsize's one-sample (or paired) Hedges' g
    interval: J(n - 1) times the exact noncentral-t interval for delta. It covers
    delta exactly when the ncp interval covers delta sqrt(n) / J, so the coverage is
    a noncentral-t probability at ncp delta sqrt(n) (Steiger & Fouladi 1997).

    The noncentral t is computed here by quadrature (``noncentral_t_cdf``), not by
    SciPy's ``nct``, which is inaccurate in places at the oldest supported SciPy."""
    df = n - 1.0
    shifted = delta * math.sqrt(n) / _hedges_j(df)
    true = delta * math.sqrt(n)

    def quantile(p: float) -> float:
        def gap(t: float) -> float:
            return noncentral_t_cdf(t, df, shifted) - p

        return float(optimize.brentq(gap, shifted - 40.0, shifted + 40.0, xtol=1e-12))

    if side == "two-sided":
        lo, hi = quantile(ALPHA / 2.0), quantile(1.0 - ALPHA / 2.0)
        return noncentral_t_cdf(hi, df, true) - noncentral_t_cdf(lo, df, true)
    return noncentral_t_cdf(quantile(1.0 - ALPHA), df, true)


def noncentral_t_cdf(t: float, df: float, ncp: float) -> float:
    """P(T' <= t) = int_0^inf f_S(s) Phi(t s - ncp) ds, with S = chi_df / sqrt(df),
    by adaptive quadrature over the bulk of S (moderate df and ncp only)."""
    half = df / 2.0
    log_c = math.log(2.0) + half * math.log(half) - math.lgamma(half)

    def integrand(s: float) -> float:
        density = math.exp(log_c + (df - 1.0) * math.log(s) - half * s * s)
        return density * float(sps.norm.cdf(t * s - ncp))

    sd = 1.0 / math.sqrt(2.0 * df)
    lo, hi = max(1e-12, 1.0 - 12.0 * sd), 1.0 + 14.0 * sd
    value, _ = integrate.quad(integrand, lo, hi, epsabs=1e-14, epsrel=1e-12, limit=200)
    return value


def mann_whitney_exact_rate(n1: int, n2: int) -> float:
    """Exact Type I error at alpha 0.05 of R's corrected normal approximation for
    the Mann-Whitney U without ties: U's null distribution by the Mann & Whitney
    (1947) recursion, p = 2 P(Z > (|U - n1 n2 / 2| - 1/2) / sd)."""
    # counts[i][j][u]: arrangements of i x's and j y's with u (x > y) pairs.
    counts = np.zeros((n1 + 1, n2 + 1, n1 * n2 + 1))
    counts[0, :, 0] = 1.0
    counts[:, 0, 0] = 1.0
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            # The largest value is an x (beats all j y's) or a y.
            counts[i, j, j:] += counts[i - 1, j, : n1 * n2 + 1 - j]
            counts[i, j, :] += counts[i, j - 1, :]
    dist = counts[n1, n2] / counts[n1, n2].sum()
    u = np.arange(n1 * n2 + 1, dtype=np.float64)
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    dev = np.maximum(np.abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    p = 2.0 * sps.norm.sf(dev / sd)
    return float(dist[p < ALPHA].sum())


def signed_rank_exact_rate(n: int, side: str = "two-sided") -> float:
    """Exact Type I error at alpha 0.05 of R's corrected normal approximation for
    the signed-rank V without zeros or ties (Wilcoxon 1945): V's null distribution
    by the subset-sum recursion over the ranks 1..n."""
    top = n * (n + 1) // 2
    dist = np.zeros(top + 1)
    dist[0] = 1.0
    for r in range(1, n + 1):
        dist[r:] = dist[r:] + dist[: top + 1 - r].copy()
    dist /= dist.sum()
    v = np.arange(top + 1, dtype=np.float64)
    mean = top / 2.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if side == "two-sided":
        dev = np.maximum(np.abs(v - mean) - 0.5, 0.0)
        p = 2.0 * sps.norm.sf(dev / sd)
    elif side == "less":
        p = sps.norm.cdf((v - mean + 0.5) / sd)
    else:
        p = sps.norm.sf((v - mean - 0.5) / sd)
    return float(dist[p < ALPHA].sum())


# ------------------------------------------------------------------ data


def normal_groups(
    sizes: Sequence[int], sds: Sequence[float], means: Sequence[float]
) -> Callable[[np.random.Generator, int], Array]:
    """Rows of independent normal groups, concatenated in group order."""
    scale = np.repeat(np.asarray(sds, dtype=np.float64), sizes)
    loc = np.repeat(np.asarray(means, dtype=np.float64), sizes)

    def draw(rng: np.random.Generator, reps: int) -> Array:
        return rng.standard_normal((reps, int(sum(sizes)))) * scale + loc

    return draw


def split(row: Array, sizes: Sequence[int]) -> list[Array]:
    edges = np.cumsum(sizes)[:-1]
    return [np.ascontiguousarray(g) for g in np.split(row, edges)]


def multivariate(
    n: int, means: Sequence[float], cov: Array
) -> Callable[[np.random.Generator, int], Array]:
    """n subjects by k conditions, multivariate normal rows."""
    chol = np.linalg.cholesky(cov)
    loc = np.asarray(means, dtype=np.float64)

    def draw(rng: np.random.Generator, reps: int) -> Array:
        z = rng.standard_normal((reps, n, len(means)))
        return z @ chol.T + loc

    return draw


def compound_symmetry(k: int, subject_var: float = 1.0) -> Array:
    return np.eye(k) + subject_var * np.ones((k, k))


def conditions(row: Array) -> list[Array]:
    return [np.ascontiguousarray(row[:, j]) for j in range(row.shape[1])]


def covers(low: float, high: float, value: float) -> bool:
    return low <= value <= high


def any_below(p_values: Sequence[float] | tuple[float, ...]) -> bool:
    return any(p < ALPHA for p in p_values)


# ------------------------------------------------------------------ betweenstats

_PAIR_HOLM = "Holm-adjusted pairs, any rejection under the complete null"
_BOOT = (
    "percentile bootstrap of 2,000 resamples at small n: its coverage is not "
    "known to be close to nominal there, so it is established by the reference"
)
_RANK_APPROX = (
    "normal (or chi-square) approximation to a discrete rank statistic, with R's "
    "continuity correction where it has one: its rate at this n is established "
    "(exactly where the null distribution can be enumerated)"
)

GH_SIZES, GH_SDS = (8, 15, 30), (1.0, 2.0, 4.0)


def _games_howell(row: Array, _: int) -> Outcome:
    groups = split(row, GH_SIZES)
    pairs = st.games_howell(groups)
    welch = st.welch_anova(groups)
    omega = st.partial_omega_squared(welch.f, welch.df1, welch.df2)
    return {
        "raw": any_below([p.p_value for p in pairs]),
        "holm": any_below([p.p_adjusted for p in pairs]),
        "omega0": covers(omega.conf_low, omega.conf_high, 0.0),
    }


OMEGA_N, OMEGA_MEANS = 20, (0.0, 0.5, 1.0)
OMEGA_TRUE = (sum((m - 0.5) ** 2 for m in OMEGA_MEANS) / 3.0) / (
    sum((m - 0.5) ** 2 for m in OMEGA_MEANS) / 3.0 + 1.0
)


def _omega_effect(row: Array, _: int) -> Outcome:
    welch = st.welch_anova(split(row, (OMEGA_N,) * 3))
    omega = st.partial_omega_squared(welch.f, welch.df1, welch.df2)
    return {"cover": covers(omega.conf_low, omega.conf_high, OMEGA_TRUE)}


MW_SIZES = (10, 25)


def _mann_whitney_api(row: Array, _: int) -> Outcome:
    import pandas as pd

    frame = pd.DataFrame(
        {"g": np.repeat(["a", "b"], MW_SIZES), "y": row}
    )  # group a is the first level
    result = ip.analyze.betweenstats(frame, "g", "y", type="nonparametric")
    assert result.test is not None
    assert result.effect_size is not None
    es = result.effect_size
    return {
        "p": result.test.p_value < ALPHA,
        "r0": covers(es.conf_low, es.conf_high, 0.0),
    }


RB_SHIFT = 0.8
RB_TRUE = 2.0 * phi(RB_SHIFT / math.sqrt(2.0)) - 1.0


def _rank_biserial_shift(row: Array, _: int) -> Outcome:
    a, b = split(row, MW_SIZES)
    es = st.rank_biserial(a, b)
    return {"cover": covers(es.conf_low, es.conf_high, RB_TRUE)}


KW_SIZES = (8, 15, 30)


def _kruskal(row: Array, _: int) -> Outcome:
    groups = split(row, KW_SIZES)
    dunn = st.dunn(groups)
    return {
        "p": st.kruskal_wallis(groups).p_value < ALPHA,
        "holm": any_below(st.p_adjust([d.p_value for d in dunn], "holm")),
    }


EPS_MEANS = (0.0, 0.5, 1.0)
EPS_TRUE = epsilon_squared_limit(EPS_MEANS, KW_SIZES)


def _epsilon(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, seed: int) -> Outcome:
        es = st.rank_epsilon_squared(split(row, KW_SIZES), seed=seed)
        return {
            "cover": covers(es.conf_low, es.conf_high, true),
            "inside": covers(es.conf_low, es.conf_high, es.estimate),
        }

    return outcome


YUEN_SIZES, YUEN_SDS = (10, 25), (3.0, 1.0)


def _yuen(row: Array, _: int) -> Outcome:
    a, b = split(row, YUEN_SIZES)
    return {"p": st.yuen_t(a, b).p_value < ALPHA}


AKP_SIZES, AKP_SDS = (12, 24), (1.0, 2.0)


def _akp(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, seed: int) -> Outcome:
        a, b = split(row, AKP_SIZES)
        es = st.akp_effect(a, b, seed=seed)
        return {"cover": covers(es.conf_low, es.conf_high, true)}

    return outcome


T1_SIZES, T1_SDS = (8, 15, 30), (1.0, 2.0, 4.0)


def _t1way(row: Array, _: int) -> Outcome:
    groups = split(row, T1_SIZES)
    pairs = st.lincon(groups)
    return {
        "p": st.trimmed_means_anova(groups).p_value < ALPHA,
        "holm": any_below([p.p_adjusted for p in pairs]),
        "ci": all(covers(p.conf_low, p.conf_high, 0.0) for p in pairs),
    }


XI_N, XI_MEANS = 15, (0.0, 0.5, 1.0)
XI_TRUE = xi_of_normal_groups(XI_MEANS)


def _xi(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, seed: int) -> Outcome:
        es = st.explanatory_effect(split(row, (XI_N,) * 3), seed=seed)
        return {"cover": covers(es.conf_low, es.conf_high, true)}

    return outcome


def _bayes_two_sample(sizes: tuple[int, int]) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        a, b = split(row, sizes)
        bf10, _log = st.jzs_bf10_two_sample(a, b, prior_scale=0.707)
        return {"null": bf10 < 1.0, "alt": bf10 > 1.0}

    return outcome


BETWEEN: tuple[Setting, ...] = (
    Setting(
        name="between_games_howell_null",
        function="betweenstats",
        level="stats",
        params="normal n (8, 15, 30), sd (1, 2, 4), means 0",
        draw=normal_groups(GH_SIZES, GH_SDS, (0.0, 0.0, 0.0)),
        outcome=_games_howell,
        quantities=(
            Quantity(
                "raw",
                "Games-Howell pairs before Holm: any false rejection",
                0.05,
                1,
                "Games-Howell is family-wise by itself, a Welch-type approximation "
                "known to hold its level with 6 or more values per group (Games & "
                "Howell 1976)",
            ),
            Quantity(
                "holm",
                "Games-Howell pairs after ggstatsplot's Holm: any false rejection",
                0.05,
                2,
                "Holm on top of p-values that are already family-wise (ggstatsplot) "
                "is conservative by construction",
            ),
            Quantity(
                "omega0",
                "partial omega squared interval [lower, 1] covers 0",
                0.95,
                2,
                "the bound is computed from F_w = (F - 1) df2 / (df1 + df2 + 1) < F, "
                "not from F, so under the null it is 0 more often than 95%",
            ),
        ),
        reps=2_000,
        seed=20261201,
        reference_reps=20_000,
        reference_seed=30261201,
    ),
    Setting(
        name="between_omega_effect",
        function="betweenstats",
        level="stats",
        params="normal n (20, 20, 20), sd 1, means (0, 0.5, 1); omega^2 = 1/7",
        draw=normal_groups((OMEGA_N,) * 3, (1.0,) * 3, OMEGA_MEANS),
        outcome=_omega_effect,
        quantities=(
            Quantity(
                "cover",
                "partial omega squared interval [lower, 1] covers the true 1/7",
                0.95,
                2,
                "the estimate uses Welch's df2 (about 2 (n - 1) here) where omega "
                "squared's formula assumes N - k (3 (n - 1)), so it converges to 0.2, "
                "not 1/7",
            ),
        ),
        reps=2_000,
        seed=20261202,
        reference_reps=20_000,
        reference_seed=30261202,
    ),
    Setting(
        name="between_mann_whitney_null",
        function="betweenstats",
        level="api",
        params="normal n (10, 25), sd 1, means 0; ip.analyze.betweenstats",
        draw=normal_groups(MW_SIZES, (1.0, 1.0), (0.0, 0.0)),
        outcome=_mann_whitney_api,
        quantities=(
            Quantity(
                "p", "Mann-Whitney rejects equal distributions", 0.05, 2, _RANK_APPROX
            ),
            Quantity(
                "r0",
                "rank-biserial r interval covers 0",
                0.95,
                2,
                "the Fisher-z interval uses r's null standard error on the atanh "
                "scale, where it is too narrow: tanh(z se) < z se",
            ),
        ),
        reps=2_000,
        seed=20261203,
        reference_reps=10_000,
        reference_seed=30261203,
        exact={"p": mann_whitney_exact_rate(*MW_SIZES)},
    ),
    Setting(
        name="between_rank_biserial_shift",
        function="betweenstats",
        level="stats",
        params=f"normal n (10, 25), sd 1, means ({RB_SHIFT}, 0); r = {RB_TRUE:.6f}",
        draw=normal_groups(MW_SIZES, (1.0, 1.0), (RB_SHIFT, 0.0)),
        outcome=_rank_biserial_shift,
        quantities=(
            Quantity(
                "cover",
                "rank-biserial r interval covers the true r (0.43)",
                0.95,
                2,
                "the standard error is r's null one, used at every r",
            ),
        ),
        reps=2_000,
        seed=20261204,
        reference_reps=20_000,
        reference_seed=30261204,
    ),
    Setting(
        name="between_kruskal_null",
        function="betweenstats",
        level="stats",
        params="normal n (8, 15, 30), sd 1, means 0",
        draw=normal_groups(KW_SIZES, (1.0,) * 3, (0.0,) * 3),
        outcome=_kruskal,
        quantities=(
            Quantity(
                "p", "Kruskal-Wallis rejects equal distributions", 0.05, 2, _RANK_APPROX
            ),
            Quantity(
                "holm",
                "Dunn pairs after Holm: " + _PAIR_HOLM,
                0.05,
                2,
                "Holm bounds the family-wise error at 5%; at or below",
            ),
        ),
        reps=2_000,
        seed=20261205,
        reference_reps=20_000,
        reference_seed=30261205,
    ),
    Setting(
        name="between_epsilon_null",
        function="betweenstats",
        level="stats",
        params="normal n (8, 15, 30), sd 1, means 0; rank epsilon squared",
        draw=normal_groups(KW_SIZES, (1.0,) * 3, (0.0,) * 3),
        outcome=_epsilon(0.0),
        quantities=(
            Quantity(
                "cover",
                "rank epsilon squared interval [lower, 1] covers 0",
                0.95,
                2,
                "epsilon squared is positive in every resample, so the lower bound "
                "is above 0 and the interval never covers no effect",
            ),
            Quantity(
                "inside",
                "rank epsilon squared lies inside its own interval",
                1.0,
                2,
                "the bootstrap of H / (N - 1) is biased upward; under no effect its "
                "5% point is often above the estimate (finding U6)",
            ),
        ),
        reps=400,
        seed=20261206,
        reference_reps=10_000,
        reference_seed=30261206,
    ),
    Setting(
        name="between_epsilon_effect",
        function="betweenstats",
        level="stats",
        params=(
            f"normal n (8, 15, 30), sd 1, means (0, 0.5, 1); epsilon^2 = {EPS_TRUE:.6f}"
        ),
        draw=normal_groups(KW_SIZES, (1.0,) * 3, EPS_MEANS),
        outcome=_epsilon(EPS_TRUE),
        quantities=(
            Quantity(
                "cover",
                "rank epsilon squared interval covers the true value (0.12)",
                0.95,
                2,
                _BOOT,
            ),
            Quantity(
                "inside",
                "rank epsilon squared lies inside its own interval",
                1.0,
                2,
                _BOOT,
            ),
        ),
        reps=400,
        seed=20261207,
        reference_reps=10_000,
        reference_seed=30261207,
    ),
    Setting(
        name="between_yuen_null",
        function="betweenstats",
        level="stats",
        params="normal n (10, 25), sd (3, 1), means 0",
        draw=normal_groups(YUEN_SIZES, YUEN_SDS, (0.0, 0.0)),
        outcome=_yuen,
        quantities=(
            Quantity(
                "p",
                "Yuen's test rejects equal trimmed means",
                0.05,
                2,
                "Yuen's Welch-type df for trimmed means is approximate with h = 6 "
                "values left after trimming the smaller group (Wilcox 2017, 5.3)",
            ),
        ),
        reps=2_000,
        seed=20261208,
        reference_reps=20_000,
        reference_seed=30261208,
    ),
    Setting(
        name="between_akp_null",
        function="betweenstats",
        level="stats",
        params="normal n (12, 24), sd (1, 2), means 0; AKP",
        draw=normal_groups(AKP_SIZES, AKP_SDS, (0.0, 0.0)),
        outcome=_akp(0.0),
        quantities=(Quantity("cover", "AKP interval covers 0", 0.95, 2, _BOOT),),
        reps=1_000,
        seed=20261209,
        reference_reps=10_000,
        reference_seed=30261209,
    ),
    Setting(
        name="between_akp_effect",
        function="betweenstats",
        level="stats",
        params="normal n (12, 24), sd (1, 2), means (0.8, 0); AKP = 0.8",
        draw=normal_groups(AKP_SIZES, AKP_SDS, (0.8, 0.0)),
        outcome=_akp(0.8),
        quantities=(
            Quantity("cover", "AKP interval covers the true 0.8", 0.95, 2, _BOOT),
        ),
        reps=1_000,
        seed=20261210,
        reference_reps=10_000,
        reference_seed=30261210,
    ),
    Setting(
        name="between_t1way_null",
        function="betweenstats",
        level="stats",
        params="normal n (8, 15, 30), sd (1, 2, 4), means 0",
        draw=normal_groups(T1_SIZES, T1_SDS, (0.0,) * 3),
        outcome=_t1way,
        quantities=(
            Quantity(
                "p",
                "trimmed-means ANOVA rejects equal trimmed means",
                0.05,
                2,
                "Welch-type F for trimmed means with h = 4 in the smallest group: "
                "approximate (Wilcox 2017, 7.1.1)",
            ),
            Quantity(
                "holm",
                "lincon pairs after Holm: " + _PAIR_HOLM,
                0.05,
                2,
                "Holm on Yuen-type pairs at small h",
            ),
            Quantity(
                "ci",
                "lincon's pair intervals all cover 0 (simultaneously)",
                0.95,
                2,
                "Studentized maximum modulus critical values at Yuen's df: "
                "approximate at small h",
            ),
        ),
        reps=2_000,
        seed=20261211,
        reference_reps=20_000,
        reference_seed=30261211,
    ),
    Setting(
        name="between_xi_null",
        function="betweenstats",
        level="stats",
        params="normal n (15, 15, 15), sd 1, means 0; xi",
        draw=normal_groups((XI_N,) * 3, (1.0,) * 3, (0.0,) * 3),
        outcome=_xi(0.0),
        quantities=(
            Quantity(
                "cover",
                "explanatory effect size xi interval covers 0",
                0.95,
                2,
                "xi is positive in every resample, so its percentile interval "
                "never reaches 0",
            ),
        ),
        reps=1_000,
        seed=20261212,
        reference_reps=10_000,
        reference_seed=30261212,
    ),
    Setting(
        name="between_xi_effect",
        function="betweenstats",
        level="stats",
        params=f"normal n (15, 15, 15), sd 1, means (0, 0.5, 1); xi = {XI_TRUE:.6f}",
        draw=normal_groups((XI_N,) * 3, (1.0,) * 3, XI_MEANS),
        outcome=_xi(XI_TRUE),
        quantities=(
            Quantity(
                "cover",
                "explanatory effect size xi interval covers the true xi (0.46)",
                0.95,
                2,
                _BOOT,
            ),
        ),
        reps=1_000,
        seed=20261213,
        reference_reps=10_000,
        reference_seed=30261213,
    ),
    Setting(
        name="between_bayes_null",
        function="betweenstats",
        level="stats",
        params="normal n (100, 100), sd 1, means 0; JZS r 0.707",
        draw=normal_groups((100, 100), (1.0, 1.0), (0.0, 0.0)),
        outcome=_bayes_two_sample((100, 100)),
        quantities=(
            Quantity(
                "null",
                "Bayes factor favours no difference (BF10 < 1) under no difference",
                None,
                2,
                "a Bayes factor has no error rate to hold: sanity check only",
            ),
        ),
        reps=1_000,
        seed=20261214,
        reference_reps=10_000,
        reference_seed=30261214,
    ),
    Setting(
        name="between_bayes_effect",
        function="betweenstats",
        level="stats",
        params="normal n (30, 30), sd 1, means (0.8, 0); JZS r 0.707",
        draw=normal_groups((30, 30), (1.0, 1.0), (0.8, 0.0)),
        outcome=_bayes_two_sample((30, 30)),
        quantities=(
            Quantity(
                "alt",
                "Bayes factor favours a difference (BF10 > 1) at d = 0.8",
                None,
                2,
                "a Bayes factor has no error rate to hold: sanity check only",
            ),
        ),
        reps=1_000,
        seed=20261215,
        reference_reps=10_000,
        reference_seed=30261215,
    ),
)


# ------------------------------------------------------------------ withinstats

PAIR_N = 15
PAIR_COV = np.array([[1.0, 1.0], [1.0, 4.0]])  # sds 1 and 2, correlation 0.5
PAIRED_DELTA = 0.6  # mean difference / sd of the differences
PAIR_SD_DIFF = math.sqrt(1.0 + 4.0 - 2.0 * 1.0)  # sd of a - b = sqrt(3)


def _paired_api(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "id": np.tile(np.arange(PAIR_N), 2),
                "cond": np.repeat(["a", "b"], PAIR_N),
                "y": np.concatenate([row[:, 0], row[:, 1]]),
            }
        )
        result = ip.analyze.withinstats(
            frame, "cond", "y", subject_id="id", bf_message=False
        )
        assert result.test is not None
        assert result.effect_size is not None
        es = result.effect_size
        return {
            "p": result.test.p_value < ALPHA,
            "cover": covers(es.conf_low, es.conf_high, true),
        }

    return outcome


def _paired_one_sided(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        a, b = row[:, 0].copy(), row[:, 1].copy()
        es = st.paired_hedges_g(a, b, alternative="greater")
        return {"cover": covers(es.conf_low, es.conf_high, true)}

    return outcome


RM_N, RM_K = 12, 4
NONSPHERICAL_SDS = np.array([1.0, 1.5, 2.0, 3.0])
NONSPHERICAL_COV = np.outer(NONSPHERICAL_SDS, NONSPHERICAL_SDS) * 0.8 ** np.abs(
    np.subtract.outer(np.arange(RM_K), np.arange(RM_K))
)


def greenhouse_geisser_epsilon(cov: Array) -> float:
    """Population Greenhouse-Geisser epsilon of a covariance matrix (Box 1954)."""
    k = cov.shape[0]
    centring = np.eye(k) - np.ones((k, k)) / k
    s = centring @ cov @ centring
    return float(np.trace(s) ** 2 / ((k - 1) * np.sum(s * s)))


def _rm_parametric(row: Array, _: int) -> Outcome:
    conds = conditions(row)
    omega = st.rm_partial_omega_squared(conds)
    return {
        "p": st.rm_anova(conds).p_value < ALPHA,
        "omega0": covers(omega.conf_low, omega.conf_high, 0.0),
        "holm": any_below([p.p_adjusted for p in st.paired_t_pairs(conds)]),
    }


RM_OMEGA_N, RM_OMEGA_EFFECTS = 20, (-0.5, 0.0, 0.5)
RM_OMEGA_TRUE = (sum(a * a for a in RM_OMEGA_EFFECTS) / 3.0) / (
    sum(a * a for a in RM_OMEGA_EFFECTS) / 3.0 + 2.0
)


def _rm_omega_effect(row: Array, _: int) -> Outcome:
    omega = st.rm_partial_omega_squared(conditions(row))
    return {"cover": covers(omega.conf_low, omega.conf_high, RM_OMEGA_TRUE)}


SR_SHIFT = 0.5  # mean of the differences, sd 1
SR_TRUE = 2.0 * phi(math.sqrt(2.0) * SR_SHIFT) - 1.0


def _signed_rank(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        a, b = row[:, 0].copy(), row[:, 1].copy()
        es = st.paired_rank_biserial(a, b)
        return {
            "p": st.signed_rank(a, b).p_value < ALPHA,
            "cover": covers(es.conf_low, es.conf_high, true),
        }

    return outcome


def _friedman(row: Array, _: int) -> Outcome:
    conds = conditions(row)
    return {
        "p": st.friedman(conds).p_value < ALPHA,
        "holm": any_below([p.p_adjusted for p in st.durbin_conover(conds)]),
    }


KENDALL_EFFECTS = (0.0, 0.25, 0.5, 0.75)
W_TRUE = kendall_w_limit(KENDALL_EFFECTS, 1.0)


def _kendall(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, seed: int) -> Outcome:
        es = st.kendalls_w(conditions(row), seed=seed)
        return {
            "cover": covers(es.conf_low, es.conf_high, true),
            "inside": covers(es.conf_low, es.conf_high, es.estimate),
        }

    return outcome


def _yuend(row: Array, _: int) -> Outcome:
    test = st.yuen_dependent_t(row[:, 0].copy(), row[:, 1].copy())
    return {"p": test.p_value < ALPHA}


DEP_AKP_DELTA = 0.8  # mean of the differences / their sd


def _dep_akp(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, seed: int) -> Outcome:
        es = st.dependent_akp(row[:, 0].copy(), row[:, 1].copy(), seed=seed)
        return {"cover": covers(es.conf_low, es.conf_high, true)}

    return outcome


def _rm_robust(true: Sequence[float]) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        conds = conditions(row)
        pairs = st.rmmcp(conds)
        values = {
            "holm": any_below([p.p_adjusted for p in pairs]),
            "ci": all(
                covers(p.conf_low, p.conf_high, true[p.i] - true[p.j]) for p in pairs
            ),
        }
        if not any(true):
            values["p"] = st.trimmed_rm_anova(conds).p_value < ALPHA
        return values

    return outcome


RMMCP_EFFECTS = (0.0, 0.5, 1.0, 1.5)
AVG_AKP_N, AVG_AKP_EFFECTS = 12, (0.0, 0.5, 1.0)
AVG_AKP_TRUE = (
    sum(
        (AVG_AKP_EFFECTS[i] - AVG_AKP_EFFECTS[j]) / math.sqrt(2.0)
        for i in range(3)
        for j in range(i + 1, 3)
    )
    / 3.0
)


def _avg_akp(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, seed: int) -> Outcome:
        es = st.average_dependent_akp(conditions(row), seed=seed)
        return {"cover": covers(es.conf_low, es.conf_high, true)}

    return outcome


def _bayes_paired(row: Array, _: int) -> Outcome:
    bf10, _log = st.jzs_bf10_paired(
        row[:, 0].copy(), row[:, 1].copy(), prior_scale=0.707
    )
    return {"null": bf10 < 1.0, "alt": bf10 > 1.0}


_SANITY = "a Bayes factor has no error rate to hold: sanity check only"
_HOLM_BOUND = "Holm bounds the family-wise error at 5%: at or below"
_GG = (
    "the Greenhouse-Geisser correction is approximate: conservative when the "
    "population epsilon is near 1 (its estimate is biased down; Huynh & Feldt "
    "1976), an approximation when it is not"
)
_OMEGA_WITHIN = (
    "the bound is computed from F_w, derived from the estimate, at the "
    "uncorrected df: not an exact interval"
)
_PAIR_COV_TEXT = "15 pairs, sds (1, 2), correlation 0.5"
_CS = "compound symmetry (subject variance 1, error variance 1)"

WITHIN: tuple[Setting, ...] = (
    Setting(
        name="within_paired_null",
        function="withinstats",
        level="api",
        params=f"{_PAIR_COV_TEXT}, means 0; ip.analyze.withinstats",
        draw=multivariate(PAIR_N, (0.0, 0.0), PAIR_COV),
        outcome=_paired_api(0.0),
        quantities=(
            Quantity(
                "p", "paired t rejects equal means", 0.05, 1, "exact under normality"
            ),
            Quantity(
                "cover",
                "paired Hedges' g interval covers 0",
                0.95,
                1,
                "at delta 0 the interval is J times the exact noncentral-t "
                "interval and covers 0 exactly when p >= 0.05",
            ),
        ),
        reps=2_000,
        seed=20261301,
    ),
    Setting(
        name="within_paired_effect",
        function="withinstats",
        level="api",
        params=(
            f"{_PAIR_COV_TEXT}, mean difference {PAIRED_DELTA} x sqrt(3); "
            "ip.analyze.withinstats"
        ),
        draw=multivariate(PAIR_N, (PAIRED_DELTA * PAIR_SD_DIFF, 0.0), PAIR_COV),
        outcome=_paired_api(PAIRED_DELTA),
        quantities=(
            Quantity(
                "cover",
                f"paired Hedges' g interval covers the true delta ({PAIRED_DELTA})",
                0.95,
                2,
                "effectsize multiplies the exact interval for delta by J(n - 1) = "
                "0.946, which pulls it toward 0; its exact coverage is computed",
            ),
        ),
        reps=2_000,
        seed=20261302,
        exact={"cover": hedges_g_coverage(PAIR_N, PAIRED_DELTA, "two-sided")},
    ),
    Setting(
        name="within_paired_one_sided",
        function="withinstats",
        level="stats",
        params=(
            f"{_PAIR_COV_TEXT}, mean difference {PAIRED_DELTA} x sqrt(3); "
            "alternative greater"
        ),
        draw=multivariate(PAIR_N, (PAIRED_DELTA * PAIR_SD_DIFF, 0.0), PAIR_COV),
        outcome=_paired_one_sided(PAIRED_DELTA),
        quantities=(
            Quantity(
                "cover",
                f"one-sided paired Hedges' g interval [low, Inf] covers {PAIRED_DELTA}",
                0.95,
                2,
                "J(n - 1) pulls the lower bound toward 0, so the one-sided interval "
                "covers more than 95% when delta > 0; exact coverage computed",
            ),
        ),
        reps=2_000,
        seed=20261303,
        exact={"cover": hedges_g_coverage(PAIR_N, PAIRED_DELTA, "greater")},
    ),
    Setting(
        name="within_rm_spherical",
        function="withinstats",
        level="stats",
        params=f"12 subjects x 4 conditions, {_CS}, means 0",
        draw=multivariate(RM_N, (0.0,) * RM_K, compound_symmetry(RM_K)),
        outcome=_rm_parametric,
        quantities=(
            Quantity(
                "p",
                "repeated-measures ANOVA (Greenhouse-Geisser) rejects",
                0.05,
                2,
                _GG,
            ),
            Quantity(
                "omega0",
                "partial omega squared interval covers 0",
                0.95,
                2,
                _OMEGA_WITHIN,
            ),
            Quantity(
                "holm", "paired t pairs after Holm: " + _PAIR_HOLM, 0.05, 2, _HOLM_BOUND
            ),
        ),
        reps=2_000,
        seed=20261304,
        reference_reps=20_000,
        reference_seed=30261304,
    ),
    Setting(
        name="within_rm_nonspherical",
        function="withinstats",
        level="stats",
        params=(
            "12 subjects x 4 conditions, sds (1, 1.5, 2, 3), correlation "
            "0.8^|i - j|, means 0; population epsilon "
            f"{greenhouse_geisser_epsilon(NONSPHERICAL_COV):.4f}"
        ),
        draw=multivariate(RM_N, (0.0,) * RM_K, NONSPHERICAL_COV),
        outcome=_rm_parametric,
        quantities=(
            Quantity(
                "p",
                "repeated-measures ANOVA (Greenhouse-Geisser) rejects",
                0.05,
                2,
                _GG,
            ),
            Quantity(
                "omega0",
                "partial omega squared interval covers 0",
                0.95,
                2,
                _OMEGA_WITHIN,
            ),
            Quantity(
                "holm", "paired t pairs after Holm: " + _PAIR_HOLM, 0.05, 2, _HOLM_BOUND
            ),
        ),
        reps=2_000,
        seed=20261305,
        reference_reps=20_000,
        reference_seed=30261305,
    ),
    Setting(
        name="within_rm_omega_effect",
        function="withinstats",
        level="stats",
        params=(
            f"20 subjects x 3 conditions, {_CS}, effects (-0.5, 0, 0.5); "
            f"omega^2 = {RM_OMEGA_TRUE:.6f}"
        ),
        draw=multivariate(RM_OMEGA_N, RM_OMEGA_EFFECTS, compound_symmetry(3)),
        outcome=_rm_omega_effect,
        quantities=(
            Quantity(
                "cover",
                "partial omega squared interval covers the true value (0.077)",
                0.95,
                2,
                _OMEGA_WITHIN,
            ),
        ),
        reps=2_000,
        seed=20261306,
        reference_reps=20_000,
        reference_seed=30261306,
    ),
    Setting(
        name="within_signed_rank_null",
        function="withinstats",
        level="stats",
        params=f"{_PAIR_COV_TEXT}, means 0",
        draw=multivariate(PAIR_N, (0.0, 0.0), PAIR_COV),
        outcome=_signed_rank(0.0),
        quantities=(
            Quantity("p", "Wilcoxon signed-rank rejects", 0.05, 2, _RANK_APPROX),
            Quantity(
                "cover",
                "paired rank-biserial r interval covers 0",
                0.95,
                2,
                "Fisher-z interval with r's null standard error: too narrow on the "
                "atanh scale",
            ),
        ),
        reps=2_000,
        seed=20261307,
        reference_reps=20_000,
        reference_seed=30261307,
        exact={"p": signed_rank_exact_rate(PAIR_N)},
    ),
    Setting(
        name="within_signed_rank_shift",
        function="withinstats",
        level="stats",
        params=(
            f"15 pairs, sds 1, correlation 0.5 (differences sd 1), mean "
            f"difference {SR_SHIFT}; r = {SR_TRUE:.6f}"
        ),
        draw=multivariate(PAIR_N, (SR_SHIFT, 0.0), np.array([[1.0, 0.5], [0.5, 1.0]])),
        outcome=_signed_rank(SR_TRUE),
        quantities=(
            Quantity(
                "cover",
                "paired rank-biserial r interval covers the true r (0.52)",
                0.95,
                2,
                "the standard error is r's null one, used at every r",
            ),
        ),
        reps=2_000,
        seed=20261308,
        reference_reps=20_000,
        reference_seed=30261308,
    ),
    Setting(
        name="within_friedman_null",
        function="withinstats",
        level="stats",
        params=f"12 subjects x 4 conditions, {_CS}, means 0",
        draw=multivariate(RM_N, (0.0,) * RM_K, compound_symmetry(RM_K)),
        outcome=_friedman,
        quantities=(
            Quantity("p", "Friedman rejects", 0.05, 2, _RANK_APPROX),
            Quantity(
                "holm",
                "Durbin-Conover pairs after Holm: " + _PAIR_HOLM,
                0.05,
                2,
                _HOLM_BOUND,
            ),
        ),
        reps=2_000,
        seed=20261309,
        reference_reps=20_000,
        reference_seed=30261309,
    ),
    Setting(
        name="within_kendall_null",
        function="withinstats",
        level="stats",
        params=f"12 subjects x 4 conditions, {_CS}, means 0; Kendall's W",
        draw=multivariate(RM_N, (0.0,) * RM_K, compound_symmetry(RM_K)),
        outcome=_kendall(0.0),
        quantities=(
            Quantity(
                "cover",
                "Kendall's W interval [lower, 1] covers 0",
                0.95,
                2,
                "W is positive in almost every resample, so the interval almost "
                "never covers no effect",
            ),
            Quantity(
                "inside",
                "Kendall's W lies inside its own interval",
                1.0,
                2,
                "the bootstrap of W is biased upward: under no effect the lower "
                "bound can exceed W itself (finding W3)",
            ),
        ),
        reps=400,
        seed=20261310,
        reference_reps=10_000,
        reference_seed=30261310,
    ),
    Setting(
        name="within_kendall_effect",
        function="withinstats",
        level="stats",
        params=(
            f"12 subjects x 4 conditions, {_CS}, effects (0, 0.25, 0.5, 0.75); "
            f"W = {W_TRUE:.6f}"
        ),
        draw=multivariate(RM_N, KENDALL_EFFECTS, compound_symmetry(RM_K)),
        outcome=_kendall(W_TRUE),
        quantities=(
            Quantity(
                "cover",
                f"Kendall's W interval covers the true W ({W_TRUE:.2f})",
                0.95,
                2,
                _BOOT,
            ),
            Quantity(
                "inside", "Kendall's W lies inside its own interval", 1.0, 2, _BOOT
            ),
        ),
        reps=400,
        seed=20261311,
        reference_reps=10_000,
        reference_seed=30261311,
    ),
    Setting(
        name="within_yuend_null",
        function="withinstats",
        level="stats",
        params=f"{_PAIR_COV_TEXT}, means 0",
        draw=multivariate(PAIR_N, (0.0, 0.0), PAIR_COV),
        outcome=_yuend,
        quantities=(
            Quantity(
                "p",
                "Yuen's dependent test rejects equal trimmed means",
                0.05,
                2,
                "t on h - 1 = 8 df for trimmed means: approximate (Wilcox 2017, 5.9.5)",
            ),
        ),
        reps=2_000,
        seed=20261312,
        reference_reps=20_000,
        reference_seed=30261312,
    ),
    Setting(
        name="within_dep_akp_null",
        function="withinstats",
        level="stats",
        params=f"{_PAIR_COV_TEXT}, means 0; dep.effect AKP",
        draw=multivariate(PAIR_N, (0.0, 0.0), PAIR_COV),
        outcome=_dep_akp(0.0),
        quantities=(Quantity("cover", "paired AKP interval covers 0", 0.95, 2, _BOOT),),
        reps=1_000,
        seed=20261313,
        reference_reps=10_000,
        reference_seed=30261313,
    ),
    Setting(
        name="within_dep_akp_effect",
        function="withinstats",
        level="stats",
        params=(
            f"{_PAIR_COV_TEXT}, mean difference {DEP_AKP_DELTA} x sqrt(3); "
            f"AKP = {DEP_AKP_DELTA}"
        ),
        draw=multivariate(PAIR_N, (DEP_AKP_DELTA * PAIR_SD_DIFF, 0.0), PAIR_COV),
        outcome=_dep_akp(DEP_AKP_DELTA),
        quantities=(
            Quantity(
                "cover",
                f"paired AKP interval covers the true {DEP_AKP_DELTA}",
                0.95,
                2,
                _BOOT,
            ),
        ),
        reps=1_000,
        seed=20261314,
        reference_reps=10_000,
        reference_seed=30261314,
    ),
    Setting(
        name="within_rmanova_null",
        function="withinstats",
        level="stats",
        params=f"12 subjects x 4 conditions, {_CS}, means 0; rmanova, rmmcp",
        draw=multivariate(RM_N, (0.0,) * RM_K, compound_symmetry(RM_K)),
        outcome=_rm_robust((0.0,) * RM_K),
        quantities=(
            Quantity(
                "p",
                "trimmed-means repeated-measures ANOVA rejects",
                0.05,
                2,
                "F for trimmed means with a Huynh-Feldt df correction at h = 8: "
                "approximate (Wilcox 2017, 8.1)",
            ),
            Quantity(
                "holm",
                "rmmcp pairs after Holm: " + _PAIR_HOLM,
                0.05,
                2,
                "Holm on trimmed-mean pairs at h - 1 = 7 df",
            ),
            Quantity(
                "ci",
                "rmmcp's pair intervals all cover 0 (simultaneously)",
                0.95,
                2,
                "Bonferroni t intervals: at or above 95% by construction",
            ),
        ),
        reps=2_000,
        seed=20261315,
        reference_reps=20_000,
        reference_seed=30261315,
    ),
    Setting(
        name="within_rmmcp_effect",
        function="withinstats",
        level="stats",
        params=f"12 subjects x 4 conditions, {_CS}, effects (0, 0.5, 1, 1.5); rmmcp",
        draw=multivariate(RM_N, RMMCP_EFFECTS, compound_symmetry(RM_K)),
        outcome=_rm_robust(RMMCP_EFFECTS),
        quantities=(
            Quantity(
                "ci",
                "rmmcp's pair intervals all cover the true differences",
                0.95,
                2,
                "Bonferroni t intervals: at or above 95% by construction",
            ),
        ),
        reps=2_000,
        seed=20261316,
        reference_reps=20_000,
        reference_seed=30261316,
    ),
    Setting(
        name="within_avg_akp_null",
        function="withinstats",
        level="stats",
        params=f"12 subjects x 3 conditions, {_CS}, means 0; wmcpAKP",
        draw=multivariate(AVG_AKP_N, (0.0,) * 3, compound_symmetry(3)),
        outcome=_avg_akp(0.0),
        quantities=(
            Quantity(
                "cover",
                "average AKP interval covers 0",
                0.95,
                2,
                "the interval is the mean of the pairs' own bounds, wider than an "
                "interval of the mean (finding W4)",
            ),
        ),
        reps=1_000,
        seed=20261317,
        reference_reps=10_000,
        reference_seed=30261317,
    ),
    Setting(
        name="within_avg_akp_effect",
        function="withinstats",
        level="stats",
        params=(
            f"12 subjects x 3 conditions, {_CS}, effects (0, 0.5, 1); "
            f"average AKP = {AVG_AKP_TRUE:.6f}"
        ),
        draw=multivariate(AVG_AKP_N, AVG_AKP_EFFECTS, compound_symmetry(3)),
        outcome=_avg_akp(AVG_AKP_TRUE),
        quantities=(
            Quantity(
                "cover",
                "average AKP interval covers the true value (-0.47)",
                0.95,
                2,
                "the interval is the mean of the pairs' own bounds, wider than an "
                "interval of the mean (finding W4)",
            ),
        ),
        reps=1_000,
        seed=20261318,
        reference_reps=10_000,
        reference_seed=30261318,
    ),
    Setting(
        name="within_bayes_null",
        function="withinstats",
        level="stats",
        params="100 pairs, sds (1, 2), correlation 0.5, means 0; JZS r 0.707",
        draw=multivariate(100, (0.0, 0.0), PAIR_COV),
        outcome=_bayes_paired,
        quantities=(
            Quantity(
                "null",
                "Bayes factor favours no difference (BF10 < 1) under no difference",
                None,
                2,
                _SANITY,
            ),
        ),
        reps=1_000,
        seed=20261319,
        reference_reps=10_000,
        reference_seed=30261319,
    ),
    Setting(
        name="within_bayes_effect",
        function="withinstats",
        level="stats",
        params=(
            "20 pairs, sds (1, 2), correlation 0.5, mean difference 0.8 x sqrt(3); "
            "JZS r 0.707"
        ),
        draw=multivariate(20, (0.8 * PAIR_SD_DIFF, 0.0), PAIR_COV),
        outcome=_bayes_paired,
        quantities=(
            Quantity(
                "alt",
                "Bayes factor favours a difference (BF10 > 1) at d_z = 0.8",
                None,
                2,
                _SANITY,
            ),
        ),
        reps=1_000,
        seed=20261320,
        reference_reps=10_000,
        reference_seed=30261320,
    ),
)


# ------------------------------------------------------------------ histostats

ONE_N = 20
ONE_DELTA = 0.5
ONE_RB_TRUE = 2.0 * phi(math.sqrt(2.0) * ONE_DELTA) - 1.0
LOGNORMAL_TM = lognormal_trimmed_mean()


def normal_sample(n: int, mean: float) -> Callable[[np.random.Generator, int], Array]:
    def draw(rng: np.random.Generator, reps: int) -> Array:
        return rng.standard_normal((reps, n)) + mean

    return draw


def lognormal_sample(n: int) -> Callable[[np.random.Generator, int], Array]:
    def draw(rng: np.random.Generator, reps: int) -> Array:
        return np.exp(rng.standard_normal((reps, n)))

    return draw


def _one_sample_api(row: Array, _: int) -> Outcome:
    import pandas as pd

    result = ip.analyze.histostats(pd.DataFrame({"x": row}), "x", bf_message=False)
    assert result.test is not None
    assert result.effect_size is not None
    es = result.effect_size
    return {
        "p": result.test.p_value < ALPHA,
        "cover": covers(es.conf_low, es.conf_high, 0.0),
    }


def _one_sample_g(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        two = st.one_sample_hedges_g(row)
        one = st.one_sample_hedges_g(row, alternative="greater")
        return {
            "p_greater": st.one_sample_t(row, alternative="greater").p_value < ALPHA,
            "two": covers(two.conf_low, two.conf_high, true),
            "one": covers(one.conf_low, one.conf_high, true),
        }

    return outcome


def _one_sample_rank(true: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        two = st.one_sample_rank_biserial(row)
        one = st.one_sample_rank_biserial(row, alternative="less")
        less = st.one_sample_signed_rank(row, alternative="less")
        return {
            "p": st.one_sample_signed_rank(row).p_value < ALPHA,
            "p_less": less.p_value < ALPHA,
            "two": covers(two.conf_low, two.conf_high, true),
            "one": covers(one.conf_low, one.conf_high, true),
        }

    return outcome


def _trimcibt(mu: float) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, seed: int) -> Outcome:
        res = st.trimmed_mean_bootstrap_t(row, mu=mu, seed=seed)
        return {
            "p": res.p_value < ALPHA,
            "cover": covers(res.conf_low, res.conf_high, mu),
        }

    return outcome


def _bayes_one_sample(row: Array, _: int) -> Outcome:
    bf10, _log = st.jzs_bf10_one_sample(row, prior_scale=0.707)
    return {"null": bf10 < 1.0, "alt": bf10 > 1.0}


_ONE_RB = (
    "Fisher-z interval with r's null standard error on the atanh scale: approximate"
)
_TRIMCIBT = (
    "symmetric bootstrap-t of the 20% trimmed mean, 2,000 resamples at n = 20: "
    "approximate (Wilcox 2017, 4.4.4)"
)

HISTOSTATS: tuple[Setting, ...] = (
    Setting(
        name="one_sample_null",
        function="histostats",
        level="api",
        params="n 20, standard normal, test value 0; ip.analyze.histostats",
        draw=normal_sample(ONE_N, 0.0),
        outcome=_one_sample_api,
        quantities=(
            Quantity(
                "p", "one-sample t rejects mean 0", 0.05, 1, "exact under normality"
            ),
            Quantity(
                "cover",
                "Hedges' g interval covers 0",
                0.95,
                1,
                "at delta 0 J times the exact interval covers 0 exactly when p >= 0.05",
            ),
        ),
        reps=2_000,
        seed=20261401,
    ),
    Setting(
        name="one_sample_null_one_sided",
        function="histostats",
        level="stats",
        params="n 20, standard normal, test value 0; alternative greater",
        draw=normal_sample(ONE_N, 0.0),
        outcome=_one_sample_g(0.0),
        quantities=(
            Quantity(
                "p_greater",
                "one-sided one-sample t rejects",
                0.05,
                1,
                "exact under normality",
            ),
            Quantity(
                "one",
                "one-sided Hedges' g interval [low, Inf] covers 0",
                0.95,
                1,
                "exact at delta 0",
            ),
        ),
        reps=2_000,
        seed=20261402,
    ),
    Setting(
        name="one_sample_effect",
        function="histostats",
        level="stats",
        params=f"n 20, normal mean {ONE_DELTA}, sd 1, test value 0",
        draw=normal_sample(ONE_N, ONE_DELTA),
        outcome=_one_sample_g(ONE_DELTA),
        quantities=(
            Quantity(
                "two",
                f"Hedges' g interval covers the true delta ({ONE_DELTA})",
                0.95,
                2,
                "J(n - 1) = 0.960 times the exact interval: exact coverage computed",
            ),
            Quantity(
                "one",
                f"one-sided Hedges' g interval [low, Inf] covers {ONE_DELTA}",
                0.95,
                2,
                "J(n - 1) pulls the lower bound toward 0: exact coverage computed",
            ),
        ),
        reps=2_000,
        seed=20261403,
        exact={
            "two": hedges_g_coverage(ONE_N, ONE_DELTA, "two-sided"),
            "one": hedges_g_coverage(ONE_N, ONE_DELTA, "greater"),
        },
    ),
    Setting(
        name="one_sample_signed_rank_null",
        function="histostats",
        level="stats",
        params="n 20, standard normal, test value 0",
        draw=normal_sample(ONE_N, 0.0),
        outcome=_one_sample_rank(0.0),
        quantities=(
            Quantity("p", "signed-rank test rejects", 0.05, 2, _RANK_APPROX),
            Quantity(
                "p_less",
                "one-sided (less) signed-rank test rejects",
                0.05,
                2,
                _RANK_APPROX,
            ),
            Quantity("two", "rank-biserial r interval covers 0", 0.95, 2, _ONE_RB),
            Quantity(
                "one",
                "one-sided rank-biserial interval [-1, high] covers 0",
                0.95,
                2,
                _ONE_RB,
            ),
        ),
        reps=2_000,
        seed=20261404,
        reference_reps=20_000,
        reference_seed=30261404,
        exact={
            "p": signed_rank_exact_rate(ONE_N),
            "p_less": signed_rank_exact_rate(ONE_N, "less"),
        },
    ),
    Setting(
        name="one_sample_rank_biserial_shift",
        function="histostats",
        level="stats",
        params=(
            f"n 20, normal mean {ONE_DELTA}, sd 1, test value 0; r = {ONE_RB_TRUE:.6f}"
        ),
        draw=normal_sample(ONE_N, ONE_DELTA),
        outcome=_one_sample_rank(ONE_RB_TRUE),
        quantities=(
            Quantity(
                "two",
                "rank-biserial r interval covers the true r (0.52)",
                0.95,
                2,
                _ONE_RB,
            ),
            Quantity(
                "one",
                "one-sided rank-biserial interval [-1, high] covers the true r",
                0.95,
                2,
                _ONE_RB,
            ),
        ),
        reps=2_000,
        seed=20261405,
        reference_reps=20_000,
        reference_seed=30261405,
    ),
    Setting(
        name="trimcibt_normal",
        function="histostats",
        level="stats",
        params="n 20, standard normal, test value 0 (the true trimmed mean)",
        draw=normal_sample(ONE_N, 0.0),
        outcome=_trimcibt(0.0),
        quantities=(
            Quantity(
                "p",
                "bootstrap-t test rejects the true trimmed mean",
                0.05,
                2,
                _TRIMCIBT,
            ),
            Quantity(
                "cover",
                "trimmed-mean interval covers the true trimmed mean",
                0.95,
                2,
                _TRIMCIBT,
            ),
        ),
        reps=1_000,
        seed=20261406,
        reference_reps=10_000,
        reference_seed=30261406,
    ),
    Setting(
        name="trimcibt_lognormal",
        function="histostats",
        level="stats",
        params=(
            f"n 20, standard lognormal, test value {LOGNORMAL_TM:.6f} (the true "
            "trimmed mean)"
        ),
        draw=lognormal_sample(ONE_N),
        outcome=_trimcibt(LOGNORMAL_TM),
        quantities=(
            Quantity(
                "p",
                "bootstrap-t test rejects the true trimmed mean (skewed data)",
                0.05,
                2,
                _TRIMCIBT,
            ),
            Quantity(
                "cover",
                "trimmed-mean interval covers the true trimmed mean (skewed data)",
                0.95,
                2,
                _TRIMCIBT,
            ),
        ),
        reps=1_000,
        seed=20261407,
        reference_reps=10_000,
        reference_seed=30261407,
    ),
    Setting(
        name="one_sample_bayes_null",
        function="histostats",
        level="stats",
        params="n 100, standard normal, test value 0; JZS r 0.707",
        draw=normal_sample(100, 0.0),
        outcome=_bayes_one_sample,
        quantities=(
            Quantity(
                "null", "Bayes factor favours mean 0 (BF10 < 1)", None, 2, _SANITY
            ),
        ),
        reps=1_000,
        seed=20261408,
        reference_reps=10_000,
        reference_seed=30261408,
    ),
    Setting(
        name="one_sample_bayes_effect",
        function="histostats",
        level="stats",
        params="n 20, normal mean 0.8, sd 1, test value 0; JZS r 0.707",
        draw=normal_sample(20, 0.8),
        outcome=_bayes_one_sample,
        quantities=(
            Quantity(
                "alt",
                "Bayes factor favours a nonzero mean (BF10 > 1) at d = 0.8",
                None,
                2,
                _SANITY,
            ),
        ),
        reps=1_000,
        seed=20261409,
        reference_reps=10_000,
        reference_seed=30261409,
    ),
)


# ------------------------------------------------------------------ scatterstats


def bivariate(
    n: int, rho: float, *, t_df: float | None = None
) -> Callable[[np.random.Generator, int], Array]:
    """n pairs of a standard bivariate normal with correlation rho, or, with
    ``t_df``, of two independent Student t variables."""

    def draw(rng: np.random.Generator, reps: int) -> Array:
        if t_df is not None:
            return rng.standard_t(t_df, size=(reps, n, 2))
        z = rng.standard_normal((reps, n, 2))
        y = rho * z[..., 0] + math.sqrt(1.0 - rho * rho) * z[..., 1]
        return np.stack([z[..., 0], y], axis=-1)

    return draw


def _pearson_api(row: Array, _: int) -> Outcome:
    import pandas as pd

    frame = pd.DataFrame({"x": row[:, 0], "y": row[:, 1]})
    result = ip.analyze.scatterstats(frame, "x", "y", bf_message=False)
    assert result.test is not None
    assert result.effect_size is not None
    es = result.effect_size
    return {
        "p": result.test.p_value < ALPHA,
        "cover": covers(es.conf_low, es.conf_high, 0.0),
    }


def _correlation(
    method: Callable[[Array, Array], st.Correlation], true: float
) -> Callable[[Array, int], Outcome]:
    def outcome(row: Array, _: int) -> Outcome:
        res = method(row[:, 0].copy(), row[:, 1].copy())
        return {
            "p": res.p_value < ALPHA,
            "cover": covers(res.conf_low, res.conf_high, true),
        }

    return outcome


def _bayes_correlation(row: Array, _: int) -> Outcome:
    bf10, _log = st.correlation_bf10(
        row[:, 0].copy(), row[:, 1].copy(), prior_scale=0.707
    )
    return {"null": bf10 < 1.0, "alt": bf10 > 1.0}


CORR_N, SPEARMAN_SMALL_N, WINCOR_N = 15, 10, 20
_FISHER = (
    "Fisher's z is asymptotically exact; at n 15 its coverage is within 0.002 of "
    "95% at rho 0, 0.3 and 0.8 (Fisher 1921: bias rho / (2 (n - 1)))"
)
_SPEARMAN_CI = (
    "Fieller, Hartley & Pearson's (1957) variance 1.06 / (n - 3) on the z scale: "
    "approximate"
)
_WINCOR = (
    "the winsorized r tested as Pearson's r, t on n - 2 df (not Wilcox's h - 2): "
    "approximate"
)
_WINCOR_CI = "Fisher-z interval at 1 / sqrt(n - 3) for a winsorized r: approximate"


def _pearson(x: Array, y: Array) -> st.Correlation:
    return st.pearson(x, y)


def _spearman(x: Array, y: Array) -> st.Correlation:
    return st.spearman(x, y)


def _wincor(x: Array, y: Array) -> st.Correlation:
    return st.winsorized_correlation(x, y)


def _scatter_settings() -> tuple[Setting, ...]:
    out: list[Setting] = [
        Setting(
            name="pearson_null",
            function="scatterstats",
            level="api",
            params=f"n {CORR_N}, bivariate normal rho 0; ip.analyze.scatterstats",
            draw=bivariate(CORR_N, 0.0),
            outcome=_pearson_api,
            quantities=(
                Quantity(
                    "p",
                    "Pearson's t rejects rho = 0",
                    0.05,
                    1,
                    "exact for a bivariate normal",
                ),
                Quantity("cover", "Fisher interval of r covers 0", 0.95, 1, _FISHER),
            ),
            reps=2_000,
            seed=20261501,
        ),
    ]
    for index, rho in enumerate((0.3, 0.8)):
        out.append(
            Setting(
                name=f"pearson_rho{rho}",
                function="scatterstats",
                level="stats",
                params=f"n {CORR_N}, bivariate normal rho {rho}",
                draw=bivariate(CORR_N, rho),
                outcome=_correlation(_pearson, rho),
                quantities=(
                    Quantity(
                        "cover",
                        f"Fisher interval of r covers rho = {rho}",
                        0.95,
                        1,
                        _FISHER,
                    ),
                ),
                reps=2_000,
                seed=20261502 + index,
            )
        )
    out.append(
        Setting(
            name="spearman_small_null",
            function="scatterstats",
            level="stats",
            params=f"n {SPEARMAN_SMALL_N}, bivariate normal rho 0",
            draw=bivariate(SPEARMAN_SMALL_N, 0.0),
            outcome=_correlation(_spearman, 0.0),
            quantities=(
                Quantity(
                    "p",
                    "Spearman's t approximation rejects rho = 0 at n 10",
                    0.05,
                    2,
                    "t approximation to the discrete permutation distribution of "
                    "rho at small n (R's cor.test(exact = FALSE))",
                ),
                Quantity(
                    "cover", "Spearman interval covers 0 at n 10", 0.95, 2, _SPEARMAN_CI
                ),
            ),
            reps=2_000,
            seed=20261504,
            reference_reps=20_000,
            reference_seed=30261504,
        )
    )
    for index, rho in enumerate((0.0, 0.3, 0.8)):
        true = spearman_rho_of_normal(rho)
        out.append(
            Setting(
                name=f"spearman_rho{rho}",
                function="scatterstats",
                level="stats",
                params=f"n {CORR_N}, bivariate normal rho {rho}; rho_S = {true:.6f}",
                draw=bivariate(CORR_N, rho),
                outcome=_correlation(_spearman, true),
                quantities=(
                    Quantity(
                        "cover",
                        f"Spearman interval covers the true rho_S = {true:.3f}",
                        0.95,
                        2,
                        _SPEARMAN_CI,
                    ),
                ),
                reps=2_000,
                seed=20261505 + index,
                reference_reps=20_000,
                reference_seed=30261505 + index,
            )
        )
    out.append(
        Setting(
            name="wincor_t3_null",
            function="scatterstats",
            level="stats",
            params=f"n {WINCOR_N}, independent Student t (3 df)",
            draw=bivariate(WINCOR_N, 0.0, t_df=3.0),
            outcome=_correlation(_wincor, 0.0),
            quantities=(
                Quantity(
                    "p",
                    "winsorized correlation test rejects (heavy tails)",
                    0.05,
                    2,
                    _WINCOR,
                ),
                Quantity(
                    "cover",
                    "winsorized r interval covers 0 (heavy tails)",
                    0.95,
                    2,
                    _WINCOR_CI,
                ),
            ),
            reps=2_000,
            seed=20261508,
            reference_reps=20_000,
            reference_seed=30261508,
        )
    )
    for index, rho in enumerate((0.0, 0.3, 0.8)):
        true = winsorized_rho_of_normal(rho)
        quantities = [
            Quantity(
                "cover",
                f"winsorized r interval covers the true rho_w = {true:.3f}",
                0.95,
                2,
                _WINCOR_CI,
            )
        ]
        if rho == 0.0:
            quantities.insert(
                0,
                Quantity(
                    "p",
                    "winsorized correlation test rejects (normal)",
                    0.05,
                    2,
                    _WINCOR,
                ),
            )
        out.append(
            Setting(
                name=f"wincor_rho{rho}",
                function="scatterstats",
                level="stats",
                params=f"n {WINCOR_N}, bivariate normal rho {rho}; rho_w = {true:.6f}",
                draw=bivariate(WINCOR_N, rho),
                outcome=_correlation(_wincor, true),
                quantities=tuple(quantities),
                reps=2_000,
                seed=20261509 + index,
                reference_reps=20_000,
                reference_seed=30261509 + index,
            )
        )
    out.extend(
        [
            Setting(
                name="correlation_bayes_null",
                function="scatterstats",
                level="stats",
                params="n 100, bivariate normal rho 0; stretched beta prior 0.707",
                draw=bivariate(100, 0.0),
                outcome=_bayes_correlation,
                quantities=(
                    Quantity(
                        "null",
                        "Bayes factor favours rho = 0 (BF10 < 1)",
                        None,
                        2,
                        _SANITY,
                    ),
                ),
                reps=1_000,
                seed=20261512,
                reference_reps=10_000,
                reference_seed=30261512,
            ),
            Setting(
                name="correlation_bayes_effect",
                function="scatterstats",
                level="stats",
                params="n 30, bivariate normal rho 0.5; stretched beta prior 0.707",
                draw=bivariate(30, 0.5),
                outcome=_bayes_correlation,
                quantities=(
                    Quantity(
                        "alt",
                        "Bayes factor favours rho != 0 (BF10 > 1) at rho 0.5",
                        None,
                        2,
                        _SANITY,
                    ),
                ),
                reps=1_000,
                seed=20261513,
                reference_reps=10_000,
                reference_seed=30261513,
            ),
        ]
    )
    return tuple(out)


SCATTERSTATS = _scatter_settings()

ALL: tuple[Setting, ...] = BETWEEN + WITHIN + HISTOSTATS + SCATTERSTATS
BY_NAME = {s.name: s for s in ALL}
