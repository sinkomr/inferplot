"""Independent reference implementations from the November 2026 design audit.

These are cross-checks, not production code. Each function is written from the
published method, uses only NumPy and SciPy, and has a known-answer check below
whose source is stated. Production code in ``src/inferplot`` must agree with the
R oracle first; agreeing with this file is a second, independent check.

Run the known-answer checks with:  python -I tools/reference/stats_reference.py

Sources of known answers:
  * ggstatsplot 1.1.1 output frozen in the plotsalot repository (MIT), dataset
    ``m3-between``: https://github.com/joshuamyers22/plotsalot
  * pingouin's documented JZS example (validated there against JASP and the
    BayesFactor R package):
    https://pingouin-stats.org/build/html/generated/pingouin.bayesfactor_ttest.html
  * plotsalot 1.0.0 golden results (cross-implementation agreement only; not R)
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special, stats

FloatArray = NDArray[np.float64]


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm (1979) step-down adjustment, identical to R's p.adjust(method="holm")."""
    m = len(p_values)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted.tolist()


def welch_anova(groups: Sequence[ArrayLike]) -> tuple[float, float, float, float]:
    """Welch (1951) heteroscedastic one-way ANOVA, as R's oneway.test(var.equal=FALSE).

    Returns (F, df1, df2, p).
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    k = len(arrays)
    n = np.array([a.size for a in arrays], dtype=float)
    means = np.array([a.mean() for a in arrays])
    variances = np.array([a.var(ddof=1) for a in arrays])
    weights = n / variances
    total_weight = weights.sum()
    weighted_mean = (weights * means).sum() / total_weight
    between = (weights * (means - weighted_mean) ** 2).sum() / (k - 1)
    lam = (((1 - weights / total_weight) ** 2) / (n - 1)).sum()
    f_stat = between / (1 + 2 * (k - 2) / (k**2 - 1) * lam)
    df2 = (k**2 - 1) / (3 * lam)
    return f_stat, float(k - 1), df2, float(stats.f.sf(f_stat, k - 1, df2))


def omega2_partial_from_f(f_stat: float, df1: float, df2: float) -> float:
    """Partial omega squared from an F statistic, floored at zero.

    Same conversion effectsize applies to oneway.test output; reproduces
    ggstatsplot 1.1.1's omega estimate on the m3-between dataset.
    """
    return max(0.0, (f_stat - 1) * df1 / (f_stat * df1 + df2 + 1))


def games_howell(groups: Sequence[ArrayLike]) -> list[tuple[int, int, float, float]]:
    """Games & Howell (1976) pairwise test via the studentized range distribution.

    Returns (i, j, q, unadjusted_p) for each pair i < j, with q unsigned. ggstatsplot
    additionally applies Holm to these p-values; use ``holm`` on the result to
    reproduce it. ggstatsplot 1.1.4 signs q as mean(group j) - mean(group i)
    (tests/test_reference_vs_oracle.py); the m3-between known answers below cannot
    show a sign, because every pair there has mean(group j) > mean(group i).
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    k = len(arrays)
    out: list[tuple[int, int, float, float]] = []
    for i, j in itertools.combinations(range(k), 2):
        a, b = arrays[i], arrays[j]
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        se = math.sqrt(va + vb)
        df = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
        q = abs(a.mean() - b.mean()) / se * math.sqrt(2)
        out.append((i, j, q, float(stats.studentized_range.sf(q, k, df))))
    return out


def hedges_j(df: float) -> float:
    """Exact small-sample bias correction for Cohen's d (Hedges 1981)."""
    return math.exp(
        special.gammaln(df / 2) - 0.5 * math.log(df / 2) - special.gammaln((df - 1) / 2)
    )


def _trimmed_kernel(x: FloatArray, tr: float) -> tuple[float, float, int]:
    """Return (trimmed mean, Yuen squared standard error d, h) for one sample."""
    n = x.size
    g = math.floor(tr * n)
    h = n - 2 * g
    ordered = np.sort(x)
    winsorized = np.clip(x, ordered[g], ordered[n - g - 1])
    d = (n - 1) * winsorized.var(ddof=1) / (h * (h - 1))
    return float(stats.trim_mean(x, tr)), d, h


def yuen(a: ArrayLike, b: ArrayLike, tr: float = 0.2) -> tuple[float, float, float]:
    """Yuen (1974) two-sample trimmed-means test. Returns (t, df, p)."""
    ma, da, ha = _trimmed_kernel(np.asarray(a, dtype=float), tr)
    mb, db, hb = _trimmed_kernel(np.asarray(b, dtype=float), tr)
    t = (ma - mb) / math.sqrt(da + db)
    df = (da + db) ** 2 / (da**2 / (ha - 1) + db**2 / (hb - 1))
    return t, df, float(2 * stats.t.sf(abs(t), df))


def t1way(
    groups: Sequence[ArrayLike], tr: float = 0.2
) -> tuple[float, float, float, float]:
    """Wilcox's heteroscedastic trimmed-means ANOVA. Returns (F, df1, df2, p)."""
    kernels = [_trimmed_kernel(np.asarray(g, dtype=float), tr) for g in groups]
    j = len(kernels)
    means = np.array([k[0] for k in kernels])
    weights = np.array([1 / k[1] for k in kernels])
    h = np.array([k[2] for k in kernels], dtype=float)
    u = weights.sum()
    grand = (weights * means).sum() / u
    a = (weights * (means - grand) ** 2).sum() / (j - 1)
    lam = (((1 - weights / u) ** 2) / (h - 1)).sum()
    f_stat = a / (1 + 2 * (j - 2) / (j**2 - 1) * lam)
    df2 = (j**2 - 1) / (3 * lam)
    return f_stat, float(j - 1), df2, float(stats.f.sf(f_stat, j - 1, df2))


def rm_anova_gg(y: ArrayLike) -> dict[str, float]:
    """One-way repeated-measures ANOVA with Greenhouse-Geisser (1959) correction.

    ``y`` has one row per subject and one column per condition, complete cases only.
    """
    data = np.asarray(y, dtype=float)
    n, k = data.shape
    grand = data.mean()
    ss_condition = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_subject = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_error = ((data - grand) ** 2).sum() - ss_condition - ss_subject
    df1, df2 = k - 1, (k - 1) * (n - 1)
    f_stat = (ss_condition / df1) / (ss_error / df2)
    centering = np.eye(k) - 1.0 / k
    centered_cov = centering @ np.cov(data, rowvar=False) @ centering
    epsilon = np.trace(centered_cov) ** 2 / ((k - 1) * (centered_cov**2).sum())
    epsilon = max(float(epsilon), 1.0 / (k - 1))
    return {
        "F": f_stat,
        "epsilon": epsilon,
        "df1": df1 * epsilon,
        "df2": df2 * epsilon,
        "p": float(stats.f.sf(f_stat, df1 * epsilon, df2 * epsilon)),
    }


def chisq_ncp_interval(
    chisq: float, df: float, level: float = 0.95
) -> tuple[float, float]:
    """Two-sided CI for a chi-square noncentrality parameter by test inversion."""
    alpha = 1 - level

    def bound(target: float) -> float:
        def gap(lam: float) -> float:
            return float(stats.ncx2.cdf(chisq, df, lam)) - target

        if float(stats.chi2.cdf(chisq, df)) < target:
            return 0.0
        upper = max(10.0, 10 * chisq)
        while gap(upper) > 0:
            upper *= 2
        return float(optimize.brentq(gap, 1e-12, upper))

    return bound(1 - alpha / 2), bound(alpha / 2)


def jzs_bf10_t(t: float, n_effective: float, df: float, r: float = 0.707) -> float:
    """JZS Bayes factor for a t statistic (Rouder et al. 2009), Cauchy scale r.

    One-sample or paired: n_effective = n, df = n - 1.
    Two-sample: n_effective = n1 * n2 / (n1 + n2), df = n1 + n2 - 2.
    The default r = 0.707 is ggstatsplot's numeric bf.prior default, not sqrt(2)/2;
    the two give Bayes factors that differ in the fifth significant figure.

    BF10 is the integral over g of the alternative's t likelihood divided by the
    null's, times the inverse-gamma(1/2, 1/2) density of g. Taking the ratio inside
    the integral (in logs) keeps the integrand near the size of BF10 even when both
    likelihoods underflow, so a purely relative quadrature tolerance applies.
    """
    log_null = -(df + 1) / 2 * math.log1p(t * t / df)

    def integrand(g: float) -> float:
        if g <= 0.0:
            return 0.0
        scale = 1 + n_effective * g * r**2
        log_alt = -0.5 * math.log(scale) - (df + 1) / 2 * math.log1p(
            t * t / (scale * df)
        )
        log_prior = -0.5 * math.log(2 * math.pi) - 1.5 * math.log(g) - 1 / (2 * g)
        return math.exp(log_alt - log_null + log_prior)

    value, _ = integrate.quad(
        integrand, 0.0, math.inf, epsabs=0.0, epsrel=1e-12, limit=500
    )
    return value


def _known_answer_checks() -> None:
    def close(name: str, got: float, want: float, rtol: float, source: str) -> None:
        ok = math.isclose(got, want, rel_tol=rtol, abs_tol=0.0)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {name:44s} got {got:.12g}  want {want:.12g}  ({source})")
        if not ok:
            raise SystemExit(1)

    gg = "ggstatsplot 1.1.1 via plotsalot oracle"
    m3 = [
        np.array([1.0, 2, 4, 8]),
        np.array([2.0, 3, 5, 7]),
        np.array([8.0, 9, 11, 15]),
    ]
    f_stat, df1, df2, _ = welch_anova(m3)
    close("Welch F, m3-between", f_stat, 6.25235033101128, 1e-12, gg)
    close("Welch df2, m3-between", df2, 5.8315699017133, 1e-12, gg)
    close(
        "partial omega2, m3-between",
        omega2_partial_from_f(f_stat, df1, df2),
        0.543264050189883,
        1e-12,
        gg,
    )
    gh = games_howell(m3)
    gh_holm = holm([p for *_, p in gh])
    for (i, j, q, _), adj, want_q, want_p in zip(
        gh,
        gh_holm,
        (0.371390676354104, 4.52240797728859, 4.82807879260335),
        (0.962951855907413, 0.112912416213043, 0.112912416213043),
        strict=True,
    ):
        close(f"Games-Howell q, pair {i}-{j}", q, want_q, 1e-12, gg)
        close(
            f"Games-Howell + Holm p, pair {i}-{j}",
            adj,
            want_p,
            1e-6,
            gg + "; R ptukey vs SciPy differ ~2.6e-7",
        )

    close(
        "JZS BF10, two-sample t=3.5, n=20+20, r=0.707",
        jzs_bf10_t(3.5, 10.0, 38.0),
        26.743,
        2e-5,
        "pingouin docs, printed to 3 decimals",
    )

    plotsalot = "plotsalot 1.0.0 golden output (not R)"
    between = [
        np.array([float(i + g * 1.5) + float(i % 2) * 0.2 for i in range(6)])
        for g in range(3)
    ]
    close(
        "t1way F, golden between frame",
        t1way(between)[0],
        2.838785046728971,
        1e-12,
        plotsalot,
    )
    close(
        "Yuen t, golden between a-b",
        yuen(between[0], between[1])[0],
        -1.2558275374183818,
        1e-12,
        plotsalot,
    )
    within = np.array(
        [[float(s + c) + float((s % 3) * c) * 0.2 for c in range(3)] for s in range(8)]
    )
    close(
        "RM ANOVA F, golden within frame",
        rm_anova_gg(within)["F"],
        396.48717948719354,
        1e-10,
        plotsalot,
    )
    table = np.array([[6.0, 5.0], [8.0, 7.0]])
    chisq = stats.chi2_contingency(table, correction=False).statistic
    _, upper = chisq_ncp_interval(chisq, 1.0)
    close(
        "Cramer's V upper 95% bound (sqrt(ncp/N))",
        math.sqrt(upper / table.sum()),
        0.22711259502032116,
        1e-9,
        plotsalot,
    )
    close(
        "Hedges J(10)",
        hedges_j(10.0),
        math.gamma(5.0) / (math.sqrt(5.0) * math.gamma(4.5)),
        1e-12,
        "Hedges 1981 gamma ratio via math.gamma",
    )


if __name__ == "__main__":
    _known_answer_checks()
