"""Independent cross-checks of the oracle fixtures against tools/reference and SciPy.

These tests do not exercise inferplot. They confirm that the fixture pipeline (CSV,
R, JSON, loader) carries numbers that an independent implementation of the
published method reproduces, and they pin down which variant R ran: where a
plausible variant exists, a negative control shows that it fails.

Every check follows the case's own arguments: `conf.level` (read from each row),
`alternative`, `p.adjust.method`, `bf.prior`, a declared level order (groups come in
the order R analyzed them), and the outputs a case does not draw (no subtitle with
`results.subtitle = FALSE`, no Bayes caption with `bf.message = FALSE`, no pairs with
`pairwise.display = "none"`): a check runs on the cases that have its quantity, and
`test_absent_outputs` asserts the quantity is absent from the others.

Two groups (alternative two-sided, greater or less):
    parametric     Welch t (SciPy); Hedges' g with the unpooled SD, the exact
                   correction at the Welch df and a noncentral-t CI (inline); the
                   Bayes caption's JZS Bayes factor (reference)
    nonparametric  Mann-Whitney U, always the normal approximation with continuity
                   correction (SciPy); rank-biserial r with a Fisher-z CI (inline)
    robust         Yuen, |t| and p as WRS2 computes it (reference)
    bayes          JZS Bayes factor with Student's t and r = 0.707 (reference)

Three or more groups:
    parametric     Welch ANOVA, partial omega squared with its one-sided
                   noncentral-F CI (inline), Games-Howell then the case's
                   p.adjust method (reference; p against SciPy's studentized range)
    nonparametric  Kruskal-Wallis (SciPy); Dunn then Holm (inline, Dunn 1964)
    robust         trimmed-means ANOVA (reference t1way); lincon pairwise Yuen tests
                   then Holm, with the Studentized maximum modulus CI (inline)
    bayes          JZS Bayes factor of every pair (reference); the Bayes ANOVA
                   itself arrives in v0.2
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, optimize, special, stats
from tools.reference import stats_reference as ref

from tests.oracle_fixtures import (
    OracleCase,
    Row,
    cases_for,
    number,
    oracle_params,
    probe,
    r_double,
    r_level_label,
    read_dataset,
    require_cases,
)
from tests.tolerances import assert_close, is_close

FloatArray = NDArray[np.float64]

# R's `alternative` values and SciPy's names for them.
SCIPY_ALTERNATIVE: dict[str, Literal["two-sided", "less", "greater"]] = {
    "two.sided": "two-sided",
    "greater": "greater",
    "less": "less",
}
# ggstatsplot's bf.prior default, and the prior of every pairwise Bayes factor.
JZS_PRIOR = 0.707

# What each output needs, in ggbetweenstats 1.1.4 terms (SPEC.md section 2).
Output = Literal["subtitle", "caption", "pairwise"]


def _draws(case: OracleCase, output: Output) -> bool:
    """Whether ggbetweenstats computes ``output`` for this case's arguments."""
    subtitle = case.args.get("results.subtitle", True) is not False
    if output == "subtitle":
        return subtitle
    if output == "caption":
        return (
            subtitle
            and case.type == "parametric"
            and case.args.get("bf.message", True) is not False
        )
    return (
        len(case.analysis_levels) > 2
        and case.args.get("pairwise.display", "significant") != "none"
    )


def _params(
    type_: str, *, two_groups: bool, needs: Output | None = "subtitle"
) -> list[Any]:
    """Cases of one type and design that have the output a check reads."""
    cases = [
        case
        for case in cases_for("ggbetweenstats")
        if case.type == type_
        and (len(case.analysis_levels) == 2) == two_groups
        and (needs is None or _draws(case, needs))
    ]
    design = "two groups" if two_groups else "three or more groups"
    require_cases(cases, f"ggbetweenstats {type_}, {design}, with {needs}")
    return [pytest.param(case, id=case.name) for case in cases]


def _prior(case: OracleCase) -> float:
    return float(case.args.get("bf.prior", JZS_PRIOR))


def _groups(case: OracleCase) -> list[FloatArray]:
    """The groups R analyzed, in R's level order (declared order when declared)."""
    frame = read_dataset(case)
    x, y = case.args["x"], case.args["y"]
    used = frame[[x, y]].dropna()
    labels = used[x].map(r_level_label)
    declared = case.factor_levels.get(x)
    if declared is not None:
        # A declared order is kept, minus levels without complete rows.
        present = set(labels)
        assert case.analysis_levels == tuple(v for v in declared if v in present)
    else:
        assert case.analysis_levels == tuple(sorted(set(labels)))
    return [
        used.loc[labels == level, y].to_numpy(dtype=np.float64)
        for level in case.analysis_levels
    ]


def _value(row: Row, column: str) -> float:
    assert column in row, f"no {column!r} column; columns are {sorted(row)}"
    return number(row[column])


def _r_upper_tail(p: float) -> float:
    """An upper-tail probability as R gets it from ``1 - cdf``.

    WRS2 computes the p-values of t1way and lincon (and presumably yuen) as 1 - pt or
    1 - pf, so they are the tail rounded to the spacing of doubles just below 1
    (2**-53).
    """
    return 1.0 - (1.0 - p)


def _alternative(case: OracleCase) -> str:
    return str(case.args.get("alternative", "two.sided"))


def _one_sided_quantile(case: OracleCase, row: Row) -> float:
    """Normal quantile for the CI's level: alpha/2 per side, or alpha when one-sided."""
    alpha = 1 - _value(row, "conf.level")
    tail = alpha / 2 if _alternative(case) == "two.sided" else alpha
    return float(stats.norm.isf(tail))


@pytest.mark.parametrize("case", _params("parametric", two_groups=True))
def test_welch_t(case: OracleCase) -> None:
    a, b = _groups(case)
    result = stats.ttest_ind(
        a, b, equal_var=False, alternative=SCIPY_ALTERNATIVE[_alternative(case)]
    )
    row = case.subtitle_data
    assert row["alternative"] == _alternative(case)
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="Welch t"
    )
    assert_close(
        "closed_form", float(result.df), _value(row, "df.error"), what="Welch df"
    )
    assert_close(
        "closed_form", float(result.pvalue), _value(row, "p.value"), what="Welch p"
    )


def _ncp_at(t: float, df: float, cdf: float) -> float:
    """Noncentrality parameter at which a noncentral t(df) has P(T <= t) = ``cdf``."""

    def excess(ncp: float) -> float:
        return float(stats.nct.cdf(t, df, ncp)) - cdf

    # P(T <= t) falls as the noncentrality parameter grows. Widen the bracket a
    # step at a time: far in the tails SciPy's nct.cdf returns NaN.
    low, high = t - 1.0, t + 1.0
    while excess(low) <= 0.0:
        low -= 1.0
    while excess(high) >= 0.0:
        high += 1.0
    return float(optimize.brentq(excess, low, high, xtol=1e-14, rtol=1e-15))


@pytest.mark.parametrize("case", _params("parametric", two_groups=True))
def test_hedges_g(case: OracleCase) -> None:
    # effectsize::hedges_g(pooled_sd = FALSE), as statsExpressions 2.2.1 calls it:
    #   d = (mean1 - mean2) / s with s = sqrt((var1 + var2) / 2), not the pooled SD
    #   g = d x J(df_W): Hedges' exact gamma-ratio correction at the Welch df
    #   CI: the noncentral-t interval for the noncentrality parameter at the Welch t
    #   and df_W, each bound times (Welch SE / s) x J(df_W). A one-sided
    #   `alternative` gives a one-sided interval with the far bound infinite.
    a, b = _groups(case)
    n1, n2 = a.size, b.size
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    difference = float(a.mean() - b.mean())
    s = math.sqrt((v1 + v2) / 2)
    w1, w2 = v1 / n1, v2 / n2
    se = math.sqrt(w1 + w2)
    df = (w1 + w2) ** 2 / (w1**2 / (n1 - 1) + w2**2 / (n2 - 1))
    j = ref.hedges_j(df)
    row = case.subtitle_data
    assert row["effectsize"] == "Hedges' g"
    g = _value(row, "estimate")
    assert_close("closed_form", difference / s * j, g, what="Hedges' g")

    # Negative controls: the pooled SD (equal to s only for equal group sizes) and
    # the plain average of the two SDs are both off.
    pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if n1 != n2:
        assert not is_close("closed_form", difference / pooled * j, g)
    assert not is_close(
        "closed_form", difference / ((math.sqrt(v1) + math.sqrt(v2)) / 2) * j, g
    )

    alpha = 1 - _value(row, "conf.level")
    side = _alternative(case)
    tail = alpha / 2 if side == "two.sided" else alpha
    scale = se / s * j
    t = difference / se
    low = -math.inf if side == "less" else _ncp_at(t, df, 1 - tail) * scale
    high = math.inf if side == "greater" else _ncp_at(t, df, tail) * scale
    assert_close("noncentral_ci", low, _value(row, "conf.low"), what="g CI low")
    assert_close("noncentral_ci", high, _value(row, "conf.high"), what="g CI high")


@pytest.mark.parametrize("case", _params("parametric", two_groups=False))
def test_welch_anova_and_omega(case: OracleCase) -> None:
    _welch_anova_checks(_groups(case), case.subtitle_data)


def _ncf_cdf(x: float, df1: float, df2: float, ncp: float) -> float:
    """P(F' <= x) for the noncentral F(df1, df2, ncp), as a Poisson mixture.

    P = sum_j Pois(j; ncp / 2) I_z(df1 / 2 + j, df2 / 2) with z = df1 x / (df1 x +
    df2) (Johnson, Kotz & Balakrishnan 1995, ch. 30, eq. 30.4), summed over j within
    12 Poisson SDs of the mode (the rest is below 1e-30). SciPy's own ncfdtr is
    off by up to 1.6e-4 relative in the omega squared bounds at the SciPy floor
    (1.11.1, InsectSprays), so the reference uses only the regularized incomplete
    beta.
    """
    z = df1 * x / (df1 * x + df2)
    if ncp == 0.0:
        return float(special.betainc(df1 / 2, df2 / 2, z))
    half = ncp / 2
    width = int(12 * math.sqrt(half) + 30)
    mode = int(half)
    j = np.arange(max(0, mode - width), mode + width + 1, dtype=np.float64)
    log_weights = -half + j * math.log(half) - special.gammaln(j + 1)
    terms = np.exp(log_weights) * special.betainc(df1 / 2 + j, df2 / 2, z)
    return float(np.sum(terms))


def _f_ncp(f_obs: float, df1: float, df2: float, prob: float) -> float:
    """Noncentrality lambda >= 0 with P(F'(df1, df2, lambda) <= f_obs) = ``prob``.

    0 when even the central F gives a probability at or below ``prob``.
    """

    def excess(ncp: float) -> float:
        return _ncf_cdf(f_obs, df1, df2, ncp) - prob

    if f_obs <= 0.0 or excess(0.0) <= 0.0:
        return 0.0
    high = 1.0
    while excess(high) > 0.0:
        high *= 2.0
    return float(optimize.brentq(excess, 0.0, high, xtol=1e-14, rtol=1e-15))


@pytest.mark.parametrize("case", _params("parametric", two_groups=False))
def test_omega_squared_ci(case: OracleCase) -> None:
    # effectsize::omega_squared(oneway.test(...), ci = conf.level), as statsExpressions
    # 2.2.1 calls it: alternative "greater", so the upper bound is 1. The lower bound
    # is not the noncentral-F interval at the observed F: the estimate is converted
    # back to the F value it would be as a partial eta squared,
    # F_w = omega df2 / ((1 - omega) df1), the noncentral F is inverted there,
    # P(F'(df1, df2, lambda) <= F_w) = conf.level, and lambda / (lambda + df2) is the
    # bound (black-box finding, Phase 1). A negative control inverts at the
    # observed F instead.
    row = case.subtitle_data
    assert row["effectsize"] == "Omega2"
    f_stat = _value(row, "statistic")
    df1, df2 = _value(row, "df"), _value(row, "df.error")
    omega = _value(row, "estimate")
    level = _value(row, "conf.level")
    assert level == float(case.args.get("conf.level", 0.95))
    assert _value(row, "conf.high") == 1.0
    f_w = omega * df2 / ((1 - omega) * df1) if omega > 0 else 0.0
    lam = _f_ncp(f_w, df1, df2, level)
    low = lam / (lam + df2)
    assert_close(
        "noncentral_ci", low, _value(row, "conf.low"), what="omega squared CI low"
    )
    at_observed = _f_ncp(f_stat, df1, df2, level)
    if low > 0:
        assert not is_close(
            "noncentral_ci", at_observed / (at_observed + df2), _value(row, "conf.low")
        )


def _p_adjust(p_values: Sequence[float], method: str) -> list[float]:
    """R's p.adjust methods from their published definitions (independent reference).

    Bonferroni; Holm (1979) step-down; Hochberg (1988) step-up; Benjamini & Hochberg
    (1995) ("BH", alias "fdr"); Benjamini & Yekutieli (2001) ("BY", the BH quotients
    times sum(1/k, k = 1..m)); Hommel (1988) as closed testing with Simes local tests
    (Wright 1992): the adjusted p of H_i is the largest Simes p over intersections
    that contain H_i, and since the Simes p only grows with each p-value, the
    largest one for a given size takes the other largest p-values. Missing values
    (NaN) stay missing and m counts the rest, as R's default ``n`` does.
    """
    values = [float(v) for v in p_values]
    kept = [v for v in values if not math.isnan(v)]
    adjusted = iter(_p_adjust_complete(kept, method))
    return [math.nan if math.isnan(v) else next(adjusted) for v in values]


def _p_adjust_complete(p: list[float], method: str) -> list[float]:
    m = len(p)
    if m == 0 or method == "none":
        return list(p)
    if method == "bonferroni":
        return [min(1.0, m * v) for v in p]
    if method == "holm":
        return ref.holm(p)
    order = sorted(range(m), key=lambda i: p[i])  # ascending
    out = [0.0] * m
    if method in {"hochberg", "BH", "fdr", "BY"}:
        harmonic = math.fsum(1.0 / k for k in range(1, m + 1)) if method == "BY" else 1
        running = math.inf
        for rank in range(m, 0, -1):  # from the largest p down
            index = order[rank - 1]
            if method == "hochberg":
                candidate = (m - rank + 1) * p[index]
            else:
                candidate = harmonic * m / rank * p[index]
            running = min(running, candidate)
            out[index] = min(1.0, running)
        return out
    if method == "hommel":
        for i in range(m):
            others = sorted((p[j] for j in range(m) if j != i), reverse=True)
            worst = 0.0
            for size in range(1, m + 1):
                members = sorted([p[i], *others[: size - 1]])
                simes = min(size * v / k for k, v in enumerate(members, start=1))
                worst = max(worst, simes)
            out[i] = min(1.0, worst)
        return out
    raise ValueError(f"unknown p.adjust method {method!r}")


# insight::format_capitalize of the method, with BH and fdr shown as FDR
# (statsExpressions 2.2.1 `.pairwise_p_adjust_expr`).
P_ADJUST_LABELS = {
    "holm": "Holm",
    "hochberg": "Hochberg",
    "hommel": "Hommel",
    "bonferroni": "Bonferroni",
    "BH": "FDR",
    "BY": "BY",
    "fdr": "FDR",
    "none": "None",
}


def _hex(text: str) -> float:
    """A probe double that must not be R's NA."""
    value = r_double(text)
    assert value is not None, "unexpected NA"
    return value


def _welch_anova_checks(groups: list[FloatArray], row: Row) -> None:
    f_stat, df1, df2, p = ref.welch_anova(groups)
    assert_close("closed_form", f_stat, _value(row, "statistic"), what="Welch F")
    assert_close("closed_form", df1, _value(row, "df"), what="Welch df1")
    assert_close("closed_form", df2, _value(row, "df.error"), what="Welch df2")
    assert_close("closed_form", p, _value(row, "p.value"), what="Welch p")
    omega = ref.omega2_partial_from_f(f_stat, df1, df2)
    assert_close(
        "closed_form", omega, _value(row, "estimate"), what="partial omega squared"
    )


@pytest.mark.parametrize("case", oracle_params("ggbetweenstats", failed=True))
def test_failed_case_partial_subtitle(case: OracleCase) -> None:
    # ggstatsplot failed after computing the subtitle (edge_n2: Welch df below 2 in
    # every pair stops the pairwise table); what it computed is still right.
    rows = case.partial("subtitle")
    assert len(rows) == 1
    _welch_anova_checks(_groups(case), rows[0])


@pytest.mark.parametrize(
    "case", _params("parametric", two_groups=False, needs="pairwise")
)
def test_games_howell(case: OracleCase) -> None:
    # ggstatsplot 1.1.4 reports q with the sign of mean(group2) - mean(group1);
    # the reference returns |q|. The adjustment is the case's p.adjust.method over
    # all k(k - 1)/2 pairs.
    groups = _groups(case)
    method = str(case.args.get("p.adjust.method", "holm"))
    levels = case.analysis_levels
    pairs = ref.games_howell(groups)
    adjusted = _p_adjust([p_raw for *_, p_raw in pairs], method)
    if method == "holm":
        assert adjusted == ref.holm([p_raw for *_, p_raw in pairs])
    expected = {
        (levels[i], levels[j]): (
            math.copysign(q, float(groups[j].mean() - groups[i].mean())),
            p_adj,
        )
        for (i, j, q, _), p_adj in zip(pairs, adjusted, strict=True)
    }
    assert len(case.pairwise) == len(expected)
    for pair in case.pairwise:
        q, p_adj = expected[(str(pair["group1"]), str(pair["group2"]))]
        what = f"Games-Howell {pair['group1']}-{pair['group2']}"
        assert pair["test"] == "Games-Howell"
        assert pair["p.adjust.method"] == P_ADJUST_LABELS[method]
        assert_close("closed_form", q, number(pair["statistic"]), what=f"{what} q")
        # The reference uses SciPy's studentized range; R's ptukey has its own
        # absolute error in the far tail (games_howell_p, DECISIONS.md D31).
        assert_close(
            "games_howell_p",
            p_adj,
            number(pair["p.value"]),
            what=f"{what} {method} p",
        )


@pytest.mark.parametrize("case", _params("nonparametric", two_groups=True))
def test_mann_whitney(case: OracleCase) -> None:
    # statsExpressions 2.2.1 calls wilcox.test(exact = FALSE): always the normal
    # approximation with continuity correction, even without ties and with fewer
    # than 50 values per group, where R's own default would be exact. The method
    # string in the fixture does not say which was used.
    a, b = _groups(case)
    side = SCIPY_ALTERNATIVE[_alternative(case)]
    result = stats.mannwhitneyu(
        a, b, alternative=side, method="asymptotic", use_continuity=True
    )
    row = case.subtitle_data
    assert row["alternative"] == _alternative(case)
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="W"
    )
    assert_close("closed_form", float(result.pvalue), _value(row, "p.value"), what="p")
    ties = np.unique(np.concatenate([a, b])).size < a.size + b.size
    if not ties and a.size < 50 and b.size < 50:
        # Negative control: here R's default rule would have used the exact p.
        exact = stats.mannwhitneyu(a, b, alternative=side, method="exact")
        assert not is_close("closed_form", float(exact.pvalue), _value(row, "p.value"))


@pytest.mark.parametrize("case", _params("nonparametric", two_groups=True))
def test_rank_biserial(case: OracleCase) -> None:
    # effectsize::rank_biserial, as statsExpressions 2.2.1 calls it: r = 2U/(n1 n2)
    # - 1 with U the Mann-Whitney U of group 1 (Glass 1965; Kerby 2014). CI: tanh of
    # atanh(r) -/+ z x sqrt((n1 + n2 + 1) / (3 n1 n2)), with ties ignored; a
    # one-sided `alternative` gives a one-sided interval ending at -1 or 1.
    a, b = _groups(case)
    n1, n2 = a.size, b.size
    u = float(stats.mannwhitneyu(a, b, method="asymptotic").statistic)
    r = 2 * u / (n1 * n2) - 1
    row = case.subtitle_data
    assert row["effectsize"] == "r (rank biserial)"
    assert_close("closed_form", r, _value(row, "estimate"), what="rank-biserial r")
    z = math.atanh(r)
    se = math.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
    q = _one_sided_quantile(case, row)
    side = _alternative(case)
    low = -1.0 if side == "less" else math.tanh(z - q * se)
    high = 1.0 if side == "greater" else math.tanh(z + q * se)
    assert_close("closed_form", low, _value(row, "conf.low"), what="r CI low")
    assert_close("closed_form", high, _value(row, "conf.high"), what="r CI high")


@pytest.mark.parametrize("case", _params("nonparametric", two_groups=False))
def test_kruskal_wallis(case: OracleCase) -> None:
    groups = _groups(case)
    result = stats.kruskal(*groups)
    row = case.subtitle_data
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="H"
    )
    assert_close("closed_form", len(groups) - 1.0, _value(row, "df.error"), what="H df")
    assert_close("closed_form", float(result.pvalue), _value(row, "p.value"), what="p")


@pytest.mark.parametrize("case", _params("robust", two_groups=True))
def test_yuen(case: OracleCase) -> None:
    # WRS2::yuen reports |t|: mtcars am and edge_k2 have the smaller trimmed mean in
    # group 1, yet a positive t, while the AKP effect size beside it is negative.
    a, b = _groups(case)
    t, df, _ = ref.yuen(a, b, tr=0.2)
    row = case.subtitle_data
    assert_close("closed_form", abs(t), _value(row, "statistic"), what="Yuen |t|")
    if t < 0:
        assert _value(row, "estimate") < 0, "the effect size keeps its sign"
    assert_close("closed_form", df, _value(row, "df.error"), what="Yuen df")
    p = 2 * _r_upper_tail(float(stats.t.sf(abs(t), df)))
    assert_close("wrs2_p", p, _value(row, "p.value"), what="Yuen p", atol_scale=2)


@pytest.mark.parametrize("case", _params("robust", two_groups=False))
def test_t1way(case: OracleCase) -> None:
    f_stat, df1, df2, _ = ref.t1way(_groups(case), tr=0.2)
    row = case.subtitle_data
    assert_close("closed_form", f_stat, _value(row, "statistic"), what="t1way F")
    assert_close("closed_form", df1, _value(row, "df"), what="t1way df1")
    assert_close("closed_form", df2, _value(row, "df.error"), what="t1way df2")
    p = _r_upper_tail(float(stats.f.sf(f_stat, df1, df2)))
    assert_close("wrs2_p", p, _value(row, "p.value"), what="t1way p")


def _check_jzs(
    row: Row, a: FloatArray, b: FloatArray, *, prior: float, what: str
) -> None:
    """BayesFactor::ttestBF on two groups: Student's t, Cauchy scale ``prior``."""
    assert number(row["prior.scale"]) == prior, f"{what}: {row['prior.scale']}"
    bf10 = _value(row, "bf10")
    n_eff = a.size * b.size / (a.size + b.size)
    df = float(a.size + b.size - 2)
    student = float(stats.ttest_ind(a, b, equal_var=True).statistic)
    got = ref.jzs_bf10_t(student, n_eff, df, r=prior)
    assert_close("ttest_bayes_factor", got, bf10, what=f"{what} BF10")
    # Negative controls: another prior (BayesFactor's own "medium" sqrt(2)/2 for
    # ggstatsplot's 0.707, and 0.707 for any other scale), and Welch's t (which
    # equals Student's t only for equal group sizes), are both off.
    other = math.sqrt(2) / 2 if prior == JZS_PRIOR else JZS_PRIOR
    wrong_prior = ref.jzs_bf10_t(student, n_eff, df, r=other)
    assert not is_close("ttest_bayes_factor", wrong_prior, bf10), f"{what}: {other}"
    if a.size != b.size:
        welch = float(stats.ttest_ind(a, b, equal_var=False).statistic)
        welch_bf = ref.jzs_bf10_t(welch, n_eff, df, r=prior)
        assert not is_close("ttest_bayes_factor", welch_bf, bf10), f"{what}: Welch"


@pytest.mark.parametrize("case", _params("bayes", two_groups=True))
def test_jzs_bayes_factor(case: OracleCase) -> None:
    a, b = _groups(case)
    _check_jzs(case.subtitle_data, a, b, prior=_prior(case), what=case.name)


@pytest.mark.parametrize(
    "case", _params("parametric", two_groups=True, needs="caption")
)
def test_jzs_bayes_factor_caption(case: OracleCase) -> None:
    # ggstatsplot's caption for a two-group parametric test is the same ttestBF call
    # with the case's bf.prior, without `alternative`: the Bayes factor stays
    # two-sided.
    caption = case.caption_data
    assert caption is not None, "no Bayes caption"
    a, b = _groups(case)
    _check_jzs(caption, a, b, prior=_prior(case), what=f"{case.name} caption")


@pytest.mark.parametrize(
    "case",
    [
        *_params("parametric", two_groups=True, needs=None),
        *_params("parametric", two_groups=False, needs=None),
    ],
)
def test_absent_outputs(case: OracleCase) -> None:
    # What a case's arguments switch off is absent, not empty-but-present: no
    # subtitle and its table without results.subtitle, no Bayes caption and its
    # table without bf.message (or the subtitle), no pairwise table with
    # pairwise.display = "none".
    if not _draws(case, "subtitle"):
        assert case.subtitle is None
        assert case.subtitle_rows == []
    if not _draws(case, "caption"):
        assert case.caption is None
        assert case.caption_data is None
    if not _draws(case, "pairwise"):
        assert case.pairwise == []


@pytest.mark.parametrize("case", _params("bayes", two_groups=False))
def test_jzs_bayes_factor_pairwise(case: OracleCase) -> None:
    # Every pair gets its own ttestBF, always with prior 0.707 (SPEC.md section 4).
    groups = dict(zip(case.analysis_levels, _groups(case), strict=True))
    assert len(case.pairwise) == len(groups) * (len(groups) - 1) // 2
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        _check_jzs(
            row, groups[g1], groups[g2], prior=JZS_PRIOR, what=f"{case.name} {g1}-{g2}"
        )


@pytest.mark.parametrize("case", _params("nonparametric", two_groups=False))
def test_dunn_holm(case: OracleCase) -> None:
    # Dunn (1964): rank all N values together (mid-ranks for ties); for groups i < j,
    # z = (Rbar_i - Rbar_j) / sqrt(S2 * (1/n_i + 1/n_j)), with the tie correction
    # S2 = N(N + 1)/12 - sum(t^3 - t) / (12(N - 1)) over sets of t tied values.
    # ggstatsplot 1.1.4 (PMCMRplus::kwAllPairsDunnTest) reports |z|, not a signed z,
    # and Holm-adjusts two-sided normal p-values across all k(k - 1)/2 pairs.
    groups = _groups(case)
    pooled = np.concatenate(groups)
    n = pooled.size
    sizes = [g.size for g in groups]
    ranks = stats.rankdata(pooled)
    mean_ranks = [float(r.mean()) for r in np.split(ranks, np.cumsum(sizes)[:-1])]
    ties = np.unique(pooled, return_counts=True)[1].astype(np.float64)
    s2 = n * (n + 1) / 12 - float((ties**3 - ties).sum()) / (12 * (n - 1))
    pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
    z_values = [
        abs(mean_ranks[i] - mean_ranks[j])
        / math.sqrt(s2 * (1 / sizes[i] + 1 / sizes[j]))
        for i, j in pairs
    ]
    adjusted = ref.holm([float(2 * stats.norm.sf(z)) for z in z_values])
    levels = case.analysis_levels
    expected = {
        (levels[i], levels[j]): (z, p_adj)
        for (i, j), z, p_adj in zip(pairs, z_values, adjusted, strict=True)
    }
    assert len(case.pairwise) == len(expected)
    for pair in case.pairwise:
        z, p_adj = expected[(str(pair["group1"]), str(pair["group2"]))]
        what = f"Dunn {pair['group1']}-{pair['group2']}"
        assert pair["test"] == "Dunn"
        assert pair["distribution"] == "z"
        assert pair["p.adjust.method"] == "Holm"
        assert_close("closed_form", z, number(pair["statistic"]), what=f"{what} z")
        assert_close(
            "closed_form", p_adj, number(pair["p.value"]), what=f"{what} Holm p"
        )


def _smm_cdf(c: float, n_comparisons: int, df: float) -> float:
    """CDF at ``c`` of the Studentized maximum modulus with ``n_comparisons`` terms.

    The statistic is max_i |Z_i| / S with Z_1..Z_C iid N(0, 1) and, independently,
    S**2 ~ chi2(df) / df, so P(max <= c) = E[erf(c S / sqrt(2)) ** C]. Integrated
    over the density of S = chi(df) / sqrt(df).
    """
    log_norm = (1 - df / 2) * math.log(2) - math.lgamma(df / 2) + math.log(df) / 2

    def integrand(s: float) -> float:
        if s <= 0.0:
            return 0.0
        x = s * math.sqrt(df)
        density = math.exp(log_norm + (df - 1) * math.log(x) - x * x / 2)
        return math.erf(c * s / math.sqrt(2)) ** n_comparisons * density

    upper = float(stats.chi.isf(1e-15, df)) / math.sqrt(df)
    value, _ = integrate.quad(integrand, 0.0, upper, limit=200)
    return value


# The df rows of the two-decimal Studentized maximum modulus table lincon reads: every
# df from 2 to 12, then 14, 16, 18, 20, 24, 30, 40 and 60, and a last row placed at
# df 200 that holds the infinite-df (normal) values. lincon interpolates linearly in
# 1/df between the rows on either side of a pair's Yuen df. The fixtures confirm
# rows 12/14, 18/20 and 40/60, and the 60/200 step (bugs_long, df 99 to 110: the
# critical values lie on the line through the df-60 entry and the infinite-df entry
# at 200 to 1.6e-14).
SMM_TABLE_DF = (*range(2, 13), 14, 16, 18, 20, 24, 30, 40, 60, 200)
SMM_INFINITE_DF_ROW = 200
# Entries of the published table that differ from the rounded quantile, as the
# fixtures show them: (alpha, C, df) -> entry. The quantile for C = 3 at df 5 is
# 3.3992, which rounds to 3.40; edge_unbalanced a-b reproduces R only with 3.39.
SMM_TABLE_EXCEPTIONS = {(0.05, 3, 5): 3.39}


def _smm_table_entry(alpha: float, n_comparisons: int, df: int) -> float:
    """Upper ``alpha`` point of the Studentized maximum modulus, to two decimals."""
    known = SMM_TABLE_EXCEPTIONS.get((round(alpha, 10), n_comparisons, df))
    if known is not None:
        return known
    if df == SMM_INFINITE_DF_ROW:
        # max |Z_i| over C independent standard normals: (2 Phi(c) - 1)^C = 1 - alpha
        exact = float(stats.norm.isf((1 - (1 - alpha) ** (1 / n_comparisons)) / 2))
    else:
        exact = float(
            optimize.brentq(
                lambda c: _smm_cdf(c, n_comparisons, df) - (1 - alpha),
                1.0,
                50.0,
                xtol=1e-13,
            )
        )
    assert abs(exact * 100 % 1 - 0.5) > 1e-6, f"{exact} is at a rounding boundary"
    return round(exact, 2)


def _lincon_critical_value(alpha: float, n_comparisons: int, df: float) -> float:
    """Critical value of WRS2::lincon's pairwise CIs, as the fixtures show it."""
    if df in SMM_TABLE_DF:
        return _smm_table_entry(alpha, n_comparisons, int(df))
    if not SMM_TABLE_DF[0] < df < SMM_TABLE_DF[-1]:
        raise NotImplementedError(f"no fixture shows lincon at Yuen df {df}")
    hi = next(row for row in SMM_TABLE_DF if row > df)
    lo = max(row for row in SMM_TABLE_DF if row < df)
    at_lo = _smm_table_entry(alpha, n_comparisons, lo)
    at_hi = _smm_table_entry(alpha, n_comparisons, hi)
    return at_lo + (1 / lo - 1 / df) / (1 / lo - 1 / hi) * (at_hi - at_lo)


@pytest.mark.parametrize("case", _params("robust", two_groups=False))
def test_lincon_holm(case: OracleCase) -> None:
    # ggstatsplot 1.1.4's robust pairwise rows (WRS2::lincon, 20% trimming):
    #   estimate  trimmed mean of group1 minus trimmed mean of group2
    #   p.value   Holm over the per-pair Yuen p-values; Hochberg (lincon's own
    #             default) does not match the PlantGrowth fixture
    #   conf.*    estimate -/+ critical value x Yuen standard error, with the
    #             critical value at the pair's Yuen df and C = number of pairs
    #             (edge_k5: 10 pairs of 5 groups; bugs_long: 6 pairs of 4)
    levels = case.analysis_levels
    groups = dict(zip(levels, _groups(case), strict=True))
    pairs = [
        (levels[i], levels[j])
        for i in range(len(levels))
        for j in range(i + 1, len(levels))
    ]
    yuen = {(g1, g2): ref.yuen(groups[g1], groups[g2], tr=0.2) for g1, g2 in pairs}
    raw = [
        2 * _r_upper_tail(float(stats.t.sf(abs(yuen[pair][0]), yuen[pair][1])))
        for pair in pairs
    ]
    adjusted = dict(zip(pairs, ref.holm(raw), strict=True))
    assert len(case.pairwise) == len(pairs)
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        assert (g1, g2) in yuen, f"unexpected pair {g1}-{g2}"
        what = f"lincon {g1}-{g2}"
        assert row["test"] == "Yuen's trimmed means"
        assert row["p.adjust.method"] == "Holm"
        mean1, d1, _ = ref._trimmed_kernel(groups[g1], 0.2)
        mean2, d2, _ = ref._trimmed_kernel(groups[g2], 0.2)
        estimate = mean1 - mean2
        _, df, _ = yuen[(g1, g2)]
        alpha = 1 - _value(row, "conf.level")
        half_width = _lincon_critical_value(alpha, len(pairs), df) * math.sqrt(d1 + d2)
        assert_close(
            "closed_form", estimate, _value(row, "estimate"), what=f"{what} estimate"
        )
        # Two-sided, then Holm over the pairs: one grid step of a one-sided tail
        # can become 2 x pairs steps here (wrs2_p).
        assert_close(
            "wrs2_p",
            adjusted[(g1, g2)],
            _value(row, "p.value"),
            what=f"{what} Holm p",
            atol_scale=2 * len(pairs),
        )
        assert_close(
            "closed_form",
            estimate - half_width,
            _value(row, "conf.low"),
            what=f"{what} conf.low",
        )
        assert_close(
            "closed_form",
            estimate + half_width,
            _value(row, "conf.high"),
            what=f"{what} conf.high",
        )


# Probes ----------------------------------------------------------------------------
#
# The probe files record R functions on fixed grids (oracle/generate.R, "Probes").
# These checks confirm the independent references above against them.

# The brute-force closed-testing Hommel reference costs O(m^3); the 1007- and
# 1008-value vectors exist for BY's harmonic sum and are checked for every other
# method.
HOMMEL_REFERENCE_MAX_M = 60


@pytest.mark.parametrize(
    "vector",
    [pytest.param(c, id=c["name"]) for c in probe("probe-padjust").rows("cases")],
)
def test_p_adjust_probe(vector: Row) -> None:
    # stats::p.adjust against the published definitions, every method, including
    # ties, exact 0 and 1, values within an ulp of 1, and NA (kept in place, with m
    # counting the rest).
    p = [r_double(h) for h in vector["p_hex"]]
    values = [math.nan if v is None else v for v in p]
    for method, out in vector["adjusted"].items():
        if method == "hommel" and len(values) > HOMMEL_REFERENCE_MAX_M:
            continue
        expected = [r_double(h) for h in out["hex"]]
        got = _p_adjust(values, method)
        for i, (g, e) in enumerate(zip(got, expected, strict=True)):
            what = f"{vector['name']} {method} [{i}]"
            if e is None:
                assert math.isnan(g), what
            else:
                assert_close("closed_form", g, e, what=what)


def _games_howell_entries() -> list[Any]:
    entries = probe("probe-ptukey").rows("games_howell")
    return [pytest.param(e, id=str(e["case"])) for e in entries]


@pytest.mark.parametrize("entry", _games_howell_entries())
def test_games_howell_raw_p_probe(entry: Row) -> None:
    # PMCMRplus::gamesHowellTest's unadjusted p for each pair, on the rows of a
    # parametric case: |q| and p from the reference (SciPy's studentized range,
    # games_howell_p as for the adjusted values), and the fixture's adjusted p is
    # p.adjust of these with the case's method. Below 2 df R's ptukey is undefined
    # and returns NaN (edge_n2), where ggstatsplot then fails.
    cases = {c.name: c for c in cases_for("ggbetweenstats")}
    failed = {c.name: c for c in cases_for("ggbetweenstats", failed=True)}
    case = cases.get(entry["case"]) or failed[entry["case"]]
    groups = _groups(case)
    levels = case.analysis_levels
    assert tuple(entry["levels"]) == levels
    reference = {
        (levels[i], levels[j]): (q, p_raw)
        for i, j, q, p_raw in ref.games_howell(groups)
    }
    raw: dict[tuple[str, str], float] = {}
    for pair in entry["pairs"]:
        key = (str(pair["group1"]), str(pair["group2"]))
        q, p_raw = reference[key]
        what = f"{case.name} {key[0]}-{key[1]}"
        assert_close(
            "closed_form", q, abs(_hex(pair["statistic_hex"])), what=f"{what} q"
        )
        welch_df = _hex(pair["welch_df_hex"])
        p_r = _hex(pair["p_value_hex"])
        if welch_df < 2:
            assert math.isnan(p_r), what
        else:
            assert_close("games_howell_p", p_raw, p_r, what=f"{what} raw p")
        raw[key] = p_r
    if case.error is not None or not _draws(case, "pairwise"):
        return
    method = str(case.args.get("p.adjust.method", "holm"))
    keys = list(raw)
    adjusted = dict(zip(keys, _p_adjust([raw[k] for k in keys], method), strict=True))
    for row in case.pairwise:
        key = (str(row["group1"]), str(row["group2"]))
        # Same R values, adjusted twice in R: equal to rounding.
        assert_close(
            "closed_form",
            adjusted[key],
            number(row["p.value"]),
            what=f"{case.name} {key} adjusted from raw",
        )


def test_omega_probe() -> None:
    # effectsize::F_to_omega2's CI is effectsize::F_to_eta2's CI at the F value the
    # omega estimate corresponds to, F_w = omega df2 / ((1 - omega) df1), for every
    # F, df, level and alternative on the grid; and the accurate one-sided lower
    # bound R's uniroot finds (tol 1e-12) is what a SciPy noncentral-F inversion at
    # F_w finds. effectsize's own lower bound misses that root on some grid points
    # (probe rows; DECISIONS.md), which is why the comparison is with the uniroot
    # bound, not with effectsize's.
    rows = probe("probe-omega").rows("rows")
    for row in rows:
        what = (
            f"F={row['f']} df=({row['df1']}, {row['df2']}) ci={row['ci']} "
            f"{row['alternative']}"
        )
        assert row["omega2_error"] is None, what
        assert row["eta2_error"] is None, what
        for omega_key, eta_key in (
            ("omega2_low", "eta2_low"),
            ("omega2_high", "eta2_high"),
        ):
            omega_bound = _hex(row[omega_key]["hex"])
            eta_bound = _hex(row[eta_key]["hex"])
            assert_close(
                "closed_form", eta_bound, omega_bound, what=f"{what} {omega_key}"
            )
        accurate = row["accurate_low"]
        if accurate is None:
            continue
        assert accurate["error"] is None, what
        f_w = _hex(row["f_w"]["hex"])
        lam = _f_ncp(f_w, float(row["df1"]), _hex(row["df2_hex"]), _hex(row["ci_hex"]))
        bound = lam / (lam + _hex(row["df2_hex"]))
        assert_close(
            "noncentral_ci",
            bound,
            _hex(accurate["bound"]["hex"]),
            what=f"{what} accurate low",
        )
