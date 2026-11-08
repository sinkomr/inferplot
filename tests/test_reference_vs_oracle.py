"""Independent cross-checks of the oracle fixtures against tools/reference and SciPy.

These tests do not exercise inferplot. They confirm that the fixture pipeline (CSV,
R, JSON, loader) carries numbers that an independent implementation of the
published method reproduces, and they pin down which variant R ran: where a
plausible variant exists, a negative control shows that it fails.

Two groups (alternative two-sided, greater or less):
    parametric     Welch t (SciPy); Hedges' g with the unpooled SD, the exact
                   correction at the Welch df and a noncentral-t CI (inline); the
                   Bayes caption's JZS Bayes factor (reference)
    nonparametric  Mann-Whitney U, always the normal approximation with continuity
                   correction (SciPy); rank-biserial r with a Fisher-z CI (inline)
    robust         Yuen, |t| and p as WRS2 computes it (reference)
    bayes          JZS Bayes factor with Student's t and r = 0.707 (reference)

Three or more groups:
    parametric     Welch ANOVA, partial omega squared, Games-Howell then Holm
                   (reference; p against SciPy's studentized range)
    nonparametric  Kruskal-Wallis (SciPy); Dunn then Holm (inline, Dunn 1964)
    robust         trimmed-means ANOVA (reference t1way); lincon pairwise Yuen tests
                   then Holm, with the Studentized maximum modulus CI (inline)
    bayes          JZS Bayes factor of every pair (reference); the Bayes ANOVA
                   itself arrives in v0.2
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import integrate, optimize, stats
from tools.reference import stats_reference as ref

from tests.oracle_fixtures import (
    OracleCase,
    Row,
    cases_for,
    number,
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


def _params(type_: str, *, two_groups: bool) -> list[Any]:
    cases = [
        case
        for case in cases_for("ggbetweenstats")
        if case.type == type_ and (len(case.analysis_levels) == 2) == two_groups
    ]
    design = "two groups" if two_groups else "three or more groups"
    require_cases(cases, f"ggbetweenstats {type_}, {design}")
    return [pytest.param(case, id=case.name) for case in cases]


def _groups(case: OracleCase) -> list[FloatArray]:
    frame = read_dataset(case)
    x, y = case.args["x"], case.args["y"]
    used = frame[[x, y]].dropna()
    labels = used[x].map(r_level_label)
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
def test_welch_anova_omega_and_games_howell(case: OracleCase) -> None:
    groups = _groups(case)
    f_stat, df1, df2, p = ref.welch_anova(groups)
    row = case.subtitle_data
    assert_close("closed_form", f_stat, _value(row, "statistic"), what="Welch F")
    assert_close("closed_form", df1, _value(row, "df"), what="Welch df1")
    assert_close("closed_form", df2, _value(row, "df.error"), what="Welch df2")
    assert_close("closed_form", p, _value(row, "p.value"), what="Welch p")
    omega = ref.omega2_partial_from_f(f_stat, df1, df2)
    assert_close(
        "closed_form", omega, _value(row, "estimate"), what="partial omega squared"
    )

    # ggstatsplot 1.1.4 reports q with the sign of mean(group2) - mean(group1);
    # the reference returns |q|.
    levels = case.analysis_levels
    pairs = ref.games_howell(groups)
    adjusted = ref.holm([p_raw for *_, p_raw in pairs])
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
        assert pair["p.adjust.method"] == "Holm"
        assert_close("closed_form", q, number(pair["statistic"]), what=f"{what} q")
        # The reference uses SciPy's studentized range; R's ptukey has its own
        # absolute error in the far tail (games_howell_p, DECISIONS.md D31).
        assert_close(
            "games_howell_p", p_adj, number(pair["p.value"]), what=f"{what} Holm p"
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


def _check_jzs(row: Row, a: FloatArray, b: FloatArray, *, what: str) -> None:
    """BayesFactor::ttestBF on two groups: Student's t, r = 0.707, two-sided."""
    assert number(row["prior.scale"]) == JZS_PRIOR, f"{what}: {row['prior.scale']}"
    bf10 = _value(row, "bf10")
    n_eff = a.size * b.size / (a.size + b.size)
    df = float(a.size + b.size - 2)
    student = float(stats.ttest_ind(a, b, equal_var=True).statistic)
    got = ref.jzs_bf10_t(student, n_eff, df, r=JZS_PRIOR)
    assert_close("ttest_bayes_factor", got, bf10, what=f"{what} BF10")
    # Negative controls: BayesFactor's own "medium" prior sqrt(2)/2, and Welch's t
    # (which equals Student's t only for equal group sizes), are both off.
    medium = ref.jzs_bf10_t(student, n_eff, df, r=math.sqrt(2) / 2)
    assert not is_close("ttest_bayes_factor", medium, bf10), f"{what}: sqrt(2)/2"
    if a.size != b.size:
        welch = float(stats.ttest_ind(a, b, equal_var=False).statistic)
        welch_bf = ref.jzs_bf10_t(welch, n_eff, df, r=JZS_PRIOR)
        assert not is_close("ttest_bayes_factor", welch_bf, bf10), f"{what}: Welch"


@pytest.mark.parametrize("case", _params("bayes", two_groups=True))
def test_jzs_bayes_factor(case: OracleCase) -> None:
    a, b = _groups(case)
    _check_jzs(case.subtitle_data, a, b, what=case.name)


@pytest.mark.parametrize("case", _params("parametric", two_groups=True))
def test_jzs_bayes_factor_caption(case: OracleCase) -> None:
    # ggstatsplot's caption for a two-group parametric test is the same ttestBF call,
    # without `alternative`: the Bayes factor stays two-sided.
    caption = case.caption_data
    assert caption is not None, "no Bayes caption"
    a, b = _groups(case)
    _check_jzs(caption, a, b, what=f"{case.name} caption")


@pytest.mark.parametrize("case", _params("bayes", two_groups=False))
def test_jzs_bayes_factor_pairwise(case: OracleCase) -> None:
    # Every pair gets its own ttestBF, always with prior 0.707 (SPEC.md section 4).
    groups = dict(zip(case.analysis_levels, _groups(case), strict=True))
    assert len(case.pairwise) == len(groups) * (len(groups) - 1) // 2
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        _check_jzs(row, groups[g1], groups[g2], what=f"{case.name} {g1}-{g2}")


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
