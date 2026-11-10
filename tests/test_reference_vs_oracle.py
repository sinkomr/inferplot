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

ggwithinstats (paired designs) has its own section at the end ("Within subjects"),
read from the subjects-by-conditions table each fixture records.
"""

from __future__ import annotations

import functools
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
    expected_bf10,
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


def _tr(case: OracleCase) -> float:
    """The trimming of the robust tests and labels (ggstatsplot's default 0.2)."""
    return float(case.args.get("tr", 0.2))


def _adjust_method(case: OracleCase) -> str:
    return str(case.args.get("p.adjust.method", "holm"))


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
    # ggstatsplot failed after computing the subtitle (edge_n2: the pairwise table
    # stops below df 2 for parametric and robust, and on the pairwise Bayes factors);
    # what it computed is still right.
    rows = case.partial("subtitle")
    groups = _groups(case)
    if case.type == "parametric":
        assert len(rows) == 1
        _welch_anova_checks(groups, rows[0])
    elif case.type == "robust":
        assert len(rows) == 1
        f_stat, df1, df2, _ = ref.t1way(groups, tr=_tr(case))
        assert_close("closed_form", f_stat, _value(rows[0], "statistic"), what="F")
        assert_close("closed_form", df1, _value(rows[0], "df"), what="df1")
        assert_close("closed_form", df2, _value(rows[0], "df.error"), what="df2")
        p = _r_upper_tail(float(stats.f.sf(f_stat, df1, df2)))
        assert_close("wrs2_p", p, _value(rows[0], "p.value"), what="t1way p")
    else:
        # The Bayes ANOVA (v0.2): one row per posterior parameter, one Bayes factor.
        assert case.type == "bayes"
        assert rows
        assert len({number(r["bf10"]) for r in rows}) == 1


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
    t, df, _ = ref.yuen(a, b, tr=_tr(case))
    row = case.subtitle_data
    assert_close("closed_form", abs(t), _value(row, "statistic"), what="Yuen |t|")
    if t < 0:
        assert _value(row, "estimate") < 0, "the effect size keeps its sign"
    assert_close("closed_form", df, _value(row, "df.error"), what="Yuen df")
    p = 2 * _r_upper_tail(float(stats.t.sf(abs(t), df)))
    assert_close("wrs2_p", p, _value(row, "p.value"), what="Yuen p", atol_scale=2)


@pytest.mark.parametrize("case", _params("robust", two_groups=False))
def test_t1way(case: OracleCase) -> None:
    f_stat, df1, df2, _ = ref.t1way(_groups(case), tr=_tr(case))
    row = case.subtitle_data
    assert_close("closed_form", f_stat, _value(row, "statistic"), what="t1way F")
    assert_close("closed_form", df1, _value(row, "df"), what="t1way df1")
    assert_close("closed_form", df2, _value(row, "df.error"), what="t1way df2")
    p = _r_upper_tail(float(stats.f.sf(f_stat, df1, df2)))
    assert_close("wrs2_p", p, _value(row, "p.value"), what="t1way p")


def _check_jzs(
    row: Row,
    a: FloatArray,
    b: FloatArray,
    *,
    prior: float,
    what: str,
    pair: tuple[str, str, str] | None = None,
) -> None:
    """BayesFactor::ttestBF on two groups: Student's t, Cauchy scale ``prior``.

    ``pair`` (case, group1, group2) names a pairwise row, compared with the accurate
    value where BayesFactor misses it (`ACCURATE_BAYES_FACTORS`).
    """
    assert number(row["prior.scale"]) == prior, f"{what}: {row['prior.scale']}"
    bf10 = _value(row, "bf10")
    if pair is not None:
        bf10 = expected_bf10(*pair, bf10)
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


@pytest.mark.parametrize("case", _params("bayes", two_groups=False, needs="pairwise"))
def test_jzs_bayes_factor_pairwise(case: OracleCase) -> None:
    # Every pair gets its own ttestBF, always with prior 0.707 (SPEC.md section 4).
    groups = dict(zip(case.analysis_levels, _groups(case), strict=True))
    assert len(case.pairwise) == len(groups) * (len(groups) - 1) // 2
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        _check_jzs(
            row,
            groups[g1],
            groups[g2],
            prior=JZS_PRIOR,
            what=f"{case.name} {g1}-{g2}",
            pair=(case.name, g1, g2),
        )


def _dunn(groups: Sequence[FloatArray]) -> tuple[list[tuple[int, int]], list[float]]:
    """Pairs (i < j) and Dunn's (1964) |z| for each.

    Rank all N values together (mid-ranks for ties); for groups i < j,
    z = (Rbar_i - Rbar_j) / sqrt(S2 * (1/n_i + 1/n_j)), with the tie correction
    S2 = N(N + 1)/12 - sum(t^3 - t) / (12(N - 1)) over sets of t tied values.
    """
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
    return pairs, z_values


@pytest.mark.parametrize(
    "case", _params("nonparametric", two_groups=False, needs="pairwise")
)
def test_dunn_holm(case: OracleCase) -> None:
    # Dunn (1964), `_dunn`. ggstatsplot 1.1.4 (PMCMRplus::kwAllPairsDunnTest) reports
    # |z|, not a signed z, and adjusts two-sided normal p-values across all
    # k(k - 1)/2 pairs with the case's p.adjust method (Holm by default).
    pairs, z_values = _dunn(_groups(case))
    method = _adjust_method(case)
    adjusted = _p_adjust([float(2 * stats.norm.sf(z)) for z in z_values], method)
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
        assert pair["p.adjust.method"] == P_ADJUST_LABELS[method]
        assert_close("closed_form", z, number(pair["statistic"]), what=f"{what} z")
        assert_close(
            "closed_form", p_adj, number(pair["p.value"]), what=f"{what} {method} p"
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


# lincon's critical values (probe-smm). WRS2::lincon reads a two-decimal table of the
# Studentized maximum modulus, one for alpha 0.05 (WRS2's internal smmcrit) and one
# for alpha 0.01 (smmcrit01), with rows at every df from 2 to 12, then 14, 16, 18, 20,
# 24, 30, 40 and 60, and a last row at df 200 that holds the infinite-df values, for
# C = 2 to 28 comparisons. C = 1 is Student's t quantile at the pair's df. Between rows
# the value is linear in 1/df; from df 200 on it is the last row; below df 2 and above
# C = 28 the table stops with an error. For any other alpha, including 1 - 0.95
# computed in floating point, and for C above 28, lincon calls a simulation instead
# (WRS2's smmvalv2). The table's entries are the published ones, not recomputed
# quantiles: `test_smm_table_against_quantiles` lists where they differ.
SMM_TABLE_DF = (*range(2, 13), 14, 16, 18, 20, 24, 30, 40, 60, 200)
SMM_LAST_ROW = 200
SMM_MAX_COMPARISONS = 28
SMM_TABLES = {0.05: "smmcrit", 0.01: "smmcrit01"}


def _smm_grid(name: str) -> Row:
    grids = {str(g["name"]): g for g in probe("probe-smm").rows("grids")}
    return grids[name]


@functools.cache
def _smm_table(alpha: float) -> dict[tuple[int, int], float]:
    """The table lincon reads at ``alpha``, {(df row, C): entry}, from probe-smm."""
    grid = _smm_grid(SMM_TABLES[alpha])
    table: dict[tuple[int, int], float] = {}
    for row in grid["rows"]:
        df = _hex(row["df_hex"])
        if df not in SMM_TABLE_DF:
            continue
        for c, text in zip(grid["c_grid"], row["values_hex"], strict=True):
            if 2 <= c <= SMM_MAX_COMPARISONS:
                table[(int(df), int(c))] = _hex(text)
    assert len(table) == len(SMM_TABLE_DF) * (SMM_MAX_COMPARISONS - 1)
    return table


def _lincon_critical_value(alpha: float, n_comparisons: int, df: float) -> float:
    """Critical value of WRS2::lincon's pairwise CIs at a pair's Yuen ``df``."""
    if alpha not in SMM_TABLES or n_comparisons > SMM_MAX_COMPARISONS:
        raise NotImplementedError(
            f"lincon simulates its critical value at alpha {alpha!r}, C {n_comparisons}"
        )
    if df < SMM_TABLE_DF[0]:
        raise ValueError(f"lincon's table starts at df 2, not {df}")
    if n_comparisons == 1:
        return ref.t_isf(alpha / 2, df)
    table = _smm_table(alpha)
    if df >= SMM_LAST_ROW:
        return table[(SMM_LAST_ROW, n_comparisons)]
    if df in SMM_TABLE_DF:
        return table[(int(df), n_comparisons)]
    hi = next(row for row in SMM_TABLE_DF if row > df)
    lo = max(row for row in SMM_TABLE_DF if row < df)
    at_lo = table[(lo, n_comparisons)]
    at_hi = table[(hi, n_comparisons)]
    return at_lo + (1 / lo - 1 / df) / (1 / lo - 1 / hi) * (at_hi - at_lo)


def _adjustment_scale(method: str, m: int) -> float:
    """The largest factor ``p.adjust`` multiplies a p-value by over ``m`` tests."""
    if method == "none":
        return 1.0
    if method == "BY":
        return m * math.fsum(1 / k for k in range(1, m + 1))
    return float(m)


def _lincon_alpha(case: OracleCase, row: Row) -> float:
    """The alpha lincon ran at: always its default 0.05.

    ggstatsplot 1.1.4 does not pass `conf.level` to lincon: with 0.90
    (PlantGrowth-cl90-robust) the subtitle's CI is 90% but every pairwise row says
    0.95 and has the 95% interval. Had it passed alpha = 1 - conf.level, lincon would
    have simulated its critical value (probe-smm), since 1 - 0.95 is not 0.05.
    """
    assert _value(row, "conf.level") == 0.95, case.name
    return 0.05


@pytest.mark.parametrize("case", _params("robust", two_groups=False, needs="pairwise"))
def test_lincon(case: OracleCase) -> None:
    # ggstatsplot 1.1.4's robust pairwise rows (WRS2::lincon, the case's trimming):
    #   estimate  trimmed mean of group1 minus trimmed mean of group2
    #   p.value   the case's p.adjust method over the per-pair Yuen p-values, which
    #             WRS2 computes as 1 - pt; Hochberg (lincon's own default) does not
    #             match the PlantGrowth fixture
    #   conf.*    estimate -/+ critical value x Yuen standard error, with the
    #             critical value at the pair's Yuen df and C = number of pairs
    #             (edge_k5: 10 pairs of 5 groups; bugs_long: 6 pairs of 4)
    levels = case.analysis_levels
    tr = _tr(case)
    method = _adjust_method(case)
    groups = dict(zip(levels, _groups(case), strict=True))
    pairs = [
        (levels[i], levels[j])
        for i in range(len(levels))
        for j in range(i + 1, len(levels))
    ]
    yuen = {(g1, g2): ref.yuen(groups[g1], groups[g2], tr=tr) for g1, g2 in pairs}
    raw = [
        2 * _r_upper_tail(float(stats.t.sf(abs(yuen[pair][0]), yuen[pair][1])))
        for pair in pairs
    ]
    adjusted = dict(zip(pairs, _p_adjust(raw, method), strict=True))
    assert len(case.pairwise) == len(pairs)
    for row in case.pairwise:
        g1, g2 = str(row["group1"]), str(row["group2"])
        assert (g1, g2) in yuen, f"unexpected pair {g1}-{g2}"
        what = f"lincon {g1}-{g2}"
        assert row["test"] == "Yuen's trimmed means"
        assert row["p.adjust.method"] == P_ADJUST_LABELS[method]
        mean1, d1, _ = ref._trimmed_kernel(groups[g1], tr)
        mean2, d2, _ = ref._trimmed_kernel(groups[g2], tr)
        estimate = mean1 - mean2
        _, df, _ = yuen[(g1, g2)]
        alpha = _lincon_alpha(case, row)
        assert_close(
            "closed_form", estimate, _value(row, "estimate"), what=f"{what} estimate"
        )
        # Two-sided, then the adjustment: one grid step of a one-sided tail can
        # become 2 x the adjustment's largest factor here (wrs2_p).
        assert_close(
            "wrs2_p",
            adjusted[(g1, g2)],
            _value(row, "p.value"),
            what=f"{what} {method} p",
            atol_scale=2 * _adjustment_scale(method, len(pairs)),
        )
        low, high = _value(row, "conf.low"), _value(row, "conf.high")
        if len(pairs) > SMM_MAX_COMPARISONS:
            # Past the table (edge_k9, 36 pairs) lincon simulates its critical
            # value with a seed of its own (probe-smm): the interval is still
            # estimate -/+ c x Yuen SE, but c has no closed form to compare with.
            assert_close("closed_form", estimate - low, high - estimate, what=what)
            continue
        critical = _lincon_critical_value(alpha, len(pairs), df)
        half_width = critical * math.sqrt(d1 + d2)
        assert_close("closed_form", estimate - half_width, low, what=f"{what} conf.low")
        assert_close(
            "closed_form", estimate + half_width, high, what=f"{what} conf.high"
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


# probe-wrs2p: WRS2's p-values are 1 - cdf --------------------------------------------


def _wrs2p_rows(section: str) -> list[Row]:
    return probe("probe-wrs2p").rows(section)


def test_wrs2_p_is_one_minus_cdf() -> None:
    # yuen and t1way on constructed data over |t| from 0 to about 40 at small,
    # moderate and large df, and lincon's unadjusted pairs: every p is exactly R's
    # 1 - pt (doubled) or 1 - pf, and the exact upper tail differs on some rows, so
    # the probe tells the two apart. inferplot's rule, `_r_upper_tail` of the exact
    # tail, matches each within `wrs2_p`.
    differ = 0
    for row in _wrs2p_rows("yuen"):
        p = _hex(row["p_value_hex"])
        assert p == 2 * _hex(row["one_minus_cdf_hex"]), row
        differ += p != 2 * _hex(row["upper_tail_hex"])
        t, df = abs(_hex(row["test_hex"])), _hex(row["df_hex"])
        expected = 2 * _r_upper_tail(float(stats.t.sf(t, df)))
        assert_close(
            "wrs2_p", expected, p, what=f"yuen {row['n']} {row['shift']}", atol_scale=2
        )
    assert differ > 0
    differ = 0
    for row in _wrs2p_rows("t1way"):
        p = _hex(row["p_value_hex"])
        assert p == _hex(row["one_minus_cdf_hex"]), row
        differ += p != _hex(row["upper_tail_hex"])
        f_stat, df1, df2 = (_hex(row[k]) for k in ("test_hex", "df1_hex", "df2_hex"))
        expected = _r_upper_tail(float(stats.f.sf(f_stat, df1, df2)))
        assert_close("wrs2_p", expected, p, what=f"t1way {row['n']} {row['shift']}")
    assert differ > 0
    for row in _wrs2p_rows("lincon"):
        for pair in row["pairs"]:
            # lincon's unadjusted p is the pair's Yuen p; its default (Hochberg) is
            # not what ggstatsplot reports.
            p = _hex(pair["p_value_hex"])
            assert p == _hex(pair["yuen_p_value_hex"]), pair
            assert p == 2 * _hex(pair["one_minus_cdf_hex"]), pair


# probe-akp: the AKP scaling constant ------------------------------------------------


def _winsorized_normal_sd(tr: float) -> float:
    """SD of a standard normal winsorized at its tr and 1 - tr quantiles.

    With z the upper tr quantile: sigma^2 = 1 - 2 tr - 2 z phi(z) + 2 tr z^2.
    """
    if tr == 0:
        return 1.0
    z = float(stats.norm.isf(tr))
    return math.sqrt(1 - 2 * tr - 2 * z * float(stats.norm.pdf(z)) + 2 * tr * z * z)


def test_akp_constant() -> None:
    # akp.effect(EQVAR = FALSE) = c(tr) x (trimmed mean 1 - trimmed mean 2) /
    # winsorized SD of group 1, with c(tr) the same on every dataset (to a few ulps)
    # and within akp_constant of the winsorized-normal SD, but not equal
    # to it: at tr 0.2 the offset is 1.2e-9, above closed_form.
    akp = probe("probe-akp")
    by_tr: dict[float, list[float]] = {}
    for row in akp.rows("rows"):
        effect = row["akp"][0]
        assert effect["EQVAR"] is False
        ratio = effect["ratio_to_group1_hex"]
        if ratio is None or not math.isfinite(_hex(ratio)):
            continue  # a winsorized SD of 0 (ties at tr 0.4)
        by_tr.setdefault(float(row["tr"]), []).append(_hex(ratio))
    for tr, ratios in by_tr.items():
        assert max(ratios) - min(ratios) <= 4 * math.ulp(max(ratios)), (tr, ratios)
    constants = [
        (_hex(r["tr_hex"]), _hex(r["ratio_to_group1_hex"]))
        for r in akp.rows("constants")
    ]
    assert len(constants) == 49
    for tr, constant in [*constants, *((t, v[0]) for t, v in by_tr.items())]:
        exact = _winsorized_normal_sd(tr)
        assert_close("akp_constant", exact, constant, what=f"AKP constant at tr {tr}")
    at_02 = by_tr[0.2][0]
    assert not is_close("closed_form", at_02, _winsorized_normal_sd(0.2))


# probe-smm: the table behind lincon's CIs -------------------------------------------


@pytest.mark.parametrize("alpha", sorted(SMM_TABLES))
def test_smm_table_rule(alpha: float) -> None:
    # WRS2's table function on its whole df x C grid follows `_lincon_critical_value`:
    # C = 1 is Student's t quantile, rows in between are linear in 1/df, df 200 and
    # above give the last row, and df below 2 or C above 28 is an error.
    grid = _smm_grid(SMM_TABLES[alpha])
    assert not grid["random"]
    assert not grid["small_grid"]
    c_grid = [int(c) for c in grid["c_grid"]]
    assert c_grid == list(range(1, 51))
    for row in grid["rows"]:
        df = _hex(row["df_hex"])
        errors = {int(e["C"]): str(e["error"]) for e in row["errors"]}
        for c, text in zip(c_grid, row["values_hex"], strict=True):
            what = f"alpha {alpha} C {c} df {df}"
            # C is checked first: above 28 that error wins at any df.
            if c > SMM_MAX_COMPARISONS:
                assert text == "NA", what
                assert "C must be" in errors[c], what
                continue
            if df < 2:
                assert text == "NA", what
                assert "degrees of freedom" in errors[c], what
                continue
            assert c not in errors, what
            expected = _lincon_critical_value(alpha, c, df)
            assert_close("closed_form", expected, _hex(text), what=what)
            if c >= 2 and df in SMM_TABLE_DF:
                assert abs(_hex(text) * 100 - round(_hex(text) * 100)) < 1e-9, what


def test_smm_lincon_backout() -> None:
    # lincon on 2 to 10 groups: its half-width over the pair's Yuen SE is the table's
    # value at alpha 0.05 and 0.01 with C <= 28. Any other alpha (1 - 0.95 included)
    # and C = 45 give something else: lincon's simulation.
    entries = probe("probe-smm").rows("lincon")
    assert len(entries) == 9 * 6
    tabled = 0
    for entry in entries:
        result = entry["result"]
        assert result["error"] is None, entry
        assert len(result["pairs"]) == entry["comparisons"]
        alpha = _hex(entry["alpha_hex"])
        in_table = alpha in SMM_TABLES and entry["comparisons"] <= SMM_MAX_COMPARISONS
        for pair in result["pairs"]:
            got = _hex(pair["critical_value_hex"])
            what = f"{entry['groups']} groups alpha {entry['alpha']} {pair['group1']}"
            if in_table:
                tabled += 1
                want = _lincon_critical_value(
                    alpha, int(entry["comparisons"]), _hex(pair["yuen_df_hex"])
                )
                assert_close("closed_form", want, got, what=what)
            elif alpha in SMM_TABLES:
                with pytest.raises(NotImplementedError):
                    _lincon_critical_value(
                        alpha, int(entry["comparisons"]), _hex(pair["yuen_df_hex"])
                    )
    assert tabled == 2 * sum(k * (k - 1) // 2 for k in range(2, 9))
    # The simulation does not depend on R's seed: lincon seeds it itself.
    runs: dict[tuple[int, str], set[tuple[str, ...]]] = {}
    for run in probe("probe-smm").rows("seed_dependence"):
        assert run["error"] is None, run
        key = (int(run["groups"]), str(run["alpha"]))
        runs.setdefault(key, set()).add(tuple(run["ci_upper_hex"]))
    assert len(runs) == 3
    assert all(len(bounds) == 1 for bounds in runs.values()), runs


# The table entries that differ from round(quantile, 2) by more than one unit in the
# last place, where they look like transcription errors: (alpha, C, df) -> entry
# (df 200 is the infinite-df row). Measured with `_smm_cdf` against probe-smm.
SMM_TABLE_OUTLIERS = {
    (0.01, 6, 12): 0.99,
    (0.01, 9, 14): 0.05,
    (0.01, 15, 2): 20.99,
    (0.01, 16, 2): 20.99,
    (0.01, 16, 200): 0.42,
    (0.01, 17, 2): 20.99,
    (0.01, 17, 200): 0.44,
    (0.01, 18, 2): 20.99,
    (0.01, 19, 2): 20.99,
    (0.01, 26, 3): 12.07,
    (0.05, 9, 2): 8.83,
}


def _smm_quantile(alpha: float, n_comparisons: int, df: int) -> float:
    if df == SMM_LAST_ROW:
        # max |Z_i| over C independent standard normals: (2 Phi(c) - 1)^C = 1 - alpha
        return float(stats.norm.isf((1 - (1 - alpha) ** (1 / n_comparisons)) / 2))
    return float(
        optimize.brentq(
            lambda c: _smm_cdf(c, n_comparisons, df) - (1 - alpha),
            0.5,
            80.0,
            xtol=1e-12,
        )
    )


@pytest.mark.parametrize("alpha", sorted(SMM_TABLES))
def test_smm_table_against_quantiles(alpha: float) -> None:
    # Every entry of the published table is the Studentized maximum modulus quantile
    # to within 0.01 (one unit in the last place, from rounding), except the
    # outliers above. With all pairs compared ggstatsplot uses alpha 0.05 and
    # C = 3, 6, 10, 15, 21 or 28, where no outlier lies.
    table = _smm_table(alpha)
    outliers = {}
    for (df, c), entry in table.items():
        quantile = _smm_quantile(alpha, c, df)
        if abs(entry - quantile) > 0.01 + 1e-9:
            outliers[(alpha, c, df)] = entry
    assert outliers == {k: v for k, v in SMM_TABLE_OUTLIERS.items() if k[0] == alpha}


# probe-ranks: Mann-Whitney, rank-biserial, Kruskal-Wallis and Dunn on ties ----------


def test_ranks_probe_two_samples() -> None:
    # wilcox.test(exact = FALSE) with and without the continuity correction, every
    # alternative, against SciPy's normal approximation with the tie-corrected
    # variance; effectsize::rank_biserial against 2U/(n1 n2) - 1 with the Fisher-z CI
    # that ignores ties.
    for entry in probe("probe-ranks").rows("two_sample"):
        a = np.array([_hex(h) for h in entry["a_hex"]])
        b = np.array([_hex(h) for h in entry["b_hex"]])
        for test in entry["wilcox"]:
            what = f"{entry['name']} {test['alternative']} correct={test['correct']}"
            assert test["error"] is None, what
            got = stats.mannwhitneyu(
                a,
                b,
                alternative=SCIPY_ALTERNATIVE[str(test["alternative"])],
                method="asymptotic",
                use_continuity=bool(test["correct"]),
            )
            assert_close(
                "closed_form",
                float(got.statistic),
                _hex(test["statistic_hex"]),
                what=what,
            )
            assert_close(
                "closed_form", float(got.pvalue), _hex(test["p_value_hex"]), what=what
            )
        n1, n2 = a.size, b.size
        u = float(stats.mannwhitneyu(a, b, method="asymptotic").statistic)
        r = 2 * u / (n1 * n2) - 1
        se = math.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
        for out in entry["rank_biserial"]:
            what = f"{entry['name']} rank-biserial {out['alternative']}"
            assert out["error"] is None, what
            assert_close("closed_form", r, _hex(out["estimate_hex"]), what=what)
            side = str(out["alternative"])
            q = float(stats.norm.isf(0.025 if side == "two.sided" else 0.05))
            z = math.atanh(r)
            low = -1.0 if side == "less" else math.tanh(z - q * se)
            high = 1.0 if side == "greater" else math.tanh(z + q * se)
            assert_close("closed_form", low, _hex(out["conf_low_hex"]), what=what)
            assert_close("closed_form", high, _hex(out["conf_high_hex"]), what=what)


def test_ranks_probe_k_samples() -> None:
    entries = probe("probe-ranks").rows("k_sample")
    for entry in entries:
        levels = [str(v) for v in entry["levels"]]
        groups = [np.array([_hex(h) for h in g]) for g in entry["groups_hex"]]
        assert len(groups) == len(levels)
        h = stats.kruskal(*groups)
        assert_close(
            "closed_form",
            float(h.statistic),
            _hex(entry["kruskal_statistic_hex"]),
            what=f"{entry['name']} H",
        )
        assert_close(
            "closed_form",
            float(h.pvalue),
            _hex(entry["kruskal_p_value_hex"]),
            what=f"{entry['name']} p",
        )
        pairs, z_values = _dunn(groups)
        assert len(entry["dunn_pairs"]) == len(pairs)
        for (i, j), z, pair in zip(pairs, z_values, entry["dunn_pairs"], strict=True):
            what = f"{entry['name']} Dunn {pair['group1']}-{pair['group2']}"
            assert (pair["group1"], pair["group2"]) == (levels[i], levels[j])
            assert_close("closed_form", z, abs(_hex(pair["statistic_hex"])), what=what)
            assert_close(
                "closed_form",
                float(2 * stats.norm.sf(z)),
                _hex(pair["p_value_hex"]),
                what=what,
            )


# Within subjects (ggwithinstats) ----------------------------------------------------
#
# The same checks for ggwithinstats, on the subjects-by-conditions table the fixture
# records (`within_data`; tests/test_oracle_fixtures.py rebuilds it from the CSV by
# subject id on every backend). Each column is one condition, in R's level order,
# subjects in the same order in every column.
#
# Two conditions (x = first level, y = second, as statsExpressions passes them):
#     parametric     paired t (SciPy); Hedges' g of the differences, d_z x J(n - 1),
#                    with a noncentral-t CI (inline); the Bayes caption (reference)
#     nonparametric  Wilcoxon signed rank, normal approximation with continuity and
#                    tie corrections, zeros dropped (inline); the matched-pairs
#                    rank-biserial r with a Fisher-z CI (inline)
#     robust         Yuen's test for dependent trimmed means (inline, Wilcox 2017)
#     bayes          paired JZS Bayes factor (reference)
# Three or more conditions:
#     parametric     repeated-measures ANOVA with the Greenhouse-Geisser correction
#                    (reference); partial omega squared from the error strata with
#                    its CI at the uncorrected df (inline); paired t per pair, then
#                    the case's p.adjust method
#     nonparametric  Friedman (SciPy); Kendall's W; Durbin-Conover per pair
#                    (inline, Conover 1999), then the p.adjust method
#     robust         rmmcp per pair: Tukey-McLaughlin trimmed mean of the differences
#                    (inline), then the p.adjust method
#     bayes          paired JZS Bayes factor of every pair (reference)


def _within_params(
    type_: str, *, two: bool, needs: Output | None = "subtitle"
) -> list[Any]:
    """ggwithinstats cases of one type and design that have the output a check reads."""
    cases = [
        case
        for case in cases_for("ggwithinstats")
        if case.type == type_
        and (len(case.analysis_levels) == 2) == two
        and (needs is None or _draws(case, needs))
    ]
    design = "two conditions" if two else "three or more conditions"
    require_cases(cases, f"ggwithinstats {type_}, {design}, with {needs}")
    return [pytest.param(case, id=case.name) for case in cases]


def _conditions(case: OracleCase) -> list[FloatArray]:
    """One array per condition, in R's level order, subjects in R's order."""
    columns = case.within_columns()
    assert tuple(columns) == case.analysis_levels
    return [
        np.array(columns[level], dtype=np.float64) for level in case.analysis_levels
    ]


def _signed_rank(d: FloatArray, alternative: str) -> tuple[float, float]:
    """Wilcoxon's (1945) signed-rank V and its p as stats::wilcox.test(exact = FALSE)
    computes it: zero differences dropped, mid-ranks of |d|, V the sum of the
    positive ranks, the normal approximation with the tie correction
    sum(t^3 - t) / 48 in the variance and a continuity correction of 0.5 toward the
    null (its sign follows z when two-sided)."""
    nonzero = d[d != 0]
    n = nonzero.size
    ranks = stats.rankdata(np.abs(nonzero))
    v = float(ranks[nonzero > 0].sum())
    ties = np.unique(np.abs(nonzero), return_counts=True)[1].astype(np.float64)
    sigma = math.sqrt(
        n * (n + 1) * (2 * n + 1) / 24 - float((ties**3 - ties).sum()) / 48
    )
    z = v - n * (n + 1) / 4
    correction = {"two.sided": 0.5 * math.copysign(1.0, z) if z else 0.0}.get(
        alternative, 0.5 if alternative == "greater" else -0.5
    )
    z = (z - correction) / sigma
    if alternative == "two.sided":
        p = 2 * min(float(stats.norm.cdf(z)), float(stats.norm.sf(z)))
    elif alternative == "greater":
        p = float(stats.norm.sf(z))
    else:
        p = float(stats.norm.cdf(z))
    return v, p


def _trimmed_kernel_paired(
    x: FloatArray, y: FloatArray, tr: float
) -> tuple[float, float, float]:
    """Yuen's test for dependent trimmed means (Wilcox 2017, sec. 5.9.5): winsorize
    each condition at g = floor(tr n), d_j = (n - 1) s_wj^2 / (h (h - 1)) and
    d_12 = (n - 1) s_w12 / (h (h - 1)) with h = n - 2g; t = (tm_x - tm_y) /
    sqrt(d_1 + d_2 - 2 d_12) on h - 1 df. Returns (t, df, difference)."""
    n = x.size
    g = math.floor(tr * n)
    h = n - 2 * g

    def winsorize(v: FloatArray) -> FloatArray:
        ordered = np.sort(v)
        return np.clip(v, ordered[g], ordered[n - g - 1])

    wx, wy = winsorize(x), winsorize(y)
    q = (n - 1) * np.cov(wx, wy)
    se = math.sqrt((q[0, 0] + q[1, 1] - 2 * q[0, 1]) / (h * (h - 1)))
    difference = float(stats.trim_mean(x, tr) - stats.trim_mean(y, tr))
    return difference / se, float(h - 1), difference


def _trimmed_one_sample(d: FloatArray, tr: float) -> tuple[float, float, float]:
    """The Tukey-McLaughlin test of a trimmed mean (Wilcox 2017, sec. 4.4.1):
    t = tm / (s_w / ((1 - 2 tr) sqrt(n))) on n - 2g - 1 df. Returns (tm, t, df)."""
    n = d.size
    g = math.floor(tr * n)
    ordered = np.sort(d)
    winsorized = np.clip(d, ordered[g], ordered[n - g - 1])
    se = math.sqrt(float(winsorized.var(ddof=1))) / ((1 - 2 * tr) * math.sqrt(n))
    tm = float(stats.trim_mean(d, tr))
    return tm, tm / se, float(n - 2 * g - 1)


@pytest.mark.parametrize("case", _within_params("parametric", two=True))
def test_paired_t(case: OracleCase) -> None:
    a, b = _conditions(case)
    result = stats.ttest_rel(a, b, alternative=SCIPY_ALTERNATIVE[_alternative(case)])
    row = case.subtitle_data
    assert row["alternative"] == _alternative(case)
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="t"
    )
    assert_close("closed_form", a.size - 1.0, _value(row, "df.error"), what="df")
    assert_close("closed_form", float(result.pvalue), _value(row, "p.value"), what="p")


@pytest.mark.parametrize("case", _within_params("parametric", two=True))
def test_paired_hedges_g(case: OracleCase) -> None:
    # effectsize::hedges_g(x, y, paired = TRUE): the one-sample effect size of the
    # differences, d_z = mean(d) / sd(d), times Hedges' J at n - 1 df; the CI is the
    # noncentral-t interval at the paired t and n - 1 df, each bound / sqrt(n) x J.
    # Negative control: d_av, the mean difference over the average SD of the two
    # conditions.
    a, b = _conditions(case)
    d = a - b
    n = d.size
    sd = float(d.std(ddof=1))
    j = ref.hedges_j(n - 1.0)
    row = case.subtitle_data
    assert row["effectsize"] == "Hedges' g"
    g = _value(row, "estimate")
    assert_close("closed_form", float(d.mean()) / sd * j, g, what="Hedges' g")
    d_av = float(d.mean()) / math.sqrt(
        (float(a.var(ddof=1)) + float(b.var(ddof=1))) / 2
    )
    assert not is_close("closed_form", d_av * j, g)
    alpha = 1 - _value(row, "conf.level")
    side = _alternative(case)
    tail = alpha / 2 if side == "two.sided" else alpha
    t = float(d.mean()) / (sd / math.sqrt(n))
    scale = j / math.sqrt(n)
    low = -math.inf if side == "less" else _ncp_at(t, n - 1.0, 1 - tail) * scale
    high = math.inf if side == "greater" else _ncp_at(t, n - 1.0, tail) * scale
    assert_close("noncentral_ci", low, _value(row, "conf.low"), what="g CI low")
    assert_close("noncentral_ci", high, _value(row, "conf.high"), what="g CI high")


@pytest.mark.parametrize("case", _within_params("nonparametric", two=True))
def test_wilcoxon_signed_rank(case: OracleCase) -> None:
    a, b = _conditions(case)
    v, p = _signed_rank(a - b, _alternative(case))
    row = case.subtitle_data
    assert_close("closed_form", v, _value(row, "statistic"), what="V")
    assert_close("closed_form", p, _value(row, "p.value"), what="p")


@pytest.mark.parametrize("case", _within_params("nonparametric", two=True))
def test_paired_rank_biserial(case: OracleCase) -> None:
    # effectsize::rank_biserial(paired = TRUE): the matched-pairs rank-biserial r =
    # (W+ - W-) / (W+ + W-) over the nonzero differences (Kerby 2014); its CI is
    # tanh(atanh(r) -/+ z SE) with the null SE of r, sqrt(n(n + 1)(2n + 1) / 6) /
    # (n(n + 1) / 2), where n counts every pair, zero differences included
    # (probe-signrank: the nonzero count misses edge_within2_ties by 0.037). With
    # |r| = 1 both bounds are r itself.
    a, b = _conditions(case)
    d = a - b
    nonzero = d[d != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    plus, minus = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    r = (plus - minus) / (plus + minus)
    row = case.subtitle_data
    assert_close("closed_form", r, _value(row, "estimate"), what="r")
    n = d.size
    se = math.sqrt(n * (n + 1) * (2 * n + 1) / 6) / (n * (n + 1) / 2)
    q = _one_sided_quantile(case, row)
    side = _alternative(case)
    if abs(r) < 1:
        low, high = math.tanh(math.atanh(r) - q * se), math.tanh(math.atanh(r) + q * se)
    else:
        low = high = r
    low = -1.0 if side == "less" else low
    high = 1.0 if side == "greater" else high
    assert_close("closed_form", low, _value(row, "conf.low"), what="r CI low")
    assert_close("closed_form", high, _value(row, "conf.high"), what="r CI high")


@pytest.mark.parametrize("case", _within_params("robust", two=True))
def test_yuend(case: OracleCase) -> None:
    # WRS2::yuend; p = 2 (1 - pt(|t|, df)) as WRS2 computes it (probe-wrs2within:
    # every yuend p is that bit for bit). `alternative` has no effect.
    a, b = _conditions(case)
    t, df, _ = _trimmed_kernel_paired(a, b, _tr(case))
    row = case.subtitle_data
    assert_close("closed_form", t, _value(row, "statistic"), what="yuend t")
    assert_close("closed_form", df, _value(row, "df.error"), what="yuend df")
    p = 2 * _r_upper_tail(float(stats.t.sf(abs(t), df)))
    assert_close("wrs2_p", p, _value(row, "p.value"), what="yuend p", atol_scale=2)


def _check_paired_jzs(
    row: Row, a: FloatArray, b: FloatArray, *, prior: float, what: str
) -> None:
    """BayesFactor::ttestBF(x, y, paired = TRUE): the one-sample JZS Bayes factor of
    the differences, n = number of pairs, df = n - 1."""
    assert number(row["prior.scale"]) == prior, f"{what}: {row['prior.scale']}"
    d = a - b
    n = d.size
    t = float(stats.ttest_1samp(d, 0.0).statistic)
    got = ref.jzs_bf10_t(t, float(n), n - 1.0, r=prior)
    assert_close("ttest_bayes_factor", got, _value(row, "bf10"), what=f"{what} BF10")
    other = math.sqrt(2) / 2 if prior == JZS_PRIOR else JZS_PRIOR
    wrong = ref.jzs_bf10_t(t, float(n), n - 1.0, r=other)
    assert not is_close("ttest_bayes_factor", wrong, _value(row, "bf10")), what


@pytest.mark.parametrize("case", _within_params("bayes", two=True))
def test_paired_jzs_bayes_factor(case: OracleCase) -> None:
    a, b = _conditions(case)
    _check_paired_jzs(case.subtitle_data, a, b, prior=_prior(case), what=case.name)


@pytest.mark.parametrize(
    "case", _within_params("parametric", two=True, needs="caption")
)
def test_paired_jzs_bayes_factor_caption(case: OracleCase) -> None:
    caption = case.caption_data
    assert caption is not None, "no Bayes caption"
    a, b = _conditions(case)
    _check_paired_jzs(caption, a, b, prior=_prior(case), what=f"{case.name} caption")


@pytest.mark.parametrize("case", _within_params("parametric", two=False))
def test_rm_anova_greenhouse_geisser(case: OracleCase) -> None:
    # afex::aov_ez as statsExpressions calls it: the F test with the
    # Greenhouse-Geisser correction (afex's default), df and p corrected.
    # Negative control: the uncorrected df.
    columns = _conditions(case)
    data = np.column_stack(columns)
    result = ref.rm_anova_gg(data)
    row = case.subtitle_data
    assert_close("closed_form", result["F"], _value(row, "statistic"), what="F")
    assert_close("closed_form", result["df1"], _value(row, "df"), what="GG df1")
    assert_close("closed_form", result["df2"], _value(row, "df.error"), what="GG df2")
    assert_close("closed_form", result["p"], _value(row, "p.value"), what="GG p")
    if result["epsilon"] < 1:
        k = data.shape[1]
        assert not is_close("closed_form", k - 1.0, _value(row, "df"))


def _rm_omega(data: FloatArray) -> tuple[float, float, float, float]:
    """Partial omega squared of a one-way repeated-measures design, as
    effectsize::omega_squared gives it for an afex model (Olejnik & Algina 2003):
    df_c (MS_c - MS_e) / (SS_c + SS_e + SS_s + MS_s), floored at 0, with the
    condition (c), error (e) and subject (s) sums of squares. Returns (omega, F,
    df_c, df_e), the F test uncorrected."""
    n, k = data.shape
    grand = data.mean()
    ss_c = n * float(((data.mean(axis=0) - grand) ** 2).sum())
    ss_s = k * float(((data.mean(axis=1) - grand) ** 2).sum())
    ss_e = float(((data - grand) ** 2).sum()) - ss_c - ss_s
    df_c, df_e, df_s = k - 1.0, (k - 1.0) * (n - 1.0), n - 1.0
    ms_c, ms_e, ms_s = ss_c / df_c, ss_e / df_e, ss_s / df_s
    omega = max(0.0, df_c * (ms_c - ms_e) / (ss_c + ss_e + ss_s + ms_s))
    return omega, ms_c / ms_e, df_c, df_e


@pytest.mark.parametrize("case", _within_params("parametric", two=False))
def test_rm_omega_squared(case: OracleCase) -> None:
    # The estimate from the error strata (`_rm_omega`); the one-sided CI as for
    # betweenstats (F_w = omega df2 / ((1 - omega) df1), invert the noncentral F at
    # conf.level, lambda / (lambda + df2)), but at the uncorrected df, although the
    # subtitle reports the Greenhouse-Geisser df (probe-rmanova). Negative control:
    # the F test's own omega from F (F_to_omega2) is far off.
    data = np.column_stack(_conditions(case))
    omega, f_stat, df1, df2 = _rm_omega(data)
    row = case.subtitle_data
    assert row["effectsize"] == "Omega2_partial" or "Omega" in str(row["effectsize"])
    assert_close("closed_form", omega, _value(row, "estimate"), what="omega")
    assert not is_close(
        "closed_form",
        ref.omega2_partial_from_f(f_stat, df1, df2),
        _value(row, "estimate"),
    )
    level = _value(row, "conf.level")
    assert level == float(case.args.get("conf.level", 0.95))
    assert _value(row, "conf.high") == 1.0
    f_w = omega * df2 / ((1 - omega) * df1) if omega > 0 else 0.0
    lam = _f_ncp(f_w, df1, df2, level)
    assert_close(
        "noncentral_ci", lam / (lam + df2), _value(row, "conf.low"), what="omega CI low"
    )


def _pairwise_rows(case: OracleCase) -> dict[tuple[str, str], Row]:
    rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    k = len(case.analysis_levels)
    assert len(rows) == k * (k - 1) // 2
    return rows


def _check_adjusted(
    case: OracleCase, raw: dict[tuple[str, str], float], *, kind: str = "closed_form"
) -> None:
    method = _adjust_method(case)
    keys = list(raw)
    adjusted = dict(zip(keys, _p_adjust([raw[k] for k in keys], method), strict=True))
    rows = _pairwise_rows(case)
    # wrs2_p's absolute part scales with what multiplies the one-sided tail: 2 for
    # two-sided, times at most the number of pairs after the adjustment.
    scale = 2.0 * len(keys) if kind == "wrs2_p" else 1.0
    for key, row in rows.items():
        assert row["p.adjust.method"] == P_ADJUST_LABELS[method]
        assert_close(
            kind,
            adjusted[key],
            number(row["p.value"]),
            what=f"{case.name} {key} {method} p",
            atol_scale=scale,
        )


def _level_pairs(case: OracleCase) -> list[tuple[int, int]]:
    k = len(case.analysis_levels)
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


@pytest.mark.parametrize(
    "case", _within_params("parametric", two=False, needs="pairwise")
)
def test_paired_t_pairwise(case: OracleCase) -> None:
    # stats::pairwise.t.test(paired = TRUE, p.adjust.method = "none"), then the
    # case's method over all pairs.
    columns = _conditions(case)
    levels = case.analysis_levels
    raw = {
        (levels[i], levels[j]): float(stats.ttest_rel(columns[i], columns[j]).pvalue)
        for i, j in _level_pairs(case)
    }
    assert {str(r["test"]) for r in case.pairwise} == {"Student's t"}
    _check_adjusted(case, raw)


@pytest.mark.parametrize("case", _within_params("nonparametric", two=False))
def test_friedman_and_kendalls_w(case: OracleCase) -> None:
    # stats::friedman.test (with its tie correction) and Kendall's W = chi^2 /
    # (n (k - 1)), the tie-corrected W (effectsize::kendalls_w).
    columns = _conditions(case)
    result = stats.friedmanchisquare(*columns)
    row = case.subtitle_data
    n, k = columns[0].size, len(columns)
    assert_close(
        "closed_form", float(result.statistic), _value(row, "statistic"), what="chi2"
    )
    assert_close("closed_form", k - 1.0, _value(row, "df.error"), what="df")
    assert_close("closed_form", float(result.pvalue), _value(row, "p.value"), what="p")
    assert_close(
        "closed_form",
        float(result.statistic) / (n * (k - 1)),
        _value(row, "estimate"),
        what="Kendall's W",
    )


def _durbin_conover(
    columns: Sequence[FloatArray],
) -> dict[tuple[int, int], tuple[float, float]]:
    """Durbin-Conover pairwise tests for complete blocks (Conover 1999, sec. 5.9):
    ranks within each subject, R_j the condition rank sums, A = sum r^2, C = b k (k
    + 1)^2 / 4, T = (k - 1) / (A - C) sum (R_j - b (k + 1) / 2)^2; for i < j,
    t = |R_i - R_j| / sqrt((A - C) 2 b / ((b - 1)(k - 1)) (1 - T / (b (k - 1)))) on
    (b - 1)(k - 1) df, two-sided."""
    data = np.column_stack(columns)
    b, k = data.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, data)
    sums = ranks.sum(axis=0)
    a = float((ranks**2).sum())
    c = b * k * (k + 1) ** 2 / 4
    t_stat = (k - 1) / (a - c) * float(((sums - b * (k + 1) / 2) ** 2).sum())
    df = (b - 1) * (k - 1)
    se = math.sqrt((a - c) * 2 * b / df * (1 - t_stat / (b * (k - 1))))
    out = {}
    for i in range(k):
        for j in range(i + 1, k):
            t = abs(float(sums[i] - sums[j])) / se
            out[(i, j)] = (t, float(2 * stats.t.sf(t, df)))
    return out


@pytest.mark.parametrize(
    "case", _within_params("nonparametric", two=False, needs="pairwise")
)
def test_durbin_conover_pairwise(case: OracleCase) -> None:
    levels = case.analysis_levels
    reference = _durbin_conover(_conditions(case))
    rows = _pairwise_rows(case)
    raw = {}
    for (i, j), (t, p) in reference.items():
        key = (levels[i], levels[j])
        assert rows[key]["test"] == "Durbin-Conover"
        assert_close(
            "closed_form", t, abs(number(rows[key]["statistic"])), what=f"{key} t"
        )
        raw[key] = p
    _check_adjusted(case, raw)


@pytest.mark.parametrize("case", _within_params("robust", two=False, needs="pairwise"))
def test_rmmcp_pairwise(case: OracleCase) -> None:
    # WRS2::rmmcp: per pair, the trimmed mean of the differences (group1 - group2)
    # and its Tukey-McLaughlin test, p = 2 (1 - pt) (probe-wrs2within); ggstatsplot
    # then applies the case's p.adjust method (rmmcp's own p.crit is not used).
    columns = _conditions(case)
    levels = case.analysis_levels
    rows = _pairwise_rows(case)
    tr = _tr(case)
    raw = {}
    for i, j in _level_pairs(case):
        key = (levels[i], levels[j])
        tm, t, df = _trimmed_one_sample(columns[i] - columns[j], tr)
        assert_close(
            "closed_form", tm, number(rows[key]["estimate"]), what=f"{key} psihat"
        )
        raw[key] = 2 * _r_upper_tail(float(stats.t.sf(abs(t), df)))
    _check_adjusted(case, raw, kind="wrs2_p")


@pytest.mark.parametrize("case", _within_params("bayes", two=False, needs="pairwise"))
def test_paired_jzs_bayes_factor_pairwise(case: OracleCase) -> None:
    columns = dict(zip(case.analysis_levels, _conditions(case), strict=True))
    rows = _pairwise_rows(case)
    for (g1, g2), row in rows.items():
        _check_paired_jzs(
            row,
            columns[g1],
            columns[g2],
            prior=JZS_PRIOR,
            what=f"{case.name} {g1}-{g2}",
        )
