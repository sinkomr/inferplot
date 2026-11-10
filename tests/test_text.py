"""Text rendering, symbol tables, APA sentence, alt text and caption lines."""

from __future__ import annotations

import dataclasses
import math
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot import text as tx
from inferplot.result import (
    Descriptive,
    EffectSize,
    PairwiseComparison,
    Settings,
    StatsResult,
    TestResult,
)
from tests.oracle_fixtures import OracleCase, load_cases, read_csv
from tests.plotmath import parse
from tests.test_text_expressions import ADJUSTED, result_from_fixture


def _case(name: str) -> OracleCase:
    (case,) = [c for c in load_cases() if c.name == f"ggbetweenstats-{name}"]
    return case


# --------------------------------------------------------------------------- #
# symbol tables equal statsExpressions 2.2.1's quoted expressions
# --------------------------------------------------------------------------- #

# R/switch-functions.R, extract_statistic_text()
STATISTIC_SOURCES = {
    "welch_t": 'italic("t")["Welch"]',
    "fisher_f": 'italic("F")["Fisher"]',
    "bootstrap_t": 'italic("t")["bootstrapped"]',
    "spearman_s": 'italic("S")',
    "student_t": 'italic("t")["Student"]',
    "mann_whitney_w": 'italic("W")["Mann-Whitney"]',
    "wilcoxon_v": 'italic("V")["Wilcoxon"]',
    "welch_f": 'italic("F")["Welch"]',
    "yuen_t": 'italic("t")["Yuen"]',
    "trimmed_means_f": 'italic("F")["trimmed-means"]',
    "meta_z": 'italic("z")',
    "friedman_chi2": 'chi["Friedman"]^2',
    "kruskal_wallis_chi2": 'chi["Kruskal-Wallis"]^2',
    "gof_chi2": 'chi["gof"]^2',
    "pearson_chi2": 'chi["Pearson"]^2',
    "mcnemar_chi2": 'chi["McNemar"]^2',
}

# extract_estimate_type()
EFFECT_SIZE_SOURCES = {
    "pearsons_c": 'widehat(italic("C"))["Pearson"]',
    "cohens_d": 'widehat(italic("d"))["Cohen"]',
    "hedges_g": 'widehat(italic("g"))["Hedges"]',
    "rank_biserial": 'widehat(italic("r"))["biserial"]^"rank"',
    "cramers_v": 'widehat(italic("V"))["Cramer"]',
    "cohens_g": 'widehat(italic("g"))["Cohen"]',
    "bayes_r2": 'widehat(italic(R^"2"))["Bayesian"]',
    "spearman_rho": 'widehat(rho)["Spearman"]',
    "bayes_pearson_rho": 'widehat(rho)["Pearson"]',
    "posterior_difference": 'widehat(delta)["difference"]',
    "akp_delta": 'widehat(delta)["R"]^"AKP"',
    "akp_delta_avg": 'widehat(delta)["R-avg"]^"AKP"',
    "partial_eta_squared": 'widehat(eta["p"]^2)',
    "partial_omega_squared": 'widehat(omega["p"]^2)',
    "trimmed_mean": 'widehat(mu)["trimmed"]',
    "rank_epsilon_squared": 'widehat(epsilon)["ordinal"]^2',
    "explanatory_xi": "widehat(xi)",
    "meta_beta": 'widehat(beta)["summary"]^"meta"',
    "pearson_r": 'widehat(italic("r"))["Pearson"]',
    "winsorized_r": 'widehat(italic("r"))["Winsorized"]',
    "kendalls_w": 'widehat(italic("W"))["Kendall"]',
}

# prior_switch()
PRIOR_SOURCES = {
    "gunel_dickey": 'italic("a")["Gunel-Dickey"]',
    "beta": 'italic("r")["beta"]^"JZS"',
    "cauchy": 'italic("r")["Cauchy"]^"JZS"',
}


@pytest.mark.parametrize(
    ("table", "sources"),
    [
        (tx.STATISTIC_SYMBOLS, STATISTIC_SOURCES),
        (tx.EFFECT_SIZE_SYMBOLS, EFFECT_SIZE_SOURCES),
        (tx.PRIOR_SYMBOLS, PRIOR_SOURCES),
    ],
    ids=["statistics", "effect_sizes", "priors"],
)
def test_symbol_tables_equal_statsexpressions(
    table: dict[str, tx.Node], sources: dict[str, str]
) -> None:
    assert set(table) == set(sources)
    for key, source in sources.items():
        assert table[key] == parse(source), key


def test_posterior_superscript_attaches_as_r_parses_it() -> None:
    """``{es}^'posterior'`` pasted after a term with ``^`` nests (right-assoc)."""
    built = tx.bayes_expression(
        log_bf01=0.5,
        effect_size="rank_biserial",
        estimate=0.1,
        conf_level=0.95,
        conf_method="ETI",
        conf_low=-0.2,
        conf_high=0.3,
        prior="cauchy",
        prior_scale=0.707,
    )
    source = (
        'list(log[e]*(BF["01"])=="0.50", '
        'widehat(italic("r"))["biserial"]^"rank"^"posterior"=="0.10", '
        'CI["95%"]^ETI~"["*"-0.20", "0.30"*"]", italic("r")["Cauchy"]^"JZS"=="0.71")'
    )
    assert built == parse(source)


# --------------------------------------------------------------------------- #
# templates beyond the fixtures
# --------------------------------------------------------------------------- #


def test_frequentist_template_df_variants() -> None:
    common = {
        "statistic_value": 1.5,
        "p_value": 0.2,
        "estimate": 0.3,
        "conf_level": 0.9,
        "conf_low": -0.1,
        "conf_high": 0.7,
        "n_obs": 1234,
    }
    no_df = tx.frequentist_expression(
        statistic="mann_whitney_w", effect_size="rank_biserial", **common
    )
    assert no_df == parse(
        'list(italic("W")["Mann-Whitney"]=="1.50", italic(p)=="0.20", '
        'widehat(italic("r"))["biserial"]^"rank"=="0.30", '
        'CI["90%"]~"["*"-0.10", "0.70"*"]", italic("n")["obs"]=="1,234")'
    )
    chi = tx.frequentist_expression(
        statistic="pearson_chi2", effect_size="cramers_v", df=4.0, paired=True, **common
    )
    assert chi == parse(
        'list(chi["Pearson"]^2*"("*4*")"=="1.50", italic(p)=="0.20", '
        'widehat(italic("V"))["Cramer"]=="0.30", '
        'CI["90%"]~"["*"-0.10", "0.70"*"]", italic("n")["pairs"]=="1,234")'
    )
    na_df = tx.frequentist_expression(
        statistic="welch_t",
        effect_size="hedges_g",
        df_error=math.nan,
        digits_df_error=2,
        **common,
    )
    assert na_df == parse(
        'list(italic("t")["Welch"]*"("*NA*")"=="1.50", italic(p)=="0.20", '
        'widehat(italic("g"))["Hedges"]=="0.30", '
        'CI["90%"]~"["*"-0.10", "0.70"*"]", italic("n")["obs"]=="1,234")'
    )


def test_unknown_names_raise() -> None:
    with pytest.raises(ValueError, match="unknown statistic"):
        tx.frequentist_expression(
            statistic="z_test",
            statistic_value=1,
            p_value=1,
            effect_size="hedges_g",
            estimate=0,
            conf_level=0.95,
            conf_low=0,
            conf_high=1,
            n_obs=3,
        )
    with pytest.raises(ValueError, match="unknown p_adjust_method"):
        tx.p_adjust_label("holm-bonferroni")
    with pytest.raises(ValueError, match="unknown centrality"):
        tx.centrality_expression(centrality="mode", value=1.0)
    with pytest.raises(ValueError, match="unknown CI method"):
        tx.bayes_expression(
            log_bf01=0,
            effect_size="posterior_difference",
            estimate=0,
            conf_level=0.95,
            conf_method="BOGUS",
            conf_low=0,
            conf_high=1,
            prior="cauchy",
            prior_scale=0.707,
        )
    test = TestResult("kruskal", "K", "m", "chi2", 1.0, 2.0, None, 0.5, "two-sided")
    es = EffectSize("hedges_g", "g", 0.1, 0.95, 0.0, 0.2, "x", "two-sided")
    with pytest.raises(NotImplementedError, match="kruskal"):
        tx.subtitle_expression(test, es, n_obs=10, digits=2)


@pytest.mark.parametrize(
    ("method", "plain"),
    [
        ("holm", "p_Holm-adj. = 0.01"),
        ("hochberg", "p_Hochberg-adj. = 0.01"),
        ("hommel", "p_Hommel-adj. = 0.01"),
        ("bonferroni", "p_Bonferroni-adj. = 0.01"),
        ("BH", "p_FDR-adj. = 0.01"),
        ("fdr", "p_FDR-adj. = 0.01"),
        ("BY", "p_BY-adj. = 0.01"),
        ("none", "p_unadj. = 0.01"),
    ],
)
def test_pairwise_p_labels(method: str, plain: str) -> None:
    node = tx.pairwise_p_expression(p_adjusted=0.0123, adjust_method=method)
    assert tx.to_plain(node) == plain
    sub = "unadj." if method == "none" else f'"{tx.p_adjust_label(method)}"-adj.'
    assert node == parse(f'list(italic(p)[{sub}]=="0.01")')


# --------------------------------------------------------------------------- #
# to_mathtext and to_plain golden examples
# --------------------------------------------------------------------------- #

GOLDEN = [
    (
        'list(italic("t")["Welch"] * "(" * 18.33 * ")" == "-3.77", italic(p) == '
        '"1.37e-03", widehat(italic("g"))["Hedges"] == "-1.35", CI["95%"] ~ "[" * '
        '"-2.17", "-0.51" * "]", italic("n")["obs"] == "32")',
        r"${\mathit{t}}_{\text{Welch}}\text{(}\text{18.33}\text{)} = \text{-3.77},"
        r"\ \mathit{p} = \text{1.37e-03},\ {\widehat{\mathit{g}}}_{\text{Hedges}} = "
        r"\text{-1.35},\ {\mathrm{CI}}_{\text{95%}}\ \text{[}\text{-2.17},\ "
        r"\text{-0.51}\text{]},\ {\mathit{n}}_{\text{obs}} = \text{32}$",
        "t_Welch(18.33) = -3.77, p = 1.37e-03, g_Hedges = -1.35, "
        "CI_95% [-2.17, -0.51], n_obs = 32",
    ),
    (
        'list(italic("F")["Welch"](2, 17.13) == "5.18", italic(p) == "0.02", '
        'widehat(omega["p"]^2) == "0.29", CI["95%"] ~ "[" * "0.00", "1.00" * "]", '
        'italic("n")["obs"] == "30")',
        r"${\mathit{F}}_{\text{Welch}}(\text{2},\ \text{17.13}) = \text{5.18},\ "
        r"\mathit{p} = \text{0.02},\ \widehat{{\mathrm{\omega}}_{\text{p}}^{\text{2}}}"
        r" = \text{0.29},\ {\mathrm{CI}}_{\text{95%}}\ \text{[}\text{0.00},\ "
        r"\text{1.00}\text{]},\ {\mathit{n}}_{\text{obs}} = \text{30}$",
        "F_Welch(2, 17.13) = 5.18, p = 0.02, ω_p^2 = 0.29, CI_95% [0.00, 1.00], "
        "n_obs = 30",
    ),
    (
        'list(log[e] * (BF["01"]) == "-0.18", italic("r")["Cauchy"]^"JZS" == "0.71")',
        r"${\mathrm{log}}_{\mathrm{e}}({\mathrm{BF}}_{\text{01}}) = \text{-0.18},\ "
        r"{\mathit{r}}_{\text{Cauchy}}^{\text{JZS}} = \text{0.71}$",
        "log_e(BF_01) = -0.18, r_Cauchy^JZS = 0.71",
    ),
    (
        'list(italic(p)["Holm" - adj.] == "0.47")',
        r"${\mathit{p}}_{\text{Holm}-\mathrm{adj.}} = \text{0.47}$",
        "p_Holm-adj. = 0.47",
    ),
    (
        'list(widehat(mu)[mean] == "5.03")',
        r"${\widehat{\mathrm{\mu}}}_{\mathrm{mean}} = \text{5.03}$",
        "μ_mean = 5.03",
    ),
    (
        'list(widehat(italic(R^"2"))["Bayesian"]^"posterior" == "0.16")',
        r"${\widehat{{\mathit{R}}^{\mathit{2}}}}_{\text{Bayesian}}^{\text{posterior}}"
        r" = \text{0.16}$",
        "R^2_Bayesian^posterior = 0.16",
    ),
    (
        'list(chi["Kruskal-Wallis"]^2 * "(" * 2 * ")" == "25.75")',
        r"${\mathrm{\chi}}_{\text{Kruskal-Wallis}}^{\text{2}}\text{(}\text{2}\text{)}"
        r" = \text{25.75}$",
        "χ_Kruskal-Wallis^2(2) = 25.75",
    ),
]


@pytest.mark.parametrize(("source", "mathtext", "plain"), GOLDEN)
def test_golden_renderings(source: str, mathtext: str, plain: str) -> None:
    node = parse(source)
    assert tx.to_mathtext(node) == mathtext
    assert tx.to_plain(node) == plain


def test_mathtext_escapes_and_rejects() -> None:
    assert tx.to_mathtext(tx.Str(r"a\b {c} $d")) == r"$\text{a\\b \{c\} \$d}$"
    with pytest.raises(ValueError, match="line break"):
        tx.to_mathtext(tx.Str("a\nb"))
    with pytest.raises(ValueError, match="letters and digits"):
        tx.to_mathtext(tx.Call(tx.Sym("italic"), (tx.Str("a-b"),)))
    with pytest.raises(NotImplementedError, match="bold"):
        tx.to_mathtext(tx.Call(tx.Sym("bold"), (tx.Str("x"),)))


def test_tick_label_is_plain_text() -> None:
    assert tx.tick_label("$5 a_b", 1234) == "$5 a_b\n(n = 1,234)"


# --------------------------------------------------------------------------- #
# APA sentence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "apa"),
    [
        (
            "ToothGrowth-parametric",
            "Welch's t-test: t(55.31) = 1.92, p = .061, Hedges' g = 0.49, "
            "95% CI [-0.02, 0.99], n = 60.",
        ),
        (
            "ToothGrowth-greater-parametric",
            "Welch's t-test (one-sided: OJ greater than VC): t(55.31) = 1.92, "
            "p = .030, Hedges' g = 0.49, 95% CI [0.06, Inf], n = 60.",
        ),
        (
            "ToothGrowth-less-parametric",
            "Welch's t-test (one-sided: OJ less than VC): t(55.31) = 1.92, p = .970, "
            "Hedges' g = 0.49, 95% CI [-Inf, 0.91], n = 60.",
        ),
        (
            "PlantGrowth-parametric",
            "Welch's ANOVA: F(2, 17.13) = 5.18, p = .017, partial omega squared = "
            ".29, 95% CI [.00, 1.00], n = 30.",
        ),
        (
            "iris-parametric",
            "Welch's ANOVA: F(2, 97.40) = 45.01, p < .001, partial omega squared = "
            ".47, 95% CI [.35, 1.00], n = 150.",
        ),
        (
            "mtcars-am-parametric",
            "Welch's t-test: t(18.33) = -3.77, p = .001, Hedges' g = -1.35, "
            "95% CI [-2.17, -0.51], n = 32.",
        ),
        (
            "ToothGrowth-cl90-parametric",
            "Welch's t-test: t(55.31) = 1.92, p = .061, Hedges' g = 0.49, "
            "90% CI [0.06, 0.91], n = 60.",
        ),
        (
            "PlantGrowth-cl90-parametric",
            "Welch's ANOVA: F(2, 17.13) = 5.18, p = .017, partial omega squared = "
            ".29, 90% CI [.04, 1.00], n = 30.",
        ),
        (
            "edge_large_n-parametric",
            "Welch's ANOVA: F(2, 2372.96) = 16.51, p < .001, partial omega squared = "
            ".01, 95% CI [.01, 1.00], n = 3,600.",
        ),
        # results.subtitle FALSE: no subtitle table, so result_from_fixture computes
        # these numbers with inferplot.stats; they read as their siblings' above.
        (
            "ToothGrowth-nosubtitle-parametric",
            "Welch's t-test: t(55.31) = 1.92, p = .061, Hedges' g = 0.49, "
            "95% CI [-0.02, 0.99], n = 60.",
        ),
        (
            "PlantGrowth-nosubtitle-parametric",
            "Welch's ANOVA: F(2, 17.13) = 5.18, p = .017, partial omega squared = "
            ".29, 95% CI [.00, 1.00], n = 30.",
        ),
    ],
)
def test_apa_sentence_from_fixtures(name: str, apa: str) -> None:
    result = result_from_fixture(_case(name))
    assert result.text.apa == apa
    assert result.text.apa.isascii()


def _welch(**changes: object) -> TestResult:
    base = TestResult(
        name="welch_t",
        label="Welch's t-test",
        method="Welch Two Sample t-test",
        statistic_name="t",
        statistic=2.0,
        df=None,
        df_error=10.0,
        p_value=0.05,
        alternative="two-sided",
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


def _g(**changes: object) -> EffectSize:
    base = EffectSize(
        name="hedges_g",
        label="Hedges' g",
        estimate=0.5,
        conf_level=0.95,
        conf_low=-0.1,
        conf_high=1.1,
        ci_method="noncentral t",
        sidedness="two-sided",
    )
    return dataclasses.replace(base, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("p", "text"),
    [
        (0.0607, "p = .061"),
        (0.001, "p = .001"),
        (0.00099996, "p < .001"),
        (0.0004, "p < .001"),
        (0.0, "p < .001"),
        (0.5, "p = .500"),
        (0.99996, "p = 1.000"),
        (1.0, "p = 1.000"),
    ],
)
def test_apa_p_values(p: float, text: str) -> None:
    sentence = tx.apa_sentence(_welch(p_value=p), _g(), n_obs=12)
    assert f", {text}, " in sentence


def _control_treatment() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    y = np.r_[rng.normal(1.0, 1, 15), rng.normal(0, 1, 15)]
    return pd.DataFrame({"g": np.repeat(["treatment", "control"], 15), "y": y})


@pytest.mark.parametrize("kind", ["parametric", "nonparametric"])
def test_one_sided_direction_names_the_levels(kind: str) -> None:
    """U2: "greater" tests whether the first level (here "control", first in sorted
    order) is greater than the second; the sentence says so with the names."""
    data = _control_treatment()
    analyze: Any = ip.analyze.betweenstats
    result = analyze(data, "g", "y", type=kind, alternative="greater")
    assert result.levels == ("control", "treatment")
    label = "Welch's t-test" if kind == "parametric" else "Mann-Whitney U test"
    heading = f"{label} (one-sided: control greater than treatment): "
    assert result.text.apa.startswith(heading)
    assert heading in result.text.alt_text
    # The subtitle keeps ggstatsplot's text, which names no direction.
    assert "control" not in (result.text.subtitle or "")
    # A program reads the direction from test.alternative and levels.
    assert result.test is not None
    assert (result.test.alternative, *result.levels) == (
        "greater",
        "control",
        "treatment",
    )
    flipped = analyze(
        data, "g", "y", type=kind, alternative="greater", order=["treatment", "control"]
    )
    assert flipped.text.apa.startswith(
        f"{label} (one-sided: treatment greater than control): "
    )
    less = analyze(data, "g", "y", type=kind, alternative="less")
    assert less.text.apa.startswith(
        f"{label} (one-sided: control less than treatment): "
    )


def test_one_sided_direction_shortens_long_names() -> None:
    data = _control_treatment()
    data["g"] = data["g"].replace({"control": "c" * 60, "treatment": "t" * 60})
    analyze: Any = ip.analyze.betweenstats
    result = analyze(data, "g", "y", alternative="less")
    first, second = "c" * 39 + "…", "t" * 39 + "…"
    assert result.text.apa.startswith(
        f"Welch's t-test (one-sided: {first} less than {second}): "
    )
    assert len(result.text.alt_text) <= tx.ALT_TEXT_MAX_CHARS
    assert "(one-sided: " in result.text.alt_text
    assert " less than " in result.text.alt_text


def test_one_sided_direction_within_subjects() -> None:
    # The paired t-test (sleep: group 1 against group 2), one direction each.
    sleep = read_csv("sleep.csv")
    analyze: Any = ip.analyze.withinstats
    result = analyze(sleep, "group", "extra", subject_id="ID", alternative="less")
    assert result.text.apa.startswith("Paired t-test (one-sided: 1 less than 2): ")
    assert "(one-sided: 1 less than 2)" in result.text.alt_text


def test_one_sided_needs_the_level_names() -> None:
    with pytest.raises(ValueError, match="cannot be named"):
        tx.apa_sentence(_welch(alternative="greater"), _g(), n_obs=12)
    assert tx.apa_sentence(
        _welch(alternative="greater"), _g(), n_obs=12, levels=("a", "b")
    ).startswith("Welch's t-test (one-sided: a greater than b): ")


def test_apa_numbers() -> None:
    sentence = tx.apa_sentence(_welch(), _g(estimate=-0.001), n_obs=1234)
    assert sentence == (
        "Welch's t-test: t(10) = 2.00, p = .050, Hedges' g = 0.00, "
        "95% CI [-0.10, 1.10], n = 1,234."
    )
    omega = EffectSize(
        "partial_omega_squared",
        "partial omega squared",
        0.123,
        0.9,
        0.0,
        1.0,
        "nc",
        "greater",
    )
    anova = _welch(
        name="welch_anova",
        label="Welch's ANOVA",
        statistic_name="F",
        df=3.0,
        df_error=17.126,
        statistic=12.345,
    )
    assert tx.apa_sentence(anova, omega, n_obs=30) == (
        "Welch's ANOVA: F(3, 17.13) = 12.35, p = .050, partial omega squared = .12, "
        "90% CI [.00, 1.00], n = 30."
    )
    biserial = EffectSize(
        "rank_biserial", "rank-biserial r", -0.5, 0.95, -0.9, 0.2, "", ""
    )
    assert "rank-biserial r = -.50, 95% CI [-.90, .20]" in tx.apa_sentence(
        _welch(), biserial, n_obs=10
    )


# --------------------------------------------------------------------------- #
# caption lines for every pairwise_display and p_adjust_method
# --------------------------------------------------------------------------- #

DISPLAY_TAIL = {
    "significant": "showing 1 of 3 pairs (significant only).",
    "non-significant": "showing 2 of 3 pairs (non-significant only).",
    "all": "showing all 3 pairs.",
}


def _pairs(display: str, method: str) -> tuple[PairwiseComparison, ...]:
    significant = (True, False, False)
    shown = {
        "significant": significant,
        "non-significant": tuple(not s for s in significant),
        "all": (True, True, True),
        "none": (False, False, False),
    }[display]
    return tuple(
        PairwiseComparison(
            group1=a,
            group2=b,
            statistic_name="q",
            statistic=1.0,
            df=10.0,
            p_value=0.01,
            p_adjusted=p,
            adjust_method=method,
            test="games_howell",
            test_label="Games-Howell",
            significant=s,
            displayed=d,
        )
        for (a, b), p, s, d in zip(
            [("a", "b"), ("a", "c"), ("b", "c")],
            (0.01, 0.2, 0.7),
            significant,
            shown,
            strict=True,
        )
    )


@pytest.mark.parametrize("method", sorted(ADJUSTED))
@pytest.mark.parametrize("display", ["significant", "non-significant", "all", "none"])
def test_caption_lines(display: str, method: str) -> None:
    base = result_from_fixture(_case("PlantGrowth-parametric"))
    settings = dataclasses.replace(
        base.settings, pairwise_display=display, p_adjust_method=method
    )
    pairs = _pairs(display, method)
    texts = tx.build_texts(
        x=base.x,
        y=base.y,
        descriptives=base.descriptives,
        test=base.test,
        effect_size=base.effect_size,
        bayes=None,
        pairwise=pairs,
        n_obs=base.n_obs,
        settings=settings,
    )
    result = dataclasses.replace(base, pairwise=pairs, settings=settings, text=texts)
    fig = tx.figure_texts(result)
    if display == "none":
        assert texts.caption is None
        assert fig.caption_lines == ()
    else:
        line = f"Pairwise: {ADJUSTED[method]}; {DISPLAY_TAIL[display]}"
        assert texts.caption == line
        assert fig.caption_lines == (line,)
    assert len(fig.pairwise_labels) == 3
    label = tx.p_adjust_label(method)
    want = "p_unadj." if label == "None" else f"p_{label}-adj."
    assert texts.pairwise_labels[0] == f"{want} = 0.01"


@pytest.mark.parametrize(
    ("display", "method", "alpha", "tail"),
    [
        ("significant", "holm", 0.05, "showing 1 of 3 pairs (significant only)."),
        (
            "significant",
            "holm",
            0.5,
            "showing 1 of 3 pairs (significant only, adjusted p < 0.5).",
        ),
        (
            "non-significant",
            "BH",
            0.1,
            "showing 2 of 3 pairs (non-significant only, adjusted p >= 0.1).",
        ),
        (
            "significant",
            "none",
            0.01,
            "showing 1 of 3 pairs (significant only, p < 0.01).",
        ),
        ("all", "holm", 0.5, "showing all 3 pairs."),
    ],
)
def test_caption_states_a_non_default_threshold(
    display: str, method: str, alpha: float, tail: str
) -> None:
    """U5: which pairs "significant only" draws depends on pairwise_alpha, so the
    caption names any threshold but ggstatsplot's 0.05."""
    line = tx.pairwise_caption_line(
        _pairs(display, method), display=display, adjust_method=method, alpha=alpha
    )
    assert line is not None
    assert line.endswith("; " + tail)


def test_caption_threshold_end_to_end() -> None:
    plants = read_csv("PlantGrowth.csv")
    analyze: Any = ip.analyze.betweenstats
    default = analyze(plants, "group", "weight")
    assert default.text.caption == (
        "Pairwise: Games-Howell, Holm adjusted; showing 0 of 3 pairs (significant "
        "only)."
    )
    loose = analyze(plants, "group", "weight", pairwise_alpha=0.5)
    drawn = sum(p.displayed for p in loose.pairwise)
    assert loose.text.caption == (
        f"Pairwise: Games-Howell, Holm adjusted; showing {drawn} of 3 pairs "
        "(significant only, adjusted p < 0.5)."
    )
    assert tx.figure_texts(loose).caption_lines == (loose.text.caption,)


def test_no_subtitle_and_no_bayes_line_without_results_subtitle() -> None:
    base = result_from_fixture(_case("ToothGrowth-parametric"))
    settings = dataclasses.replace(base.settings, results_subtitle=False)
    texts = tx.build_texts(
        x=base.x,
        y=base.y,
        descriptives=base.descriptives,
        test=base.test,
        effect_size=base.effect_size,
        bayes=base.bayes,
        pairwise=base.pairwise,
        n_obs=base.n_obs,
        settings=settings,
    )
    assert texts.subtitle is None
    assert texts.caption is None
    assert texts.apa == base.text.apa
    fig = tx.figure_texts(dataclasses.replace(base, settings=settings, text=texts))
    assert fig.subtitle is None
    assert fig.caption_lines == ()


@pytest.mark.parametrize("bf_message", [None, True, False])
def test_bf_message_false_never_draws_the_bayes_line(bf_message: bool | None) -> None:
    """ggstatsplot's bf.message = FALSE (fixture ToothGrowth-nobf) drops the caption,
    even when a result carries a Bayes factor."""
    base = result_from_fixture(_case("ToothGrowth-parametric"))
    assert base.bayes is not None
    settings = dataclasses.replace(base.settings, bf_message=bf_message)
    texts = tx.build_texts(
        x=base.x,
        y=base.y,
        descriptives=base.descriptives,
        test=base.test,
        effect_size=base.effect_size,
        bayes=base.bayes,
        pairwise=base.pairwise,
        n_obs=base.n_obs,
        settings=settings,
    )
    fig = tx.figure_texts(dataclasses.replace(base, settings=settings, text=texts))
    if bf_message is False:
        assert texts.caption is None
        assert fig.caption_lines == ()
    else:
        assert texts.caption == base.text.caption
        assert texts.caption == "log_e(BF_01) = -0.18, r_Cauchy^JZS = 0.71"
        assert len(fig.caption_lines) == 1
    assert texts.subtitle == base.text.subtitle


def test_digits_three() -> None:
    base = result_from_fixture(_case("PlantGrowth-parametric"))
    settings = dataclasses.replace(base.settings, digits=3)
    fig = tx.figure_texts(dataclasses.replace(base, settings=settings))
    assert fig.subtitle is not None
    assert r"(\text{2},\ \text{17.128})" in fig.subtitle
    assert r"\text{5.181}" in fig.subtitle
    assert fig.centrality_labels[0].endswith(r"\text{5.032}$")


# --------------------------------------------------------------------------- #
# alt text
# --------------------------------------------------------------------------- #

_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def _sentences(text: str) -> int:
    return len(_SENTENCE_END.findall(text))


def test_alt_text_on_every_parametric_fixture() -> None:
    # ggbetweenstats; tests/test_text_within.py covers every ggwithinstats fixture.
    for case in load_cases():
        if case.function != "ggbetweenstats":
            continue
        if case.type != "parametric" or case.error is not None:
            continue
        result = result_from_fixture(case)
        alt = result.text.alt_text
        assert len(alt) <= 400, case.name
        assert 2 <= _sentences(alt) <= 4, (case.name, alt)
        assert result.y in alt
        assert result.x in alt
        # Only what the figure shows (U12): the test when the subtitle is drawn,
        # the pair count when brackets are.
        if result.settings.results_subtitle:
            assert result.text.apa in alt
        else:
            n = tx.format_n(result.n_obs)
            assert f"No test result is shown (results_subtitle=False), n = {n}." in alt
            assert result.text.apa not in alt
        for d in result.descriptives:
            assert f"{d.level} (n = {tx.format_n(d.n)}, mean " in alt
        if result.pairwise and result.settings.pairwise_display != "none":
            significant = sum(1 for p in result.pairwise if p.significant)
            assert f"{significant} of {len(result.pairwise)} pairwise" in alt
        else:
            assert "pairwise comparison" not in alt


@pytest.mark.parametrize("kind", ["parametric", "nonparametric", "robust"])
def test_alt_text_describes_only_what_is_drawn(kind: str) -> None:
    """U12: without the results subtitle the figure shows no test, and with
    pairwise_display="none" no brackets; the alt text says neither. Brackets are
    drawn with results_subtitle=False, so their count stays."""
    plants = read_csv("PlantGrowth.csv")
    analyze: Any = ip.analyze.betweenstats
    full = analyze(plants, "group", "weight", type=kind)
    hidden = "No test result is shown (results_subtitle=False), n = 30."
    pairs = " pairwise comparisons "
    assert full.text.apa in full.text.alt_text
    assert pairs in full.text.alt_text

    quiet = analyze(plants, "group", "weight", type=kind, results_subtitle=False)
    assert quiet.text.apa == full.text.apa  # the result still reports the test
    assert quiet.text.alt_text == full.text.alt_text.replace(full.text.apa, hidden)
    assert pairs in quiet.text.alt_text

    for subtitle in (True, False):
        bare = analyze(
            plants,
            "group",
            "weight",
            type=kind,
            results_subtitle=subtitle,
            pairwise_display="none",
        )
        assert "pairwise" not in bare.text.alt_text
        assert (full.text.apa in bare.text.alt_text) == subtitle
        assert (hidden in bare.text.alt_text) == (not subtitle)
        assert 2 <= _sentences(bare.text.alt_text) <= 4


def test_alt_text_without_subtitle_two_groups_and_within() -> None:
    analyze: Any = ip.analyze.betweenstats
    teeth = read_csv("ToothGrowth.csv")
    for kind, words in (
        ("parametric", "No test result is shown (results_subtitle=False), n = 60."),
        ("bayes", "No test was computed (results_subtitle=False), n = 60."),
    ):
        result = analyze(teeth, "supp", "len", type=kind, results_subtitle=False)
        assert result.text.alt_text.endswith(words)
    sleep = read_csv("sleep.csv")
    within: Any = ip.analyze.withinstats
    result = within(sleep, "group", "extra", subject_id="ID", results_subtitle=False)
    assert result.text.alt_text.endswith(
        "No test result is shown (results_subtitle=False), n = 10 pairs."
    )
    assert result.text.apa.startswith("Paired t-test: ")


def test_non_ascii_level_names_stay_as_they_are() -> None:
    """U12: level names are the user's data; the APA sentence prints them
    unchanged, and JSON, str() and the embedded copy carry them."""
    plants = read_csv("PlantGrowth.csv")
    plants["group"] = plants["group"].replace(
        {"ctrl": "Kontrolle", "trt1": "Behandlung ä", "trt2": "处理"}
    )
    analyze: Any = ip.analyze.betweenstats
    result = analyze(plants, "group", "weight", type="bayes", results_subtitle=False)
    assert "Behandlung ä vs 处理, BF10 = " in result.text.apa
    assert not result.text.apa.isascii()
    text = result.to_json(include_values=False)
    assert text.isascii()  # escaped, as strict JSON allows
    assert StatsResult.from_json(text).text.apa == result.text.apa
    assert StatsResult.from_json(result.to_json()) == result
    assert str(result).startswith(result.text.apa)
    long = "Behandlungsgruppe mit einer sehr langen Bezeichnung ä"
    two = plants[plants["group"] != "处理"].replace({"Behandlung ä": long})
    one_sided = analyze(two, "group", "weight", alternative="greater")
    assert one_sided.levels == (long, "Kontrolle")
    assert one_sided.text.apa.startswith(
        f"Welch's t-test (one-sided: {long[:39]}… greater than Kontrolle): "
    )


def test_alt_text_plant_growth() -> None:
    result = result_from_fixture(_case("PlantGrowth-parametric"))
    assert result.text.alt_text == (
        "Plot comparing weight across the 3 groups of group: ctrl (n = 10, mean 5.03), "
        "trt1 (n = 10, mean 4.66), trt2 (n = 10, mean 5.53). Welch's ANOVA: "
        "F(2, 17.13) = 5.18, p = .017, partial omega squared = .29, 95% CI [.00, "
        "1.00], n = 30. 0 of 3 pairwise comparisons are significant (Games-Howell, "
        "Holm adjusted, alpha = 0.05)."
    )


@pytest.mark.parametrize("k", [6, 12, 40])
@pytest.mark.parametrize("name_length", [8, 60, 500])
def test_alt_text_summarizes_many_or_long_groups(k: int, name_length: int) -> None:
    base = result_from_fixture(_case("PlantGrowth-parametric"))
    descriptives = tuple(
        Descriptive(
            level=f"{i}" + "g" * name_length, n=10 + i, centrality="mean", value=i
        )
        for i in range(k)
    )
    texts = tx.build_texts(
        x="x" * name_length,
        y="y" * name_length,
        descriptives=descriptives,
        test=base.test,
        effect_size=base.effect_size,
        bayes=None,
        pairwise=base.pairwise,
        n_obs=sum(d.n for d in descriptives),
        settings=base.settings,
    )
    alt = texts.alt_text
    assert len(alt) <= 400
    assert 2 <= _sentences(alt) <= 4
    assert base.text.apa.split(", n = ")[0] in alt
    assert f"0 of {len(base.pairwise)} pairwise comparisons" in alt


def test_alt_text_never_cuts_the_result_sentence() -> None:
    # A result sentence too long for the limit by itself (an F printed with 251
    # digits) and 500-character names: the text keeps its start and the whole
    # result, leaves the groups out, and runs past 400 characters rather than cut.
    base = result_from_fixture(_case("PlantGrowth-parametric"))
    assert base.test is not None
    test = dataclasses.replace(base.test, statistic=1e250)
    long_name = "x" * 500
    texts = tx.build_texts(
        x=long_name,
        y=long_name,
        descriptives=tuple(
            dataclasses.replace(d, level=d.level + long_name) for d in base.descriptives
        ),
        test=test,
        effect_size=base.effect_size,
        bayes=None,
        pairwise=base.pairwise,
        n_obs=base.n_obs,
        settings=base.settings,
    )
    head = f"Plot comparing {'x' * 11}… across the 3 groups of {'x' * 11}…. "
    assert texts.alt_text.startswith(head)
    assert texts.alt_text == head + texts.apa + (
        " 0 of 3 pairwise comparisons are significant (Games-Howell, Holm adjusted, "
        "alpha = 0.05)."
    )
    assert len(texts.alt_text) > tx.ALT_TEXT_MAX_CHARS


def test_settings_in_figure_texts() -> None:
    base = result_from_fixture(_case("PlantGrowth-parametric"))
    settings = Settings(**{**dataclasses.asdict(base.settings), "digits": 1})
    fig = tx.figure_texts(dataclasses.replace(base, settings=settings))
    assert (
        fig.pairwise_labels[0]
        == r"${\mathit{p}}_{\text{Holm}-\mathrm{adj.}} = \text{0.5}$"
    )
