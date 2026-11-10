"""Every fixture expression, rebuilt by the production builders, is AST-equal to R's.

For every fixture of every function and type (a fixture added later is picked up
without new code), each plotmath expression ggstatsplot built is parsed with
``tests/plotmath.py`` and compared with the expression inferplot's builders make
from the *numbers* in the same fixture: subtitles, full Bayes captions, inferplot's
v0.1 two-group caption (the first and last items of ggstatsplot's), pairwise labels,
bracket annotations, both centrality label layers and the n tick labels. A fixture
whose expression has a structure no builder makes fails; nothing is skipped. A case
where ggstatsplot failed (edge_n2) is checked on what its hidden calls computed
before the failure (``OracleCase.partial``).

The completed parametric fixtures are also turned into hand-built ``StatsResult``
objects that honour each case's arguments (the analysis layer arrives separately),
and ``build_texts`` and ``figure_texts`` must give the fixture's texts as plain text
and as mathtext: no subtitle and no Bayes caption where the case asks for none, the
case's adjustment in every bracket label and in the caption line, and exactly the
pairs ggstatsplot drew.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import pytest

from inferplot import text as tx
from inferplot.result import (
    BayesFactor,
    Descriptive,
    EffectSize,
    PairwiseComparison,
    Settings,
    StatsResult,
    TestResult,
    current_provenance,
    tied_count,
)
from inferplot.stats import (
    games_howell,
    hedges_g,
    partial_omega_squared,
    welch_anova,
    welch_t,
)
from tests.oracle_fixtures import (
    OracleCase,
    number,
    oracle_params,
    r_level_label,
)
from tests.plotmath import parse
from tests.stats.fixture_data import P_ADJUST, groups
from tests.stats.fixture_data import alternative as case_alternative
from tests.test_text_format import (
    all_case_params,
    caption_rows,
    case_digits,
    centrality_rows,
    centrality_value,
    df_digits,
    paired,
    subtitle_rows,
    text_cases,
)
from tests.within_np_cases import HIDDEN

Row = Mapping[str, Any]


# --------------------------------------------------------------------------- #
# statsExpressions' switches, mirrored on the test side (R/switch-functions.R)
# --------------------------------------------------------------------------- #


def statistic_key(method: str) -> str:
    """``extract_statistic_text(tolower(method))`` -> STATISTIC_SYMBOLS key."""
    x = method.lower()
    rules: list[tuple[bool, str]] = [
        (bool(re.search(r"^welch", x)), "welch_t"),
        (bool(re.search(r"afex| of means$", x)), "fisher_f"),
        (x.startswith("boot"), "bootstrap_t"),
        (x.startswith("spearman"), "spearman_s"),
        (bool(re.search(r"t-test$|correlation$", x)), "student_t"),
        ("wilcoxon rank" in x, "mann_whitney_w"),
        ("wilcoxon signed" in x, "wilcoxon_v"),
        ("variances" in x, "welch_f"),
        ("yuen" in x, "yuen_t"),
        ("heteroscedastic" in x, "trimmed_means_f"),
        ("meta" in x, "meta_z"),
        ("friedman" in x, "friedman_chi2"),
        ("kruskal" in x, "kruskal_wallis_chi2"),
        ("probabilities" in x, "gof_chi2"),
        ("pearson's chi" in x, "pearson_chi2"),
        ("mcnemar's chi" in x, "mcnemar_chi2"),
    ]
    for hit, key in rules:
        if hit:
            return key
    raise KeyError(f"no statistic symbol for method {method!r}")


def effect_size_key(effectsize: str) -> str:
    """``extract_estimate_type(tolower(effectsize))`` -> EFFECT_SIZE_SYMBOLS key."""
    x = effectsize.lower()
    rules: list[tuple[bool, str]] = [
        ("pearson's c" in x, "pearsons_c"),
        ("cohen's d" in x, "cohens_d"),
        ("hedges' g" in x, "hedges_g"),
        ("biserial" in x, "rank_biserial"),
        ("cramer" in x, "cramers_v"),
        ("cohen's g" in x, "cohens_g"),
        ("r-squared" in x, "bayes_r2"),
        ("spearman" in x, "spearman_rho"),
        (x.startswith("bayesian pearson"), "bayes_pearson_rho"),
        (bool(re.search(r"posterior|t-", x)), "posterior_difference"),
        (x.endswith("difference"), "akp_delta"),
        (x.endswith("average"), "akp_delta_avg"),
        (x.startswith("eta2"), "partial_eta_squared"),
        (x.startswith("omega2"), "partial_omega_squared"),
        (x.startswith("trimmed"), "trimmed_mean"),
        (x.startswith("epsilon2"), "rank_epsilon_squared"),
        (x.startswith("explanatory"), "explanatory_xi"),
        (x.startswith("meta"), "meta_beta"),
        (x.startswith("pearson"), "pearson_r"),
        (x.startswith("winsor"), "winsorized_r"),
        (x.startswith("kendall"), "kendalls_w"),
    ]
    for hit, key in rules:
        if hit:
            return key
    raise KeyError(f"no effect-size symbol for {effectsize!r}")


def prior_key(method: str) -> str:
    """``prior_switch(tolower(method))``."""
    x = method.lower()
    if "contingency" in x:
        return "gunel_dickey"
    if "correlation" in x:
        return "beta"
    return "cauchy"


_CENTRALITY = {
    "parametric": "mean",
    "nonparametric": "median",
    "robust": "trimmed_mean",
    "bayes": "map",
}


# --------------------------------------------------------------------------- #
# rebuilding one fixture's expressions from its numbers
# --------------------------------------------------------------------------- #


def rebuild_row(case: OracleCase, row: Row) -> tx.Node:
    """The subtitle or caption expression of a stats row, from its numbers."""
    digits = case_digits(case)
    if "bf10" in row:
        return tx.bayes_expression(
            log_bf01=-math.log(number(row["bf10"])),
            effect_size=effect_size_key(str(row["effectsize"])),
            estimate=number(row["estimate"]),
            conf_level=number(row["conf.level"]),
            conf_method=str(row["conf.method"]),
            conf_low=number(row["conf.low"]),
            conf_high=number(row["conf.high"]),
            prior=prior_key(str(row["method"])),
            prior_scale=number(row["prior.scale"]),
            digits=digits,
        )
    d_df, d_dfe = df_digits(str(row["method"]), digits)
    return tx.frequentist_expression(
        statistic=statistic_key(str(row["method"])),
        statistic_value=number(row["statistic"]),
        p_value=number(row["p.value"]),
        effect_size=effect_size_key(str(row["effectsize"])),
        estimate=number(row["estimate"]),
        conf_level=number(row["conf.level"]),
        conf_low=number(row["conf.low"]),
        conf_high=number(row["conf.high"]),
        n_obs=int(row["n.obs"]),
        df=number(row["df"]) if "df" in row else None,
        df_error=number(row["df.error"]) if "df.error" in row else None,
        digits=digits,
        digits_df=d_df,
        digits_df_error=d_dfe,
        paired=paired(case),
    )


def rebuild_pair(case: OracleCase, row: Row) -> tx.Node:
    """A pairwise label from its row: log BF01 for Bayes rows, adjusted p otherwise."""
    digits = case_digits(case)
    if "bf10" in row:
        return tx.pairwise_bf_expression(
            log_bf01=-math.log(number(row["bf10"])), digits=digits
        )
    return tx.pairwise_p_expression(
        p_adjusted=number(row["p.value"]),
        adjust_method=str(case.args.get("p.adjust.method", "holm")),
        digits=digits,
    )


def rebuild_centrality(case: OracleCase, row: Row) -> tx.Node:
    return tx.centrality_expression(
        centrality=_CENTRALITY[case.type],
        value=centrality_value(case, row),
        digits=case_digits(case),
    )


def v01_caption(case: OracleCase) -> tx.Node | None:
    """inferplot's v0.1 caption for a two-group parametric fixture, from its numbers.

    None for other fixtures and where ggstatsplot computed no Bayes caption
    (``results.subtitle`` or ``bf.message`` FALSE).
    """
    if case.type != "parametric" or len(case.analysis_levels) != 2:
        return None
    rows = caption_rows(case)
    if not rows:
        return None
    row = rows[0]
    return tx.bayes_factor_expression(
        log_bf01=-math.log(number(row["bf10"])),
        prior=prior_key(str(row["method"])),
        prior_scale=number(row["prior.scale"]),
        digits=case_digits(case),
    )


def comparisons(case: OracleCase) -> list[tuple[str, str]]:
    """The pairs ggstatsplot drew brackets for, in its order."""
    out: list[tuple[str, str]] = []
    for params in case.layer_params("GeomSignif"):
        out.extend((str(a), str(b)) for a, b in params["stat"]["comparisons"])
    return out


def checked_expressions(case: OracleCase) -> list[tuple[str, str, tx.Node]]:
    """(location, R's deparsed string, inferplot's rebuilt AST) for one fixture.

    A case where ggstatsplot failed is checked on what its hidden calls computed
    before the failure (``OracleCase.partial``): subtitle, caption and centrality
    labels.
    """
    out: list[tuple[str, str, tx.Node]] = []
    subtitles = subtitle_rows(case)
    for i, row in enumerate(subtitles):
        out.append(
            (f"subtitle_data[{i}]", str(row["expression"]), rebuild_row(case, row))
        )
    if case.subtitle is not None:
        out.append(("subtitle", case.subtitle, rebuild_row(case, subtitles[0])))
    captions = caption_rows(case)
    for i, row in enumerate(captions):
        out.append(
            (f"caption_data[{i}]", str(row["expression"]), rebuild_row(case, row))
        )
    if case.caption is not None:
        out.append(("caption", case.caption, rebuild_row(case, captions[0])))
    for geom, row in centrality_rows(case):
        out.append(
            (
                f"{geom}.expression",
                str(row["expression"]),
                rebuild_centrality(case, row),
            )
        )
    if case.error is not None:
        return out  # ggstatsplot failed in the pairwise call: no pairs, no plot
    rows = case.pairwise
    by_pair = {(str(r["group1"]), str(r["group2"])): r for r in rows}
    for i, row in enumerate(rows):
        out.append((f"pairwise[{i}]", str(row["expression"]), rebuild_pair(case, row)))
    for params in case.layer_params("GeomSignif"):
        stat = params["stat"]
        annotations = stat["annotations"]
        annotations = (
            [annotations] if isinstance(annotations, str) else list(annotations)
        )
        pairs = [(str(a), str(b)) for a, b in stat["comparisons"]]
        assert len(pairs) == len(annotations)
        for pair, ann in zip(pairs, annotations, strict=True):
            out.append(
                (f"annotation{pair}", str(ann), rebuild_pair(case, by_pair[pair]))
            )
    for layer in case.plot_layers:
        for j, built in enumerate(layer["built"] or []):
            # group "setosa-versicolor-1": the bracket's 1-based comparison index
            index = int(str(built["group"]).rsplit("-", 1)[1]) - 1
            pair = comparisons(case)[index]
            out.append(
                (
                    f"built[{j}]",
                    str(built["annotation"]),
                    rebuild_pair(case, by_pair[pair]),
                )
            )
    return out


def results_subtitle(case: OracleCase) -> bool:
    return bool(case.args.get("results.subtitle", True))


def bf_message(case: OracleCase) -> bool | None:
    value = case.args.get("bf.message")
    return None if value is None else bool(value)


def bayes_caption_drawn(case: OracleCase) -> bool:
    """inferplot's rule for ggstatsplot's Bayes caption (SPEC section 2, D28).

    Parametric type, ``results.subtitle`` TRUE and ``bf.message`` not FALSE.
    """
    return (
        case.type == "parametric"
        and results_subtitle(case)
        and bf_message(case) is not False
    )


@pytest.mark.parametrize("case", all_case_params())
def test_expressions_equal_ggstatsplot(case: OracleCase) -> None:
    checked = checked_expressions(case)
    assert checked
    where = {w.split("[")[0] for w, _, _ in checked}
    # Each fixture holds the texts its own arguments ask for, except where
    # ggstatsplot swallowed an error and dropped one (tests/within_np_cases.py).
    has_subtitle = "subtitle_data" in where
    dropped = HIDDEN.get(case.name) == "subtitle"
    assert has_subtitle == (results_subtitle(case) and not dropped)
    assert ("subtitle" in where) == (has_subtitle and case.error is None)
    if case.type == "parametric":
        assert ("caption_data" in where) == bayes_caption_drawn(case)
    for location, source, rebuilt in checked:
        assert rebuilt == parse(source), f"{location}: {source}"


@pytest.mark.parametrize("case", oracle_params("ggbetweenstats", failed=True))
def test_failed_case_is_checked_on_partial_outputs(case: OracleCase) -> None:
    """Where ggstatsplot fails (edge_n2: in the pairwise call), every expression its
    hidden calls computed before the failure is still rebuilt and checked."""
    assert case.expected_error is not None
    where = [w for w, _, _ in checked_expressions(case)]
    for component in ("subtitle", "caption"):
        computed = component in case.partial_components
        rows = (case.partial(component) if computed else None) or []
        found = sum(w.startswith(f"{component}_data[") for w in where)
        assert found == len(rows)
    labels = case.partial("centrality_description")
    assert labels
    assert where.count("centrality_description.expression") == len(labels)
    assert len(labels) == len(case.analysis_levels)


@pytest.mark.parametrize("case", all_case_params())
def test_tick_labels_equal_ggstatsplot(case: OracleCase) -> None:
    x = str(case.args["x"])
    rows = centrality_rows(case)
    assert rows
    for _, row in rows:
        level = r_level_label(row[x])
        assert tx.tick_label(level, int(row["n.obs"])) == row["n.expression"]


@pytest.mark.parametrize("case", all_case_params())
def test_v01_caption_is_first_and_last_items(case: OracleCase) -> None:
    """Two-group parametric: the v0.1 caption is ggstatsplot's first and last items."""
    built = v01_caption(case)
    two_group_parametric = case.type == "parametric" and len(case.analysis_levels) == 2
    if not two_group_parametric:
        assert built is None
        return
    if not bayes_caption_drawn(case):
        assert case.caption is None
        assert built is None
        return
    assert built is not None
    assert case.caption is not None
    full = parse(case.caption)
    assert isinstance(full, tx.Call)
    assert full == rebuild_row(case, caption_rows(case)[0])
    assert built == tx.Call(full.fn, (full.args[0], full.args[-1]))


def test_every_structure_is_covered() -> None:
    """The fixtures exercise the templates this module claims to check."""
    seen: set[str] = set()
    for case in text_cases():
        for where, _, rebuilt in checked_expressions(case):
            seen.add(where.split("[")[0].split("(")[0])
            assert rebuilt is not None
    assert {"subtitle", "caption", "pairwise", "annotation", "built"} <= seen


# --------------------------------------------------------------------------- #
# hand-built results from the parametric fixtures
# --------------------------------------------------------------------------- #

_ALTERNATIVES = {"two.sided": "two-sided", "less": "less", "greater": "greater"}

# The caption's pairwise line, written out on the test side (SPEC section 7), by
# p_adjust_method and by pairwise_display.
ADJUSTED = {
    "holm": "Games-Howell, Holm adjusted",
    "hochberg": "Games-Howell, Hochberg adjusted",
    "hommel": "Games-Howell, Hommel adjusted",
    "bonferroni": "Games-Howell, Bonferroni adjusted",
    "BH": "Games-Howell, Benjamini-Hochberg adjusted",
    "fdr": "Games-Howell, Benjamini-Hochberg adjusted",
    "BY": "Games-Howell, Benjamini-Yekutieli adjusted",
    "none": "Games-Howell, unadjusted",
}
DISPLAY_NOTES = {
    "significant": "significant only",
    "non-significant": "non-significant only",
}
_DISPLAY_ALIASES = {"s": "significant", "ns": "non-significant"}


def pairwise_display(case: OracleCase) -> str:
    value = str(case.args.get("pairwise.display", "significant"))
    return _DISPLAY_ALIASES.get(value, value)


def is_displayed(display: str, significant: bool) -> bool:
    """ggstatsplot's ``pairwise.display`` rule: which pairs get a bracket."""
    rules = {
        "significant": significant,
        "non-significant": not significant,
        "all": True,
        "none": False,
    }
    return rules[display]


def fixture_settings(case: OracleCase) -> Settings:
    """The case's arguments as inferplot settings (ggstatsplot's defaults elsewhere)."""
    return Settings(
        conf_level=float(case.args.get("conf.level", 0.95)),
        digits=case_digits(case),
        alternative=_ALTERNATIVES[str(case.args.get("alternative", "two.sided"))],
        p_adjust_method=str(case.args.get("p.adjust.method", "holm")),
        pairwise_display=pairwise_display(case),
        pairwise_alpha=float(case.args.get("pairwise.alpha", 0.05)),
        bf_prior=float(case.args.get("bf.prior", 0.707)),
        bf_message=bf_message(case),
        results_subtitle=results_subtitle(case),
        seed=0,
        tr=float(case.args.get("tr", 0.2)),
    )


_WELCH_T = ("welch_t", "Welch's t-test", "t", "Welch Two Sample t-test")
_WELCH_F = (
    "welch_anova",
    "Welch's ANOVA",
    "F",
    "One-way analysis of means (not assuming equal variances)",
)


def _main_test(
    case: OracleCase, settings: Settings, data: list[Any]
) -> tuple[TestResult, EffectSize]:
    """The omnibus test and effect size: the fixture's, or inferplot.stats' when the
    fixture has no subtitle table (``results.subtitle`` FALSE)."""
    two = len(data) == 2
    name, label, statistic_name, method = _WELCH_T if two else _WELCH_F
    es_name = "hedges_g" if two else "partial_omega_squared"
    es_label = "Hedges' g" if two else "partial omega squared"
    ci_method = "noncentral t" if two else "noncentral F"
    alternative = settings.alternative
    if case.subtitle_rows:
        row = case.subtitle_data
        assert str(row["method"]) == method
        test = TestResult(
            name=name,
            label=label,
            method=method,
            statistic_name=statistic_name,
            statistic=number(row["statistic"]),
            df=None if two else number(row["df"]),
            df_error=number(row["df.error"]),
            p_value=number(row["p.value"]),
            alternative=alternative,
        )
        effect_size = EffectSize(
            name=es_name,
            label=es_label,
            estimate=number(row["estimate"]),
            conf_level=number(row["conf.level"]),
            conf_low=number(row["conf.low"]),
            conf_high=number(row["conf.high"]),
            ci_method=ci_method,
            sidedness=alternative if two else "greater",
        )
        return test, effect_size
    if two:
        a, b = data
        w = welch_t(a, b, alternative=case_alternative(case))
        g = hedges_g(
            a, b, conf_level=settings.conf_level, alternative=case_alternative(case)
        )
        statistic, df, df_error, p = w.t, None, w.df, w.p_value
        ci = g
    else:
        anova = welch_anova(data)
        ci = partial_omega_squared(
            anova.f, anova.df1, anova.df2, conf_level=settings.conf_level
        )
        statistic, df, df_error, p = anova.f, anova.df1, anova.df2, anova.p_value
    test = TestResult(
        name=name,
        label=label,
        method=method,
        statistic_name=statistic_name,
        statistic=statistic,
        df=df,
        df_error=df_error,
        p_value=p,
        alternative=alternative,
    )
    effect_size = EffectSize(
        name=es_name,
        label=es_label,
        estimate=ci.estimate,
        conf_level=ci.conf_level,
        conf_low=ci.conf_low,
        conf_high=ci.conf_high,
        ci_method=ci_method,
        sidedness=ci.sidedness,
    )
    return test, effect_size


def _pairs(
    case: OracleCase, settings: Settings, data: list[Any]
) -> tuple[PairwiseComparison, ...]:
    """Games-Howell pairs: the fixture's (raw p NaN: not recorded), or with
    ``pairwise.display`` "none", where ggstatsplot computes none, inferplot.stats'."""
    levels = case.analysis_levels
    if len(levels) == 2:
        return ()
    adjust = settings.p_adjust_method
    alpha = settings.pairwise_alpha
    display = settings.pairwise_display
    rows = case.pairwise
    found: list[tuple[str, str, float, float | None, float, float]] = []
    if rows:
        found = [
            (
                str(r["group1"]),
                str(r["group2"]),
                number(r["statistic"]),
                None,
                math.nan,
                number(r["p.value"]),
            )
            for r in rows
        ]
    else:
        assert display == "none", f"{case.name}: no pairwise table"
        found = [
            (levels[r.i], levels[r.j], r.statistic, r.df, r.p_value, r.p_adjusted)
            for r in games_howell(data, p_adjust_method=P_ADJUST[adjust])
        ]
    return tuple(
        PairwiseComparison(
            group1=group1,
            group2=group2,
            statistic_name="q",
            statistic=statistic,
            df=df,
            p_value=p_value,
            p_adjusted=p_adjusted,
            adjust_method=adjust,
            test="games_howell",
            test_label="Games-Howell",
            significant=p_adjusted < alpha,
            displayed=is_displayed(display, p_adjusted < alpha),
        )
        for group1, group2, statistic, df, p_value, p_adjusted in found
    )


def result_from_fixture(case: OracleCase) -> StatsResult:
    """A StatsResult built by hand from a completed parametric fixture, honouring the
    case's own arguments (``fixture_settings``).

    Numbers come from the fixture where it records them: the omnibus test and effect
    size from ``subtitle_data``, the Bayes factor from ``caption_data``, the pairs
    from ``pairwise_comparisons_data`` (whose p.value is the adjusted p; the raw p is
    not recorded, so ``p_value`` is NaN there) and the centrality values and n from
    the label layer. Where the fixture has no subtitle table (``results.subtitle``
    FALSE), the test and effect-size numbers are computed by inferplot.stats on the
    case's dataset (``tests.stats.fixture_data.groups``), and no Bayes factor is
    computed. With ``pairwise.display`` "none" ggstatsplot computes no pairs, so they
    come from ``inferplot.stats.games_howell`` with the case's adjustment, every one
    hidden (``displayed`` False; ``result.check()`` reports nothing for them, U11).
    ``displayed`` follows the case's ``pairwise.display`` and ``pairwise.alpha``;
    the tests compare it with the brackets ggstatsplot drew. ``n_dropped`` is the
    dataset's rows minus the analyzed ones.
    """
    assert case.type == "parametric"
    assert case.error is None, f"{case.name}: ggstatsplot failed"
    settings = fixture_settings(case)
    levels = case.analysis_levels
    data = groups(case)
    two = len(levels) == 2
    test, effect_size = _main_test(case, settings, data)
    bayes = None
    captions = caption_rows(case)
    if two and captions:
        bf10 = number(captions[0]["bf10"])
        bayes = BayesFactor(
            bf10=bf10,
            log_bf01=-math.log(bf10),
            prior="cauchy",
            prior_scale=number(captions[0]["prior.scale"]),
            test="jzs_two_sample",
        )
    elif two:
        assert not bayes_caption_drawn(case), f"{case.name}: no Bayes caption"
    x = str(case.args["x"])
    label_rows = {
        r_level_label(r[x]): r
        for geom, r in centrality_rows(case)
        if geom == "GeomLabelRepel"
    }
    descriptives = tuple(
        Descriptive(
            level=level,
            n=int(label_rows[level]["n.obs"]),
            centrality="mean",
            value=centrality_value(case, label_rows[level]),
        )
        for level in levels
    )
    pairwise = _pairs(case, settings, data)
    n_obs = sum(d.n for d in descriptives)
    texts = tx.build_texts(
        x=x,
        y=str(case.args["y"]),
        descriptives=descriptives,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        n_obs=n_obs,
        settings=settings,
    )
    return StatsResult(
        analysis="betweenstats",
        design="between",
        type="parametric",
        x=x,
        y=str(case.args["y"]),
        subject_id=None,
        levels=levels,
        # The analyzed observations, read from the fixture's dataset.
        values=tuple(tuple(g.tolist()) for g in data),
        subjects=(),
        descriptives=descriptives,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        n_obs=n_obs,
        n_dropped=int(case.raw["dataset"]["rows"]) - n_obs,
        dropped_subjects=0,
        dropped_subject_rows=0,
        n_tied=tied_count(data),
        warnings=(),
        provenance=current_provenance(seed=0, citations=("Welch 1947",)),
        settings=settings,
        text=texts,
    )


def parametric_params() -> list[Any]:
    # ggbetweenstats only: these results are built by hand with Welch's tests.
    # The ggwithinstats texts are compared on inferplot's own analyses in
    # tests/test_withinstats_texts.py.
    cases = [
        c
        for c in text_cases()
        if c.function == "ggbetweenstats" and c.type == "parametric" and c.error is None
    ]
    assert cases
    return [pytest.param(c, id=c.name) for c in cases]


def expected_pairwise_line(case: OracleCase, n_pairs: int) -> str | None:
    """The caption's pairwise line for a case, from its arguments and its brackets."""
    display = pairwise_display(case)
    if n_pairs == 0 or display == "none":
        return None
    adjusted = ADJUSTED[str(case.args.get("p.adjust.method", "holm"))]
    # Every fixture uses ggstatsplot's pairwise.alpha 0.05, where the caption has
    # no threshold note (U5; test_text covers the others).
    assert float(case.args.get("pairwise.alpha", 0.05)) == 0.05, case.name
    if display == "all":
        return f"Pairwise: {adjusted}; showing all {n_pairs} pairs."
    drawn = len(comparisons(case))
    return (
        f"Pairwise: {adjusted}; showing {drawn} of {n_pairs} pairs "
        f"({DISPLAY_NOTES[display]})."
    )


@pytest.mark.parametrize("case", parametric_params())
def test_result_texts_equal_fixture(case: OracleCase) -> None:
    result = result_from_fixture(case)
    fig = tx.figure_texts(result)
    texts = result.text

    # subtitle: present exactly when the case asks for one
    if case.subtitle is None:
        assert not results_subtitle(case)
        assert texts.subtitle is None
        assert fig.subtitle is None
    else:
        subtitle = parse(case.subtitle)
        assert fig.subtitle == tx.to_mathtext(subtitle)
        assert texts.subtitle == tx.to_plain(subtitle)
        assert result.n_obs == int(case.subtitle_data["n.obs"])

    # centrality labels and tick labels
    x = str(case.args["x"])
    labels = {
        r_level_label(r[x]): r
        for geom, r in centrality_rows(case)
        if geom == "GeomLabelRepel"
    }
    assert set(labels) == set(result.levels)
    for i, level in enumerate(result.levels):
        node = parse(str(labels[level]["expression"]))
        assert fig.centrality_labels[i] == tx.to_mathtext(node)
        assert texts.centrality_labels[i] == tx.to_plain(node)
        assert fig.tick_labels[i] == labels[level]["n.expression"]
        assert texts.tick_labels[i] == labels[level]["n.expression"]

    # pairwise labels, adjustment labels and the pairs drawn
    assert len(fig.pairwise_labels) == len(result.pairwise)
    assert len(texts.pairwise_labels) == len(result.pairwise)
    rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    if rows:
        assert set(rows) == {(p.group1, p.group2) for p in result.pairwise}
    else:
        assert all(not p.displayed for p in result.pairwise)
    for i, pair in enumerate(result.pairwise):
        row = rows.get((pair.group1, pair.group2))
        if row is None:
            continue
        node = parse(str(row["expression"]))
        assert fig.pairwise_labels[i] == tx.to_mathtext(node)
        assert texts.pairwise_labels[i] == tx.to_plain(node)
        assert tx.p_adjust_label(pair.adjust_method) == row["p.adjust.method"]
    drawn = [(p.group1, p.group2) for p in result.pairwise if p.displayed]
    assert drawn == comparisons(case)

    # caption: the v0.1 Bayes line (two groups) or the pairwise line
    if len(result.levels) == 2:
        if case.caption is None:
            assert not bayes_caption_drawn(case)
            assert result.bayes is None
            assert fig.caption_lines == ()
            assert texts.caption is None
        else:
            full = parse(case.caption)
            assert isinstance(full, tx.Call)
            v01 = tx.Call(full.fn, (full.args[0], full.args[-1]))
            assert fig.caption_lines == (tx.to_mathtext(v01),)
            assert texts.caption == tx.to_plain(v01)
    else:
        line = expected_pairwise_line(case, len(result.pairwise))
        assert fig.caption_lines == (() if line is None else (line,))
        assert texts.caption == line

    # result checks follow from the case
    codes = {issue.code for issue in result.check()}
    # An explicit pairwise_display="none" hides every pair without a report.
    hidden = any(not p.displayed for p in result.pairwise)
    asked = result.settings.pairwise_display == "none"
    assert ("pairs_hidden" in codes) == (hidden and not asked)
    assert ("small_n" in codes) == any(n < 5 for n in case.n_per_level.values())
    assert ("rows_dropped" in codes) == (result.n_dropped > 0)
