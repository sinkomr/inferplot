"""Every fixture expression, rebuilt by the production builders, is AST-equal to R's.

For every fixture of every function and type (a fixture added later is picked up
without new code), each plotmath expression ggstatsplot built is parsed with
``tests/plotmath.py`` and compared with the expression inferplot's builders make
from the *numbers* in the same fixture: subtitles, full Bayes captions, inferplot's
v0.1 two-group caption (the first and last items of ggstatsplot's), pairwise labels,
bracket annotations, both centrality label layers and the n tick labels. A fixture
whose expression has a structure no builder makes fails; nothing is skipped.

The parametric fixtures are also turned into hand-built ``StatsResult`` objects
(the analysis layer arrives separately), and ``build_texts`` and ``figure_texts``
must give the fixture's texts as plain text and as mathtext.
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
)
from tests.oracle_fixtures import OracleCase, load_cases, number, r_level_label
from tests.plotmath import parse
from tests.test_text_format import (
    all_case_params,
    case_digits,
    centrality_rows,
    centrality_value,
    df_digits,
)

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
    """inferplot's v0.1 caption for a two-group parametric fixture, from its numbers."""
    row = case.caption_data
    if row is None or len(case.analysis_levels) != 2:
        return None
    return tx.bayes_factor_expression(
        log_bf01=-math.log(number(row["bf10"])),
        prior=prior_key(str(row["method"])),
        prior_scale=number(row["prior.scale"]),
        digits=case_digits(case),
    )


def comparisons(case: OracleCase) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for params in case.layer_params("GeomSignif"):
        out.extend((str(a), str(b)) for a, b in params["stat"]["comparisons"])
    return out


def checked_expressions(case: OracleCase) -> list[tuple[str, str, tx.Node]]:
    """(location, R's deparsed string, inferplot's rebuilt AST) for one fixture."""
    if case.error is not None:
        return []
    out: list[tuple[str, str, tx.Node]] = []
    for i, row in enumerate(case.subtitle_rows):
        out.append(
            (f"subtitle_data[{i}]", str(row["expression"]), rebuild_row(case, row))
        )
    if case.subtitle is not None:
        out.append(
            ("subtitle", case.subtitle, rebuild_row(case, case.subtitle_rows[0]))
        )
    caption_rows = case.table("caption_data")
    for i, row in enumerate(caption_rows):
        out.append(
            (f"caption_data[{i}]", str(row["expression"]), rebuild_row(case, row))
        )
    if case.caption is not None:
        out.append(("caption", case.caption, rebuild_row(case, caption_rows[0])))
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
    for geom, row in centrality_rows(case):
        out.append(
            (
                f"{geom}.expression",
                str(row["expression"]),
                rebuild_centrality(case, row),
            )
        )
    return out


@pytest.mark.parametrize("case", all_case_params())
def test_expressions_equal_ggstatsplot(case: OracleCase) -> None:
    checked = checked_expressions(case)
    if case.error is None:
        assert checked
        assert case.subtitle is not None
    for where, source, rebuilt in checked:
        assert rebuilt == parse(source), f"{where}: {source}"


@pytest.mark.parametrize("case", all_case_params())
def test_tick_labels_equal_ggstatsplot(case: OracleCase) -> None:
    x = str(case.args["x"])
    rows = centrality_rows(case)
    if case.error is None:
        assert rows
    for _, row in rows:
        level = r_level_label(row[x])
        assert tx.tick_label(level, int(row["n.obs"])) == row["n.expression"]


@pytest.mark.parametrize("case", all_case_params())
def test_v01_caption_is_first_and_last_items(case: OracleCase) -> None:
    """Two-group parametric: the v0.1 caption is ggstatsplot's first and last items."""
    built = v01_caption(case)
    two_group_parametric = case.type == "parametric" and len(case.analysis_levels) == 2
    if not two_group_parametric or case.error is not None:
        assert built is None or case.type != "parametric"
        return
    assert built is not None
    assert case.caption is not None
    full = parse(case.caption)
    assert isinstance(full, tx.Call)
    assert full == rebuild_row(case, case.table("caption_data")[0])
    assert built == tx.Call(full.fn, (full.args[0], full.args[-1]))


def test_every_structure_is_covered() -> None:
    """The fixtures exercise the templates this module claims to check."""
    seen: set[str] = set()
    for case in load_cases():
        for where, _, rebuilt in checked_expressions(case):
            seen.add(where.split("[")[0].split("(")[0])
            assert rebuilt is not None
    assert {"subtitle", "caption", "pairwise", "annotation", "built"} <= seen


# --------------------------------------------------------------------------- #
# hand-built results from the parametric fixtures
# --------------------------------------------------------------------------- #

_ALTERNATIVES = {"two.sided": "two-sided", "less": "less", "greater": "greater"}


def result_from_fixture(case: OracleCase) -> StatsResult:
    """A StatsResult built by hand from a parametric fixture's own numbers.

    The fixture's pairwise p.value is the adjusted p; the raw p is not recorded, so
    ``p_value`` is NaN here.
    """
    assert case.type == "parametric"
    row = case.subtitle_data
    levels = case.analysis_levels
    two = len(levels) == 2
    alternative = _ALTERNATIVES[str(case.args.get("alternative", "two.sided"))]
    test = TestResult(
        name="welch_t" if two else "welch_anova",
        label="Welch's t-test" if two else "Welch's ANOVA",
        method=str(row["method"]),
        statistic_name="t" if two else "F",
        statistic=number(row["statistic"]),
        df=None if two else number(row["df"]),
        df_error=number(row["df.error"]),
        p_value=number(row["p.value"]),
        alternative=alternative,
    )
    effect_size = EffectSize(
        name="hedges_g" if two else "partial_omega_squared",
        label="Hedges' g" if two else "partial omega squared",
        estimate=number(row["estimate"]),
        conf_level=number(row["conf.level"]),
        conf_low=number(row["conf.low"]),
        conf_high=number(row["conf.high"]),
        ci_method="noncentral t" if two else "noncentral F",
        sidedness=alternative if two else "greater",
    )
    caption_row = case.caption_data
    bayes = None
    if two and caption_row is not None:
        bf10 = number(caption_row["bf10"])
        bayes = BayesFactor(
            bf10=bf10,
            log_bf01=-math.log(bf10),
            prior="cauchy",
            prior_scale=number(caption_row["prior.scale"]),
            test="jzs_two_sample",
        )
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
    drawn = set(comparisons(case))
    alpha = float(case.args.get("pairwise.alpha", 0.05))
    adjust = str(case.args.get("p.adjust.method", "holm"))
    pairwise = tuple(
        PairwiseComparison(
            group1=str(p["group1"]),
            group2=str(p["group2"]),
            statistic_name="q",
            statistic=number(p["statistic"]),
            df=None,
            p_value=math.nan,
            p_adjusted=number(p["p.value"]),
            adjust_method=adjust,
            test="games_howell",
            test_label="Games-Howell",
            significant=number(p["p.value"]) < alpha,
            displayed=(str(p["group1"]), str(p["group2"])) in drawn,
        )
        for p in case.pairwise
    )
    settings = Settings(
        conf_level=float(case.args.get("conf.level", 0.95)),
        digits=case_digits(case),
        alternative=alternative,
        p_adjust_method=adjust,
        pairwise_display=str(case.args.get("pairwise.display", "significant")),
        pairwise_alpha=alpha,
        bf_prior=float(case.args.get("bf.prior", 0.707)),
        bf_message=None,
        results_subtitle=bool(case.args.get("results.subtitle", True)),
        seed=0,
        tr=float(case.args.get("tr", 0.2)),
    )
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
        descriptives=descriptives,
        test=test,
        effect_size=effect_size,
        bayes=bayes,
        pairwise=pairwise,
        n_obs=n_obs,
        n_dropped=0,
        dropped_subjects=0,
        warnings=(),
        provenance=current_provenance(seed=0, citations=("Welch 1947",)),
        settings=settings,
        text=texts,
    )


def parametric_params() -> list[Any]:
    cases = [c for c in load_cases() if c.type == "parametric" and c.error is None]
    assert cases
    return [pytest.param(c, id=c.name) for c in cases]


@pytest.mark.parametrize("case", parametric_params())
def test_result_texts_equal_fixture(case: OracleCase) -> None:
    result = result_from_fixture(case)
    fig = tx.figure_texts(result)
    texts = result.text
    assert case.subtitle is not None
    subtitle = parse(case.subtitle)
    assert fig.subtitle == tx.to_mathtext(subtitle)
    assert texts.subtitle == tx.to_plain(subtitle)

    x = str(case.args["x"])
    labels = {
        r_level_label(r[x]): r
        for geom, r in centrality_rows(case)
        if geom == "GeomLabelRepel"
    }
    for i, level in enumerate(result.levels):
        node = parse(str(labels[level]["expression"]))
        assert fig.centrality_labels[i] == tx.to_mathtext(node)
        assert texts.centrality_labels[i] == tx.to_plain(node)
        assert fig.tick_labels[i] == labels[level]["n.expression"]
        assert texts.tick_labels[i] == labels[level]["n.expression"]

    rows = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    for i, pair in enumerate(result.pairwise):
        node = parse(str(rows[pair.group1, pair.group2]["expression"]))
        assert fig.pairwise_labels[i] == tx.to_mathtext(node)
        assert texts.pairwise_labels[i] == tx.to_plain(node)
    assert [p.displayed for p in result.pairwise] == [
        p.significant for p in result.pairwise
    ]

    if len(result.levels) == 2:
        assert case.caption is not None
        full = parse(case.caption)
        assert isinstance(full, tx.Call)
        v01 = tx.Call(full.fn, (full.args[0], full.args[-1]))
        assert fig.caption_lines == (tx.to_mathtext(v01),)
        assert texts.caption == tx.to_plain(v01)
    else:
        drawn = sum(p.displayed for p in result.pairwise)
        line = (
            f"Pairwise: Games-Howell, Holm adjusted; showing {drawn} of "
            f"{len(result.pairwise)} pairs (significant only)."
        )
        assert fig.caption_lines == (line,)
        assert texts.caption == line
    assert result.check() == tuple(result.check())
