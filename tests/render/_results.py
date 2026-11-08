"""StatsResult objects for the rendering tests.

The analysis layer (``ip.analyze.betweenstats``) is written in parallel, so these
tests build their results here, directly from ``inferplot.stats`` and
``inferplot.text.build_texts``: one result per completed parametric ggbetweenstats
fixture, from the fixture's own dataset and arguments, plus synthetic results with
many levels or many points. Once the analysis layer is merged the tests switch to it.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from inferplot import stats
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
from inferplot.stats import Alternative, PAdjustMethod
from inferplot.text import build_texts
from tests.oracle_fixtures import OracleCase, cases_for, read_dataset
from tests.stats import fixture_data

FloatArray = NDArray[np.float64]

THREE_GROUP_BF_WARNING = (
    "The Bayes factor caption for three or more groups arrives in inferplot 0.2."
)
ONE_SIDED_BF_WARNING = (
    "The Bayes factor in the caption is two-sided; it does not use alternative="
)


def parametric_cases() -> tuple[OracleCase, ...]:
    """Every parametric ggbetweenstats fixture ggstatsplot completed."""
    return tuple(c for c in cases_for("ggbetweenstats") if c.type == "parametric")


def _displayed(display: str, significant: bool) -> bool:
    return {
        "all": True,
        "none": False,
        "significant": significant,
        "non-significant": not significant,
    }[display]


def build_result(
    levels: Sequence[str],
    groups: Sequence[FloatArray],
    *,
    x: str = "group",
    y: str = "value",
    conf_level: float = 0.95,
    digits: int = 2,
    alternative: Alternative = "two-sided",
    p_adjust_method: PAdjustMethod = "holm",
    pairwise_display: str = "significant",
    pairwise_alpha: float = 0.05,
    bf_message: bool | None = None,
    bf_prior: float = 0.707,
    results_subtitle: bool = True,
    seed: int = 0,
    n_dropped: int = 0,
    pair_order: Sequence[tuple[str, str]] | None = None,
) -> StatsResult:
    """A parametric betweenstats result for ``groups`` (one array per level).

    ``pair_order`` lists the pairs (group1, group2) in the order to report them;
    by default every pair i < j in level order. Pairs are skipped entirely when
    ``pairwise_display`` is "none", as ggstatsplot does.
    """
    k = len(levels)
    descriptives = tuple(
        Descriptive(level=level, n=int(g.size), centrality="mean", value=stats.mean(g))
        for level, g in zip(levels, groups, strict=True)
    )
    warnings: list[str] = []
    bayes = None
    if k == 2:
        welch = stats.welch_t(groups[0], groups[1], alternative=alternative)
        test = TestResult(
            name="welch_t",
            label="Welch's t-test",
            method="Welch Two Sample t-test",
            statistic_name="t",
            statistic=welch.t,
            df=None,
            df_error=welch.df,
            p_value=welch.p_value,
            alternative=alternative,
        )
        g = stats.hedges_g(
            groups[0], groups[1], conf_level=conf_level, alternative=alternative
        )
        effect = EffectSize(
            name="hedges_g",
            label="Hedges' g",
            estimate=g.estimate,
            conf_level=g.conf_level,
            conf_low=g.conf_low,
            conf_high=g.conf_high,
            ci_method="noncentral t",
            sidedness=g.sidedness,
        )
        if results_subtitle and bf_message is not False:
            bf10, log_bf01 = stats.jzs_bf10_two_sample(
                groups[0], groups[1], prior_scale=bf_prior
            )
            bayes = BayesFactor(
                bf10=bf10,
                log_bf01=log_bf01,
                prior="cauchy",
                prior_scale=bf_prior,
                test="jzs_two_sample",
            )
            if alternative != "two-sided":
                warnings.append(f"{ONE_SIDED_BF_WARNING}{alternative!r}.")
        pairs: tuple[PairwiseComparison, ...] = ()
    else:
        anova = stats.welch_anova(groups)
        test = TestResult(
            name="welch_anova",
            label="Welch's ANOVA",
            method="One-way analysis of means (not assuming equal variances)",
            statistic_name="F",
            statistic=anova.f,
            df=anova.df1,
            df_error=anova.df2,
            p_value=anova.p_value,
            alternative="two-sided",
        )
        omega = stats.partial_omega_squared(
            anova.f, anova.df1, anova.df2, conf_level=conf_level
        )
        effect = EffectSize(
            name="partial_omega_squared",
            label="partial omega squared",
            estimate=omega.estimate,
            conf_level=omega.conf_level,
            conf_low=omega.conf_low,
            conf_high=omega.conf_high,
            ci_method="noncentral F",
            sidedness=omega.sidedness,
        )
        if results_subtitle and bf_message is None:
            warnings.append(THREE_GROUP_BF_WARNING)
        pairs = ()
        if pairwise_display != "none":
            found = {
                (levels[p.i], levels[p.j]): p
                for p in stats.games_howell(groups, p_adjust_method=p_adjust_method)
            }
            order = list(found) if pair_order is None else list(pair_order)
            assert sorted(order) == sorted(found), (order, list(found))
            pairs = tuple(
                PairwiseComparison(
                    group1=a,
                    group2=b,
                    statistic_name="q",
                    statistic=found[a, b].statistic,
                    df=found[a, b].df,
                    p_value=found[a, b].p_value,
                    p_adjusted=found[a, b].p_adjusted,
                    adjust_method=p_adjust_method,
                    test="games_howell",
                    test_label="Games-Howell",
                    significant=found[a, b].p_adjusted < pairwise_alpha,
                    displayed=_displayed(
                        pairwise_display, found[a, b].p_adjusted < pairwise_alpha
                    ),
                )
                for a, b in order
            )
    settings = Settings(
        conf_level=conf_level,
        digits=digits,
        alternative=alternative,
        p_adjust_method=p_adjust_method,
        pairwise_display=pairwise_display,
        pairwise_alpha=pairwise_alpha,
        bf_prior=bf_prior,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        seed=seed,
        tr=0.2,
    )
    n_obs = sum(d.n for d in descriptives)
    return StatsResult(
        analysis="betweenstats",
        design="between",
        type="parametric",
        x=x,
        y=y,
        subject_id=None,
        levels=tuple(levels),
        values=tuple(tuple(float(v) for v in g) for g in groups),
        descriptives=descriptives,
        test=test,
        effect_size=effect,
        bayes=bayes,
        pairwise=pairs,
        n_obs=n_obs,
        n_dropped=n_dropped,
        dropped_subjects=0,
        warnings=tuple(warnings),
        provenance=current_provenance(seed=seed, citations=("Welch 1947",)),
        settings=settings,
        text=build_texts(
            x=x,
            y=y,
            descriptives=descriptives,
            test=test,
            effect_size=effect,
            bayes=bayes,
            pairwise=pairs,
            n_obs=n_obs,
            settings=settings,
        ),
    )


def fixture_result(case: OracleCase) -> StatsResult:
    """The result inferplot computes for ``case``, honoring its arguments."""
    return _fixture_result(case.name)


@functools.cache
def _fixture_result(name: str) -> StatsResult:
    (case,) = [c for c in parametric_cases() if c.name == name]
    args = case.args
    frame = read_dataset(case)
    x, y = str(args["x"]), str(args["y"])
    n_dropped = len(frame) - len(frame[[x, y]].dropna())
    order = [(str(r["group1"]), str(r["group2"])) for r in case.pairwise]
    display = str(args.get("pairwise.display", "significant"))
    return build_result(
        case.analysis_levels,
        fixture_data.groups(case),
        x=x,
        y=y,
        conf_level=fixture_data.conf_level(case),
        digits=int(args.get("digits", 2)),
        alternative=fixture_data.alternative(case),
        p_adjust_method=fixture_data.p_adjust_method(case),
        pairwise_display={"s": "significant", "ns": "non-significant"}.get(
            display, display
        ),
        bf_message=args.get("bf.message"),
        bf_prior=fixture_data.bf_prior(case),
        results_subtitle=bool(args.get("results.subtitle", True)),
        n_dropped=n_dropped,
        pair_order=order or None,
    )


def many_levels_result(k: int = 12, *, n: int = 15, seed: int = 1) -> StatsResult:
    """``k`` groups drawn from the same normal distribution, with long-ish names."""
    rng = np.random.default_rng(seed)
    levels = [f"site {i + 1:02d}" for i in range(k)]
    groups = [rng.normal(50.0, 10.0, size=n) for _ in range(k)]
    return build_result(levels, groups, x="site", y="yield")


def many_points_result(n_per_group: int = 2_200, *, seed: int = 2) -> StatsResult:
    """Three groups with ``3 * n_per_group`` values in all (6,600 by default)."""
    rng = np.random.default_rng(seed)
    groups = [rng.normal(mu, 1.0, size=n_per_group) for mu in (0.0, 0.3, 0.5)]
    return build_result(["a", "b", "c"], groups, x="arm", y="score")
