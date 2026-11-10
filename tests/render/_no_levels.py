"""histostats and scatterstats results with stand-in numbers, from any frame.

For data the analyses refuse (a constant column, values too large to draw) and
for the random layout tests, which need many shapes of data quickly, the results
are assembled with the analysis' own ``_prepare`` (every argument checked, the
data read) and ``_assemble``, from stand-in parts: the test, effect size and Bayes
factor carry placeholder numbers, and the centrality value is the type's
centrality of the values (mean, median, 20% trimmed mean; the median standing in
for the MAP). The renderer draws every number from the result, so the figures
have the shape the real analyses give.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from scipy import stats

from inferplot.result import BayesFactor, EffectSize, StatsResult, TestResult

histo: Any = importlib.import_module("inferplot.analyses.histostats")
scatter: Any = importlib.import_module("inferplot.analyses.scatterstats")

_CENTRALITY = {
    "parametric": "mean",
    "nonparametric": "median",
    "robust": "trimmed_mean",
    "bayes": "map",
}


def _center(values: Any, type_name: str, tr: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    if type_name == "parametric":
        return float(np.mean(x))
    if type_name == "robust":
        return float(stats.trim_mean(x, tr))
    return float(np.median(x))


def _test(type_name: str, statistic: str, label: str, side: str) -> TestResult:
    return TestResult(
        name=f"stand_in_{type_name}",
        label=label,
        method=label,
        statistic_name=statistic,
        statistic=1.25,
        df=None,
        df_error=30.0 if statistic == "t" else None,
        p_value=0.22,
        alternative=side,
    )


def _effect(name: str, label: str, level: float, side: str) -> EffectSize:
    return EffectSize(
        name=name,
        label=label,
        estimate=0.3,
        conf_level=level,
        conf_low=-float("inf") if side == "less" else -0.1,
        conf_high=float("inf") if side == "greater" else 0.6,
        ci_method="stand-in",
        sidedness=side,
    )


def hist_result(frame: Any, x: str, **kwargs: Any) -> StatsResult:
    """A histostats result of ``frame[x]`` with stand-in numbers (see above)."""
    data, args, digits = histo._prepare(frame, x, **kwargs)
    side = args.alternative
    test = effect = bayes = None
    if args.type != "bayes":
        test = _test(args.type, "t", "One-sample t-test", side)
        effect = _effect("hedges_g", "Hedges' g", args.conf_level, side)
    if args.results_subtitle and args.type in ("parametric", "bayes"):
        bayes = BayesFactor(
            bf10=2.5,
            log_bf01=-float(np.log(2.5)),
            prior="cauchy",
            prior_scale=args.prior_scale,
            test="jzs_one_sample",
        )
    parts = histo._Parts(
        test=test,
        effect_size=effect,
        bayes=bayes,
        centrality=_CENTRALITY[args.type],
        center=_center(data.values, args.type, args.tr),
        warnings=(),
        citations=("stand-in",),
    )
    return histo._assemble(data, args, parts, digits=digits)


def scatter_result(frame: Any, x: str, y: str, **kwargs: Any) -> StatsResult:
    """A scatterstats result of ``frame[x]`` and ``frame[y]`` with stand-in
    numbers (see above)."""
    data, args, digits = scatter._prepare(frame, x, y, **kwargs)
    test = effect = bayes = None
    if args.type != "bayes":
        test = _test(args.type, "t", "Pearson correlation", "two-sided")
        effect = _effect("pearson_r", "Pearson's r", args.conf_level, "two-sided")
    if args.results_subtitle and args.type in ("parametric", "bayes"):
        bayes = BayesFactor(
            bf10=2.5,
            log_bf01=-float(np.log(2.5)),
            prior="beta",
            prior_scale=args.prior_scale,
            test="correlation",
        )
    parts = scatter._Parts(
        test=test,
        effect_size=effect,
        bayes=bayes,
        warnings=(),
        citations=("stand-in",),
    )
    return scatter._assemble(data, args, parts, digits=digits)
