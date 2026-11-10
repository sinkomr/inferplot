"""Results of histostats and scatterstats built before their statistics exist.

``prepared(case)`` runs the analysis' own argument checks and data reading
(``_prepare``) on a fixture's dataset with the fixture's arguments; ``parts(case,
args)`` makes the parts a type's analysis would return, from the numbers the fixture
records (its subtitle and caption rows, its centrality line) or, where ggstatsplot
printed none (``results.subtitle = FALSE``), from stand-in numbers; ``assembled``
builds the result with the analysis' own ``_assemble``. The names of the tests and
effect sizes here are placeholders: the statistics package chooses the real ones.
"""

from __future__ import annotations

import importlib
import math
import statistics
from typing import Any

from inferplot.result import BayesFactor, EffectSize, StatsResult, TestResult
from tests.oracle_args import apply_label_filter, split
from tests.oracle_fixtures import Backend, OracleCase, number, read_dataset

histo: Any = importlib.import_module("inferplot.analyses.histostats")
scatter: Any = importlib.import_module("inferplot.analyses.scatterstats")


def module(case: OracleCase) -> Any:
    return histo if case.function == "gghistostats" else scatter


def prepared(case: OracleCase, backend: Backend = "pandas") -> tuple[Any, Any, int]:
    """``_prepare`` on the case's dataset with its analysis arguments: the data
    read, the validated arguments and digits. A ``label.expression`` case gets the
    labels of the other points set to missing (pandas only)."""
    call = split(case.function, case.args)
    frame = read_dataset(case, backend)
    if call.label_filter is not None:
        assert backend == "pandas", "label filters are applied with pandas"
        frame = apply_label_filter(
            frame, str(call.analysis["label_var"]), call.label_filter
        )
    return module(case)._prepare(frame, *call.args, **call.analysis)


def _row(case: OracleCase) -> dict[str, Any] | None:
    rows = case.subtitle_rows
    return dict(rows[0]) if len(rows) == 1 else None


def _sidedness(alternative: str) -> str:
    return alternative if alternative in ("less", "greater") else "two-sided"


def _bayes(row: dict[str, Any] | None, prior: str, scale: float, test: str) -> Any:
    bf10 = number(row["bf10"]) if row is not None else 2.5
    log_bf01 = -math.log(bf10) if bf10 > 0 and math.isfinite(bf10) else math.nan
    return BayesFactor(
        bf10=bf10, log_bf01=log_bf01, prior=prior, prior_scale=scale, test=test
    )


_CENTRALITY = {
    "parametric": "mean",
    "nonparametric": "median",
    "robust": "trimmed_mean",
    "bayes": "map",
}


def histostats_parts(case: OracleCase, data: Any, args: Any) -> Any:
    """The parts of a histostats type, from the case's numbers."""
    row = _row(case)
    values = data.values.tolist()
    lines = case.centrality
    center = number(lines[0]["var"]) if lines else statistics.median(values)
    get = (lambda key, default: number(row[key])) if row else (lambda _k, d: d)
    side = args.alternative
    test = effect = bayes = None
    if args.type != "bayes":
        name, statistic, label = {
            "parametric": ("one_sample_t", "t", "One-sample t-test"),
            "nonparametric": ("wilcoxon_one_sample", "V", "Wilcoxon signed-rank test"),
            "robust": ("trimmed_mean_bootstrap_t", "t", "Bootstrap-t test"),
        }[args.type]
        test = TestResult(
            name=name,
            label=label,
            method=str(row["method"]) if row else label,
            statistic_name=statistic,
            statistic=get("statistic", 1.25),
            df=None,
            df_error=get("df.error", len(values) - 1.0)
            if args.type == "parametric"
            else None,
            p_value=get("p.value", 0.25),
            alternative=side,
        )
        boot = args.type == "robust"
        effect = EffectSize(
            name={"parametric": "hedges_g", "nonparametric": "rank_biserial"}.get(
                args.type, "trimmed_mean"
            ),
            label={"parametric": "Hedges' g", "nonparametric": "rank-biserial r"}.get(
                args.type, "trimmed mean"
            ),
            estimate=get("estimate", 0.4),
            conf_level=args.conf_level,
            conf_low=get("conf.low", -math.inf if side == "less" else 0.1),
            conf_high=get("conf.high", math.inf if side == "greater" else 0.7),
            ci_method="bootstrap-t" if boot else "noncentral t",
            sidedness="two-sided" if boot else _sidedness(side),
            bootstrap_resamples=2000 if boot else None,
            bootstrap_seed=args.seed if boot else None,
            undefined_resamples=0 if boot else None,
            infinite_resamples=0 if boot else None,
        )
    caption = case.caption_data
    if args.type == "bayes" and args.results_subtitle:
        bayes = _bayes(row, "cauchy", args.prior_scale, "jzs_one_sample")
    elif args.type == "parametric" and caption is not None:
        bayes = _bayes(dict(caption), "cauchy", args.prior_scale, "jzs_one_sample")
    return histo._Parts(
        test=test,
        effect_size=effect,
        bayes=bayes,
        centrality=_CENTRALITY[args.type],
        center=center,
        warnings=(),
        citations=("placeholder",),
    )


def scatterstats_parts(case: OracleCase, data: Any, args: Any) -> Any:
    """The parts of a scatterstats type, from the case's numbers."""
    row = _row(case)
    get = (lambda key, default: number(row[key])) if row else (lambda _k, d: d)
    test = effect = bayes = None
    if args.type != "bayes":
        statistic = "S" if args.type == "nonparametric" else "t"
        name, label = {
            "parametric": ("pearson_r", "Pearson's r"),
            "nonparametric": ("spearman_rho", "Spearman's rho"),
            "robust": ("winsorized_r", "winsorized r"),
        }[args.type]
        test = TestResult(
            name=name.replace("_r", "_t").replace("_rho", "_s"),
            label=f"{label} correlation test",
            method=str(row["method"]) if row else label,
            statistic_name=statistic,
            statistic=get("statistic", -2.5),
            df=None,
            df_error=None if statistic == "S" else get("df.error", data.n_obs - 2.0),
            p_value=get("p.value", 0.02),
            alternative="two-sided",
        )
        effect = EffectSize(
            name=name,
            label=label,
            estimate=get("estimate", -0.5),
            conf_level=args.conf_level,
            conf_low=get("conf.low", -0.8),
            conf_high=get("conf.high", -0.1),
            ci_method="normal approximation (Fisher z)",
            sidedness="two-sided",
        )
    caption = case.caption_data
    if args.type == "bayes" and args.results_subtitle:
        bayes = _bayes(row, "beta", args.prior_scale, "correlation")
    elif args.type == "parametric" and caption is not None:
        bayes = _bayes(dict(caption), "beta", args.prior_scale, "correlation")
    return scatter._Parts(
        test=test,
        effect_size=effect,
        bayes=bayes,
        warnings=(),
        citations=("placeholder",),
    )


def assembled(case: OracleCase, backend: Backend = "pandas") -> StatsResult:
    """The result ``_assemble`` builds on the case's data with parts made from its
    numbers."""
    data, args, digits = prepared(case, backend)
    make = histostats_parts if case.function == "gghistostats" else scatterstats_parts
    return module(case)._assemble(data, args, make(case, data, args), digits=digits)
