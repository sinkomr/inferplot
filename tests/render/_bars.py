"""barstats results with stand-in numbers, from any count table.

The random layout tests and the figures built to trigger a check need tables of
every shape quickly, before (and independently of) the statistics. A result is
assembled with the analysis' own ``_prepare`` (every argument checked, the table
read) and ``_assemble``, from stand-in parts: the main test and each bar's test
carry placeholder numbers (a p-value that varies with the table, so the labels
above the bars have realistic widths). The renderer draws every number from the
result, so the figures have the shape the real analysis gives.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import pandas as pd

from inferplot.result import EffectSize, StatsResult, TestResult

bars: Any = importlib.import_module("inferplot.analyses.barstats")


def _test(name: str, label: str, statistic: float, df: float, p: float) -> TestResult:
    return TestResult(
        name=name,
        label=label,
        method=label,
        statistic_name="chi2",
        statistic=statistic,
        df=None,
        df_error=df,
        p_value=p,
        alternative="two-sided",
    )


def _effect(name: str, label: str, estimate: float, level: float) -> EffectSize:
    return EffectSize(
        name=name,
        label=label,
        estimate=estimate,
        conf_level=level,
        conf_low=0.0,
        conf_high=min(1.0, estimate + 0.2),
        ci_method="placeholder",
        sidedness="two-sided",
    )


def frame_of(
    counts: Sequence[Sequence[int]],
    x_levels: Sequence[str],
    y_levels: Sequence[str] | None,
    *,
    x: str = "x",
    y: str = "y",
) -> pd.DataFrame:
    """A frame of one row per cell with its count in column "n"; the levels are
    declared, so empty ones stay in the table."""
    rows: list[tuple[str, str | None, int]] = []
    for i, level in enumerate(x_levels):
        for j, column in enumerate(y_levels or [None]):
            rows.append((level, column, int(counts[i][j])))
    frame = pd.DataFrame(
        {
            x: pd.Categorical([r[0] for r in rows], categories=list(x_levels)),
            "n": [r[2] for r in rows],
        }
    )
    if y_levels is not None:
        frame[y] = pd.Categorical([r[1] for r in rows], categories=list(y_levels))
    return frame


def table_result(
    counts: Sequence[Sequence[int]],
    x_levels: Sequence[str] | None = None,
    y_levels: Sequence[str] | None = None,
    *,
    x: str = "x",
    y: str = "y",
    one_way: bool = False,
    **kwargs: Any,
) -> StatsResult:
    """The result of a table of ``counts`` (rows: levels of x; columns: levels of
    y, one column with ``one_way``), with stand-in numbers; ``kwargs`` are
    analysis arguments (``label``, ``digits_perc``, ``proportion_test``,
    ``results_subtitle``, ...)."""
    rows = len(counts)
    columns = len(counts[0])
    xs = list(x_levels) if x_levels is not None else [f"x{i}" for i in range(rows)]
    ys: list[str] | None
    if one_way:
        ys = None
    else:
        ys = (
            list(y_levels)
            if y_levels is not None
            else [f"y{j}" for j in range(columns)]
        )
    frame = frame_of(counts, xs, ys, x=x, y=y)
    table, args, digits = bars._prepare(
        frame, x, None if one_way else y, counts="n", **kwargs
    )
    totals = table.column_totals()
    n = sum(totals)
    design = "one-way" if one_way else ("paired" if args.paired else "two-way")
    test = effect = None
    if args.results_subtitle:
        name, label = {
            "two-way": ("pearson_chi_square", "Pearson's chi-square test"),
            "paired": ("mcnemar", "McNemar's test"),
            "one-way": ("chi_square_gof", "Chi-square goodness-of-fit test"),
        }[design]
        es_name, es_label = {
            "two-way": ("cramers_v", "Cramer's V"),
            "paired": ("cohens_g", "Cohen's g"),
            "one-way": ("pearsons_c", "Pearson's C"),
        }[design]
        test = _test(name, label, 12.3456, float(max(rows - 1, 1)), 10.0 ** -(n % 9))
        effect = _effect(es_name, es_label, 0.31, args.conf_level)
    bar_tests = []
    if not one_way:
        for j, total in enumerate(totals):
            if total == 0:
                continue
            p = 10.0 ** -((total * 7 + j) % 12) * 0.537
            bar_tests.append(
                bars._BarTest(
                    test=_test(
                        "chi_square_gof",
                        "Chi-square goodness-of-fit test",
                        3.21,
                        float(rows - 1),
                        p,
                    ),
                    effect_size=_effect("pearsons_c", "Pearson's C", 0.2, 0.95),
                )
            )
    parts = bars._Parts(
        test=test,
        effect_size=effect,
        bar_tests=tuple(bar_tests),
        pairwise=(),
        warnings=(),
        citations=("placeholder",),
    )
    return bars._assemble(table, args, parts, digits=digits)
