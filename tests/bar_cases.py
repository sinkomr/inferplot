"""The ggbarstats and ggpiestats fixtures, and how inferplot replays each.

``prepared(case, backend)`` runs barstats' own argument checks and table reading
(``_prepare``) on a fixture's dataset with the fixture's analysis arguments;
``parts(case, table, args)`` makes the parts a design's analysis would return
from the numbers the fixture records (its subtitle row, its per-bar rows
``one_sample_data``, its pairwise rows) or, where ggstatsplot printed none
(``results.subtitle = FALSE``; a table it leaves without a subtitle), from
stand-in numbers; ``assembled`` builds the result with the analysis' own
``_assemble``. The names of the tests and effect sizes here are placeholders:
the statistics package chooses the real ones.

Where inferplot refuses a call ggstatsplot answers, by design (SPEC.md section 4,
DECISIONS.md D101), ``REFUSALS`` says how it refuses and which changed call, if
any, gives ggstatsplot's output:

* type robust raises ``ValueError`` (no robust method for counts; ggstatsplot
  runs the parametric test), so the replay with type parametric must give
  ggstatsplot's output;
* type bayes raises ``NotImplementedError`` (the Bayes factor of counts arrives
  in inferplot 0.2), with no replay;
* a table with one level of x or of y with observations, or one category
  without y (edge_bar_degenerate), raises ``ValueError`` (ggstatsplot leaves the
  subtitle out silently), with no replay;
* paired variables with different levels (edge_bar_paired_unequal) raise
  ``ValueError``; declaring both on the same levels (the -declared case) is the
  replay, a fixture of its own;
* a ``ratio`` of the wrong length or one that does not sum to 1 raises
  ``ValueError`` (``chisq.test`` stops and ggstatsplot leaves the subtitle out);
  the rescaled ratio of mtcars-cyl-ratio-sum4 is mtcars-cyl-ratio's.
"""

from __future__ import annotations

import dataclasses
import importlib
import math
from collections.abc import Mapping
from typing import Any

from inferplot.result import (
    EffectSize,
    PairwiseComparison,
    StatsResult,
    TestResult,
)
from tests.oracle_args import split
from tests.oracle_fixtures import Backend, OracleCase, cases_for, number, read_dataset

bars: Any = importlib.import_module("inferplot.analyses.barstats")

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
CASES: tuple[OracleCase, ...] = (*cases_for("ggbarstats"), *cases_for("ggpiestats"))
FAILED: tuple[OracleCase, ...] = tuple(cases_for("ggbarstats", failed=True))
BY_NAME: dict[str, OracleCase] = {c.name: c for c in (*CASES, *FAILED)}


@dataclasses.dataclass(frozen=True)
class Refusal:
    """inferplot refuses the fixture's call with ``error`` whose message contains
    ``match``; ``replay`` (keyword changes, None drops one) reproduces
    ggstatsplot's output, or is None where no call does."""

    error: type[Exception]
    match: str
    replay: Mapping[str, Any] | None


_ROBUST = Refusal(ValueError, "there is no robust method for counts", {"type": None})
_BAYES = Refusal(NotImplementedError, "arrives in inferplot 0.2", None)
_ONE_LEVEL = Refusal(
    ValueError, "needs observations in two or more levels of x='", None
)


def _refusal(case: OracleCase) -> Refusal | None:
    name, kind = case.name, case.type
    if kind == "robust":
        return _ROBUST
    if kind == "bayes":
        return _BAYES
    if name.startswith("ggbarstats-edge_bar_degenerate-oneway-one"):
        return Refusal(ValueError, "has a single category ('a')", None)
    if name.startswith(
        (
            "ggbarstats-edge_bar_degenerate-one-cell",
            "ggbarstats-edge_bar_degenerate-one-column",
            "ggbarstats-edge_bar_degenerate-one-row",
        )
    ):
        return _ONE_LEVEL
    if name == "ggbarstats-edge_bar_paired_unequal-parametric":
        return Refusal(ValueError, "('maybe' only in 'after')", None)
    if name == "ggbarstats-mtcars-cyl-ratio-short-parametric":
        return Refusal(ValueError, "ratio has 2 shares but x='cyl' has 3 levels", None)
    if name == "ggbarstats-mtcars-cyl-ratio-sum4-parametric":
        return Refusal(
            ValueError,
            "If these are relative sizes, write ratio=[0.5, 0.25, 0.25].",
            {"ratio": [0.5, 0.25, 0.25]},
        )
    return None


REFUSALS: dict[str, Refusal] = {
    case.name: refusal for case in CASES if (refusal := _refusal(case)) is not None
}
ANSWERED: tuple[OracleCase, ...] = tuple(c for c in CASES if c.name not in REFUSALS)


def call_arguments(
    case: OracleCase, changes: Mapping[str, Any] | None = None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The fixture's analysis call as positional and keyword arguments, with
    ``changes`` applied (None drops a keyword)."""
    call = split(case.function, case.args)
    keywords = dict(call.analysis)
    for key, value in (changes or {}).items():
        if value is None:
            keywords.pop(key, None)
        else:
            keywords[key] = value
    return call.args, keywords


def prepared(
    case: OracleCase,
    backend: Backend = "pandas",
    changes: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, int]:
    """``_prepare`` on the case's dataset with its analysis arguments: the table
    read, the validated arguments and digits."""
    args, keywords = call_arguments(case, changes)
    return bars._prepare(read_dataset(case, backend), *args, **keywords)


def _sidedness(alternative: str) -> str:
    return alternative if alternative in ("less", "greater") else "two-sided"


_TESTS = {
    "two-way": ("pearson_chi_square", "Pearson's chi-square test"),
    "paired": ("mcnemar", "McNemar's test"),
    "one-way": ("chi_square_gof", "Chi-square goodness-of-fit test"),
}
_EFFECTS = {
    "two-way": ("cramers_v", "Cramer's V"),
    "paired": ("cohens_g", "Cohen's g"),
    "one-way": ("pearsons_c", "Pearson's C"),
}


def _row_getter(row: Mapping[str, Any] | None) -> Any:
    if row is None:
        return lambda _key, default: default
    return lambda key, default: number(row[key])


def _main(case: OracleCase, args: Any) -> tuple[Any, Any]:
    """The main test and effect size from the subtitle row, or stand-ins."""
    if not args.results_subtitle:
        return None, None
    rows = case.subtitle_rows
    row = dict(rows[0]) if len(rows) == 1 else None
    get = _row_getter(row)
    design = args.design
    name, label = _TESTS[design]
    test = TestResult(
        name=name,
        label=label,
        method=str(row["method"]) if row else label,
        statistic_name="chi2",
        statistic=get("statistic", 4.0),
        df=None,
        df_error=get("df", 1.0),
        p_value=get("p.value", 0.05),
        alternative="two-sided",
    )
    es_name, es_label = _EFFECTS[design]
    side = args.alternative
    effect = EffectSize(
        name=es_name,
        label=es_label,
        estimate=get("estimate", 0.25),
        conf_level=args.conf_level,
        conf_low=get("conf.low", -0.5 if side == "less" else 0.0),
        conf_high=get("conf.high", 1.0 if side == "greater" else 0.5),
        ci_method="placeholder",
        sidedness=_sidedness(side),
    )
    return test, effect


def _bar_tests(case: OracleCase, table: Any) -> tuple[Any, ...]:
    """One placeholder per bar from the case's per-bar rows (keyed by the level
    of y), in the order of y's levels with observations."""
    if table.y_levels is None:
        return ()
    by_level = {str(row[table.y]): row for row in case.one_sample if table.y in row}
    out: list[Any] = []
    for level in table.y_levels.present:
        row = by_level.get(level)
        get = _row_getter(row)
        test = TestResult(
            name="chi_square_gof",
            label="Chi-square goodness-of-fit test",
            method="Chi-squared test for given probabilities",
            statistic_name="chi2",
            statistic=get("statistic", 1.0),
            df=None,
            df_error=get("df", 1.0),
            p_value=get("p.value", 0.3),
            alternative="two-sided",
        )
        effect = EffectSize(
            name="pearsons_c",
            label="Pearson's C",
            estimate=get("estimate", 0.2),
            conf_level=0.95,
            conf_low=get("conf.low", 0.0),
            conf_high=get("conf.high", 0.6),
            ci_method="placeholder",
            sidedness="two-sided",
        )
        out.append(bars._BarTest(test=test, effect_size=effect))
    return tuple(out)


def _pairs(case: OracleCase, table: Any, args: Any) -> tuple[Any, ...]:
    if not bars._pairs_wanted(table, args):
        return ()
    out: list[PairwiseComparison] = []
    for row in case.pairwise:
        adjusted = number(row["p.value.adj"])
        out.append(
            PairwiseComparison(
                group1=str(row["group1"]),
                group2=str(row["group2"]),
                statistic_name="",
                statistic=None,
                df=None,
                p_value=number(row["p.value"]),
                p_adjusted=adjusted,
                adjust_method=args.p_adjust_method,
                test="fisher_exact",
                test_label="Fisher's exact test",
                significant=adjusted < 0.05,
                displayed=False,
                estimate=number(row["estimate"]),
                conf_level=number(row["conf.level"]),
                conf_low=number(row["conf.low"]),
                conf_high=number(row["conf.high"]),
            )
        )
    return tuple(out)


def parts(case: OracleCase, table: Any, args: Any) -> Any:
    """The parts of the case's design, from its numbers."""
    test, effect = _main(case, args)
    return bars._Parts(
        test=test,
        effect_size=effect,
        bar_tests=_bar_tests(case, table),
        pairwise=_pairs(case, table, args),
        warnings=(),
        citations=("placeholder",),
    )


def assembled(
    case: OracleCase,
    backend: Backend = "pandas",
    changes: Mapping[str, Any] | None = None,
) -> StatsResult:
    """The result ``_assemble`` builds on the case's table with parts made from
    its numbers (the replay of a refused case where ``changes`` give one)."""
    table, args, digits = prepared(case, backend, changes)
    return bars._assemble(table, args, parts(case, table, args), digits=digits)


def finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)
