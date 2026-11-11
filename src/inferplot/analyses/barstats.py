"""``barstats``: count the categories of x within each level of y, and test them.

The analysis that ``ip.analyze.barstats`` exposes, the counterpart of ggstatsplot
1.1.4's ``ggbarstats()``. It reads the table of counts through
:func:`inferplot._data.read_table`, validates every argument, and returns a
:class:`~inferplot.result.StatsResult` with ``design="contingency"``. It never
imports matplotlib.

What each design reproduces (SPEC.md section 4, "barstats"; every type but bayes
runs the same tests, as in ggstatsplot):

* two-way (``y`` given): Pearson's chi-square test of independence without
  Yates' correction, with the bias-corrected Cramér's V and its noncentral
  chi-square interval; Fisher's exact test between every two levels of x (in the
  result, never drawn);
* paired (``paired=True``): McNemar's test (Bowker's test of symmetry beyond 2 x
  2) without continuity correction, with Cohen's g and its Wilson interval;
* one-way (``y=None``): the chi-square goodness-of-fit test against ``ratio``
  with Pearson's C and its interval;
* with ``y``, each bar's own goodness-of-fit test of x against ``ratio``
  (always two-sided, 95%, unadjusted), whose p labels the bar.

Each design's analysis is a function ``(table, args) -> _Parts`` in
``_ANALYSES``, run after every argument has been validated and the table read
(``_prepare``); ``_assemble`` builds the result from its parts. With
``results_subtitle=False`` only the bars' tests are computed (no main test, no
pairs), as ggstatsplot computes them.

Where ggstatsplot drops a result silently, inferplot raises plainly (D101): no
pair of units that changes level, or two levels between which none changes
(McNemar's and Bowker's tests are 0/0). A Fisher pair without a test (its table
has one column; its exact test is beyond the cost limit) does not fail the
analysis: it stays in ``pairwise`` with p None and ``no_test_reason``
(ggstatsplot loses the whole pairwise table there). The other pairs are adjusted
among themselves, except that a pair too large to compute still counts in the
adjustment, as a test with p = 1 (``stats.pairwise_fisher``).
"""

from __future__ import annotations

import math
import numbers
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Final, Literal, cast

import numpy as np

from inferplot import stats
from inferplot._data import (
    TableData,
    TableLevels,
    declare_order,
    read_table,
    table_notes,
)
from inferplot._validate import (
    canonical_type,
    check_alternative,
    check_bool,
    check_conf_level,
    check_digits,
    check_optional_bool,
    check_p_adjust_method,
    check_seed,
    checked,
    closest_name,
    describe_value,
)
from inferplot.analyses._no_levels import canonical_alternative
from inferplot.result import (
    CELL_LABELS,
    Bar,
    Cell,
    EffectSize,
    PairwiseComparison,
    Settings,
    StatsResult,
    TestResult,
    current_provenance,
)
from inferplot.text import contingency_texts, r_number_text, zero_p_note

__all__ = ["barstats"]

Design = Literal["two-way", "paired", "one-way"]

# ggstatsplot's defaults of the arguments that can have no effect.
_DEFAULT_CONF_LEVEL: Final = 0.95
_DEFAULT_P_ADJUST: Final = "holm"
_DEFAULT_DIGITS_PERC: Final = 0
_DEFAULT_SEED: Final = 0
# What the result records for the settings barstats does not have.
_NO_BF_PRIOR: Final = 0.707
_NO_TR: Final = 0.2
_NO_PAIRWISE_ALPHA: Final = 0.05
# R's chisq.test() refuses probabilities whose sum is further than this from 1.
_RATIO_TOLERANCE: Final = math.sqrt(sys.float_info.epsilon)
# The level of the per-bar tests' intervals: ggstatsplot does not pass
# conf.level to them (onesample_data()).
BAR_CONF_LEVEL: Final = 0.95


@dataclass(frozen=True, slots=True)
class _Args:
    """The validated arguments every design's analysis reads.

    ``proportion_test`` is resolved: None became ``results_subtitle`` (with y;
    False without). ``passed`` holds the arguments as the caller gave them, for
    the warnings that name them.
    """

    x: str
    y: str | None
    counts: str | None
    type: str
    paired: bool
    ratio: tuple[float, ...] | None
    label: str
    digits_perc: int
    proportion_test: bool
    p_adjust_method: str
    conf_level: float
    alternative: stats.Alternative
    bf_message: bool | None
    results_subtitle: bool
    seed: int
    passed: Mapping[str, object] = field(default_factory=dict[str, object])

    @property
    def design(self) -> Design:
        if self.y is None:
            return "one-way"
        return "paired" if self.paired else "two-way"


@dataclass(frozen=True, slots=True)
class _BarTest:
    """One bar's proportion test (a level of y with observations): the
    goodness-of-fit test of x within the bar and Pearson's C, or None with the
    reason."""

    test: TestResult | None
    effect_size: EffectSize | None
    no_test_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Parts:
    """What one design's analysis computed.

    ``bar_tests`` holds one entry per level of y that holds observations, in
    order (empty without y); ``pairwise`` the Fisher pairs where ggstatsplot
    computes them (``_pairs_wanted``). ``warnings`` are the design's own:
    ``_assemble`` adds the data notes and the arguments without effect (D39).
    """

    test: TestResult | None
    effect_size: EffectSize | None
    bar_tests: tuple[_BarTest, ...]
    pairwise: tuple[PairwiseComparison, ...]
    warnings: tuple[str, ...]
    citations: tuple[str, ...]


# --------------------------------------------------------------------------- #
# the designs (each takes the table and the validated arguments)
# --------------------------------------------------------------------------- #

# What the result records for each test and effect size: ``name`` and ``label``
# are inferplot's, ``method`` is R's ``method`` string (statsExpressions' row).
_CHI2: Final = "chi2"
_NONCENTRAL_CHI2: Final = "noncentral chi-square"
_WILSON: Final = "Wilson score"
_GOF_NAME: Final = "chi_square_gof"
_GOF_LABEL: Final = "Chi-square goodness-of-fit test"
_GOF_METHOD: Final = "Chi-squared test for given probabilities"
_FISHER_NAME: Final = "fisher_exact"
_FISHER_LABEL: Final = "Fisher's exact test"
# ggstatsplot's threshold for a pair's ``significant`` (pairs are never drawn).
_PAIR_ALPHA: Final = 0.05
# Every count is held exactly in a double below this table total.
_MAX_TOTAL: Final = 2**53

_PEARSON_1900: Final = (
    "Pearson, K. (1900). Philosophical Magazine 50: 157-175 (the chi-square test)."
)
_BERGSMA_2013: Final = (
    "Bergsma, W. (2013). Journal of the Korean Statistical Society 42: 323-328 "
    "(bias-corrected Cramér's V)."
)
_SMITHSON_2003: Final = (
    "Smithson, M. (2003). Confidence Intervals. Sage, chapter 5 (noncentral "
    "chi-square intervals of V and C)."
)
_FISHER_1935: Final = (
    "Fisher, R. A. (1935). Journal of the Royal Statistical Society 98: 39-82 "
    "(the exact test of a contingency table)."
)
_MEHTA_PATEL_1983: Final = (
    "Mehta, C. R. & Patel, N. R. (1983). Journal of the American Statistical "
    "Association 78: 427-434 (network algorithm for exact tests of r x c tables)."
)
_MCNEMAR_1947: Final = "McNemar, Q. (1947). Psychometrika 12: 153-157 (McNemar's test)."
_BOWKER_1948: Final = (
    "Bowker, A. H. (1948). Journal of the American Statistical Association 43: "
    "572-574 (the test of symmetry)."
)
_COHEN_1988: Final = (
    "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences, "
    "2nd ed. Erlbaum, chapter 5 (Cohen's g)."
)
_WILSON_1927: Final = (
    "Wilson, E. B. (1927). Journal of the American Statistical Association 22: "
    "209-212 (the score interval)."
)
_PEARSON_1904: Final = (
    "Pearson, K. (1904). Mathematical Contributions to the Theory of Evolution "
    "XIII. Drapers' Company Research Memoirs (the contingency coefficient C)."
)


def _check_total(table: TableData) -> None:
    """Every count and margin must be an exact double (the statistics' limit)."""
    if table.n >= _MAX_TOTAL:
        raise ValueError(
            f"The table holds {table.n:,} observations; the tests of counts need "
            "fewer than 2^53 (9,007,199,254,740,992), so that every count and total "
            "is held exactly. Divide the counts by a common factor only if they "
            "are not counts of observations (a test of counts reads them as "
            "observations)."
        )


def _gof_test(test: stats.GoodnessOfFit) -> TestResult:
    return TestResult(
        name=_GOF_NAME,
        label=_GOF_LABEL,
        method=_GOF_METHOD,
        statistic_name=_CHI2,
        statistic=test.statistic,
        df=None,
        df_error=float(test.df),
        p_value=test.p_value,
        alternative="two-sided",
    )


def _effect(
    name: str, label: str, effect: stats.EffectSizeCI, ci_method: str
) -> EffectSize:
    return EffectSize(
        name=name,
        label=label,
        estimate=effect.estimate,
        conf_level=effect.conf_level,
        conf_low=effect.conf_low,
        conf_high=effect.conf_high,
        ci_method=ci_method,
        sidedness=effect.sidedness,
    )


def _bar_tests(table: TableData, args: _Args) -> tuple[tuple[_BarTest, ...], list[str]]:
    """The proportion test of each bar (the levels of y that hold observations,
    in order): the goodness-of-fit test of x within the bar against ``ratio``,
    every level of x kept, two-sided at 95% and unadjusted (ggstatsplot's
    ``onesample_data()``), with Pearson's C. A bar's test fails only when x has
    a single level, which ``_check_two_way`` refuses first (D101, review B21):
    that branch keeps the entry with the reason (R prints NA there) but is not
    reached through the API, so it adds no warning."""
    if table.y_levels is None:
        return (), []
    out: list[_BarTest] = []
    for bar in stats.proportion_tests(table.table(), args.ratio):
        if bar.test is None or bar.effect is None:
            reason = (
                f"x={args.x!r} has a single level, and a goodness-of-fit test "
                "compares two or more (ggstatsplot prints NA)"
            )
            out.append(_BarTest(test=None, effect_size=None, no_test_reason=reason))
            continue
        out.append(
            _BarTest(
                test=_gof_test(bar.test),
                effect_size=_effect(
                    "pearsons_c", "Pearson's C", bar.effect, _NONCENTRAL_CHI2
                ),
            )
        )
    return tuple(out), []


def _two_way(table: TableData, args: _Args) -> _Parts:
    """Pearson's chi-square test of independence (no Yates' correction) on the
    table without its empty rows and columns, with Cramér's V (Bergsma's bias
    correction, noncentral chi-square interval at ``conf_level``, following
    ``alternative``); the proportion test of each bar; and Fisher's exact test
    between every two levels of x where ggstatsplot computes them
    (``_pairs_wanted``). Without the results subtitle only the bars' tests."""
    _check_total(table)
    bar_tests, warnings = _bar_tests(table, args)
    if not args.results_subtitle:
        return _Parts(
            test=None,
            effect_size=None,
            bar_tests=bar_tests,
            pairwise=(),
            warnings=tuple(warnings),
            citations=(_PEARSON_1900, _PEARSON_1904, _SMITHSON_2003),
        )
    full = table.table()
    kept = full[full.sum(axis=1) > 0][:, full.sum(axis=0) > 0]
    rows, columns = kept.shape
    if table.n <= max(rows, columns):
        raise ValueError(
            f"The bias-corrected Cramér's V needs more observations than levels of "
            f"x and of y: the {rows} x {columns} table of x={args.x!r} by "
            f"y={args.y!r} holds {table.n} (R gives NaN for it, and ggstatsplot "
            "prints it). Collect more observations; with this few, no test of "
            "independence can say much."
        )
    chi2 = stats.chisq_independence(kept)
    v = stats.cramers_v(kept, conf_level=args.conf_level, alternative=args.alternative)
    test = TestResult(
        name="pearson_chi_square",
        label="Pearson's chi-square test",
        method="Pearson's Chi-squared test",
        statistic_name=_CHI2,
        statistic=chi2.statistic,
        df=None,
        df_error=float(chi2.df),
        p_value=chi2.p_value,
        alternative="two-sided",
    )
    effect = _effect("cramers_v", "Cramer's V", v, _NONCENTRAL_CHI2)
    citations = [_PEARSON_1900, _BERGSMA_2013, _SMITHSON_2003, _PEARSON_1904]
    pairs: tuple[PairwiseComparison, ...] = ()
    if _pairs_wanted(table, args):
        pairs, pair_warnings = _fisher_pairs(table, args)
        warnings.extend(pair_warnings)
        citations += [_FISHER_1935, _MEHTA_PATEL_1983]
    return _Parts(
        test=test,
        effect_size=effect,
        bar_tests=bar_tests,
        pairwise=pairs,
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


def _paired(table: TableData, args: _Args) -> _Parts:
    """McNemar's test (Bowker's test of symmetry beyond 2 x 2) without
    continuity correction on the square table with every declared level, with
    Cohen's g (Wilson score interval at ``conf_level``, following
    ``alternative``), and the proportion test of each bar; no pairs. Without the
    results subtitle only the bars' tests."""
    _check_total(table)
    bar_tests, warnings = _bar_tests(table, args)
    if not args.results_subtitle:
        return _Parts(
            test=None,
            effect_size=None,
            bar_tests=bar_tests,
            pairwise=(),
            warnings=tuple(warnings),
            citations=(_PEARSON_1900, _PEARSON_1904, _SMITHSON_2003),
        )
    full = table.table()
    symmetry = stats.mcnemar(full)
    g = stats.cohens_g(full, conf_level=args.conf_level, alternative=args.alternative)
    test = TestResult(
        name="mcnemar",
        label="McNemar's test",
        method="McNemar's Chi-squared test",
        statistic_name=_CHI2,
        statistic=symmetry.statistic,
        df=None,
        df_error=float(symmetry.df),
        p_value=symmetry.p_value,
        alternative="two-sided",
    )
    citations = [_MCNEMAR_1947]
    if full.shape[0] > 2:
        citations.append(_BOWKER_1948)
    citations += [_COHEN_1988, _WILSON_1927, _PEARSON_1900, _PEARSON_1904]
    return _Parts(
        test=test,
        effect_size=_effect("cohens_g", "Cohen's g", g, _WILSON),
        bar_tests=bar_tests,
        pairwise=(),
        warnings=tuple(warnings),
        citations=tuple(citations),
    )


def _check_discordant(table: TableData, args: _Args) -> None:
    """McNemar's test needs pairs that change level, and Bowker's needs some
    between every two levels (each term is 0 / 0 otherwise: R returns NaN, and
    ggstatsplot leaves out its subtitle silently)."""
    levels = table.x_levels.levels
    counts = table.counts
    k = len(levels)
    quiet = [
        (levels[i], levels[j])
        for i in range(k)
        for j in range(i + 1, k)
        if counts[i][j] + counts[j][i] == 0
    ]
    if len(quiet) == k * (k - 1) // 2:
        held = ", ".join(
            f"{level!r}: {counts[i][i]:,}" for i, level in enumerate(levels)
        )
        raise ValueError(
            f"McNemar's test compares the pairs whose level changes from "
            f"{args.x!r} to {args.y!r}, and none does: each of the {table.n:,} "
            f"pairs has the same level in both ({held}), so the test is not "
            "defined (R's statistic is 0/0, and ggstatsplot leaves out its subtitle "
            "silently). Report the agreement as it is, or pass "
            "results_subtitle=False to draw the bars without a test."
        )
    unused = [
        level
        for i, level in enumerate(levels)
        if sum(counts[i]) == 0 and sum(row[i] for row in counts) == 0
    ]
    if quiet and all(a in unused or b in unused for a, b in quiet):
        _refuse_unused(table, args, unused, quiet)
    if quiet:
        between = [f"between {a!r} and {b!r}" for a, b in quiet]
        listed = (
            between[0]
            if len(between) == 1
            else ", ".join(between[:-1]) + " or " + between[-1]
        )
        many = len(quiet) > 1
        raise ValueError(
            f"Bowker's test of symmetry (McNemar's test beyond two levels) needs "
            f"pairs that change between every two levels, and no pair changes "
            f"{listed}, in either direction: "
            f"{'those terms are' if many else 'that term is'} 0/0, so the test is "
            "not defined (R's statistic is NaN, and ggstatsplot leaves out its "
            "subtitle silently). Merge levels so that every two exchange some "
            "pairs, or compare two levels that do exchange pairs on their own "
            "(keep the units whose both measurements are among those two: a 2 x 2 "
            "McNemar test)."
        )


def _refuse_unused(
    table: TableData,
    args: _Args,
    unused: Sequence[str],
    quiet: Sequence[tuple[str, str]],
) -> None:
    """Bowker's test fails only for levels without observations in either
    measurement (review B19): say so, and give the call that leaves them out
    of both columns, in the frame's own library."""
    one = len(unused) == 1
    names = _levels_text(unused)
    kept = [level for level in table.x_levels.levels if level not in unused]
    names_ = [name for name in (args.x, args.y) if name is not None]
    if table.library == "pandas":  # declared, so both are categoricals
        calls = "; ".join(
            f"data[{name!r}] = data[{name!r}].cat.remove_categories({list(unused)!r})"
            for name in names_
        )
    else:
        calls = "; ".join(declare_order(table.library, name, kept) for name in names_)
    test = "McNemar's test" if len(kept) == 2 else "Bowker's test"
    terms = ", ".join(f"{a!r} and {b!r}" for a, b in quiet)
    raise ValueError(
        f"The declared {'level' if one else 'levels'} {names} "
        f"{'has' if one else 'have'} no observations in either measurement "
        f"({args.x!r} or {args.y!r}), so "
        f"Bowker's test of symmetry has a term of 0/0 for every pair with "
        f"{'it' if one else 'them'} ({terms}) and is not defined (R's statistic "
        "is NaN, and ggstatsplot leaves out its subtitle silently). If "
        f"{'it is' if one else 'they are'} not a possible answer, leave "
        f"{'it' if one else 'them'} out of both columns, and {test} runs on the "
        f"other levels: {calls}."
    )


def _one_way(table: TableData, args: _Args) -> _Parts:
    """The chi-square goodness-of-fit test of every level of x (declared empty
    ones as a count of 0) against ``ratio`` (equal shares by default), with
    Pearson's C (noncentral chi-square interval at ``conf_level``, following
    ``alternative``). Without the results subtitle there is no test."""
    _check_total(table)
    citations = (_PEARSON_1900, _PEARSON_1904, _SMITHSON_2003)
    if not args.results_subtitle:
        return _Parts(
            test=None,
            effect_size=None,
            bar_tests=(),
            pairwise=(),
            warnings=(),
            citations=citations,
        )
    counts = table.table()[:, 0]
    fit = stats.goodness_of_fit(counts, args.ratio)
    c = stats.pearsons_c(
        counts, args.ratio, conf_level=args.conf_level, alternative=args.alternative
    )
    return _Parts(
        test=_gof_test(fit),
        effect_size=_effect("pearsons_c", "Pearson's C", c, _NONCENTRAL_CHI2),
        bar_tests=(),
        pairwise=(),
        warnings=(),
        citations=citations,
    )


_SHOWN_PAIRS: Final = 6


def _fisher_pairs(
    table: TableData, args: _Args
) -> tuple[tuple[PairwiseComparison, ...], list[str]]:
    """Fisher's exact test between every two levels of x that hold observations
    (level order), each on its own 2 x c table (the levels of y either holds),
    with Cramér's V; the p-values adjusted by ``p_adjust_method`` over the pairs
    that have a test (``stats.pairwise_fisher``, which holds the tests of one
    table to a shared work budget, cheapest first). A pair without a test (one
    column; an exact test beyond the budget) stays with p None and the reason,
    and ``warnings`` names it (ggstatsplot loses the whole table where
    fisher.test stops)."""
    assert table.y_levels is not None
    levels = table.x_levels.levels
    y_levels = table.y_levels.levels
    method = cast("stats.PAdjustMethod", args.p_adjust_method)
    found = stats.pairwise_fisher(
        table.table(),
        p_adjust_method=method,
        conf_level=args.conf_level,
        alternative=args.alternative,
    )
    out: list[PairwiseComparison] = []
    warnings: list[str] = []
    single = 0
    costly: list[stats.PairComparison] = []
    for pair in found:
        name = f"{levels[pair.row1]!r} and {levels[pair.row2]!r}"
        reason = None
        if pair.p_value is None and len(pair.columns) == 1:
            single += 1
            column = y_levels[pair.columns[0]]
            reason = (
                f"both levels occur only in {column!r} of {args.y!r}, so their "
                "table has a single column and no test; R's fisher.test stops there"
            )
            warnings.append(
                f"The pair {name} of {args.x!r} has no Fisher test: both levels "
                f"occur only in {column!r} of {args.y!r}, so their table has a "
                "single column, and their shares cannot differ across the levels of "
                f"{args.y!r}. Nothing needs fixing unless you need that comparison."
            )
        elif pair.p_value is None:
            costly.append(pair)
            reason = (
                "too large to compute exactly within inferplot's limits (a 2 x "
                f"{len(pair.columns)} table of {pair.n:,}), not approximated"
            )
        elif pair.effect is None:
            warnings.append(
                f"The pair {name} of {args.x!r} has no Cramér's V: its 2 x "
                f"{len(pair.columns)} table holds {pair.n} observations, and the "
                "bias-corrected V needs more than the number of its columns (R gives "
                "NaN). Its Fisher test is reported."
            )
        effect = pair.effect
        out.append(
            PairwiseComparison(
                group1=levels[pair.row1],
                group2=levels[pair.row2],
                statistic_name="",
                statistic=None,
                df=None,
                p_value=pair.p_value,
                p_adjusted=pair.p_adjusted,
                adjust_method=args.p_adjust_method,
                test=_FISHER_NAME,
                test_label=_FISHER_LABEL,
                significant=None
                if pair.p_adjusted is None
                else pair.p_adjusted < _PAIR_ALPHA,
                displayed=False,
                estimate=None if effect is None else effect.estimate,
                conf_level=None if effect is None else effect.conf_level,
                conf_low=None if effect is None else effect.conf_low,
                conf_high=None if effect is None else effect.conf_high,
                no_test_reason=reason,
            )
        )
    if costly:
        warnings.append(_costly_pairs_warning(costly, len(found), levels, args))
    missing = single + len(costly)
    if missing:
        rest = len(found) - missing
        lost = (
            " ggstatsplot loses the whole pairwise table here, silently."
            if single
            else ""
        )
        # A pair too large to compute is still a comparison: it is counted with
        # p = 1 (stats.pairwise_fisher); a single-column pair is none.
        family = (
            f"adjusts the other {rest} as {rest + len(costly)} comparisons: a pair "
            "too large to compute counts as a test with p = 1, so leaving it out "
            "makes no computed pair look more significant"
            if costly
            else f"adjusts the other {rest} among themselves"
        )
        warnings.append(
            f"{missing} of the {len(found)} pairwise Fisher tests "
            f"{'has' if missing == 1 else 'have'} no p-value (above); "
            f"{'it stays' if missing == 1 else 'they stay'} in result.pairwise "
            f"without one, and p_adjust_method={args.p_adjust_method!r} {family}."
            f"{lost}"
            if rest
            else f"None of the {len(found)} pairwise Fisher tests has a p-value "
            f"(above); the pairs stay in result.pairwise without one.{lost}"
        )
    return tuple(out), warnings


def _costly_pairs_warning(
    costly: list[stats.PairComparison],
    total: int,
    levels: tuple[str, ...],
    args: _Args,
) -> str:
    """The pairs whose exact test was too large to compute: how many, which, why,
    and what would give them a test."""
    names = [f"{levels[p.row1]!r} and {levels[p.row2]!r}" for p in costly]
    shown = "; ".join(names[:_SHOWN_PAIRS])
    if len(names) > _SHOWN_PAIRS:
        shown += f"; and {len(names) - _SHOWN_PAIRS} more"
    tested = total - len(costly)
    dense = all(p.n >= 50 * len(p.columns) ** 2 for p in costly)
    advice = (
        f"with this many observations the chi-square test answers the same "
        f"question: compare two levels of {args.x!r} in a call of their own, whose "
        "main test is the chi-square test of their table"
        if dense
        else f"merge sparse levels of {args.y!r} (levels with few observations), or "
        f"compare two levels of {args.x!r} in a call of their own"
    )
    return (
        f"Fisher's exact test was computed for {tested} of the {total} pairs of "
        f"levels of {args.x!r}; the other {len(costly)} ({shown}) "
        f"{'is' if len(costly) == 1 else 'are'} too large to compute exactly within "
        "inferplot's limits (about a second of work for one test and two seconds "
        "for all the pairs of one table, the cheapest computed first), and "
        "inferplot does not approximate them. Their Cramér's V is reported. To "
        f"test them, {advice}."
    )


_ANALYSES: Final[dict[Design, Callable[[TableData, _Args], _Parts]]] = {
    "two-way": _two_way,
    "paired": _paired,
    "one-way": _one_way,
}


def _pairs_wanted(table: TableData, args: _Args) -> bool:
    """Whether ggstatsplot computes the pairwise Fisher tests: with the results
    subtitle, unpaired, more than two levels of x and more than one of y
    (declared empty levels count, as ggstatsplot counts factor levels; the pairs
    themselves are between the levels that hold observations)."""
    return (
        args.results_subtitle
        and not args.paired
        and table.y_levels is not None
        and len(table.x_levels.levels) > 2
        and len(table.y_levels.levels) > 1
    )


def _bayes_caption_possible(args: _Args) -> bool:
    """Where ggstatsplot draws a Bayes factor caption (SPEC.md section 2,
    ``bf_message``): with the results subtitle and unpaired (one level of y,
    where it draws none, is refused)."""
    return args.results_subtitle and not args.paired


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #


def _levels_text(levels: Sequence[str]) -> str:
    """``'a'``, ``'a' and 'b'``, ``'a', 'b' and 'c'``."""
    quoted = [repr(level) for level in levels]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _empty_level_notes(table: TableData, args: _Args) -> list[str]:
    """What a declared level without observations means in each design: a row
    of zeros in the legend (x), no bar (y), a category with a count of 0 (one
    way), a level of both measurements (paired)."""
    notes: list[str] = []
    for column, role in ((table.x_levels, "x"), (table.y_levels, "y")):
        if column is None or not column.empty:
            continue
        one = len(column.empty) == 1
        start = (
            f"The declared {'level' if one else 'levels'} "
            f"{_levels_text(column.empty)} of {column.name!r} "
            f"{'has' if one else 'have'} no observations"
        )
        if args.design == "one-way":
            rest = (
                f"; {'it is a category' if one else 'they are categories'} with a "
                "count of 0 in the goodness-of-fit test, as ggstatsplot counts "
                f"{'it' if one else 'them'}."
            )
        elif args.design == "paired":
            # Review B19: McNemar's test is named only where it is computed.
            where = " in McNemar's test" if args.results_subtitle else ""
            rest = (
                f"; {'it stays a level' if one else 'they stay levels'} of both "
                f"measurements{where}, with a count of 0."
                + (_empty_in_bar_tests(table, args, column) if role == "x" else "")
            )
        elif role == "x":
            rest = (
                f"; {'it stays' if one else 'they stay'} in the legend with no share "
                "of any bar, and the chi-square test leaves the empty "
                f"{'row' if one else 'rows'} out (as ggstatsplot does)."
                + _empty_in_bar_tests(table, args, column)
            )
        else:
            rest = (
                f", so {'it has no bar' if one else 'they have no bars'}, and the "
                f"chi-square test leaves the empty {'column' if one else 'columns'} "
                "out (as ggstatsplot does)."
            )
        notes.append(start + rest)
    return notes


def remove_unused(table: TableData, column: TableLevels) -> str:
    """The statement that leaves the unused categories of ``column`` out, in the
    frame's own library (review B18)."""
    if table.library == "pandas":
        name = column.name
        return f"data[{name!r}] = data[{name!r}].cat.remove_unused_categories()"
    return declare_order(table.library, column.name, column.present)


def _empty_in_bar_tests(table: TableData, args: _Args, column: TableLevels) -> str:
    """What declared empty levels of x do to the bars' proportion tests (review
    B18): each bar's goodness-of-fit test counts them as categories expected
    to hold their share of the bar, so a bar's p can be small although its
    drawn shares are close; and how to leave them out."""
    if table.y_levels is None or not args.proportion_test:
        return ""
    k = len(column.levels)
    one = len(column.empty) == 1
    if args.ratio is None:
        share = f"1/{k} of the bar" if one else f"1/{k} of the bar each"
    else:
        shares = [
            r_number_text(args.ratio[column.levels.index(level)])
            for level in column.empty
        ]
        share = f"the share {', '.join(shares)} of the bar"
        if not one:
            share = f"the shares {', '.join(shares)} of the bar"
    return (
        f" Each bar's proportion test, though, counts "
        f"{'it as a category' if one else 'them as categories'} expected to hold "
        f"{share} ({k} categories in all), so the p above a bar can be small "
        "although its drawn shares are close. If "
        f"{'it is not a possible answer' if one else 'they are not possible answers'}"
        f", leave {'it' if one else 'them'} out: {remove_unused(table, column)}."
    )


def _no_effect_warnings(table: TableData, args: _Args, parts: _Parts) -> list[str]:
    """One plain-English entry per argument that has no effect on these data, or
    a narrower one than the caller may expect (D39), naming the value as the
    caller gave it."""
    out: list[str] = []
    passed = args.passed
    adjust = _p_adjust_note(table, args, parts)
    if adjust is not None:
        out.append(adjust)
    level = passed.get("conf_level")
    if args.conf_level == _DEFAULT_CONF_LEVEL:
        pass
    elif not args.results_subtitle:
        bars = (
            "; the proportion test of each bar reports a 95% interval whatever "
            "conf_level says (as in ggstatsplot)"
            if args.y is not None
            else ""
        )
        out.append(
            f"conf_level={level!r} has no effect: results_subtitle=False computes "
            f"no main test and no pairs{bars}."
        )
    elif args.y is not None:
        applies = (
            "the main effect size and the pairs"
            if parts.pairwise
            else "the main effect size"
        )
        out.append(
            "The proportion test of each bar reports Pearson's C with a 95% "
            f"interval whatever conf_level says (as in ggstatsplot): "
            f"conf_level={level!r} applies to {applies} only."
        )
    if args.digits_perc != _DEFAULT_DIGITS_PERC and args.label == "counts":
        out.append(
            f"digits_perc={passed.get('digits_perc')!r} has no effect: "
            "label='counts' shows no percentages."
        )
    if passed.get("proportion_test") is False and args.y is None:
        out.append(
            "proportion_test=False has no effect: without y there is a single bar, "
            "whose test is the main test."
        )
    if args.ratio is not None and args.y is not None:
        out.append(
            "ratio= sets the expected shares of the proportion test of each bar; "
            + (
                "McNemar's test"
                if args.paired
                else "the chi-square test of independence"
            )
            + " does not use it (as in ggstatsplot)."
        )
    if args.alternative != "two-sided":
        out.append(_alternative_note(table, args, parts))
    if args.seed != _DEFAULT_SEED:
        out.append(
            f"seed={args.seed!r} has no effect: no test of counts draws random "
            "numbers (the chi-square, McNemar and exact Fisher tests use the counts "
            "as they are); it is recorded in the result."
        )
    if _bayes_caption_possible(args):
        if args.bf_message is None:
            what = (
                "a Bayesian goodness-of-fit test"
                if args.y is None
                else "a Bayesian contingency-table test"
            )
            out.append(
                "ggstatsplot draws a Bayes factor caption here; the Bayes factor "
                f"of counts ({what}) arrives in inferplot 0.2, so there is none. "
                "Pass bf_message=False to leave it out without this note."
            )
    elif args.bf_message is False:
        why = (
            "results_subtitle=False leaves the caption out"
            if not args.results_subtitle
            else "ggstatsplot draws none for paired data"
        )
        out.append(f"bf_message=False has no effect: {why}.")
    return out


def _p_adjust_note(table: TableData, args: _Args, parts: _Parts) -> str | None:
    """``p_adjust_method`` changed from Holm where it adjusts nothing: no pairs
    are computed, or at most one has a test (a single p-value is its own
    adjusted p-value under every method)."""
    if args.p_adjust_method == _DEFAULT_P_ADJUST:
        return None
    head = (
        f"p_adjust_method={args.passed.get('p_adjust_method')!r} has no effect: it "
        "adjusts the pairwise Fisher tests between the levels of x, and "
    )
    if not _pairs_wanted(table, args):
        if not args.results_subtitle:
            why = "results_subtitle=False leaves them out, as in ggstatsplot"
        elif args.paired:
            why = "paired data have none"
        elif args.y is None:
            why = "without y there are none"
        else:
            why = (
                f"x={args.x!r} has {len(table.x_levels.levels)} levels, and they are "
                "computed for three or more"
            )
        return f"{head}{why}."
    tested = [pair for pair in parts.pairwise if pair.p_value is not None]
    if len(tested) > 1:
        return None
    if not tested:
        why = "none of the pairs has a test"
    else:
        (pair,) = tested
        why = (
            f"only one pair has a test ({pair.group1!r} and {pair.group2!r}), and "
            "a single p-value is its own adjusted p-value under every method"
        )
    return f"{head}{why}."


def _two_by_two_pairs(table: TableData, parts: _Parts) -> int:
    """The pairs whose table (their two rows, over the levels of y either holds)
    is 2 x 2: the only ones whose Fisher test follows ``alternative``."""
    index = {level: i for i, level in enumerate(table.x_levels.levels)}
    out = 0
    for pair in parts.pairwise:
        first = table.counts[index[pair.group1]]
        second = table.counts[index[pair.group2]]
        if sum(1 for a, b in zip(first, second, strict=True) if a or b) == 2:
            out += 1
    return out


def _alternative_note(table: TableData, args: _Args, parts: _Parts) -> str:
    """What a one-sided ``alternative`` changes in barstats: the effect-size
    intervals and the Fisher tests of 2 x 2 pair tables, never the chi-square
    or McNemar p-value (ggstatsplot passes it to effectsize and fisher.test
    only)."""
    side = args.alternative
    if not args.results_subtitle:
        bars = (
            ", and the proportion test of each bar is always two-sided"
            if args.y is not None
            else ""
        )
        return (
            f"alternative={side!r} has no effect: results_subtitle=False computes no "
            f"interval and no pairwise test{bars}."
        )
    bound = "[low, 1]" if side == "greater" else "[0, high]"
    pairs = ""
    if args.design == "two-way" and parts.pairwise:
        effect = f"Cramér's V interval one-sided, {bound}, also for each pair,"
        square = _two_by_two_pairs(table, parts)
        if square == len(parts.pairwise):
            pairs = (
                " and makes the Fisher test of each pair whose table is 2 x 2 one-sided"
            )
        elif square:
            pairs = (
                f" and makes the Fisher test of the {square} "
                f"{'pair' if square == 1 else 'pairs'} whose table is 2 x 2 "
                "one-sided (fisher.test ignores alternative beyond 2 x 2)"
            )
        else:
            effect = effect.rstrip(",")
            pairs = (
                "; the pairs' Fisher tests stay two-sided, as no pair's table is "
                "2 x 2 (fisher.test ignores alternative beyond 2 x 2)"
            )
    else:
        g_bound = "[low, 0.5]" if side == "greater" else "[-0.5, high]"
        effect = {
            "two-way": f"Cramér's V interval one-sided, {bound}",
            "paired": f"Cohen's g interval one-sided, {g_bound}",
            "one-way": f"Pearson's C interval one-sided, {bound}",
        }[args.design]
    test = {
        "two-way": "the chi-square test",
        "paired": "McNemar's test",
        "one-way": "the goodness-of-fit test",
    }[args.design]
    bars = ", as are the proportion tests of the bars" if args.y is not None else ""
    return (
        f"alternative={side!r} makes the {effect}{pairs}; {test} has no direction, "
        f"so its p-value is the same{bars} (as in ggstatsplot)."
    )


def _bars_and_cells(
    table: TableData, parts: _Parts
) -> tuple[tuple[Bar, ...], tuple[Cell, ...]]:
    """One bar per level of y with observations (one bar without y) with its
    test, and one cell per bar and level of x with its share of the bar."""
    totals = table.column_totals()
    if table.y_levels is None:
        bars = (
            Bar(
                level=None,
                n=totals[0],
                test=None,
                effect_size=None,
                no_test_reason=(
                    "Without y there is a single bar; its goodness-of-fit test is "
                    "the main test (result.test)."
                ),
            ),
        )
        columns: list[int] = [0]
    else:
        columns = [j for j, total in enumerate(totals) if total > 0]
        if len(parts.bar_tests) != len(columns):
            raise ValueError(
                f"barstats: {len(parts.bar_tests)} bar tests for {len(columns)} bars"
            )
        levels = table.y_levels.levels
        bars = tuple(
            Bar(
                level=levels[j],
                n=totals[j],
                test=bar.test,
                effect_size=bar.effect_size,
                no_test_reason=bar.no_test_reason,
            )
            for j, bar in zip(columns, parts.bar_tests, strict=True)
        )
    cells = tuple(
        Cell(
            bar=bar.level,
            level=level,
            count=table.counts[i][j],
            percent=table.counts[i][j] / bar.n * 100,
        )
        for bar, j in zip(bars, columns, strict=True)
        for i, level in enumerate(table.x_levels.levels)
    )
    return bars, cells


def _assemble(
    table: TableData, args: _Args, parts: _Parts, *, digits: int
) -> StatsResult:
    """The result of a barstats analysis from the parts its design computed.

    ``levels`` and ``y_levels`` are the levels of x and y with declared empty
    ones; ``counts`` the table; ``bars`` and ``cells`` what the figure draws;
    ``n_obs`` the table total and ``n_dropped`` the rows with a missing x, y or
    counts.
    """
    bars, cells = _bars_and_cells(table, parts)
    settings = Settings(
        conf_level=args.conf_level,
        digits=digits,
        alternative=args.alternative,
        p_adjust_method=args.p_adjust_method,
        # The pairs are never drawn; barstats has no prior scale and no trimming.
        pairwise_display="none",
        pairwise_alpha=_NO_PAIRWISE_ALPHA,
        bf_prior=_NO_BF_PRIOR,
        bf_message=args.bf_message,
        results_subtitle=args.results_subtitle,
        seed=args.seed,
        tr=_NO_TR,
        paired=args.paired,
        ratio=args.ratio,
        label=args.label,
        digits_perc=args.digits_perc,
        proportion_test=args.proportion_test,
    )
    levels = table.x_levels.levels
    y_levels = () if table.y_levels is None else table.y_levels.levels
    texts = contingency_texts(
        x=table.x,
        y=table.y,
        levels=levels,
        y_levels=y_levels,
        test=parts.test,
        effect_size=parts.effect_size,
        bars=bars,
        cells=cells,
        pairwise=parts.pairwise,
        n_obs=table.n,
        settings=settings,
    )
    warnings = (
        *table_notes(table),
        *_empty_level_notes(table, args),
        *parts.warnings,
        *_no_effect_warnings(table, args, parts),
        *_zero_p_notes(parts, bars, settings),
    )
    return StatsResult(
        analysis="barstats",
        design="contingency",
        type=args.type,
        x=table.x,
        y=table.y,
        subject_id=None,
        levels=levels,
        values=(),
        subjects=(),
        descriptives=(),
        test=parts.test,
        effect_size=parts.effect_size,
        bayes=None,
        pairwise=parts.pairwise,
        n_obs=table.n,
        n_dropped=table.n_dropped,
        dropped_subjects=0,
        dropped_subject_rows=0,
        n_tied=0,
        warnings=warnings,
        provenance=current_provenance(seed=args.seed, citations=parts.citations),
        settings=settings,
        text=texts,
        y_levels=y_levels,
        counts=table.counts,
        cells=cells,
        bars=bars,
        counts_var=table.counts_var,
    )


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def _zero_p_notes(parts: _Parts, bars: Sequence[Bar], settings: Settings) -> list[str]:
    """What a p-value of exactly 0 means where a text prints it as 2.23e-308
    (D108, review B15): the main test's in the subtitle, the bars' in the
    labels above them and in the alt text."""
    notes: list[str] = []
    test = parts.test
    if settings.results_subtitle and test is not None and test.p_value == 0.0:
        notes.append(zero_p_note(test.label))
    zero = [
        bar.level
        for bar in bars
        if bar.test is not None and bar.test.p_value == 0.0 and bar.level is not None
    ]
    if zero and settings.proportion_test:
        names = ", ".join(repr(level) for level in zero)
        many = len(zero) > 1
        notes.append(
            zero_p_note(
                f"the goodness-of-fit test of the bar{'s' if many else ''} {names}",
                "The labels above those bars print"
                if many
                else "The label above that bar prints",
            )
        )
    return notes


def _check_type(value: object) -> str:
    """The canonical type. "nonparametric" runs the same tests as "parametric",
    as in ggstatsplot; "robust" has no method for counts and "bayes" arrives in
    inferplot 0.2."""
    type_name = canonical_type(value)
    if type_name == "robust":
        raise ValueError(
            f"type={value!r}: there is no robust method for counts. barstats' tests "
            "(Pearson's chi-square, McNemar's, goodness of fit) use the counts "
            "themselves, and ggstatsplot silently runs the same test for "
            "type='robust'. Use type='parametric' (type='nonparametric' runs the "
            "same test)."
        )
    if type_name == "bayes":
        raise NotImplementedError(
            f"type={value!r}: the Bayes factor of counts (ggstatsplot's Bayesian "
            "contingency-table and goodness-of-fit tests) arrives in inferplot 0.2. "
            "Use type='parametric' for the chi-square test."
        )
    return type_name


def check_label(value: object, name: str = "label") -> str:
    """What each cell's label shows: "percentage" (its share of the bar),
    "counts" or "both"."""
    allowed = ", ".join(repr(label) for label in CELL_LABELS)
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, one of {allowed}; got {describe_value(value)}."
        )
    if value in CELL_LABELS:
        return str(value)
    close = closest_name(value, CELL_LABELS)
    hint = f" Did you mean {name}={close!r}?" if close is not None else ""
    raise ValueError(f"{name}={value!r} is not allowed; use one of {allowed}.{hint}")


def _ratio_entry(item: object) -> float:
    """One share of ``ratio``: a real number (not a bool), a float32 read as it
    prints (as every number argument is)."""
    if isinstance(item, bool | np.bool_) or not isinstance(item, numbers.Real):
        raise TypeError(
            "ratio must hold numbers, the expected share of each level of x, for "
            "example ratio=[0.5, 0.25, 0.25]; got "
            f"{describe_value(cast('object', item))}."
        )
    if isinstance(item, np.floating) and item.dtype.itemsize < 8:
        return float(str(item))
    return float(item)


def check_ratio(value: object, name: str = "ratio") -> tuple[float, ...] | None:
    """``None`` (equal shares) or the expected share of each level of x, in the
    order of the levels: positive numbers summing to 1 within R's tolerance,
    sqrt(machine epsilon). Never rescaled (``chisq.test`` refuses such
    probabilities and ggstatsplot then leaves out its subtitle silently); the
    error shows the rescaled values. The length is checked against the levels
    once the data are read."""
    if value is None:
        return None
    example = f"for example {name}=[0.5, 0.25, 0.25]"
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        raise TypeError(
            f"{name} must be a list of the expected shares of the levels of x, in "
            f"their order, {example}; got {describe_value(value)}."
        )
    if isinstance(value, Set | Mapping):
        why = (
            "a set has no order"
            if isinstance(value, Set)
            else "a mapping's order is not the order of the levels"
        )
        raise TypeError(
            f"{name} must list the shares in the order of the levels of x, but got "
            f"a {type(cast('object', value)).__name__} ({why}). Pass a list, {example}."
        )
    shares = tuple(_ratio_entry(item) for item in cast("Iterable[object]", value))
    bad = [s for s in shares if not (math.isfinite(s) and s > 0)]
    if bad:
        zero = (
            " A share of 0 expects no observations in its level: the "
            "goodness-of-fit statistic divides by it (R returns Inf or NaN). If "
            "the level cannot occur, leave it out of the data (do not declare it) "
            "and give the shares of the others."
            if bad[0] == 0
            else ""
        )
        raise ValueError(
            f"{name} must hold positive numbers, the expected share of each level "
            f"of x ({example}); got {bad[0]!r}.{zero}"
        )
    total = math.fsum(shares)
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        rescaled = [float(f"{s / total:.12g}") for s in shares]
        raise ValueError(
            f"{name} must sum to 1: it holds the expected share of each level of x, "
            f"and {list(shares)!r} sums to {total:g}. inferplot does not rescale it "
            "(R's chisq.test refuses it, and ggstatsplot then leaves out its "
            f"subtitle silently). If these are relative sizes, write "
            f"{name}={rescaled!r}."
        )
    return shares


def _check_bf_message(*, show_bf: bool | None, subtitle_on: bool, paired: bool) -> None:
    """``bf_message=True``: ggstatsplot draws the caption only with the subtitle
    and for unpaired data; there, the Bayes factor of counts arrives in 0.2."""
    if show_bf is not True:
        return
    if not subtitle_on:
        raise ValueError(
            "bf_message=True asks for the Bayes factor caption, which is drawn only "
            "together with the results subtitle; results_subtitle=False turns both "
            "off. Pass results_subtitle=True, or drop bf_message=True."
        )
    if paired:
        raise ValueError(
            "bf_message=True asks for a Bayes factor caption, which ggstatsplot does "
            "not draw for paired data (McNemar's test has no Bayes factor there). "
            "Remove bf_message=True."
        )
    raise NotImplementedError(
        "bf_message=True: the Bayes factor caption of counts (a Bayesian "
        "contingency-table or goodness-of-fit test) arrives in inferplot 0.2. "
        "Remove bf_message=True; the default (None) leaves the caption out and "
        "says so in result.warnings."
    )


def _check_design(table: TableData, args: _Args) -> None:
    """The tables no test of the design can use (D101; ggstatsplot leaves its
    subtitle out silently for each): one category without y, one level of x or
    of y with observations (one cell) with y, fewer than two levels paired, no
    unit that changes level (or two levels between which none does) where
    McNemar's test is computed, and a ``ratio`` of the wrong length."""
    x_levels = table.x_levels
    paired_levels = list(x_levels.levels)
    if args.design == "one-way":
        _check_one_way(x_levels)
    elif args.design == "two-way":
        assert table.y_levels is not None
        _check_two_way(x_levels, table.y_levels)
    elif len(paired_levels) < 2:
        raise ValueError(
            f"paired=True: x={args.x!r} and y={args.y!r} have a single level "
            f"({paired_levels[0]!r}), and McNemar's test compares the pairs that "
            "change between two or more levels. Declare both columns on every "
            "possible category if some have no rows."
        )
    elif args.results_subtitle:
        _check_discordant(table, args)
    if args.ratio is not None and len(args.ratio) != len(x_levels.levels):
        declared = (
            f", the declared empty {'one' if len(x_levels.empty) == 1 else 'ones'} "
            f"included ({_levels_text(x_levels.empty)})"
            if x_levels.empty
            else ""
        )
        raise ValueError(
            f"ratio has {len(args.ratio)} shares but x={args.x!r} has "
            f"{len(x_levels.levels)} levels{declared}: "
            f"{', '.join(repr(level) for level in x_levels.levels)}. Give one share "
            "per level, in that order (R's chisq.test refuses it too, and "
            "ggstatsplot then leaves out its subtitle silently)."
        )


def _check_one_way(x_levels: TableLevels) -> None:
    levels = list(x_levels.levels)
    if len(levels) < 2:
        raise ValueError(
            f"x={x_levels.name!r} has a single category ({levels[0]!r}), and the "
            "goodness-of-fit test compares the counts of two or more (ggstatsplot "
            "leaves out its subtitle silently here). If other categories are "
            "possible but have no rows, declare them, for example "
            f"pd.Categorical(data[{x_levels.name!r}], categories=[{levels[0]!r}, "
            "...]), and they count as 0."
        )


def _check_two_way(x_levels: TableLevels, y_levels: TableLevels) -> None:
    x_present = list(x_levels.present)
    y_present = list(y_levels.present)
    if len(x_present) >= 2 and len(y_present) >= 2:
        return
    single = [
        f"{column.name!r} has observations only in {present[0]!r}"
        for column, present in ((x_levels, x_present), (y_levels, y_present))
        if len(present) < 2
    ]
    x, y = x_levels.name, y_levels.name
    if len(single) == 2:
        advice = "Every observation is in one cell, so there is nothing to compare."
    elif len(x_present) < 2:
        advice = (
            f"To test how the observations divide among the levels of {y!r}, count "
            f"that column alone: barstats(data, x={y!r})."
        )
    else:
        advice = (
            f"To test how the observations divide among the levels of {x!r}, count "
            f"that column alone: barstats(data, x={x!r})."
        )
    raise ValueError(
        f"The chi-square test of independence needs observations in two or more "
        f"levels of x={x!r} and of y={y!r}, and {' and '.join(single)}. "
        "(ggstatsplot turns such "
        "a table into a goodness-of-fit test, finds no effect size for it and "
        f"leaves out its subtitle silently.) {advice}"
    )


def _prepare(
    data: object,
    x: str,
    y: str | None = None,
    *,
    counts: str | None = None,
    paired: bool = False,
    ratio: Sequence[float] | None = None,
    label: str = "percentage",
    proportion_test: bool | None = None,
    p_adjust_method: str = "holm",
    digits_perc: int = 0,
    type: str = "parametric",
    conf_level: float = 0.95,
    digits: int = 2,
    alternative: str = "two-sided",
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    seed: int = 0,
) -> tuple[TableData, _Args, int]:
    """Every argument checked and the table read: the table, the validated
    arguments and ``digits``. The public function runs the design's analysis on
    them; tests use it to build results before the statistics exist."""
    type_name = _check_type(type)
    pair = check_bool(paired, "paired")
    shares = check_ratio(ratio)
    cell_label = check_label(label)
    wants_tests = check_optional_bool(proportion_test, "proportion_test")
    adjust = check_p_adjust_method(p_adjust_method)
    n_digits_perc = check_digits(digits_perc, "digits_perc")
    level = check_conf_level(conf_level)
    n_digits = check_digits(digits)
    side = canonical_alternative(check_alternative(alternative, directionless=True))
    show_bf = check_optional_bool(bf_message, "bf_message")
    subtitle_on = check_bool(results_subtitle, "results_subtitle")
    rng_seed = check_seed(seed)
    if counts is not None and not isinstance(cast("object", counts), str):
        raise TypeError(
            "counts must be a column name (a string): the column that says how many "
            "observations each row stands for, for example counts='n'; got "
            f"{describe_value(counts)}."
        )
    if y is None:
        if pair:
            raise ValueError(
                f"paired=True needs y: x={x!r} and y are the two measurements of "
                "each unit (McNemar's test), for example barstats(data, "
                f"x={x!r}, y='<second measurement>', paired=True)."
            )
        if wants_tests is True:
            raise ValueError(
                "proportion_test=True asks for a proportion test above each bar, "
                "and without y there is a single bar, whose test is the main test. "
                "Remove proportion_test=True, or pass y."
            )
    _check_bf_message(show_bf=show_bf, subtitle_on=subtitle_on, paired=pair)

    table = read_table(data, x, y, counts=counts, paired=pair)
    resolved = (subtitle_on if wants_tests is None else wants_tests) and y is not None
    args = _Args(
        x=x,
        y=y,
        counts=counts,
        type=type_name,
        paired=pair,
        ratio=shares,
        label=cell_label,
        digits_perc=n_digits_perc,
        proportion_test=resolved,
        p_adjust_method=adjust,
        conf_level=level,
        alternative=side,
        bf_message=show_bf,
        results_subtitle=subtitle_on,
        seed=rng_seed,
        passed={
            "p_adjust_method": p_adjust_method,
            "conf_level": level,
            "digits_perc": n_digits_perc,
            "proportion_test": wants_tests,
        },
    )
    _check_design(table, args)
    return table, args, n_digits


@checked
def barstats(
    data: object,
    x: str,
    y: str | None = None,
    *,
    counts: str | None = None,
    paired: bool = False,
    ratio: Sequence[float] | None = None,
    label: str = "percentage",
    proportion_test: bool | None = None,
    p_adjust_method: str = "holm",
    digits_perc: int = 0,
    type: str = "parametric",
    conf_level: float = 0.95,
    digits: int = 2,
    alternative: str = "two-sided",
    bf_message: bool | None = None,
    results_subtitle: bool = True,
    seed: int = 0,
) -> StatsResult:
    """Count the categories of ``x`` within each level of ``y``, and test whether
    they divide the same way in every level; no plotting.

    The counterpart of ggstatsplot 1.1.4's ``ggbarstats()``: one stacked bar per
    level of ``y``, filled by the shares of the levels of ``x``. Rows with a
    missing ``x``, ``y`` or ``counts`` are left out (and counted in
    ``n_dropped``); n is the number of observations counted (pairs with
    ``paired=True``).

    The test follows the design (SPEC.md section 4); every type but "bayes" runs
    the same test, as in ggstatsplot:

    * ``y`` given: Pearson's chi-square test of independence (without Yates'
      correction) with Cramér's V (bias-corrected) and its interval; the
      proportion test of each bar (x's shares within the bar against ``ratio``);
      Fisher's exact test between every two levels of x, adjusted by
      ``p_adjust_method``, in ``result.pairwise`` (never drawn).
    * ``paired=True``: ``x`` and ``y`` are two measurements of each unit (one row
      per unit, or per combination with ``counts``): McNemar's test (Bowker's
      test of symmetry beyond two levels) with Cohen's g.
    * ``y=None``: the chi-square goodness-of-fit test of x's shares against
      ``ratio`` (equal shares by default) with Pearson's C.

    Args:
        data: a pandas, polars or pyarrow data frame.
        x: the column whose categories fill the bars (text, categorical, booleans
            or whole-number codes).
        y: the column whose categories make the bars; None for a single bar
            (the one-way test).
        counts: a column of whole numbers, how many observations each row stands
            for (data given as a table); None counts the rows.
        paired: x and y are two measurements of the same units (McNemar's test);
            they need the same categories in the same order.
        ratio: the expected share of each level of x, in their order, summing to
            1 (the one-way test, and the proportion test of each bar); None for
            equal shares. Never rescaled.
        label: what each cell's label shows: "percentage" (its share of the bar),
            "counts" or "both".
        proportion_test: label each bar with its proportion test's p; None
            follows ``results_subtitle``, as ggstatsplot's default does. Needs
            ``y``.
        p_adjust_method: adjustment of the pairwise Fisher tests: one of R's
            p.adjust() methods.
        digits_perc: decimals of a percentage label.
        type: "parametric" ("p") or "nonparametric" ("np"), the same tests;
            "robust" raises (no robust method for counts) and "bayes" arrives in
            inferplot 0.2.
        conf_level: confidence level of the effect-size intervals (the main one
            and the pairs'); the proportion tests of the bars are always 95%.
        digits: decimals in the subtitle and the bars' p labels.
        alternative: "two-sided", "less" or "greater": makes the effect-size
            intervals one-sided (and a pair's Fisher test, where its table is 2 x
            2); the chi-square p-value has no direction.
        bf_message: the Bayes factor caption arrives in inferplot 0.2; None notes
            its absence where ggstatsplot draws one, False leaves the note out,
            True raises.
        results_subtitle: compute the main test and the pairs (False: only the
            counts and the bars' proportion tests).
        seed: recorded in the result; no barstats test draws random numbers.

    Returns:
        A :class:`~inferplot.result.StatsResult` holding every number and text.

    A pair of levels of x without a Fisher test (both occur in one level of y
    only, or the exact test would take more than about two seconds) stays in
    ``result.pairwise`` with p None and ``no_test_reason``; the other pairs are
    adjusted among themselves, and ``result.warnings`` and ``result.check()``
    say so.

    Raises:
        TypeError, ValueError: a bad argument or unusable data (a measurement
            given as x or y, counts that are not whole, a single category, no
            unit that changes level in paired data); the message says how to
            fix the call.
        KeyError: a column that the data frame does not have.
        NotImplementedError: type "bayes" and ``bf_message=True`` arrive in
            inferplot 0.2.

    Example:
        >>> import pandas as pd
        >>> import inferplot as ip
        >>> df = pd.DataFrame({
        ...     "answer": ["yes", "no", "yes", "no"],
        ...     "group": ["a", "a", "b", "b"],
        ...     "n": [12, 8, 5, 15],
        ... })
        >>> result = ip.analyze.barstats(df, "answer", counts="n")
        >>> print(result.text.apa)
        Chi-square goodness-of-fit test (shares of answer against equal shares): chi2(1,
        N = 40) = 0.90, p = .343, Pearson's C = .15, 95% CI [.00, .42].
    """
    table, args, n_digits = _prepare(
        data,
        x,
        y,
        counts=counts,
        paired=paired,
        ratio=ratio,
        label=label,
        proportion_test=proportion_test,
        p_adjust_method=p_adjust_method,
        digits_perc=digits_perc,
        type=type,
        conf_level=conf_level,
        digits=digits,
        alternative=alternative,
        bf_message=bf_message,
        results_subtitle=results_subtitle,
        seed=seed,
    )
    parts = _ANALYSES[args.design](table, args)
    return _assemble(table, args, parts, digits=n_digits)
