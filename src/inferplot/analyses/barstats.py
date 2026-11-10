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
(``_prepare``); ``_assemble`` builds the result from its parts. The statistics
are not in this version yet: every design raises ``NotImplementedError`` naming
itself.
"""

from __future__ import annotations

import math
import numbers
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from typing import Final, Literal, cast

from inferplot import stats
from inferplot._data import TableData, TableLevels, read_table, table_notes
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
from inferplot.text import contingency_texts

__all__ = ["barstats"]

Design = Literal["two-way", "paired", "one-way"]

# ggstatsplot's defaults of the arguments that can have no effect.
_DEFAULT_CONF_LEVEL: Final = 0.95
_DEFAULT_P_ADJUST: Final = "holm"
_DEFAULT_DIGITS_PERC: Final = 0
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

# The methods of each design, for the messages of the designs not implemented yet.
_METHODS: Final[dict[Design, str]] = {
    "two-way": "Pearson's chi-square test of independence with Cramér's V",
    "paired": "McNemar's test with Cohen's g",
    "one-way": "the chi-square goodness-of-fit test with Pearson's C",
}


def _shape(table: TableData) -> str:
    rows = len(table.x_levels.levels)
    if table.y_levels is None:
        return f"{rows} categories of x={table.x!r}"
    columns = len(table.y_levels.levels)
    return f"a {rows} x {columns} table of x={table.x!r} by y={table.y!r}"


def _not_implemented(table: TableData, args: _Args) -> NotImplementedError:
    """The error of a design whose statistics are not in this version yet
    (CLAUDE.md rule 3: no fallback to another method)."""
    unit = "pairs" if args.paired else "observations"
    return NotImplementedError(
        f"barstats, {args.design} design ({_METHODS[args.design]}), is not "
        "implemented yet in this version of inferplot; it arrives with the Phase 4 "
        f"barstats statistics. The data were read: {_shape(table)}, {table.n:,} "
        f"{unit}."
    )


def _two_way(table: TableData, args: _Args) -> _Parts:
    """Pearson's chi-square test of independence (no Yates' correction) with
    Cramér's V (Bergsma's bias correction, noncentral chi-square interval), the
    proportion test of each bar, and Fisher's exact test between every two
    levels of x where ggstatsplot computes them."""
    raise _not_implemented(table, args)


def _paired(table: TableData, args: _Args) -> _Parts:
    """McNemar's test (Bowker's beyond 2 x 2) without continuity correction with
    Cohen's g (Wilson interval), and the proportion test of each bar."""
    raise _not_implemented(table, args)


def _one_way(table: TableData, args: _Args) -> _Parts:
    """The chi-square goodness-of-fit test against ``ratio`` with Pearson's C
    (noncentral chi-square interval)."""
    raise _not_implemented(table, args)


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
            rest = (
                f"; {'it stays a level' if one else 'they stay levels'} of both "
                "measurements in McNemar's test, with a count of 0."
            )
        elif role == "x":
            rest = (
                f"; {'it stays' if one else 'they stay'} in the legend with no share "
                "of any bar, and the chi-square test leaves the empty "
                f"{'row' if one else 'rows'} out (as ggstatsplot does)."
            )
        else:
            rest = (
                f", so {'it has no bar' if one else 'they have no bars'}, and the "
                f"chi-square test leaves the empty {'column' if one else 'columns'} "
                "out (as ggstatsplot does)."
            )
        notes.append(start + rest)
    return notes


def _no_effect_warnings(table: TableData, args: _Args) -> list[str]:
    """One plain-English entry per argument that has no effect on these data, or
    a narrower one than the caller may expect (D39), naming the value as the
    caller gave it."""
    out: list[str] = []
    passed = args.passed
    if args.p_adjust_method != _DEFAULT_P_ADJUST and not _pairs_wanted(table, args):
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
        out.append(
            f"p_adjust_method={passed.get('p_adjust_method')!r} has no effect: it "
            "adjusts the pairwise Fisher tests between the levels of x, and "
            f"{why}."
        )
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
        out.append(
            "The proportion test of each bar reports Pearson's C with a 95% "
            f"interval whatever conf_level says (as in ggstatsplot): "
            f"conf_level={level!r} applies to the main effect size and the pairs "
            "only."
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
        out.append(_alternative_note(args))
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


def _alternative_note(args: _Args) -> str:
    """What a one-sided ``alternative`` changes in barstats: the effect-size
    intervals and the Fisher tests of 2 x 2 pair tables, never the chi-square
    p-value (ggstatsplot passes it to effectsize and fisher.test only)."""
    side = args.alternative
    if not args.results_subtitle:
        return (
            f"alternative={side!r} has no effect: results_subtitle=False computes no "
            "interval and no pairwise test, and the proportion test of each bar is "
            "always two-sided."
        )
    bound = "[low, 1]" if side == "greater" else "[0, high]"
    effect = {
        "two-way": f"Cramér's V interval one-sided, {bound}, also for each pair,",
        "paired": "Cohen's g interval one-sided",
        "one-way": f"Pearson's C interval one-sided, {bound}",
    }[args.design]
    test = {
        "two-way": "the chi-square test",
        "paired": "McNemar's test",
        "one-way": "the goodness-of-fit test",
    }[args.design]
    pairs = (
        " and makes the Fisher test of each pair whose table is 2 x 2 one-sided"
        if args.design == "two-way"
        else ""
    )
    return (
        f"alternative={side!r} makes the {effect}{pairs}; {test} has no direction, "
        "so its p-value is the same, as are the proportion tests of the bars "
        "(as in ggstatsplot)."
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
        *_no_effect_warnings(table, args),
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
    import numpy as np

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
        raise ValueError(
            f"{name} must hold positive numbers, the expected share of each level "
            f"of x ({example}); got {bad[0]!r}."
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
    of y with observations (one cell) with y, fewer than two levels paired, and
    a ``ratio`` of the wrong length."""
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
    side = canonical_alternative(check_alternative(alternative))
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

    Raises:
        TypeError, ValueError: a bad argument or unusable data (a measurement
            given as x or y, counts that are not whole, a single category); the
            message says how to fix the call.
        KeyError: a column that the data frame does not have.
        NotImplementedError: the statistics of each design are not in this
            version yet; type "bayes" and ``bf_message=True`` arrive in 0.2.
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
