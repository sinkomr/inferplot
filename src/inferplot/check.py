"""Problems with a result or a figure, as :class:`Issue` records.

``result.check()`` returns the statistical issues (:func:`result_issues`);
``fig.check()`` adds the rendering issues (SPEC section 8). An empty tuple means no
problems were found.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from inferplot.result import StatsResult

__all__ = [
    "HEAVY_TIES",
    "SMALL_EXPECTED",
    "SMALL_N",
    "Issue",
    "Severity",
    "expected_counts",
    "result_issues",
]

Severity = Literal["warning", "info"]

# A group with fewer observations than this gets a small_n warning.
SMALL_N: Final = 5
# rows_dropped is a warning when more than this share of the subjects (within) or
# of the rows (between) was left out.
_MOST_LEFT_OUT: Final = 0.5
# A rank-based test on data with more than this share of tied values gets a
# heavy_ties warning (SPEC section 8).
HEAVY_TIES: Final = 0.25


@dataclass(frozen=True, slots=True)
class Issue:
    """One problem: a stable ``code``, its ``severity``, what is wrong, the fix."""

    code: str
    severity: Severity
    message: str
    fix: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.message} {self.fix}"


def _small_n(result: StatsResult) -> Issue | None:
    if result.design == "contingency":
        return None  # counts: small_expected judges the expected counts instead
    if result.design in ("one-sample", "correlation"):
        return _small_sample(result)
    small = [d for d in result.descriptives if d.n < SMALL_N]
    if not small:
        return None
    if result.design == "within":
        # Every level holds the same complete subjects, so name them once.
        n = small[0].n
        return Issue(
            code="small_n",
            severity="warning",
            message=(
                f"Only {n} {'subject has' if n == 1 else 'subjects have'} a value at "
                f"every level of {result.x!r}, fewer than {SMALL_N}. Tests, "
                "intervals and plotted distributions are unreliable with so few "
                "subjects."
            ),
            fix="Collect more data, or report the result as exploratory.",
        )
    groups = ", ".join(f"{d.level!r} (n = {d.n})" for d in small)
    noun = "group has" if len(small) == 1 else "groups have"
    return Issue(
        code="small_n",
        severity="warning",
        message=(
            f"{len(small)} {noun} fewer than {SMALL_N} observations: {groups}. "
            "Tests, intervals and plotted distributions are unreliable with so few "
            "values."
        ),
        fix="Collect more data, or report the result as exploratory.",
    )


def _small_sample(result: StatsResult) -> Issue | None:
    """``small_n`` for the designs without groups: fewer than ``SMALL_N`` values
    (one-sample) or complete pairs (correlation), reported once."""
    n = result.n_obs
    if n >= SMALL_N:
        return None
    if result.design == "one-sample":
        what = f"{n} {'value' if n == 1 else 'values'} of {result.x!r}"
        unit = "values"
    else:
        what = (
            f"{n} complete {'pair' if n == 1 else 'pairs'} of {result.x!r} and "
            f"{result.y!r}"
        )
        unit = "pairs"
    # Type bayes reports a Bayes factor alone (D45): no test, no interval.
    if result.test is not None:
        reported = "The test and its interval are unreliable"
    elif result.bayes is not None:
        reported = "The Bayes factor can say little either way"
    else:
        reported = "Any test is unreliable"
    return Issue(
        code="small_n",
        severity="warning",
        message=(
            f"Only {what} {'was' if n == 1 else 'were'} analyzed, fewer than "
            f"{SMALL_N}. {reported} with so few {unit}."
        ),
        fix="Collect more data, or report the result as exploratory.",
    )


def _rows_dropped(result: StatsResult) -> Issue | None:
    if result.n_dropped <= 0 and result.dropped_subjects <= 0:
        return None
    parts: list[str] = []
    if result.n_dropped > 0:
        rows = "row was" if result.n_dropped == 1 else "rows were"
        if result.design == "contingency":
            columns = _either(result.x, result.y, result.counts_var)
        elif result.y is None:  # one-sample: a single column
            columns = repr(result.x)
        elif result.subject_id is None:
            columns = f"{result.x!r} or {result.y!r}"
        else:
            columns = f"{result.x!r}, {result.y!r} or {result.subject_id!r}"
        parts.append(f"{result.n_dropped} {rows} left out because {columns} is missing")
    if result.dropped_subjects > 0:
        one = result.dropped_subjects == 1
        rows = result.dropped_subject_rows
        their_rows = f" ({rows} {'row' if rows == 1 else 'rows'})" if rows else ""
        parts.append(
            f"{result.dropped_subjects} {'subject' if one else 'subjects'}"
            f"{their_rows} {'was' if one else 'were'} left out because "
            f"{'it lacks' if one else 'they lack'} a value at some level of "
            f"{result.x!r}"
        )
    # More than half of the subjects (within) or of the rows (between) left out:
    # the analysis describes a minority of the data, which a reader must know.
    if result.subject_id is not None:
        out, unit = result.dropped_subjects, "subjects"
        total = result.n_obs + out
    elif result.counts_var is not None:
        # A table of counts: its rows are cells, not observations, and a row
        # with a missing count stands for an unknown number of them.
        out, unit, total = 0, "rows", 0
    else:
        out, unit = result.n_dropped, "rows"
        total = result.n_obs + out
    share = out / total if total else 0.0
    if share > _MOST_LEFT_OUT:
        parts.append(
            f"that is {share:.0%} of the {total:,} {unit}, so the analysis describes "
            f"only the {result.n_obs:,} that are left"
        )
    return Issue(
        code="rows_dropped",
        severity="warning" if share > _MOST_LEFT_OUT else "info",
        message="; ".join(parts) + ".",
        fix="Check that the missing values are expected; impute or remove them "
        "yourself to silence this.",
    )


def _pairs_hidden(result: StatsResult) -> Issue | None:
    """Pairs computed but not drawn. Not for ``pairwise_display="none"``: that is
    never a default (the default is "significant"), so the setting itself says the
    caller asked for no brackets."""
    display = result.settings.pairwise_display
    if display == "none":
        return None
    hidden = [pair for pair in result.pairwise if not pair.displayed]
    if not hidden:
        return None
    total = len(result.pairwise)
    verb = "is" if len(hidden) == 1 else "are"
    return Issue(
        code="pairs_hidden",
        severity="info",
        message=(
            f"{len(hidden)} of {total} pairwise comparisons {verb} not drawn "
            f"(pairwise_display={display!r}); all of them are in result.pairwise."
        ),
        fix="Pass pairwise_display='all' to draw every pair.",
    )


def _heavy_ties(result: StatsResult) -> Issue | None:
    """Rank-based tests (type nonparametric) where more than ``HEAVY_TIES`` of
    what the test ranks are ties (DECISIONS.md D47): between groups, the values
    beyond the first of each distinct value (n minus the distinct values); the
    signed-rank test, the paired differences that are 0 (dropped) or repeat an
    earlier one's size (subjects minus distinct nonzero sizes); Friedman's test,
    the values tied with another of the same subject.

    Read from ``result.n_tied``, stored at analysis time
    (:func:`inferplot.result.paired_tied_count` for within designs), so the copy
    without observations (``to_json(include_values=False)``) gives the same
    answer.
    """
    if result.type != "nonparametric" or result.design == "contingency":
        return None  # barstats ranks nothing: nonparametric is the chi-square test
    if result.design == "within":
        return _paired_heavy_ties(result)
    if result.design == "one-sample":
        return _one_sample_heavy_ties(result)
    if result.design == "correlation":
        return _correlation_heavy_ties(result)
    total = sum(d.n for d in result.descriptives)
    tied = result.n_tied
    distinct = total - tied
    if total == 0 or tied / total <= HEAVY_TIES:
        return None
    return Issue(
        code="heavy_ties",
        severity="warning",
        message=(
            f"{result.y!r} has only {distinct} distinct values among {total}, so "
            f"{tied} of them ({tied / total:.0%}) are ties. "
            "The rank tests correct for ties, but with this many their normal "
            "approximations are coarse and the ranks carry less information."
        ),
        fix=(
            "Report the ties. If y takes only a few values (a rating or a count), "
            "consider analyzing it as categories with barstats."
        ),
    )


def _paired_heavy_ties(result: StatsResult) -> Issue | None:
    """``heavy_ties`` for the signed-rank test (two conditions) and Friedman's
    test (three or more), from ``n_tied`` (``paired_tied_count``)."""
    n, k, tied = result.n_obs, len(result.levels), result.n_tied
    total = n if k == 2 else n * k
    if total == 0 or tied / total <= HEAVY_TIES:
        return None
    share = f"{tied / total:.0%}"
    if k == 2:
        message = (
            f"{tied} of the {n} paired differences ({share}) are 0 or repeat "
            "another difference's size, so the signed-rank test drops the zero "
            f"ones and ranks only {n - tied} distinct sizes. Its normal "
            "approximation corrects for ties, but with this many the p-value is "
            "coarse and the ranks carry less information."
        )
    else:
        message = (
            f"{tied} of the {total} values ({share}) tie with another value of the "
            "same subject, so Friedman's test, which ranks each subject's values, "
            "gives many shared ranks. Its chi-square approximation corrects for "
            "ties, but with this many the p-value is coarse and the ranks carry "
            "less information."
        )
    return Issue(
        code="heavy_ties",
        severity="warning",
        message=message,
        fix=(
            "Report the ties. If y takes only a few values (a rating or a count), "
            "say so when you report the test."
        ),
    )


def _one_sample_heavy_ties(result: StatsResult) -> Issue | None:
    """``heavy_ties`` for the one-sample signed-rank test, which ranks the sizes
    of the nonzero x - test_value (``one_sample_tied_count``)."""
    n, tied = result.n_obs, result.n_tied
    if n == 0 or tied / n <= HEAVY_TIES:
        return None
    test_value = result.settings.test_value
    shown = "the test value" if test_value is None else f"{test_value:g}"
    return Issue(
        code="heavy_ties",
        severity="warning",
        message=(
            f"{tied} of the {n} values of {result.x!r} ({tied / n:.0%}) equal "
            f"{shown} or lie as far from it as another value, so the signed-rank "
            f"test drops the equal ones and ranks only {n - tied} distinct "
            "distances. Its normal approximation corrects for ties, but with this "
            "many the p-value is coarse and the ranks carry less information."
        ),
        fix=(
            "Report the ties. If x takes only a few values (a rating or a count), "
            "say so when you report the test."
        ),
    )


def _correlation_heavy_ties(result: StatsResult) -> Issue | None:
    """``heavy_ties`` for Spearman's rho, which ranks x and y separately
    (``correlation_tied_count``: ties within x plus ties within y, of 2 n)."""
    total, tied = 2 * result.n_obs, result.n_tied
    if total == 0 or tied / total <= HEAVY_TIES:
        return None
    return Issue(
        code="heavy_ties",
        severity="warning",
        message=(
            f"{tied} of the {total} values of {result.x!r} and {result.y!r} "
            f"({tied / total:.0%}) tie with another value of the same column, so "
            "Spearman's rho gives many values a shared (average) rank. With this "
            "many ties its t approximation and interval are coarse and the ranks "
            "carry less information."
        ),
        fix=(
            "Report the ties. If x or y takes only a few values (a rating or a "
            "count), say so when you report the correlation."
        ),
    )


def _pairwise_conf_level(result: StatsResult) -> Issue | None:
    """Pairwise intervals at another level than ``settings.conf_level`` (type
    robust: always 95%, as in ggstatsplot; DECISIONS.md D53)."""
    level = result.settings.conf_level
    levels = sorted(
        {
            pair.conf_level
            for pair in result.pairwise
            if pair.conf_level is not None and pair.conf_level != level
        }
    )
    if not levels:
        return None
    shown = ", ".join(f"{100 * v:g}%" for v in levels)
    es = result.effect_size
    if (
        result.design == "within"
        and es is not None
        and es.conf_level != level
        and {es.conf_level} == set(levels)
    ):
        # Type robust with conditions (D66): the effect size's interval and the
        # pairs' are both at the fixed level; conf_level changes neither.
        return Issue(
            code="pairwise_conf_level",
            severity="info",
            message=(
                f"Every interval of this result is a {shown} interval: the "
                f"{es.label}'s bootstrap interval and the pairwise intervals. "
                f"conf_level={level!r} changes none of them; ggstatsplot does not "
                "pass it to WRS2 here."
            ),
            fix=f"Report the intervals as {shown} intervals, or pass conf_level=0.95.",
        )
    return Issue(
        code="pairwise_conf_level",
        severity="info",
        message=(
            f"The pairwise confidence intervals are {shown} intervals, while "
            f"conf_level={level!r} applies to the effect size, as in ggstatsplot."
        ),
        fix="Report the pairwise intervals' own level, or pass conf_level=0.95.",
    )


def _infinite_effect_size(result: StatsResult) -> Issue | None:
    """An infinite effect size, or bootstrap resamples where it is infinite (type
    robust: a winsorized variance of 0 under trimmed means that differ). The values
    are kept (WRS2's formulas give the same values; no oracle fixture has one), so
    the estimate or a CI bound can be -Inf or Inf."""
    es = result.effect_size
    if es is None:
        return None
    infinite = es.infinite_resamples or 0
    estimate_infinite = math.isinf(es.estimate)
    if not infinite and not estimate_infinite:
        return None
    parts: list[str] = []
    if estimate_infinite:
        parts.append(f"{_named(es.label)} is infinite.")
    one_sample = result.design == "one-sample"
    if infinite:
        total = es.bootstrap_resamples or 0
        interval = (
            f"{_percent(es.conf_level)} CI [{_bound(es.conf_low)}, "
            f"{_bound(es.conf_high)}] may be unbounded"
        )
        if one_sample:
            # The bootstrap-t test of the trimmed mean: a resample whose
            # winsorized values are all equal has a standard error of 0. The
            # interval is unbounded when its critical value falls among them.
            unbounded = math.isinf(es.conf_low) or math.isinf(es.conf_high)
            shown = (
                f"{_percent(es.conf_level)} CI [{_bound(es.conf_low)}, "
                f"{_bound(es.conf_high)}]"
            )
            parts.append(
                f"{infinite:,} of {total:,} bootstrap resamples have a standard "
                "error of 0 and an infinite bootstrap t; they are kept, so the "
                f"{es.label}'s {shown} "
                + (
                    "is unbounded."
                    if unbounded
                    else "would be unbounded at a conf_level high enough to reach them."
                )
            )
        elif es.name == "akp_delta_avg":
            # Counted per pair, in resamples: the most of any one pair.
            parts.append(
                f"In up to {infinite:,} of the {total:,} bootstrap resamples a "
                "pair's AKP is infinite (the most of any pair), so the "
                f"{es.label}'s {interval}."
            )
        else:
            parts.append(
                f"{infinite:,} of {total:,} bootstrap resamples give an infinite "
                f"{es.label}, so its {interval}."
            )
    cause = (
        " A resample whose winsorized values are all equal has a winsorized "
        "variance of 0."
        if one_sample
        else " A winsorized variance of 0 next to trimmed means that differ makes "
        "the effect size infinite."
    )
    fix = (
        # A one-sample robust result has values that vary (else it is refused),
        # so the signed-rank test can run; a smaller tr does not remove resamples
        # whose values are all one number.
        "Report the interval as it is, unbounded where it is; collect more values, "
        "or use type='nonparametric', which does not resample."
        if one_sample
        else "Report the effect size as unbounded, pass a smaller tr, or use "
        "type='nonparametric'."
    )
    return Issue(
        code="infinite_effect_size",
        severity="warning",
        message=" ".join(parts) + cause,
        fix=fix,
    )


def _estimate_outside_interval(result: StatsResult) -> Issue | None:
    """An effect size outside its own CI. Percentile bootstrap intervals (rank
    epsilon squared, the explanatory effect size xi) are read off the resampled
    values, which a small effect's bootstrap bias can shift past the estimate
    (12 groups of 6 noise values: epsilon squared 0.074 with [0.107, 1]). Read
    from stored fields only, so the copy without values reports it too."""
    es = result.effect_size
    if es is None:
        return None
    numbers = (es.estimate, es.conf_low, es.conf_high)
    if any(math.isnan(x) for x in numbers):
        return None
    if es.conf_low <= es.estimate <= es.conf_high:
        return None
    # Outside by less than the three decimals the message prints (a bound that is
    # the estimate up to its last digits): nothing a reader could see or act on.
    crossed = es.conf_low if es.estimate < es.conf_low else es.conf_high
    if _three(es.estimate) == _three(crossed):
        return None
    level = _percent(es.conf_level)
    head = (
        f"{_named(es.label)}, {_three(es.estimate)}, lies outside its own {level} "
        f"CI [{_three(es.conf_low)}, {_three(es.conf_high)}]."
    )
    if es.ci_method == "percentile bootstrap":
        between = {"within": "conditions", "between": "groups"}.get(result.design)
        larger = (
            "resampling adds chance variation"
            if between is None
            else f"resampling adds chance differences between the {between}"
        )
        shift = (
            f"{larger}, so the resampled effect sizes run larger than the estimate"
            if es.estimate < es.conf_low
            else "the resampled effect sizes run smaller than the estimate"
        )
        why = (
            " The CI is a percentile bootstrap interval, read off the effect sizes "
            "of resampled data. When the effect is small next to the bootstrap's "
            f"own bias ({shift}), the whole interval can lie on one side of the "
            "estimate."
        )
    else:
        why = f" The CI ({es.ci_method}) should contain its estimate."
    return Issue(
        code="estimate_outside_interval",
        severity="warning",
        message=head + why,
        fix=(
            "Report the estimate, and the interval only as a rough range of "
            "plausible values (say that it does not contain the estimate); the "
            "test's p-value does not come from the bootstrap and is unaffected."
        ),
    )


def _named(label: str) -> str:
    """An effect size's name at the start of a sentence: "Kendall's W" and
    "Hedges' g" as they are (a name with an owner takes no article), "The partial
    omega squared"."""
    first = label.split(" ", 1)[0]
    if first.endswith(("'s", "'")):
        return label
    return f"The {label}"


def _three(x: float) -> str:
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.3f}"


def _percent(level: float) -> str:
    return f"{100 * level:g}%"


def _bound(x: float) -> str:
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.2f}"


def _either(*names: str | None) -> str:
    """``'a'``, ``'a' or 'b'``, ``'a', 'b' or 'c'`` of the names given."""
    quoted = [repr(name) for name in names if name is not None]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


# --------------------------------------------------------------------------- #
# small_expected: the chi-square approximations of barstats (design contingency)
# --------------------------------------------------------------------------- #

# Cochran's (1954) rule for a chi-square approximation: no expected count below
# MIN_EXPECTED, and at most SMALL_EXPECTED_SHARE of them below SMALL_EXPECTED.
SMALL_EXPECTED: Final = 5.0
MIN_EXPECTED: Final = 1.0
SMALL_EXPECTED_SHARE: Final = 0.2


def _shares(result: StatsResult) -> list[float]:
    """The expected share of each level of x in a goodness-of-fit test:
    ``settings.ratio``, else equal shares."""
    k = len(result.levels)
    ratio = result.settings.ratio
    return list(ratio) if ratio is not None else [1.0 / k] * k


def expected_counts(result: StatsResult) -> list[float]:
    """The expected counts behind a barstats result's main test (design
    "contingency"): one-way, n times each level's share (``settings.ratio``,
    else equal; every level kept); two-way, row total times column total over
    n for the rows and columns that hold observations (R's ``chisq.test`` drops
    the others); paired, half the discordant pairs of each pair of levels that
    has any ((n_ij + n_ji) / 2, what McNemar's and Bowker's tests compare each
    count with)."""
    counts = result.counts
    n = result.n_obs
    if not result.y_levels:
        return [n * share for share in _shares(result)]
    if result.settings.paired:
        k = len(counts)
        out: list[float] = []
        for i in range(k):
            for j in range(i + 1, k):
                discordant = counts[i][j] + counts[j][i]
                if discordant:
                    out.extend([discordant / 2] * 2)
        return out
    rows = [sum(row) for row in counts]
    columns = [sum(column) for column in zip(*counts, strict=True)]
    return [r * c / n for r in rows if r for c in columns if c]


def _fails_cochran(expected: list[float]) -> bool:
    if not expected:
        return False
    small = sum(e < SMALL_EXPECTED for e in expected)
    return min(expected) < MIN_EXPECTED or small > SMALL_EXPECTED_SHARE * len(expected)


def _small_expected(result: StatsResult) -> Issue | None:
    """Expected counts too small for the chi-square approximation of a barstats
    p-value (Cochran's rule: none below 1 and at most 20% below 5), in the main
    test or, where the figure labels them, a bar's proportion test. R's
    ``chisq.test`` warns ("Chi-squared approximation may be incorrect") and
    ggstatsplot mutes the warning; inferplot computes no exact test."""
    if result.design != "contingency":
        return None
    parts: list[str] = []
    main = expected_counts(result) if result.test is not None else []
    if _fails_cochran(main):
        small = sum(e < SMALL_EXPECTED for e in main)
        tiny = sum(e < MIN_EXPECTED for e in main)
        if result.settings.paired:
            test = "McNemar's test (half the pairs that change between two levels)"
        elif result.y_levels:
            test = "the chi-square test of independence"
        else:
            test = "the goodness-of-fit test"
        below_one = f", {tiny} below 1" if tiny else ""
        parts.append(
            f"{small} of the {len(main)} expected counts of {test} are below 5 (the "
            f"smallest is {min(main):.2f}{below_one}), so its p-value, a chi-square "
            "approximation, may be inaccurate"
        )
    if result.settings.proportion_test:
        shares = _shares(result)
        bars = [
            bar.level
            for bar in result.bars
            if bar.test is not None
            and bar.level is not None
            and _fails_cochran([bar.n * share for share in shares])
        ]
        if bars:
            one = len(bars) == 1
            names = ", ".join(repr(bar) for bar in bars)
            tail = (
                "too"
                if parts
                else f"so {'its p-value' if one else 'their p-values'} may be "
                "inaccurate"
            )
            parts.append(
                f"the proportion {'test of the bar' if one else 'tests of the bars'} "
                f"{names} {'has' if one else 'have'} expected counts below 5, {tail}"
            )
    if not parts:
        return None
    message = "; ".join(parts)
    return Issue(
        code="small_expected",
        severity="warning",
        message=(
            message[0].upper()
            + message[1:]
            + ". R's chisq.test warns about this and ggstatsplot does not pass the "
            "warning on."
        ),
        fix=(
            "Report the p-values as approximate, combine sparse categories, or "
            "collect more data."
        ),
    )


def result_issues(result: StatsResult) -> tuple[Issue, ...]:
    """The statistical issues of ``result``: small_n, small_expected, heavy_ties,
    infinite_effect_size, estimate_outside_interval, rows_dropped, pairs_hidden,
    pairwise_conf_level."""
    found = (
        _small_n(result),
        _small_expected(result),
        _heavy_ties(result),
        _infinite_effect_size(result),
        _estimate_outside_interval(result),
        _rows_dropped(result),
        _pairs_hidden(result),
        _pairwise_conf_level(result),
    )
    return tuple(issue for issue in found if issue is not None)
