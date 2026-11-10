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

__all__ = ["HEAVY_TIES", "SMALL_N", "Issue", "Severity", "result_issues"]

Severity = Literal["warning", "info"]

# A group with fewer observations than this gets a small_n warning.
SMALL_N: Final = 5
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


def _rows_dropped(result: StatsResult) -> Issue | None:
    if result.n_dropped <= 0 and result.dropped_subjects <= 0:
        return None
    parts: list[str] = []
    if result.n_dropped > 0:
        rows = "row was" if result.n_dropped == 1 else "rows were"
        columns = (
            f"{result.x!r} or {result.y!r}"
            if result.subject_id is None
            else f"{result.x!r}, {result.y!r} or {result.subject_id!r}"
        )
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
    return Issue(
        code="rows_dropped",
        severity="info",
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
    """Rank-based tests (type nonparametric) on data where more than
    ``HEAVY_TIES`` of the values are ties: values beyond the first of each
    distinct value, n minus the number of distinct values (DECISIONS.md D47).

    Read from ``result.n_tied``, stored at analysis time, so the copy without
    observations (``to_json(include_values=False)``) gives the same answer. The
    total is the sum of the levels' n (within designs: complete subjects times
    levels). Every level's values are pooled, also in within designs, where the
    signed-rank and Friedman tests rank differences or within-subject values
    instead.
    """
    if result.type != "nonparametric":
        return None
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
        parts.append(f"The {es.label} is infinite.")
    if infinite:
        total = es.bootstrap_resamples or 0
        parts.append(
            f"{infinite:,} of {total:,} bootstrap resamples give an infinite "
            f"{es.label}, so its {_percent(es.conf_level)} CI "
            f"[{_bound(es.conf_low)}, {_bound(es.conf_high)}] may be unbounded."
        )
    return Issue(
        code="infinite_effect_size",
        severity="warning",
        message=" ".join(parts)
        + " A winsorized variance of 0 next to trimmed means that differ makes the "
        "effect size infinite.",
        fix=(
            "Report the effect size as unbounded, pass a smaller tr, or use "
            "type='nonparametric'."
        ),
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
    level = _percent(es.conf_level)
    head = (
        f"The {es.label}, {_three(es.estimate)}, lies outside its own {level} CI "
        f"[{_three(es.conf_low)}, {_three(es.conf_high)}]."
    )
    if es.bootstrap_resamples is not None:
        shift = (
            "resampling adds chance differences between the groups, so the "
            "resampled effect sizes run larger than the estimate"
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


def result_issues(result: StatsResult) -> tuple[Issue, ...]:
    """The statistical issues of ``result``: small_n, heavy_ties,
    infinite_effect_size, estimate_outside_interval, rows_dropped, pairs_hidden,
    pairwise_conf_level."""
    found = (
        _small_n(result),
        _heavy_ties(result),
        _infinite_effect_size(result),
        _estimate_outside_interval(result),
        _rows_dropped(result),
        _pairs_hidden(result),
        _pairwise_conf_level(result),
    )
    return tuple(issue for issue in found if issue is not None)
