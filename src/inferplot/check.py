"""Problems with a result or a figure, as :class:`Issue` records.

``result.check()`` returns the statistical issues (:func:`result_issues`);
``fig.check()`` adds the rendering issues (SPEC section 8). An empty tuple means no
problems were found.
"""

from __future__ import annotations

import math
from collections import Counter
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
        parts.append(
            f"{result.n_dropped} {rows} left out because {result.x!r} or "
            f"{result.y!r} is missing"
        )
    if result.dropped_subjects > 0:
        subjects = "subject was" if result.dropped_subjects == 1 else "subjects were"
        parts.append(
            f"{result.dropped_subjects} {subjects} left out because they lack a "
            "value in some condition"
        )
    return Issue(
        code="rows_dropped",
        severity="info",
        message="; ".join(parts) + ".",
        fix="Check that the missing values are expected; impute or remove them "
        "yourself to silence this.",
    )


def _pairs_hidden(result: StatsResult) -> Issue | None:
    hidden = [pair for pair in result.pairwise if not pair.displayed]
    if not hidden:
        return None
    total = len(result.pairwise)
    display = result.settings.pairwise_display
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

    Read from ``result.values``; a result read back without its observations
    (``to_json(include_values=False)``) cannot be checked for ties.
    """
    if result.type != "nonparametric" or not result.values:
        return None
    counts = Counter(v for level in result.values for v in level)
    total = sum(counts.values())
    tied = total - len(counts)
    if total == 0 or tied / total <= HEAVY_TIES:
        return None
    return Issue(
        code="heavy_ties",
        severity="warning",
        message=(
            f"{result.y!r} has only {len(counts)} distinct values among {total}, so "
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
    are kept, as in WRS2, so the estimate or a CI bound can be -Inf or Inf."""
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
        "effect size infinite (as WRS2 computes it).",
        fix=(
            "Report the effect size as unbounded, pass a smaller tr, or use "
            "type='nonparametric'."
        ),
    )


def _percent(level: float) -> str:
    return f"{100 * level:g}%"


def _bound(x: float) -> str:
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.2f}"


def result_issues(result: StatsResult) -> tuple[Issue, ...]:
    """The statistical issues of ``result``: small_n, heavy_ties,
    infinite_effect_size, rows_dropped, pairs_hidden, pairwise_conf_level."""
    found = (
        _small_n(result),
        _heavy_ties(result),
        _infinite_effect_size(result),
        _rows_dropped(result),
        _pairs_hidden(result),
        _pairwise_conf_level(result),
    )
    return tuple(issue for issue in found if issue is not None)
