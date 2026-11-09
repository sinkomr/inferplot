"""Problems with a result or a figure, as :class:`Issue` records.

``result.check()`` returns the statistical issues (:func:`result_issues`);
``fig.check()`` adds the rendering issues (SPEC section 8). An empty tuple means no
problems were found.
"""

from __future__ import annotations

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
    ``HEAVY_TIES`` of the values share their value with another observation.

    Read from ``result.values``; a result read back without its observations
    (``to_json(include_values=False)``) cannot be checked for ties.
    """
    if result.type != "nonparametric" or not result.values:
        return None
    counts = Counter(v for level in result.values for v in level)
    total = sum(counts.values())
    tied = sum(c for c in counts.values() if c > 1)
    if total == 0 or tied / total <= HEAVY_TIES:
        return None
    return Issue(
        code="heavy_ties",
        severity="warning",
        message=(
            f"{tied} of {total} values of {result.y!r} ({tied / total:.0%}) share "
            f"their value with another observation ({len(counts)} distinct values). "
            "The rank tests correct for ties, but with this many their normal "
            "approximations are coarse and the ranks carry less information."
        ),
        fix=(
            "Report the ties. If y takes only a few values (a rating or a count), "
            "consider analyzing it as categories with barstats."
        ),
    )


def result_issues(result: StatsResult) -> tuple[Issue, ...]:
    """The statistical issues of ``result``: small_n, heavy_ties, rows_dropped,
    pairs_hidden."""
    found = (
        _small_n(result),
        _heavy_ties(result),
        _rows_dropped(result),
        _pairs_hidden(result),
    )
    return tuple(issue for issue in found if issue is not None)
