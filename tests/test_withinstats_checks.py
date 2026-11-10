"""The result checks of within-subjects designs (SPEC.md section 8).

``heavy_ties`` counts the ties among what the rank test ranks: the absolute
paired differences of the signed-rank test (a zero difference, which the test
drops, and each repeat of a size count as tied) and each subject's values in
Friedman's test, not every value pooled. ``small_n`` and ``rows_dropped`` speak
of complete subjects. Every check reads stored fields, so the copy embedded in a
saved figure gives the same list (D69).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot import check
from inferplot.result import StatsResult, paired_tied_count
from tests import within_pb_cases as wc

_within: Any = ip.analyze.withinstats


def _long(columns: dict[str, list[float]]) -> pd.DataFrame:
    rows = [
        {"subject": f"s{i:02d}", "condition": level, "score": value}
        for level, values in columns.items()
        for i, value in enumerate(values)
    ]
    return pd.DataFrame(rows)


def _np(columns: dict[str, list[float]]) -> StatsResult:
    return _within(
        _long(columns), "condition", "score", subject_id="subject", type="np"
    )


def _codes(result: StatsResult) -> list[str]:
    return [issue.code for issue in result.check()]


def _same_without_values(result: StatsResult) -> None:
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert embedded.n_tied == result.n_tied
    assert embedded.check() == result.check()


def test_paired_tied_count() -> None:
    # |differences| 1, 1, 0, 2, 3: one zero and one repeat of 1 -> 5 - 3 = 2.
    assert paired_tied_count([[1, 2, 3, 4, 5], [2, 1, 3, 6, 2]]) == 2
    # Within subjects: (1, 1, 2) has one tie, (3, 4, 5) none, (7, 7, 7) two.
    assert paired_tied_count([[1, 3, 7], [1, 4, 7], [2, 5, 7]]) == 3
    with pytest.raises(ValueError, match="2 or more levels"):
        paired_tied_count([[1.0, 2.0]])


def test_signed_rank_ties_are_the_paired_differences() -> None:
    # Likert-like values (pooled: 6 of 20 repeat, 30%), but every paired
    # difference has its own size: the signed-rank test sees no tie.
    pre = [1.0, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    post = [p + 0.1 * (i + 1) for i, p in enumerate(pre)]
    distinct = _np({"pre": pre, "post": post})
    assert distinct.n_tied == 0
    assert "heavy_ties" not in _codes(distinct)
    # Distinct values (pooled: no tie), but 6 of 10 differences are exactly 1.0
    # and one is 0: 10 - 4 = 6 of the 10 paired differences are tied.
    pre = [1.5, 2.25, 3.75, 4.5, 5.25, 6.5, 7.75, 8.5, 9.25, 10.5]
    shift = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, -3.0, 4.0]
    post = [p + s for p, s in zip(pre, shift, strict=True)]
    tied = _np({"pre": pre, "post": post})
    assert tied.n_tied == 6
    (issue,) = [i for i in tied.check() if i.code == "heavy_ties"]
    assert issue.message.startswith(
        "6 of the 10 paired differences (60%) are 0 or repeat another difference's "
        "size, so the signed-rank test drops the zero ones and ranks only 4 "
        "distinct sizes."
    )
    _same_without_values(tied)


def test_friedman_ties_are_within_subjects() -> None:
    # Every subject ranks 1, 2, 3 in turn: pooled, 27 of 30 values repeat, but no
    # subject has a tie.
    rows = [[1.0, 2, 3], [2.0, 3, 1], [3.0, 1, 2]] * 3 + [[1.0, 3, 2]]
    cycled = _np({c: [r[j] for r in rows] for j, c in enumerate("abc")})
    assert cycled.n_tied == 0
    assert "heavy_ties" not in _codes(cycled)
    # Distinct values pooled per subject, but each subject repeats one value at
    # two conditions: 10 of the 30 values tie within their subject (33%).
    patterns = ([0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0])
    rows = [[i + v for v in patterns[i % 3]] for i in range(10)]
    paired = _np({c: [r[j] for r in rows] for j, c in enumerate("abc")})
    assert paired.n_tied == 10
    (issue,) = [i for i in paired.check() if i.code == "heavy_ties"]
    assert issue.message.startswith(
        "10 of the 30 values (33%) tie with another value of the same subject, so "
        "Friedman's test, which ranks each subject's values, gives many shared "
        "ranks."
    )
    _same_without_values(paired)


def test_heavy_ties_threshold_is_more_than_a_quarter() -> None:
    assert check.HEAVY_TIES == 0.25
    # 8 subjects: 2 tied differences (25%) do not fire, 3 (37.5%) do.
    pre = [1.5, 2.25, 3.75, 4.5, 5.25, 6.5, 7.75, 8.5]
    for shift, fires in (
        ([1.0, 1.0, 1.0, 2.5, -3.5, 4.5, 5.5, 6.25], False),
        ([1.0, 1.0, 1.0, 1.0, -3.5, 4.5, 5.5, 6.25], True),
    ):
        post = [p + s for p, s in zip(pre, shift, strict=True)]
        result = _np({"pre": pre, "post": post})
        assert ("heavy_ties" in _codes(result)) == fires


def test_only_type_nonparametric() -> None:
    pre = [1.5, 2.25, 3.75, 4.5, 5.25, 6.5, 7.75, 8.5, 9.25, 10.5]
    shift = [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 2.0, -3.0, 4.0, 1.5]
    post = [p + s for p, s in zip(pre, shift, strict=True)]
    frame = _long({"pre": pre, "post": post})
    assert "heavy_ties" in _codes(
        _within(frame, "condition", "score", subject_id="subject", type="np")
    )
    for kind in ("parametric", "robust", "bayes"):
        result = _within(frame, "condition", "score", subject_id="subject", type=kind)
        assert "heavy_ties" not in _codes(result)


# The nonparametric ggwithinstats fixtures where more than a quarter of what the
# test ranks is tied: edge_within2_ties (75% of the paired differences are 0 or a
# repeated size) and edge_within2_large_n (45%: values with few decimals). Pooling
# every value instead fired on bugs_long (94% pooled: a 0-10 rating, but only 23%
# of the values tie within their subject) and edge_within_ties (89% pooled, 18%
# within subjects), and missed edge_within2_large_n (23% pooled).
HEAVY_TIES_FIXTURES = frozenset(
    {
        "ggwithinstats-edge_within2_large_n-nonparametric",
        "ggwithinstats-edge_within2_ties-nonparametric",
    }
)


def test_heavy_ties_on_the_fixtures() -> None:
    fired = {
        case.name
        for case in wc.cases_of("nonparametric")
        if "heavy_ties" in _codes(wc.analyze(case.name, "pandas"))
    }
    assert fired == HEAVY_TIES_FIXTURES


def test_small_n_and_rows_dropped_read_right_end_to_end() -> None:
    """The within wording of small_n (complete subjects, said once) and
    rows_dropped (rows missing a value, then whole subjects with their rows), on
    the fixtures' own data."""
    small = wc.analyze("ggwithinstats-edge_within_small-parametric", "pandas")
    (issue,) = [i for i in small.check() if i.code == "small_n"]
    assert str(issue) == (
        "[warning] small_n: Only 4 subjects have a value at every level of "
        "'condition', fewer than 5. Tests, intervals and plotted distributions are "
        "unreliable with so few subjects. Collect more data, or report the result "
        "as exploratory."
    )
    bugs = wc.analyze("ggwithinstats-bugs_long-parametric", "pandas")
    (issue,) = [i for i in bugs.check() if i.code == "rows_dropped"]
    assert issue.message == (
        "5 rows were left out because 'condition', 'desire' or 'subject' is "
        "missing; 5 subjects (15 rows) were left out because they lack a value at "
        "some level of 'condition'."
    )
    # The rows read add up: complete subjects x levels + dropped rows + the rows
    # of the dropped subjects.
    assert bugs.n_obs * 4 + bugs.n_dropped + bugs.dropped_subject_rows == 372
    one = wc.analyze("ggwithinstats-edge_within_missing-parametric", "pandas")
    (issue,) = [i for i in one.check() if i.code == "rows_dropped"]
    assert issue.message == (
        "1 subject (2 rows) was left out because it lacks a value at some level of "
        "'condition'."
    )
