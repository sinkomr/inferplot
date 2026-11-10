"""rows_dropped stays "info" and becomes "warning" when more than half of the
subjects (withinstats) or of the rows (betweenstats) were left out, saying the
share (review of phase 3, finding 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import inferplot as ip


def _within(complete: int, partial: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        (f"s{i:02d}", level, float(rng.normal()))
        for i in range(complete)
        for level in ("a", "b", "c")
    ]
    rows += [(f"p{i:02d}", "a", float(rng.normal())) for i in range(partial)]
    return pd.DataFrame(rows, columns=["id", "cond", "y"])


def _issue(result: ip.StatsResult) -> ip.Issue:
    (issue,) = [c for c in result.check() if c.code == "rows_dropped"]
    return issue


def test_within_most_subjects_left_out_is_a_warning() -> None:
    result = ip.analyze.withinstats(_within(4, 5), "cond", "y", subject_id="id")
    issue = _issue(result)
    assert issue.severity == "warning"
    assert issue.message == (
        "5 subjects (5 rows) were left out because they lack a value at some level "
        "of 'cond'; that is 56% of the 9 subjects, so the analysis describes only "
        "the 4 that are left."
    )
    half = ip.analyze.withinstats(_within(5, 5), "cond", "y", subject_id="id")
    assert _issue(half).severity == "info"
    assert "%" not in _issue(half).message


def test_between_most_rows_left_out_is_a_warning() -> None:
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] + [np.nan] * 7
    data = pd.DataFrame({"g": ["a", "b"] * 6 + ["a"], "y": y})
    issue = _issue(ip.analyze.betweenstats(data, "g", "y"))
    assert issue.severity == "warning"
    assert issue.message.endswith(
        "that is 54% of the 13 rows, so the analysis describes only the 6 that are "
        "left."
    )
    few = pd.DataFrame(
        {"g": ["a", "b"] * 6, "y": [1.0, 2.0, 3.0, 5.0, 4.0, 7.0] * 2}
    ).assign(y=lambda d: d["y"].where(d.index < 10))
    assert _issue(ip.analyze.betweenstats(few, "g", "y")).severity == "info"
