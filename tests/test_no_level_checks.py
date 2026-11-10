"""``check()`` on real histostats and scatterstats results (SPEC.md section 8):
each check's trigger and wording on the designs without levels.

Over every fixture the statistical checks that fire are ``heavy_ties`` (type
nonparametric with many ties), ``rows_dropped`` (missing values) and
``infinite_effect_size`` (two robust histostats cases); ``estimate_outside_interval``
never fires (a bootstrap-t interval is symmetric about its estimate, and the
Fisher z intervals contain theirs), and neither does ``small_n`` (the smallest
fixture has 5 values or pairs).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import inferplot as ip
from inferplot.result import StatsResult
from tests import no_level_cases as nl

H: Any = ip.analyze.histostats
S: Any = ip.analyze.scatterstats


def _codes(result: StatsResult) -> list[str]:
    return [issue.code for issue in result.check()]


def test_checks_over_every_fixture() -> None:
    fired: dict[str, set[str]] = {}
    for case in (*nl.HISTO, *nl.SCATTER):
        result = nl.analyzed(case.name)
        if result is None:
            continue
        for code in _codes(result):
            fired.setdefault(code, set()).add(case.name.split("-", 1)[1])
    assert set(fired) == {"heavy_ties", "rows_dropped", "infinite_effect_size"}
    assert fired["infinite_effect_size"] == {
        "edge_one_small-robust",
        "edge_one_ties-robust",
    }
    # heavy_ties only for the rank tests.
    assert all(name.endswith("nonparametric") for name in fired["heavy_ties"])
    assert {n.rsplit("-", 1)[0] for n in fired["rows_dropped"]} == {
        "bugs_long",
        "edge_one_missing",
        "edge_corr_missing",
    }


@pytest.mark.parametrize("type_", ["parametric", "nonparametric", "robust", "bayes"])
def test_small_n_one_sample(type_: str) -> None:
    four = pd.DataFrame({"v": [1.0, 2.0, 4.0, 7.0]})
    (issue,) = [i for i in H(four, "v", type=type_).check() if i.code == "small_n"]
    reported = (
        "The Bayes factor can say little either way"
        if type_ == "bayes"
        else "The test and its interval are unreliable"
    )
    assert issue.message == (
        f"Only 4 values of 'v' were analyzed, fewer than 5. {reported} with so few "
        "values."
    )
    assert issue.severity == "warning"
    five = pd.DataFrame({"v": [1.0, 2.0, 4.0, 7.0, 3.0]})
    assert "small_n" not in _codes(H(five, "v", type=type_))
    nothing = H(four, "v", type="bayes", results_subtitle=False)
    assert nothing.check()[0].message.endswith(
        "Any test is unreliable with so few values."
    )


@pytest.mark.parametrize("type_", ["parametric", "nonparametric", "robust", "bayes"])
def test_small_n_correlation(type_: str) -> None:
    four = pd.DataFrame({"a": [1.0, 2, 4, 3], "b": [2.0, 1, 5, 3]})
    (issue,) = [i for i in S(four, "a", "b", type=type_).check() if i.code == "small_n"]
    assert issue.message.startswith(
        "Only 4 complete pairs of 'a' and 'b' were analyzed, fewer than 5."
    )
    assert issue.message.endswith("with so few pairs.")
    five = pd.DataFrame({"a": [1.0, 2, 4, 3, 5], "b": [2.0, 1, 5, 3, 4]})
    assert "small_n" not in _codes(S(five, "a", "b", type=type_))


def test_rows_dropped() -> None:
    result = nl.analyzed("gghistostats-edge_one_missing-parametric")
    assert result is not None
    (issue,) = [i for i in result.check() if i.code == "rows_dropped"]
    assert (issue.severity, issue.message) == (
        "info",
        "2 rows were left out because 'value' is missing.",
    )
    result = nl.analyzed("ggscatterstats-edge_corr_missing-parametric")
    assert result is not None
    (issue,) = [i for i in result.check() if i.code == "rows_dropped"]
    assert (issue.severity, issue.message) == (
        "info",
        "3 rows were left out because 'x' or 'y' is missing.",
    )
    # More than half of the rows left out: a warning with the share.
    frame = pd.DataFrame(
        {"a": [1.0, 2, 3, 4, 5, None, None, None, None, None, None], "b": range(11)}
    )
    (issue,) = [i for i in S(frame, "a", "b").check() if i.code == "rows_dropped"]
    assert issue.severity == "warning"
    assert issue.message == (
        "6 rows were left out because 'a' or 'b' is missing; that is 55% of the 11 "
        "rows, so the analysis describes only the 5 that are left."
    )
    (issue,) = [i for i in H(frame, "a").check() if i.code == "rows_dropped"]
    assert issue.message.startswith("6 rows were left out because 'a' is missing;")


def test_heavy_ties() -> None:
    result = nl.analyzed("gghistostats-ToothGrowth-nonparametric")
    assert result is not None
    (issue,) = [i for i in result.check() if i.code == "heavy_ties"]
    assert issue.message.startswith(
        "20 of the 60 values of 'len' (33%) equal 18 or lie as far from it as "
        "another value, so the signed-rank test drops the equal ones and ranks only "
        "40 distinct distances."
    )
    case = nl.BY_NAME["ggscatterstats-edge_corr_ties-nonparametric"]
    result = nl.analyzed(case.name)
    assert result is not None
    x, y = case.analyzed_values("x"), case.analyzed_values("y")
    tied = (len(x) - len(set(x))) + (len(y) - len(set(y)))
    assert (tied, len(x)) == (29, 20)
    (issue,) = [i for i in result.check() if i.code == "heavy_ties"]
    assert issue.message.startswith("29 of the 40 values of 'x' and 'y' (72%) tie")
    # The same data with a test that ranks nothing: no heavy_ties.
    for name in (
        "gghistostats-ToothGrowth-parametric",
        "ggscatterstats-edge_corr_ties-parametric",
    ):
        result = nl.analyzed(name)
        assert result is not None
        assert "heavy_ties" not in _codes(result)


def test_infinite_effect_size_wording() -> None:
    result = nl.analyzed("gghistostats-edge_one_small-robust")
    assert result is not None
    (issue,) = [i for i in result.check() if i.code == "infinite_effect_size"]
    assert issue.message == (
        "169 of 2,000 bootstrap resamples have a standard error of 0 and an infinite "
        "bootstrap t; they are kept, so the trimmed mean's 95% CI [-Inf, Inf] is "
        "unbounded. A resample whose winsorized values are all equal has a "
        "winsorized variance of 0."
    )
    assert issue.fix == (
        "Report the interval as it is, unbounded where it is; collect more values, "
        "or use type='nonparametric', which does not resample."
    )
    result = nl.analyzed("gghistostats-edge_one_ties-robust")
    assert result is not None
    (issue,) = [i for i in result.check() if i.code == "infinite_effect_size"]
    assert "CI [2.39, 4.01] would be unbounded at a conf_level high enough" in (
        issue.message
    )
    # The copy without values reports the same.
    embedded = StatsResult.from_json(result.to_json(include_values=False))
    assert embedded.check() == result.check()
