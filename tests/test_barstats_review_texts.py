"""barstats' texts after the review (visible half): what the APA sentence, the
alt text and ``str(result)`` say where the earlier wording misled."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

import inferplot as ip
from inferplot.result import StatsResult
from inferplot.text import contingency_tests_sentence


def table(counts: list[list[int]]) -> pd.DataFrame:
    rows = len(counts)
    cols = len(counts[0])
    xs = [f"x{i + 1}" for i in range(rows)]
    ys = [f"y{j + 1}" for j in range(cols)]
    frame = pd.DataFrame(
        [(xs[i], ys[j], counts[i][j]) for i in range(rows) for j in range(cols)],
        columns=["x", "y", "n"],
    )
    frame["x"] = pd.Categorical(frame["x"], categories=xs)
    frame["y"] = pd.Categorical(frame["y"], categories=ys)
    return frame


# -- B4: pairs that were never tested are not counted --------------------------------


def test_pairs_without_a_test_are_not_counted_as_tested() -> None:
    # x1 and x2 occur only in y1: their pair's table keeps one column (no test).
    result = ip.analyze.barstats(
        table([[5, 0], [4, 0], [0, 9]]), "x", "y", counts="n", bf_message=False
    )
    untested = [p for p in result.pairwise if p.p_value is None]
    assert len(untested) == 1
    sentence = (
        "Fisher's exact tests between the levels of x (Holm adjusted): adjusted "
        "p < .05 in 2 of the 2 tested pairs; 1 pair has no test."
    )
    assert result.text.apa.endswith(sentence), result.text.apa
    assert sentence in str(result)
    # The untested pair's label stays as it is (it is never drawn).
    assert len(result.text.pairwise_labels) == 3


def test_all_pairs_tested_keep_the_short_wording() -> None:
    result = ip.analyze.barstats(
        table([[30, 10], [10, 30], [20, 20]]), "x", "y", counts="n", bf_message=False
    )
    assert all(p.p_value is not None for p in result.pairwise)
    assert "adjusted p < .05 in " in result.text.apa
    assert " of 3 pairs." in result.text.apa
    assert "tested" not in result.text.apa


def test_no_pair_tested_says_so() -> None:
    result = ip.analyze.barstats(
        table([[30, 10], [10, 30], [20, 20]]), "x", "y", counts="n", bf_message=False
    )
    pairs = [
        dataclasses.replace(
            p,
            p_value=None,
            p_adjusted=None,
            significant=None,
            no_test_reason="beyond the cost limit",
        )
        for p in result.pairwise
    ]
    sentence = contingency_tests_sentence(
        result.bars, pairs, x="x", y="y", settings=result.settings
    )
    assert sentence is not None
    assert sentence.endswith(
        "Fisher's exact tests between the levels of x: none of the 3 pairs has a test."
    )


# -- B5: an effect size outside its noncentral chi-square interval ---------------


def _outside(result: StatsResult) -> str:
    (issue,) = [i for i in result.check() if i.code == "estimate_outside_interval"]
    assert "bootstrap" not in issue.message + issue.fix
    assert "unaffected" in issue.fix
    return issue.message


def test_cramers_v_above_its_interval_is_explained() -> None:
    # A 2 x 35 table of few observations per cell (the reviewer's c02 case):
    # V = 0.236 with CI [0.000, 0.201].
    rng = np.random.default_rng(7)
    found = None
    for trial in range(40):
        cols = int(rng.integers(10, 100)) if trial % 2 else int(rng.integers(3, 12))
        counts = rng.integers(0, 9, size=(2, cols))
        counts[:, 0] += 1
        counts[0, :] += 1
        frame = pd.DataFrame(
            [
                (f"x{i}", f"y{j:03d}", int(counts[i, j]))
                for i in range(2)
                for j in range(cols)
            ],
            columns=["x", "y", "n"],
        )
        result = ip.analyze.barstats(
            frame, "x", "y", counts="n", bf_message=False, proportion_test=False
        )
        es = result.effect_size
        assert es is not None
        if es.estimate > es.conf_high > 0:
            found = result
            break
    assert found is not None
    message = _outside(found)
    assert "property of the method" in message
    assert "noncentrality" in message
    assert "few observations per cell" in message


def test_cramers_v_beside_a_zero_interval_is_explained_by_the_correction() -> None:
    # A 2 x 20 table (chi-square 19.70 on 19 df): the bias correction takes the
    # upper bound to 0, not the lower tail of the statistic.
    first = [7, 1, 9, 4, 7, 6, 4, 7, 6, 2, 1, 1, 5, 8, 2, 9, 9, 3, 6, 6]
    second = [4, 5, 5, 4, 3, 3, 1, 0, 3, 2, 0, 3, 2, 8, 0, 5, 8, 2, 1, 6]
    frame = pd.DataFrame(
        [("x0", f"y{j:03d}", n) for j, n in enumerate(first)]
        + [("x1", f"y{j:03d}", n) for j, n in enumerate(second)],
        columns=["x", "y", "n"],
    )
    result = ip.analyze.barstats(
        frame, "x", "y", counts="n", bf_message=False, proportion_test=False
    )
    es = result.effect_size
    assert es is not None
    assert es.conf_low == es.conf_high == 0.0 < es.estimate
    message = _outside(result)
    assert "property of the method" in message
    assert "both bounds are 0" not in message


def test_pearsons_c_beside_a_zero_interval_is_explained() -> None:
    # A fit closer than chance (chi-square 0.58 on 9 df, n = 1,000): C = 0.024
    # with CI [0, 0].
    counts = [100, 104, 96, 100, 102, 98, 100, 103, 97, 100]
    frame = pd.DataFrame({"x": [f"c{i}" for i in range(10)], "w": counts})
    result = ip.analyze.barstats(frame, "x", counts="w", bf_message=False)
    es = result.effect_size
    assert es is not None
    assert es.conf_low == es.conf_high == 0.0 < es.estimate
    message = _outside(result)
    assert "both bounds are 0" in message
    assert "about 2.5% of tables" in message


# -- B16: expected shares print by a stated rule ---------------------------------


def _one_way(**kwargs: Any) -> StatsResult:
    frame = pd.DataFrame({"g": list("aaabbbbccccc" * 10)})
    return ip.analyze.barstats(frame, "g", bf_message=False, **kwargs)


def test_equal_shares_read_equal_however_written() -> None:
    result = _one_way(ratio=[1 / 3, 1 / 3, 1 / 3])
    assert "(shares of g against equal shares)" in result.text.apa
    assert "against equal shares" in result.text.alt_text
    assert "333" not in result.text.apa + result.text.alt_text


def test_other_shares_are_rounded_and_marked() -> None:
    result = _one_way(ratio=[1 / 6, 1 / 3, 1 / 2])
    assert "against expected shares of about 0.17, 0.33 and 0.5)" in result.text.apa
    assert "666" not in result.text.alt_text
    result = _one_way(ratio=[1 / 6, 1 / 3, 1 / 2], digits=3)
    assert "expected shares of about 0.167, 0.333 and 0.5)" in result.text.apa
    result = _one_way(ratio=[0.2, 0.3, 0.5])
    assert "against expected shares 0.2, 0.3 and 0.5)" in result.text.apa
    result = _one_way(ratio=[0.001, 0.499, 0.5])
    assert "expected shares of about 0.001, 0.5 and 0.5)" in result.text.apa
