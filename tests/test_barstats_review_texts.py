"""barstats' texts after the review (visible half): what the APA sentence, the
alt text and ``str(result)`` say where the earlier wording misled."""

from __future__ import annotations

import dataclasses
from typing import Any

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
