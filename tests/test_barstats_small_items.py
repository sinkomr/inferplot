"""The small items of the barstats review (B17), one test each."""

from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np
import pandas as pd
import pytest

import inferplot as ip

FRAME = pd.DataFrame({"x": list("aabbab"), "y": list("uvuvvu")})
anything: Any = ip.barstats


def _codes(fig: Any) -> list[str]:
    return [i.code for i in fig.check()]


def test_the_docstring_names_every_error() -> None:
    doc = ip.barstats.__doc__ or ""
    raises = doc[doc.index("Raises:") :]
    for error in ("TypeError", "ValueError", "KeyError", "NotImplementedError"):
        assert error in raises


def test_a_level_named_empty_text_gets_a_stand_in() -> None:
    frame = pd.DataFrame({"x": ["", "a", "", "a", "a", ""], "y": list("uvuvvu")})
    fig = ip.barstats(frame, "x", "y", bf_message=False)
    legend = fig.axes["main"].get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["(empty)", "a"]
    assert any("empty text ('')" in w and "'(empty)'" in w for w in fig.result.warnings)
    assert "(empty) " in fig.result.text.alt_text
    bars = pd.DataFrame({"x": list("abab"), "y": ["", "", "v", "v"]})
    fig = ip.barstats(bars, "x", "y", bf_message=False)
    ticks = [t.get_text() for t in fig.axes["main"].get_xticklabels()]
    assert ticks[0].startswith("(empty)\n(n = 2)")
    assert fig.result.text.tick_labels[0] == "(empty)"


def test_precomposed_and_decomposed_levels_are_named() -> None:
    composed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")
    assert composed != decomposed
    frame = pd.DataFrame(
        {"x": [composed, decomposed, composed, decomposed], "y": list("uvvu")}
    )
    result = ip.analyze.barstats(frame, "x", "y", bf_message=False)
    (note,) = [w for w in result.warnings if "how their characters are" in w]
    assert "'caf\\xe9'" in note
    assert "'cafe\\u0301'" in note
    assert "NFC" in note


def test_a_removed_legend_is_reported() -> None:
    fig = ip.barstats(FRAME, "x", "y", bf_message=False)
    assert "legend_missing" not in _codes(fig)
    legend = fig.axes["main"].get_legend()
    assert legend is not None
    legend.remove()
    (issue,) = [i for i in fig.check() if i.code == "legend_missing"]
    assert "alt text" in issue.message


def test_one_expected_count_below_5_is_singular() -> None:
    frame = pd.DataFrame(
        {"x": ["a", "a", "b", "b"], "y": ["u", "v", "u", "v"], "n": [10, 940, 0, 50]}
    )
    result = ip.analyze.barstats(frame, "x", "y", counts="n", bf_message=False)
    (issue,) = [i for i in result.check() if i.code == "small_expected"]
    assert "1 of the 4 expected counts of the chi-square test of independence is " in (
        issue.message
    )


def test_small_expected_covers_the_bars_tests_the_apa_sentence_reports() -> None:
    frame = pd.DataFrame(
        {
            "x": ["a", "b", "c"] * 2,
            "y": ["u"] * 3 + ["v"] * 3,
            "n": [40, 40, 40, 2, 1, 1],
        }
    )
    for drawn in (True, False):
        result = ip.analyze.barstats(
            frame, "x", "y", counts="n", proportion_test=drawn, bf_message=False
        )
        assert "Goodness-of-fit tests within each level of y" in result.text.apa
        issues = [i for i in result.check() if i.code == "small_expected"]
        assert issues
        assert "the proportion test of the bar 'v'" in issues[0].message


def test_the_mixed_object_message_starts_its_sentence() -> None:
    frame = pd.DataFrame({"x": [1, "a", 2.5, None], "y": list("uvuv")})
    with pytest.raises(TypeError) as caught:
        ip.analyze.barstats(frame, "x", "y")
    assert ". Convert it to text or to a categorical column first." in str(caught.value)


def test_a_dropped_row_of_a_table_says_what_it_stood_for() -> None:
    frame = pd.DataFrame(
        {"x": ["a", "b", None, "a", "b"], "y": list("uvuvu"), "n": [10, 20, 50, 5, 8]}
    )
    result = ip.analyze.barstats(frame, "x", "y", counts="n", bf_message=False)
    assert result.warnings[0].startswith(
        "1 row (standing for 50 observations) was left out because"
    )


def test_line_breaks_in_names_are_spaces_in_the_sentences() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "co\nst": rng.choice(["low\nend", "high"], 60),
            "new\nline": rng.choice(["p", "q\nr"], 60),
        }
    )
    result = ip.analyze.barstats(frame, "co\nst", "new\nline", bf_message=False)
    for text in (result.text.apa, result.text.alt_text):
        assert "\n" not in text
    assert "co st" in result.text.apa
    assert "new line" in result.text.apa
    assert "low end" in result.text.alt_text


@pytest.mark.parametrize(
    ("keyword", "value", "advised"),
    [
        ("cmap", "viridis", {"palette": "viridis"}),
        ("figsize", (8, 5), None),
        ("hue_order", ["b", "a"], None),
        ("width", 0.5, None),
        ("edgecolor", "black", {"bar_args": {"edgecolor": "black"}}),
        ("bayes", True, {"bf_message": False}),
    ],
)
def test_targeted_messages(keyword: str, value: object, advised: Any) -> None:
    with pytest.raises(TypeError) as caught:
        anything(FRAME, "x", "y", **{keyword: value})
    text = str(caught.value)
    assert f"{keyword}= is not an argument of barstats()" in text
    if keyword == "bayes":
        assert "no call to make for it yet" in text
        assert "type='bayes' and bf_message=True are refused" in text
    if advised is not None:
        anything(FRAME, "x", "y", **advised)
