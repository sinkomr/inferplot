"""Shortened level names stay distinguishable (review of phase 3, finding 3):
names that would shorten to the same text keep what they do not share, in the
alt text (groups, one-sided direction, Bayes pair ranges) and in the APA
sentence's direction; no name shortens to what another shortens to."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot.text import _shortened  # pyright: ignore[reportPrivateUsage]

LONG = [
    "Mean reaction time in milliseconds, congruent trials",
    "Mean reaction time in milliseconds, incongruent trials",
]


def test_names_that_differ_early_are_cut_at_the_end() -> None:
    assert _shortened(["control group", "treatment group"], 8) == {
        "control group": "control…",
        "treatment group": "treatme…",
    }
    assert _shortened(["a", "b"], 8) == {"a": "a", "b": "b"}


def test_names_that_share_a_start_keep_where_they_differ() -> None:
    assert _shortened(LONG, 40) == {
        LONG[0]: "Mean reaction time in milliseconds, con…",
        LONG[1]: "Mean reaction time in milliseconds, inc…",
    }
    assert _shortened(LONG, 34) == {
        LONG[0]: "Mean reaction \u2026 congruent trials",
        LONG[1]: "Mean reaction \u2026 incongruent trials",
    }
    short = _shortened(LONG, 20)
    assert short == {
        LONG[0]: "…congruent trials",
        LONG[1]: "…incongruent trials",
    }


@pytest.mark.parametrize("limit", [40, 34, 20, 12, 8, 4])
def test_no_two_names_ever_look_alike(limit: int) -> None:
    names = [
        *LONG,
        "Mean reaction time in milliseconds, neutral trials",
        "week 1 of the study, morning",
        "week 1 of the study, evening",
        "week 10 of the study, morning",
    ]
    short = _shortened(names, limit)
    assert len(set(short.values())) == len(names)
    for name, text in short.items():
        assert len(text) <= limit or text == name


def _frame(names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.normal(500, 40, 14)
    columns = [a + 30 * j + rng.normal(0, 15, 14) for j in range(len(names))]
    return pd.DataFrame(
        [
            (f"s{i:02d}", name, float(col[i]))
            for i in range(14)
            for name, col in zip(names, columns, strict=True)
        ],
        columns=["id", "cond", "y"],
    )


def test_the_alt_text_and_the_direction_tell_the_conditions_apart() -> None:
    result = ip.analyze.withinstats(
        _frame(LONG), "cond", "y", subject_id="id", alternative="less"
    )
    alt = result.text.alt_text
    assert "…congruent trials (mean" in alt
    assert "…incongruent trials (mean" in alt
    assert "one-sided: …congruent trials less than …incongruent trials" in alt
    assert "Mean reaction time … (mean" not in alt
    apa = result.text.apa
    assert apa.startswith(
        "Paired t-test (one-sided: Mean reaction time in milliseconds, con… less "
        "than Mean reaction time in milliseconds, inc…)"
    )


def test_bayes_pair_ranges_tell_the_conditions_apart() -> None:
    names = [*LONG, "Mean reaction time in milliseconds, neutral trials"]
    result = ip.analyze.withinstats(
        _frame(names), "cond", "y", subject_id="id", type="bayes",
        results_subtitle=False,
    )  # fmt: skip
    alt = result.text.alt_text
    assert "Mean reacti… vs Mean reacti…" not in alt
    assert "…congruent" in alt
    assert "…incongruen" in alt or "…incongruent" in alt
