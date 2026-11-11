"""Declared levels without observations (review B18, B19).

A category kept after filtering a categorical column counts in each bar's
goodness-of-fit test as a category expected to hold its share of the bar, as
in ggstatsplot, so the p drawn above a bar can be tiny although the bar's
drawn shares are close. The numbers stay ggstatsplot's; the user is told: in
``warnings`` (with the call that leaves the category out, in the frame's own
library, executed here), by ``check()`` and in the APA sentence and alt text.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow
import pyarrow.compute
import pytest

import inferplot as ip

LEVELS = ["drug", "placebo", "waitlist"]


def _frame(backend: str) -> Any:
    rng = np.random.default_rng(3)
    treatment = [str(v) for v in rng.choice(LEVELS, 120)]
    outcome = [str(v) for v in rng.choice(["better", "same"], 120)]
    kept = [i for i, t in enumerate(treatment) if t != "waitlist"]
    treatment = [treatment[i] for i in kept]
    outcome = [outcome[i] for i in kept]
    if backend == "pandas":
        return pd.DataFrame(
            {
                "treatment": pd.Categorical(treatment, categories=LEVELS),
                "outcome": outcome,
            }
        )
    if backend == "polars":
        return pl.DataFrame(
            {
                "treatment": pl.Series(treatment, dtype=pl.Enum(LEVELS)),
                "outcome": outcome,
            }
        )
    codes = pyarrow.array([LEVELS.index(t) for t in treatment], pyarrow.int32())
    dictionary = pyarrow.DictionaryArray.from_arrays(
        codes, pyarrow.array(LEVELS), ordered=True
    )
    return pyarrow.table({"treatment": dictionary, "outcome": outcome})


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_an_unused_category_in_the_bars_tests_is_explained(backend: str) -> None:
    frame = _frame(backend)
    result = ip.analyze.barstats(frame, "treatment", "outcome", bf_message=False)
    assert result.levels == tuple(LEVELS)
    (note,) = [w for w in result.warnings if "'waitlist'" in w]
    assert "expected to hold 1/3 of the bar (3 categories in all)" in note
    # The texts next to the p-values say what they test.
    clause = "against equal shares of 3 categories ('waitlist' with no observations)"
    assert clause in result.text.apa
    assert clause in result.text.alt_text
    (issue,) = [i for i in result.check() if i.code == "empty_level_tested"]
    assert issue.severity == "warning"
    assert "'better': p = " in issue.message
    # The call the note gives leaves the category out, and the bars' tests are
    # then those of the two categories.
    found = re.search(r"leave it out: (.*)\.$", note)
    assert found is not None, note
    space: dict[str, Any] = {"data": frame, "pd": pd, "pl": pl, "pyarrow": pyarrow}
    if isinstance(frame, pd.DataFrame):
        space["data"] = frame.copy()
    exec(found.group(1), space)
    again = ip.analyze.barstats(space["data"], "treatment", "outcome", bf_message=False)
    assert again.levels == ("drug", "placebo")
    assert again.n_obs == result.n_obs
    assert not [w for w in again.warnings if "'waitlist'" in w]
    assert "empty_level_tested" not in [i.code for i in again.check()]
    assert "with no observations" not in again.text.apa
    tested = [b.test.p_value for b in again.bars if b.test is not None]
    before = [b.test.p_value for b in result.bars if b.test is not None]
    assert min(tested) > max(before)


def test_without_drawn_bar_tests_there_is_no_check() -> None:
    frame = _frame("pandas")
    result = ip.analyze.barstats(
        frame, "treatment", "outcome", bf_message=False, proportion_test=False
    )
    assert "empty_level_tested" not in [i.code for i in result.check()]
