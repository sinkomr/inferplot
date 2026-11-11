"""barstats' pairwise Fisher tests at their limits: what the warnings and the
``pairs_without_test`` check say in every combination of the adjustment method
("none" or one that adjusts), pairs whose table has a single column (present or
not) and pairs computed (all, some, one or none); pairs whose Cramér's V the
intervals' limit did not reach; and ``ip.barstats`` refusing more bars than its
figure holds before the tests run (review P5B).
"""

from __future__ import annotations

from typing import Final, cast

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import inferplot as ip
from inferplot import stats
from inferplot.result import StatsResult
from inferplot.stats import contingency

# x0 and x1: 840 observations over 20 levels of y, far apart (their exact test is
# beyond the limits); a small row tested cheaply against either; two rows held
# only in the last level of y (their pair has a single column).
FAR0: Final = [2 * j + 2 for j in range(20)]
FAR1: Final = [40 - 2 * j for j in range(20)]
CHEAP: Final = [3, 2] + [0] * 18
LAST3: Final = [0] * 19 + [3]
LAST4: Final = [0] * 19 + [4]


def _frame(table: list[list[int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"x": f"x{i}", "y": f"y{j:02d}", "n": v}
            for i, row in enumerate(table)
            for j, v in enumerate(row)
            if v
        ]
    )


def _run(
    table: list[list[int]],
    method: str,
    limit: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> StatsResult:
    if limit is not None:
        # A smaller budget for the tests of one table: the same table with fewer
        # pairs computed (the units are counted, so this is the same on any
        # machine).
        monkeypatch.setattr(contingency, "_PAIRWISE_WORK_LIMIT", limit)
    return ip.analyze.barstats(
        _frame(table), "x", "y", counts="n", p_adjust_method=method
    )


def _check(result: StatsResult) -> str:
    (issue,) = [i for i in result.check() if i.code == "pairs_without_test"]
    return issue.message


# (table, budget, pairs with a single column, too large, tested)
CASES: Final = {
    "costly-some": ([FAR0, FAR1, CHEAP], None, 0, 1, 2),
    "costly-one": ([FAR0, FAR1, CHEAP], 1e8, 0, 2, 1),
    "costly-none": ([FAR0, FAR1, CHEAP], 1e4, 0, 3, 0),
    "single-costly-some": ([FAR0, FAR1, LAST3, LAST4], None, 1, 1, 4),
    "single-costly-one": ([FAR0, FAR1, LAST3, LAST4], 7e7, 1, 4, 1),
    "single-costly-none": ([FAR0, FAR1, LAST3, LAST4], 1e4, 1, 5, 0),
    "single-some": ([[3, 0], [4, 0], [0, 5]], None, 1, 0, 2),
    "single-none": ([[3, 0], [4, 0], [0, 5]], 1e4, 1, 2, 0),
}


@pytest.mark.parametrize("method", ["holm", "bonferroni", "none"])
@pytest.mark.parametrize("case", list(CASES))
def test_each_sentence_is_true_in_every_combination(
    case: str, method: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, limit, single, costly, tested = CASES[case]
    result = _run(table, method, limit, monkeypatch)
    pairs = result.pairwise
    total = len(pairs)
    assert total == single + costly + tested
    assert sum(p.p_value is not None for p in pairs) == tested
    have = [p for p in pairs if p.p_value is not None]
    texts = [
        w
        for w in result.warnings
        if "pair" in w and not w.startswith("p_adjust_method=")
    ] + [_check(result)]
    joined = " ".join(texts)
    # What the adjustment did, as it did it.
    raw = [p.p_value for p in have if p.p_value is not None]
    if method == "none":
        assert [p.p_adjusted for p in have] == raw
        assert "adjusts" not in joined
        assert "adjusted as" not in joined
        assert "among themselves" not in joined
        if tested:
            assert "unadjusted" in joined
    else:
        family = [*raw, *[1.0] * costly]
        adjusted = stats.p_adjust(family, cast("stats.PAdjustMethod", method))
        assert [p.p_adjusted for p in have] == list(adjusted)[: len(raw)]
        assert "unadjusted" not in joined
        if tested and costly:
            assert f"as {tested + costly} comparisons" in joined
        elif tested > 1:
            assert "among themselves" in joined
    # The counts each sentence states.
    missing = single + costly
    if costly:
        (warning,) = [w for w in texts if w.startswith("Fisher's exact test was")]
        comparable = total - single
        among = f"of the {comparable} pairs of levels of 'x'"
        if single:
            among += " whose table has two or more columns"
        if tested:
            assert f"computed for {tested} {among};" in warning
        else:
            assert f"computed for none {among}:" in warning
    (summary,) = [w for w in texts if "Fisher test p-value (above)" in w]
    if tested:
        verb = "has" if missing == 1 else "have"
        assert summary.startswith(
            f"{missing} of the {total} pairs of levels of 'x' {verb}"
        )
    else:
        assert summary.startswith(f"None of the {total} pairs of levels of 'x' has")
    assert _check(result).startswith(f"{missing} of the {total} pairwise comparisons")


def test_the_sentences_of_one_combination(monkeypatch: pytest.MonkeyPatch) -> None:
    # One tested pair among two too large, Holm: the tested pair is adjusted
    # with the other two counted as p = 1 (three comparisons).
    result = _run([FAR0, FAR1, CHEAP], "holm", 1e8, monkeypatch)
    assert [w for w in result.warnings if "pair" in w] == [
        "Fisher's exact test was computed for 1 of the 3 pairs of levels of 'x'; "
        "the other 2 ('x0' and 'x1'; 'x1' and 'x2') are too large to compute "
        "exactly within inferplot's limits (about a second of work for one test "
        "and two seconds for all the pairs of one table, the cheapest computed "
        "first), and inferplot does not approximate them. Their Cramér's V is "
        "reported. To test them, merge sparse levels of 'y' (levels with few "
        "observations), or compare two levels of 'x' in a call of their own.",
        "2 of the 3 pairs of levels of 'x' have no Fisher test p-value (above); "
        "they stay in result.pairwise without one, and p_adjust_method='holm' "
        "adjusts the other one as 3 comparisons: a pair too large to compute "
        "counts as a test with p = 1, so leaving them out makes no computed pair "
        "look more significant.",
    ]
    assert _check(result).endswith(
        "The other one is adjusted as 3 comparisons: a pair too large to compute "
        "counts as a test with p = 1."
    )
    # p_adjust_method has an effect here (three comparisons), so no note says
    # it has none.
    assert not any("has no effect" in w for w in result.warnings)
    none = _run([FAR0, FAR1, CHEAP], "none", 1e8, monkeypatch)
    assert none.warnings[1].endswith(
        "and p_adjust_method='none' leaves the p-value of the other one unadjusted."
    )
    assert _check(none).endswith(
        "The other one keeps its p-value unadjusted (p_adjust_method='none')."
    )


def test_the_check_names_six_pairs_and_counts_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no budget at all, every one of 28 pairs is too large; the check names
    # six of them and counts the others (a table of 200 levels can leave
    # thousands).
    rng = np.random.default_rng(1)
    table = rng.integers(1, 9, size=(8, 4)).tolist()
    result = _run(table, "holm", 1e3, monkeypatch)
    message = _check(result)
    assert message.startswith("28 of the 28 pairwise comparisons of 'x' have no")
    assert message.endswith("; and 22 more.")
    assert message.count("too large to compute") == 6


def test_pairs_whose_v_the_limit_did_not_reach(monkeypatch: pytest.MonkeyPatch) -> None:
    # An allowance for the intervals that reaches no pair: every V is missing
    # with its reason, the Fisher tests are not affected, and a warning says so.
    rng = np.random.default_rng(2)
    table = rng.integers(5, 30, size=(5, 3)).tolist()
    full = _run(table, "holm", None, monkeypatch)
    spent = contingency._Budget(contingency._PAIRWISE_WORK_LIMIT)
    contingency._pairwise_fisher(np.array(table), "holm", 0.95, "two-sided", spent)
    # The tests' budget leaves 1e7 units, the intervals' own allowance none.
    monkeypatch.setattr(contingency, "_PAIRWISE_EFFECT_LIMIT", 0.0)
    monkeypatch.setattr(contingency, "_PAIRWISE_WORK_LIMIT", spent.spent + 1e7)
    result = _run(table, "holm", None, monkeypatch)
    assert [p.p_value for p in result.pairwise] == [p.p_value for p in full.pairwise]
    assert all(p.p_value is not None for p in result.pairwise)
    # What the tests left reaches some pairs: the smallest tables.
    reached = [p for p in result.pairwise if p.estimate is not None]
    missed = [p for p in result.pairwise if p.estimate is None]
    assert reached
    assert missed
    for pair in reached:
        twin = next(
            f
            for f in full.pairwise
            if (f.group1, f.group2) == (pair.group1, pair.group2)
        )
        assert (pair.estimate, pair.conf_low, pair.conf_high) == (
            twin.estimate,
            twin.conf_low,
            twin.conf_high,
        )
    (warning,) = [w for w in result.warnings if w.startswith("Cramér's V was")]
    assert warning.startswith(
        f"Cramér's V was computed for {len(reached)} of the 10 pairs of levels of "
        f"'x' whose V is defined; the other {len(missed)} ("
    )
    assert warning.endswith(
        "none (estimate None): the V intervals of all the pairs of one table share "
        "a limit of work (about half a second, more when the Fisher tests leave "
        "some), the smallest tables first, and inferplot does not approximate them. "
        f"{'Its' if len(missed) == 1 else 'Their'} Fisher test is not affected. To "
        "get every V, compare fewer levels of 'x' in one call."
    )
    monkeypatch.undo()
    stats_pairs = contingency.pairwise_fisher(np.array(table))
    assert all(p.effect is not None for p in stats_pairs)


def test_too_many_bars_are_refused_before_the_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 40 levels of y: inferplot's own figure holds 37 bars (DECISIONS D123).
    # ip.barstats refuses as soon as the data are read, with the renderer's
    # message, without running the tests; ip.analyze.barstats is unchanged, and
    # with ax= the analysis runs and the figure is drawn there.
    rng = np.random.default_rng(3)
    n = 4000
    frame = pd.DataFrame(
        {
            "x": [f"x{i}" for i in rng.integers(0, 5, n)],
            "y": [f"y{j:02d}" for j in rng.integers(0, 40, n)],
        }
    )
    result = ip.analyze.barstats(frame, "x", "y")
    with pytest.raises(ValueError, match="at most 37 bars fit") as rendered:
        ip.render.barstats(result)

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("the pairwise tests ran")

    monkeypatch.setattr(stats, "pairwise_fisher", refuse)
    with pytest.raises(ValueError, match="at most 37 bars fit") as early:
        ip.barstats(frame, "x", "y")
    assert str(early.value) == str(rendered.value)
    with pytest.raises(AssertionError, match="the pairwise tests ran"):
        ip.analyze.barstats(frame, "x", "y")
    with pytest.raises(AssertionError, match="the pairwise tests ran"):
        ip.barstats(frame, "x", "y", ax=Figure(figsize=(40, 6)).add_subplot())
    monkeypatch.undo()
    fig = ip.barstats(frame, "x", "y", ax=Figure(figsize=(40, 6)).add_subplot())
    assert fig.result.pairwise == result.pairwise
    # Fewer bars: drawn as before.
    kept = frame[frame["y"] < "y30"]
    assert len(ip.barstats(kept, "x", "y").result.bars) == 30
