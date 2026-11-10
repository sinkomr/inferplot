"""Text levels are sorted as text, as R sorts them, so "week 10" comes before
"week 2"; the order stays (parity), and a ``warnings`` entry says where it differs
from the order by the numbers in the names and how ``order=`` sets that order
(review of phase 3, finding 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot._data import read_between, read_within

WEEKS = [f"week {w}" for w in range(1, 13)]


def _within(names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        (f"p{i:02d}", name, 50.0 + 1.5 * j + float(rng.normal(0, 2)))
        for i in range(8)
        for j, name in enumerate(names)
    ]
    return pd.DataFrame(rows, columns=["id", "time", "score"])


def _notes(warnings: tuple[str, ...]) -> list[str]:
    return [w for w in warnings if "sorted as text" in w]


def test_numbered_levels_keep_the_text_order_and_say_so() -> None:
    result = ip.analyze.withinstats(
        _within(WEEKS), "time", "score", subject_id="id", pairwise_display="none"
    )
    assert result.levels[:5] == (
        "week 1",
        "week 10",
        "week 11",
        "week 12",
        "week 2",
    )
    (note,) = _notes(result.warnings)
    assert note.startswith(
        "The levels of 'time' are text, sorted as text (as R sorts them), so "
        "'week 10' comes before 'week 2': 'week 1', 'week 10', 'week 11', "
        "'week 12', 'week 2', ... (12 levels). The analysis and the figure use this "
        "order. To order them by the numbers in their names, pass order="
    )
    # Twelve names make an example too long to write out.
    assert note.endswith("(for example by the numbers in their names).")


def test_a_short_natural_order_is_the_example() -> None:
    names = ["dose 5", "dose 10", "dose 20"]
    data = _within(names)
    result = ip.analyze.withinstats(data, "time", "score", subject_id="id")
    (note,) = _notes(result.warnings)
    assert "so 'dose 10' comes before 'dose 5'" in note
    assert note.endswith("pass order=['dose 5', 'dose 10', 'dose 20'].")
    between = ip.analyze.betweenstats(data, "time", "score")
    assert _notes(between.warnings) == [note]


@pytest.mark.parametrize(
    "names",
    [
        ["a", "b", "c"],  # no numbers
        ["week 1", "week 2", "week 3"],  # numbers that sort alike either way
        ["t01", "t02", "t10"],  # zero-padded
        # Names that can be read two ways get no advice: decimals already in
        # order (as whole numbers 0.5 would go before 0.25), and "1.10", which is
        # after "1.5" as a version and before it as a decimal.
        ["0.25", "0.5", "1"],
        ["1.10", "1.5"],
        ["v1.10", "v1.2"],
    ],
)
def test_no_note_where_the_orders_agree(names: list[str]) -> None:
    result = ip.analyze.withinstats(_within(names), "time", "score", subject_id="id")
    assert _notes(result.warnings) == []


def test_no_note_with_order_or_declared_or_numeric_levels() -> None:
    names = ["dose 5", "dose 10", "dose 20"]
    data = _within(names)
    given = ip.analyze.withinstats(data, "time", "score", subject_id="id", order=names)
    assert _notes(given.warnings) == []
    declared = data.assign(time=pd.Categorical(data["time"], categories=names))
    assert read_within(declared, "time", "score", "id").natural_order == ()
    numeric = data.assign(time=data["time"].str.removeprefix("dose ").astype(int))
    assert read_between(numeric, "time", "score").natural_order == ()
    assert read_between(numeric, "time", "score").levels == ("5", "10", "20")


def test_decimals_and_signs_are_read_as_numbers() -> None:
    data = _within(["2.5", "10"])
    (note,) = _notes(ip.analyze.betweenstats(data, "time", "score").warnings)
    assert note.endswith("pass order=['2.5', '10'].")
    # A minus sign at the start of a name is part of the number; a hyphen inside
    # a name ("1-5") is not.
    signed = read_between(_within(["-1", "-10", "2"]), "time", "score")
    assert signed.levels == ("-1", "-10", "2")
    assert signed.natural_order == ("-10", "-1", "2")
    ranges = read_between(_within(["1-5", "6-10", "11-15"]), "time", "score")
    assert ranges.levels == ("1-5", "11-15", "6-10")
    assert ranges.natural_order == ("1-5", "6-10", "11-15")
