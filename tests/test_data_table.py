"""Reading a count table for barstats (``read_table``, SPEC.md section 4 "barstats").

The table, levels, n and dropped rows of every ggbarstats and ggpiestats fixture on
pandas, polars and pyarrow, against what ggstatsplot's ``.pie_bar_data_prep()``
left (``contingency_data``); then each rule on small frames.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from inferplot._data import TableData, read_table, table_notes
from tests.oracle_fixtures import Backend, OracleCase, cases_for, read_dataset

BACKENDS: tuple[Backend, ...] = ("pandas", "polars", "pyarrow")
CASES: tuple[OracleCase, ...] = (
    *cases_for("ggbarstats"),
    *cases_for("ggpiestats"),
    *cases_for("ggbarstats", failed=True),
)
# Paired variables with different levels: ggstatsplot drops the subtitle silently
# ("'x' must be square"); inferplot refuses while reading (D101).
PAIRED_UNEQUAL = "ggbarstats-edge_bar_paired_unequal-parametric"


def test_every_bar_fixture_is_read() -> None:
    assert len(CASES) == 90


def _read_case(case: OracleCase, backend: Backend) -> TableData:
    data = case.contingency_data
    return read_table(
        read_dataset(case, backend),
        data["x"],
        data["y"],
        counts=data["counts"],
        paired=bool(case.args.get("paired", False)),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_table_matches_the_oracle(case: OracleCase, backend: Backend) -> None:
    data = case.contingency_data
    if data["error"] is not None:
        # A count that is not whole: ggbarstats fails while preparing the data.
        with pytest.raises(ValueError, match="ggbarstats fails on such counts too"):
            _read_case(case, backend)
        return
    if case.name == PAIRED_UNEQUAL:
        with pytest.raises(ValueError, match=r"\('maybe' only in 'after'\)"):
            _read_case(case, backend)
        return
    table = _read_case(case, backend)
    assert list(table.x_levels.levels) == data["x_levels"]
    y_levels = [] if table.y_levels is None else list(table.y_levels.levels)
    assert y_levels == data["y_levels"]
    assert [list(row) for row in table.counts] == case.counts_table()
    assert table.n == data["n"]
    assert table.n_dropped == data["rows_dropped"]
    assert table.n_rows == len(data["rows"])
    assert all(type(v) is int for row in table.counts for v in row)


# --- small frames ----------------------------------------------------------------


def frame(backend: Backend, columns: Mapping[str, Sequence[object]]) -> Any:
    """``columns`` as a frame of ``backend`` (None is missing)."""
    pandas = pd.DataFrame(dict(columns))
    if backend == "pandas":
        return pandas
    if backend == "polars":
        return pl.DataFrame(dict(columns), strict=False)
    return pa.table(dict(columns))


def counts_of(table: TableData) -> list[list[int]]:
    return [list(row) for row in table.counts]


@pytest.mark.parametrize("backend", BACKENDS)
def test_text_numbers_and_booleans_are_categories(backend: Backend) -> None:
    data = frame(
        backend,
        {
            "g": ["b", "a", "b", "c", "a", "b"],
            "n": [10, 9, 10, 9, 2, 2],
            "f": [4.0, 6.0, 4.0, 8.0, 8.0, 4.0],
            "t": [True, False, True, True, False, True],
        },
    )
    table = read_table(data, "g", "n")
    assert table.x_levels.levels == ("a", "b", "c")
    assert table.y_levels is not None
    assert table.y_levels.levels == ("2", "9", "10")  # numbers numerically
    assert counts_of(table) == [[1, 1, 0], [1, 0, 2], [0, 1, 0]]
    whole = read_table(data, "f")
    assert whole.x_levels.levels == ("4", "6", "8")
    assert counts_of(whole) == [[3], [1], [2]]
    logical = read_table(data, "t", "g")
    assert logical.x_levels.levels == ("False", "True")
    assert counts_of(logical) == [[2, 0, 0], [0, 3, 1]]


def test_declared_levels_keep_their_order_and_empty_levels() -> None:
    pandas = pd.DataFrame(
        {
            "x": pd.Categorical(["lo", "hi", "lo"], categories=["lo", "mid", "hi"]),
            "y": pd.Categorical(["p", "p", "q"], categories=["q", "none", "p"]),
        }
    )
    polars = pl.DataFrame(
        {
            "x": pl.Series(["lo", "hi", "lo"], dtype=pl.Enum(["lo", "mid", "hi"])),
            "y": pl.Series(["p", "p", "q"], dtype=pl.Enum(["q", "none", "p"])),
        }
    )
    arrow = pa.Table.from_pandas(
        pandas.assign(x=pandas["x"].cat.as_ordered(), y=pandas["y"].cat.as_ordered())
    )
    for data in (pandas, polars, arrow):
        table = read_table(data, "x", "y")
        assert table.x_levels.levels == ("lo", "mid", "hi")
        assert table.x_levels.empty == ("mid",)
        assert table.x_levels.present == ("lo", "hi")
        assert table.y_levels is not None
        assert table.y_levels.levels == ("q", "none", "p")
        assert table.y_levels.empty == ("none",)
        assert counts_of(table) == [[1, 0, 1], [0, 0, 0], [0, 0, 1]]
        assert table.column_totals() == (1, 0, 2)
        assert table.row_totals() == (2, 0, 1)


def test_a_declared_categorical_of_booleans() -> None:
    data = pd.DataFrame(
        {"b": pd.Categorical([True, True, False], categories=[True, False])}
    )
    table = read_table(data, "b")
    assert table.x_levels.levels == ("True", "False")
    assert counts_of(table) == [[2], [1]]


def test_an_unordered_dictionary_is_read_as_its_values() -> None:
    data = pa.table(
        {"x": pa.array(["b", "a", "b", "c"]).dictionary_encode(), "y": [1, 1, 2, 2]}
    )
    table = read_table(data, "x", "y")
    assert table.x_levels.levels == ("a", "b", "c")
    assert table.x_levels.dictionary_order == ("b", "a", "c")
    notes = table_notes(table)
    assert any("not in the order of its pyarrow dictionary" in n for n in notes)
    assert not any("order=" in n for n in notes)  # barstats has no order=


@pytest.mark.parametrize("backend", BACKENDS)
def test_only_x_y_and_counts_drop_rows(backend: Backend) -> None:
    data = frame(
        backend,
        {
            "x": ["a", None, "b", "b", "a", "c"],
            "y": ["p", "p", None, "q", "q", "q"],
            "w": [1.0, 2.0, 3.0, None, 5.0, 0.0],
            "other": [None, None, None, None, None, None],
        },
    )
    unweighted = read_table(data, "x", "y")
    assert unweighted.n_dropped == 2
    assert (unweighted.n_missing_x, unweighted.n_missing_y) == (1, 1)
    assert unweighted.n == 4  # 'other' drops nothing
    weighted = read_table(data, "x", "y", counts="w")
    assert weighted.n_dropped == 3
    assert weighted.n_missing_counts == 1
    assert weighted.n_zero_rows == 1
    assert weighted.n_rows == 3
    assert weighted.x_levels.levels == ("a",)  # c has a count of 0 only
    assert weighted.x_levels.vanished == ("b", "c")
    assert counts_of(weighted) == [[1, 5]]
    notes = table_notes(weighted)
    assert notes[0] == (
        "3 rows were left out because 'x', 'y' or 'w' is missing (1 without 'x', "
        "1 without 'y', 1 without 'w'); 6 observations remain."
    )
    assert notes[1] == (
        "The levels 'b' and 'c' of 'x' were left out of the analysis: every row of "
        "them has a missing 'y' or 'w', or a count of 0."
    )


def test_a_declared_level_held_only_by_zero_counts_stays() -> None:
    data = pd.DataFrame(
        {
            "x": pd.Categorical(["a", "b", "c"], categories=["a", "b", "c"]),
            "w": [3, 4, 0],
        }
    )
    table = read_table(data, "x", counts="w")
    assert table.x_levels.levels == ("a", "b", "c")
    assert table.x_levels.empty == ("c",)
    assert counts_of(table) == [[3], [4], [0]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_counts_must_be_whole_numbers_at_least_zero(backend: Backend) -> None:
    base = {"x": ["a", "b", "a", "b"]}
    whole_floats = read_table(
        frame(backend, {**base, "w": [1.0, 2.0, 3.0, 4.0]}), "x", counts="w"
    )
    assert counts_of(whole_floats) == [[4], [6]]
    with pytest.raises(ValueError, match=r"not whole numbers \(for example 2.5, not"):
        read_table(frame(backend, {**base, "w": [1.0, 2.5, 3.0, 4.0]}), "x", counts="w")
    with pytest.raises(ValueError, match=r"positions 1\)\. ggbarstats fails"):
        read_table(
            frame(backend, {**base, "w": [1.0, math.inf, 3.0, 4.0]}), "x", counts="w"
        )
    with pytest.raises(ValueError, match=r"1 of its values are negative \(row"):
        read_table(frame(backend, {**base, "w": [1, -2, 3, 4]}), "x", counts="w")
    with pytest.raises(TypeError, match="it holds True/False values"):
        read_table(
            frame(backend, {**base, "w": [True, False, True, True]}), "x", counts="w"
        )
    with pytest.raises(TypeError, match=r"numbers stored as text \(for example '1'\)"):
        read_table(frame(backend, {**base, "w": ["1", "2", "3", "4"]}), "x", counts="w")


def test_large_counts_are_added_exactly_without_repeating_rows() -> None:
    big = 10**12
    data = pd.DataFrame({"x": ["a", "b", "a"], "y": ["p", "p", "q"], "w": [big] * 3})
    table = read_table(data, "x", "y", counts="w")
    assert counts_of(table) == [[big, big], [big, 0]]
    assert table.n == 3 * big
    # Beyond what int64 can add: Python ints.
    huge = 2**62
    exact = read_table(
        pd.DataFrame({"x": ["a", "a", "b"], "w": [huge, huge, 1]}), "x", counts="w"
    )
    assert counts_of(exact) == [[2 * huge], [1]]
    # (The dtype is given: polars 1.0 does not infer UInt64 from Python ints.)
    beyond = pl.DataFrame(
        {"x": ["a", "a"], "w": pl.Series("w", [2**63 + 1, 2**63 + 1], dtype=pl.UInt64)}
    )
    assert counts_of(read_table(beyond, "x", counts="w")) == [[2**64 + 2]]
    floats = pd.DataFrame({"x": ["a", "b"], "w": [1e15, 3.0]})
    assert counts_of(read_table(floats, "x", counts="w")) == [[10**15], [3]]


def test_counts_column_must_differ_from_x_and_y() -> None:
    data = pd.DataFrame({"x": ["a", "b"], "y": ["p", "q"]})
    with pytest.raises(ValueError, match=r"both are 'x'\. To count the categories"):
        read_table(data, "x", "x")
    with pytest.raises(ValueError, match=r"counts='y' is also y="):
        read_table(data, "x", "y", counts="y")


def test_no_observation_left_raises() -> None:
    data = pd.DataFrame({"x": ["a", "b"], "w": [0, 0]})
    with pytest.raises(ValueError, match=r"no row of data has 'x' and 'w' present, an"):
        read_table(data, "x", counts="w")
    with pytest.raises(ValueError, match="no row of data has 'x' and 'y' present"):
        read_table(pd.DataFrame({"x": ["a", None], "y": [None, "q"]}), "x", "y")


# --- measurements -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (
            ("group", "score"),
            "y='score' holds numbers that are not whole (1.5), so it looks like a "
            "measurement: barstats counts observations in categories (text, "
            "categorical, booleans or whole-number codes). To compare 'score' "
            "across the groups of 'group' (for example its mean in each group), use "
            "betweenstats(data, x='group', y='score').",
        ),
        (
            ("score", "group"),
            "To compare 'score' across the groups of 'group', use "
            "betweenstats(data, x='group', y='score').",
        ),
        (
            ("score", "code"),
            "To compare 'score' across the groups of 'code', use "
            "betweenstats(data, x='code', y='score').",
        ),
        (
            ("score", "other"),
            "To relate two numeric columns use scatterstats(data, x='score', "
            "y='other').",
        ),
        (
            ("score", None),
            "To test one numeric column against a value use histostats(data, "
            "x='score', test_value=...).",
        ),
    ],
)
def test_a_measurement_points_to_the_function_that_fits(
    columns: tuple[str, str | None], message: str
) -> None:
    data = pd.DataFrame(
        {
            "group": ["a", "b"] * 6,
            "code": [1, 2, 3] * 4,
            "score": [1.5, 2.0, 3.25, 4.0] * 3,
            "other": [0.5 * k for k in range(12)],
        }
    )
    x, y = columns
    with pytest.raises(TypeError) as info:
        read_table(data, x, y)
    assert message in str(info.value)


def test_many_distinct_whole_numbers_are_a_measurement() -> None:
    ages = pd.DataFrame({"age": list(range(20, 45)), "g": ["a", "b"] * 12 + ["a"]})
    with pytest.raises(TypeError) as info:
        read_table(ages, "g", "age")
    assert str(info.value) == (
        "y='age' holds 25 different numbers over 25 observations, so it looks like "
        "a measurement, not a column of categories (most categories would hold a "
        "single observation). To compare 'age' across the groups of 'g' (for "
        "example its mean in each group), use betweenstats(data, x='g', y='age'). "
        "To count these values as categories anyway, convert them to text first, "
        "for example data['age'].astype(str) in pandas."
    )
    # Eleven codes over a table with counts are categories: n is the total.
    codes = pd.DataFrame({"code": list(range(11)), "n": [5] * 11})
    assert read_table(codes, "code", counts="n").n == 55
    # Ten codes are categories whatever n is.
    assert read_table(pd.DataFrame({"code": list(range(10))}), "code").n == 10


def test_dates_are_told_to_become_text() -> None:
    data = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    with pytest.raises(TypeError, match="cannot be counted as categories as they are"):
        read_table(data, "d")


# --- paired -------------------------------------------------------------------------


def test_paired_needs_the_same_levels_in_the_same_order() -> None:
    same = pd.DataFrame({"a": ["no", "yes", "yes"], "b": ["yes", "yes", "no"]})
    table = read_table(same, "a", "b", paired=True)
    assert counts_of(table) == [[0, 1], [1, 1]]
    differ = pd.DataFrame({"a": ["no", "yes", "no"], "b": ["maybe", "yes", "no"]})
    with pytest.raises(ValueError, match="paired=True") as info:
        read_table(differ, "a", "b", paired=True)
    assert str(info.value) == (
        "paired=True: x='a' and y='b' are two measurements of the same units, so "
        "they need the same categories in the same order, and they differ ('maybe' "
        "only in 'b'), so the table of the pairs is not square and McNemar's test "
        "cannot run (ggstatsplot leaves out its subtitle silently). If every "
        "category is possible in both, declare both columns on the same "
        "categories, which keeps a category without rows as a level with a count "
        "of 0: categories = ['maybe', 'no', 'yes']; data['a'] = "
        "pd.Categorical(data['a'], categories=categories); data['b'] = "
        "pd.Categorical(data['b'], categories=categories)."
    )
    with pytest.raises(ValueError, match="paired=True") as info:
        read_table(pl.from_pandas(differ), "a", "b", paired=True)
    assert str(info.value).endswith(
        "data = data.with_columns(pl.col('a').cast(pl.String).cast(pl.Enum(['maybe', "
        "'no', 'yes'])), pl.col('b').cast(pl.String).cast(pl.Enum(['maybe', 'no', "
        "'yes'])))."
    )
    with pytest.raises(ValueError, match="paired=True") as info:
        read_table(pa.Table.from_pandas(differ), "a", "b", paired=True)
    assert ": data = data.to_pandas(); categories = ['maybe'" in str(info.value)
    # The fix the message gives works.
    cats = ["maybe", "no", "yes"]
    declared = differ.assign(
        a=pd.Categorical(differ["a"], categories=cats),
        b=pd.Categorical(differ["b"], categories=cats),
    )
    square = read_table(declared, "a", "b", paired=True)
    assert counts_of(square) == [[0, 0, 0], [1, 1, 0], [0, 0, 1]]
    order = pd.DataFrame(
        {
            "a": pd.Categorical(["no", "yes"], categories=["no", "yes"]),
            "b": pd.Categorical(["no", "yes"], categories=["yes", "no"]),
        }
    )
    with pytest.raises(ValueError, match="same categories in different orders"):
        read_table(order, "a", "b", paired=True)


def test_paired_counts_count_units() -> None:
    # ggstatsplot uncounts before pairing: each row is a combination of the two
    # measurements and its count the units that have it.
    data = pd.DataFrame(
        {
            "a": ["no", "no", "yes", "yes"],
            "b": ["no", "yes", "no", "yes"],
            "n": [14, 4, 12, 10],
        }
    )
    table = read_table(data, "a", "b", counts="n", paired=True)
    assert counts_of(table) == [[14, 4], [12, 10]]
    assert table.n == 40


def test_paired_numeric_codes_show_numbers() -> None:
    data = pd.DataFrame({"a": [0, 1, 1], "b": [1, 2, 1]})
    with pytest.raises(ValueError, match=r"categories = \[0, 1, 2\]; data\['a'\]"):
        read_table(data, "a", "b", paired=True)
    with pytest.raises(ValueError, match=r"pl.col\('a'\).cast\(pl.Int64\)"):
        read_table(pl.from_pandas(data), "a", "b", paired=True)


# --- notes --------------------------------------------------------------------------


Note = Callable[[TableData], list[str]]


def test_numbered_text_levels_get_a_note_without_order() -> None:
    data = pd.DataFrame({"week": ["week 10", "week 2", "week 1"], "y": ["p", "q", "p"]})
    table = read_table(data, "y", "week")
    notes = table_notes(table)
    assert notes == [
        "The levels of 'week' are text, sorted as text (as R sorts them), so 'week "
        "10' comes before 'week 2': 'week 1', 'week 10', 'week 2'. The analysis and "
        "the figure use this order. To order them by the numbers in their names, "
        "pass the column as a pandas Categorical (pd.Categorical(data['week'], "
        "categories=['week 1', 'week 2', 'week 10'])) or a polars Enum in that order."
    ]


def test_lookalike_levels_and_control_characters_are_noted() -> None:
    data = pd.DataFrame({"x\tname": ["a", "a ", "b\x07"], "y": ["p", "q", "p"]})
    notes = table_notes(read_table(data, "x\tname", "y"))
    assert any("'a' and 'a '" in n and "counted as separate" in n for n in notes)
    assert any("The level 'b\\x07'" in n for n in notes)
    assert any("The column name x='x\\tname'" in n for n in notes)


def test_table_is_a_read_only_int64_array() -> None:
    table = read_table(pd.DataFrame({"x": ["a", "b", "b"]}), "x")
    array = table.table()
    assert array.dtype == np.int64
    assert not array.flags.writeable
    assert array.tolist() == [[1], [2]]
