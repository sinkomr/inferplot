"""``read_pair``: two numeric columns for scatterstats (SPEC.md section 3).

A row is analyzed when both x and y are present, in row order, on pandas, polars and
pyarrow alike; every ggscatterstats fixture records the pairs R analyzed. The
``label_var`` column labels the analyzed points as text and never drops a row.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from inferplot._data import ColumnNotFoundError, data_notes, read_pair
from tests.oracle_fixtures import OracleCase, oracle_params, read_dataset

BACKENDS = ("pandas", "polars", "pyarrow")


def make(backend: str, columns: dict[str, Any]) -> Any:
    """A frame of ``backend`` from plain Python lists (None is missing)."""
    if backend == "pandas":
        return pd.DataFrame(columns)
    if backend == "polars":
        return pl.DataFrame(columns, strict=False)
    return pa.table(columns)


def message(error: type[Exception], thunk: Any) -> str:
    with pytest.raises(error) as info:
        thunk()
    assert type(info.value) is error
    return str(info.value)


# --- oracle fixtures ---------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", oracle_params("ggscatterstats"))
def test_reads_the_pairs_r_analyzed(case: OracleCase, backend: str) -> None:
    label_var = case.args.get("label.var")
    pair = read_pair(
        read_dataset(case, backend),  # type: ignore[arg-type]
        str(case.args["x"]),
        str(case.args["y"]),
        label_var=None if label_var is None else str(label_var),
    )
    assert pair.x_values.tolist() == case.analyzed_values("x")
    assert pair.y_values.tolist() == case.analyzed_values("y")
    assert pair.n_obs == case.analyzed_data["n"]
    assert pair.n_dropped == case.analyzed_data["rows_dropped"]
    assert not pair.x_values.flags.writeable
    assert not pair.y_values.flags.writeable
    if label_var is None:
        assert pair.labels == ()
        assert pair.label_var is None
        return
    # The labels ggstatsplot drew: every analyzed point's label.var, or, with
    # label.expression, the points it selects (inferplot: the others missing).
    ((drawn),) = case.layer_data("GeomLabelRepel")
    assert pair.label_var == label_var
    assert len(pair.labels) == pair.n_obs
    drawn_labels = [str(row[label_var]) for row in drawn]
    if "label.expression" in case.args:
        assert set(drawn_labels) < set(pair.labels)  # mtcars: 9 of 32
    else:
        assert list(pair.labels) == drawn_labels


@pytest.mark.parametrize("backend", BACKENDS)
def test_complete_pairs_in_row_order(backend: str) -> None:
    frame = make(
        backend,
        {
            "a": [1.0, None, 3.0, 4.0, float("nan"), 6.0],
            "b": [10.0, 20.0, None, 40.0, None, 60.0],
        },
    )
    pair = read_pair(frame, "a", "b")
    assert pair.x_values.tolist() == [1.0, 4.0, 6.0]
    assert pair.y_values.tolist() == [10.0, 40.0, 60.0]
    assert (pair.n_dropped, pair.n_missing_x, pair.n_missing_y) == (3, 2, 2)
    assert data_notes(pair) == [
        "3 rows were left out because 'a' or 'b' is missing (2 without 'a', 2 "
        "without 'b'); 3 complete pairs remain."
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_fewer_than_three_pairs_raise(backend: str) -> None:
    frame = make(backend, {"a": [1.0, 2.0, None], "b": [3.0, 1.0, 2.0]})
    text = message(ValueError, lambda: read_pair(frame, "a", "b"))
    assert text == (
        "scatterstats needs at least 3 complete pairs (rows with both x='a' and "
        "y='b' present) to test a correlation; found 2 after dropping 1 row with a "
        "missing value (data has 3 rows)."
    )
    full = make(backend, {"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    assert read_pair(full, "a", "b").n_obs == 3


def test_the_same_column_twice_is_refused() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    text = message(ValueError, lambda: read_pair(frame, "a", "a"))
    assert text == (
        "x and y must be different columns; both are 'a'. A column's correlation "
        "with itself is 1 by construction."
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_infinite_values_raise_naming_the_column(backend: str) -> None:
    frame = make(backend, {"a": [1.0, 2.0, 3.0], "b": [1.0, float("inf"), 2.0]})
    text = message(ValueError, lambda: read_pair(frame, "a", "b"))
    assert text.startswith("y='b' has 1 infinite value(s) (row positions 1)")
    frame = make(backend, {"a": [float("-inf"), 2.0, 3.0], "b": [1.0, 3.0, 2.0]})
    assert message(ValueError, lambda: read_pair(frame, "a", "b")).startswith("x='a'")
    # An infinite value in a row that is dropped anyway is not analyzed.
    frame = make(
        backend, {"a": [float("inf"), 2.0, 3.0, 4.0], "b": [None, 3.0, 2.0, 1.0]}
    )
    assert read_pair(frame, "a", "b").n_obs == 3


def test_text_points_to_betweenstats_with_the_other_column() -> None:
    frame = pd.DataFrame({"g": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    text = message(TypeError, lambda: read_pair(frame, "g", "v"))
    assert text == (
        "x='g' holds text or categories, not numbers. scatterstats() relates two "
        "numeric columns. To compare 'v' across the groups of 'g', use "
        "betweenstats(data, x='g', y='v'); to relate two columns of categories, use "
        "barstats(data, x=..., y=...)."
    )
    # A text y names the swap that works.
    text = message(TypeError, lambda: read_pair(frame, "v", "g"))
    assert text.startswith("y='g' holds text or categories")
    assert "betweenstats(data, x='g', y='v')" in text
    both = pd.DataFrame({"g": ["a", "b", "c"], "h": ["x", "y", "z"]})
    text = message(TypeError, lambda: read_pair(both, "g", "h"))
    assert "betweenstats(data, x='g', y='<numeric column>')" in text


def test_booleans_and_dates_are_refused() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [True, False, True],
            "d": [datetime.date(2024, 1, d) for d in (1, 2, 3)],
        }
    )
    assert "True/False" in message(TypeError, lambda: read_pair(frame, "a", "b"))
    pl_frame = pl.from_pandas(frame)
    assert "dates or times" in message(TypeError, lambda: read_pair(pl_frame, "d", "a"))


# --- labels ------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_labels_of_the_analyzed_points(backend: str) -> None:
    frame = make(
        backend,
        {
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [1.0, 3.0, 2.0, 5.0, 4.0],
            "name": ["p", None, "r", "", "t"],
        },
    )
    pair = read_pair(frame, "a", "b", label_var="name")
    # Row 3 is dropped (a missing); a missing label keeps its row, unlabelled.
    assert pair.labels == ("p", "", "", "t")
    assert pair.label_var == "name"


def test_labels_of_numbers_categories_and_booleans_are_text() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 1.0, 2.0],
            "i": [1, 20, 300],
            "f": [2.5, 4.0, float("nan")],
            "c": pd.Categorical(["x", None, "z"]),
            "t": [True, False, True],
        }
    )
    assert read_pair(frame, "a", "b", label_var="i").labels == ("1", "20", "300")
    assert read_pair(frame, "a", "b", label_var="f").labels == ("2.5", "4", "")
    assert read_pair(frame, "a", "b", label_var="c").labels == ("x", "", "z")
    assert read_pair(frame, "a", "b", label_var="t").labels == ("True", "False", "True")
    # A label column may be x or y itself: its values as text.
    assert read_pair(frame, "a", "b", label_var="a").labels == ("1", "2", "3")
    polars = pl.from_pandas(frame[["a", "b", "c"]])
    assert read_pair(polars, "a", "b", label_var="c").labels == ("x", "", "z")
    arrow = pa.table(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 1.0, 2.0],
            "c": pa.array(["x", None, "z"]).dictionary_encode(),
        }
    )
    assert read_pair(arrow, "a", "b", label_var="c").labels == ("x", "", "z")


def test_a_label_column_must_exist() -> None:
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0], "model": list("xyz")}
    )
    text = message(
        ColumnNotFoundError, lambda: read_pair(frame, "a", "b", label_var="Model")
    )
    assert text.startswith(
        "label_var='Model' is not a column of data. Did you mean 'model'?"
    )
    text = message(TypeError, lambda: read_pair(frame, "a", "b", label_var=3))  # type: ignore[arg-type]
    assert text == "label_var must be a column name (a string), not 3."


def test_control_characters_in_either_column_name_are_noted() -> None:
    frame = pd.DataFrame({"a\x07": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 2.0]})
    notes = data_notes(read_pair(frame, "a\x07", "b"))
    assert len(notes) == 1
    assert notes[0].startswith("The column name x='a\\x07' holds control characters")
