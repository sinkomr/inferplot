"""``read_one``: one numeric column for histostats (SPEC.md section 3).

The analyzed values are the non-missing values of x in row order, on pandas, polars
and pyarrow alike; every gghistostats fixture records the values R analyzed. Columns
that are not measurements are refused with a message that names the function that
fits them.
"""

from __future__ import annotations

import datetime
import decimal
import math
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from inferplot._data import ColumnNotFoundError, data_notes, read_one
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
@pytest.mark.parametrize("case", oracle_params("gghistostats"))
def test_reads_the_values_r_analyzed(case: OracleCase, backend: str) -> None:
    # gghistostats drops the rows without x (tidyr::drop_na on x alone) and keeps
    # the others in file order: bugs_long desire, 367 of 372.
    one = read_one(read_dataset(case, backend), str(case.args["x"]))  # type: ignore[arg-type]
    assert one.values.tolist() == case.analyzed_values("x")
    assert one.n_obs == case.analyzed_data["n"]
    assert one.n_dropped == case.analyzed_data["rows_dropped"]
    assert one.values.dtype == np.float64
    assert not one.values.flags.writeable


# --- missing values ----------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_values_are_dropped_and_counted(backend: str) -> None:
    frame = make(backend, {"v": [1.0, None, 3.0, float("nan"), 5.0]})
    one = read_one(frame, "v")
    assert one.values.tolist() == [1.0, 3.0, 5.0]
    assert one.n_dropped == 2
    assert data_notes(one) == [
        "2 rows were left out because 'v' is missing; 3 values remain."
    ]


def test_pandas_nullable_integers_and_na() -> None:
    frame = pd.DataFrame({"v": pd.array([4, None, 6], dtype="Int64")})
    one = read_one(frame, "v")
    assert one.values.tolist() == [4.0, 6.0]
    assert one.n_dropped == 1
    frame = pd.DataFrame({"v": pd.array([1.5, None, 2.5], dtype="Float64")})
    assert read_one(frame, "v").values.tolist() == [1.5, 2.5]


@pytest.mark.parametrize("backend", BACKENDS)
def test_integers_and_decimals_are_read_as_doubles(backend: str) -> None:
    one = read_one(make(backend, {"v": [1, 2, 4]}), "v")
    assert one.values.tolist() == [1.0, 2.0, 4.0]
    if backend == "pyarrow":
        values = [decimal.Decimal("1.25"), decimal.Decimal("2.50"), None]
        table = pa.table({"v": pa.array(values, pa.decimal128(5, 2))})
        decimals = read_one(table, "v")
        assert decimals.values.tolist() == [1.25, 2.5]
        assert decimals.n_dropped == 1


def test_large_integers_a_double_holds_are_read_others_refused() -> None:
    big = 2**60
    one = read_one(pd.DataFrame({"v": [big, big + 2**10, big + 2**11]}), "v")
    assert one.values.tolist() == [float(big), float(big + 2**10), float(big + 2**11)]
    text = message(
        ValueError, lambda: read_one(pd.DataFrame({"v": [big + 1, big]}), "v")
    )
    assert text.startswith("x='v' holds integers too large to analyze exactly")
    assert "x - x.min()" in text


@pytest.mark.parametrize("backend", BACKENDS)
def test_infinite_values_raise(backend: str) -> None:
    frame = make(backend, {"v": [1.0, float("inf"), 2.0, None, float("-inf")]})
    text = message(ValueError, lambda: read_one(frame, "v"))
    assert text == (
        "x='v' has 2 infinite value(s) (row positions 1, 4); remove them or replace "
        "them with missing values."
    )


# --- too little data ---------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_fewer_than_two_values_raise(backend: str) -> None:
    text = message(ValueError, lambda: read_one(make(backend, {"v": [1.0, None]}), "v"))
    assert text == (
        "histostats needs at least 2 values of x='v' to test against test_value; "
        "found 1 after dropping 1 row with a missing value (data has 2 rows)."
    )
    assert read_one(make(backend, {"v": [1.0, 2.0]}), "v").n_obs == 2


def test_an_empty_frame_raises() -> None:
    empty = pd.DataFrame({"v": pd.Series([], dtype="float64")})
    text = message(ValueError, lambda: read_one(empty, "v"))
    assert "found none (data has 0 rows)" in text
    text = message(ColumnNotFoundError, lambda: read_one(pd.DataFrame(), "v"))
    assert text == "x='v' is not a column of data. Available columns: (none)."


# --- columns that are not measurements ---------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_text_points_to_betweenstats_and_barstats(backend: str) -> None:
    frame = make(backend, {"g": ["a", "b", "a"]})
    text = message(TypeError, lambda: read_one(frame, "g"))
    assert text == (
        "x='g' holds text or categories, not numbers. histostats() compares one "
        "numeric column with test_value. To compare a numeric column across the "
        "groups of 'g', use betweenstats(data, x='g', y='<numeric column>'); to "
        "count its categories, use barstats(data, x='g')."
    )


def test_categories_are_refused_like_text() -> None:
    frames = [
        pd.DataFrame({"g": pd.Categorical(["a", "b"])}),
        pd.DataFrame({"g": pd.Categorical([1.5, 2.5])}),
        pl.DataFrame({"g": ["a", "b"]}).with_columns(pl.col("g").cast(pl.Categorical)),
        pl.DataFrame({"g": ["a", "b"]}).with_columns(
            pl.col("g").cast(pl.Enum(["a", "b"]))
        ),
        pa.table({"g": pa.array(["a", "b"]).dictionary_encode()}),
    ]
    for frame in frames:
        text = message(TypeError, lambda frame=frame: read_one(frame, "g"))
        assert "holds text or categories, not numbers" in text, text


@pytest.mark.parametrize("backend", BACKENDS)
def test_booleans_are_refused(backend: str) -> None:
    text = message(
        TypeError, lambda: read_one(make(backend, {"b": [True, False]}), "b")
    )
    assert text.startswith("x='b' holds True/False values, which are not measurements.")
    assert "barstats(data, x='b')" in text


@pytest.mark.parametrize("backend", BACKENDS)
def test_numbers_stored_as_text_are_told_to_convert(backend: str) -> None:
    text = message(TypeError, lambda: read_one(make(backend, {"v": ["1.5", "2"]}), "v"))
    assert "holds numbers stored as text (for example '1.5')" in text
    assert "pandas.to_numeric(data['v'])" in text


def test_an_object_column_of_numbers_is_told_to_convert() -> None:
    frame = pd.DataFrame({"v": pd.Series([1.5, 2.5, None], dtype=object)})
    text = message(TypeError, lambda: read_one(frame, "v"))
    assert "holds numbers in a column of Python objects" in text
    assert "pandas.to_numeric(data['v'])" in text


def test_dates_are_refused() -> None:
    frames = [
        pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-02-01"])}),
        pl.DataFrame({"d": [datetime.date(2024, 1, 1), datetime.date(2024, 2, 1)]}),
        pa.table({"d": pa.array([datetime.date(2024, 1, 1)] * 2)}),
        pd.DataFrame({"d": pd.to_timedelta([1, 2], unit="D")}),
    ]
    for frame in frames:
        text = message(TypeError, lambda frame=frame: read_one(frame, "d"))
        assert "It holds dates or times" in text, text


def test_column_names_get_a_did_you_mean_and_types_are_named() -> None:
    frame = pd.DataFrame({"length": [1.0, 2.0]})
    text = message(ColumnNotFoundError, lambda: read_one(frame, "Length"))
    assert "Did you mean 'length'?" in text
    text = message(TypeError, lambda: read_one(frame, frame["length"]))  # type: ignore[arg-type]
    assert text == (
        "x must be a column name (a string), not a pandas.Series: pass the column's "
        "name, for example x='length'."
    )


def test_lazy_frames_and_arrays_are_refused() -> None:
    lazy = pl.DataFrame({"v": [1.0, 2.0]}).lazy()
    assert ".collect()" in message(TypeError, lambda: read_one(lazy, "v"))
    assert "long format" in message(TypeError, lambda: read_one([1.0, 2.0], "v"))


def test_control_characters_in_the_column_name_are_noted() -> None:
    one = read_one(pd.DataFrame({"a\tb": [1.0, 2.0]}), "a\tb")
    assert data_notes(one) == [
        "The column name x='a\\tb' holds control characters (U+0009), which the "
        "figure draws a tab as a space in the axis label; the result keeps the name "
        "as given."
    ]


def test_values_keep_row_order_and_exact_doubles() -> None:
    values = [0.1 + 0.2, -0.0, 5e-324, 1.7976931348623157e308, 3.0]
    one = read_one(pd.DataFrame({"v": values}), "v")
    assert one.values.tolist() == values
    assert math.copysign(1.0, one.values[1]) == -1.0
