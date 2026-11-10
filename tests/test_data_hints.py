"""Every conversion a message prints works when executed as printed.

A review followed the hints literally and met another error: pandas.to_numeric
on text with "NA" entries, a polars cast of a Categorical, the "column of Python
objects" advice for a pandas sparse column, and a pandas-only recipe for dates
in a polars frame. Here each message's statement is executed on the frame (bound
to ``data``, with pandas, polars as ``pl`` and pyarrow imported), and the call
that failed must then work, with the entries that are not numbers missing.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute
import pytest

import inferplot as ip

STATEMENT = re.compile(
    r"(?:first|seconds|own unit|first date): ((?:data\[|data =).*?)\.(?=\s|$)"
)


def _run(message: str, frame: Any) -> Any:
    """Execute the statement the message prints; the frame it leaves."""
    found = STATEMENT.search(message)
    assert found is not None, message
    space: dict[str, Any] = {"data": frame, "pandas": pd, "pl": pl, "pyarrow": pa}
    if isinstance(frame, pd.DataFrame):
        space["data"] = frame.copy()
    exec(found.group(1), space)
    return space["data"]


def _error(call: Any) -> str:
    with pytest.raises(TypeError) as caught:
        call()
    return str(caught.value)


TEXT = ["1.5", "2.5", "", "4.0", "NA", None, " 6 ", "7.25"]


def _frame(backend: str, column: Any) -> Any:
    if backend == "pandas":
        return pd.DataFrame({"rt": column})
    if backend == "polars":
        return pl.DataFrame({"rt": column})
    return pa.table({"rt": column})


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_text_with_entries_that_are_not_numbers(backend: str) -> None:
    frame = _frame(backend, TEXT)
    message = _error(lambda: ip.histostats(frame, "rt"))
    assert "holds numbers stored as text (for example '1.5')" in message
    assert "The entries that are not numbers ('', 'NA') become missing values" in (
        message
    )
    fixed = _run(message, frame)
    result = ip.analyze.histostats(fixed, "rt")
    assert result.n_obs == 5
    assert sorted(result.values[0]) == [1.5, 2.5, 4.0, 6.0, 7.25]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_text_that_is_all_numbers(backend: str) -> None:
    frame = _frame(backend, ["1.5", "2", "3.5", "8"])
    message = _error(lambda: ip.histostats(frame, "rt"))
    assert "become missing" not in message
    assert ip.analyze.histostats(_run(message, frame), "rt").n_obs == 4


def _categories(backend: str) -> Any:
    digits = ["1", "2", "3", "2", "NA"]
    if backend == "pandas":
        return pd.DataFrame({"x": pd.Categorical(digits)})
    if backend == "polars":
        return pl.DataFrame({"x": pl.Series(digits, dtype=pl.Categorical)})
    if backend == "polars enum":
        return pl.DataFrame(
            {"x": pl.Series(digits, dtype=pl.Enum(["1", "2", "3", "NA"]))}
        )
    return pa.table({"x": pa.array(digits).dictionary_encode()})


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars enum", "pyarrow"])
def test_categories_of_digits(backend: str) -> None:
    frame = _categories(backend)
    message = _error(lambda: ip.histostats(frame, "x"))
    assert "categories whose names are numbers" in message
    assert "('NA')" in message
    result = ip.analyze.histostats(_run(message, frame), "x")
    assert sorted(result.values[0]) == [1.0, 2.0, 2.0, 3.0]


def test_pandas_sparse_column() -> None:
    frame = pd.DataFrame({"x": pd.arrays.SparseArray([0.0, 0.0, 1.0, 2.0, 3.0])})
    message = _error(lambda: ip.histostats(frame, "x"))
    assert message.startswith("x='x' is a pandas sparse column (Sparse[float64")
    assert "Python objects" not in message
    result = ip.analyze.histostats(_run(message, frame), "x")
    assert sorted(result.values[0]) == [0.0, 0.0, 1.0, 2.0, 3.0]
    # betweenstats' y gets the same advice.
    between = pd.DataFrame(
        {
            "g": ["a", "b"] * 3,
            "y": pd.arrays.SparseArray([0.0, 1.0, 0.0, 2.0, 3.0, 0.0]),
        }
    )
    message = _error(lambda: ip.betweenstats(between, "g", "y"))
    assert "is a pandas sparse column" in message


def test_pandas_object_numbers() -> None:
    frame = pd.DataFrame({"x": pd.Series([1.5, 2.5, None, 4.0], dtype=object)})
    message = _error(lambda: ip.histostats(frame, "x"))
    assert "column of Python objects" in message
    assert ip.analyze.histostats(_run(message, frame), "x").n_obs == 3


def _dates(backend: str) -> Any:
    days = [
        datetime.datetime(2024, 1, 1) + datetime.timedelta(days=3 * i) for i in range(6)
    ]
    if backend == "pandas":
        return pd.DataFrame({"d": pd.to_datetime(days)})
    if backend == "polars":
        return pl.DataFrame({"d": days})
    if backend == "polars date":
        return pl.DataFrame({"d": [d.date() for d in days]})
    return pa.table({"d": pa.array(days)})


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars date", "pyarrow"])
def test_dates_get_their_library_recipe(backend: str) -> None:
    frame = _dates(backend)
    message = _error(lambda: ip.histostats(frame, "d"))
    assert "the days since the first date" in message
    if backend != "pandas":
        assert "dt.days" not in message
    result = ip.analyze.histostats(_run(message, frame), "d")
    assert sorted(result.values[0]) == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0]


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_durations(backend: str) -> None:
    spans = [datetime.timedelta(seconds=s) for s in (5, 9, 12, 30)]
    frame = (
        pd.DataFrame({"d": spans})
        if backend == "pandas"
        else pl.DataFrame({"d": spans})
    )
    message = _error(lambda: ip.histostats(frame, "d"))
    result = ip.analyze.histostats(_run(message, frame), "d")
    assert sorted(result.values[0]) == [5.0, 9.0, 12.0, 30.0]


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_between_y_numbers_as_text(backend: str) -> None:
    columns = {
        "g": ["a", "b", "a", "b", "a", "b"],
        "rt": ["1", "2", "NA", "4", "5", "6"],
    }
    frame = (
        pd.DataFrame(columns)
        if backend == "pandas"
        else pl.DataFrame(columns)
        if backend == "polars"
        else pa.table(columns)
    )
    message = _error(lambda: ip.analyze.betweenstats(frame, "g", "rt"))
    result = ip.analyze.betweenstats(_run(message, frame), "g", "rt")
    assert result.n_obs == 5
    assert np.isfinite(result.test.statistic if result.test else 0.0)
