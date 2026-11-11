"""barstats' paired hints lead to a working call that analyses what was meant.

Each refusal of ``paired=True`` prints a statement; it is executed as printed on
the frame (bound to ``data``, with pandas as ``pd``, polars as ``pl`` and
pyarrow as ``pa``), and the call must then work with every row kept.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import inferplot as ip

DECLARE = re.compile(
    r": ((?:data = data\.to_pandas\(\); )?categories = .*|"
    r"data = data\.with_columns\(.*)\.$"
)


def _refusal(call: Any) -> str:
    with pytest.raises(ValueError) as caught:  # noqa: PT011 (the message is read)
        call()
    return str(caught.value)


def _run(statement: str, frame: Any) -> Any:
    space: dict[str, Any] = {"data": frame, "pd": pd, "pl": pl, "pa": pa}
    if isinstance(frame, pd.DataFrame):
        space["data"] = frame.copy()
    exec(statement, space)
    return space["data"]


def _frames() -> dict[str, tuple[Sequence[object], Sequence[object]]]:
    rng = np.random.default_rng(3)
    ints = [int(v) for v in rng.choice([1, 2], 40)]
    texts = [str(v) for v in rng.choice(["1", "2", "3"], 40)]
    bools = [bool(v) for v in rng.choice([True, False], 40)]
    bool_texts = [str(v) for v in rng.choice(["True", "False", "unknown"], 40)]
    return {
        "int vs text": (ints, texts),
        "bool vs text": (bools, bool_texts),
    }


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
@pytest.mark.parametrize("case", ["int vs text", "bool vs text"])
def test_declaring_both_columns_keeps_every_row(backend: str, case: str) -> None:
    xs, ys = _frames()[case]
    columns = {"x": xs, "y": ys}
    frame: Any = (
        pd.DataFrame(columns)
        if backend == "pandas"
        else pl.DataFrame(columns)
        if backend == "polars"
        else pa.table(columns)
    )
    text = _refusal(lambda: ip.analyze.barstats(frame, "x", "y", paired=True))
    found = DECLARE.search(text)
    assert found is not None, text
    data = _run(found.group(1), frame)
    result = ip.analyze.barstats(data, "x", "y", paired=True, bf_message=False)
    assert result.n_obs == len(xs)
    assert result.n_dropped == 0


def test_a_pandas_integer_column_with_a_missing_value() -> None:
    # A pandas integer column with a missing value is float: its labels are
    # "1" and "2", and the recipe must keep them.
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 1.0, 2.0, 1.0, 2.0, 1.0],
            "y": ["2", "1", "1", "3", "3", "1", "2", "1"],
        }
    )
    text = _refusal(lambda: ip.analyze.barstats(frame, "x", "y", paired=True))
    found = DECLARE.search(text)
    assert found is not None, text
    data = _run(found.group(1), frame)
    result = ip.analyze.barstats(data, "x", "y", paired=True, bf_message=False)
    assert result.n_obs == 7


# -- B6: paired=True on long data -------------------------------------------------

RESHAPE = re.compile(
    r": ((?:data = data\.to_pandas\(\); )?wide = .*?); (ip\.barstats\(wide, .*?\))\."
)


def _long(backend: str) -> Any:
    rng = np.random.default_rng(1)
    ids = np.arange(60)
    first = rng.choice(["yes", "no"], 60, p=[0.4, 0.6])
    second = rng.choice(["yes", "no"], 60, p=[0.6, 0.4])
    columns = {
        "id": np.r_[ids, ids].tolist(),
        "wave": ["w1"] * 60 + ["w2"] * 60,
        "ans": np.r_[first, second].tolist(),
    }
    if backend == "pandas":
        return pd.DataFrame(columns)
    if backend == "polars":
        return pl.DataFrame(columns)
    return pa.table(columns)


def _follow(message: str, frame: Any) -> Any:
    found = RESHAPE.search(message)
    assert found is not None, message
    space: dict[str, Any] = {"data": frame, "pd": pd, "pl": pl, "pa": pa, "ip": ip}
    exec(found.group(1), space)
    return eval(found.group(2), space)


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_long_paired_data_are_shown_the_reshape(backend: str) -> None:
    frame = _long(backend)
    text = _refusal(lambda: ip.barstats(frame, "ans", "wave", paired=True))
    assert "The data look long" in text
    assert "share no category" in text
    fig = _follow(text, frame)
    result = fig.result
    assert result.n_obs == 60  # one pair per unit
    assert result.test is not None
    assert result.test.name == "mcnemar"


@pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
def test_subject_id_is_shown_the_reshape(backend: str) -> None:
    frame = _long(backend)
    call: Any = ip.barstats
    with pytest.raises(TypeError) as caught:
        call(frame, "ans", "wave", paired=True, subject_id="id")
    text = str(caught.value)
    assert "subject_id= is not an argument of barstats()" in text
    assert "If the data are long" in text
    assert _follow(text, frame).result.n_obs == 60
