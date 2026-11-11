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
