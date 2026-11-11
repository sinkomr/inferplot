"""barstats' conversion hints, in the frame's own library, executed as printed.

The natural-order note, the many-codes refusal and the counts-stored-as-text
refusal printed pandas code for polars and pyarrow frames, and "1,000" failed
the pandas recipe itself (review B14). Each statement is executed on the frame
(bound to ``data``, with pandas as ``pd`` and ``pandas``, polars as ``pl`` and
pyarrow as ``pyarrow``), and the call must then work as meant.
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

BACKENDS = ["pandas", "polars", "pyarrow"]


def _frame(backend: str, columns: dict[str, list[Any]]) -> Any:
    if backend == "pandas":
        return pd.DataFrame(columns)
    if backend == "polars":
        return pl.DataFrame(columns, strict=False)
    return pyarrow.table(columns)


def _run(statement: str, frame: Any) -> Any:
    space: dict[str, Any] = {
        "data": frame.copy() if isinstance(frame, pd.DataFrame) else frame,
        "pd": pd,
        "pandas": pd,
        "pl": pl,
        "pyarrow": pyarrow,
    }
    exec(statement, space)
    return space["data"]


def _statement(text: str, after: str) -> str:
    found = re.search(re.escape(after) + r" (.*)\.$", text)
    assert found is not None, text
    return str(found.group(1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_natural_order_note(backend: str) -> None:
    rng = np.random.default_rng(1)
    columns = {
        "score": [str(v) for v in rng.choice(["1", "2", "10"], 90)],
        "g": [str(v) for v in rng.choice(["a", "b"], 90)],
    }
    frame = _frame(backend, columns)
    result = ip.analyze.barstats(frame, "score", "g", bf_message=False)
    assert result.levels == ("1", "10", "2")
    (note,) = [w for w in result.warnings if "sorted as text" in w]
    assert "pd." not in note or backend == "pandas"
    data = _run(_statement(note, "in that order:"), frame)
    again = ip.analyze.barstats(data, "score", "g", bf_message=False)
    assert again.levels == ("1", "2", "10")
    assert again.counts == tuple(result.counts[i] for i in (0, 2, 1))
    assert not [w for w in again.warnings if "sorted as text" in w]


@pytest.mark.parametrize("backend", BACKENDS)
def test_many_codes(backend: str) -> None:
    rng = np.random.default_rng(2)
    years = [int(v) for v in rng.integers(2000, 2015, 20)]
    groups = [str(v) for v in rng.choice(["a", "b"], 20)]
    frame = _frame(backend, {"year": years, "g": groups})
    with pytest.raises(TypeError) as caught:
        ip.analyze.barstats(frame, "year", "g")
    text = str(caught.value)
    assert "pandas" not in text or backend == "pandas"
    data = _run(_statement(text, "convert them to text first:"), frame)
    result = ip.analyze.barstats(data, "year", "g", results_subtitle=False)
    assert result.n_obs == 20
    assert set(result.levels) == {str(y) for y in years}


def test_many_codes_with_a_missing_value_in_pandas() -> None:
    years = [2000.0 + v for v in range(12)] + [None] + [2000.0 + v for v in range(7)]
    frame = pd.DataFrame({"year": years, "g": list("ab") * 10})
    with pytest.raises(TypeError) as caught:
        ip.analyze.barstats(frame, "year", "g")
    data = _run(_statement(str(caught.value), "convert them to text first:"), frame)
    result = ip.analyze.barstats(data, "year", "g", results_subtitle=False)
    assert result.n_obs == 19
    assert "2000" in result.levels
    assert "2000.0" not in result.levels


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "counts", [["10", "20", "30", "5"], ["1,000", "2,000", "30", "5"]]
)
def test_counts_stored_as_text(backend: str, counts: list[str]) -> None:
    frame = _frame(backend, {"x": list("aabb"), "y": list("uvuv"), "n": counts})
    with pytest.raises(TypeError) as caught:
        ip.analyze.barstats(frame, "x", "y", counts="n")
    text = str(caught.value)
    assert "pandas" not in text or backend == "pandas"
    data = _run(_statement(text, "convert it first:"), frame)
    result = ip.analyze.barstats(data, "x", "y", counts="n", bf_message=False)
    assert result.n_obs == sum(int(c.replace(",", "")) for c in counts)


# -- B10: tables of counts given without counts= -----------------------------------

AGGREGATED = {
    "smoke": ["never", "former", "current"] * 2,
    "sex": ["F"] * 3 + ["M"] * 3,
    "n": [34, 18, 9, 28, 18, 13],
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_table_without_counts_is_named(backend: str) -> None:
    frame = _frame(backend, AGGREGATED)
    result = ip.analyze.barstats(frame, "smoke", "sex", bf_message=False)
    assert result.n_obs == 6
    (note,) = [w for w in result.warnings if "occurs in one row only" in w]
    assert "pass counts='n'" in note
    assert "(n = 6)" in note
    found = re.search(r": (barstats\(data, .*?\))\.", note)
    assert found is not None, note
    fig = eval(found.group(1), {"barstats": ip.barstats, "data": frame})
    assert fig.result.n_obs == 120


def test_raw_data_get_no_counts_note() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        {
            "x": rng.choice(["a", "b"], 50),
            "y": rng.choice(["u", "v"], 50),
            "age": rng.integers(20, 60, 50),
        }
    )
    result = ip.analyze.barstats(frame, "x", "y", bf_message=False)
    assert not [w for w in result.warnings if "occurs in one row only" in w]


WIDE = {
    "region": [f"r{i}" for i in range(8)],
    "yes": [12, 30, 22, 12, 30, 41, 22, 9],
    "no": [20, 20, 31, 20, 9, 31, 12, 30],
}
LONG_RECIPE = re.compile(
    r": ((?:data = data\.to_pandas\(\); )?long = .*?); (ip\..*)\.$"
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("x", "y"), [("region", "yes"), ("yes", "region")])
def test_a_wide_table_of_counts_is_shown_the_reshape(
    backend: str, x: str, y: str
) -> None:
    frame = _frame(backend, WIDE)
    with pytest.raises(ValueError, match="wide table of counts") as caught:
        ip.analyze.barstats(frame, x, y)
    text = str(caught.value)
    found = LONG_RECIPE.search(text)
    assert found is not None, text
    space: dict[str, Any] = {"data": frame, "ip": ip, "pd": pd, "pl": pl}
    exec(found.group(1), space)
    fig = eval(found.group(2), space)
    assert fig.result.n_obs == sum(WIDE["yes"]) + sum(WIDE["no"])
    assert set(fig.result.levels) == {"yes", "no"}


def test_counts_that_sum_to_100_are_noted() -> None:
    frame = pd.DataFrame(
        {"x": ["a", "b", "a", "b"], "y": ["u", "u", "v", "v"], "pct": [30, 20, 15, 35]}
    )
    result = ip.analyze.barstats(frame, "x", "y", counts="pct", bf_message=False)
    (note,) = [w for w in result.warnings if "sum to exactly 100" in w]
    assert "percentages" in note
