"""Paired data in wide format (one row per subject, one column per condition)
passed to withinstats: the message says so and shows the reshape to long format,
in the frame's own library; betweenstats does not advise turning a measurement
into groups (review of phase 3, finding 4)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import inferplot as ip


def _wide() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pre = rng.normal(70, 10, 12).round(1)
    return pd.DataFrame(
        {"id": range(1, 13), "pre": pre, "post": (pre + rng.normal(3, 4, 12)).round(1)}
    )


def _reshape(message: str) -> str:
    found = re.search(r"Reshape [^:]*: long = (data\.[^\n]*?\)), then", message)
    assert found is not None, message
    return found.group(1)


def test_pandas_wide_data_get_the_melt_call() -> None:
    wide = _wide()
    with pytest.raises(TypeError) as raised:
        ip.analyze.withinstats(wide, "pre", "post", subject_id="id")
    message = str(raised.value)
    assert "very likely in wide format" in message
    assert "scatterstats" not in message
    assert "convert" not in message
    assert (
        "long = data.melt(id_vars='id', value_vars=['pre', 'post'], "
        "var_name='condition', value_name='value'), then withinstats(long, "
        "'condition', 'value', subject_id='id')." in message
    )
    # The call shown works.
    long = eval(_reshape(message), {"data": wide})
    result = ip.analyze.withinstats(long, "condition", "value", subject_id="id")
    assert result.levels == ("post", "pre")
    assert result.n_obs == 12


def test_polars_and_pyarrow_get_their_own_reshape() -> None:
    wide = pl.from_pandas(_wide())
    with pytest.raises(TypeError) as raised:
        ip.analyze.withinstats(wide, "pre", "post", subject_id="id")
    code = _reshape(str(raised.value))
    assert code.startswith("data.unpivot(index='id', on=['pre', 'post']")
    long = eval(code, {"data": wide})
    assert (
        ip.analyze.withinstats(long, "condition", "value", subject_id="id").n_obs == 12
    )
    table = pa.Table.from_pandas(_wide())
    with pytest.raises(TypeError) as raised:
        ip.analyze.withinstats(table, "pre", "post", subject_id="id")
    assert _reshape(str(raised.value)).startswith("data.to_pandas().melt(")


def test_whole_number_scores_in_wide_format_are_named_too() -> None:
    wide = _wide()
    wide = wide.assign(pre=wide["pre"].round().astype(int))
    with pytest.raises(ValueError, match="very likely in wide format") as raised:
        ip.analyze.withinstats(wide, "pre", "post", subject_id="id")
    assert "data.melt(id_vars='id'" in str(raised.value)


def test_betweenstats_does_not_turn_a_measurement_into_groups() -> None:
    with pytest.raises(TypeError) as raised:
        ip.analyze.betweenstats(_wide(), "pre", "post")
    message = str(raised.value)
    assert "looks like a measurement, not a column of groups" in message
    assert "convert them to text" not in message
    assert "scatterstats" in message
    # Repeated codes that are not whole (doses) may still be grouped as text.
    doses = pd.DataFrame({"dose": [0.5, 1.5, 2.5] * 4, "y": np.arange(12.0)})
    with pytest.raises(TypeError, match="convert them to text first"):
        ip.analyze.betweenstats(doses, "dose", "y")
