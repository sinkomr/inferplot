"""The data boundary (``inferplot._data``): frames in, NumPy arrays out (SPEC.md §3).

Each test guards one behaviour or one version-specific pitfall; the tests also run at
the dependency floors (DECISIONS.md D22) in the lowest-direct CI job:

* pyarrow 14: a row pointing at a null dictionary entry is not null in ``is_null()``
  (``test_dictionary_nulls``);
* pandas 2: a nullable float column holds NA and NaN separately
  (``test_pandas_nullable_float_na_and_nan``, ``test_null_and_nan_both_missing``);
* pandas 3: ``to_numpy()`` is a read-only view under copy-on-write;
* pandas 2.1: nullable and Arrow-backed columns give object arrays holding ``<NA>``
  (``test_pandas_nullable_and_arrow_dtypes``);
* pandas ``ArrowDtype`` dictionary columns have no ``.cat`` accessor in narwhals
  (``test_pandas_arrow_dictionary_x_rejected_clearly``);
* polars ``Categorical`` order changed in polars 1.32
  (``test_polars_categorical_is_sorted_not_encounter_order``).
"""

from __future__ import annotations

import csv
import decimal
import math
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from inferplot._data import ColumnNotFoundError, level_label, read_between
from tests.oracle_fixtures import (
    DATASET_DIR,
    OracleCase,
    cases_for,
    fixture_paths,
    r_level_label,
    read_dataset,
    require_cases,
)
from tests.tolerances import assert_close

BACKENDS = ("pandas", "polars", "pyarrow")
CASES = require_cases(cases_for("ggbetweenstats"), "ggbetweenstats")
# Cases where ggstatsplot itself failed (declared in the fixture): the data were read
# and counted all the same, so their levels and counts are checked too.
FAILED = require_cases(
    cases_for("ggbetweenstats", failed=True), "ggbetweenstats (failed in R)"
)


def make(backend: str, columns: dict[str, Any]) -> Any:
    """A frame of ``backend`` from plain Python lists (None is missing)."""
    if backend == "pandas":
        return pd.DataFrame(columns)
    if backend == "polars":
        return pl.DataFrame(columns, strict=False)
    return pa.table(columns)


# --- oracle fixtures ---------------------------------------------------------------


def _csv_missing(case: OracleCase) -> tuple[int, int, int]:
    """Rows missing x, missing y, and missing either, counted from the CSV text."""
    x, y = str(case.args["x"]), str(case.args["y"])
    with (DATASET_DIR / case.dataset_file).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mx = sum(r[x] == "" for r in rows)
    my = sum(r[y] == "" for r in rows)
    either = sum(r[x] == "" or r[y] == "" for r in rows)
    return mx, my, either


def _check_levels_and_counts(case: OracleCase, backend: str) -> Any:
    frame = read_dataset(case, backend)  # type: ignore[arg-type]
    x = str(case.args["x"])
    out = read_between(frame, x, str(case.args["y"]))
    assert out.levels == case.analysis_levels
    assert out.counts() == case.n_per_level
    mx, my, either = _csv_missing(case)
    assert (out.n_missing_x, out.n_missing_y, out.n_dropped) == (mx, my, either)
    assert out.n_obs + out.n_dropped == case.raw["dataset"]["rows"]
    declared = case.factor_levels.get(x)
    if declared is None:
        assert out.level_source == "sorted"
        assert out.empty_levels == ()
    else:
        # Declared levels keep their order; a declared level without rows is
        # dropped from the analysis and reported (edge_empty_level).
        assert out.level_source == "declared"
        assert out.empty_levels == tuple(
            level for level in declared if level not in case.analysis_levels
        )
        assert out.levels == tuple(
            level for level in declared if level in case.analysis_levels
        )
    return out


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", FAILED, ids=[c.name for c in FAILED])
def test_failed_fixture_levels_and_counts(case: OracleCase, backend: str) -> None:
    assert case.expected_error is not None
    _check_levels_and_counts(case, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_fixture_levels_and_counts(case: OracleCase, backend: str) -> None:
    out = _check_levels_and_counts(case, backend)
    # ggstatsplot's own centrality label layer: its level order, n, SD, min and max.
    (labels,) = case.layer_data("GeomLabelRepel")
    x = str(case.args["x"])
    assert tuple(str(row[x]) for row in labels) == out.levels
    for row, level, values in zip(labels, out.levels, out.values, strict=True):
        assert row["n.obs"] == values.size
        # statsExpressions' .prettyNum: a comma every three digits from 1,000 on.
        assert row["n.expression"] == f"{level}\n(n = {values.size:,})"
        assert row["min"] == values.min()
        assert row["max"] == values.max()
        if "std.dev" in row:  # the bayes label layer has none
            sd = float(np.std(values, ddof=1))
            assert_close(
                "closed_form", sd, row["std.dev"], what=f"{case.name} {level} SD"
            )
        if case.type == "parametric":
            assert_close(
                "closed_form",
                float(values.mean()),
                row[case.args["y"]],
                what=f"{case.name} {level} mean",
            )
        assert values.dtype == np.float64
        assert not values.flags.writeable


def test_fixture_selection_is_complete() -> None:
    # Every ggbetweenstats fixture is read above: completed ones with ggstatsplot's
    # label layer, failed ones for their levels and counts.
    names = {p.stem for p in fixture_paths() if p.stem.startswith("ggbetweenstats-")}
    assert {c.name for c in CASES} | {c.name for c in FAILED} == names
    assert not {c.name for c in CASES} & {c.name for c in FAILED}


# --- what data may be --------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [np.zeros((3, 2)), [1, 2], {"g": ["a"], "y": [1.0]}, ("a", 1), None, "data.csv"],
    ids=["ndarray", "list", "dict", "tuple", "None", "str"],
)
def test_non_frames_rejected(bad: object) -> None:
    with pytest.raises(TypeError, match="pandas or polars DataFrame in long format"):
        read_between(bad, "g", "y")


def test_lazy_and_series_get_specific_hints() -> None:
    with pytest.raises(TypeError, match=r"\.collect\(\)"):
        read_between(pl.LazyFrame({"g": ["a"], "y": [1.0]}), "g", "y")
    for series in (pd.Series([1.0]), pl.Series([1.0]), pa.chunked_array([[1.0]])):
        with pytest.raises(TypeError, match="whole data frame"):
            read_between(series, "g", "y")


def test_narwhals_frame_accepted() -> None:
    native = pd.DataFrame({"g": ["a", "b"], "y": [1.0, 2.0]})
    assert read_between(nw.from_native(native, eager_only=True), "g", "y").levels == (
        "a",
        "b",
    )


# --- columns -------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_column(backend: str) -> None:
    frame = make(backend, {"group": ["a", "b"], "score": [1.0, 2.0]})
    with pytest.raises(KeyError) as info:
        read_between(frame, "group", "scroe")
    message = str(info.value)
    assert isinstance(info.value, ColumnNotFoundError)
    assert message.startswith("y='scroe' is not a column")  # no KeyError quoting
    assert "Did you mean 'score'?" in message
    assert "Available columns: 'group', 'score'." in message
    with pytest.raises(KeyError, match="Did you mean 'group'"):
        read_between(frame, "Group", "score")  # case-insensitive match first
    with pytest.raises(KeyError) as info:
        read_between(frame, "zzz", "score")
    assert "Did you mean" not in str(info.value)


def test_column_name_must_be_text_and_distinct() -> None:
    frame = pd.DataFrame({"g": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(TypeError, match="column name"):
        read_between(frame, 0, "y")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="different columns"):
        read_between(frame, "y", "y")


# --- y -------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_y_bool_and_text_rejected(backend: str) -> None:
    frame = make(backend, {"g": ["a", "b"], "flag": [True, False], "t": ["1", "2"]})
    with pytest.raises(TypeError, match="Booleans are not measurements"):
        read_between(frame, "g", "flag")
    with pytest.raises(TypeError, match="must be numeric"):
        read_between(frame, "g", "t")


def test_y_pandas_object_numbers_rejected() -> None:
    frame = pd.DataFrame({"g": ["a", "b"], "y": pd.Series([1.0, 2.0], dtype=object)})
    with pytest.raises(TypeError, match="to_numeric"):
        read_between(frame, "g", "y")


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"g": ["a", "b"], "y": np.array([1, 2], dtype=np.uint8)}),
        pd.DataFrame({"g": ["a", "b"], "y": pd.array([1, 2], dtype="Int16")}),
        pl.DataFrame(
            {"g": ["a", "b"], "y": [decimal.Decimal("1.5"), decimal.Decimal("2.0")]}
        ),
        pa.table(
            {
                "g": ["a", "b"],
                "y": pa.array([decimal.Decimal("1.5"), decimal.Decimal("2.0")]),
            }
        ),
        pl.DataFrame({"g": ["a", "b"], "y": pl.Series([1.5, 2.0], dtype=pl.Float32)}),
    ],
    ids=["uint8", "Int16", "pl-decimal", "pa-decimal", "float32"],
)
def test_y_numeric_types(frame: Any) -> None:
    out = read_between(frame, "g", "y")
    assert [v.tolist() for v in out.values] in ([[1.0], [2.0]], [[1.5], [2.0]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_infinite_y_raises(backend: str) -> None:
    frame = make(
        backend, {"g": ["a", "a", "b", "b"], "y": [1.0, math.inf, 2.0, -math.inf]}
    )
    with pytest.raises(
        ValueError, match=r"2 infinite value\(s\) \(row positions 1, 3\)"
    ):
        read_between(frame, "g", "y")


@pytest.mark.parametrize("backend", BACKENDS)
def test_infinite_y_in_a_dropped_row_is_ignored(backend: str) -> None:
    # R drops the row for its missing x before any test sees the Inf.
    frame = make(backend, {"g": ["a", None, "b"], "y": [1.0, math.inf, 2.0]})
    out = read_between(frame, "g", "y")
    assert out.n_dropped == 1
    assert out.n_obs == 2


# --- x -------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_x_bool_rejected(backend: str) -> None:
    frame = make(backend, {"g": [True, False], "y": [1.0, 2.0]})
    with pytest.raises(TypeError, match="text labels"):
        read_between(frame, "g", "y")


def test_x_datetime_rejected() -> None:
    frame = pd.DataFrame(
        {"g": pd.to_datetime(["2020-01-01", "2021-01-01"]), "y": [1.0, 2.0]}
    )
    with pytest.raises(TypeError, match="Datetime"):
        read_between(frame, "g", "y")


@pytest.mark.parametrize("backend", BACKENDS)
def test_x_float_whole_numbers(backend: str) -> None:
    # A pandas integer column with a missing value becomes float64.
    frame = make(
        backend,
        {"g": [6.0, 4.0, None, 8.0, 4.0, -0.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
    )
    out = read_between(frame, "g", "y")
    assert out.levels == ("0", "4", "6", "8")
    assert out.counts() == {"0": 1, "4": 2, "6": 1, "8": 1}
    assert out.n_missing_x == 1


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value", [0.5, math.inf])
def test_x_float_not_whole_rejected(backend: str, value: float) -> None:
    frame = make(backend, {"g": [1.0, value], "y": [1.0, 2.0]})
    with pytest.raises(TypeError, match="not whole"):
        read_between(frame, "g", "y")


@pytest.mark.parametrize("backend", BACKENDS)
def test_x_integers_sort_numerically(backend: str) -> None:
    frame = make(
        backend, {"g": [10, 9, -1, 10, 2**53 + 1], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    out = read_between(frame, "g", "y")
    assert out.levels == ("-1", "9", "10", str(2**53 + 1))  # exact beyond float64


@pytest.mark.parametrize("backend", BACKENDS)
def test_x_text_sorts_by_code_point(backend: str) -> None:
    values = ["b", "a", "B", "é", "z", "a10", "a9", ""]
    frame = make(backend, {"g": values, "y": [float(i) for i in range(len(values))]})
    out = read_between(frame, "g", "y")
    assert (
        out.levels
        == tuple(sorted(values))
        == ("", "B", "a", "a10", "a9", "b", "z", "é")
    )


def test_pandas_nullable_and_arrow_dtypes() -> None:
    frame = pd.DataFrame(
        {
            "s": pd.array(["b", None, "a", "a"], dtype="string"),
            "i": pd.array([4, 6, None, 4], dtype="Int64"),
            "y": pd.array([1.0, 2.0, 3.0, None], dtype=pd.ArrowDtype(pa.float64())),
        }
    )
    out = read_between(frame, "s", "y")
    assert (out.levels, out.n_missing_x, out.n_missing_y, out.n_dropped) == (
        ("a", "b"),
        1,
        1,
        2,
    )
    out = read_between(frame, "i", "y")
    assert (out.levels, out.n_dropped) == (("4", "6"), 2)


def test_pandas_arrow_dictionary_x_rejected_clearly() -> None:
    dtype = pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
    frame = pd.DataFrame(
        {
            "g": pd.Series(pa.array(["b", "a"]).dictionary_encode(), dtype=dtype),
            "y": [1.0, 2.0],
        }
    )
    with pytest.raises(TypeError, match=r"astype\('category'\)"):
        read_between(frame, "g", "y")


# --- missing values ------------------------------------------------------------------


def test_pandas_object_text_with_nan_and_none() -> None:
    frame = pd.DataFrame(
        {
            "g": pd.Series(["b", np.nan, "a", None, "a"], dtype=object),
            "y": [1.0, 2.0, 3.0, 4.0, np.nan],
        }
    )
    out = read_between(frame, "g", "y")
    assert (out.levels, out.n_per_level) == (("a", "b"), (1, 1))
    assert (out.n_missing_x, out.n_missing_y, out.n_dropped) == (2, 1, 3)


def test_pandas_nullable_float_na_and_nan() -> None:
    # 0/0 is NaN (distinct from NA) in pandas 2 and NA in pandas 3; both are missing.
    num = pd.Series(pd.array([0.0, 1.0, None, 2.0, 3.0], dtype="Float64"))
    den = pd.Series(pd.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype="Float64"))
    frame = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "y": num / den})
    out = read_between(frame, "g", "y")
    assert (out.n_missing_y, out.counts()) == (2, {"a": 1, "b": 2})


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
def test_null_and_nan_both_missing(backend: str) -> None:
    frame = make(
        backend,
        {
            "g": ["a", "a", "a", "b", "b", None],
            "y": [None, math.nan, 1.0, 2.0, 3.0, 4.0],
        },
    )
    out = read_between(frame, "g", "y")
    assert (out.n_missing_x, out.n_missing_y, out.n_dropped) == (1, 2, 3)
    assert out.counts() == {"a": 1, "b": 2}


def test_dictionary_nulls() -> None:
    # A null index, and an index pointing at a null dictionary entry (pyarrow 14's
    # is_null() misses the second).
    x = pa.DictionaryArray.from_arrays(
        pa.array([0, 1, None, 2, 0, 2], type=pa.int8()), pa.array(["b", None, "a"])
    )
    out = read_between(
        pa.table({"g": x, "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}), "g", "y"
    )
    assert (out.levels, out.n_per_level, out.n_missing_x) == (("b", "a"), (2, 2), 2)
    assert [v.tolist() for v in out.values] == [[1.0, 5.0], [4.0, 6.0]]


# --- level order ---------------------------------------------------------------------


def _check(
    out: Any, levels: tuple[str, ...], empty: tuple[str, ...], source: str
) -> None:
    assert (out.levels, out.empty_levels, out.level_source) == (levels, empty, source)


@pytest.mark.parametrize("ordered", [False, True])
def test_pandas_categorical_declared_order(ordered: bool) -> None:
    g = pd.Categorical(
        ["b", "a", "c", None], categories=["c", "unused", "b", "a"], ordered=ordered
    )
    out = read_between(pd.DataFrame({"g": g, "y": [1.0, 2.0, 3.0, 4.0]}), "g", "y")
    _check(out, ("c", "b", "a"), ("unused",), "declared")


def test_pandas_categorical_number_categories() -> None:
    g = pd.Categorical([6, 4, 8, 4], categories=[8, 6, 4])
    out = read_between(pd.DataFrame({"g": g, "y": [1.0, 2.0, 3.0, 4.0]}), "g", "y")
    _check(out, ("8", "6", "4"), (), "declared")
    g = pd.Categorical([0.5, 4.0], categories=[4.0, 0.5])
    out = read_between(pd.DataFrame({"g": g, "y": [1.0, 2.0]}), "g", "y")
    _check(out, ("4", "0.5"), (), "declared")


def test_level_emptied_by_missing_y_is_reported() -> None:
    g = pd.Categorical(["a", "b", "c", "c"], categories=["a", "b", "c"])
    out = read_between(pd.DataFrame({"g": g, "y": [1.0, 2.0, np.nan, None]}), "g", "y")
    _check(out, ("a", "b"), ("c",), "declared")


def test_polars_enum_declared_order() -> None:
    g = pl.Series(["b", "a", None, "b"], dtype=pl.Enum(["z", "b", "a"]))
    out = read_between(pl.DataFrame({"g": g, "y": [1.0, 2.0, 3.0, 4.0]}), "g", "y")
    _check(out, ("b", "a"), ("z",), "declared")


def test_polars_categorical_is_sorted_not_encounter_order() -> None:
    # polars 1.0 sorts a Categorical in encounter order, polars 2 lexically; inferplot
    # treats it as text on every version.
    g = pl.Series(["b", "c", "a", "b"], dtype=pl.Categorical)
    out = read_between(pl.DataFrame({"g": g, "y": [1.0, 2.0, 3.0, 4.0]}), "g", "y")
    _check(out, ("a", "b", "c"), (), "sorted")


def test_pyarrow_dictionary_order() -> None:
    from_pandas = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "g": pd.Categorical(["b", "a"], categories=["c", "b", "a"]),
                "y": [1.0, 2.0],
            }
        ),
        preserve_index=False,
    )
    _check(read_between(from_pandas, "g", "y"), ("b", "a"), ("c",), "declared")
    encoded = pa.table(
        {"g": pa.array(["b", "a", "b"]).dictionary_encode(), "y": [1.0, 2.0, 3.0]}
    )
    _check(
        read_between(encoded, "g", "y"), ("b", "a"), (), "declared"
    )  # encounter order
    chunked = pa.table(
        {
            "g": pa.chunked_array(
                [
                    pa.array(["b", None]).dictionary_encode(),
                    pa.array(["c", "a"]).dictionary_encode(),
                ]
            ),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = read_between(chunked, "g", "y")
    _check(out, ("b", "c", "a"), (), "declared")
    assert [v.tolist() for v in out.values] == [[1.0], [3.0], [4.0]]


def test_label_collision_raises() -> None:
    g = pd.Categorical(["1", 1], categories=["1", 1])
    with pytest.raises(ValueError, match="both print as '1'"):
        read_between(pd.DataFrame({"g": g, "y": [1.0, 2.0]}), "g", "y")


def test_row_order_kept_within_levels() -> None:
    frame = pl.DataFrame(
        {"g": ["b", "a", "b", "a", "b"], "y": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )
    out = read_between(frame, "g", "y")
    assert [v.tolist() for v in out.values] == [[4.0, 2.0], [5.0, 3.0, 1.0]]


# --- order= and level count -----------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_order(backend: str) -> None:
    frame = make(backend, {"g": [4, 6, 8, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    out = read_between(frame, "g", "y", order=["8", 4, "6"])
    assert (out.levels, out.n_per_level, out.level_source) == (
        ("8", "4", "6"),
        (1, 2, 1),
        "order",
    )
    with pytest.raises(ValueError, match=r"\['6'\] are missing from order"):
        read_between(frame, "g", "y", order=["8", "4"])
    with pytest.raises(ValueError, match=r"\['5'\] are not levels of x"):
        read_between(frame, "g", "y", order=["8", "4", "6", "5"])
    with pytest.raises(ValueError, match="more than once"):
        read_between(frame, "g", "y", order=["8", "4", "6", "4"])
    with pytest.raises(TypeError, match="list of level names"):
        read_between(frame, "g", "y", order="846")
    with pytest.raises(TypeError, match="list of level names"):
        read_between(frame, "g", "y", order=846)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="level names, not True"):
        read_between(frame, "g", "y", order=[True, 4, 6])
    with pytest.raises(TypeError, match="level names, not None"):
        read_between(frame, "g", "y", order=[None, 4, 6])
    with pytest.raises(TypeError, match="level names, not nan"):
        read_between(frame, "g", "y", order=[math.nan, 4, 6])
    with pytest.raises(ValueError, match=r"\['8.5'\] are not levels of x"):
        read_between(frame, "g", "y", order=[8.5, 4, 6])


def _unique_descending(frame: Any, backend: str) -> Any:
    """The distinct values of column g, largest first, as the backend returns them."""
    if backend == "pandas":
        return sorted(frame["g"].unique(), reverse=True)  # NumPy scalars
    if backend == "polars":
        return frame["g"].unique().sort(descending=True)  # a polars Series
    return frame["g"].unique().sort(order="descending")  # pyarrow scalars


@pytest.mark.parametrize("backend", BACKENDS)
def test_order_takes_the_values_x_holds(backend: str) -> None:
    # order=sorted(df[x].unique()) and the like: NumPy integers and floats, whole
    # Python floats, arrays, pandas Index, polars Series and pyarrow scalars.
    ints = make(backend, {"g": [4, 6, 8, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    floats = make(backend, {"g": [4.0, 6.0, 8.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    want = ("8", "6", "4")
    orders: list[Any] = [
        _unique_descending(ints, backend),
        [np.int64(8), np.int32(6), np.uint8(4)],
        [8.0, 6.0, 4.0],
        [np.float64(8.0), np.float32(6.0), 4],
        np.array([8, 6, 4]),
        np.array([8.0, 6.0, 4.0]),
        np.array(["8", "6", "4"]),
        pd.Index([8, 6, 4]),
        pl.Series([8, 6, 4]),
        pa.array([8, 6, 4]),
        (n for n in (8, 6, 4)),
    ]
    for frame in (ints, floats):
        for order in orders:
            if hasattr(order, "__next__"):
                order = (n for n in (8, 6, 4))  # a generator is used up once
            out = read_between(frame, "g", "y", order=order)
            assert out.levels == want, order
            assert out.level_source == "order"
            assert out.n_per_level == (1, 1, 2)
    text = make(backend, {"g": ["a", "b", "c", "a"], "y": [1.0, 2.0, 3.0, 4.0]})
    for order in (np.array(["c", "a", "b"]), pd.Index(["c", "a", "b"])):
        out = read_between(text, "g", "y", order=order)
        assert out.levels == ("c", "a", "b")
        assert all(type(level) is str for level in out.levels)


def test_order_from_the_column_through_the_analysis() -> None:
    from inferplot import analyze

    num = pd.DataFrame(
        {"g": [4, 6, 8] * 4, "y": np.random.default_rng(1).normal(size=12)}
    )
    result = analyze.betweenstats(
        num, "g", "y", order=sorted(num["g"].unique(), reverse=True)
    )
    assert result.levels == ("8", "6", "4")
    result = analyze.betweenstats(
        num.astype({"g": float}),
        "g",
        "y",
        order=[8.0, 6.0, 4.0],  # type: ignore[list-item]
    )
    assert result.levels == ("8", "6", "4")


def test_order_naming_an_empty_level() -> None:
    g = pd.Categorical(["a", "b"], categories=["a", "b", "c"])
    frame = pd.DataFrame({"g": g, "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"\['c'\] have no rows with both x and y"):
        read_between(frame, "g", "y", order=["c", "b", "a"])


@pytest.mark.parametrize("backend", BACKENDS)
def test_fewer_than_two_levels(backend: str) -> None:
    frame = make(backend, {"g": ["a", "a", "b"], "y": [1.0, 2.0, None]})
    with pytest.raises(ValueError, match="at least 2 groups"):
        read_between(frame, "g", "y")


# --- labels --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "label"),
    [
        ("x", "x"),
        (4, "4"),
        (4.0, "4"),
        (-0.0, "0"),
        (0.5, "0.5"),
        (np.int64(7), "7"),
        (1 / 3, "0.333333333333333"),
        (np.float32(2.0), "2"),
    ],
)
def test_level_label(value: object, label: str) -> None:
    assert level_label(value) == label


@pytest.mark.parametrize("value", [True, np.bool_(False), None, math.nan, b"x"])
def test_level_label_rejects(value: object) -> None:
    with pytest.raises(TypeError):
        level_label(value)


def test_matches_repository_helper() -> None:
    for value in ["a", 4, 4.0, 0.5, 12.25, -3]:
        assert level_label(value) == r_level_label(value)
