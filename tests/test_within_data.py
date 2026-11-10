"""The within-subjects data boundary (``inferplot._data.read_within``, SPEC.md §3).

The rules: rows missing ``x``, ``y`` or ``subject_id`` are dropped and counted; a
subject repeated within a level raises; subjects without a value at every level are
left out and listed; the analyzed data are a subjects-by-levels matrix of complete
subjects, in the subject column's order, the same on pandas, polars and pyarrow.

The pivot is checked against R where fixtures allow it: every ``ggwithinstats``
fixture records the subjects-by-conditions data ggstatsplot analyzed
(``_recorded_pivot``). Until the first such fixture exists,
``test_fixture_pivot`` asserts that there is none (it is not skipped), and the
hand-built expectations below on ``bugs_long``, ``sleep`` and
``edge_within_missing`` check the same rules.
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from inferplot._data import ColumnNotFoundError, WithinData, read_within
from tests.oracle_fixtures import (
    DATASET_DIR,
    OracleCase,
    cases_for,
    fixture_paths,
    read_csv,
    read_dataset,
)

BACKENDS = ("pandas", "polars", "pyarrow")
# The exact message is compared after each raise.
VALUE_ERROR: type[Exception] = ValueError


def make(backend: str, columns: dict[str, Any]) -> Any:
    """A frame of ``backend`` from plain Python lists (None is missing)."""
    if backend == "pandas":
        return pd.DataFrame(columns)
    if backend == "polars":
        return pl.DataFrame(columns, strict=False)
    return pa.table(columns)


def long(rows: Sequence[tuple[object, object, object]]) -> dict[str, list[object]]:
    """Columns s, g, y from (subject, level, value) rows."""
    return {
        "s": [r[0] for r in rows],
        "g": [r[1] for r in rows],
        "y": [r[2] for r in rows],
    }


def as_dict(data: WithinData) -> dict[str, tuple[float, ...]]:
    """``{subject: values in level order}``."""
    return {
        subject: tuple(float(v) for v in row)
        for subject, row in zip(data.subjects, data.values, strict=True)
    }


def check_shape(data: WithinData) -> None:
    """Invariants of every WithinData."""
    k = len(data.levels)
    assert data.values.shape == (data.n_subjects, k)
    assert data.values.dtype == np.float64
    assert data.values.flags.c_contiguous
    assert not data.values.flags.writeable
    assert len(set(data.subjects)) == data.n_subjects
    assert not set(data.subjects) & set(data.dropped_subjects)
    assert all(type(s) is str for s in data.subjects + data.dropped_subjects)
    columns = data.by_level()
    assert len(columns) == k
    for j, column in enumerate(columns):
        assert not column.flags.writeable
        assert column.tolist() == data.values[:, j].tolist()


# --- oracle fixtures (ggwithinstats) ------------------------------------------------


def _recorded_pivot(case: OracleCase) -> Mapping[str, Any]:
    """The subjects-by-conditions data a ggwithinstats fixture records.

    Proposed schema (to settle with the oracle generator): ``within_data`` with
    ``subject_id`` (the column), ``levels`` (the analyzed order), ``subjects`` (the
    complete subjects as R prints them, one per row), ``values`` (one row per
    subject, one column per level) and ``dropped_subjects`` (the subjects left out
    as incomplete). A fixture without it fails here; it is never skipped.
    """
    return case.raw["within_data"]


def _within_params() -> list[Any]:
    cases = cases_for("ggwithinstats") + cases_for("ggwithinstats", failed=True)
    if not cases:
        return [pytest.param(None, id="no-ggwithinstats-fixtures-yet")]
    return [pytest.param(case, id=case.name) for case in cases]


def _csv_rows_missing(file: str, columns: Sequence[str]) -> int:
    with (DATASET_DIR / file).open(encoding="utf-8", newline="") as f:
        return sum(any(r[c] == "" for c in columns) for r in csv.DictReader(f))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", _within_params())
def test_fixture_pivot(case: OracleCase | None, backend: str) -> None:
    if case is None:
        # Not a skip: this asserts the state the hand-built tests below stand in
        # for. The first ggwithinstats fixture replaces this parameter.
        assert not [p for p in fixture_paths() if p.stem.startswith("ggwithinstats-")]
        return
    recorded = _recorded_pivot(case)
    x, y = str(case.args["x"]), str(case.args["y"])
    subject_id = str(recorded["subject_id"])
    data = read_within(read_dataset(case, backend), x, y, subject_id)  # type: ignore[arg-type]
    check_shape(data)
    assert data.levels == case.analysis_levels == tuple(recorded["levels"])
    assert set(case.n_per_level.values()) == {data.n_subjects}
    # Row order is inferplot's own (the subject column's order); the pairing of
    # every value with its subject and level is R's.
    expected = {
        str(s): tuple(float(v) for v in row)
        for s, row in zip(recorded["subjects"], recorded["values"], strict=True)
    }
    assert as_dict(data) == expected
    assert set(data.dropped_subjects) == {str(s) for s in recorded["dropped_subjects"]}
    assert data.n_dropped == _csv_rows_missing(case.dataset_file, (x, y, subject_id))


# --- hand-built expectations on the oracle datasets --------------------------------


def _complete_from_csv(
    file: str, x: str, y: str, subject_id: str
) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Complete subjects ({subject: {level: value}}) and all subjects, read from
    the CSV text with the csv module, independently of inferplot."""
    cells: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    levels: set[str] = set()
    with (DATASET_DIR / file).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row[subject_id] != "":
                seen.add(row[subject_id])
            if "" in (row[x], row[y], row[subject_id]):
                continue
            levels.add(row[x])
            cells.setdefault(row[subject_id], {})[row[x]] = float(row[y])
    complete = {s: v for s, v in cells.items() if set(v) == levels}
    return complete, seen


@pytest.mark.parametrize("backend", BACKENDS)
def test_bugs_long(backend: str) -> None:
    frame = read_csv("bugs_long.csv", backend)  # type: ignore[arg-type]
    data = read_within(frame, "condition", "desire", "subject")
    check_shape(data)
    assert data.levels == ("HDHF", "HDLF", "LDHF", "LDLF")
    assert data.level_source == "sorted"
    # 93 subjects; 5 have one missing desire each, so 88 complete subjects remain.
    assert data.n_subjects == 88
    assert data.dropped_subjects == ("2", "10", "42", "64", "80")  # numeric order
    assert (data.n_dropped, data.n_missing_y, data.n_rows_of_dropped_subjects) == (
        5,
        5,
        15,
    )
    assert (data.n_missing_x, data.n_missing_subject) == (0, 0)
    assert data.n_subjects * 4 + data.n_dropped + data.n_rows_of_dropped_subjects == 372
    assert list(data.subjects) == sorted(data.subjects, key=int)
    complete, seen = _complete_from_csv(
        "bugs_long.csv", "condition", "desire", "subject"
    )
    assert len(seen) == 93
    assert as_dict(data) == {
        s: tuple(v[level] for level in data.levels) for s, v in complete.items()
    }
    assert as_dict(data)["1"] == (10.0, 9.0, 6.0, 6.0)  # HDHF, HDLF, LDHF, LDLF


@pytest.mark.parametrize("backend", BACKENDS)
def test_sleep(backend: str) -> None:
    frame = read_csv("sleep.csv", backend)  # type: ignore[arg-type]
    data = read_within(frame, "group", "extra", "ID")
    check_shape(data)
    assert data.levels == ("1", "2")
    assert data.subjects == tuple(str(i) for i in range(1, 11))
    assert (data.n_dropped, data.dropped_subjects) == (0, ())
    # R's sleep data: drug 1 then drug 2 for each of the 10 patients.
    assert data.values[:, 0].tolist() == [
        0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0,
    ]  # fmt: skip
    assert data.values[:, 1].tolist() == [
        1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4,
    ]  # fmt: skip


@pytest.mark.parametrize("backend", BACKENDS)
def test_edge_within_missing(backend: str) -> None:
    file = "edge_within_missing.csv"
    data = read_within(read_csv(file, backend), "condition", "value", "subject")  # type: ignore[arg-type]
    check_shape(data)
    assert data.levels == ("mid", "post", "pre")  # sorted by code point
    assert data.n_subjects == 11
    assert data.dropped_subjects == ("s05",)  # it has no "mid" row at all
    assert (data.n_dropped, data.n_rows_of_dropped_subjects) == (0, 2)
    assert as_dict(data)["s01"] == (30.51, 34.24, 33.06)
    complete, _ = _complete_from_csv(file, "condition", "value", "subject")
    assert as_dict(data) == {
        s: tuple(v[level] for level in data.levels) for s, v in complete.items()
    }
    declared = read_csv(file, backend, {"condition": ["pre", "mid", "post"]})  # type: ignore[arg-type]
    ordered = read_within(declared, "condition", "value", "subject")
    assert (ordered.levels, ordered.level_source) == (
        ("pre", "mid", "post"),
        "declared",
    )
    assert as_dict(ordered)["s01"] == (33.06, 30.51, 34.24)
    by_order = read_within(
        read_csv(file, backend),  # type: ignore[arg-type]
        "condition",
        "value",
        "subject",
        order=["pre", "mid", "post"],
    )
    assert by_order.level_source == "order"
    assert as_dict(by_order) == as_dict(ordered)


# --- data and columns --------------------------------------------------------------


def test_non_frames_rejected() -> None:
    with pytest.raises(TypeError, match="pandas or polars DataFrame in long format"):
        read_within({"s": [1], "g": ["a"], "y": [1.0]}, "g", "y", "s")


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_subject_column(backend: str) -> None:
    frame = make(backend, {"subject": [1, 1], "g": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(ColumnNotFoundError) as info:
        read_within(frame, "g", "y", "subjcet")
    assert str(info.value).startswith("subject_id='subjcet' is not a column of data.")
    assert "Did you mean 'subject'?" in str(info.value)
    with pytest.raises(TypeError, match="subject_id must be a column name"):
        read_within(frame, "g", "y", 1)  # type: ignore[arg-type]


def test_subject_column_must_differ_from_x_and_y() -> None:
    frame = pd.DataFrame({"s": [1, 1], "g": ["a", "b"], "y": [1.0, 2.0]})
    with pytest.raises(VALUE_ERROR) as info:
        read_within(frame, "g", "y", "g")
    assert str(info.value) == (
        "subject_id must name the column that identifies each subject, a column "
        "other than x and y; subject_id='g' is also x."
    )
    with pytest.raises(ValueError, match="subject_id='y' is also y"):
        read_within(frame, "g", "y", "y")
    with pytest.raises(ValueError, match="different columns"):
        read_within(frame, "y", "y", "s")


@pytest.mark.parametrize("backend", BACKENDS)
def test_y_rules_are_betweenstats_rules(backend: str) -> None:
    frame = make(backend, {"s": [1, 1], "g": ["a", "b"], "y": [True, False]})
    with pytest.raises(TypeError, match="Booleans are not measurements"):
        read_within(frame, "g", "y", "s")


# --- subject_id types --------------------------------------------------------------

PAIRS = [(1.0, 3.0), (2.0, 5.0), (4.0, 4.5)]  # (a, b) per subject


def _paired(subjects: Sequence[object]) -> dict[str, list[object]]:
    rows = [
        (s, g, pair[j])
        for s, pair in zip(subjects, PAIRS, strict=True)
        for j, g in enumerate("ab")
    ]
    return long(rows)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("subjects", "labels"),
    [
        (["p1", "p10", "p2"], ("p1", "p10", "p2")),  # text: code-point order
        ([10, 2, 1], ("1", "2", "10")),  # integers: numeric order
        ([10.0, 2.0, -0.0], ("0", "2", "10")),  # whole floats, labelled as ints
        ([2**53 + 1, 3, 2], ("2", "3", str(2**53 + 1))),  # exact beyond float64
    ],
)
def test_subject_types_and_order(
    backend: str, subjects: list[str | int | float], labels: tuple[str, ...]
) -> None:
    data = read_within(make(backend, _paired(subjects)), "g", "y", "s")
    check_shape(data)
    assert data.subjects == labels
    by_label = dict(
        zip(
            [str(int(s)) if not isinstance(s, str) else s for s in subjects],
            PAIRS,
            strict=True,
        )
    )
    assert as_dict(data) == by_label


def test_subject_nullable_and_categorical_columns() -> None:
    base = _paired(["b", "c", "a"])
    expected_sorted = {"a": PAIRS[2], "b": PAIRS[0], "c": PAIRS[1]}
    frames: list[tuple[Any, tuple[str, ...]]] = [
        (
            pd.DataFrame({**base, "s": pd.array(base["s"], dtype="string")}),
            ("a", "b", "c"),
        ),
        (
            pd.DataFrame({**base, "s": pd.array([2, 2, 3, 3, 1, 1], dtype="Int64")}),
            ("1", "2", "3"),
        ),
        (
            pd.DataFrame(
                {
                    **base,
                    "s": pd.Categorical(base["s"], categories=["c", "z", "a", "b"]),
                }
            ),
            ("c", "a", "b"),  # declared order; the empty category is ignored
        ),
        (
            pl.DataFrame(
                {**base, "s": pl.Series(base["s"], dtype=pl.Enum(["c", "a", "b"]))}
            ),
            ("c", "a", "b"),
        ),
        (
            pl.DataFrame({**base, "s": pl.Series(base["s"], dtype=pl.Categorical)}),
            ("a", "b", "c"),  # polars Categorical is text, as for x
        ),
        (
            pa.table({**base, "s": pa.array(base["s"]).dictionary_encode()}),
            ("a", "b", "c"),  # an unordered dictionary is text: sorted
        ),
        (
            pa.table(
                {
                    **base,
                    "s": pa.DictionaryArray.from_arrays(
                        pa.array([0, 0, 1, 1, 2, 2], type=pa.int8()),
                        pa.array(["b", "c", "a"]),
                        ordered=True,
                    ),
                }
            ),
            ("b", "c", "a"),  # a dictionary marked ordered declares its order
        ),
        (
            pd.DataFrame(
                {**base, "s": pd.array(base["s"], dtype=pd.ArrowDtype(pa.string()))}
            ),
            ("a", "b", "c"),
        ),
    ]
    for frame, order in frames:
        data = read_within(frame, "g", "y", "s")
        check_shape(data)
        assert data.subjects == order, frame
        if order != ("1", "2", "3"):
            assert as_dict(data) == expected_sorted


@pytest.mark.parametrize("backend", BACKENDS)
def test_subject_bool_float_and_datetime_rejected(backend: str) -> None:
    rows = _paired([True, False, True])
    with pytest.raises(TypeError) as info:
        read_within(make(backend, rows), "g", "y", "s")
    assert (
        "subject_id='s' must hold text, categories or whole numbers to identify "
        "subjects" in str(info.value)
    )
    assert "cannot tell subjects apart" in str(info.value)
    with pytest.raises(TypeError) as info:
        read_within(make(backend, _paired([1.5, 2.0, 3.0])), "g", "y", "s")
    assert str(info.value).startswith(
        "subject_id='s' holds numbers that are not whole (1.5); a subject column"
    )
    assert "not a measurement" in str(info.value)
    dates = pd.DataFrame(
        {**_paired([1, 2, 3]), "s": pd.to_datetime(["2020-01-01"] * 6)}
    )
    with pytest.raises(TypeError, match="to identify subjects"):
        read_within(dates, "g", "y", "s")


def test_subject_label_collision_raises() -> None:
    s = pd.Categorical(["1", "1", 1, 1], categories=["1", 1])
    frame = pd.DataFrame({"s": s, "g": list("abab"), "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(
        ValueError,
        match="subject_id='s' has two different values that both print as '1'",
    ):
        read_within(frame, "g", "y", "s")


# --- missing values, duplicates, infinite values -----------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_values_drop_rows_and_incomplete_subjects(backend: str) -> None:
    rows = [
        ("s1", "a", 1.0), ("s1", "b", 2.0),
        ("s2", "a", 3.0), ("s2", "b", None),  # missing y: s2 is incomplete
        ("s3", "a", 5.0), ("s3", None, 6.0),  # missing x: s3 is incomplete
        (None, "b", 7.0),  # missing subject: no subject to drop
        ("s4", "a", 8.0), ("s4", "b", 9.0),
        ("s5", "a", None), ("s5", "b", None),  # every y missing: still a subject
        ("s6", "a", 1.5),  # no row at b
        ("s7", "a", 2.5), ("s7", "b", 3.5),
    ]  # fmt: skip
    data = read_within(make(backend, long(rows)), "g", "y", "s")
    check_shape(data)
    assert data.subjects == ("s1", "s4", "s7")
    assert data.dropped_subjects == ("s2", "s3", "s5", "s6")
    assert (data.n_missing_subject, data.n_missing_x, data.n_missing_y) == (1, 1, 3)
    assert data.n_dropped == 5  # rows, each once
    assert data.n_rows_of_dropped_subjects == 3  # s2 a, s3 a, s6 a
    assert (
        data.n_subjects * 2 + data.n_dropped + data.n_rows_of_dropped_subjects
        == len(rows)
    )
    assert as_dict(data) == {"s1": (1.0, 2.0), "s4": (8.0, 9.0), "s7": (2.5, 3.5)}


@pytest.mark.parametrize("backend", ["polars", "pyarrow"])
def test_nan_y_is_missing(backend: str) -> None:
    rows = [
        ("s1", "a", 1.0),
        ("s1", "b", math.nan),
        ("s2", "a", 3.0),
        ("s2", "b", 4.0),
        ("s3", "a", 1.0),
        ("s3", "b", 0.0),
    ]
    data = read_within(make(backend, long(rows)), "g", "y", "s")
    assert (data.subjects, data.dropped_subjects, data.n_dropped) == (
        ("s2", "s3"),
        ("s1",),
        1,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_duplicate_subject_within_a_level_raises(backend: str) -> None:
    rows = [
        ("s1", "a", 1.0), ("s1", "b", 2.0),
        ("s2", "a", 3.0), ("s2", "b", 4.0),
        ("s1", "a", 5.0),
        ("s3", "b", 6.0), ("s3", "b", 7.0), ("s3", "b", 8.0),
    ]  # fmt: skip
    with pytest.raises(VALUE_ERROR) as info:
        read_within(make(backend, long(rows)), "g", "y", "s")
    assert str(info.value) == (
        "subject_id='s' must identify each subject at most once per level of x='g', "
        "but 2 subject-level combinations have more than one row: subject 's1' at "
        "'a' (row positions 0, 4); subject 's3' at 'b' (row positions 5, 6, 7). "
        "Combine the repeated rows first (for example their mean), or make the "
        "repetition part of x."
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_duplicate_with_a_missing_y_still_raises(backend: str) -> None:
    # Checked before rows with a missing y are dropped: the second row still
    # records s1 at a.
    rows = [
        ("s1", "a", 1.0),
        ("s1", "b", 2.0),
        ("s1", "a", None),
        ("s2", "a", 3.0),
        ("s2", "b", 4.0),
    ]
    with pytest.raises(VALUE_ERROR) as info:
        read_within(make(backend, long(rows)), "g", "y", "s")
    message = str(info.value)
    assert (
        "1 subject-level combination has more than one row: subject 's1' at 'a' "
        "(row positions 0, 2)" in message
    )
    assert message.endswith(
        "A row whose y is missing counts too: it still records that subject at that "
        "level, so remove it if it is a placeholder."
    )
    # Rows without a subject or a level cannot repeat anything.
    rows = [
        ("s1", "a", 1.0),
        ("s1", "b", 2.0),
        ("s2", "a", 3.0),
        ("s2", "b", 4.0),
        (None, "a", 5.0),
        (None, "a", 6.0),
        ("s1", None, 7.0),
    ]
    assert read_within(make(backend, long(rows)), "g", "y", "s").n_dropped == 3


def test_duplicates_shown_are_capped() -> None:
    rows = [(f"s{i}", "a", 1.0) for i in range(7)] * 2 + [
        (f"s{i}", "b", 1.0) for i in range(7)
    ]
    with pytest.raises(ValueError, match=r"; and 2 more\. Combine"):
        read_within(pd.DataFrame(long(rows)), "g", "y", "s")


@pytest.mark.parametrize("backend", BACKENDS)
def test_infinite_y_raises(backend: str) -> None:
    rows = [
        ("s1", "a", 1.0),
        ("s1", "b", math.inf),
        ("s2", "a", 3.0),
        ("s2", "b", 4.0),
        ("s3", "a", -math.inf),
    ]
    # Raises also for s3, which is left out later for lacking b.
    with pytest.raises(
        ValueError, match=r"y='y' has 2 infinite value\(s\) \(row positions 1, 4\)"
    ):
        read_within(make(backend, long(rows)), "g", "y", "s")
    # A row with a missing subject is dropped before any value is looked at.
    rows = [
        ("s1", "a", 1.0),
        ("s1", "b", 2.0),
        ("s2", "a", 3.0),
        ("s2", "b", 4.0),
        (None, "a", math.inf),
    ]
    assert read_within(make(backend, long(rows)), "g", "y", "s").n_dropped == 1


# --- levels and subject counts -----------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_too_few_complete_subjects(backend: str) -> None:
    rows = [("s1", "a", 1.0), ("s2", "b", 2.0), ("s3", "a", None), ("s3", "b", 3.0)]
    with pytest.raises(VALUE_ERROR) as info:
        read_within(make(backend, long(rows)), "g", "y", "s")
    assert str(info.value) == (
        "withinstats needs at least 2 subjects with a value of y at every level of "
        "x='g' ('a', 'b'); of the 3 subjects in subject_id='s', none has a value at "
        "every level after rows with missing values were dropped. For example, "
        "subject 's1' has no value at 'b'. Subjects missing a level are left out of "
        "a within-subjects analysis; check the levels and the subject column."
    )
    rows += [("s4", "a", 4.0), ("s4", "b", 5.0)]
    with pytest.raises(ValueError, match=r"only 1 \('s4'\) has a value at every level"):
        read_within(make(backend, long(rows)), "g", "y", "s")
    rows += [("s5", "a", 4.0), ("s5", "b", 5.0)]
    assert read_within(make(backend, long(rows)), "g", "y", "s").subjects == (
        "s4",
        "s5",
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fewer_than_two_levels(backend: str) -> None:
    rows = [("s1", "a", 1.0), ("s1", "b", None), ("s2", "a", 2.0)]
    with pytest.raises(
        ValueError, match=r"x='g' needs at least 2 conditions with data; found 1"
    ):
        read_within(make(backend, long(rows)), "g", "y", "s")


def test_order_and_empty_declared_levels() -> None:
    g = pd.Categorical(["b", "a", "b", "a"], categories=["c", "b", "a"])
    frame = pd.DataFrame({"s": [1, 1, 2, 2], "g": g, "y": [1.0, 2.0, 3.0, 4.0]})
    data = read_within(frame, "g", "y", "s")
    assert (data.levels, data.empty_levels, data.level_source) == (
        ("b", "a"),
        ("c",),
        "declared",
    )
    assert as_dict(data) == {"1": (1.0, 2.0), "2": (3.0, 4.0)}
    flipped = read_within(frame, "g", "y", "s", order=["a", "b"])
    assert as_dict(flipped) == {"1": (2.0, 1.0), "2": (4.0, 3.0)}
    with pytest.raises(
        ValueError, match=r"\['c'\] have no rows with x, y and subject_id all present"
    ):
        read_within(frame, "g", "y", "s", order=["c", "a", "b"])


# --- the same matrix whatever the row order and library ----------------------------


def test_row_order_and_backend_do_not_change_the_matrix() -> None:
    frame = read_csv("bugs_long.csv")
    reference = read_within(frame, "condition", "desire", "subject")
    rng = random.Random(7)
    for _ in range(3):
        rows = list(range(len(frame)))
        rng.shuffle(rows)
        shuffled = frame.iloc[rows].reset_index(drop=True)
        for native in (
            shuffled,
            pl.from_pandas(
                shuffled.astype(
                    dict.fromkeys(
                        ("gender", "region", "education", "condition"), "string"
                    )
                )
            ),
            pa.Table.from_pandas(shuffled, preserve_index=False),
        ):
            data = read_within(native, "condition", "desire", "subject")
            assert data.subjects == reference.subjects
            assert data.dropped_subjects == reference.dropped_subjects
            assert data.values.tobytes() == reference.values.tobytes()


@pytest.mark.parametrize("backend", BACKENDS)
def test_integer_y_too_large_for_a_double_raises(backend: str) -> None:
    # As for read_between: integers above 2**53 that float64 would round raise.
    big = 1_700_000_000_000_000_000
    frame = make(
        backend,
        {
            "s": [1, 1, 2, 2, 3, 3],
            "g": list("ababab"),
            "y": [big + v for v in (10, 40, 70, 150, 180, 210)],
        },
    )
    with pytest.raises(ValueError, match="too large to analyze exactly"):
        read_within(frame, "g", "y", "s")


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_condition_without_complete_rows_is_named(backend: str) -> None:
    from inferplot.analyses.withinstats import (
        _data_warnings,  # pyright: ignore[reportPrivateUsage]
    )

    # Condition 'c' has rows, but each misses y or the subject: it is left out
    # and the warnings say so; 'a' and 'b' are analyzed.
    rows = [
        (1, "a", 1.0),
        (1, "b", 2.0),
        (2, "a", 3.0),
        (2, "b", 5.0),
        (3, "a", 4.0),
        (3, "b", 4.5),
        (1, "c", None),
        (None, "c", 7.0),
    ]
    data = read_within(make(backend, long(rows)), "g", "y", "s")
    assert (data.levels, data.vanished_levels, data.dropped_subjects) == (
        ("a", "b"),
        ("c",),
        (),
    )
    assert _data_warnings(data) == [
        "The level 'c' of 'g' was left out of the analysis: every row of it has a "
        "missing 'y' or 's'."
    ]
    # A condition with complete rows only for subjects that lack another condition
    # keeps every subject from being complete: that raises, naming a subject.
    rows = [
        (1, "a", 1.0),
        (1, "b", 2.0),
        (2, "a", 3.0),
        (2, "b", 5.0),
        (3, "c", 4.0),
    ]
    with pytest.raises(ValueError, match="needs at least 2 subjects") as info:
        read_within(make(backend, long(rows)), "g", "y", "s")
    assert "none has a value at every level" in str(info.value)


def test_within_lookalike_and_dictionary_notes() -> None:
    from inferplot.analyses.withinstats import (
        _data_warnings,  # pyright: ignore[reportPrivateUsage]
    )

    rows = [(s, g, float(s + i)) for s in (1, 2, 3) for i, g in enumerate(["b", "b "])]
    data = read_within(
        pa.table({**long(rows), "g": pa.array(long(rows)["g"]).dictionary_encode()}),
        "g",
        "y",
        "s",
    )
    assert data.lookalike_levels == (("b", "b "),)
    assert data.dictionary_order == ()  # first appearance is sorted here
    notes = _data_warnings(data)
    assert len(notes) == 1
    assert notes[0].startswith("The levels 'b' and 'b ' of 'g' differ only in spaces")
    rows = [(s, g, float(s + i)) for s in (1, 2, 3) for i, g in enumerate("ba")]
    table = pa.table({**long(rows), "g": pa.array(long(rows)["g"]).dictionary_encode()})
    data = read_within(table, "g", "y", "s")
    assert (data.levels, data.dictionary_order) == (("a", "b"), ("b", "a"))
    assert _data_warnings(data)[0].startswith(
        "The levels of 'g' are in sorted order ('a', 'b'), not in the order of its "
        "pyarrow dictionary ('b', 'a')"
    )


def test_dictionary_encoded_columns_do_not_follow_row_order() -> None:
    # The review's example: dictionary_encode() and polars .to_arrow() list the
    # values in order of first appearance, which is not a declared order; the
    # levels and the subjects are sorted, so shuffling the rows changes nothing.
    subjects = np.repeat([f"s{i}" for i in range(5)], 2)
    x = np.tile(["pre", "post"], 5)
    y = np.arange(10.0)
    reference = read_within(
        pd.DataFrame({"s": subjects, "x": x, "y": y}), "x", "y", "s"
    )
    assert reference.levels == ("post", "pre")
    for seed in range(4):
        perm = np.random.default_rng(seed).permutation(10)
        frames = [
            pa.table(
                {
                    "s": pa.array(subjects[perm]).dictionary_encode(),
                    "x": pa.array(x[perm]).dictionary_encode(),
                    "y": y[perm],
                }
            ),
            pl.DataFrame(
                {
                    "s": pl.Series(subjects[perm], dtype=pl.Categorical),
                    "x": pl.Series(x[perm], dtype=pl.Categorical),
                    "y": y[perm],
                }
            ).to_arrow(),
        ]
        for frame in frames:
            data = read_within(frame, "x", "y", "s")
            assert (data.levels, data.subjects) == (
                reference.levels,
                reference.subjects,
            )
            assert data.values.tobytes() == reference.values.tobytes()
            assert data.level_source == "sorted"
