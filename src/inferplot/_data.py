"""The data boundary: user data frames in, clean NumPy arrays out (SPEC.md section 3).

Every public function takes any eager frame that ``narwhals.from_native(data,
eager_only=True)`` accepts (pandas, polars, pyarrow) and returns plain NumPy arrays and
Python values. Nothing downstream of this module sees a data frame. pandas, polars and
pyarrow are optional extras, so none of them is imported here; narwhals' ``stable.v2``
API does all the work.

What it does for ``betweenstats`` (``read_between``):

* ``y`` must be numeric (integers, floats, decimals; booleans are rejected).
* ``x`` may be text, categorical, integer, or float holding only whole numbers (a pandas
  integer column with a missing value becomes float).
* Rows with a missing ``x`` or ``y`` are dropped and counted, as ggstatsplot's
  ``tidyr::drop_na()`` does: pandas NaN, None and NA; polars null and NaN; pyarrow
  null and NaN (R's ``is.na()`` is true for NaN).
* An infinite ``y`` in a row that is otherwise complete raises ``ValueError``.
* Level order: the declared category order (pandas Categorical, polars Enum, pyarrow
  dictionary marked ordered), otherwise the sorted unique values (numbers
  numerically, text as Python's ``sorted()``, which is R's C-locale order). A polars
  ``Categorical`` has no stable declared order across polars versions, so it is
  treated as text. A pyarrow dictionary not marked ordered lists values as they
  first appear in the rows, so it is read as its values (``dictionary_order`` keeps
  its order when that differs, for the result's warnings).
* Declared levels left without rows are dropped and reported (``empty_levels``);
  ggstatsplot does the same with ``droplevels()``. Other levels whose rows all miss
  ``y`` are reported too (``vanished_levels``), and so are levels that differ only
  in surrounding whitespace (``lookalike_levels``); ``data_notes`` words them.
* Labels are the names R gives factor levels: whole numbers without ``.0``.
* ``order=`` must be a permutation of the levels present.

What it adds for ``withinstats`` (``read_within``): ``subject_id`` follows ``x``'s
type rules; rows missing any of the three columns are dropped and counted; a subject
repeated within a level raises; subjects without a value at every level are left out
and listed; the result is a subjects-by-levels matrix of complete subjects in the
subject column's order (see ``read_within``).
"""

from __future__ import annotations

import contextlib
import datetime
import difflib
import math
import numbers
import re
import unicodedata
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final, Literal, cast

import narwhals.stable.v2 as nw
import numpy as np
import numpy.typing as npt
from narwhals.exceptions import DuplicateError

__all__ = [
    "BetweenData",
    "ColumnNotFoundError",
    "OneData",
    "PairData",
    "TableData",
    "TableLevels",
    "WithinData",
    "data_notes",
    "level_label",
    "read_between",
    "read_one",
    "read_pair",
    "read_table",
    "read_within",
    "table_notes",
    "to_frame",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
LevelSource = Literal["declared", "sorted", "order"]

_LONG_FORMAT_HINT = (
    "build a pandas or polars DataFrame in long format: one row per observation, "
    "one column per variable, for example "
    'pandas.DataFrame({"group": groups, "score": scores})'
)
# Cutoff used for every did-you-mean suggestion (SPEC.md section 2).
_CLOSE_MATCH_CUTOFF = 0.6
# How many offending row positions an error message names.
_MAX_ROWS_SHOWN = 5


class ColumnNotFoundError(KeyError):
    """A column name that the data frame does not have.

    Subclasses ``KeyError`` (SPEC.md section 3) but prints its message as written:
    ``str(KeyError("..."))`` would wrap it in quotes and escape newlines.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class BetweenData:
    """The analyzed data of a between-groups design, one array per level.

    ``values[i]`` holds the ``y`` values of ``levels[i]`` in the frame's row order.
    The arrays are float64 and read-only.
    """

    x: str
    y: str
    levels: tuple[str, ...]
    values: tuple[FloatArray, ...]
    n_per_level: tuple[int, ...]
    n_obs: int
    n_dropped: int
    """Rows dropped because ``x`` or ``y`` was missing (a row counts once)."""
    n_missing_x: int
    n_missing_y: int
    """Rows missing ``x`` and rows missing ``y``; a row missing both counts in each."""
    empty_levels: tuple[str, ...]
    """Declared levels without a complete row, in declared order; not analyzed."""
    level_source: LevelSource
    dictionary_order: tuple[str, ...] = ()
    """``levels`` in the order of ``x``'s unordered pyarrow dictionary, when that
    differs from the sorted order used (empty otherwise, and with ``order=``)."""
    vanished_levels: tuple[str, ...] = ()
    """Levels of an ``x`` without declared categories whose rows all miss ``y``,
    sorted; not analyzed. (Declared ones are in ``empty_levels``.)"""
    lookalike_levels: tuple[tuple[str, ...], ...] = ()
    """Sets of analyzed levels that differ only in leading or trailing whitespace."""
    natural_order: tuple[str, ...] = ()
    """``levels`` ordered by the numbers in their names (digits compared as
    numbers), when the levels are text in sorted order and that order differs
    ("week 2" before "week 10"); empty otherwise, and with ``order=``."""

    def counts(self) -> dict[str, int]:
        """``{level: n}`` in level order."""
        return dict(zip(self.levels, self.n_per_level, strict=True))


@dataclass(frozen=True, slots=True)
class WithinData:
    """The analyzed data of a within-subjects design: one row per complete subject.

    ``values[i, j]`` is the ``y`` value of ``subjects[i]`` at ``levels[j]``; the
    matrix is float64, C-contiguous and read-only. Only complete subjects (a value
    at every level) are in it. Rows are in the subject column's order (declared
    categories, else sorted: text by code point, numbers numerically), so they do
    not depend on the order of the input rows or on the data frame library.
    """

    x: str
    y: str
    subject_id: str
    levels: tuple[str, ...]
    subjects: tuple[str, ...]
    values: FloatArray
    n_dropped: int
    """Rows dropped because ``x``, ``y`` or ``subject_id`` was missing (once each)."""
    n_missing_x: int
    n_missing_y: int
    n_missing_subject: int
    """Rows missing each column; a row missing several counts in each."""
    dropped_subjects: tuple[str, ...]
    """Subjects with at least one row but without a complete row at every level,
    in the subject column's order; not analyzed."""
    n_rows_of_dropped_subjects: int
    """Complete rows that belong to ``dropped_subjects`` (not in ``n_dropped``)."""
    empty_levels: tuple[str, ...]
    """Declared levels of ``x`` without a complete row, in declared order."""
    level_source: LevelSource
    dictionary_order: tuple[str, ...] = ()
    """As ``BetweenData.dictionary_order``."""
    vanished_levels: tuple[str, ...] = ()
    """Levels of an ``x`` without declared categories whose rows all miss ``y`` or
    ``subject_id``, sorted; not analyzed."""
    lookalike_levels: tuple[tuple[str, ...], ...] = ()
    """As ``BetweenData.lookalike_levels``."""
    natural_order: tuple[str, ...] = ()
    """As ``BetweenData.natural_order``."""

    @property
    def n_subjects(self) -> int:
        """Complete subjects: the matrix's rows."""
        return len(self.subjects)

    def by_level(self) -> tuple[FloatArray, ...]:
        """One read-only array per level, each in subject order (the matrix's
        columns), so entry i of every level belongs to ``subjects[i]``."""
        columns: list[FloatArray] = []
        for j in range(len(self.levels)):
            column = np.ascontiguousarray(self.values[:, j], dtype=np.float64)
            column.setflags(write=False)
            columns.append(column)
        return tuple(columns)


# --- frames and columns -----------------------------------------------------------


def to_frame(data: object) -> nw.DataFrame[Any]:
    """Wrap ``data`` as an eager narwhals DataFrame or raise a ``TypeError`` that says
    what to pass instead."""
    name = _type_name(data)
    if isinstance(data, np.ndarray | list | tuple | dict):
        raise TypeError(
            f"data must be a data frame, not a {name}; {_LONG_FORMAT_HINT}."
        )
    try:
        return nw.from_native(data, eager_only=True)
    except TypeError:
        pass
    except DuplicateError:
        _raise_duplicate_columns(data)
        raise
    if "Lazy" in name:
        hint = "collect it into an eager DataFrame first, for example with .collect()"
    elif name.endswith(("Series", "ChunkedArray", "Array")):
        hint = "pass the whole data frame and name the columns as x and y"
    else:
        hint = _LONG_FORMAT_HINT
    raise TypeError(
        f"data must be a pandas, polars or pyarrow DataFrame, not a {name}; {hint}."
    ) from None


def _raise_duplicate_columns(data: object) -> None:
    """Raise ``ValueError`` naming the first label ``data`` uses for two columns
    (pandas ``columns``, pyarrow ``column_names``); return if none is found."""
    labels: object = getattr(data, "column_names", None)
    if labels is None:
        labels = getattr(data, "columns", None)
    if not isinstance(labels, Iterable):
        return
    seen: set[object] = set()
    for label in cast("Iterable[object]", labels):
        if label in seen:
            raise ValueError(
                f"data has more than one column named {label!r}; rename the columns "
                "so that each name is used once."
            ) from None
        seen.add(label)


def _type_name(obj: object) -> str:
    cls = type(obj)
    module = cls.__module__.split(".")[0]
    return cls.__name__ if module == "builtins" else f"{module}.{cls.__name__}"


# How many column labels a message about the labels shows.
_MAX_LABELS_SHOWN: Final = 6


def _non_string_labels(frame: nw.DataFrame[Any]) -> str | None:
    """The way forward when the frame's column labels are not all strings, as
    ``pandas.DataFrame(array)`` makes them; None when they are strings."""
    labels: list[object] = list(frame.columns)
    if all(isinstance(c, str) for c in labels):
        return None
    shown = ", ".join(repr(c) for c in labels[:_MAX_LABELS_SHOWN])
    more = ", ..." if len(labels) > _MAX_LABELS_SHOWN else ""
    return (
        f"The column labels of data are not strings ({shown}{more}), as "
        "pandas.DataFrame(array) makes them, and inferplot names columns by "
        "strings: name the columns, for example pandas.DataFrame(array, "
        "columns=['group', 'score']), or turn the labels into strings with "
        "data.rename(columns=str)"
    )


def _column(frame: nw.DataFrame[Any], name: object, role: str) -> nw.Series[Any]:
    """The column called ``name``, or ``ColumnNotFoundError`` with the closest match."""
    if not isinstance(name, str):
        raise _not_a_name(frame, name, role)
    columns = [str(c) for c in frame.columns]
    if name in frame.columns:
        return frame.get_column(name)
    labels_fix = _non_string_labels(frame)
    if labels_fix is not None and name in columns:
        raise ColumnNotFoundError(
            f"{role}={name!r} is not a column of data. {labels_fix} and keep "
            f"{role}={name!r}."
        )
    folded = [c for c in columns if c.casefold() == name.casefold()]
    close = folded or difflib.get_close_matches(
        name, columns, n=1, cutoff=_CLOSE_MATCH_CUTOFF
    )
    suggestion = f" Did you mean {close[0]!r}?" if close else ""
    available = ", ".join(repr(c) for c in columns) or "(none)"
    raise ColumnNotFoundError(
        f"{role}={name!r} is not a column of data.{suggestion} "
        f"Available columns: {available}."
    )


_EXAMPLE_COLUMN: Final = {
    "x": "group",
    "y": "score",
    "subject_id": "subject",
    "label_var": "label",
}


def _not_a_name(frame: nw.DataFrame[Any], name: object, role: str) -> TypeError:
    """The error for a column argument that is not a string: a column itself (a
    Series, array or list) is named by its type, never printed."""
    start = f"{role} must be a column name (a string), not"
    if isinstance(name, str | bytes | numbers.Number) or name is None:
        message = f"{start} {name!r}."
        labels_fix = _non_string_labels(frame)
        if labels_fix is not None and name in list(frame.columns):
            message += f" {labels_fix} and pass {role}={str(name)!r}."
        return TypeError(message)
    own = getattr(name, "name", None)
    if isinstance(own, str) and own in frame.columns:
        fix = f"pass the column's name, for example {role}={own!r}."
    else:
        example = _EXAMPLE_COLUMN.get(role, "name")
        fix = f"pass the name of a column of data, for example {role}={example!r}."
    return TypeError(f"{start} a {_type_name(name)}: {fix}")


# --- labels -----------------------------------------------------------------------


def level_label(value: object) -> str:
    """The name R gives a factor level made from ``value``.

    Text is kept as is; integers and whole-number floats print without ``.0`` (R reads
    ``4`` from a CSV as the integer ``4L``, labelled ``"4"``); other floats use 15
    significant digits, as R's ``as.character()``. Booleans, missing values and other
    types raise ``TypeError``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"a level cannot be a boolean ({value!r})")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, np.integer):
        return str(int(cast("np.integer[Any]", value)))
    if isinstance(value, float | np.floating):
        number = float(cast("float | np.floating[Any]", value))
        if not math.isfinite(number):
            raise TypeError(f"a level cannot be {number!r}")
        if number.is_integer():
            return str(int(number))
        return format(number, ".15g")
    raise TypeError(
        f"a level must be text or a number, not {type(value).__name__} ({value!r})"
    )


# --- missing values ---------------------------------------------------------------


def _mask(series: nw.Series[Any]) -> BoolArray:
    """``series`` (boolean, without nulls) as a writable NumPy bool array.

    Always a copy: pandas 3 (copy-on-write) returns a read-only view from
    ``to_numpy()``, so in-place ``|=`` on it fails.
    """
    return np.array(series.to_numpy(), dtype=np.bool_, copy=True)


def _missing(series: nw.Series[Any]) -> BoolArray:
    """True where ``series`` is null, or NaN in a float column (R's ``is.na()``).

    pandas nullable floats can hold NA and NaN separately (pandas 2), and ``is_nan``
    is null at a null, hence ``fill_null(False)``.
    """
    missing = _mask(series.is_null())
    if series.dtype.is_float():
        missing |= _mask(series.is_nan().fill_null(value=False))
    return missing


def _float_values(series: nw.Series[Any]) -> FloatArray:
    """Every value as float64; entries that are missing hold an arbitrary number.

    Nulls are filled with 0.0 before ``to_numpy`` because nullable and Arrow-backed
    pandas columns at the pandas floor (2.1) return object arrays holding ``<NA>``, and
    filling with NaN would turn back into NA in pandas 3. Callers mask with
    ``_missing``.

    Integers beyond 2**53 are converted one by one (round to nearest, as a cast
    does): pyarrow's cast refuses them even when the double is exact.
    ``_check_exact_integers`` is what refuses the ones a double cannot hold.
    """
    if series.dtype.is_integer() and _beyond_exact_integers(series):
        return np.array(
            [
                float(v) if isinstance(v, numbers.Integral) else 0.0
                for v in series.to_list()
            ],
            dtype=np.float64,
        )
    as_float = series.cast(nw.Float64()).fill_null(value=0.0)
    return np.asarray(as_float.to_numpy(), dtype=np.float64)


# Every integer of magnitude up to 2**53 is a float64 exactly; above it, only some.
_EXACT_INTEGER_LIMIT: Final = 2**53


def _beyond_exact_integers(series: nw.Series[Any]) -> bool:
    """Whether an integer column holds a value of magnitude above 2**53."""
    low, high = series.min(), series.max()
    if low is None or high is None:  # every value missing
        return False
    return int(low) < -_EXACT_INTEGER_LIMIT or int(high) > _EXACT_INTEGER_LIMIT


# --- grouping columns: x and subject_id ---------------------------------------------

XKind = Literal["declared", "text", "integer", "float"]


@dataclass(frozen=True, slots=True)
class _Role:
    """How the messages about a grouping column name it and what it is for."""

    arg: str  # the argument that names the column: "x" or "subject_id"
    purpose: str
    bool_hint: str
    not_whole_hint: str


_X_ROLE: Final = _Role(
    arg="x",
    purpose="define groups",
    bool_hint=(
        "convert it to text labels first, for example "
        "df[x].map({True: 'yes', False: 'no'})"
    ),
    not_whole_hint=(
        "a grouping column must hold text, categories or whole numbers. For a "
        "numeric x use scatterstats; to group by these values, convert them to text "
        "first."
    ),
)
_SUBJECT_ROLE: Final = _Role(
    arg="subject_id",
    purpose="identify subjects",
    bool_hint=(
        "a column with two values cannot tell subjects apart; check that "
        "subject_id names the column of subject identifiers"
    ),
    not_whole_hint=(
        "a subject column must hold text, categories or whole numbers. Check that "
        "subject_id names the column of subject identifiers, not a measurement; to "
        "use these values as identifiers, convert them to text first."
    ),
)


def _undeclared_dictionary(
    series: nw.Series[Any],
) -> tuple[nw.Series[Any], list[object] | None]:
    """A pyarrow dictionary column whose type is not marked ordered, decoded to its
    values, and its dictionary values in dictionary order; any other column
    unchanged, with ``None``.

    Arrow declares an order only with ``ordered=True``. ``dictionary_encode()``,
    ``pyarrow.csv``'s ``auto_dict_encode`` and polars' ``.to_arrow()`` make
    unordered dictionaries listing the values as they first appear, which depends
    on the row order; decoded, such a column is read as its values are (text sorted,
    numbers numerically). The dictionary of a column of several chunks is unified
    first, so it is one list.
    """
    if not series.implementation.is_pyarrow():
        return series, None
    native: Any = series.to_native()
    dtype = native.type
    if not hasattr(dtype, "index_type") or dtype.ordered:
        return series, None
    unified = native.unify_dictionaries()
    dictionary: list[object] = (
        unified.chunk(0).dictionary.to_pylist() if unified.num_chunks else []
    )
    decoded = nw.from_native(native.cast(dtype.value_type), series_only=True)
    return decoded, dictionary


def _dictionary_order(
    dictionary: list[object] | None, levels: Sequence[str]
) -> tuple[str, ...]:
    """``levels`` in the order of an unordered dictionary, when that differs from
    ``levels``; otherwise empty."""
    if dictionary is None:
        return ()
    present = set(levels)
    ordered: list[str] = []
    for value in dictionary:
        if value is None:
            continue
        try:
            label = level_label(value)
        except TypeError:
            return ()
        if label in present and label not in ordered:
            ordered.append(label)
    return tuple(ordered) if ordered != list(levels) else ()


def _what_values(value: object) -> Literal["dates or times", "intervals"] | None:
    """Whether ``value`` is a date, time or duration (Python, NumPy or pandas), or
    a pandas Interval (``pandas.cut``, ``pandas.qcut``): values that are neither
    text nor numbers, for which converting to text is the fix."""
    if isinstance(
        value,
        datetime.date
        | datetime.time
        | datetime.timedelta
        | np.datetime64
        | np.timedelta64,
    ):
        return "dates or times"
    cls = type(value)
    if cls.__module__.split(".")[0] == "pandas":
        if cls.__name__ == "Interval":
            return "intervals"
        if cls.__name__ == "Period":
            return "dates or times"
    return None


def _first_value(series: nw.Series[Any]) -> object:
    """The first value of ``series`` that is not missing, or None."""
    try:
        values = series.drop_nulls().head(1).to_list()
    except Exception:  # an unusual column type narwhals cannot handle
        return None
    return values[0] if values else None


def _unusable_values(
    role: _Role, name: str, what: Literal["dates or times", "intervals"], shown: str
) -> TypeError:
    """The error for a grouping column of dates, times or intervals, also as a
    categorical: converting to text is what makes them usable."""
    if what == "intervals":
        detail = f"intervals (bins made by pandas.cut or pandas.qcut, {shown})"
        fix = (
            f"example data[{name!r}].astype(str) in pandas; "
            f"data[{name!r}].cat.rename_categories(str) does so and keeps the bins "
            "in their order."
        )
    else:
        detail = f"dates or times ({shown})"
        fix = (
            f"example data[{name!r}].astype(str) in pandas or "
            f"pl.col({name!r}).cast(pl.String) in polars."
        )
    return TypeError(
        f"{role.arg}={name!r} holds {detail}, which cannot {role.purpose} as they "
        f"are. Convert them to text first, for {fix}"
    )


def _label_error(error: TypeError, value: object, name: str, role: _Role) -> TypeError:
    """``level_label``'s error for ``value`` with the column named, or the message
    for dates, times and intervals."""
    what = _what_values(value)
    if what is not None:
        return _unusable_values(role, name, what, repr(value))
    return TypeError(f"{role.arg}={name!r}: {error}")


def _x_kind(series: nw.Series[Any], name: str, role: _Role = _X_ROLE) -> XKind:
    dtype = series.dtype
    if isinstance(dtype, nw.Enum):
        return "declared"
    if isinstance(dtype, nw.Categorical):
        # polars Categorical: categories in encounter order up to polars 1.31, then
        # global and sorted lexically; neither is an order the user declared.
        return "text" if series.implementation.is_polars() else "declared"
    if isinstance(dtype, nw.String):
        return "text"
    if dtype.is_integer():
        return "integer"
    if dtype.is_float():
        return "float"
    if isinstance(dtype, nw.Datetime | nw.Date | nw.Time | nw.Duration):
        raise _unusable_values(role, name, "dates or times", str(dtype))
    if isinstance(dtype, nw.Boolean):
        hint = role.bool_hint
    else:
        what = _what_values(_first_value(series))
        if what is not None:
            raise _unusable_values(role, name, what, str(dtype))
        hint = "convert it to text or to a categorical column first"
    raise TypeError(
        f"{role.arg}={name!r} must hold text, categories or whole numbers to "
        f"{role.purpose}; its type is {dtype}. {hint}."
    )


def _declared_categories(
    series: nw.Series[Any], name: str, role: _Role = _X_ROLE
) -> list[object]:
    try:
        categories = series.cat.get_categories().to_list()
    except AttributeError:
        # pandas ArrowDtype(dictionary): narwhals reports Categorical but has no
        # .cat accessor for it (narwhals 2.0 to 2.28).
        raise TypeError(
            f"{role.arg}={name!r} is a pandas Arrow dictionary column, which is not "
            "supported; convert it with .astype('category')."
        ) from None
    return [c for c in categories if c is not None]


def _labels_in_order(
    values: Sequence[object], name: str, role: _Role = _X_ROLE
) -> list[str]:
    """Distinct labels of ``values`` with the label of each first occurrence."""
    labels: list[str] = []
    seen: set[str] = set()
    origin: dict[str, object] = {}
    for value in values:
        try:
            label = level_label(value)
        except TypeError as error:
            raise _label_error(error, value, name, role) from None
        if label in seen:
            if origin[label] != value:
                raise ValueError(
                    f"{role.arg}={name!r} has two different values that both print "
                    f"as {label!r} ({origin[label]!r} and {value!r})."
                )
            continue
        seen.add(label)
        origin[label] = value
        labels.append(label)
    return labels


def _row_labels(raw: Sequence[object], name: str, role: _Role = _X_ROLE) -> list[str]:
    """Each row's label, computing the label once per distinct value.

    Values are str, int or float here (categories or text), all hashable.
    """
    cache: dict[Hashable, str] = {}
    out: list[str] = []
    for value in raw:
        key: Hashable = value  # pyright: ignore[reportAssignmentType]
        label = cache.get(key)
        if label is None:
            try:
                label = level_label(value)
            except TypeError as error:
                raise _label_error(error, value, name, role) from None
            cache[key] = label
        out.append(label)
    return out


@dataclass(frozen=True, slots=True)
class _Grouping:
    """A grouping column (``x`` or ``subject_id``) read once: its kind, which rows
    are missing, its declared categories and, for text and categories, the raw
    value of every row."""

    name: str
    role: _Role
    kind: XKind
    missing: BoolArray
    categories: list[object]
    raw: list[object]


def _read_grouping(series: nw.Series[Any], name: str, role: _Role) -> _Grouping:
    kind = _x_kind(series, name, role)
    missing = _missing(series)
    categories = _declared_categories(series, name, role) if kind == "declared" else []
    # Raw values. Never to_numpy() on a categorical: pyarrow puts a dictionary
    # value where a null was. to_list() is right everywhere but slow on pyarrow
    # dictionaries (1.2 s per million rows); casting text categories to String first
    # takes 0.04 s. Numeric categories must not be cast (4.0 would become "4.0").
    raw: list[object] = []
    if kind == "text" or (
        kind == "declared" and all(isinstance(c, str) for c in categories)
    ):
        raw = series.cast(nw.String()).to_list()
    elif kind == "declared":
        raw = series.to_list()
    if kind == "declared":
        # A row pointing at a null dictionary entry is null in to_list() but not in
        # is_null() at pyarrow 14 (it is at pyarrow 26).
        missing |= np.fromiter((v is None for v in raw), np.bool_, len(raw))
    return _Grouping(name, role, kind, missing, categories, raw)


def _grouping_labels(
    group: _Grouping, series: nw.Series[Any], rows: IntArray
) -> list[str]:
    """The label of each of ``rows`` (row positions where the column is present)."""
    if group.kind in ("declared", "text"):
        return _row_labels([group.raw[i] for i in rows], group.name, group.role)
    if group.kind == "integer":
        # Python ints from to_list() are exact for any 64-bit integer; float64
        # is not above 2**53.
        full: list[Any] = series.to_list()
        return [str(int(full[i])) for i in rows]
    values = _float_values(series)[rows]
    bad = ~np.isfinite(values) | (values != np.round(values))
    if bad.any():
        example = float(values[bad][0])
        hint = group.role.not_whole_hint
        distinct = len(set(values.tolist()))
        if group.role is _X_ROLE and 2 * distinct > values.size:
            # As text, most groups would hold one row: a measurement, not groups.
            hint = (
                f"its {values.size} values hold {distinct} different numbers, so "
                "it looks like a measurement, not a column of groups (as text, most "
                "groups would hold a single row). To relate two measurements use "
                "scatterstats; if each row holds one subject's values in several "
                "columns (wide format), reshape them to long format first and "
                "compare them with withinstats(data, x, y, subject_id=...)."
            )
        raise TypeError(
            f"{group.role.arg}={group.name!r} holds numbers that are not whole "
            f"({example!r}); {hint}"
        )
    return [str(int(v)) for v in values]


def _reshape_example(
    frame: nw.DataFrame[Any], subject_id: str, columns: Sequence[str]
) -> str:
    """The call that reshapes wide ``columns`` of ``frame`` to long format, in the
    frame's own library, with the withinstats call that follows."""
    melt = (
        f"id_vars={subject_id!r}, value_vars={list(columns)!r}, "
        "var_name='condition', value_name='value'"
    )
    implementation = frame.implementation
    if implementation.is_polars():
        reshape = (
            f"long = data.unpivot(index={subject_id!r}, on={list(columns)!r}, "
            "variable_name='condition', value_name='value')"
        )
    elif implementation.is_pyarrow():
        reshape = f"long = data.to_pandas().melt({melt})"
    else:
        reshape = f"long = data.melt({melt})"
    return (
        f"{reshape}, then withinstats(long, 'condition', 'value', "
        f"subject_id={subject_id!r})"
    )


def _wide_format(
    frame: nw.DataFrame[Any], group: _Grouping, x_col: nw.Series[Any], y: str,
    subject_id: str,
) -> TypeError | None:  # fmt: skip
    """withinstats with an ``x`` of measurements (numbers that are not whole) next
    to a numeric ``y``: the data are very likely in wide format, one column per
    condition. The error that says so and shows the reshape; None otherwise."""
    if group.kind != "float":
        return None
    values = _float_values(x_col)[~group.missing]
    bad = np.isfinite(values) & (values != np.round(values))
    if not bad.any():
        return None
    x = group.name
    return TypeError(
        f"x={x!r} holds numbers that are not whole ({float(values[bad][0])!r}), "
        f"like y={y!r}: it looks like a measurement, so the data are very likely in "
        f"wide format, one row per subject and one column per condition ({x!r}, "
        f"{y!r}). withinstats() needs long format: one row per subject and "
        "condition, with a column naming the condition and a column of values. "
        f"Reshape first: {_reshape_example(frame, subject_id, [x, y])}."
    )


def _grouping_order(
    group: _Grouping, present: set[str]
) -> tuple[list[str], list[str], LevelSource]:
    """The ``present`` labels in the column's order, the declared labels that are
    not present, and where the order comes from.

    Declared categories keep their order; text sorts by code point (Python's
    ``sorted()``, R's C-locale order); numbers sort numerically.
    """
    if group.kind == "declared":
        declared = _labels_in_order(group.categories, group.name, group.role)
        undeclared = sorted(present.difference(declared))
        if undeclared:  # cannot happen for a valid categorical; never guess
            raise ValueError(
                f"{group.role.arg}={group.name!r} has values outside its "
                f"categories: {undeclared!r}."
            )
        ordered = [label for label in declared if label in present]
        empty = [label for label in declared if label not in present]
        return ordered, empty, "declared"
    if group.kind == "text":
        return sorted(present), [], "sorted"
    return sorted(present, key=int), [], "sorted"


# A number in a level name: a run of ASCII digits, with a minus sign where the name
# or a word starts with it ("-10", not the hyphen of "group-10" or "1-5"); read as
# a whole number, or together with the decimals that follow it.
_WHOLE_NUMBERS: Final = re.compile(r"((?:(?<!\S)-)?[0-9]+)")
_DECIMAL_NUMBERS: Final = re.compile(r"((?:(?<!\S)-)?[0-9]+(?:\.[0-9]+)?)")


def _natural_key(label: str, numbers: re.Pattern[str]) -> tuple[str | Fraction, ...]:
    """``label`` split into text and the numbers ``numbers`` finds, the numbers as
    exact values: text at even places, numbers at odd ones, so two keys compare
    place by place."""
    parts = numbers.split(label)
    return tuple(Fraction(part) if n % 2 else part for n, part in enumerate(parts))


def _by_numbers(levels: Sequence[str], numbers: re.Pattern[str]) -> list[str]:
    return sorted(levels, key=lambda label: (_natural_key(label, numbers), label))


def _natural_order(group: _Grouping, levels: Sequence[str]) -> tuple[str, ...]:
    """Text ``levels`` in sorted order, ordered by the numbers in their names when
    that differs ("week 2" before "week 10"); empty otherwise. Numbers sort
    numerically and declared categories keep their order, so neither has one.

    Empty too where the names can be read two ways: "1.10" is after "1.5" as a
    version and before it as a decimal, and "0.25", "0.5" are in order as
    decimals. The order is given only when reading every run of digits as a
    whole number and reading digits, a point and digits as one decimal number
    agree.
    """
    if group.kind != "text":
        return ()
    natural = _by_numbers(levels, _WHOLE_NUMBERS)
    if natural == list(levels) or natural != _by_numbers(levels, _DECIMAL_NUMBERS):
        return ()
    return tuple(natural)


def _quiet_labels(group: _Grouping, series: nw.Series[Any], rows: IntArray) -> set[str]:
    """The labels of ``rows`` that can be made, without raising for the others.

    For the rows ``read_between`` drops because ``y`` is missing: their ``x`` is
    not analyzed, so a value that cannot be a level (a float that is not whole)
    is not an error there; it is only not reported as a level.
    """
    if rows.size == 0:
        return set()
    labels: set[str] = set()
    if group.kind in ("declared", "text"):
        for i in rows.tolist():
            with contextlib.suppress(TypeError):
                labels.add(level_label(group.raw[i]))
    elif group.kind == "integer":
        full: list[Any] = series.to_list()
        labels.update(str(int(full[i])) for i in rows.tolist())
    else:
        values = _float_values(series)[rows]
        whole = values[np.isfinite(values) & (values == np.round(values))]
        labels.update(str(int(v)) for v in whole.tolist())
    return labels


def _vanished(group: _Grouping, seen: set[str], present: set[str]) -> list[str]:
    """Labels ``seen`` in rows with ``x`` present but not ``present`` among the
    analyzed rows, in the column's order. Declared levels are reported as empty
    levels instead, so they are left out here."""
    if group.kind == "declared":
        return []
    return _grouping_order(group, seen - present)[0]


def _lookalikes(levels: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Sets of ``levels`` that are equal once leading and trailing whitespace is
    removed, in level order."""
    by_core: dict[str, list[str]] = {}
    for level in levels:
        by_core.setdefault(level.strip(), []).append(level)
    return tuple(tuple(same) for same in by_core.values() if len(same) > 1)


def _order_item(item: object) -> object:
    """A pyarrow scalar as its Python value; anything else unchanged."""
    if type(item).__module__.split(".")[0] == "pyarrow":
        as_py = getattr(item, "as_py", None)
        if callable(as_py):
            return as_py()
    return item


def _order_label(item: object) -> str:
    """The level label of one ``order`` entry: text, or a number ``x`` can hold.

    Integers of any kind (Python, NumPy) and finite floats are accepted and labelled
    as ``x``'s values are (``8.0`` and ``np.int64(8)`` both name level ``"8"``).
    """
    value = _order_item(item)
    if isinstance(value, str):
        return str(value)  # a plain str, also for numpy.str_
    if isinstance(value, bool | np.bool_):
        pass
    elif isinstance(value, numbers.Integral):
        return str(int(value))
    elif isinstance(value, float | np.floating):
        number = float(cast("float | np.floating[Any]", value))
        if math.isfinite(number):
            return level_label(number)
    raise TypeError(f"order entries must be level names, not {value!r}.")


def _check_order(
    order: object,
    present: Sequence[str],
    empty: Sequence[str],
    *,
    complete: str = "both x and y",
) -> list[str]:
    """``order`` as labels, if it is a permutation of ``present``.

    ``order`` may be any iterable other than a string, a set or a mapping: a list,
    a tuple, a generator, a NumPy array, a pandas Index or Series, a polars Series
    or a pyarrow array. A set (also a frozenset or a dict's keys view) is refused
    because its iteration order is not one the caller chose, and a mapping because
    it is ambiguous (keys or values). Typed ``object``: callers that skip type
    checking can pass anything. ``complete`` names the columns a row needs to be
    analyzed, for the message about declared levels without such rows.
    """
    kind = type(order).__name__
    if isinstance(order, str | bytes) or not isinstance(order, Iterable):
        raise TypeError(
            "order must be a list of level names, for example "
            f"order={list(present)!r}; got {order!r}."
        )
    if isinstance(order, Set | Mapping):
        why = (
            "a set has no order, so the levels would come out in an arbitrary one"
            if isinstance(order, Set)
            else "a mapping is ambiguous: its keys or its values could be meant"
        )
        raise TypeError(
            f"order must list the levels in the order to use, but got a "
            f"{kind}: {why}. Pass a list, for example "
            f"order={list(present)!r}."
        )
    labels = [_order_label(item) for item in cast("Iterable[object]", order)]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"order names {duplicates!r} more than once.")
    unknown = [label for label in labels if label not in present]
    missing = [label for label in present if label not in labels]
    if unknown or missing:
        parts: list[str] = []
        if unknown:
            no_rows = [u for u in unknown if u in empty]
            other = [u for u in unknown if u not in empty]
            if no_rows:
                parts.append(f"{no_rows!r} have no rows with {complete} present")
            if other:
                parts.append(f"{other!r} are not levels of x")
        if missing:
            parts.append(f"{missing!r} are missing from order")
        raise ValueError(
            "order must list every level present exactly once: "
            + "; ".join(parts)
            + f". The levels are {list(present)!r}."
        )
    return labels


# --- y ----------------------------------------------------------------------------


def _check_y(series: nw.Series[Any], name: str, x: nw.Series[Any], x_name: str) -> None:
    """``y`` must be numeric; booleans are refused. A text ``y`` next to a numeric
    ``x`` is named as a likely swap."""
    dtype = series.dtype
    if isinstance(dtype, nw.Boolean):
        hint = (
            "Booleans are not measurements; convert to 0/1 deliberately if that is "
            "what you mean."
        )
    elif not dtype.is_numeric():
        x_dtype = x.dtype
        if x_dtype.is_numeric() and not isinstance(x_dtype, nw.Boolean):
            hint = (
                f"x={x_name!r} is numeric, so x and y may be swapped: x is the "
                f"grouping column and y the measurement, as in x={name!r}, "
                f"y={x_name!r}."
            )
        else:
            _check_sparse(series, name, "y")
            library = _library(series)
            first = _first_value(series)
            if _numbers_as_text(first):
                categories = isinstance(dtype, nw.Categorical | nw.Enum)
                hint = (
                    f"It holds numbers stored as text (for example {first!r}). "
                    "Convert it first: "
                    f"{_to_numbers(library, name, categories=categories)}. Entries "
                    "that are not numbers become missing values, which are left out "
                    "like the other missing values."
                )
            else:
                hint = _generic_numbers(library, name)
    else:
        return
    raise TypeError(f"y={name!r} must be numeric; its type is {dtype}. {hint}")


def _check_finite(
    name: str, values: FloatArray, rows: IntArray, arg: str = "y"
) -> None:
    """An infinite value of a measurement (``arg``, default ``y``) in a row that is
    otherwise complete raises."""
    infinite = ~np.isfinite(values)
    if infinite.any():
        shown = ", ".join(str(r) for r in rows[infinite][:_MAX_ROWS_SHOWN])
        raise ValueError(
            f"{arg}={name!r} has {int(infinite.sum())} infinite value(s) (row "
            f"positions {shown}{', ...' if infinite.sum() > _MAX_ROWS_SHOWN else ''}); "
            "remove them or replace them with missing values."
        )


def _check_exact_integers(
    series: nw.Series[Any], name: str, rows: IntArray, arg: str = "y"
) -> None:
    """An integer measurement (``arg``, default ``y``) that float64 cannot hold
    exactly raises.

    The statistics run in float64. An integer above 2**53 (a nanosecond timestamp,
    a large identifier or count) is rounded to a multiple of a power of two there,
    so nearby values collapse into ties and every statistic changes, the rank
    tests included. Only the analyzed ``rows`` are checked.
    """
    if not series.dtype.is_integer() or rows.size == 0:
        return
    if not _beyond_exact_integers(series):
        return
    full: list[Any] = series.to_list()
    kept = [int(full[r]) for r in rows.tolist()]
    rounded = [v for v in kept if int(float(v)) != v]
    if not rounded:
        return
    example = rounded[0]
    raise ValueError(
        f"{arg}={name!r} holds integers too large to analyze exactly: "
        f"{len(rounded)} of its {len(kept)} values change when stored as "
        "double-precision numbers "
        f"(for example {example} becomes {int(float(example))}), so different values "
        "would become ties. Subtract a common offset or change the unit first, for "
        f"example {arg} - {arg}.min() (smallest value: {min(kept)}); that leaves "
        "every test and standardized effect size as it is."
    )


# --- betweenstats -----------------------------------------------------------------


def read_between(
    data: object,
    x: str,
    y: str,
    *,
    order: Iterable[object] | None = None,
    min_levels: int = 2,
) -> BetweenData:
    """Read a between-groups design: ``y`` split by the levels of ``x``.

    Raises ``TypeError`` for unsupported data or column types, ``ColumnNotFoundError``
    (a ``KeyError``) for a missing column, and ``ValueError`` for infinite ``y``
    values, fewer than ``min_levels`` levels with data, or a bad ``order``.
    """
    frame = to_frame(data)
    x_col, x_dictionary = _undeclared_dictionary(_column(frame, x, "x"))
    y_col = _column(frame, y, "y")
    if x == y:
        raise ValueError(f"x and y must be different columns; both are {x!r}.")

    _check_y(y_col, y, x_col, x)
    group = _read_grouping(x_col, x, _X_ROLE)
    missing_x = group.missing
    missing_y = _missing(y_col)
    keep = ~(missing_x | missing_y)
    rows = np.nonzero(keep)[0]

    _check_exact_integers(y_col, y, rows)
    y_kept = _float_values(y_col)[keep]
    _check_finite(y, y_kept, rows)

    row_labels = _grouping_labels(group, x_col, rows)
    levels, empty, source = _grouping_order(group, set(row_labels))
    dictionary_order = _dictionary_order(x_dictionary, levels)
    natural_order = _natural_order(group, levels)
    lost = np.nonzero(~missing_x & missing_y)[0]
    vanished = _vanished(group, _quiet_labels(group, x_col, lost), set(levels))

    if order is not None:
        levels = _check_order(order, levels, [*empty, *vanished])
        source = "order"
        dictionary_order = ()
        natural_order = ()
    if len(levels) < min_levels:
        raise ValueError(
            f"x={x!r} needs at least {min_levels} groups with data; found "
            f"{len(levels)} ({levels!r}) after dropping rows with missing values."
        )

    index = {label: i for i, label in enumerate(levels)}
    codes = np.fromiter(
        (index[label] for label in row_labels), np.intp, len(row_labels)
    )
    values: list[FloatArray] = []
    for i in range(len(levels)):
        group = np.ascontiguousarray(y_kept[codes == i], dtype=np.float64)
        group.setflags(write=False)
        values.append(group)
    n_per_level = tuple(int(v.size) for v in values)
    return BetweenData(
        x=x,
        y=y,
        levels=tuple(levels),
        values=tuple(values),
        n_per_level=n_per_level,
        n_obs=int(sum(n_per_level)),
        n_dropped=int(len(keep) - rows.size),
        n_missing_x=int(missing_x.sum()),
        n_missing_y=int(missing_y.sum()),
        empty_levels=tuple(empty),
        level_source=source,
        dictionary_order=dictionary_order,
        vanished_levels=tuple(vanished),
        lookalike_levels=_lookalikes(levels),
        natural_order=natural_order,
    )


# --- withinstats ------------------------------------------------------------------

_COMPLETE_WITHIN: Final = "x, y and subject_id all"


def _duplicates(
    subject_id: str,
    x: str,
    keys: Sequence[tuple[str, str]],
    rows: IntArray,
    missing_y: BoolArray,
) -> None:
    """Raise if a subject has more than one row at a level of ``x``.

    ``keys`` are the (subject, level) labels of ``rows``: every row where both
    ``subject_id`` and ``x`` are present, whether or not ``y`` is.
    """
    positions: dict[tuple[str, str], list[int]] = {}
    repeated: list[tuple[str, str]] = []
    for key, row in zip(keys, rows.tolist(), strict=True):
        found = positions.setdefault(key, [])
        found.append(row)
        if len(found) == 2:
            repeated.append(key)
    if not repeated:
        return
    shown = repeated[:_MAX_ROWS_SHOWN]
    cells = "; ".join(
        f"subject {subject!r} at {level!r} (row positions "
        f"{', '.join(str(r) for r in positions[subject, level])})"
        for subject, level in shown
    )
    more = (
        f"; and {len(repeated) - len(shown)} more" if len(repeated) > len(shown) else ""
    )
    with_missing = any(missing_y[r] for key in shown for r in positions[key])
    note = (
        " A row whose y is missing counts too: it still records that subject at "
        "that level, so remove it if it is a placeholder."
        if with_missing
        else ""
    )
    count = len(repeated)
    raise ValueError(
        f"subject_id={subject_id!r} must identify each subject at most once per "
        f"level of x={x!r}, but {count} subject-level "
        f"{'combination has' if count == 1 else 'combinations have'} more than one "
        f"row: {cells}{more}. Combine the repeated rows first (for example their "
        "mean), or make the repetition part of x." + note
    )


def _too_few_subjects(
    *,
    subject_id: str,
    x: str,
    levels: Sequence[str],
    complete: Sequence[str],
    seen: int,
    first_lacking: tuple[str, list[str]] | None,
    min_subjects: int,
    wide: str = "",
) -> ValueError:
    names = ", ".join(repr(level) for level in levels)
    if complete:
        found = (
            f"only {len(complete)} ({', '.join(repr(s) for s in complete)}) "
            f"{'has' if len(complete) == 1 else 'have'}"
        )
    else:
        found = "none has"
    example = ""
    if first_lacking is not None:
        subject, lacking = first_lacking
        example = (
            f" For example, subject {subject!r} has no value at "
            f"{', '.join(repr(level) for level in lacking)}."
        )
    return ValueError(
        f"withinstats needs at least {min_subjects} subjects with a value of y at "
        f"every level of x={x!r} ({names}); of the {seen} subjects in "
        f"subject_id={subject_id!r}, {found} a value at every level after rows "
        f"with missing values were dropped.{example} Subjects missing a level are "
        "left out of a within-subjects analysis; check the levels and the subject "
        f"column.{wide}"
    )


def read_within(
    data: object,
    x: str,
    y: str,
    subject_id: str,
    *,
    order: Iterable[object] | None = None,
    min_levels: int = 2,
    min_subjects: int = 2,
) -> WithinData:
    """Read a within-subjects design: ``y`` for each subject at each level of ``x``.

    The data are in long format, one row per subject and level. The rules
    (SPEC.md section 3):

    * ``y`` as for ``read_between``. ``x`` and ``subject_id`` may each be text,
      categorical, integer, or float holding only whole numbers; booleans are
      refused. Subject labels are made as level labels are (whole numbers without
      ``.0``), and two values that print alike raise.
    * Rows with a missing ``x``, ``y`` or ``subject_id`` are dropped and counted
      (``n_dropped``, rows).
    * A subject with more than one row at the same level raises ``ValueError``
      naming the first repeats. The check covers every row with a subject and a
      level, also one whose ``y`` is missing.
    * An infinite ``y`` in a row with all three columns present raises, also when
      its subject is left out later.
    * Levels: as for ``read_between``, from the complete rows, then ``order=``.
    * Subjects without a complete row at every level are left out of the analysis
      and listed in ``dropped_subjects`` (subjects, not rows). A subject is listed
      when it has any row with ``subject_id`` present, so a subject whose rows all
      miss ``y`` is listed too.
    * At least ``min_levels`` levels and ``min_subjects`` complete subjects.

    Raises ``TypeError`` for unsupported data or column types, ``ColumnNotFoundError``
    (a ``KeyError``) for a missing column, and ``ValueError`` for the data errors
    above, a column used twice, or a bad ``order``.
    """
    frame = to_frame(data)
    x_col, x_dictionary = _undeclared_dictionary(_column(frame, x, "x"))
    y_col = _column(frame, y, "y")
    s_col, _ = _undeclared_dictionary(_column(frame, subject_id, "subject_id"))
    if x == y:
        raise ValueError(f"x and y must be different columns; both are {x!r}.")
    if subject_id in (x, y):
        role = "x" if subject_id == x else "y"
        raise ValueError(
            f"subject_id must name the column that identifies each subject, a "
            f"column other than x and y; subject_id={subject_id!r} is also {role}."
        )
    _check_y(y_col, y, x_col, x)
    x_group = _read_grouping(x_col, x, _X_ROLE)
    wide = _wide_format(frame, x_group, x_col, y, subject_id)
    if wide is not None:
        raise wide
    s_group = _read_grouping(s_col, subject_id, _SUBJECT_ROLE)
    for group in (x_group, s_group):
        # Two categories that print alike would otherwise look like a repeat.
        _labels_in_order(group.categories, group.name, group.role)
    missing_y = _missing(y_col)

    # Labels of every row that has the column, so subjects and duplicates can be
    # found among rows whose y is missing too.
    s_rows = np.nonzero(~s_group.missing)[0]
    x_rows = np.nonzero(~x_group.missing)[0]
    subject_of: dict[int, str] = dict(
        zip(s_rows.tolist(), _grouping_labels(s_group, s_col, s_rows), strict=True)
    )
    level_of: dict[int, str] = dict(
        zip(x_rows.tolist(), _grouping_labels(x_group, x_col, x_rows), strict=True)
    )
    keyed = np.nonzero(~(s_group.missing | x_group.missing))[0]
    _duplicates(
        subject_id,
        x,
        [(subject_of[r], level_of[r]) for r in keyed.tolist()],
        keyed,
        missing_y,
    )

    keep = ~(s_group.missing | x_group.missing | missing_y)
    rows = np.nonzero(keep)[0]
    _check_exact_integers(y_col, y, rows)
    y_kept = _float_values(y_col)[keep]
    _check_finite(y, y_kept, rows)

    row_levels = [level_of[r] for r in rows.tolist()]
    row_subjects = [subject_of[r] for r in rows.tolist()]
    levels, empty, source = _grouping_order(x_group, set(row_levels))
    dictionary_order = _dictionary_order(x_dictionary, levels)
    natural_order = _natural_order(x_group, levels)
    vanished = _vanished(x_group, set(level_of.values()), set(levels))
    if order is not None:
        levels = _check_order(
            order, levels, [*empty, *vanished], complete=_COMPLETE_WITHIN
        )
        source = "order"
        dictionary_order = ()
        natural_order = ()
    if len(levels) < min_levels:
        raise ValueError(
            f"x={x!r} needs at least {min_levels} conditions with data; found "
            f"{len(levels)} ({levels!r}) after dropping rows with missing values."
        )

    level_index = {label: j for j, label in enumerate(levels)}
    present: dict[str, set[str]] = {}
    for subject, level in zip(row_subjects, row_levels, strict=True):
        present.setdefault(subject, set()).add(level)
    complete_set = {s for s, have in present.items() if len(have) == len(levels)}
    seen_set = set(subject_of.values())
    subjects, _, _ = _grouping_order(s_group, complete_set)
    dropped, _, _ = _grouping_order(s_group, seen_set - complete_set)
    if len(subjects) < min_subjects:
        first_lacking = None
        if dropped:
            have = present.get(dropped[0], set())
            first_lacking = (dropped[0], [lv for lv in levels if lv not in have])
        wide = ""
        if x_group.kind in ("integer", "float") and all(
            len(have) <= 1 for have in present.values()
        ):
            # One row per subject and numbers in x: very likely wide format.
            wide = (
                f" Every subject has a single row and x={x!r} holds numbers, so the "
                "data are very likely in wide format, one row per subject and one "
                f"column per condition ({x!r}, {y!r}). Reshape them to long format "
                f"first: {_reshape_example(frame, subject_id, [x, y])}."
            )
        raise _too_few_subjects(
            subject_id=subject_id,
            x=x,
            levels=levels,
            complete=subjects,
            seen=len(seen_set),
            first_lacking=first_lacking,
            min_subjects=min_subjects,
            wide=wide,
        )

    subject_index = {label: i for i, label in enumerate(subjects)}
    matrix = np.empty((len(subjects), len(levels)), dtype=np.float64)
    in_dropped = 0
    for value, subject, level in zip(
        y_kept.tolist(), row_subjects, row_levels, strict=True
    ):
        i = subject_index.get(subject)
        if i is None:
            in_dropped += 1
        else:
            matrix[i, level_index[level]] = value
    matrix.setflags(write=False)
    return WithinData(
        x=x,
        y=y,
        subject_id=subject_id,
        levels=tuple(levels),
        subjects=tuple(subjects),
        values=matrix,
        n_dropped=int(len(keep) - rows.size),
        n_missing_x=int(x_group.missing.sum()),
        n_missing_y=int(missing_y.sum()),
        n_missing_subject=int(s_group.missing.sum()),
        dropped_subjects=tuple(dropped),
        n_rows_of_dropped_subjects=in_dropped,
        empty_levels=tuple(empty),
        level_source=source,
        dictionary_order=dictionary_order,
        vanished_levels=tuple(vanished),
        lookalike_levels=_lookalikes(levels),
        natural_order=natural_order,
    )


# --- histostats and scatterstats: measurements without groups ----------------------


@dataclass(frozen=True, slots=True)
class OneData:
    """The analyzed values of one numeric column (``histostats``).

    ``values`` holds the non-missing values of ``x`` in the frame's row order, as a
    read-only float64 array.
    """

    x: str
    values: FloatArray
    n_dropped: int
    """Rows dropped because ``x`` was missing."""

    @property
    def n_obs(self) -> int:
        """The analyzed values."""
        return int(self.values.size)


@dataclass(frozen=True, slots=True)
class PairData:
    """The complete pairs of two numeric columns (``scatterstats``).

    A row is analyzed when both ``x`` and ``y`` are present; ``x_values[i]`` and
    ``y_values[i]`` are the i-th such row's values, in the frame's row order
    (read-only float64 arrays). ``labels[i]`` is that row's ``label_var`` value as
    text, "" where it is missing or empty; ``labels`` is empty without
    ``label_var``.
    """

    x: str
    y: str
    x_values: FloatArray
    y_values: FloatArray
    n_dropped: int
    """Rows dropped because ``x`` or ``y`` was missing (a row counts once)."""
    n_missing_x: int
    n_missing_y: int
    """Rows missing ``x`` and rows missing ``y``; a row missing both counts in each."""
    label_var: str | None = None
    labels: tuple[str, ...] = ()

    @property
    def n_obs(self) -> int:
        """The complete pairs."""
        return int(self.x_values.size)


# The function each measurement reader serves, for its messages.
_OneOrPair = Literal["histostats", "scatterstats"]


Library = Literal["pandas", "polars", "pyarrow"]


def _library(series: nw.Series[Any]) -> Library:
    """The library of the user's frame, for recipes written in it."""
    impl = series.implementation
    if impl.is_polars():
        return "polars"
    if impl.is_pyarrow():
        return "pyarrow"
    return "pandas"


def _replace_column(
    library: Library, name: str, pandas: str, polars: str, arrow: str
) -> str:
    """A statement that replaces column ``name`` of ``data`` with the
    expression for ``library`` (pandas: a Series; polars: an expression;
    pyarrow: an array)."""
    if library == "polars":
        return f"data = data.with_columns(({polars}).alias({name!r}))"
    if library == "pyarrow":
        return (
            f"data = data.set_column(data.column_names.index({name!r}), {name!r}, "
            f"{arrow})"
        )
    return f"data[{name!r}] = {pandas}"


# A number as text, after trimming spaces (pyarrow's recipe keeps only these).
_NUMBER_PATTERN: Final = r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$"


def _to_numbers(library: Library, name: str, *, categories: bool) -> str:
    """A statement that converts the text (or categories) of column ``name`` to
    numbers in ``library``, entries that are not numbers becoming missing."""
    col = f"data[{name!r}]"
    pandas = f"pandas.to_numeric({col}, errors='coerce')"
    text = f"pl.col({name!r})" + (".cast(pl.String)" if categories else "")
    polars = f"{text}.str.strip_chars().cast(pl.Float64, strict=False)"
    arrow_text = f"pyarrow.compute.cast({col}, pyarrow.string())" if categories else col
    trimmed = f"pyarrow.compute.utf8_trim_whitespace({arrow_text})"
    arrow = (
        "pyarrow.compute.cast(pyarrow.compute.if_else("
        f"pyarrow.compute.match_substring_regex({trimmed}, r'{_NUMBER_PATTERN}'), "
        f"{trimmed}, None), pyarrow.float64())"
    )
    return _replace_column(library, name, pandas, polars, arrow)


def _not_numbers(series: nw.Series[Any]) -> list[str]:
    """The distinct non-missing entries of a text column that do not read as
    numbers, in order of appearance."""
    try:
        values = series.drop_nulls().to_list()
    except Exception:  # an unusual column type narwhals cannot handle
        return []
    return list(dict.fromkeys(str(v) for v in values if not _numbers_as_text(str(v))))


def _numbers_as_text(value: object) -> bool:
    """Whether ``value`` is text that reads as a number ("1.5", " 2 ")."""
    if not isinstance(value, str):
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _grouping_hint(
    name: str, fn: _OneOrPair, other: tuple[str, nw.Series[Any]] | None
) -> str:
    """Where a column of groups belongs: betweenstats (a measurement across its
    groups) or barstats (counting its categories)."""
    if fn == "histostats":
        return (
            "histostats() compares one numeric column with test_value. To compare "
            f"a numeric column across the groups of {name!r}, use "
            f"betweenstats(data, x={name!r}, y='<numeric column>'); to count its "
            f"categories, use barstats(data, x={name!r})."
        )
    start = "scatterstats() relates two numeric columns."
    if other is not None:
        other_name, other_col = other
        dtype = other_col.dtype
        if dtype.is_numeric() and not isinstance(dtype, nw.Boolean):
            return (
                f"{start} To compare {other_name!r} across the groups of {name!r}, "
                f"use betweenstats(data, x={name!r}, y={other_name!r}); to relate "
                "two columns of categories, use barstats(data, x=..., y=...)."
            )
    return (
        f"{start} To compare a numeric column across the groups of {name!r}, use "
        f"betweenstats(data, x={name!r}, y='<numeric column>'); to relate two "
        "columns of categories, use barstats(data, x=..., y=...)."
    )


def _check_measurement(
    series: nw.Series[Any],
    name: str,
    arg: str,
    fn: _OneOrPair,
    other: tuple[str, nw.Series[Any]] | None = None,
) -> None:
    """A column that ``fn`` analyzes as numbers (``arg``: "x" or "y") must be
    numeric. Text and categories are told where groups belong, booleans that they
    are not measurements, dates that they need converting, and numbers held as
    text or Python objects how to convert them. ``other`` is the other column of
    scatterstats, so the advice can name a call that works."""
    dtype = series.dtype
    if dtype.is_numeric() and not isinstance(dtype, nw.Boolean):
        return
    library = _library(series)
    _check_sparse(series, name, arg)
    start = f"{arg}={name!r} must be numeric; its type is {dtype}."
    if isinstance(dtype, nw.Boolean):
        raise TypeError(
            f"{arg}={name!r} holds True/False values, which are not measurements. To "
            "count them, convert them to text labels first, for example "
            f"data[{name!r}].map({{True: 'yes', False: 'no'}}), and use "
            f"barstats(data, x={name!r}); to analyze them as 0 and 1, convert them "
            "deliberately first."
        )
    if isinstance(dtype, nw.Datetime | nw.Date | nw.Time | nw.Duration):
        raise TypeError(
            f"{start} It holds dates or times, which {fn}() cannot analyze as "
            "numbers. Convert them to numbers in the unit you mean first"
            + _time_recipe(library, name, dtype)
        )
    first = _first_value(series)
    if _numbers_as_text(first):
        bad = _not_numbers(series)
        categories = isinstance(dtype, nw.Categorical | nw.Enum)
        held = (
            "categories whose names are numbers"
            if categories
            else "numbers stored as text"
        )
        missing = ""
        if bad:
            shown = ", ".join(repr(v) for v in bad[:3])
            more = f" and {len(bad) - 3} more" if len(bad) > 3 else ""
            missing = (
                f" The entries that are not numbers ({shown}{more}) become missing "
                "values, which are left out like the other missing values."
            )
        raise TypeError(
            f"{start} It holds {held} (for example {first!r}). "
            f"Convert it first: {_to_numbers(library, name, categories=categories)}."
            + missing
        )
    if isinstance(dtype, nw.String | nw.Categorical | nw.Enum) or isinstance(
        first, str
    ):
        raise TypeError(
            f"{arg}={name!r} holds text or categories, not numbers. "
            f"{_grouping_hint(name, fn, other)}"
        )
    if _what_values(first) == "dates or times":
        raise TypeError(
            f"{start} It holds dates or times, which {fn}() cannot analyze as "
            "numbers. Convert them to numbers in the unit you mean first."
        )
    if isinstance(first, numbers.Real) and not isinstance(first, bool | np.bool_):
        raise TypeError(
            f"{start} It holds numbers in a column of Python objects; convert it "
            f"first: {_to_numbers(library, name, categories=False)}."
        )
    raise TypeError(f"{start} {_generic_numbers(library, name)}")


def _generic_numbers(library: Library, name: str) -> str:
    """Advice to convert a column of another type to numbers, in ``library``."""
    if library == "polars":
        how = f"pl.col({name!r}).cast(pl.Float64)"
    elif library == "pyarrow":
        how = f"pyarrow.compute.cast(data[{name!r}], pyarrow.float64())"
    else:
        how = f"pandas.to_numeric(data[{name!r}])"
    return f"Convert it to numbers first, for example with {how}."


def _check_sparse(series: nw.Series[Any], name: str, arg: str) -> None:
    """A pandas sparse column (which narwhals reads as of an unknown type) is
    told what it is and how to make it dense."""
    native = nw.to_native(series)
    dtype = getattr(native, "dtype", None)
    if type(dtype).__name__ != "SparseDtype":
        return
    raise TypeError(
        f"{arg}={name!r} is a pandas sparse column ({dtype}): it stores only the "
        "entries that differ from its fill value, and inferplot reads dense "
        f"columns only. Make it dense first: data[{name!r}] = "
        f"data[{name!r}].sparse.to_dense()."
    )


def _time_recipe(library: Library, name: str, dtype: object) -> str:
    """A recipe that turns dates or durations into numbers in ``library``,
    after "Convert them to numbers in the unit you mean first"."""
    col = f"data[{name!r}]"
    if isinstance(dtype, nw.Duration):
        how = _replace_column(
            library,
            name,
            f"{col}.dt.total_seconds()",
            f"pl.col({name!r}).dt.total_seconds()",
            f"pyarrow.compute.cast({col}, pyarrow.int64())",
        )
        if library == "pyarrow":
            return f", for example the durations in the column's own unit: {how}."
        return f", for example the seconds: {how}."
    if isinstance(dtype, nw.Datetime | nw.Date):
        how = _replace_column(
            library,
            name,
            f"({col} - {col}.min()).dt.days",
            f"(pl.col({name!r}) - pl.col({name!r}).min()).dt.total_days()",
            f"pyarrow.compute.days_between(pyarrow.compute.min({col}), {col})",
        )
        return f", for example the days since the first date: {how}."
    return "."


def _measurement(
    series: nw.Series[Any], name: str, arg: str, rows: IntArray
) -> FloatArray:
    """The values of ``rows`` as a read-only float64 array, after the checks every
    measurement gets: integers a double cannot hold exactly and infinite values
    raise."""
    _check_exact_integers(series, name, rows, arg)
    values = np.ascontiguousarray(_float_values(series)[rows], dtype=np.float64)
    _check_finite(name, values, rows, arg)
    values.setflags(write=False)
    return values


def read_one(data: object, x: str, *, min_values: int = 2) -> OneData:
    """Read one numeric column for ``histostats``: its non-missing values in row
    order.

    Rules (SPEC.md section 3): ``x`` must be numeric (integers, floats, decimals;
    booleans, text, categories and dates are refused, with a message that says
    which function fits); rows with a missing ``x`` (null, NaN) are dropped and
    counted, as ggstatsplot's ``tidyr::drop_na()`` on ``x`` does; an infinite
    value, or an integer a double cannot hold exactly, raises. Fewer than
    ``min_values`` values raise: no type can test fewer than two.

    Raises ``TypeError`` for unsupported data or column types,
    ``ColumnNotFoundError`` (a ``KeyError``) for a missing column, and
    ``ValueError`` for the data errors above.
    """
    frame = to_frame(data)
    column = _column(frame, x, "x")
    _check_measurement(column, x, "x", "histostats")
    missing = _missing(column)
    rows = np.nonzero(~missing)[0]
    values = _measurement(column, x, "x", rows)
    n_dropped = int(missing.sum())
    if values.size < min_values:
        found = f"found {values.size}" if values.size else "found none"
        dropped = (
            f" after dropping {n_dropped} {'row' if n_dropped == 1 else 'rows'} "
            "with a missing value"
            if n_dropped
            else ""
        )
        raise ValueError(
            f"histostats needs at least {min_values} values of x={x!r} to test "
            f"against test_value; {found}{dropped} (data has {len(missing)} "
            f"{'row' if len(missing) == 1 else 'rows'})."
        )
    return OneData(x=x, values=values, n_dropped=n_dropped)


def _label_text(value: object) -> str:
    """A label value as text: "" for a missing one; text as given; whole numbers
    without ".0" and other numbers with 15 significant digits (R's labels);
    anything else (booleans, dates) as Python prints it."""
    if value is None:
        return ""
    if isinstance(value, str):
        return str(value)
    try:
        if bool(value != value):  # NaN, NaT, pandas NA
            return ""
    except (TypeError, ValueError):  # pandas NA's comparison is NA
        return ""
    if isinstance(value, bool | np.bool_):
        return "True" if value else "False"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return level_label(float(cast("float | np.floating[Any]", value)))
    return str(value)


def _point_labels(series: nw.Series[Any], rows: IntArray) -> tuple[str, ...]:
    """The label of each of ``rows``; "" where the label is missing or only
    white space (nothing would be drawn: no label)."""
    raw: list[object] = series.to_list()
    texts = (_label_text(raw[i]) for i in rows.tolist())
    return tuple(text if text.strip() else "" for text in texts)


def read_pair(
    data: object,
    x: str,
    y: str,
    *,
    label_var: str | None = None,
    min_pairs: int = 3,
) -> PairData:
    """Read two numeric columns for ``scatterstats``: the complete pairs in row
    order.

    Rules (SPEC.md section 3): ``x`` and ``y`` are different numeric columns,
    each checked as ``read_one`` checks its column; a row is analyzed when both
    are present (``n_dropped`` counts the others, each once); ``label_var`` is any
    column, never read as numbers, whose values label the analyzed points as text
    ("" where missing or empty: no label); a missing label does not drop the row.
    Fewer than ``min_pairs`` complete pairs raise: a correlation's test needs at
    least three.

    Raises ``TypeError`` for unsupported data or column types,
    ``ColumnNotFoundError`` (a ``KeyError``) for a missing column, and
    ``ValueError`` for the data errors above or ``x`` and ``y`` naming one column.
    """
    frame = to_frame(data)
    x_col = _column(frame, x, "x")
    y_col = _column(frame, y, "y")
    label_col = None if label_var is None else _column(frame, label_var, "label_var")
    if x == y:
        raise ValueError(
            f"x and y must be different columns; both are {x!r}. A column's "
            "correlation with itself is 1 by construction."
        )
    _check_measurement(x_col, x, "x", "scatterstats", (y, y_col))
    _check_measurement(y_col, y, "y", "scatterstats", (x, x_col))
    missing_x = _missing(x_col)
    missing_y = _missing(y_col)
    keep = ~(missing_x | missing_y)
    rows = np.nonzero(keep)[0]
    x_values = _measurement(x_col, x, "x", rows)
    y_values = _measurement(y_col, y, "y", rows)
    n_dropped = int(len(keep) - rows.size)
    if rows.size < min_pairs:
        found = f"found {rows.size}" if rows.size else "found none"
        dropped = (
            f" after dropping {n_dropped} {'row' if n_dropped == 1 else 'rows'} "
            "with a missing value"
            if n_dropped
            else ""
        )
        raise ValueError(
            f"scatterstats needs at least {min_pairs} complete pairs (rows with both "
            f"x={x!r} and y={y!r} present) to test a correlation; {found}{dropped} "
            f"(data has {len(keep)} {'row' if len(keep) == 1 else 'rows'})."
        )
    labels = () if label_col is None else _point_labels(label_col, rows)
    return PairData(
        x=x,
        y=y,
        x_values=x_values,
        y_values=y_values,
        n_dropped=n_dropped,
        n_missing_x=int(missing_x.sum()),
        n_missing_y=int(missing_y.sum()),
        label_var=label_var,
        labels=labels,
    )


# --- notes for the result's warnings ----------------------------------------------

# How many levels a note names before it shortens the list.
_MAX_LEVELS_SHOWN: Final = 6


def _quoted(levels: Sequence[str]) -> str:
    """``'a'``, ``'a' and 'b'``, ``'a', 'b' and 'c'``."""
    quoted = [repr(level) for level in levels]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def _listed(levels: Sequence[str]) -> str:
    """``'a', 'b', 'c'``, shortened after ``_MAX_LEVELS_SHOWN`` levels."""
    if len(levels) <= _MAX_LEVELS_SHOWN:
        return ", ".join(repr(level) for level in levels)
    shown = ", ".join(repr(level) for level in levels[: _MAX_LEVELS_SHOWN - 1])
    return f"{shown}, ... ({len(levels)} levels)"


def data_notes(data: BetweenData | WithinData | OneData | PairData) -> list[str]:
    """Plain-English notes on how the data were read, for the result's
    ``warnings``: an unordered dictionary whose order was not used, levels left out
    because every row of them misses a value, and levels that differ only in
    surrounding whitespace. They name levels of ``x``, never subjects (D36).
    Column names with control characters get a note in every design; the
    designs without levels (``OneData``, ``PairData``) get only that note and one
    on the rows left out for a missing value, which never names a row's values."""
    if isinstance(data, OneData | PairData):
        return _measurement_notes(data)
    notes: list[str] = []
    x = data.x
    if data.dictionary_order:
        notes.append(
            f"The levels of {x!r} are in sorted order ({_listed(data.levels)}), not "
            f"in the order of its pyarrow dictionary "
            f"({_listed(data.dictionary_order)}): a dictionary that is not marked "
            "ordered, as dictionary_encode(), pyarrow.csv and polars' .to_arrow() "
            "make, lists values in the order they first appear in the rows, so it "
            "does not declare an order. To choose the order, pass order= (a list "
            "of the levels) or mark the dictionary ordered."
        )
    if data.natural_order:
        notes.append(_natural_order_note(x, data.levels, data.natural_order))
    vanished = data.vanished_levels
    if vanished:
        needs = (
            f"{data.y!r} or {data.subject_id!r}"
            if isinstance(data, WithinData)
            else repr(data.y)
        )
        one = len(vanished) == 1
        notes.append(
            f"The {'level' if one else 'levels'} {_quoted(vanished)} of {x!r} "
            f"{'was' if one else 'were'} left out of the analysis: every row of "
            f"{'it' if one else 'them'} has a missing {needs}."
        )
    for level in data.levels:
        drawn = _control_characters(level)
        if drawn is not None:
            notes.append(
                f"The level {level!r} of {x!r} holds control characters {drawn}; the "
                "result keeps the name as given."
            )
    for arg, name in (("x", x), ("y", data.y)):
        drawn = _control_characters(name)
        if drawn is not None:
            notes.append(
                f"The column name {arg}={name!r} holds control characters {drawn} "
                "in the axis label; the result keeps the name as given."
            )
    for same in data.lookalike_levels:
        notes.append(
            f"The levels {_quoted(same)} of {x!r} differ only in spaces at their "
            "start or end, so they are analyzed as separate groups whose labels "
            "look alike. If they are the same group, remove the spaces first, for "
            "example with .str.strip() in pandas or .str.strip_chars() in polars."
        )
    return notes


def _measurement_notes(data: OneData | PairData) -> list[str]:
    """``data_notes`` for one column (histostats) or a pair of columns
    (scatterstats): the rows left out and control characters in column names."""
    notes: list[str] = []
    dropped = data.n_dropped
    if dropped:
        rows = f"{dropped:,} {'row was' if dropped == 1 else 'rows were'}"
        if isinstance(data, OneData):
            notes.append(
                f"{rows} left out because {data.x!r} is missing; "
                f"{data.n_obs:,} values remain."
            )
        else:
            notes.append(
                f"{rows} left out because {data.x!r} or {data.y!r} is missing "
                f"({data.n_missing_x:,} without {data.x!r}, {data.n_missing_y:,} "
                f"without {data.y!r}); {data.n_obs:,} complete pairs remain."
            )
    names = [("x", data.x)]
    if isinstance(data, PairData):
        names.append(("y", data.y))
    for arg, name in names:
        drawn = _control_characters(name)
        if drawn is not None:
            notes.append(
                f"The column name {arg}={name!r} holds control characters {drawn} "
                "in the axis label; the result keeps the name as given."
            )
    return notes


def _control_characters(name: str) -> str | None:
    """How the figure draws the control characters of ``name`` (all but the line
    break), "(U+0007), which the figure draws each as ...", or None without any."""
    controls = sorted(
        {c for c in name if c != "\n" and unicodedata.category(c) == "Cc"}
    )
    if not controls:
        return None
    codes = ", ".join(f"U+{ord(c):04X}" for c in controls)
    drawn = (
        "a tab as a space"
        if controls == ["\t"]
        else "a tab as a space and any other as \ufffd (U+FFFD)"
        if "\t" in controls
        else "each as \ufffd (U+FFFD)"
    )
    return f"({codes}), which the figure draws {drawn}"


# An order= example longer than this is described instead of written out.
_MAX_EXAMPLE: Final = 120


def _natural_order_note(x: str, levels: Sequence[str], natural: Sequence[str]) -> str:
    """The note for text levels sorted as text where the numbers in their names
    give another order: the first place they differ, and how order= sets it."""
    first = next(
        n for n, (a, b) in enumerate(zip(levels, natural, strict=True)) if a != b
    )
    example = f"order={list(natural)!r}"
    how = (
        f"pass {example}"
        if len(example) <= _MAX_EXAMPLE
        else "pass order= with the levels in the order you want (for example by "
        "the numbers in their names)"
    )
    return (
        f"The levels of {x!r} are text, sorted as text (as R sorts them), so "
        f"{levels[first]!r} comes before {natural[first]!r}: {_listed(levels)}. "
        "The analysis and the figure use this order. To order them by the numbers "
        f"in their names, {how}."
    )


# --- barstats: a table of counts from categorical columns ---------------------------
#
# ``read_table`` reads what ggstatsplot 1.1.4's ``.pie_bar_data_prep()`` keeps for
# ggbarstats and ggpiestats: the rows with x, y and counts present
# (``tidyr::drop_na`` on those columns only), each repeated ``counts`` times
# (``tidyr::uncount``: a count of 0 removes the row), then every column
# ``as.factor()`` (declared levels kept, empty ones included; otherwise the sorted
# values). Rows are never repeated here: each row adds its count to its cell.


@dataclass(frozen=True, slots=True)
class TableLevels:
    """The levels of one column of a count table (``x`` or ``y``) and how they
    were read.

    ``levels`` holds every level in order: the declared categories, empty ones
    included (also listed in ``empty``), else the sorted values that hold a
    count. ``vanished`` names undeclared values seen only in rows left out (a
    missing value in another used column, or a count of 0), which R's
    ``as.factor()`` never sees.
    """

    name: str
    levels: tuple[str, ...]
    source: LevelSource
    empty: tuple[str, ...] = ()
    """Declared levels without any count, in declared order (kept in ``levels``)."""
    vanished: tuple[str, ...] = ()
    dictionary_order: tuple[str, ...] = ()
    """As ``BetweenData.dictionary_order`` (of the levels that hold a count)."""
    natural_order: tuple[str, ...] = ()
    """As ``BetweenData.natural_order`` (of the levels that hold a count)."""
    lookalike_levels: tuple[tuple[str, ...], ...] = ()
    """Sets of levels that differ only in leading or trailing whitespace."""

    @property
    def present(self) -> tuple[str, ...]:
        """The levels that hold a count, in order."""
        return tuple(level for level in self.levels if level not in self.empty)


@dataclass(frozen=True, slots=True)
class TableData:
    """The table of counts a barstats design analyzes.

    ``counts[i][j]`` is the number of observations at level i of ``x`` (the fill;
    rows of the table) and level j of ``y`` (the bars; columns), as Python ints;
    without ``y`` (one-way) every row has a single entry. Rows and columns follow
    ``x_levels.levels`` and ``y_levels.levels``, declared empty levels included, as
    ggstatsplot holds them. With ``paired`` the table is square over the same
    levels in the same order (``x`` and ``y`` are the two measurements of each
    unit).
    """

    x: str
    y: str | None
    counts_var: str | None
    paired: bool
    x_levels: TableLevels
    y_levels: TableLevels | None
    counts: tuple[tuple[int, ...], ...]
    n_rows: int
    """Rows with ``x``, ``y`` and ``counts`` present (zero counts included)."""
    n_dropped: int
    """Rows left out because ``x``, ``y`` or ``counts`` is missing (once each)."""
    n_missing_x: int
    n_missing_y: int
    n_missing_counts: int
    """Rows missing each column; a row missing several counts in each."""
    n_zero_rows: int
    """Rows with every column present whose count is 0: they add nothing
    (``tidyr::uncount``)."""
    library: Library = "pandas"
    """The library of the user's frame, for recipes written in it."""

    @property
    def n(self) -> int:
        """The table total: observations (pairs with ``paired``)."""
        return sum(sum(row) for row in self.counts)

    def table(self) -> npt.NDArray[np.int64]:
        """``counts`` as a read-only int64 array (rows ``x``, columns ``y``)."""
        out = np.array(self.counts, dtype=np.int64)
        out.setflags(write=False)
        return out

    def column_totals(self) -> tuple[int, ...]:
        """The observations in each level of ``y`` (one entry without ``y``)."""
        return tuple(sum(column) for column in zip(*self.counts, strict=True))

    def row_totals(self) -> tuple[int, ...]:
        """The observations in each level of ``x``."""
        return tuple(sum(row) for row in self.counts)


_TABLE_ROLES: Final = {
    arg: _Role(
        arg=arg,
        purpose="be counted as categories",
        bool_hint="",  # booleans are categories here (_read_category reads them)
        not_whole_hint=(
            "barstats counts observations in categories (text, categorical, "
            "booleans or whole-number codes), not measurements."
        ),
    )
    for arg in ("x", "y")
}
# Whole numbers with more distinct values than this, over fewer than twice as many
# observations, look like a measurement rather than category codes.
_MANY_CODES: Final = 10


def _boolean_label(value: object) -> str:
    return "True" if value else "False"


def _read_category(series: nw.Series[Any], name: str, role: _Role) -> _Grouping:
    """A column of categories: text, categoricals, booleans or numbers.

    Booleans are the categories "False" and "True" (in that order, as R sorts its
    logical values; Python's names); a declared categorical of booleans keeps its
    declared order. Everything else is read as betweenstats reads ``x``.
    """
    if isinstance(series.dtype, nw.Boolean):
        missing = _missing(series)
        values: list[object] = series.to_list()
        raw: list[object] = [
            None if gone else _boolean_label(v)
            for v, gone in zip(values, missing.tolist(), strict=True)
        ]
        return _Grouping(name, role, "text", missing, [], raw)
    group = _read_grouping(series, name, role)
    categories = group.categories
    if (
        group.kind == "declared"
        and categories
        and all(isinstance(c, bool | np.bool_) for c in categories)
    ):
        labels: list[object] = [_boolean_label(c) for c in categories]
        raw = [
            None if gone else _boolean_label(v)
            for v, gone in zip(group.raw, group.missing.tolist(), strict=True)
        ]
        return _Grouping(name, role, "declared", group.missing, labels, raw)
    return group


def _looks_measured(series: nw.Series[Any]) -> bool:
    """Whether a column looks like a measurement rather than categories: numeric
    (not boolean) with a value that is not whole, or with more than
    ``_MANY_CODES`` distinct values over fewer than twice as many rows (the rule
    ``_check_codes`` applies, here on every value present)."""
    dtype = series.dtype
    if not dtype.is_numeric() or isinstance(dtype, nw.Boolean):
        return False
    values = _float_values(series)[~_missing(series)]
    if values.size == 0:
        return False
    if bool((~np.isfinite(values) | (values != np.round(values))).any()):
        return True
    distinct = len(set(values.tolist()))
    return distinct > _MANY_CODES and 2 * distinct > values.size


def _measurement_advice(
    arg: str, x: str, y: str | None, other: nw.Series[Any] | None
) -> str:
    """Which function fits when ``arg`` holds measurements, given the other
    column (``other``; None without y)."""
    if other is not None and _looks_measured(other):
        return (
            f"To relate two numeric columns use scatterstats(data, x={x!r}, y={y!r})."
        )
    if arg == "y":
        return (
            f"To compare {y!r} across the groups of {x!r} (for example its mean in "
            f"each group), use betweenstats(data, x={x!r}, y={y!r})."
        )
    if y is None:
        return (
            f"To test one numeric column against a value use histostats(data, "
            f"x={x!r}, test_value=...)."
        )
    return (
        f"To compare {x!r} across the groups of {y!r}, use betweenstats(data, "
        f"x={y!r}, y={x!r})."
    )


def _check_codes(
    group: _Grouping, series: nw.Series[Any], rows: IntArray, n: int, advice: str
) -> None:
    """A numeric column of ``read_table`` must hold category codes: whole numbers
    with few distinct values. Numbers that are not whole, or more than
    ``_MANY_CODES`` distinct whole numbers over fewer than twice as many of the
    ``n`` observations, are a measurement: ``TypeError`` with ``advice``."""
    if group.kind not in ("integer", "float") or rows.size == 0:
        return
    arg, name = group.role.arg, group.name
    if group.kind == "float":
        values = _float_values(series)[rows]
        bad = ~np.isfinite(values) | (values != np.round(values))
        if bad.any():
            raise TypeError(
                f"{arg}={name!r} holds numbers that are not whole "
                f"({float(values[bad][0])!r}), so it looks like a measurement: "
                "barstats counts observations in categories (text, categorical, "
                f"booleans or whole-number codes). {advice}"
            )
        distinct = len(set(values.tolist()))
    else:
        full: list[Any] = series.to_list()
        distinct = len({int(full[i]) for i in rows.tolist()})
    if distinct > _MANY_CODES and 2 * distinct > n:
        raise TypeError(
            f"{arg}={name!r} holds {distinct} different numbers over {n:,} "
            "observations, so it looks like a measurement, not a column of "
            "categories (most categories would hold a single observation). "
            f"{advice} To count these values as categories anyway, convert them to "
            f"text first: {_codes_to_text(_library(series), name)}."
        )


def _codes_to_text(library: Library, name: str) -> str:
    """A statement that writes whole-number codes as text, missing values kept
    missing (a pandas float column of whole numbers included), in ``library``."""
    pandas = f"data[{name!r}].astype('Int64').astype('string')"
    polars = f"pl.col({name!r}).cast(pl.Int64).cast(pl.String)"
    arrow = (
        f"pyarrow.compute.cast(pyarrow.compute.cast(data[{name!r}], pyarrow.int64()), "
        "pyarrow.string())"
    )
    return _replace_column(library, name, pandas, polars, arrow)


def _counts_to_numbers(library: Library, name: str) -> str:
    """A statement that reads counts stored as text as numbers, thousands
    separators ("1,000") and surrounding spaces removed, entries that are not
    numbers becoming missing (rows then left out and counted), in ``library``."""
    col = f"data[{name!r}]"
    pandas = (
        f"pandas.to_numeric({col}.str.replace(',', '', regex=False).str.strip(), "
        "errors='coerce')"
    )
    polars = (
        f"pl.col({name!r}).cast(pl.String).str.replace_all(',', '', literal=True)"
        ".str.strip_chars().cast(pl.Float64, strict=False)"
    )
    text = (
        "pyarrow.compute.utf8_trim_whitespace(pyarrow.compute.replace_substring("
        f"pyarrow.compute.cast({col}, pyarrow.string()), pattern=',', "
        "replacement=''))"
    )
    arrow = (
        "pyarrow.compute.cast(pyarrow.compute.if_else("
        f"pyarrow.compute.match_substring_regex({text}, r'{_NUMBER_PATTERN}'), "
        f"{text}, None), pyarrow.float64())"
    )
    return _replace_column(library, name, pandas, polars, arrow)


def _bad_rows(rows: IntArray, bad: BoolArray) -> str:
    shown = ", ".join(str(r) for r in rows[bad][:_MAX_ROWS_SHOWN])
    return shown + (", ..." if bad.sum() > _MAX_ROWS_SHOWN else "")


def _table_weights(series: nw.Series[Any], name: str, rows: IntArray) -> Any:
    """The counts of ``rows`` (present in every used column): an int64 array, or
    an object array of Python ints where a total could pass 2**62.

    Whole numbers only: an integer column, or floats that are whole (a pandas
    integer column with a missing value is float). Booleans, text and other
    types raise ``TypeError``; a negative, infinite or non-whole count raises
    ``ValueError`` naming its row positions (ggbarstats fails outright on 2.5:
    ``tidyr::uncount`` cannot repeat a row 2.5 times).
    """
    dtype = series.dtype
    what = (
        f"counts={name!r} names the column that says how many observations each "
        "row stands for (data given as a table, one row per combination of "
        "categories), so it must hold whole numbers >= 0"
    )
    if isinstance(dtype, nw.Boolean):
        raise TypeError(
            f"{what}; it holds True/False values. To count the rows themselves, "
            "leave counts= out."
        )
    if not dtype.is_numeric():
        first = _first_value(series)
        library = _library(series)
        text = isinstance(first, str)
        if text and _numbers_as_text(first.replace(",", "")):
            hint = (
                f" It holds numbers stored as text (for example {first!r}); convert "
                f"it first: {_counts_to_numbers(library, name)}."
            )
        elif text:
            hint = (
                f" Its entries are not numbers (for example {first!r}): name the "
                "column that holds the numbers of observations."
            )
        else:
            hint = f" {_generic_numbers(library, name)}"
        raise TypeError(f"{what}; its type is {dtype}.{hint}")
    values: Any
    if dtype.is_integer() and _beyond_exact_integers(series):
        full: list[Any] = series.to_list()
        exact = [int(full[r]) for r in rows.tolist()]
        values = np.array(exact, dtype=object)
        negative = np.array([v < 0 for v in exact], dtype=np.bool_)
    elif dtype.is_integer():
        as_int = series.cast(nw.Int64()).fill_null(value=0)
        values = np.asarray(as_int.to_numpy(), dtype=np.int64)[rows]
        negative = values < 0
    else:
        floats = _float_values(series)[rows]
        problem = ~np.isfinite(floats) | (floats != np.round(floats))
        if problem.any():
            first_bad = float(floats[problem][0])
            kind = "infinite" if math.isinf(first_bad) else "not whole"
            raise ValueError(
                f"{what}; {int(problem.sum())} of its values are not whole numbers "
                f"(for example {first_bad!r}, {kind}; row positions "
                f"{_bad_rows(rows, problem)}). ggbarstats fails on such counts too. "
                "If they are weights or proportions rather than numbers of "
                "observations, barstats cannot use them: its tests need counts."
            )
        negative = floats < 0
        if floats.size and float(np.abs(floats).max()) >= 2.0**62:
            values = np.array([int(v) for v in floats.tolist()], dtype=object)
        else:
            values = floats.astype(np.int64)
    if negative.any():
        raise ValueError(
            f"{what}; {int(negative.sum())} of its values are negative (row "
            f"positions {_bad_rows(rows, negative)})."
        )
    # Totals stay exact in int64 below 2**62; beyond that, Python ints.
    if (
        values.dtype == np.int64
        and values.size
        and int(values.max()) > (2**62) // values.size
    ):
        values = np.array([int(v) for v in values.tolist()], dtype=object)
    return values


def _table_levels(
    group: _Grouping,
    series: nw.Series[Any],
    labels: Sequence[str],
    lost: IntArray,
    dictionary: list[object] | None,
) -> TableLevels:
    """The levels of one column: the declared categories (empty ones kept) or
    the sorted labels that hold a count, and the notes about them. ``lost`` are
    the rows with this column present that add nothing to the table."""
    present = set(labels)
    ordered, empty, source = _grouping_order(group, present)
    if group.kind == "declared":
        levels = _labels_in_order(group.categories, group.name, group.role)
    else:
        levels = ordered
    vanished = _vanished(group, _quiet_labels(group, series, lost), present)
    return TableLevels(
        name=group.name,
        levels=tuple(levels),
        source=source,
        empty=tuple(empty),
        vanished=tuple(vanished),
        dictionary_order=_dictionary_order(dictionary, ordered),
        natural_order=_natural_order(group, ordered),
        lookalike_levels=_lookalikes(levels),
    )


def _declare_example(
    frame: nw.DataFrame[Any],
    columns: tuple[str, str],
    categories: Sequence[str],
    numeric: bool,
) -> str:
    """The code that declares both ``columns`` on the same ``categories``, in the
    frame's own library."""
    x, y = columns
    kinds = {name: _text_kind(frame[name].dtype) for name in columns}
    if frame.implementation.is_polars():
        enum = f"pl.Enum({list(categories)!r})"

        def polars_text(name: str) -> str:
            kind = kinds[name]
            if kind == "number":
                return ".cast(pl.Int64).cast(pl.String)"
            if kind == "bool":  # polars writes true/false; the levels are True/False
                return ".cast(pl.String).str.to_titlecase()"
            return ".cast(pl.String)"

        return (
            f"data = data.with_columns(pl.col({x!r}){polars_text(x)}.cast({enum}), "
            f"pl.col({y!r}){polars_text(y)}.cast({enum}))"
        )
    shown: list[object] = [int(c) for c in categories] if numeric else [*categories]
    start = "data = data.to_pandas(); " if frame.implementation.is_pyarrow() else ""

    def pandas_column(name: str) -> str:
        # Text categories need text values: numbers (a whole float with missing
        # values included) and booleans are written as their level labels first,
        # or pd.Categorical would make every value missing.
        if numeric or kinds[name] == "text":
            return f"data[{name!r}]"
        if kinds[name] == "number":
            return f"data[{name!r}].astype('Int64').astype(str)"
        return f"data[{name!r}].astype(str)"

    return (
        f"{start}categories = {shown!r}; "
        f"data[{x!r}] = pd.Categorical({pandas_column(x)}, categories=categories); "
        f"data[{y!r}] = pd.Categorical({pandas_column(y)}, categories=categories)"
    )


def _text_kind(dtype: Any) -> Literal["number", "bool", "text", "other"]:
    """What a column holds, for the recipes that turn it into text levels:
    numbers, booleans, text, or other (categoricals: written as text by
    ``astype(str)``)."""
    if isinstance(dtype, nw.Boolean):
        return "bool"
    if dtype.is_numeric():
        return "number"
    if isinstance(dtype, nw.String):
        return "text"
    return "other"


def _paired_categories(
    x_levels: TableLevels, y_levels: TableLevels, numeric: bool
) -> list[str]:
    """The categories both paired columns can be declared on: a declared
    column's order first, then the other's further levels; else sorted."""
    xs, ys = x_levels.levels, y_levels.levels
    if x_levels.source == "declared":
        return [*xs, *(level for level in ys if level not in xs)]
    if y_levels.source == "declared":
        return [*ys, *(level for level in xs if level not in ys)]
    union = set(xs) | set(ys)
    return sorted(union, key=int) if numeric else sorted(union)


def _check_paired_levels(
    frame: nw.DataFrame[Any],
    x_levels: TableLevels,
    y_levels: TableLevels,
    numeric: bool,
) -> None:
    """``paired=True`` needs the same levels in the same order in ``x`` and
    ``y``: McNemar's test reads the table's cells by position, and
    ``mcnemar.test`` needs a square table (ggstatsplot leaves the subtitle out
    silently when it is not, D101)."""
    xs, ys = x_levels.levels, y_levels.levels
    if xs == ys:
        return
    x, y = x_levels.name, y_levels.name
    if not set(xs) & set(ys):
        recipe = long_paired_recipe(frame.to_native(), x, y, None, ys)
        raise ValueError(
            f"paired=True: x={x!r} and y={y!r} share no category ({x!r}: "
            f"{_listed(xs)}; {y!r}: {_listed(ys)}). paired=True reads each row as "
            "one unit measured twice, x the first measurement and y the second, "
            f"both on the same categories. The data look long (one row per unit "
            f"and occasion, with {y!r} naming the occasion): reshape them to one "
            "row per unit, with one column per occasion, and pass two occasions as "
            f"x and y: {recipe}. If {x!r} and {y!r} are two measurements coded "
            "differently, recode one to the other's categories first."
        )
    start = (
        f"paired=True: x={x!r} and y={y!r} are two measurements of the same units, "
        "so they need the same categories in the same order, and they differ"
    )
    if set(xs) == set(ys):
        detail = (
            f": they hold the same categories in different orders ({x!r}: "
            f"{_listed(xs)}; {y!r}: {_listed(ys)}). McNemar's test pairs the "
            "categories by position, so declare both in the same order"
        )
        categories = list(xs)
    else:
        only_x = [level for level in xs if level not in ys]
        only_y = [level for level in ys if level not in xs]
        parts: list[str] = []
        if only_x:
            parts.append(f"{_quoted(only_x)} only in {x!r}")
        if only_y:
            parts.append(f"{_quoted(only_y)} only in {y!r}")
        detail = (
            f" ({'; '.join(parts)}), so the table of the pairs is not square and "
            "McNemar's test cannot run (ggstatsplot leaves out its subtitle "
            "silently). If every category is possible in both, declare both "
            "columns on the same categories, which keeps a category without rows "
            "as a level with a count of 0"
        )
        categories = _paired_categories(x_levels, y_levels, numeric)
    example = _declare_example(frame, (x, y), categories, numeric)
    raise ValueError(f"{start}{detail}: {example}.")


def _unit_column(frame: nw.DataFrame[Any], x: str, y: str) -> str | None:
    """The column that names each unit in long data (one row per unit and
    occasion ``y``): another column whose values repeat, and that holds each
    occasion at most once per value; None where no column does."""
    for name in frame.columns:
        if name in (x, y):
            continue
        try:
            pair = frame.select(name, y).drop_nulls()
            repeats = pair[name].n_unique() < pair.shape[0]
            if repeats and not bool(pair.is_duplicated().any()):
                return name
        except Exception:  # a column narwhals cannot compare: not a unit column
            continue
    return None


def long_paired_recipe(
    data: object, x: str, y: str, unit: str | None, occasions: Sequence[str]
) -> str:
    """The code that reshapes long data (one row per unit and occasion, ``y``
    naming the occasion and ``x`` holding the answer) to one row per unit, and
    the paired call on two occasions, in the frame's own library. ``unit`` names
    the unit column (found in the frame when None; '<unit>' where none is)."""
    frame = to_frame(data)
    unit = unit or _unit_column(frame, x, y) or "<column naming each unit>"
    first, second = occasions[0], occasions[1] if len(occasions) > 1 else "..."
    call = f"ip.barstats(wide, x={first!r}, y={second!r}, paired=True)"
    if frame.implementation.is_polars():
        return f"wide = data.pivot(on={y!r}, index={unit!r}, values={x!r}); {call}"
    start = "data = data.to_pandas(); " if frame.implementation.is_pyarrow() else ""
    return (
        f"{start}wide = data.pivot(index={unit!r}, columns={y!r}, values={x!r})"
        f".rename(columns=str).reset_index(); {call}"
    )


def _sum(values: Any) -> int:
    """The exact total of an int64 or object array of whole counts."""
    if values.dtype == object:
        return sum(int(v) for v in values.tolist())
    return int(values.sum())


def read_table(
    data: object,
    x: str,
    y: str | None = None,
    *,
    counts: str | None = None,
    paired: bool = False,
) -> TableData:
    """Read a count table for ``barstats``: ``x`` (rows; the fill) by ``y``
    (columns; the bars), or ``x`` alone (one-way).

    Rules (SPEC.md section 4, "barstats", as ggstatsplot 1.1.4 prepares the data):

    * ``x`` and ``y`` hold categories: text, categoricals, booleans ("False",
      "True") or whole numbers (labelled without ".0", ordered numerically).
      Numbers that are not whole, or more than 10 distinct whole numbers over
      fewer than twice as many observations, are a measurement and raise
      ``TypeError`` naming the function that fits.
    * Only ``x``, ``y`` and ``counts`` are checked for missing values: a row
      missing any of them is dropped and counted (``n_dropped``).
    * ``counts`` names a column of whole numbers >= 0 (integers, or floats that are
      whole): each row adds its count to its cell, so very large counts work
      without repeating rows. A count of 0 adds nothing, and a value seen only in
      such rows is not a level (unless declared). A negative, infinite or
      non-whole count raises ``ValueError`` (ggbarstats fails on 2.5).
    * Levels: the declared categories (pandas Categorical, polars Enum, pyarrow
      dictionary marked ordered), empty ones included; otherwise the sorted
      values that hold a count (text as R sorts it in the C locale, numbers
      numerically). An unordered pyarrow dictionary is read as its values.
    * ``paired=True``: ``x`` and ``y`` are the two measurements of each unit
      (``counts`` then counts units, as ggstatsplot reads it: it uncounts before
      pairing), and the table must be square over the same levels in the same
      order; otherwise ``ValueError`` shows how to declare both columns on the
      same categories in the frame's own library.

    Raises ``TypeError`` for unsupported data or column types,
    ``ColumnNotFoundError`` (a ``KeyError``) for a missing column, and
    ``ValueError`` for the data errors above, a column named twice, or no
    observation left.
    """
    frame = to_frame(data)
    x_raw = _column(frame, x, "x")
    y_raw = None if y is None else _column(frame, y, "y")
    counts_col = None if counts is None else _column(frame, counts, "counts")
    if y is not None and x == y:
        raise ValueError(
            f"x and y must be different columns; both are {x!r}. To count the "
            f"categories of one column, leave y out: barstats(data, x={x!r})."
        )
    if counts is not None and counts in (x, y):
        role = "x" if counts == x else "y"
        raise ValueError(
            f"counts={counts!r} is also {role}=; counts names the column of how "
            "many observations each row stands for, not a column of categories."
        )
    x_col, x_dictionary = _undeclared_dictionary(x_raw)
    x_group = _read_category(x_col, x, _TABLE_ROLES["x"])
    y_col: nw.Series[Any] | None = None
    y_dictionary: list[object] | None = None
    y_group: _Grouping | None = None
    if y_raw is not None and y is not None:
        y_col, y_dictionary = _undeclared_dictionary(y_raw)
        y_group = _read_category(y_col, y, _TABLE_ROLES["y"])

    size = len(x_group.missing)
    none = np.zeros(size, dtype=np.bool_)
    missing_x = x_group.missing
    missing_y = none if y_group is None else y_group.missing
    missing_counts = none if counts_col is None else _missing(counts_col)
    rows = np.nonzero(~(missing_x | missing_y | missing_counts))[0]
    weights: Any = np.ones(rows.size, dtype=np.int64)
    if counts_col is not None and counts is not None:
        weights = _table_weights(counts_col, counts, rows)
    positive = np.asarray(weights > 0, dtype=np.bool_)
    used, used_weights = rows[positive], weights[positive]
    n_total = _sum(used_weights)

    _check_codes(x_group, x_col, used, n_total, _measurement_advice("x", x, y, y_col))
    if y_group is not None and y_col is not None:
        advice = _measurement_advice("y", x, y, x_col)
        _check_codes(y_group, y_col, used, n_total, advice)

    unused = np.ones(size, dtype=np.bool_)
    unused[used] = False
    x_labels = _grouping_labels(x_group, x_col, used)
    x_levels = _table_levels(
        x_group, x_col, x_labels, np.nonzero(~missing_x & unused)[0], x_dictionary
    )
    y_levels: TableLevels | None = None
    y_labels: list[str] = []
    if y_group is not None and y_col is not None:
        y_labels = _grouping_labels(y_group, y_col, used)
        y_levels = _table_levels(
            y_group, y_col, y_labels, np.nonzero(~missing_y & unused)[0], y_dictionary
        )
        if paired:
            numeric = {x_group.kind, y_group.kind} <= {"integer", "float"}
            _check_paired_levels(frame, x_levels, y_levels, numeric)

    if n_total == 0:
        needs = _quoted([c for c in (x, y, counts) if c is not None])
        zero = f", and a count above 0 in {counts!r}" if counts is not None else ""
        raise ValueError(
            f"barstats needs at least one observation; no row of data has {needs} "
            f"present{zero} (data has {size:,} {'row' if size == 1 else 'rows'})."
        )

    x_index = {label: i for i, label in enumerate(x_levels.levels)}
    cells = np.fromiter((x_index[v] for v in x_labels), np.intp, len(x_labels))
    n_columns = 1
    if y_levels is not None:
        n_columns = len(y_levels.levels)
        y_index = {label: j for j, label in enumerate(y_levels.levels)}
        y_codes = np.fromiter((y_index[v] for v in y_labels), np.intp, len(y_labels))
        cells = cells * n_columns + y_codes
    n_cells = len(x_levels.levels) * n_columns
    flat: Any
    if counts_col is None:
        flat = np.bincount(cells, minlength=n_cells).astype(np.int64)
    else:
        addends: npt.NDArray[Any] = used_weights
        flat = np.zeros(n_cells, dtype=addends.dtype)
        if flat.dtype == object:
            flat[:] = 0  # Python ints, not the float 0.0 of np.zeros
        np.add.at(flat, cells, addends)  # pyright: ignore[reportUnknownMemberType]
    as_rows: list[list[Any]] = flat.reshape(len(x_levels.levels), n_columns).tolist()
    return TableData(
        x=x,
        y=y,
        counts_var=counts,
        paired=paired,
        x_levels=x_levels,
        y_levels=y_levels,
        counts=tuple(tuple(int(v) for v in row) for row in as_rows),
        n_rows=int(rows.size),
        n_dropped=int(size - rows.size),
        n_missing_x=int(missing_x.sum()),
        n_missing_y=int(missing_y.sum()),
        n_missing_counts=int(missing_counts.sum()),
        n_zero_rows=int(rows.size - used.size),
        library=_library(x_raw),
    )


def _or_listed(names: Sequence[str]) -> str:
    """``'a'``, ``'a' or 'b'``, ``'a', 'b' or 'c'``."""
    quoted = [repr(name) for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def table_notes(data: TableData) -> list[str]:
    """Plain-English notes on how a count table was read, for the result's
    ``warnings``: the rows left out for a missing value, an unordered dictionary
    whose order was not used, text levels in another order than their numbers,
    levels seen only in rows left out, levels that differ only in surrounding
    whitespace, and control characters in level and column names. barstats has
    no ``order=``: a declared categorical sets the order of its levels. Declared
    empty levels are the analysis' to word (what they mean depends on the
    design)."""
    notes: list[str] = []
    used = [(data.x, data.n_missing_x)]
    if data.y is not None:
        used.append((data.y, data.n_missing_y))
    if data.counts_var is not None:
        used.append((data.counts_var, data.n_missing_counts))
    dropped = data.n_dropped
    if dropped:
        detail = ""
        if len(used) > 1:
            detail = (
                " ("
                + ", ".join(f"{count:,} without {name!r}" for name, count in used)
                + ")"
            )
        rows = f"{dropped:,} {'row was' if dropped == 1 else 'rows were'}"
        notes.append(
            f"{rows} left out because {_or_listed([name for name, _ in used])} is "
            f"missing{detail}; {data.n:,} observations remain."
        )
    for column in (data.x_levels, data.y_levels):
        if column is None:
            continue
        notes.extend(_table_level_notes(data, column))
    names = [("x", data.x)]
    if data.y is not None:
        names.append(("y", data.y))
    for arg, name in names:
        drawn = _control_characters(name)
        if drawn is not None:
            notes.append(
                f"The column name {arg}={name!r} holds control characters {drawn} "
                "in the figure's labels; the result keeps the name as given."
            )
    return notes


def _table_level_notes(data: TableData, column: TableLevels) -> list[str]:
    """``table_notes`` for the levels of one column."""
    notes: list[str] = []
    name = column.name
    if column.dictionary_order:
        notes.append(
            f"The levels of {name!r} are in sorted order ({_listed(column.present)}), "
            f"not in the order of its pyarrow dictionary "
            f"({_listed(column.dictionary_order)}): a dictionary that is not marked "
            "ordered, as dictionary_encode(), pyarrow.csv and polars' .to_arrow() "
            "make, lists values in the order they first appear in the rows, so it "
            "does not declare an order. To choose the order, mark the dictionary "
            "ordered, or pass the column as a pandas Categorical or a polars Enum "
            "with the categories in the order you want."
        )
    if column.natural_order:
        notes.append(_table_natural_order_note(column, data.library))
    vanished = column.vanished
    if vanished:
        one = len(vanished) == 1
        others = [c for c in (data.x, data.y, data.counts_var) if c not in (None, name)]
        causes = [f"a missing {_or_listed([str(c) for c in others])}"] if others else []
        if data.counts_var is not None:
            causes.append("a count of 0")
        notes.append(
            f"The {'level' if one else 'levels'} {_quoted(vanished)} of {name!r} "
            f"{'was' if one else 'were'} left out of the analysis: every row of "
            f"{'it' if one else 'them'} has {', or '.join(causes)}."
        )
    for level in column.levels:
        drawn = _control_characters(level)
        if drawn is not None:
            notes.append(
                f"The level {level!r} of {name!r} holds control characters {drawn}; "
                "the result keeps the name as given."
            )
    for same in column.lookalike_levels:
        notes.append(
            f"The levels {_quoted(same)} of {name!r} differ only in spaces at their "
            "start or end, so they are counted as separate categories whose labels "
            "look alike. If they are the same category, remove the spaces first, "
            "for example with .str.strip() in pandas or .str.strip_chars() in polars."
        )
    return notes


def _declare_order(library: Library, name: str, order: Sequence[str]) -> str:
    """A statement that declares column ``name`` (text) on the categories
    ``order``, in ``library``: a pandas Categorical, a polars Enum, an ordered
    pyarrow dictionary."""
    shown = list(order)
    if library == "polars":
        return (
            f"data = data.with_columns(pl.col({name!r}).cast(pl.String)"
            f".cast(pl.Enum({shown!r})))"
        )
    if library == "pyarrow":
        arrow = (
            "pyarrow.DictionaryArray.from_arrays(pyarrow.compute.index_in("
            f"data[{name!r}].combine_chunks(), value_set=pyarrow.array({shown!r})), "
            f"pyarrow.array({shown!r}), ordered=True)"
        )
        return _replace_column(library, name, "", "", arrow)
    return f"data[{name!r}] = pd.Categorical(data[{name!r}], categories={shown!r})"


def _table_natural_order_note(column: TableLevels, library: Library) -> str:
    """The natural-order note of a barstats column: barstats has no ``order=``,
    so another order comes from a declared categorical, written in the frame's
    own library (review B14)."""
    levels, natural = column.present, column.natural_order
    first = next(
        n for n, (a, b) in enumerate(zip(levels, natural, strict=True)) if a != b
    )
    what = {
        "pandas": "a pandas Categorical",
        "polars": "a polars Enum",
        "pyarrow": "an ordered pyarrow dictionary",
    }[library]
    if len(f"{list(natural)!r}") <= _MAX_EXAMPLE:
        how = f"declare the column as {what} in that order: " + _declare_order(
            library, column.name, natural
        )
    else:
        how = f"declare the column as {what} with the categories in that order"
    return (
        f"The levels of {column.name!r} are text, sorted as text (as R sorts "
        f"them), so {levels[first]!r} comes before {natural[first]!r}: "
        f"{_listed(levels)}. The analysis and the figure use this order. To order "
        f"them by the numbers in their names, {how}."
    )
