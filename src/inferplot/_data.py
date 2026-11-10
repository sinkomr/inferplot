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
from collections.abc import Hashable, Iterable, Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import narwhals.stable.v2 as nw
import numpy as np
import numpy.typing as npt
from narwhals.exceptions import DuplicateError

__all__ = [
    "BetweenData",
    "ColumnNotFoundError",
    "WithinData",
    "data_notes",
    "level_label",
    "read_between",
    "read_within",
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


_EXAMPLE_COLUMN: Final = {"x": "group", "y": "score", "subject_id": "subject"}


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
        raise TypeError(
            f"{group.role.arg}={group.name!r} holds numbers that are not whole "
            f"({example!r}); {group.role.not_whole_hint}"
        )
    return [str(int(v)) for v in values]


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


_DIGITS: Final = re.compile(r"([0-9]+)")


def _natural_key(label: str) -> tuple[str | int, ...]:
    """``label`` split into text and runs of ASCII digits, the digits as numbers:
    text at even places, numbers at odd ones, so two keys compare place by place."""
    parts = _DIGITS.split(label)
    return tuple(int(part) if n % 2 else part for n, part in enumerate(parts))


def _natural_order(group: _Grouping, levels: Sequence[str]) -> tuple[str, ...]:
    """Text ``levels`` in sorted order, ordered by the numbers in their names when
    that differs ("week 2" before "week 10"); empty otherwise. Numbers sort
    numerically and declared categories keep their order, so neither has one."""
    if group.kind != "text":
        return ()
    natural = sorted(levels, key=lambda label: (_natural_key(label), label))
    return tuple(natural) if natural != list(levels) else ()


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
            hint = "Convert it with pandas.to_numeric() or .cast(float) first."
    else:
        return
    raise TypeError(f"y={name!r} must be numeric; its type is {dtype}. {hint}")


def _check_finite(name: str, values: FloatArray, rows: IntArray) -> None:
    """An infinite ``y`` in a row that is otherwise complete raises."""
    infinite = ~np.isfinite(values)
    if infinite.any():
        shown = ", ".join(str(r) for r in rows[infinite][:_MAX_ROWS_SHOWN])
        raise ValueError(
            f"y={name!r} has {int(infinite.sum())} infinite value(s) (row positions "
            f"{shown}{', ...' if infinite.sum() > _MAX_ROWS_SHOWN else ''}); "
            "remove them or replace them with missing values."
        )


def _check_exact_integers(series: nw.Series[Any], name: str, rows: IntArray) -> None:
    """An integer ``y`` that float64 cannot hold exactly raises.

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
        f"y={name!r} holds integers too large to analyze exactly: {len(rounded)} of "
        f"its {len(kept)} values change when stored as double-precision numbers "
        f"(for example {example} becomes {int(float(example))}), so different values "
        "would become ties. Subtract a common offset or change the unit first, for "
        f"example y - y.min() (smallest value: {min(kept)}); that leaves every test "
        "and standardized effect size as it is."
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
        "column."
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
        raise _too_few_subjects(
            subject_id=subject_id,
            x=x,
            levels=levels,
            complete=subjects,
            seen=len(seen_set),
            first_lacking=first_lacking,
            min_subjects=min_subjects,
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


def data_notes(data: BetweenData | WithinData) -> list[str]:
    """Plain-English notes on how the levels of ``x`` were read, for the result's
    ``warnings``: an unordered dictionary whose order was not used, levels left out
    because every row of them misses a value, and levels that differ only in
    surrounding whitespace. They name levels of ``x``, never subjects (D36)."""
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
    for same in data.lookalike_levels:
        notes.append(
            f"The levels {_quoted(same)} of {x!r} differ only in spaces at their "
            "start or end, so they are analyzed as separate groups whose labels "
            "look alike. If they are the same group, remove the spaces first, for "
            "example with .str.strip() in pandas or .str.strip_chars() in polars."
        )
    return notes


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
