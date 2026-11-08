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
  dictionary), otherwise the sorted unique values (numbers numerically, text as
  Python's ``sorted()``, which is R's C-locale order). A polars ``Categorical`` has no
  stable declared order across polars versions, so it is treated as text.
* Declared levels left without rows are dropped and reported (``empty_levels``);
  ggstatsplot does the same with ``droplevels()``.
* Labels are the names R gives factor levels: whole numbers without ``.0``.
* ``order=`` must be a permutation of the levels present.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import narwhals.stable.v2 as nw
import numpy as np
import numpy.typing as npt

__all__ = [
    "BetweenData",
    "ColumnNotFoundError",
    "level_label",
    "read_between",
    "to_frame",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
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

    def counts(self) -> dict[str, int]:
        """``{level: n}`` in level order."""
        return dict(zip(self.levels, self.n_per_level, strict=True))


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
    if "Lazy" in name:
        hint = "collect it into an eager DataFrame first, for example with .collect()"
    elif name.endswith(("Series", "ChunkedArray", "Array")):
        hint = "pass the whole data frame and name the columns as x and y"
    else:
        hint = _LONG_FORMAT_HINT
    raise TypeError(
        f"data must be a pandas, polars or pyarrow DataFrame, not a {name}; {hint}."
    ) from None


def _type_name(obj: object) -> str:
    cls = type(obj)
    module = cls.__module__.split(".")[0]
    return cls.__name__ if module == "builtins" else f"{module}.{cls.__name__}"


def _column(frame: nw.DataFrame[Any], name: object, role: str) -> nw.Series[Any]:
    """The column called ``name``, or ``ColumnNotFoundError`` with the closest match."""
    if not isinstance(name, str):
        raise TypeError(f"{role} must be a column name (a string), not {name!r}.")
    columns = [str(c) for c in frame.columns]
    if name in frame.columns:
        return frame.get_column(name)
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
    """
    as_float = series.cast(nw.Float64()).fill_null(value=0.0)
    return np.asarray(as_float.to_numpy(), dtype=np.float64)


# --- x: levels --------------------------------------------------------------------

XKind = Literal["declared", "text", "integer", "float"]


def _x_kind(series: nw.Series[Any], name: str) -> XKind:
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
    if isinstance(dtype, nw.Boolean):
        hint = (
            "convert it to text labels first, for example "
            "df[x].map({True: 'yes', False: 'no'})"
        )
    else:
        hint = "convert it to text or to a categorical column first"
    raise TypeError(
        f"x={name!r} must hold text, categories or whole numbers to define groups; "
        f"its type is {dtype}. {hint}."
    )


def _declared_categories(series: nw.Series[Any], name: str) -> list[object]:
    try:
        categories = series.cat.get_categories().to_list()
    except AttributeError:
        # pandas ArrowDtype(dictionary): narwhals reports Categorical but has no
        # .cat accessor for it (narwhals 2.0 to 2.28).
        raise TypeError(
            f"x={name!r} is a pandas Arrow dictionary column, which is not supported; "
            "convert it with .astype('category')."
        ) from None
    return [c for c in categories if c is not None]


def _labels_in_order(values: Sequence[object], name: str) -> list[str]:
    """Distinct labels of ``values`` with the label of each first occurrence."""
    labels: list[str] = []
    seen: set[str] = set()
    origin: dict[str, object] = {}
    for value in values:
        try:
            label = level_label(value)
        except TypeError as error:
            raise TypeError(f"x={name!r}: {error}") from None
        if label in seen:
            if origin[label] != value:
                raise ValueError(
                    f"x={name!r} has two different values that both print as "
                    f"{label!r} ({origin[label]!r} and {value!r})."
                )
            continue
        seen.add(label)
        origin[label] = value
        labels.append(label)
    return labels


def _row_labels(raw: Sequence[object], name: str) -> list[str]:
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
                raise TypeError(f"x={name!r}: {error}") from None
            cache[key] = label
        out.append(label)
    return out


def _check_order(
    order: object, present: Sequence[str], empty: Sequence[str]
) -> list[str]:
    """``order`` as labels, if it is a permutation of ``present``.

    Typed ``object``: callers that skip type checking can pass anything.
    """
    if isinstance(order, str) or not isinstance(order, Sequence):
        raise TypeError(
            "order must be a list of level names, for example "
            f"order={list(present)!r}; got {order!r}."
        )
    labels: list[str] = []
    for item in cast("Sequence[object]", order):
        if isinstance(item, bool) or not isinstance(item, str | int):
            raise TypeError(f"order entries must be level names, not {item!r}.")
        labels.append(level_label(item))
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
                parts.append(f"{no_rows!r} have no rows with both x and y present")
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


# --- betweenstats -----------------------------------------------------------------


def read_between(
    data: object,
    x: str,
    y: str,
    *,
    order: Sequence[str | int] | None = None,
    min_levels: int = 2,
) -> BetweenData:
    """Read a between-groups design: ``y`` split by the levels of ``x``.

    Raises ``TypeError`` for unsupported data or column types, ``ColumnNotFoundError``
    (a ``KeyError``) for a missing column, and ``ValueError`` for infinite ``y``
    values, fewer than ``min_levels`` levels with data, or a bad ``order``.
    """
    frame = to_frame(data)
    x_col = _column(frame, x, "x")
    y_col = _column(frame, y, "y")
    if x == y:
        raise ValueError(f"x and y must be different columns; both are {x!r}.")

    y_dtype = y_col.dtype
    if isinstance(y_dtype, nw.Boolean) or not y_dtype.is_numeric():
        raise TypeError(
            f"y={y!r} must be numeric; its type is {y_dtype}. "
            + (
                "Booleans are not measurements; convert to 0/1 deliberately if "
                "that is what you mean."
                if isinstance(y_dtype, nw.Boolean)
                else "Convert it with pandas.to_numeric() or .cast(float) first."
            )
        )
    kind = _x_kind(x_col, x)

    missing_x = _missing(x_col)
    categories = _declared_categories(x_col, x) if kind == "declared" else []
    # Raw x values. Never to_numpy() on a categorical: pyarrow puts a dictionary
    # value where a null was. to_list() is right everywhere but slow on pyarrow
    # dictionaries (1.2 s per million rows); casting text categories to String first
    # takes 0.04 s. Numeric categories must not be cast (4.0 would become "4.0").
    raw_x: list[object] = []
    if kind == "text" or (
        kind == "declared" and all(isinstance(c, str) for c in categories)
    ):
        raw_x = x_col.cast(nw.String()).to_list()
    elif kind == "declared":
        raw_x = x_col.to_list()
    if kind == "declared":
        # A row pointing at a null dictionary entry is null in to_list() but not in
        # is_null() at pyarrow 14 (it is at pyarrow 26).
        missing_x |= np.fromiter((v is None for v in raw_x), np.bool_, len(raw_x))
    missing_y = _missing(y_col)
    keep = ~(missing_x | missing_y)
    rows = np.nonzero(keep)[0]

    y_all = _float_values(y_col)
    y_kept = y_all[keep]
    infinite = ~np.isfinite(y_kept)
    if infinite.any():
        shown = ", ".join(str(r) for r in rows[infinite][:_MAX_ROWS_SHOWN])
        raise ValueError(
            f"y={y!r} has {int(infinite.sum())} infinite value(s) (row positions "
            f"{shown}{', ...' if infinite.sum() > _MAX_ROWS_SHOWN else ''}); "
            "remove them or replace them with missing values."
        )

    empty: list[str] = []
    if kind in ("declared", "text"):
        kept_raw = [raw_x[i] for i in rows]
        row_labels = _row_labels(kept_raw, x)
        present = set(row_labels)
        if kind == "declared":
            declared = _labels_in_order(categories, x)
            undeclared = sorted(present.difference(declared))
            if undeclared:  # cannot happen for a valid categorical; never guess
                raise ValueError(
                    f"x={x!r} has values outside its categories: {undeclared!r}."
                )
            levels = [label for label in declared if label in present]
            empty = [label for label in declared if label not in present]
            source: LevelSource = "declared"
        else:
            levels = sorted(present)  # text: Python's sorted() is R's C-locale order
            source = "sorted"
    else:
        if kind == "integer":
            # Python ints from to_list() are exact for any 64-bit integer; float64
            # is not above 2**53.
            full: list[Any] = x_col.to_list()
            ints = [int(full[i]) for i in rows]
        else:
            x_values = _float_values(x_col)[keep]
            bad = ~np.isfinite(x_values) | (x_values != np.round(x_values))
            if bad.any():
                example = float(x_values[bad][0])
                raise TypeError(
                    f"x={x!r} holds numbers that are not whole ({example!r}); a "
                    "grouping column must hold text, categories or whole numbers. "
                    "For a numeric x use scatterstats; to group by these values, "
                    "convert them to text first."
                )
            ints = [int(v) for v in x_values]
        row_labels = [str(v) for v in ints]
        levels = [str(v) for v in sorted(set(ints))]
        source = "sorted"

    if order is not None:
        levels = _check_order(order, levels, empty)
        source = "order"
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
    )
