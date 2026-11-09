"""Frozen result types: every number on a figure, with JSON and frame export.

All classes are frozen, slotted dataclasses holding plain Python types only (str,
int, float, bool, None and tuples of these or of other result classes). NumPy
scalars and lists passed in are converted on construction, so a result always
prints and serializes cleanly.

JSON (:meth:`StatsResult.to_json`) is strict JSON: infinite values are written as
the strings "Inf" and "-Inf" and NaN as null, never as the ``Infinity`` or ``NaN``
literals Python's ``json`` module writes by default. :meth:`StatsResult.from_json`
reads it back into an equal result.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.metadata
import json
import math
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypeVar, Union, cast

if TYPE_CHECKING:
    from inferplot.check import Issue

__all__ = [
    "PARITY_TARGET",
    "BayesFactor",
    "Descriptive",
    "EffectSize",
    "PairwiseComparison",
    "Provenance",
    "Settings",
    "StatsResult",
    "TestResult",
    "Texts",
    "current_provenance",
]

PARITY_TARGET: Final = "ggstatsplot 1.1.4"


# --------------------------------------------------------------------------- #
# plain-type normalization
# --------------------------------------------------------------------------- #


def _plain_value(value: object) -> object:
    """Convert NumPy scalars to Python scalars, and lists and NumPy arrays to
    tuples, recursively."""
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, list | tuple):
        items = cast("list[object] | tuple[object, ...]", value)
        return tuple(_plain_value(v) for v in items)
    if dataclasses.is_dataclass(value):
        return value
    tolist = getattr(value, "tolist", None)
    ndim = getattr(value, "ndim", None)
    if callable(tolist) and isinstance(ndim, int) and ndim >= 1:
        return _plain_value(tolist())
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:
        return _plain_value(item())
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return str(value)
    raise TypeError(
        f"results hold plain Python values only, got {type(value).__name__}"
    )


@functools.cache
def _field_hints(cls: type) -> dict[str, object]:
    return typing.get_type_hints(cls)


def _as_float(hint: object, value: object, where: str) -> object:
    """Make the numbers of a float-typed field floats (``3`` -> ``3.0``).

    Follows ``float``, ``float | None`` and ``tuple[float, ...]`` (nested) hints; a
    bool where a number belongs raises ``TypeError``. Other values are returned as
    they are.
    """
    if _is_union(hint):
        if value is None:
            return None
        non_none = [a for a in typing.get_args(hint) if a is not type(None)]
        return _as_float(non_none[0], value, where) if len(non_none) == 1 else value
    if hint is float:
        if isinstance(value, bool):
            raise TypeError(f"{where} must be a number, got {value!r}")
        return float(value) if type(value) is int else value
    if typing.get_origin(hint) is tuple and isinstance(value, tuple):
        items = cast("tuple[object, ...]", value)
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis and _holds_float(args[0]):
            return tuple(_as_float(args[0], v, where) for v in items)
        return items
    return value


def _holds_float(hint: object) -> bool:
    if hint is float:
        return True
    if _is_union(hint):
        return any(_holds_float(a) for a in typing.get_args(hint))
    if typing.get_origin(hint) is tuple:
        return any(_holds_float(a) for a in typing.get_args(hint) if a is not Ellipsis)
    return False


class _Plain:
    """Mixin: normalize every field to plain Python types after construction.

    NumPy scalars and arrays and lists become Python scalars and tuples, and the
    numbers of float-typed fields become floats, so ``values`` built from integer
    data still holds floats and a result equals itself read back from JSON.
    """

    __slots__ = ()

    def __post_init__(self) -> None:
        hints = _field_hints(type(self))
        for field in dataclasses.fields(cast("Any", self)):
            value = getattr(self, field.name)
            plain = _plain_value(value)
            if _holds_float(hints[field.name]):
                plain = _as_float(
                    hints[field.name], plain, f"{type(self).__name__}.{field.name}"
                )
            if plain is not value:
                object.__setattr__(self, field.name, plain)


# --------------------------------------------------------------------------- #
# result parts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Descriptive(_Plain):
    """One analyzed level: its size and centrality: "mean", "median", "trimmed_mean"
    (at the settings' tr) or "map", with its value. ``value`` is None for a MAP
    that has no kernel density (R's bandwidth fails; ``warnings`` says why)."""

    level: str
    n: int
    centrality: str
    value: float | None


@dataclass(frozen=True, slots=True)
class TestResult(_Plain):
    """The main test.

    ``df_error`` is statsExpressions' ``df.error`` (Welch's t df; the denominator df
    of an F test) and ``df`` its ``df`` (numerator df), None where absent.
    ``alternative`` is "two-sided", "less" or "greater".
    """

    # Not a pytest test class, although its name starts with "Test".
    __test__: ClassVar[bool] = False

    name: str
    label: str
    method: str
    statistic_name: str
    statistic: float
    df: float | None
    df_error: float | None
    p_value: float
    alternative: str


@dataclass(frozen=True, slots=True)
class EffectSize(_Plain):
    """The effect size and its CI; ``sidedness`` "greater" means [low, max].

    A bootstrap CI (``ci_method`` "percentile bootstrap") records its number of
    resamples and its seed in ``bootstrap_resamples`` and ``bootstrap_seed``, and
    in ``infinite_resamples`` how many resamples gave an infinite effect size (the
    AKP and explanatory effect sizes when a resample's winsorized variance is 0;
    kept, so a bound can be infinite); all three are None for a CI
    without resampling. With unequal group sizes the explanatory effect size's
    estimate is itself the median of the bootstrap values, so it depends on the
    seed too.
    """

    name: str
    label: str
    estimate: float
    conf_level: float
    conf_low: float
    conf_high: float
    ci_method: str
    sidedness: str
    bootstrap_resamples: int | None = None
    bootstrap_seed: int | None = None
    infinite_resamples: int | None = None


@dataclass(frozen=True, slots=True)
class BayesFactor(_Plain):
    """A Bayes factor in favour of the alternative and log(BF01) = -log(bf10)."""

    bf10: float
    log_bf01: float
    prior: str
    prior_scale: float
    test: str


@dataclass(frozen=True, slots=True)
class PairwiseComparison(_Plain):
    """One pair: ``significant`` is ``p_adjusted < pairwise_alpha``.

    ``estimate`` with ``conf_level``, ``conf_low`` and ``conf_high`` is the pair's
    own estimate and interval where the pairwise method reports one (lincon: the
    trimmed-mean difference group1 - group2), else None. ``bayes`` is the pair's
    Bayes factor where the comparison is Bayesian (type "bayes"); such a pair has
    no p-value, so ``p_value``, ``p_adjusted`` and ``significant`` are None and
    ``adjust_method`` is "none" (Bayes factors are not adjusted).
    """

    group1: str
    group2: str
    statistic_name: str
    statistic: float
    df: float | None
    p_value: float | None
    p_adjusted: float | None
    adjust_method: str
    test: str
    test_label: str
    significant: bool | None
    displayed: bool
    estimate: float | None = None
    conf_level: float | None = None
    conf_low: float | None = None
    conf_high: float | None = None
    bayes: BayesFactor | None = None


@dataclass(frozen=True, slots=True)
class Texts(_Plain):
    """Every text of the result as plain Unicode (mathtext is built by the renderer)."""

    subtitle: str | None
    caption: str | None
    apa: str
    alt_text: str
    centrality_labels: tuple[str, ...]
    tick_labels: tuple[str, ...]
    pairwise_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Provenance(_Plain):
    """Versions, seed, parity target and method citations behind a result."""

    inferplot: str
    numpy: str
    scipy: str
    matplotlib: str
    narwhals: str
    seed: int
    parity_target: str
    citations: tuple[str, ...]


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def current_provenance(*, seed: int, citations: tuple[str, ...]) -> Provenance:
    """Provenance for a result computed now (versions from installed metadata).

    Reads versions with ``importlib.metadata``, so it never imports matplotlib.
    """
    return Provenance(
        inferplot=_version("inferplot"),
        numpy=_version("numpy"),
        scipy=_version("scipy"),
        matplotlib=_version("matplotlib"),
        narwhals=_version("narwhals"),
        seed=seed,
        parity_target=PARITY_TARGET,
        citations=citations,
    )


@dataclass(frozen=True, slots=True)
class Settings(_Plain):
    """The analysis settings that shaped the numbers and texts."""

    conf_level: float
    digits: int
    alternative: str
    p_adjust_method: str
    pairwise_display: str
    pairwise_alpha: float
    bf_prior: float
    bf_message: bool | None
    results_subtitle: bool
    seed: int
    tr: float


# --------------------------------------------------------------------------- #
# JSON encoding and decoding
# --------------------------------------------------------------------------- #


def _encode(value: object) -> object:
    """Plain value -> JSON-ready value (dicts, lists; inf as "Inf", nan as None)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encode(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return [_encode(v) for v in cast("tuple[object, ...]", value)]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
    return value


def _to_dict(value: object) -> object:
    """Plain value -> dicts and lists, floats kept as floats (inf and nan kept)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        hints = typing.get_type_hints(type(value))
        out: dict[str, object] = {}
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            if _accepts_float(hints[field.name]) and type(item) is int:
                item = float(item)
            out[field.name] = _to_dict(item)
        return out
    if isinstance(value, tuple):
        return [_to_dict(v) for v in cast("tuple[object, ...]", value)]
    return value


def _accepts_float(hint: object) -> bool:
    if hint is float:
        return True
    return float in typing.get_args(hint) and _is_union(hint)


def _is_union(hint: object) -> bool:
    origin = typing.get_origin(hint)
    return origin is Union or origin is types.UnionType


_T = TypeVar("_T")


def _decode(hint: object, value: object, where: str) -> object:
    """JSON value -> plain value of type ``hint`` (the inverse of :func:`_encode`)."""
    if _is_union(hint):
        args = typing.get_args(hint)
        if value is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _decode(non_none[0], value, where)
        raise TypeError(f"{where}: unsupported union {hint!r}")
    if hint is float:
        if value is None:
            return math.nan
        if value in ("Inf", "-Inf"):
            return math.inf if value == "Inf" else -math.inf
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        raise TypeError(f"{where}: expected a number, got {value!r}")
    if hint is int or hint is str or hint is bool:
        if type(value) is not hint:
            raise TypeError(f"{where}: expected {hint}, got {value!r}")
        return value
    if typing.get_origin(hint) is tuple:
        (item_hint, _ellipsis) = typing.get_args(hint)
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        items = cast("list[object]", value)
        return tuple(
            _decode(item_hint, v, f"{where}[{i}]") for i, v in enumerate(items)
        )
    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return _from_mapping(hint, value, where)
    raise TypeError(f"{where}: unsupported type {hint!r}")


def _from_mapping(cls: type[_T], value: object, where: str) -> _T:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where}: expected an object, got {value!r}")
    mapping = cast("Mapping[str, object]", value)
    hints = typing.get_type_hints(cls)
    names = [field.name for field in dataclasses.fields(cast("Any", cls))]
    if set(mapping) != set(names):
        missing = sorted(set(names) - set(mapping))
        extra = sorted(set(mapping) - set(names))
        raise ValueError(f"{where}: missing fields {missing}, unknown fields {extra}")
    kwargs = {
        name: _decode(hints[name], mapping[name], f"{where}.{name}") for name in names
    }
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #

_FRAME_COLUMNS: Final = (
    ("test", "str"),
    ("label", "str"),
    ("group1", "str"),
    ("group2", "str"),
    ("statistic_name", "str"),
    ("statistic", "float"),
    ("df", "float"),
    ("df_error", "float"),
    ("p_value", "float"),
    ("p_adjusted", "float"),
    ("adjust_method", "str"),
    ("alternative", "str"),
    ("effect_size", "str"),
    ("estimate", "float"),
    ("conf_level", "float"),
    ("conf_low", "float"),
    ("conf_high", "float"),
    ("bf10", "float"),
    ("log_bf01", "float"),
    ("significant", "bool"),
    ("displayed", "bool"),
)


@dataclass(frozen=True, slots=True)
class StatsResult(_Plain):
    """Everything an analysis computed, as plain Python values (SPEC section 6).

    ``values`` holds the analyzed observations: for each entry of ``levels``, the
    ``y`` values of the complete rows in that level, in data order, as floats. The
    renderer draws the points, boxes and violins from them, so a result read back
    with :meth:`from_json` draws the same figure (DECISIONS.md D36).

    ``test`` and ``effect_size`` are None for type "bayes", which has no
    frequentist test: its two-group Bayes factor is in ``bayes`` and, with three or
    more groups, each pair's in ``pairwise`` (the Bayesian ANOVA and posterior
    estimates arrive in inferplot 0.2, DECISIONS.md D41).
    """

    analysis: str
    design: str
    type: str
    x: str
    y: str
    subject_id: str | None
    levels: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    descriptives: tuple[Descriptive, ...]
    test: TestResult | None
    effect_size: EffectSize | None
    bayes: BayesFactor | None
    pairwise: tuple[PairwiseComparison, ...]
    n_obs: int
    n_dropped: int
    dropped_subjects: int
    warnings: tuple[str, ...]
    provenance: Provenance
    settings: Settings
    text: Texts

    # -- export -------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Nested dicts and lists; floats stay floats (``inf`` and ``nan`` kept)."""
        return cast("dict[str, Any]", _to_dict(self))

    def to_json(self, indent: int | None = 2, *, include_values: bool = True) -> str:
        """Strict, ASCII-only JSON: inf as "Inf", -inf as "-Inf", nan as null.

        ``include_values=False`` writes ``values`` as an empty list: every number
        and text except the analyzed observations, which can be large and are the
        user's data. ``save()`` embeds this form in figure files; a result read
        back from it has ``values == ()`` and cannot be drawn again.
        """
        encoded = _encode(self)
        if not include_values:
            cast("dict[str, object]", encoded)["values"] = []
        return json.dumps(encoded, indent=indent, allow_nan=False, ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StatsResult:
        """Rebuild a result from :meth:`to_dict` or parsed :meth:`to_json` output."""
        return _from_mapping(cls, data, "result")

    @classmethod
    def from_json(cls, text: str) -> StatsResult:
        """Rebuild a result from :meth:`to_json` output ("Inf" becomes inf again)."""
        return cls.from_dict(json.loads(text))

    def frame_rows(self) -> list[dict[str, Any]]:
        """The rows of :meth:`to_frame`: the main test, then each pair.

        Type "bayes" has no frequentist test: its main row is the two-group Bayes
        factor, and with three or more groups there is no main row (the Bayesian
        ANOVA arrives in inferplot 0.2).
        """
        test, es, bayes = self.test, self.effect_size, self.bayes
        rows: list[dict[str, Any]] = []
        if test is not None:
            rows.append(
                {
                    "test": test.name,
                    "label": test.label,
                    "group1": None,
                    "group2": None,
                    "statistic_name": test.statistic_name,
                    "statistic": test.statistic,
                    "df": test.df,
                    "df_error": test.df_error,
                    "p_value": test.p_value,
                    "p_adjusted": None,
                    "adjust_method": None,
                    "alternative": test.alternative,
                    "effect_size": None if es is None else es.name,
                    "estimate": None if es is None else es.estimate,
                    "conf_level": None if es is None else es.conf_level,
                    "conf_low": None if es is None else es.conf_low,
                    "conf_high": None if es is None else es.conf_high,
                    "bf10": None if bayes is None else bayes.bf10,
                    "log_bf01": None if bayes is None else bayes.log_bf01,
                    "significant": None,
                    "displayed": None,
                }
            )
        elif bayes is not None:
            rows.append(
                {
                    **dict.fromkeys(name for name, _ in _FRAME_COLUMNS),
                    "test": bayes.test,
                    "label": "Bayesian t-test",
                    "alternative": "two-sided",
                    "bf10": bayes.bf10,
                    "log_bf01": bayes.log_bf01,
                }
            )
        for pair in self.pairwise:
            rows.append(
                {
                    "test": pair.test,
                    "label": pair.test_label,
                    "group1": pair.group1,
                    "group2": pair.group2,
                    "statistic_name": pair.statistic_name,
                    "statistic": pair.statistic,
                    "df": pair.df,
                    "df_error": None,
                    "p_value": pair.p_value,
                    "p_adjusted": pair.p_adjusted,
                    "adjust_method": pair.adjust_method,
                    "alternative": "two-sided",
                    "effect_size": None,
                    "estimate": pair.estimate,
                    "conf_level": pair.conf_level,
                    "conf_low": pair.conf_low,
                    "conf_high": pair.conf_high,
                    "bf10": None if pair.bayes is None else pair.bayes.bf10,
                    "log_bf01": None if pair.bayes is None else pair.bayes.log_bf01,
                    "significant": pair.significant,
                    "displayed": pair.displayed,
                }
            )
        return rows

    def to_frame(self, backend: Literal["pandas", "polars"] = "pandas") -> Any:
        """One row per test: the main test first, then each pair.

        Missing values are nulls (pandas: NaN in float columns, None in text
        columns, ``<NA>`` in the nullable boolean columns).
        """
        rows = self.frame_rows()
        columns = {name: [row[name] for row in rows] for name, _ in _FRAME_COLUMNS}
        if backend == "pandas":
            try:
                import pandas as pd  # pyright: ignore[reportMissingTypeStubs]
            except ImportError as error:
                raise ImportError(
                    "to_frame(backend='pandas') needs pandas: "
                    "pip install 'inferplot[pandas]'"
                ) from error
            dtypes = {"str": "object", "float": "float64", "bool": "boolean"}
            return pd.DataFrame(
                {
                    name: pd.Series(
                        [
                            math.nan if v is None and kind == "float" else v
                            for v in columns[name]
                        ],
                        dtype=dtypes[kind],
                    )
                    for name, kind in _FRAME_COLUMNS
                }
            )
        if backend == "polars":
            try:
                import polars as pl
            except ImportError as error:
                raise ImportError(
                    "to_frame(backend='polars') needs polars: "
                    "pip install 'inferplot[polars]'"
                ) from error
            dtypes = {"str": pl.String, "float": pl.Float64, "bool": pl.Boolean}
            return pl.DataFrame(
                columns, schema={name: dtypes[kind] for name, kind in _FRAME_COLUMNS}
            )
        raise ValueError(f"backend must be 'pandas' or 'polars', got {backend!r}")

    # -- checks and printing ------------------------------------------------- #

    def check(self) -> tuple[Issue, ...]:
        """Statistical problems with this result; empty when there are none."""
        from inferplot.check import result_issues

        return result_issues(self)

    def __str__(self) -> str:
        """The APA sentence, then a table of the pairwise comparisons if any."""
        if not self.pairwise:
            return self.text.apa
        return self.text.apa + "\n\n" + _pairwise_table(self)


def _p_cell(p: float | None) -> str:
    if p is None or math.isnan(p):
        return "NA"
    if p < 0.001:
        return "<.001"
    text = f"{p:.3f}"
    return text[1:] if text.startswith("0.") else text


def _num_cell(x: float | None) -> str:
    if x is None or math.isnan(x):
        return "NA"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return f"{x:.2f}"


# Column heading of the adjusted p by p_adjust_method (the caption's names).
_ADJUSTED_HEADINGS: Final = {
    "holm": "p (Holm)",
    "hochberg": "p (Hochberg)",
    "hommel": "p (Hommel)",
    "bonferroni": "p (Bonferroni)",
    "BH": "p (Benjamini-Hochberg)",
    "fdr": "p (Benjamini-Hochberg)",
    "BY": "p (Benjamini-Yekutieli)",
    "none": "p (unadj.)",
}


def _bf_cell(bayes: BayesFactor) -> str:
    """BF10 to two decimals, scientific from 1e5 and below 0.01, and printed from
    log(BF01) where it is beyond a double (``text.format_bf10``)."""
    from inferplot.text import format_bf10

    return format_bf10(bayes.bf10, bayes.log_bf01, 2)


def _table(title: str, rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [
        "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    ]
    return "\n".join([title, *lines])


def _bayes_pairwise_table(result: StatsResult) -> str:
    first = result.pairwise[0]
    rows: list[tuple[str, ...]] = [
        ("group1", "group2", first.statistic_name, "df", "BF10", "log BF01", "shown")
    ]
    scale = ""
    for pair in result.pairwise:
        bayes = pair.bayes
        assert bayes is not None
        scale = f"{bayes.prior_scale:g}"
        rows.append(
            (
                pair.group1,
                pair.group2,
                _num_cell(pair.statistic),
                _num_cell(pair.df),
                _bf_cell(bayes),
                _num_cell(bayes.log_bf01),
                "yes" if pair.displayed else "no",
            )
        )
    title = f"Pairwise Bayes factors ({first.test_label}, JZS prior {scale}):"
    return _table(title, rows)


def _yes_no(flag: bool | None) -> str:
    return "NA" if flag is None else ("yes" if flag else "no")


def _pairwise_table(result: StatsResult) -> str:
    first = result.pairwise[0]
    if first.bayes is not None:
        return _bayes_pairwise_table(result)
    method = result.settings.p_adjust_method
    adjusted = _ADJUSTED_HEADINGS.get(method, f"p ({method})")
    header = (
        "group1",
        "group2",
        first.statistic_name,
        "df",
        "p",
        adjusted,
        "significant",
        "shown",
    )
    rows = [header] + [
        (
            pair.group1,
            pair.group2,
            _num_cell(pair.statistic),
            _num_cell(pair.df),
            _p_cell(pair.p_value),
            _p_cell(pair.p_adjusted),
            _yes_no(pair.significant),
            _yes_no(pair.displayed),
        )
        for pair in result.pairwise
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    ]
    title = f"Pairwise comparisons ({first.test_label}):"
    return "\n".join([title, *lines])
