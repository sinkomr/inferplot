"""Frozen result types: every number on a figure, with JSON and frame export.

All classes are frozen, slotted dataclasses holding plain Python types only (str,
int, float, bool, None and tuples of these or of other result classes). NumPy
scalars and lists passed in are converted on construction, so a result always
prints and serializes cleanly.

JSON (:meth:`StatsResult.to_json`) is strict JSON: infinite values are written as
the strings "Inf" and "-Inf" and NaN as null, never as the ``Infinity`` or ``NaN``
literals Python's ``json`` module writes by default. Its first key is
``schema_version`` (:data:`SCHEMA_VERSION`), the version of the field set;
:meth:`StatsResult.from_json` reads JSON of this version back into an equal result
and refuses any other version with a ``ValueError``.

Schema versions: 1 is the field set before the version was written (JSON without
``schema_version``); 2 added ``schema_version`` itself,
``effect_size.undefined_resamples``, ``dropped_subject_rows`` and ``n_tied``; 3
added ``point_labels``, ``label_var`` and ``settings.test_value``, made ``y``
optional (None for histostats) and added the designs "one-sample" and
"correlation"; 4 added the design "contingency" (barstats) with ``y_levels``,
``counts``, ``cells``, ``bars``, ``counts_var`` and the settings ``paired``,
``ratio``, ``label``, ``digits_perc`` and ``proportion_test``, and made a
pair's ``statistic`` optional (Fisher's exact test has none).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.metadata
import json
import math
import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, TypeVar, Union, cast

if TYPE_CHECKING:
    from inferplot.check import Issue

__all__ = [
    "CELL_LABELS",
    "DESIGNS",
    "PARITY_TARGET",
    "SCHEMA_VERSION",
    "Bar",
    "BayesFactor",
    "Cell",
    "Descriptive",
    "EffectSize",
    "PairwiseComparison",
    "Provenance",
    "Settings",
    "StatsResult",
    "TestResult",
    "Texts",
    "correlation_tied_count",
    "current_provenance",
    "one_sample_tied_count",
    "paired_tied_count",
    "tied_count",
]

PARITY_TARGET: Final = "ggstatsplot 1.1.4"

# The version of the JSON field set, written as the first key of to_json() and
# to_dict(). Raise it whenever a field is added, removed, renamed or changes meaning.
SCHEMA_VERSION: Final = 4

# Each design and the analysis that makes it (SPEC section 6).
DESIGNS: Final[dict[str, str]] = {
    "between": "betweenstats",
    "within": "withinstats",
    "one-sample": "histostats",
    "correlation": "scatterstats",
    "contingency": "barstats",
}


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

    Sign of ``statistic`` and place of the degrees of freedom, by ``name`` (first
    and second are ``StatsResult.levels[0]`` and ``levels[1]``), as ggstatsplot
    reports them:

    * "welch_t": signed, mean(first) - mean(second); Welch's df in ``df_error``.
    * "mann_whitney": W, the Mann-Whitney U of the first level (pairs in which its
      value is the larger, ties counted 1/2), never negative; no df. The
      rank-biserial r is positive when the first level tends to be larger.
    * "yuen_t": the absolute value \\|t\\| (WRS2's yuen as ggstatsplot prints it),
      whichever trimmed mean is larger; the AKP effect size keeps the sign of
      trimmed mean(first) - trimmed mean(second). Yuen's df in ``df_error``.
    * "welch_anova", "trimmed_means_anova": F, numerator df in ``df`` and
      denominator df in ``df_error``.
    * "kruskal_wallis": chi2 (H); its df, k - 1, in ``df_error`` (``df`` None).
    * "paired_t": signed, mean of the differences first - second; n - 1 in
      ``df_error``.
    * "rm_anova": F with the Greenhouse-Geisser corrected df in ``df`` and
      ``df_error``.
    * Within-subjects tests merged later follow SPEC section 4: the signed-rank V
      sums the ranks of the positive differences first - second, Friedman's chi2
      has its df in ``df_error``, and dependent Yuen's t is signed (unlike
      "yuen_t").

    ``alternative`` is "two-sided", "less" or "greater". A one-sided test compares
    the two entries of ``StatsResult.levels`` in that order: "greater" tests
    whether ``levels[0]`` is greater than ``levels[1]`` (for paired tests: whether
    the differences ``levels[0]`` minus ``levels[1]`` are positive), "less"
    whether it is less. ``alternative`` and ``levels`` together say the direction;
    the APA sentence names it in words.
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
    """The effect size and its CI.

    ``sidedness`` says which interval ``conf_low`` and ``conf_high`` are:
    "two-sided"; "greater", a one-sided [low, max] interval (max is 1 for bounded
    effect sizes, Inf otherwise); "less", a one-sided [min, high] interval.

    A bootstrap CI (``ci_method`` "percentile bootstrap") records its number of
    resamples and its seed in ``bootstrap_resamples`` and ``bootstrap_seed``; in
    ``undefined_resamples`` how many resamples had no effect size and were left
    out (rank epsilon squared when every value of a resample is tied, D54), so the
    interval comes from ``bootstrap_resamples - undefined_resamples`` values; and
    in ``infinite_resamples`` how many resamples gave an infinite effect size (the
    AKP and explanatory effect sizes when a resample's winsorized variance is 0;
    kept, so a bound can be infinite). The average AKP over pairs of conditions
    (``akp_delta_avg``) computes every pair's AKP on the same resamples and
    averages the pairs' bounds: its two counts are the largest of any one pair,
    in resamples, so every pair's bounds come from at least
    ``bootstrap_resamples - undefined_resamples`` values (the ``warnings`` entry
    gives the totals over all pairs, in values). All four are None for a CI
    without resampling, and a bootstrap CI must give the two counts. With unequal
    group
    sizes the explanatory effect size's estimate is itself the median of the
    bootstrap values, so it depends on the seed too.
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
    undefined_resamples: int | None = None
    infinite_resamples: int | None = None

    def __post_init__(self) -> None:
        _Plain.__post_init__(self)
        counts = (self.undefined_resamples, self.infinite_resamples)
        if self.bootstrap_resamples is None:
            if any(count is not None for count in counts):
                raise ValueError(
                    "EffectSize: undefined_resamples and infinite_resamples are "
                    "counts of bootstrap resamples; give bootstrap_resamples too, or "
                    "leave all three None for an interval without resampling"
                )
        elif any(count is None for count in counts):
            raise ValueError(
                "EffectSize: a bootstrap interval must record undefined_resamples "
                "and infinite_resamples (0 when there are none), so that "
                "bootstrap_resamples - undefined_resamples is the number of values "
                "the interval comes from"
            )


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

    ``group1`` is the earlier of the two in ``StatsResult.levels``. Sign of
    ``statistic`` and its df (``df``; pairs have no ``df_error``), by ``test``, as
    ggstatsplot reports them:

    * "games_howell": q signed as mean(group2) - mean(group1) (the opposite of
      the other pairwise tests); the pair's Welch df.
    * "dunn": z = \\|mean-rank difference\\| / SE, never negative; no df.
    * "yuen_trimmed_means" (lincon): t signed as trimmed mean(group1) - trimmed
      mean(group2), like ``estimate``; the pair's Yuen df.
    * "paired_t" (within subjects): t signed as the mean of the differences
      group1 - group2; df n - 1.
    * Bayes pairs: Student's t (pooled; between subjects) or the paired t
      (within), signed group1 - group2; df n1 + n2 - 2 or n - 1.
    * "fisher_exact" (barstats, design "contingency"): no test statistic
      (``statistic`` None, ``statistic_name`` ""), no df; the pair's table is
      group1 and group2 by the levels of y that either holds. ``estimate`` is
      the pair's Cramér's V with its interval (None where V is not defined);
      ``displayed`` is always False (ggstatsplot computes the pairs but never
      draws them). A pair without a test (its table has a single column, or
      its exact test would take too long) keeps its place with ``p_value``,
      ``p_adjusted`` and ``significant`` None and ``no_test_reason`` saying
      why; the other pairs are adjusted among themselves, a pair too large to
      compute counting in that adjustment as a test with p = 1.

    ``no_test_reason`` is None for every pair that has a p-value.
    """

    group1: str
    group2: str
    statistic_name: str
    statistic: float | None
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
    no_test_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Cell(_Plain):
    """One cell of a barstats figure (design "contingency"): level ``level`` of x
    within the bar of level ``bar`` of y (None for the single bar without y).

    ``percent`` is the cell's share of its bar as ggstatsplot computes it, (count
    / bar total) * 100 in double precision, before the label rounds it to
    ``digits_perc``; it need not sum to 100 over a bar once rounded.
    """

    bar: str | None
    level: str
    count: int
    percent: float


@dataclass(frozen=True, slots=True)
class Bar(_Plain):
    """One bar of a barstats figure (design "contingency"): a level of y that
    holds observations (None for the single bar without y), its ``n``, and its
    proportion test: the goodness-of-fit test of x within the bar against
    ``settings.ratio`` (equal shares by default), always two-sided and at 95%,
    never adjusted (ggstatsplot's ``onesample_data()``), with Pearson's C and
    its interval in ``effect_size``. ``test`` and ``effect_size`` are None where
    no test exists, ``no_test_reason`` says why (the single bar without y: its
    test is the result's main test).
    """

    level: str | None
    n: int
    test: TestResult | None
    effect_size: EffectSize | None
    no_test_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Texts(_Plain):
    """Every text of the result as plain Unicode (mathtext is built by the renderer).

    Design "contingency" (barstats) alone fills ``bar_labels`` (one per entry of
    ``StatsResult.bars``: its proportion test's "p = 0.09", drawn above the bar
    when ``settings.proportion_test``; "" for a bar without a test),
    ``n_labels`` (one per bar: "(n = 13)", drawn under it) and ``cell_labels``
    (one per entry of ``StatsResult.cells``: "16%", "3" or "3\\n(16%)" as
    ``settings.label`` says; "" for a zero cell, which has no label); they are
    empty in every other design.
    """

    subtitle: str | None
    caption: str | None
    apa: str
    alt_text: str
    centrality_labels: tuple[str, ...]
    tick_labels: tuple[str, ...]
    pairwise_labels: tuple[str, ...]
    bar_labels: tuple[str, ...] = ()
    n_labels: tuple[str, ...] = ()
    cell_labels: tuple[str, ...] = ()


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
    """The analysis settings that shaped the numbers and texts.

    ``test_value`` is the value histostats tests against (ggstatsplot's
    ``test.value``); None in every other design. The pairwise settings keep their
    defaults in the designs without pairs (histostats, scatterstats), and
    ``alternative`` is "two-sided" in scatterstats, whose correlations are always
    tested two-sided.

    Design "contingency" (barstats) alone sets ``paired`` (McNemar's test of the
    two measurements x and y), ``ratio`` (the expected shares of the levels of x,
    in their order, or None for equal shares), ``label`` ("percentage",
    "counts" or "both": what each cell's label shows), ``digits_perc`` (the
    decimals of a percentage label) and ``proportion_test`` (whether the figure
    labels each bar with its proportion test's p); they are None in every other
    design. Its ``pairwise_display`` is "none" (the pairs are never drawn), and
    ``bf_prior`` and ``tr`` keep their defaults (barstats has neither).
    """

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
    test_value: float | None = None
    paired: bool | None = None
    ratio: tuple[float, ...] | None = None
    label: str | None = None
    digits_perc: int | None = None
    proportion_test: bool | None = None


# The settings only design "contingency" (barstats) sets.
_CONTINGENCY_SETTINGS: Final = (
    "paired",
    "ratio",
    "label",
    "digits_perc",
    "proportion_test",
)
# What a barstats cell label may show (Settings.label).
CELL_LABELS: Final = ("percentage", "counts", "both")


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


def _check_schema_version(data: Mapping[str, object]) -> None:
    """Refuse a result whose field set is not this version's (SCHEMA_VERSION)."""
    rerun = (
        "Results cannot be converted between schema versions: compute the result "
        "again with this version of inferplot (the analysis call, or the figure "
        "file's script)."
    )
    if "schema_version" not in data:
        raise ValueError(
            "This result has no schema_version, so it was written by an inferplot "
            f"that used schema version 1; this inferplot reads schema version "
            f"{SCHEMA_VERSION} only. {rerun}"
        )
    version = data["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(
            f"This result has schema_version {version!r}; this inferplot reads "
            f"schema version {SCHEMA_VERSION} only. {rerun}"
        )


def tied_count(values: Iterable[Iterable[float]]) -> int:
    """The tied values among ``values`` (every level pooled): their number minus
    the number of distinct ones (``StatsResult.n_tied``; DECISIONS.md D47)."""
    total = 0
    distinct: set[float] = set()
    for level in values:
        for value in level:
            total += 1
            distinct.add(value)
    return total - len(distinct)


def paired_tied_count(columns: Sequence[Sequence[float]]) -> int:
    """The ties among what a within-subjects rank test ranks (``StatsResult.n_tied``
    of design "within"; ``columns`` holds each level's values in subject order).

    Two conditions: the signed-rank test ranks the absolute paired differences
    a - b and drops the zero ones, so the count is the subjects minus the distinct
    nonzero |a - b| (a zero difference and each repeat of a size count once).
    Three or more: Friedman's test ranks each subject's values, so the count is
    the sum over subjects of the levels minus the subject's distinct values.
    """
    k = len(columns)
    if k < 2:
        raise ValueError(f"a within-subjects design has 2 or more levels, got {k}")
    rows = list(zip(*columns, strict=True))
    if k == 2:
        sizes = {abs(a - b) for a, b in rows}
        sizes.discard(0.0)
        return len(rows) - len(sizes)
    return sum(k - len(set(row)) for row in rows)


def one_sample_tied_count(values: Sequence[float], test_value: float) -> int:
    """The ties among what the one-sample signed-rank test ranks
    (``StatsResult.n_tied`` of design "one-sample"): it ranks the sizes of the
    nonzero x - test_value, so the count is the values minus the distinct nonzero
    |x - test_value| (a value equal to test_value and each repeat of a size
    count once), as for paired differences (D71)."""
    sizes = {abs(v - test_value) for v in values}
    sizes.discard(0.0)
    return len(values) - len(sizes)


def correlation_tied_count(x: Sequence[float], y: Sequence[float]) -> int:
    """The ties among what Spearman's rho ranks (``StatsResult.n_tied`` of design
    "correlation"): x and y are ranked separately, so the count is the pairs minus
    the distinct values of x plus the pairs minus the distinct values of y, out of
    2 n ranked values."""
    if len(x) != len(y):
        raise ValueError(f"x and y must pair up; got {len(x)} and {len(y)} values")
    return (len(x) - len(set(x))) + (len(y) - len(set(y)))


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
    ("ci_sidedness", "str"),
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

    Design "within" (``withinstats``): ``subject_id`` names the subject column and
    ``subjects`` lists the complete subjects; ``values[j][i]`` is the value of
    ``subjects[i]`` at ``levels[j]``, so every level has one value per subject in
    the same order and the renderer can join each subject's points.
    ``dropped_subjects`` counts the subjects left out because they lack a value at
    some level; ``n_obs`` is the number of complete subjects (ggstatsplot's
    ``n.obs``, printed as the number of pairs), and ``n_dropped`` counts rows
    dropped for a missing ``x``, ``y`` or ``subject_id``. ``dropped_subject_rows``
    counts the rows, complete in themselves, that belong to the dropped subjects,
    so the rows read are ``n_obs * len(levels) + n_dropped +
    dropped_subject_rows``. Design "between" has ``subject_id`` None,
    ``subjects`` empty and ``dropped_subjects`` and ``dropped_subject_rows`` 0;
    its rows read are ``n_obs + n_dropped``.

    ``n_tied`` counts the ties among what the design's rank test ranks
    (:func:`tied_count`, :func:`paired_tied_count`): design "between", every
    analyzed value pooled minus the number of distinct ones; design "within" with
    two conditions, the subjects minus the distinct nonzero absolute paired
    differences (the signed-rank test drops zero differences and ranks the
    others' sizes); with three or more, the sum over subjects of the levels minus
    the subject's distinct values (Friedman's test ranks within each subject).
    The denominators are ``n_obs``, ``n_obs`` and ``n_obs * len(levels)``. It is
    stored with the result, not derived
    from ``values``, so ``check()`` gives the same answer on the copy without
    values that ``save()`` embeds (``heavy_ties``, SPEC section 8).

    ``test`` and ``effect_size`` are None for type "bayes", which has no
    frequentist test: its two-group Bayes factor is in ``bayes`` and, with three or
    more groups, each pair's in ``pairwise`` (the Bayesian ANOVA and posterior
    estimates arrive in inferplot 0.2, DECISIONS.md D41).

    Design "one-sample" (``histostats``): ``x`` is the analyzed column and ``y``
    None; ``levels``, ``subjects`` and ``pairwise`` are empty; ``values`` holds one
    tuple, the analyzed values of ``x`` in row order; ``descriptives`` holds one
    entry whose ``level`` is the column name; ``settings.test_value`` is the value
    tested against. ``n_obs`` counts the values, ``n_dropped`` the rows with a
    missing ``x``, and ``n_tied`` the values minus the distinct nonzero |x -
    test_value| (:func:`one_sample_tied_count`).

    Design "correlation" (``scatterstats``): ``x`` and ``y`` are the two columns;
    ``levels``, ``subjects``, ``descriptives`` and ``pairwise`` are empty;
    ``values`` holds two tuples of equal length, x and y of the complete pairs in
    row order. ``point_labels`` holds one text per pair, the ``label_var`` value
    ("" for no label), and is empty without ``label_var``. ``n_obs`` counts the
    complete pairs, ``n_dropped`` the rows with a missing ``x`` or ``y``, and
    ``n_tied`` the ties within x plus those within y (:func:`correlation_tied_count`).

    Design "contingency" (``barstats``): ``x`` is the column whose levels fill
    the bars and ``levels`` its levels; ``y`` is the column whose levels are the
    bars and ``y_levels`` its levels (``y`` None and ``y_levels`` empty in the
    one-way design, a single bar). Declared levels without observations are
    kept in both, as ggstatsplot keeps them. ``counts`` is the table of counts,
    one row per level of x and one column per level of y (a single column
    without y), as ints; it is aggregate data, so ``to_json(include_values=
    False)`` keeps it. ``bars`` holds one entry per bar drawn (the levels of y
    that hold observations, in order) with its n and proportion test, and
    ``cells`` one entry per bar and level of x, in that order, with the count
    and its share of the bar. ``counts_var`` is the column of counts when the
    data are a table (``counts=``), else None. ``values``, ``subjects`` and
    ``descriptives`` are empty; ``n_obs`` is the table total (pairs with
    ``settings.paired``), ``n_dropped`` the rows with a missing ``x``, ``y`` or
    ``counts``, and ``n_tied`` 0 (no rank test). ``pairwise`` holds Fisher's
    exact test between every two levels of x that hold observations (when
    ggstatsplot computes them), never drawn.

    ``point_labels`` is the user's data, like ``values`` and ``subjects``:
    ``to_json(include_values=False)`` leaves all three out. Every result is
    checked on construction: a field set that does not fit its design raises
    ``ValueError``.
    """

    analysis: str
    design: str
    type: str
    x: str
    y: str | None
    subject_id: str | None
    levels: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    subjects: tuple[str, ...]
    descriptives: tuple[Descriptive, ...]
    test: TestResult | None
    effect_size: EffectSize | None
    bayes: BayesFactor | None
    pairwise: tuple[PairwiseComparison, ...]
    n_obs: int
    n_dropped: int
    dropped_subjects: int
    dropped_subject_rows: int
    n_tied: int
    warnings: tuple[str, ...]
    provenance: Provenance
    settings: Settings
    text: Texts
    point_labels: tuple[str, ...] = ()
    label_var: str | None = None
    y_levels: tuple[str, ...] = ()
    counts: tuple[tuple[int, ...], ...] = ()
    cells: tuple[Cell, ...] = ()
    bars: tuple[Bar, ...] = ()
    counts_var: str | None = None

    def __post_init__(self) -> None:
        _Plain.__post_init__(self)
        problem = _shape_problem(self)
        if problem is not None:
            raise ValueError(f"StatsResult (design {self.design!r}): {problem}")

    # -- export -------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Nested dicts and lists; floats stay floats (``inf`` and ``nan`` kept).

        The first key is ``schema_version`` (:data:`SCHEMA_VERSION`), then the
        fields in order.
        """
        fields = cast("dict[str, Any]", _to_dict(self))
        return {"schema_version": SCHEMA_VERSION, **fields}

    def to_json(self, indent: int | None = 2, *, include_values: bool = True) -> str:
        """Strict, ASCII-only JSON: inf as "Inf", -inf as "-Inf", nan as null.

        The first key is ``schema_version`` (:data:`SCHEMA_VERSION`), the version of
        the field set, which :meth:`from_json` checks.

        ``include_values=False`` writes ``values``, ``subjects`` and
        ``point_labels`` as empty lists: every number and text except the analyzed
        observations, the subject labels and the point labels, which can be large
        and are the user's data. ``save()`` embeds this form in figure files; a
        result read back from it has ``values == ()``, ``subjects == ()`` and
        ``point_labels == ()`` and cannot be drawn again, but ``check()`` reports
        the same issues (every check reads stored fields, not the observations).

        Example:
            >>> import pandas as pd
            >>> import inferplot as ip
            >>> df = pd.DataFrame({
            ...     "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            ...     "score": [1.0, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9.5],
            ... })
            >>> result = ip.analyze.betweenstats(df, "group", "score")
            >>> import json
            >>> data = json.loads(result.to_json())
            >>> data["schema_version"], data["test"]["name"], len(data["pairwise"])
            (4, 'welch_anova', 3)
        """
        fields = cast("dict[str, object]", _encode(self))
        if not include_values:
            fields["values"] = []
            fields["subjects"] = []
            fields["point_labels"] = []
        encoded = {"schema_version": SCHEMA_VERSION, **fields}
        return json.dumps(encoded, indent=indent, allow_nan=False, ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StatsResult:
        """Rebuild a result from :meth:`to_dict` or parsed :meth:`to_json` output.

        Raises ``ValueError`` when ``schema_version`` is missing (a result written
        by an inferplot before version 2 of the field set) or is not
        :data:`SCHEMA_VERSION`: such a result has to be computed again.
        """
        if not isinstance(cast("object", data), Mapping):
            raise TypeError(f"result: expected an object, got {data!r}")
        _check_schema_version(data)
        fields = {key: value for key, value in data.items() if key != "schema_version"}
        return _from_mapping(cls, fields, "result")

    @classmethod
    def from_json(cls, text: str) -> StatsResult:
        """Rebuild a result from :meth:`to_json` output ("Inf" becomes inf again).

        Also reads the copy ``fig.save()`` embeds in a figure file. Raises
        ``ValueError`` for JSON of another schema version (see :meth:`from_dict`).

        Example:
            >>> import pandas as pd
            >>> import inferplot as ip
            >>> df = pd.DataFrame({
            ...     "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            ...     "score": [1.0, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9.5],
            ... })
            >>> result = ip.analyze.betweenstats(df, "group", "score")
            >>> ip.StatsResult.from_json(result.to_json()) == result
            True
        """
        return cls.from_dict(json.loads(text))

    def frame_rows(self) -> list[dict[str, Any]]:
        """The rows of :meth:`to_frame`: the main test, then each pair.

        Type "bayes" has no frequentist test: its main row is the two-group Bayes
        factor, and with three or more groups there is no main row (the Bayesian
        ANOVA arrives in inferplot 0.2).

        Design "contingency" (barstats): the main test, then each bar's
        proportion test (``group1`` names the bar's level of y, ``displayed``
        says whether the figure labels it), then the pairs. A Fisher pair's
        ``alternative`` is the analysis' only where its table is 2 x 2 (R's
        ``fisher.test`` ignores it otherwise), and its interval's sidedness
        follows ``alternative`` (effectsize's ``cramers_v`` as ggstatsplot calls
        it).
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
                    "ci_sidedness": None if es is None else es.sidedness,
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
                    "label": _BAYES_ROW_LABELS.get(self.design, "Bayesian t-test"),
                    "alternative": "two-sided",
                    "bf10": bayes.bf10,
                    "log_bf01": bayes.log_bf01,
                }
            )
        rows.extend(_bar_rows(self))
        contingency = self.design == "contingency"
        for pair in self.pairwise:
            rows.append(
                {
                    "test": pair.test,
                    "label": pair.test_label,
                    "group1": pair.group1,
                    "group2": pair.group2,
                    "statistic_name": None
                    if pair.statistic is None
                    else pair.statistic_name,
                    "statistic": pair.statistic,
                    "df": pair.df,
                    "df_error": None,
                    "p_value": pair.p_value,
                    "p_adjusted": pair.p_adjusted,
                    "adjust_method": pair.adjust_method,
                    "alternative": _pair_alternative(self, pair)
                    if contingency
                    else "two-sided",
                    "effect_size": None,
                    "estimate": pair.estimate,
                    "conf_level": pair.conf_level,
                    "conf_low": pair.conf_low,
                    "conf_high": pair.conf_high,
                    # Pairwise intervals (lincon) are two-sided; a barstats
                    # pair's Cramér's V interval follows alternative.
                    "ci_sidedness": None
                    if pair.conf_low is None
                    else self.settings.alternative
                    if contingency
                    else "two-sided",
                    "bf10": None if pair.bayes is None else pair.bayes.bf10,
                    "log_bf01": None if pair.bayes is None else pair.bayes.log_bf01,
                    "significant": pair.significant,
                    "displayed": pair.displayed,
                }
            )
        return rows

    def to_frame(self, backend: Literal["pandas", "polars"] = "pandas") -> Any:
        """One row per test: the main test first, then each pair.

        ``alternative`` is the test's hypothesis; ``ci_sidedness`` is the interval's
        own sidedness (``EffectSize.sidedness`` on the main row, "two-sided" for a
        pair's interval, null without an interval): a two-sided test can carry a
        one-sided interval, such as partial omega squared's [low, 1].

        Missing values are nulls (pandas: NaN in float columns, None in text
        columns, ``<NA>`` in the nullable boolean columns).

        Example:
            >>> import pandas as pd
            >>> import inferplot as ip
            >>> df = pd.DataFrame({
            ...     "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            ...     "score": [1.0, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9.5],
            ... })
            >>> result = ip.analyze.betweenstats(df, "group", "score")
            >>> frame = result.to_frame()
            >>> list(frame["test"])
            ['welch_anova', 'games_howell', 'games_howell', 'games_howell']
            >>> list(frame["group2"][1:])
            ['b', 'c', 'c']
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
        """Statistical problems with this result; empty when there are none.

        Each is an :class:`~inferplot.check.Issue` with a stable ``code``, its
        ``severity`` ("warning" or "info"), a ``message`` and a ``fix``.
        ``fig.check()`` adds the problems of the drawn figure.

        Example:
            >>> import pandas as pd
            >>> import inferplot as ip
            >>> df = pd.DataFrame({
            ...     "group": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            ...     "score": [1.0, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9.5],
            ... })
            >>> result = ip.analyze.betweenstats(df, "group", "score")
            >>> [issue.code for issue in result.check()]
            ['small_n', 'pairs_hidden']
        """
        from inferplot.check import result_issues

        return result_issues(self)

    def __str__(self) -> str:
        """The APA sentence, then a table of the pairwise comparisons if any."""
        if not self.pairwise:
            return self.text.apa
        return self.text.apa + "\n\n" + _pairwise_table(self)


def _bar_rows(result: StatsResult) -> list[dict[str, Any]]:
    """The :meth:`StatsResult.to_frame` rows of the per-bar proportion tests
    (design "contingency"), in bar order; bars without a test have none."""
    rows: list[dict[str, Any]] = []
    for bar in result.bars:
        test, es = bar.test, bar.effect_size
        if test is None:
            continue
        rows.append(
            {
                **dict.fromkeys(name for name, _ in _FRAME_COLUMNS),
                "test": test.name,
                "label": test.label,
                "group1": bar.level,
                "statistic_name": test.statistic_name,
                "statistic": test.statistic,
                "df": test.df,
                "df_error": test.df_error,
                "p_value": test.p_value,
                "alternative": test.alternative,
                "effect_size": None if es is None else es.name,
                "estimate": None if es is None else es.estimate,
                "conf_level": None if es is None else es.conf_level,
                "conf_low": None if es is None else es.conf_low,
                "conf_high": None if es is None else es.conf_high,
                "ci_sidedness": None if es is None else es.sidedness,
                "displayed": bool(result.settings.proportion_test),
            }
        )
    return rows


def _pair_alternative(result: StatsResult, pair: PairwiseComparison) -> str:
    """The hypothesis of a barstats Fisher pair: the analysis' ``alternative``
    where the pair's table (its two rows of ``counts``, over the levels of y
    either holds) is 2 x 2, else "two-sided"."""
    side = result.settings.alternative
    if side == "two-sided" or not result.counts:
        return "two-sided"
    index = {level: i for i, level in enumerate(result.levels)}
    first = result.counts[index[pair.group1]]
    second = result.counts[index[pair.group2]]
    held = sum(1 for a, b in zip(first, second, strict=True) if a or b)
    return side if held == 2 else "two-sided"


# The to_frame() label of a Bayes factor without a frequentist test, by design.
_BAYES_ROW_LABELS: Final = {
    "between": "Bayesian t-test",
    "within": "Bayesian paired t-test",
    "one-sample": "Bayesian one-sample t-test",
    "correlation": "Bayesian Pearson correlation",
}


def _shape_problem(result: StatsResult) -> str | None:
    """What makes ``result``'s fields unfit for its design, or None.

    Every design: a known design and analysis; ``point_labels`` and
    ``label_var`` only in design "correlation", ``test_value`` only in
    "one-sample", ``y`` None only there. Designs "one-sample" and "correlation"
    are checked in full (SPEC section 6): no levels, subjects or pairs, the
    values and labels in their shape. The checks allow the copy without
    observations (``values``, ``subjects`` and ``point_labels`` empty).
    """
    design, analysis = result.design, result.analysis
    if design not in DESIGNS:
        known = ", ".join(repr(d) for d in DESIGNS)
        return f"unknown design; known: {known}"
    if analysis not in DESIGNS.values():
        known = ", ".join(repr(a) for a in DESIGNS.values())
        return f"unknown analysis {analysis!r}; known: {known}"
    one_sample, correlation = design == "one-sample", design == "correlation"
    if not correlation and (result.point_labels or result.label_var is not None):
        return "point_labels and label_var belong to design 'correlation' only"
    if one_sample != (result.settings.test_value is not None):
        return (
            "settings.test_value is required in design 'one-sample' and must be None "
            "in every other design"
        )
    if design == "contingency":
        return _contingency_problem(result)
    contingency_only = _contingency_fields_problem(result)
    if contingency_only is not None:
        return contingency_only
    if one_sample != (result.y is None):
        return "y is None in design 'one-sample' and a column name in every other"
    if not (one_sample or correlation):
        return None
    if analysis != DESIGNS[design]:
        return f"comes from {DESIGNS[design]}, not {analysis}"
    n = result.n_obs
    empty: dict[str, tuple[object, ...]] = {
        "levels": result.levels,
        "subjects": result.subjects,
        "pairwise": result.pairwise,
    }
    if correlation:
        empty["descriptives"] = result.descriptives
    for name, value in empty.items():
        if value:
            return f"{name} must be empty"
    if result.subject_id is not None:
        return "subject_id must be None"
    if result.dropped_subjects or result.dropped_subject_rows:
        return "dropped_subjects and dropped_subject_rows must be 0"
    columns = 1 if one_sample else 2
    if result.values:
        if len(result.values) != columns:
            return f"values must hold {columns} tuple(s), got {len(result.values)}"
        lengths = {len(v) for v in result.values}
        if lengths != {n}:
            return f"every tuple of values must hold n_obs = {n} values, got {lengths}"
    if one_sample:
        test_value = result.settings.test_value
        if test_value is not None and not math.isfinite(test_value):
            return f"settings.test_value must be finite, got {test_value!r}"
        if len(result.descriptives) != 1:
            return f"descriptives must hold one entry, got {len(result.descriptives)}"
        (descriptive,) = result.descriptives
        if descriptive.level != result.x or descriptive.n != n:
            return (
                f"the descriptive must be of the column {result.x!r} with n = {n}; "
                f"got {descriptive.level!r} with n = {descriptive.n}"
            )
        return None
    if result.point_labels:
        if result.label_var is None:
            return "point_labels need label_var, the column they come from"
        if len(result.point_labels) != n:
            return (
                f"point_labels must hold one text per pair (n_obs = {n}), got "
                f"{len(result.point_labels)}"
            )
    if result.settings.alternative != "two-sided":
        return "correlations are tested two-sided; settings.alternative must say so"
    return None


def _contingency_fields_problem(result: StatsResult) -> str | None:
    """The fields of design "contingency" set in another design, or None."""
    filled = [
        name
        for name in ("y_levels", "counts", "cells", "bars")
        if getattr(result, name)
    ]
    if result.counts_var is not None:
        filled.append("counts_var")
    filled += [
        f"text.{name}"
        for name in ("bar_labels", "n_labels", "cell_labels")
        if getattr(result.text, name)
    ]
    filled += [
        f"settings.{name}"
        for name in _CONTINGENCY_SETTINGS
        if getattr(result.settings, name) is not None
    ]
    if filled:
        return f"{', '.join(filled)} belong to design 'contingency' only"
    return None


def _contingency_problem(result: StatsResult) -> str | None:
    """What makes a barstats result's fields unfit for design "contingency"
    (SPEC section 6), or None: the table, its levels, bars and cells agree with
    each other and with ``n_obs``; the settings of the design are set; nothing
    of the other designs is."""
    if result.analysis != "barstats":
        return f"comes from barstats, not {result.analysis}"
    for name in ("values", "subjects", "descriptives"):
        if getattr(result, name):
            return f"{name} must be empty"
    if result.subject_id is not None:
        return "subject_id must be None"
    if result.dropped_subjects or result.dropped_subject_rows or result.n_tied:
        return "dropped_subjects, dropped_subject_rows and n_tied must be 0"
    problem = _contingency_settings_problem(result)
    if problem is not None:
        return problem
    levels, y_levels = result.levels, result.y_levels
    if len(levels) < 2 or len(set(levels)) != len(levels):
        return f"levels must hold two or more distinct levels of x, got {levels!r}"
    if (result.y is None) != (not y_levels):
        return "y_levels name the levels of y, and are empty only without y"
    if y_levels and (len(y_levels) < 2 or len(set(y_levels)) != len(y_levels)):
        return f"y_levels must hold two or more distinct levels, got {y_levels!r}"
    columns = len(y_levels) if y_levels else 1
    counts = result.counts
    if len(counts) != len(levels) or any(len(row) != columns for row in counts):
        return (
            f"counts must hold {len(levels)} rows (levels of x) of {columns} "
            "count(s) each"
        )
    if any(type(v) is not int or v < 0 for row in counts for v in row):
        return "counts must be whole numbers >= 0"
    if sum(sum(row) for row in counts) != result.n_obs:
        return f"counts must add up to n_obs = {result.n_obs}"
    if result.settings.paired and y_levels != levels:
        return "a paired table has the same levels, in the same order, in x and y"
    problem = _bars_problem(result, columns)
    if problem is not None:
        return problem
    for pair in result.pairwise:
        if pair.displayed:
            return "barstats pairs are never drawn: displayed must be False"
        if pair.group1 not in levels or pair.group2 not in levels:
            return f"the pair {pair.group1!r}, {pair.group2!r} must be levels of x"
        untested = pair.p_value is None
        if (pair.no_test_reason is not None) != untested or (
            untested and (pair.p_adjusted is not None or pair.significant is not None)
        ):
            return (
                f"the pair {pair.group1!r}, {pair.group2!r}: a pair without a "
                "p-value has p_adjusted and significant None and says why in "
                "no_test_reason; a pair with one has no no_test_reason"
            )
    return None


def _contingency_settings_problem(result: StatsResult) -> str | None:
    settings = result.settings
    unset = [
        name
        for name in _CONTINGENCY_SETTINGS
        if name != "ratio" and getattr(settings, name) is None
    ]
    if unset:
        return f"settings.{', settings.'.join(unset)} must be set"
    if settings.label not in CELL_LABELS:
        return f"settings.label must be one of {CELL_LABELS!r}, got {settings.label!r}"
    if settings.paired and result.y is None:
        return "settings.paired needs y, the second measurement"
    ratio = settings.ratio
    if ratio is not None and (
        len(ratio) != len(result.levels)
        or not all(math.isfinite(r) and r > 0 for r in ratio)
    ):
        return (
            "settings.ratio must hold one positive share per level of x "
            f"({len(result.levels)}), got {ratio!r}"
        )
    if settings.pairwise_display != "none":
        return "barstats pairs are never drawn: settings.pairwise_display is 'none'"
    return None


def _bars_problem(result: StatsResult, columns: int) -> str | None:
    """``bars`` and ``cells`` against ``counts``: a bar per level of y that
    holds observations (one bar without y), a cell per bar and level of x."""
    totals = [sum(row[j] for row in result.counts) for j in range(columns)]
    if result.y_levels:
        expected = [
            (level, n)
            for level, n in zip(result.y_levels, totals, strict=True)
            if n > 0
        ]
    else:
        expected = [(None, totals[0])]
    got = [(bar.level, bar.n) for bar in result.bars]
    if got != expected:
        return f"bars must be the bars with observations, {expected!r}; got {got!r}"
    index = {level: j for j, level in enumerate(result.y_levels)}
    cells = [
        (
            bar.level,
            level,
            result.counts[i][0 if bar.level is None else index[bar.level]],
        )
        for bar in result.bars
        for i, level in enumerate(result.levels)
    ]
    if [(c.bar, c.level, c.count) for c in result.cells] != cells:
        return "cells must hold every bar's count of each level of x, in order"
    by_bar = {bar.level: bar.n for bar in result.bars}
    for cell in result.cells:
        if cell.percent != cell.count / by_bar[cell.bar] * 100:
            return (
                f"the cell {cell.bar!r}, {cell.level!r} must hold its share of the "
                "bar, count / n * 100"
            )
    text = result.text
    if not (len(text.bar_labels) == len(text.n_labels) == len(result.bars)):
        return "text.bar_labels and text.n_labels must hold one label per bar"
    if len(text.cell_labels) != len(result.cells):
        return "text.cell_labels must hold one label per cell"
    return None


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
    if all(pair.statistic is None for pair in result.pairwise):
        # Fisher's exact test (barstats) has no statistic and no df.
        rows = [(*row[:2], *row[4:]) for row in rows]
        header = rows[0]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(
            cell.ljust(width) for cell, width in zip(row, widths, strict=True)
        ).rstrip()
        for row in rows
    ]
    title = f"Pairwise comparisons ({first.test_label}):"
    return "\n".join([title, *lines])
