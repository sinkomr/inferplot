"""Argument checks for the public functions (SPEC.md section 2, "API errors").

Two parts:

* ``checked``: a decorator that checks keyword *names* and the positional count
  before the function runs. It is the only place in the package with
  ``*args, **kwargs``; ``functools.wraps`` keeps the real signature visible to
  ``inspect.signature``, and the ``ParamSpec`` typing keeps it visible to type
  checkers.
* value validators (``canonical_type``, ``check_conf_level``, ...): each takes the
  value and the argument name, returns it as a plain Python value, and otherwise
  raises ``TypeError`` (wrong kind of value) or ``ValueError`` (right kind, not
  allowed).

Every message is written for an AI agent reading a traceback: it says what was
passed, what is allowed, and how to fix the call.
"""

from __future__ import annotations

import difflib
import functools
import inspect
import math
import numbers
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ParamSpec, TypeVar, cast

import numpy as np

__all__ = [
    "ALTERNATIVES",
    "LAYERS",
    "PAIRWISE_DISPLAYS",
    "P_ADJUST_METHODS",
    "TYPES",
    "canonical_pairwise_display",
    "canonical_type",
    "check_alternative",
    "check_artist_args",
    "check_bool",
    "check_conf_level",
    "check_digits",
    "check_layers",
    "check_optional_bool",
    "check_optional_text",
    "check_p_adjust_method",
    "check_positive",
    "check_probability",
    "check_seed",
    "check_tr",
    "checked",
]

P = ParamSpec("P")
R = TypeVar("R")

# Cutoff used for every did-you-mean suggestion (SPEC.md section 2).
_CLOSE_MATCH_CUTOFF = 0.6
# Longest repr an error message quotes; longer values are named by their type.
_MAX_REPR = 40

# Canonical names and their accepted aliases (ggstatsplot's).
TYPES: Mapping[str, str] = {
    "parametric": "parametric",
    "p": "parametric",
    "nonparametric": "nonparametric",
    "np": "nonparametric",
    "robust": "robust",
    "r": "robust",
    "bayes": "bayes",
    "bf": "bayes",
}
PAIRWISE_DISPLAYS: Mapping[str, str] = {
    "significant": "significant",
    "s": "significant",
    "non-significant": "non-significant",
    "ns": "non-significant",
    "all": "all",
    "none": "none",
}
# R's p.adjust() names, case-sensitive as in R.
P_ADJUST_METHODS = (
    "holm",
    "hochberg",
    "hommel",
    "bonferroni",
    "BH",
    "BY",
    "fdr",
    "none",
)
ALTERNATIVES = ("two-sided", "less", "greater")
LAYERS = ("violin", "box", "points")


# --- helpers ----------------------------------------------------------------------


def _show(value: object) -> str:
    """``repr(value)`` when short, else its type name (never a whole data frame)."""
    text = repr(value)
    if len(text) <= _MAX_REPR:
        return text
    cls = type(value)
    module = cls.__module__.split(".")[0]
    name = cls.__name__ if module == "builtins" else f"{module}.{cls.__name__}"
    return f"a {name}"


def _closest(word: str, candidates: Iterable[str]) -> str | None:
    """The candidate closest to ``word``: a case-insensitive exact match first, then
    difflib with cutoff 0.6 on case-folded text."""
    options = list(candidates)
    folded = {c.casefold(): c for c in reversed(options)}
    key = word.casefold()
    if key in folded:
        return folded[key]
    match = difflib.get_close_matches(
        key, list(folded), n=1, cutoff=_CLOSE_MATCH_CUTOFF
    )
    return folded[match[0]] if match else None


def _snake(name: str) -> str:
    """``name`` in snake_case: R's dots, hyphens and camelCase become underscores."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[.\-\s]+", "_", name).lower()


def _choices(mapping: Mapping[str, str]) -> str:
    """``'parametric' (alias 'p'), ...`` from a name -> canonical mapping."""
    parts: list[str] = []
    for canonical in dict.fromkeys(mapping.values()):
        aliases = [a for a, c in mapping.items() if c == canonical and a != canonical]
        alias = f" (alias {', '.join(repr(a) for a in aliases)})" if aliases else ""
        parts.append(f"{canonical!r}{alias}")
    return ", ".join(parts)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool | np.bool_)


def _real(value: object, name: str, example: str) -> float:
    """``value`` as a float if it is a real number (not a bool), else TypeError."""
    if _is_bool(value) or not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name} must be a number, for example {name}={example}; "
            f"got {_show(value)}."
        )
    return float(value)


def _integer(value: object, name: str, example: str) -> int:
    """``value`` as an int if it is an integer (not a bool), else TypeError."""
    if _is_bool(value) or not isinstance(value, numbers.Integral):
        hint = ""
        if isinstance(value, numbers.Real) and not _is_bool(value):
            number = float(value)
            if math.isfinite(number) and number.is_integer():
                hint = f" (write {name}={int(number)})"
        raise TypeError(
            f"{name} must be a whole number (an int), for example {name}={example}; "
            f"got {_show(value)}{hint}."
        )
    return int(value)


def _text(value: object, name: str, allowed: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string, one of {allowed}; got {_show(value)}."
        )
    return value


# --- keyword names ----------------------------------------------------------------


def _removed(arg: str, version: str, rest: str) -> str:
    return f"{arg}= was removed in ggstatsplot {version}; {rest} Remove {arg}=."


def _targeted(
    arg: str, key: str, value: object, fn: str, valid: frozenset[str]
) -> TypeError | NotImplementedError | None:
    """SPEC.md section 2's message for a removed or borrowed keyword, if ``key`` (the
    snake_case spelling of ``arg``) is one of them."""
    if key == "var_equal":
        return TypeError(
            _removed(
                arg,
                "1.0.0",
                "Welch's test is always used (it does not assume equal variances), "
                "so there is nothing to set.",
            )
        )
    if key == "nboot":
        return TypeError(
            f"{arg}= was removed in ggstatsplot 1.0.0; inferplot uses 2,000 seeded "
            f"bootstrap resamples wherever a method needs them. Remove {arg}=; pass "
            "seed= (an int >= 0) to change the random stream."
        )
    if key == "effsize_type":
        return TypeError(
            _removed(
                arg,
                "1.0.0",
                "Hedges' g and partial omega squared are always used (parametric "
                "type), so there is nothing to choose.",
            )
        )
    if key == "k":
        example = _show(value) if isinstance(value, int) else "2"
        return TypeError(
            f"{arg}= was renamed digits in ggstatsplot 0.12.3; pass digits={example} "
            "instead."
        )
    # A function without ax= only computes; drawing options belong elsewhere.
    draws = "ax" in valid
    drawing = f"ip.{fn}(...) or ip.render.{fn}(result, ...)"
    if key == "package":
        if "palette" in valid or draws:
            return TypeError(
                f"{arg}= was removed in ggstatsplot 1.0.0; pass palette= instead: a "
                "list of colors or a matplotlib colormap name, for example "
                f"palette='tab10'. Remove {arg}=."
            )
        return TypeError(
            f"{arg}= was removed in ggstatsplot 1.0.0; colors are a drawing option: "
            "pass palette= (a list of colors or a matplotlib colormap name) to "
            f"{drawing}; ip.analyze.{fn}() only computes the statistics. "
            f"Remove {arg}=."
        )
    if key == "plot_type":
        if "layers" in valid:
            rest = (
                "use layers=, a tuple of 'violin', 'box' and 'points', for example "
                "layers=('box', 'points')."
            )
        elif draws:
            rest = f"{fn}() draws a fixed set of layers and has no plot type option."
        else:
            rest = (
                f"ip.analyze.{fn}() only computes the statistics. What is drawn is "
                f"chosen when drawing, with {drawing}."
            )
        return TypeError(_removed(arg, "0.12.0", rest))
    if key == "pairwise_comparisons":
        if "pairwise_display" in valid:
            rest = (
                "pairwise comparisons are computed whenever there are three or more "
                "groups; use pairwise_display='none' to hide them or "
                "pairwise_display='all' to show every pair."
            )
        else:
            rest = f"{fn}() has no pairwise comparisons."
        return TypeError(_removed(arg, "0.12.0", rest))
    if key == "output":
        return TypeError(
            _removed(
                arg,
                "0.10.0",
                "use fig.result instead: it holds every number (fig.result.to_json() "
                f"exports them), and ip.analyze.{fn}(...) computes it without drawing.",
            )
        )
    if key == "messages":
        return TypeError(
            f"{arg}= is not needed: inferplot never prints; notes about the analysis "
            f"are in fig.result.warnings. Remove {arg}=."
        )
    if key in {"sampling_plan", "fixed_margin", "prior_concentration"}:
        return TypeError(
            f"{arg}= was removed from ggbarstats in ggstatsplot 1.0.0 (a Bayesian "
            f"contingency-table setting) and inferplot has no equivalent. "
            f"Remove {arg}=."
        )
    if key == "grouping_var":
        return NotImplementedError(
            f"{arg}= (ggstatsplot's grouped_* functions, one plot per value of a "
            "column) arrives in inferplot v0.2. For now call "
            f"{fn}() once per subset of the data."
        )
    if key == "hue":
        return TypeError(
            f"hue= is a seaborn argument; there is no second grouping in {fn}(): "
            "colors already follow x. grouping_var (one plot per value of a column) "
            f"arrives in inferplot v0.2; for now call {fn}() once per subset of the "
            "data. Remove hue=."
        )
    if key in {"ci", "errorbar"}:
        return TypeError(
            f"{arg}= is a seaborn argument; intervals in inferplot are always "
            f"conf_level confidence intervals. Remove {arg}= and pass conf_level= "
            "to change the level, for example conf_level=0.90."
        )
    if key in {"subject", "id", "subject_id"}:
        if "subject_id" in valid:
            example = _show(value) if isinstance(value, str) else "'subject'"
            return TypeError(
                f"{arg}= is not an argument of {fn}(); use subject_id={example}, the "
                "column that identifies each subject."
            )
        return TypeError(
            f"{fn}() compares independent groups and takes no {arg}=. For repeated "
            "measures of the same subjects use "
            "withinstats(data, x, y, subject_id=...); otherwise remove "
            f"{arg}=."
        )
    reason = _NO_EQUIVALENT.get(key)
    if reason is not None:
        return TypeError(
            f"{arg}= is a ggstatsplot argument that inferplot does not have: "
            f"{reason} Remove {arg}=."
        )
    return None


# ggbetweenstats 1.1.4 arguments outside SPEC.md's table that have no inferplot
# equivalent. Without these, difflib would suggest an unrelated argument
# (centrality.type -> centrality_plotting, ggsignif.args -> point_args).
_NO_EQUIVALENT: Mapping[str, str] = {
    "centrality_type": (
        "the centrality measure always follows type (the mean for "
        "type='parametric'), which is ggstatsplot's default."
    ),
    "centrality_point_args": (
        "the centrality point is not styled by an argument; point_args, "
        "boxplot_args and violin_args style the data layers."
    ),
    "centrality_label_args": (
        "the centrality label is not styled by an argument; point_args, "
        "boxplot_args and violin_args style the data layers."
    ),
    "ggsignif_args": (
        "the significance brackets are not styled by an argument; "
        "pairwise_display chooses which pairs are drawn."
    ),
    "ggtheme": (
        "there are no ggplot2 themes in matplotlib; adjust the returned "
        "matplotlib figure instead."
    ),
    "ggplot_component": (
        "there are no ggplot2 components in matplotlib; adjust the returned "
        "matplotlib figure instead."
    ),
}


def _keyword_problem(
    arg: str, value: object, fn: str, valid: frozenset[str], order: Sequence[str]
) -> TypeError | NotImplementedError:
    key = _snake(arg)
    if key in valid:
        return TypeError(
            f"{fn}() has no argument {arg!r}: inferplot spells every argument in "
            f"lowercase snake_case. Write {key}={_show(value)}."
        )
    targeted = _targeted(arg, key, value, fn, valid)
    if targeted is not None:
        return targeted
    close = _closest(key, order)
    if close is not None:
        return TypeError(
            f"{fn}() got an unexpected keyword argument {arg!r}. Did you mean "
            f"{close}={_show(value)}?"
        )
    return TypeError(
        f"{fn}() got an unexpected keyword argument {arg!r}. Its arguments are: "
        f"{', '.join(order)}."
    )


def _positional_problem(
    fn: str, positional: Sequence[str], extra: Sequence[object], valid: frozenset[str]
) -> TypeError:
    names = ", ".join(positional)
    given = len(positional) + len(extra)
    shown = ", ".join(_show(v) for v in extra)
    hint = ""
    first = extra[0]
    if isinstance(first, str):
        if first in TYPES and "type" in valid:
            hint = f" For example {fn}(..., type={first!r})."
        elif first in PAIRWISE_DISPLAYS and "pairwise_display" in valid:
            hint = f" For example {fn}(..., pairwise_display={first!r})."
    plural = "argument" if len(positional) == 1 else "arguments"
    return TypeError(
        f"{fn}() takes {len(positional)} positional {plural} ({names}) but {given} "
        f"were given. Everything after {names} is keyword-only; name each extra "
        f"argument. Extra positional values: {shown}.{hint}"
    )


def _type_name(value: object) -> str:
    """``a pandas.DataFrame``; a short repr for builtins such as dicts."""
    cls = type(value)
    if cls.__module__ == "builtins":
        return _show(value)
    return f"a {cls.__module__.split('.')[0]}.{cls.__name__}"


def _not_a_result(fn: str, value: object) -> TypeError:
    return TypeError(
        f"render.{fn}() draws the result of ip.analyze.{fn}(data, x, y); got "
        f"{_type_name(value)}. Compute it first with "
        f"result = ip.analyze.{fn}(data, x, y, ...) and pass result, or use "
        f"ip.{fn}(data, x, y, ...) to analyze and draw in one call."
    )


def checked(fn: Callable[P, R]) -> Callable[P, R]:
    """Check keyword names and the positional count of every call to ``fn``.

    Unknown keywords raise ``TypeError`` (``NotImplementedError`` for
    ``grouping_var``) with SPEC.md section 2's targeted message, the snake_case
    spelling of an R-style name, or the closest valid keyword. When several
    keywords are wrong, the first one in call order is reported. A drawing
    function (first parameter ``result``) given something other than a
    ``StatsResult`` says so before the positional count is checked. Values are not
    checked here otherwise; the value validators below do that.
    """
    signature = inspect.signature(fn)
    kinds = inspect.Parameter
    for parameter in signature.parameters.values():
        if parameter.kind in (kinds.VAR_POSITIONAL, kinds.VAR_KEYWORD):
            raise TypeError(
                f"checked() needs an explicit signature; {fn.__name__}() takes "
                f"{parameter}."
            )
    positional = tuple(
        p.name
        for p in signature.parameters.values()
        if p.kind in (kinds.POSITIONAL_ONLY, kinds.POSITIONAL_OR_KEYWORD)
    )
    order = tuple(
        p.name
        for p in signature.parameters.values()
        if p.kind in (kinds.POSITIONAL_OR_KEYWORD, kinds.KEYWORD_ONLY)
    )
    valid = frozenset(order)
    name = fn.__name__
    draws_result = bool(positional) and positional[0] == "result"

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        for arg, value in kwargs.items():
            if arg not in valid:
                raise _keyword_problem(arg, value, name, valid, order)
        if draws_result:
            from inferplot.result import StatsResult

            if args or "result" in kwargs:
                given = args[0] if args else kwargs["result"]
                if not isinstance(given, StatsResult):
                    raise _not_a_result(name, given)
        if len(args) > len(positional):
            raise _positional_problem(name, positional, args[len(positional) :], valid)
        return fn(*args, **kwargs)

    return wrapper


# --- values -----------------------------------------------------------------------


def _choice(
    value: object, name: str, mapping: Mapping[str, str], hints: Mapping[str, str]
) -> str:
    allowed = _choices(mapping)
    if value is None and "none" in mapping:
        raise TypeError(
            f"{name} must be a string, one of {allowed}; got None. To turn it off "
            f"write {name}='none'."
        )
    text = _text(value, name, allowed)
    if text in mapping:
        return mapping[text]
    hint = hints.get(text)
    if hint is None:
        close = _closest(text, mapping)
        hint = f" Did you mean {name}={close!r}?" if close is not None else ""
    raise ValueError(f"{name}={text!r} is not allowed; use one of {allowed}.{hint}")


def canonical_type(value: object, name: str = "type") -> str:
    """The canonical statistical type: 'parametric', 'nonparametric', 'robust' or
    'bayes' (ggstatsplot's aliases 'p', 'np', 'r', 'bf' accepted)."""
    return _choice(value, name, TYPES, {})


def canonical_pairwise_display(value: object, name: str = "pairwise_display") -> str:
    """'significant', 'non-significant', 'all' or 'none' (aliases 's', 'ns')."""
    return _choice(value, name, PAIRWISE_DISPLAYS, {})


def check_p_adjust_method(value: object, name: str = "p_adjust_method") -> str:
    """One of R's ``p.adjust()`` method names, spelled exactly as in R."""
    methods = {m: m for m in P_ADJUST_METHODS}
    return _choice(value, name, methods, {})


def check_alternative(value: object, name: str = "alternative") -> str:
    """'two-sided', 'less' or 'greater'; R's 'two.sided' gets a pointed hint."""
    hints = {
        "two.sided": (f" R spells it 'two.sided'; inferplot writes {name}='two-sided'.")
    }
    return _choice(value, name, {a: a for a in ALTERNATIVES}, hints)


def check_conf_level(value: object, name: str = "conf_level") -> float:
    """A confidence level strictly between 0 and 1 (bool rejected)."""
    number = _real(value, name, "0.95")
    if not 0.0 < number < 1.0:
        hint = ""
        if 1.0 < number < 100.0:
            hint = (
                f" It is a proportion, not a percentage: write {name}={number / 100:g}."
            )
        raise ValueError(
            f"{name} must be strictly between 0 and 1, for example {name}=0.95; "
            f"got {_show(value)}.{hint}"
        )
    return number


def check_probability(value: object, name: str) -> float:
    """A probability strictly between 0 and 1, such as ``pairwise_alpha``."""
    number = _real(value, name, "0.05")
    if not 0.0 < number < 1.0:
        raise ValueError(
            f"{name} must be strictly between 0 and 1, for example {name}=0.05; "
            f"got {_show(value)}."
        )
    return number


def check_positive(value: object, name: str, *, example: str = "0.707") -> float:
    """A finite number greater than 0, such as ``bf_prior`` or ``dpi``.

    ``example`` is the value the error messages suggest (``bf_prior``'s default
    unless given).
    """
    number = _real(value, name, example)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(
            f"{name} must be a finite number greater than 0, for example "
            f"{name}={example}; got {_show(value)}."
        )
    return number


def check_tr(value: object, name: str = "tr") -> float:
    """A trimming proportion with 0 <= tr < 0.5."""
    number = _real(value, name, "0.2")
    if not 0.0 <= number < 0.5:
        raise ValueError(
            f"{name} is the proportion trimmed from each end and must satisfy "
            f"0 <= {name} < 0.5, for example {name}=0.2; got {_show(value)}."
        )
    return number


def check_digits(value: object, name: str = "digits") -> int:
    """A whole number of decimal places, >= 0 (bool rejected)."""
    number = _integer(value, name, "2")
    if number < 0:
        raise ValueError(
            f"{name} is a number of decimal places and must be >= 0, for example "
            f"{name}=2; got {number}."
        )
    return number


def check_seed(value: object, name: str = "seed") -> int:
    """A random seed: a whole number >= 0 (bool rejected)."""
    number = _integer(value, name, "0")
    if number < 0:
        raise ValueError(
            f"{name} must be a whole number >= 0, for example {name}=0; got {number}."
        )
    return number


def check_bool(value: object, name: str) -> bool:
    """``True`` or ``False`` (NumPy booleans accepted; 0 and 1 are not)."""
    if not _is_bool(value):
        raise TypeError(f"{name} must be True or False; got {_show(value)}.")
    return bool(value)


def check_optional_bool(value: object, name: str) -> bool | None:
    """``None``, ``True`` or ``False``."""
    if value is None:
        return None
    if not _is_bool(value):
        raise TypeError(f"{name} must be None, True or False; got {_show(value)}.")
    return bool(value)


def check_optional_text(value: object, name: str) -> str | None:
    """``None`` or a string, such as ``title`` or ``xlab``."""
    if value is None:
        return None
    if isinstance(value, str):
        return str(value)  # a plain str, also for str subclasses such as numpy.str_
    raise TypeError(
        f"{name} must be a string or None; got {_show(value)}. Convert it with "
        f"str() first if you mean its text."
    )


def check_layers(value: object, name: str = "layers") -> tuple[str, ...]:
    """A non-empty tuple (or list) of distinct names from 'violin', 'box', 'points'."""
    allowed = ", ".join(repr(layer) for layer in LAYERS)
    if isinstance(value, str) or not isinstance(value, Sequence):
        example = f"({value!r},)" if isinstance(value, str) else "('box', 'points')"
        raise TypeError(
            f"{name} must be a tuple of layer names from {allowed}, for example "
            f"{name}={example}; got {_show(value)}."
        )
    items = list(cast("Sequence[object]", value))
    if not items:
        raise ValueError(
            f"{name} must name at least one of {allowed}; got an empty sequence."
        )
    layers: list[str] = []
    for item in items:
        if not isinstance(item, str) or item not in LAYERS:
            close = _closest(item, LAYERS) if isinstance(item, str) else None
            hint = f" Did you mean {close!r}?" if close is not None else ""
            raise ValueError(
                f"{name} contains {_show(item)}, which is not a layer; use names "
                f"from {allowed}.{hint}"
            )
        if item in layers:
            raise ValueError(f"{name} names {item!r} more than once.")
        layers.append(str(item))
    return tuple(layers)


def check_artist_args(value: object, name: str) -> dict[str, object] | None:
    """``None``, or a mapping of matplotlib property names to values, as a new dict.

    The properties themselves are checked by ``Artist.set`` when drawing.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be a dict of matplotlib properties, for example "
            f"{name}={{'alpha': 0.5}}; got {_show(value)}."
        )
    out: dict[str, object] = {}
    for key, item in cast("Mapping[object, object]", value).items():
        if not isinstance(key, str):
            raise TypeError(
                f"{name} keys must be matplotlib property names (strings); got "
                f"{_show(key)}."
            )
        out[key] = item
    return out
