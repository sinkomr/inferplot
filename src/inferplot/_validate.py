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
from typing import Final, ParamSpec, TypeVar, cast

import numpy as np

__all__ = [
    "ALTERNATIVES",
    "ANALYSIS_KEYWORDS",
    "DRAWING_KEYWORDS",
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
    "check_finite_number",
    "check_layers",
    "check_optional_bool",
    "check_optional_text",
    "check_p_adjust_method",
    "check_positive",
    "check_probability",
    "check_seed",
    "check_tr",
    "checked",
    "closest_name",
    "describe_value",
    "subject_id_required",
]

P = ParamSpec("P")
R = TypeVar("R")

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


_SEPARATORS: Final = re.compile(r"[\s._\-]+")

# Other names for keywords that are very likely meant as these (did-you-mean).
_KEYWORD_SYNONYMS: Final[Mapping[str, str]] = {
    "trim": "tr",
    "trimming": "tr",
    "conf": "conf_level",
    "confidence": "conf_level",
    "confidence_level": "conf_level",
    "ci_level": "conf_level",
    "level": "conf_level",
    "decimals": "digits",
    "prior": "bf_prior",
    "display": "pairwise_display",
    "axes": "ax",
}


def _plain(word: str) -> str:
    """``word`` case-folded, with each run of spaces, dots, hyphens and underscores
    turned into one underscore and none at either end."""
    return _SEPARATORS.sub("_", word.casefold()).strip("_")


def _edits(a: str, b: str) -> int:
    """The optimal string alignment distance: insertions, deletions, substitutions
    and swaps of two neighbouring characters, each counting 1."""
    previous2: list[int] = []
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(
                previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                current[j] = min(current[j], previous2[j - 2] + 1)
        previous2, previous = previous, current
    return previous[len(b)]


def _misspelled(word: str, candidate: str) -> bool:
    """Whether ``word`` is a misspelling or a longer form of ``candidate`` (both
    ``_plain``).

    A misspelling: at most one edit for a name of 4 to 6 letters, two for a longer
    one, none for a shorter one (separators ignored). A longer form: ``word`` is
    ``candidate`` plus at most 4 letters, not a separate word (``bayesian``,
    ``boxplot``, ``seeds``; not ``fdr_tsbh``). Similar-looking names of different
    things (``tail`` and ``title``, ``one-sided`` and ``two-sided``,
    ``holm-bonferroni`` and ``bonferroni``) are neither.
    """
    bare, target = word.replace("_", ""), candidate.replace("_", "")
    allowed = 0 if len(target) <= 3 else 1 if len(target) <= 6 else 2
    if _edits(bare, target) <= allowed:
        return True
    return (
        len(target) >= 3
        and word.startswith(candidate)
        and 0 < len(word) - len(candidate) <= 4
        and word[len(candidate)] != "_"  # an added word is another name
    )


def _abbreviated(word: str, candidate: str) -> bool:
    """Whether ``word`` is the start of ``candidate`` (both ``_plain``; separators
    ignored): the whole of its first two or more parts (``p_adjust``, ``padjust``),
    or at least 5 letters ending inside a part (``param``, ``centrality_plot``).
    Exactly its first part is not enough: ``centrality`` could mean
    ``centrality_plotting`` or a centrality measure, ``pairwise`` several arguments.
    """
    bare, target = word.replace("_", ""), candidate.replace("_", "")
    if bare == target or not target.startswith(bare):
        return False
    ends: list[int] = []
    total = 0
    for part in candidate.split("_"):
        total += len(part)
        ends.append(total)
    if len(bare) in ends:
        return ends.index(len(bare)) >= 1
    return len(bare) >= 5


def _closest(word: str, candidates: Iterable[str]) -> str | None:
    """The candidate ``word`` very likely meant, or None (SPEC.md section 2).

    In turn: a match ignoring case and separators (``Type``, ``two_sided``,
    ``nonsignificant``); a keyword synonym (``trim`` for ``tr``); the misspellings
    and longer forms ``_misspelled`` accepts, the closest by difflib's ratio; the
    one candidate ``word`` abbreviates (none when it abbreviates several). A
    did-you-mean is followed literally by an AI agent, so an unlikely guess is worse
    than none: without one the message lists the valid names.
    """
    options = list(candidates)
    key = _plain(word)
    if not key:
        return None
    plain = {c: _plain(c) for c in options}
    for c in options:
        if plain[c].replace("_", "") == key.replace("_", ""):
            return c
    synonym = _KEYWORD_SYNONYMS.get(key)
    if synonym is not None and synonym in options:
        return synonym
    misspelled = [c for c in options if _misspelled(key, plain[c])]
    if misspelled:
        return max(
            misspelled,
            key=lambda c: difflib.SequenceMatcher(None, key, plain[c]).ratio(),
        )
    abbreviated = [c for c in options if _abbreviated(key, plain[c])]
    return abbreviated[0] if len(abbreviated) == 1 else None


def describe_value(value: object) -> str:
    """A value as an error message quotes it: its repr when short, else its type."""
    return _show(value)


def closest_name(word: str, candidates: Iterable[str]) -> str | None:
    """The name a did-you-mean suggests for ``word``, or None (see ``_closest``)."""
    return _closest(word, candidates)


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
    if isinstance(value, np.floating) and value.dtype.itemsize < 8:
        # A float32 or float16 0.9 is 0.8999999761581421 as a double; read it as
        # the number it prints as (its shortest repr), which is what was meant.
        return float(str(value))
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


# The functions whose drawing takes layers= (SPEC.md section 2), so their analysis
# function can name it for plot.type= without importing the renderers.
_LAYERED_DRAWINGS: Final = frozenset({"betweenstats", "withinstats"})

# Keywords only the drawing functions (ip.<name>, ip.render.<name>) take, by
# function, so ip.analyze.<name> can point to them without importing the renderers.
# Functions not listed get the common ones. tests/test_validate.py checks these
# against the real signatures.
_COMMON_DRAWING: Final = frozenset(
    {"ax", "palette", "title", "subtitle", "caption", "xlab", "ylab"}
)
_LAYERED_DRAWING: Final = _COMMON_DRAWING | {
    "layers",
    "point_args",
    "boxplot_args",
    "violin_args",
    "centrality_plotting",
}
DRAWING_KEYWORDS: Final[Mapping[str, frozenset[str]]] = {
    "betweenstats": _LAYERED_DRAWING,
    "withinstats": _LAYERED_DRAWING | {"point_path", "centrality_path"},
    "histostats": _COMMON_DRAWING | {"binwidth", "centrality_plotting", "bar_args"},
    "scatterstats": _COMMON_DRAWING
    | {"marginal", "smooth", "point_args", "smooth_args", "marginal_args"},
    "barstats": _COMMON_DRAWING | {"legend_title", "bar_args"},
}
# Keywords only the analysis functions (ip.analyze.<name>, ip.<name>) take, so a
# drawing function ip.render.<name> can point to them.
ANALYSIS_KEYWORDS: Final = frozenset(
    {
        "data",
        "x",
        "y",
        "type",
        "pairwise_display",
        "p_adjust_method",
        "pairwise_alpha",
        "conf_level",
        "digits",
        "tr",
        "bf_prior",
        "bf_message",
        "results_subtitle",
        "alternative",
        "seed",
        "order",
        "subject_id",
        "test_value",
        "label_var",
        "counts",
        "paired",
        "ratio",
        "proportion_test",
    }
)
# What a drawing function given data by keyword was given instead of a result.
_DATA_KEYWORDS: Final = ("data", "x", "y")


def _wrong_kind_of_function(
    arg: str, key: str, fn: str, valid: frozenset[str]
) -> TypeError | None:
    """An analysis keyword given to a drawing function, or a drawing keyword given
    to an analysis function: point to the function that takes it."""
    if "result" in valid and key in ANALYSIS_KEYWORDS:
        columns = "data, x" if fn == "histostats" else "data, x, y"
        return TypeError(
            f"{arg}= is an argument of the analysis, ip.analyze.{fn}({columns}, "
            f"...); render.{fn}() draws a result that already holds it. Pass {key}= "
            f"to ip.analyze.{fn}() or ip.{fn}() and draw that result. Remove {arg}=."
        )
    drawing = DRAWING_KEYWORDS.get(fn, _COMMON_DRAWING)
    if "ax" not in valid and "result" not in valid and key in drawing:
        return TypeError(
            f"{arg}= is a drawing option of ip.{fn}(...) and "
            f"ip.render.{fn}(result, ...); ip.analyze.{fn}() only computes the "
            f"statistics. Pass {key}= when drawing, or remove it."
        )
    return None


def _targeted(
    arg: str,
    key: str,
    value: object,
    fn: str,
    valid: frozenset[str],
    filled: frozenset[str] = frozenset(),
) -> TypeError | NotImplementedError | None:
    """SPEC.md section 2's message for a removed or borrowed keyword, if ``key`` (the
    snake_case spelling of ``arg``) is one of them. ``filled`` names the arguments
    the call already gives."""
    if fn in _NO_LEVEL_FUNCTIONS:
        own = _no_level_targeted(arg, key, value, fn, valid)
        if own is not None:
            return own
    if fn == "barstats":
        bar = _bar_targeted(arg, key, value, valid)
        if bar is not None:
            return bar
    if key == "var_equal":
        if fn in _SUBJECT_FUNCTIONS:
            why = (
                f"{fn}() compares each subject's paired values (the paired t-test, "
                "or the repeated-measures ANOVA with the Greenhouse-Geisser "
                "correction), which has no equal-variance option, so there is "
                "nothing to set."
            )
        else:
            why = (
                "Welch's test is always used (it does not assume equal variances), "
                "so there is nothing to set."
            )
        return TypeError(_removed(arg, "1.0.0", why))
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
        elif fn in _LAYERED_DRAWINGS:
            rest = (
                f"ip.analyze.{fn}() only computes the statistics; choose the layers "
                "when drawing with layers=, a tuple of 'violin', 'box' and 'points': "
                f"ip.{fn}(..., layers=('box', 'points')) or "
                f"ip.render.{fn}(result, layers=('box', 'points'))."
            )
        else:
            rest = (
                f"ip.analyze.{fn}() only computes the statistics. What is drawn is "
                f"chosen when drawing, with {drawing}."
            )
        return TypeError(_removed(arg, "0.12.0", rest))
    if key == "pairwise_comparisons":
        levels = "conditions" if fn in _SUBJECT_FUNCTIONS else "groups"
        if "pairwise_display" in valid:
            rest = (
                "pairwise comparisons are computed whenever there are three or more "
                f"{levels}; use pairwise_display='none' to hide them or "
                "pairwise_display='all' to show every pair."
            )
        elif "result" in valid and fn in _PAIRWISE_FUNCTIONS:
            rest = (
                f"the pairwise comparisons are part of the result (result.pairwise), "
                f"computed whenever there are three or more {levels}; which pairs "
                f"render.{fn}() draws is set by the analysis' pairwise_display=: "
                f"pass pairwise_display='none' or 'all' to ip.analyze.{fn}() or "
                f"ip.{fn}() and draw that result."
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
        if "subject_id" in valid and "subject_id" in filled:
            return TypeError(
                f"{arg}= is not an argument of {fn}(), and subject_id= (the column "
                f"that identifies each subject) is already given. Remove {arg}=."
            )
        if "subject_id" in valid:
            example = _show(value) if isinstance(value, str) else "'subject'"
            return TypeError(
                f"{arg}= is not an argument of {fn}(); use subject_id={example}, the "
                "column that identifies each subject."
            )
        if fn in _SUBJECT_FUNCTIONS:  # its drawing function: the result holds it
            return TypeError(
                f"{arg}= is not an argument of render.{fn}(): the subject column is "
                f"an argument of the analysis, subject_id=, and the result already "
                f"holds it (result.subject_id). Pass subject_id= to "
                f"ip.analyze.{fn}(data, x, y, subject_id=...) or ip.{fn}(...) and "
                f"draw that result. Remove {arg}=."
            )
        return TypeError(
            f"{fn}() compares independent groups and takes no {arg}=. For repeated "
            "measures of the same subjects use "
            "withinstats(data, x, y, subject_id=...); otherwise remove "
            f"{arg}=."
        )
    if key == "paired" and fn in _SUBJECT_FUNCTIONS:
        if "result" in valid:
            return TypeError(
                f"{arg}= is not needed: render.{fn}() draws a result whose "
                f"measurements are already paired by subject_id. Remove {arg}=."
            )
        return TypeError(
            f"{arg}= is not needed: {fn}() always pairs the measurements of each "
            f"subject, matched by subject_id. Remove {arg}=."
        )
    common = _common_name(arg, key, value, fn, valid)
    if common is not None:
        return common
    reason = _NO_EQUIVALENT.get(key)
    if reason is None and fn in _SUBJECT_FUNCTIONS:
        reason = _NO_EQUIVALENT_WITHIN.get(key)
    if reason is not None:
        return TypeError(
            f"{arg}= is a ggstatsplot argument that inferplot does not have: "
            f"{reason} Remove {arg}=."
        )
    return None


def _instead(arg: str, fn: str, valid: frozenset[str], target: str, what: str) -> str:
    """The message for ``arg``, a common name for what ``target`` does: ``what``
    explains it, and the message says where ``target`` belongs when ``fn`` does not
    take it (the analysis, or the drawing)."""
    if target in valid:
        return f"{arg}= is not an argument of {fn}(); {what} Remove {arg}=."
    if "result" in valid and target in ANALYSIS_KEYWORDS:
        return (
            f"{arg}= is not an argument of render.{fn}(); {what} {target}= is an "
            f"argument of the analysis: pass it to ip.analyze.{fn}() or ip.{fn}() "
            f"and draw that result. Remove {arg}=."
        )
    if target in DRAWING_KEYWORDS.get(fn, _COMMON_DRAWING):
        return (
            f"{arg}= is not an argument of {fn}(); {what} {target}= is a drawing "
            f"option of ip.{fn}(...) and ip.render.{fn}(result, ...). "
            f"Remove {arg}=."
        )
    return (
        f"{arg}= is not an argument of {fn}(), which has no {target}=. Remove {arg}=."
    )


# Common names for alternative=, order= and the labels (keys in snake_case).
_DIRECTION_NAMES: Final = frozenset(
    {"tail", "tails", "sided", "sides", "alternative_hypothesis", "one_sided"}
)
_ORDER_NAMES: Final = frozenset({"levels", "categories", "x_order", "level_order"})
_LABEL_NAMES: Final = frozenset({"label", "labels"})
_PATH_NAMES: Final = frozenset({"point_path", "centrality_path"})


def _common_name(
    arg: str, key: str, value: object, fn: str, valid: frozenset[str]
) -> TypeError | None:
    """The message for a common name of something inferplot spells otherwise, or
    has not (``tail``, ``alpha``, ``levels``, ``legend``, ...), whose nearest valid
    name would be another argument (STATUS U3)."""
    if key in _DIRECTION_NAMES:
        what = (
            "the direction of the test is set by alternative=: 'two-sided' (the "
            "default), 'greater' (the first level of x is greater than the second) "
            "or 'less' (it is less). The first level is the first in order= when "
            "given, else the first declared category, else the first in sorted "
            "order."
        )
        return TypeError(_instead(arg, fn, valid, "alternative", what))
    if key == "alpha":
        transparency = (
            "For transparency, use point_args={'alpha': 0.5}, boxplot_args or "
            "violin_args."
        )
        if "pairwise_alpha" in valid:
            drawn = f" {transparency}" if "point_args" in valid else ""
            return TypeError(
                f"{arg}= is not an argument of {fn}(): pairwise_alpha= sets the "
                "threshold below which a pair's adjusted p-value counts as "
                "significant, and conf_level= sets the level of the confidence "
                f"intervals (conf_level=0.95 for 95%).{drawn} Remove {arg}= and pass "
                "the one you mean."
            )
        if "result" in valid:
            drawn = f" {transparency}" if "point_args" in valid else ""
            return TypeError(
                f"{arg}= is not an argument of render.{fn}().{drawn} pairwise_alpha= "
                "and conf_level= are arguments of the analysis: pass them to "
                f"ip.analyze.{fn}() or ip.{fn}() and draw that result. Remove {arg}=."
            )
        return None
    if key == "paired" and fn not in _SUBJECT_FUNCTIONS:
        # paired= belongs to barstats' analysis only (McNemar's test); every
        # form of the comparison functions refuses it with this message.
        if "result" in valid:
            return TypeError(
                f"render.{fn}() draws independent groups and takes no {arg}=. For "
                "repeated measures of the same subjects use "
                "withinstats(data, x, y, subject_id=...), or "
                "ip.analyze.withinstats() and ip.render.withinstats(); otherwise "
                f"remove {arg}=."
            )
        return TypeError(
            f"{fn}() compares independent groups and takes no {arg}=. For repeated "
            "measures of the same subjects use "
            f"withinstats(data, x, y, subject_id=...); otherwise remove {arg}=."
        )
    if key in _ORDER_NAMES:
        example = (
            _show(list(cast("Iterable[object]", value)))
            if isinstance(value, list | tuple)
            else "['control', 'treatment']"
        )
        what = (
            "the order of the levels of x is set by order=, a list of every level "
            f"present, for example order={example}."
        )
        return TypeError(_instead(arg, fn, valid, "order", what))
    if key == "legend":
        return TypeError(
            f"{arg}= is not an argument of {fn}(): there is no legend; the colors "
            f"follow the levels of x, which the x axis names. Remove {arg}=."
        )
    if key == "bayes":
        what = (
            "for Bayesian statistics pass type='bayes', and for the Bayes factor "
            "caption under a parametric subtitle pass bf_message=True or False."
        )
        return TypeError(_instead(arg, fn, valid, "type", what))
    if key in _LABEL_NAMES:
        what = (
            "the labels are set by xlab= (the x axis), ylab= (the y axis) and title=."
        )
        return TypeError(_instead(arg, fn, valid, "xlab", what))
    if key in _PATH_NAMES and fn not in _SUBJECT_FUNCTIONS:
        return TypeError(
            f"{arg}= is an argument of withinstats(), which joins each subject's "
            "points (point_path) or the centrality points (centrality_path) across "
            f"conditions; {fn}() compares independent groups and draws no paths. "
            f"Remove {arg}=."
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


# ggwithinstats 1.1.4 arguments without an inferplot equivalent, beyond those of
# ggbetweenstats. Only withinstats() reports them this way, so betweenstats()'s
# messages for these names stay as they were.
_NO_EQUIVALENT_WITHIN: Mapping[str, str] = {
    "centrality_path_args": (
        "the path joining the centrality points is not styled by an argument; "
        "centrality_path=False leaves it out."
    ),
    "point_path_args": (
        "the paths joining each subject's points are not styled by an argument; "
        "point_path=False leaves them out."
    ),
}
# Functions without levels of x: one column (histostats) or a pair of columns
# (scatterstats). Many messages of the comparison functions speak of groups and
# levels; these functions get their own (``_no_level_targeted``).
_NO_LEVEL_FUNCTIONS: Final = frozenset({"histostats", "scatterstats"})
_PAIRWISE_KEYS: Final = frozenset(
    {"pairwise_display", "p_adjust_method", "pairwise_alpha"}
)
_COLOUR_KEYS: Final = frozenset({"color", "colour", "colors", "colours"})


def _drawing_note(fn: str, valid: frozenset[str]) -> str:
    """Where a drawing option of ``fn`` is passed, said from ``fn``'s own form."""
    if "ax" in valid or "result" in valid:
        return f"a drawing option of {fn}()"
    return f"a drawing option of ip.{fn}(...) and ip.render.{fn}(result, ...)"


# Other packages' names of histostats' test_value (R's t.test and wilcox.test,
# SciPy's ttest_1samp), and what each message says of them.
_TEST_VALUE_NAMES: Final[dict[str, str]] = {
    "mu": "R's t.test() and wilcox.test() call it mu",
    "popmean": "SciPy's ttest_1samp() calls it popmean",
}
# pandas' DataFrame.corr(method=...) and SciPy's names of the correlations, and
# the scatterstats type that computes each.
_CORRELATION_METHODS: Final[dict[str, tuple[str, str]]] = {
    "pearson": ("parametric", "Pearson's r"),
    "spearman": ("nonparametric", "Spearman's rho"),
}


def _correlation_method_hint(value: object, analysis: str) -> str:
    """scatterstats' message for method= (pandas' DataFrame.corr and SciPy
    spell the choice of correlation that way): the type that computes it, and for
    Kendall's tau, which ggscatterstats does not have, the rank correlation it
    does have."""
    types = (
        "'parametric' (Pearson's r), 'nonparametric' (Spearman's rho), 'robust' "
        "(the winsorized correlation) or 'bayes' (the Bayes factor of Pearson's r)"
    )
    given = value.strip().lower() if isinstance(value, str) else None
    if given in _CORRELATION_METHODS:
        type_name, coefficient = _CORRELATION_METHODS[given]
        return (
            f"scatterstats() chooses the correlation with type={analysis}; "
            f"{coefficient} is type={type_name!r}. Write type={type_name!r}."
        )
    if given == "kendall":
        return (
            "Kendall's tau is not available: ggscatterstats has no Kendall "
            f"correlation either. scatterstats() chooses the correlation with type="
            f"{analysis}: {types}. For a rank correlation write type='nonparametric' "
            "(Spearman's rho)."
        )
    return f"scatterstats() chooses the correlation with type={analysis}: {types}."


def _no_level_targeted(
    arg: str, key: str, value: object, fn: str, valid: frozenset[str]
) -> TypeError | None:
    """The messages of histostats and scatterstats for keywords whose usual
    message would speak of groups or levels, for ggstatsplot's own arguments of
    gghistostats and ggscatterstats, and for seaborn and matplotlib habits."""
    histo = fn == "histostats"
    drawing = _drawing_note(fn, valid)
    what = (
        "histostats() tests one numeric column, x, against test_value"
        if histo
        else "scatterstats() tests the correlation of two numeric columns, x and y"
    )

    def no(reason: str) -> TypeError:
        return TypeError(
            f"{arg}= is not an argument of {fn}(): {reason} Remove {arg}=."
        )

    # In ip.render.<fn>() an analysis argument is already part of the result.
    analysis = (
        f", an argument of the analysis: pass it to ip.analyze.{fn}() or ip.{fn}() "
        f"(render.{fn}() draws a result that already holds it)"
        if "result" in valid
        else ""
    )
    if key in _TEST_VALUE_NAMES:
        habit = _TEST_VALUE_NAMES[key]
        if not histo:
            return no(
                f"it names the value a one-sample test compares a column with "
                f"({habit}): histostats(data, x, test_value=...) tests one column "
                "against a value; scatterstats() tests whether the correlation of x "
                "and y is 0."
            )
        number = (
            _show(value)
            if isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(value)
            else "0"
        )
        return no(
            f"the value x is tested against is test_value= ({habit}; ggstatsplot "
            f"calls it test.value){analysis}. Write test_value={number}."
        )
    if key == "method" and not histo:
        return no(_correlation_method_hint(value, analysis))

    if key == "y" and histo:
        return no(
            f"{what}. To relate x to another column use scatterstats(data, x, y); "
            "to compare a column across the groups of x use "
            "betweenstats(data, x, y)."
        )
    if key in {"subject", "id", "subject_id"}:
        pairing = "" if histo else ", pairing them by row,"
        return no(
            f"{what}{pairing} and has no subjects. For repeated measures of the "
            "same subjects use withinstats(data, x, y, subject_id=...)."
        )
    if key == "paired":
        if histo:
            return no(
                f"{what}. To test paired measurements, use withinstats(data, x, y, "
                "subject_id=...), or pass a column of the paired differences as x "
                "with test_value=0."
            )
        return no(f"{what}; it always pairs x and y by row.")
    if key in _PAIRWISE_KEYS or key == "pairwise_comparisons":
        if key == "pairwise_comparisons":
            return None  # its removal message says the function has none
        return no(
            f"{what} and has no pairwise comparisons; {arg}= belongs to "
            "betweenstats() and withinstats(), which compare groups pair by pair."
        )
    if key == "order" or key in _ORDER_NAMES:
        return no(f"{what} and has no levels to order.")
    if key in _PATH_NAMES:
        return no(
            "it is an argument of withinstats(), which joins each subject's "
            f"points across conditions; {fn}() draws no paths."
        )
    if key in _DIRECTION_NAMES or (key == "alternative" and not histo):
        if histo:
            return TypeError(
                f"{arg}= is not an argument of histostats(); the direction of the "
                "test is set by alternative=: 'two-sided' (the default), 'greater' "
                "(x tends to be greater than test_value) or 'less'. Remove "
                f"{arg}=."
            )
        return no(
            "correlations are tested two-sided in ggscatterstats and in inferplot, "
            "so there is no direction to choose."
        )
    if key == "test_value" and not histo:
        return no(
            "it is an argument of histostats(), which tests one column against a "
            "value; scatterstats() tests whether the correlation of x and y is 0."
        )
    if key == "label_var" and histo:
        return no(
            "it is an argument of scatterstats(), whose points it labels; a "
            "histogram has no points to label."
        )
    if key == "label_expression":
        where = (
            "and pass label_var='label'"
            if not histo
            else "and pass label_var='label' to scatterstats()"
        )
        return TypeError(
            f"{arg}= is a ggstatsplot argument (a dplyr filter) that inferplot does "
            "not have: inferplot labels every point whose label_var value is "
            "present. To label only some points, set the labels of the others to "
            "missing first, for example data.assign(label=data['model'].where("
            f"(data['mpg'] > 25) | (data['wt'] > 5))), {where}. Remove {arg}=."
        )
    if key in {"normal_curve", "normal_curve_args"}:
        return TypeError(
            f"{arg}= is not an argument of ggstatsplot 1.1.4's gghistostats (its ... "
            "accepts it and draws nothing), and inferplot draws no normal curve. "
            f"Remove {arg}=."
        )
    if key == "bin_args" and histo:
        return no(
            "the bars are styled with bar_args=, matplotlib properties such as "
            f"bar_args={{'facecolor': 'grey', 'alpha': 0.7}}, {drawing}."
        )
    if key == "centrality_line_args" and histo:
        return no(
            "the centrality line is not styled by an argument; "
            "centrality_plotting=False leaves the line and its label out."
        )
    if key == "point_label_args" and not histo:
        return no(
            "the point labels are not styled by an argument; label_var= chooses "
            "the column they come from."
        )
    if key == "smooth_line_args" and not histo:
        return no(
            "inferplot always draws the least-squares line with its conf_level "
            "band (ggstatsplot's default, method 'lm') and has no other smoothing "
            "method such as 'loess'. Style it with smooth_args=, matplotlib "
            f"properties, {drawing}, or leave it out with smooth=False."
        )
    if key in {"xsidehistogram_args", "ysidehistogram_args"} and not histo:
        return no(
            "the marginal histograms are styled together with marginal_args=, "
            f"matplotlib properties, {drawing}; marginal=False leaves them out."
        )
    if key == "bins":
        if histo:
            return no(
                "the width of the bars is set by binwidth=, "
                f"{drawing}; by default it is (max - min) / sqrt(n), as in "
                "ggstatsplot."
            )
        return no("the marginal histograms use 30 bins each, as in ggstatsplot.")
    if key in {"kde", "kde_kws"}:
        return no("it is a seaborn argument; inferplot draws no density curve.")
    if key in {"stat", "density"} and histo:
        return no(
            "the left axis counts the values in each bin and the right axis gives "
            "the same as a percentage of n, as in ggstatsplot; there is no density "
            "scale."
        )
    if key == "fit_reg" and not histo:
        return no(
            f"it is a seaborn argument; smooth=False, {drawing}, leaves out the "
            "line and its band."
        )
    if key in {"x_jitter", "y_jitter", "jitter"} and not histo:
        return no("every point is drawn at its values; there is no jitter.")
    if key == "line_kws" and not histo:
        return no(
            f"it is a seaborn argument; style the line with smooth_args=, {drawing}."
        )
    if key == "scatter_kws" and not histo:
        return no(
            f"it is a seaborn argument; style the points with point_args=, {drawing}."
        )
    if key in _COLOUR_KEYS:
        artists = "bar_args" if histo else "point_args, smooth_args, marginal_args"
        return no(f"colours are set with palette= or {artists}, {drawing}.")
    if key == "alpha":
        artist = "bar_args" if histo else "point_args"
        return no(
            "conf_level= sets the level of the confidence interval (conf_level=0.95 "
            f"for 95%); for transparency use {artist}={{'alpha': 0.5}}, {drawing}."
        )
    if key == "hue":
        return TypeError(
            f"hue= is a seaborn argument; {fn}() has no grouping by colour. "
            "grouping_var (one plot per value of a column) arrives in inferplot "
            f"v0.2; for now call {fn}() once per subset of the data. Remove hue=."
        )
    if key == "legend":
        return no("there is no legend.")
    if key in _LABEL_NAMES and not histo:
        return no(
            "point labels come from a column, label_var='name'; the axis labels and "
            f"the title are xlab=, ylab= and title=, {drawing}."
        )
    if key in {"centrality_point_args", "centrality_label_args", "ggsignif_args"} or (
        key == "centrality_type" and not histo
    ):
        return TypeError(
            f"{arg}= is a ggstatsplot argument that inferplot does not have: it "
            "styles a part of ggbetweenstats' and ggwithinstats' plots that "
            f"{fn}() does not draw. Remove {arg}=."
        )
    if key == "var_equal":
        test = "the one-sample t-test" if histo else "a correlation"
        return TypeError(
            _removed(
                arg,
                "1.0.0",
                f"{test} has no equal-variance option, so there is nothing to set.",
            )
        )
    if key == "effsize_type":
        rest = (
            "Hedges' g is always used (type parametric), so there is nothing to choose."
            if histo
            else "the correlation coefficient is itself the effect size, so there "
            "is nothing to choose."
        )
        return TypeError(_removed(arg, "1.0.0", rest))
    return None


# --- barstats' keywords -------------------------------------------------------------
#
# barstats counts the categories of x within each level of y (ggbarstats). Many
# messages of the comparison functions speak of measurements, groups and
# brackets; barstats gets its own for those keywords, for ggbarstats' and
# ggpiestats' own arguments, and for the habits of seaborn (countplot, histplot),
# pandas (crosstab, value_counts, plot) and matplotlib users.

# Other names of a column of counts (seaborn's and pandas' weights, dplyr's wt).
_COUNTS_NAMES: Final = frozenset(
    {"weights", "weight", "wt", "freq", "frequency", "frequencies", "n_col"}
)
# Other names of the expected shares of the one-way test (R's chisq.test p,
# SciPy's chisquare f_exp).
_RATIO_NAMES: Final = frozenset(
    {"p", "probs", "prob", "probabilities", "proportions", "expected", "f_exp"}
)
# barstats' analysis arguments that ANALYSIS_KEYWORDS does not list (they are
# valid keywords of other functions' drawings or of none).
_BAR_ANALYSIS: Final = frozenset({"label", "digits_perc"})
# Drawing arguments of the comparison functions that barstats does not have.
_OTHER_LAYERS: Final = frozenset(
    {
        "layers",
        "point_args",
        "boxplot_args",
        "violin_args",
        "centrality_plotting",
        "binwidth",
        "marginal",
        "smooth",
    }
)


# Other names of the axis labels and of the legend title (review B9).
_AXIS_LABEL_NAMES: Final = frozenset(
    {"x_label", "y_label", "x_title", "y_title", "xtitle", "ytitle", "x_lab", "y_lab"}
)
_LEGEND_TITLE_NAMES: Final = frozenset(
    {"legend_label", "fill_label", "legend_name", "fill_title", "fill_name"}
)

# What a direction means in barstats (review B8): its tests have no direction.
CONTINGENCY_DIRECTION: Final = (
    "barstats' tests have no direction: the chi-square and McNemar tests compare "
    "all the cells at once, so their p-values are the same whatever is asked. "
    "alternative='greater' or 'less' only makes the effect-size intervals "
    "one-sided ([low, 1] or [0, high]: the main one and the pairs'; the bars' "
    "tests stay two-sided) "
    "and the Fisher test of a pair whose table is 2 x 2; alternative='two-sided' "
    "is the default. barstats has no order=: the levels follow their declared "
    "categories, else their sorted values."
)


def _bar_targeted(
    arg: str, key: str, value: object, valid: frozenset[str]
) -> TypeError | None:
    """barstats' message for ``arg`` (snake_case ``key``), or None."""
    analysis_form = "data" in valid  # ip.analyze.barstats or ip.barstats
    render_form = "result" in valid
    drawing = (
        "a drawing option of barstats()"
        if "ax" in valid
        else "a drawing option of ip.barstats(...) and ip.render.barstats(result, ...)"
    )

    def no(reason: str) -> TypeError:
        return TypeError(
            f"{arg}= is not an argument of barstats(): {reason} Remove {arg}=."
        )

    def analysis_argument(name: str, what: str) -> TypeError:
        if analysis_form:
            return no(what)
        return TypeError(
            f"{arg}= is not an argument of render.barstats(): {what} {name}= is an "
            "argument of the analysis: pass it to ip.analyze.barstats() or "
            f"ip.barstats() and draw that result. Remove {arg}=."
        )

    shown = _show(value) if isinstance(value, str) else None
    if render_form and key in _BAR_ANALYSIS | {"paired"}:
        return TypeError(
            f"{arg}= is an argument of the analysis, ip.analyze.barstats(data, x, "
            f"y, ...); render.barstats() draws a result that already holds it. Pass "
            f"{key}= to ip.analyze.barstats() or ip.barstats() and draw that "
            f"result. Remove {arg}=."
        )
    if key == "perc_k":
        number = _show(value) if isinstance(value, int) else "0"
        return analysis_argument(
            "digits_perc",
            f"perc.k was renamed digits.perc in ggstatsplot 0.12.0; inferplot "
            f"writes digits_perc={number}.",
        )
    if key in _COUNTS_NAMES:
        column = shown or "'<column>'"
        return analysis_argument(
            "counts",
            "a column that says how many observations each row stands for (data "
            f"given as a table) is passed as counts={column}.",
        )
    if key in _RATIO_NAMES:
        return analysis_argument(
            "ratio",
            "the expected shares of the levels of x, in their order and summing to "
            "1, are passed as ratio=, for example ratio=[0.5, 0.25, 0.25]; equal "
            "shares by default.",
        )
    if key in {"correct", "yates", "correction", "continuity_correction"}:
        return no(
            "barstats uses Pearson's chi-square and McNemar's test without a "
            "continuity correction, as ggbarstats does (chisq.test(correct = "
            "FALSE)), so there is nothing to set."
        )
    if key == "simulate_p_value":
        return no(
            "barstats computes the chi-square p-value from its distribution, as "
            "ggbarstats does; there is no simulated p-value."
        )
    if key == "hue":
        column = shown or "'<hue column>'"
        return TypeError(
            "hue= is a seaborn argument. barstats() colours each bar by x: the "
            "levels of x fill the bars and y makes the bars, so seaborn's "
            f"countplot(data, x='<bars>', hue={column}) is barstats(data, "
            f"x={column}, y='<bars>'). Remove hue=."
        )
    if key == "stacked":
        return no(
            "the bars are always stacked to 100%: each bar shows how the levels of "
            "x divide one level of y, as in ggbarstats."
        )
    if key in {"normalize", "stat", "percent", "percentage"}:
        return analysis_argument(
            "label",
            "each bar already shows the shares of 100% of the levels of x; label= "
            "chooses what the cells say: 'percentage' (the default), 'counts' or "
            "'both'.",
        )
    if key == "dodge":
        return no("the bars are always stacked, not side by side, as in ggbarstats.")
    if key in {"orient", "orientation", "horizontal"}:
        return no("the bars are vertical, one for each level of y, as in ggbarstats.")
    if key == "kind":
        return no(
            "it is an argument of pandas' plot(); barstats() always draws a "
            "stacked bar chart of percentages, as ggbarstats."
        )
    if key in {"index", "columns"}:
        role = "x" if key == "index" else "y"
        return no(
            f"pandas.crosstab(index=a, columns=b) puts a in the rows and b in the "
            "columns of the table; barstats(data, x=a, y=b) counts the same table, "
            f"with x as the rows (the fill) and y as the columns (the bars). Pass "
            f"the column as {role}=."
        )
    if key in {"values", "aggfunc"}:
        return analysis_argument(
            "counts",
            "pandas.crosstab(..., values=<counts>, aggfunc='sum') adds a column of "
            "counts in each cell; barstats does the same with counts='<column>'.",
        )
    if key == "margins":
        return no(
            "the totals of the table are each bar's n (result.bars) and n "
            "(result.n_obs); the counts are in result.counts."
        )
    if key == "legend":
        return no(
            "the legend of the levels of x is always drawn; its title is "
            f"legend_title=, {drawing}."
        )
    if key in {"label_args", "sample_size_label_args"}:
        what = (
            "the labels inside the bars are not styled by an argument; label= "
            "chooses what they say ('percentage', 'counts' or 'both') and "
            "digits_perc= the decimals of a percentage."
            if key == "label_args"
            else "the n below each bar is not styled by an argument."
        )
        return TypeError(
            f"{arg}= is a ggstatsplot argument that inferplot does not have: {what} "
            f"Remove {arg}=."
        )
    if key == "label_repel":
        return TypeError(
            f"{arg}= is an argument of ggstatsplot's ggpiestats (labels beside the "
            "slices of a pie); barstats() draws stacked bars with the labels inside, "
            f"so there is nothing to repel. Remove {arg}=."
        )
    if key == "labels":
        return no(
            "label= chooses what the cells say ('percentage', 'counts' or 'both'); "
            f"the axis labels, title and legend title are xlab=, ylab=, title= and "
            f"legend_title=, {drawing}."
        )
    if key in {"xlabel", "ylabel"}:
        # Without this, label= (one edit away) would be the did-you-mean.
        axis, text = key[0], shown or "'...'"
        where = f"Write {axis}lab={text}." if "ax" in valid else f"It is {drawing}."
        return no(
            f"it is matplotlib's name for the {axis} axis label, which inferplot "
            f"calls {axis}lab=. {where}"
        )
    if key in _AXIS_LABEL_NAMES:
        # Without this, label= (a few edits away) would be the did-you-mean.
        axis, text = key[0], shown or "'...'"
        where = f"Write {axis}lab={text}." if "ax" in valid else f"It is {drawing}."
        return no(f"the {axis} axis label is {axis}lab=. {where}")
    if key in _LEGEND_TITLE_NAMES:
        text = shown or "'...'"
        where = f"Write legend_title={text}." if "ax" in valid else f"It is {drawing}."
        return no(
            f"the title of the legend of the levels of x is legend_title=. {where}"
        )
    if key in {"order", "x_order", "level_order", "levels", "categories"}:
        return no(
            "barstats has no order=, as ggbarstats has none: the levels of x and y "
            "follow their declared categories, else their sorted values. To choose "
            "an order, pass the column as a pandas Categorical, for example "
            "data['x'] = pd.Categorical(data['x'], categories=['low', 'mid', "
            "'high']), or a polars Enum."
        )
    if key in _DIRECTION_NAMES:
        return no(CONTINGENCY_DIRECTION)
    if key in {"subject", "id", "subject_id"}:
        return no(
            "with paired=True each row holds one unit's two measurements, x and y "
            "(McNemar's test), so there are no subjects to name. For a numeric "
            "measurement repeated over conditions use withinstats(data, x, y, "
            "subject_id=...)."
        )
    if key in {"test_value", "mu"}:
        return no(
            "without y, barstats tests the shares of the levels of x against "
            "ratio=, equal shares by default; histostats() tests one numeric "
            "column against test_value=."
        )
    if key in {"tr", "trim", "trimming"}:
        return no(
            "barstats has no trimmed methods: a test of counts has no robust "
            "variant (type='robust' is refused)."
        )
    if key in {"bf_prior", "prior"}:
        return no(
            "ggbarstats has no bf.prior: its Bayes factor for contingency tables "
            "(arriving in inferplot 0.2) uses ggstatsplot's fixed prior "
            "concentration of 1."
        )
    if key in {"pairwise_display", "pairwise_alpha", "pairwise_comparisons"}:
        return no(
            "barstats computes Fisher's exact test between every two levels of x "
            "(result.pairwise) but never draws them, as ggbarstats; "
            "p_adjust_method= sets their adjustment."
        )
    if key in {"var_equal", "effsize_type"}:
        rest = (
            "the tests of counts have no equal-variance option, so there is nothing "
            "to set."
            if key == "var_equal"
            else "the effect size follows the design (Cramér's V, Cohen's g or "
            "Pearson's C), so there is nothing to choose."
        )
        return TypeError(_removed(arg, "1.0.0", rest))
    if key in {"method", "test"}:
        return no(
            "the test follows the design, as in ggbarstats: Pearson's chi-square "
            "test of independence with y, McNemar's test with paired=True, and the "
            "goodness-of-fit test against ratio= without y."
        )
    if key in _COLOUR_KEYS:
        return no(
            "colours are set with palette=, one per level of x in their order (for "
            f"example palette=['#0072B2', '#E69F00']), {drawing}; the colour of "
            "the segments is the only key to the legend, so bar_args= styles only "
            "their edges, hatching and transparency."
        )
    if key == "alpha":
        return no(
            "conf_level= sets the level of the confidence intervals (conf_level=0.95 "
            "for 95%); for transparency use bar_args={'alpha': 0.5}, "
            f"{drawing}."
        )
    if key in _OTHER_LAYERS:
        return no(
            "barstats draws stacked bars, not the points, boxes, violins or "
            f"histograms of the other functions; style the bars with bar_args=, "
            f"{drawing}."
        )
    if key == "centrality_type":
        return TypeError(
            f"{arg}= is a ggstatsplot argument that inferplot does not have: "
            f"barstats counts categories and has no centrality. Remove {arg}=."
        )
    return None


def _with_long_recipe(
    problem: TypeError | NotImplementedError,
    arg: str,
    value: object,
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> TypeError | NotImplementedError:
    """barstats given a subject column (``subject_id=`` and its other names):
    the data are very likely long, one row per unit and occasion; the message
    then shows the reshape to one row per unit and the paired call, in the
    frame's own library (review B6). The message as it was where the call
    gives no data, x and y to build it from."""
    if _snake(arg) not in {"subject", "id", "subject_id"} or not isinstance(value, str):
        return problem
    given = dict(zip(("data", "x", "y"), args, strict=False)) | dict(kwargs)
    data, x, y = given.get("data"), given.get("x"), given.get("y")
    if not isinstance(x, str) or not isinstance(y, str):
        return problem
    from inferplot._data import long_paired_recipe, to_frame

    try:
        frame = to_frame(data)
        occasions = [str(v) for v in frame[y].drop_nulls().unique().to_list()]
        recipe = long_paired_recipe(data, x, y, value, sorted(occasions))
    except Exception:  # whatever is wrong with the data is reported later
        return problem
    return TypeError(
        f"{problem} If the data are long, one row per unit and occasion with "
        f"{value!r} naming the unit and {y!r} the occasion, reshape them to one row "
        "per unit with one column per occasion, and compare two occasions with "
        f"McNemar's test: {recipe}."
    )


# Functions whose analysis takes subject_id (their drawing functions share the name).
_SUBJECT_FUNCTIONS: Final = frozenset({"withinstats"})
# Functions whose analysis computes pairwise comparisons.
_PAIRWISE_FUNCTIONS: Final = frozenset({"betweenstats", "withinstats"})


def subject_id_required(fn: str, *, given: str | None = None) -> TypeError:
    """The error for a call to ``fn`` without a subject column (SPEC.md section 2):
    measurements are never paired by row order. ``given`` is the value passed, if
    any, as the message shows it (``subject_id=None``)."""
    start = (
        f"{fn}() needs subject_id="
        if given is None
        else f"subject_id={given} is not a column name: {fn}() needs subject_id="
    )
    return TypeError(
        f"{start}, the column that identifies the subject of each row, for example "
        f"{fn}(data, x, y, subject_id='subject'). Measurements are paired by subject "
        "and never by row order, which silently pairs the wrong values once rows "
        "are sorted, filtered or missing. Put the data in long format: one row per "
        "subject and level of x, with a column naming the subject."
    )


def _keyword_problem(
    arg: str,
    value: object,
    fn: str,
    valid: frozenset[str],
    order: Sequence[str],
    filled: frozenset[str],
) -> TypeError | NotImplementedError:
    """The error for keyword ``arg``, which ``fn`` does not take. ``filled`` names
    the arguments the call already gives; they are never suggested."""
    key = _snake(arg)
    if key in filled:
        return TypeError(
            f"{fn}() got an unexpected keyword argument {arg!r}, and {key} is already "
            f"given. Remove {arg}=."
        )
    if key in valid:
        return TypeError(
            f"{fn}() has no argument {arg!r}: inferplot spells every argument in "
            f"lowercase snake_case. Write {key}={_show(value)}."
        )
    targeted = _targeted(arg, key, value, fn, valid, filled)
    if targeted is not None:
        return targeted
    elsewhere = _wrong_kind_of_function(arg, key, fn, valid)
    if elsewhere is not None:
        return elsewhere
    close = _closest(key, [name for name in order if name not in filled])
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
    fn: str,
    positional: Sequence[str],
    extra: Sequence[object],
    valid: frozenset[str],
    *,
    subject_given: bool,
) -> TypeError:
    names = ", ".join(positional)
    given = len(positional) + len(extra)
    shown = ", ".join(_show(v) for v in extra)
    hint = ""
    first = extra[0]
    other = None  # the keyword a string value is a valid value of
    if isinstance(first, str):
        if first in TYPES and "type" in valid:
            other = "type"
        elif first in PAIRWISE_DISPLAYS and "pairwise_display" in valid:
            other = "pairwise_display"
    if "subject_id" in valid and not subject_given:
        # The required subject column is missing, so it comes first whatever the
        # value: a column may well be called 's' or 'np'.
        if isinstance(first, str):
            hint = (
                f" If {first!r} is the column that identifies each subject, write "
                f"{fn}(..., subject_id={first!r})"
            )
            hint += (
                f"; if it is {other}, write {fn}(..., {other}={first!r})."
                if other is not None
                else "."
            )
        else:
            own = getattr(first, "name", None)
            column = own if isinstance(own, str) else "subject"
            hint = (
                f" {fn}() takes the column that identifies each subject by its "
                f"name: write {fn}(..., subject_id={column!r})."
            )
    elif other is not None:
        hint = f" For example {fn}(..., {other}={first!r})."
    elif fn == "barstats" and "counts" in valid and isinstance(first, str):
        # A column name after x and y: very likely the column of counts.
        hint = (
            f" If {first!r} is the column of counts of a table (how many "
            f"observations each row stands for), write barstats(..., "
            f"counts={first!r})."
        )
    elif fn == "histostats" and positional == ("data", "x") and isinstance(first, str):
        # A column name after x: very likely a y for another function.
        hint = (
            " histostats() tests one column against test_value; to relate x to "
            f"{first!r} use scatterstats(data, x, {first!r}), and to compare "
            f"{first!r} across the groups of x use betweenstats(data, x, {first!r})."
        )
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


def _not_a_result(fn: str, got: str) -> TypeError:
    columns = "data, x" if fn == "histostats" else "data, x, y"
    return TypeError(
        f"render.{fn}() draws the result of ip.analyze.{fn}({columns}); got "
        f"{got}. Compute it first with "
        f"result = ip.analyze.{fn}({columns}, ...) and pass result, or use "
        f"ip.{fn}({columns}, ...) to analyze and draw in one call."
    )


def _render_named(
    error: TypeError | NotImplementedError, name: str
) -> TypeError | NotImplementedError:
    """A drawing function's error that starts with its bare name ("histostats()
    got ..."), named as it is called: "render.histostats() got ..."."""
    text = str(error)
    if not text.startswith(f"{name}("):
        return error
    return type(error)(f"render.{text}")


def checked(fn: Callable[P, R]) -> Callable[P, R]:
    """Check keyword names and the positional count of every call to ``fn``.

    Unknown keywords raise ``TypeError`` (``NotImplementedError`` for
    ``grouping_var``) with SPEC.md section 2's targeted message, the snake_case
    spelling of an R-style name, or the closest valid keyword. When several
    keywords are wrong, the first one in call order is reported. A drawing
    function (first parameter ``result``) given something other than a
    ``StatsResult`` says so before the positional count is checked. A function with
    a required ``subject_id`` called without one gets ``subject_id_required``'s
    message instead of Python's. Values are not checked here otherwise; the value
    validators below do that.
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
    required = tuple(
        p.name
        for p in signature.parameters.values()
        if p.kind is kinds.KEYWORD_ONLY and p.default is p.empty
    )
    name = fn.__name__
    draws_result = bool(positional) and positional[0] == "result"

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # A drawing function given data, positionally or as data=, x= or y=, says
        # it draws a result before anything else (x= is not a typo of ax=).
        if draws_result:
            from inferplot.result import StatsResult

            if args or "result" in kwargs:
                given = args[0] if args else kwargs["result"]
                if not isinstance(given, StatsResult):
                    raise _not_a_result(name, _type_name(given))
            elif "data" in kwargs:
                raise _not_a_result(name, _type_name(kwargs["data"]))
            else:
                data_like = [k for k in _DATA_KEYWORDS if k in kwargs]
                if data_like:
                    shown = ", ".join(f"{k}={_show(kwargs[k])}" for k in data_like)
                    raise _not_a_result(name, f"{shown} and no result")
        filled = frozenset(positional[: len(args)]) | (valid & kwargs.keys())
        for arg, value in kwargs.items():
            if arg not in valid:
                problem = _keyword_problem(arg, value, name, valid, order, filled)
                if name == "barstats" and not draws_result:
                    problem = _with_long_recipe(problem, arg, value, args, kwargs)
                raise _render_named(problem, name) if draws_result else problem
        if len(args) > len(positional):
            problem = _positional_problem(
                name,
                positional,
                args[len(positional) :],
                valid,
                subject_given="subject_id" in kwargs,
            )
            raise _render_named(problem, name) if draws_result else problem
        if "subject_id" in required and "subject_id" not in kwargs:
            raise subject_id_required(name)
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
        hint = hints.get(_plain(text))
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


# Other names of R's p.adjust() methods (keys as ``_plain`` spells them): the
# procedure's name and R's spelling.
_BH: Final = "The Benjamini-Hochberg procedure (the false discovery rate)"
_BY: Final = "The Benjamini-Yekutieli procedure"
_HOLM: Final = "The Holm-Bonferroni procedure"
_ADJUSTMENT_NAMES: Final[Mapping[str, tuple[str, str]]] = {
    "benjamini_hochberg": (_BH, "BH"),
    "fdr_bh": (_BH, "BH"),
    "b_h": (_BH, "BH"),
    "benjamini_yekutieli": (_BY, "BY"),
    "fdr_by": (_BY, "BY"),
    "b_y": (_BY, "BY"),
    "holm_bonferroni": (_HOLM, "holm"),
    "bonferroni_holm": (_HOLM, "holm"),
}
# Adjustments R's p.adjust() does not have.
_ADJUSTMENTS_NOT_AVAILABLE: Final[Mapping[str, str]] = {
    "sidak": "Sidak's correction",
    "holm_sidak": "The Holm-Sidak procedure",
    "tukey": "Tukey's method",
    "tukey_hsd": "Tukey's method",
}


def check_p_adjust_method(value: object, name: str = "p_adjust_method") -> str:
    """One of R's ``p.adjust()`` method names, spelled exactly as in R. Other names
    of the same procedures are told R's spelling; procedures R does not have are
    said to be unavailable."""
    methods = {m: m for m in P_ADJUST_METHODS}
    hints = {
        key: f" {procedure} is R's {method!r}: write {name}={method!r}."
        for key, (procedure, method) in _ADJUSTMENT_NAMES.items()
    }
    for key, procedure in _ADJUSTMENTS_NOT_AVAILABLE.items():
        hints[key] = (
            f" {procedure} is not one of R's p.adjust() methods, which inferplot "
            "follows, so it is not available; choose one of those listed."
        )
    return _choice(value, name, methods, hints)


def direction_note(name: str = "alternative") -> str:
    """What 'greater' and 'less' compare, as the messages state it (SPEC.md
    section 2): the first level of x against the second."""
    return (
        f"{name}='greater' tests whether the first level of x is greater than the "
        f"second, and {name}='less' whether it is less. The first level is the "
        "first in order= when given, else the first declared category, else the "
        "first in sorted order."
    )


_ONE_SIDED: Final = ("one_sided", "onesided", "one_side", "one_tailed", "onetailed")
_TWO_TAILED: Final = ("two_tailed", "twotailed")


def check_alternative(
    value: object, name: str = "alternative", *, directionless: bool = False
) -> str:
    """'two-sided', 'less' or 'greater'. R's 'two.sided' and 'two-tailed' are told
    the spelling; 'one-sided' is told that the direction must be given, or with
    ``directionless`` (barstats) what a direction does there."""
    one_sided = (
        f" {CONTINGENCY_DIRECTION}"
        if directionless
        else f" A one-sided test needs its direction: {direction_note(name)}"
    )
    two_tailed = f" A two-tailed test is {name}='two-sided': write {name}='two-sided'."
    hints = {
        "two.sided": f" R spells it 'two.sided'; inferplot writes {name}='two-sided'.",
        **dict.fromkeys(_ONE_SIDED, one_sided),
        **dict.fromkeys(_TWO_TAILED, two_tailed),
    }
    return _choice(value, name, {a: a for a in ALTERNATIVES}, hints)


# Confidence levels written as percentages that the message may divide by 100.
_COMMON_PERCENTAGES: Final = frozenset({90.0, 95.0, 99.0, 99.9})


def check_conf_level(value: object, name: str = "conf_level") -> float:
    """A confidence level strictly between 0 and 1 (bool rejected)."""
    number = _real(value, name, "0.95")
    if not 0.0 < number < 1.0:
        hint = ""
        if number in _COMMON_PERCENTAGES:
            # Only the levels in common use: 5 or 1.96 would become 0.05 or 0.0196,
            # a different analysis, so they get the rule alone.
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


def check_finite_number(value: object, name: str, *, example: str = "0") -> float:
    """A finite real number, such as ``test_value`` (bool rejected; a NumPy
    float32 is read as the number it prints as)."""
    number = _real(value, name, example)
    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be a finite number, for example {name}={example}; got "
            f"{_show(value)}."
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


# A double holds about 15 significant decimal digits; more decimals print noise.
_MAX_DIGITS: Final = 15


def check_digits(value: object, name: str = "digits") -> int:
    """A whole number of decimal places, 0 to 15 (bool rejected)."""
    number = _integer(value, name, "2")
    if number < 0:
        raise ValueError(
            f"{name} is a number of decimal places and must be >= 0, for example "
            f"{name}=2; got {number}."
        )
    if number > _MAX_DIGITS:
        raise ValueError(
            f"{name} is a number of decimal places and must be at most "
            f"{_MAX_DIGITS}, the most a double-precision number holds meaningfully, "
            f"for example {name}=2; got {number}."
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
