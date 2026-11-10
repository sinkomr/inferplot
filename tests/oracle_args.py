"""Translate a fixture's R arguments into an inferplot call (SPEC.md section 9).

One table maps every ggstatsplot argument the oracle uses to its inferplot keyword:
R's dotted names become snake_case, and JSON already holds TRUE, FALSE and NULL as
Python booleans and None. An argument without a mapping raises, so a new fixture
argument fails loudly until it is added here.

``split`` separates the keywords of the analysis (``ip.analyze.<name>``) from those
of the drawing (``ip.render.<name>``), and handles the two ggscatterstats arguments
that have no inferplot keyword: ``label.expression`` (a dplyr filter of the points
to label; inferplot labels the points whose ``label_var`` is present, so the test
sets the others' labels to missing, ``apply_label_filter``) and
``smooth.line.args`` (ggplot2's smoothing settings; kept in ``unsupported`` for
the renderer's tests to decide on).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from inferplot._validate import DRAWING_KEYWORDS

FUNCTIONS: Mapping[str, str] = MappingProxyType(
    {
        "ggbetweenstats": "betweenstats",
        "ggwithinstats": "withinstats",
        "gghistostats": "histostats",
        "ggscatterstats": "scatterstats",
        "ggbarstats": "barstats",
        # ggpiestats computes the same statistics as ggbarstats (both call
        # statsExpressions::contingency_table()); the oracle records its one-way
        # cases next to ggbarstats' as a check that they agree.
        "ggpiestats": "barstats",
    }
)

# Column arguments that inferplot takes positionally after `data`, in order.
POSITIONAL: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ggbetweenstats": ("x", "y"),
        "ggwithinstats": ("x", "y"),
        "gghistostats": ("x",),
        "ggscatterstats": ("x", "y"),
        "ggbarstats": ("x", "y"),
        "ggpiestats": ("x", "y"),
    }
)

KEYWORDS: Mapping[str, str] = MappingProxyType(
    {
        "type": "type",
        "conf.level": "conf_level",
        "digits": "digits",
        "tr": "tr",
        "bf.prior": "bf_prior",
        "bf.message": "bf_message",
        "results.subtitle": "results_subtitle",
        "alternative": "alternative",
        "pairwise.display": "pairwise_display",
        "p.adjust.method": "p_adjust_method",
        "centrality.plotting": "centrality_plotting",
        "subject.id": "subject_id",
        "point.path": "point_path",
        "centrality.path": "centrality_path",
        "test.value": "test_value",
        "binwidth": "binwidth",
        "marginal": "marginal",
        "label.var": "label_var",
        "counts": "counts",
        "paired": "paired",
        "ratio": "ratio",
        "proportion.test": "proportion_test",
        "label": "label",
        "digits.perc": "digits_perc",
    }
)

# Argument values spelled differently in inferplot. A vector argument (a JSON array,
# such as ``ratio``) is translated element by element.
VALUES: Mapping[str, Mapping[Any, Any]] = MappingProxyType(
    {"alternative": MappingProxyType({"two.sided": "two-sided"})}
)


@dataclass(frozen=True)
class InferplotCall:
    """``function(data, *args, **kwargs)`` in inferplot terms."""

    function: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any] = field(default_factory=dict)


def translate(r_function: str, r_args: Mapping[str, Any]) -> InferplotCall:
    """Map one fixture's R function and arguments to an inferplot call."""
    if r_function not in FUNCTIONS:
        raise KeyError(f"no inferplot function for {r_function!r}; add it to FUNCTIONS")
    positional = POSITIONAL[r_function]
    unknown = sorted(set(r_args) - set(positional) - set(KEYWORDS))
    if unknown:
        raise KeyError(
            f"no inferplot keyword for R argument(s) {unknown}; add them to KEYWORDS"
        )
    args = tuple(r_args[name] for name in positional if name in r_args)
    kwargs: dict[str, Any] = {}
    for r_name, value in r_args.items():
        if r_name in positional:
            continue
        kwargs[KEYWORDS[r_name]] = _respell(VALUES.get(r_name, {}), value)
    return InferplotCall(function=FUNCTIONS[r_function], args=args, kwargs=kwargs)


def _respell(spellings: Mapping[Any, Any], value: Any) -> Any:
    """``value`` in inferplot's spelling; lists element by element."""
    if isinstance(value, list):
        return [_respell(spellings, item) for item in value]
    if isinstance(value, Hashable):
        return spellings.get(value, value)
    return value


# ggscatterstats arguments without an inferplot keyword (see the module docstring).
LABEL_FILTER = "label.expression"
UNSUPPORTED = frozenset({"smooth.line.args"})


@dataclass(frozen=True)
class SplitCall:
    """A fixture's call as ``ip.analyze.<function>(data, *args, **analysis)`` and
    ``ip.render.<function>(result, **drawing)``. ``label_filter`` is the R filter of
    the points to label (``label.expression``), ``unsupported`` the R arguments
    inferplot has no keyword for."""

    function: str
    args: tuple[Any, ...]
    analysis: Mapping[str, Any]
    drawing: Mapping[str, Any]
    label_filter: str | None = None
    unsupported: Mapping[str, Any] = field(default_factory=dict)


def split(r_function: str, r_args: Mapping[str, Any]) -> SplitCall:
    """``translate`` with the analysis and drawing keywords apart."""
    unsupported = {k: v for k, v in r_args.items() if k in UNSUPPORTED}
    label_filter = r_args.get(LABEL_FILTER)
    rest = {
        k: v for k, v in r_args.items() if k not in UNSUPPORTED and k != LABEL_FILTER
    }
    call = translate(r_function, rest)
    drawing_names = DRAWING_KEYWORDS[call.function]
    return SplitCall(
        function=call.function,
        args=call.args,
        analysis={k: v for k, v in call.kwargs.items() if k not in drawing_names},
        drawing={k: v for k, v in call.kwargs.items() if k in drawing_names},
        label_filter=None if label_filter is None else str(label_filter),
        unsupported=unsupported,
    )


def apply_label_filter(frame: Any, label_var: str, expression: str) -> Any:
    """A pandas ``frame`` whose ``label_var`` is missing where the R filter
    ``expression`` is not true: inferplot's way to label only some points.
    ``pandas.eval`` reads the fixtures' expressions (``mpg > 25 | wt > 5``) as R
    does."""
    keep = frame.eval(expression)
    return frame.assign(**{label_var: frame[label_var].where(keep)})
