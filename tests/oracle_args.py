"""Translate a fixture's R arguments into an inferplot call (SPEC.md section 9).

One table maps every ggstatsplot argument the oracle uses to its inferplot keyword:
R's dotted names become snake_case, and JSON already holds TRUE, FALSE and NULL as
Python booleans and None. An argument without a mapping raises, so a new fixture
argument fails loudly until it is added here.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

FUNCTIONS: Mapping[str, str] = MappingProxyType(
    {
        "ggbetweenstats": "betweenstats",
        "ggwithinstats": "withinstats",
        "gghistostats": "histostats",
        "ggscatterstats": "scatterstats",
        "ggbarstats": "barstats",
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
