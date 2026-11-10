"""inferplot: statistical plots with the right test computed and its results annotated.

``ip.betweenstats(df, "group", "score")`` analyzes and draws; ``ip.analyze.<name>``
computes the :class:`StatsResult` (every number and text) without importing
matplotlib; ``ip.render.<name>(result)`` draws an existing result. ``import
inferplot`` imports no matplotlib: ``render`` and ``StatsFigure`` load it on first
use.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from inferplot import analyze
from inferplot._api import barstats, betweenstats, histostats, scatterstats, withinstats
from inferplot.check import Issue
from inferplot.result import StatsResult

if TYPE_CHECKING:
    from inferplot import render
    from inferplot.render import StatsFigure

__version__ = "0.1.0.dev0"

__all__ = [
    "Issue",
    "StatsFigure",
    "StatsResult",
    "__version__",
    "analyze",
    "barstats",
    "betweenstats",
    "histostats",
    "render",
    "scatterstats",
    "withinstats",
]

# Names that import matplotlib, loaded on first access (PEP 562).
_LAZY = {"render": "inferplot.render", "StatsFigure": "inferplot.render.figure"}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'inferplot' has no attribute {name!r}")
    loaded = importlib.import_module(module)
    value = loaded if name == "render" else getattr(loaded, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
