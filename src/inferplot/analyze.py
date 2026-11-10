"""``ip.analyze``: compute a result without drawing anything.

``ip.analyze.betweenstats(df, "group", "score")`` returns the same
:class:`~inferplot.result.StatsResult` that the figure functions annotate, so the
numbers and texts are available without matplotlib. This module, and everything it
imports, never imports matplotlib.
"""

from __future__ import annotations

from inferplot.analyses import (
    barstats,
    betweenstats,
    histostats,
    scatterstats,
    withinstats,
)

__all__ = ["barstats", "betweenstats", "histostats", "scatterstats", "withinstats"]
