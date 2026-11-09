"""Draw results with matplotlib (object-oriented API only; never pyplot).

``inferplot.render.betweenstats(result)`` draws a result of
``inferplot.analyze.betweenstats`` and returns a :class:`StatsFigure`.
"""

from __future__ import annotations

from inferplot.render.betweenstats import betweenstats
from inferplot.render.figure import StatsFigure

__all__ = ["StatsFigure", "betweenstats"]
