"""Draw results with matplotlib (object-oriented API only; never pyplot).

``inferplot.render.betweenstats(result)`` draws a result of
``inferplot.analyze.betweenstats`` and returns a :class:`StatsFigure`;
``inferplot.render.withinstats(result)`` does the same for
``inferplot.analyze.withinstats``. ``inferplot.render.histostats(result)`` and
``inferplot.render.scatterstats(result)`` check their arguments; their drawing is
not implemented yet.
"""

from __future__ import annotations

from inferplot.render.betweenstats import betweenstats
from inferplot.render.figure import StatsFigure
from inferplot.render.histostats import histostats
from inferplot.render.scatterstats import scatterstats
from inferplot.render.withinstats import withinstats

__all__ = ["StatsFigure", "betweenstats", "histostats", "scatterstats", "withinstats"]
