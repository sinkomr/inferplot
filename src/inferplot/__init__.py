"""inferplot: statistical plots with the right test computed and its results annotated.

``inferplot.analyze.<name>`` computes a :class:`StatsResult` (every number and text of
a figure) without importing matplotlib. The figure functions arrive with the renderer;
see ``docs/STATUS.md``.
"""

from inferplot import analyze
from inferplot.check import Issue
from inferplot.result import StatsResult

__version__ = "0.1.0.dev0"

__all__ = ["Issue", "StatsResult", "__version__", "analyze"]
