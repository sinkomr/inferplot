"""Analyses: data in, :class:`~inferplot.result.StatsResult` out (no plotting).

Each module holds one public analysis, exposed as ``inferplot.analyze.<name>``.
Nothing here imports matplotlib.
"""

from __future__ import annotations

from inferplot.analyses.betweenstats import betweenstats

__all__ = ["betweenstats"]
