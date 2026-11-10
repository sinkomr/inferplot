"""Drawing into a given Axes: refused kinds of Axes, shared axes, other panels."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from inferplot.render import betweenstats
from inferplot.result import StatsResult
from tests.render._results import frame_result


def _three() -> StatsResult:
    rng = np.random.default_rng(1)
    groups = [rng.normal(m, 1, 20) for m in (5, 6, 7.5)]
    return frame_result(["a", "b", "c"], groups, pairwise_display="all")


@pytest.mark.parametrize("projection", ["polar", "3d", "aitoff"])
def test_an_axes_that_is_not_rectangular_2d_is_refused(projection: str) -> None:
    figure = Figure()
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(projection=projection)
    before = list(ax.get_children())
    with pytest.raises(ValueError, match="rectangular") as info:
        betweenstats(_three(), ax=ax)
    assert projection in str(info.value)
    assert list(ax.get_children()) == before
