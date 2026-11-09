"""Image comparisons of a few figures against stored baselines (pytest-mpl).

The comparison runs only with ``--mpl``, in the Linux CI job that runs the newest
Python; the baselines in ``tests/render/baseline`` are generated in that same
environment by the "Image baselines" workflow, because text renders differently on
other systems. Without ``--mpl`` each test still draws its figure and checks it.

Every figure is deterministic: the data come from the oracle datasets or a closed
form, and jitter and subsampling use the default seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from scipy.stats import norm

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.render.betweenstats import MAX_POINTS, SUBSAMPLE_SIZE

DATASETS = Path(__file__).resolve().parents[2] / "oracle" / "datasets"
FIGURE_CODES = frozenset({"label_overlap", "text_clipped"})

image = pytest.mark.mpl_image_compare


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATASETS / f"{name}.csv", keep_default_na=False, na_values=[""])


def _clean(fig: StatsFigure) -> Figure:
    """The matplotlib Figure, after checking that no text is clipped or overlaps."""
    issues = [i for i in fig.check() if i.code in FIGURE_CODES]
    assert not issues, issues
    figure = fig.figure
    assert isinstance(figure, Figure)
    return figure


@image
def test_toothgrowth_two_groups() -> Figure:
    """Welch's t test of tooth length by supplement, with the Bayes factor caption."""
    fig = ip.betweenstats(_read("ToothGrowth"), "supp", "len", order=["OJ", "VC"])
    caption = fig.result.text.caption
    assert caption is not None
    assert caption.startswith("log_e(BF_01)")
    return _clean(fig)


@image
def test_plantgrowth_all_pairs() -> Figure:
    """Welch's ANOVA of plant weight, with all three Games-Howell pairs bracketed."""
    fig = ip.betweenstats(
        _read("PlantGrowth"),
        "group",
        "weight",
        order=["ctrl", "trt1", "trt2"],
        pairwise_display="all",
    )
    assert len(fig.brackets) == 3
    return _clean(fig)


@image
def test_insectsprays_all_pairs() -> Figure:
    """Six sprays with all 15 pairs bracketed: the tallest bracket stack."""
    fig = ip.betweenstats(
        _read("InsectSprays"),
        "spray",
        "count",
        order=list("ABCDEF"),
        pairwise_display="all",
    )
    assert len(fig.brackets) == 15
    return _clean(fig)


def _many_values() -> pd.DataFrame:
    """Three groups of 2,200 normal quantiles: 6,600 values, no random numbers."""
    n = 2_200
    quantiles = norm.ppf((np.arange(n) + 0.5) / n)
    # Interleave the quantiles so that the values are not sorted within a group.
    order = np.argsort((np.arange(n) * 7_919) % n)
    # Means away from zero: a mean of 0 would print as rounding noise (about 1e-17).
    groups = {"a": 1.0, "b": 1.3, "c": 1.5}
    return pd.DataFrame(
        {
            "arm": np.repeat(list(groups), n),
            "score": np.concatenate([mu + quantiles[order] for mu in groups.values()]),
        }
    )


@image
def test_subsampled_points() -> Figure:
    """6,600 values: violins and boxes use all of them, the points a 2,000 subsample."""
    data = _many_values()
    assert len(data) > MAX_POINTS
    fig = ip.betweenstats(data, "arm", "score", order=["a", "b", "c"])
    assert fig.subsample is not None
    drawn = sum(
        np.asarray(points.get_offsets()).shape[0]
        for points in fig.figure.findobj(PathCollection)
        if str(points.get_gid()).startswith("points-")
    )
    assert drawn == SUBSAMPLE_SIZE
    return _clean(fig)


@image
def test_drawn_into_a_given_axes() -> Figure:
    """A figure of the user's own, with their title, drawn into with ax=."""
    figure = Figure(figsize=(7.5, 7.0), dpi=100)
    FigureCanvasAgg(figure)
    figure.suptitle("Plant growth experiment, 2026", fontweight="bold")
    ax = figure.add_subplot()
    fig = ip.betweenstats(
        _read("PlantGrowth"), "group", "weight", order=["ctrl", "trt1", "trt2"], ax=ax
    )
    assert fig.figure is figure
    assert figure.get_size_inches().tolist() == [7.5, 7.0]
    return _clean(fig)
