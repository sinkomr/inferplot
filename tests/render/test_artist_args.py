"""Artist dicts that would move or hide the data, and palette entries."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from inferplot.render import betweenstats, withinstats
from inferplot.result import StatsResult
from tests.render._results import frame_result
from tests.render._within import two_conditions


def _three() -> StatsResult:
    rng = np.random.default_rng(1)
    groups = [rng.normal(m, 1, 20) for m in (5, 6, 7.5)]
    return frame_result(["a", "b", "c"], groups)


@pytest.mark.parametrize(
    ("name", "key", "value", "why"),
    [
        ("point_args", "offsets", np.zeros((3, 2)), "positions"),
        ("point_args", "paths", [], "positions"),
        ("point_args", "transform", None, "coordinates"),
        ("point_args", "offset_transform", None, "coordinates"),
        ("point_args", "animated", True, "out of every"),
        ("point_args", "figure", None, "detach"),
        ("point_args", "clip_path", None, "clip"),
        ("point_args", "clip_box", None, "clip"),
        ("boxplot_args", "path", None, "positions"),
        ("boxplot_args", "transform", None, "coordinates"),
        ("violin_args", "xy", np.zeros((3, 2)), "positions"),
        ("violin_args", "axes", None, "detach"),
        ("violin_args", "data", None, "positions"),
        ("point_args", "xdata", None, "positions"),
        ("point_args", "verts", None, "positions"),
        ("point_args", "segments", None, "positions"),
    ],
)
def test_properties_that_move_or_replace_the_data_are_refused(
    name: str, key: str, value: object, why: str
) -> None:
    figure = Figure()
    FigureCanvasAgg(figure)
    ax = figure.add_subplot()
    before = list(ax.get_children())
    with pytest.raises(ValueError, match=rf"^{name} sets {key!r}") as info:
        betweenstats(_three(), ax=ax, **{name: {key: value}})  # pyright: ignore[reportArgumentType]
    assert why in str(info.value)
    assert list(ax.get_children()) == before


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("point_args", {"visible": False}),
        ("point_args", {"alpha": 0}),
        ("point_args", {"sizes": [0.0]}),
        ("point_args", {"color": "none"}),
        ("boxplot_args", {"visible": False}),
        ("boxplot_args", {"linewidth": 0, "fill": False}),
        ("boxplot_args", {"alpha": 0.0}),
        ("violin_args", {"facecolor": "none", "edgecolor": "none"}),
    ],
)
def test_a_layer_made_invisible_is_a_warning(
    name: str, args: dict[str, object]
) -> None:
    result = _three()
    fig = betweenstats(result, **{name: args})  # pyright: ignore[reportArgumentType]
    added = fig.result.warnings[len(result.warnings) :]
    assert len(added) == 1, added
    assert added[0].startswith(f"{name}=")
    assert "invisible" in added[0]
    assert "layers=" in added[0]  # says how to leave the layer out instead


def test_a_visible_style_adds_no_warning() -> None:
    result = _three()
    fig = betweenstats(
        result,
        point_args={"alpha": 0.1, "sizes": [2.0]},
        boxplot_args={"linewidth": 0},  # the box face still shows
        violin_args={"alpha": 0.05},
    )
    assert fig.result.warnings == result.warnings


def test_withinstats_refuses_and_warns_the_same_way() -> None:
    result = two_conditions()
    with pytest.raises(ValueError, match=r"^point_args sets 'offsets'"):
        withinstats(result, point_args={"offsets": np.zeros((2, 2))})
    fig = withinstats(result, point_args={"visible": False})
    assert any("invisible" in w for w in fig.result.warnings)


@pytest.mark.parametrize(
    "palette",
    [
        ["red", "green", "blue", "notacolor"],
        ["red", "green", "blue", "red", 5],
    ],
)
def test_every_palette_entry_is_validated(palette: list[object]) -> None:
    with pytest.raises(ValueError, match="not a matplotlib color"):
        betweenstats(_three(), palette=palette)
