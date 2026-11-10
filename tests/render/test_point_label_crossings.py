"""scatterstats: no line runs through a point label.

A review found point labels crossed by the least-squares line and by other
labels' link lines (mtcars with every car labelled: 7 and 5 of 32 labels; 20
random labels: every figure), and ``fig.check()`` silent. Measured here
independently of the placement: each drawn line (the least-squares line, every
link from its point to its label) is sampled every 0.25 px, and a sample more
than 1 px inside another label's rendered box is a crossing.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import ConnectionPatch
from matplotlib.text import Text

import inferplot as ip
from inferplot.render import StatsFigure
from inferplot.render.point_labels import place_labels

DATASETS = Path(__file__).resolve().parents[2] / "oracle" / "datasets"
INSIDE_PX: Final = 1.0


def _labels(fig: StatsFigure) -> list[Text]:
    ax = fig.axes["main"]
    return [
        t
        for t in ax.texts
        if str(t.get_gid()).startswith("point-label-")
        and "link" not in str(t.get_gid())
    ]


def _samples(a: Any, b: Any) -> np.ndarray[Any, Any]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = max(2, int(np.hypot(*(b - a)) / 0.25) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return a + t * (b - a)


def _inside(points: np.ndarray[Any, Any], box: Any) -> bool:
    return bool(
        (
            (points[:, 0] > box.x0 + INSIDE_PX)
            & (points[:, 0] < box.x1 - INSIDE_PX)
            & (points[:, 1] > box.y0 + INSIDE_PX)
            & (points[:, 1] < box.y1 - INSIDE_PX)
        ).any()
    )


def crossed_labels(fig: StatsFigure) -> tuple[list[str], list[str]]:
    """The labels the least-squares line crosses, and those another label's
    link crosses."""
    fig.draw()
    ax = fig.axes["main"]
    labels = _labels(fig)
    boxes = [t.get_window_extent() for t in labels]
    by_line: list[str] = []
    for line in ax.lines:
        if line.get_gid() != "smooth-line":
            continue
        xy = ax.transData.transform(np.asarray(line.get_xydata(), dtype=float))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        pts = np.vstack([_samples(p, q) for p, q in itertools.pairwise(xy)])
        by_line = [
            t.get_text() for t, b in zip(labels, boxes, strict=True) if _inside(pts, b)
        ]
    by_link: list[str] = []
    for link in (a for a in ax.get_children() if isinstance(a, ConnectionPatch)):
        start = link.coords1.transform(link.xy1)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportOptionalMemberAccess]
        end = link.coords2.transform(link.xy2)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportOptionalMemberAccess]
        pts = _samples(start, end)
        for t, b in zip(labels, boxes, strict=True):
            own = np.allclose(t.get_transform().transform(t.get_position()), end)
            if not own and _inside(pts, b):
                by_link.append(t.get_text())
    return by_line, by_link


def _mtcars() -> pd.DataFrame:
    return pd.read_csv(DATASETS / "mtcars.csv")


def test_mtcars_every_car_labelled() -> None:
    fig = ip.scatterstats(_mtcars(), "wt", "mpg", label_var="model")
    assert crossed_labels(fig) == ([], [])
    assert "point_labels_crowded" not in [i.code for i in fig.check()]


def test_mtcars_selected_cars() -> None:
    cars = _mtcars()
    keep = (cars["mpg"] > 25) | (cars["wt"] > 5)
    fig = ip.scatterstats(
        cars.assign(model=cars["model"].where(keep)), "wt", "mpg", label_var="model"
    )
    assert crossed_labels(fig) == ([], [])


@pytest.mark.parametrize("n", [10, 20, 32])
@pytest.mark.parametrize("seed", range(3))
def test_random_labelled_points(n: int, seed: int) -> None:
    rng = np.random.default_rng([12, n, seed])
    x = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "x": x,
            "y": 0.6 * x + 0.8 * rng.normal(0, 1, n),
            "lab": [f"p{i}" for i in range(n)],
        }
    )
    fig = ip.scatterstats(frame, "x", "y", label_var="lab")
    assert crossed_labels(fig) == ([], []), (n, seed)
    assert "point_labels_crowded" not in [i.code for i in fig.check()]


def test_crossings_are_reported() -> None:
    cars = _mtcars()
    fig = ip.scatterstats(cars, "wt", "mpg", label_var="model")
    labels = {t.get_text(): t for t in _labels(fig)}
    # A label moved onto the least-squares line, at its middle.
    label = labels["Valiant"]
    label.set_transform(fig.axes["main"].transData)
    label.set_position((3.3, 37.285 - 5.344 * 3.3))
    label.set_horizontalalignment("center")
    label.set_verticalalignment("center")
    (issue,) = [i for i in fig.check() if i.code == "point_labels_crowded"]
    assert (
        "1 point label lies across the least-squares line: point label 'Valiant'"
        in (issue.message)
    )
    assert "smooth=False" in issue.fix
    # A label moved onto another label's link.
    links = [
        a for a in fig.axes["main"].get_children() if isinstance(a, ConnectionPatch)
    ]
    assert links
    start = links[0].coords1.transform(links[0].xy1)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportOptionalMemberAccess]
    end = links[0].coords2.transform(links[0].xy2)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportOptionalMemberAccess]
    middle = fig.axes["main"].transData.inverted().transform((start + end) / 2)
    other = labels["Fiat 128"]
    other.set_transform(fig.axes["main"].transData)
    other.set_position(tuple(middle))
    other.set_horizontalalignment("center")
    other.set_verticalalignment("center")
    (issue,) = [i for i in fig.check() if i.code == "point_labels_crowded"]
    assert "crossed by the line joining another label to its point" in issue.message
    assert "'Fiat 128'" in issue.message


FRAME: Final = (0.0, 0.0, 600.0, 400.0)


def _box(anchor: Any, size: Any, place: Any) -> tuple[float, float, float, float]:
    x, y = anchor[0] + place.dx, anchor[1] + place.dy
    w, h = size
    x0 = {"left": x, "right": x - w, "center": x - w / 2}[place.ha]
    y0 = {"bottom": y, "top": y - h, "center": y - h / 2}[place.va]
    return x0, y0, x0 + w, y0 + h


def test_a_label_keeps_off_the_line() -> None:
    # One point on a rising line: right of it (the first choice) the line runs
    # through the label, so it goes to the first place the line misses.
    anchors = np.array([[300.0, 200.0]])
    sizes = np.array([[80.0, 12.0]])
    xs = np.linspace(0.0, 600.0, 61)
    ys = 200.0 + (xs - 300.0) * 0.05
    line = np.column_stack([xs[:-1], ys[:-1], xs[1:], ys[1:], np.full(60, 1.0)])
    (free,) = place_labels(
        anchors=anchors, sizes=sizes, points=anchors, marker=2.8, gap=2.0, frame=FRAME
    )
    assert (free.ha, free.va) == ("left", "center")
    (place,) = place_labels(
        anchors=anchors, sizes=sizes, points=anchors, marker=2.8, gap=2.0,
        frame=FRAME, lines=line,
    )  # fmt: skip
    x0, y0, x1, y1 = _box(anchors[0], sizes[0], place)
    on_line = 200.0 + (np.linspace(x0, x1, 50) - 300.0) * 0.05
    assert ((on_line + 1.0 < y0) | (on_line - 1.0 > y1)).all()


@pytest.mark.parametrize("seed", range(6))
def test_links_go_around_labels(seed: int) -> None:
    # 25 labels on 150 points in the middle of the plot: many are moved and
    # joined to their point; no link may run through another label.
    rng = np.random.default_rng([7, seed])
    points = np.column_stack([rng.uniform(200, 400, 150), rng.uniform(130, 270, 150)])
    anchors = points[:25]
    sizes = np.column_stack([rng.uniform(30, 90, 25), np.full(25, 12.0)])
    places = place_labels(
        anchors=anchors, sizes=sizes, points=points, marker=2.8, gap=2.0,
        frame=FRAME, link_width=0.8,
    )  # fmt: skip
    assert sum(p.ring > 0 for p in places) >= 5
    boxes = [_box(a, s, p) for a, s, p in zip(anchors, sizes, places, strict=True)]
    for i, (point, place) in enumerate(zip(anchors, places, strict=True)):
        if place.ring == 0:
            continue
        step = np.array([place.dx, place.dy])
        # The link is drawn from the edge of the point's marker.
        start = point + step / np.hypot(*step) * 2.8
        pts = _samples(start, point + step)
        for j, (x0, y0, x1, y1) in enumerate(boxes):
            if j == i:
                continue
            inside = (
                (pts[:, 0] > x0 + INSIDE_PX) & (pts[:, 0] < x1 - INSIDE_PX)
                & (pts[:, 1] > y0 + INSIDE_PX) & (pts[:, 1] < y1 - INSIDE_PX)
            )  # fmt: skip
            assert not inside.any(), (seed, i, j)
