"""Shared helpers for the rendering tests: artists by gid, SVG and PDF readers."""

from __future__ import annotations

import functools
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.text import Text

from inferplot.render import StatsFigure, betweenstats
from tests.oracle_fixtures import OracleCase, require_cases
from tests.render._results import fixture_result, parametric_cases

SVG_NS = "{http://www.w3.org/2000/svg}"
DC_NS = "{http://purl.org/dc/elements/1.1/}"
FIGURE_CODES = frozenset({"label_overlap", "text_clipped"})


def short(case: OracleCase) -> str:
    return case.name.removeprefix("ggbetweenstats-").removesuffix("-parametric")


def case_params() -> list[Any]:
    cases = require_cases(parametric_cases(), "parametric ggbetweenstats")
    return [pytest.param(case, id=short(case)) for case in cases]


def case_named(name: str) -> OracleCase:
    (case,) = [c for c in parametric_cases() if short(c) == name]
    return case


@functools.cache
def default_figure(name: str) -> StatsFigure:
    """The figure of a fixture with default arguments (rendered once per session)."""
    return betweenstats(fixture_result(case_named(name)))


def gids(fig: StatsFigure) -> set[str]:
    return {
        str(a.get_gid()) for a in fig.figure.findobj(Artist) if a.get_gid() is not None
    }


def by_gid(fig: StatsFigure, gid: str) -> Artist | None:
    found = [a for a in fig.figure.findobj(Artist) if a.get_gid() == gid]
    assert len(found) <= 1, f"{len(found)} artists with gid {gid!r}"
    return found[0] if found else None


def text_of(fig: StatsFigure, gid: str) -> str | None:
    artist = by_gid(fig, gid)
    if artist is None:
        return None
    assert isinstance(artist, Text)
    return artist.get_text()


def numbered_texts(fig: StatsFigure, prefix: str) -> list[str]:
    """Texts with gids ``prefix-0``, ``prefix-1``, ... in order."""
    out: list[str] = []
    while (text := text_of(fig, f"{prefix}-{len(out)}")) is not None:
        out.append(text)
    return out


def numbered_count(fig: StatsFigure, prefix: str) -> int:
    n = 0
    while by_gid(fig, f"{prefix}-{n}") is not None:
        n += 1
    return n


def figure_codes(fig: StatsFigure) -> list[str]:
    return [issue.code for issue in fig.check() if issue.code in FIGURE_CODES]


def stroke_reach(fig: StatsFigure, *, near: float = 8.0) -> tuple[float, float]:
    """How the drawn bracket lines meet the other brackets' labels, in pixels.

    An independent measure for ``check()``: each bracket line is sampled as drawn,
    every 0.05 px along its pieces and at 9 offsets across its width, with
    projecting caps; a label's box is its rendered text extent. Returns the
    deepest sample inside another bracket's label box (its distance to the box's
    nearest edge; 0.0 when none is inside) and the smallest distance from a
    sample outside to such a box (inf when no line comes within ``near``).
    """
    fig.draw()
    renderer = fig.figure.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
    dpi = float(fig.figure.dpi)
    deepest, clearance = 0.0, float("inf")
    for n, (label, _, _) in enumerate(fig.brackets):
        box = label.get_window_extent(renderer)
        for m, (_, line, _) in enumerate(fig.brackets):
            if m == n:
                continue
            data = np.asarray(line.get_xydata(), dtype=np.float64)  # pyright: ignore[reportUnknownArgumentType]
            xy = np.asarray(line.get_transform().transform(data), dtype=np.float64)
            half = float(line.get_linewidth()) * dpi / 72.0 / 2.0
            for (x0, y0), (x1, y1) in itertools.pairwise(xy):
                if (
                    min(x0, x1) - half - near > box.x1
                    or max(x0, x1) + half + near < box.x0
                    or min(y0, y1) - half - near > box.y1
                    or max(y0, y1) + half + near < box.y0
                ):
                    continue
                length = float(np.hypot(x1 - x0, y1 - y0))
                ux, uy = (x1 - x0) / length, (y1 - y0) / length
                # Only the stretch of the line within ``near`` of the box.
                along = [
                    (cx - x0) * ux + (cy - y0) * uy
                    for cx in (box.x0, box.x1)
                    for cy in (box.y0, box.y1)
                ]
                start = max(-half, min(along) - near)
                stop = min(length + half, max(along) + near)
                t, s = np.meshgrid(
                    np.append(np.arange(start, stop, 0.05), stop),
                    np.linspace(-half, half, 9),
                )
                px = x0 + ux * t - uy * s
                py = y0 + uy * t + ux * s
                inside = (px > box.x0) & (px < box.x1) & (py > box.y0) & (py < box.y1)
                if inside.any():
                    depth = np.minimum.reduce(
                        [px - box.x0, box.x1 - px, py - box.y0, box.y1 - py]
                    )[inside]
                    deepest = max(deepest, float(depth.max()))
                dx = np.maximum(np.maximum(box.x0 - px, px - box.x1), 0.0)
                dy = np.maximum(np.maximum(box.y0 - py, py - box.y1), 0.0)
                gaps = np.hypot(dx, dy)[~inside]
                if gaps.size:
                    clearance = min(clearance, float(gaps.min()))
    return deepest, clearance


def svg_texts(path: Path) -> list[str]:
    """The text of every SVG <text> element, in document order."""
    root = ET.parse(path).getroot()
    return ["".join(el.itertext()) for el in root.iter(f"{SVG_NS}text")]


def svg_ids(path: Path) -> set[str]:
    root = ET.parse(path).getroot()
    return {str(el.get("id")) for el in root.iter() if el.get("id") is not None}


def svg_description(path: Path) -> str:
    root = ET.parse(path).getroot()
    found = root.find(f".//{DC_NS}description")
    assert found is not None, "no dc:description"
    assert found.text is not None
    return found.text


_PDF_ESCAPES = {
    ord("n"): b"\n",
    ord("r"): b"\r",
    ord("t"): b"\t",
    ord("b"): b"\b",
    ord("f"): b"\f",
    ord("("): b"(",
    ord(")"): b")",
    ord("\\"): b"\\",
}


def pdf_info_string(data: bytes, key: str) -> str:
    """A literal string of the PDF Info dictionary, with PDF escapes decoded."""
    marker = f"/{key} (".encode("ascii")
    assert data.count(marker) == 1, f"expected one {marker!r}"
    i = data.index(marker) + len(marker)
    out = bytearray()
    depth = 0
    while True:
        c = data[i]
        if c == ord("\\"):
            n = data[i + 1]
            if n in _PDF_ESCAPES:
                out += _PDF_ESCAPES[n]
                i += 2
            elif ord("0") <= n <= ord("7"):
                j = i + 1
                while j < i + 4 and ord("0") <= data[j] <= ord("7"):
                    j += 1
                out.append(int(data[i + 1 : j], 8) & 0xFF)
                i = j
            elif n in (ord("\n"), ord("\r")):  # line continuation
                i += 2
                if n == ord("\r") and data[i] == ord("\n"):
                    i += 1
            else:
                out.append(n)
                i += 2
        elif c == ord("("):
            depth += 1
            out.append(c)
            i += 1
        elif c == ord(")"):
            if depth == 0:
                break
            depth -= 1
            out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1
    return out.decode("latin-1")
