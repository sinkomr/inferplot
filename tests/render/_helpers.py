"""Shared helpers for the rendering tests: artists by gid, SVG and PDF readers."""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

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
