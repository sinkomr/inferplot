"""Fonts: DejaVu Sans first, the user's sans-serif fonts as fallbacks, a generic
fallback in SVG, and glyph_missing for characters no font can draw.

DejaVu Serif ships with matplotlib and has glyphs DejaVu Sans lacks (U+2900
"⤀"), so it stands in for a user's CJK font on every machine.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import matplotlib
import numpy as np
import pytest

from inferplot.export import _write  # pyright: ignore[reportPrivateUsage]
from inferplot.render import StatsFigure, betweenstats
from inferplot.result import StatsResult
from tests.render._results import frame_result

ARROW = "⤀"  # in DejaVu Serif, not in DejaVu Sans


def _result(first: str = "a") -> StatsResult:
    rng = np.random.default_rng(1)
    groups = [rng.normal(m, 1, 20) for m in (5, 6, 7.5)]
    return frame_result([first, "b", "c"], groups, pairwise_display="all")


def _missing(fig: StatsFigure) -> str | None:
    found = [i for i in fig.check() if i.code == "glyph_missing"]
    return None if not found else found[0].message + " " + found[0].fix


def _bytes(fig: StatsFigure, fmt: str) -> bytes:
    buffer = io.BytesIO()
    _write(fig, buffer, fmt, dpi=100, transparent=False)  # pyright: ignore[reportArgumentType]
    return buffer.getvalue()


def _with_serif() -> dict[Any, Any]:
    return {
        "font.sans-serif": [
            "DejaVu Serif",
            "No Such Font Family",
            *matplotlib.rcParamsDefault["font.sans-serif"],
        ]
    }


def test_text_uses_dejavu_sans_then_a_generic_fallback() -> None:
    fig = betweenstats(_result())
    label = fig.axes["main"].get_xticklabels()[0]
    assert label.get_fontfamily() == ["DejaVu Sans", "sans-serif"]
    assert _missing(fig) is None


# matplotlib warns about each missing glyph as it draws.
missing_glyph = pytest.mark.filterwarnings("ignore:Glyph .* missing:UserWarning")


@missing_glyph
def test_a_character_no_font_has_is_reported() -> None:
    fig = betweenstats(_result(f"x{ARROW}y"), title=f"see {ARROW}")
    message = _missing(fig)
    assert message is not None
    assert ARROW in message
    assert "font.sans-serif" in message


@missing_glyph
@pytest.mark.filterwarnings("ignore:Font 'rm' does not have a glyph")
def test_a_character_missing_in_mathtext_is_reported() -> None:
    # Neither DejaVu Sans nor STIX has CJK characters.
    fig = betweenstats(_result(), title="$x$ \u6771")
    message = _missing(fig)
    assert message is not None
    assert "\u6771" in message
    assert "$" not in message.split(".")[0]


def test_the_users_sans_serif_fonts_are_fallbacks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    plain = betweenstats(_result())
    with matplotlib.rc_context(_with_serif()), caplog.at_level(logging.WARNING):
        latin = betweenstats(_result())
        fig = betweenstats(_result(f"x{ARROW}y"))
    assert not [r for r in caplog.records if "findfont" in r.getMessage()]
    label = fig.axes["main"].get_xticklabels()[0]
    assert label.get_fontfamily() == ["DejaVu Sans", "DejaVu Serif", "sans-serif"]
    assert _missing(fig) is None
    assert b"DejaVuSerif" in _bytes(fig, "pdf")  # the arrow's font is embedded
    # Latin text is drawn exactly as without the fallback.
    for fmt in ("png", "pdf"):
        assert _bytes(latin, fmt) == _bytes(plain, fmt)


def test_svg_names_the_fallbacks_and_a_generic_family() -> None:
    with matplotlib.rc_context(_with_serif()):
        fig = betweenstats(_result())
    svg = _bytes(fig, "svg").decode()
    families = set(re.findall(r"font-family: ([^;\"]*)", svg))
    plain = [f for f in families if f.startswith("'DejaVu Sans', 'DejaVu Serif'")]
    assert plain
    assert all(f.rstrip().endswith("sans-serif") for f in plain)
