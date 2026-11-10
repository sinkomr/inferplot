"""Level names with control characters, and characters no font has (review of
phase 3, finding 10): the result keeps the names; the figure draws a tab as a
space and any other control character as U+FFFD, says so in ``warnings``, and a
saved SVG is well-formed XML. matplotlib's "Glyph ... missing" warning does not
reach the caller for a character glyph_missing reports; a check that quotes
another panel's text quotes its plain wording."""

from __future__ import annotations

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import inferplot as ip
from inferplot.render.figure import describe_text

NAMES = ["bell\x07", "tab\there", "nul\x00char"]


def _long(names: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    m = rng.normal(0, 1, (8, len(names))) + np.arange(len(names))
    return pd.DataFrame(
        [
            (f"s{i}", name, float(m[i, j]))
            for i in range(8)
            for j, name in enumerate(names)
        ],
        columns=["id", "cond", "y"],
    )


def _glyph_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if str(w.message).startswith("Glyph ")]


def test_control_characters_are_drawn_replaced_and_said(tmp_path: Path) -> None:
    fig = ip.withinstats(_long(NAMES), "cond", "y", subject_id="id", type="np")
    result = fig.result
    assert set(result.levels) == set(NAMES)
    drawn = [t.get_text() for t in fig.axes["main"].get_xticklabels()]
    by_level = dict(zip(result.levels, drawn, strict=True))
    assert by_level["bell\x07"].startswith("bell�\n")
    assert by_level["tab\there"].startswith("tab here\n")
    assert by_level["nul\x00char"].startswith("nul�char\n")
    assert list(result.text.tick_labels) == [
        ip.render.withinstats(result).axes["main"].get_xticklabels()[n].get_text()
        for n in range(3)
    ]
    notes = [w for w in result.warnings if "control characters" in w]
    assert notes == [
        "The level 'bell\\x07' of 'cond' holds control characters (U+0007), which "
        "the figure draws each as � (U+FFFD); the result keeps the name as "
        "given.",
        "The level 'nul\\x00char' of 'cond' holds control characters (U+0000), "
        "which the figure draws each as � (U+FFFD); the result keeps the name "
        "as given.",
        "The level 'tab\\there' of 'cond' holds control characters (U+0009), which "
        "the figure draws a tab as a space; the result keeps the name as given.",
    ]
    path = tmp_path / "names.svg"
    fig.save(path)
    ET.parse(path)  # well-formed
    assert "bell�" in path.read_text(encoding="utf-8")


def test_missing_glyphs_are_reported_once_by_check(tmp_path: Path) -> None:
    names = ["日本語", "naïve"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = ip.withinstats(_long(names), "cond", "y", subject_id="id")
        codes = [c.code for c in fig.check()]
        for ext in ("png", "svg", "pdf"):
            fig.save(tmp_path / f"names.{ext}", dpi=60)
    assert "glyph_missing" in codes
    assert _glyph_warnings(caught) == []


def test_missing_glyphs_check_does_not_report_still_warn() -> None:
    # A private-use character (category Co): glyph_missing leaves it out, so
    # matplotlib's own warning is the only notice and passes through.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = ip.withinstats(
            _long(["a", "b"]), "cond", "y", subject_id="id", title="mark "
        )
        fig.check()
    assert "glyph_missing" not in [c.code for c in fig.check()]
    assert any("57344" in message for message in _glyph_warnings(caught))


def test_another_panels_subtitle_is_quoted_as_text() -> None:
    figure = Figure(figsize=(5.0, 3.0), dpi=100)
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    data = _long(["a", "b", "c"])
    first = ip.withinstats(data, "cond", "y", subject_id="id", ax=left)
    second = ip.withinstats(data, "cond", "y", subject_id="id", ax=right)
    (issue,) = [c for c in second.check() if c.code == "label_overlap"]
    subtitle = first.result.text.subtitle
    assert subtitle is not None
    assert f"{describe_text('subtitle', subtitle)} of another panel" in issue.message
