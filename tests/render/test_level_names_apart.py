"""Level names that would be shortened alike stay apart (D77 in the figures).

A figure wraps a long level name into at most three lines and shortens what
does not fit with an ellipsis. Two names that differ only at their end would
then be drawn alike: the tick labels of betweenstats, withinstats and barstats
and the legend entries of barstats (the only key to its colours). Such names
keep the part where they differ, after an elided shared start; names that do
not collide are drawn as before. ``fig.check()`` reports any two levels whose
drawn labels read alike (``level_labels_alike``), whoever made them so.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import inferplot as ip
from inferplot.render import StatsFigure, comparison

LONG = "A very long level name that goes on and on about the category it names"
PERIODS = [
    "Measured in the first half of the year 2023",
    "Measured in the first half of the year 2024",
]
_N_LINE = re.compile(r"\n\(n = [^\n]*\)$")


def _reads(text: str) -> str:
    """A drawn label as it reads: its line breaks as spaces, without the n line."""
    return " ".join(_N_LINE.sub("", text).split())


def _ticks(fig: StatsFigure) -> list[str]:
    return [_reads(t.get_text()) for t in fig.axes["main"].get_xticklabels()]


def _legend(fig: StatsFigure) -> list[str]:
    legend = fig.axes["main"].get_legend()
    assert legend is not None
    return [_reads(t.get_text()) for t in legend.get_texts()]


def _codes(fig: StatsFigure) -> list[str]:
    return [i.code for i in fig.check()]


def _bars_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.choice([LONG + " one", LONG + " two", "short"], 240)
    y = rng.choice(PERIODS, 240)
    return pd.DataFrame({"answer": x, "period": y})


def _keeps_its_end(drawn: str, name: str) -> bool:
    """The drawn text is an abbreviation of ``name`` that keeps its last word."""
    return drawn.endswith(name.split()[-1]) and "…" in drawn


# -- the wrapping itself -----------------------------------------------------------


def test_wrap_names_keeps_names_that_differ_at_their_end_apart() -> None:
    names = [LONG + " one", LONG + " two", "short"]
    drawn = comparison.wrap_names(names, lambda line: len(line) <= 16)
    reads = [" ".join(d.split()) for d in drawn]
    assert len(set(reads)) == 3
    assert drawn[2] == "short"
    for d, name in zip(drawn[:2], names[:2], strict=True):
        assert _keeps_its_end(" ".join(d.split()), name)
        assert len(d.split("\n")) <= comparison.TICK_LINES
        assert all(len(line) <= 16 for line in d.split("\n"))


def test_wrap_names_leaves_names_that_do_not_collide_as_they_were() -> None:
    names = [
        "control group receiving placebo",
        "a very long level name that keeps going and going",
        "x" * 40,
    ]
    fits = lambda line: len(line) <= 16  # noqa: E731
    expected = [comparison.wrap_name(n, fits) for n in names]
    assert comparison.wrap_names(names, fits) == expected


def test_wrap_names_draws_a_name_whole_when_shortening_cannot_keep_it_apart() -> None:
    # One word each, too long for a line and differing at its end: no shared
    # start can be elided, so they are drawn whole.
    names = ["a" * 30 + "1", "a" * 30 + "2"]
    drawn = comparison.wrap_names(names, lambda line: len(line) <= 16)
    reads = [" ".join(d.split()) for d in drawn]
    assert reads[0] != reads[1]
    assert all("…" not in d for d in drawn)


# -- the figures --------------------------------------------------------------------


def test_barstats_legend_and_ticks_keep_names_apart() -> None:
    fig = ip.barstats(_bars_frame(), "answer", "period", bf_message=False)
    legend = _legend(fig)
    assert len(set(legend)) == 3, legend
    assert _keeps_its_end(legend[0], LONG + " one")
    assert _keeps_its_end(legend[1], LONG + " two")
    assert legend[2] == "short"
    ticks = _ticks(fig)
    assert len(set(ticks)) == 2, ticks
    assert ticks[0].endswith("2023")
    assert ticks[1].endswith("2024")
    assert "level_labels_alike" not in _codes(fig)
    # The texts name the levels apart too (the alt text shortens them by D77).
    alt = fig.result.text.alt_text
    assert "2023" in alt
    assert "2024" in alt
    assert fig.result.text.tick_labels == tuple(PERIODS)


def test_barstats_one_way_legend_keeps_names_apart() -> None:
    fig = ip.barstats(_bars_frame(), "answer", bf_message=False)
    assert len(set(_legend(fig))) == 3
    assert "level_labels_alike" not in _codes(fig)


def test_betweenstats_ticks_keep_names_apart() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {"g": np.repeat([*PERIODS, "short"], 12), "v": rng.normal(size=36)}
    )
    fig = ip.betweenstats(frame, "g", "v")
    ticks = _ticks(fig)
    assert len(set(ticks)) == 3, ticks
    assert ticks[0].endswith("2023")
    assert ticks[1].endswith("2024")
    assert "level_labels_alike" not in _codes(fig)


def test_withinstats_ticks_keep_names_apart() -> None:
    rng = np.random.default_rng(2)
    n = 10
    frame = pd.DataFrame(
        {
            "when": np.tile(PERIODS, n),
            "id": np.repeat(np.arange(n), 2),
            "v": rng.normal(size=2 * n),
        }
    )
    fig = ip.withinstats(frame, "when", "v", subject_id="id")
    ticks = _ticks(fig)
    assert len(set(ticks)) == 2, ticks
    assert ticks[0].endswith("2023")
    assert ticks[1].endswith("2024")
    assert "level_labels_alike" not in _codes(fig)


# -- the check ---------------------------------------------------------------------


def _alike(fig: StatsFigure) -> str:
    (issue,) = [i for i in fig.check() if i.code == "level_labels_alike"]
    assert issue.severity == "warning"
    return issue.message


def test_check_reports_legend_entries_made_alike() -> None:
    fig = ip.barstats(_bars_frame(), "answer", "period", bf_message=False)
    legend = fig.axes["main"].get_legend()
    assert legend is not None
    texts = legend.get_texts()
    texts[1].set_text(texts[0].get_text())
    message = _alike(fig)
    assert "legend" in message
    assert repr(LONG + " one") in message
    assert repr(LONG + " two") in message


def test_check_reports_tick_labels_made_alike() -> None:
    fig = ip.barstats(_bars_frame(), "answer", "period", bf_message=False)
    ax = fig.axes["main"]
    ax.set_xticks([0, 1], labels=["same\n(n = 117)", "same\n(n = 123)"])
    message = _alike(fig)
    assert repr(PERIODS[0]) in message
    assert repr(PERIODS[1]) in message


def test_check_reports_alike_tick_labels_of_a_comparison_figure() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 8), "v": rng.normal(size=24)})
    fig = ip.betweenstats(frame, "g", "v")
    ax = fig.axes["main"]
    ax.set_xticks(ax.get_xticks(), labels=["a\n(n = 8)", "a\n(n = 8)", "c\n(n = 8)"])
    message = _alike(fig)
    assert "'a' and 'b'" in message


def test_check_reports_names_that_only_whitespace_tells_apart() -> None:
    frame = pd.DataFrame(
        {"x": ["a b", "a  b"] * 10, "y": ["u", "v"] * 5 + ["v", "u"] * 5}
    )
    fig = ip.barstats(frame, "x", "y", bf_message=False)
    assert "level_labels_alike" in _codes(fig)


@pytest.mark.parametrize("function", ["histostats", "scatterstats"])
def test_figures_without_level_labels_report_nothing(function: str) -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"a": rng.normal(size=30), "b": rng.normal(size=30)})
    fig = (
        ip.histostats(frame, "a")
        if function == "histostats"
        else ip.scatterstats(frame, "a", "b")
    )
    assert "level_labels_alike" not in _codes(fig)
