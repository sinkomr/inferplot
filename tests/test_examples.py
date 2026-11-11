"""Every script in docs/examples runs as written: in a temporary folder, headless,
without the network, in a few seconds, and writes the files its docstring names."""

from __future__ import annotations

import ast
import re
import runpy
import socket
import time
from pathlib import Path

import pytest

from tests.timing import allowed

EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"
SCRIPTS = sorted(
    p for p in EXAMPLES.glob("*.py") if p.name != "readme_images.py"
)  # readme_images.py runs these and is checked in test_docs.py
FUNCTIONS = ("betweenstats", "withinstats", "histostats", "scatterstats", "barstats")


def _writes(script: Path) -> list[str]:
    """The file names after "Writes:" in the script's docstring."""
    doc = ast.get_docstring(ast.parse(script.read_text(encoding="utf-8"))) or ""
    found = re.search(r"Writes:\s*(.+)", doc, re.DOTALL)
    assert found is not None, f"{script.name}: the docstring names no files"
    return re.findall(r"[\w.-]+\.(?:svg|pdf|png)", found.group(1))


def test_every_function_and_task_has_an_example() -> None:
    names = {p.stem for p in SCRIPTS}
    assert set(FUNCTIONS) <= names
    assert {"two_panels", "analyze_then_render", "read_back"} <= names


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs(
    script: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = script.read_text(encoding="utf-8")
    assert len(source.splitlines()) < 40
    assert "pyplot" not in source

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("the example tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.chdir(tmp_path)
    start = time.perf_counter()
    runpy.run_path(str(script), run_name="__main__")
    elapsed = time.perf_counter() - start
    # A few seconds on a developer machine; generous for slow CI runners.
    assert elapsed < allowed(30), f"{script.name} took {elapsed:.1f} s"
    printed = capsys.readouterr().out
    assert printed.strip(), f"{script.name} printed nothing"
    written = _writes(script)
    assert written
    for name in written:
        assert (tmp_path / name).stat().st_size > 0, name
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(written)
