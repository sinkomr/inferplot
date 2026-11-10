"""The renderer never touches matplotlib.pyplot (rendering rules, CLAUDE.md rule 6)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

PYPLOT = re.compile(r"matplotlib\.pyplot|import pyplot|pyplot\.|\bplt\b")
ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    *sorted((ROOT / "src" / "inferplot" / "render").rglob("*.py")),
    ROOT / "src" / "inferplot" / "export.py",
]


def test_sources_never_mention_pyplot_or_show() -> None:
    assert len(SOURCES) >= 6
    for path in SOURCES:
        text = path.read_text(encoding="utf-8")
        assert PYPLOT.search(text) is None, path
        assert ".show(" not in text, path
        assert "rcParams[" not in text, path
        assert "usetex=True" not in text, path


def test_import_render_save_and_check_leave_pyplot_unimported(
    tmp_path: Path,
) -> None:
    script = f"""
import sys
import inferplot.render
import inferplot.export
assert "matplotlib.pyplot" not in sys.modules, "import"
from tests.render._results import many_points_result
from tests.render._within import many_subjects
for fig in (
    inferplot.render.betweenstats(many_points_result()),
    inferplot.render.withinstats(many_subjects(3_000)),
):
    fig.check()
    for suffix in (".svg", ".pdf", ".png"):
        fig.save({str(tmp_path)!r} + "/f" + suffix, dpi=40)
    fig._repr_png_()
assert "matplotlib.pyplot" not in sys.modules, "render"
print("ok")
"""
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
