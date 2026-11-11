"""Regenerate the README's figures: run each function's example, save it as PNG.

Run from the repository root: ``uv run python docs/examples/readme_images.py``.
Writes docs/img/<function>.png. The files are identical byte for byte on every
run with the same platform and package versions.
"""

import os
import runpy
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FUNCTIONS = ("betweenstats", "withinstats", "histostats", "scatterstats", "barstats")


def main(out_dir: Path) -> None:
    """Run each example in a scratch folder and save its figure in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start = Path.cwd()
    for name in FUNCTIONS:
        with tempfile.TemporaryDirectory() as scratch:
            os.chdir(scratch)  # the example writes its own SVG here
            try:
                namespace = runpy.run_path(str(HERE / f"{name}.py"))
            finally:
                os.chdir(start)
        namespace["fig"].save(out_dir / f"{name}.png", dpi=100)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "img")
