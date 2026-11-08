"""Packaging and import-time guarantees."""

import importlib.metadata
import subprocess
import sys

import inferplot


def test_version_matches_installed_metadata() -> None:
    assert inferplot.__version__ == importlib.metadata.version("inferplot")


def test_import_does_not_load_pyplot() -> None:
    # CLAUDE.md rule 6: library code never imports matplotlib.pyplot.
    code = "import sys, inferplot; assert 'matplotlib.pyplot' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
