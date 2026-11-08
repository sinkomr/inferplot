"""Packaging and import-time guarantees."""

import importlib.metadata
import subprocess
import sys
import textwrap

import inferplot

# Imports every inferplot module, so a module that only some code path loads is
# checked too.
IMPORT_ALL = """
import importlib, pkgutil, sys
import inferplot
for module in pkgutil.walk_packages(inferplot.__path__, "inferplot."):
    importlib.import_module(module.name)
"""


def _run(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_version_matches_installed_metadata() -> None:
    assert inferplot.__version__ == importlib.metadata.version("inferplot")


def test_import_does_not_load_pyplot() -> None:
    # CLAUDE.md rule 6: library code never imports matplotlib.pyplot.
    _run(IMPORT_ALL + "assert 'matplotlib.pyplot' not in sys.modules\n")


def test_import_works_without_optional_backends() -> None:
    # CLAUDE.md rule 5: pandas, polars and pyarrow are optional extras. With all
    # three unimportable (a None entry in sys.modules makes `import` raise
    # ImportError), every inferplot module must still import.
    block = "import sys\nfor name in ('pandas', 'polars', 'pyarrow'):\n"
    block += "    sys.modules[name] = None\n"
    _run(block + IMPORT_ALL)
