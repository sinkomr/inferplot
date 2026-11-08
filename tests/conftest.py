"""Shared pytest configuration."""

import importlib.metadata
import os

from hypothesis import settings

# Wall-clock deadlines make property tests flaky on slow CI machines (Windows).
settings.register_profile("default", deadline=None)
# CI: the same examples on every run, and a reproduction blob on failure.
settings.register_profile("ci", deadline=None, derandomize=True, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

_REPORTED = ("numpy", "scipy", "matplotlib", "narwhals", "pandas", "polars", "pyarrow")


def pytest_report_header() -> str:
    """Show the installed versions that the statistics depend on."""
    versions: list[str] = []
    for name in _REPORTED:
        try:
            versions.append(f"{name} {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"{name} (not installed)")
    return "inferplot dependencies: " + ", ".join(versions)
