"""Shared pytest configuration."""

import os

from hypothesis import settings

# Wall-clock deadlines make property tests flaky on slow CI machines (Windows).
settings.register_profile("default", deadline=None)
# CI: the same examples on every run, and a reproduction blob on failure.
settings.register_profile("ci", deadline=None, derandomize=True, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
