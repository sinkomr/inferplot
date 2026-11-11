"""Build hook: the README as PyPI shows it.

README.md links to files of the repository by relative path (the example scripts,
the figures in docs/img, docs/api.md), which is right on GitHub and broken on PyPI,
where the long description stands alone. This hook hands hatchling the README with
every relative link made absolute: figures to raw.githubusercontent.com, everything
else to the file's page on GitHub, and a link to a heading of the README itself
(``#not-in-v01``; PyPI gives headings no anchors) to that heading on GitHub, at the
tag of the version being built (``v0.1.0``) or, for a development or pre-release
version, which has no tag, at ``main``. The figures are served from the public
repository.
README.md itself is not changed. tools/release_check.py refuses a built package
whose description still holds a relative link; tests/test_packaging.py tests
``absolute_links``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface

REPOSITORY = "sinkomr/inferplot"
_TARGET = re.compile(r"\]\(([^)\s]+)\)")
_FINAL_VERSION = re.compile(r"\d+\.\d+\.\d+")
_ABSOLUTE = ("http://", "https://", "mailto:")
_IMAGES = (".png", ".svg", ".jpg", ".jpeg", ".gif")


def absolute_links(text: str, version: str) -> str:
    """``text`` (Markdown) with every relative link target made absolute."""
    ref = f"v{version}" if _FINAL_VERSION.fullmatch(version) else "main"

    def absolute(match: re.Match[str]) -> str:
        target = match[1]
        if target.startswith(_ABSOLUTE):
            return match[0]
        if target.startswith("#"):  # a heading of the README itself
            return f"](https://github.com/{REPOSITORY}/blob/{ref}/README.md{target})"
        path = target.removeprefix("./")
        if path.lower().endswith(_IMAGES):
            return f"](https://raw.githubusercontent.com/{REPOSITORY}/{ref}/{path})"
        # GitHub redirects /blob/<ref>/<directory> to the directory's page.
        return f"](https://github.com/{REPOSITORY}/blob/{ref}/{path})"

    return _TARGET.sub(absolute, text)


class ReadmeHook(MetadataHookInterface):
    """Sets the project's ``readme`` (declared dynamic in pyproject.toml)."""

    def update(self, metadata: dict[str, Any]) -> None:
        text = Path(self.root, "README.md").read_text(encoding="utf-8")
        metadata["readme"] = {
            "content-type": "text/markdown",
            "text": absolute_links(text, str(metadata["version"])),
        }
