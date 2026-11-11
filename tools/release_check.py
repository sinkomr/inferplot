"""Checks the release workflow runs on the built packages (docs/RELEASING.md).

Run from the repository root:

    python -I tools/release_check.py dists dist/
    python -I tools/release_check.py version dist/ --tag v0.1.0
    python -I tools/release_check.py quickstart README.md > quickstart.py

``dists`` checks that the wheel holds the package and nothing else and that the
sdist holds what the tests and a downstream packager need; ``version`` checks the
tag, the built version and the CHANGELOG against each other; ``quickstart`` prints
the README's quick start so that it can be run against an installed wheel. Each
prints what it checked and exits 1 with the reasons on the first failure. Standard
library only, so it runs before anything is installed. tests/test_packaging.py
tests these functions on hand-built archives.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import HeaderParser
from pathlib import Path, PurePosixPath

# A wheel may hold the package and its metadata directory, nothing else.
WHEEL_PACKAGE = "inferplot"
WHEEL_REQUIRED = ("inferplot/__init__.py", "inferplot/py.typed")
WHEEL_LICENSES = ("licenses/LICENSE", "licenses/NOTICE")

# Files the sdist must carry, relative to its top directory: the build inputs,
# the licenses, and what `pytest` needs when run from the unpacked sdist (the
# oracle fixtures and datasets, the reference implementations, the documents the
# documentation tests execute, the evaluation tasks and the release workflow that
# tests/test_packaging.py reads).
SDIST_REQUIRED = (
    "pyproject.toml",
    "hatch_build.py",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CHANGELOG.md",
    "llms.txt",
    "src/inferplot/__init__.py",
    "src/inferplot/py.typed",
    "tests/conftest.py",
    "tests/tolerances.py",
    "tests/test_packaging.py",
    "tests/test_evals.py",
    "tests/fixtures/oracle/manifest.json",
    "oracle/datasets/PlantGrowth.csv",
    "oracle/generate.R",
    "tools/reference/stats_reference.py",
    "tools/release_check.py",
    "docs/api.md",
    "docs/intervals.md",
    "skills/inferplot/SKILL.md",
    "evals/run.py",
    "evals/README.md",
    ".github/workflows/release.yml",
    ".github/workflows/oracle.yml",
)
# Path parts that never belong in either archive.
FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".claude",
        "build",
        "dist",
        ".DS_Store",
    }
)
# Words that mark test data; none may appear in a wheel path.
WHEEL_FORBIDDEN_WORDS = ("tests", "fixtures", "oracle", "evals", "conftest")

LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")
CHANGELOG_ENTRY = re.compile(
    r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}|Unreleased))?\s*$",
    re.MULTILINE,
)


class CheckError(Exception):
    """A release check failed; the message says why."""


def _one(dist_dir: Path, pattern: str) -> Path:
    found = sorted(dist_dir.glob(pattern))
    if len(found) != 1:
        names = [p.name for p in found]
        raise CheckError(f"expected one {pattern} in {dist_dir}, found {names}")
    return found[0]


def check_wheel(path: Path) -> list[str]:
    """Check a wheel's file list and metadata; return its file names."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        info_dirs = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
        if len(info_dirs) != 1:
            raise CheckError(f"{path.name}: expected one .dist-info, found {info_dirs}")
        info = info_dirs.pop()
        metadata = archive.read(f"{info}/METADATA").decode("utf-8")
    problems: list[str] = []
    for name in names:
        top = name.split("/", 1)[0]
        parts = PurePosixPath(name).parts
        if top not in (WHEEL_PACKAGE, info):
            problems.append(f"{name}: outside the package")
        if FORBIDDEN_PARTS.intersection(parts) or name.endswith((".pyc", ".pyo")):
            problems.append(f"{name}: build or cache file")
        lowered = [p.lower() for p in parts[1:]]
        if any(w in p for p in lowered for w in WHEEL_FORBIDDEN_WORDS):
            problems.append(f"{name}: test data in the wheel")
    for required in WHEEL_REQUIRED:
        if required not in names:
            problems.append(f"{required}: missing")
    for license_file in WHEEL_LICENSES:
        if f"{info}/{license_file}" not in names:
            problems.append(f"{info}/{license_file}: missing")
    problems.extend(_metadata_problems(metadata, path.name))
    if problems:
        raise CheckError(f"{path.name}:\n  " + "\n  ".join(problems))
    return names


def _metadata_problems(text: str, label: str) -> list[str]:
    meta = HeaderParser().parsestr(text)
    problems: list[str] = []
    expected = {
        "Name": "inferplot",
        "License-Expression": "MIT",
        "Description-Content-Type": "text/markdown",
        "Requires-Python": ">=3.11",
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            problems.append(
                f"{label} METADATA {key} is {meta.get(key)!r}, not {value!r}"
            )
    requires = set(meta.get_all("Requires-Dist") or [])
    runtime = {r.split(";")[0].strip() for r in requires if "extra ==" not in r}
    names = {re.split(r"[<>=!~ \[]", r, maxsplit=1)[0].lower() for r in runtime}
    if names != {"numpy", "scipy", "matplotlib", "narwhals"}:
        problems.append(
            f"{label} runtime dependencies are {sorted(names)} (CLAUDE.md rule 5)"
        )
    body = meta.get_payload()
    if not isinstance(body, str) or "## Quick start" not in body:
        problems.append(f"{label} long description lacks the README's quick start")
        return problems
    # PyPI shows the description on its own: a relative link leads nowhere there
    # (hatch_build.py makes them absolute at build time).
    relative = sorted(
        {
            target
            for target in LINK_TARGET.findall(body)
            if not target.startswith(("http://", "https://", "mailto:", "#"))
        }
    )
    if relative:
        problems.append(
            f"{label} long description has relative links: {', '.join(relative)}"
        )
    return problems


def check_sdist(path: Path) -> list[str]:
    """Check an sdist's file list; return the names relative to its top directory."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    tops = {PurePosixPath(m.name).parts[0] for m in members}
    if len(tops) != 1:
        raise CheckError(f"{path.name}: expected one top directory, found {tops}")
    top = tops.pop()
    names = [str(PurePosixPath(m.name).relative_to(top)) for m in members if m.isfile()]
    problems: list[str] = []
    for member in members:
        if not (member.isfile() or member.isdir()):
            problems.append(f"{member.name}: a link or special file")
    for name in names:
        if FORBIDDEN_PARTS.intersection(PurePosixPath(name).parts) or name.endswith(
            (".pyc", ".pyo")
        ):
            problems.append(f"{name}: build or cache file")
    present = set(names)
    for required in SDIST_REQUIRED:
        if required not in present:
            problems.append(f"{required}: missing")
    if problems:
        raise CheckError(f"{path.name}:\n  " + "\n  ".join(problems))
    return names


def check_dists(dist_dir: Path) -> str:
    wheel = _one(dist_dir, "*.whl")
    sdist = _one(dist_dir, "*.tar.gz")
    wheel_names = check_wheel(wheel)
    sdist_names = check_sdist(sdist)
    wheel_version = wheel.name.split("-")[1]
    sdist_version = sdist.name.removesuffix(".tar.gz").split("-")[1]
    if wheel_version != sdist_version:
        raise CheckError(f"wheel {wheel_version} and sdist {sdist_version} differ")
    return (
        f"{wheel.name}: {len(wheel_names)} files, package and metadata only\n"
        f"{sdist.name}: {len(sdist_names)} files, every required file present"
    )


def changelog_latest(text: str) -> tuple[str, str | None]:
    """The first versioned entry of a Keep a Changelog file: (version, date)."""
    for match in CHANGELOG_ENTRY.finditer(text):
        if match["version"] != "Unreleased":
            return match["version"], match["date"]
    raise CheckError("CHANGELOG.md has no versioned entry")


def release_base(version: str) -> str:
    """0.1.0 for 0.1.0, 0.1.0.dev0 and 0.1.0rc1: the release a version prepares."""
    match = re.match(r"^(\d+\.\d+\.\d+)", version)
    if match is None:
        raise CheckError(f"version {version!r} is not X.Y.Z or a pre-release of it")
    return match[1]


def check_version(dist_dir: Path, tag: str | None, changelog: Path) -> str:
    wheel = _one(dist_dir, "*.whl")
    version = wheel.name.split("-")[1]
    lines = [f"built version {version}"]
    if tag is not None:
        if tag != f"v{version}":
            raise CheckError(f"tag {tag} does not match the built version {version}")
        lines.append(f"tag {tag} matches")
    entry, date = changelog_latest(changelog.read_text(encoding="utf-8"))
    if entry != release_base(version):
        raise CheckError(f"CHANGELOG.md's latest entry is {entry}, not {version}")
    if (
        tag is not None
        and re.fullmatch(r"\d+\.\d+\.\d+", version)
        and (date is None or date == "Unreleased")
    ):
        raise CheckError(f"CHANGELOG.md's entry {entry} has no release date")
    lines.append(f"CHANGELOG.md's latest entry is {entry} ({date})")
    return "\n".join(lines)


def quickstart(readme: Path) -> str:
    """The first Python block under the README's "## Quick start" heading."""
    text = readme.read_text(encoding="utf-8")
    section = text.split("\n## Quick start\n", 1)
    if len(section) != 2:
        raise CheckError("README.md has no '## Quick start' section")
    match = re.search(r"```python\n(.*?)\n```", section[1], re.DOTALL)
    if match is None:
        raise CheckError("README.md's quick start has no python block")
    return match[1] + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dists = sub.add_parser("dists")
    dists.add_argument("dist_dir", type=Path)
    version = sub.add_parser("version")
    version.add_argument("dist_dir", type=Path)
    version.add_argument("--tag", default=None)
    version.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    quick = sub.add_parser("quickstart")
    quick.add_argument("readme", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "dists":
            print(check_dists(args.dist_dir))
        elif args.command == "version":
            print(check_version(args.dist_dir, args.tag, args.changelog))
        else:
            sys.stdout.write(quickstart(args.readme))
    except CheckError as error:
        print(f"release check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
