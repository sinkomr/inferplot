"""Release metadata stays true (docs/RELEASING.md), without network access.

The version, the CHANGELOG and the release workflow are checked against each other
here so that a disagreement fails CI before anyone pushes a tag. The release
checks in tools/release_check.py, which the workflow runs on the built packages,
are tested on small hand-built archives.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import io
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

import inferplot

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CHANGELOG = ROOT / "CHANGELOG.md"


def _load_release_check() -> Any:
    spec = importlib.util.spec_from_file_location(
        "release_check", ROOT / "tools" / "release_check.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_check"] = module
    spec.loader.exec_module(module)
    return module


release_check = _load_release_check()

PUBLISHING_JOBS = {"publish-testpypi", "publish-pypi"}


def _workflow() -> dict[Any, Any]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


# --- version and CHANGELOG ---------------------------------------------------------


def test_version_matches_built_metadata() -> None:
    assert inferplot.__version__ == importlib.metadata.version("inferplot")


def test_version_has_one_source() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in project["project"]["dynamic"]
    assert "version" not in project["project"]
    assert project["tool"]["hatch"]["version"]["path"] == "src/inferplot/__init__.py"


def test_changelog_latest_entry_names_the_version() -> None:
    version = inferplot.__version__
    entry, date = release_check.changelog_latest(CHANGELOG.read_text(encoding="utf-8"))
    assert entry == release_check.release_base(version)
    if re.fullmatch(r"\d+\.\d+\.\d+", version):
        # A final version is a release commit: its entry carries the release date.
        assert date is not None
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), date


def test_changelog_has_a_link_for_the_entry() -> None:
    text = CHANGELOG.read_text(encoding="utf-8")
    entry, _ = release_check.changelog_latest(text)
    assert f"[{entry}]: https://github.com/sinkomr/inferplot/" in text


@pytest.mark.parametrize(
    ("version", "base"),
    [
        ("0.1.0", "0.1.0"),
        ("0.1.0.dev0", "0.1.0"),
        ("0.1.0rc1", "0.1.0"),
        ("1.2.3a4", "1.2.3"),
    ],
)
def test_release_base(version: str, base: str) -> None:
    assert release_check.release_base(version) == base


# --- the release workflow ----------------------------------------------------------


def test_workflow_parses_and_starts_only_by_tag_or_by_hand() -> None:
    workflow = _workflow()
    # YAML 1.1 reads the key `on` as True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"] == {"tags": ["v*"]}
    target = triggers["workflow_dispatch"]["inputs"]["target"]
    assert target["default"] == "build-only"
    assert target["options"] == ["build-only", "testpypi", "pypi"]


def test_workflow_actions_are_pinned_by_sha() -> None:
    uses = re.findall(
        r"^\s*(?:-\s*)?uses:\s*(\S+)", WORKFLOW.read_text(encoding="utf-8"), re.M
    )
    assert uses
    for ref in uses:
        action, _, sha = ref.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", sha), (
            f"{action} is not pinned by a SHA: {ref}"
        )
    # Each pinned line names its version in a comment.
    for line in WORKFLOW.read_text(encoding="utf-8").splitlines():
        if "uses:" in line:
            assert re.search(r"# v\d", line), line


def test_only_publishing_jobs_get_an_identity_token() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}
    for name, job in workflow["jobs"].items():
        permissions = job.get("permissions", {})
        if name in PUBLISHING_JOBS:
            assert permissions == {"id-token": "write"}, name
        else:
            assert "id-token" not in permissions, name


def test_publishing_jobs_need_environments_and_the_checks() -> None:
    jobs = _workflow()["jobs"]
    assert set(jobs) == {"build", "install", "sdist", *PUBLISHING_JOBS}
    testpypi, pypi = jobs["publish-testpypi"], jobs["publish-pypi"]
    assert testpypi["environment"]["name"] == "testpypi"
    assert set(testpypi["needs"]) == {"build", "install", "sdist"}
    assert "build-only" in testpypi["if"]
    # PyPI only after TestPyPI, only from a version tag, only through the
    # protected environment whose name PyPI's trusted publisher holds.
    assert pypi["environment"]["name"] == "pypi"
    assert pypi["needs"] == "publish-testpypi"
    assert "startsWith(github.ref, 'refs/tags/v')" in pypi["if"]
    for job in (testpypi, pypi):
        run = " ".join(step.get("run", "") for step in job["steps"])
        assert "--trusted-publishing always" in run
        assert "secrets." not in str(job)


def test_releases_do_not_race() -> None:
    concurrency = _workflow()["concurrency"]
    assert concurrency == {"group": "release", "cancel-in-progress": False}


def test_checkouts_keep_no_credentials() -> None:
    for job in _workflow()["jobs"].values():
        for step in job["steps"]:
            if str(step.get("uses", "")).startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False


# --- the release checks on hand-built archives -------------------------------------

METADATA = """\
Metadata-Version: 2.4
Name: inferplot
Version: 0.1.0
License-Expression: MIT
Requires-Python: >=3.11
Requires-Dist: matplotlib>=3.8
Requires-Dist: narwhals>=2.0
Requires-Dist: numpy>=1.26
Requires-Dist: scipy>=1.11
Requires-Dist: pandas>=2.1; extra == 'pandas'
Description-Content-Type: text/markdown

# inferplot

## Quick start
"""


def _wheel(path: Path, extra: tuple[str, ...] = (), metadata: str = METADATA) -> Path:
    info = "inferplot-0.1.0.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        for name in (
            "inferplot/__init__.py",
            "inferplot/py.typed",
            "inferplot/stats/contingency.py",
            f"{info}/licenses/LICENSE",
            f"{info}/licenses/NOTICE",
            f"{info}/RECORD",
            *extra,
        ):
            archive.writestr(name, "")
        archive.writestr(f"{info}/METADATA", metadata)
    return path


def _sdist(path: Path, names: tuple[str, ...]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            info = tarfile.TarInfo(f"inferplot-0.1.0/{name}")
            info.size = 0
            archive.addfile(info, io.BytesIO(b""))
    return path


def test_wheel_check_accepts_the_package_only(tmp_path: Path) -> None:
    release_check.check_wheel(_wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl"))


@pytest.mark.parametrize(
    "extra",
    [
        "tests/test_api.py",
        "inferplot/tests/fixtures/oracle/case.json",
        "oracle/datasets/iris.csv",
        "inferplot/__pycache__/x.cpython-314.pyc",
        "conftest.py",
    ],
)
def test_wheel_check_refuses_anything_else(tmp_path: Path, extra: str) -> None:
    wheel = _wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl", (extra,))
    with pytest.raises(release_check.CheckError):
        release_check.check_wheel(wheel)


def test_wheel_check_needs_every_module_of_the_source_tree(tmp_path: Path) -> None:
    # A wheel that lacks a module imports and runs the quick start (renderers
    # load on first use), so the file list is compared with the source tree.
    source = tmp_path / "src" / "inferplot"
    (source / "stats").mkdir(parents=True)
    for name in ("__init__.py", "py.typed", "stats/contingency.py"):
        (source / name).write_text("")
    wheel = _wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl")
    release_check.check_wheel(wheel, source)
    (source / "render").mkdir()
    (source / "render" / "barstats.py").write_text("")
    with pytest.raises(
        release_check.CheckError, match=r"inferplot/render/barstats\.py: in the source"
    ):
        release_check.check_wheel(wheel, source)
    # The real tree: every module is where the sdist rules put it.
    real = ROOT / "src" / "inferplot"
    assert (real / "render" / "barstats.py").is_file()


def test_wheel_check_refuses_a_wrong_name_or_version(tmp_path: Path) -> None:
    platform = _wheel(tmp_path / "inferplot-0.1.0-cp314-cp314-macosx_11_0_arm64.whl")
    with pytest.raises(release_check.CheckError, match="not a pure-Python wheel"):
        release_check.check_wheel(platform)
    renamed = _wheel(tmp_path / "inferplot-0.1.1-py3-none-any.whl")
    with pytest.raises(
        release_check.CheckError, match=r"METADATA Version is '0\.1\.0'"
    ):
        release_check.check_wheel(renamed)


def test_a_final_version_may_not_say_it_is_unreleased(tmp_path: Path) -> None:
    stale = METADATA + "\n**Status:** not released. inferplot is not on PyPI yet.\n"
    with pytest.raises(release_check.CheckError, match="still says 'not released'"):
        release_check.check_wheel(
            _wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl", metadata=stale)
        )
    # A development or pre-release version may: it is not the release.
    dev = stale.replace("Version: 0.1.0", "Version: 0.1.0rc1")
    release_check.check_wheel(
        _wheel(tmp_path / "inferplot-0.1.0rc1-py3-none-any.whl", metadata=dev)
    )
    # The README of this repository says so today, which is why a release commit
    # has to change it (docs/RELEASING.md step 1.3).
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (release_check.UNRELEASED.search(readme) is None) == bool(
        re.fullmatch(r"\d+\.\d+\.\d+", inferplot.__version__)
    )


def test_smoke_draws_one_figure_with_each_function(tmp_path: Path) -> None:
    report = release_check.smoke(tmp_path)
    for name in (
        "betweenstats",
        "withinstats",
        "histostats",
        "scatterstats",
        "barstats",
    ):
        assert f"\n{name}: " in report
        assert (tmp_path / f"smoke_{name}.svg").stat().st_size > 0
        assert (tmp_path / f"smoke_{name}.png").stat().st_size > 0


def test_workflow_stops_unless_the_pypi_environment_requires_a_reviewer() -> None:
    # GitHub creates a missing environment without protection on first use, so
    # the first job checks the protection rules before anything is built.
    steps = _workflow()["jobs"]["build"]["steps"]
    names = [step.get("name", "") for step in steps]
    guard = next(i for i, name in enumerate(names) if "requires a reviewer" in name)
    build = names.index("Build the sdist and the wheel")
    assert guard < build
    step = steps[guard]
    assert "github.event_name == 'push'" in step["if"]
    assert "inputs.target == 'pypi'" in step["if"]
    assert "environments/pypi" in step["run"]
    assert "required_reviewers" in step["run"]
    assert "exit 1" in step["run"]


def test_wheel_check_refuses_another_runtime_dependency(tmp_path: Path) -> None:
    metadata = METADATA.replace(
        "Requires-Dist: scipy", "Requires-Dist: seaborn\nRequires-Dist: scipy"
    )
    wheel = _wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl", metadata=metadata)
    with pytest.raises(release_check.CheckError, match="runtime dependencies"):
        release_check.check_wheel(wheel)


def test_wheel_check_refuses_relative_links_in_the_description(tmp_path: Path) -> None:
    metadata = METADATA + "\n[the API](docs/api.md) and ![a figure](docs/img/x.png)\n"
    wheel = _wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl", metadata=metadata)
    with pytest.raises(release_check.CheckError, match=r"relative links: docs/api\.md"):
        release_check.check_wheel(wheel)
    anchored = METADATA + "\n[below](#usage)\n"
    with pytest.raises(release_check.CheckError, match="relative links: #usage"):
        release_check.check_wheel(
            _wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl", metadata=anchored)
        )
    absolute = METADATA + "\n[the API](https://example.org/api.md)\n"
    release_check.check_wheel(
        _wheel(tmp_path / "inferplot-0.1.0-py3-none-any.whl", metadata=absolute)
    )


def _load_hatch_build() -> Any:
    spec = importlib.util.spec_from_file_location(
        "hatch_build", ROOT / "hatch_build.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_links_are_made_absolute_for_pypi() -> None:
    # The build hook (hatch_build.py) hands hatchling the README with absolute
    # links: figures from raw.githubusercontent.com, files from their GitHub page,
    # at the version's tag, or at main for a version that has no tag.
    hook = _load_hatch_build()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    relative = [
        target
        for target in release_check.LINK_TARGET.findall(readme)
        if not target.startswith(("http", "#"))
    ]
    assert "docs/img/betweenstats.png" in relative
    assert "docs/api.md" in relative
    for target in relative:  # the README's own links lead to files of the repository
        assert (ROOT / target).exists(), target

    released = hook.absolute_links(readme, "0.1.0")
    assert not [
        target
        for target in release_check.LINK_TARGET.findall(released)
        if not target.startswith("https://")
    ]
    base = "https://raw.githubusercontent.com/sinkomr/inferplot/v0.1.0/"
    assert f"]({base}docs/img/betweenstats.png)" in released
    assert "](https://github.com/sinkomr/inferplot/blob/v0.1.0/docs/api.md)" in released
    # A link to one of the README's own headings: PyPI gives headings no anchors.
    assert "](#not-in-v01)" in readme
    assert (
        "](https://github.com/sinkomr/inferplot/blob/v0.1.0/README.md#not-in-v01)"
        in released
    )
    # Only link targets change.
    assert release_check.LINK_TARGET.sub("", released) == release_check.LINK_TARGET.sub(
        "", readme
    )
    for version in ("0.1.0.dev0", "0.1.0rc1"):
        assert "/blob/main/docs/api.md)" in hook.absolute_links(readme, version)


def test_the_readme_hook_is_declared() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert sorted(project["project"]["dynamic"]) == ["readme", "version"]
    assert "readme" not in project["project"]
    assert project["tool"]["hatch"]["metadata"]["hooks"]["custom"] == {
        "path": "hatch_build.py"
    }


def test_sdist_check_needs_the_test_data(tmp_path: Path) -> None:
    complete = release_check.SDIST_REQUIRED
    release_check.check_sdist(_sdist(tmp_path / "ok.tar.gz", complete))
    missing = tuple(n for n in complete if not n.startswith("tests/fixtures"))
    with pytest.raises(release_check.CheckError, match=r"manifest\.json: missing"):
        release_check.check_sdist(_sdist(tmp_path / "missing.tar.gz", missing))
    cached = (*complete, "src/inferplot/__pycache__/x.pyc")
    with pytest.raises(release_check.CheckError, match="cache"):
        release_check.check_sdist(_sdist(tmp_path / "cached.tar.gz", cached))


def test_sdist_rules_include_every_required_file() -> None:
    # Every file the sdist must carry exists in the repository and is matched by
    # an include rule of pyproject.toml (hatchling builds from these rules).
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    rules = project["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    for name in release_check.SDIST_REQUIRED:
        assert (ROOT / name).is_file(), name
        path = PurePosixPath(name)
        assert any(
            path.match(rule.lstrip("/")) or name.startswith(rule.lstrip("/") + "/")
            for rule in rules
        ), name


def test_version_check(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist / "inferplot-0.1.0-py3-none-any.whl")
    dated = tmp_path / "dated.md"
    dated.write_text("# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-12-01\n")
    undated = tmp_path / "undated.md"
    undated.write_text("# Changelog\n\n## [0.1.0] - Unreleased\n")
    assert "matches" in release_check.check_version(dist, "v0.1.0", dated)
    with pytest.raises(release_check.CheckError, match="does not match"):
        release_check.check_version(dist, "v0.1.1", dated)
    with pytest.raises(release_check.CheckError, match="no release date"):
        release_check.check_version(dist, "v0.1.0", undated)
    # Without a tag (a practice or build-only run) the entry may be undated.
    release_check.check_version(dist, None, undated)


def test_quickstart_is_the_readmes_block() -> None:
    code = release_check.quickstart(ROOT / "README.md")
    assert code.startswith("import pandas as pd\n")
    assert 'fig.save("weight.svg")' in code
    compile(code, "quickstart.py", "exec")


def test_command_line_entry_point(tmp_path: Path) -> None:
    # The command-line entry point, on the hand-built archives (the workflow runs it
    # on the real ones).
    dist = tmp_path / "dist"
    dist.mkdir()
    _wheel(dist / "inferplot-0.1.0-py3-none-any.whl")
    _sdist(dist / "inferplot-0.1.0.tar.gz", release_check.SDIST_REQUIRED)
    # (The hand-built wheel holds three files, so it is not compared with this
    # repository's source tree; the workflow's real wheel is.)
    none = str(tmp_path / "no-source")
    assert release_check.main(["dists", str(dist), "--source", none]) == 0
    assert release_check.main(["dists", str(dist)]) == 1  # modules missing
    (dist / "inferplot-0.1.0.tar.gz").unlink()
    _sdist(dist / "inferplot-0.1.0.tar.gz", ("README.md",))
    assert release_check.main(["dists", str(dist), "--source", none]) == 1
