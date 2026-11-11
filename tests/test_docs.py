"""The user documentation is kept true by these tests.

Every fenced code block in README.md, docs/api.md, docs/intervals.md and
skills/inferplot/SKILL.md is checked, by its info string:

* ``python``: executed, the blocks of one document in order in one namespace, in
  a temporary folder. A ``text`` block right after it is its output, compared
  with what the block prints.
* ``python signature``: a function signature, compared with the function's own.
* ``sh``: install commands; each line must be one of the known commands, with
  extras that pyproject.toml defines and the project's own repository URL.

docs/api.md must stay under 32 KB and name every public function, keyword,
``check()`` code and result field (read from the code, so it cannot drift).
llms.txt and the skill's front matter are checked for their formats, and every
relative link in the documents must point to a file in the repository.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import io
import re
import runpy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import inferplot as ip
from inferplot.render.figure import StatsFigure
from inferplot.result import _FRAME_COLUMNS, SCHEMA_VERSION, StatsResult, Texts

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "docs" / "api.md"
SKILL = ROOT / "skills" / "inferplot" / "SKILL.md"
LLMS = ROOT / "llms.txt"
DOCUMENTS = (
    "README.md",
    "docs/api.md",
    "docs/intervals.md",
    "skills/inferplot/SKILL.md",
)
FUNCTIONS = ("betweenstats", "withinstats", "histostats", "scatterstats", "barstats")
API_LIMIT = 32 * 1024

# The data frames docs/api.md and SKILL.md name in their examples (both say so).
PRELUDE = """
import pandas as pd
import inferplot as ip

_weights = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
            4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
            6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]
df = pd.DataFrame({"group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10,
                   "score": _weights})
_extra = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0,
          1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
long = pd.DataFrame({"subject": [f"s{i}" for i in range(1, 11)] * 2,
                     "time": ["pre"] * 10 + ["post"] * 10, "score": _extra})
cars = pd.DataFrame({
    "wt": [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
           3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835],
    "mpg": [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2,
            17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9]})
answers = pd.DataFrame({
    "answer": ["yes"] * 14 + ["no"] * 6 + ["yes"] * 7 + ["no"] * 13,
    "group": ["a"] * 20 + ["b"] * 20,
    "before": ["yes"] * 12 + ["no"] * 8 + ["yes"] * 9 + ["no"] * 11,
    "after": ["yes"] * 18 + ["no"] * 2 + ["yes"] * 15 + ["no"] * 5})
"""

_FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_LINK = re.compile(r"\]\(([^)\s]+)\)")


@dataclass(frozen=True)
class Block:
    info: str
    body: str
    line: int

    @property
    def language(self) -> str:
        return self.info.split()[0] if self.info.split() else ""

    @property
    def is_signature(self) -> bool:
        return self.info.split()[1:] == ["signature"]


def blocks(path: Path) -> list[Block]:
    text = path.read_text(encoding="utf-8")
    return [
        Block(m.group(1).strip(), m.group(2), text.count("\n", 0, m.start()) + 1)
        for m in _FENCE.finditer(text)
    ]


def _run_document(name: str, folder: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """Run every python block of a document; compare each output block."""
    monkeypatch.chdir(folder)
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    if name != "README.md":  # the README runs as written, with no setup
        exec(PRELUDE, namespace)
    found = blocks(ROOT / name)
    ran = 0
    for i, block in enumerate(found):
        where = f"{name}:{block.line}"
        if block.language == "text":
            before = found[i - 1] if i else None
            assert before is not None, f"{where}: an output block follows no code"
            assert before.language == "python", f"{where}: output after non-Python"
            assert not before.is_signature, f"{where}: output after a signature"
            continue
        if block.language != "python" or block.is_signature:
            continue
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(block.body, where, "exec"), namespace)
        ran += 1
        after = found[i + 1] if i + 1 < len(found) else None
        if after is not None and after.language == "text":
            assert printed.getvalue().rstrip() == after.body.rstrip(), where
    return ran


@pytest.mark.parametrize("name", DOCUMENTS)
def test_code_blocks_run_and_print_what_the_document_shows(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = _run_document(name, tmp_path, monkeypatch)
    if name in ("README.md", "skills/inferplot/SKILL.md", "docs/api.md"):
        assert ran > 0


@pytest.mark.parametrize("name", DOCUMENTS)
def test_every_block_has_a_known_kind(name: str) -> None:
    for block in blocks(ROOT / name):
        assert block.info in ("python", "python signature", "text", "sh"), (
            f"{name}:{block.line}: code block {block.info!r} is neither run nor "
            "compared with output"
        )


def test_readme_quick_start_is_under_40_lines() -> None:
    code = [b for b in blocks(ROOT / "README.md") if b.language == "python"]
    assert len(code[0].body.splitlines()) < 40
    assert "fig.check()" in code[0].body
    assert "fig.save(" in code[0].body
    assert "fig.result.text.apa" in code[0].body


# --- install commands ------------------------------------------------------------


def test_install_commands_use_this_project() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(project["project"]["optional-dependencies"])
    homepage = project["project"]["urls"]["Homepage"]
    commands = [
        line.strip()
        for name in DOCUMENTS
        for block in blocks(ROOT / name)
        if block.language == "sh"
        for line in block.body.splitlines()
        if line.strip()
    ]
    assert commands
    for command in commands:
        assert re.match(r"(git clone|cd|pip install|uv pip install|uv sync) ", command)
        for url in re.findall(r"https?://\S+", command):
            assert url == homepage, command
        for group in re.findall(r"\[([a-z,]+)\]", command):
            assert set(group.split(",")) <= extras, command
        if command.startswith("cd "):
            assert command == "cd " + homepage.rsplit("/", 1)[1]


# --- signatures and coverage of docs/api.md --------------------------------------


def _signature_text(qualified: str, fn: Any) -> str:
    parts: list[str] = []
    star = False
    for p in inspect.signature(fn).parameters.values():
        if p.kind is p.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        parts.append(p.name if p.default is p.empty else f"{p.name}={p.default!r}")
    return f"{qualified}({', '.join(parts)})"


def _squash(text: str) -> str:
    return "".join(text.split())


def test_api_signatures_are_the_functions_own() -> None:
    shown = {
        _squash(b.body).split("(", 1)[0]: _squash(b.body)
        for b in blocks(API)
        if b.is_signature
    }
    expected: dict[str, str] = {}
    for name in FUNCTIONS:
        for module, prefix in ((ip.analyze, "ip.analyze."), (ip.render, "ip.render.")):
            fn = getattr(module, name)
            expected[prefix + name] = _squash(_signature_text(prefix + name, fn))
    assert shown == expected


@pytest.mark.parametrize("name", FUNCTIONS)
def test_one_call_takes_the_analysis_then_the_drawing_keywords(name: str) -> None:
    # docs/api.md says so instead of printing a third signature per function.
    def params(fn: Any) -> list[tuple[str, object]]:
        return [(p.name, p.default) for p in inspect.signature(fn).parameters.values()]

    render = params(getattr(ip.render, name))
    assert render[0][0] == "result"
    combined = params(getattr(ip.analyze, name)) + render[1:]
    assert params(getattr(ip, name)) == combined


def _api_text() -> str:
    return API.read_text(encoding="utf-8")


def test_api_is_under_32_kb() -> None:
    size = len(API.read_bytes())
    assert size < API_LIMIT, f"docs/api.md is {size:,} bytes; the limit is 32 KB"


def test_api_describes_every_keyword_in_one_table_row() -> None:
    rows = {
        m.group(1)
        for m in re.finditer(r"^\| `([a-z_]+)` \|", _api_text(), re.MULTILINE)
    }
    names = {
        p
        for fn_name in FUNCTIONS
        for module in (ip, ip.analyze, ip.render)
        for p in inspect.signature(getattr(module, fn_name)).parameters
    }
    assert names - rows == set()


def _check_codes() -> set[str]:
    source = ROOT / "src" / "inferplot"
    return {
        code
        for path in source.rglob("*.py")
        for code in re.findall(r'code="([a-z_]+)"', path.read_text(encoding="utf-8"))
    }


def test_api_lists_every_check_code() -> None:
    codes = _check_codes()
    assert len(codes) >= 25
    rows = set(re.findall(r"^\| `([a-z_]+)` \| (?:info|warning)", _api_text(), re.M))
    assert codes - rows == set()
    assert rows - codes == set()  # no code the package no longer emits


def test_api_names_every_result_field_and_figure_member() -> None:
    text = _api_text()
    names = [f.name for f in dataclasses.fields(StatsResult)]
    names += [f.name for f in dataclasses.fields(Texts)]
    names += [c for c, _ in _FRAME_COLUMNS]
    names += ["to_dict", "to_json", "from_dict", "from_json", "to_frame", "check"]
    names += ["figure", "axes", "result", "save", "schema_version"]
    missing = [n for n in names if not re.search(rf"[`.]{n}[`(]", text)]
    assert missing == []
    assert f"`schema_version` ({SCHEMA_VERSION})" in text
    assert hasattr(StatsFigure, "save")
    assert hasattr(StatsFigure, "check")


def test_api_names_every_public_function() -> None:
    text = _api_text()
    for name in FUNCTIONS:
        for prefix in ("ip.", "ip.analyze.", "ip.render."):
            assert f"{prefix}{name}" in text
    for name in ip.__all__:
        assert f"`{name}`" in text or f"ip.{name}" in text, name


# --- links, llms.txt, the skill ----------------------------------------------------


@pytest.mark.parametrize("name", [*DOCUMENTS, "llms.txt"])
def test_relative_links_point_to_files(name: str) -> None:
    path = ROOT / name
    for target in _LINK.findall(path.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        assert (path.parent / target.split("#")[0]).exists(), f"{name}: {target}"


def test_llms_txt_format() -> None:
    lines = LLMS.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# inferplot"
    summary = next(line for line in lines[1:] if line.strip())
    assert summary.startswith("> ")
    sections = [line for line in lines if line.startswith("## ")]
    assert sections
    links = [line for line in lines if line.startswith("- [")]
    targets = {re.match(r"- \[[^\]]+\]\(([^)]+)\): \S", line) for line in links}
    assert None not in targets
    named = {m.group(1) for m in targets if m is not None}
    for required in (
        "README.md",
        "docs/api.md",
        "docs/intervals.md",
        "docs/examples/",
        "skills/inferplot/SKILL.md",
    ):
        assert required in named


def test_skill_front_matter() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front, _, body = text[4:].partition("\n---\n")
    fields = dict(line.split(": ", 1) for line in front.splitlines())
    assert fields["name"] == "inferplot"
    description = fields["description"]
    assert 50 < len(description) <= 1024
    for word in ("paired", "correlation", "contingency", "one-sample"):
        assert word in description
    assert len(body.split()) < 2000
    for rule in ("subject_id", "paired=True", "fig.check()", "warnings", "fig.save("):
        assert rule in body


# --- the README's figures ------------------------------------------------------------


def _embedded(path: Path) -> StatsResult:
    with Image.open(path) as image:
        text: dict[str, str] = image.text  # type: ignore[attr-defined]
    return StatsResult.from_json(text["Description"])


def _make_readme_images(folder: Path) -> None:
    script = runpy.run_path(str(ROOT / "docs" / "examples" / "readme_images.py"))
    with contextlib.redirect_stdout(io.StringIO()):
        script["main"](folder)


def test_readme_images_show_the_current_results(tmp_path: Path) -> None:
    _make_readme_images(tmp_path)
    for name in FUNCTIONS:
        # The committed image shows the numbers the code computes now (its
        # pixels may differ by platform; its embedded texts may not).
        committed = _embedded(ROOT / "docs" / "img" / f"{name}.png").text
        assert committed == _embedded(tmp_path / f"{name}.png").text, name


@pytest.mark.slow
def test_readme_images_are_reproducible_byte_for_byte(tmp_path: Path) -> None:
    # On one platform and set of versions, as the README says.
    _make_readme_images(tmp_path / "first")
    _make_readme_images(tmp_path / "second")
    for name in FUNCTIONS:
        first = (tmp_path / "first" / f"{name}.png").read_bytes()
        assert first == (tmp_path / "second" / f"{name}.png").read_bytes(), name


def test_save_advice_holds(tmp_path: Path) -> None:
    # README.md and docs/api.md: save with fig.save(), not with pyplot; pyplot
    # writes its own, empty figure, and fig.figure.savefig() writes the text as
    # outlines without the result (STATUS V11).
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.DataFrame(
        {"g": ["a"] * 5 + ["b"] * 5, "y": [1.0, 2, 3, 4, 5, 3, 4, 5, 6, 8]}
    )
    fig = ip.betweenstats(data, "g", "y")
    plt.switch_backend("agg")  # headless, whatever the machine's default
    try:
        plt.savefig(tmp_path / "pyplot.svg")
    finally:
        plt.close("all")
    fig.figure.savefig(tmp_path / "figure.svg")
    fig.save(tmp_path / "save.svg")
    pyplot, own, saved = (
        (tmp_path / f"{n}.svg").read_text(encoding="utf-8")
        for n in ("pyplot", "figure", "save")
    )
    assert "Welch" not in pyplot
    assert "<text" not in pyplot
    assert "<text" not in own
    assert "schema_version" not in own
    assert "<text" in saved
    assert "schema_version" in saved
