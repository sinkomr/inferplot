"""save(): formats, embedded result JSON, editable text, deterministic bytes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest
from PIL import Image

from inferplot.export import FORMATS, METADATA_KEYS
from inferplot.render import betweenstats
from inferplot.result import StatsResult
from tests.render._helpers import (
    case_named,
    default_figure,
    pdf_info_string,
    svg_description,
)
from tests.render._results import fixture_result, many_levels_result

NAMES = ["ToothGrowth", "PlantGrowth-display-all", "ToothGrowth-greater"]


def _same_result(embedded: str, result: StatsResult) -> None:
    assert embedded == result.to_json(indent=None)
    assert StatsResult.from_json(embedded).to_dict() == result.to_dict()


def test_metadata_keys() -> None:
    assert set(FORMATS) == {".svg", ".pdf", ".png"}
    assert METADATA_KEYS == {
        "svg": "Description",
        "pdf": "Subject",
        "png": "Description",
    }


@pytest.mark.parametrize("name", NAMES)
def test_png_round_trip(name: str, tmp_path: Path) -> None:
    fig = default_figure(name)
    path = tmp_path / "f.png"
    fig.save(path, dpi=120)
    with Image.open(path) as image:
        text = image.text  # pyright: ignore[reportAttributeAccessIssue]
        want = [s * 120 for s in fig.figure.get_size_inches()]
        assert all(abs(a - b) < 1 for a, b in zip(image.size, want, strict=True))
    _same_result(text["Description"], fig.result)


@pytest.mark.parametrize("name", NAMES)
def test_svg_round_trip(name: str, tmp_path: Path) -> None:
    fig = default_figure(name)
    path = tmp_path / "f.svg"
    fig.save(path)
    _same_result(svg_description(path), fig.result)


@pytest.mark.parametrize("name", NAMES)
def test_pdf_round_trip_and_truetype_text(name: str, tmp_path: Path) -> None:
    fig = default_figure(name)
    path = tmp_path / "f.pdf"
    fig.save(path)
    data = path.read_bytes()
    _same_result(pdf_info_string(data, "Subject"), fig.result)
    assert b"/CreationDate" not in data
    # pdf.fonttype 42 embeds TrueType fonts (Type 3 would draw glyph procedures).
    assert b"/TrueType" in data or b"/CIDFontType2" in data
    assert b"/Type3" not in data


def test_one_sided_inf_survives_the_round_trip(tmp_path: Path) -> None:
    fig = default_figure("ToothGrowth-greater")
    assert fig.result.effect_size.conf_high == float("inf")
    path = tmp_path / "f.svg"
    fig.save(path)
    assert '"conf_high": "Inf"' in svg_description(path)
    back = StatsResult.from_json(svg_description(path))
    assert back.effect_size.conf_high == float("inf")


@pytest.mark.parametrize("suffix", [".svg", ".pdf", ".png"])
def test_saving_twice_gives_identical_bytes(suffix: str, tmp_path: Path) -> None:
    fig = default_figure("PlantGrowth-display-all")
    first, second = tmp_path / f"a{suffix}", tmp_path / f"b{suffix}"
    fig.save(first, dpi=72)
    fig.save(second, dpi=72)
    assert first.read_bytes() == second.read_bytes()
    # A fresh rendering of the same result writes the same bytes too.
    third = tmp_path / f"c{suffix}"
    betweenstats(fixture_result(case_named("PlantGrowth-display-all"))).save(
        third, dpi=72
    )
    assert third.read_bytes() == first.read_bytes()


def test_svg_has_no_date(tmp_path: Path) -> None:
    path = tmp_path / "f.svg"
    default_figure("ToothGrowth").save(path)
    assert "<dc:date>" not in path.read_text(encoding="utf-8")


def test_uppercase_extension_and_pathlike(tmp_path: Path) -> None:
    fig = default_figure("ToothGrowth")
    fig.save(str(tmp_path / "F.SVG"))
    assert (tmp_path / "F.SVG").read_bytes().startswith(b"<?xml")


@pytest.mark.parametrize("name", ["f.jpg", "f", "f.svgz", "f.eps"])
def test_other_extensions_raise(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.svg, \.pdf or \.png"):
        default_figure("ToothGrowth").save(tmp_path / name)
    assert not (tmp_path / name).exists()


def test_bad_save_arguments(tmp_path: Path) -> None:
    fig = default_figure("ToothGrowth")
    with pytest.raises(ValueError, match="dpi"):
        fig.save(tmp_path / "f.png", dpi=0)
    with pytest.raises(TypeError, match="transparent"):
        fig.save(tmp_path / "f.png", transparent=1)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="path"):
        fig.save(5)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="positional"):
        fig.save(tmp_path / "f.png", 300)  # pyright: ignore[reportCallIssue]


def test_transparent_png(tmp_path: Path) -> None:
    path = tmp_path / "f.png"
    default_figure("ToothGrowth").save(path, dpi=50, transparent=True)
    with Image.open(path) as image:
        assert image.mode == "RGBA"
        corner = image.getpixel((0, 0))
        assert isinstance(corner, tuple)
        assert corner[3] == 0


def test_repr_png() -> None:
    fig = default_figure("ToothGrowth")
    data = fig._repr_png_()  # pyright: ignore[reportPrivateUsage]
    assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_global_rcparams_untouched(tmp_path: Path) -> None:
    before = _raw_rcparams()
    fig = betweenstats(many_levels_result())
    fig.check()
    for suffix in (".svg", ".pdf", ".png"):
        fig.save(tmp_path / f"f{suffix}", dpi=50)
    assert _raw_rcparams() == before


def _raw_rcparams() -> dict[str, object]:
    """rcParams without resolving the backend (reading it would import pyplot)."""
    rc = matplotlib.rcParams
    return {key: dict.__getitem__(rc, key) for key in dict.keys(rc)}


def test_embedded_json_is_the_full_result(tmp_path: Path) -> None:
    fig = default_figure("PlantGrowth-display-all")
    path = tmp_path / "f.svg"
    fig.save(path)
    data = json.loads(svg_description(path))
    assert data["analysis"] == "betweenstats"
    assert len(data["pairwise"]) == 3
    assert data["text"]["subtitle"] == fig.result.text.subtitle
