# matplotlib's and NumPy's stubs type many ``**kwargs`` and overloads as Unknown,
# which strict mode reports on every call; the arguments we pass are all typed.
# pyright: reportUnknownMemberType=false
"""Write a StatsFigure to .svg, .pdf or .png with its result embedded.

Text stays editable: SVG text is written as ``<text>`` elements
(``svg.fonttype="none"``) and PDF fonts as TrueType (``pdf.fonttype=42``). The
result's JSON (:meth:`~inferplot.result.StatsResult.to_json`, compact, without the
analyzed observations: they can be large and are the user's data) is embedded as
the PNG text chunk "Description", the SVG metadata "Description" and the PDF
"Subject". Output is deterministic: a fixed ``svg.hashsalt`` and no creation date,
so saving twice gives identical bytes. These settings apply inside an
``rc_context`` only; global ``rcParams`` are never changed.
"""

from __future__ import annotations

import io
import os
from contextlib import ExitStack
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Final, Literal, cast

import matplotlib

from inferplot import _validate as v
from inferplot.render import theme
from inferplot.render.figure import reported_glyphs, root_figure

if TYPE_CHECKING:
    from inferplot.render.figure import StatsFigure

__all__ = ["FORMATS", "METADATA_KEYS", "png_bytes", "save"]

Format = Literal["svg", "pdf", "png"]

FORMATS: Final[dict[str, Format]] = {".svg": "svg", ".pdf": "pdf", ".png": "png"}
# The metadata key that holds the result's JSON, per format.
METADATA_KEYS: Final[dict[Format, str]] = {
    "svg": "Description",
    "pdf": "Subject",
    "png": "Description",
}
# The largest dpi save() accepts, and the most pixels it writes as PNG (an RGBA
# buffer of 400 MB).
MAX_DPI: Final = 2_400
MAX_PNG_PIXELS: Final = 100_000_000
# The least dpi save() writes as PNG: FreeType fails on inferplot's figures
# below 7 dpi ("invalid ppem value"), and at 10 the texts are a few pixels high.
MIN_PNG_DPI: Final = 10
SAVE_RC: Final[dict[str, object]] = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "svg.hashsalt": "inferplot",
}


def _metadata(fig: StatsFigure, fmt: Format) -> dict[str, str | None]:
    # Every number and text, without the analyzed observations (DECISIONS D36).
    payload = fig.result.to_json(indent=None, include_values=False)
    metadata: dict[str, str | None] = {METADATA_KEYS[fmt]: payload}
    if fmt == "svg":
        metadata["Date"] = None
    elif fmt == "pdf":
        metadata["CreationDate"] = None
    return metadata


def _write(
    fig: StatsFigure,
    target: str | os.PathLike[str] | IO[bytes],
    fmt: Format,
    *,
    dpi: float,
    transparent: bool,
) -> None:
    with ExitStack() as stack:
        stack.enter_context(reported_glyphs())
        if fig.own:
            stack.enter_context(theme.context())
        stack.enter_context(matplotlib.rc_context(cast("Any", SAVE_RC)))
        root_figure(fig.figure).savefig(
            target,
            format=fmt,
            dpi=dpi,
            transparent=transparent,
            metadata=_metadata(fig, fmt),
        )


def save(
    fig: StatsFigure,
    path: str | os.PathLike[str],
    *,
    dpi: float = 300,
    transparent: bool = False,
) -> None:
    """Write ``fig`` to ``path``; the extension (.svg, .pdf, .png) picks the format.

    The figure is not pyplot's current figure, so saving with pyplot after an
    inferplot call writes pyplot's own, empty figure; save with this function or
    ``fig.save(path)``.

    Raises:
        ValueError: the extension is not .svg, .pdf or .png, ``dpi`` is not a
            positive number or is above MAX_DPI, or a PNG is asked for below
            MIN_PNG_DPI or would have more than MAX_PNG_PIXELS pixels.
        TypeError: ``path`` is not a string or path, or ``transparent`` not a bool.
    """
    if not isinstance(path, str | os.PathLike):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"path must be a file name such as 'figure.svg'; got {type(path).__name__}."
        )
    suffix = Path(path).suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(
            f"save() writes .svg, .pdf or .png files; got {os.fspath(path)!r}. Use "
            "one of these extensions (.svg or .pdf keep text editable)."
        )
    dots = v.check_positive(dpi, "dpi", example="300")
    clear = v.check_bool(transparent, "transparent")
    _check_dpi(fig, dots, dpi, FORMATS[suffix])
    _write(fig, path, FORMATS[suffix], dpi=dots, transparent=clear)


def _check_dpi(fig: StatsFigure, dots: float, given: object, fmt: Format) -> None:
    """``dpi`` at most MAX_DPI, and a PNG of at most MAX_PNG_PIXELS pixels.

    Larger values make matplotlib allocate an image of gigabytes and fail with a
    MemoryError (or Agg's size error) after a long wait.
    """
    if dots > MAX_DPI:
        raise ValueError(
            f"dpi={given!r} is more than {MAX_DPI:,}, the most save() writes; "
            "journals ask for 300 to 1,200. Pass a smaller dpi, or save as .svg or "
            ".pdf, whose text and lines do not depend on dpi."
        )
    if fmt != "png":
        return
    if dots < MIN_PNG_DPI:
        raise ValueError(
            f"dpi={given!r} is less than {MIN_PNG_DPI}, the least save() writes as "
            "PNG: below about 7 dpi the font renderer (FreeType) cannot draw the "
            "texts at all, and below 10 they are unreadable. Pass dpi=10 or more "
            "(300 for print), or save as .svg or .pdf, which do not depend on dpi."
        )
    width, height = (float(s) for s in root_figure(fig.figure).get_size_inches())
    pixels = width * dots * height * dots
    if pixels > MAX_PNG_PIXELS:
        most = int((MAX_PNG_PIXELS / (width * height)) ** 0.5)
        raise ValueError(
            f"At dpi={given!r} this {width:.1f} x {height:.1f} in figure would be a "
            f"PNG of {pixels / 1e6:,.0f} million pixels, more than the "
            f"{MAX_PNG_PIXELS / 1e6:,.0f} million save() writes. Pass dpi={most} or "
            "less, or save as .svg or .pdf, which do not depend on dpi."
        )


def png_bytes(fig: StatsFigure, *, dpi: float = 100) -> bytes:
    """The figure as PNG bytes (for notebooks), with the result embedded."""
    buffer = io.BytesIO()
    _write(fig, buffer, "png", dpi=dpi, transparent=False)
    return buffer.getvalue()
