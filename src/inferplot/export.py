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
from inferplot.render.figure import root_figure

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

    Raises:
        ValueError: the extension is not .svg, .pdf or .png, or ``dpi`` is not a
            positive number.
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
    dots = v.check_positive(dpi, "dpi")
    clear = v.check_bool(transparent, "transparent")
    _write(fig, path, FORMATS[suffix], dpi=dots, transparent=clear)


def png_bytes(fig: StatsFigure, *, dpi: float = 100) -> bytes:
    """The figure as PNG bytes (for notebooks), with the result embedded."""
    buffer = io.BytesIO()
    _write(fig, buffer, "png", dpi=dpi, transparent=False)
    return buffer.getvalue()
