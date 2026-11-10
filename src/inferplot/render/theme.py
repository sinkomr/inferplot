"""inferplot's look: palette, fonts, sizes and the rcParams used while drawing.

The look is ours (DECISIONS.md D19); only the content is ggstatsplot's. Global
``rcParams`` are never changed: :func:`context` applies matplotlib's defaults plus
:data:`RC` inside ``matplotlib.style.context`` (an ``rc_context``), so a user's
global style cannot leak into an inferplot figure.
"""

from __future__ import annotations

import difflib
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final, cast

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.style
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

__all__ = [
    "FONT_FAMILY",
    "MATH_FONT_FAMILY",
    "OKABE_ITO",
    "RC",
    "Color",
    "context",
    "font",
    "resolve_palette",
]

Color = tuple[float, float, float, float]

# The font every text uses; DejaVu Sans ships with matplotlib, so every machine
# renders the same glyphs.
FONT_FAMILY: Final = "DejaVu Sans"
MATH_FONT_FAMILY: Final = "dejavusans"

# Okabe & Ito (2008), "Color Universal Design": orange, sky blue, bluish green,
# yellow, blue, vermillion, reddish purple, black (black last so that up to seven
# levels never get it).
OKABE_ITO: Final = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)
MANY_LEVELS_COLORMAP: Final = "tab20"

INK: Final = "#222222"
SOFT_INK: Final = "#4D4D4D"
LIGHT_INK: Final = "#EEEEEE"  # inferplot's ink on a dark background (ax=)
CENTRALITY_COLOR: Final = "#8B0000"  # ggstatsplot's "darkred" centrality point

RC: Final[dict[str, object]] = {
    "font.family": FONT_FAMILY,
    "mathtext.fontset": MATH_FONT_FAMILY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": SOFT_INK,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.7,
    "xtick.color": SOFT_INK,
    "ytick.color": SOFT_INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "text.color": INK,
    "text.usetex": False,
}

# Font sizes in points.
SIZE_TITLE: Final = 13.0
SIZE_SUBTITLE: Final = 10.5
SIZE_CAPTION: Final = 9.0
SIZE_AXIS_LABEL: Final = 11.0
SIZE_TICK: Final = 10.0
SIZE_LABEL: Final = 9.0  # centrality and bracket labels


@contextmanager
def context() -> Generator[None]:
    """matplotlib's defaults plus :data:`RC`, restored on exit."""
    with matplotlib.style.context(["default", RC]):
        yield


# The user's fallback fonts for the drawing in progress (see fallback_fonts).
_FALLBACKS: ContextVar[tuple[str, ...]] = ContextVar(
    "inferplot_fallback_fonts", default=()
)
# A family name every SVG viewer knows, after the named fonts (matplotlib writes
# it into the SVG's font-family lists; for drawing it resolves to DejaVu Sans
# inside :func:`context`).
GENERIC_FAMILY: Final = "sans-serif"


def _installed(name: str) -> bool:
    """Whether matplotlib finds a font of family ``name`` (without falling back)."""
    try:
        manager = cast("Any", font_manager)
        manager.findfont(FontProperties(family=name), fallback_to_default=False)
    except ValueError:
        return False
    return True


def user_fallbacks() -> tuple[str, ...]:
    """The installed families of the user's ``font.sans-serif`` rcParam that
    are not in matplotlib's default list, in their order.

    Read from the global rcParams (never changed) before :func:`context` resets
    them. matplotlib's default list is left out: its fonts are Latin ones, and
    naming whichever of them a machine has would make saved files differ by
    machine.
    """
    names = cast("list[str]", matplotlib.rcParams.get("font.sans-serif", []))
    default = set(cast("list[str]", matplotlib.rcParamsDefault["font.sans-serif"]))
    found: list[str] = []
    for name in names:
        if name in default or name in font_manager.font_family_aliases:
            continue
        if name not in found and name != FONT_FAMILY and _installed(name):
            found.append(name)
    return tuple(found)


@contextmanager
def fallback_fonts(names: tuple[str, ...]) -> Generator[None]:
    """Use ``names`` after DejaVu Sans in :func:`font` and :func:`families`."""
    token = _FALLBACKS.set(names)
    try:
        yield
    finally:
        _FALLBACKS.reset(token)


def families() -> list[str]:
    """The font families of every text inferplot draws: DejaVu Sans, then the
    user's fallback fonts (matplotlib's font fallback draws a character missing
    from DejaVu Sans with the first that has it), then a generic family."""
    return [FONT_FAMILY, *_FALLBACKS.get(), GENERIC_FAMILY]


def font(size: float, *, weight: str = "normal") -> FontProperties:
    """DejaVu Sans (then :func:`families`' fallbacks) at ``size`` points, with
    DejaVu Sans for mathtext."""
    return FontProperties(
        family=families(), size=size, weight=weight, math_fontfamily=MATH_FONT_FAMILY
    )


def luminance(color: object) -> float:
    """Relative luminance of an opaque colour (WCAG 2.1, "relative luminance")."""
    rgb = mcolors.to_rgb(color)  # pyright: ignore[reportArgumentType]

    def channel(u: float) -> float:
        return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(float(u)) for u in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: object, b: object) -> float:
    """WCAG 2.1 contrast ratio of two opaque colours, from 1 to 21."""
    la, lb = sorted((luminance(a), luminance(b)))
    return (lb + 0.05) / (la + 0.05)


def over(color: object, background: object) -> tuple[float, float, float]:
    """``color`` (with its alpha) composited over an opaque ``background``."""
    r, g, b, a = mcolors.to_rgba(color)  # pyright: ignore[reportArgumentType]
    br, bg, bb = mcolors.to_rgb(background)  # pyright: ignore[reportArgumentType]
    return (r * a + br * (1 - a), g * a + bg * (1 - a), b * a + bb * (1 - a))


def ink_for(background: object) -> str:
    """inferplot's ink for text and lines on ``background``: :data:`INK`, or
    :data:`LIGHT_INK` where that has the higher contrast (a dark background)."""
    if contrast(LIGHT_INK, background) > contrast(INK, background):
        return LIGHT_INK
    return INK


def _rgba(color: object, where: str) -> Color:
    try:
        return cast("Color", tuple(float(c) for c in mcolors.to_rgba(color)))  # pyright: ignore[reportArgumentType]
    except (ValueError, TypeError):
        raise ValueError(
            f"palette contains {color!r} at {where}, which is not a matplotlib color; "
            "use names such as 'tab:blue' or hex strings such as '#0072B2'."
        ) from None


def _colormap_colors(name: str, k: int) -> list[Color]:
    try:
        cmap = matplotlib.colormaps[name]
    except KeyError:
        close = difflib.get_close_matches(name, list(matplotlib.colormaps), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"palette={name!r} is not a matplotlib colormap name. Pass a colormap "
            f"name such as 'viridis' or 'Set2', or a list of colors such as "
            f"['#E69F00', '#56B4E9'].{hint}"
        ) from None
    if isinstance(cmap, mcolors.ListedColormap) and cmap.N < 64:
        # A qualitative colormap: use its colors in order.
        if k > cmap.N:
            raise ValueError(
                f"palette={name!r} has {cmap.N} colors but the plot has {k} levels; "
                "choose a colormap with more colors or pass a list of colors."
            )
        return [_rgba(cmap(i), f"index {i}") for i in range(k)]
    # A continuous colormap: k colors at the centres of k equal bins, (i + 0.5) / k,
    # so that no level gets an end of the map (the near-white end of "Blues").
    return [_rgba(cmap((i + 0.5) / k), f"{(i + 0.5) / k:.3f}") for i in range(k)]


def resolve_palette(palette: object, k: int) -> tuple[list[Color], str | None]:
    """``k`` colors for ``palette`` and a warning when the default had to change.

    ``palette`` is None (Okabe-Ito; more than 8 levels use ``tab20``), a matplotlib
    colormap name, or a sequence of at least ``k`` matplotlib colors.
    """
    if palette is None:
        if k <= len(OKABE_ITO):
            return [_rgba(c, f"index {i}") for i, c in enumerate(OKABE_ITO[:k])], None
        cmap = matplotlib.colormaps[MANY_LEVELS_COLORMAP]
        colors = [_rgba(cmap(i % cmap.N), f"index {i}") for i in range(k)]
        repeat = f"; with more than {cmap.N} levels colors repeat" if k > cmap.N else ""
        warning = (
            f"The plot has {k} levels, more than the {len(OKABE_ITO)} colors of the "
            f"default Okabe-Ito palette, so its colors come from matplotlib's "
            f"'{MANY_LEVELS_COLORMAP}' colormap{repeat}. Pass palette= to choose "
            "your own colors."
        )
        return colors, warning
    if isinstance(palette, str):
        return _colormap_colors(palette, k), None
    if isinstance(palette, Sequence):
        items = list(cast("Sequence[object]", palette))
        if len(items) < k:
            raise ValueError(
                f"palette has {len(items)} colors but the plot has {k} levels; pass "
                f"at least {k} colors, or a colormap name such as 'viridis'."
            )
        # Every entry is checked, also those beyond the k used.
        colors = [_rgba(c, f"index {i}") for i, c in enumerate(items)]
        return colors[:k], None
    raise TypeError(
        "palette must be None, a matplotlib colormap name or a list of colors; got "
        f"{type(palette).__name__}."
    )
