"""The mathtext inferplot draws parses and shows exactly the glyphs plotmath shows.

Every distinct expression in the fixtures (plus inferplot's v0.1 Bayes captions, the
plain tick labels and every symbol in the tables) is rendered with matplotlib on an
Agg canvas, which raises on any mathtext error, and saved as SVG inside
``rc_context({"svg.fonttype": "none"})`` so text stays text. The glyphs in the SVG
must be exactly those ``tests/plotmath.expected_glyphs`` predicts for the plotmath
expression, as a multiset (the SVG backend groups glyphs by style, so document order
is not reading order). No pyplot.
"""

from __future__ import annotations

import collections
import functools
import html
import io
import re

import matplotlib as mpl
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from inferplot import text as tx
from tests.oracle_fixtures import load_cases
from tests.plotmath import expected_glyphs, parse
from tests.test_text_expressions import checked_expressions, v01_caption
from tests.test_text_format import centrality_rows

# matplotlib 3.11 writes one <tspan x=".." y=".." style=".."> per glyph; 3.8 writes
# runs of same-style glyphs as <tspan style=".." x="x1 x2 .." y="..">run</tspan>.
_TSPAN_RE = re.compile(r"<tspan ([^>]*)>([^<]*)</tspan>")


def _attr(attrs: str, name: str) -> str:
    m = re.search(rf'\b{name}="([^"]*)"', attrs)
    return m.group(1) if m else ""


def _draw(fig: Figure, items: list[tuple[str, str, bool]]) -> None:
    rows = len(items)
    for k, (gid, s, parse_math) in enumerate(items):
        fig.text(
            0.005,
            1 - (k + 0.8) / (rows + 0.3),
            s,
            fontsize=8,
            gid=gid,
            parse_math=parse_math,
            math_fontfamily="dejavusans",
        )


def render_glyphs(items: list[tuple[str, str, bool]]) -> dict[str, str]:
    """Draw (gid, string, parse_math) on Agg, then as SVG; the glyphs per gid."""
    size = (9.0, 0.32 * len(items) + 0.3)
    agg = Figure(figsize=size, dpi=100)
    FigureCanvasAgg(agg)
    _draw(agg, items)
    agg.canvas.draw()  # raises on any mathtext parse error
    buf = io.StringIO()
    with mpl.rc_context({"svg.fonttype": "none"}):
        svg_fig = Figure(figsize=size, dpi=100)
        FigureCanvasSVG(svg_fig)
        _draw(svg_fig, items)
        svg_fig.savefig(buf, format="svg")
    svg = buf.getvalue()
    out: dict[str, str] = {}
    for gid, _, _ in items:
        start = svg.index(f'<g id="{gid}">')
        chunk = svg[start : svg.index("</text>", start)]
        glyphs = [html.unescape(m.group(2)) for m in _TSPAN_RE.finditer(chunk)]
        if not glyphs:  # plain text: one <text> element per line
            group = svg[start : svg.index("</g>", start)]
            glyphs = [
                html.unescape(m.group(1))
                for m in re.finditer(r"<text[^>]*>([^<]*)</text>", group)
            ]
        out[gid] = "".join(glyphs)
    return out


def _normalized(glyphs: str) -> collections.Counter[str]:
    # mathtext writes a space inside \text{} as U+00A0 in the SVG
    return collections.Counter(glyphs.replace(" ", "").replace("\xa0", ""))


def render_items() -> tuple[
    list[tuple[str, str, bool]], dict[str, str], dict[str, str]
]:
    """(items to draw, expected glyphs per gid, source per gid)."""
    sources: dict[str, tx.Node] = {}
    ticks: set[str] = set()
    for case in load_cases():
        if case.function not in ("ggbetweenstats", "ggwithinstats"):
            # gghistostats and ggscatterstats join with their text layers (Phase 4).
            continue
        for _, source, rebuilt in checked_expressions(case):
            sources.setdefault(source, rebuilt)
        if case.type == "parametric":
            caption = v01_caption(case)
            if caption is not None:
                sources.setdefault(f"[v0.1 caption] {case.name}", caption)
        for _, row in centrality_rows(case):
            ticks.add(str(row["n.expression"]))
    symbols: list[tx.Node] = [
        *tx.STATISTIC_SYMBOLS.values(),
        *tx.EFFECT_SIZE_SYMBOLS.values(),
        *tx.PRIOR_SYMBOLS.values(),
    ]
    sources["[all symbols]"] = tx.Call(tx.Sym("list"), tuple(symbols))
    sources["[escapes]"] = tx.Str(r"a\b {c} $d %e #f &g _h ^i ~j")
    for method in ("hochberg", "hommel", "bonferroni", "BH", "BY", "none"):
        sources[f"[p label {method}]"] = tx.pairwise_p_expression(
            p_adjusted=0.0123, adjust_method=method
        )
    items: list[tuple[str, str, bool]] = []
    expected: dict[str, str] = {}
    where: dict[str, str] = {}
    for k, (source, node) in enumerate(sorted(sources.items())):
        gid = f"e{k}"
        items.append((gid, tx.to_mathtext(node), True))
        expected[gid] = expected_glyphs(node)
        where[gid] = source
    for k, label in enumerate(sorted(ticks)):
        gid = f"t{k}"
        items.append((gid, label, False))
        expected[gid] = label.replace("\n", "")
        where[gid] = repr(label)
    return items, expected, where


_PAGE = 50


@functools.cache
def _all_items() -> tuple[list[tuple[str, str, bool]], dict[str, str], dict[str, str]]:
    return render_items()


def _page_params() -> list[int]:
    items, _, _ = _all_items()
    return list(range(0, len(items), _PAGE))


def test_every_fixture_expression_is_rendered() -> None:
    items, expected, _ = _all_items()
    assert len(items) > 250
    assert set(expected) == {gid for gid, _, _ in items}


@pytest.mark.parametrize("start", _page_params())
def test_mathtext_renders_with_plotmath_glyphs(start: int) -> None:
    items, expected, where = _all_items()
    page = items[start : start + _PAGE]
    glyphs = render_glyphs(page)
    failures = []
    for gid, _, _ in page:
        got, wanted = _normalized(glyphs[gid]), _normalized(expected[gid])
        if got != wanted:
            failures.append(
                f"{where[gid]}: missing {wanted - got}, extra {got - wanted}"
            )
    assert not failures


def test_parse_errors_would_be_caught() -> None:
    """A malformed mathtext string raises on the Agg draw (the check is real)."""
    with pytest.raises(ValueError):  # noqa: PT011 - matplotlib's parse error
        render_glyphs([("bad", "$\\mathit{$", True)])


def test_text_module_never_imports_matplotlib() -> None:
    import subprocess
    import sys

    code = (
        "import sys, inferplot.text, inferplot.result, inferplot.check; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_expected_glyphs_rules() -> None:
    node = parse('list(italic(p)["Holm" - adj.] == "0.47")')
    assert expected_glyphs(node) == "pHolm\u2212adj.=0.47"
    node = parse('list(widehat(mu)[mean] == "5.03")')
    assert expected_glyphs(node) == "\u0302\u03bcmean=5.03"
