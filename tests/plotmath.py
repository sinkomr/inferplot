"""Test-only parser for deparsed R plotmath, producing ``inferplot.text`` AST nodes.

The oracle fixtures store ggstatsplot's expressions as deparsed R
(``list(italic("t")["Welch"] * "(" * 55.31 * ")" == "1.92", ...)``). This parser
reads them with R's operator precedence into the same node types the production
builders return, so text parity is plain AST equality.

Precedence, low to high: ``~``; ``==`` (and the other comparisons); ``+ -``;
``* /``; unary ``+ -``; ``^`` (right-associative); calls ``f(...)`` and subscripts
``x[...]``. Unary ``+`` and ``-`` do not occur in statsExpressions' templates and
raise; a leading unary ``~`` (ggbarstats' bar labels) is the call `~`(x).

``expected_glyphs`` predicts the characters plotmath draws for an expression, the
multiset the mathtext rendering's SVG must contain.
"""

from __future__ import annotations

import re

from inferplot.text import BinOp, Call, Index, Node, Num, Paren, Str, Sym

__all__ = ["expected_glyphs", "parse", "skeleton", "tokenize"]

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?L?)
  | (?P<name>[A-Za-z.][A-Za-z0-9._]*)
  | (?P<op>==|!=|<=|>=|[~*/+\-^()\[\],<>])
    """,
    re.VERBOSE,
)


def _unescape(raw: str) -> str:
    return re.sub(
        r"\\(.)",
        lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)),
        raw[1:-1],
    )


def tokenize(src: str) -> list[tuple[str, str]]:
    pos, out = 0, []
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if m is None:
            raise SyntaxError(f"cannot tokenize at {pos}: {src[pos : pos + 20]!r}")
        pos = m.end()
        kind = m.lastgroup
        assert kind is not None
        if kind != "ws":
            out.append((kind, m.group()))
    return out


class _Parser:
    def __init__(self, src: str) -> None:
        self.toks = tokenize(src)
        self.i = 0

    def peek(self) -> tuple[str, str] | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, value: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None or (value is not None and tok[1] != value):
            raise SyntaxError(f"expected {value!r}, got {tok!r}")
        self.i += 1
        return tok

    def at(self, *values: str) -> bool:
        tok = self.peek()
        return tok is not None and tok[0] == "op" and tok[1] in values

    def parse(self) -> Node:
        node = self.tilde()
        if self.peek() is not None:
            raise SyntaxError(f"trailing tokens: {self.toks[self.i :]}")
        return node

    def tilde(self) -> Node:
        if self.at("~"):
            # A leading ``~x`` is R's call `~`(x) (ggbarstats' bar labels,
            # ``list(~italic(p)=='0.09')``); plotmath draws a space before x.
            self.take()
            node: Node = Call(Sym("~"), (self.comparison(),))
        else:
            node = self.comparison()
        while self.at("~"):
            self.take()
            node = BinOp("~", node, self.comparison())
        return node

    def comparison(self) -> Node:
        node = self.additive()
        if self.at("==", "!=", "<", ">", "<=", ">="):
            op = self.take()[1]
            node = BinOp(op, node, self.additive())
        return node

    def additive(self) -> Node:
        node = self.multiplicative()
        while self.at("+", "-"):
            op = self.take()[1]
            node = BinOp(op, node, self.multiplicative())
        return node

    def multiplicative(self) -> Node:
        node = self.unary()
        while self.at("*", "/"):
            op = self.take()[1]
            node = BinOp(op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.at("-", "+"):
            raise SyntaxError(
                "unary operators are not part of statsExpressions' templates"
            )
        return self.power()

    def power(self) -> Node:
        node = self.postfix()
        if self.at("^"):
            self.take()
            node = BinOp("^", node, self.unary())  # right-associative
        return node

    def postfix(self) -> Node:
        node = self.primary()
        while self.at("(", "["):
            if self.take()[1] == "(":
                args: list[Node] = []
                if not self.at(")"):
                    args.append(self.tilde())
                    while self.at(","):
                        self.take()
                        args.append(self.tilde())
                self.take(")")
                node = Call(node, tuple(args))
            else:
                idx = self.tilde()
                self.take("]")
                node = Index(node, idx)
        return node

    def primary(self) -> Node:
        kind, value = self.take()
        if kind == "str":
            return Str(_unescape(value))
        if kind == "num":
            return Num(value)
        if kind == "name":
            return Sym(value)
        if value == "(":
            inner = self.tilde()
            self.take(")")
            return Paren(inner)
        raise SyntaxError(f"unexpected token {value!r}")


def parse(src: str) -> Node:
    """Parse one deparsed plotmath expression."""
    return _Parser(src).parse()


_GREEK_GLYPHS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",  # noqa: RUF001
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",  # noqa: RUF001
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",  # noqa: RUF001
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "ϕ", "chi": "χ",  # noqa: RUF001
    "psi": "ψ", "omega": "ω",
}  # fmt: skip


def expected_glyphs(node: Node) -> str:
    """The characters plotmath draws, in reading order, spaces omitted.

    A binary minus is U+2212; the hat of ``widehat`` is the combining U+0302 that
    mathtext emits (drawn from a STIX size-variant font).
    """
    if isinstance(node, Str):
        return node.value.replace(" ", "")
    if isinstance(node, Num):
        return node.token
    if isinstance(node, Sym):
        return _GREEK_GLYPHS.get(node.name, node.name)
    if isinstance(node, Index):
        return expected_glyphs(node.base) + expected_glyphs(node.index)
    if isinstance(node, BinOp):
        sep = {"*": "", "~": "", "==": "=", "^": "", "-": "\u2212", "+": "+"}[node.op]
        return expected_glyphs(node.lhs) + sep + expected_glyphs(node.rhs)
    if isinstance(node, Paren):
        return "(" + expected_glyphs(node.inner) + ")"
    fn = node.fn
    args = [expected_glyphs(a) for a in node.args]
    if isinstance(fn, Sym) and fn.name == "list":
        return ",".join(args)
    if isinstance(fn, Sym) and fn.name in ("italic", "plain", "bold", "~"):
        return args[0]
    if isinstance(fn, Sym) and fn.name == "widehat":
        return "\u0302" + args[0]
    return expected_glyphs(fn) + "(" + ",".join(args) + ")"


_NUMBERISH = re.compile(r"-?[\d.,]+(e[-+]\d+)?|-?Inf|NA|NaN")


def skeleton(node: Node) -> str:
    """The structure with every number-like string and number replaced by '#'."""
    if isinstance(node, Str):
        return '"#"' if _NUMBERISH.fullmatch(node.value) else f'"{node.value}"'
    if isinstance(node, Num):
        return "#"
    if isinstance(node, Sym):
        return node.name
    if isinstance(node, Index):
        return f"{skeleton(node.base)}[{skeleton(node.index)}]"
    if isinstance(node, BinOp):
        return f"{skeleton(node.lhs)} {node.op} {skeleton(node.rhs)}"
    if isinstance(node, Paren):
        return f"({skeleton(node.inner)})"
    return f"{skeleton(node.fn)}({', '.join(skeleton(a) for a in node.args)})"
