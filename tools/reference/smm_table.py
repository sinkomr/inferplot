"""Regenerate ``src/inferplot/stats/_smm_table.py``, lincon's critical values.

WRS2's ``lincon`` takes its critical value from a two-decimal table of the
Studentized maximum modulus distribution at alpha = 0.05 (its ``smmcrit``; SPEC.md
section 4): rows at df 2..12, 14, 16, 18, 20, 24, 30, 40, 60 and a last row,
labelled df 200, that holds the infinite-df values; one column per number of
comparisons C = 2..28. (C = 1 is Student's t quantile at the pair's df, and above
28 comparisons lincon simulates: neither reads the table.) The entries written are
WRS2's own, read from the oracle's probe-smm (``smmcrit`` on its df x C grid), with
one exception (DECISIONS.md D51): C = 9 at df 2 reads 8.83 in WRS2, a transposition
of the quantile 8.38 (8.3846), and is written as 8.38, following the
accuracy-over-bug-parity precedent (D31); C = 9 never occurs in betweenstats, whose
C is k (k - 1) / 2.

The module docstring also lists every entry that differs from the quantile rounded
to two decimals. The quantile is computed here independently of
``inferplot.stats.smm``: with M = max_i |Z_i| / S, Z_1..Z_C independent standard
normal and S = sqrt(chi2_df / df) independent of them,

    P(M <= c) = int_0^inf (2 Phi(c s) - 1)^C f_S(s) ds,
    f_S(s) = 2 (df/2)^(df/2) / Gamma(df/2) s^(df - 1) exp(-df s^2 / 2),

and at df = infinity (2 Phi(c) - 1)^C (Stoline & Ury 1979, Technometrics 21:
87-93; Hochberg & Tamhane 1987, Multiple Comparison Procedures, section 2.1).
The integral is evaluated with SciPy's adaptive quadrature over a split at the
mode of f_S (relative error 1e-13) and the quantile found with Brent's method.

Run from the repository root (takes about 30 seconds):

    uv run python -I tools/reference/smm_table.py
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

from scipy import integrate, optimize, special

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "src" / "inferplot" / "stats" / "_smm_table.py"
PROBE = ROOT / "tests" / "fixtures" / "oracle" / "probe-smm.json"
ROWS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 30, 40, 60, 200)
INFINITE_ROW = 200
COMPARISONS = range(2, 29)
ALPHA = 0.05

# (C, df row) -> (WRS2's entry, the entry written): WRS2's transcription errors.
CORRECTED: dict[tuple[int, int], tuple[float, float]] = {(9, 2): (8.83, 8.38)}


def smm_cdf(c: float, comparisons: int, df: float) -> float:
    """P(M <= c) for the Studentized maximum modulus (see the module docstring)."""
    if math.isinf(df):
        return float(special.erf(c / math.sqrt(2.0))) ** comparisons
    log_norm = math.log(2.0) + (df / 2) * math.log(df / 2) - special.gammaln(df / 2)

    def density(s: float) -> float:
        if s <= 0.0:
            return 0.0
        log_f = log_norm + (df - 1) * math.log(s) - df * s * s / 2
        inner = float(special.erf(c * s / math.sqrt(2.0))) ** comparisons
        return math.exp(log_f) * inner

    mode = math.sqrt(max(df - 1.0, 0.5) / df)
    cuts = (0.0, 0.5 * mode, mode, 1.5 * mode, 3.0 * mode, 50.0)
    total = 0.0
    for lo, hi in itertools.pairwise(cuts):
        part = integrate.quad(density, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)
        total += float(part[0])
    return total


def smm_quantile(comparisons: int, df: float, alpha: float = ALPHA) -> float:
    """The upper ``alpha`` point of the Studentized maximum modulus."""
    return float(
        optimize.brentq(
            lambda c: smm_cdf(c, comparisons, df) - (1.0 - alpha),
            0.5,
            200.0,
            xtol=1e-14,
            rtol=1e-14,
        )
    )


def published() -> dict[tuple[int, int], float]:
    """WRS2's alpha-0.05 table from probe-smm: {(C, df row): entry}."""
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    (grid,) = [g for g in probe["grids"] if g["name"] == "smmcrit"]
    columns = [int(c) for c in grid["c_grid"]]
    table: dict[tuple[int, int], float] = {}
    for row in grid["rows"]:
        df = float.fromhex(row["df_hex"])
        if df not in ROWS:
            continue
        for c, text in zip(columns, row["values_hex"], strict=True):
            if c in COMPARISONS:
                table[(c, int(df))] = float.fromhex(text)
    if len(table) != len(ROWS) * len(COMPARISONS):
        raise SystemExit(f"probe-smm has {len(table)} table entries")
    return table


def build() -> dict[int, tuple[float, ...]]:
    """The table written to the module: WRS2's entries with ``CORRECTED`` applied."""
    entries = published()
    for key, (wrs2, corrected) in CORRECTED.items():
        if entries[key] != wrs2:
            raise SystemExit(f"probe-smm has {entries[key]} at {key}, not {wrs2}")
        entries[key] = corrected
    return {c: tuple(entries[(c, df)] for df in ROWS) for c in COMPARISONS}


def not_rounded_quantiles(
    table: dict[int, tuple[float, ...]],
) -> list[tuple[int, int, float, float]]:
    """(C, df row, entry, quantile) wherever the entry is not round(quantile, 2)."""
    out: list[tuple[int, int, float, float]] = []
    for c, row in table.items():
        for df, entry in zip(ROWS, row, strict=True):
            exact = smm_quantile(c, math.inf if df == INFINITE_ROW else df)
            if entry != round(exact, 2):
                out.append((c, df, entry, round(exact, 5)))
    return out


def render(table: dict[int, tuple[float, ...]]) -> str:
    differ = not_rounded_quantiles(table)
    lines = [
        '"""lincon\'s Studentized maximum modulus table at alpha = 0.05 (generated).',
        "",
        "Written by tools/reference/smm_table.py from the oracle's probe-smm; do",
        "not edit by hand. One tuple per number of comparisons C = 2..28, one entry",
        "per row of ``SMM_ROWS``; the row labelled 200 holds the infinite-df values.",
        "The",
        "entries are WRS2's published ones (its ``smmcrit``), except C = 9 at df 2,",
        "where WRS2 has 8.83 and this table the quantile 8.38 (DECISIONS.md D51).",
        f"{len(differ)} entries are not the quantile rounded to two decimals",
        "(C, df row, entry, quantile):",
        "",
    ]
    lines += [f"    {item!r}" for item in differ]
    lines += [
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Final",
        "",
        "# fmt: off",
        "SMM_ROWS: Final[tuple[int, ...]] = (",
        "    " + ", ".join(str(r) for r in ROWS[:12]) + ",",
        "    " + ", ".join(str(r) for r in ROWS[12:]) + ",",
        ")",
        "",
        "SMM_TABLE_05: Final[dict[int, tuple[float, ...]]] = {",
    ]
    for comparisons, row in table.items():
        first = ", ".join(f"{v:5.2f}" for v in row[:10])
        second = ", ".join(f"{v:5.2f}" for v in row[10:])
        lines.append(f"    {comparisons}: (")
        lines.append(f"        {first},")
        lines.append(f"        {second},")
        lines.append("    ),")
    lines.append("}")
    lines.append("# fmt: on")
    return "\n".join(lines) + "\n"


def main() -> int:
    TARGET.write_text(render(build()), encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
