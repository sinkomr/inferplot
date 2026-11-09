"""Regenerate ``src/inferplot/stats/_smm_table.py``, lincon's critical values.

WRS2's ``lincon`` takes its critical value from a two-decimal table of the
Studentized maximum modulus distribution at alpha = 0.05 (SPEC.md section 4): rows
at df 2..12, 14, 16, 18, 20, 24, 30, 40, 60 and a last row, labelled df 200, that
holds the infinite-df values; one column per number of comparisons C. The table is
rebuilt here from the distribution itself and rounded to two decimals, and the
entries where WRS2's published table is known to differ from the rounded quantile
(oracle evidence) are written as WRS2 has them (``PUBLISHED_DIFFERS``).

The Studentized maximum modulus with C comparisons and df degrees of freedom is
M = max_i |Z_i| / S, with Z_1..Z_C independent standard normal and
S = sqrt(chi2_df / df) independent of them, so

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
import math
import sys
from pathlib import Path

from scipy import integrate, optimize, special

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "src" / "inferplot" / "stats" / "_smm_table.py"
ROWS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 30, 40, 60, 200)
INFINITE_ROW = 200
MAX_COMPARISONS = 45
ALPHA = 0.05

# (C, df row) -> WRS2's entry, where it is not the rounded quantile. Evidence: the
# lincon CIs of the oracle fixtures (tests/stats/test_robust_parity.py,
# edge_unbalanced a-b: C = 3 at df 5.16 interpolates 3.39 at df 5; the quantile is
# 3.39924).
PUBLISHED_DIFFERS: dict[tuple[int, int], float] = {(3, 5): 3.39}


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


def build() -> dict[int, tuple[float, ...]]:
    table: dict[int, tuple[float, ...]] = {}
    for comparisons in range(1, MAX_COMPARISONS + 1):
        row: list[float] = []
        for df in ROWS:
            exact = smm_quantile(comparisons, math.inf if df == INFINITE_ROW else df)
            rounded = round(exact, 2)
            row.append(PUBLISHED_DIFFERS.get((comparisons, df), rounded))
        table[comparisons] = tuple(row)
    return table


def render(table: dict[int, tuple[float, ...]]) -> str:
    lines = [
        '"""lincon\'s Studentized maximum modulus table at alpha = 0.05 (generated).',
        "",
        "Written by tools/reference/smm_table.py; do not edit by hand. One tuple per",
        "number of comparisons C = 1..45, one entry per row of ``SMM_ROWS``; the row",
        "labelled 200 holds the infinite-df values. Entries are the quantiles rounded",
        "to two decimals, except where WRS2's published table differs:",
        f"{sorted(PUBLISHED_DIFFERS.items())!r}.",
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
