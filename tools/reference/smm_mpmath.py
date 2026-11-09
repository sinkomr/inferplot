# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision quantiles of the Studentized maximum modulus distribution.

Regenerates ``tests/data/smm_reference.json``, which ``tests/stats/test_smm.py``
reads under the tolerance kind ``smm_quantile``. mpmath is used here only; it is not
a dependency of inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tools/reference/smm_mpmath.py

M = max_i |Z_i| / S over C independent standard normals and S = sqrt(chi2_df / df);
P(M <= c) = int_0^inf erf(c s / sqrt 2)^C f_S(s) ds (Stoline & Ury 1979). Here the
integral is a composite 24-point Gauss-Legendre rule at 30 digits on 240 equal
panels in t = log s, over the range outside which the density of S is below about
1e-30; it does not share the variable, panels or density form of
``inferplot.stats.smm``. The quantile is the root of P(M <= c) = level by mpmath's
secant method from inferplot's value; the script stops unless P(M <= root) is within
1e-24 of the level both with these panels and with twice as many. At df = inf
the quantile is closed form: Phi^-1((1 + level^(1/C)) / 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp

from inferplot.stats.smm import smm_quantile

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "tests" / "data" / "smm_reference.json"

mp.mp.dps = 30

COMPARISONS = (1, 2, 3, 6, 10, 28, 29, 36, 45, 66, 120, 300)
DFS = ("1", "1.5", "2", "3.7", "5.16", "10", "17.85", "45.2", "120", "1490", "1e5")
LEVELS = ("0.95",)
EXTRA = (  # (C, df, level): other levels
    (36, "12.3", "0.9"),
    (36, "12.3", "0.99"),
    (45, "2", "0.99"),
    (3, "5", "0.5"),
)


PANELS = 240
NODES = 24
_GL: dict[int, list[tuple[mp.mpf, mp.mpf]]] = {}


def _gauss_legendre(n: int) -> list[tuple[mp.mpf, mp.mpf]]:
    """Gauss-Legendre nodes and weights on [-1, 1] at the working precision."""
    if n not in _GL:
        out = []
        for i in range(1, n + 1):
            x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
            dp = mp.mpf(1)
            for _ in range(100):
                p0, p1 = mp.mpf(1), x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                step = p1 / dp
                x -= step
                if abs(step) < mp.mpf(10) ** (-mp.mp.dps + 2):
                    break
            out.append((x, 2 / ((1 - x * x) * dp * dp)))
        _GL[n] = out
    return _GL[n]


def cdf(c: mp.mpf, comparisons: int, df: mp.mpf, panels: int = PANELS) -> mp.mpf:
    """P(M <= c) by composite Gauss-Legendre in t = log s over [t_lo, t_hi]."""
    log_norm = mp.log(2) + (df / 2) * mp.log(df / 2) - mp.loggamma(df / 2)
    root2 = mp.sqrt(2)
    spread = 1 / mp.sqrt(2 * df)
    mode = mp.log(mp.sqrt((df - 1) / df)) if df > 1 else mp.mpf(-1)
    t_lo = mode - (75 / df + 14 * spread)
    t_hi = mp.log1p(170 / df) / 2 + 14 * spread
    width = (t_hi - t_lo) / panels
    total = mp.mpf(0)
    rule = _gauss_legendre(NODES)
    for i in range(panels):
        mid = t_lo + (i + mp.mpf(1) / 2) * width
        for x, w in rule:
            t = mid + x * width / 2
            s = mp.exp(t)
            dens = mp.exp(log_norm + df * t - df * s * s / 2)
            total += w * dens * mp.erf(c * s / root2) ** comparisons
    return total * width / 2


def quantile(comparisons: int, df_text: str, level_text: str) -> mp.mpf:
    level = mp.mpf(level_text)
    df = mp.mpf(df_text)
    start = mp.mpf(smm_quantile(comparisons, float(df), conf_level=float(level)))
    root = mp.findroot(
        lambda c: cdf(c, comparisons, df) - level,
        (start, start * (1 + mp.mpf("1e-9"))),
        solver="secant",
        tol=mp.mpf("1e-50"),
        verify=False,
    )
    miss = abs(cdf(root, comparisons, df) - level)
    finer = abs(cdf(root, comparisons, df, panels=2 * PANELS) - level)
    if max(miss, finer) > mp.mpf("1e-24"):
        raise SystemExit(f"C {comparisons} df {df_text}: residual {miss}, {finer}")
    return root


def main() -> int:
    points = [
        (c, df, level) for c in COMPARISONS for df in DFS for level in LEVELS
    ] + list(EXTRA)
    rows = []
    for comparisons, df, level in points:
        value = quantile(comparisons, df, level)
        rows.append(
            {
                "comparisons": comparisons,
                "df": df,
                "conf_level": level,
                "quantile": mp.nstr(value, 30),
            }
        )
        print(comparisons, df, level, mp.nstr(value, 20), flush=True)
    for comparisons in COMPARISONS:
        level = mp.mpf("0.95")
        value = mp.sqrt(2) * mp.erfinv((level) ** (mp.mpf(1) / comparisons))
        rows.append(
            {
                "comparisons": comparisons,
                "df": "inf",
                "conf_level": "0.95",
                "quantile": mp.nstr(value, 30),
            }
        )
    payload = {
        "description": (
            "Studentized maximum modulus quantiles, 40-digit mpmath "
            "(tools/reference/smm_mpmath.py)"
        ),
        "rows": rows,
    }
    TARGET.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
