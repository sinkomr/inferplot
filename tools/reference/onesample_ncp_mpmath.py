# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision noncentral-t inversions for the one-sample Hedges' g interval.

Regenerates ``tests/data/onesample_ncp_reference.json``, which
``tests/stats/test_one_sample.py`` reads. mpmath is used here only; it is not a
dependency of inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tools/reference/onesample_ncp_mpmath.py \
        [--rows 0:105]

``--rows start:stop`` recomputes only those rows and keeps the others already in
the file (the full grid takes a few minutes; run it in parts of under two minutes).

The one-sample Hedges' g interval (``effectsize::hedges_g(x, mu)``) is the interval
for the noncentrality parameter of the observed one-sample t on df = n - 1, each
bound divided by sqrt(n) and multiplied by Hedges' J(n - 1). This script solves, at
40 significant digits, P(T'(df, ncp) > t) = tail for the lower bound and
P(T'(df, ncp) <= t) = tail for the upper bound, with tail 0.025 (a two-sided 95%
interval) and 0.05 (a one-sided 95% bound, or a two-sided 90% interval), on a grid
of t = 0, 0.5, 2, 10, 50, 200, 1,000 and n = 2, 5, 30, 200, 2,000 (df = n - 1).

Both tails are integrals over the scaled chi variable S = chi_df / sqrt(df):
P(T' <= t) = int f_S(s) Phi(t s - ncp) ds and P(T' > t) = int f_S(s) Phi(ncp - t s)
ds, by mpmath's tanh-sinh quadrature in s with breakpoints at the density's bulk
and around s = ncp / t, where Phi steps (as ``tests/data/noncentral_mpmath.py``);
the root by mpmath's Anderson-Bjorck bracketing solver from a bracket found by
stepping out from the normal approximation ncp = t -/+ z sqrt(1 + t^2 / (2 df)).
inferplot integrates in log s with fixed Gauss-Legendre panels and solves in
double precision, so the two share the formula only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "onesample_ncp_reference.json"
DPS = 40
T_VALUES = ["0", "0.5", "2", "10", "50", "200", "1000"]
NS = [2, 5, 30, 200, 2000]
# (side, tail): "lower" solves P(T' > t) = tail, "upper" P(T' <= t) = tail.
BOUNDS = [("lower", "0.025"), ("upper", "0.025"), ("lower", "0.05")]


def _tail(t: Any, df: Any, ncp: Any, *, upper: bool) -> Any:
    x = df / 2
    log_c = mp.log(2) + x * mp.log(x) - mp.loggamma(x)

    def density(s: Any) -> Any:
        return mp.exp(log_c + (df - 1) * mp.log(s) - x * s * s)

    sd = 1 / mp.sqrt(2 * df)
    pts = {mp.mpf(0), mp.mpf(1)}
    for k in (-12, -6, -3, -1, 1, 3, 6, 12, 30, 60):
        if 1 + k * sd > 0:
            pts.add(1 + k * sd)
    if t != 0:
        s0 = ncp / t
        if s0 > 0:
            for k in (-40, -20, -8, -3, -1, 0, 1, 3, 8, 20, 40):
                if s0 + k / abs(t) > 0:
                    pts.add(s0 + k / abs(t))
    edges = [*sorted(pts), mp.inf]
    if upper:
        return mp.quad(lambda s: density(s) * mp.ncdf(ncp - t * s), edges)
    return mp.quad(lambda s: density(s) * mp.ncdf(t * s - ncp), edges)


def solve(t: Any, df: Any, side: str, tail: Any) -> Any:
    upper_tail = side == "lower"

    def g(ncp: Any) -> Any:
        # Increasing in ncp for the upper tail, decreasing for the lower one.
        value = _tail(t, df, ncp, upper=upper_tail)
        return mp.log(value) - mp.log(tail)

    z = mp.sqrt(2) * mp.erfinv(1 - 2 * tail)
    spread = mp.sqrt(1 + t * t / (2 * df))
    guess = t - z * spread if side == "lower" else t + z * spread
    step = spread
    a, b = guess - step, guess + step
    for _ in range(200):
        ga, gb = g(a), g(b)
        if ga * gb < 0:
            break
        # Move the bracket toward the root.
        if (ga > 0) == upper_tail:
            a, b = a - step, a
        else:
            a, b = b, b + step
        step *= 2
    else:  # pragma: no cover
        raise RuntimeError(f"no bracket at t={t}, df={df}, {side} {tail}")
    return mp.findroot(g, (a, b), solver="anderson", tol=mp.mpf(10) ** (-DPS + 5))


def grid() -> list[tuple[str, int, str, str]]:
    return [(t, n, side, tail) for t in T_VALUES for n in NS for side, tail in BOUNDS]


def main() -> None:
    parser = argparse.ArgumentParser()
    rows = grid()
    parser.add_argument("--rows", default=f"0:{len(rows)}", help="start:stop")
    start, stop = (int(v) for v in parser.parse_args().rows.split(":"))
    mp.mp.dps = DPS
    kept: dict[int, dict[str, object]] = {}
    if OUT.exists():
        kept = {int(r["row"]): r for r in json.loads(OUT.read_text())["rows"]}
    for index, (t, n, side, tail) in enumerate(rows):
        if not start <= index < stop:
            continue
        ncp = solve(mp.mpf(t), mp.mpf(n - 1), side, mp.mpf(tail))
        kept[index] = {
            "row": index,
            "t": t,
            "n": n,
            "side": side,
            "tail": tail,
            "ncp": mp.nstr(ncp, 30),
        }
        print(index, t, n, side, tail, kept[index]["ncp"], file=sys.stderr)
    payload = {
        "description": (
            "Noncentrality bounds of the one-sample t (df = n - 1): 'lower' solves "
            "P(T' > t) = tail, 'upper' P(T' <= t) = tail, at 40 digits with mpmath "
            "(tools/reference/onesample_ncp_mpmath.py), printed to 30 digits."
        ),
        "mpmath": mp.__version__,
        "dps": DPS,
        "rows": [kept[i] for i in sorted(kept)],
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
