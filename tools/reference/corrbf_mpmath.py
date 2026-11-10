# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision references for the Bayes factor of a Pearson correlation.

Regenerates ``tests/data/corrbf_reference.json``, which
``tests/stats/test_correlation.py`` reads. mpmath is used here only; it is not a
dependency of inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tools/reference/corrbf_mpmath.py

The Bayes factor of ``BayesFactor::correlationBF(rscale = kappa)``: a stretched
beta(1/kappa, 1/kappa) prior on rho over (-1, 1) (Ly, Verhagen & Wagenmakers 2016,
Journal of Mathematical Psychology 72: 19-32, eq. 13, Jeffreys's exact test with
width kappa):

    BF10 = 2^((kappa - 2) / kappa) sqrt(pi) / B(1/kappa, 1/kappa)
           Gamma((n + 2/kappa - 1) / 2) / Gamma((n + 2/kappa) / 2)
           2F1((n - 1) / 2, (n - 1) / 2; (n + 2/kappa) / 2; r^2).

At 40 significant digits with mpmath's ``hyp2f1``, and again with Euler's
transformation 2F1(a, b; c; z) = (1 - z)^(c - a - b) 2F1(c - a, c - b; c; z), which
evaluates other series; the script stops unless the two agree to 1e-30 in the log.
At r = +/-1, Gauss's 2F1(a, b; c; 1) = Gamma(c) Gamma(c - a - b) / (Gamma(c - a)
Gamma(c - b)) where c > a + b, and BF10 = inf otherwise. inferplot computes the
Euler integral of the same function by a trapezoidal sum, so the two share the
formula only. r^2 is formed from r at 40 digits (r is given as a double).

Grid: n = 3, 4, 5, 10, 30, 100, 500, 2000; r = 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99,
0.999, 0.9999999 and 1 (BF10 is even in r); kappa = 0.707, 0.8 and 1/3.
"""

from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "corrbf_reference.json"
DPS = 40
NS = [3, 4, 5, 10, 30, 100, 500, 2000]
RS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999999, 1.0]
KAPPAS = [0.707, 0.8, 1.0 / 3.0]
AGREE = mp.mpf("1e-30")


def log_bf10(n: int, r: float, kappa: float) -> mp.mpf:
    alpha = 1 / mp.mpf(kappa)
    a = mp.mpf(n - 1) / 2
    c = (mp.mpf(n) + 2 * alpha) / 2
    z = mp.mpf(r) ** 2
    front = (
        (1 - 2 * alpha) * mp.log(2)
        + mp.log(mp.pi) / 2
        - mp.log(mp.beta(alpha, alpha))
        + mp.loggamma(c - mp.mpf(1) / 2)
        - mp.loggamma(c)
    )
    if z == 1:
        if c - 2 * a <= 0:
            return mp.inf
        gauss = mp.loggamma(c) + mp.loggamma(c - 2 * a) - 2 * mp.loggamma(c - a)
        return front + gauss
    one = mp.log(mp.hyp2f1(a, a, c, z))
    two = (c - 2 * a) * mp.log(1 - z) + mp.log(mp.hyp2f1(c - a, c - a, c, z))
    if abs(one - two) > AGREE * max(1, abs(one)):
        raise SystemExit(f"n {n} r {r} kappa {kappa}: forms disagree: {one} {two}")
    return front + one


def main() -> None:
    mp.mp.dps = DPS
    rows = []
    for n in NS:
        for r in RS:
            for kappa in KAPPAS:
                value = log_bf10(n, r, kappa)
                rows.append(
                    {
                        "n": n,
                        "r": r,
                        "kappa_hex": float(kappa).hex(),
                        "log_bf10": "inf" if value == mp.inf else mp.nstr(value, 30),
                    }
                )
    payload = {
        "description": (
            "log BF10 of BayesFactor::correlationBF's stretched beta prior (Ly, "
            "Verhagen & Wagenmakers 2016, eq. 13) at 40 digits with mpmath "
            "(tools/reference/corrbf_mpmath.py), printed to 30 digits; 'inf' where "
            "the limit at r = +/-1 diverges."
        ),
        "mpmath": mp.__version__,
        "dps": DPS,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
