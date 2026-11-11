# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Noncentrality bounds of chi-squares from 1e6 to 5e8, at 40 digits.

Regenerates ``tests/data/ncx2_mid_reference.json``, which
``tests/stats/test_noncentral_chi2.py`` reads. mpmath is used here only; it is not
a dependency of inferplot or of its tests. Run from the repository root (a few
minutes):

    uv run --with mpmath python -I tools/reference/ncx2_mid_mpmath.py

Each row is the noncentrality lambda with P(X' <= x) = prob for X' ~ chi'^2(df,
lambda). In this range the Poisson mixture has too many terms, and
``ncx2_large_mpmath.py`` covers df 1 and 3 only, so another exact representation
is used (Johnson, Kotz & Balakrishnan 1995, ch. 29): X' = (Z + sqrt(lambda))^2 + W
with Z standard normal and W an independent central chi-square on df - 1 degrees
of freedom, so that

    P(X' <= x) = E_W [ Phi(sqrt(x - W) - sqrt(lambda))
                       - Phi(-sqrt(x - W) - sqrt(lambda)) ],

integrated over W by quadrature (for df = 2, over the half-normal U with W = U^2,
whose density has no singularity). The root is found by the secant method from
(sqrt(x - df + 1) - z)^2 to a relative 1e-30. This is where chi2_ncp_bound changes
from its search to the Cornish-Fisher solution (DECISIONS.md D119): the search's
lower tail loses accuracy from about 2e6 on (5e-11 at prob 0.025 and 1e8).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "ncx2_mid_reference.json"
DPS = 40
XS = [1e6, 1.9e6, 2.1e6, 5e6, 3e7, 5e8]
DFS = [2, 5, 60, 1000]
PROBS = [0.0005, 0.025, 0.25, 0.45, 0.55, 0.975, 0.9995]


def cdf(x: float, df: int, lam: mp.mpf) -> mp.mpf:
    x_, root_lam = mp.mpf(x), mp.sqrt(lam)
    k = df - 1

    def inner(w: mp.mpf) -> mp.mpf:
        r = mp.sqrt(x_ - w)
        return mp.ncdf(r - root_lam) - mp.ncdf(-r - root_lam)

    if k == 1:  # W = U^2, U half-normal
        return mp.quad(
            lambda u: 2 * mp.npdf(u) * inner(u * u), [0, 1, 3, 6, 10, 20, 40]
        )
    half = mp.mpf(k) / 2
    log_norm = -half * mp.log(2) - mp.loggamma(half)

    def density(w: mp.mpf) -> mp.mpf:
        if w <= 0:
            return mp.mpf(0)
        return mp.exp(log_norm + (half - 1) * mp.log(w) - w / 2)

    sd = math.sqrt(2 * k)
    steps = (-30, -12, -6, -3, -1, 0, 1, 3, 6, 12, 30, 60)
    points = sorted({0.0, *[max(0.0, k + j * sd) for j in steps], k + 60 * sd + 400})
    return mp.quad(lambda w: density(w) * inner(w), points)


def root(x: float, df: int, prob: float) -> mp.mpf:
    z = mp.sqrt(2) * mp.erfinv(2 * mp.mpf(prob) - 1)
    guess = (mp.sqrt(mp.mpf(x) - (df - 1)) - z) ** 2
    return mp.findroot(
        lambda lam: cdf(x, df, lam) - prob,
        (guess, guess * (1 + mp.mpf(10) ** -12)),
        solver="secant",
        tol=mp.mpf(10) ** -30,
        maxsteps=60,
    )


def main() -> None:
    mp.mp.dps = DPS
    rows = []
    for x in XS:
        for df in DFS:
            for prob in PROBS:
                ncp = root(x, df, prob)
                rows.append(
                    {
                        "x": float(x).hex(),
                        "df": df,
                        "prob": float(prob).hex(),
                        "ncp": mp.nstr(ncp, 30),
                    }
                )
    payload = {
        "description": (
            "lambda with P(chi'^2(df, lambda) <= x) = prob; x and prob as exact "
            "doubles (float.hex), lambda to 30 digits; (Z + sqrt(lambda))^2 plus an "
            "independent chi-square on df - 1 degrees of freedom, by quadrature"
        ),
        "mpmath": mp.__version__,
        "dps": DPS,
        "roots": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} roots written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
