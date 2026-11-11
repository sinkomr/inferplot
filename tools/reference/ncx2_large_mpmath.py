# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Noncentrality bounds of very large chi-squares, at 50 digits beyond sqrt(x).

Regenerates ``tests/data/ncx2_large_reference.json``, which
``tests/stats/test_noncentral_chi2.py`` reads. mpmath is used here only; it is not
a dependency of inferplot or of its tests. Run from the repository root (a few
seconds):

    uv run --with mpmath python -I tools/reference/ncx2_large_mpmath.py

Each row is the noncentrality lambda with P(X' <= x) = prob for X' ~ chi'^2(df,
lambda), at x from 1e9 to 1.7e308 (a ``ratio`` share of 1e-100 gives a
goodness-of-fit statistic of about 2.5e100), where the Poisson mixture
(``ncx2_mpmath.py``) has too many terms. Two exact representations instead
(Johnson, Kotz & Balakrishnan 1995, ch. 29):

* df = 1: X' = (Z + sqrt(lambda))^2, so P(X' <= x) = Phi(sqrt(x) - sqrt(lambda)) -
  Phi(-sqrt(x) - sqrt(lambda));
* df = 3: P(X'_3 <= x) = P(X'_1 <= x) - 2 f_3(x) (the recurrence F_(k + 2) = F_k -
  2 f_(k + 2) in the degrees of freedom), with the density f_3(x) = e^(-(x +
  lambda) / 2) sinh(sqrt(lambda x)) / sqrt(2 pi lambda) (I_1/2 in elementary
  form), that is (phi(sqrt(x) - sqrt(lambda)) - phi(sqrt(x) + sqrt(lambda))) /
  (2 sqrt(lambda)). Checked once against the convolution of the df = 1 law with the
  exponential of mean 2 by quadrature (agreement to 1e-45 at x = 1e12).

The root is solved on the log of the smaller tail by bisection to 1e-35 relative,
in a bracket stepped out from (sqrt(x) - z)^2 - (df - 1) (z the normal
quantile of prob). ``x`` and ``prob`` are the
doubles the test passes, written exactly (``float.hex``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "ncx2_large_reference.json"
DPS = 50
XS = [1e9, 3.7e10, 1e12, 3.3e14, 1e16, 1e18, 2.5e20, 1e30, 2.5e100, 1e200, 1.7e308]
DFS = [1, 3]
PROBS = [1e-100, 1e-12, 0.0005, 0.025, 0.5, 0.975, 0.9995, 1.0 - 1e-12]


# Phi(-s) and phi(s) for s = sqrt(x) + sqrt(lambda) beyond this are below e^-5e7,
# nothing next to the other terms (at least 1e-100 times the other tail); mpmath's
# erfc overflows near s = 1e154, so they are left out.
FAR: Final = mp.mpf(10) ** 4


def far_ncdf(s: mp.mpf) -> mp.mpf:
    return mp.ncdf(-s) if s < FAR else mp.mpf(0)


def far_npdf(s: mp.mpf) -> mp.mpf:
    return mp.npdf(s) if s < FAR else mp.mpf(0)


def lower_df1(x: mp.mpf, lam: mp.mpf) -> mp.mpf:
    a, b = mp.sqrt(x), mp.sqrt(lam)
    return mp.ncdf(a - b) - far_ncdf(a + b)


def upper_df1(x: mp.mpf, lam: mp.mpf) -> mp.mpf:
    a, b = mp.sqrt(x), mp.sqrt(lam)
    return mp.ncdf(b - a) + far_ncdf(a + b)


def twice_density_df3(x: mp.mpf, lam: mp.mpf) -> mp.mpf:
    a, b = mp.sqrt(x), mp.sqrt(lam)
    return (mp.npdf(a - b) - far_npdf(a + b)) / b


def tail(x: mp.mpf, df: int, lam: mp.mpf, *, upper: bool) -> mp.mpf:
    if df == 1:
        return upper_df1(x, lam) if upper else lower_df1(x, lam)
    assert df == 3
    if upper:
        return upper_df1(x, lam) + twice_density_df3(x, lam)
    return lower_df1(x, lam) - twice_density_df3(x, lam)


def root(x: float, df: int, prob: float) -> mp.mpf:
    xm, pm = mp.mpf(x), mp.mpf(prob)
    upper = pm > mp.mpf(1) / 2
    target = mp.log(1 - pm) if upper else mp.log(pm)
    with mp.extradps(120):  # 2 prob - 1 keeps the digits of a prob of 1e-100
        z = mp.sqrt(2) * mp.erfinv(2 * pm - 1)
    start = (mp.sqrt(xm) - z) ** 2 - (df - 1)

    def f(lam: mp.mpf) -> mp.mpf:
        return mp.log(tail(xm, df, lam, upper=upper)) - target

    # f falls with lambda (the lower tail) or rises (the upper one): step out a
    # bracket of the root from the start value, then bisect it.
    step = 2 * mp.sqrt(start) + 10
    lo, hi = start - step, start + step
    while mp.sign(f(lo)) == mp.sign(f(hi)):
        step *= 2
        lo, hi = start - step, start + step
    f_lo = f(lo)
    while hi - lo > abs(hi) * mp.mpf(10) ** -35:
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if mp.sign(f_mid) == mp.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    rows = []
    for x in XS:
        # sqrt(x) - sqrt(lambda) is about z: the digits of sqrt(x) come on top.
        mp.mp.dps = DPS + int(mp.log10(x) / 2)
        for df in DFS:
            for prob in PROBS:
                lam = root(x, df, prob)
                rows.append(
                    {
                        "x": float.hex(x),
                        "df": df,
                        "prob": float.hex(prob),
                        "ncp": mp.nstr(lam, 30),
                    }
                )
                print(rows[-1])
    payload = {
        "description": (
            "Noncentrality lambda with P(chi'^2(df, lambda) <= x) = prob for very "
            "large x, at 50 digits beyond those of sqrt(x) "
            "(tools/reference/ncx2_large_mpmath.py); x and "
            "prob as float.hex."
        ),
        "mpmath": mp.__version__,
        "dps": DPS,
        "roots": rows,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
