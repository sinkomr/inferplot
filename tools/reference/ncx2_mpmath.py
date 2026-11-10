# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision noncentral chi-square probabilities and inversions.

Regenerates ``tests/data/ncx2_reference.json``, which ``tests/stats/test_noncentral.py``
and ``tests/stats/test_contingency.py`` read. mpmath is used here only; it is not a
dependency of inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tools/reference/ncx2_mpmath.py \
        [--section roots|cdf] [--rows 0:300]

``--rows start:stop`` recomputes only those rows of the section and keeps the others
already in the file (the full grid takes several minutes; run it in parts of under
two minutes).

The distribution: P(X' <= x) = sum_j Pois(j; lambda / 2) P(df / 2 + j, x / 2)
(Johnson, Kotz & Balakrishnan 1995, ch. 29), P the regularized lower incomplete
gamma function, summed at 40 digits over j within 40 standard deviations (plus 60
terms) of lambda / 2 and of sqrt(lambda x) / 2, where the summand peaks far in a
tail. The Poisson weights and the gamma factors are not evaluated one by one: the
lower factors by the downward recurrence P(a, y) = P(a + 1, y) + y^a e^-y /
Gamma(a + 1) from the top of the window, the upper ones Q by the upward recurrence
Q(a + 1, y) = Q(a, y) + y^a e^-y / Gamma(a + 1) from the bottom, both adding
positive terms (no cancellation), each started from one value of Kummer's series
or Legendre's continued fraction, whichever converges there; the weights and the
added terms go by their ratios from one direct value. inferplot sums a window of
SciPy gamma values in double precision (subsampled from lambda / 2 = 1e4 on), so
the two share the formula only.

Section ``roots``: the noncentrality lambda >= 0 with P(X' <= x) = prob on a grid of
x = 0.1, 0.5, 2, 10, 50, 300, 2,000, 10,000, df = 1, 2, 3, 5, 10, 20, 50 and every
probability a two-sided or one-sided interval at levels 0.5, 0.9, 0.95 and 0.999
solves for (0.0005, 0.001, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.999,
0.9995); "0" where P(X' <= x) at lambda = 0 is already at most prob. Solved by
mpmath's Anderson-Bjorck method on the log probability from a bracket stepped out
from the normal approximation.

Section ``cdf``: log10 P(X' <= x) and log10 P(X' > x) at x, df, lambda chosen to
reach both far tails and Poisson means beyond 1e4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "ncx2_reference.json"
DPS = 40
XS = ["0.1", "0.5", "2", "10", "50", "300", "2000", "10000"]
DFS = [1, 2, 3, 5, 10, 20, 50]
PROBS = [
    "0.0005",
    "0.001",
    "0.025",
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "0.75",
    "0.9",
    "0.95",
    "0.975",
    "0.999",
    "0.9995",
]
# (x, df, ncp) for the cdf section.
CDF_POINTS = [
    (x, df, ncp)
    for x in ("0.5", "10", "300", "10000")
    for df in (1, 5, 50)
    for ncp in ("0", "1", "30", "1000", "20000")
] + [
    ("200000", 3, "200000"),
    ("195000", 3, "200000"),
    ("206000", 3, "200000"),
    ("1000000", 10, "1010000"),
    ("990000", 10, "1000000"),
    ("50000", 2, "40000"),
]


def _window(mu: Any, y: Any) -> tuple[int, int]:
    """Poisson indices to sum: 40 standard deviations (plus 60 terms) around both
    the Poisson mode mu and sqrt(mu y), where the summand peaks far in a tail (there
    the gamma factor behaves like y^j / j!, so the summand like (mu y)^j / j!^2)."""
    far = mp.sqrt(mu * y)
    lo = min(mu, far)
    hi = max(mu, far)
    lo = max(0, int(mp.floor(lo - 40 * mp.sqrt(lo) - 60)))
    hi = int(mp.ceil(hi + 40 * mp.sqrt(hi) + 60))
    return lo, hi


def _series(a: Any, y: Any) -> Any:
    """P(a, y) by Kummer's series y^a e^-y / Gamma(a + 1) sum_k y^k / ((a + 1)
    ... (a + k)) (DLMF 8.7.1), for y < a + 1, where its terms fall."""
    term = mp.mpf(1)
    total = mp.mpf(1)
    k = 0
    eps = mp.mpf(10) ** (-DPS - 5)
    while True:
        k += 1
        term = term * y / (a + k)
        total += term
        if term < eps * total:
            break
    return mp.exp(a * mp.log(y) - y - mp.loggamma(a + 1)) * total


def _fraction(a: Any, y: Any) -> Any:
    """Q(a, y) by Legendre's continued fraction (DLMF 8.9.2) with the modified
    Lentz method, for y > a + 1, where it converges quickly."""
    tiny = mp.mpf(10) ** (-300)
    eps = mp.mpf(10) ** (-DPS - 5)
    b = y + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    i = 0
    while True:
        i += 1
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if d == 0 else d
        c = b + an / c
        c = tiny if c == 0 else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return mp.exp(a * mp.log(y) - y - mp.loggamma(a)) * h


def lower_gamma(a: Any, y: Any) -> Any:
    """Regularized lower incomplete gamma P(a, y)."""
    return _series(a, y) if y < a + 1 else 1 - _fraction(a, y)


def upper_gamma(a: Any, y: Any) -> Any:
    """Regularized upper incomplete gamma Q(a, y)."""
    return _fraction(a, y) if y > a + 1 else 1 - _series(a, y)


def tails(x: Any, df: Any, ncp: Any) -> tuple[Any, Any]:
    """P(X' <= x) and P(X' > x) at DPS digits."""
    y = x / 2
    a0 = df / 2
    mu = ncp / 2
    if mu == 0:
        return lower_gamma(a0, y), upper_gamma(a0, y)
    lo, hi = _window(mu, y)

    def weight(j: int) -> Any:
        return mp.exp(-mu + j * mp.log(mu) - mp.loggamma(j + 1))

    def step(a: Any) -> Any:
        # y^a e^-y / Gamma(a + 1)
        return mp.exp(a * mp.log(y) - y - mp.loggamma(a + 1))

    # Lower: downward from the top of the window; the weights and the steps by
    # their ratios, w_(j-1) = w_j j / mu and s(a - 1) = s(a) a / y.
    p = lower_gamma(a0 + hi, y)
    w = weight(hi)
    s = step(a0 + hi - 1)
    lower = mp.mpf(0)
    for j in range(hi, lo - 1, -1):
        lower += w * p
        p += s
        w = w * j / mu
        s = s * (a0 + j - 1) / y
    # Upper: upward from the bottom of the window, w_(j+1) = w_j mu / (j + 1) and
    # s(a + 1) = s(a) y / (a + 1).
    q = upper_gamma(a0 + lo, y)
    w = weight(lo)
    s = step(a0 + lo)
    upper = mp.mpf(0)
    for j in range(lo, hi + 1):
        upper += w * q
        q += s
        w = w * mu / (j + 1)
        s = s * y / (a0 + j + 1)
    return lower, upper


def solve(x: Any, df: Any, prob: Any) -> Any:
    low0, _ = tails(x, df, mp.mpf(0))
    if low0 <= prob:
        return mp.mpf(0)
    upper_side = prob > mp.mpf("0.5")
    target = mp.log(1 - prob) if upper_side else mp.log(prob)

    def g(ncp: Any) -> Any:
        low, up = tails(x, df, ncp)
        # increasing in ncp
        return mp.log(up) - target if upper_side else target - mp.log(low)

    z = mp.sqrt(2) * mp.erfinv(2 * prob - 1)
    d = x - df
    centre = d + 2 * z * z
    disc = centre * centre - d * d + 2 * z * z * df
    guess = centre - mp.sqrt(disc) if z > 0 else centre + mp.sqrt(disc)
    guess = max(guess, mp.mpf("1e-6"))
    step = mp.sqrt(2 * (df + 2 * guess)) / 4 + mp.mpf("0.01")
    a, b = max(guess - step, mp.mpf(0)), guess + step
    for _ in range(400):
        ga, gb = g(a), g(b)
        if ga <= 0 <= gb:
            break
        if ga > 0:
            a, b = max(a - step, mp.mpf(0)), a
        else:
            a, b = b, b + step
        step *= 2
    else:  # pragma: no cover
        raise RuntimeError(f"no bracket at x={x}, df={df}, prob={prob}")
    if ga == 0:
        return a
    return mp.findroot(g, (a, b), solver="anderson", tol=mp.mpf(10) ** (-DPS + 10))


def root_grid() -> list[tuple[str, int, str]]:
    return [(x, df, prob) for x in XS for df in DFS for prob in PROBS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=("roots", "cdf"), default="roots")
    parser.add_argument("--rows", default=None, help="start:stop")
    args = parser.parse_args()
    mp.mp.dps = DPS
    payload: dict[str, Any] = {"roots": [], "cdf": []}
    if OUT.exists():
        payload = json.loads(OUT.read_text())
    section = args.section
    grid: list[Any] = root_grid() if section == "roots" else CDF_POINTS
    start, stop = (
        (0, len(grid)) if args.rows is None else map(int, args.rows.split(":"))
    )
    kept = {int(r["row"]): r for r in payload.get(section, [])}
    for index, point in enumerate(grid):
        if not start <= index < stop:
            continue
        if section == "roots":
            x, df, prob = point
            ncp = solve(mp.mpf(x), mp.mpf(df), mp.mpf(prob))
            kept[index] = {
                "row": index,
                "x": x,
                "df": df,
                "prob": prob,
                "ncp": mp.nstr(ncp, 30),
            }
        else:
            x, df, ncp = point
            low, up = tails(mp.mpf(x), mp.mpf(df), mp.mpf(ncp))
            kept[index] = {
                "row": index,
                "x": x,
                "df": df,
                "ncp": ncp,
                "log10_lower": mp.nstr(mp.log10(low), 25),
                "log10_upper": mp.nstr(mp.log10(up), 25),
            }
        print(section, kept[index], file=sys.stderr)
    payload[section] = [kept[i] for i in sorted(kept)]
    payload["description"] = (
        "Noncentral chi-square at 40 digits with mpmath "
        "(tools/reference/ncx2_mpmath.py): 'roots' holds the noncentrality with "
        "P(X' <= x) = prob (0 where none is positive), 'cdf' log10 of both tails."
    )
    payload["mpmath"] = mp.__version__
    payload["dps"] = DPS
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
