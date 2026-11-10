# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision references for the paired JZS Bayes factors of probe-pairedbf.

Regenerates ``tests/data/pairedbf_reference.json``, which
``tests/stats/test_bayes_paired.py`` reads. mpmath is used here only; it is not a
dependency of inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tools/reference/pairedbf_mpmath.py [--rows 0:76]

Each value takes about a second; ``--rows start:stop`` recomputes only those rows
and keeps the others already in the file (the full run is two such halves).

probe-pairedbf holds 152 calls of ``BayesFactor::ttestBF(x = a, y = b, paired =
TRUE, rscale = r)``: n = 5, 10, 30 and 200 pairs, |t| from 0 to 1,000, r = 0.707 and
0.8, with the data (``a_hex``, ``b_hex``), R's t and R's log BF10. For each row
this script computes, at 40 significant digits:

* ``t_exact``: the paired t of the data in exact rational arithmetic (the exact
  differences, their exact mean and variance), then the square root at 60 digits;
* ``log_bf10``: log BF10 at R's own t (``t_hex``, a double), which isolates
  BayesFactor's integration error from the rounding of its t. R's t and the exact
  t differ by at most a few units of 2^-53 relative, which moves log BF10 by at
  most about (n - 1) times that (2e-13 relative at n = 200), far below every
  comparison made with these values.

The Bayes factor is Rouder et al. (2009), eq. 1, with N = n and nu = n - 1:

    BF10 = int_0^inf (1 + N g r^2)^(-1/2) (1 + t^2 / ((1 + N g r^2) nu))^(-(nu+1)/2)
           (1 + t^2 / nu)^((nu+1)/2) (2 pi)^(-1/2) g^(-3/2) e^(-1/(2g)) dg,

integrated over u = log g, where the integrand is smooth and falls off at least
like e^-u to the right and like exp(-e^-u / 2) to the left. The integrand is
divided by its value at the peak (found by a scan and a root of the derivative), and
mpmath's tanh-sinh quadrature runs over panels one curvature width wide near the
peak and wider further out, to where the integrand is e^-200 below its peak. Each
value is computed twice, with tanh-sinh and with Gauss-Legendre quadrature on panels
half as wide, and the script stops unless the two agree to 1e-30 in the log. This
shares nothing with inferplot's trapezoidal sum in u (``inferplot.stats.bayes``) but
the formula.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests" / "fixtures" / "oracle" / "probe-pairedbf.json"
OUT = ROOT / "tests" / "data" / "pairedbf_reference.json"
DPS = 40
DROP = 200  # panels extend until the log integrand is this far below its peak
AGREE = mp.mpf("1e-30")


def _log_integrand(u: mp.mpf, t2: mp.mpf, n: int, r: mp.mpf) -> mp.mpf:
    nu = mp.mpf(n - 1)
    scale = 1 + n * r * r * mp.exp(u)
    return (
        -mp.log(scale) / 2
        - (nu + 1) / 2 * mp.log1p(t2 / (scale * nu))
        + (nu + 1) / 2 * mp.log1p(t2 / nu)
        - mp.log(2 * mp.pi) / 2
        - u / 2
        - mp.exp(-u) / 2
    )


def _peak(t2: mp.mpf, n: int, r: mp.mpf) -> mp.mpf:
    """The u where the log integrand is largest: a scan, then a root of its
    derivative."""
    reach = float(mp.log(1 + t2)) + 30.0
    grid = [mp.mpf(-30) + mp.mpf(k) / 4 for k in range(int((reach + 30) * 4) + 1)]
    values = [_log_integrand(u, t2, n, r) for u in grid]
    best = grid[max(range(len(grid)), key=lambda i: values[i])]

    def slope(u: mp.mpf) -> mp.mpf:
        return mp.diff(lambda v: _log_integrand(v, t2, n, r), u)

    return mp.findroot(
        slope, (best - mp.mpf(0.25), best + mp.mpf(0.25)), solver="anderson"
    )


def _panels(
    centre: mp.mpf, width: mp.mpf, t2: mp.mpf, n: int, r: mp.mpf, peak: mp.mpf
) -> list[mp.mpf]:
    """Panel edges: ``width`` wide within 8 widths of the centre, then doubling,
    until the log integrand is DROP below ``peak`` on both sides."""
    edges = [centre]
    for direction in (-1, 1):
        step, u = width, centre
        for i in range(10_000):
            u = u + direction * step
            edges.append(u)
            if i >= 8:
                step = min(step * 2, mp.mpf(4))
            if _log_integrand(u, t2, n, r) < peak - DROP:
                break
        else:  # pragma: no cover
            raise RuntimeError("tail not reached")
    return sorted(edges)


def log_bf10(t: mp.mpf, n: int, r: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """log BF10 by tanh-sinh and by Gauss-Legendre on halved panels."""
    t2 = t * t
    centre = _peak(t2, n, r)
    peak = _log_integrand(centre, t2, n, r)
    curvature = -mp.diff(lambda v: _log_integrand(v, t2, n, r), centre, 2)
    width = min(mp.mpf(1), 1 / mp.sqrt(curvature)) if curvature > 0 else mp.mpf(1)

    def f(u: mp.mpf) -> mp.mpf:
        return mp.exp(_log_integrand(u, t2, n, r) - peak)

    first = mp.quad(f, _panels(centre, width, t2, n, r, peak), method="tanh-sinh")
    second = mp.quad(
        f, _panels(centre, width / 2, t2, n, r, peak), method="gauss-legendre"
    )
    return peak + mp.log(first), peak + mp.log(second)


def exact_t(a_hex: list[str], b_hex: list[str]) -> mp.mpf:
    a = [Fraction(float.fromhex(v)) for v in a_hex]
    b = [Fraction(float.fromhex(v)) for v in b_hex]
    d = [x - y for x, y in zip(a, b, strict=True)]
    n = len(d)
    mean = sum(d, Fraction(0)) / n
    var = sum(((v - mean) ** 2 for v in d), Fraction(0)) / (n - 1)
    with mp.workdps(60):
        return (
            mp.mpf(mean.numerator)
            / mean.denominator
            / mp.sqrt(mp.mpf(var.numerator) / var.denominator / n)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="0:152", help="start:stop, row indices")
    start, stop = (int(v) for v in parser.parse_args().rows.split(":"))
    mp.mp.dps = DPS
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    kept: dict[int, dict[str, object]] = {}
    if OUT.exists():
        kept = {int(r["row"]): r for r in json.loads(OUT.read_text())["rows"]}
    for index, row in enumerate(probe["rows"]):
        n = int(row["n"])
        r = mp.mpf(float.fromhex(row["rscale_hex"]))
        t_r = mp.mpf(float.fromhex(row["t_hex"]))
        t_x = exact_t(row["a_hex"], row["b_hex"])
        if not start <= index < stop:
            continue
        one, two = log_bf10(t_r, n, r)
        if abs(one - two) > AGREE * max(1, abs(one)):
            raise SystemExit(f"row {index}: quadratures disagree: {one} {two}")
        kept[index] = {
            "row": index,
            "n": n,
            "t_target": row["t_target"],
            "rscale_hex": row["rscale_hex"],
            "t_hex": row["t_hex"],
            "t_exact": mp.nstr(t_x, 30),
            "log_bf10": mp.nstr(one, 30),
        }
        print(index, n, row["t_target"], kept[index]["log_bf10"], file=sys.stderr)
    out = [kept[i] for i in sorted(kept)]
    payload = {
        "description": (
            "Paired JZS log BF10 (Rouder et al. 2009, eq. 1, N = n, nu = n - 1) for "
            "the 152 rows of probe-pairedbf, 40-digit mpmath quadrature "
            "(tools/reference/pairedbf_mpmath.py), printed to 30 digits, at R's t "
            "(t_hex); t_exact is the exact paired t of the probe's data."
        ),
        "mpmath": mp.__version__,
        "dps": DPS,
        "rows": out,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
