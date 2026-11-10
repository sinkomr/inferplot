# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision references for the one-sample JZS Bayes factors of
probe-onesamplebf.

Regenerates ``tests/data/onesamplebf_reference.json``, which
``tests/stats/test_one_sample.py`` reads. mpmath is used here only; it is not a
dependency of inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tools/reference/onesamplebf_mpmath.py [--rows 0:304]

Each distinct value takes about a second; ``--rows start:stop`` recomputes only
those rows and keeps the others already in the file (the full run is three such
parts, each under two minutes).

probe-onesamplebf holds 304 calls of ``BayesFactor::ttestBF(x = x, mu = mu, rscale =
r)``: n = 5, 10, 30 and 200 values, |t| from 0 to 1,000, mu 0 and 3, r = 0.707 and
0.8, with the data (``x_hex``), R's t and R's log BF10. For each row this script
computes, at 40 significant digits:

* ``t_exact``: the one-sample t of the data against mu in exact rational
  arithmetic, then the square root at 60 digits;
* ``log_bf10``: log BF10 at R's own t (``t_hex``, a double), with N = n and nu =
  n - 1, by the quadrature of ``tools/reference/pairedbf_mpmath.py`` (the paired
  Bayes factor is the one-sample Bayes factor of the differences, so the formula
  and its two independent quadratures, tanh-sinh and Gauss-Legendre, which must
  agree to 1e-30 in the log, are the same). It shares nothing with inferplot's
  trapezoidal sum (``inferplot.stats.bayes``) but the formula.

Rows with the same n, R's t and r share one computation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "tests" / "fixtures" / "oracle" / "probe-onesamplebf.json"
OUT = ROOT / "tests" / "data" / "onesamplebf_reference.json"
DPS = 40
AGREE = mp.mpf("1e-30")


def _paired_module():  # type: ignore[no-untyped-def]
    """``pairedbf_mpmath`` by path (``python -I`` leaves the script's folder off
    sys.path)."""
    path = Path(__file__).resolve().parent / "pairedbf_mpmath.py"
    spec = importlib.util.spec_from_file_location("pairedbf_mpmath", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_t(x_hex: list[str], mu: float) -> mp.mpf:
    d = [Fraction(float.fromhex(v)) - Fraction(mu) for v in x_hex]
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
    parser.add_argument("--rows", default="0:304", help="start:stop, row indices")
    start, stop = (int(v) for v in parser.parse_args().rows.split(":"))
    mp.mp.dps = DPS
    paired = _paired_module()
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    kept: dict[int, dict[str, Any]] = {}
    if OUT.exists():
        kept = {int(r["row"]): r for r in json.loads(OUT.read_text())["rows"]}
    done: dict[tuple[int, str, str], str] = {
        (int(r["n"]), str(r["t_hex"]), str(r["rscale_hex"])): str(r["log_bf10"])
        for r in kept.values()
    }
    for index, row in enumerate(probe["rows"]):
        if not start <= index < stop:
            continue
        n = int(row["n"])
        key = (n, str(row["t_hex"]), str(row["rscale_hex"]))
        if key not in done:
            r = mp.mpf(float.fromhex(row["rscale_hex"]))
            t_r = mp.mpf(float.fromhex(row["t_hex"]))
            one, two = paired.log_bf10(t_r, n, r)
            if abs(one - two) > AGREE * max(1, abs(one)):
                raise SystemExit(f"row {index}: quadratures disagree: {one} {two}")
            done[key] = mp.nstr(one, 30)
        kept[index] = {
            "row": index,
            "n": n,
            "mu": row["mu"],
            "t_target": row["t_target"],
            "rscale_hex": row["rscale_hex"],
            "t_hex": row["t_hex"],
            "t_exact": mp.nstr(exact_t(row["x_hex"], float(row["mu"])), 30),
            "log_bf10": done[key],
        }
        print(index, n, row["t_target"], kept[index]["log_bf10"], file=sys.stderr)
    out = [kept[i] for i in sorted(kept)]
    payload = {
        "description": (
            "One-sample JZS log BF10 (Rouder et al. 2009, eq. 1, N = n, nu = n - 1) "
            "for the 304 rows of probe-onesamplebf, 40-digit mpmath quadrature "
            "(tools/reference/onesamplebf_mpmath.py), printed to 30 digits, at R's "
            "t (t_hex); t_exact is the exact one-sample t of the probe's data "
            "against mu."
        ),
        "mpmath": mp.__version__,
        "dps": DPS,
        "rows": out,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
