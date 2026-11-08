# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision references for the noncentral t and F distribution functions.

Regenerates ``tests/data/noncentral_reference.json`` for
``tests/stats/test_noncentral.py``. mpmath is used only here; it is not a dependency
of inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tests/data/noncentral_mpmath.py [--fresh]

Without ``--fresh`` the points already in the file are kept and only their values are
recomputed. Points are chosen with inferplot's own functions so that the lower-tail
probability lands near a list of targets; the references are exact at whatever point
they are given.

* Noncentral t: P(T' <= t) = integral_0^inf f_S(s) Phi(t s - ncp) ds and
  P(T' > t) = integral f_S(s) Phi(ncp - t s) ds with S = chi_df / sqrt(df), by
  mpmath's tanh-sinh quadrature in s (not log s) at 40 digits, with breakpoints at
  the density's bulk and around s = ncp / t where Phi steps. inferplot integrates in
  log s with fixed Gauss-Legendre panels, so the two share no quadrature.
* Noncentral F: the Poisson(ncp / 2) mixture of regularized incomplete beta functions
  (Johnson, Kotz & Balakrishnan 1995, ch. 30), summed term by term at 40 digits until
  the Poisson weights beyond the mean fall below 10^-45. It is the formula inferplot
  uses, so it checks inferplot's floating point (weights, window, double-precision
  incomplete beta), not the formula.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests" / "data" / "noncentral_reference.json"
DPS = 40
DIGITS = 22
T_VALUES = [-50.0, -8.0, -1.5, 0.3, 2.0, 8.0, 50.0, 500.0]
T_DFS = [1.0, 2.5, 10.0, 37.3, 200.0, 1e4]
T_TARGETS = [1e-12, 5e-4, 0.025, 0.5, 0.975, 0.9995, 1.0 - 1e-12]
F_VALUES = [0.3, 1.5, 5.0, 40.0]
F_DF1 = [1.0, 2.0, 4.5]
F_DF2 = [3.5, 17.13, 100.0, 5000.0]
F_NCPS = [0.5, 5.0, 60.0, 800.0]


def nct_tails(t: float, df: float, ncp: float) -> tuple[Any, Any]:
    with mp.workdps(DPS):
        t_, df_, nc = mp.mpf(t), mp.mpf(df), mp.mpf(ncp)
        x = df_ / 2
        log_c = mp.log(2) + x * mp.log(x) - mp.loggamma(x)

        def density(s: Any) -> Any:
            return mp.exp(log_c + (df_ - 1) * mp.log(s) - x * s * s)

        sd = 1 / mp.sqrt(2 * df_)
        pts = {mp.mpf(0), mp.mpf(1)}
        for k in (-12, -6, -3, -1, 1, 3, 6, 12, 30):
            if 1 + k * sd > 0:
                pts.add(1 + k * sd)
        s0 = nc / t_
        if s0 > 0:
            for k in (-40, -20, -8, -3, -1, 0, 1, 3, 8, 20, 40):
                if s0 + k / abs(t_) > 0:
                    pts.add(s0 + k / abs(t_))
        edges = [*sorted(pts), mp.inf]
        lower = mp.quad(lambda s: density(s) * mp.ncdf(t_ * s - nc), edges)
        upper = mp.quad(lambda s: density(s) * mp.ncdf(nc - t_ * s), edges)
        return lower, upper


def ncf_tails(f: float, df1: float, df2: float, ncp: float) -> tuple[Any, Any]:
    with mp.workdps(DPS):
        f_, d1, d2, lam = (mp.mpf(v) for v in (f, df1, df2, ncp))
        x = d1 * f_ / (d1 * f_ + d2)
        y = d2 / (d1 * f_ + d2)
        mu = lam / 2
        lower = upper = mp.mpf(0)
        j = 0
        while True:
            w = mp.exp(-mu + j * mp.log(mu) - mp.loggamma(j + 1))
            lower += w * mp.betainc(d1 / 2 + j, d2 / 2, 0, x, regularized=True)
            upper += w * mp.betainc(d2 / 2, d1 / 2 + j, 0, y, regularized=True)
            if j > mu and w < mp.mpf(10) ** -45:
                break
            j += 1
        return lower, upper


def _fresh_t_points() -> list[dict[str, Any]]:
    from scipy import optimize

    from inferplot.stats.noncentral import noncentral_t_logcdf

    out: list[dict[str, Any]] = []
    for t in T_VALUES:
        for df in T_DFS:
            for target in T_TARGETS:
                upper = target > 0.5
                goal = math.log1p(-target) if upper else math.log(target)

                def g(
                    ncp: float,
                    t: float = t,
                    df: float = df,
                    upper: bool = upper,
                    goal: float = goal,
                ) -> float:
                    return noncentral_t_logcdf(t, df, ncp, upper=upper) - goal

                lo, hi = t - 50.0 - 2.0 * abs(t), t + 50.0 + 2.0 * abs(t)
                try:
                    ncp = optimize.brentq(g, lo, hi, xtol=1e-12)
                except ValueError:
                    continue  # the target is out of reach in this bracket
                out.append({"t": t, "df": df, "ncp": ncp, "target": target})
    return out


def _fresh_f_points() -> list[dict[str, Any]]:
    return [
        {"f": f, "df1": d1, "df2": d2, "ncp": lam}
        for f in F_VALUES
        for d1 in F_DF1
        for d2 in F_DF2
        for lam in F_NCPS
    ]


def _compute(args: tuple[str, dict[str, Any]]) -> tuple[str, str]:
    kind, p = args
    if kind == "t":
        lower, upper = nct_tails(p["t"], p["df"], p["ncp"])
    else:
        lower, upper = ncf_tails(p["f"], p["df1"], p["df2"], p["ncp"])
    return mp.nstr(lower, DIGITS), mp.nstr(upper, DIGITS)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    old = json.loads(OUTPUT.read_text()) if OUTPUT.exists() and not args.fresh else None
    sets: dict[str, Any] = {}
    for kind, fresh in (("t", _fresh_t_points), ("f", _fresh_f_points)):
        name = f"noncentral_{kind}"
        if old is None:
            points = fresh()
        else:
            points = [
                {k: v for k, v in p.items() if k not in ("lower", "upper")}
                for p in old["sets"][name]["points"]
            ]
        with ProcessPoolExecutor(args.workers) as pool:
            values = list(pool.map(_compute, [(kind, p) for p in points]))
        sets[name] = {
            "points": [
                {**p, "lower": lo, "upper": up}
                for p, (lo, up) in zip(points, values, strict=True)
            ]
        }
        print(f"{name}: {len(points)} points", flush=True)
    data = {
        "description": (
            "High-precision lower and upper tails of the noncentral t and F "
            "distributions as decimal strings (22 significant digits), written by "
            "tests/data/noncentral_mpmath.py with mpmath at 40 digits; see that "
            "file for the methods. Used by tests/stats/test_noncentral.py."
        ),
        "generator": "uv run --with mpmath python -I tests/data/noncentral_mpmath.py",
        "sets": sets,
    }
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
