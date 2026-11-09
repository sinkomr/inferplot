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
* Noncentral F with df1 = 2 and a large df2 (set noncentral_f_df1_two): exact
  sums (``ncf_tails_df1_two``).
* Noncentral F with a large ncp, and far tails (sets noncentral_f_large and
  noncentral_f_deep): F' = (X1 / df1) / (X2 / df2) with X1 ~ chi'^2(df1, ncp) and
  X2 ~ chi^2(df2), so P(F' <= f) = integral of the chi'^2 density
  1/2 e^(-(u + ncp)/2) (u / ncp)^(df1/4 - 1/2) I_(df1/2 - 1)(sqrt(ncp u)) times
  Q(df2 / 2, u / (2 c)), c = f df1 / df2 (the upper tail with P instead of Q),
  by tanh-sinh quadrature at 40 digits over the region where the integrand is
  within e^-120 of its peak (found on a grid first). It shares nothing with the
  Poisson mixture inferplot sums, and works at any ncp (1e12 here). The deep set
  stores log probabilities, down to e^-2000.
"""

from __future__ import annotations

import argparse
import itertools
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
# noncentral_f_large: Poisson means mu = ncp / 2 from 1e5 to 1e12, and df2 up to
# 1e8, at f where each tail is about each target.
LARGE_MUS = [1e5, 1e8, 1e12]
LARGE_DFS = [(2.0, 6.77), (5.0, 60.0), (3.0, 2300.0)]
LARGE_TARGETS = [1e-3, 1e-20]
# noncentral_f_df1_two: df1 = 2 with a large df2 (mpmath's incomplete gamma takes
# about 2 s per value at shape 5e7, so the integral is out of reach there; the
# exact sum below needs none). (df2, mu) pairs.
DF1_TWO_CASES = [(1e6, 1e4), (1e8, 1e4), (1e8, 3e4), (1e8, 1e5)]
DF1_TWO_TARGETS = [1e-3, 1e-10, 1e-20]
# noncentral_f_deep: (df1, df2, ncp) and target log tail probabilities.
DEEP_CASES = [(2.0, 30.0, 4e4), (5.0, 60.0, 1e3), (3.0, 2300.0, 2e5), (2.0, 6.77, 2e7)]
DEEP_LOG_TARGETS = [-300.0, -2000.0]


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


def ncf_tails_integral(f: float, df1: float, df2: float, ncp: float) -> tuple[Any, Any]:
    """log P(F' <= f) and log P(F' > f) by the chi'^2 x chi^2 integral."""
    with mp.workdps(DPS):
        k, lam = mp.mpf(df1), mp.mpf(ncp)
        c = mp.mpf(f) * k / mp.mpf(df2)
        s = mp.mpf(df2) / 2
        nu = k / 2 - 1

        def log_density(u: Any) -> Any:
            return (
                -mp.log(2)
                - (u + lam) / 2
                + (k / 4 - mp.mpf(1) / 2) * mp.log(u / lam)
                + mp.log(mp.besseli(nu, mp.sqrt(lam * u)))
            )

        def log_q(u: Any) -> Any:
            return mp.log(mp.gammainc(s, u / (2 * c), mp.inf, regularized=True))

        def log_p(u: Any) -> Any:
            return mp.log(mp.gammainc(s, 0, u / (2 * c), regularized=True))

        sd = mp.sqrt(2 * k + 4 * lam)
        top = lam + k + 400 * sd + 400 * c * s
        out = []
        for tail in (log_q, log_p):

            def g(u: Any, tail: Any = tail) -> Any:
                return log_density(u) + tail(u)

            # A fine grid over the chi'^2 bulk and a coarse one over everything.
            bulk_lo = max(lam + k - 100 * sd, (lam + k) / 10**6)
            bulk = [bulk_lo + (200 * sd) * mp.mpf(i) / 500 for i in range(501)]
            broad = [top * mp.mpf(i) / 500 for i in range(1, 501)]
            grid = sorted(set(bulk + broad))
            values = [g(u) for u in grid]
            peak = max(values)
            if not mp.isfinite(peak):
                raise ValueError("the integrand vanishes on the grid")
            keep = [i for i, v in enumerate(values) if v > peak - 120]
            assert keep[-1] < len(grid) - 1, "integrand not contained in the grid"
            # The grid intervals around the region where the integrand is within
            # e^-120 of its peak, split so that there are at least 40.
            span = grid[max(keep[0] - 1, 0) : keep[-1] + 2]
            if keep[0] == 0:
                span = [mp.mpf(0), *span]
            parts = max(1, math.ceil(40 / (len(span) - 1)))
            edges = [
                left + (right - left) * mp.mpf(i) / parts
                for left, right in itertools.pairwise(span)
                for i in range(parts)
            ]
            edges.append(span[-1])
            total = mp.quad(lambda u, g=g, peak=peak: mp.exp(g(u) - peak), edges)
            out.append(peak + mp.log(total))
        return out[0], out[1]


def _target_f(
    df1: float, df2: float, ncp: float, log_target: float, upper: bool
) -> float:
    """f where inferplot's log tail is log_target (the reference is exact at it)."""
    from scipy import optimize

    from inferplot.stats.noncentral import noncentral_f_logcdf

    def g(log_f: float) -> float:
        return (
            noncentral_f_logcdf(math.exp(log_f), df1, df2, ncp, upper=upper)
            - log_target
        )

    lo, hi = -5.0, 5.0
    while (g(lo) > 0.0) != upper:
        lo -= 5.0
    while (g(hi) < 0.0) != upper:
        hi += 5.0
    return math.exp(optimize.brentq(g, lo, hi, xtol=1e-12))


def _fresh_large_points() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cases = [(d1, d2, 2.0 * mu) for mu in LARGE_MUS for d1, d2 in LARGE_DFS]
    for df1, df2, ncp in cases:
        for target in LARGE_TARGETS:
            for upper in (False, True):
                f = _target_f(df1, df2, ncp, math.log(target), upper)
                out.append({"f": f, "df1": df1, "df2": df2, "ncp": ncp})
    return out


def ncf_tails_df1_two(f: float, df2: float, ncp: float) -> tuple[Any, Any]:
    """P(F' <= f) and P(F' > f) for df1 = 2, as exact sums.

    With a = 1 + j an integer, I_y(b, 1 + j) = y^b sum_{i <= j} (b)_i / i! x^i
    (a negative binomial distribution function), so the Poisson(mu) mixture is
    P(F' > f) = y^b sum_i (b)_i / i! x^i P(J >= i) and P(F' <= f) the same with
    P(J < i), J ~ Poisson(mu): positive terms, by recurrences at 80 digits.
    """
    with mp.workdps(80):
        b, mu = mp.mpf(df2) / 2, mp.mpf(ncp) / 2
        den = 2 * mp.mpf(f) + mp.mpf(df2)
        x, y = 2 * mp.mpf(f) / den, mp.mpf(df2) / den
        mean = b * x / y
        top = int(max(mean + 80 * mp.sqrt(mean / y), mu + 80 * mp.sqrt(mu)) + 200)
        pmf = [mp.exp(-mu)]
        for i in range(1, top + 2):
            pmf.append(pmf[-1] * mu / i)
        at_least = [mp.mpf(0)] * (top + 2)  # P(J >= i), summed from the top
        running = mp.mpf(0)
        for i in range(top + 1, -1, -1):
            running += pmf[i]
            at_least[i] = running
        below = mp.mpf(0)  # P(J < i), summed from 0
        term = mp.exp(b * mp.log(y))
        upper = lower = mp.mpf(0)
        for i in range(top + 1):
            upper += term * at_least[i]
            lower += term * below
            below += pmf[i]
            term = term * (b + i) * x / (i + 1)
        return lower, upper


def _fresh_df1_two_points() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for df2, mu in DF1_TWO_CASES:
        for target in DF1_TWO_TARGETS:
            for upper in (False, True):
                f = _target_f(2.0, df2, 2.0 * mu, math.log(target), upper)
                out.append({"f": f, "df1": 2.0, "df2": df2, "ncp": 2.0 * mu})
    return out


def _fresh_deep_points() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for df1, df2, ncp in DEEP_CASES:
        for log_target in DEEP_LOG_TARGETS:
            for upper in (False, True):
                try:
                    f = _target_f(df1, df2, ncp, log_target, upper)
                except (ValueError, OverflowError, ArithmeticError):
                    continue  # out of reach of the bracket (f beyond the doubles)
                out.append({"f": f, "df1": df1, "df2": df2, "ncp": ncp})
    return out


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


def _compute(args: tuple[str, dict[str, Any]]) -> tuple[str, str] | None:
    kind, p = args
    if kind == "t":
        lower, upper = nct_tails(p["t"], p["df"], p["ncp"])
    elif kind == "f":
        lower, upper = ncf_tails(p["f"], p["df1"], p["df2"], p["ncp"])
    elif kind == "f_df1_two":
        lower, upper = ncf_tails_df1_two(p["f"], p["df2"], p["ncp"])
    else:  # large and deep: log tails by the integral
        try:
            log_lower, log_upper = ncf_tails_integral(
                p["f"], p["df1"], p["df2"], p["ncp"]
            )
        except (AssertionError, IndexError, ValueError) as error:
            # The grid did not resolve the integrand (f near 1e-261): left out.
            print(f"left out {p}: {error!r}", flush=True)
            return None
        if kind == "f_deep":
            return mp.nstr(log_lower, DIGITS), mp.nstr(log_upper, DIGITS)
        lower, upper = mp.exp(log_lower), mp.exp(log_upper)
    return mp.nstr(lower, DIGITS), mp.nstr(upper, DIGITS)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--sets",
        help="comma-separated sets to (re)compute, e.g. f_large,f_deep; the others "
        "are copied from the file (default: all)",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    old = json.loads(OUTPUT.read_text()) if OUTPUT.exists() and not args.fresh else None
    chosen = None if args.sets is None else set(args.sets.split(","))
    sets: dict[str, Any] = {}
    kinds = (
        ("t", _fresh_t_points),
        ("f", _fresh_f_points),
        ("f_large", _fresh_large_points),
        ("f_df1_two", _fresh_df1_two_points),
        ("f_deep", _fresh_deep_points),
    )
    for kind, fresh in kinds:
        name = f"noncentral_{kind}"
        if chosen is not None and kind not in chosen and old is not None:
            sets[name] = old["sets"][name]
            continue
        keys = ("log_lower", "log_upper") if kind == "f_deep" else ("lower", "upper")
        if old is None or name not in old["sets"]:
            points = fresh()
        else:
            points = [
                {k: v for k, v in p.items() if k not in keys}
                for p in old["sets"][name]["points"]
            ]
        with ProcessPoolExecutor(args.workers) as pool:
            values = list(pool.map(_compute, [(kind, p) for p in points]))
        sets[name] = {
            "points": [
                {**p, keys[0]: value[0], keys[1]: value[1]}
                for p, value in zip(points, values, strict=True)
                if value is not None
            ]
        }
        print(f"{name}: {len(sets[name]['points'])} points", flush=True)
    data = {
        "description": (
            "High-precision lower and upper tails of the noncentral t and F "
            "distributions as decimal strings (22 significant digits; their "
            "natural logs in noncentral_f_deep), written by "
            "tests/data/noncentral_mpmath.py with mpmath at 40 digits; see that "
            "file for the methods. Used by tests/stats/test_noncentral.py."
        ),
        "generator": "uv run --with mpmath python -I tests/data/noncentral_mpmath.py",
        "sets": sets,
    }
    OUTPUT.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
