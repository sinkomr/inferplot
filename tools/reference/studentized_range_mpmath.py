# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""High-precision references for P(Q_{k,nu} > q), the studentized range upper tail.

Regenerates ``tests/data/studentized_range_reference.json``, which
``tests/stats/test_srange.py`` reads under the tolerance kind
``studentized_range_tail``. mpmath is used here only; it is not a dependency of
inferplot or of its tests. Run from the repository root:

    uv run --with mpmath python -I tools/reference/studentized_range_mpmath.py \
        [--sets grid,df_inf,k2_identity,fixture_pairs,failed_fixture_pairs,\
                added_fixture_pairs] \
        [--workers 8] [--fresh]

By default the points (k, df, q) already in the JSON file are kept and only their
reference values are recomputed, so a rerun reproduces the file; ``--fresh`` picks
new q values from ``inferplot.stats.srange`` (any q will do: the references are
exact at whatever q they are given). The finite-df sets are slow, one to six
minutes per point; the full grid takes hours on 8 processes.

Three routes, none sharing srange.py's algebra (log1mexp in log space) or its
quadrature (fixed-step trapezoid sums):

* ``ref_t2``: exact for k = 2. Q_{2,nu} = |Z1 - Z2| / S = sqrt(2) |T_nu|, so
  P(Q > q) = P(|T_nu| > q / sqrt 2) = I_x(nu / 2, 1 / 2) with x = nu / (nu + q^2 / 2)
  (regularized incomplete beta); nu = inf gives erfc(q / 2).
* ``ref_range_cancel``: P(W_k > w) from the textbook max form
  k int phi(z) [Phi(z)^m - (Phi(z) - Phi(z - w))^m] dz, which cancels catastrophically
  in the tail; the integrand is evaluated with enough extra digits to absorb that.
* ``ref_srange``: the chi mixture by nested composite Gauss-Legendre quadrature
  (24 nodes per panel; 48 agree to 3.6e-21) over t = log s, with P(W > w) from the
  min form written as the positive geometric sum
  k int phi(z) b sum_{i<m} a^(m-1-i) (a - b)^i dz, a = P(Z > z), b = P(Z > z + w),
  at 24 digits. (Nested adaptive tanh-sinh took over 3 minutes per point.)

mpmath's quad stops on an absolute error estimate of about 10^-dps, so the
tanh-sinh integrand of ``ref_range_cancel`` is divided by its largest breakpoint value
first; without that, values near 1e-99 came back 4.5e-7 off.

Fixture pairs: q = |mean_i - mean_j| / sqrt((v_i/n_i + v_j/n_j) / 2) and the pair's
Welch df for every Games-Howell pair of the parametric ggbetweenstats fixtures with
three or more groups, as ``inferplot.stats.games_howell`` computes them from the
fixture datasets when a set is built fresh (``fixture_pairs`` was first built with
NumPy's mean and variance; the reference is exact at whatever q and df it is
given, and the tests check that the frozen points agree with games_howell within
``closed_form``). ``fixture_pairs`` holds the 31 pairs of the cases ggstatsplot
completed when the set was first built; ``failed_fixture_pairs`` holds the pairs of
the cases where ggstatsplot failed (edge_n2: every Welch df is below 2 and R's
ptukey returns NaN, so the probe has no R value to compare and the reference is the
only check of inferplot's p there). ``added_fixture_pairs`` holds the pairs of the
completed cases that ``fixture_pairs`` does not cover (the cases added to the oracle
after it was built), so that the two sets together cover every Games-Howell pair of
every completed parametric case with three or more groups. Points that share
(k, df, q), such as identical pairs of two cases, are computed once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mpmath as mp
from mpmath.calculus.quadrature import GaussLegendre

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tests" / "data" / "studentized_range_reference.json"
QDPS = 24  # quadrature precision: ~1e-24 relative target, far below what is tested
DIGITS = 22  # significant digits stored per reference value
TAILS = [0.9, 0.5, 0.1, 1e-2, 1e-4, 1e-7, 1e-10, 1e-15, 1e-20, 1e-30, 1e-50, 1e-100]
GRID_KS = [3, 5, 10, 50]
GRID_NUS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0]
INF_KS = [2, 3, 5, 10, 50, 100]
K2_NUS = [
    1.0,
    1.5,
    2.0,
    3.7,
    5.0,
    10.0,
    20.0,
    50.0,
    94.6978,
    100.0,
    1000.0,
    1e4,
    1e6,
    1e9,
    1e12,
    math.inf,
]
K2_TAILS = [*TAILS, 1e-200, 1e-300]

# --------------------------------------------------------------------- references


def ref_t2(q: float, nu: float, dps: int = 60) -> Any:
    """Exact P(Q_{2,nu} > q) via the Student t (or normal) tail."""
    with mp.workdps(dps):
        q_ = mp.mpf(q)
        if math.isinf(nu):
            return +mp.erfc(q_ / 2)
        nu_ = mp.mpf(nu)
        x = nu_ / (nu_ + q_ * q_ / 2)
        return +mp.betainc(nu_ / 2, mp.mpf(1) / 2, 0, x, regularized=True)


def _breakpoints_min_form(w: float, k: int) -> list[float]:
    mu = math.sqrt(2.0 * math.log(k))
    c = w / 2.0
    return sorted({-c - 8, -c - 2, -c, -c + 2, -c + 8, -mu - 5, -mu, -mu + 4})


def ref_range_cancel(w: float, k: int, extra_digits: int) -> Any:
    """P(W_k > w), cancelling max form; integrand at QDPS + 10 + extra digits."""
    m = k - 1
    hi = QDPS + 10 + extra_digits
    with mp.workdps(hi):
        w_ = mp.mpf(w)

    def raw(z: Any) -> Any:
        with mp.workdps(hi):
            z = mp.mpf(z)
            p = mp.ncdf(z)
            return mp.npdf(z) * (p**m - (p - mp.ncdf(z - w_)) ** m)

    pts = [-p for p in reversed(_breakpoints_min_form(w, k))]  # max = -min by symmetry
    with mp.workdps(QDPS):
        scale = max(raw(mp.mpf(p)) for p in pts)
        val = mp.quad(
            lambda z: raw(z) / scale, [-mp.inf, *[mp.mpf(p) for p in pts], mp.inf]
        )
        return k * val * scale


def _gl_rule(degree: int) -> list[tuple[Any, Any]]:
    """Gauss-Legendre nodes and weights on [-1, 1] (3 * 2^(degree - 1) points)."""
    return GaussLegendre(mp.mp).calc_nodes(degree, mp.mp.prec + 20)


def _gl(
    f: Callable[[Any], Any], edges: Sequence[Any], rule: list[tuple[Any, Any]]
) -> Any:
    total = mp.mpf(0)
    for a, b in itertools.pairwise(edges):
        half, mid = (b - a) / 2, (a + b) / 2
        total += half * mp.fsum(wt * f(mid + half * x) for x, wt in rule)
    return total


def _range_pos_gl(w: Any, k: int, rule: list[tuple[Any, Any]]) -> Any:
    """P(W_k > w): positive geometric-sum min form, fixed Gauss-Legendre panels."""
    m = k - 1

    def integrand(z: Any) -> Any:
        a = mp.ncdf(-z)
        b = mp.ncdf(-(z + w))
        ratio = (a - b) / a
        s, term = mp.mpf(0), a ** (m - 1)
        for _ in range(m):
            s += term
            term *= ratio
        return k * mp.npdf(z) * b * s

    pts = _breakpoints_min_form(float(w), k)
    edges = [pts[0] - 12, *pts, pts[-1] + 12]
    fine: list[float] = []
    for a, b in itertools.pairwise(edges):  # panels at most 3 wide
        n = max(1, math.ceil((b - a) / 3.0))
        fine.extend(a + (b - a) * i / n for i in range(n))
    fine.append(edges[-1])
    return _gl(integrand, [mp.mpf(e) for e in fine], rule)


def ref_srange(q: float, k: int, nu: float, degree: int = 4) -> Any:
    """P(Q_{k,nu} > q), finite nu: Gauss-Legendre panels over t = log s and z."""
    with mp.workdps(QDPS):
        rule = _gl_rule(degree)
        q_, nu_ = mp.mpf(q), mp.mpf(nu)
        x = nu_ / 2
        log_const = mp.log(2) + x * mp.log(x) - mp.loggamma(x)

        def g(t: Any) -> Any:
            r = _range_pos_gl(q_ * mp.exp(t), k, rule)
            return mp.exp(log_const + 2 * x * t - x * mp.exp(2 * t)) * r

        sigma = 1.0 / math.sqrt(2.0 * nu)
        tc = 0.5 * math.log(nu / (nu + 0.5 * q * q))
        core = [
            tc + sigma * o for o in (-14, -10, -7, -5, -3, -1.5, 0, 1.5, 3, 5, 8, 12)
        ]
        # To the left the integrand decays like e^(nu t) (P(W > w) -> 1): continue
        # 48 / nu + 1 below the centre, in panels of max(4 / nu, 2 sigma).
        left_len = max(0.0, 48.0 / nu + 1.0 - 14.0 * sigma)
        width = max(4.0 / nu, 2.0 * sigma)
        n_left = math.ceil(left_len / width)
        left_edges = [core[0] - width * j for j in range(n_left, 0, -1)]
        edges = [mp.mpf(e) for e in sorted({*left_edges, *core})]
        return _gl(g, edges, rule)


# --------------------------------------------------------------------- point sets


def _q_for(target: float, k: int, nu: float) -> float:
    """q with srange_sf(q, k, nu) = target (any q will do; the reference is exact)."""
    from scipy import optimize

    from inferplot.stats.srange import srange_logsf

    def f(lq: float) -> float:
        return srange_logsf(math.exp(lq), k, nu) - math.log(target)

    return math.exp(optimize.brentq(f, -30.0, 705.0, xtol=1e-14))


def _fixture_points(*, failed: bool = False) -> list[dict[str, Any]]:
    """|q| and Welch df of every Games-Howell pair, as games_howell computes them."""
    import numpy as np

    sys.path.insert(0, str(ROOT))
    from tests.oracle_fixtures import cases_for, r_level_label, read_dataset

    from inferplot.stats import games_howell

    out: list[dict[str, Any]] = []
    for case in cases_for("ggbetweenstats", failed=failed):
        if case.type != "parametric" or len(case.analysis_levels) < 3:
            continue
        frame = read_dataset(case)
        x, y = case.args["x"], case.args["y"]
        used = frame[[x, y]].dropna()
        labels = used[x].astype(object).map(r_level_label)
        groups = [
            used.loc[labels == lv, y].to_numpy(dtype=np.float64)
            for lv in case.analysis_levels
        ]
        for pair in games_howell(groups, p_adjust_method="none"):
            out.append(
                {
                    "case": case.name,
                    "pair": [
                        case.analysis_levels[pair.i],
                        case.analysis_levels[pair.j],
                    ],
                    "k": len(groups),
                    "df": pair.df,
                    "q": abs(pair.statistic),
                }
            )
    return out


def _fresh_points(name: str) -> list[dict[str, Any]]:
    if name == "grid":
        return [
            {"k": k, "df": nu, "tail": t, "q": _q_for(t, k, nu)}
            for k in GRID_KS
            for nu in GRID_NUS
            for t in TAILS
        ]
    if name == "df_inf":
        return [
            {"k": k, "df": math.inf, "tail": t, "q": _q_for(t, k, math.inf)}
            for k in INF_KS
            for t in TAILS
        ]
    if name == "k2_identity":
        return [
            {"k": 2, "df": nu, "tail": t, "q": _q_for(t, 2, nu)}
            for nu in K2_NUS
            for t in K2_TAILS
        ]
    if name == "fixture_pairs":
        return _fixture_points()
    if name == "failed_fixture_pairs":
        return _fixture_points(failed=True)
    raise KeyError(name)


def _added_fixture_points(covered: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """The completed fixture pairs whose (case, level, level) is not in ``covered``."""
    return [
        p
        for p in _fixture_points()
        if (p["case"], p["pair"][0], p["pair"][1]) not in covered
    ]


def _compute(args: tuple[str, dict[str, Any]]) -> str:
    name, point = args
    k, nu, q = int(point["k"]), _df(point["df"]), float(point["q"])
    if name == "k2_identity":
        value = ref_t2(q, nu)
    elif math.isinf(nu):
        tail = float(point.get("tail", 1e-15))
        value = ref_range_cancel(q, k, int(-math.log10(tail)) + 10)
    else:
        value = ref_srange(q, k, nu)
    return mp.nstr(value, DIGITS)


def _df(value: Any) -> float:
    return math.inf if value == "Inf" else float(value)


def _encode(point: dict[str, Any]) -> dict[str, Any]:
    out = dict(point)
    if math.isinf(float(out["df"])):
        out["df"] = "Inf"
    return out


METHODS = {
    "grid": "ref_srange: nested composite Gauss-Legendre (24 nodes per panel) over "
    "t = log s and z, positive geometric-sum form, 24 digits",
    "df_inf": "ref_range_cancel: tanh-sinh quadrature of the textbook max form with "
    "the integrand at 34 + 10 + |log10 tail| digits",
    "k2_identity": "ref_t2: exact P(|T_nu| > q / sqrt 2) = I_x(nu/2, 1/2), "
    "x = nu / (nu + q^2 / 2), 60 digits (erfc(q / 2) for nu = Inf)",
    "fixture_pairs": "ref_srange as for grid, at the 31 Games-Howell pairs (|q|, Welch "
    "df) of the parametric ggbetweenstats fixtures with three or more groups",
    "failed_fixture_pairs": "ref_srange as for grid, at the Games-Howell pairs (|q|, "
    "Welch df) of the parametric ggbetweenstats fixtures with three or more groups "
    "where ggstatsplot failed (edge_n2: Welch df below 2, R's ptukey NaN)",
    "added_fixture_pairs": "ref_srange as for grid, at the Games-Howell pairs (|q|, "
    "Welch df) of the completed parametric ggbetweenstats fixtures with three or more "
    "groups that fixture_pairs does not cover",
}


def _key(point: dict[str, Any]) -> tuple[int, str, str]:
    return int(point["k"]), repr(_df(point["df"])), repr(float(point["q"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sets", default=",".join(METHODS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    existing = json.loads(args.output.read_text()) if args.output.exists() else None
    data: dict[str, Any] = existing or {
        "description": (
            "High-precision values of P(Q_{k,df} > q), the studentized range upper "
            "tail, written by tools/reference/studentized_range_mpmath.py with "
            "mpmath; p is a decimal string with 22 significant digits, df 'Inf' is "
            "the range of k standard normals. Used by tests/stats/test_srange.py "
            "(tolerance studentized_range_tail)."
        ),
        "generator": "uv run --with mpmath python -I "
        "tools/reference/studentized_range_mpmath.py",
        "sets": {},
    }
    for name in args.sets.split(","):
        if name == "added_fixture_pairs" and (args.fresh or name not in data["sets"]):
            covered = {
                (p["case"], p["pair"][0], p["pair"][1])
                for p in data["sets"]["fixture_pairs"]["points"]
            }
            points = [_encode(p) for p in _added_fixture_points(covered)]
        elif args.fresh or name not in data["sets"]:
            points = [_encode(p) for p in _fresh_points(name)]
        else:
            points = [
                {k: v for k, v in p.items() if k != "p"}
                for p in data["sets"][name]["points"]
            ]
        unique = list({_key(p): p for p in points}.values())
        print(f"{name}: {len(points)} points, {len(unique)} distinct", flush=True)
        with ProcessPoolExecutor(args.workers) as pool:
            done = list(pool.map(_compute, [(name, p) for p in unique]))
        by_key = {_key(p): v for p, v in zip(unique, done, strict=True)}
        values = [by_key[_key(p)] for p in points]
        data["sets"][name] = {
            "method": METHODS[name],
            "points": [{**p, "p": v} for p, v in zip(points, values, strict=True)],
        }
        args.output.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
