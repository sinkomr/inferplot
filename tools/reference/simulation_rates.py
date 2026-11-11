"""Established rates for the simulation tests of kind 2 (SPEC.md section 11).

Regenerates ``tests/data/simulation_reference.json``, which the slow simulation
tests read through ``tests/simulation_settings.py``. Run from the repository root,
one setting (or a few) at a time, each invocation in under about two minutes:

    uv run python -I tools/reference/simulation_rates.py --only between_akp_null
    uv run python -I tools/reference/simulation_rates.py --true-values

Every setting with a quantity of kind 2 is run once at its ``reference_reps``
(10,000 or 20,000 replications: ten times the test's or more) from
``reference_seed``, a seed the test does not use, with bootstrap seeds 1,000,000 +
i, so the reference and the test are independent simulations of the same
setting. The file records each setting's ``params`` (the tests refuse a reference
made for other parameters), its replications, seed and counts, the exact rate
where the setting knows one (the rank tests and the Hedges' g intervals, whose
simulated rate must then agree with it), and the run time.

``--true-values`` checks the derived population values the coverage settings use
against large Monte Carlo samples (one sample of 2,000,000 per value, or 400,000
subjects): the 20% winsorized correlation and Spearman's rho of a bivariate
normal, xi, the limit of rank epsilon squared and of Kendall's W, the 20%
trimmed mean of a standard lognormal, and (from one table of 2,000,000
observations each) the bias-corrected Cramer's V of barstats' two 2 x 3 designs
and Pearson's C of its one-way design. These estimates are computed with NumPy
and SciPy, not with inferplot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests import simulation_settings as ss  # noqa: E402

BOOTSTRAP_OFFSET = 1_000_000


def _load() -> dict[str, Any]:
    return ss.load_reference()


def _save(data: dict[str, Any]) -> None:
    ordered = {key: data[key] for key in sorted(data)}
    with ss.REFERENCE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=1, sort_keys=False)
        handle.write("\n")


def run_setting(name: str) -> dict[str, Any]:
    setting = ss.BY_NAME[name]
    if setting.reference_reps == 0:
        raise SystemExit(f"{name} has only quantities of kind 1: no reference")
    start = time.perf_counter()
    hits = ss.run(
        setting,
        setting.reference_reps,
        setting.reference_seed,
        offset=BOOTSTRAP_OFFSET,
    )
    seconds = time.perf_counter() - start
    entry: dict[str, Any] = {
        "params": setting.params,
        "reps": setting.reference_reps,
        "seed": setting.reference_seed,
        "hits": hits,
        "seconds": round(seconds, 1),
    }
    if setting.exact:
        entry["exact"] = setting.exact
    return entry


def _winsorized_corr(x: np.ndarray, y: np.ndarray, tr: float = 0.2) -> float:
    def wins(v: np.ndarray) -> np.ndarray:
        s = np.sort(v)
        g = math.floor(tr * v.size)
        return np.clip(v, s[g], s[v.size - g - 1])

    return float(np.corrcoef(wins(x), wins(y))[0, 1])


def _winsorized_var(v: np.ndarray, tr: float = 0.2) -> float:
    s = np.sort(v)
    g = math.floor(tr * v.size)
    return float(np.var(np.clip(v, s[g], s[v.size - g - 1]), ddof=1))


def true_values() -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(4040))
    big = 2_000_000
    out: dict[str, Any] = {}
    for rho in (0.3, 0.8):
        z = rng.standard_normal((big, 2))
        x, y = z[:, 0], rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1]
        out[f"winsorized_rho_{rho}"] = {
            "formula": ss.winsorized_rho_of_normal(rho),
            "monte_carlo": _winsorized_corr(x, y),
        }
        out[f"spearman_rho_{rho}"] = {
            "formula": ss.spearman_rho_of_normal(rho),
            "monte_carlo": float(sps.spearmanr(x, y)[0]),
        }
    groups = [rng.standard_normal(big) + m for m in ss.XI_MEANS]
    tmeans = [float(sps.trim_mean(g, 0.2)) for g in groups]
    pooled = _winsorized_var(np.concatenate(groups))
    xi = math.sqrt(np.var(tmeans, ddof=1) / (pooled / ss.winsorized_normal_variance()))
    out["xi"] = {"formula": ss.XI_TRUE, "monte_carlo": xi}
    sizes = [s * 50_000 for s in ss.KW_SIZES]
    kw_groups = [
        rng.standard_normal(n) + m for n, m in zip(sizes, ss.EPS_MEANS, strict=True)
    ]
    h = float(sps.kruskal(*kw_groups)[0])
    out["epsilon_squared"] = {
        "formula": ss.EPS_TRUE,
        "monte_carlo": h / (sum(sizes) - 1),
    }
    subjects = 400_000
    table = rng.standard_normal((subjects, len(ss.KENDALL_EFFECTS))) + np.asarray(
        ss.KENDALL_EFFECTS
    )
    chi2 = float(sps.friedmanchisquare(*table.T)[0])
    out["kendall_w"] = {
        "formula": ss.W_TRUE,
        "monte_carlo": chi2 / (subjects * (len(ss.KENDALL_EFFECTS) - 1)),
    }
    out["lognormal_trimmed_mean"] = {
        "formula": ss.LOGNORMAL_TM,
        "monte_carlo": float(sps.trim_mean(np.exp(rng.standard_normal(big)), 0.2)),
    }
    # barstats, from a generator of its own (the values above stay as they were).
    bar_rng = np.random.Generator(np.random.PCG64(4041))
    for label, cells, true in (
        ("weak", ss.V_WEAK, ss.V_WEAK_TRUE),
        ("strong", ss.V_STRONG, ss.V_STRONG_TRUE),
    ):
        table = bar_rng.multinomial(big, np.ravel(cells)).reshape(2, 3)
        out[f"cramers_v_{label}"] = {
            "formula": true,
            "monte_carlo": _adjusted_cramers_v(table.astype(np.float64)),
        }
    counts = bar_rng.multinomial(big, ss.GOF_C_PROBS).astype(np.float64)
    expected = big / len(counts)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    out["pearsons_c"] = {
        "formula": ss.GOF_C_TRUE,
        "monte_carlo": math.sqrt(chi2 / (chi2 + big)),
    }
    return out


def _adjusted_cramers_v(table: np.ndarray) -> float:
    """Bergsma's (2013) bias-corrected Cramer's V of a table."""
    n = float(table.sum())
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    r, c = table.shape
    phi2 = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
    rows = r - (r - 1) ** 2 / (n - 1)
    cols = c - (c - 1) ** 2 / (n - 1)
    return math.sqrt(phi2 / (min(rows, cols) - 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--true-values", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for s in ss.ALL:
            if s.reference_reps:
                print(s.name, s.reference_reps)
        return
    data = _load()
    if args.true_values:
        data["_true_values"] = true_values()
        _save(data)
        print(json.dumps(data["_true_values"], indent=1))
    for name in args.only:
        entry = run_setting(name)
        data = _load()
        data[name] = entry
        _save(data)
        rates = {k: v / entry["reps"] for k, v in entry["hits"].items()}
        print(name, entry["seconds"], "s", rates, entry.get("exact", ""), flush=True)


if __name__ == "__main__":
    main()
