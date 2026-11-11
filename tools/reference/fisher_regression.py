"""Regression table for Fisher's exact test of 2 x c tables (tests/data/
fisher_regression.json).

Generated once from the implementation that preceded the bounded-cost search (main
at commit 2c015dd, a breadth-first network search over the columns): every table
below that the earlier code answered, with its p-value, so that
tests/stats/test_fisher_cost.py can check that the new search gives the same
p-values (``closed_form``) on everything the earlier one answered. Tables are drawn
from fixed seeds: 2 x 2 to 2 x 9, totals from 4 to 20,000, near and far from
independence, equal and unequal column totals (equal totals make mirror-image
ties), and sparse rows.

usage (from the repository root, at the commit named above):
    uv run python -I tools/reference/fisher_regression.py BATCH [BATCH ...]
writes tests/data/fisher_regression.json, merging the given batches (0 to 7) into
what is there.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from inferplot.stats.contingency import fisher_exact

OUT = Path("tests/data/fisher_regression.json")
PER_BATCH = 60
SECONDS = 1.5


def _table(rng: np.random.Generator) -> np.ndarray:
    c = int(rng.choice([2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9]))
    kind = int(rng.integers(0, 5))
    if c <= 3:
        n = int(rng.choice([8, 30, 100, 400, 2_000, 8_000, 20_000]))
    elif c <= 5:
        n = int(rng.choice([8, 30, 80, 200, 500, 1_200]))
    else:
        n = int(rng.choice([6, 20, 50, 100, 160, 240] if c <= 6 else [6, 20, 50, 100, 140]))
    if kind == 0:  # equal column totals: mirror-image ties
        cols = np.full(c, max(1, n // c))
    elif kind == 1:  # one sparse row
        cols = rng.multinomial(n, np.full(c, 1 / c)) + 1
        first = rng.binomial(cols, 0.08)
        return np.array([first, cols - first], dtype=np.int64)
    else:
        cols = rng.multinomial(n, rng.dirichlet(np.full(c, 2.0))) + 1
    tilt = float(rng.choice([0.0, 0.0, 0.05, 0.15, 0.3, 0.45]))
    shares = np.clip(0.5 + tilt * np.linspace(-1, 1, c), 0.0, 1.0)
    first = rng.binomial(cols, shares)
    return np.array([first, cols - first], dtype=np.int64)


def main(batches: list[int]) -> None:
    rows = json.loads(OUT.read_text())["tables"] if OUT.exists() else []
    seen = {(r["batch"], r["index"]) for r in rows}
    for batch in batches:
        rng = np.random.default_rng(20261110 + batch)
        for index in range(PER_BATCH):
            table = _table(rng)
            if (batch, index) in seen:
                continue
            sides = ["two-sided"]
            if table.shape[1] == 2:
                sides += ["less", "greater"]
            for side in sides:
                start = time.perf_counter()
                try:
                    p = fisher_exact(table, alternative=side).p_value  # type: ignore[arg-type]
                except ValueError:
                    continue
                seconds = time.perf_counter() - start
                if seconds > SECONDS:
                    continue
                rows.append(
                    {
                        "batch": batch,
                        "index": index,
                        "table": table.tolist(),
                        "alternative": side,
                        "p_value": p,
                        "p_value_hex": float(p).hex(),
                    }
                )
            print(batch, index, table.shape[1], int(table.sum()), flush=True)
    rows.sort(key=lambda r: (r["batch"], r["index"], r["alternative"]))
    OUT.write_text(
        json.dumps(
            {
                "source": "inferplot main at 2c015dd, stats.contingency.fisher_exact",
                "generator": "tools/reference/fisher_regression.py",
                "tables": rows,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main([int(v) for v in sys.argv[1:]])
