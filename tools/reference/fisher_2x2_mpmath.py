# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""40-digit references for Fisher's exact test of 2 x 2 tables at totals from 10 to
1e10 (tests/data/fisher_2x2_reference.json), for tests/stats/test_fisher_cost.py.

The first cell A of a 2 x 2 table with fixed margins is hypergeometric. From the
observed cell a, the probabilities are walked outward in both directions by the
exact ratio of consecutive terms, P(k + 1) / P(k) = (c0 - k)(r1 - k) / ((k + 1)(c1
- r1 + k + 1)), in 40-digit arithmetic, P(a) itself from log-gamma; each walk stops
at the end of the range, or once past the mode the terms are below 1e-45 of the
sum. "greater" is P(A >= a), "less" P(A <= a), and the two-sided p sums the terms
at most P(a) (1 + 1e-7) (R's fisher.test). The tables: balanced margins with the
observed cell 0.3 and 3 standard deviations from the mode, unequal margins 2
standard deviations out, an exactly symmetric table (a mirror-image tie), and a
far tail (8 standard deviations).

usage: uv run --with mpmath python -I tools/reference/fisher_2x2_mpmath.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import mpmath as mp

mp.mp.dps = 40
OUT = Path("tests/data/fisher_2x2_reference.json")


def _log_choose(m: int, k: int) -> mp.mpf:
    return mp.loggamma(m + 1) - mp.loggamma(k + 1) - mp.loggamma(m - k + 1)


def reference(a: int, b: int, c: int, d: int) -> dict[str, str]:
    r1, c0 = a + b, a + c
    c1 = b + d
    n = r1 + c + d
    lo, hi = max(0, r1 - c1), min(r1, c0)
    p_obs = mp.exp(_log_choose(c0, a) + _log_choose(c1, r1 - a) - _log_choose(n, r1))
    limit = p_obs * (1 + mp.mpf(10) ** -7)
    up = mp.mpf(0)
    up_counted = mp.mpf(0)
    p, k = p_obs, a
    while True:
        up += p
        if p <= limit:
            up_counted += p
        if k == hi:
            break
        nxt = p * (c0 - k) * (r1 - k) / (mp.mpf(k + 1) * (c1 - r1 + k + 1))
        if nxt < p and nxt < up * mp.mpf(10) ** -45:
            break
        p, k = nxt, k + 1
    down = mp.mpf(0)
    down_counted = mp.mpf(0)
    p, k = p_obs, a
    while k > lo:
        nxt = p * k * (c1 - r1 + k) / (mp.mpf(c0 - k + 1) * (r1 - k + 1))
        if nxt < p and nxt < (down + p_obs) * mp.mpf(10) ** -45:
            break
        p, k = nxt, k - 1
        down += p
        if p <= limit:
            down_counted += p
    return {
        "two-sided": mp.nstr(min(mp.mpf(1), up_counted + down_counted), 30),
        "greater": mp.nstr(min(mp.mpf(1), up), 30),
        "less": mp.nstr(min(mp.mpf(1), down + p_obs), 30),
    }


def tables(n: int) -> dict[str, tuple[int, int, int, int]]:
    s = math.sqrt(n)
    q = n // 4
    return {
        "balanced, 0.3 sd": (q + round(0.15 * s), q, q, q),
        "balanced, 3 sd": (q + round(1.5 * s), q, q, q),
        "unequal margins, 2 sd": (
            n // 50 + round(0.26 * s),
            n // 5 - n // 50,
            n // 10 - n // 50,
            n - n // 5 - n // 10 + n // 50,
        ),
        "exactly symmetric (ties)": (
            q + round(0.2 * s),
            q - round(0.2 * s),
            q - round(0.2 * s),
            q + round(0.2 * s),
        ),
        "far tail, 8 sd": (q + round(4 * s), q, q, q),
    }


def main() -> None:
    rows = []
    start = time.time()
    for e in range(1, 11):
        n = 10**e
        for name, cells in tables(n).items():
            if min(cells) < 0:
                continue
            a, b, c, d = cells
            rows.append(
                {
                    "n_power": e,
                    "case": name,
                    "table": [[a, b], [c, d]],
                    "p": reference(a, b, c, d),
                }
            )
            print(e, name, f"{time.time() - start:.1f} s", flush=True)
    OUT.write_text(
        json.dumps(
            {"generator": "tools/reference/fisher_2x2_mpmath.py", "rows": rows},
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
