"""Exact-decimal replay of every robust bootstrap in inferplot (Phase 4 review, Z1).

Each function replays one of inferplot's robust bootstraps on the same random
stream (``numpy.random.Generator(PCG64(seed))``, the same draw order) but computes
every resample in exact rational arithmetic on the numbers as written:
``Fraction(Decimal(repr(v)))``, the shortest decimal that reads back as the double.
It classifies each resample as 0 / 0, infinite or finite, counts the resamples
whose T* is at least |t| (the one-sample test), and returns the percentile bounds
from floating-point values of the exact resamples. inferplot must reproduce every
classification and count exactly, and the bounds to rounding (DECISIONS.md D74:
numbers that are the same as written are the same).

The draw order is inferplot's documented one (the stream is part of the method):
a group's sorted values (betweenstats); the groups of the explanatory measure in
the order of (size, sorted values); subjects in lexicographic order of their
values (withinstats); the one-sample deviations from the trimmed mean in the order
(|d|, d) of the deviations as written, ties in the order of the sorted values.

Run the invariance table (each method under about two minutes):

    uv run python -I tools/reference/decimal_replay.py one_sample
    uv run python -I tools/reference/decimal_replay.py akp xi dependent average

For every method it draws 60 rating-scale and 60 one-decimal data sets, writes
each in whole numbers, in tenths and shifted by a tenth, compares inferplot with
this replay (classification and counts must agree exactly, bounds to 1e-9
relative) and every writing with the whole numbers after rescaling; and compares
four computed writings (times 0.3 and 2.54, parsed from decimal text and computed
in binary) with the whole numbers.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

from inferplot.stats import robust, within_robust
from inferplot.stats.one_sample import trimmed_mean_bootstrap_t
from inferplot.stats.robust import winsorized_normal_variance

B = 2000
_CHUNK_VALUES = 1 << 22


def as_written(value: float) -> Fraction:
    """The number as written: the shortest decimal that reads back as the double."""
    return Fraction(Decimal(repr(float(value))))


def _trim(values: Sequence[Fraction], g: int) -> tuple[Fraction, Fraction]:
    """Exact trimmed mean and winsorized sum of squares."""
    s = sorted(values)
    m = len(s)
    mid = s[g : m - g]
    tm = sum(mid, Fraction(0)) / len(mid)
    w = [s[g]] * g + mid + [s[m - g - 1]] * g
    centre = sum(w, Fraction(0)) / m
    return tm, sum(((v - centre) ** 2 for v in w), Fraction(0))


def _row_chunks(values_per_row: int) -> list[int]:
    size = max(1, min(B, _CHUNK_VALUES // max(1, values_per_row)))
    full, rest = divmod(B, size)
    return [size] * full + ([rest] if rest else [])


def _draw(seed: int, counts: Sequence[int], sizes: Sequence[int]) -> list[Any]:
    """The positions inferplot draws: per chunk, group by group."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out: list[list[Any]] = [[] for _ in counts]
    for rows in _row_chunks(sum(sizes)):
        for j, (count, size) in enumerate(zip(counts, sizes, strict=True)):
            out[j].append(rng.integers(0, count, size=(rows, size)))
    return [np.concatenate(parts) for parts in out]


def _sorted_defined(values: list[float]) -> list[float]:
    return sorted(v for v in values if not math.isnan(v))


def one_sample(
    x: Sequence[float],
    mu: float,
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """``trimmed_mean_bootstrap_t`` (WRS2::trimcibt) replayed exactly."""
    doubles = sorted(float(v) for v in x)
    n = len(doubles)
    g = math.floor(tr * n)
    w = [as_written(v) for v in doubles]
    tm, wss = _trim(w, g)
    mu_w = as_written(mu)
    d = [v - tm for v in w]
    order = sorted(range(n), key=lambda i: (abs(d[i]), d[i]))
    source = [w[i] for i in order]
    shrink = (1 - 2 * tr) * math.sqrt(n)
    target = (tm - mu_w) ** 2 / wss  # t^2 / ((n - 1) shrink^2)
    (positions,) = _draw(seed, [n], [n])
    t_star: list[float] = []
    undefined = infinite = at_least = 0
    for row in positions.tolist():
        tm_s, wss_s = _trim([source[i] for i in row], g)
        if wss_s == 0:
            if tm_s == tm:
                undefined += 1
                t_star.append(math.nan)
            else:
                infinite += 1
                at_least += 1
                t_star.append(math.inf)
            continue
        ratio = (tm_s - tm) ** 2 / wss_s
        at_least += ratio >= target
        t_star.append(shrink * math.sqrt(float(ratio * (n - 1))))
    defined = _sorted_defined(t_star)
    k = round((1.0 - (1.0 - conf_level)) * len(defined))
    critical = defined[k - 1]
    se = math.sqrt(float(wss / (n - 1))) / shrink
    centre = float(tm)
    return {
        "undefined": undefined,
        "infinite": infinite,
        "at_least": at_least,
        "p": at_least / len(defined),
        "t_zero": tm == mu_w,
        "conf_low": centre - critical * se,
        "conf_high": centre + critical * se,
    }


def akp(
    a: Sequence[float],
    b: Sequence[float],
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """``akp_effect`` (WRS2::akp.effect) replayed exactly."""
    wa = [as_written(v) for v in sorted(float(v) for v in a)]
    wb = [as_written(v) for v in sorted(float(v) for v in b)]
    na, nb = len(wa), len(wb)
    ga, gb = math.floor(tr * na), math.floor(tr * nb)
    c = math.sqrt(winsorized_normal_variance(tr))
    ia, ib = _draw(seed, [na, nb], [na, nb])
    values: list[float] = []
    for ra, rb in zip(ia.tolist(), ib.tolist(), strict=True):
        tm_a, wss_a = _trim([wa[i] for i in ra], ga)
        tm_b, _ = _trim([wb[i] for i in rb], gb)
        num = tm_a - tm_b
        if wss_a == 0:
            values.append(math.nan if num == 0 else math.copysign(math.inf, num))
        else:
            values.append(c * float(num) / math.sqrt(float(wss_a / (na - 1))))
    return _floor_bounds(values, conf_level)


def _counts(values: list[float]) -> dict[str, Any]:
    return {
        "undefined": sum(math.isnan(v) for v in values),
        "infinite": sum(math.isinf(v) for v in values),
        "zero": sum(v == 0.0 for v in values),
    }


def _floor_bounds(values: list[float], conf_level: float) -> dict[str, Any]:
    defined = _sorted_defined(values)
    alpha = 1.0 - conf_level
    low = math.floor(alpha / 2.0 * len(defined))
    high = math.floor((1.0 - alpha / 2.0) * len(defined))
    return {**_counts(values), "conf_low": defined[low], "conf_high": defined[high - 1]}


def _rounded_bounds(values: list[float], alpha: float) -> tuple[float, float]:
    defined = _sorted_defined(values)
    low = round(alpha / 2.0 * len(defined))
    return defined[low], defined[len(defined) - low - 1]


def xi(
    groups: Sequence[Sequence[float]],
    tr: float = 0.2,
    conf_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """``explanatory_effect`` (WRS2::t1way's effect size) replayed exactly."""
    ordered = sorted(
        (sorted(float(v) for v in grp) for grp in groups), key=lambda s: (len(s), s)
    )
    written = [[as_written(v) for v in s] for s in ordered]
    size = min(len(s) for s in ordered)
    k = len(ordered)
    g = math.floor(tr * size)
    gp = math.floor(tr * size * k)
    normal = Fraction(winsorized_normal_variance(tr))
    positions = _draw(seed, [len(s) for s in ordered], [size] * k)
    values: list[float] = []
    for r in range(B):
        rows = [[written[j][i] for i in positions[j][r].tolist()] for j in range(k)]
        tms = [_trim(row, g)[0] for row in rows]
        _, pooled = _trim([v for row in rows for v in row], gp)
        mean_tm = sum(tms, Fraction(0)) / k
        top = sum(((t - mean_tm) ** 2 for t in tms), Fraction(0)) / (k - 1)
        if pooled == 0:
            values.append(math.nan if top == 0 else math.inf)
        else:
            values.append(math.sqrt(float(top / (pooled / (k * size - 1) / normal))))
    low, high = _rounded_bounds(values, 1.0 - conf_level)
    return {**_counts(values), "conf_low": low, "conf_high": high}


def dependent(
    conditions: Sequence[Sequence[float]], tr: float = 0.2, seed: int = 0
) -> dict[str, Any]:
    """``dependent_akp`` (two conditions) or ``average_dependent_akp`` (three or
    more; WRS2::dep.effect and wmcpAKP) replayed exactly: subjects in
    lexicographic order of their values, the same resamples for every pair."""
    columns = [[float(v) for v in c] for c in conditions]
    n = len(columns[0])
    rows = sorted(range(n), key=lambda i: tuple(c[i] for c in columns))
    written = [[as_written(c[i]) for i in rows] for c in columns]
    k = len(columns)
    g = math.floor(tr * n)
    c = math.sqrt(winsorized_normal_variance(tr))
    (positions,) = _draw(seed, [n], [n])
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    alpha = float(Decimal(1) - Decimal(repr(0.95)))
    lows: list[float] = []
    highs: list[float] = []
    undefined: list[int] = []
    infinite: list[int] = []
    zero: list[int] = []
    for i, j in pairs:
        d = [x - y for x, y in zip(written[i], written[j], strict=True)]
        values: list[float] = []
        for row in positions.tolist():
            tm, wss = _trim([d[s] for s in row], g)
            if wss == 0:
                values.append(math.nan if tm == 0 else math.copysign(math.inf, tm))
            else:
                values.append(c * float(tm) / math.sqrt(float(wss / (n - 1))))
        low, high = _rounded_bounds(values, alpha)
        lows.append(low)
        highs.append(high)
        counts = _counts(values)
        undefined.append(counts["undefined"])
        infinite.append(counts["infinite"])
        zero.append(counts["zero"])

    def average(v: list[float]) -> float:
        return float(sum(v)) if any(math.isinf(x) for x in v) else math.fsum(v) / len(v)

    return {
        "undefined": tuple(undefined),
        "infinite": tuple(infinite),
        "zero": tuple(zero),
        "conf_low": average(lows),
        "conf_high": average(highs),
    }


# ------------------------------------------------------------------ the table


def _same(x: float, y: float, rel: float = 1e-9) -> bool:
    if math.isinf(x) or math.isinf(y) or math.isnan(x) or math.isnan(y):
        return x == y or (math.isnan(x) and math.isnan(y))
    return abs(x - y) <= rel * max(1.0, abs(x), abs(y))


WRITINGS: dict[str, tuple[Callable[[float], float], float, float, bool]] = {
    # name: (map of a whole number, unit, shift, replayed exactly)
    "whole": (lambda v: v, 1.0, 0.0, True),
    "tenths": (lambda v: v / 10, 0.1, 0.0, True),
    "shifted": (
        lambda v: float(Decimal(repr(v)) / 10 + Decimal("0.1")),
        0.1,
        0.1,
        True,
    ),
    "x0.3 text": (lambda v: float(Decimal(repr(v)) * Decimal("0.3")), 0.3, 0.0, False),
    "x0.3 binary": (lambda v: v * 0.3, 0.3, 0.0, False),
    "x2.54 text": (
        lambda v: float(Decimal(repr(v)) * Decimal("2.54")),
        2.54,
        0.0,
        False,
    ),
    "x2.54 binary": (lambda v: v * 2.54, 2.54, 0.0, False),
    "+0.1 binary": (lambda v: v / 10 + 0.1, 0.1, 0.1, False),
}


def _apply(f: Callable[[float], float], values: np.ndarray) -> np.ndarray:
    return np.array([f(float(v)) for v in values])


def _ratings(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(1, 6, n).astype(float)


def _one_decimal(rng: np.random.Generator, n: int) -> np.ndarray:
    # one-decimal data as whole numbers of tenths (body temperatures)
    return np.round(rng.normal(368, 4, n)).astype(float)


def _data(method: str, kind: str, rng: np.random.Generator) -> list[np.ndarray]:
    make = _ratings if kind == "ratings" else _one_decimal
    if method == "one_sample":
        return [make(rng, int(rng.integers(6, 21)))]
    if method == "akp":
        return [
            make(rng, int(rng.integers(6, 16))),
            make(rng, int(rng.integers(6, 16))),
        ]
    if method == "xi":
        return [make(rng, int(rng.integers(6, 13))) for _ in range(3)]
    n = int(rng.integers(6, 21))
    a = make(rng, n)
    step = 1.0
    shifts = rng.choice([-step, 0.0, step, 2 * step], n, p=[0.15, 0.5, 0.25, 0.1])
    b = np.clip(a + shifts, 1, 5) if kind == "ratings" else a + shifts
    if method == "dependent":
        return [a, b]
    c = (
        np.clip(b + rng.choice([-step, 0.0, step], n), 1, 5)
        if kind == "ratings"
        else b + rng.choice([-step, 0.0, step], n)
    )
    return [a, b, c]


def _inferplot(method: str, data: list[np.ndarray], mu: float) -> dict[str, Any]:
    if method == "one_sample":
        r = trimmed_mean_bootstrap_t(data[0], mu=mu)
        return {
            "undefined": r.undefined_resamples,
            "infinite": r.infinite_resamples,
            "p": r.p_value,
            "t_zero": r.statistic == 0.0,
            "conf_low": r.conf_low,
            "conf_high": r.conf_high,
            "estimate": r.estimate,
        }
    if method == "akp":
        r = robust.akp_effect(data[0], data[1])
    elif method == "xi":
        r = robust.explanatory_effect(data)
    elif method == "dependent":
        r = within_robust.dependent_akp(data[0], data[1])
    else:
        r = within_robust.average_dependent_akp(data)
    undefined: Any = r.undefined_resamples
    infinite: Any = r.infinite_resamples
    if method in ("dependent", "average"):
        undefined = getattr(r, "pair_undefined", None) or (undefined,)
        infinite = getattr(r, "pair_infinite", None) or (infinite,)
    return {
        "undefined": undefined,
        "infinite": infinite,
        "conf_low": r.conf_low,
        "conf_high": r.conf_high,
        "estimate": r.estimate,
    }


def _replay(method: str, data: list[np.ndarray], mu: float) -> dict[str, Any]:
    if method == "one_sample":
        return one_sample(data[0].tolist(), mu)
    if method == "akp":
        return akp(data[0].tolist(), data[1].tolist())
    if method == "xi":
        return xi([d.tolist() for d in data])
    return dependent([d.tolist() for d in data])


def _rescaled(method: str, value: float, unit: float, shift: float) -> float:
    if method != "one_sample" or math.isinf(value):
        return value
    return (value - shift) / unit


def _shown(method: str, key: str, value: Any, unit: float, shift: float) -> Any:
    if key.startswith(("conf", "est")):
        return _rescaled(method, value, unit, shift)
    return value


def table(method: str, per_kind: int = 60, seed: int = 2026) -> list[str]:
    """Rows of the invariance table for ``method``."""
    rng = np.random.default_rng([seed, sorted(METHODS).index(method)])
    lines: list[str] = []
    for kind in ("ratings", "one-decimal"):
        computed = refused = 0
        replay_diff = {name: 0 for name, w in WRITINGS.items() if w[3]}
        invariance_diff = dict.fromkeys(WRITINGS, 0)
        examples: list[str] = []
        while computed + refused < per_kind:
            base = _data(method, kind, rng)
            mu_whole = float(
                rng.choice([2.5, 3.0])
                if kind == "ratings"
                else rng.choice([368.0, 370.0, 372.5])
            )
            try:
                reference = _inferplot(method, base, mu_whole)
            except ValueError:
                refused += 1
                continue
            computed += 1
            for name, (f, unit, shift, replayed) in WRITINGS.items():
                data = [_apply(f, d) for d in base]
                mu = f(mu_whole)
                try:
                    got = _inferplot(method, data, mu)
                except ValueError as error:
                    invariance_diff[name] += 1
                    examples.append(f"{kind} {name}: refused ({error})"[:160])
                    continue
                if replayed:
                    want = _replay(method, data, mu)
                    keys = ["undefined", "infinite"] + (
                        ["p", "t_zero"] if method == "one_sample" else []
                    )
                    bad = [k for k in keys if got[k] != want[k]] + [
                        k
                        for k in ("conf_low", "conf_high")
                        if not _same(got[k], want[k])
                    ]
                    if bad:
                        replay_diff[name] += 1
                        if len(examples) < 6:
                            examples.append(
                                f"{kind} {name} replay: "
                                + ", ".join(f"{k} {got[k]} vs {want[k]}" for k in bad)
                            )
                keys = ["undefined", "infinite"] + (
                    ["p", "t_zero"] if method == "one_sample" else []
                )
                bad = [k for k in keys if got[k] != reference[k]] + [
                    k
                    for k in ("conf_low", "conf_high", "estimate")
                    if not _same(_rescaled(method, got[k], unit, shift), reference[k])
                ]
                if bad:
                    invariance_diff[name] += 1
                    if len(examples) < 6:
                        examples.append(
                            f"{kind} {name} vs whole: "
                            + ", ".join(
                                f"{k} {_shown(method, k, got[k], unit, shift)}"
                                f" vs {reference[k]}"
                                for k in bad
                            )
                        )
        lines.append(
            f"{method:10s} {kind:11s} {computed} data sets ({refused} refused): "
            + "replay differences "
            + ", ".join(f"{k} {v}" for k, v in replay_diff.items())
            + " | differences from whole numbers "
            + ", ".join(f"{k} {v}" for k, v in invariance_diff.items() if k != "whole")
        )
        lines.extend("    " + e for e in examples)
    return lines


METHODS: tuple[str, ...] = ("one_sample", "akp", "xi", "dependent", "average")

if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")
    for name in sys.argv[1:] or METHODS:
        for line in table(name):
            print(line, flush=True)
