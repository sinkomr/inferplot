# ruff: noqa: E501  (published tables are kept one row per line)
"""p_adjust against R outputs, the fixtures, exact rationals and its properties.

Sections (from the reviewed prototype; the statsmodels and mpmath cross-checks ran
during development and are not repeated here, since neither is a dependency):

    reference Holm     tools/reference holm, bit for bit
    fixtures           Holm over Games-Howell, Dunn and lincon p-values reproduces
                       every adjusted pairwise p in the k >= 3 fixtures
    published values   an R p.adjust(p, "hommel") output; R multtest outputs for
                       Bonferroni, Holm, Hochberg, BH and BY; Wikipedia's Holm
                       example; Benjamini & Hochberg's (1995) example
    exact rationals    every method from its definition in Fractions (Hommel by
                       full closed testing)
    properties         bounds, order, ties, permutation equivariance, n = 1 and 2
    operation order    the BH quotient orderings differ by at most 2 ulp
    errors             argument validation
    harmonic sum       BY's q: math.fsum against an emulated 64-bit significand sum
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats
from tools.reference import stats_reference as ref

from inferplot.stats import P_ADJUST_METHODS, PAdjustMethod, games_howell, p_adjust
from tests.oracle_fixtures import cases_for, number
from tests.stats.fixture_data import groups, p_adjust_method
from tests.tolerances import assert_close

FloatArray = NDArray[np.float64]


def ulps(a: float, b: float) -> float:
    """Distance in units of the last place of ``b``."""
    if a == b:
        return 0.0
    return abs(a - b) / math.ulp(b if b != 0.0 else a)


def random_vectors(seed: int, count: int) -> list[FloatArray]:
    """p-value vectors with ties, values at 0 and 1, values near 1, tiny values."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out: list[FloatArray] = []
    for _ in range(count):
        n = int(rng.integers(1, 41))
        kind = int(rng.integers(0, 5))
        if kind == 0:
            p = rng.uniform(size=n)
        elif kind == 1:  # mostly small: signals
            p = rng.uniform(size=n) ** 6
        elif kind == 2:  # ties from rounding
            p = np.round(rng.uniform(size=n), 2)
        elif kind == 3:  # near 1
            p = 1.0 - rng.uniform(size=n) * 1e-3
        else:  # a mixture with exact 0, exact 1 and repeats
            pool = np.array([0.0, 1.0, 1e-300, 0.05, 0.05, 0.5, 1 - 2**-53])
            p = np.where(
                rng.uniform(size=n) < 0.5, rng.choice(pool, size=n), rng.uniform(size=n)
            )
        out.append(p.astype(np.float64))
    return out


# --------------------------------------------------------------- reference Holm


def test_reference_holm_bit_for_bit() -> None:
    for p in random_vectors(1, 5000):
        assert p_adjust(p, "holm") == tuple(ref.holm(p.tolist()))


# --------------------------------------------------------------------- fixtures


def _raw_pairwise_p(type_: str, data: list[FloatArray], tr: float = 0.2) -> list[float]:
    """Unadjusted pairwise p-values, pairs i < j in level order."""
    index_pairs = list(itertools.combinations(range(len(data)), 2))
    if type_ == "parametric":
        return [pair.p_value for pair in games_howell(data, p_adjust_method="none")]
    if type_ == "nonparametric":  # Dunn (1964) with the tie correction
        pooled = np.concatenate(data)
        n = pooled.size
        sizes = [g.size for g in data]
        ranks = stats.rankdata(pooled)
        mean_ranks = [float(r.mean()) for r in np.split(ranks, np.cumsum(sizes)[:-1])]
        ties = np.unique(pooled, return_counts=True)[1].astype(np.float64)
        s2 = n * (n + 1) / 12 - float((ties**3 - ties).sum()) / (12 * (n - 1))
        return [
            float(
                2
                * stats.norm.sf(
                    abs(mean_ranks[i] - mean_ranks[j])
                    / math.sqrt(s2 * (1 / sizes[i] + 1 / sizes[j]))
                )
            )
            for i, j in index_pairs
        ]
    # robust: lincon's per-pair Yuen p, which WRS2 computes as 1 - pt
    raw: list[float] = []
    for i, j in index_pairs:
        t, df, _ = ref.yuen(data[i], data[j], tr=tr)
        raw.append(2 * (1.0 - (1.0 - float(stats.t.sf(abs(t), df)))))
    return raw


FIXTURE_KINDS = {
    "parametric": "games_howell_p",
    "nonparametric": "closed_form",
    "robust": "wrs2_p",
}


def _pairwise_cases() -> list[object]:
    cases = [
        case
        for case in cases_for("ggbetweenstats")
        if case.error is None
        and len(case.analysis_levels) >= 3
        and case.type in FIXTURE_KINDS
        and case.pairwise
    ]
    assert cases
    return [pytest.param(case, id=case.name) for case in cases]


@pytest.mark.parametrize("case", _pairwise_cases())
def test_fixture_adjusted_p_values(case: object) -> None:
    from tests.oracle_fixtures import OracleCase

    assert isinstance(case, OracleCase)
    levels = case.analysis_levels
    pairs = list(itertools.combinations(levels, 2))
    adjusted = dict(
        zip(
            pairs,
            p_adjust(
                _raw_pairwise_p(
                    case.type, groups(case), float(case.args.get("tr", 0.2))
                ),
                p_adjust_method(case),
            ),
            strict=True,
        )
    )
    by_pair = {(str(r["group1"]), str(r["group2"])): r for r in case.pairwise}
    assert set(by_pair) == set(pairs), case.name
    # wrs2_p: one grid step of a one-sided tail becomes 2 x (the adjustment's largest
    # factor) steps after doubling and adjusting: m for most methods, m times the
    # harmonic sum for BY, 1 for none.
    m = len(pairs)
    method = p_adjust_method(case)
    factor = {"none": 1.0, "BY": m * math.fsum(1 / k for k in range(1, m + 1))}.get(
        method, float(m)
    )
    scale = 2.0 * factor if case.type == "robust" else 1.0
    for pair in pairs:
        assert_close(
            FIXTURE_KINDS[case.type],
            adjusted[pair],
            number(by_pair[pair]["p.value"]),
            what=f"{case.name} {pair}",
            atol_scale=scale,
        )


# ------------------------------------------------------------- published values

# R 'stats::p.adjust(pval0, method = "hommel")', printed with 16 significant digits,
# as recorded in statsmodels 0.15.0, statsmodels/stats/tests/test_multi.py,
# test_hommel (BSD-3; R output, not R source).
SM_HOMMEL_P = [
    0.00116, 0.00924, 0.01075, 0.01437, 0.01784, 0.01918, 0.02751, 0.02871, 0.03054,
    0.03246, 0.04259, 0.06879, 0.0691, 0.08081, 0.08593, 0.08993, 0.09386, 0.09412,
    0.09718, 0.09758, 0.09781, 0.09788, 0.13282, 0.20191, 0.21757, 0.24031, 0.26061,
    0.26762, 0.29474, 0.32901, 0.41386, 0.51479, 0.52461, 0.53389, 0.56276, 0.62967,
    0.72178, 0.73403, 0.87182, 0.95384,
]  # fmt: skip
SM_HOMMEL_R = [
    0.0464, 0.25872, 0.29025, 0.3495714285714286, 0.41032, 0.44114, 0.57771,
    0.60291, 0.618954, 0.6492, 0.7402725000000001, 0.86749, 0.86749,
    0.8889100000000001, 0.8971477777777778, 0.8993, 0.9175374999999999,
    0.9175374999999999, 0.9175374999999999, 0.9175374999999999, 0.9175374999999999,
    0.9175374999999999, 0.95384, *[0.9538400000000001] * 17,
]  # fmt: skip

# R multtest::mt.rawp2adjp outputs (columns rawp, Bonferroni, Holm, Hochberg,
# SidakSS, SidakSD, BH, BY, ...), as recorded in the same statsmodels file
# (res_multtest1/2/3). multtest is not p.adjust, so these pin the definitions, not
# R's rounding.
MT_COLUMNS: dict[PAdjustMethod, int] = {
    "bonferroni": 1,
    "holm": 2,
    "hochberg": 3,
    "BH": 6,
    "BY": 7,
}
MT1 = [
    [5.2365677720800003e-05, 5.2365677720800005e-04, 5.2365677720800005e-04, 5.2365677720800005e-04, 0, 0, 5.2365677720800005e-04, 1.5337740764175588e-03],
    [2.8140506198000000e-04, 2.8140506197999998e-03, 2.5326455578199999e-03, 2.5326455578199999e-03, 0, 0, 1.4070253098999999e-03, 4.1211324652269442e-03],
    [1.4987731079600001e-02, 1.4987731079600000e-01, 1.1990184863680001e-01, 1.1990184863680001e-01, 0, 0, 4.9959103598666670e-02, 1.4632862843720582e-01],
    [2.0273268879800001e-02, 2.0273268879799999e-01, 1.4191288215860001e-01, 1.4191288215860001e-01, 0, 0, 5.0683172199499998e-02, 1.4844940238274187e-01],
    [1.7709695272300000e-01, 1.0, 1.0, 9.6783382430900000e-01, 0, 0, 3.5419390544599999e-01, 1.0],
    [6.4219392379499995e-01, 1.0, 1.0, 9.6783382430900000e-01, 0, 0, 8.9533672797500008e-01, 1.0],
    [6.8084594763299999e-01, 1.0, 1.0, 9.6783382430900000e-01, 0, 0, 8.9533672797500008e-01, 1.0],
    [7.1626938238000004e-01, 1.0, 1.0, 9.6783382430900000e-01, 0, 0, 8.9533672797500008e-01, 1.0],
    [8.3854136755300002e-01, 1.0, 1.0, 9.6783382430900000e-01, 0, 0, 9.3171263061444454e-01, 1.0],
    [9.6783382430900000e-01, 1.0, 1.0, 9.6783382430900000e-01, 0, 0, 9.6783382430900000e-01, 1.0],
]  # fmt: skip
_MT2_FLAT = [
    0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.012, 0.024, 0.036, 0.048,
    0.06, 0.072, 0.012, 0.02, 0.024, 0.024, 0.024, 0.024, 0.012, 0.012,
    0.012, 0.012, 0.012, 0.012, 0.01194015976019192, 0.02376127616613988,
    0.03546430060660932, 0.04705017875634587, 0.058519850599,
    0.06987425045000606, 0.01194015976019192, 0.01984063872102404,
    0.02378486270400004, 0.023808512, 0.023808512, 0.023808512, 0.012,
    0.012, 0.012, 0.012, 0.012, 0.012, 0.0294, 0.0294, 0.0294, 0.0294,
    0.0294, 0.0294,
]  # fmt: skip
MT2 = np.array(_MT2_FLAT + [0.0] * 12).reshape(6, 10, order="F")
_MT3_FLAT = [
    0.001, 0.002, 0.003, 0.004, 0.005, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01,
    0.02, 0.03, 0.04, 0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 0.01, 0.018, 0.024,
    0.028, 0.03, 0.25, 0.25, 0.25, 0.25, 0.25, 0.01, 0.018, 0.024, 0.028,
    0.03, 0.09, 0.09, 0.09, 0.09, 0.09, 0.00995511979025177,
    0.01982095664805061, 0.02959822305108317, 0.03928762649718986,
    0.04888986953422814, 0.4012630607616213, 0.4613848859051006,
    0.5160176928207072, 0.5656115457763677, 0.6105838818818925,
    0.00995511979025177, 0.0178566699880266, 0.02374950634358763,
    0.02766623106147537, 0.02962749064373438, 0.2262190625000001,
    0.2262190625000001, 0.2262190625000001, 0.2262190625000001,
    0.2262190625000001, 0.01, 0.01, 0.01, 0.01, 0.01, 0.08333333333333334,
    0.0857142857142857, 0.0875, 0.0888888888888889, 0.09,
    0.02928968253968254, 0.02928968253968254, 0.02928968253968254,
    0.02928968253968254, 0.02928968253968254, 0.2440806878306878,
    0.2510544217687075, 0.2562847222222222, 0.2603527336860670,
    0.2636071428571428,
]  # fmt: skip
MT3 = np.array(_MT3_FLAT + [0.0] * 20).reshape(10, 10, order="F")


def test_r_hommel_output() -> None:
    ours = p_adjust(SM_HOMMEL_P, "hommel")
    assert [f"{v:.16g}" for v in ours] == [f"{v:.16g}" for v in SM_HOMMEL_R]


def test_r_multtest_outputs() -> None:
    for table in (np.array(MT1), MT2, MT3):
        raw = table[:, 0]
        for method, column in MT_COLUMNS.items():
            ours = np.array(p_adjust(raw, method))
            expected = table[:, column]
            # The tables print 10 to 17 significant digits; 1e-12 relative is
            # inside every one of them.
            assert np.all(np.abs(ours - expected) <= 1e-12 * expected), method


def test_wikipedia_holm_example() -> None:
    # https://en.wikipedia.org/wiki/Holm-Bonferroni_method, "Adjusted p-values":
    # p = 0.01, 0.04, 0.03, 0.005 -> 0.03, 0.06, 0.06, 0.02.
    ours = p_adjust([0.01, 0.04, 0.03, 0.005], "holm")
    assert np.allclose(ours, [0.03, 0.06, 0.06, 0.02], rtol=1e-15, atol=0)


def test_benjamini_hochberg_1995_example() -> None:
    # Benjamini & Hochberg (1995), section 4 (Neuhaus et al. 1992, 15 endpoints):
    # Bonferroni at 0.05 rejects 3, the BH procedure rejects 4.
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
         0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]  # fmt: skip
    rejected = {m: sum(v <= 0.05 for v in p_adjust(p, m)) for m in ("bonferroni", "BH")}
    assert rejected == {"bonferroni": 3, "BH": 4}


# -------------------------------------------------------------- exact rationals


def _exact(p: Sequence[float], method: str) -> list[Fraction]:
    """The definitions in exact rational arithmetic, straight from the papers."""
    n = len(p)
    f = [Fraction(v) for v in p]
    order = sorted(range(n), key=lambda i: f[i])
    s = [f[i] for i in order]
    one = Fraction(1)
    if method == "bonferroni":
        return [min(one, n * v) for v in f]
    if method == "none":
        return f
    if method == "holm":
        out_sorted = [
            min(one, max((n - j) * s[j] for j in range(i + 1))) for i in range(n)
        ]
    elif method == "hochberg":
        out_sorted = [
            min(min(one, (n - j) * s[j]) for j in range(i, n)) for i in range(n)
        ]
    elif method in ("BH", "fdr", "BY"):
        q = (
            sum((Fraction(1, k) for k in range(1, n + 1)), Fraction(0))
            if method == "BY"
            else one
        )
        out_sorted = [
            min(min(one, q * n * s[j] / (j + 1)) for j in range(i, n)) for i in range(n)
        ]
    elif method == "hommel":
        # Hommel (1988): closed testing with Simes' test; the adjusted p-value of
        # H_i is the maximum Simes p-value over all index sets containing i.
        def simes(subset: tuple[int, ...]) -> Fraction:
            vals = sorted(f[i] for i in subset)
            m = len(vals)
            return min(m * v / (k + 1) for k, v in enumerate(vals))

        adj = [Fraction(0)] * n
        for size in range(1, n + 1):
            for subset in itertools.combinations(range(n), size):
                value = simes(subset)
                for i in subset:
                    adj[i] = max(adj[i], value)
        return [min(one, a) for a in adj]
    else:
        raise AssertionError(method)
    out = [Fraction(0)] * n
    for rank, i in enumerate(order):
        out[i] = out_sorted[rank]
    return out


def test_exact_rationals() -> None:
    vectors = [v for v in random_vectors(4, 1500) if v.size <= 9]
    vectors += [
        np.array([0.5]),
        np.array([0.02, 0.02]),
        np.array([0.01, 0.04, 0.03, 0.005]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1 - 2**-53] * 5),
    ]
    worst: dict[str, float] = {}
    for p in vectors:
        for method in P_ADJUST_METHODS:
            ours = p_adjust(p, method)
            for a, e in zip(ours, _exact(p.tolist(), method), strict=True):
                worst[method] = max(worst.get(method, 0.0), ulps(a, float(e)))
    for method, w in worst.items():
        # Integer multipliers round once; the quotients of Hommel, BH and BY are
        # within 1, 1 and 2 ulp of the correctly rounded value.
        limit = 0 if method in ("none", "bonferroni", "holm", "hochberg") else 2
        assert w <= limit, (method, w)


# ------------------------------------------------------------------- properties


def test_properties() -> None:
    chain = [
        ("bonferroni", "holm"),
        ("holm", "hochberg"),
        ("hochberg", "hommel"),
        ("BY", "BH"),
        ("hochberg", "BH"),
    ]
    for p in random_vectors(5, 2000):
        a = {m: np.array(p_adjust(p, m)) for m in P_ADJUST_METHODS}
        assert np.array_equal(a["none"], p)
        assert np.array_equal(a["fdr"], a["BH"])
        idx = np.argsort(p, kind="stable")
        for m, v in a.items():
            assert np.all(v >= p), m
            assert np.all(v <= 1.0), m
            assert np.all(np.diff(v[idx]) >= 0), m  # order preserving
            for value in np.unique(p):
                tied = v[p == value]
                assert np.all(tied == tied[0]), m
        if p.size == 1:
            for m, v in a.items():
                assert v[0] == min(1.0, p[0]), m
        if p.size == 2:
            # Hommel's procedure reduces to Hochberg's for two hypotheses.
            assert np.array_equal(a["hommel"], a["hochberg"])
        for hi, lo in chain:
            # The ordering holds to rounding (Hommel may sit 1 ulp above Hochberg,
            # as in R's own output).
            gap = a[lo] - a[hi]
            assert np.all(gap <= 4 * np.spacing(np.maximum(a[hi], 1e-300))), (hi, lo)
        perm = np.random.Generator(np.random.PCG64(p.size)).permutation(p.size)
        for m in P_ADJUST_METHODS:
            assert np.array_equal(np.array(p_adjust(p[perm], m)), a[m][perm]), m


# -------------------------------------------------------------- operation order


def test_bh_operation_orders_differ_by_at_most_two_ulp() -> None:
    forms: dict[str, Callable[[float, float, float], float]] = {
        "(n*p)/i": lambda n, p, i: (n * p) / i,
        "n*(p/i)": lambda n, p, i: n * (p / i),
        "p/(i/n)": lambda n, p, i: p / (i / n),
    }
    worst = 0.0
    for p in random_vectors(7, 3000):
        n = float(p.size)
        for idx, v in enumerate(np.sort(p)):
            i = float(idx + 1)
            base = (n / i) * float(v)  # the ordering p_adjust uses
            for form in forms.values():
                worst = max(worst, ulps(form(n, float(v), i), base))
    assert worst <= 2.0
    # (n / n) * p is exactly p, so the largest p-value is never adjusted below itself.
    assert p_adjust([0.3, 0.7], "BH")[1] == 0.7


# ----------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("p", "method"),
    [
        ([0.1, math.nan], "holm"),
        ([0.1, 1.5], "holm"),
        ([-0.1], "BH"),
        ([0.1, math.inf], "BY"),
        ([[0.1, 0.2]], "holm"),
        ([0.1], "bonf"),
        ([0.1], "bh"),
    ],
)
def test_errors(p: object, method: object) -> None:
    with pytest.raises(ValueError, match=r"p-value|p_adjust_method"):
        p_adjust(p, method)  # type: ignore[arg-type]


def test_empty_input() -> None:
    assert p_adjust([], "hommel") == ()


# ----------------------------------------------------------------- harmonic sum


def _round_to_64_bits(x: Fraction) -> Fraction:
    """Round a positive rational to a 64-bit significand, ties to even."""
    exponent = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** exponent > x:
        exponent -= 1
    scale = Fraction(2) ** (63 - exponent)  # x * scale in [2^63, 2^64)
    scaled = x * scale
    whole = scaled.numerator // scaled.denominator
    rest = scaled - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2 == 1):
        whole += 1
    return Fraction(whole) / scale


def test_by_harmonic_sum_matches_a_64_bit_accumulator() -> None:
    # R's sum() accumulates in long double where the platform has one (?sum): the
    # x87 80-bit format on x86_64 Linux, where the oracle runs, has a 64-bit
    # significand. math.fsum (correctly rounded) gives the same double for every
    # n < 1008; a plain double loop differs from n = 4 on.
    acc = Fraction(0)
    terms: list[float] = []
    naive = 0.0
    fsum_mismatch: list[int] = []
    naive_mismatch: list[int] = []
    for n in range(1, 1200):
        terms.append(1.0 / n)
        acc = _round_to_64_bits(acc + Fraction(terms[-1]))
        long_double = float(acc)
        naive += terms[-1]
        if math.fsum(terms) != long_double:
            fsum_mismatch.append(n)
            assert ulps(math.fsum(terms), long_double) == 1.0
        if naive != long_double:
            naive_mismatch.append(n)
    assert fsum_mismatch == [1008, 1138]
    assert naive_mismatch[0] == 4
