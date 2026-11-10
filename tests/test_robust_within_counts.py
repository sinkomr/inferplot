"""The robust within type's checks say what is true there (review of phase 3,
findings 5 and 6): every interval is a 95% interval whatever conf_level is
(D66), and the average AKP's undefined and infinite counts are in resamples (the
most of any pair), while the warnings entry totals the pairs' values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import inferplot as ip
from inferplot.stats.within_robust import average_dependent_akp


def _long(m: NDArray[np.float64]) -> pd.DataFrame:
    n, k = m.shape
    return pd.DataFrame(
        [
            (f"s{i:02d}", f"c{j + 1}", float(m[i, j]))
            for i in range(n)
            for j in range(k)
        ],
        columns=["id", "cond", "y"],
    )


# Ratings 1 to 3 of 8 subjects at 4 conditions: many resamples have every
# winsorized difference of a pair equal (0/0 or infinite).
RATINGS = np.array(
    [[1, 1, 1, 1], [1, 1, 3, 2], [3, 2, 2, 3], [3, 2, 2, 2],
     [3, 1, 3, 3], [1, 2, 3, 2], [1, 3, 3, 3], [1, 1, 3, 1]],
    dtype=float,
)  # fmt: skip


def test_the_statistics_object_counts_each_pair() -> None:
    akp = average_dependent_akp([RATINGS[:, j] for j in range(4)])
    assert len(akp.pair_undefined) == len(akp.pair_infinite) == 6
    assert akp.undefined_resamples == max(akp.pair_undefined)
    assert akp.infinite_resamples == max(akp.pair_infinite)
    assert sum(akp.pair_undefined) == 679
    assert sum(akp.pair_infinite) == 92
    assert 0 < akp.infinite_resamples <= akp.resamples


def test_the_result_counts_resamples_and_the_warning_totals_values() -> None:
    result = ip.analyze.withinstats(
        _long(RATINGS), "cond", "y", subject_id="id", type="robust",
        pairwise_display="none",
    )  # fmt: skip
    es = result.effect_size
    assert es is not None
    akp = average_dependent_akp([RATINGS[:, j] for j in range(4)])
    assert (es.undefined_resamples, es.infinite_resamples) == (
        akp.undefined_resamples,
        akp.infinite_resamples,
    )
    undefined, infinite = (w for w in result.warnings if "bootstrap values" in w)
    assert undefined.startswith(
        "679 of the 12,000 bootstrap values (2,000 resamples of the subjects for "
        f"each of the 6 pairs) give an undefined (0/0) AKP (at most "
        f"{es.undefined_resamples:,} of one pair's 2,000)"
    )
    assert infinite.startswith(
        "92 of the 12,000 bootstrap values (2,000 resamples of the subjects for "
        f"each of the 6 pairs) give an infinite AKP (at most "
        f"{es.infinite_resamples:,} of one pair's 2,000)"
    )
    (issue,) = [c for c in result.check() if c.code == "infinite_effect_size"]
    assert issue.message.startswith(
        f"In up to {es.infinite_resamples:,} of the 2,000 bootstrap resamples a "
        "pair's AKP is infinite (the most of any pair), so the average AKP robust "
        "standardized difference's 95% CI"
    )


def test_pairwise_conf_level_says_every_interval_is_95_percent() -> None:
    rng = np.random.default_rng(2)
    m = rng.normal(50, 8, (15, 1)) + rng.normal(0, 2, (15, 3)) + np.arange(3.0)
    result = ip.analyze.withinstats(
        _long(m), "cond", "y", subject_id="id", type="robust", conf_level=0.9
    )
    (issue,) = [c for c in result.check() if c.code == "pairwise_conf_level"]
    assert issue.message == (
        "Every interval of this result is a 95% interval: the average AKP robust "
        "standardized difference's bootstrap interval and the pairwise intervals. "
        "conf_level=0.9 changes none of them; ggstatsplot does not pass it to WRS2 "
        "here."
    )
    assert (
        issue.fix == "Report the intervals as 95% intervals, or pass conf_level=0.95."
    )
    # Between groups the effect size follows conf_level and only the pairs do not.
    frame = _long(m)
    between = ip.analyze.betweenstats(frame, "cond", "y", type="robust", conf_level=0.9)
    (issue,) = [c for c in between.check() if c.code == "pairwise_conf_level"]
    assert issue.message.startswith("The pairwise confidence intervals are 95%")
