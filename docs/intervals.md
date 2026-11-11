# How far to trust each interval

inferplot reproduces ggstatsplot's numbers, including the confidence intervals R
computes. A 95% interval should contain the true value in about 95 of 100 studies, and
a test at 0.05 should reject a true "no effect" in about 5 of 100. We checked both by
simulation for every test and interval inferplot reports (89 rates; the tests are in
`tests/test_simulation*.py` and run with `pytest -m slow`). Most hold. The ones below
do not, at the small to moderate sample sizes we simulated. None of this is a
difference between inferplot and R: each rate was reproduced with an independent
implementation of R's method.

| Interval or test | Target | Measured | In words |
| --- | --- | --- | --- |
| Partial omega squared, betweenstats (3+ groups, parametric), with an effect | 95% | 92% | The estimate runs high (0.20 where the truth was 0.14) |
| Partial omega squared, withinstats | 95% | 99.8 to 100% | Very wide; the lower bound is almost always near 0 |
| Rank-biserial r (Mann-Whitney, signed-rank, one-sample) | 95% | 88 to 93% | Too narrow; read it as roughly a 90% interval |
| Rank epsilon squared (Kruskal-Wallis) | 95% | 85%; never contains 0 | Runs high in small samples |
| Kendall's W (Friedman) | 95% | 68%; never contains 0 | Runs high in small samples |
| xi (robust, 3+ groups) | 95% | 92%; never contains 0 | |
| Average AKP (robust, 3+ conditions) | 95% | 99.6% | Wider than needed; errs on the safe side |
| Winsorized correlation at a strong correlation (0.75, 20 pairs) | 95% | 90% | Holds at weak correlations |
| Yuen's test, 10 against 25 values, the small group three times as spread out | 5% | 6% | Slightly too many false alarms |
| Games-Howell with Holm's adjustment on top, no differences at all | 5% | 1.6% | Conservative: a real difference is harder to detect |

What to do about it:

- **Effect sizes that cannot be negative** (rank epsilon squared, Kendall's W, xi,
  partial omega squared) never get an interval that contains a negative value.
  Rank epsilon squared, Kendall's W and partial omega squared have a one-sided
  interval from a lower bound to 1 (`effect_size.sidedness` is "greater"); xi has a
  two-sided percentile interval. The bootstrap intervals (rank epsilon squared,
  Kendall's W, xi) can never contain 0, even when there is no effect at all. Use
  the test's p-value to judge whether there is an effect, and read the interval as a
  range of plausible sizes, not as a test. Under no effect the estimate itself falls
  outside its own interval in 4 to 5% of data sets; `result.check()` reports that as
  `estimate_outside_interval`.
- **Rank-biserial r**: report it, and treat its interval as approximate. A bound that
  just excludes 0 is weaker evidence than it looks; the test's p-value holds its
  level.
- **Partial omega squared with three or more independent groups** is computed from
  Welch's ANOVA with a formula meant for the classical ANOVA. Treat it as a rough
  indication and describe group differences with the pairwise comparisons.
- **Pairwise comparisons** are computed and adjusted exactly as ggstatsplot does
  (Holm on top of Games-Howell for the parametric type), which is stricter than
  either adjustment alone.
- **The average AKP** (withinstats, type robust, three or more conditions) is the
  mean of the pairs' AKP values, and its interval is the mean of the pairs' own
  95% intervals, as in R. That interval covered the true value in 99.6% of
  simulated data sets: it is wider than a 95% interval needs to be. The average is
  signed (each pair is earlier level minus later level), so it changes with the
  order of the conditions; fix the order with `order=` before the analysis.
- **AKP with two conditions** (withinstats, type robust): listing the two
  conditions the other way round negates the estimate exactly, but the bootstrap
  interval is mirrored only approximately, because the resamples are drawn anew.
  Robust within-subjects intervals are always 95%, whatever `conf_level` says.

Everything not listed held its level in the simulations: Welch's t-test and ANOVA,
the paired and one-sample t-tests, Hedges' g and its interval, the Mann-Whitney,
Kruskal-Wallis, signed-rank and Friedman tests, Pearson's and Spearman's
correlations with their intervals (Spearman's slightly narrow at strong
correlations, 94%), the trimmed-means ANOVA, the dependent Yuen test, the
bootstrap-t test of the trimmed mean, the between-groups AKP interval, and the
robust pairwise intervals.

## Bootstrap intervals and `seed`

Bootstrap intervals use 2,000 resamples drawn from `seed` (default 0, recorded in
`effect_size.bootstrap_seed`); ggstatsplot uses 100. Another seed gives a slightly
different interval:

```python
import pandas as pd
import inferplot as ip

weights = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14,
           4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69,
           6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]
df = pd.DataFrame({"group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10,
                   "weight": weights})
for seed in (0, 1):
    result = ip.analyze.betweenstats(df, "group", "weight", type="nonparametric",
                                     seed=seed)
    print(seed, result.text.apa)
```

```text
0 Kruskal-Wallis test: chi2(2) = 7.99, p = .018, rank epsilon squared = .28, 95% CI [.10, 1.00], n = 30.
1 Kruskal-Wallis test: chi2(2) = 7.99, p = .018, rank epsilon squared = .28, 95% CI [.11, 1.00], n = 30.
```

Choose the seed before the analysis and report it; never try seeds until an
interval excludes 0. The bootstrap-t test of histostats (type robust) takes its
p-value from the resamples too: a p of 0 means that no resample was as extreme,
that is p below about 1 in 2,000, and `warnings` says so.
