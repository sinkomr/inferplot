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
  partial omega squared) have a one-sided interval from a lower bound to 1. Their
  bootstrap interval can never contain 0, even when there is no effect at all. Use
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

Everything not listed held its level in the simulations: Welch's t-test and ANOVA,
the paired and one-sample t-tests, Hedges' g and its interval, the Mann-Whitney,
Kruskal-Wallis, signed-rank and Friedman tests, Pearson's and Spearman's
correlations with their intervals (Spearman's slightly narrow at strong
correlations, 94%), the trimmed-means ANOVA, the dependent Yuen test, the
bootstrap-t test of the trimmed mean, the between-groups AKP interval, and the
robust pairwise intervals.
