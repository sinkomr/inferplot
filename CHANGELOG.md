# Changelog

All notable changes to inferplot are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor
version (0.2, 0.3) may change the public API; each such change is listed here.

## [0.1.0] - Unreleased

The first release. inferplot draws a statistical plot, runs the test that fits the
design and writes the full result on the figure, after R's ggstatsplot 1.1.4, against
whose output every number is tested.

### Added

- Five functions, each in three forms: `ip.<name>` analyzes and draws,
  `ip.analyze.<name>` returns the result without importing matplotlib, and
  `ip.render.<name>` draws a result.
  - `betweenstats`: independent groups. Welch's t-test and ANOVA with Hedges' g and
    partial omega squared; Mann-Whitney and Kruskal-Wallis with the rank-biserial r
    and rank epsilon squared; Yuen's trimmed-means tests with robust effect sizes;
    JZS Bayes factors for two groups. Pairwise comparisons (Games-Howell, Dunn,
    Yuen) with a choice of multiplicity adjustment.
  - `withinstats`: the same subjects measured repeatedly, paired by a required
    `subject_id` column. Paired t-test and repeated-measures ANOVA
    (Greenhouse-Geisser corrected), Wilcoxon signed-rank and Friedman tests,
    trimmed-means tests, the paired JZS Bayes factor, with subject paths drawn.
  - `histostats`: one sample against a value. One-sample t-test, Wilcoxon
    signed-rank, bootstrap-t test of the trimmed mean, one-sample Bayes factor.
  - `scatterstats`: two measurements. Pearson, Spearman, winsorized and Bayesian
    correlations, with a least-squares line, its band and marginal histograms.
  - `barstats`: counts of categories. Pearson's chi-square test of independence
    with Cramér's V and Fisher's exact test between every two levels; McNemar's and
    Bowker's tests for paired categories; goodness of fit against given shares.
    Data may be one row per observation or a table with a `counts` column.
- A result object (`StatsResult`) with every number and text on the figure: the
  test, the effect size and its interval, Bayes factors, every pair, descriptives,
  the observations used, notes in `warnings`, and an APA-style sentence and alt
  text. `to_json()`, `from_json()` and `to_frame()` (pandas or polars).
- `fig.check()` and `result.check()`: 29 kinds of statistical and drawing problems
  (small samples, heavy ties, small expected counts, overlapping or clipped text,
  level names or colours that read alike, low contrast, missing glyphs and others),
  each with a fix.
- `fig.save()` to SVG, PDF or PNG with editable text and the result's JSON embedded
  in the file; saving twice gives identical bytes.
- pandas, polars and pyarrow data frames as input (optional extras); missing values
  are dropped and counted.
- Documentation: README, `docs/api.md` (the whole API in one file),
  `docs/intervals.md` (how far to trust each interval), `llms.txt`, and a Claude
  Code skill (`skills/inferplot/SKILL.md`).

### Differences from ggstatsplot 1.1.4

These are deliberate; each is recorded in `docs/design/DECISIONS.md`.

- `withinstats` requires `subject_id`; values are never paired by row order.
- Bootstrap intervals use 2,000 resamples drawn from `seed` (default 0), not 100, so
  their bounds differ from ggstatsplot's by random noise.
- Where R's numerical routines are inaccurate, inferplot reports the accurate value:
  Games-Howell p-values in the far tail (D31, D37), WRS2's p-values near zero (D43),
  Bayes factors where BayesFactor's quadrature misses (D50, D61, D81), the exact
  critical values for nine or more robust groups (D52), and intervals of Cramér's V
  and Pearson's C where R's root search stops short (D99). Exactly collinear data
  give r = 1 or -1 and an infinite t (D85).
- Fewer subjects than conditions: the repeated-measures ANOVA keeps its
  Greenhouse-Geisser correction where ggstatsplot prints the uncorrected F (D62).
- Counts that fit exactly print as 0 (chi-square, Cramér's V, Pearson's C) where R's
  arithmetic leaves rounding noise such as "V = 5.59e-09" (D118).
- Pairwise Fisher tests are exact and held to about two seconds per table; a pair
  too large to compute is named, not approximated, and still counts in the
  adjustment for multiple comparisons (D120). ggstatsplot computes every pair or
  fails.
- Nothing is ignored silently. Unknown arguments raise an error; settings that would
  change nothing raise or add a note; where ggstatsplot drops its subtitle without a
  word (for example a table with one category), inferplot raises an error that says
  why (D101).
- When only significant pairs are drawn, the caption says how many were tested.
- The figures have their own look (Okabe-Ito colours) with ggstatsplot's layers and
  texts.

### Known limits

- Not in this release (planned for 0.2): Bayes factors for three or more groups or
  conditions and for counts, posterior estimates, `grouping_var`, and counterparts
  of `ggpiestats`, `ggdotplotstats`, `ggcorrmat` and `ggcoefstats`. Asking for them
  raises `NotImplementedError` with a message that says what to do instead.
- Robust pairwise comparisons in `betweenstats` raise when a pair has fewer than 2
  degrees of freedom, as ggstatsplot does.
- Some intervals do not hold their nominal level in every setting (for example
  rank-based intervals and the robust average AKP); `docs/intervals.md` gives the
  simulated coverage of each.
- The signed-rank test ranks differences as R rounds them, so the same tenths entered
  two ways can give different ranks, as in R.

[0.1.0]: https://github.com/sinkomr/inferplot/releases/tag/v0.1.0
