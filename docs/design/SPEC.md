# inferplot specification

Status: draft 4, 2026-11-08 (Phase 0 oracle findings and both Phase 0 reviews; ggbetweenstats variants read from the statsExpressions 2.2.1 source, MIT, and confirmed by fixtures where noted). Parity target: **ggstatsplot 1.1.4** (CRAN, 2026-10-22).
When this document and the R oracle disagree, the oracle wins: fix this document and
log the change in `DECISIONS.md`. Rows marked *(verify)* name the R function but not yet
the exact variant (continuity correction, exact vs asymptotic p, CI method, sidedness);
read the variant from the oracle fixture before implementing and replace the marker.

## 1. Scope

**v0.1:** `betweenstats`, `withinstats`, `histostats`, `scatterstats`, `barstats`;
types parametric, nonparametric and robust everywhere ggstatsplot has them; Bayes
factors for t-tests and correlations.

**v0.2:** Bayes factors for ANOVA and contingency tables, posterior estimates in
Bayesian captions, `dotplotstats`, `piestats`, `corrmat`, `coefstats`, `grouping_var`
(ggstatsplot's `grouped_*` functions), replicate-aware SuperPlots.

**Out of scope:** curve fitting with uncertainty bands, residual plots, interactive output.

## 2. Public API

```python
import inferplot as ip

fig = ip.betweenstats(df, "group", "score")            # analyze + render
res = ip.analyze.betweenstats(df, "group", "score")    # StatsResult only, no matplotlib
fig = ip.render.betweenstats(res)                      # draw an existing result
fig.result.text.apa; fig.result.to_json(); fig.check()
fig.save("score.svg")
```

`inferplot.analyze` must not import matplotlib. `inferplot.render.<name>` accepts only a
`StatsResult` produced by the matching analysis.

### Common keyword arguments (all keyword-only)

| Argument | Default | Notes |
| --- | --- | --- |
| `type` | `"parametric"` | also `"nonparametric"`, `"robust"`, `"bayes"`; aliases `"p"`, `"np"`, `"r"`, `"bf"` |
| `conf_level` | `0.95` | |
| `digits` | `2` | subtitle formatting, as ggstatsplot |
| `tr` | `0.2` | trimming for robust methods |
| `bf_prior` | `0.707` | numeric, exactly as ggstatsplot (not sqrt(2)/2) |
| `bf_message` | `None` | Bayes factor caption. ggstatsplot draws one only when `results_subtitle` is true and either the type is parametric (betweenstats, withinstats, histostats, scatterstats) or, in barstats, the type is not bayes, `paired=False` and `y` is None or has more than one level (ggstatsplot: `y` does not have exactly one level); otherwise it draws none and says nothing. Within that condition `None` draws the caption when v0.1 computes the Bayes factor (two groups or conditions, one sample, correlation) and otherwise adds a `warnings` entry saying it arrives in v0.2 (ANOVA with three or more groups or conditions, contingency tables, one-way goodness of fit); outside it, `None` draws no caption and adds no warning. `True` raises `NotImplementedError` for the v0.2 designs and `ValueError` outside the condition; `False` never draws it. The caption's Bayes factor is two-sided whatever `alternative` says, as ggstatsplot's; with a one-sided `alternative`, `warnings` says so. See §5 item 5 |
| `results_subtitle` | `True` | |
| `alternative` | `"two-sided"` | `"less"`, `"greater"` only for two-group, paired and one-sample tests with type parametric or nonparametric; otherwise `ValueError` (ggstatsplot silently ignores it for robust and bayes types, for correlations, which have no `alternative` in ggscatterstats or `corr_test`, for the Bayes caption and for pairwise comparisons) |
| `seed` | `0` | every random step; recorded in provenance |
| `xlab`, `ylab`, `title` | `None` | user text |
| `subtitle` | `None` | used only when `results_subtitle=False`, as ggstatsplot |
| `caption` | `None` | user text drawn below the generated caption lines |
| `palette` | `None` | list of colors or a matplotlib colormap name; default Okabe-Ito |
| `ax` | `None` | draw into an existing Axes; subtitle and caption go inside that Axes; not allowed with marginal panels |

### Function-specific arguments

| Function | Positional | Specific keywords (defaults) |
| --- | --- | --- |
| `betweenstats` | `data, x, y` | `pairwise_display="significant"`, `p_adjust_method="holm"`, `pairwise_alpha=0.05`, `centrality_plotting=True`, `layers=("violin", "box", "points")`, `order=None`, `point_args`, `boxplot_args`, `violin_args` (dicts applied with `Artist.set`, so unknown properties raise) |
| `withinstats` | `data, x, y` | as `betweenstats`, plus **required** `subject_id`, and `point_path=True`, `centrality_path=True` |
| `histostats` | `data, x` | `test_value=0.0`, `binwidth=None` (ggstatsplot default rule), `centrality_plotting=True` |
| `scatterstats` | `data, x, y` | `marginal=True`, `smooth=True` (OLS line with `conf_level` band), `label_var=None` |
| `barstats` | `data, x, y=None` | `counts=None`, `paired=False`, `ratio=None`, `label="percentage"` (`"counts"`, `"both"`), `proportion_test=True`, `p_adjust_method="holm"`, `digits_perc=0` |

`p_adjust_method` accepts R's names: `holm`, `hochberg`, `hommel`, `bonferroni`, `BH`,
`BY`, `fdr`, `none`. `pairwise_display`: `significant`, `non-significant`, `all`, `none`
(aliases `s`, `ns`).

### API errors (exact behavior)

Unknown keyword → `TypeError` naming the closest valid keyword (difflib, cutoff 0.6).
These names get a targeted message instead:

| Keyword (dotted or snake_case) | Message must say |
| --- | --- |
| `var.equal` | removed in ggstatsplot 1.0.0; Welch's test is always used |
| `nboot` | removed in 1.0.0; inferplot uses 2,000 seeded resamples; pass `seed` |
| `effsize.type` | removed in 1.0.0; Hedges' g and partial omega squared are always used |
| `k` | renamed `digits` in ggstatsplot 0.12.3 |
| `package` | removed in 1.0.0; pass `palette` |
| `plot.type` | removed in 0.12.0; use `layers` |
| `pairwise.comparisons` | removed in 0.12.0; use `pairwise_display="none"` |
| `output` | removed in 0.10.0; use `fig.result` |
| `messages` | inferplot never prints |
| `sampling.plan`, `fixed.margin`, `prior.concentration` | removed from ggbarstats in 1.0.0 |
| `grouping.var`, `grouping_var` | arrives in v0.2 (`NotImplementedError`) |
| `hue` | no second grouping in this function; `grouping_var` arrives in v0.2 |
| `ci`, `errorbar` | intervals are always `conf_level` confidence intervals |
| `subject`, `id`, `subject.id` | use `subject_id` |

Passing a numpy array, list or dict as `data` → `TypeError` saying to build a pandas
or polars DataFrame in long format. `withinstats` without `subject_id` → `TypeError`
explaining that pairing by row order is not allowed.

## 3. Data handling

- Accept any eager frame `narwhals.from_native(data, eager_only=True)` accepts (pandas,
  polars, pyarrow). Convert used columns to float64/str NumPy arrays at one boundary
  (`_data.py`); nothing downstream sees a dataframe.
- Missing column → `KeyError` listing available columns and the closest match.
- `y` must be numeric (bool rejected). `x` may be string, categorical or integer.
- Rows with missing values in used columns are dropped and counted (`n_dropped`).
  Infinite values raise `ValueError`.
- Level order: the frame's categorical order if declared, else sorted unique values;
  `order=` must be a permutation of the present levels. Empty categorical levels are
  dropped before analysis and listed in `warnings`; they must never suppress results
  (ggstatsplot 1.1.4 fixed a bug where they did).
- `withinstats`: pivot `subject_id × x`. A repeated subject within a level raises,
  naming the first duplicates. Subjects missing any level are removed from statistics
  **and** the plot and counted (ggstatsplot 1.1.3 behavior).
- More than 5,000 points in one panel: draw a density layer plus a seeded subsample of
  2,000 points; statistics always use all data; record it in `warnings`.

## 4. Statistical methods

R reference = what the oracle runs (via ggstatsplot 1.1.4). Implement from the method
references below; never translate R source. Bayes rows marked v0.2 raise
`NotImplementedError` in v0.1.

**Method references** (compiled from memory: confirm each citation before quoting it in
a docstring):

| Method | Reference |
| --- | --- |
| Welch t; Welch ANOVA | Welch 1947, *Biometrika*; Welch 1951, *Biometrika* |
| Games-Howell | Games & Howell 1976, *Journal of Educational Statistics* |
| Holm and other adjustments | Holm 1979; Hochberg 1988; Hommel 1988; Benjamini & Hochberg 1995; Benjamini & Yekutieli 2001 |
| Hedges' g and its CI | Hedges 1981; Hedges & Olkin 1985; unpooled standardizer: Delacre, Lakens, Ley, Liu & Leys 2021; noncentral CIs: Steiger & Fouladi 1997, Smithson 2003 |
| Omega squared and its CI | Olejnik & Algina 2003; Steiger 2004 |
| Mann-Whitney, Wilcoxon | Mann & Whitney 1947; Wilcoxon 1945 |
| Rank-biserial r | Kerby 2014; Glass 1965 |
| Kruskal-Wallis, rank epsilon squared | Kruskal & Wallis 1952; Tomczak & Tomczak 2014 |
| Dunn | Dunn 1964 |
| Friedman, Kendall's W | Friedman 1937; Kendall & Babington Smith 1939 |
| Durbin-Conover | Conover 1999, *Practical Nonparametric Statistics* |
| Yuen; trimmed means, t1way, lincon, yuend, rmanova, rmmcp, trimcibt, Winsorized correlation | Yuen 1974; Wilcox 2017, *Introduction to Robust Estimation and Hypothesis Testing*, 4th ed. |
| AKP effect size | Algina, Keselman & Penfield 2005, *Psychological Methods* |
| Explanatory measure of effect size | Wilcox & Tian 2011, *Journal of Applied Statistics* |
| Repeated measures, Greenhouse-Geisser | Greenhouse & Geisser 1959 |
| JZS Bayes factor, t-tests | Rouder, Speckman, Sun, Morey & Iverson 2009 |
| Bayes factor, ANOVA (v0.2) | Rouder, Morey, Speckman & Province 2012 |
| Bayes factor, correlation | Ly, Verhagen & Wagenmakers 2016 |
| Bayes factor, contingency (v0.2) | Gunel & Dickey 1974 |
| Pearson r, Fisher z | Pearson 1895; Fisher 1915 |
| Spearman rho | Spearman 1904 |
| Chi-square, Cramér's V, Pearson's C | Pearson 1900; Cramér 1946 |
| McNemar, Cohen's g | McNemar 1947; Cohen 1988 |

### betweenstats, two groups

| type | Test (R reference) | Effect size + CI (R reference) | Centrality |
| --- | --- | --- | --- |
| parametric | Welch's t (`stats::t.test`, var.equal = FALSE); p follows `alternative` | Hedges' g (`effectsize::hedges_g(pooled_sd = FALSE)`): d = (mean1 - mean2) / sqrt((var1 + var2) / 2), the unpooled standardizer, not the pooled SD and not the average of the two SDs; g = d x J(df_W), with J Hedges' exact gamma-ratio correction at the Welch df df_W. CI: the noncentral-t interval for the noncentrality parameter at the Welch t and df_W, each bound times (Welch SE / sqrt((var1 + var2) / 2)) x J(df_W). The CI follows `alternative`: "greater" gives [lower bound, Inf] and "less" [-Inf, upper bound], one-sided at `conf_level`; the subtitle prints "Inf" and "-Inf". Reproduced in every two-group fixture, balanced and unbalanced (mtcars `am`, `edge_k2`): g to 1e-10, CI bounds to 7.5e-7 | mean |
| nonparametric | Mann-Whitney U (`stats::wilcox.test(exact = FALSE)`): always the normal approximation with continuity correction, never exact, also without ties and with fewer than 50 values per group (`edge_k2`: ggstatsplot reports p = 0.0025; the exact p would be 0.0013); the fixture's method string omits "with continuity correction"; p follows `alternative` | rank-biserial r = 2U / (n1 n2) - 1 (`effectsize::rank_biserial`), U the Mann-Whitney U of group 1; CI tanh(atanh(r) -/+ z x sqrt((n1 + n2 + 1) / (3 n1 n2))), ties ignored; follows `alternative` ("greater" gives [lower bound, 1], "less" [-1, upper bound]); reproduced to 1e-10 | median |
| robust | Yuen, 20% trim (`WRS2::yuen`): t is reported unsigned, \|t\|, even when group 1's trimmed mean is the smaller (mtcars `am`, `edge_k2`), while the AKP effect size keeps its sign; p presumably 1 - pt like lincon's *(verify: every two-group robust fixture has p of 0.0047 or more, where the two computations agree)* (§10, `wrs2_p`) | AKP robust standardized difference, `WRS2::akp.effect(EQVAR = FALSE)`: trimmed-mean difference scaled by group 1's winsorized SD, not the pooled SD (oracle: 0.6827 vs pooled 0.6393 on ToothGrowth); bootstrap CI (ggstatsplot: 100 resamples) | trimmed mean |
| bayes | JZS Bayes factor, r = 0.707 (`BayesFactor::ttestBF(formula, rscale = bf.prior)`) on Student's (pooled) t, two-sided whatever `alternative` says. `tools/reference`'s one-dimensional quadrature reproduces all 35 distinct two-sample values in the fixtures (subtitles, parametric captions, pairs) to 1.5e-9; Welch's t differs on every unbalanced pair, by 1.5e-4 or more | v0.2: posterior median of the mean difference with a 95% equal-tailed interval (`conf.method` "ETI" in the fixtures; 4,000 posterior draws per `parameters::model_parameters()`'s help page, to confirm) | MAP (`bayestestR::map_estimate`: kernel density, Sheather-Jones bandwidth, 2^10 grid points) |

### betweenstats, three or more groups

| type | Omnibus | Effect size + CI | Pairwise, then `p_adjust_method` | Centrality |
| --- | --- | --- | --- | --- |
| parametric | Welch ANOVA (`stats::oneway.test`) | partial omega squared (`effectsize::omega_squared`); oracle shows CI [0, 1], so one-sided | Games-Howell (`PMCMRplus::gamesHowellTest`) **then Holm on top**; q is signed, mean(group2) - mean(group1). p comes from R's `ptukey`, whose upper tail has absolute error up to about 5e-10: iris setosa-versicolor reads p = 1.41e-9 where the true value is about 2.2e-14, and ggstatsplot prints that number. SciPy's `studentized_range` is no ground truth out there either: 1.9e-14 for this pair, 14% low Parity in the far tail means reproducing `ptukey` (STATUS) | mean |
| nonparametric | Kruskal-Wallis (`stats::kruskal.test`) | rank epsilon squared (`effectsize::rank_epsilon_squared`), one-sided percentile bootstrap CI [lower, 1] (ggstatsplot: 100 iterations) | Dunn (`PMCMRplus::kwAllPairsDunnTest`): z = abs(mean-rank difference) / tie-corrected SE, reported unsigned; two-sided normal p (reproduced to 2.4e-15) | median |
| robust | trimmed-means ANOVA (`WRS2::t1way`); p computed as 1 - pf (§10, `wrs2_p`) | explanatory effect size (`WRS2::t1way`), bootstrap CI (ggstatsplot: 100 resamples, `alpha = 1 - conf.level` computed in floating point, 0.050000000000000044: with 100 resamples only that value, not 0.05, reproduces ggstatsplot's bounds, so the CI's order statistics depend on alpha / 2 x resamples rounding past 2.5); with unequal group sizes the point estimate is itself random (oracle replicate SD 0.0048 on mtcars), so compare it within Monte Carlo error | `WRS2::lincon`: estimate = trimmed mean(group1) - trimmed mean(group2); p = per-pair Yuen p (1 - pt), then `p_adjust_method` (lincon's own Hochberg is not used); CI = estimate -/+ c x Yuen SE. c comes from a two-decimal table of the Studentized maximum modulus at (alpha, C = number of pairs, not groups: `edge_k5` has 10, `bugs_long` 6), whose rows are every df from 2 to 12, then 14, 16, 18, 20, 24, 30, 40, 60 and a last row at df 200 holding the infinite-df (normal) values; c is interpolated linearly in 1/df between the rows on either side of the pair's Yuen df (fixtures: df 5.2 to 110, rows up to 60/200; reproduced to 1e-14). The entries are the published table's, not recomputed quantiles: for C = 3 at df 5 the table has 3.39 where the quantile, 3.3992, rounds to 3.40 *(verify: take the table from its published source; no fixture has df above 200 or alpha other than 0.05)* | trimmed mean |
| bayes | v0.2: `BayesFactor::anovaBF` (method and precision to measure when implemented) | v0.2: Bayesian R squared, posterior median with 95% HDI (`conf.method` "HDI" in the fixtures) | v0.2: `ttestBF` per pair as for two groups, unadjusted, always with prior 0.707 and 95% intervals whatever `bf_prior` and `conf_level` say (every pair's Bayes factor is reproduced by `tools/reference`) | MAP |

### withinstats, two conditions

| type | Test | Effect size + CI | Centrality |
| --- | --- | --- | --- |
| parametric | paired t (`stats::t.test`, paired) | Hedges' g, paired *(verify d_z vs d_av)*; the CI follows `alternative` as for two groups (ggwithinstats passes it to the same `two_sample_test`) | mean |
| nonparametric | Wilcoxon signed-rank (`stats::wilcox.test`, paired) *(verify exact/correct)* | rank-biserial r *(verify CI)*; follows `alternative` as for two groups | median |
| robust | Yuen for dependent samples (`WRS2::yuend`) | robust effect size for dependent groups, `WRS2::dep.effect` (statsExpressions 2.2.1 calls it for paired robust tests, not `wmcpAKP`), bootstrap CI (ggstatsplot: 100 resamples); 1.1.3 fixed this value | trimmed mean |
| bayes | paired JZS Bayes factor (`ttestBF`) | posterior difference: v0.2 | MAP |

### withinstats, three or more conditions

| type | Omnibus | Effect size + CI | Pairwise, then `p_adjust_method` | Centrality |
| --- | --- | --- | --- | --- |
| parametric | one-way repeated-measures ANOVA (`afex::aov_ez`) *(verify Greenhouse-Geisser default)* | partial omega squared | paired t | mean |
| nonparametric | Friedman (`stats::friedman.test`) | Kendall's W (`effectsize::kendalls_w`), bootstrap CI | Durbin-Conover (`PMCMRplus::durbinAllPairsTest`) | median |
| robust | trimmed-means RM ANOVA (`WRS2::rmanova`) | AKP average (`WRS2::wmcpAKP`) | `WRS2::rmmcp` | trimmed mean |
| bayes | v0.2 | v0.2 | v0.2 | MAP |

### histostats (one sample against `test_value`)

| type | Test | Effect size + CI | Centrality |
| --- | --- | --- | --- |
| parametric | one-sample t | Hedges' g (1.1.4: one-sided CI when `alternative` is one-sided) | mean |
| nonparametric | one-sample Wilcoxon signed-rank | rank-biserial r (1.1.4: one-sided CI when `alternative` is one-sided, statsExpressions 2.2.1) | median |
| robust | bootstrap-t trimmed-mean test (`WRS2::trimcibt`); statistic exact, p and CI bootstrap | trimmed mean with CI | trimmed mean |
| bayes | one-sample JZS Bayes factor | posterior: v0.2 | MAP |

### scatterstats

| type | Test | Estimate + CI |
| --- | --- | --- |
| parametric | Pearson r, t test (`correlation::correlation`) | r, Fisher z CI |
| nonparametric | Spearman rho | rho *(verify CI method)* |
| robust | 20% Winsorized Pearson | r_w *(verify df and CI)* |
| bayes | Bayesian Pearson Bayes factor (`correlation`, BayesFactor) *(verify prior scale)* | posterior rho: v0.2 |

Plot: points, OLS line with `conf_level` band, marginal histograms.

### barstats (`x` = rows / fill, `y` = columns / bars)

| Design | type | Test | Effect size + CI |
| --- | --- | --- | --- |
| two-way | parametric = nonparametric (documented equivalence) | Pearson chi-square of independence (`stats::chisq.test`) *(verify continuity correction)* | Cramér's V (`effectsize::cramers_v`) *(verify adjusted, sidedness)* |
| two-way, `paired=True` | parametric = nonparametric | McNemar (`stats::mcnemar.test`) *(verify correction)* | Cohen's g |
| one-way (`y=None`) | parametric = nonparametric | chi-square goodness of fit against `ratio` | Pearson's C |
| any | bayes | v0.2: `contingencyTableBF` (independent multinomial, rows fixed, a = 1) for two-way tables; one-way goodness of fit is a Monte Carlo estimate from 100,000 Dirichlet draws (statsExpressions 2.2.1), random, so it needs its own tolerance | v0.2 |

Per-bar proportion tests (goodness of fit of `x` within each `y` level) are adjusted with
`p_adjust_method` (added in ggstatsplot 1.0.0). Pairwise Fisher exact tests between `x`
levels appear in `extract_stats` since 1.0.0 *(verify)*; include them in the result.
`type="robust"` has no distinct method: raise `ValueError` (ggstatsplot silently treats
it as parametric).

### Defaults that look odd but are parity requirements

- Games-Howell p-values get Holm adjustment on top (double adjustment). Replicate it.
  Their far upper tail is R's `ptukey` value, which can be orders of magnitude above
  the true one (1.41e-9 for about 2.2e-14 on iris), and the bracket label prints it.
- `bf_prior=0.707` numeric. sqrt(2)/2 changes Bayes factors in the fifth significant figure.
- Mann-Whitney p is always the corrected normal approximation, also where R's own
  `wilcox.test` default would compute the exact p.
- Yuen's t and Dunn's z are reported unsigned; Games-Howell's q and every effect size
  keep their sign.
- WRS2's p-values (t1way and lincon; presumably Yuen) are 1 - cdf, rounded to the
  spacing of doubles just below 1; lincon's CI uses a published two-decimal table with at least one entry
  that is not the rounded quantile.
- One-sided `alternative` makes the two-group effect-size CIs one-sided; the Bayes
  caption stays two-sided.
- Pairwise display defaults to significant pairs only: pairs whose adjusted p is strictly
  below `pairwise_alpha` get a bracket; with `type="bayes"` every pair does. Brackets
  exist only for three or more groups.
- Centrality labels show the mean, median, 20% trimmed mean or MAP to `digits`; x tick
  labels read `<level>\n(n = <n>)`. ggstatsplot also bootstraps a 95% CI for each
  centrality value (100 replicates) but never displays it; inferplot does not compute it.

## 5. Deliberate differences from ggstatsplot

Statistical and behavioral differences are exactly these:

1. `subject_id` is required in `withinstats`; pairing never comes from row order.
2. Bootstrap CIs use 2,000 seeded resamples (ggstatsplot uses 100), so bootstrap-derived
   numbers in the subtitle differ from ggstatsplot's by Monte Carlo noise. §10 says how
   they are tested.
3. When only significant pairs are drawn, the caption states how many pairs were tested.
4. Nothing is silently dropped or aliased: unknown arguments and `type` values without a
   distinct method raise.
5. v0.1 only: where ggstatsplot's default adds a Bayes factor that inferplot computes only
   from v0.2 (the Bayes ANOVA caption after a parametric test with three or more groups
   or conditions; unpaired contingency tables and one-way goodness of fit with a type
   other than bayes), no Bayes
   caption is drawn by default and `warnings` says so; `bf_message=True` raises
   `NotImplementedError` for those designs (§2). Posterior estimates in Bayes captions
   also arrive in v0.2 (§1).

Visual styling is not a parity target: same semantic layers (points, box, violin,
centrality label, brackets, subtitle content, caption content), our own look. Default
palette Okabe-Ito; more than 8 levels uses `tab20` and adds a warning.

## 6. Result contract

`StatsFigure`: `figure`, `axes` (dict: `"main"`, plus `"xside"`/`"yside"` for marginals),
`result`, `check()`, `save(path, *, dpi=300, transparent=False)` (format from the
extension: `.svg`, `.pdf` or `.png`; anything else raises), `_repr_png_` for notebooks.
No `show()`.

`StatsResult` (frozen dataclass, plain Python types only, JSON-serializable):

| Field | Content |
| --- | --- |
| `analysis`, `design`, `type` | e.g. `"betweenstats"`, `"between"`, `"parametric"` (canonical) |
| `x`, `y`, `subject_id`, `levels` | column names and analyzed level order |
| `descriptives` | per level: `n`, centrality name and value |
| `test` | name, R-style method string, statistic name and value, `df`, `df_error`, p-value, alternative |
| `effect_size` | name, estimate, `conf_level`, CI bounds, CI method, sidedness |
| `bayes` | `bf10`, `log_bf01`, prior name and scale; `None` when absent |
| `pairwise` | per pair: groups, statistic, p, adjusted p, adjust method, test name, significant, displayed |
| `n_obs`, `n_dropped`, `dropped_subjects` | counts |
| `warnings` | plain-English strings, no internal jargon |
| `provenance` | inferplot, numpy, scipy, matplotlib versions; seed; parity target `"ggstatsplot 1.1.4"`; method citations |
| `text` | `subtitle`, `caption`, `apa`, `alt_text` |

Methods: `to_dict()`, `to_json()`, `to_frame(backend="pandas" | "polars")` (one row per
test: the main test first, then each pair), `check()`, `__str__` (APA sentence plus
pairwise table).

`save()` sets `svg.fonttype="none"` and `pdf.fonttype=42` inside an `rc_context`, and
embeds `result.to_json()` in file metadata (PNG text chunk; SVG and PDF description
fields; confirm which keys each matplotlib backend accepts).

## 7. Text formats

- **Subtitle:** ggstatsplot's expression content in matplotlib mathtext. Numbers must
  equal the formatted strings in the oracle's `expression` output exactly (same
  rounding, same switch to scientific notation, same `digits`). Bootstrap-derived
  numbers are the exception described in §10.
- **Degrees of freedom** are numbers, not strings, in ggstatsplot's expression:
  statsExpressions rounds `df` and `df.error` to `digits` decimals where they are
  fractional (Welch t and F, Yuen, t1way; within-subjects parametric and robust) and to
  0 otherwise, and pastes them unquoted, so they print as R prints a number: without
  trailing zeros or a bare decimal point (iris Welch F `(2, 97.4)` for 97.40; mtcars
  `am` x `drat` Welch t `(27.2)`; mtcars `cyl` x `disp` t1way `(2, 10.4)`). Every other
  number is a quoted string and keeps its zeros ("1.00"). An infinite one-sided bound
  prints as "Inf" or "-Inf".
- **Caption:** Bayes factor line as ggstatsplot (log_e BF01, prior r) where §2's
  `bf_message` rule draws it; then the pairwise line, e.g. "Pairwise: Games-Howell, Holm
  adjusted; showing 1 of 3 pairs (significant only)."
- **APA sentence (APA 7):** "Welch's t-test: t(9.62) = -2.35, p = .041, Hedges' g =
  -1.07, 95% CI [-2.01, -0.08], n = 12." Two decimals for statistics; p to three
  decimals without leading zero, "p < .001" below .001; non-integer df to two decimals;
  ASCII hyphen-minus in all text fields.
- **Alt text:** two to four sentences, at most 400 characters: chart type and
  variables, groups with n, centrality values, omnibus result, count of significant
  pairs.

## 8. `check()`

`result.check()` returns the statistical issues; `fig.check()` returns those plus the
rendering issues. Each is a tuple of `Issue(code, severity, message, fix)`; empty means
no problems.

| Code | Level | Severity | Trigger |
| --- | --- | --- | --- |
| `small_n` | result | warning | any group with n < 5 |
| `heavy_ties` | result | warning | rank-based test with more than 25% tied values |
| `rows_dropped` | result | info | `n_dropped > 0` or subjects dropped |
| `pairs_hidden` | result | info | some pairs not displayed |
| `label_overlap` | figure | warning | any two text artists' rendered bounding boxes intersect |
| `text_clipped` | figure | warning | any text extends outside the figure |
| `points_subsampled` | figure | info | large-n subsampling used |

## 9. R oracle

- `oracle/generate.R` runs ggstatsplot 1.1.4 on `oracle/datasets/*.csv` and writes one
  JSON file per case: inputs, arguments, R and package versions (ggstatsplot,
  statsExpressions, effectsize, WRS2, BayesFactor, PMCMRplus, afex, correlation),
  `extract_stats()` tables, and the subtitle and caption expression strings. It handles
  ggbetweenstats. Each other function first needs its own bookkeeping (which rows it
  analyzes: complete subjects for withinstats, one sample for histostats), its own
  bootstrap replay, and its own rules for naming the hidden calls, for when a caption
  is expected and for checking `n_per_level`; until then the script stops on its cases
  instead of writing an incomplete fixture.
- For bootstrap-based CIs, the script first replays ggstatsplot's own bootstrap: the
  effect-size call is the first random draw inside `ggbetweenstats()`, so `set.seed()`
  with the case seed followed by the same call, with 100 resamples and the rows in
  statsExpressions' order (sorted by group), must reproduce ggstatsplot's estimate and
  both bounds exactly. That proves the call is the method and variant ggstatsplot ran;
  when several calls are plausible (`akp.effect`'s `EQVAR`) the one that reproduces it
  is kept. The script then runs that call with 20,000 resamples for a precise reference,
  and 50 more times with 2,000 resamples to measure the Monte Carlo SD of the estimate
  and of each bound. The bootstrapped designs are listed explicitly (AKP for two robust
  groups, t1way's explanatory effect size for three or more, rank epsilon squared for
  three or more nonparametric groups); a non-default `conf.level` or `tr` stops the
  script.
- ggstatsplot 1.1.4 computes the subtitle and the caption through an internal helper
  that turns an error into a missing subtitle or caption and mutes warnings and
  messages, and it mutes warnings around `centrality_description()` and
  `pairwise_comparisons()`. The generator replaces these with recorders that make the
  same calls (same arguments, same random-number stream) and logs what ggstatsplot hid
  in `suppressed`. It stops when ggstatsplot swallows an error the case does not
  declare, when a subtitle or Bayes caption ggstatsplot should draw is missing, or when
  `n_per_level` differs from ggstatsplot's own `n.obs`.
- `.github/workflows/oracle.yml` (manual dispatch, on the branch that changes the
  generator): `r-lib/actions/setup-r@v2` with R 4.5.3, CRAN pinned to a Posit Package
  Manager snapshot dated on or after 2026-10-23, and hard checks of R, ggstatsplot and
  every installed package version. It uploads the JSON files as artifact
  `oracle-fixtures`, and a separate job that runs no R code signs them with a
  build-provenance attestation (`actions/attest`). They are committed to
  `tests/fixtures/oracle/` on the same branch after review.
- Datasets: mtcars, iris, ToothGrowth, PlantGrowth, morley, sleep, ggstatsplot's
  `bugs_long` and `Titanic_full`, plus edge cases: n = 2 per group, ties, all-negative
  outcome, empty factor level, missing values, unbalanced groups, k = 2, 3 and 5, a
  subject missing one condition, heavy outliers. `oracle/export_datasets.R` writes the R
  datasets (UTF-8, LF, quoted strings, empty field for missing, up to 15 significant
  digits); every oracle run re-exports them and fails if a committed byte changed. The
  edge cases are hand-written `edge_*.csv` files; an empty factor level is declared in the
  case (`factor_levels`) because CSV cannot hold one.
- Both sides read CSVs the same way: only empty fields are missing, and R sorts strings
  in the C locale, so factor levels from text columns follow Python's `sorted()`.
- plotsalot's `oracle/` (MIT) is a usable starting point; credit it in `NOTICE`.
- Python tests load the same CSV from `oracle/datasets/`, translate the fixture's R
  arguments to inferplot keywords with one mapping table (`tests/oracle_args.py`: dots to
  underscores, R `TRUE`/`FALSE`/`NULL` to Python), run the analysis, and compare every
  number the fixture holds, using `tests/tolerances.py`. One parametrized test per
  function collects all its fixtures, so a new fixture is tested without new code.

**Fixture schema (version 2).** One JSON object per case, named `<name>.json`:

| Field | Content |
| --- | --- |
| `schema_version`, `name` | 2; `<R function>-<dataset label>-<type>` |
| `function`, `call`, `args`, `seed` | the R function, the deparsed call, its arguments (column names as strings), the `set.seed()` value used before it |
| `dataset` | `file`, `sha256` of the CSV R read, `rows`, declared `factor_levels` or null |
| `analysis_levels`, `n_per_level` | group order and counts after dropping incomplete rows and empty levels; the generator checks the counts against ggstatsplot's own `n.obs` |
| `error`, `r_warnings`, `r_messages` | conditions that reached the generator; `error` is null unless the case expects ggstatsplot to fail |
| `suppressed` | what ggstatsplot hid, one entry per hidden call in call order: `component` (`subtitle`, `caption`, `centrality_description`, `pairwise_comparisons`), `error` (null unless ggstatsplot swallowed one; errors of the last two are not swallowed and end up in `error`), and the muted `warnings` and `messages` |
| `expected_hidden_errors` | components whose error the case declares ggstatsplot swallows; empty for every current case |
| `subtitle`, `caption` | deparsed plotmath from `extract_subtitle()` and `extract_caption()` |
| `stats` | every `extract_stats()` table as an array of row objects; plotmath columns deparsed. In 1.1.4 `descriptive_data` is null |
| `plot_layers` | the ggplot layers in order: `name`, `geom`, and the layer's own data (the centrality labels: value, n, `expression`, `n.expression`), null for layers that draw the main data. Bracket layers (`GeomSignif`) also carry `params` (`stat`, `geom`, `aes`; `stat.comparisons` lists the drawn pairs, `stat.annotations` their labels) and `built`, the rows ggplot2 computes for drawing |
| `bootstrap_reference` | null, or: `call`; `ggstatsplot` (its estimate and bounds, 100 resamples, seed); `replay` (`matches` and each candidate's replayed values); the 20,000-resample `estimate`, `conf_low`, `conf_high` and `raw` output; `replicates` (mean and SD of the estimate and each bound over 50 runs of 2,000) |
| `versions` | R version, platform, OS, collation, repositories, package versions |

Numbers carry 15 significant digits; R's NA is null; NaN, Inf and -Inf are the strings
`"NaN"`, `"Inf"`, `"-Inf"`. `manifest.json` (schema version 2) holds the SHA-256 of
every fixture and of the generator files (`generator_files`: `oracle/generate.R`,
`oracle/export_datasets.R`, `.github/workflows/oracle.yml`), the GitHub Actions run that
produced them (`run`: repository, run id, attempt, commit), R's version and RNG kind,
and every installed R package with its version (`r_library`). On every test run the
loader (`tests/oracle_fixtures.py`) refuses fixtures whose own hash, whose dataset's
hash or whose generator files' hashes differ from the manifest. On every push and pull
request `.github/workflows/fixtures.yml` verifies the manifest's attestation (signed by
`oracle.yml` in this repository as that file stood at the commit the manifest names,
by the run it names), that this commit is in the branch's history, and that its
generator files equal the committed ones. Together these catch a hand edit, even with
a re-hashed manifest; fixtures from another run or an older generator; and a generator
changed without a new oracle run. Attestations do not expire, unlike artifacts (90
days). Because the oracle runs on the branch that changes the generator, that branch
must land with a merge commit or a fast-forward: a squash or rebase merge drops the
attested commit from main's history, and the check then fails until the oracle runs
again.

## 10. Tolerances

`tests/tolerances.py` maps each quantity kind to `(rtol, atol, reason)`. Starting values:

| Quantity | Tolerance | Reason |
| --- | --- | --- |
| closed-form statistics, df, t/F/chi-square/normal p-values | rtol 1e-10 | both sides exact to double precision |
| Games-Howell p (`games_howell_p`) | rtol 1e-6 | R ptukey vs SciPy studentized_range measured 2.8e-7 away from the far tail; in the tail ptukey has absolute error up to about 5e-10, so inferplot must reproduce ptukey there |
| R ptukey against SciPy's studentized range (`ptukey_vs_scipy`, reference cross-check only) | rtol 1e-6, atol 3e-9 | measured up to 2.5e-9 absolute after Holm (`edge_k5`, iris); SciPy's own far tail is off by order 1e-14 |
| p-values WRS2 computes as 1 - pt or 1 - pf (`wrs2_p`: t1way, lincon, presumably Yuen) | rtol 1e-10, atol 2^-53 x the factor applied to the one-sided tail afterwards (2 for two-sided, times the number of pairs after Holm) | the one-sided tail rounded to the spacing of doubles just below 1 (2^-53); rounding the tail the same way matches all fixtures but iris t1way, which lands one grid step away |
| noncentral-distribution CIs | start rtol 1e-6; set from measurement | R root-finding precision; Hedges' g bounds agree with a tight SciPy inversion to 7.5e-7 relative and up to 1.9e-8 absolute, so a bound near 0 will need an absolute part |
| JZS Bayes factors of t tests (`ttest_bayes_factor`) | rtol 1e-8 | measured on two-sample values only: `tools/reference` reproduces all 35 distinct `ttestBF` values in the fixtures to at most 1.5e-9, which is BayesFactor's own integration error; a prior of sqrt(2)/2 instead of 0.707 moves them by 3.7e-6 or more, Welch's t instead of Student's moves every unbalanced pair by 1.5e-4 or more. Measure paired and one-sample values when their fixtures arrive |
| other Bayes factors (correlation; v0.2 ANOVA and contingency) | own entry, measured when first implemented | not yet measured |
| bootstrap CIs and p-values | within 4 Monte Carlo SDs of the 20,000-resample R reference, never less than the closed-form tolerance | randomness; a quantity whose replicate SD is 0 does not depend on the resamples (AKP and rank epsilon squared estimates, an upper bound of 1) and is stored to 15 significant digits |
| MAP centrality | one grid step of the density grid | kernel density grid |
| subtitle numbers | exact string match | formatting parity; see below for bootstrap-derived numbers |

Never loosen an entry without a measured reason written in the registry.

**Subtitle numbers from a bootstrap.** Bootstrap CI bounds, and point estimates that are
themselves random (t1way's explanatory effect size with unequal group sizes), cannot
match ggstatsplot's strings: ggstatsplot computes them from 100 resamples, inferplot from
2,000. For these quantities parity is checked in two separate parts:

1. the value inferplot computes is within the bootstrap tolerance of the 20,000-resample
   R reference (`bootstrap_reference` in the fixture);
2. inferplot's formatter, applied to ggstatsplot's own value from `stats`, reproduces
   ggstatsplot's string exactly.

Every other subtitle number is compared as the exact string inferplot produces.

## 11. Other tests

- **Reference cross-checks:** where `tools/reference/stats_reference.py` covers a method,
  production code must agree with it too.
- **Simulation (marker `slow`):** under the null with unequal variances and unequal n,
  2,000 replications: Type I error at alpha 0.05 inside the binomial 99% band; 95% CI
  coverage likewise.
- **Properties (hypothesis):** row-order invariance; level relabeling invariance; pandas,
  polars and pyarrow inputs give identical results; standardized effect sizes invariant
  to location and positive scale.
- **Rendering:** structural SVG assertions (layers present, label text equals result
  text); a few pytest-mpl image baselines generated on Linux only.

## 12. Documentation for AI users

- README quick start with a pandas example, under 40 lines.
- `docs/api.md`: the complete public API in under 32 KB (about 8,000 tokens); a test
  enforces the size.
- `llms.txt` at the repository root in llmstxt.org format, pointing to `docs/api.md`
  and examples.
- `skills/inferplot/SKILL.md`: a Claude Code skill for users' projects. It says: use
  inferplot for comparison and association plots; declare the design explicitly; read
  `fig.result.text` and run `fig.check()` instead of inspecting the image; never try
  several `type` values to find significance.
- Every public function has a runnable docstring example, executed in CI.

## 13. Agent evaluation (Phase 5)

`evals/` holds about 30 tasks: a natural-language request, a dataset, and expected key
numbers from the oracle. Two arms: a Python environment without inferplot (matplotlib,
seaborn, scipy, statsmodels available) and one with inferplot plus the skill. Each task
runs `claude -p` in a fresh temporary directory. Score: runs without error; correct test
for the design; numbers within tolerance; `check()` clean (inferplot arm); turns and
tokens. Running it spends the human's Claude usage: ask before each run.

## 14. Release

Version 0.1.0 after Phase 5. Tag-triggered `release.yml` with PyPI trusted publishing,
TestPyPI first. The human creates the PyPI and TestPyPI projects and trusted-publisher
entries; the agent prepares everything else.
