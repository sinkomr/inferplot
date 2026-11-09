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
| `tr` | `0.2` | trimming for robust methods; another value with a type that trims nothing raises `ValueError` (ggstatsplot silently ignores it) |
| `bf_prior` | `0.707` | numeric, exactly as ggstatsplot (not sqrt(2)/2) |
| `bf_message` | `None` | Bayes factor caption. ggstatsplot draws one only when `results_subtitle` is true and either the type is parametric (betweenstats, withinstats, histostats, scatterstats) or, in barstats, the type is not bayes, `paired=False` and `y` is None or has more than one level (ggstatsplot: `y` does not have exactly one level); otherwise it draws none and says nothing. Within that condition `None` draws the caption when v0.1 computes the Bayes factor (two groups or conditions, one sample, correlation) and otherwise adds a `warnings` entry saying it arrives in v0.2 (ANOVA with three or more groups or conditions, contingency tables, one-way goodness of fit); outside it, `None` draws no caption and adds no warning. `True` raises `NotImplementedError` for the v0.2 designs and `ValueError` outside the condition; `False` never draws it. The caption's Bayes factor is two-sided whatever `alternative` says, as ggstatsplot's; with a one-sided `alternative`, `warnings` says so. See §5 item 5 |
| `results_subtitle` | `True` | |
| `alternative` | `"two-sided"` | `"less"`, `"greater"` only for two-group, paired and one-sample tests with type parametric or nonparametric; otherwise `ValueError` (ggstatsplot silently ignores it for robust and bayes types, for correlations, which have no `alternative` in ggscatterstats or `corr_test`, for the Bayes caption and for pairwise comparisons) |
| `seed` | `0` | every random step; recorded in provenance |
| `xlab`, `ylab`, `title` | `None` | user text (see below the table) |
| `subtitle` | `None` | used only when `results_subtitle=False`; passing it with `results_subtitle=True` raises `ValueError` (ggstatsplot silently ignores it there) |
| `caption` | `None` | user text drawn below the generated caption lines |
| `palette` | `None` | list of colors or a matplotlib colormap name; default Okabe-Ito |
| `ax` | `None` | draw into an existing Axes; subtitle and caption go inside that Axes; not allowed with marginal panels |

User texts (`title`, `subtitle`, `caption`, `xlab`, `ylab`) are matplotlib text and
follow its convention: text between two `$` signs is mathtext (`r"Dose ($\mu$g)"`),
and `\$` is a literal dollar sign. Each is laid out once before anything is drawn, so
an expression matplotlib cannot read raises `ValueError` at the call, also with `ax=`,
and leaves a given Axes unchanged (D40). Default labels from column names are never
read as mathtext. With `ax=`, inferplot's font applies only to the axis labels, tick
labels and the texts it draws; texts already in that Axes keep their font.

### Function-specific arguments

| Function | Positional | Specific keywords (defaults) |
| --- | --- | --- |
| `betweenstats` | `data, x, y` | `pairwise_display="significant"`, `p_adjust_method="holm"`, `pairwise_alpha=0.05`, `centrality_plotting=True`, `layers=("violin", "box", "points")`, `order=None`, `point_args`, `boxplot_args`, `violin_args` (dicts applied with `Artist.set`; each needs its layer, `"points"`, `"box"` or `"violin"`, in `layers`, or the call raises `ValueError` naming the layer to add; an empty dict is accepted. Before anything is drawn, each is applied to a scratch artist made as the drawing makes it, which is drawn and written as SVG and PDF (matplotlib checks some values only then); a property name the artist does not have raises `TypeError` with the closest property name, and any value matplotlib rejects, when setting it or only when drawing, raises `ValueError` naming the argument, with or without `ax=`, leaving a given Axes unchanged; `violin_args` does not style the density outlines drawn instead of violins) |
| `withinstats` | `data, x, y` | as `betweenstats`, plus **required** `subject_id`, and `point_path=True`, `centrality_path=True` |
| `histostats` | `data, x` | `test_value=0.0`, `binwidth=None` (ggstatsplot default rule), `centrality_plotting=True` |
| `scatterstats` | `data, x, y` | `marginal=True`, `smooth=True` (OLS line with `conf_level` band), `label_var=None` |
| `barstats` | `data, x, y=None` | `counts=None`, `paired=False`, `ratio=None`, `label="percentage"` (`"counts"`, `"both"`), `proportion_test=True`, `p_adjust_method="holm"`, `digits_perc=0` |

`p_adjust_method` accepts R's names: `holm`, `hochberg`, `hommel`, `bonferroni`, `BH`,
`BY`, `fdr`, `none`. `pairwise_display`: `significant`, `non-significant`, `all`, `none`
(aliases `s`, `ns`).

Arguments without effect on the given data warn (D39). With two groups there are no
pairwise comparisons, so a non-default `p_adjust_method`, `pairwise_display` or
`pairwise_alpha` adds one `warnings` entry each, and a non-default `bf_prior` does so
wherever no Bayes factor is computed (`bf_message=False`, `results_subtitle=False`, or
three or more groups in v0.1). Defaults and their aliases add nothing. `tr` with a
type that trims nothing and a one-sided `alternative` with three or more groups raise
instead, because they would change what the reported numbers mean.

### API errors (exact behavior)

Unknown keyword → `TypeError` naming the closest valid keyword (difflib, cutoff 0.6).
These names get a targeted message instead:

| Keyword (dotted or snake_case) | Message must say |
| --- | --- |
| `var.equal` | removed in ggstatsplot 1.0.0; Welch's test is always used |
| `nboot` | removed in 1.0.0; inferplot uses 2,000 seeded resamples; pass `seed` |
| `effsize.type` | removed in 1.0.0; Hedges' g and partial omega squared are always used |
| `k` | renamed `digits` in ggstatsplot 0.12.3 |
| `package` | removed in 1.0.0; pass `palette` (given to `ip.analyze.<name>`: colors are a drawing option of `ip.<name>` and `ip.render.<name>`) |
| `plot.type` | removed in 0.12.0; use `layers` (given to `ip.analyze.<name>`: pass `layers=` when drawing, to `ip.<name>` or `ip.render.<name>`) |
| `pairwise.comparisons` | removed in 0.12.0; use `pairwise_display="none"` |
| `output` | removed in 0.10.0; use `fig.result` |
| `messages` | inferplot never prints |
| `sampling.plan`, `fixed.margin`, `prior.concentration` | removed from ggbarstats in 1.0.0 |
| `grouping.var`, `grouping_var` | arrives in v0.2 (`NotImplementedError`) |
| `hue` | no second grouping in this function; `grouping_var` arrives in v0.2 |
| `ci`, `errorbar` | intervals are always `conf_level` confidence intervals |
| `subject`, `id`, `subject.id` | use `subject_id` |

Other ggbetweenstats 1.1.4 arguments with no inferplot equivalent (`centrality.type`,
`centrality.point.args`, `centrality.label.args`, `ggsignif.args`, `ggtheme`,
`ggplot.component`, dotted or snake_case) raise `TypeError` saying so and why, never a
did-you-mean. A keyword whose snake_case form (dots, hyphens and camelCase turned into
lowercase underscores) is a valid argument raises `TypeError` naming that spelling.
`subject`, `id`, `subject.id` point to `subject_id` in functions that have it; in
`betweenstats` they, and `subject_id`, say the groups are independent and point to
`withinstats(..., subject_id=...)`. When several keywords are wrong, the first in call
order is reported. Did-you-mean never suggests an argument the call already gives; a
snake_case spelling of one says it is already given. A drawing keyword given to
`ip.analyze.<name>` (`ax`, `palette`, `layers`, `title`, `subtitle`, `caption`, `xlab`,
`ylab`, `centrality_plotting`, `point_args`, `boxplot_args`, `violin_args`) says it is a
drawing option of `ip.<name>` and `ip.render.<name>`; an analysis keyword given to
`ip.render.<name>` (`data`, `x`, `y`, `type`, `pairwise_display`, `seed`, ...) says the
result already holds it and points to `ip.analyze.<name>`. More positional arguments
than `data, x[, y]` raise `TypeError`
naming the extra values and saying everything after them is keyword-only. A drawing
function (`ip.render.<name>`) given anything but a `StatsResult`, positionally, as
`result=`, or as data by keyword (`data=`, or `x=`/`y=` without a result), raises
`TypeError`
saying it draws the result of `ip.analyze.<name>(data, x, y)`, or to use
`ip.<name>(data, x, y, ...)` to analyze and draw in one call; this comes before the
keyword and positional checks.

Passing a numpy array, list or dict as `data` → `TypeError` saying to build a pandas
or polars DataFrame in long format. `withinstats` without `subject_id` → `TypeError`
explaining that pairing by row order is not allowed.

## 3. Data handling

- Accept any eager frame `narwhals.from_native(data, eager_only=True)` accepts (pandas,
  polars, pyarrow). Convert used columns to float64/str NumPy arrays at one boundary
  (`_data.py`); nothing downstream sees a dataframe.
- Missing column → `KeyError` listing available columns and the closest match.
- `y` must be numeric (boolean rejected). `x` may be text, categorical, integer, or
  float when every value is whole (boolean rejected) (D35).
- Rows with missing values in used columns are dropped and counted (`n_dropped`).
  An infinite `y` in a complete row raises `ValueError`. At least two levels need data.
- Level order: the declared order of a pandas `Categorical`, polars `Enum` or pyarrow
  dictionary column, else sorted unique values (a polars `Categorical` counts as text:
  its category order changed within polars 1.x); `order=` must be a permutation of the
  present levels. It is any ordered collection other than a string (list, tuple, NumPy
  array, pandas Index or Series, polars Series, pyarrow array, generator) of level
  names or of the values `x` holds: any integer (Python or NumPy), any finite float
  (8.0 names level "8") or a pyarrow scalar. A set or a mapping raises `TypeError`
  (no order, or an ambiguous one). Empty declared levels are dropped before analysis and listed in
  `warnings`; they must never suppress results (ggstatsplot 1.1.4 fixed a bug where
  they did).
- Parametric type (Welch's t and ANOVA, Games-Howell): every level needs n >= 2, two
  groups may not both be constant, and with three or more groups every level needs a
  variance above 0. The error names the levels (D35). A group that varies by less
  than double precision can resolve next to the largest |y| (a standard deviation
  below roughly 1e-154 times it; §4, "Scale and range") raises too, naming the
  levels; with two groups this happens only when the group with the largest |y| is
  constant. A group whose values span more than the largest double raises.
- `withinstats`: pivot `subject_id × x`. A repeated subject within a level raises,
  naming the first duplicates. Subjects missing any level are removed from statistics
  **and** the plot and counted (ggstatsplot 1.1.3 behavior).
- More than 5,000 points in one panel: draw a density layer plus a seeded subsample of
  2,000 points; statistics always use all data; record it in `warnings`.

## 4. Statistical methods

R reference = what the oracle runs (via ggstatsplot 1.1.4). Implement from the method
references below; never translate R source (D5's exceptions, ggstatsplot and
statsExpressions, cover the texts, not the statistics). Bayes rows marked v0.2 raise
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
| parametric | Welch's t (`stats::t.test`, var.equal = FALSE); p follows `alternative` | Hedges' g (`effectsize::hedges_g(pooled_sd = FALSE)`): d = (mean1 - mean2) / sqrt((var1 + var2) / 2), the unpooled standardizer, not the pooled SD and not the average of the two SDs; g = d x J(df_W), with J Hedges' exact gamma-ratio correction at the Welch df df_W. CI: the noncentral-t interval for the noncentrality parameter at the Welch t and df_W, each bound times (Welch SE / sqrt((var1 + var2) / 2)) x J(df_W). The CI follows `alternative`: "greater" gives [lower bound, Inf] and "less" [-Inf, upper bound], one-sided at `conf_level`; the subtitle prints "Inf" and "-Inf". Reproduced in every two-group fixture, balanced and unbalanced (mtcars `am`, `edge_k2`), at 90% and 95% and one-sided: g to 1e-10, CI bounds to 1.9e-8 absolute (`noncentral_ci`). inferplot computes the noncentral t itself: SciPy's `nctdtr` is off by up to 1.5e-3 at the SciPy 1.11 floor and returns NaN far in the tails at 1.18 | mean |
| nonparametric | Mann-Whitney U (`stats::wilcox.test(exact = FALSE)`): always the normal approximation with continuity correction, never exact, also without ties and with fewer than 50 values per group (`edge_k2`: ggstatsplot reports p = 0.0025; the exact p would be 0.0013); the fixture's method string omits "with continuity correction"; p follows `alternative` | rank-biserial r = 2U / (n1 n2) - 1 (`effectsize::rank_biserial`), U the Mann-Whitney U of group 1; CI tanh(atanh(r) -/+ z x sqrt((n1 + n2 + 1) / (3 n1 n2))), ties ignored; follows `alternative` ("greater" gives [lower bound, 1], "less" [-1, upper bound]); reproduced to 1e-10 | median |
| robust | Yuen, 20% trim (`WRS2::yuen`): t is reported unsigned, \|t\|, even when group 1's trimmed mean is the smaller (mtcars `am`, `edge_k2`), while the AKP effect size keeps its sign; WRS2 computes p as 2 (1 - pt(|t|, df)) (every robust omnibus p in the fixtures lies on the 1 - cdf rounding grid); inferplot reports the accurate tail, 2 P(T > |t|), which agrees within that rounding and stays positive where 1 - cdf is 0 (D43; §10, `wrs2_p`) | AKP robust standardized difference, `WRS2::akp.effect(EQVAR = FALSE)`: trimmed-mean difference scaled by group 1's winsorized SD, not the pooled SD (oracle: 0.6827 vs pooled 0.6393 on ToothGrowth); percentile bootstrap CI: each resample draws group 1 then group 2 with replacement at full size; the bounds are the (floor(alpha/2 B) + 1)-th and floor((1 - alpha/2) B)-th smallest of B values (alpha = 1 - `conf_level`; ggstatsplot: 100 resamples, inferplot 2,000), proved by replaying ggstatsplot's own 100-resample values exactly. WRS2's constant is 1.2e-9 relative above the exact square root of the winsorized normal variance, which inferplot uses (D42) | trimmed mean |
| bayes | JZS Bayes factor, r = 0.707 (`BayesFactor::ttestBF(formula, rscale = bf.prior)`) on Student's (pooled) t, two-sided whatever `alternative` says. `tools/reference`'s one-dimensional quadrature reproduces all 36 distinct two-sample values in the fixtures (subtitles, parametric captions, pairs) to 1.5e-9; Welch's t differs on every unbalanced pair, by 1.5e-4 or more | v0.2: posterior median of the mean difference with a 95% equal-tailed interval (`conf.method` "ETI" in the fixtures; 4,000 posterior draws per `parameters::model_parameters()`'s help page, to confirm) | MAP (`bayestestR::map_estimate`): Gaussian kernel density with `bw.SJ` (Sheather & Jones 1991, solve-the-equation; pair distances binned in 1,000 bins of 1.01 x range / 1000 anchored at zero, root by Brent's zero finder at tol 0.1 x lower) by `stats::density` (linear binning on 1,024 points from min - 4 bw to max + 4 bw, FFT, linear interpolation), read at 1,024 points from min(x) to max(x); the first point with the largest density. No density (fewer than 2 values, no spread, zero IQR) raises `ValueError` |

### betweenstats, three or more groups

| type | Omnibus | Effect size + CI | Pairwise, then `p_adjust_method` | Centrality |
| --- | --- | --- | --- | --- |
| parametric | Welch ANOVA (`stats::oneway.test`) | partial omega squared = max(0, (F - 1) df1 / (F df1 + df2 + 1)) with Welch's F and df (`effectsize::omega_squared`), one-sided CI [lower, 1] at `conf_level`: the estimate read as a partial eta squared gives F_w = omega df2 / ((1 - omega) df1), lambda solves P(F'(df1, df2, lambda) <= F_w) = `conf_level`, the lower bound is lambda / (lambda + df2), and 0 when no lambda > 0 does (D33, probe-omega). inferplot computes the noncentral F itself (SciPy 1.11's `ncfdtr` is off by up to 1.2 relative near P = 1) | Games-Howell (`PMCMRplus::gamesHowellTest`) **then Holm on top**; q is signed, mean(group2) - mean(group1); group1 is the earlier level in the analysis order, and rows are sorted by group1 and then group2 as text in the C locale (statsExpressions 2.2.1 `pairwise_comparisons()`: `dplyr::arrange(group1, group2)`), the order the brackets keep. p is the accurate studentized range upper tail at the pair's Welch df and \|q\| (`inferplot.stats.srange_sf`, D31), then `p_adjust_method` over all k(k-1)/2 pairs (ggstatsplot's adjusted p is exactly `p.adjust` of gamesHowellTest's raw p). ggstatsplot's p is R's `ptukey` value, whose error grows as the df falls (probe-ptukey: up to 1.35e-3 absolute at df 2, 1.8e-7 at df 10, 1.9e-8 at df 20 to 1000, a floor near 4.7e-10 in the far tail) and which is NaN below 2 df, where ggstatsplot stops with an error (fixture `edge_n2`, D37). SciPy's `studentized_range` is no ground truth in the far tail either (1.9e-14 for iris setosa-versicolor, 14% low) | mean |
| nonparametric | Kruskal-Wallis (`stats::kruskal.test`) | rank epsilon squared = H / (N - 1) (`effectsize::rank_epsilon_squared`; Tomczak & Tomczak 2014), one-sided percentile bootstrap CI [lower, 1]: each resample draws n_g values with replacement within every group (group sizes fixed; whole-row resampling misses the oracle's 20,000-resample references by up to about 220 replicate SDs); the lower bound is the bootstrap (1 - conf_level) quantile with `boot::boot.ci`'s percentile rule, order statistic (R + 1) alpha interpolated on the normal quantile scale (Davison & Hinkley 1997, ch. 5); ggstatsplot: 100 iterations, inferplot: 2,000 seeded, drawn from each group's sorted values with groups in content order, so row order and level order do not change it | Dunn (`PMCMRplus::kwAllPairsDunnTest`): z = abs(mean-rank difference) / tie-corrected SE, reported unsigned; two-sided normal p (reproduced to 2.4e-15); inferplot computes raw p for i < j in input order and adjusts with `p_adjust` | median |
| robust | trimmed-means ANOVA (`WRS2::t1way`); WRS2 computes p as 1 - pf, inferplot the accurate upper tail (D43; §10, `wrs2_p`) | explanatory effect size (`WRS2::t1way`; Wilcox & Tian 2011): xi = sqrt(var of the trimmed means, J - 1 denominator, / (pooled winsorized variance / v_w)), v_w the winsorized standard normal variance (exact; WRS2 divides by 0.4120867, 3.2e-8 relative in xi, D42). Bootstrap: every group resampled with replacement to the smallest group's size; bounds the (l + 1)-th and (B - l)-th smallest, l = round(alpha/2 B), alpha = 1 - `conf.level` computed in floating point (0.050000000000000044, so alpha/2 x 100 = 2.5000000000000022 rounds to 3); with unequal group sizes the point estimate is the median of the bootstrap values, so it is itself random (oracle replicate SD 0.0048 on mtcars) and compared within Monte Carlo error. Proved by replaying ggstatsplot's own 100-resample values and the 20,000-resample references exactly | `WRS2::lincon`: estimate = trimmed mean(group1) - trimmed mean(group2); p = per-pair Yuen p (1 - pt), then `p_adjust_method` (lincon's own Hochberg is not used); CI = estimate -/+ c x Yuen SE. c comes from a two-decimal table of the Studentized maximum modulus at (alpha, C = number of pairs, not groups: `edge_k5` has 10, `bugs_long` 6), whose rows are every df from 2 to 12, then 14, 16, 18, 20, 24, 30, 40, 60 and a last row at df 200 holding the infinite-df (normal) values; c is interpolated linearly in 1/df between the rows on either side of the pair's Yuen df (fixtures: df 5.2 to 110, rows up to 60/200; reproduced to 1e-14). The entries are the published table's, not recomputed quantiles: for C = 3 at df 5 the table has 3.39 where the quantile, 3.3992, rounds to 3.40 inferplot rebuilds the table from the distribution (`tools/reference/smm_table.py`, Stoline & Ury 1979) and rounds it; 21 entries are pinned by the fixtures, of which only C = 3 at df 5 differs (3.39), kept as published. From df 200 the infinite-df row (unconfirmed); df below 2, `conf_level` other than 0.95 and C above 45 raise `NotImplementedError` until probe-smm pins them | trimmed mean |
| bayes | v0.2: `BayesFactor::anovaBF` (method and precision to measure when implemented) | v0.2: Bayesian R squared, posterior median with 95% HDI (`conf.method` "HDI" in the fixtures) | v0.2: `ttestBF` per pair as for two groups, unadjusted, always with prior 0.707 and 95% intervals whatever `bf_prior` and `conf_level` say (every pair's Bayes factor is reproduced by `tools/reference`) | MAP |

**Scale and range (parametric).** Every parametric statistic, interval and Bayes
factor is computed on y times one power of two that brings the largest |y| of all
groups into [1/2, 1). Multiplying by a power of two is exact, so the results do not
depend on the units of y: the data times 2^k give bit-identical numbers for every k
that keeps them normal doubles (§11). The Welch-Satterthwaite df is computed from the
variance shares w_i / (w_a + w_b), so no square of a variance term can overflow or
underflow. In that scale a variance term v / n below 2^-1022 (the smallest normal
double) is not accurate, so inferplot raises a plain `ValueError` naming the group
instead of reporting a number: for two groups the term is the squared standard
error the t statistic divides by (Welch's; also Student's when the Bayes factor is
drawn), for three or more groups each group's v / n. In units of y that is roughly a
group standard deviation below 1e-154 times the largest |y| (§3). Consequently a
two-sample |t| stays below 2^512; `t_ncp_interval` accepts |t| up to 1e300 and the
JZS Bayes factor any finite t. The noncentral F behind the omega squared interval is
a Poisson mixture of beta distributions; from a Poisson mean of 1e4 on it is summed
on every h-th term, h = floor(sqrt(mu) / 8), the trapezoidal rule on an analytic
summand (Trefethen & Weideman 2014) with Loader's (2000) saddle-point weights, and
beta factors with a >= 1e5 max(b, 1) use an expansion through a^-2, so the interval
takes at most about 0.04 s at any F (DECISIONS log).

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
  ggstatsplot's far upper tail is R's `ptukey` value, which can be orders of magnitude
  above the true one (1.41e-9 for about 2.2e-14 on iris); inferplot prints the accurate
  value (§5 item 6, D31).
- `bf_prior=0.707` numeric. sqrt(2)/2 changes Bayes factors in the fifth significant figure.
- Mann-Whitney p is always the corrected normal approximation, also where R's own
  `wilcox.test` default would compute the exact p.
- Yuen's t and Dunn's z are reported unsigned; Games-Howell's q and every effect size
  keep their sign.
- WRS2's p-values (Yuen, t1way and lincon) are 1 - cdf, rounded to the spacing of
  doubles just below 1 and 0 below about 1e-16; inferplot reports the accurate tail
  (D43); lincon's CI uses a published two-decimal table with at least one entry
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
4. Nothing is silently dropped or aliased: unknown arguments, `type` values without a
   distinct method, a `subtitle` passed while `results_subtitle=True`, and a `tr`
   other than 0.2 with a type that trims nothing raise; arguments that have no effect
   on the given data (pairwise arguments with two groups, `bf_prior` where no Bayes
   factor is computed) add a `warnings` entry each (§2, D39).
5. v0.1 only: where ggstatsplot's default adds a Bayes factor that inferplot computes only
   from v0.2 (the Bayes ANOVA caption after a parametric test with three or more groups
   or conditions; unpaired contingency tables and one-way goodness of fit with a type
   other than bayes), no Bayes
   caption is drawn by default and `warnings` says so; `bf_message=True` raises
   `NotImplementedError` for those designs (§2). Posterior estimates in Bayes captions
   also arrive in v0.2 (§1). With `type="bayes"` (D41), two groups get a subtitle of
   log_e(BF01) and the prior scale with a `warnings` entry that the posterior estimate
   arrives in v0.2; three or more groups raise `NotImplementedError` unless
   `results_subtitle=False`, which draws ggstatsplot's own output without its
   subtitle: MAP labels and a Bayes factor bracket for every pair.
6. Games-Howell p-values are computed from an accurate upper tail of the studentized
   range, not from R's `ptukey`, whose error grows as the df falls (up to 1.35e-3
   absolute at df 2, 1.8e-7 at df 10, about 5e-10 in the far tail) and which is NaN
   below 2 df (D31, D37). Where that shows, the printed p differs from ggstatsplot's:
   3 of the 106 pairwise rows of the parametric fixtures: iris setosa-versicolor,
   where ggstatsplot prints 1.41e-09 and inferplot about 2.24e-14, iris
   setosa-virginica (2.83e-08 against 2.74e-08) and edge_negative b-c (1.27e-08
   against 1.28e-08; an mpmath reference confirms inferplot's raw p). Where every pair's Welch
   df is below 2 (two values per group), ggstatsplot 1.1.4 stops with an error and
   inferplot reports the accurate p-values. §10 says how they are tested.
7. The partial omega squared lower bound is the accurate root of ggstatsplot's own
   construction. Where effectsize's root search stops early (up to 2.1e-3 on the
   probe grid, never in the fixtures), inferplot's bound differs (D33).
8. At the limits of double precision the parametric statistics are computed in one
   power-of-two scale of all groups (§4, "Scale and range"), so data at any
   magnitude give the same numbers, and a group whose standard deviation is below
   roughly 1e-154 times the largest |y| raises a plain `ValueError`, also for some
   data a computation in the data's own units could handle (one constant group at
   1e160 against [1, 2, 3]). No fixture comes near either limit.

Visual styling is not a parity target: same semantic layers (points, box, violin,
centrality label, brackets, subtitle content, caption content), our own look. Default
palette Okabe-Ito; more than 8 levels uses `tab20` and adds a warning. Violins are
Gaussian kernel densities with Silverman's bandwidth (R's `bw.nrd0`) trimmed to the
data range; boxes use type-7 quartiles and Tukey whiskers without outlier markers (the
points show them). A panel with more than 5,000 values draws a uniform seeded
2,000-point subsample of the points; the violins (or, without a violin layer, unfilled
density outlines) and boxes show all values. A level with fewer than two distinct
values gets no violin, and `warnings` says so. inferplot sizes its own figure from the
rendered text extents, so no text is clipped or overlaps (D38): significance brackets
are stacked in points on top of a data region of at least 2.5 in, so the figure grows
only by the height the brackets need; with `ax=` nothing is
resized and `check()` reports what does not fit.

## 6. Result contract

`StatsFigure`: `figure`, `axes` (read-only mapping: `"main"`, plus `"xside"`/`"yside"`
for marginals), `result` (the result drawn, with any rendering warnings appended to its
`warnings`), `check()`, `save(path, *, dpi=300, transparent=False)` (format from the
extension, case-insensitive: `.svg`, `.pdf` or `.png`; anything else raises),
`_repr_png_` for notebooks. No `show()`.

`StatsResult` (frozen dataclass, plain Python types only, JSON-serializable):

| Field | Content |
| --- | --- |
| `analysis`, `design`, `type` | e.g. `"betweenstats"`, `"between"`, `"parametric"` (canonical) |
| `x`, `y`, `subject_id`, `levels` | column names and analyzed level order |
| `values` | per level, the analyzed `y` values in data order; the renderer draws from them (D36) |
| `descriptives` | per level: `n`, centrality name (`mean`, `median`, `trimmed_mean`, `map`) and value |
| `test` | name, R-style method string, statistic name and value, `df`, `df_error`, p-value, alternative |
| `effect_size` | name, estimate, `conf_level`, CI bounds, CI method, sidedness |
| `bayes` | `bf10`, `log_bf01`, prior name and scale; `None` when absent |
| `pairwise` | per pair: groups, statistic, p, adjusted p, adjust method, test name, significant, displayed |
| `n_obs`, `n_dropped`, `dropped_subjects` | counts |
| `warnings` | plain-English strings, no internal jargon |
| `provenance` | inferplot, numpy, scipy, matplotlib and narwhals versions (read from package metadata, so matplotlib is not imported); seed; parity target `"ggstatsplot 1.1.4"`; method citations |
| `settings` | the analysis arguments that shaped the numbers and texts, canonical values |
| `text` | plain Unicode: `subtitle`, `caption` (the generated lines joined by a newline; the user's caption is not part of it), `apa`, `alt_text`, `centrality_labels`, `tick_labels`, `pairwise_labels` (one per entry of `pairwise`, in that order) |

Methods: `to_dict()`, `to_json()`, `from_dict()` and `from_json()` (rebuild an equal
result), `to_frame(backend="pandas" | "polars")` (one row per test: the main test first,
then each pair), `check()`, `__str__` (APA sentence plus pairwise table). `to_json()` is
strict ASCII JSON: inf as `"Inf"`, -inf as `"-Inf"`, NaN as null. `to_frame()` columns:
test, label, group1, group2, statistic_name, statistic, df, df_error, p_value,
p_adjusted, adjust_method, alternative, effect_size, estimate, conf_level, conf_low,
conf_high, bf10, log_bf01, significant, displayed; missing values are nulls (pandas uses
nullable `boolean` columns).

`save()` sets `svg.fonttype="none"`, `pdf.fonttype=42` and a fixed `svg.hashsalt`
inside an `rc_context`, drops the SVG date and PDF creation date (saving twice writes
identical bytes), and embeds `result.to_json(include_values=False)` in file metadata:
the PNG text chunk "Description", the SVG Dublin Core description and the PDF Info
"Subject". The embedded copy holds every number and text but not the analyzed
observations, which can be large and are the user's data (D36); a result read back
from it has `values == ()` and cannot be drawn again.

## 7. Text formats

- **Subtitle, Bayes caption, bracket and centrality labels (D34):** inferplot builds
  the plotmath expression ggstatsplot builds (statsExpressions 2.2.1's templates) as an
  AST and renders it as mathtext for drawing and as plain Unicode for `result.text`
  (`t_Welch(55.31) = 1.92, p = 0.06, g_Hedges = 0.49, CI_95% [-0.02, 0.99],
  n_obs = 60`; `log_e(BF_01) = -0.18, r_Cauchy^JZS = 0.71`). Parity is AST equality with
  the fixture's parsed expression, so numbers must equal the formatted strings in the
  oracle's `expression` output exactly (same rounding, same switch to scientific
  notation, same `digits`). Bootstrap-derived numbers are the exception described in
  §10. The v0.1 two-group Bayes caption is the first and last items of ggstatsplot's
  (D32). For `type="bayes"` the two-group subtitle is the same two items (D41); with
  three or more groups and `results_subtitle=False` each bracket reads
  `log_e(BF_01) = <value>` and the caption's pairwise line reads "Pairwise: Bayes
  factors (Student's t, JZS prior 0.707), unadjusted; showing all <m> pairs." Tick labels are plain `<level>\n(n = <n>)`.
- **Numbers (probe-format):** `insight::format_value(x, digits)`: fixed notation with
  `digits` decimals, scientific (`digits` mantissa decimals, signed two-digit exponent)
  when x is not 0 and log10(|x|) < -digits or |x| >= 1e5, judged on the unrounded
  value; NA and NaN print as "" in bracket and centrality labels and the log Bayes
  factor (direct `format_value`) and as "NA" in subtitle tables (`.to_char`); Inf and
  -Inf as "Inf" and "-Inf". Rounding is C `sprintf` rounding of the binary double. n
  gets a comma every three digits from 1,000 on (`.prettyNum`). The CI level is
  `paste0(conf.level * 100, "%")` as R prints it (89%, 97.5%, 99.99%). An exact p of 0
  prints as 2.23e-308 in the subtitle (`.Machine$double.xmin`). Negative zero is not
  pinned.
- **Degrees of freedom** are numbers, not strings, in ggstatsplot's expression:
  statsExpressions rounds `df` and `df.error` to `digits` decimals where they are
  fractional (Welch t and F, Yuen, t1way; within-subjects parametric and robust) and to
  0 otherwise, and pastes them unquoted, so they print as R prints a number: without
  trailing zeros or a bare decimal point (iris Welch F `(2, 97.4)` for 97.40; mtcars
  `am` x `drat` Welch t `(27.2)`; mtcars `cyl` x `disp` t1way `(2, 10.4)`). Every other
  number is a quoted string and keeps its zeros ("1.00"). An infinite one-sided bound
  prints as "Inf" or "-Inf".
- **Caption:** Bayes factor line as ggstatsplot (log_e BF01, prior r) where §2's
  `bf_message` rule draws it; then the pairwise line for three or more groups unless
  `pairwise_display="none"`: "Pairwise: Games-Howell, Holm adjusted; showing 1 of 3
  pairs (significant only).", "... showing 2 of 3 pairs (non-significant only)." or
  "... showing all 3 pairs."; adjustments read Holm, Hochberg, Hommel, Bonferroni,
  Benjamini-Hochberg (BH, fdr), Benjamini-Yekutieli (BY), and "Games-Howell,
  unadjusted" for none. Bracket labels keep ggstatsplot's `p.adjust.method` labels
  (Holm, Bonferroni, Hochberg, Hommel, FDR for BH and fdr, BY, and `p_unadj.`).
- **APA sentence (APA 7):** "Welch's t-test: t(9.62) = -2.35, p = .041, Hedges' g =
  -1.07, 95% CI [-2.01, -0.08], n = 12."; "Welch's ANOVA: F(2, 17.13) = 5.18, p = .017,
  partial omega squared = .29, 95% CI [.00, 1.00], n = 30." Two decimals for statistics;
  p to three decimals without leading zero, "p < .001" below .001; no leading zero for
  quantities bounded by 1 (omega and eta squared, epsilon squared, correlations,
  rank-biserial r, Cramér's V, Kendall's W); non-integer df to two decimals, integer df
  bare; a one-sided test reads "Welch's t-test (one-sided, greater): ..."; ASCII only.
- **Alt text:** two to four sentences, at most 400 characters: "Plot comparing <y>
  across the <k> groups of <x>: <level> (n = <n>, mean <value>), ...", then the APA
  sentence, then "<s> of <m> pairwise comparisons are significant (<test>,
  <adjustment>, alpha = <alpha>)." When that is longer than 400 characters the groups
  are summarized as ranges and long names are shortened with an ellipsis (U+2026).

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
| `label_overlap` | figure | warning | any two text artists' rendered bounding boxes intersect, or a bracket label's box reaches more than 0.5 px into the bar or a leg of another bracket |
| `text_clipped` | figure | warning | any text extends outside the figure |
| `points_subsampled` | figure | info | large-n subsampling used |

`label_overlap` compares the rendered boxes of every drawn, non-empty text: title,
subtitle, caption lines, axis labels, visible tick labels, centrality labels with their
label boxes, bracket labels; with `ax=` only the texts of that Axes. `text_clipped`
fires when a text extends more than 0.5 px outside the figure, or a text drawn inside
the Axes (centrality and bracket labels; with `ax=` also the subtitle and caption)
extends outside it. For bracket labels only the drawn lines of the other brackets
count: a label inside the area an enclosing bracket spans is fine, and so is contact
within anti-aliasing slack: the label's box must reach more than 0.5 px into the
line's stroke (caps included) in both directions, or past its centre when the stroke
is thinner than 1 px. With `ax=` nothing is resized: the bracket levels keep their
spacing while the data keep 2.5 in or half the Axes height left after the subtitle
and caption, whichever is less; a shorter Axes pushes the levels closer together,
which can put a label on another bracket's line, and `check()` reports it. Messages name texts
in plain words, never mathtext source.

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
  script for those designs, and so does a case that declares a swallowed error.
- ggstatsplot 1.1.4 computes the subtitle and the caption through an internal helper
  that turns an error into a missing subtitle or caption and mutes warnings and
  messages, and it mutes warnings around `centrality_description()` and
  `pairwise_comparisons()`. The generator replaces these with recorders that make the
  same calls (same arguments, same random-number stream) and logs what ggstatsplot hid
  in `suppressed`. It stops when ggstatsplot swallows an error the case does not
  declare, when a subtitle or Bayes caption ggstatsplot should draw is missing, or when
  `n_per_level` differs from ggstatsplot's own `n.obs`. It knows exactly the arguments
  in `KNOWN_ARGS` and stops on any other; from each case's arguments it derives the
  hidden calls (subtitle and caption unless `results.subtitle = FALSE`, the caption only
  for type parametric with `bf.message`; `centrality_description()` always;
  `pairwise_comparisons()` for three or more groups unless `pairwise.display = "none"`),
  which outputs exist and their order, and stops on any difference. A case where
  ggstatsplot itself fails declares `expect_error` (the failing hidden call, text of the
  message and the reason); the generator checks all three and keeps what the hidden
  calls returned before the failure (`partial`). `edge_n2` is such a case (D37).
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
- Cases: Phase 0's 45 (all four types on ten dataset and variable pairs, one-sided
  alternatives) and Phase 1's 28 parametric ggbetweenstats cases: edge data
  (`edge_missing`; `edge_empty_level` with declared levels c, empty, a, b;
  `PlantGrowth-declared` with trt2, ctrl, trt1; `edge_n2`; `edge_ties`; `edge_negative`;
  `edge_outliers`; `InsectSprays`, k = 6; `edge_null`, Welch F 0.65; `edge_large_n`,
  n 1,200, 1,300 and 1,100 from a formula recorded in `generate.R`) and arguments
  (`PlantGrowth-padj-*` for every other `p.adjust` method; `conf.level` 0.90 on
  ToothGrowth and PlantGrowth; `digits` 3 on mtcars `am` and `cyl`;
  `pairwise.display` all, non-significant and none; `results.subtitle = FALSE` on
  ToothGrowth and PlantGrowth; `bf.message = FALSE`; `bf.prior` 0.8).
- Probe files (`probe-<topic>.json`) record what R functions behind ggstatsplot return
  on fixed grids the cases cannot cover. Doubles are C99 hex text (`sprintf("%a")`,
  read exactly with `r_double()`); R's NA is "NA". `probe-format`:
  `insight::format_value`, statsExpressions' `.to_char` and `.prettyNum`, the df route
  (`deparse(str2lang(.to_char(df, d)))`), CI level labels, a few whole expressions and
  centrality doubles on tie data, over 1,002 values and digits 0 to 4. `probe-padjust`:
  `stats::p.adjust`, all 8 methods, on 20 vectors. `probe-ptukey`: `ptukey`'s upper tail
  and 1 - cdf on a grid of q, k and df, and `PMCMRplus::gamesHowellTest`'s unadjusted p
  for every pair of every parametric case with three or more groups.
  `probe-omega`: `effectsize::F_to_omega2` and `F_to_eta2` at F_w on a grid of F, df,
  levels and alternatives, the noncentral F probability at each reported bound, and an
  accurate `uniroot` lower bound (tol 1e-12) for "greater". They are hashed in the
  manifest (`probes`) and checked like the fixtures; tests read them with
  `load_probes()` / `probe()` and `OracleProbe.section()` / `rows()`.
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
| `expected_error` | null, or the failure the case declares: `component`, `message`, `reason` |
| `partial` | null unless ggstatsplot failed; then one entry per hidden call that ran (`component`, `value`, data frames as row objects with plotmath deparsed): the subtitle, caption and centrality tables computed before the failure, and last the failing call itself (`pairwise_comparisons` in `edge_n2`) with value null |
| `suppressed` | what ggstatsplot hid, one entry per hidden call in call order: `component` (`subtitle`, `caption`, `centrality_description`, `pairwise_comparisons`), `error` (null unless ggstatsplot swallowed one; errors of the last two are not swallowed and end up in `error`), and the muted `warnings` and `messages` |
| `expected_hidden_errors` | components whose error the case declares ggstatsplot swallows; empty for every current case. An error raised inside `centrality_description()` or `pairwise_comparisons()` is also logged on its `suppressed` entry (it is not swallowed and ends in `error` as well) |
| `subtitle`, `caption` | deparsed plotmath from `extract_subtitle()` and `extract_caption()` |
| `stats` | every `extract_stats()` table as an array of row objects; plotmath columns deparsed. In 1.1.4 `descriptive_data` is null |
| `plot_layers` | the ggplot layers in order: `name`, `geom`, and the layer's own data (the centrality labels: value, n, `expression`, `n.expression`), null for layers that draw the main data. Bracket layers (`GeomSignif`) also carry `params` (`stat`, `geom`, `aes`; `stat.comparisons` lists the drawn pairs, `stat.annotations` their labels) and `built`, the rows ggplot2 computes for drawing |
| `bootstrap_reference` | null, or: `call`; `ggstatsplot` (its estimate and bounds, 100 resamples, seed); `replay` (`matches` and each candidate's replayed values); the 20,000-resample `estimate`, `conf_low`, `conf_high` and `raw` output; `replicates` (mean and SD of the estimate and each bound over 50 runs of 2,000) |
| `versions` | R version, platform, OS, collation, repositories, package versions |

Numbers carry 15 significant digits; R's NA is null; NaN, Inf and -Inf are the strings
`"NaN"`, `"Inf"`, `"-Inf"`. `manifest.json` (schema version 2) holds the SHA-256 of
every fixture and of the generator files (`generator_files`: `oracle/generate.R`,
`oracle/export_datasets.R`, `.github/workflows/oracle.yml`), of every probe file
(`probes`), the GitHub Actions run that produced them (`run`: repository, run id, attempt, commit), R's version and RNG kind,
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
| Games-Howell p against ggstatsplot (`games_howell_p`) | rtol 1e-6, atol 3e-9 | R's ptukey error (probe-ptukey) grows as the df falls; on the fixtures' pairs it is at most about 4.7e-10 raw and, times the Holm multiplier, 2.8e-9 after Holm (InsectSprays B-E, 15 pairs); inferplot computes the accurate value (§5 item 6), checked against high-precision references under `studentized_range_tail` |
| studentized range tail against mpmath references (`studentized_range_tail`) | rtol 4e-13 | 792 frozen mpmath values (`tests/data/studentized_range_reference.json`, regenerated by `tools/reference/studentized_range_mpmath.py`): a grid, df = Inf, exact k = 2 values, and every Games-Howell pair of the completed parametric fixtures (109 pairs) plus the 3 of `edge_n2`; measured at most 1.45e-13, the rounding of exp(log P) at P = 1e-300, and 5.2e-15 on the fixture pairs |
| noncentral t and F distribution functions (`noncentral_cdf`) | rtol 2e-11 | 521 frozen mpmath points (`tests/data/noncentral_reference.json`); measured at most 2.5e-12 (F at SciPy 1.11), 1.6e-14 for t |
| p-values WRS2 computes as 1 - pt or 1 - pf (`wrs2_p`: Yuen, t1way, lincon), against inferplot's accurate tail (D43) | rtol 1e-10, atol 2^-53 x the factor applied to the one-sided tail afterwards (2 for two-sided, times the number of pairs after Holm) | the one-sided tail rounded to the spacing of doubles just below 1 (2^-53); rounding the tail the same way matches all fixtures but iris t1way, which lands one grid step away |
| noncentral-distribution CIs (`noncentral_ci`) | rtol 1e-6, atol 4e-8 | R's root finding is accurate to about 1e-8 absolute: Hedges' g bounds at most 1.9e-8 absolute and 7.5e-7 relative (ToothGrowth lower bound, -0.0217), omega lower bounds at most 1.5e-8 absolute; an omega bound where effectsize's search stopped early is compared with the accurate root instead (D33) |
| JZS Bayes factors of t tests (`ttest_bayes_factor`) | rtol 1e-8 | measured on two-sample values only: `tools/reference` reproduces all 36 distinct `ttestBF` values in the fixtures to at most 1.5e-9, which is BayesFactor's own integration error; a prior of sqrt(2)/2 instead of 0.707 moves them by 3.7e-6 or more, Welch's t instead of Student's moves every unbalanced pair by 1.5e-4 or more. Measure paired and one-sample values when their fixtures arrive |
| other Bayes factors (correlation; v0.2 ANOVA and contingency) | own entry, measured when first implemented | not yet measured |
| bootstrap CIs and p-values | within 4 Monte Carlo SDs of the 20,000-resample R reference, never less than the closed-form tolerance | randomness; a quantity whose replicate SD is 0 does not depend on the resamples (AKP and rank epsilon squared estimates, an upper bound of 1) and is stored to 15 significant digits |
| MAP centrality (`MAP_CENTRALITY`) | the same grid point (0 steps; the oracle value must lie on the grid within 1e-6 steps) | all 30 fixture labels on ggstatsplot's grid point; each method variant tried moves 1 to 14 of them |
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

**Games-Howell p (§5 item 6).** Checked the same two ways: the value inferplot computes
is within `games_howell_p` (rtol 1e-6, atol 3e-9: R's measured error after Holm) of
ggstatsplot's value and within `studentized_range_tail` of a high-precision reference;
and the formatter applied to ggstatsplot's own value reproduces ggstatsplot's label.
Where both values format to the same string, inferplot's own string must equal it. The
pipeline is checked in pieces too: inferplot's raw p against gamesHowellTest's raw p
(probe-ptukey) within `games_howell_p`, and `p_adjust` applied to gamesHowellTest's raw
p reproduces ggstatsplot's adjusted p within `closed_form`.

Every other subtitle number is compared as the exact string inferplot produces.

## 11. Other tests

- **Reference cross-checks:** where `tools/reference/stats_reference.py` covers a method,
  production code must agree with it too.
- **Simulation (marker `slow`):** 2,000 seeded replications each, unequal variances and
  unequal n, every rate inside the binomial 99% band of its nominal value: Type I error
  at alpha 0.05 under the null; coverage of the 95% Hedges' g interval at nonzero true
  standardized differences delta = (mu1 - mu2) / sqrt((s1^2 + s2^2) / 2) (0.263, 0.788
  and -0.525). At delta = 0 the interval covers 0 exactly when the Welch p is at least
  alpha, so coverage there repeats the Type I error test and is kept only as that.
- **Scale:** the data times 2^k give bit-identical statistics, intervals, Bayes factors
  and Games-Howell rows for every k that keeps them normal doubles; times 10^k (k =
  -300..300) they agree within `closed_form` (`noncentral_ci` for interval bounds)
  (`tests/stats/test_scale.py`).
- **Properties (hypothesis):** row-order invariance; level relabeling invariance; pandas,
  polars and pyarrow inputs give identical results; standardized effect sizes invariant
  to location and positive scale.
- **Rendering:** structural SVG assertions (layers present, label text equals result
  text) on every OS; five pytest-mpl image baselines (`tests/render/test_images.py`:
  ToothGrowth, PlantGrowth and InsectSprays with every pair bracketed, a subsampled
  panel, an `ax=` figure), generated by ci.yml's `mpl-baselines` job and compared only
  in the `--mpl` job (ubuntu-latest, newest Python) at RMS tolerance 0.1 on the 0-255
  scale (D44). For a figure changed on purpose: `gh workflow run ci.yml --ref <branch>
  -f mpl_generate=true`, then `gh run download <run-id> -n mpl-baselines -D
  tests/render/baseline`, look at every image and commit them on that branch.

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
