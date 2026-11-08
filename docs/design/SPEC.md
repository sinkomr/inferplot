# inferplot specification

Status: draft 1, 2026-11-07. Parity target: **ggstatsplot 1.1.4** (CRAN, 2026-10-22).
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
| `bf_message` | `True` | Bayes factor caption when a Bayes factor exists for the design |
| `results_subtitle` | `True` | |
| `alternative` | `"two-sided"` | `"less"`, `"greater"` only for two-group, paired, one-sample and correlation tests; otherwise `ValueError` |
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
| Hedges' g and its CI | Hedges 1981; Hedges & Olkin 1985; noncentral CIs: Steiger & Fouladi 1997, Smithson 2003 |
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
| parametric | Welch's t (`stats::t.test`, var.equal = FALSE) | Hedges' g (`effectsize::hedges_g`) *(verify pooled vs unpooled SD and CI method)* | mean |
| nonparametric | Mann-Whitney U (`stats::wilcox.test`) *(verify exact/correct)* | rank-biserial r (`effectsize::rank_biserial`) *(verify CI)* | median |
| robust | Yuen, 20% trim (`WRS2::yuen`) | AKP robust standardized difference (`WRS2::akp.effect`), bootstrap CI | trimmed mean |
| bayes | JZS Bayes factor, r = 0.707 (`BayesFactor::ttestBF`) | posterior difference: v0.2 | MAP *(verify density settings)* |

### betweenstats, three or more groups

| type | Omnibus | Effect size + CI | Pairwise, then `p_adjust_method` | Centrality |
| --- | --- | --- | --- | --- |
| parametric | Welch ANOVA (`stats::oneway.test`) | partial omega squared (`effectsize::omega_squared`); oracle shows CI [0, 1], so one-sided | Games-Howell (`PMCMRplus::gamesHowellTest`) **then Holm on top** | mean |
| nonparametric | Kruskal-Wallis (`stats::kruskal.test`) | rank epsilon squared (`effectsize::rank_epsilon_squared`), bootstrap CI | Dunn (`PMCMRplus::kwAllPairsDunnTest`) | median |
| robust | trimmed-means ANOVA (`WRS2::t1way`) | explanatory effect size (`WRS2::t1way`), bootstrap CI | `WRS2::lincon` | trimmed mean |
| bayes | v0.2: `BayesFactor::anovaBF` | v0.2: Bayesian R squared | v0.2: `ttestBF` per pair, unadjusted | MAP |

### withinstats, two conditions

| type | Test | Effect size + CI | Centrality |
| --- | --- | --- | --- |
| parametric | paired t (`stats::t.test`, paired) | Hedges' g, paired *(verify d_z vs d_av)* | mean |
| nonparametric | Wilcoxon signed-rank (`stats::wilcox.test`, paired) *(verify exact/correct)* | rank-biserial r *(verify CI)* | median |
| robust | Yuen for dependent samples (`WRS2::yuend`) | AKP (`WRS2::wmcpAKP`); 1.1.3 fixed this value | trimmed mean |
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
| nonparametric | one-sample Wilcoxon signed-rank | rank-biserial r | median |
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
| any | bayes | v0.2: `contingencyTableBF` (independent multinomial, rows fixed, a = 1) | v0.2 |

Per-bar proportion tests (goodness of fit of `x` within each `y` level) are adjusted with
`p_adjust_method` (added in ggstatsplot 1.0.0). Pairwise Fisher exact tests between `x`
levels appear in `extract_stats` since 1.0.0 *(verify)*; include them in the result.
`type="robust"` has no distinct method: raise `ValueError` (ggstatsplot silently treats
it as parametric).

### Defaults that look odd but are parity requirements

- Games-Howell p-values get Holm adjustment on top (double adjustment). Replicate it.
- `bf_prior=0.707` numeric. sqrt(2)/2 changes Bayes factors in the fifth significant figure.
- Pairwise display defaults to significant pairs only.

## 5. Deliberate differences from ggstatsplot

Statistical and behavioral differences are exactly these:

1. `subject_id` is required in `withinstats`; pairing never comes from row order.
2. Bootstrap CIs use 2,000 seeded resamples (ggstatsplot hard-wires 100).
3. When only significant pairs are drawn, the caption states how many pairs were tested.
4. Nothing is silently dropped or aliased: unknown arguments and `type` values without a
   distinct method raise.

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
  rounding, same switch to scientific notation, same `digits`).
- **Caption:** Bayes factor line as ggstatsplot (log_e BF01, prior r) when `bf_message`
  and a Bayes factor exists; then the pairwise line, e.g. "Pairwise: Games-Howell, Holm
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

- `oracle/generate.R` runs ggstatsplot 1.1.4 on `oracle/datasets/*.csv` for every
  function × type × dataset ggstatsplot accepts and writes one JSON file per case:
  inputs, arguments, R and package versions (ggstatsplot, statsExpressions, effectsize,
  WRS2, BayesFactor, PMCMRplus, afex, correlation), `extract_stats()` tables, and the
  subtitle and caption expression strings.
- For bootstrap-based CIs, the script also calls the underlying function directly with
  at least 20,000 resamples to give a precise reference.
- `.github/workflows/oracle.yml` (manual dispatch): `r-lib/actions/setup-r@v2` with R
  4.5, CRAN pinned to a Posit Package Manager snapshot dated on or after 2026-10-23, and
  a hard check `packageVersion("ggstatsplot") == "1.1.4"`. It uploads the JSON files as
  artifact `oracle-fixtures`; they are committed to `tests/fixtures/oracle/` after review.
- Datasets: mtcars, iris, ToothGrowth, PlantGrowth, morley, sleep, ggstatsplot's
  `bugs_long` and `Titanic_full`, plus edge cases: n = 2 per group, ties, all-negative
  outcome, empty factor level, missing values, unbalanced groups, k = 2, 3 and 5, a
  subject missing one condition, heavy outliers.
- plotsalot's `oracle/` (MIT) is a usable starting point; credit it in `NOTICE`.
- Python tests load the same CSV from `oracle/datasets/`, translate the fixture's R
  arguments to inferplot keywords with one mapping table (`tests/oracle_args.py`: dots to
  underscores, R `TRUE`/`FALSE`/`NULL` to Python), run the analysis, and compare every
  number the fixture holds, using `tests/tolerances.py`. One parametrized test per
  function collects all its fixtures, so a new fixture is tested without new code.

## 10. Tolerances

`tests/tolerances.py` maps each quantity kind to `(rtol, atol, reason)`. Starting values:

| Quantity | Tolerance | Reason |
| --- | --- | --- |
| closed-form statistics, df, t/F/chi-square/normal p-values | rtol 1e-10 | both sides exact to double precision |
| Games-Howell p | rtol 1e-6 | R ptukey vs SciPy studentized_range measured 2.6e-7 |
| noncentral-distribution CIs | start rtol 1e-6; set from measurement | R root-finding precision |
| Bayes factors | rtol 1e-4 | BayesFactor numerical integration error |
| bootstrap CIs and p-values | within 4 Monte Carlo SE of the 20,000-resample R reference | randomness |
| MAP centrality | one grid step of the density grid | kernel density grid |
| subtitle numbers | exact string match | formatting parity |

Never loosen an entry without a measured reason written in the registry.

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
