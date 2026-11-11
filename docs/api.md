# inferplot API reference (v0.1)

`import inferplot as ip`. Every function exists three ways:

- `ip.<name>(data, x, y, ...)` analyzes and draws; returns a `StatsFigure`.
- `ip.analyze.<name>(data, x, y, ...)` computes a `StatsResult` (every number and
  text) and imports no matplotlib.
- `ip.render.<name>(result, ...)` draws a result of `ip.analyze.<name>`; returns a
  `StatsFigure`.

`ip.<name>` takes the keywords of `ip.analyze.<name>` followed by those of
`ip.render.<name>`, with the same defaults. `data`, `x` and `y` are positional;
everything else is keyword-only. `data` is a pandas, polars or pyarrow data frame
in long format (one row per observation). The public names are `betweenstats`,
`withinstats`, `histostats`, `scatterstats`, `barstats`, `analyze`, `render`,
`StatsResult`, `StatsFigure`, `Issue` and `__version__`.

Examples below use `df` (columns `group`, `score`), `long` (`subject`, `time`,
`score`), `cars` (`wt`, `mpg`) and `answers` (`answer`, `group`, `before`,
`after`), pandas data frames.

## Signatures

The one-call functions are `ip.betweenstats`, `ip.withinstats`, `ip.histostats`,
`ip.scatterstats` and `ip.barstats`; their two halves follow.

```python signature
ip.analyze.betweenstats(data, x, y, *, type='parametric',
    pairwise_display='significant', p_adjust_method='holm', pairwise_alpha=0.05,
    conf_level=0.95, digits=2, tr=0.2, bf_prior=0.707, bf_message=None,
    results_subtitle=True, alternative='two-sided', seed=0, order=None)
```

```python signature
ip.render.betweenstats(result, *, centrality_plotting=True,
    layers=('violin', 'box', 'points'), xlab=None, ylab=None, title=None,
    subtitle=None, caption=None, palette=None, point_args=None,
    boxplot_args=None, violin_args=None, ax=None)
```

```python signature
ip.analyze.withinstats(data, x, y, *, subject_id, type='parametric',
    pairwise_display='significant', p_adjust_method='holm', pairwise_alpha=0.05,
    conf_level=0.95, digits=2, tr=0.2, bf_prior=0.707, bf_message=None,
    results_subtitle=True, alternative='two-sided', seed=0, order=None)
```

```python signature
ip.render.withinstats(result, *, point_path=None, centrality_path=None,
    centrality_plotting=True, layers=('violin', 'box', 'points'), xlab=None,
    ylab=None, title=None, subtitle=None, caption=None, palette=None,
    point_args=None, boxplot_args=None, violin_args=None, ax=None)
```

```python signature
ip.analyze.histostats(data, x, *, type='parametric', test_value=0.0,
    conf_level=0.95, digits=2, tr=0.2, bf_prior=0.707, bf_message=None,
    results_subtitle=True, alternative='two-sided', seed=0)
```

```python signature
ip.render.histostats(result, *, binwidth=None, centrality_plotting=True,
    xlab=None, ylab=None, title=None, subtitle=None, caption=None, palette=None,
    bar_args=None, ax=None)
```

```python signature
ip.analyze.scatterstats(data, x, y, *, type='parametric', conf_level=0.95,
    digits=2, tr=0.2, bf_prior=0.707, bf_message=None, results_subtitle=True,
    label_var=None, seed=0)
```

```python signature
ip.render.scatterstats(result, *, marginal=True, smooth=True, xlab=None,
    ylab=None, title=None, subtitle=None, caption=None, palette=None,
    point_args=None, smooth_args=None, marginal_args=None, ax=None)
```

```python signature
ip.analyze.barstats(data, x, y=None, *, counts=None, paired=False, ratio=None,
    label='percentage', proportion_test=None, p_adjust_method='holm',
    digits_perc=0, type='parametric', conf_level=0.95, digits=2,
    alternative='two-sided', bf_message=None, results_subtitle=True, seed=0)
```

```python signature
ip.render.barstats(result, *, xlab=None, ylab=None, title=None, subtitle=None,
    caption=None, legend_title=None, palette=None, bar_args=None, ax=None)
```

## Arguments

Functions: B betweenstats, W withinstats, H histostats, S scatterstats, C barstats
(counts).

Analysis arguments (`ip.<name>`, `ip.analyze.<name>`):

| Argument | Used by | Values and effect |
| --- | --- | --- |
| `data` | all | pandas, polars or pyarrow data frame, long format. A NumPy array, list or dict raises `TypeError`. |
| `x` | all | Column name. B, W: the groups or conditions; H: the numeric column tested; S: the first numeric column; C: the categories that fill each bar. |
| `y` | B W S C | Column name. B, W: the numeric outcome; S: the second numeric column; C: the categories that make the bars, or `None` for one bar (goodness of fit). |
| `subject_id` | W | Required. The column naming each subject; values are paired by it, never by row order. Subjects missing a level are dropped and counted. |
| `type` | all | `"parametric"` (`"p"`), `"nonparametric"` (`"np"`), `"robust"` (`"r"`), `"bayes"` (`"bf"`). The result records the full name. C: parametric and nonparametric run the same test; robust raises `ValueError`; bayes `NotImplementedError`. |
| `pairwise_display` | B W | Which pairs get a bracket (3+ levels): `"significant"` (`"s"`), `"non-significant"` (`"ns"`), `"all"`, `"none"`. Every pair is computed either way. |
| `p_adjust_method` | B W C | Multiplicity adjustment of the pairs: `"holm"`, `"hochberg"`, `"hommel"`, `"bonferroni"`, `"BH"`, `"BY"`, `"fdr"`, `"none"`. C: the pairwise Fisher tests, never the bars' tests. |
| `pairwise_alpha` | B W | A pair is significant when its adjusted p is below this (0.05). |
| `conf_level` | all | Level of the effect-size interval (0.95). Not the robust pairwise intervals or the robust within-subjects intervals (always 95%), nor C's bar tests. |
| `digits` | all | Decimals in the subtitle, caption and labels, 0 to 15. |
| `tr` | B W H S | Trimming (or winsorizing, S) proportion of `type="robust"`, 0 <= tr < 0.5 (0.2). Another value with another type raises `ValueError`. |
| `bf_prior` | B W H S | Prior scale of the Bayes factor (0.707; S: the beta prior's width). |
| `bf_message` | all | Bayes factor caption under a parametric subtitle. `None`: drawn where v0.1 computes it, otherwise a note in `warnings`; `True`: required (raises where it is not available); `False`: never. |
| `results_subtitle` | all | `False` draws no results subtitle (and no Bayes caption); then `subtitle=` may be given. |
| `alternative` | B W H C | `"two-sided"`, `"greater"`, `"less"`. B, W: two levels, types parametric and nonparametric only; `"greater"` tests whether the first level is greater than the second. H: x against `test_value`. C: makes the effect-size intervals one-sided, not the chi-square p. Elsewhere `ValueError`. S has none: correlations are tested two-sided. |
| `seed` | all | Seed of every random step (default 0), recorded in the result. |
| `order` | B W | Level order: a list (or other ordered collection) of the levels present. Default: declared categories, else sorted. Sets which level is "first" for `alternative`. |
| `test_value` | H | The value x is tested against (0.0). |
| `label_var` | S | A column whose values label the points; missing or empty values give no label. |
| `counts` | C | A column of whole numbers: how many observations each row stands for (data given as a table). |
| `paired` | C | `True`: x and y are two measurements of the same units (McNemar's test); one row per unit. |
| `ratio` | C | Expected shares of the levels of x, in order, summing to 1 (equal by default). Never rescaled. |
| `label` | C | Cell labels: `"percentage"`, `"counts"` or `"both"`. |
| `proportion_test` | C | Label each bar with its goodness-of-fit p. `None` follows `results_subtitle`. Needs `y`. |
| `digits_perc` | C | Decimals of percentage labels (0). |

Drawing arguments (`ip.<name>`, `ip.render.<name>`):

| Argument | Used by | Values and effect |
| --- | --- | --- |
| `result` | all | The `StatsResult` of the matching `ip.analyze` function. A copy read back from a saved figure has no observations; only C can draw it again. |
| `centrality_plotting` | B W H | Draw each level's centrality point and label (H: the dashed line). |
| `layers` | B W | Any of `"violin"`, `"box"`, `"points"`, drawn back to front. |
| `point_path` | W | Join each subject's points. `None`/`True`: with two conditions and the points layer; `False`: never. |
| `centrality_path` | W | Join the centrality points. `None`/`True`: with `centrality_plotting=True`. |
| `binwidth` | H | Bar width; default (max - min) / sqrt(n). At most 10,000 bars. |
| `marginal` | S | Histograms of x and y beside the plot. |
| `smooth` | S | Least-squares line with its `conf_level` band. |
| `xlab` | all | x axis title; default the column name (C: the name of y). Text between `$` signs is mathtext. |
| `ylab` | all | y axis title; default the column name (H: "count"; C: none). |
| `title` | all | Title above the subtitle. |
| `subtitle` | all | Own subtitle; only for a result with `results_subtitle=False`, else `ValueError`. |
| `caption` | all | Own text below the generated caption. |
| `legend_title` | C | Title of the legend of x (default the name of x). |
| `palette` | all | A matplotlib colormap name or a list of colours. Default Okabe-Ito; `tab20` with a note above 8 levels. |
| `point_args` | B W S | matplotlib properties of the points (`Artist.set`); B, W: needs the points layer. |
| `boxplot_args` | B W | Properties of the boxes; needs the box layer. |
| `violin_args` | B W | Properties of the violins; needs the violin layer. |
| `bar_args` | H C | Properties of every bar (`Rectangle`); properties that move, rotate or reorder the bars are refused; C: also the fill colour (use `palette`), edges over 2 points, alpha under 0.4. |
| `smooth_args` | S | Properties of the line (`Line2D`); the band takes its colour. Needs `smooth=True`. |
| `marginal_args` | S | Properties of the marginal bars. Needs `marginal=True`. |
| `ax` | all | Draw into this matplotlib Axes; nothing is resized, so run `check()`. B, W, C: its x axis may not be shared. S needs `marginal=False`. |

## What each function computes

The methods are ggstatsplot 1.1.4's. Effect sizes come with a `conf_level`
interval; "one-sided" means [lower bound, 1]. Pairwise p-values are adjusted with
`p_adjust_method`, except Bayes factors.

**`betweenstats`**, independent groups:

| type | 2 groups | 3+ groups | Pairs (3+) | Centre |
| --- | --- | --- | --- | --- |
| parametric | Welch's t; Hedges' g (noncentral t CI); Bayes factor caption | Welch's ANOVA; partial omega squared (one-sided) | Games-Howell, then `p_adjust_method` | mean |
| nonparametric | Mann-Whitney U (normal approximation); rank-biserial r | Kruskal-Wallis; rank epsilon squared (bootstrap, one-sided) | Dunn | median |
| robust | Yuen's trimmed-means t; AKP difference (bootstrap) | trimmed-means ANOVA; explanatory xi (bootstrap) | Yuen (lincon), 95% intervals | trimmed mean |
| bayes | JZS Bayes factor of Student's t | needs `results_subtitle=False` | Bayes factor per pair, unadjusted | MAP |

**`withinstats`**, the same subjects measured repeatedly:

| type | 2 conditions | 3+ conditions | Pairs (3+) | Centre |
| --- | --- | --- | --- | --- |
| parametric | paired t; Hedges' g of the differences; Bayes caption | repeated-measures ANOVA, Greenhouse-Geisser corrected; partial omega squared (one-sided) | paired t | mean |
| nonparametric | Wilcoxon signed-rank; matched-pairs rank-biserial r | Friedman; Kendall's W (bootstrap, one-sided) | Durbin-Conover | median |
| robust | Yuen's dependent t; AKP of the differences (bootstrap, 95%) | trimmed-means repeated-measures ANOVA; average AKP (95%) | Yuen (rmmcp), 95% intervals | trimmed mean |
| bayes | paired JZS Bayes factor | needs `results_subtitle=False` | paired Bayes factor per pair | MAP |

**`histostats`**, one sample against `test_value`: parametric, one-sample t with
Hedges' g and a Bayes caption (mean); nonparametric, Wilcoxon signed-rank with the
rank-biserial r (median); robust, bootstrap-t test of the trimmed mean, whose
effect size is the trimmed mean with its bootstrap-t interval; bayes, one-sample
JZS Bayes factor (MAP).

**`scatterstats`**, two measurements: parametric, Pearson's r (t test, Fisher z
interval) with a Bayes caption; nonparametric, Spearman's rho (S statistic);
robust, winsorized Pearson r; bayes, the Bayesian Pearson correlation. Always
two-sided.

**`barstats`**, counts: with `y`, Pearson's chi-square test of independence (no
continuity correction) with Cramér's V (bias-corrected), a goodness-of-fit test in
each bar, and Fisher's exact test between every two levels of x in
`result.pairwise` (never drawn); with `paired=True`, McNemar's test (Bowker's
beyond two levels) with Cohen's g; with `y=None`, the goodness-of-fit test against
`ratio` with Pearson's C.

## Results: `StatsResult`

A frozen dataclass of plain Python values. `fig.result` is the result drawn, with
any drawing notes added to `warnings`.

| Field | Content |
| --- | --- |
| `analysis`, `design`, `type` | `"betweenstats"`; `"between"`, `"within"`, `"one-sample"`, `"correlation"` or `"contingency"`; the full type name. |
| `x`, `y`, `subject_id`, `label_var`, `counts_var` | The columns used (`None` where absent). |
| `levels`, `y_levels` | Levels of x (and of y in C) in analysis order. |
| `values` | The analyzed observations: one tuple per level (W: in `subjects` order; H: one tuple; S: x and y). |
| `subjects`, `point_labels` | W: the complete subjects; S: one label per pair. |
| `descriptives` | Per level: `level`, `n`, `centrality` name and `value`. |
| `test` | The test: `name`, `label`, `method`, `statistic_name`, `statistic`, `df`, `df_error`, `p_value`, `alternative`; `None` for type bayes. The sign of each statistic and where its df go: `help(inferplot.result.TestResult)`. |
| `effect_size` | `name`, `label`, `estimate`, `conf_level`, `conf_low`, `conf_high`, `ci_method`, `sidedness`, and for bootstrap intervals `bootstrap_resamples`, `bootstrap_seed`, `undefined_resamples`, `infinite_resamples`; `None` for type bayes. |
| `bayes` | `bf10`, `log_bf01`, `prior`, `prior_scale`, `test`; `None` when not computed. |
| `pairwise` | Per pair: `group1`, `group2`, `statistic_name`, `statistic`, `df`, `p_value`, `p_adjusted`, `adjust_method`, `test`, `test_label`, `significant`, `displayed`, the pair's `estimate` and interval where it has one, `bayes`, `no_test_reason`. |
| `n_obs` | Observations (W: complete subjects; S: complete pairs; C: the table total). |
| `n_dropped`, `dropped_subjects`, `dropped_subject_rows` | Rows dropped for missing values; subjects (and their rows) dropped for missing a level. |
| `n_tied` | Ties among what the rank test ranks (read by `heavy_ties`). |
| `counts`, `bars`, `cells` | C: the table of counts; each bar's n and goodness-of-fit test; each cell's count and percent. |
| `warnings` | Notes in plain English: anything dropped, refused defaults, what v0.1 leaves out. Report them. |
| `provenance` | inferplot, numpy, scipy, matplotlib and narwhals versions, `seed`, parity target, citations. |
| `settings` | The analysis arguments that shaped the numbers, in canonical form. |
| `text` | Every text, plain Unicode (below). |

`text` fields: `apa` (an APA-style results sentence), `subtitle`, `caption`,
`alt_text` (2 to 4 sentences describing the figure, at most 400 characters where
possible), `centrality_labels`, `tick_labels`, `pairwise_labels` (one per pair),
and in C `bar_labels`, `n_labels`, `cell_labels`.

Methods: `to_dict()`; `to_json(indent=2, *, include_values=True)`, strict ASCII
JSON whose first key is `schema_version` (4), infinities as `"Inf"`/`"-Inf"`;
`StatsResult.from_dict()` and `StatsResult.from_json()` rebuild an equal result
and refuse another schema version with `ValueError`; `to_frame(backend="pandas")`
(or `"polars"`), one row per test (main test, then C's bars, then each pair) with
columns `test`, `label`, `group1`, `group2`, `statistic_name`, `statistic`, `df`,
`df_error`, `p_value`, `p_adjusted`, `adjust_method`, `alternative`,
`effect_size`, `estimate`, `conf_level`, `conf_low`, `conf_high`,
`ci_sidedness`, `bf10`, `log_bf01`, `significant`, `displayed`; `check()`, the
statistical issues; `str(result)`, the APA sentence and a table of the pairs.

```python
result = ip.analyze.betweenstats(df, "group", "score")
print(result.test.name, result.effect_size.name, result.effect_size.sidedness)
print([(p.group1, p.group2, p.significant) for p in result.pairwise])
print(result.to_frame()["test"].tolist())
```

```text
welch_anova partial_omega_squared greater
[('ctrl', 'trt1', False), ('ctrl', 'trt2', False), ('trt1', 'trt2', False)]
['welch_anova', 'games_howell', 'games_howell', 'games_howell']
```

## Figures: `StatsFigure`

- `figure`: the matplotlib Figure (inferplot's own, or the one holding `ax`).
- `axes`: read-only mapping of panels: `"main"`, and in S with marginals
  `"xside"` and `"yside"`.
- `result`: the `StatsResult` drawn.
- `check()`: `result.check()` plus the drawing issues; an empty tuple when clean.
- `save(path, *, dpi=300, transparent=False)`: `.svg`, `.pdf` or `.png` (any other
  extension raises). Text stays editable, saving twice gives identical bytes, and
  the result's JSON without the raw data (`values`, `subjects`, `point_labels`) is
  embedded: SVG metadata description, PNG text chunk `Description`, PDF `Subject`.
  `StatsResult.from_json()` reads it back.

Save with `fig.save()`. The figure is not pyplot's current figure:
`plt.savefig()` writes pyplot's own, empty figure, and `fig.figure.savefig()`
writes text as outlines and embeds no result. With `ax=`, `fig.save()` writes the
whole figure and embeds the result of the panel it is called on.

## `check()` codes

Each issue is an `Issue(code, severity, message, fix)`; `str(issue)` reads
`[severity] code: message fix`. Codes are stable.

| Code | Severity | Meaning |
| --- | --- | --- |
| `small_n` | warning | A group, the complete subjects, the sample or the pairs number fewer than 5. |
| `small_expected` | warning | C: expected counts too small for the chi-square approximation (one below 1, or over 20% below 5). |
| `pairs_without_test` | info | C: a pair of levels has no Fisher test; the others are adjusted among themselves. |
| `empty_level_tested` | warning | C: a level of x without observations counts as a category in the bars' drawn tests. |
| `heavy_ties` | warning | Nonparametric: over 25% of the ranked values are ties. |
| `rows_dropped` | info | Rows or subjects were dropped for missing values; a warning when over half were. |
| `pairs_hidden` | info | Some computed pairs have no bracket (`pairwise_display`). |
| `infinite_effect_size` | warning | Robust: the effect size or some bootstrap resamples are infinite; a bound may be Inf. |
| `estimate_outside_interval` | warning | The effect size lies outside its own interval (a bootstrap one, or C: the noncentral chi-square interval of V or C); the p-value is unaffected. |
| `pairwise_conf_level` | info | The pairwise (or robust within) intervals are 95% whatever `conf_level` says. |
| `label_overlap` | warning | Two texts overlap, or a bracket label crosses another bracket. |
| `text_clipped` | warning | A text extends outside the figure or its Axes. |
| `points_subsampled` | info | Over 5,000 points: a seeded subsample is drawn; statistics use all data. |
| `path_hidden` | warning | W: a text covers a subject or centrality path. |
| `paths_crowded` | warning | W: so many subject paths that they hide the plot; pass `point_path=False`. |
| `centrality_label_overlap` | warning | A text covers half of a centrality point. |
| `centrality_label_ambiguous` | warning | A centrality label reads as another level's. |
| `point_labels_crowded` | warning | S: point labels overlap, cross the line, or more than 200 are labelled. |
| `figure_oversized` | warning | inferplot's own figure grew far beyond its panels (a very long text). |
| `cell_labels_hidden` | warning | C: a cell label left out, or drawn where it does not read as its cell's. |
| `level_labels_alike` | warning | B, W, C: two levels drawn with tick labels or legend entries that read alike. |
| `level_colors_alike` | warning | C: two levels of x drawn in colours that cannot be told apart (the legend's only key). |
| `line_label_ambiguous` | warning | H: a line's label is nearer the other line or crossed by it. |
| `low_contrast` | warning | A text or a level colour can hardly be told from its background. |
| `glyph_missing` | warning | Characters no available font can draw. |
| `axis_changed` | warning | An axis was inverted or made non-linear after drawing. |
| `data_outside_view` | warning | Values, levels or brackets lie outside the axis limits. |
| `flat_data` | warning | Values that differ are drawn within 2 pixels of height. |

## Errors

Every message names the fix; follow it literally.

- `TypeError`: an unknown or misspelt keyword (with a did-you-mean when one is very
  likely), an argument from ggstatsplot or seaborn that does not exist here
  (`nboot`, `hue`, `paired` in B, `method` in S), a drawing keyword given to
  `ip.analyze.<name>` or an analysis keyword given to `ip.render.<name>`, a missing
  `subject_id`, data that are not a data frame, a column of the wrong kind.
- `KeyError`: a column the data frame does not have (the message lists the columns).
- `ValueError`: a value outside its range, a setting that would change nothing or
  change what a number means (`tr` with a type that trims nothing, a one-sided
  `alternative` with 3+ levels, `subtitle` with `results_subtitle=True`), data the
  method cannot use (too few values, constant data, a subject repeated within a
  level), a `save()` path that is not .svg, .pdf or .png.
- `NotImplementedError`: what arrives in v0.2. Bayes factors for 3+ groups or
  conditions (`type="bayes"` there needs `results_subtitle=False`;
  `bf_message=True` raises), Bayes factors for counts (C with `type="bayes"` or
  `bf_message=True`), `grouping_var`; also B's robust pairs when a pair has under 2
  degrees of freedom.

inferplot never switches tests, drops arguments or prints. Notes go to
`result.warnings`.

## Bootstraps, Bayes factors and printed p-values

- Bootstrap intervals (robust effect sizes, rank epsilon squared, Kendall's W, the
  bootstrap-t test of H) use 2,000 resamples drawn from `seed`. The result records
  `bootstrap_resamples` and `bootstrap_seed`. ggstatsplot uses 100 resamples, so
  its bounds differ by random noise. Pick the seed before the analysis; do not try
  seeds. [intervals.md](intervals.md) says how far each interval holds its level.
- Bayes factors: `bf10` > 1 favours an effect, < 1 its absence. Subtitles and
  captions print log_e(BF01) = -log(BF10), so a negative value favours an effect.
  The prior is a Cauchy of scale `bf_prior` (0.707; S: a stretched beta, whose
  parameter 1 / `bf_prior` is printed). Bayes factors are two-sided whatever
  `alternative` says.
- A p-value of exactly 0 (an infinite t, or no bootstrap resample as extreme) prints
  as `p = 2.23e-308` in the subtitle, as in ggstatsplot, and as `p < .001` in the
  APA sentence. For a bootstrap p, `warnings` says it shows only that p is below
  about 1 in 2,000.

## Rules for agents

1. State the design: `subject_id=` for repeated measures, `paired=True` for two
   categorical measurements of the same units, `y=None` for one-way counts.
2. Choose `type` from the data and the question before looking at results. Never
   loop over types or settings to find significance.
3. Read `fig.result.text.apa` and `fig.result.to_json()`, not the image.
4. Run `fig.check()`; fix each issue with its `fix`, or report it.
5. Report every entry of `fig.result.warnings`.
6. Save with `fig.save("name.svg")` (or .pdf, .png).
