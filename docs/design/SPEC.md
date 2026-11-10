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
| `tr` | `0.2` | trimming for robust methods (0 <= tr < 0.5); another value with a type that trims nothing raises `ValueError` (ggstatsplot silently ignores it). Type robust with three or more groups raises `NotImplementedError` when a pair has fewer than 2 df (WRS2's lincon stops there); its pairwise intervals are 95% at any `conf_level`, as ggstatsplot's, with a `warnings` entry (D53) |
| `bf_prior` | `0.707` | numeric, exactly as ggstatsplot (not sqrt(2)/2) |
| `bf_message` | `None` | Bayes factor caption. ggstatsplot draws one only when `results_subtitle` is true and either the type is parametric (betweenstats, withinstats, histostats, scatterstats) or, in barstats, the type is not bayes, `paired=False` and `y` is None or has more than one level (ggstatsplot: `y` does not have exactly one level); otherwise it draws none and says nothing. Within that condition `None` draws the caption when v0.1 computes the Bayes factor (two groups or conditions, one sample, correlation) and otherwise adds a `warnings` entry saying it arrives in v0.2 (ANOVA with three or more groups or conditions, contingency tables, one-way goodness of fit); outside it, `None` draws no caption and adds no warning. `True` raises `NotImplementedError` for the v0.2 designs and `ValueError` outside the condition; `False` never draws it. The caption's Bayes factor is two-sided whatever `alternative` says, as ggstatsplot's; with a one-sided `alternative`, `warnings` says so. See §5 item 5 |
| `results_subtitle` | `True` | |
| `alternative` | `"two-sided"` | `"less"`, `"greater"` only for two-group, paired and one-sample tests with type parametric or nonparametric; otherwise `ValueError` (ggstatsplot silently ignores it for robust and bayes types, for correlations, which have no `alternative` in ggscatterstats or `corr_test`, for the Bayes caption and for pairwise comparisons) |
| `seed` | `0` | every random step; recorded in provenance |
| `xlab`, `ylab`, `title` | `None` | user text (see below the table) |
| `subtitle` | `None` | used only when `results_subtitle=False`; passing it with `results_subtitle=True` raises `ValueError` (ggstatsplot silently ignores it there) |
| `caption` | `None` | user text drawn below the generated caption lines |
| `palette` | `None` | list of colors or a matplotlib colormap name; default Okabe-Ito |
| `ax` | `None` | draw into an existing Axes; subtitle and caption go inside that Axes; not allowed with marginal panels. It must be a plain rectangular 2D Axes whose x axis is not shared (each tick label carries that panel's n, so sharing them would show one panel's n under another's groups); both are refused before anything is drawn. A shared y axis gets limits that fit every panel sharing it, at one scale. inferplot's ink is near-black on a light background and near-white on a dark one |

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
| `withinstats` | `data, x, y` | as `betweenstats`, plus **required** `subject_id`, and `point_path=True` (join each subject's points with a line; drawn only with two conditions and the `"points"` layer, as ggwithinstats draws it only for two conditions), `centrality_path=True` (join the centrality points; drawn only with `centrality_plotting=True`, as in ggwithinstats). `point_path=False` where no subject paths are drawn and `centrality_path=False` with `centrality_plotting=False` have no effect and add a `warnings` entry each (D39); the defaults add nothing (D57) |
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

Unknown keyword → `TypeError` with a did-you-mean only when the suggestion is very
likely what was meant; otherwise the message lists the function's arguments. An AI
agent follows a suggestion literally, so an unlikely one is worse than none (D59). Very
likely means, in turn: equal ignoring case and separators (`.`, `-`, `_`, spaces); a
synonym (`trim`, `trimming` → `tr`; `conf`, `confidence`, `confidence_level`,
`ci_level`, `level` → `conf_level`; `decimals` → `digits`; `prior` → `bf_prior`;
`display` → `pairwise_display`; `axes` → `ax`); a misspelling within one edit for names
of 4 to 6 letters or two for longer names, where swapping two neighbouring letters
counts as one edit; the name plus up to 4 letters, not a separate word (`seeds`,
`bayesian`, `boxplot`); or the start of exactly one name, either its first two or more
whole parts (`p_adjust`) or at least 5 letters ending inside a part (`param`,
`centrality_plot`). The same rule applies to the values of `type`, `pairwise_display`,
`p_adjust_method`, `alternative` and `layers`, and to matplotlib property names in
`point_args`, `boxplot_args` and `violin_args`. Column names keep a case-insensitive
match first, then difflib with cutoff 0.6.
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
| `paired` | withinstats: not needed; measurements are always paired by `subject_id`. betweenstats: the groups are independent; for repeated measures use `withinstats(data, x, y, subject_id=...)` |
| `tail`, `tails`, `sided`, `sides`, `alternative_hypothesis`, `one_sided` | use `alternative=`: 'two-sided' (default), 'greater' (the first level of x is greater than the second), 'less'; the first level is the first in `order=`, else the first declared category, else the first sorted |
| `alpha` | `pairwise_alpha=` sets the threshold for the brackets, `conf_level=` the interval level; for transparency `point_args={'alpha': 0.5}`, `boxplot_args`, `violin_args` (where the function draws) |
| `levels`, `categories`, `x_order`, `level_order` | use `order=` (the given list as the example) |
| `legend` | there is no legend; colors follow the levels of x |
| `bayes` | `type='bayes'`, or `bf_message=` for the caption |
| `label`, `labels` | `xlab=`, `ylab=`, `title=` |
| `point_path`, `centrality_path` | betweenstats: arguments of withinstats; independent groups have no paths |
| `centrality.path.args`, `point.path.args` | withinstats: no equivalent; `centrality_path` and `point_path` switch the paths |

Other ggbetweenstats 1.1.4 arguments with no inferplot equivalent (`centrality.type`,
`centrality.point.args`, `centrality.label.args`, `ggsignif.args`, `ggtheme`,
`ggplot.component`, dotted or snake_case) raise `TypeError` saying so and why, never a
did-you-mean. A keyword whose snake_case form (dots, hyphens and camelCase turned into
lowercase underscores) is a valid argument raises `TypeError` naming that spelling.
`subject`, `id`, `subject.id` point to `subject_id` in functions that have it; in
`betweenstats` they, and `subject_id`, say the groups are independent and point to
`withinstats(..., subject_id=...)`. `ip.render.withinstats` says the subject column is an
argument of the analysis (`subject_id=`). An extra positional argument to a
function with `subject_id`, when `subject_id=` is not given, gets the hint
`subject_id='<value>'` whatever the value (a non-string gets `subject_id='<its name>'`
or `'subject'`); a value that is also a valid `type` or `pairwise_display` adds that
option after the subject hint. For each row of the table, a function that lacks the
named argument points to the analysis function or the drawing function that has it. When several keywords are wrong, the first in call
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

Values (D59, D60). `p_adjust_method`: other names of R's methods get R's spelling,
ignoring case and separators (`benjamini-hochberg`, `fdr_bh`, `b-h` → 'BH';
`benjamini-yekutieli`, `fdr_by`, `b-y` → 'BY'; `holm-bonferroni`, `bonferroni-holm` →
'holm'); `sidak`, `holm-sidak`, `tukey` and `tukey-hsd` are said to be unavailable.
`alternative`: `one-sided` and its spellings (`one.sided`, `onesided`, `one-side`,
`one-tailed`) say that the direction must be given and that 'greater' tests the first
level of x against the second; `two-tailed` → 'two-sided'. `conf_level` at or above 1:
only 90, 95, 99 and 99.9 are told the value divided by 100; other values (5, 1.96) get
the rule alone. `digits` is 0 to 15. A NumPy float32 or float16 number is read as the
number it prints as (`np.float32(0.9)` is 0.9). A column argument that is not a string
is named by its type, never printed; a frame whose column labels are not strings is
told to use `data.rename(columns=str)` or to name the columns. Duplicate column labels
raise `ValueError` naming the label.

## 3. Data handling

- Accept any eager frame `narwhals.from_native(data, eager_only=True)` accepts (pandas,
  polars, pyarrow). Convert used columns to float64/str NumPy arrays at one boundary
  (`_data.py`); nothing downstream sees a dataframe.
- Missing column → `KeyError` listing available columns and the closest match.
- `y` must be numeric (boolean rejected). `x` may be text, categorical, integer, or
  float when every value is whole (boolean rejected) (D35). Dates, times, durations
  and pandas Intervals (`pandas.cut`, `pandas.qcut`), also as a categorical, raise
  `TypeError` saying to convert them to text (`.astype(str)`; for bins,
  `cat.rename_categories(str)` keeps their order). A non-numeric `y` next to a numeric
  `x` says that x and y may be swapped.
- Rows with missing values in used columns are dropped and counted (`n_dropped`).
  An infinite `y` in a complete row raises `ValueError`, and so does an integer `y`
  above 2^53 that a double cannot hold exactly (it would be rounded into ties; the
  message says to subtract an offset). At least two levels need data.
- Level order (D35, D58): the declared order of a pandas `Categorical`, a polars
  `Enum`, or a pyarrow dictionary whose type is marked `ordered=True`; otherwise the
  sorted unique values. A polars `Categorical` counts as text, because its category
  order changed within polars 1.x. A pyarrow dictionary not marked ordered counts as
  its values (text sorted, numbers numerically, floats whole), because
  `dictionary_encode()`, `pyarrow.csv` with `auto_dict_encode` and polars `.to_arrow()`
  list values in order of first appearance. When such a dictionary's order differs
  from the sorted levels and `order=` is not given, `warnings` names both orders
  (shortened after 6 levels) and says that `order=` or an ordered dictionary sets the
  order. A pandas `Categorical` declares its order whether or not it is marked ordered,
  as pandas, seaborn and R read it; a categorical whose categories were built in row
  order (polars `.to_pandas()` of a Categorical, pyarrow `.to_pandas()` of an unordered
  dictionary) cannot be told apart from a declared one, so it follows the row order.
  `order=` must be a permutation of the present levels. It is any ordered collection other than a string (list, tuple, NumPy
  array, pandas Index or Series, polars Series, pyarrow array, generator) of level
  names or of the values `x` holds: any integer (Python or NumPy), any finite float
  (8.0 names level "8") or a pyarrow scalar. A set or a mapping raises `TypeError`
  (no order, or an ambiguous one). Empty declared levels are dropped before analysis and listed in
  `warnings`; they must never suppress results (ggstatsplot 1.1.4 fixed a bug where
  they did). A level of an `x` without declared categories whose rows all miss `y` (in
  withinstats `y` or `subject_id`) is also dropped and named in `warnings` (D60); with
  fewer than two levels left, the usual error applies. Levels that differ only in
  leading or trailing whitespace stay separate groups and are named in `warnings`.
- Parametric type (Welch's t and ANOVA, Games-Howell): every level needs n >= 2, two
  groups may not both be constant, and with three or more groups every level needs a
  variance above 0. The error names the levels (D35). A group that varies by less
  than double precision can resolve next to the largest |y| (a standard deviation
  below roughly 1e-154 times it; §4, "Scale and range") raises too, naming the
  levels; with two groups this happens only when the group with the largest |y| is
  constant. A group whose values span more than the largest double raises.
- `withinstats` (D56): pivot `subject_id × x`. `subject_id` follows `x`'s type rules and
  may not be `x` or `y`. Rows with a missing `x`, `y` or `subject_id` are dropped and
  counted in `n_dropped` (rows). A subject with more than one row at a level raises,
  naming the first repeats; the check covers every row with a subject and a level, also
  one whose `y` is missing. An infinite `y` in a row with all three columns present
  raises, also when its subject is left out later. Subjects without a value at every
  level are removed from statistics **and** the plot and counted in `dropped_subjects`
  (subjects; ggstatsplot 1.1.3 behavior), including a subject whose `y` values are all
  missing. Fewer than two complete subjects raise. The analyzed data are a
  subjects-by-levels matrix whose rows follow the subject column's order (declared
  categories, else sorted: text by code point, numbers numerically), so results do not
  depend on the input row order or the data frame library, except when the subject
  column or `x` is a pandas categorical whose categories were built in row order (see
  Level order). An unordered pyarrow dictionary `subject_id` is sorted.
- More than 5,000 points in one panel: draw a density layer plus a seeded subsample of
  the points, stratified by level: level g of n_g values (N in all) gets
  max(round(2000 n_g / N), min(n_g, 50)) points, its smallest and largest value always
  among them, so every level has marks and every extreme is drawn (D63); statistics
  always use all data; record it in `warnings`. In `withinstats` whole subjects are
  sampled: every value of a seeded random `2000 // k` of the subjects (1,000 at two
  conditions, 666 at three), always including the subjects that hold each condition's
  smallest and largest value, so every drawn subject path is complete; `warnings` and
  `points_subsampled` name the subjects drawn (D57).

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
| robust | Yuen, 20% trim (`WRS2::yuen`): t is reported unsigned, \|t\|, even when group 1's trimmed mean is the smaller (mtcars `am`, `edge_k2`), while the AKP effect size keeps its sign; WRS2 computes p as 2 (1 - pt(|t|, df)) (probe-wrs2p: all 84 yuen p-values equal it bit for bit; iris_no_setosa-petal: 5.77e-15 where the exact tail is 5.93e-15); inferplot reports the accurate tail, 2 P(T > |t|), which agrees within that rounding and stays positive where 1 - cdf is 0 (D43; §10, `wrs2_p`). `alternative` has no effect (ToothGrowth-greater-robust equals the two-sided case) | AKP robust standardized difference, `WRS2::akp.effect(EQVAR = FALSE)`: trimmed-mean difference scaled by group 1's winsorized SD, not the pooled SD (oracle: 0.6827 vs pooled 0.6393 on ToothGrowth); percentile bootstrap CI: each resample draws group 1 then group 2 with replacement at full size; the bounds are the (floor(alpha/2 B) + 1)-th and floor((1 - alpha/2) B)-th smallest of B values (alpha = 1 - `conf_level`; ggstatsplot: 100 resamples, inferplot 2,000), proved by replaying ggstatsplot's own 100-resample values exactly. AKP = c(tr) x trimmed-mean difference / group 1's winsorized SD, with c(tr) the square root of the winsorized normal variance: WRS2 evaluates it numerically, 1.2e-9 relative above the exact value at tr 0.2, 4.6e-9 at 0.1 and up to 4.2e-7 at 0.49 (probe-akp); inferplot uses the exact value (D42) | trimmed mean |
| bayes | JZS Bayes factor, r = 0.707 (`BayesFactor::ttestBF(formula, rscale = bf.prior)`) on Student's (pooled) t, two-sided whatever `alternative` says. `tools/reference`'s one-dimensional quadrature reproduces 75 of the 77 distinct two-sample values in the fixtures (subtitles, parametric captions, pairs) to 2.3e-9; BayesFactor misses the other two (iris Petal.Length setosa-versicolor and setosa-virginica, t -39.5 and -50.0) by 1.2e-3 and 6.1e-5, where inferplot reports the accurate value (D50, §5 item 10); Welch's t differs on every unbalanced pair, by 1.5e-4 or more | v0.2: posterior median of the mean difference with a 95% equal-tailed interval (`conf.method` "ETI" in the fixtures; 4,000 posterior draws per `parameters::model_parameters()`'s help page, to confirm) | MAP (`bayestestR::map_estimate`): Gaussian kernel density with `bw.SJ` (Sheather & Jones 1991, solve-the-equation; pair distances binned in 1,000 bins of 1.01 x range / 1000 anchored at zero, root by Brent's zero finder at tol 0.1 x lower) by `stats::density` (linear binning on 1,024 points from min - 4 bw to max + 4 bw, FFT, linear interpolation), read at 1,024 points from min(x) to max(x); the first point with the largest density (probe-map: 49 of its 52 vectors on R's grid point or the same double, among them negative values, data spanning zero, values near 1e8 and 1.7e9, n up to 1e5 and tight clusters; normal_n1e5 is one step away, where R's two densities differ by 6.2e-15; for values near 1e8 and above R anchors the bins at min(x), so its bandwidth differs by up to 9.6e-4 relative, the MAP not). Constant data: the value itself. Where bw.SJ fails on data with spread (zero IQR), the group's MAP is `None`, no label is drawn and `warnings` says why; ggstatsplot prints an empty label (D55, §5 item 12). Fewer than 2 values raise `ValueError`. bw.SJ's pair distances are binned from zero with truncation toward zero (pinned by probe-map's 17 vectors that span zero; floor misses by up to 0.53 relative), and from min(x) once max\|x\| / bin width reaches 2^31 (probe-map's two far vectors, 2.4e10 and 7.5e11 bin widths, match to 1e-14; the threshold between 2.8e4 and 2.4e10 is inferred, likely R's integer bin index) |

### betweenstats, three or more groups

| type | Omnibus | Effect size + CI | Pairwise, then `p_adjust_method` | Centrality |
| --- | --- | --- | --- | --- |
| parametric | Welch ANOVA (`stats::oneway.test`) | partial omega squared = max(0, (F - 1) df1 / (F df1 + df2 + 1)) with Welch's F and df (`effectsize::omega_squared`), one-sided CI [lower, 1] at `conf_level`: the estimate read as a partial eta squared gives F_w = omega df2 / ((1 - omega) df1), lambda solves P(F'(df1, df2, lambda) <= F_w) = `conf_level`, the lower bound is lambda / (lambda + df2), and 0 when no lambda > 0 does (D33, probe-omega). inferplot computes the noncentral F itself (SciPy 1.11's `ncfdtr` is off by up to 1.2 relative near P = 1) | Games-Howell (`PMCMRplus::gamesHowellTest`) **then Holm on top**; q is signed, mean(group2) - mean(group1); group1 is the earlier level in the analysis order, and rows are sorted by group1 and then group2 as text in the C locale (statsExpressions 2.2.1 `pairwise_comparisons()`: `dplyr::arrange(group1, group2)`), the order the brackets keep. p is the accurate studentized range upper tail at the pair's Welch df and \|q\| (`inferplot.stats.srange_sf`, D31), then `p_adjust_method` over all k(k-1)/2 pairs (ggstatsplot's adjusted p is exactly `p.adjust` of gamesHowellTest's raw p). ggstatsplot's p is R's `ptukey` value, whose error grows as the df falls (probe-ptukey: up to 1.35e-3 absolute at df 2, 1.8e-7 at df 10, 1.9e-8 at df 20 to 1000, a floor near 4.7e-10 in the far tail) and which is NaN below 2 df, where ggstatsplot stops with an error (fixture `edge_n2`, D37). SciPy's `studentized_range` is no ground truth in the far tail either (1.9e-14 for iris setosa-versicolor, 14% low) | mean |
| nonparametric | Kruskal-Wallis (`stats::kruskal.test`) | rank epsilon squared = H / (N - 1) (`effectsize::rank_epsilon_squared`; Tomczak & Tomczak 2014), one-sided percentile bootstrap CI [lower, 1]: each resample draws n_g values with replacement within every group (group sizes fixed; whole-row resampling misses the oracle's 20,000-resample references by up to about 220 replicate SDs); the lower bound is the bootstrap (1 - conf_level) quantile with `boot::boot.ci`'s percentile rule, order statistic (R + 1) alpha interpolated on the normal quantile scale (Davison & Hinkley 1997, ch. 5); ggstatsplot: 100 iterations, inferplot: 2,000 seeded, drawn from each group's sorted values with groups in content order, so row order and level order do not change it. A resample with every value tied has no epsilon squared: it is left out and the quantile taken over the others, counted in `undefined_resamples` with a `warnings` entry (D54; edge_ties_small) | Dunn (`PMCMRplus::kwAllPairsDunnTest`): z = abs(mean-rank difference) / tie-corrected SE, reported unsigned; two-sided normal p (reproduced to 2.4e-15); inferplot computes raw p for i < j in input order and adjusts with `p_adjust` | median |
| robust | trimmed-means ANOVA (`WRS2::t1way`); WRS2 computes p as 1 - pf, inferplot the accurate upper tail (D43; §10, `wrs2_p`) | explanatory effect size (`WRS2::t1way`; Wilcox & Tian 2011): xi = sqrt(var of the trimmed means, J - 1 denominator, / (pooled winsorized variance / v_w)), v_w the winsorized standard normal variance (exact; WRS2 divides by 0.4120867, 3.2e-8 relative in xi, D42). Bootstrap: every group resampled with replacement to the smallest group's size; bounds the (l + 1)-th and (B - l)-th smallest, l = round(alpha/2 B), alpha = 1 - `conf.level` computed in floating point (0.050000000000000044, so alpha/2 x 100 = 2.5000000000000022 rounds to 3); with unequal group sizes the point estimate is the median of the bootstrap values, so it is itself random (oracle replicate SD 0.0048 on mtcars) and compared within Monte Carlo error. Proved by replaying ggstatsplot's own 100-resample values and the 20,000-resample references exactly | `WRS2::lincon`: estimate = trimmed mean(group1) - trimmed mean(group2); p = per-pair Yuen p (1 - pt), then `p_adjust_method` (lincon's own Hochberg is not used); CI = estimate -/+ c x Yuen SE. c comes from a two-decimal table of the Studentized maximum modulus at (alpha, C = number of pairs, not groups: `edge_k5` has 10, `bugs_long` 6), whose rows are every df from 2 to 12, then 14, 16, 18, 20, 24, 30, 40, 60 and a last row at df 200 holding the infinite-df (normal) values; c is interpolated linearly in 1/df between the rows on either side of the pair's Yuen df (fixtures: df 5.2 to 110, rows up to 60/200; reproduced to 1e-14). The entries are WRS2's published ones, read from probe-smm (`tools/reference/smm_table.py`): 24 of its alpha-0.05 entries are one unit off the rounded quantile (20 low, 4 high), 8 of them at C values betweenstats uses, and they are kept; C = 9 at df 2 is 8.38 where WRS2 has 8.83, a transposition of the quantile 8.3846 (D51). C = 1 is Student's t quantile at the pair's df. From df 200 on, the last row (edge_large_n: df about 1,490). Below df 2 lincon stops with an error (edge_n2-robust), and inferplot raises `NotImplementedError`. Above 28 comparisons (nine or more groups) WRS2 simulates c with a seed of its own; inferplot uses the exact Studentized maximum modulus quantile at the pair's df (`inferplot.stats.smm_quantile`; D52, §5 item 11). ggstatsplot calls lincon at its default alpha 0.05 whatever `conf_level` says (PlantGrowth-cl90-robust: every pairwise row says 0.95), so the pairwise intervals are always 95%: each pair records `conf_level` 0.95, `warnings` says so and `check()` reports `pairwise_conf_level` (D53) | trimmed mean |
| bayes | v0.2: `BayesFactor::anovaBF` (method and precision to measure when implemented; `bf_prior` reaches it, PlantGrowth-prior08-bayes: BF10 3.48 at 0.8 against 3.66 at 0.707) | v0.2: Bayesian R squared, posterior median with 95% HDI (`conf.method` "HDI" in the fixtures) | v0.2: `ttestBF` per pair as for two groups, unadjusted, always with prior 0.707 and 95% intervals whatever `bf_prior` and `conf_level` say (every pair's Bayes factor is reproduced by `tools/reference`, the two of D50 as accurate values); `pairwise_display` has no effect except "none" (every pair is drawn, D48) and `alternative` none at all (ToothGrowth-greater-bayes) | MAP |

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
takes at most about 0.25 s at any F (DECISIONS log). Beta factors with one
parameter at least 1e4 and at least 10 times the other are computed in log space by
quadrature, and incomplete beta values below 1e-250 by a continued fraction in log
space, so the distribution function keeps relative accuracy for tails >= 1e-20 at any
ncp up to 1e300 and df2 up to at least 1e8, and its log far below the doubles (tested
to log P = -2000); a NaN is never returned as a probability.

**Differences of means.** Every difference of means or trimmed means, and every
deviation from a weighted centre, is formed from the exact means and rounded once
(D49), in the parametric and robust statistics alike, so groups that differ by about
the rounding of their common level (timestamps, values near 2^50, last-bit
differences) get the correct t, F and q. Welch's t, df and p in betweenstats do not
depend on the magnitude of y, also when the difference of the means exceeds the
largest double.

### withinstats, two conditions

Read from the Phase 3 fixtures and probes (oracle run 42780911147). n is the number
of complete subjects; the subtitle prints it as n_pairs.

| type | Test | Effect size + CI | Centrality |
| --- | --- | --- | --- |
| parametric | paired t (`stats::t.test`, paired), t signed as first level minus second, df n - 1 | Hedges' g of the differences: d_z x J(n - 1) with d_z = mean(d) / sd(d) (not d_av, which misses every fixture). CI: the noncentral-t interval at the paired t and n - 1 df, each bound divided by sqrt(n) and multiplied by J; it follows `alternative` (sleep-less: [-Inf, -0.50]) | mean |
| nonparametric | Wilcoxon signed-rank (`stats::wilcox.test`, paired), always `exact = FALSE, correct = TRUE`: zero differences dropped, normal approximation with tie and continuity corrections, V reported (R's own default would be exact for small samples; probe-signrank) | matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-) over the nonzero differences. CI analytic, not bootstrapped: tanh(atanh r -/+ z SE), SE = sqrt(n (n + 1) (2n + 1) / 6) / (n (n + 1) / 2) with n counting every pair, zeros included; \|r\| = 1 gives [r, r] (and a `warnings` entry that the interval has no width); follows `alternative`. All-zero differences raise (R: p NaN) | median |
| robust | Yuen for dependent samples (`WRS2::yuend`): t signed (unlike `yuen`'s \|t\|), df h - 1, p = 2 (1 - pt) in R (probe-wrs2within, 84 of 84 rows); inferplot reports the accurate tail (D43) | `WRS2::dep.effect`, AKP row (statsExpressions 2.2.1 calls it for paired robust tests, not `wmcpAKP`); percentile bootstrap (ggstatsplot: 100 resamples), always a 95% interval: `conf.level` does not reach it (sleep-cl90-robust prints CI 95%); bounds can be infinite; 1.1.3 fixed this value | trimmed mean |
| bayes | paired JZS Bayes factor (`ttestBF(x, y, paired = TRUE)`): the one-sample Bayes factor of the differences. BayesFactor misses the accurate value at large \|t\| (probe-pairedbf); inferplot reports the accurate one (D61) | posterior difference: v0.2 | MAP |

`alternative` has no effect on the robust and bayes types in ggstatsplot (sleep-greater);
inferplot raises for them, as in betweenstats (section 2).

### withinstats, three or more conditions

| type | Omnibus | Effect size + CI | Pairwise, then `p_adjust_method` | Centrality |
| --- | --- | --- | --- | --- |
| parametric | one-way repeated-measures ANOVA (`afex::aov_ez`) with the Greenhouse-Geisser correction, always (never Huynh-Feldt): corrected df and p; the subtitle prints the corrected df with 2 decimals (bugs_long: F(2.63, 228.95)) | partial omega squared from the error strata, df_c (MS_c - MS_e) / (SS_c + SS_e + SS_s + MS_s) (not `F_to_omega2`: 0.078 against 0.18 on bugs_long); one-sided CI [low, 1] through the noncentral F at the uncorrected df, although the subtitle shows the corrected df: F_w = omega df2 / ((1 - omega) df1) at df1 = k - 1, df2 = (k - 1)(n - 1), lambda with P(F' <= F_w) = `conf.level`, lower bound lambda / (lambda + df2) (bugs_long, cl90, probe-rmanova). A negative estimate is reported as 0 with [0, 1]. Epsilon lies in [1/(k - 1), 1] (D62) | paired t (`pairwise.t.test(paired = TRUE)`): each pair's own differences, df n - 1, t signed as group1 minus group2, R's label "Student's t" | mean |
| nonparametric | Friedman (`stats::friedman.test`), tie-corrected chi-square on k - 1 df | Kendall's W = chi-square / (n (k - 1)) (`effectsize::kendalls_w`); bootstrap CI, one-sided [low, 1] at `conf.level` (ggstatsplot: 100 resamples): each resample draws n whole subjects; inferplot uses 2,000 seeded resamples in a canonical order of subjects and conditions and `boot.ci`'s percentile rule. When every subject ranks the conditions alike W = 1, R's bootstrap fails and ggstatsplot drops the subtitle silently (edge_within_small); inferplot reports W = 1 with [1, 1] and a `warnings` entry (D64) | Durbin-Conover (`PMCMRplus::durbinAllPairsTest`; Conover 1999, complete blocks): \|t\| = \|R_i - R_j\| / sqrt(2 (n A - sum R_j^2) / ((n - 1)(k - 1))) with A the sum of squared ranks, reported unsigned (121 fixture rows, 11 probe designs), two-sided on (n - 1)(k - 1) df; W = 1 gives t = Inf, p = 0, which inferplot reports with a `warnings` entry (D64) | median |
| robust | trimmed-means RM ANOVA (`WRS2::rmanova`): fractional df when its sphericity correction applies, p = 1 - pf in R (84 of 84 probe rows; inferplot reports the accurate tail, D43) | average AKP (`WRS2::wmcpAKP`), percentile bootstrap, always a 95% interval (`conf.level` does not reach it; bugs_long-cl90-robust) | `WRS2::rmmcp`: per pair the trimmed mean of the differences, a one-sample trimmed t on n - 2g - 1 df, p = 2 (1 - pt), then `p_adjust_method` (rmmcp's own `p.crit` is in its table but ggstatsplot does not use it); its interval uses the Bonferroni t at alpha / (2C) | trimmed mean |
| bayes | v0.2 (a Bayesian RM ANOVA with a random subject effect, `anovaBF`) | v0.2 | paired JZS Bayes factor per pair, prior 0.707 whatever `bf_prior` is (bugs_long-prior08), as in betweenstats | MAP |

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
  doubles just below 1 and 0 below about 1e-16 (probe-wrs2p, 420 of 420); ggstatsplot
  stores a t1way p of exactly 0 as 2.23e-308 and prints a Holm-adjusted lincon p of 0
  as "0.00" (iris-petal-robust); inferplot reports the accurate tail (D43, §5 item 9).
  lincon's CI uses WRS2's published two-decimal table, 24 of whose alpha-0.05 entries
  are one unit off the rounded quantile (D51), always at 95% (D53).
- An AKP or xi resample (or, for xi with equal sizes, the data) whose winsorized
  variance is 0 while the trimmed means differ gives ±inf; inferplot keeps the value
  (WRS2's formulas give the same; no fixture has one), counts it in
  `infinite_resamples`, warns and reports `infinite_effect_size`. A 0/0 resample
  raises `ValueError`.
- `alternative` has no effect on the robust and bayes types (ToothGrowth-greater-*).
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
9. WRS2's p-values (Yuen, t1way, lincon) are reported as the accurate upper tail, not
   1 - cdf (D43). Where 1 - cdf rounds visibly the printed p differs: iris-petal-robust
   (t1way 1.36e-42 where ggstatsplot prints 2.23e-308; setosa-versicolor and
   setosa-virginica 7.28e-30 and 9.12e-32 where it prints 0.00; versicolor-virginica
   5.93e-15 against 5.77e-15) and iris_no_setosa (5.93e-15 against 5.77e-15, 5.43e-16
   against 4.44e-16).
10. Pairwise Bayes factors are accurate where BayesFactor's quadrature misses (D50):
    iris Petal.Length setosa-versicolor (+1.16e-3) and setosa-virginica (-6.1e-5); one
    label differs, log_e BF01 -133.68 where ggstatsplot prints -133.69.
11. With nine or more groups (more than 28 pairs) the robust pairwise intervals use the
    exact Studentized maximum modulus quantile, the distribution WRS2's table holds
    up to 28 pairs. Above 28, WRS2 uses a seeded simulation that differs from it by
    -6.6% to +4.7%: its values follow a one-sided critical value (the quantile of
    the largest of C signed t variates, within 1% at alpha 0.05 and 0.1), so its
    intervals there are mostly narrower than simultaneous two-sided ones (D52).
    Estimates and p-values are unchanged.
12. A group whose MAP has no density estimate (bw.SJ fails on zero IQR) gets no
    centrality label and a `warnings` entry; ggstatsplot draws an empty label (D55).
13. Differences of means are exact up to one rounding (D49), where R subtracts rounded
    means; the fixtures agree within 1.5e-13, but data whose groups differ by about
    the rounding of their level (0.1 + 0.2 against 0.3) get the correct statistic.
14. `type="robust"` refuses a `conf_level` so close to 1 that the percentile rule
    would return the most extreme bootstrap resamples at any level (above 0.999 with
    two groups, from 0.9995 with three or more, at 2,000 resamples): the interval
    would stop widening while still being labelled with the level asked for. WRS2
    returns those bounds, and with ggstatsplot's 100 resamples it already does so
    from 0.98 (two groups) and 0.99 (three or more). Rank epsilon squared refuses in
    the same way.
15. Hedges' g with Welch's df between 1 and 2 (one group of two values holding most
    of the variance) gets a `warnings` entry: the small-sample correction is then
    between 0 and 0.56, so g is far below the uncorrected standardized difference.
    The number is effectsize's; at df exactly 1 it is 0, with its own warning.

Visual styling is not a parity target: same semantic layers (points, box, violin,
centrality label, brackets, subtitle content, caption content), our own look. Default
palette Okabe-Ito; more than 8 levels uses `tab20` and adds a warning. Violins are
Gaussian kernel densities with Silverman's bandwidth (R's `bw.nrd0`) trimmed to the
data range; boxes use type-7 quartiles and Tukey whiskers; values beyond the whiskers
are shown by the points or, without a points layer, by small open circles at the level
(D63). A panel with more than 5,000 values draws a stratified seeded subsample of the
points (section 3); the violins (or, without a violin layer, unfilled density outlines)
and boxes show all values. Where the kernel bandwidth is narrower than the spacing of
the density grid (one value far from thousands of others), each grid point gets the
density averaged over its cell. A level with fewer than two distinct values gets no
violin, and a violin that cannot be drawn is named in `warnings`, never claimed as
shown. A continuous colormap given as `palette` is sampled at (i + 0.5) / k, away from
its near-white end; a listed colormap gives its colours in order. Centrality labels
never cover a centrality point: each takes the first of its usual place, above or
below the point's height on that side, or the same three on the other side, that
covers no centrality point and no other centrality label. Tick labels wrap long level
names on spaces into at most three lines of the narrowest level width (1.1 in) and
shorten what does not fit with "…"; the result keeps the full name. withinstats (D57) jitters each subject's
points by one seeded offset shared across its levels (ggwithinstats draws them
unjittered), so a subject path ends on its points; subject paths are thin translucent
grey lines in one collection, under the points; the centrality path is drawn in the
centrality colour. Centrality labels avoid the paths: with two conditions the first
level's label is left of its point, and with three or more a middle level's label sits
above or below the point's height on the side the path to the next level leaves free.
inferplot sizes its own figure from the
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
| `values` | per level, the analyzed `y` values in data order (betweenstats) or in `subjects` order (withinstats: entry i of every level belongs to `subjects[i]`); the renderer draws from them (D36) |
| `subjects` | withinstats: labels of the complete subjects, in matrix row order (D56); empty for betweenstats |
| `descriptives` | per level: `n`, centrality name (`mean`, `median`, `trimmed_mean`, `map`) and value |
| `test` | name, R-style method string, statistic name and value, `df`, `df_error`, p-value, alternative; `None` for type bayes (D45) |
| `effect_size` | name, estimate, `conf_level`, CI bounds, CI method, sidedness, `bootstrap_resamples`, `bootstrap_seed`, `undefined_resamples` (resamples left out, D54) and `infinite_resamples` (resamples whose effect size is ±inf, kept) (`None` without resampling); `None` for type bayes (D45) |
| `bayes` | `bf10`, `log_bf01`, prior name and scale; `None` when absent |
| `pairwise` | per pair: groups, statistic name and value, df, p and adjusted p (`None` for Bayes pairs), adjust method ("none" for Bayes pairs), test name and label (ggstatsplot's `test` column: "Games-Howell", "Dunn", "Yuen's trimmed means", "Student's t"), significant (`None` for Bayes pairs), displayed, `estimate`, `conf_level`, `conf_low`, `conf_high` (lincon; else `None`) and `bayes` (a Bayes factor for Bayes pairs; else `None`) (D45, D46) |
| `n_obs`, `n_dropped`, `dropped_subjects` | `n_obs`: observations (betweenstats) or complete subjects (withinstats; to confirm against ggstatsplot's `n.obs` when the first ggwithinstats fixture arrives); `n_dropped`: rows with a missing value in a used column; `dropped_subjects`: subjects left out for lacking a level |
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
conf_high, bf10, log_bf01, significant, displayed; missing values are nulls. Type bayes:
the main row is the two-group Bayes factor, and with three or more groups there is none;
pair rows carry lincon's estimate and CI and a Bayes pair's bf10 and log_bf01. In a
field that may be `None`, JSON null reads back as `None` (pandas uses
nullable `boolean` columns).

`save()` sets `svg.fonttype="none"`, `pdf.fonttype=42` and a fixed `svg.hashsalt`
inside an `rc_context`, drops the SVG date and PDF creation date (saving twice writes
identical bytes), and embeds `result.to_json(include_values=False)` in file metadata:
the PNG text chunk "Description", the SVG Dublin Core description and the PDF Info
"Subject". The embedded copy holds every number and text but not the analyzed
observations or the subject labels, which can be large and are the user's data (D36,
D56); a result read back from it has `values == ()` and `subjects == ()` and cannot be
drawn again. Text is drawn in DejaVu Sans, then the installed fonts of the user's
`font.sans-serif` list that matplotlib's default list does not hold (so a configured
CJK font supplies its glyphs), then the generic `sans-serif`. An SVG looks as drawn
only where DejaVu Sans, and the STIX fonts mathtext uses for accents such as hats, are
installed (macOS and Windows do not ship them): elsewhere the viewer substitutes fonts
and an accent can land beside its letter. Its mathtext is written one positioned
character per `<tspan>`, so the subtitle is not searchable or editable as running
text. PDF embeds its fonts and is the faithful vector format. `save()` accepts a dpi up
to 2,400 and a PNG of up to 100 million pixels. The figure is not pyplot's current
figure, so `plt.savefig()` after an inferplot call writes pyplot's own, empty figure;
`fig.figure.savefig()` works but writes text as paths and embeds no result.

Drawing arguments (D63): `point_args`, `boxplot_args` and `violin_args` refuse
properties that carry or move the data or detach the artist (`offsets`, `paths`,
`path`, `xy`, `verts`, `segments`, `data`, `xdata`, `ydata`, `transform`,
`offset_transform`, `figure`, `axes`, `clip_path`, `clip_box`, `animated`) with a
`ValueError` naming the property and why; a dictionary that makes its layer invisible
(`visible=False`, `alpha=0`, zero size) adds a `warnings` entry (D39). Every `palette`
entry is validated, not only the first k.

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
  unadjusted" for none. The test name is ggstatsplot's `test` column: "Pairwise: Dunn,
  Holm adjusted; ...", "Pairwise: Yuen's trimmed means, Holm adjusted; ...", and for
  type bayes (D41, D48) "Pairwise: Bayes factors (Student's t, JZS prior 0.707),
  unadjusted; showing all <m> pairs." (ggstatsplot draws no pairwise caption; these
  lines are inferplot's own, §5 item 3). Bracket labels keep ggstatsplot's `p.adjust.method` labels
  (Holm, Bonferroni, Hochberg, Hommel, FDR for BH and fdr, BY, and `p_unadj.`).
- **APA sentence (APA 7):** "Welch's t-test: t(9.62) = -2.35, p = .041, Hedges' g =
  -1.07, 95% CI [-2.01, -0.08], n = 12."; "Welch's ANOVA: F(2, 17.13) = 5.18, p = .017,
  partial omega squared = .29, 95% CI [.00, 1.00], n = 30." Two decimals for statistics;
  p to three decimals without leading zero, "p < .001" below .001; no leading zero for
  quantities bounded by 1 (omega and eta squared, epsilon squared, correlations,
  rank-biserial r, Cramér's V, Kendall's W); non-integer df to two decimals, integer df
  bare; a one-sided test reads "Welch's t-test (one-sided, greater): ..."; ASCII only.
  The other types: "Mann-Whitney U test: W = 575.50, p = .064, rank-biserial r = .28,
  95% CI [-.01, .52], n = 60."; "Kruskal-Wallis test: chi2(2) = 7.99, p = .018, rank
  epsilon squared = .28, 95% CI [.10, 1.00], n = 30."; "Yuen's t-test (20% trimmed
  means): t(33.49) = 2.29, p = .029, AKP robust standardized difference = 0.68, 95% CI
  [0.04, 2.00], n = 60."; "Trimmed-means ANOVA (20% trimmed means): F(2, 9.64) = 8.28,
  p = .008, explanatory effect size xi = 0.67, 95% CI [0.28, 0.99], n = 30.";
  "Bayesian t-test (JZS prior, r = 0.707): BF10 = 1.20, log(BF01) = -0.18, n = 60.";
  "Bayesian t-tests of each pair (JZS prior, r = 0.707): ctrl vs trt1, BF10 = 0.65;
  ...; n = 30." BF10 has two decimals, scientific from 1e5 and below 0.01; a BF10
  beyond the range of a double (stored as inf or 0) prints from log(BF01) as
  m.mme±NNN ("BF10 = 4.19e+330"), also in the alt text and `str(result)`; xi and AKP
  keep the leading zero (not bounded by 1). Without a computed test (type bayes, two
  groups, `results_subtitle=False`): "No test was computed (results_subtitle=False), n
  = 60."
- **Alt text:** two to four sentences, at most 400 characters: "Plot comparing <y>
  across the <k> groups of <x>: <level> (n = <n>, mean <value>), ...", then the APA
  sentence, then "<s> of <m> pairwise comparisons are significant (<test>,
  <adjustment>, alpha = <alpha>)." When that is longer than 400 characters the groups
  are summarized as ranges and long names are shortened with an ellipsis (U+2026);
  if it is still too long the group description is dropped, and the result sentence
  is never cut (so only a result sentence over 400 characters exceeds the limit).
  The centrality reads "mean", "median", "<tr x 100>% trimmed mean" or "MAP
  estimate", and "no MAP estimate" for a group without one (D55). Type bayes with
  three or more groups replaces the result and pair sentences with "Bayesian t-tests
  of each of the <m> pairs (JZS prior, r = 0.707, Student's t): BF10 from <min> (<g1>
  vs <g2>) to <max> (<g1> vs <g2>); brackets show log(BF01) for every pair.", pairs
  ranked by -log(BF01), level names shortened like the groups'; with
  `pairwise_display="none"` it ends "no brackets are drawn (pairwise_display='none')."

## 8. `check()`

`result.check()` returns the statistical issues; `fig.check()` returns those plus the
rendering issues. Each is a tuple of `Issue(code, severity, message, fix)`; empty means
no problems.

| Code | Level | Severity | Trigger |
| --- | --- | --- | --- |
| `small_n` | result | warning | any group with n < 5 (withinstats: fewer than 5 complete subjects, reported once) |
| `heavy_ties` | result | warning | type nonparametric where more than 25% of the analyzed values are ties: n minus the number of distinct values (D47; read from `result.values`) |
| `rows_dropped` | result | info | `n_dropped > 0` or subjects dropped |
| `pairs_hidden` | result | info | some pairs not displayed |
| `infinite_effect_size` | result | warning | type robust: the effect size, or some of its bootstrap resamples, is infinite (a winsorized variance of 0 next to trimmed means that differ); the values are kept, so a CI bound can be ±Inf |
| `pairwise_conf_level` | result | info | the pairwise intervals are at a level other than `conf_level` (type robust with three or more groups: lincon's are always 95%, D53) |
| `label_overlap` | figure | warning | any two text artists' rendered bounding boxes intersect, or a bracket label's box reaches more than 0.5 px into the bar or a leg of another bracket |
| `text_clipped` | figure | warning | any text extends outside the figure |
| `points_subsampled` | figure | info | large-n subsampling used |
| `path_hidden` | figure | warning | withinstats: a text's rendered box covers more than 1 px of the centrality path or of a subject path (0.5 px slack at the box edges) |
| `paths_crowded` | figure | warning | withinstats: the subject paths, rendered alone as drawn, are at least half opaque over more than 15% of the area between the two conditions (about 250 subjects in inferplot's own figure); the fix is `point_path=False` |
| `centrality_label_overlap` | figure | warning | a centrality point at least half covered by a text's box |
| `low_contrast` | figure | warning | a text under 3:1 (WCAG 2.1, large text and graphics) against its background, or a level's colour under 1.1:1 against the Axes background (WCAG's 3:1 would report the default palette's yellow, 1.32:1 on white) |
| `glyph_missing` | figure | warning | characters no font of a text can draw (DejaVu Sans, then the user's `font.sans-serif` fonts; in mathtext DejaVu Sans and STIX) |
| `axis_changed` | figure | warning | an inverted x or y axis, or a non-linear scale: brackets and labels were placed for an upright linear axis |
| `data_outside_view` | figure | warning | values, levels or brackets outside the axis limits, whoever changed them |
| `flat_data` | figure | warning | values that differ drawn within 2 px of height |

`label_overlap` compares the rendered boxes of every drawn, non-empty text: title,
subtitle, caption lines, axis labels, visible tick labels, centrality labels with their
label boxes, bracket labels; with `ax=` the texts of that Axes, each also compared with
the plotting areas and texts (titles, axis and tick labels) of the figure's other Axes
and with figure texts such as a suptitle (an overlap there is a `label_overlap` naming
the other panel's text). Under constrained or tight layout the advice names the layout
engine as the cause. `text_clipped`
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
  ggbetweenstats and ggwithinstats (complete subjects through
  `statsExpressions::complete_subjects`, recorded as `within_data`; bootstrap replay
  of `dep.effect`, `wmcpAKP` and `kendalls_w`). Each other function first needs its
  own bookkeeping (which rows it analyzes: one sample for histostats), its own
  bootstrap replay, and its own rules for naming the hidden calls, for when a caption
  is expected and for checking `n_per_level`; until then the script stops on its cases
  instead of writing an incomplete fixture. In GitHub Actions the generator runs in
  ten parallel shards, and a merge job checks that every case and probe ran exactly
  once on the same R and library before it writes the manifest.
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
  three or more nonparametric groups). The replay follows the case's `conf.level`
  (statsExpressions passes alpha = 1 - conf.level to WRS2; the candidates try that
  and the rounded decimal) and `tr`; a case without a subtitle has no reference, and a
  case that fails after its subtitle is referenced from `partial` (edge_n2-robust).
  The 50 replicates run in parallel processes, each with its own seed.
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
  every installed package version. It runs OpenBLAS on one thread with Haswell kernels
  and logs the runner's CPU and kernel: BayesFactor's Bayes ANOVA depends on the kernels
  in the last digits, and its posterior sampler amplifies that (nine earlier fixtures
  changed until it was pinned). `ORACLE_SUBSET` (a regular expression) limits a local
  debugging run, which writes no manifest. It uploads the JSON files as artifact
  `oracle-fixtures`, and a separate job that runs no R code signs them with a
  build-provenance attestation (`actions/attest`). They are committed to
  `tests/fixtures/oracle/` on the same branch after review.
- Datasets: mtcars, iris, ToothGrowth, PlantGrowth, morley, sleep, ggstatsplot's
  `bugs_long` and `Titanic_full`, plus edge cases: n = 2 per group, ties, all-negative
  outcome, empty factor level, missing values, unbalanced groups, k = 2, 3 and 5, a
  subject missing one condition, heavy outliers; Phase 2 added `iris_no_setosa`
  (exported), `edge_ties_small` and `edge_k9` (nine groups). `oracle/export_datasets.R` writes the R
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
  ToothGrowth and PlantGrowth; `bf.message = FALSE`; `bf.prior` 0.8). Phase 2's 88
  nonparametric, robust and bayes cases: the same edge data and arguments for each
  type, iris Petal.Length, iris_no_setosa (Petal.Length and Petal.Width), tr 0.1,
  one-sided edge_k2 (nonparametric), ToothGrowth-greater (robust, bayes),
  edge_ties_small (nonparametric) and edge_k9 (robust); edge_n2 robust and bayes
  declare failures (lincon's df below 2; the Bayes pairwise table).
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
  accurate `uniroot` lower bound (tol 1e-12) for "greater". `probe-smm`: lincon's
  two tables (alpha 0.05 and 0.01, 48 df values x C = 1 to 50), the WRS2 functions
  lincon calls, lincon itself on 2 to 10 groups at 6 alphas, and a seed check.
  `probe-wrs2p`: 420 yuen, t1way and lincon p-values against 1 - cdf and the exact
  tail. `probe-akp`: `akp.effect` on 8 datasets at tr 0.01 to 0.49. `probe-map`: 52
  vectors with `map_estimate`, `bw.SJ` (or its error), `bw.nrd0` and density
  summaries (plus 4 vectors that pin the bin anchor threshold). `probe-ranks`:
  `wilcox.test`, `rank_biserial`, `kruskal.test` and Dunn on tied data.
  `probe-signrank`: paired `wilcox.test` for every combination of exact, correct and
  alternative, and the paired `rank_biserial`, on 12 sets of pairs. `probe-pairedbf`:
  paired `ttestBF` at \|t\| from 0 to 1,000. `probe-friedman`: `friedman.test`,
  Durbin-Conover and Kendall's W on 11 tables. `probe-rmanova`: afex with no,
  Greenhouse-Geisser and Huynh-Feldt correction, the error strata and omega squared.
  `probe-wrs2within`: yuend and rmanova p-values against 1 - cdf and the exact tail,
  the rmmcp table, dep.effect and wmcpAKP. They are hashed in the
  manifest (`probes`) and checked like the fixtures; tests read them with
  `load_probes()` / `probe()` and `OracleProbe.section()` / `rows()`.
- Both sides read CSVs the same way: only empty fields are missing, and R sorts strings
  in the C locale, so factor levels from text columns follow Python's `sorted()`.
- plotsalot's `oracle/` (MIT) is a usable starting point; credit it in `NOTICE`.
- Python tests load the same CSV from `oracle/datasets/`, translate the fixture's R
  arguments to inferplot keywords with one mapping table (`tests/oracle_args.py`: dots to
  underscores, R `TRUE`/`FALSE`/`NULL` to Python), run the analysis, and compare every
  number inferplot reports that the fixture holds, using `tests/tolerances.py`. One
  parametrized test per function collects all its fixtures, so a new fixture is tested
  without new code. Fixture numbers no test compares, because v0.1 reports none of
  them: the Bayesian ANOVA Bayes factors and posterior estimates (bayes subtitle and
  parametric caption with three or more groups; v0.2), the IQR, CI, skewness and
  kurtosis of ggstatsplot's centrality table, the bracket positions ggplot2 computed,
  and ggstatsplot's own 100-resample bootstrap bounds (only its strings are compared;
  section 10). Result fields R's tables do not carry are compared with independent
  computations instead: the unadjusted pairwise p of Dunn's test with the normal tail
  of R's z, and lincon's t, df and unadjusted p with SciPy's Yuen test
  (`test_pairwise_raw_p_and_df`). `test_r_warnings_and_messages_are_declared` fails
  when a fixture records an R warning or message that has not been read and declared
  (`DECLARED_R_NOISE`: afex's Huynh-Feldt and lme4 notes and insight's note on the
  Bayesian repeated-measures ANOVA); no other test reads these fields.

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
| `within_data` | ggwithinstats only: `subject_id`, `levels`, `subjects`, `subject_order` (the sort rule, as text), `values` (one row per complete subject), `dropped_subjects`, `rows_without_x_or_subject` |
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
changed without a new oracle run. The loader alone does not catch a fixture and the
manifest edited together (the bootstrap tolerances are read from the fixtures, so such
an edit could also widen them); `fixtures.yml` is the guard against that, and it
blocks a merge only if main is protected with it as a required check, which is a
repository setting (not set as of 2026-11-10; until then its result has to be read). Attestations do not expire, unlike artifacts (90
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
| p-values WRS2 computes as 1 - pt or 1 - pf (`wrs2_p`: Yuen, t1way, lincon), against inferplot's accurate tail (D43) | rtol 1e-10, atol 2^-53 x the factor applied to the one-sided tail afterwards (2 for two-sided, times the number of pairs after Holm) | the one-sided tail rounded to the spacing of doubles just below 1 (2^-53): probe-wrs2p's 420 yuen, t1way and lincon p-values all equal R's 1 - cdf bit for bit; rounding the tail the same way matches all fixtures but iris t1way, which lands one grid step away |
| noncentral-distribution CIs (`noncentral_ci`) | rtol 1e-6, atol 4e-8 | R's root finding is accurate to about 1e-8 absolute: Hedges' g bounds at most 1.9e-8 absolute and 7.5e-7 relative (ToothGrowth lower bound, -0.0217), omega lower bounds at most 1.5e-8 absolute; an omega bound where effectsize's search stopped early is compared with the accurate root instead (D33) |
| JZS Bayes factors of t tests (`ttest_bayes_factor`) | rtol 1e-8 | measured on two-sample values only: `tools/reference` reproduces 75 of the 77 distinct `ttestBF` values in the fixtures to at most 2.3e-9, which is BayesFactor's own integration error; the two it misses are compared with 50-digit values (D50); a prior of sqrt(2)/2 instead of 0.707 moves them by 3.7e-6 or more, Welch's t instead of Student's moves every unbalanced pair by 1.5e-4 or more. Measure paired and one-sample values when their fixtures arrive |
| other Bayes factors (correlation; v0.2 ANOVA and contingency) | own entry, measured when first implemented | not yet measured |
| bootstrap CIs and p-values | within 4 Monte Carlo SDs of the 20,000-resample R reference, never less than the closed-form tolerance | randomness; a quantity whose replicate SD is 0 does not depend on the resamples (AKP and rank epsilon squared estimates, an upper bound of 1) and is stored to 15 significant digits |
| MAP centrality (`MAP_CENTRALITY`) | the same grid point of the 1,024-point grid from min(x) to max(x) (0 steps; the oracle value must lie on the grid within 1e-6 steps) | every fixture label on ggstatsplot's grid point; each method variant tried moves 1 to 14 of the first 30; probe-map's normal_n1e5 is pinned one step away (D55 row) |
| WRS2's AKP constant (`akp_constant`) | rtol 5e-7 | WRS2's constant against the exact winsorized-normal SD: 1.2e-9 at tr 0.2, 4.6e-9 at 0.1, at most 3.4e-8 up to 0.25, 4.2e-7 at 0.49 (probe-akp). AKP parity itself scales by the probe's constant at the case's tr and uses `closed_form` (D42) |
| WRS2's xi constant (`wrs2_xi_constant`) | rtol 3.3e-8 | WRS2's divisor against the exact winsorized normal variance: -3.2404e-8 in xi at tr 0.2, +3.0242e-8 at 0.1; measured at those two only (D42) |
| Studentized maximum modulus quantile against mpmath (`smm_quantile`) | rtol 3e-14 | 148 frozen 30-digit values (`tests/data/smm_reference.json`, `tools/reference/smm_mpmath.py`): C = 1 to 300, df 1 to 1e5 and infinity; measured at most 1.1e-14 |
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
