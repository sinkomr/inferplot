---
name: inferplot
description: Statistical comparison and association plots in Python with the test computed and its full result written on the figure. Use when asked to compare groups (t-test, ANOVA, Mann-Whitney, Kruskal-Wallis), paired or repeated measures (paired t-test, Wilcoxon, Friedman), one-sample tests against a value, correlations (Pearson, Spearman), or contingency tables and counts (chi-square, McNemar), and to plot and report the result.
---

# inferplot

inferplot (`import inferplot as ip`) draws the plot, runs the test that fits the
design and writes the result on the figure: statistic, df, exact p, effect size with
its confidence interval, Bayes factor, n, adjusted pairwise comparisons. It
reproduces R's ggstatsplot 1.1.4.

## When to use it

Use it for a group comparison, a repeated-measures comparison, a one-sample test, a
correlation of two measurements or a test on counts of categories, when the user
wants a figure, a reported result, or both. Do not use it for regression models
with covariates, curve fitting, time series or interactive plots. If the package is
not installed, say so; it is not on PyPI yet (install it from
github.com/sinkomr/inferplot).

## Which function

| Question | Call |
| --- | --- |
| Independent groups differ? | `ip.betweenstats(df, "group", "score")` |
| Same subjects measured repeatedly differ? | `ip.withinstats(long, "time", "score", subject_id="subject")` |
| One sample differs from a value? | `ip.histostats(df, "score", test_value=5)` |
| Two measurements related? | `ip.scatterstats(cars, "wt", "mpg")` |
| Shares of categories differ between groups? | `ip.barstats(answers, "answer", "group")` |
| Shares differ from expected shares? | `ip.barstats(answers, "answer", ratio=[0.5, 0.5])` |
| Same units changed category? | `ip.barstats(answers, "before", "after", paired=True)` |

Data are a pandas, polars or pyarrow data frame in long format: one row per
observation; repeated measures are several rows per subject. In the examples `df`,
`long`, `cars` and `answers` are such frames with the columns named.

## Rules

1. Declare the design. Repeated measures: `withinstats(..., subject_id="subject")`;
   never rely on row order. Two categorical measurements of the same units:
   `barstats(..., paired=True)`.
2. Choose `type` from the data and the research question before looking at any
   result: `"parametric"` (means; roughly normal data or large samples),
   `"nonparametric"` (ranks; ordinal or skewed data), `"robust"` (trimmed means;
   heavy tails or outliers), `"bayes"` (Bayes factor). Never try several types,
   seeds or settings to find significance; if the user asks for several, report
   all of them.
3. Read the result from `fig.result.text.apa` and `fig.result.to_json()`, not from
   the image.
4. Run `fig.check()`. Apply each issue's `fix`, or report the issue to the user.
5. Report every entry of `fig.result.warnings`.
6. Save with `fig.save("name.svg")` (or .pdf, .png): text stays editable and the
   result is embedded. Never `plt.savefig()`: it writes an empty figure.

## Examples

```python
fig = ip.betweenstats(df, "group", "score", type="parametric")
print(fig.result.text.apa)
issues = fig.check()  # act on or report each one
notes = fig.result.warnings  # report each one
fig.save("score.svg")
```

```python
fig = ip.withinstats(long, "time", "score", subject_id="subject", type="nonparametric")
fig = ip.histostats(df, "score", test_value=5, type="parametric")
fig = ip.scatterstats(cars, "wt", "mpg", type="nonparametric")  # Spearman
fig = ip.barstats(answers, "answer", "group")  # chi-square test of independence
fig = ip.barstats(answers, "before", "after", paired=True)  # McNemar
fig = ip.barstats(answers, "answer")  # goodness of fit against equal shares
```

Compute without drawing, then draw (the numbers do not change):

```python
result = ip.analyze.betweenstats(df, "group", "score")
fig = ip.render.betweenstats(result, layers=("box", "points"))
table = result.to_frame()  # one row per test and pair
```

## Errors

Every error message names the fix: follow it literally. `TypeError` means a wrong
keyword or data shape (the message gives the right name, for example
`confidence=` becomes `conf_level=`); `KeyError` a missing column (the message lists
the columns); `ValueError` a value or data the method cannot use;
`NotImplementedError` something that arrives in v0.2 (Bayes factors for three or
more groups and for counts, `grouping_var`): use what the message suggests, never
another test without telling the user.

## Reference

- [docs/api.md](https://github.com/sinkomr/inferplot/blob/main/docs/api.md): every
  argument, result field and `check()` code.
- [docs/intervals.md](https://github.com/sinkomr/inferplot/blob/main/docs/intervals.md):
  how far to trust each interval; cite it when an interval is reported as
  approximate.
