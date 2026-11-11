#!/usr/bin/env Rscript
# inferplot R oracle: run ggstatsplot 1.1.4 and write one JSON fixture per case.
#
# Usage, from the repository root: Rscript oracle/generate.R <datasets-dir> <output-dir>
#
# Runs in GitHub Actions only (.github/workflows/oracle.yml), which pins R 4.5.3 and a
# dated Posit Package Manager CRAN snapshot, checks every installed version, and signs
# the output with a build-provenance attestation. The output is committed to
# tests/fixtures/oracle/ after review and never edited by hand: manifest.json holds
# the SHA-256 of every fixture and of the generator files below, and each fixture
# holds the SHA-256 of the CSV it read. It also writes probe files (probe-*.json, see
# "Probes" below), hashed in the manifest the same way. docs/design/SPEC.md section 9
# describes the schema.
#
# This script only calls R packages; none of their code is copied into inferplot.

suppressPackageStartupMessages({
  library(ggstatsplot)
  library(jsonlite)
  library(digest)
})

PARITY_VERSION <- "1.1.4"
SCHEMA_VERSION <- 2L
# Files that decide what the fixtures contain. The manifest records their SHA-256, and
# the Python loader refuses fixtures whose generator differs from the committed one.
GENERATOR_FILES <- c(
  "oracle/generate.R", "oracle/export_datasets.R", ".github/workflows/oracle.yml"
)
CASE_SEED <- 20261107L # set before every ggstatsplot call
BOOT_SEED <- 1L # reference run; replicate i uses BOOT_SEED + i
BOOT_REFERENCE_RESAMPLES <- 20000L
BOOT_REPLICATE_RESAMPLES <- 2000L # inferplot's resample count (SPEC.md section 5)
BOOT_REPLICATES <- 50L
# Processes for the replicates (each sets its own seed, so the count does not change
# any value). GitHub's ubuntu-24.04 runners have 4 cores.
BOOT_CORES <- max(1L, min(4L, parallel::detectCores(), na.rm = TRUE))
# Resamples ggbetweenstats uses for every bootstrap CI: statsExpressions 2.2.1's nboot
# default, which it passes as nboot (akp.effect, t1way) or iterations
# (rank_epsilon_squared).
GG_RESAMPLES <- 100L
# ggbetweenstats' defaults for the arguments the bootstrapped effect sizes take.
DEFAULT_CONF_LEVEL <- 0.95
DEFAULT_TR <- 0.2
# statsExpressions passes alpha = 1.0 - conf.level to WRS2. In floating point that is
# 0.050000000000000044 for 0.95, not 0.05 (0.09999999999999998 for 0.90, not 0.1),
# which can change which bootstrap order statistic a CI uses (alpha / 2 * 100 =
# 2.500000000000002). Candidates try both spellings, the computed one first.
alpha_spellings <- function(conf_level) {
  rounded <- format(signif(1 - conf_level, 10))
  list(
    list(text = paste("1 -", format(conf_level)), value = 1.0 - conf_level),
    list(text = rounded, value = as.numeric(rounded))
  )
}
case_conf_level <- function(case) {
  if (is.null(case$args$conf.level)) DEFAULT_CONF_LEVEL else case$args$conf.level
}
case_tr <- function(case) if (is.null(case$args$tr)) DEFAULT_TR else case$args$tr

RECORDED_PACKAGES <- c(
  "ggstatsplot", "statsExpressions", "effectsize", "parameters", "bayestestR",
  "insight", "datawizard", "correlation", "performance", "WRS2", "BayesFactor",
  "PMCMRplus", "afex", "ggplot2", "jsonlite", "digest"
)

if (as.character(packageVersion("ggstatsplot")) != PARITY_VERSION) {
  stop(
    "ggstatsplot ", as.character(packageVersion("ggstatsplot")),
    " is installed; the oracle requires ", PARITY_VERSION
  )
}
missing_generator_files <- GENERATOR_FILES[!file.exists(GENERATOR_FILES)]
if (length(missing_generator_files) > 0L) {
  stop(
    "run generate.R from the repository root; not found: ",
    paste(missing_generator_files, collapse = ", ")
  )
}

# Sort strings by code point, as Python's sorted() does, so factor levels built from
# character columns come out in the same order on both sides.
invisible(Sys.setlocale("LC_COLLATE", "C"))

args <- commandArgs(trailingOnly = TRUE)
dataset_dir <- if (length(args) >= 1L) args[[1L]] else "oracle/datasets"
out_dir <- if (length(args) >= 2L) args[[2L]] else "oracle/out"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Cases ---------------------------------------------------------------------------
#
# Each case names a dataset CSV, the R arguments (x and y are column names), and
# optionally declared factor levels (to create empty levels). `expect_error` declares
# a case where ggstatsplot itself fails: the hidden call that fails (`component`),
# text its error message contains (`message`) and why (`reason`); the fixture then
# records what the hidden calls returned before the failure (`partial`).
# `expect_hidden_errors` names the components (subtitle, caption) whose error
# ggstatsplot is expected to swallow (see "Conditions ggstatsplot hides" below). Any
# other outcome stops the run.

TYPES <- c("parametric", "nonparametric", "robust", "bayes")

between_cases <- function(dataset, x, y, label = dataset, types = TYPES,
                          extra_args = list(), factor_levels = NULL) {
  lapply(types, function(type) {
    list(
      name = paste("ggbetweenstats", label, type, sep = "-"),
      fun = "ggbetweenstats",
      dataset = dataset,
      factor_levels = factor_levels,
      args = c(list(x = x, y = y, type = type), extra_args),
      columns = c("x", "y"),
      expect_error = NULL,
      expect_hidden_errors = character()
    )
  })
}

# One parametric case, as a list of one case so that c() can combine it with the
# others. ggbetweenstats arguments other than x, y and type go in `extra_args`;
# `label` names the case.
parametric_case <- function(dataset, x, y, label = dataset, extra_args = list(),
                            factor_levels = NULL, expect_error = NULL) {
  cases <- between_cases(
    dataset, x = x, y = y, label = label, types = "parametric",
    extra_args = extra_args, factor_levels = factor_levels
  )
  cases[[1L]]$expect_error <- expect_error
  cases
}

# Phase 2 (betweenstats nonparametric, robust and bayes): one case per type in
# `types`. `expect_error` maps a type to the failure that case declares, as above.
PHASE2_TYPES <- c("nonparametric", "robust", "bayes")
phase2_cases <- function(dataset, x, y, label = dataset, types = PHASE2_TYPES,
                         extra_args = list(), factor_levels = NULL, expect_error = list()) {
  unknown <- setdiff(names(expect_error), types)
  if (length(unknown) > 0L) stop(label, ": expect_error for types without a case: ", unknown)
  cases <- between_cases(
    dataset, x = x, y = y, label = label, types = types,
    extra_args = extra_args, factor_levels = factor_levels
  )
  lapply(cases, function(case) {
    case$expect_error <- expect_error[[case$args$type]]
    case
  })
}

# Phase 3 (withinstats): ggwithinstats cases, one per type in `types`. Every case
# names its subject column (subject.id); inferplot pairs by subject, never by row
# order. `expect_error` maps a type to the failure that case declares, as above, and
# `expect_hidden_errors` a type to the components whose error ggstatsplot swallows.
within_cases <- function(dataset, x, y, subject, label = dataset, types = TYPES,
                         extra_args = list(), factor_levels = NULL, expect_error = list(),
                         expect_hidden_errors = list()) {
  unknown <- setdiff(c(names(expect_error), names(expect_hidden_errors)), types)
  if (length(unknown) > 0L) stop(label, ": expectations for types without a case: ", unknown)
  lapply(types, function(type) {
    list(
      name = paste("ggwithinstats", label, type, sep = "-"),
      fun = "ggwithinstats",
      dataset = dataset,
      factor_levels = factor_levels,
      args = c(list(x = x, y = y, subject.id = subject, type = type), extra_args),
      columns = c("x", "y", "subject.id"),
      expect_error = expect_error[[type]],
      expect_hidden_errors = expect_hidden_errors[[type]] %||% character()
    )
  })
}

# The within-subject edge datasets (oracle/export_datasets.R) share their columns.
edge_within <- function(dataset, ...) {
  within_cases(dataset, x = "condition", y = "value", subject = "subject", ...)
}

# Phase 4: gghistostats cases (one numeric column `x` against `test.value`) and
# ggscatterstats cases (a correlation of `x` and `y`), one per type in `types`, with
# `expect_error` and `expect_hidden_errors` mapping a type to what the case declares,
# as above, and `expect_empty_layers` the geoms ggplot2 builds no rows for (a smooth
# line on constant x). `columns` names the arguments that are column names
# (label.var too).
phase4_cases <- function(fun, dataset, args, label, types, extra_args, expect_error,
                         expect_hidden_errors, expect_empty_layers, expect_na_values) {
  declared <- c(
    names(expect_error), names(expect_hidden_errors), names(expect_empty_layers),
    names(expect_na_values)
  )
  unknown <- setdiff(declared, types)
  if (length(unknown) > 0L) stop(label, ": expectations for types without a case: ", unknown)
  lapply(types, function(type) {
    all_args <- c(args, list(type = type), extra_args)
    list(
      name = paste(fun, label, type, sep = "-"),
      fun = fun,
      dataset = dataset,
      factor_levels = NULL,
      args = all_args,
      columns = intersect(c("x", "y", "label.var"), names(all_args)),
      expect_error = expect_error[[type]],
      expect_hidden_errors = expect_hidden_errors[[type]] %||% character(),
      expect_empty_layers = expect_empty_layers[[type]] %||% character(),
      expect_na_values = expect_na_values[[type]] %||% character()
    )
  })
}

# `expect_na_values` maps a type to the tables (subtitle, caption) where R reports a
# statistic, p-value, estimate, bound or Bayes factor as NA: ggstatsplot prints such a
# number as "NA" or as an empty string, with at most a warning, so an NA no case
# declares stops the run.
histo_cases <- function(dataset, x, label = dataset, types = TYPES, extra_args = list(),
                        expect_error = list(), expect_hidden_errors = list(),
                        expect_empty_layers = list(), expect_na_values = list()) {
  phase4_cases(
    "gghistostats", dataset, list(x = x), label, types, extra_args, expect_error,
    expect_hidden_errors, expect_empty_layers, expect_na_values
  )
}

scatter_cases <- function(dataset, x, y, label = dataset, types = TYPES, extra_args = list(),
                          expect_error = list(), expect_hidden_errors = list(),
                          expect_empty_layers = list(), expect_na_values = list()) {
  phase4_cases(
    "ggscatterstats", dataset, list(x = x, y = y), label, types, extra_args, expect_error,
    expect_hidden_errors, expect_empty_layers, expect_na_values
  )
}

# The one-sample and correlation edge datasets share their columns.
edge_one <- function(dataset, test_value, ...) {
  histo_cases(dataset, x = "value", extra_args = list(test.value = test_value), ...)
}
edge_corr <- function(dataset, ...) scatter_cases(dataset, x = "x", y = "y", ...)

# Phase 4 (barstats): ggbarstats cases, one per type in `types`. `x` is the variable
# drawn as the fill (the rows of the table), `y` the bars (its columns); without `y`
# the design is one-way (goodness of fit), which ggbarstats 1.1.4 accepts as well as
# ggpiestats (`fun`; both compute it through statsExpressions::contingency_table()).
# `columns` names the arguments that are column names (counts too). Declarations as
# above, by type: `expect_error` (ggstatsplot fails; component "data" when it fails
# while preparing the data, before any hidden call), `expect_hidden_errors` (the
# components whose error ggstatsplot swallows: subtitle, caption,
# pairwise_comparisons, and proportion_test for the per-bar tests, which become NA
# rows), `expect_na_values` (the tables, among subtitle, caption, pairwise and
# one_sample, where R reports a number as NA) and `expect_null_outputs` (components
# that return nothing without an error).
bar_cases <- function(dataset, x, y = NULL, label = dataset, types = "parametric",
                      extra_args = list(), factor_levels = NULL, fun = "ggbarstats",
                      expect_error = list(), expect_hidden_errors = list(),
                      expect_na_values = list(), expect_null_outputs = list()) {
  declared <- c(
    names(expect_error), names(expect_hidden_errors), names(expect_na_values),
    names(expect_null_outputs)
  )
  unknown <- setdiff(declared, types)
  if (length(unknown) > 0L) stop(label, ": expectations for types without a case: ", unknown)
  lapply(types, function(type) {
    args <- c(list(x = x), if (!is.null(y)) list(y = y), list(type = type), extra_args)
    list(
      name = paste(fun, label, type, sep = "-"),
      fun = fun,
      dataset = dataset,
      factor_levels = factor_levels,
      args = args,
      columns = intersect(c("x", "y", "counts"), names(args)),
      expect_error = expect_error[[type]],
      expect_hidden_errors = expect_hidden_errors[[type]] %||% character(),
      expect_empty_layers = character(),
      expect_na_values = expect_na_values[[type]] %||% character(),
      expect_null_outputs = expect_null_outputs[[type]] %||% character()
    )
  })
}
pie_cases <- function(...) bar_cases(..., fun = "ggpiestats")
# The hand-written two-way edge datasets share their columns.
edge_bar <- function(dataset, ...) bar_cases(dataset, x = "x", y = "y", ...)
# The paired ones: one row per subject, the answer before (x) and after (y).
edge_paired <- function(dataset, extra_args = list(), ...) {
  bar_cases(
    dataset, x = "before", y = "after", extra_args = c(list(paired = TRUE), extra_args), ...
  )
}

P_ADJUST_OTHERS <- c("bonferroni", "hochberg", "hommel", "BH", "BY", "fdr", "none")

CASES <- c(
  between_cases("ToothGrowth", x = "supp", y = "len"), # k = 2, n = 30 + 30
  between_cases("PlantGrowth", x = "group", y = "weight"), # k = 3, n = 10 each
  between_cases("mtcars", x = "cyl", y = "mpg"), # k = 3, n = 11/7/14, ties
  # Added after the Phase 0 review; each settles a SPEC.md claim the cases above
  # cannot test.
  # Unbalanced two groups (19/13): pooled or unpooled SD in Hedges' g, Student or
  # Welch t in the JZS Bayes factor.
  between_cases("mtcars", x = "am", y = "mpg", label = "mtcars-am"),
  # Welch df 27.197..., which the subtitle prints as 27.2 (df are numbers there).
  between_cases(
    "mtcars", x = "am", y = "drat", label = "mtcars-am-drat", types = "parametric"
  ),
  # One-sided alternatives: the sidedness of the effect-size CIs.
  between_cases(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-greater",
    types = c("parametric", "nonparametric"), extra_args = list(alternative = "greater")
  ),
  between_cases(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-less",
    types = c("parametric", "nonparametric"), extra_args = list(alternative = "less")
  ),
  # Two groups without ties, both under 50: exact or normal-approximation
  # Mann-Whitney p.
  between_cases("edge_k2", x = "group", y = "value"), # n = 9 + 12
  # Welch F df2 97.40 (printed 97.4); lincon at Yuen df 56 to 58.
  between_cases("iris", x = "Species", y = "Sepal.Width"),
  # t1way df2 10.40 (printed 10.4); lincon at Yuen df 6.7, 11.9 and 12.9.
  between_cases("mtcars", x = "cyl", y = "disp", label = "mtcars-disp"),
  # lincon at Yuen df 5.2, 12.8 and 19.1; groups of 4, 12 and 25.
  between_cases("edge_unbalanced", x = "group", y = "value"),
  # Five groups, ten pairs: whether lincon's C counts pairs or groups.
  between_cases("edge_k5", x = "group", y = "value"),
  # Four groups, six pairs, Yuen df 99 to 110; missing values in y.
  between_cases("bugs_long", x = "condition", y = "desire"),
  # Phase 1 (betweenstats, parametric): edge data and every argument the parametric
  # slice exposes. Each pins something the cases above leave open.
  # Missing values in y and one missing group label: incomplete rows are dropped.
  parametric_case("edge_missing", x = "group", y = "value"),
  # A declared level with no rows, placed between others: ggstatsplot drops it and
  # keeps the declared order of the rest (c, a, b).
  parametric_case(
    "edge_empty_level", x = "group", y = "value",
    factor_levels = list(group = c("c", "empty", "a", "b"))
  ),
  # Declared level order on a known dataset: tick order, pair order, the sign of q.
  parametric_case(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-declared",
    factor_levels = list(group = c("trt2", "ctrl", "trt1"))
  ),
  # Two values per group: every pair's Welch df lies between 1 and 2 (1.06, 1.42,
  # 1.28). ggbetweenstats 1.1.4 fails here: R's ptukey returns NaN for df < 2, so
  # PMCMRplus::gamesHowellTest gives NaN p-values (probe-ptukey records them), and
  # statsExpressions' pairwise_comparisons() then stops while building its table.
  # The subtitle, the Bayes caption and the centrality labels were computed before
  # the failure; the fixture keeps them in `partial`.
  parametric_case(
    "edge_n2", x = "group", y = "value",
    expect_error = list(
      component = "pairwise_comparisons",
      message = "arguments imply differing number of rows",
      reason = paste(
        "Welch df below 2 in every pair: ptukey returns NaN for df < 2 and",
        "pairwise_comparisons() fails on the NaN p-values"
      )
    )
  ),
  # Integer scores with heavy ties: centrality labels at exact decimals.
  parametric_case("edge_ties", x = "group", y = "score"),
  # All-negative outcome: negative means and labels.
  parametric_case("edge_negative", x = "group", y = "value"),
  # Two extreme values in one group: unequal variances.
  parametric_case("edge_outliers", x = "group", y = "value"),
  # Six groups, fifteen pairs, counts with zeros (R's InsectSprays).
  parametric_case("InsectSprays", x = "spray", y = "count"),
  # Welch F below 1 (0.65): partial omega squared 0, its CI [0, 1].
  parametric_case("edge_null", x = "group", y = "value"),
  # n of 1100 to 1300 per group: n with a big mark, Welch df above 1000. The values
  # are round(50 + 6 sin(0.37 i) + 2 cos(1.13 i) + shift, 2) for the row number
  # i = 1..3600 over all groups, printed with two decimals: a has rows 1 to 1200
  # (shift 0), b rows 1201 to 2500 (shift 0.6), c rows 2501 to 3600 (shift 1.1).
  parametric_case("edge_large_n", x = "group", y = "value"),
  # Every other p.adjust method on the pairwise Games-Howell p-values.
  unlist(lapply(
    c("bonferroni", "hochberg", "hommel", "BH", "BY", "fdr", "none"),
    function(method) {
      parametric_case(
        "PlantGrowth", x = "group", y = "weight",
        label = paste0("PlantGrowth-padj-", method),
        extra_args = list(p.adjust.method = method)
      )
    }
  ), recursive = FALSE),
  # A 90% confidence level: Hedges' g CI, partial omega squared CI, the CI label.
  parametric_case(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-cl90",
    extra_args = list(conf.level = 0.90)
  ),
  parametric_case(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-cl90",
    extra_args = list(conf.level = 0.90)
  ),
  # Three digits in every printed number.
  parametric_case(
    "mtcars", x = "am", y = "mpg", label = "mtcars-am-digits3",
    extra_args = list(digits = 3L)
  ),
  parametric_case(
    "mtcars", x = "cyl", y = "mpg", label = "mtcars-digits3",
    extra_args = list(digits = 3L)
  ),
  # Which pairs get brackets.
  parametric_case(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-display-all",
    extra_args = list(pairwise.display = "all")
  ),
  parametric_case(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-display-ns",
    extra_args = list(pairwise.display = "non-significant")
  ),
  parametric_case(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-display-none",
    extra_args = list(pairwise.display = "none")
  ),
  # No statistical subtitle: ggstatsplot then computes no subtitle and no caption.
  parametric_case(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  parametric_case(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  # No Bayes caption.
  parametric_case(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-nobf",
    extra_args = list(bf.message = FALSE)
  ),
  # Another Cauchy prior scale for the Bayes caption.
  parametric_case(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-prior08",
    extra_args = list(bf.prior = 0.8)
  ),
  # Phase 2 (betweenstats nonparametric, robust, bayes): the Phase 1 edge data and
  # every argument these types take.
  phase2_cases("edge_missing", x = "group", y = "value"),
  phase2_cases(
    "edge_empty_level", x = "group", y = "value",
    factor_levels = list(group = c("c", "empty", "a", "b"))
  ),
  # Declared order: pair order and the sign of lincon's differences.
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-declared",
    factor_levels = list(group = c("trt2", "ctrl", "trt1"))
  ),
  # Two values per group: Dunn on tiny groups, MAP of two values. Every pair's Yuen df
  # is below 2, where lincon's critical-value table stops with an error, and
  # BayesFactor's pairwise tests leave the pairwise table without rows. The subtitle
  # (and its bootstrap) and the centrality labels come first; `partial` keeps them.
  phase2_cases(
    "edge_n2", x = "group", y = "value",
    expect_error = list(
      robust = list(
        component = "pairwise_comparisons",
        message = "The degrees of freedom must be greater than or equal to 2",
        reason = paste(
          "Yuen df below 2 in every pair: WRS2::lincon's Studentized maximum modulus",
          "table starts at df 2 and stops with an error below it"
        )
      ),
      bayes = list(
        component = "pairwise_comparisons",
        message = "arguments imply differing number of rows",
        reason = paste(
          "two values per group: pairwise_comparisons() fails while building the",
          "table of pairwise Bayes factors"
        )
      )
    )
  ),
  # Heavy ties: Dunn's tie correction, medians on ties, MAP on integers.
  phase2_cases("edge_ties", x = "group", y = "score"),
  # Small groups dominated by one value (9 of 13 are 1): some bootstrap resamples of
  # rank epsilon squared have every value tied.
  phase2_cases(
    "edge_ties_small", x = "group", y = "score", types = "nonparametric"
  ),
  phase2_cases("edge_negative", x = "group", y = "value"),
  phase2_cases("edge_outliers", x = "group", y = "value"),
  # Six groups, fifteen pairs (lincon at C = 15), counts with zeros and ties.
  phase2_cases("InsectSprays", x = "spray", y = "count"),
  phase2_cases("edge_null", x = "group", y = "value"),
  # Yuen df of about 1,400 in every lincon pair: past the last row (200) of the
  # Studentized maximum modulus table.
  phase2_cases("edge_large_n", x = "group", y = "value"),
  # Nine groups of six, 36 pairs: past the 28 comparisons of lincon's table, where
  # lincon simulates its critical values (probe-smm).
  phase2_cases("edge_k9", x = "group", y = "value", types = "robust"),
  # t1way F of about 1,500: 1 - pf is exactly 0, as are lincon's setosa pairs.
  phase2_cases("iris", x = "Species", y = "Petal.Length", label = "iris-petal"),
  # Two groups with Yuen p near 6e-15 (Petal.Length) and 5e-16 (Petal.Width), where
  # 1 - pt and the exact upper tail differ by 1% or more.
  phase2_cases(
    "iris_no_setosa", x = "Species", y = "Petal.Length", label = "iris_no_setosa-petal"
  ),
  phase2_cases(
    "iris_no_setosa", x = "Species", y = "Petal.Width",
    label = "iris_no_setosa-petalwidth", types = "robust"
  ),
  # Every other p.adjust method on Dunn's and lincon's p-values.
  unlist(lapply(
    c("bonferroni", "hochberg", "hommel", "BH", "BY", "fdr", "none"),
    function(method) {
      phase2_cases(
        "PlantGrowth", x = "group", y = "weight",
        label = paste0("PlantGrowth-padj-", method),
        types = c("nonparametric", "robust"),
        extra_args = list(p.adjust.method = method)
      )
    }
  ), recursive = FALSE),
  # A 90% confidence level: rank-biserial and rank epsilon squared CIs, the AKP and
  # t1way bootstrap CIs with alpha = 1 - 0.9, lincon's alpha, the Bayes intervals.
  phase2_cases(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-cl90",
    extra_args = list(conf.level = 0.90)
  ),
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-cl90",
    extra_args = list(conf.level = 0.90)
  ),
  # 10% trimming: Yuen, AKP, t1way, lincon and the trimmed-mean labels.
  phase2_cases(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-tr01",
    types = "robust", extra_args = list(tr = 0.1)
  ),
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-tr01",
    types = "robust", extra_args = list(tr = 0.1)
  ),
  # Three digits in every printed number.
  phase2_cases(
    "mtcars", x = "am", y = "mpg", label = "mtcars-am-digits3",
    extra_args = list(digits = 3L)
  ),
  phase2_cases(
    "mtcars", x = "cyl", y = "mpg", label = "mtcars-digits3",
    extra_args = list(digits = 3L)
  ),
  # Which pairs get brackets (bayes: whether the display argument applies at all).
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-display-all",
    extra_args = list(pairwise.display = "all")
  ),
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-display-ns",
    extra_args = list(pairwise.display = "non-significant")
  ),
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-display-none",
    extra_args = list(pairwise.display = "none")
  ),
  # No statistical subtitle: no test, no effect size, no bootstrap.
  phase2_cases(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  # One-sided Mann-Whitney on unbalanced groups without ties (9 + 12).
  phase2_cases(
    "edge_k2", x = "group", y = "value", label = "edge_k2-greater",
    types = "nonparametric", extra_args = list(alternative = "greater")
  ),
  phase2_cases(
    "edge_k2", x = "group", y = "value", label = "edge_k2-less",
    types = "nonparametric", extra_args = list(alternative = "less")
  ),
  # Whether the robust and Bayes tests take `alternative` at all.
  phase2_cases(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-greater",
    types = c("robust", "bayes"), extra_args = list(alternative = "greater")
  ),
  # Another Cauchy prior scale: the two-group Bayes factor, and with three groups the
  # Bayes ANOVA and whether the pairs keep 0.707.
  phase2_cases(
    "ToothGrowth", x = "supp", y = "len", label = "ToothGrowth-prior08",
    types = "bayes", extra_args = list(bf.prior = 0.8)
  ),
  phase2_cases(
    "PlantGrowth", x = "group", y = "weight", label = "PlantGrowth-prior08",
    types = "bayes", extra_args = list(bf.prior = 0.8)
  ),
  # Phase 3: heavy ties in small groups, so that some bootstrap resamples of AKP
  # (group a's winsorized variance 0 in about 12% of resamples) and of t1way's
  # explanatory effect size (every group's winsorized variance 0, about 5%) are
  # infinite: whether WRS2 returns +-Inf there and what ggstatsplot prints.
  phase2_cases("edge_heavy_ties2", x = "group", y = "value", types = "robust"),
  phase2_cases("edge_heavy_ties3", x = "group", y = "value", types = "robust"),

  # Phase 3 (withinstats): ggwithinstats, all four types, two conditions and three or
  # more. statsExpressions::complete_subjects() keeps the subjects observed in every
  # condition; the fixture records the subjects-by-conditions data it analyzed
  # (`within_data`).
  # Two conditions. R's sleep: ten subjects, a zero difference and tied differences.
  within_cases("sleep", x = "group", y = "extra", subject = "ID"),
  # Five subjects.
  edge_within("edge_within2_small"),
  # A declared order (pre, then post, the reverse of the sorted order) with an empty
  # level between the two: the sign of every difference.
  edge_within(
    "edge_within2_small", label = "edge_within2_small-declared",
    factor_levels = list(condition = c("pre", "empty", "post"))
  ),
  # Zero and tied absolute differences: Wilcoxon's handling of both.
  edge_within("edge_within2_ties"),
  edge_within("edge_within2_negative"),
  edge_within("edge_within2_outliers"),
  # The rows of edge_within2_outliers in a random order: the same results.
  edge_within("edge_within2_shuffled"),
  edge_within("edge_within2_null"),
  # 1,500 subjects.
  edge_within("edge_within2_large_n"),
  # A subject without a post row, a missing value, a row without a subject and a row
  # without a condition; subject ids that sort differently as text.
  edge_within("edge_within2_missing"),
  # Three or more conditions. ggstatsplot's bugs_long: four conditions, subjects with
  # missing values.
  within_cases("bugs_long", x = "condition", y = "desire", subject = "subject"),
  # A subject missing one of three conditions (s05 has no mid row).
  edge_within("edge_within_missing"),
  # Declared order with an empty level: pre, (empty), mid, post.
  edge_within(
    "edge_within_missing", label = "edge_within_missing-declared",
    factor_levels = list(condition = c("pre", "empty", "mid", "post"))
  ),
  # Four subjects, every one ranking the conditions a < b < c: Kendall's W is 1 in
  # every bootstrap resample, kendalls_w's CI fails ("arguments imply differing
  # number of rows: 1, 0"; probe-friedman) and ggstatsplot swallows the error, so
  # the nonparametric type has no subtitle; Durbin-Conover's t is infinite there.
  edge_within(
    "edge_within_small", expect_hidden_errors = list(nonparametric = "subtitle")
  ),
  # Integer scores with ties within subjects: Friedman, Durbin-Conover, Kendall's W.
  edge_within("edge_within_ties"),
  edge_within("edge_within_negative"),
  edge_within("edge_within_outliers"),
  # The rows of edge_within_outliers in a random order: the same results.
  edge_within("edge_within_shuffled"),
  # Condition means equal: F near 0.
  edge_within("edge_within_null"),
  # 1,100 subjects.
  edge_within("edge_within_large_n"),
  # Five conditions, ten pairs: sphericity correction, many pairs.
  edge_within("edge_within_k5"),
  # Greenhouse-Geisser epsilon near its lower bound.
  edge_within("edge_within_nonspherical"),
  # Every other p.adjust method on the paired t, Durbin-Conover and rmmcp p-values.
  unlist(lapply(P_ADJUST_OTHERS, function(method) {
    within_cases(
      "bugs_long", x = "condition", y = "desire", subject = "subject",
      label = paste0("bugs_long-padj-", method),
      types = c("parametric", "nonparametric", "robust"),
      extra_args = list(p.adjust.method = method)
    )
  }), recursive = FALSE),
  # A 90% confidence level: whether it reaches each effect size and its bootstrap.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-cl90",
    extra_args = list(conf.level = 0.90)
  ),
  within_cases(
    "bugs_long", x = "condition", y = "desire", subject = "subject",
    label = "bugs_long-cl90", extra_args = list(conf.level = 0.90)
  ),
  # 10% trimming: yuend, dep.effect, rmanova, wmcpAKP, rmmcp, trimmed-mean labels.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-tr01",
    types = "robust", extra_args = list(tr = 0.1)
  ),
  within_cases(
    "bugs_long", x = "condition", y = "desire", subject = "subject",
    label = "bugs_long-tr01", types = "robust", extra_args = list(tr = 0.1)
  ),
  # Which pairs get brackets.
  unlist(lapply(c("all", "non-significant", "none"), function(display) {
    within_cases(
      "bugs_long", x = "condition", y = "desire", subject = "subject",
      label = paste0(
        "bugs_long-display-",
        c(all = "all", `non-significant` = "ns", none = "none")[[display]]
      ),
      extra_args = list(pairwise.display = display)
    )
  }), recursive = FALSE),
  # Three digits in every printed number.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-digits3",
    extra_args = list(digits = 3L)
  ),
  within_cases(
    "bugs_long", x = "condition", y = "desire", subject = "subject",
    label = "bugs_long-digits3", extra_args = list(digits = 3L)
  ),
  # No statistical subtitle.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  within_cases(
    "bugs_long", x = "condition", y = "desire", subject = "subject",
    label = "bugs_long-nosubtitle", extra_args = list(results.subtitle = FALSE)
  ),
  # One-sided alternatives: the paired t and Wilcoxon p, the sidedness of their
  # effect-size CIs; and whether yuend and the paired Bayes factor take them at all.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-greater",
    extra_args = list(alternative = "greater")
  ),
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-less",
    types = c("parametric", "nonparametric"), extra_args = list(alternative = "less")
  ),
  # Another Cauchy prior scale: the paired Bayes factor (subtitle and caption), and
  # with four conditions the Bayes ANOVA and whether the pairs keep 0.707.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-prior08",
    types = c("parametric", "bayes"), extra_args = list(bf.prior = 0.8)
  ),
  within_cases(
    "bugs_long", x = "condition", y = "desire", subject = "subject",
    label = "bugs_long-prior08", types = "bayes", extra_args = list(bf.prior = 0.8)
  ),
  # No Bayes caption.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-nobf",
    types = "parametric", extra_args = list(bf.message = FALSE)
  ),
  # The arguments ggwithinstats adds: the lines joining each subject's points (two
  # conditions only) and the centrality values.
  within_cases(
    "sleep", x = "group", y = "extra", subject = "ID", label = "sleep-nopaths",
    types = "parametric", extra_args = list(point.path = FALSE, centrality.path = FALSE)
  ),
  within_cases(
    "bugs_long", x = "condition", y = "desire", subject = "subject",
    label = "bugs_long-nopaths", types = "parametric",
    extra_args = list(centrality.path = FALSE)
  ),

  # Review of Phases 0-3 (STATUS.md T7): small hand-written cases that decide open
  # rules, each with its bootstrap reference where it has one.
  # About 2% of AKP resamples are 0/0: group a's resample is all 5 after winsorizing
  # (winsorized SD 0) and its trimmed mean equals b's (D65). (edge_akp_undefined,
  # about 17%: in HELD_OUT_CASES below.)
  phase2_cases("edge_akp_undefined2", x = "group", y = "value", types = "robust"),
  # Three groups of three sharing the value 1: about 8.8% of the stratified resamples
  # of rank epsilon squared have every value tied (D54 in R itself).
  phase2_cases("edge_tied3_small", x = "group", y = "value", types = "nonparametric"),
  # tr 0.29 with n = 100: 0.29 * 100 is 28.999999999999996 in floating point, so
  # floor(tr n) trims 28 values from each end, not 29. Values: round(20 + 4 sin(0.7 i)
  # + 0.05 i, 2) for a and round(22 + 5 cos(0.53 i) + 0.03 i, 2) for b, i = 1..100.
  phase2_cases(
    "edge_tr029", x = "group", y = "value", types = "robust", extra_args = list(tr = 0.29)
  ),
  # Two values against fifty (b: round(10 + 2 sin(1.3 i), 2), i = 1..50): Welch's df
  # about 1.002, Hedges' g at a df just above 1.
  parametric_case("edge_n2_vs50", x = "group", y = "value"),
  # (Three subjects, four conditions: in HELD_OUT_CASES below.)
  # Friedman with every subject constant across conditions (0/0), and with five of
  # eight subjects constant (Kendall's W bootstrap draws resamples of only those).
  # All tied: the subtitle fails ("missing value where TRUE/FALSE needed") and
  # ggstatsplot swallows that; then Durbin-Conover's table has no rows and
  # pairwise_comparisons() stops.
  edge_within(
    "edge_rm_all_tied", types = "nonparametric",
    expect_hidden_errors = list(nonparametric = "subtitle"),
    expect_error = list(nonparametric = list(
      component = "pairwise_comparisons",
      message = "arguments imply differing number of rows: 0, 1",
      reason = paste(
        "every subject constant across conditions: Friedman's statistic is 0/0, the",
        "subtitle fails silently, and the Durbin-Conover table has no rows"
      )
    ))
  ),
  edge_within("edge_rm_some_tied", types = "nonparametric"),

  # Phase 4 (histostats): gghistostats, all four types. The fixture records the
  # values it analyzed (`analyzed_data`), the histogram as ggplot2 built it and the
  # centrality line and label (`centrality`).
  # mtcars mpg against the default test value 0, far below every value (t about 19).
  histo_cases("mtcars", x = "mpg"),
  # A test value inside the data, equal to no value (mtcars) or to seven (iris).
  histo_cases("mtcars", x = "mpg", label = "mtcars-tv20", extra_args = list(test.value = 20)),
  histo_cases(
    "iris", x = "Sepal.Length", label = "iris-sepal", extra_args = list(test.value = 5.8)
  ),
  histo_cases("ToothGrowth", x = "len", extra_args = list(test.value = 18)),
  # Far above every value: |t| about 80.
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-tv100", extra_args = list(test.value = 100)
  ),
  # Missing values in x.
  histo_cases("bugs_long", x = "desire", extra_args = list(test.value = 5)),
  # Five values.
  edge_one("edge_one_small", 3),
  # Integer scores, five of them equal to the test value (Wilcoxon's zeros and ties).
  edge_one("edge_one_ties", 3),
  edge_one("edge_one_negative", -10),
  # One value six times as far out as the rest.
  edge_one("edge_one_outlier", 10),
  # Mean 0.00625 above the test value: t near 0.
  edge_one("edge_one_null", 50),
  # 2,000 values: round(100 + 15 sin(0.37 i) + 5 cos(1.13 i), 2), i = 1..2000.
  edge_one("edge_one_large_n", 100),
  # Missing values in the analyzed column and in another one.
  edge_one("edge_one_missing", 10),
  # Arguments, on ToothGrowth against 18.
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-cl90",
    extra_args = list(test.value = 18, conf.level = 0.90)
  ),
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-tr01", types = "robust",
    extra_args = list(test.value = 18, tr = 0.1)
  ),
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-digits3",
    extra_args = list(test.value = 18, digits = 3L)
  ),
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-nosubtitle",
    extra_args = list(test.value = 18, results.subtitle = FALSE)
  ),
  # One-sided: the t and Wilcoxon p and the sidedness of their effect-size CIs, and
  # whether trimcibt and the Bayes factor take `alternative` at all.
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-greater",
    extra_args = list(test.value = 18, alternative = "greater")
  ),
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-less",
    types = c("parametric", "nonparametric"),
    extra_args = list(test.value = 18, alternative = "less")
  ),
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-prior08", types = c("parametric", "bayes"),
    extra_args = list(test.value = 18, bf.prior = 0.8)
  ),
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-nobf", types = "parametric",
    extra_args = list(test.value = 18, bf.message = FALSE)
  ),
  # A given bin width (the default is range / sqrt(n)), and no centrality line.
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-binwidth2", types = "parametric",
    extra_args = list(test.value = 18, binwidth = 2)
  ),
  histo_cases(
    "ToothGrowth", x = "len", label = "ToothGrowth-nocentrality", types = "parametric",
    extra_args = list(test.value = 18, centrality.plotting = FALSE)
  ),

  # Phase 4 (scatterstats): ggscatterstats, all four types. The fixture records the
  # complete pairs it analyzed (`analyzed_data`), correlation::correlation's own row
  # (`correlation`), and the smooth line, its band and the marginal histograms as
  # ggplot2 built them.
  scatter_cases("mtcars", x = "wt", y = "mpg"),
  # 150 pairs with ties in both columns.
  scatter_cases("iris", x = "Sepal.Length", y = "Petal.Length"),
  edge_corr("edge_corr_small"),
  edge_corr("edge_corr_ties"),
  # r = 1 and r = -1: BayesFactor's correlation Bayes factor is NaN there, and
  # ggstatsplot prints log(BF01) as an empty string.
  edge_corr(
    "edge_corr_perfect",
    expect_na_values = list(parametric = "caption", bayes = "subtitle")
  ),
  scatter_cases(
    "edge_corr_perfect", x = "x", y = "y_neg", label = "edge_corr_perfect-neg",
    expect_na_values = list(parametric = "caption", bayes = "subtitle")
  ),
  edge_corr("edge_corr_null"),
  # One point that turns a strong positive correlation negative.
  edge_corr("edge_corr_outlier"),
  # 2,000 pairs: x = round(50 + 10 sin(0.37 i) + 3 cos(1.7 i), 2), y = round(20 +
  # 3 sin(0.37 i) + 4 sin(2.3 i + 0.4), 2), i = 1..2000.
  edge_corr("edge_corr_large_n"),
  # Missing x, missing y and both.
  edge_corr("edge_corr_missing"),
  # Constant x: no correlation is defined. Pearson's r is NA ("the standard deviation
  # is zero") and the parametric subtitle prints NA; the Bayesian correlation stops
  # ("No valid values found"), which ggstatsplot swallows (the parametric caption, the
  # bayes subtitle). The OLS line has no slope, and geom_smooth(na.rm = TRUE) draws
  # nothing.
  edge_corr(
    "edge_corr_constant_x",
    expect_hidden_errors = list(parametric = "caption", bayes = "subtitle"),
    expect_na_values = list(
      parametric = "subtitle", nonparametric = "subtitle", robust = "subtitle"
    ),
    expect_empty_layers = list(
      parametric = "GeomSmooth", nonparametric = "GeomSmooth", robust = "GeomSmooth",
      bayes = "GeomSmooth"
    )
  ),
  # Arguments, on mtcars wt and mpg.
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-cl90", extra_args = list(conf.level = 0.90)
  ),
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-tr01", types = "robust",
    extra_args = list(tr = 0.1)
  ),
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-digits3", extra_args = list(digits = 3L)
  ),
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-prior08", types = c("parametric", "bayes"),
    extra_args = list(bf.prior = 0.8)
  ),
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-nobf", types = "parametric",
    extra_args = list(bf.message = FALSE)
  ),
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-nomarginal", types = "parametric",
    extra_args = list(marginal = FALSE)
  ),
  # Another smoother: smooth.line.args replaces the whole default list.
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-loess", types = "parametric",
    extra_args = list(smooth.line.args = list(method = "loess", formula = "y ~ x"))
  ),
  # Point labels: every point, and the points label.expression selects.
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-labels", types = "parametric",
    extra_args = list(label.var = "model")
  ),
  scatter_cases(
    "mtcars", x = "wt", y = "mpg", label = "mtcars-labelexpr", types = "parametric",
    extra_args = list(label.var = "model", label.expression = "mpg > 25 | wt > 5")
  ),
  # Small samples the foundation package asked for: two and three values (values of
  # edge_one_small's first rows), three and four pairs (edge_corr_small's first rows),
  # and a point label that is missing (the third of six points has no name).
  edge_one("edge_one_n2", 3),
  edge_one("edge_one_n3", 3),
  # Ten values of 5 against 3: the default bin width when max = min, and which types
  # answer on constant data; then a given bin width. t.test stops ("data are
  # essentially constant") in the parametric subtitle and caption and in the bayes
  # subtitle, and ggstatsplot swallows it; trimcibt's interval is NA (robust).
  edge_one(
    "edge_one_constant", 3,
    expect_hidden_errors = list(parametric = c("subtitle", "caption"), bayes = "subtitle"),
    expect_na_values = list(robust = "subtitle")
  ),
  histo_cases(
    "edge_one_constant", x = "value", label = "edge_one_constant-binwidth1",
    extra_args = list(test.value = 3, binwidth = 1),
    expect_hidden_errors = list(parametric = c("subtitle", "caption"), bayes = "subtitle"),
    expect_na_values = list(robust = "subtitle")
  ),
  # Three pairs: the Fisher-z interval of r has 1 / sqrt(n - 3) = Inf, and R prints
  # it as NA (parametric, robust).
  edge_corr(
    "edge_corr_n3", expect_na_values = list(parametric = "subtitle", robust = "subtitle")
  ),
  edge_corr("edge_corr_n4"),
  scatter_cases(
    "edge_corr_label_missing", x = "x", y = "y", types = "parametric",
    extra_args = list(label.var = "name")
  ),

  # Phase 4 (barstats): ggbarstats, and ggpiestats for the one-way design. The fixture
  # records the table it analyzed (`contingency_data`), every extract_stats() table
  # (descriptive_data: the labels inside the bars; one_sample_data: the per-bar
  # proportion tests), the layers as ggplot2 built them and the plot's labels. The
  # edge datasets are hand-written; each one's table is in its comment.
  # Two-way, all four types: 3 x 2 (pairwise Fisher tests between the three levels of
  # cyl, each a 2 x 2 table), 2 x 3, 2 x 2, and Titanic's 4 x 2 with n = 2,201.
  bar_cases("mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am", types = TYPES),
  bar_cases("mtcars", x = "am", y = "cyl", label = "mtcars-am-cyl", types = TYPES),
  bar_cases("mtcars", x = "am", y = "vs", label = "mtcars-am-vs", types = TYPES),
  bar_cases("Titanic_full", x = "Class", y = "Survived", label = "Titanic_full-class", types = TYPES),
  # 2 x 4: four bars, no pairwise tests (x has two levels).
  bar_cases("Titanic_full", x = "Survived", y = "Class", label = "Titanic_full-survived"),
  # The same table as counts: as.data.frame(Titanic), 32 rows of Class, Sex, Age,
  # Survived and Freq (8 of them 0), aggregated from Titanic_full.csv.
  bar_cases(
    "Titanic_counts", x = "Class", y = "Survived", label = "Titanic_counts-class",
    extra_args = list(counts = "Freq")
  ),
  # One-way (goodness of fit against equal proportions), all four types, from both
  # functions.
  bar_cases("mtcars", x = "cyl", label = "mtcars-cyl", types = TYPES),
  pie_cases("mtcars", x = "cyl", label = "mtcars-cyl", types = TYPES),
  # Paired (McNemar), all four types. edge_bar_paired: 40 subjects, before x after
  # (yes, yes) 10, (yes, no) 12, (no, yes) 4, (no, no) 14.
  edge_paired("edge_bar_paired", types = TYPES),
  # Arguments, on mtcars cyl by am.
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-cl90",
    extra_args = list(conf.level = 0.90)
  ),
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-digits3",
    extra_args = list(digits = 3L)
  ),
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  # proportion.test defaults to results.subtitle; given explicitly it is drawn without
  # the subtitle.
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-nosubtitle-proptest",
    extra_args = list(results.subtitle = FALSE, proportion.test = TRUE)
  ),
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-noproptest",
    extra_args = list(proportion.test = FALSE)
  ),
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-nobf",
    extra_args = list(bf.message = FALSE)
  ),
  # One-sided: the Cramer's V interval, and Fisher's test on each 2 x 2 pair.
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-greater",
    extra_args = list(alternative = "greater")
  ),
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-less",
    extra_args = list(alternative = "less")
  ),
  # The labels inside the bars.
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-counts",
    extra_args = list(label = "counts")
  ),
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-both",
    extra_args = list(label = "both")
  ),
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-digitsperc1",
    extra_args = list(digits.perc = 1L)
  ),
  # ratio with two variables: the per-bar proportion tests (order of cyl's levels 4,
  # 6, 8).
  bar_cases(
    "mtcars", x = "cyl", y = "am", label = "mtcars-cyl-am-ratio",
    extra_args = list(ratio = c(0.25, 0.25, 0.5))
  ),
  # Every other p.adjust method on the six pairwise Fisher tests of Titanic's classes.
  unlist(lapply(P_ADJUST_OTHERS, function(method) {
    bar_cases(
      "Titanic_full", x = "Class", y = "Survived",
      label = paste0("Titanic_full-class-padj-", method),
      extra_args = list(p.adjust.method = method)
    )
  }), recursive = FALSE),
  # One-way arguments, on mtcars cyl (levels 4, 6, 8).
  bar_cases("mtcars", x = "cyl", label = "mtcars-cyl-cl90", extra_args = list(conf.level = 0.90)),
  bar_cases("mtcars", x = "cyl", label = "mtcars-cyl-digits3", extra_args = list(digits = 3L)),
  bar_cases(
    "mtcars", x = "cyl", label = "mtcars-cyl-nosubtitle",
    extra_args = list(results.subtitle = FALSE)
  ),
  bar_cases("mtcars", x = "cyl", label = "mtcars-cyl-nobf", extra_args = list(bf.message = FALSE)),
  bar_cases(
    "mtcars", x = "cyl", label = "mtcars-cyl-greater", extra_args = list(alternative = "greater")
  ),
  bar_cases("mtcars", x = "cyl", label = "mtcars-cyl-both", extra_args = list(label = "both")),
  bar_cases(
    "mtcars", x = "cyl", label = "mtcars-cyl-digitsperc1", extra_args = list(digits.perc = 1L)
  ),
  # ratio: in the order of the levels; one that does not sum to 1 and one of the wrong
  # length.
  bar_cases(
    "mtcars", x = "cyl", label = "mtcars-cyl-ratio", extra_args = list(ratio = c(0.5, 0.25, 0.25))
  ),
  pie_cases(
    "mtcars", x = "cyl", label = "mtcars-cyl-ratio", extra_args = list(ratio = c(0.5, 0.25, 0.25))
  ),
  # chisq.test refuses both ("probabilities must sum to 1.", "'x' and 'p' must have
  # the same number of elements") and ggstatsplot swallows it: no subtitle. The
  # one-way Bayes factor normalizes c(2, 1, 1) (a caption is drawn) and stops on the
  # short one.
  bar_cases(
    "mtcars", x = "cyl", label = "mtcars-cyl-ratio-sum4", extra_args = list(ratio = c(2, 1, 1)),
    expect_hidden_errors = list(parametric = "subtitle")
  ),
  bar_cases(
    "mtcars", x = "cyl", label = "mtcars-cyl-ratio-short", extra_args = list(ratio = c(0.5, 0.5)),
    expect_hidden_errors = list(parametric = c("subtitle", "caption"))
  ),
  # Paired arguments.
  edge_paired(
    "edge_bar_paired", label = "edge_bar_paired-cl90", extra_args = list(conf.level = 0.90)
  ),
  edge_paired(
    "edge_bar_paired", label = "edge_bar_paired-less", extra_args = list(alternative = "less")
  ),
  # Edge data. edge_bar_zero_cell: x a, b by y p, q, r: a (6, 0, 4), b (3, 7, 5); a
  # zero cell and expected counts below 5.
  edge_bar("edge_bar_zero_cell"),
  # Declared levels with an empty level in x (b, empty, a) and in y (r, none, q, p).
  edge_bar(
    "edge_bar_zero_cell", label = "edge_bar_zero_cell-declared",
    factor_levels = list(x = c("b", "empty", "a"), y = c("r", "none", "q", "p"))
  ),
  # One-way with an empty declared level: a category with a count of 0.
  bar_cases(
    "edge_bar_zero_cell", x = "x", label = "edge_bar_zero_cell-oneway-declared",
    factor_levels = list(x = c("b", "empty", "a"))
  ),
  # 3 x 3, n = 12: a (3, 1, 0), b (0, 2, 1), c (1, 0, 4); expected counts below 1.
  edge_bar("edge_bar_small_expected"),
  # 2 x 2 with a count of 1: a (1, 9), b (8, 4).
  edge_bar("edge_bar_2x2_one"),
  # 5 x 4, n = 4,070: the count of (x_i, y_j) is 60 + 15 i + 25 j + 40 ((i j) mod 3),
  # i = 1..5 (a..e), j = 1..4 (w..z), rows in x-major order.
  edge_bar("edge_bar_large"),
  # Missing x (row 3), y (row 7), both (row 11) and weight w (row 15).
  edge_bar("edge_bar_missing"),
  edge_bar(
    "edge_bar_missing", label = "edge_bar_missing-counts", extra_args = list(counts = "w")
  ),
  bar_cases("edge_bar_missing", x = "x", label = "edge_bar_missing-oneway"),
  # Perfect association (3 x 3 diagonal: 8, 6, 7) and exact independence (a (4, 8,
  # 12), b (6, 12, 18): chi-squared exactly 0).
  edge_bar("edge_bar_perfect"),
  edge_bar("edge_bar_independent"),
  # Degenerate designs on twelve rows: `one` and `single` hold one level each; x3 by
  # y2 is a (3, 0), b (4, 0), c (0, 5), so bar q holds only c.
  # One category: chisq.test stops ("'x' must at least have 2 elements"), and the
  # one-way Bayes factor returns nothing for a one-cell table.
  bar_cases(
    "edge_bar_degenerate", x = "one", label = "edge_bar_degenerate-oneway-one",
    expect_hidden_errors = list(parametric = "subtitle"),
    expect_null_outputs = list(parametric = "caption")
  ),
  # One row (x has one level): chisq.test turns a 1 x 2 table into a goodness-of-fit
  # test of y, and cramers_v refuses it ("not supported for goodness-of-fit tests");
  # contingencyTableBF refuses it too; each bar's test has one category (NA rows).
  bar_cases(
    "edge_bar_degenerate", x = "one", y = "y2", label = "edge_bar_degenerate-one-row",
    expect_hidden_errors = list(parametric = c("subtitle", "caption", "proportion_test")),
    expect_na_values = list(parametric = "one_sample")
  ),
  # One column (y has one level): the same goodness-of-fit turn and refusal; no
  # caption (bf.condition: y not of exactly one level); the bar's own test runs.
  bar_cases(
    "edge_bar_degenerate", x = "x3", y = "single", label = "edge_bar_degenerate-one-column",
    expect_hidden_errors = list(parametric = "subtitle")
  ),
  # One cell.
  bar_cases(
    "edge_bar_degenerate", x = "one", y = "single", label = "edge_bar_degenerate-one-cell",
    expect_hidden_errors = list(parametric = c("subtitle", "proportion_test")),
    expect_na_values = list(parametric = "one_sample")
  ),
  # Pairs a-b: after droplevels the pair's table is 2 x 1 and fisher.test stops; the
  # whole pairwise table is lost.
  bar_cases(
    "edge_bar_degenerate", x = "x3", y = "y2", label = "edge_bar_degenerate-single-x",
    expect_hidden_errors = list(parametric = "pairwise_comparisons")
  ),
  # Percent labels: bars of 8 (a 1, b 3, c 4: 12.5%, 37.5%, 50%), 16 (1, 5, 10: 6.25%,
  # 31.25%, 62.5%), 3 (1 each: 33.3%) and 7 (1, 2, 4).
  edge_bar("edge_bar_rounding"),
  edge_bar(
    "edge_bar_rounding", label = "edge_bar_rounding-digitsperc1",
    extra_args = list(digits.perc = 1L)
  ),
  edge_bar(
    "edge_bar_rounding", label = "edge_bar_rounding-digitsperc2",
    extra_args = list(digits.perc = 2L)
  ),
  edge_bar("edge_bar_rounding", label = "edge_bar_rounding-both", extra_args = list(label = "both")),
  edge_bar(
    "edge_bar_rounding", label = "edge_bar_rounding-counts", extra_args = list(label = "counts")
  ),
  # Weights: w holds zeros (every row of c), w_frac a weight of 2.5.
  edge_bar("edge_bar_counts", label = "edge_bar_counts", extra_args = list(counts = "w")),
  # tidyr::uncount() refuses a weight of 2.5 while ggstatsplot prepares the data.
  edge_bar(
    "edge_bar_counts", label = "edge_bar_counts-frac", extra_args = list(counts = "w_frac"),
    expect_error = list(parametric = list(
      component = "data",
      message = "Can't convert from `weights` <double> to <integer> due to loss of precision",
      reason = "a weight that is not a whole number: tidyr::uncount() cannot repeat a row 2.5 times"
    ))
  ),
  # Paired edge data. 3 x 3 (low, mid, high; McNemar-Bowker): before low (6, 5, 2),
  # mid (1, 7, 6), high (1, 3, 8) after low, mid, high.
  edge_paired("edge_bar_paired3"),
  # Every subject answers the same twice (no discordant pair): McNemar's statistic is
  # 0/0 and cohens_g stops ("not enough data"), which ggstatsplot swallows.
  edge_paired(
    "edge_bar_paired_agree", expect_hidden_errors = list(parametric = "subtitle")
  ),
  # One discordant pair (yes, no) among ten subjects.
  edge_paired("edge_bar_paired_small"),
  # After has a level (maybe) that before lacks: a 2 x 3 table, which mcnemar.test
  # refuses ("'x' must be square").
  edge_paired(
    "edge_bar_paired_unequal", expect_hidden_errors = list(parametric = "subtitle")
  ),
  # The same with both variables declared on maybe, no, yes: a square table.
  edge_paired(
    "edge_bar_paired_unequal", label = "edge_bar_paired_unequal-declared",
    factor_levels = list(before = c("maybe", "no", "yes"), after = c("maybe", "no", "yes"))
  )
)

# Review round 3: cases that settle what R does where the reviews of Phases 3 and 4
# could only reason about it (W9, Z1, B5, C3, D108 and the one-sided Fisher pairs).
# They are assigned to shards after the cases above, so that adding them moves no
# earlier case to another shard (assign_shards()). Each dataset is hand-written; its
# values or table are in the comment. Cases whose R result differs from inferplot's
# are in HELD_OUT_CASES below.
ROUND3_CASES <- c(
  # Values equal to the test value 0.3 only as written. edge_one_near_mu, ten values
  # per column: `literal` is 0.3, 0.1, 0.5, 0.7, 0.2, 0.9, 0.4, 0.6, 0.8, 1.1 (the
  # literal 0.3, an exact zero of x - 0.3); `sum`, `below`, `both` and `sum_ties` are
  # in HELD_OUT_CASES.
  histo_cases(
    "edge_one_near_mu", x = "literal", label = "edge_one_near_mu-literal",
    types = "nonparametric", extra_args = list(test.value = 0.3)
  ),
  # Cramer's V and its interval at 19 to 60 df (review B5), x a and b by y01 ... yc
  # through counts n: edge_bar_wide20 a 1 + (2j mod 7), b 1 + ((5j + 3) mod 9), j =
  # 1..20 (n 187); edge_bar_wide40 and edge_bar_wide61 a 1 + (j mod 7), b 1 + ((2j +
  # 3) mod 9), j = 1..40 (n 358) and 1..61 (n 553); edge_bar_wide61r the reviewer's
  # 2 x 61 table (n 585, chi-squared 122.4, p 3.6e-06; reviewBB c02_outside.py,
  # numpy default_rng(7), trial 9).
  edge_bar("edge_bar_wide20", extra_args = list(counts = "n")),
  edge_bar("edge_bar_wide40", extra_args = list(counts = "n")),
  edge_bar("edge_bar_wide61", extra_args = list(counts = "n")),
  edge_bar("edge_bar_wide61r", extra_args = list(counts = "n")),
  # Exact zeros (review C3): counts 8, 2, 7, 8 against ratio 0.32, 0.08, 0.28, 0.32,
  # an exact fit: chi-squared is 0, R's doubles give 1.13e-31 (Pearson's C 6.71e-17),
  # and ggstatsplot prints that (probe-bartables). edge_bar_zero_v is in
  # HELD_OUT_CASES; 49 levels of 2 each (edge_bar_equal49, chi-squared 1.21e-30) only
  # in probe-bartables: ggbarstats stops on 49 levels ("`n` must be a whole number
  # between 1 and 24, not the number 49.") after computing its subtitle.
  bar_cases(
    "edge_bar_exact_fit", x = "x", label = "edge_bar_exact_fit",
    extra_args = list(counts = "n", ratio = c(0.32, 0.08, 0.28, 0.32))
  ),
  # Percent labels at rounding ties (D108): x a and b, one bar per (n, k) with a k and
  # b n - k, bar n2000k001 and so on: n 2000 with k 1, 3, 10, 21, 30, 39, 89, 91, 109;
  # n 1000 with k 5, 15; n 200 with k 1, 3; n 40 with k 1. (k / n) x 100 is a tie at
  # one digit for odd k of 2000, at none for the rest; digits.perc 1 is in
  # HELD_OUT_CASES. The per-bar tests against ratio 0.05, 0.95 (against equal shares
  # their p underflows to 0, which statsExpressions stores as 2.2e-308).
  edge_bar(
    "edge_bar_ties", label = "edge_bar_ties-digitsperc0",
    extra_args = list(counts = "n", digits.perc = 0L, ratio = c(0.05, 0.95))
  ),
  edge_bar(
    "edge_bar_ties", label = "edge_bar_ties-digitsperc2",
    extra_args = list(counts = "n", digits.perc = 2L, ratio = c(0.05, 0.95))
  ),
  # One-sided Fisher pairs on reduced tables: x a (p 6, q 1, r 0), b (2, 7, 0), c (5,
  # 0, 2), d (1, 0, 6), so the pairs a-b (p, q) and c-d (p, r) are 2 x 2 tables and the
  # other four 2 x 3. With x declared d, b, c, a they are in HELD_OUT_CASES.
  unlist(lapply(c("two.sided", "greater", "less"), function(alternative) {
    edge_bar(
      "edge_bar_reduced", label = paste0("edge_bar_reduced-", alternative),
      extra_args = list(counts = "n", alternative = alternative)
    )
  }), recursive = FALSE)
)
ROUND3_CASE_NAMES <- vapply(ROUND3_CASES, function(case) case$name, character(1L))
CASES <- c(CASES, ROUND3_CASES)

# Cases run like the others but written into probe-heldout instead of a fixture of
# their own: R's result differs from what inferplot computes today by a rule the
# lead has yet to decide, so they must not enter the parity tests (which collect
# every fixture of a function) until then (review of Phases 0-3, T7).
HELD_OUT_CASES <- c(
  # About 17% of AKP resamples are 0/0 (a: 5,5,5,5,5,5,6,7,8,9; b: ten 5s). WRS2 drops
  # them when it sorts but keeps the bound positions of all B, so the upper bound is
  # NA (probe-bootrules) and ggstatsplot prints CI [0.22, NA]; inferplot leaves them
  # out and counts positions over the rest (D65), a finite bound.
  phase2_cases("edge_akp_undefined", x = "group", y = "value", types = "robust"),
  # Three subjects, four conditions: fewer subjects than conditions, so the covariance
  # of the three contrasts is singular. afex then computes no sphericity correction
  # ("corresponding non-sphericity tests and corrections not available") and
  # ggstatsplot prints the uncorrected F(3, 6); inferplot applies the
  # Greenhouse-Geisser epsilon of the singular matrix (D62).
  edge_within("edge_rm_n3k4", types = "parametric"),
  # Review round 3 (W9, D88, D105): a value equal to the test value 0.3 as written but
  # not in binary. edge_one_near_mu: `sum` is `literal` with 0.1 + 0.2
  # (0.30000000000000004) for its 0.3, `below` with 0.7 - 0.4 (0.29999999999999993);
  # `both` is 0.3, 0.1 + 0.2, 0.1, 0.5, 0.7, 0.2, 0.9, 0.4, 0.6, 0.8; `sum_ties` is
  # 0.1 + 0.2 twice, 0.5, 0.5, 0.8, 0.8, 0.2, 0.9, 1.1, 0.1. wilcox.test ranks
  # x - 0.3 = +-2^-54 as the smallest nonzero difference (probe-nearzero); inferplot
  # counts such a value as a zero (D105).
  unlist(lapply(c("sum", "below", "both", "sum_ties"), function(column) {
    histo_cases(
      "edge_one_near_mu", x = column, label = paste0("edge_one_near_mu-", column),
      types = "nonparametric", extra_args = list(test.value = 0.3)
    )
  }), recursive = FALSE),
  # Review round 3 (Z1; D65, D105): ratings 1 to 5 against the test value in two
  # writings. edge_one_ratings: `whole` is 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 3, 2, 4, 3, 3
  # (trimmed mean 3) and `tenths` the same in tenths (0.1 to 0.5), against 2.5 and
  # 0.25; `centred_whole` is 1, 2, 3, 3, 4, 5, 2, 4, 3, 3, 1, 5, 3, 2, 4 and
  # `centred_tenths` the same in tenths, against their trimmed mean, 3 and 0.3.
  # trimcibt gives the same in both writings (probe-trimcibt): about 11% of its T*
  # are 0/0 (whole, tenths; 2.5% centred), which sort() drops while p still divides
  # by all B and the bound takes the round(0.95 B)-th, so the interval is NA (whole,
  # tenths) and p at t = 0 is 0.975 (centred). inferplot leaves 0/0 resamples out of
  # both (D65): p 0.117 with [2.35, 3.65], and p 1 at t = 0.
  histo_cases(
    "edge_one_ratings", x = "whole", label = "edge_one_ratings-whole", types = "robust",
    extra_args = list(test.value = 2.5), expect_na_values = list(robust = "subtitle")
  ),
  histo_cases(
    "edge_one_ratings", x = "tenths", label = "edge_one_ratings-tenths", types = "robust",
    extra_args = list(test.value = 0.25), expect_na_values = list(robust = "subtitle")
  ),
  histo_cases(
    "edge_one_ratings", x = "centred_whole", label = "edge_one_ratings-centred_whole",
    types = "robust", extra_args = list(test.value = 3)
  ),
  histo_cases(
    "edge_one_ratings", x = "centred_tenths", label = "edge_one_ratings-centred_tenths",
    types = "robust", extra_args = list(test.value = 0.3)
  ),
  # Review round 3 (C3; D107): a (p 0, q 1), b (p 5, q 4), whose adjusted Cramer's V is
  # exactly 0 (chi-squared 10/9 = df n / (n - 1)). R's doubles leave chi-squared / n -
  # 1 / 9 = 2^-56, and ggstatsplot prints V = 5.59e-09 (probe-bartables); inferplot's
  # own rounding prints 3.95e-09.
  edge_bar("edge_bar_zero_v", extra_args = list(counts = "n")),
  # Review round 3 (D108): R's round() at one digit picks between the two candidates
  # rounded to that digit, the nearer one to the double and the even one on a tie of
  # their distances, so (1 / 2000) x 100 prints 0%; Python's round() of the same
  # double gives 0.1 (probe-round).
  edge_bar(
    "edge_bar_ties", label = "edge_bar_ties-digitsperc1",
    extra_args = list(counts = "n", digits.perc = 1L, ratio = c(0.05, 0.95))
  ),
  # Review round 3 (one-sided Fisher pairs; SPEC section 4): edge_bar_reduced with x
  # declared d, b, c, a. Every pair's p, adjusted p and V equal inferplot's, and a pair
  # of 2 x 2 has the earlier declared level as group1 and first row (b-a, d-c); but R
  # lists the pairs sorted as text (b-a, b-c, c-a, d-a, d-b, d-c), inferplot in level
  # order (d-b, d-c, d-a, b-c, b-a, c-a).
  unlist(lapply(c("two.sided", "greater", "less"), function(alternative) {
    edge_bar(
      "edge_bar_reduced", label = paste0("edge_bar_reduced-declared-", alternative),
      extra_args = list(counts = "n", alternative = alternative),
      factor_levels = list(x = c("d", "b", "c", "a"))
    )
  }), recursive = FALSE)
)

case_names <- vapply(CASES, function(case) case$name, character(1L))
all_names <- c(case_names, vapply(HELD_OUT_CASES, function(case) case$name, character(1L)))
if (anyDuplicated(all_names)) stop("duplicate case names: ", all_names[duplicated(all_names)])

# Arguments whose consequences the bookkeeping below knows (which hidden calls
# ggbetweenstats and ggwithinstats make, which outputs they draw). Fail closed on any
# other. ggwithinstats adds subject.id (always given) and the two path flags, which
# change only the plot's layers.
KNOWN_ARGS <- c(
  "x", "y", "type", "alternative", "conf.level", "digits", "p.adjust.method",
  "pairwise.display", "results.subtitle", "bf.message", "bf.prior", "tr"
)
KNOWN_ARGS_BY_FUNCTION <- list(
  ggbetweenstats = KNOWN_ARGS,
  ggwithinstats = c(KNOWN_ARGS, "subject.id", "point.path", "centrality.path"),
  # gghistostats 1.1.4 has no pairwise or y arguments; binwidth and
  # centrality.plotting change only the plot's layers (and whether the centrality
  # label is computed).
  gghistostats = c(
    "x", "type", "test.value", "alternative", "conf.level", "digits", "results.subtitle",
    "bf.message", "bf.prior", "tr", "binwidth", "centrality.plotting"
  ),
  # ggscatterstats 1.1.4 takes no `alternative`. label.expression is R code kept as
  # text in the case and parsed when the call is built; smooth.line.args replaces
  # geom_smooth()'s arguments (its `formula` is kept as text the same way).
  ggscatterstats = c(
    "x", "y", "type", "conf.level", "digits", "results.subtitle", "bf.message", "bf.prior",
    "tr", "marginal", "smooth.line.args", "label.var", "label.expression"
  ),
  # ggbarstats and ggpiestats 1.1.4 take no bf.prior (contingency tables use the
  # Gunel-Dickey prior with concentration 1, fixed by ggstatsplot) and no tr.
  # label, digits.perc and proportion.test change only the labels and layers.
  ggbarstats = c(
    "x", "y", "counts", "type", "paired", "results.subtitle", "label", "digits",
    "proportion.test", "digits.perc", "bf.message", "ratio", "alternative", "conf.level",
    "p.adjust.method"
  ),
  ggpiestats = c(
    "x", "y", "counts", "type", "paired", "results.subtitle", "label", "digits",
    "proportion.test", "digits.perc", "bf.message", "ratio", "alternative", "conf.level",
    "p.adjust.method"
  )
)
for (case in c(CASES, HELD_OUT_CASES)) {
  known <- KNOWN_ARGS_BY_FUNCTION[[case$fun]]
  if (is.null(known)) stop(case$name, ": generate.R has no bookkeeping for ", case$fun)
  unknown <- setdiff(names(case$args), known)
  if (length(unknown) > 0L) {
    stop(case$name, ": generate.R has no bookkeeping for ", paste(unknown, collapse = ", "))
  }
  if (case$fun == "ggwithinstats" && is.null(case$args$subject.id)) {
    # Without subject.id ggwithinstats pairs rows by their order within each level;
    # inferplot never does, so every case names its subjects.
    stop(case$name, ": ggwithinstats cases must give subject.id")
  }
  if (!is.null(case$args$label.expression) && is.null(case$args$label.var)) {
    stop(case$name, ": label.expression has no effect without label.var")
  }
  # [[ ]], not $: `$` would match label.var and label.expression partially.
  label <- case$args[["label"]]
  if (!is.null(label) && !(label %in% c("percentage", "counts", "both"))) {
    # ggstatsplot matches label by regular expression ("perc|prop", "count|n|N");
    # the cases use its three documented values.
    stop(case$name, ": label must be percentage, counts or both")
  }
  flags <- c(
    "results.subtitle", "bf.message", "point.path", "centrality.path", "marginal",
    "centrality.plotting", "paired", "proportion.test"
  )
  for (flag in flags) {
    value <- case$args[[flag]]
    if (!is.null(value) && !(isTRUE(value) || isFALSE(value))) {
      stop(case$name, ": ", flag, " must be TRUE or FALSE")
    }
  }
  display <- case$args$pairwise.display
  if (!is.null(display) && !(display %in% c("significant", "non-significant", "all", "none"))) {
    # ggstatsplot also accepts abbreviations ("s", "ns"); spell them out here so the
    # bookkeeping below and the Python tests read one spelling.
    stop(case$name, ": spell pairwise.display out in full")
  }
}

# JSON conversion -------------------------------------------------------------------
#
# Numbers keep 15 significant digits. NA becomes null; NaN, Inf and -Inf become the
# strings "NaN", "Inf" and "-Inf". Data frames become arrays of row objects. Plotmath
# expressions and other language objects become their deparsed text.

deparse_text <- function(x) paste(deparse(x, width.cutoff = 500L), collapse = " ")

encode_doubles <- function(x) {
  x <- unname(as.vector(x))
  special <- is.nan(x) | is.infinite(x)
  if (!any(special)) {
    return(x)
  }
  out <- as.list(x)
  out[is.nan(x)] <- list("NaN")
  out[is.infinite(x) & x > 0] <- list("Inf")
  out[is.infinite(x) & x < 0] <- list("-Inf")
  if (length(out) == 1L) out[[1L]] else out
}

frame_rows <- function(df) {
  columns <- names(df)
  lapply(seq_len(nrow(df)), function(i) {
    row <- lapply(columns, function(column) {
      values <- df[[column]]
      cell <- if (is.data.frame(values)) {
        frame_rows(values[i, , drop = FALSE])[[1L]]
      } else if (is.list(values)) {
        values[[i]]
      } else {
        values[i]
      }
      to_json_value(cell)
    })
    names(row) <- columns
    row
  })
}

to_json_value <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (is.expression(x)) {
    parts <- lapply(as.list(x), deparse_text)
    return(if (length(parts) == 1L) parts[[1L]] else parts)
  }
  if (is.language(x) || is.function(x) || is.environment(x)) {
    return(deparse_text(x))
  }
  if (is.data.frame(x)) {
    return(frame_rows(x))
  }
  if (is.factor(x)) {
    return(as.character(x))
  }
  if (is.list(x)) {
    return(lapply(unclass(x), to_json_value))
  }
  if (is.double(x)) {
    return(encode_doubles(x))
  }
  if (is.atomic(x)) {
    return(unname(as.vector(x)))
  }
  deparse_text(x)
}

write_json_file <- function(value, path) {
  text <- jsonlite::toJSON(
    value,
    auto_unbox = TRUE, digits = NA, na = "null", null = "null", pretty = TRUE
  )
  writeLines(text, path, useBytes = TRUE)
}

# Helpers -------------------------------------------------------------------------

capture <- function(expr) {
  warnings <- character()
  messages <- character()
  value <- tryCatch(
    withCallingHandlers(
      expr,
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) {
        messages <<- c(messages, conditionMessage(m))
        invokeRestart("muffleMessage")
      }
    ),
    error = function(e) e
  )
  list(value = value, warnings = warnings, messages = messages)
}

file_sha256 <- function(path) digest::digest(path, algo = "sha256", file = TRUE)

# Conditions ggstatsplot hides ------------------------------------------------------
#
# ggstatsplot 1.1.4 computes the subtitle and the caption through its internal
# `.eval_f`, purrr::possibly() around suppressWarnings(suppressMessages(...)): an
# error there leaves the subtitle or caption out without a trace, and warnings and
# messages never reach capture(). It also wraps centrality_description() and
# pairwise_comparisons() in suppressWarnings() (their errors do reach capture()).
# The replacements below make exactly the same calls, with the same arguments and the
# same random-number stream, return the same values and pass the same conditions on,
# and log what ggstatsplot hides: one entry per call, in call order.

HIDDEN <- new.env()
HIDDEN$log <- list()
# What each logged call returned (NULL after an error), in the same order. Written to
# a fixture only when ggstatsplot fails (`partial`).
HIDDEN$values <- list()

log_hidden <- function(component, error, warnings, messages, value = NULL) {
  HIDDEN$log[[length(HIDDEN$log) + 1L]] <- list(
    component = component,
    error = error,
    warnings = as.list(warnings),
    messages = as.list(messages)
  )
  HIDDEN$values[length(HIDDEN$values) + 1L] <- list(value)
}

# Same behavior as ggstatsplot's `.eval_f`: NULL on error, warnings and messages muted.
recording_eval_f <- function(.f, ...) {
  dots <- rlang::list2(...)
  warnings <- character()
  messages <- character()
  error <- NULL
  value <- tryCatch(
    withCallingHandlers(
      rlang::exec(.f, !!!dots),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(m) {
        messages <<- c(messages, conditionMessage(m))
        invokeRestart("muffleMessage")
      }
    ),
    error = function(e) {
      error <<- conditionMessage(e)
      NULL
    }
  )
  log_hidden(".eval_f", error, warnings, messages, value)
  value
}

# Records warnings and passes them on to ggstatsplot's own suppressWarnings(). The
# original call is evaluated as written, so non-standard evaluation is unchanged. An
# error is logged and raised again unchanged (ggstatsplot does not catch it).
recording_wrapper <- function(component, f) {
  # Force both now: the caller's loop variable changes before the wrapper runs.
  force(component)
  force(f)
  wrapper <- function() {
    call <- sys.call()
    call[[1L]] <- f
    warnings <- character()
    caller <- parent.frame()
    value <- tryCatch(
      withCallingHandlers(
        eval(call, caller),
        warning = function(w) warnings <<- c(warnings, conditionMessage(w))
      ),
      # Only an error that leaves the call ends up here (one handled inside it
      # does not): log it, then raise the same condition again.
      error = function(e) {
        log_hidden(component, conditionMessage(e), warnings, character())
        stop(e)
      }
    )
    log_hidden(component, NULL, warnings, character(), value)
    value
  }
  formals(wrapper) <- formals(f)
  wrapper
}

utils::assignInNamespace(".eval_f", recording_eval_f, ns = "ggstatsplot")
local({
  imports <- parent.env(asNamespace("ggstatsplot"))
  for (name in c("centrality_description", "pairwise_comparisons")) {
    original <- get(name, envir = imports, inherits = FALSE)
    unlockBinding(name, imports)
    assign(name, recording_wrapper(name, original), envir = imports)
    lockBinding(name, imports)
  }
})

# ggbarstats and ggpiestats run the per-bar proportion tests (one goodness-of-fit test
# of x within each level of y, ggstatsplot's onesample_data()) through
# `.chisq_test_safe`: purrr::possibly() around suppressWarnings(contingency_table(data,
# x, ratio = ratio)), which turns an error into a row of NA. The replacement makes the
# same call the same way (the arguments pass through `...` exactly as in the
# original, so the non-standard evaluation of x is unchanged), returns the same value
# or the same NA row, lets messages through as the original does, and logs the error
# and the muted warnings as a hidden call "proportion_test" (one per bar, in the order
# of y's levels; run_case() names each entry's level).
recording_chisq_test_safe <- function(data, x, ratio) {
  warnings <- character()
  error <- NULL
  value <- tryCatch(
    withCallingHandlers(
      (function(...) statsExpressions::contingency_table(...))(data, x, ratio = ratio),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      error <<- conditionMessage(e)
      NULL
    }
  )
  if (!is.null(error)) {
    value <- tibble::tibble(
      statistic = NA_real_,
      p.value = NA_real_,
      df = NA_real_,
      method = "Chi-squared test for given probabilities"
    )
  }
  log_hidden("proportion_test", error, warnings, character(), value)
  value
}
utils::assignInNamespace(".chisq_test_safe", recording_chisq_test_safe, ns = "ggstatsplot")

# The functions whose hidden calls the rules below describe. ggbetweenstats and
# ggwithinstats 1.1.4 share them: both compute the subtitle and caption through
# .bw_subtitle_caption() and .subtitle_caption(), and the labels and pairs through
# .bw_decorate() (ggstatsplot's R/ggbetweenstats-helpers.R and R/0-utils.R).
HIDDEN_CALL_FUNCTIONS <- c(
  "ggbetweenstats", "ggwithinstats", "gghistostats", "ggscatterstats", "ggbarstats",
  "ggpiestats"
)
# Functions whose design has no groups: their fixtures have no levels and record the
# analyzed values instead (`analyzed_data`).
NO_LEVEL_FUNCTIONS <- c("gghistostats", "ggscatterstats")
# Functions that count categories: their fixtures record the table they analyzed
# (`contingency_data`).
BAR_FUNCTIONS <- c("ggbarstats", "ggpiestats")

# What the hidden-call rules of ggbarstats and ggpiestats depend on, as their shared
# .pie_bar_data_prep() leaves the data: the number of levels of x and of y (factor
# levels, declared empty ones included; y_levels is 0 without y) and the number of
# levels of y that hold data (the per-bar tests run once for each).
bar_design <- function(x_levels, y_levels, y_present) {
  list(x_levels = x_levels, y_levels = y_levels, y_present = y_present)
}

# Whether ggbarstats or ggpiestats draws its per-bar proportion tests: proportion.test
# defaults to results.subtitle, and ggbarstats turns it off for type bayes and for
# fewer than two levels of x or of y (so always without y); ggpiestats for type bayes
# and for one level of x with more than one level of y.
bar_proportion_test <- function(case, design) {
  wanted <- case$args$proportion.test %||% !isFALSE(case$args$results.subtitle)
  if (case$fun == "ggbarstats") {
    off <- case$args$type == "bayes" || design$x_levels < 2L || design$y_levels < 2L
  } else {
    off <- (design$x_levels == 1L && design$y_levels > 1L) || case$args$type == "bayes"
  }
  isTRUE(wanted) && !off
}

# The `.eval_f` calls each function makes, in order. All four compute the subtitle and
# caption through .subtitle_caption(): with results.subtitle (default TRUE) the
# subtitle, and then the Bayes caption when bf.condition (type "parametric") and
# bf.message (default TRUE) both hold; with results.subtitle = FALSE neither.
# ggscatterstats then builds its two marginal histogram layers through `.eval_f`
# (ggside::geom_xsidehistogram, then geom_ysidehistogram) unless marginal = FALSE, so
# an error there would drop the marginals silently.
#
# ggbarstats and ggpiestats (.pie_bar_subtitle_caption()): with results.subtitle the
# subtitle, then the Bayes caption when bf.message holds and ggstatsplot's
# bf.condition does (type not bayes, paired FALSE, and y not of exactly one level, so
# also without y), then the pairwise Fisher tests through `.eval_f` when x has more
# than two levels, y more than one and paired is FALSE. `design` is bar_design().
expected_eval_f <- function(case, design = NULL) {
  if (!(case$fun %in% HIDDEN_CALL_FUNCTIONS)) {
    stop("generate.R has no hidden-call rules for ", case$fun, "; add them first")
  }
  if (case$fun %in% BAR_FUNCTIONS) {
    if (is.null(design)) stop(case$name, ": the hidden calls depend on the table's levels")
    if (isFALSE(case$args$results.subtitle)) {
      return(character())
    }
    paired <- isTRUE(case$args$paired)
    caption <- case$args$type != "bayes" && !paired && design$y_levels != 1L &&
      !isFALSE(case$args$bf.message)
    pairwise <- design$x_levels > 2L && design$y_levels > 1L && !paired
    return(c("subtitle", if (caption) "caption", if (pairwise) "pairwise_comparisons"))
  }
  calls <- if (isFALSE(case$args$results.subtitle)) {
    character()
  } else {
    c("subtitle", if (case$args$type == "parametric" && !isFALSE(case$args$bf.message)) "caption")
  }
  if (case$fun == "ggscatterstats" && !isFALSE(case$args$marginal)) {
    calls <- c(calls, "xsidehistogram", "ysidehistogram")
  }
  calls
}

# Every hidden call each function makes, in order. ggbetweenstats and ggwithinstats:
# the `.eval_f` calls above, then centrality_description() (always, also when the
# labels are not drawn), then pairwise_comparisons() for three or more groups
# (ggwithinstats: three or more conditions after complete_subjects()) unless
# pairwise.display = "none". gghistostats: the `.eval_f` calls, then
# centrality_description() (through .histo_labeller()) unless centrality.plotting =
# FALSE. ggscatterstats: the `.eval_f` calls only. ggbarstats and ggpiestats: the
# `.eval_f` calls, then with y one per-bar proportion test (`.chisq_test_safe`) for
# each level of y that holds data, whatever results.subtitle and proportion.test say
# (onesample_data() also gives the n under each bar). `levels` is bar_design() for
# them.
expected_hidden_calls <- function(case, levels) {
  if (case$fun %in% BAR_FUNCTIONS) {
    return(c(expected_eval_f(case, levels), rep("proportion_test", levels$y_present)))
  }
  if (case$fun == "gghistostats") {
    return(c(
      expected_eval_f(case),
      if (!isFALSE(case$args$centrality.plotting)) "centrality_description"
    ))
  }
  if (case$fun == "ggscatterstats") {
    return(expected_eval_f(case))
  }
  display <- case$args$pairwise.display
  pairwise <- length(levels) > 2L && !identical(display, "none")
  c(expected_eval_f(case), "centrality_description", if (pairwise) "pairwise_comparisons")
}

# Name the `.eval_f` entries by their order. Any other number of calls than the
# arguments imply stops the run; a call that failed may have stopped early, so its
# entries are named only as far as they go (outcome_problems() then reports the
# error unless the case expects it).
name_hidden <- function(case, log, failed = FALSE, design = NULL) {
  eval_f <- which(vapply(log, function(entry) entry$component == ".eval_f", logical(1L)))
  if (case$fun %in% BAR_FUNCTIONS && is.null(design)) {
    # The data could not be prepared (ggstatsplot fails there too): no hidden call
    # can have run.
    if (length(log) > 0L) stop(case$name, ": hidden calls without prepared data")
    return(log)
  }
  expected <- expected_eval_f(case, design)
  if (length(eval_f) > length(expected) || (!failed && length(eval_f) != length(expected))) {
    stop(
      case$name, ": ", length(eval_f), " `.eval_f` calls; expected ",
      if (length(expected) == 0L) "none" else paste(expected, collapse = " and ")
    )
  }
  for (i in seq_along(eval_f)) log[[eval_f[[i]]]]$component <- expected[[i]]
  log
}

# The plot's layers in drawing order, with each layer's own data. The centrality
# labels live here (ggstatsplot 1.1.4's extract_stats() leaves descriptive_data
# empty). Layers that draw the plot's main data have data null.
#
# `build_all` (gghistostats, ggscatterstats): every layer but the raw points also
# carries `params` and `built`, so the histogram's bins and counts, the centrality
# line, the smooth line with its confidence band and the marginal histograms are
# recorded as ggplot2 computed them.
plot_layers <- function(plot, build_all = FALSE) {
  # ggplot2 4.x plots are S7 objects; fall back to `$` for older S3 plots.
  layers <- tryCatch(plot@layers, error = function(e) plot$layers)
  layer_names <- names(layers)
  if (is.null(layer_names)) layer_names <- rep("", length(layers))
  unname(Map(function(layer, name, index) {
    data <- layer$data
    geom <- class(layer$geom)[[1L]]
    detailed <- geom == "GeomSignif" || (build_all && geom != "GeomPoint")
    list(
      name = name,
      geom = geom,
      data = if (is.data.frame(data)) frame_rows(data) else NULL,
      # Significance brackets keep the drawn pairs (`comparisons`) and their labels
      # (`annotations`) in the layer's parameters, not in its data; `built` is the
      # layer as ggplot2 computes it for drawing.
      params = if (detailed) {
        list(
          stat = to_json_value(layer$stat_params),
          geom = to_json_value(layer$geom_params),
          aes = to_json_value(layer$aes_params)
        )
      } else {
        NULL
      },
      built = if (detailed) frame_rows(ggplot2::layer_data(plot, index)) else NULL
    )
  }, layers, layer_names, seq_along(layers)))
}

# The plot's texts outside the layers (ggbarstats and ggpiestats): the labels
# (ggplot2::get_labs(), where the x axis names y and the fill names x) and the fill
# legend's title, which ggstatsplot sets through guides(). Each is recorded as text
# "<error: ...>" when ggplot2 cannot give it, so a change in ggplot2's internals
# cannot stop the run.
plot_texts <- function(plot) {
  attempt <- function(expr) {
    tryCatch(to_json_value(expr), error = function(e) paste0("<error: ", conditionMessage(e), ">"))
  }
  guides <- tryCatch(plot@guides, error = function(e) plot$guides)
  list(
    labels = attempt(ggplot2::get_labs(plot)),
    legend_title = attempt(guides$guides$fill$params$title)
  )
}

VERSIONS <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  platform = R.version$platform,
  os = utils::sessionInfo()$running,
  collate = Sys.getlocale("LC_COLLATE"),
  rng_kind = as.list(RNGkind()),
  repos = as.list(getOption("repos")),
  packages = stats::setNames(
    lapply(RECORDED_PACKAGES, function(p) as.character(packageVersion(p))),
    RECORDED_PACKAGES
  )
)

# Bootstrap references --------------------------------------------------------------
#
# For effect sizes whose CI ggstatsplot bootstraps:
#
# 1. Replay ggstatsplot's own computation. Inside ggbetweenstats() and
#    ggwithinstats() the effect-size call is the first to draw random numbers
#    (statsExpressions 2.2.1 source), so set.seed(CASE_SEED) followed by the same
#    call with GG_RESAMPLES resamples, on the rows in statsExpressions' order, must
#    reproduce ggstatsplot's estimate and
#    both CI bounds exactly. A match proves the reference call below is the method
#    and variant ggstatsplot ran. A case may list several candidate calls (both
#    EQVAR settings of akp.effect); the one that matches is used, and every
#    candidate's replayed values are recorded.
# 2. Reference: the same call with BOOT_REFERENCE_RESAMPLES resamples.
# 3. Replicates: BOOT_REPLICATES runs with BOOT_REPLICATE_RESAMPLES resamples, the
#    Monte Carlo spread of an estimate and CI computed the way inferplot computes
#    them.
# Each `run` returns list(values = c(estimate, conf_low, conf_high), raw = ...).

# Rows as statsExpressions passes them to the effect-size functions:
# long_to_wide_converter(paired = FALSE, spread = FALSE) orders them by the levels of
# x and keeps the original order within a level. Bootstraps that resample rows depend
# on this order, so replays must use it.
statsexpressions_rows <- function(analysis) {
  convert <- utils::getFromNamespace("long_to_wide_converter", "statsExpressions")
  long <- convert(data = analysis, x = x, y = y, paired = FALSE, spread = FALSE)
  data.frame(y = long[["y"]], x = long[["x"]])
}

# The same for a within-subjects design, from the complete subjects ggwithinstats
# passes on (within_analysis(): columns .rowid, x, y): the long tibble
# long_to_wide_converter(paired = TRUE, spread = FALSE) gives two_sample_test(),
# oneway_anova() and pairwise_comparisons() (rows sorted by subject, then by
# condition), and the wide tibble (one column per condition, subjects in that order)
# two_sample_test() gives dep.effect for the robust type. The columns keep the
# case's own names (x_name, y_name), as in ggstatsplot: renamed to x and y, a level
# called "x" or "y" (edge_within_nonspherical) would clash with them when the data are
# spread wide.
statsexpressions_paired_rows <- function(analysis, x_name, y_name) {
  convert <- utils::getFromNamespace("long_to_wide_converter", "statsExpressions")
  data <- analysis
  names(data) <- c(".rowid", x_name, y_name)
  xs <- rlang::sym(x_name)
  ys <- rlang::sym(y_name)
  list(
    x = x_name, y = y_name,
    long = rlang::inject(convert(
      data = data, x = !!xs, y = !!ys, subject.id = .rowid, paired = TRUE, spread = FALSE
    )),
    wide = rlang::inject(convert(
      data = data, x = !!xs, y = !!ys, subject.id = .rowid, paired = TRUE, spread = TRUE
    ))
  )
}

# statsExpressions' formula y ~ x | .rowid for a design (friedman.test, kendalls_w).
blocked_formula <- function(rows) {
  stats::as.formula(sprintf("`%s` ~ `%s` | .rowid", rows$y, rows$x))
}

akp_candidate <- function(analysis, eqvar, alpha, tr) {
  list(
    call = sprintf(
      "WRS2::akp.effect(y ~ x, data, EQVAR = %s, tr = %s, nboot = R, alpha = %s)",
      eqvar, format(tr), alpha$text
    ),
    run = function(resamples) {
      r <- WRS2::akp.effect(
        y ~ x,
        data = analysis, EQVAR = eqvar, tr = tr, nboot = resamples, alpha = alpha$value
      )
      effect <- as.numeric(r$AKPeffect)
      ci <- r$AKPci
      # EQVAR = FALSE may return one effect per group's variance; keep the first and
      # its interval. The raw output is recorded so the extraction can be checked.
      pair <- if (is.matrix(ci) && nrow(ci) == length(effect)) {
        ci[1L, ]
      } else if (is.matrix(ci) && ncol(ci) == length(effect)) {
        ci[, 1L]
      } else {
        as.numeric(ci)[1:2]
      }
      # jsonlite writes non-finite doubles as null: an infinite bound is kept as text
      # (row by row for a matrix), finite output exactly as before.
      ci_raw <- if (all(is.finite(ci))) {
        ci
      } else if (is.matrix(ci)) {
        lapply(seq_len(nrow(ci)), function(i) encode_doubles(ci[i, ]))
      } else {
        encode_doubles(ci)
      }
      list(
        values = c(effect[[1L]], as.numeric(pair)),
        raw = list(AKPeffect = r$AKPeffect, AKPci = ci_raw, AKPci_dim = dim(r$AKPci))
      )
    }
  )
}

# The effect-size row statsExpressions keeps from a WRS2 object, as it keeps it:
# estimate, conf.low, conf.high from tidy_model_parameters().
tidy_effect <- function(model, ...) {
  tidy <- statsExpressions::tidy_model_parameters(model, ...)
  if (nrow(tidy) != 1L) stop("expected one effect-size row, got ", nrow(tidy))
  c(tidy$estimate, tidy$conf.low, tidy$conf.high)
}

# A WRS2 matrix or list output for the fixture, with row names kept.
raw_matrix <- function(m) {
  m <- unclass(m)
  if (!is.matrix(m)) {
    return(to_json_value(m))
  }
  df <- data.frame(
    row = rownames(m) %||% as.character(seq_len(nrow(m))), m,
    check.names = FALSE, stringsAsFactors = FALSE
  )
  names(df) <- make.unique(names(df))
  frame_rows(df)
}

# ggwithinstats' bootstrapped effect sizes (statsExpressions 2.2.1 source):
# WRS2::dep.effect for two robust conditions (its AKP row: tidy_model_parameters(keep
# = "AKP")), WRS2::wmcpAKP for three or more robust conditions, and
# effectsize::kendalls_w for three or more nonparametric conditions. statsExpressions
# passes neither alpha nor conf.level to the two WRS2 calls, only tr and nboot.
within_bootstrap_candidates <- function(case, rows, subtitle_row) {
  k <- nlevels(rows$long[[rows$x]])
  type <- case$args$type
  if (type %in% c("parametric", "bayes") || (type == "nonparametric" && k == 2L)) {
    return(list())
  }
  if (length(subtitle_row) == 0L) {
    return(list())
  }
  conf_level <- case_conf_level(case)
  tr <- case_tr(case)
  if (type == "robust" && k == 2L) {
    wide <- rows$wide
    return(list(list(
      call = sprintf(
        "WRS2::dep.effect(x = wide[[2]], y = wide[[3]], tr = %s, nboot = R)", format(tr)
      ),
      conf_level = NULL,
      run = function(resamples) {
        r <- WRS2::dep.effect(x = wide[[2L]], y = wide[[3L]], tr = tr, nboot = resamples)
        list(values = tidy_effect(r, keep = "AKP"), raw = raw_matrix(r))
      }
    )))
  }
  if (type == "robust") {
    # oneway_anova(): long_to_wide_converter(data, x, y) on its long data (whose .rowid
    # it made a factor), without .rowid.
    convert <- utils::getFromNamespace("long_to_wide_converter", "statsExpressions")
    long <- rows$long
    long$.rowid <- as.factor(long$.rowid)
    xs <- rlang::sym(rows$x)
    ys <- rlang::sym(rows$y)
    wide <- rlang::inject(convert(data = long, x = !!xs, y = !!ys))
    wide$.rowid <- NULL
    return(list(list(
      call = sprintf("WRS2::wmcpAKP(wide, tr = %s, nboot = R)", format(tr)),
      conf_level = NULL,
      run = function(resamples) {
        r <- WRS2::wmcpAKP(wide, tr = tr, nboot = resamples)
        list(values = tidy_effect(r), raw = to_json_value(unclass(r)))
      }
    )))
  }
  if (type == "nonparametric") {
    long <- rows$long
    long$.rowid <- as.factor(long$.rowid)
    formula <- blocked_formula(rows)
    return(list(list(
      call = sprintf(
        "effectsize::kendalls_w(%s, data, ci = %s, iterations = R)",
        deparse_text(formula), format(conf_level)
      ),
      run = function(resamples) {
        r <- effectsize::kendalls_w(
          formula,
          data = long, ci = conf_level, iterations = resamples, verbose = FALSE
        )
        list(values = c(r[[1L]], r$CI_low, r$CI_high), raw = r)
      }
    )))
  }
  stop(case$name, ": no bootstrap rule for type ", type)
}

# WRS2::trimcibt's documented default number of bootstrap samples (its help page:
# nboot = 200). statsExpressions 2.2.1's one_sample_test() passes no nboot, so
# gghistostats' robust subtitle uses this many; the replay checks it (a candidate with
# nboot left out and one with nboot = 200 must both reproduce ggstatsplot).
TRIMCIBT_DEFAULT_RESAMPLES <- 200L

# gghistostats' one bootstrapped quantity (statsExpressions 2.2.1 one_sample_test()):
# type robust calls WRS2::trimcibt(x, nv = test.value, tr, alpha = 1 - conf.level), a
# bootstrap-t test whose statistic and estimate are exact while its p-value and the
# interval come from the bootstrap. The reference therefore records the p-value
# (`fields`) next to the estimate and the bounds. The values are x without missing
# values, in file order, as one_sample_test() passes them.
histo_bootstrap_candidates <- function(case, values, subtitle_row) {
  if (case$args$type != "robust" || length(subtitle_row) == 0L) {
    return(list())
  }
  conf_level <- case_conf_level(case)
  tr <- case_tr(case)
  test_value <- case$args$test.value %||% 0
  candidates <- list()
  for (resamples in list(NA_integer_, TRIMCIBT_DEFAULT_RESAMPLES)) {
    for (alpha in alpha_spellings(conf_level)) {
      candidates[[length(candidates) + 1L]] <- local({
        alpha <- alpha
        gg_resamples <- resamples
        list(
          call = sprintf(
            "WRS2::trimcibt(x, nv = %s, tr = %s, alpha = %s, nboot = R)",
            format(test_value), format(tr), alpha$text
          ),
          fields = c("estimate", "conf_low", "conf_high", "p_value"),
          gg_columns = c("estimate", "conf.low", "conf.high", "p.value"),
          gg_resamples = TRIMCIBT_DEFAULT_RESAMPLES,
          replay_resamples = gg_resamples,
          run = function(resamples) {
            r <- if (is.na(resamples)) {
              WRS2::trimcibt(x = values, nv = test_value, tr = tr, alpha = alpha$value)
            } else {
              WRS2::trimcibt(
                x = values, nv = test_value, tr = tr, alpha = alpha$value, nboot = resamples
              )
            }
            raw <- Filter(is.numeric, unclass(r))
            list(
              values = c(r$estimate, as.numeric(r$ci)[1:2], r$p.value),
              raw = lapply(raw, encode_doubles)
            )
          }
        )
      })
    }
  }
  candidates
}

bootstrap_candidates <- function(case, analysis, subtitle_row) {
  # Fail closed: each function needs its own replay (paired row order, other effect
  # sizes) before its cases are added.
  if (case$fun == "ggwithinstats") {
    return(within_bootstrap_candidates(case, analysis, subtitle_row))
  }
  if (case$fun == "gghistostats") {
    return(histo_bootstrap_candidates(case, analysis, subtitle_row))
  }
  if (case$fun == "ggscatterstats") {
    # correlation::correlation computes every ggscatterstats interval analytically
    # (the Bayes type samples a posterior, a v0.2 output, replayed in `correlation`).
    return(list())
  }
  if (case$fun != "ggbetweenstats") {
    stop("generate.R has no bootstrap replay for ", case$fun, "; add one first")
  }
  k <- nlevels(analysis$x)
  type <- case$args$type
  # statsExpressions 2.2.1 bootstraps the effect-size CI of exactly these
  # ggbetweenstats designs: AKP (robust, two groups), t1way's explanatory effect size
  # (robust, three or more) and rank epsilon squared (nonparametric, three or more).
  if (type %in% c("parametric", "bayes") || (type == "nonparametric" && k == 2L)) {
    return(list())
  }
  # Without a subtitle ggstatsplot computes no effect size (results.subtitle = FALSE,
  # or a subtitle error it swallowed): there is no CI to replay or reference.
  if (length(subtitle_row) == 0L) {
    return(list())
  }
  # statsExpressions passes conf.level (to WRS2 as alpha = 1 - conf.level) and tr on
  # to these calls.
  conf_level <- case_conf_level(case)
  tr <- case_tr(case)
  alphas <- alpha_spellings(conf_level)
  if (type == "robust" && k == 2L) {
    # statsExpressions' own call first: EQVAR = FALSE, alpha = 1 - conf.level.
    return(c(
      lapply(alphas, function(alpha) akp_candidate(analysis, FALSE, alpha, tr)),
      lapply(alphas, function(alpha) akp_candidate(analysis, TRUE, alpha, tr))
    ))
  }
  if (type == "robust") {
    return(lapply(alphas, function(alpha) {
      list(
        call = sprintf(
          "WRS2::t1way(y ~ x, data, tr = %s, alpha = %s, nboot = R)", format(tr), alpha$text
        ),
        run = function(resamples) {
          r <- WRS2::t1way(
            y ~ x,
            data = analysis, tr = tr, alpha = alpha$value, nboot = resamples
          )
          list(
            values = c(r$effsize, r$effsize_ci),
            raw = list(effsize = r$effsize, effsize_ci = r$effsize_ci)
          )
        }
      )
    }))
  }
  if (type == "nonparametric") {
    # Match the sidedness ggstatsplot reported: an upper bound of exactly 1 means a
    # one-sided ("greater") interval.
    alternative <- if (isTRUE(subtitle_row$conf.high == 1)) "greater" else "two.sided"
    return(list(list(
      call = sprintf(
        "effectsize::rank_epsilon_squared(y ~ x, data, ci = %s, alternative = \"%s\", iterations = R)",
        format(conf_level), alternative
      ),
      run = function(resamples) {
        r <- effectsize::rank_epsilon_squared(
          y ~ x,
          data = analysis, ci = conf_level, alternative = alternative,
          iterations = resamples, verbose = FALSE
        )
        list(values = c(r[[1L]], r$CI_low, r$CI_high), raw = r)
      }
    )))
  }
  stop(case$name, ": no bootstrap rule for type ", type)
}

run_or_stop <- function(case, what, expr) {
  out <- capture(expr)
  if (inherits(out$value, "error")) {
    stop(what, " failed for ", case$name, ": ", conditionMessage(out$value))
  }
  out
}

# Bootstrap references already computed in this run, by call and data.
BOOT_CACHE <- new.env()

# The 20,000-resample reference and the 2,000-resample replicates of one call.
bootstrap_runs <- function(case, chosen) {
  reference <- run_or_stop(case, "bootstrap reference", {
    set.seed(BOOT_SEED)
    chosen$run(BOOT_REFERENCE_RESAMPLES)
  })
  # Each replicate sets its own seed, so running them in parallel processes gives the
  # same values as running them in turn; warnings are collected in replicate order.
  runs <- parallel::mclapply(
    seq_len(BOOT_REPLICATES),
    function(i) {
      out <- capture({
        set.seed(BOOT_SEED + i)
        unname(chosen$run(BOOT_REPLICATE_RESAMPLES)$values)
      })
      list(
        values = if (inherits(out$value, "error")) NULL else out$value,
        error = if (inherits(out$value, "error")) conditionMessage(out$value) else NULL,
        warnings = out$warnings
      )
    },
    mc.cores = BOOT_CORES
  )
  for (run in runs) {
    if (!is.list(run)) stop("bootstrap replicates failed for ", case$name, ": ", run)
    if (!is.null(run$error)) {
      stop("bootstrap replicates failed for ", case$name, ": ", run$error)
    }
  }
  list(reference = reference, runs = runs)
}

# The quantities a bootstrap reference records, as fixture field names, and the
# subtitle_data columns that hold ggstatsplot's values of them. Every effect-size
# candidate records these three; gghistostats' trimcibt adds the p-value.
DEFAULT_BOOT_FIELDS <- c("estimate", "conf_low", "conf_high")
DEFAULT_BOOT_COLUMNS <- c("estimate", "conf.low", "conf.high")

bootstrap_reference <- function(case, analysis, stats) {
  # statsExpressions computes every contingency-table interval analytically (the
  # one-way Bayes factor's Monte Carlo draws are a v0.2 output, in the caption).
  if (case$fun %in% BAR_FUNCTIONS) {
    return(NULL)
  }
  subtitle_row <- stats$subtitle_data
  within <- case$fun == "ggwithinstats"
  histo <- case$fun == "gghistostats"
  ordered <- if (within) {
    statsexpressions_paired_rows(analysis, case$args$x, case$args$y)
  } else if (histo) {
    analysis$x
  } else if (case$fun == "ggscatterstats") {
    analysis
  } else {
    statsexpressions_rows(analysis)
  }
  candidates <- bootstrap_candidates(case, ordered, subtitle_row)
  if (length(candidates) == 0L) {
    return(NULL)
  }
  fields <- candidates[[1L]]$fields %||% DEFAULT_BOOT_FIELDS
  gg_columns <- candidates[[1L]]$gg_columns %||% DEFAULT_BOOT_COLUMNS
  gg_values <- unname(vapply(gg_columns, function(column) {
    value <- subtitle_row[[column]]
    if (length(value) == 1L) as.numeric(value) else NA_real_
  }, numeric(1L)))

  tried <- list()
  chosen <- NULL
  for (candidate in candidates) {
    replay_resamples <- if ("replay_resamples" %in% names(candidate)) {
      candidate$replay_resamples
    } else {
      GG_RESAMPLES
    }
    replay <- run_or_stop(case, "ggstatsplot replay", {
      set.seed(CASE_SEED)
      candidate$run(replay_resamples)
    })
    values <- unname(replay$value$values)
    matches <- isTRUE(all.equal(values, gg_values, tolerance = 1e-10))
    entry <- list(
      call = candidate$call, values = as.list(encode_doubles(values)), matches = matches
    )
    if ("replay_resamples" %in% names(candidate)) {
      # Which nboot the replay passed: "default" when it left nboot out.
      entry <- append(entry, list(
        replay_resamples = if (is.na(replay_resamples)) "default" else replay_resamples
      ), after = 1L)
    }
    tried[[length(tried) + 1L]] <- entry
    if (matches && is.null(chosen)) chosen <- candidate
  }
  replayed <- !is.null(chosen)
  if (!replayed) chosen <- candidates[[1L]]

  # The reference and the replicates depend only on the call and its data (each run
  # sets its own seed), so cases that share both (the p.adjust, display and digits
  # variants of one dataset) share one computation; the values are the same either
  # way.
  key <- digest::digest(list(chosen$call, ordered), algo = "sha256")
  cached <- BOOT_CACHE[[key]]
  if (is.null(cached)) {
    cached <- bootstrap_runs(case, chosen)
    assign(key, cached, envir = BOOT_CACHE)
  }
  reference <- cached$reference
  runs <- cached$runs
  replicates <- list(
    value = t(vapply(runs, function(run) run$values, numeric(length(fields)))),
    warnings = unlist(lapply(runs, function(run) run$warnings))
  )
  # Infinite bounds (an effect size whose winsorized variance is 0 in enough
  # resamples) are written as "Inf" and "-Inf", like every other fixture double.
  ref <- encode_doubles(unname(reference$value$values))
  reps <- replicates$value
  spread <- function(column, f) encode_doubles(f(reps[, column]))
  gg_values <- encode_doubles(gg_values)
  # Each field: ggstatsplot's value, the reference's, and the replicates' mean and SD,
  # in the order of `fields`.
  gg_list <- stats::setNames(lapply(seq_along(fields), function(i) gg_values[[i]]), fields)
  ref_list <- stats::setNames(lapply(seq_along(fields), function(i) ref[[i]]), fields)
  spread_list <- unlist(lapply(seq_along(fields), function(i) {
    stats::setNames(
      list(spread(i, mean), spread(i, stats::sd)),
      paste0(fields[[i]], c("_mean", "_sd"))
    )
  }), recursive = FALSE)
  out <- list(
    quantity = if (histo) {
      "bootstrap-t trimmed-mean test: confidence interval and p-value"
    } else {
      "effect size confidence interval"
    },
    call = chosen$call,
    data_order = if (within) {
      paste(
        "complete subjects sorted by subject, then by condition, as statsExpressions",
        "passes them (dep.effect and wmcpAKP: one column per condition)"
      )
    } else if (histo) {
      "the values of x in file order, missing values dropped, as statsExpressions passes them"
    } else {
      "rows ordered by the levels of x, as statsExpressions passes them"
    },
    # The level of the interval the call computes. dep.effect and wmcpAKP take no
    # confidence level (statsExpressions passes none): their level is the one
    # ggstatsplot reports, and the case's own is kept as requested_conf_level.
    conf_level = if (within && "conf_level" %in% names(chosen)) {
      subtitle_row$conf.level
    } else {
      case_conf_level(case)
    },
    ggstatsplot = c(
      gg_list,
      list(resamples = chosen$gg_resamples %||% GG_RESAMPLES, seed = CASE_SEED)
    ),
    replay = list(matches = replayed, candidates = tried),
    seed = BOOT_SEED,
    resamples = BOOT_REFERENCE_RESAMPLES
  )
  out <- c(
    out,
    ref_list,
    list(
      raw = reference$value$raw,
      replicates = c(
        list(
          count = BOOT_REPLICATES,
          resamples = BOOT_REPLICATE_RESAMPLES,
          first_seed = BOOT_SEED + 1L
        ),
        spread_list
      ),
      r_warnings = as.list(unique(c(reference$warnings, replicates$warnings)))
    )
  )
  if (within) {
    out <- append(out, list(requested_conf_level = case_conf_level(case)), after = 4L)
  }
  out
}

# Running a case --------------------------------------------------------------------

read_dataset <- function(case) {
  path <- file.path(dataset_dir, paste0(case$dataset, ".csv"))
  data <- utils::read.csv(
    path,
    na.strings = "", stringsAsFactors = FALSE, check.names = FALSE,
    fileEncoding = "UTF-8"
  )
  for (column in names(case$factor_levels)) {
    data[[column]] <- factor(data[[column]], levels = case$factor_levels[[column]])
  }
  list(data = data, path = path)
}

# The data ggstatsplot analyzes for a between-groups design, as R sees it: complete
# rows of x and y, x as a factor without empty levels.
between_analysis <- function(data, x, y) {
  used <- data[!is.na(data[[x]]) & !is.na(data[[y]]), , drop = FALSE]
  groups <- if (is.factor(used[[x]])) used[[x]] else factor(used[[x]])
  data.frame(y = used[[y]], x = droplevels(groups))
}

# The data ggstatsplot analyzes for a within-subjects design: ggwithinstats() first
# calls statsExpressions::complete_subjects(data, x, y, subject.id), which drops rows
# without x or subject, stops on a subject repeated within a level, keeps the
# subjects observed (with y) in every level of x and sorts by subject (columns .rowid,
# x, y; x a factor without empty levels). Every test, label and the plot use that.
within_analysis <- function(data, x, y, subject) {
  frame <- data.frame(
    x = data[[x]], y = data[[y]], subject = data[[subject]], stringsAsFactors = FALSE
  )
  out <- statsExpressions::complete_subjects(frame, x, y, subject)
  if (!identical(names(out), c(".rowid", "x", "y"))) {
    stop("complete_subjects() returned columns ", paste(names(out), collapse = ", "))
  }
  out
}

# Which rows each function analyzes. Fail closed: every other function (ggbarstats
# counts its rows, and so on) needs its own entry before its cases.
ANALYSES <- list(
  ggbetweenstats = function(data, case) between_analysis(data, case$args$x, case$args$y),
  ggwithinstats = function(data, case) {
    within_analysis(data, case$args$x, case$args$y, case$args$subject.id)
  },
  # gghistostats: tidyr::drop_na(select(data, x)), so only a missing x drops a row;
  # statsExpressions' one_sample_test() and the histogram use the rest in file order.
  gghistostats = function(data, case) {
    keep <- !is.na(data[[case$args$x]])
    data.frame(x = as.numeric(data[[case$args$x]][keep]), row = which(keep))
  },
  # ggscatterstats: filter(data, !is.na(x), !is.na(y)), the complete pairs in file
  # order; corr_test() drops the same rows again.
  ggscatterstats = function(data, case) {
    keep <- !is.na(data[[case$args$x]]) & !is.na(data[[case$args$y]])
    data.frame(
      x = as.numeric(data[[case$args$x]][keep]),
      y = as.numeric(data[[case$args$y]][keep]),
      row = which(keep)
    )
  },
  ggbarstats = function(data, case) bar_analysis(data, case),
  ggpiestats = function(data, case) bar_analysis(data, case)
)

# ggbarstats and ggpiestats: the data as ggstatsplot's own .pie_bar_data_prep() leaves
# it (select x, y and counts, tidyr::drop_na(), tidyr::uncount() by counts, every
# column as.factor(), so declared levels, empty ones included, are kept), called the
# way both functions call it. An error there (ggstatsplot fails on such data too) is
# kept as text.
bar_analysis <- function(data, case) {
  prep_fn <- utils::getFromNamespace(".pie_bar_data_prep", "ggstatsplot")
  sym_or_null <- function(name) if (is.null(name)) NULL else rlang::sym(name)
  out <- capture(rlang::inject(prep_fn(
    data = data,
    x = !!rlang::sym(case$args$x),
    y = !!sym_or_null(case$args$y),
    counts = !!sym_or_null(case$args$counts)
  )))
  if (inherits(out$value, "error")) {
    return(list(error = conditionMessage(out$value), x = NULL, y = NULL))
  }
  prep <- out$value
  list(
    error = NULL,
    x = prep$data[[case$args$x]],
    y = if (is.null(case$args$y)) NULL else prep$data[[case$args$y]],
    n = nrow(prep$data),
    test = prep$test,
    design = bar_design(
      prep$x_levels, prep$y_levels,
      if (is.null(case$args$y)) 0L else nlevels(droplevels(prep$data[[case$args$y]]))
    )
  )
}

# The table a ggbarstats or ggpiestats case analyzed, as `within_data` records a
# within-subjects case: the columns (x, y and counts, null where not given), the
# 1-based CSV rows kept (no value missing in those columns; counts then repeat a row),
# the levels of x and y as ggstatsplot holds them (declared order, empty declared
# levels included; else sorted: numbers numerically, text in the C locale), the counts
# (one row per level of x, one column per level of y; one column without y), n (the
# rows after uncount()), and the error when the data could not be prepared.
contingency_data <- function(data, case, analysis) {
  columns <- c(case$args$x, case$args$y, case$args$counts)
  keep <- stats::complete.cases(data[columns])
  one_way <- is.null(case$args$y)
  base <- list(
    x = case$args$x,
    y = case$args$y,
    counts = case$args$counts,
    rows = as.list(which(keep)),
    rows_dropped = sum(!keep),
    rule = paste(
      "rows with x, y and counts present (tidyr::drop_na), each repeated counts times",
      "(tidyr::uncount), every column as.factor()"
    ),
    error = analysis$error
  )
  if (!is.null(analysis$error)) {
    return(c(base, list(x_levels = list(), y_levels = list(), table = list(), n = NULL)))
  }
  tab <- if (one_way) table(analysis$x) else table(analysis$x, analysis$y)
  rows <- lapply(seq_len(nlevels(analysis$x)), function(i) {
    as.list(as.integer(if (one_way) tab[[i]] else tab[i, ]))
  })
  c(base, list(
    x_levels = as.list(levels(analysis$x)),
    y_levels = if (one_way) list() else as.list(levels(analysis$y)),
    table = rows,
    n = analysis$n
  ))
}

# The values a one-sample or correlation case analyzed (gghistostats,
# ggscatterstats), as `within_data` records the subjects of a ggwithinstats case:
# the column(s), the 1-based CSV rows kept (in file order), their values, and how many
# rows were left out for a missing value.
analyzed_data <- function(data, case, analysis) {
  if (case$fun == "gghistostats") {
    return(list(
      x = case$args$x,
      rows = as.list(analysis$row),
      x_values = as.list(analysis$x),
      n = nrow(analysis),
      rows_dropped = nrow(data) - nrow(analysis),
      rule = "rows with x present, in file order (tidyr::drop_na on x alone)"
    ))
  }
  list(
    x = case$args$x,
    y = case$args$y,
    rows = as.list(analysis$row),
    x_values = as.list(analysis$x),
    y_values = as.list(analysis$y),
    n = nrow(analysis),
    rows_dropped = nrow(data) - nrow(analysis),
    rule = "rows with both x and y present, in file order (complete pairs)"
  )
}

# correlation::correlation() as statsExpressions 2.2.1's corr_test() calls it, on the
# complete pairs with the case's column names, after set.seed(CASE_SEED): inside
# ggscatterstats() the subtitle's call is the first to draw random numbers (the Bayes
# type samples its posterior), so this replays the subtitle's row exactly. Its whole
# output row is recorded (`value`), and `matches_subtitle` says whether r, its CI,
# the statistic and p equal ggstatsplot's subtitle_data.
correlation_record <- function(case, analysis, subtitle_row) {
  type <- case$args$type
  data <- stats::setNames(
    data.frame(analysis$x, analysis$y), c(case$args$x, case$args$y)
  )
  conf_level <- case_conf_level(case)
  prior <- case$args$bf.prior %||% 0.707
  winsorize <- if (type == "robust") case_tr(case) else FALSE
  set.seed(CASE_SEED)
  out <- capture(correlation::correlation(
    data = data,
    method = if (type == "nonparametric") "spearman" else "pearson",
    ci = conf_level,
    bayesian = type == "bayes",
    bayesian_prior = prior,
    winsorize = winsorize
  ))
  failed <- inherits(out$value, "error")
  value <- if (failed) NULL else as.data.frame(out$value)
  matches <- NULL
  if (!failed && length(subtitle_row) > 0L && nrow(value) == 1L) {
    estimate <- value[[intersect(c("r", "rho"), names(value))[[1L]]]]
    mine <- c(estimate, value$CI_low, value$CI_high)
    theirs <- c(subtitle_row$estimate, subtitle_row$conf.low, subtitle_row$conf.high)
    if (type != "bayes") {
      statistic <- intersect(c("t", "S"), names(value))
      # statsExpressions' add_expression_col() stores a p of exactly 0 as
      # .Machine$double.xmin (r = -1: correlation's p is 0, the subtitle's 2.2e-308).
      p <- if (isTRUE(value$p == 0)) .Machine$double.xmin else value$p
      mine <- c(mine, value[[statistic[[1L]]]], p)
      theirs <- c(theirs, subtitle_row$statistic, subtitle_row$p.value)
    } else {
      mine <- c(mine, value$BF)
      theirs <- c(theirs, subtitle_row$bf10)
    }
    matches <- identical(length(mine), length(theirs)) &&
      isTRUE(all.equal(unname(mine), unname(theirs), tolerance = 0))
  }
  list(
    call = sprintf(
      paste0(
        "correlation::correlation(data, method = \"%s\", ci = %s, bayesian = %s, ",
        "bayesian_prior = %s, winsorize = %s)"
      ),
      if (type == "nonparametric") "spearman" else "pearson", format(conf_level),
      type == "bayes", format(prior), format(winsorize)
    ),
    seed = CASE_SEED,
    value = if (failed) NULL else to_json_value(value),
    error = if (failed) conditionMessage(out$value) else NULL,
    warnings = as.list(unique(out$warnings)),
    messages = as.list(unique(out$messages)),
    matches_subtitle = matches
  )
}

# The subjects-by-conditions data a within-subjects case analyzed, as ggwithinstats
# analyzed it (within_analysis()): `subjects` in the order ggstatsplot holds them
# (complete_subjects() sorts by subject: numbers numerically, text in the C locale;
# within a subject the rows follow the level order), which is the order dep.effect,
# wmcpAKP and kendalls_w resample; `values`, one row per subject and one column per
# level; `dropped_subjects`, the subjects with a row (x and subject present) that were
# left out for a missing level or value; and the number of rows without x or subject.
within_data <- function(data, case, analysis) {
  x <- case$args$x
  subject <- case$args$subject.id
  levels <- levels(analysis$x)
  subjects <- unique(analysis$.rowid)
  columns <- lapply(levels, function(level) {
    rows <- analysis$x == level
    as.numeric(analysis$y[rows][match(subjects, analysis$.rowid[rows])])
  })
  usable <- !is.na(data[[x]]) & !is.na(data[[subject]])
  candidates <- sort(unique(data[[subject]][usable]))
  list(
    subject_id = subject,
    levels = as.list(levels),
    subjects = as.list(as.character(subjects)),
    subject_order = paste(
      "sorted by subject as complete_subjects() sorts (numbers numerically, text in",
      "the C locale); the analysis holds the rows subject by subject, levels in order"
    ),
    values = lapply(seq_along(subjects), function(i) {
      as.list(vapply(columns, function(column) column[[i]], numeric(1L)))
    }),
    dropped_subjects = as.list(as.character(setdiff(candidates, subjects))),
    rows_without_x_or_subject = sum(!usable)
  )
}

# Group counts from ggstatsplot's own centrality labels (`n.obs`), named by level.
label_counts <- function(layers, x) {
  for (layer in layers) {
    if (identical(layer$geom, "GeomLabelRepel") && length(layer$data) > 0L) {
      return(stats::setNames(
        vapply(layer$data, function(row) as.integer(row[["n.obs"]]), integer(1L)),
        vapply(layer$data, function(row) as.character(row[[x]]), character(1L))
      ))
    }
  }
  NULL
}

run_case <- function(case) {
  dataset <- read_dataset(case)
  data <- dataset$data
  call_args <- case$args
  for (name in case$columns) call_args[[name]] <- as.name(call_args[[name]])
  # R code kept as text in the case: label.expression (an expression ggscatterstats
  # passes to dplyr::filter) and the formula in smooth.line.args.
  if (!is.null(call_args$label.expression)) {
    call_args$label.expression <- str2lang(call_args$label.expression)
  }
  if (is.character(call_args$smooth.line.args$formula)) {
    call_args$smooth.line.args$formula <- stats::as.formula(
      call_args$smooth.line.args$formula, env = globalenv()
    )
  }
  fn <- call("::", as.name("ggstatsplot"), as.name(case$fun))
  cl <- as.call(c(list(fn, data = quote(data)), call_args))
  env <- environment()
  no_levels <- case$fun %in% NO_LEVEL_FUNCTIONS
  bar <- case$fun %in% BAR_FUNCTIONS

  HIDDEN$log <- list()
  HIDDEN$values <- list()
  set.seed(CASE_SEED)
  outcome <- capture({
    plot <- eval(cl, envir = env)
    list(
      stats = ggstatsplot::extract_stats(plot),
      subtitle = ggstatsplot::extract_subtitle(plot),
      caption = ggstatsplot::extract_caption(plot),
      layers = plot_layers(plot, build_all = no_levels || bar),
      texts = if (bar) plot_texts(plot)
    )
  })
  failed <- inherits(outcome$value, "error")
  # Computed after ggstatsplot's call, so it cannot touch the random-number stream
  # the call saw (it draws no random numbers anyway).
  analysis <- ANALYSES[[case$fun]](data, case)
  hidden <- name_hidden(case, HIDDEN$log, failed, design = analysis$design)
  if (bar && !is.null(analysis$y)) {
    # The per-bar tests run in the order of y's levels that hold data.
    present <- levels(droplevels(analysis$y))
    tests <- which(vapply(hidden, function(entry) entry$component == "proportion_test", logical(1L)))
    if (length(tests) == length(present)) {
      for (i in seq_along(tests)) hidden[[tests[[i]]]]$level <- present[[i]]
    }
  }
  fixture <- list(
    schema_version = SCHEMA_VERSION,
    name = case$name,
    `function` = case$fun,
    call = deparse_text(cl),
    dataset = list(
      file = basename(dataset$path),
      sha256 = file_sha256(dataset$path),
      rows = nrow(data),
      factor_levels = if (is.null(case$factor_levels)) NULL else lapply(case$factor_levels, as.list)
    ),
    args = case$args,
    seed = CASE_SEED,
    # A one-sample or correlation design has no levels: an empty list and an empty
    # object, and `analyzed_data` holds what it analyzed.
    # ggbarstats and ggpiestats: the levels of x as ggstatsplot holds them, declared
    # empty levels included (count 0); none when the data could not be prepared.
    analysis_levels = if (no_levels) list() else as.list(levels(analysis$x)),
    n_per_level = if (no_levels || is.null(analysis$x)) {
      stats::setNames(list(), character())
    } else {
      as.list(table(analysis$x))
    },
    error = if (failed) conditionMessage(outcome$value) else NULL,
    expected_error = case$expect_error,
    # What the hidden calls returned before ggstatsplot failed, one entry per call.
    partial = if (failed) {
      lapply(seq_along(hidden), function(i) {
        list(component = hidden[[i]]$component, value = to_json_value(HIDDEN$values[[i]]))
      })
    },
    r_warnings = as.list(outcome$warnings),
    r_messages = as.list(outcome$messages),
    suppressed = hidden,
    expected_hidden_errors = as.list(case$expect_hidden_errors),
    subtitle = if (failed) NULL else to_json_value(outcome$value$subtitle),
    caption = if (failed) NULL else to_json_value(outcome$value$caption),
    stats = if (failed) NULL else to_json_value(outcome$value$stats),
    plot_layers = if (failed) NULL else outcome$value$layers,
    # When ggstatsplot failed after computing the subtitle, its effect-size CI is in
    # `partial` and is referenced the same way.
    bootstrap_reference = if (failed) {
      partial_subtitle <- NULL
      for (i in seq_along(hidden)) {
        if (identical(hidden[[i]]$component, "subtitle")) partial_subtitle <- HIDDEN$values[[i]]
      }
      if (is.null(partial_subtitle)) {
        NULL
      } else {
        bootstrap_reference(case, analysis, list(subtitle_data = partial_subtitle))
      }
    } else {
      bootstrap_reference(case, analysis, outcome$value$stats)
    },
    versions = VERSIONS
  )
  if (case$fun == "ggwithinstats") {
    # After n_per_level (each level's count is the number of complete subjects).
    at <- match("n_per_level", names(fixture))
    fixture <- append(fixture, list(within_data = within_data(data, case, analysis)), after = at)
  }
  if (bar) {
    at <- match("n_per_level", names(fixture))
    fixture <- append(
      fixture, list(contingency_data = contingency_data(data, case, analysis)), after = at
    )
    at <- match("expected_hidden_errors", names(fixture))
    fixture <- append(
      fixture,
      list(
        expected_na_values = as.list(case$expect_na_values),
        expected_null_outputs = as.list(case$expect_null_outputs)
      ),
      after = at
    )
    at <- match("plot_layers", names(fixture))
    fixture <- append(
      fixture, list(plot_texts = if (failed) NULL else outcome$value$texts), after = at
    )
  }
  if (no_levels) {
    at <- match("n_per_level", names(fixture))
    fixture <- append(
      fixture, list(analyzed_data = analyzed_data(data, case, analysis)), after = at
    )
    at <- match("expected_hidden_errors", names(fixture))
    fixture <- append(
      fixture,
      list(
        expected_empty_layers = as.list(case$expect_empty_layers),
        expected_na_values = as.list(case$expect_na_values)
      ),
      after = at
    )
    extra <- if (case$fun == "gghistostats") {
      # The centrality line's value and its label (drawn on a secondary x axis, not a
      # layer): what centrality_description() returned; null without the line.
      centrality <- NULL
      for (i in seq_along(hidden)) {
        if (identical(hidden[[i]]$component, "centrality_description")) {
          centrality <- HIDDEN$values[[i]]
        }
      }
      list(centrality = to_json_value(centrality))
    } else {
      subtitle_row <- if (failed) NULL else outcome$value$stats$subtitle_data
      list(correlation = correlation_record(case, analysis, subtitle_row))
    }
    at <- match("plot_layers", names(fixture))
    fixture <- append(fixture, extra, after = at)
  }
  fixture
}

# The checks of a gghistostats or ggscatterstats case that replace the per-level
# counts: the number of analyzed values (pairs) must be ggstatsplot's own n (the
# subtitle's n.obs; gghistostats' centrality table too), and the layers the
# arguments imply must be there with their data built.
no_level_problems <- function(case, fixture) {
  out <- character()
  n <- fixture$analyzed_data$n
  geoms <- vapply(fixture$plot_layers, function(layer) layer$geom, character(1L))
  subtitle_rows <- fixture$stats$subtitle_data
  if (length(subtitle_rows) > 0L) {
    counts <- unique(vapply(subtitle_rows, function(row) as.integer(row[["n.obs"]]), integer(1L)))
    if (!identical(counts, as.integer(n))) {
      out <- c(out, sprintf("analyzed %d values; the subtitle's n.obs is %s", n, paste(counts, collapse = ",")))
    }
  }
  if (case$fun == "gghistostats") {
    centrality <- fixture$centrality
    wants_centrality <- !isFALSE(case$args$centrality.plotting)
    if (wants_centrality != !is.null(centrality)) {
      out <- c(out, if (wants_centrality) "no centrality table" else "unexpected centrality table")
    }
    if (!is.null(centrality) && !identical(as.integer(centrality[[1L]][["n.obs"]]), as.integer(n))) {
      out <- c(out, "the centrality table's n.obs differs from the analyzed values")
    }
    expected <- c("GeomBar", if (wants_centrality) "GeomVline")
    if (!identical(unname(geoms), expected)) {
      out <- c(out, sprintf("layers %s; expected %s", paste(geoms, collapse = ", "), paste(expected, collapse = ", ")))
    }
  } else {
    record <- fixture$correlation
    if (!is.null(record$error) && length(subtitle_rows) > 0L) {
      out <- c(out, paste("correlation::correlation failed although the subtitle exists:", record$error))
    }
    if (length(subtitle_rows) > 0L && !isTRUE(record$matches_subtitle)) {
      out <- c(out, "correlation::correlation's row does not reproduce the subtitle")
    }
    if (!("GeomSmooth" %in% geoms)) out <- c(out, "no smooth line layer")
    sides <- sum(grepl("side", geoms, ignore.case = TRUE))
    if (sides != (if (isFALSE(case$args$marginal)) 0L else 2L)) {
      out <- c(out, sprintf("%d marginal histogram layers (%s)", sides, paste(geoms, collapse = ", ")))
    }
    if (!is.null(case$args$label.var) != ("GeomLabelRepel" %in% geoms)) {
      out <- c(out, "point labels do not follow label.var")
    }
  }
  # No number in the subtitle or caption table is NA unless the case declares it:
  # ggstatsplot prints such a number as "NA" or "", with at most a warning.
  for (table in c("subtitle", "caption")) {
    rows <- fixture$stats[[paste0(table, "_data")]]
    numbers <- c("statistic", "p.value", "estimate", "conf.low", "conf.high", "bf10")
    missing_value <- function(v) {
      is.null(v) || identical(v, "NaN") || (length(v) == 1L && is.atomic(v) && is.na(v))
    }
    has_na <- any(vapply(rows, function(row) {
      any(vapply(intersect(numbers, names(row)), function(column) {
        missing_value(row[[column]])
      }, logical(1L)))
    }, logical(1L)))
    if (has_na != (table %in% case$expect_na_values)) {
      out <- c(out, sprintf(
        "%s table %s", table, if (has_na) "holds NA" else "holds no NA as declared"
      ))
    }
  }
  # Every layer but the raw points was built, with rows unless the case declares
  # that ggplot2 builds none.
  for (layer in fixture$plot_layers) {
    if (layer$geom == "GeomPoint") next
    empty <- length(layer$built) == 0L
    if (empty != (layer$geom %in% case$expect_empty_layers)) {
      out <- c(out, sprintf(
        "layer %s %s", layer$geom, if (empty) "has no built data" else "is not empty as declared"
      ))
    }
  }
  out
}

# The checks of a ggbarstats or ggpiestats case that replace the per-level counts:
# the table generate.R recorded must be the one ggstatsplot counted (its
# descriptive_data, every cell; its one_sample_data, every bar's n; the subtitle's
# n.obs), the per-bar tests must be NA only where the case declares a swallowed
# error, no reported number may be NA unless declared, and the layers must be the
# ones the arguments imply, each built with rows.
bar_problems <- function(case, fixture) {
  out <- character()
  data <- fixture$contingency_data
  design <- hidden_call_levels(case, fixture)
  x_levels <- unlist(data$x_levels)
  y_levels <- unlist(data$y_levels)
  one_way <- is.null(case$args$y)
  cell <- function(x, y) {
    i <- match(x, x_levels)
    j <- if (one_way) 1L else match(y, y_levels)
    if (is.na(i) || is.na(j)) NA_integer_ else as.integer(data$table[[i]][[j]])
  }
  # Every cell ggstatsplot labels (descriptive_data: one row per bar and level of x,
  # zeros included) holds the recorded count, and together they hold all n.
  described <- fixture$stats$descriptive_data
  total <- 0
  for (row in described) {
    count <- as.integer(row$counts)
    expected <- cell(as.character(row[[case$args$x]]), if (!one_way) as.character(row[[case$args$y]]))
    if (!identical(count, expected)) {
      out <- c(out, sprintf(
        "descriptive_data counts %s for %s; the table has %s", count,
        paste(row[[case$args$x]], if (!one_way) row[[case$args$y]]), expected
      ))
    }
    total <- total + count
  }
  if (!identical(as.integer(total), as.integer(data$n))) {
    out <- c(out, sprintf("descriptive_data counts %s rows; the table %s", total, data$n))
  }
  for (row in fixture$stats$subtitle_data) {
    if (!identical(as.integer(row[["n.obs"]]), as.integer(data$n))) {
      out <- c(out, sprintf("the subtitle's n.obs is %s; the table has %s", row[["n.obs"]], data$n))
    }
  }
  # The per-bar tests: one row per bar that holds data, with that bar's n.
  bars <- fixture$stats$one_sample_data
  if (one_way != (length(bars) == 0L)) {
    out <- c(out, if (one_way) "one_sample_data without y" else "no one_sample_data")
  }
  for (row in bars) {
    level <- as.character(row[[case$args$y]])
    j <- match(level, y_levels)
    n_bar <- if (is.na(j)) NA else sum(vapply(data$table, function(r) as.numeric(r[[j]]), numeric(1L)))
    if (!isTRUE(as.numeric(row$counts) == n_bar)) {
      out <- c(out, sprintf("bar %s: one_sample_data n %s, the table %s", level, row$counts, n_bar))
    }
  }
  missing_value <- function(v) {
    is.null(v) || identical(v, "NaN") || (length(v) == 1L && is.atomic(v) && is.na(v))
  }
  numbers <- c("statistic", "p.value", "estimate", "conf.low", "conf.high", "bf10")
  tables <- c(
    subtitle = "subtitle_data", caption = "caption_data",
    pairwise = "pairwise_comparisons_data", one_sample = "one_sample_data"
  )
  for (table in names(tables)) {
    rows <- fixture$stats[[tables[[table]]]]
    has_na <- any(vapply(rows, function(row) {
      any(vapply(intersect(c(numbers, "p.value.adj"), names(row)), function(column) {
        missing_value(row[[column]])
      }, logical(1L)))
    }, logical(1L)))
    if (has_na != (table %in% case$expect_na_values)) {
      out <- c(out, sprintf(
        "%s table %s", table, if (has_na) "holds NA" else "holds no NA as declared"
      ))
    }
  }
  # The layers: the bars (GeomCol) and their labels (GeomLabel), then for ggbarstats
  # with y the proportion tests' p-values above the bars when drawn and the n under
  # each bar, without y the n below the single bar (annotate()); for ggpiestats the
  # proportion tests beside each pie when drawn (only with more than one pie).
  geoms <- vapply(fixture$plot_layers, function(layer) layer$geom, character(1L))
  tests <- bar_proportion_test(case, design)
  expected <- if (case$fun == "ggbarstats") {
    c("GeomCol", "GeomLabel", if (!one_way && tests) "GeomText", "GeomText")
  } else {
    c("GeomCol", "GeomLabel", if (design$y_levels > 1L && tests) "GeomText")
  }
  if (!identical(unname(geoms), expected)) {
    out <- c(out, sprintf(
      "layers %s; expected %s", paste(geoms, collapse = ", "), paste(expected, collapse = ", ")
    ))
  }
  for (layer in fixture$plot_layers) {
    if (length(layer$built) == 0L) out <- c(out, sprintf("layer %s has no built data", layer$geom))
  }
  out
}

# What expected_hidden_calls() needs from a written fixture: the levels R analyzed,
# or for ggbarstats and ggpiestats the bar_design() of the table it recorded.
hidden_call_levels <- function(case, fixture) {
  if (!(case$fun %in% BAR_FUNCTIONS)) {
    return(unlist(fixture$analysis_levels))
  }
  data <- fixture$contingency_data
  if (!is.null(data$error)) {
    return(bar_design(0L, 0L, 0L))
  }
  column_totals <- if (length(data$y_levels) == 0L) {
    numeric()
  } else {
    vapply(seq_along(data$y_levels), function(j) {
      sum(vapply(data$table, function(row) as.numeric(row[[j]]), numeric(1L)))
    }, numeric(1L))
  }
  bar_design(length(data$x_levels), length(data$y_levels), sum(column_totals > 0))
}

# Everything about a finished case that differs from what the case declares.
outcome_problems <- function(case, fixture) {
  errored <- !is.null(fixture$error)
  expected <- case$expect_error
  if (errored != !is.null(expected)) {
    return(if (errored) paste("error:", fixture$error) else "expected an error, got none")
  }
  if (errored) {
    # The declared call must be the one that failed, with the declared message, and
    # every hidden call before it must have run.
    out <- character()
    if (!grepl(expected$message, fixture$error, fixed = TRUE)) {
      out <- c(out, paste("unexpected error:", fixture$error))
    }
    calls <- vapply(fixture$suppressed, function(entry) entry$component, character(1L))
    errors <- lapply(fixture$suppressed, function(entry) entry$error)
    last <- length(calls)
    if (identical(expected$component, "data")) {
      # ggbarstats or ggpiestats failed while preparing the data (before any hidden
      # call); generate.R's own preparation, the same function, failed the same way.
      if (last != 0L || !identical(fixture$contingency_data$error, fixture$error)) {
        out <- c(out, sprintf(
          "expected the data preparation to fail before any hidden call; hidden calls were %s",
          paste(calls, collapse = ", ")
        ))
      }
      return(out)
    }
    planned <- expected_hidden_calls(case, hidden_call_levels(case, fixture))
    # A call before the failing one may have failed only where the case declares
    # that ggstatsplot swallows its error (edge_rm_all_tied: the subtitle).
    earlier_failed <- !vapply(errors[-last], is.null, logical(1L))
    undeclared <- earlier_failed & !(calls[-last] %in% case$expect_hidden_errors)
    if (last == 0L || calls[[last]] != expected$component || is.null(errors[[last]]) ||
      !identical(unname(calls), planned[seq_len(min(last, length(planned)))]) ||
      any(undeclared) ||
      !setequal(calls[-last][earlier_failed], case$expect_hidden_errors)) {
      out <- c(out, sprintf(
        "expected %s to fail after %s; hidden calls were %s",
        expected$component, paste(planned, collapse = ", "), paste(calls, collapse = ", ")
      ))
    }
    return(out)
  }
  out <- character()
  hidden_errors <- Filter(function(entry) !is.null(entry$error), fixture$suppressed)
  failed_parts <- vapply(hidden_errors, function(entry) entry$component, character(1L))
  for (entry in hidden_errors) {
    if (!(entry$component %in% case$expect_hidden_errors)) {
      out <- c(out, sprintf("ggstatsplot swallowed a %s error: %s", entry$component, entry$error))
    }
  }
  for (part in setdiff(case$expect_hidden_errors, failed_parts)) {
    out <- c(out, sprintf("expected ggstatsplot to swallow a %s error", part))
  }
  # Every hidden call, in order, must be the one the arguments imply.
  calls <- vapply(fixture$suppressed, function(entry) entry$component, character(1L))
  expected_calls <- expected_hidden_calls(case, hidden_call_levels(case, fixture))
  if (!identical(unname(calls), expected_calls)) {
    out <- c(out, sprintf(
      "hidden calls %s; expected %s",
      paste(calls, collapse = ", "), paste(expected_calls, collapse = ", ")
    ))
  }
  # ggbetweenstats and ggwithinstats draw the subtitle unless results.subtitle =
  # FALSE, and the Bayes caption only for type "parametric" with bf.message (ggstatsplot's bf.condition),
  # never without the subtitle; the tables behind them follow the same rule.
  bar <- case$fun %in% BAR_FUNCTIONS
  evaluated <- expected_eval_f(case, if (bar) hidden_call_levels(case, fixture))
  # A component may also return nothing without an error where the case declares it
  # (ggbarstats and ggpiestats: `expect_null_outputs`).
  failed_parts <- c(failed_parts, case$expect_null_outputs)
  wants_subtitle <- "subtitle" %in% evaluated
  wants_caption <- "caption" %in% evaluated
  has_subtitle <- !is.null(fixture$subtitle)
  has_caption <- !is.null(fixture$caption)
  has_subtitle_data <- length(fixture$stats$subtitle_data) > 0L
  has_caption_data <- length(fixture$stats$caption_data) > 0L
  if ((wants_subtitle != has_subtitle) && !("subtitle" %in% failed_parts)) {
    out <- c(out, if (wants_subtitle) "no subtitle" else "unexpected subtitle")
  }
  if ((wants_caption != has_caption) && !("caption" %in% failed_parts)) {
    out <- c(out, if (wants_caption) "no Bayes caption" else "unexpected caption")
  }
  if (has_subtitle != has_subtitle_data) {
    out <- c(out, "subtitle and subtitle_data disagree")
  }
  if (has_caption != has_caption_data) {
    out <- c(out, "caption and caption_data disagree")
  }
  # Pairwise comparisons exist exactly when ggbetweenstats or ggwithinstats computes them.
  wants_pairwise <- "pairwise_comparisons" %in% expected_calls
  has_pairwise <- length(fixture$stats$pairwise_comparisons_data) > 0L
  if (wants_pairwise != has_pairwise && !("pairwise_comparisons" %in% failed_parts)) {
    out <- c(out, if (wants_pairwise) "no pairwise comparisons" else "unexpected pairwise comparisons")
  }
  if (case$fun %in% NO_LEVEL_FUNCTIONS) {
    return(c(out, no_level_problems(case, fixture)))
  }
  if (bar) {
    return(c(out, bar_problems(case, fixture)))
  }
  # n_per_level must be what ggstatsplot itself counted.
  labels <- label_counts(fixture$plot_layers, case$args$x)
  counts <- unlist(fixture$n_per_level)
  if (is.null(labels)) {
    out <- c(out, "no centrality label layer to check n_per_level against")
  } else if (!identical(sort(names(labels)), sort(names(counts))) ||
    !identical(unname(as.integer(counts[names(labels)])), unname(labels))) {
    out <- c(out, sprintf(
      "n_per_level %s differs from ggstatsplot's n.obs %s",
      paste(names(counts), counts, sep = "=", collapse = ","),
      paste(names(labels), labels, sep = "=", collapse = ",")
    ))
  }
  out
}

# Probes ----------------------------------------------------------------------------
#
# Probe files (probe-<topic>.json) record what R functions behind ggstatsplot return
# on grids the fixtures cannot cover: number formatting, p.adjust, ptukey and the
# omega squared CI. Doubles go out as C99 hex strings (sprintf("%a"); Python reads
# them with float.fromhex) next to the 15-digit numbers, because 15 digits cannot
# place a value on the right side of a decimal tie or pin a last bit; R's NA is the
# string "NA". The manifest hashes the probes like the fixtures. Functions of GPL
# packages are called as black boxes; nothing here prints or deparses their code.

hex <- function(x) sprintf("%a", x)

# Run `expr`; stop the run on an error, return its value and warnings otherwise.
probe_call <- function(what, expr) {
  out <- capture(expr)
  if (inherits(out$value, "error")) {
    stop("probe ", what, " failed: ", conditionMessage(out$value))
  }
  out
}

# Run `expr`; record an error as text instead of stopping (for probe grid points
# where an error is itself a finding).
probe_attempt <- function(expr) {
  out <- capture(expr)
  failed <- inherits(out$value, "error")
  list(
    value = if (failed) NULL else out$value,
    error = if (failed) conditionMessage(out$value) else NULL,
    warnings = as.list(unique(out$warnings))
  )
}

probe_file <- function(name, description, body, datasets = list()) {
  c(
    list(
      schema_version = SCHEMA_VERSION,
      name = name,
      kind = "probe",
      description = description,
      # CSV files the probe read, with their SHA-256 (checked like the fixtures').
      datasets = datasets
    ),
    body,
    list(versions = VERSIONS)
  )
}

# probe-format: insight::format_value and statsExpressions 2.2.1's internal .to_char
# (what every subtitle number goes through) over a grid of doubles and digits 0 to
# 4; the route degrees of freedom take (pasted unquoted into the expression, parsed
# by plotmath, deparsed when the expression is printed); .prettyNum for counts; the
# CI level label paste0(conf.level * 100L, "%"); ggstatsplot's pairwise axis label;
# a few whole expressions; and R's own doubles for centrality on tie data.
format_probe <- function() {
  se_ns <- asNamespace("statsExpressions")
  to_char <- get(".to_char", envir = se_ns)
  pretty_num <- get(".prettyNum", envir = se_ns)
  p_adjust_expr <- get(".pairwise_p_adjust_expr", envir = se_ns)
  seclabel <- get(".pairwise_seclabel", envir = asNamespace("ggstatsplot"))
  fv <- insight::format_value
  dep <- function(e) paste(deparse(e, width.cutoff = 500L), collapse = " ")
  safe <- function(expr) {
    tryCatch(expr, error = function(e) paste0("<error: ", conditionMessage(e), ">"))
  }
  ulp_up <- function(x) x * (1 + 2^-52)
  ulp_down <- function(x) x * (1 - 2^-52)

  base <- c(
    0, 5e-324, .Machine$double.xmin, 1e-300, 1e-15, 1e-5,
    0.000999, 0.001, 0.00499, 0.005, 0.00874, 0.0095, 0.00995, 0.009995, 0.0099999,
    0.01, 0.0100000001, 0.011, 0.015, 0.025, 0.045, 0.125, 0.135, 0.145, 0.285,
    0.345, 0.5, 1.5, 2.5, 3.5, 1.005, 2.675, 5.155, 5.435, 9.785, 9.995, 12.225,
    12.295, 12.905, 52.475, 52.985, 54.535, 99.995, 999.995, 9999.995, 99999.4,
    99999.995, 99999.996, 1e5, 100000.4, 123456, 123456.789, 999999.5, 1e6,
    1234567.89, 1e15, 1e16, 1e100, 1e300, .Machine$double.xmax,
    1 / 3, 2 / 3, 0.1 + 0.2, 0.707, sqrt(2) / 2, 0.95, 17.1284186166441,
    97.401587118856
  )
  # Decimal ties k + 0.5 units of the last place for digits 0 to 4, as R parses them.
  ties <- unlist(lapply(0:4, function(d) (c(0:20, 99, 999) + 0.5) / 10^d))
  grid <- c(base, -base, ties, -ties)
  grid <- c(grid, ulp_up(grid[grid != 0]), ulp_down(grid[grid != 0]))
  # unique() treats -0 as 0, so negative zero is added after it (x * (1 + 2^-52)
  # overflows to Inf at the largest double).
  grid <- c(unique(c(grid, Inf, -Inf)), -0, NA_real_, NaN)

  values <- vector("list", length(grid) * 5L)
  i <- 0L
  for (x in grid) {
    for (d in 0:4) {
      tc <- safe(to_char(x, d))
      i <- i + 1L
      values[[i]] <- list(
        hex = hex(x),
        # sprintf("%a") may print -0 as 0x0p+0; this says which zero it is.
        negative_zero = isTRUE(x == 0 && 1 / x < 0),
        digits = d,
        format_value = safe(fv(x, d)),
        to_char = tc,
        sprintf_f = sprintf("%.*f", d, x),
        sprintf_e = sprintf("%.*e", d, x),
        round_17g = sprintf("%.17g", round(x, d)),
        df_deparse = safe(dep(str2lang(tc)))
      )
    }
  }

  n_grid <- c(
    0, 1, 9, 10, 99, 999, 1000, 1234, 9999, 99999, 1e5, 123456, 999999, 1e6,
    1234567, 1e7, 1e9
  )
  counts <- c(
    lapply(n_grid, function(n) {
      list(hex = hex(n), type = "double", prettyNum = safe(pretty_num(n)))
    }),
    lapply(n_grid, function(n) {
      list(hex = hex(n), type = "integer", prettyNum = safe(pretty_num(as.integer(n))))
    })
  )

  level_grid <- c(0.5, 0.8, 0.89, 0.9, 0.95, 0.975, 0.99, 0.999, 0.9999)
  levels <- lapply(level_grid, function(l) {
    list(hex = hex(l), label = paste0(l * 100L, "%"))
  })

  alpha_grid <- c(0.05, 0.01, 0.001, 0.0001, 1e-5, 0.1, 0.123456789)
  alpha <- lapply(alpha_grid, function(a) {
    list(
      hex = hex(a),
      label = safe(dep(seclabel("Games-Howell test", pairwise.alpha = a)[[1L]]))
    )
  })

  stats_df <- data.frame(
    statistic = 123456.7, df = 1e5, df.error = 99999.996, p.value = 0, estimate = -0,
    conf.level = 0.9, conf.low = -Inf, conf.high = NA_real_, n.obs = 123456L,
    method = "One-way analysis of means (not assuming equal variances)",
    effectsize = "Omega2"
  )
  end_to_end <- list(
    subtitle = safe(dep(statsExpressions::add_expression_col(
      stats_df, digits = 2L, digits.df = 0L, digits.df.error = 2L
    )$expression[[1L]])),
    subtitle_digits3 = safe(dep(statsExpressions::add_expression_col(
      stats_df, digits = 3L, digits.df = 0L, digits.df.error = 3L
    )$expression[[1L]])),
    pairwise = safe(lapply(
      p_adjust_expr(
        data.frame(p.value = c(0, 1e-300, 0.004999, 0.00999, 1)), "none", 2L,
        "Games-Howell"
      )$expression,
      as.character
    )),
    bayes = safe(dep(statsExpressions::add_expression_col(data.frame(
      bf10 = exp(-123456.7), estimate = 0.1234, conf.level = 0.95, conf.low = -0.0049,
      conf.high = 1e6, prior.scale = 0.707, conf.method = "ETI",
      method = "Bayesian t-test", effectsize = "Bayesian t-test", n.obs = 10L
    ))$expression[[1L]]))
  )

  # R's doubles for centrality on data with decimal ties (mean() accumulates in
  # long double).
  tie_data <- data.frame(
    g = rep(c("a", "b"), each = 8L),
    y = c(
      10.1, 9.2, 9.7, 10.4, 9.5, 9.9, 9.8, 9.68,
      12.2, 12.39, 12.1, 12.3, 12.31, 12.29, 12.6, 11.9
    )
  )
  centrality <- lapply(c("parametric", "nonparametric", "robust"), function(type) {
    out <- probe_call(
      paste("centrality_description", type),
      statsExpressions::centrality_description(tie_data, g, y, type = type)
    )$value
    list(
      type = type,
      values_hex = as.list(vapply(out$y, hex, "")),
      labels = as.list(vapply(out$expression, dep, ""))
    )
  })

  probe_file(
    "probe-format",
    "insight::format_value, statsExpressions .to_char and .prettyNum, the df route, CI level labels",
    list(
      values = values, counts = counts, conf_levels = levels, pairwise_alpha = alpha,
      end_to_end = end_to_end, centrality = centrality
    )
  )
}

# probe-padjust: stats::p.adjust for every method on fixed vectors: Hommel exactly,
# the order of operations of the BH and BY quotients, BY's harmonic sum near
# n = 1008, ties, n of 1 and 2, exact 0 and 1, values within an ulp of 1, capping at
# 1, and missing values.
padjust_probe <- function() {
  methods <- c("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none")
  stopifnot(setequal(methods, p.adjust.methods))
  vectors <- list(
    n1_half = 0.5,
    n1_zero = 0,
    n1_one = 1,
    n1_small = 0.03,
    n2 = c(0.01, 0.04),
    n2_tied = c(0.02, 0.02),
    n2_far = c(0.9, 0.03),
    wikipedia_holm = c(0.01, 0.04, 0.03, 0.005),
    n5 = c(0.01, 0.04, 0.03, 0.005, 0.2),
    n5_ties = c(0.02, 0.01, 0.02, 0.5, 0.02),
    n5_all_tied = rep(0.3, 5),
    n5_near_one = c(0.999, 0.9999, 1 - 2^-53, 0.95384, 1),
    n5_zeros_and_cap = c(0, 0, 0.6, 0.7, 0.8),
    n6_hommel_differs = c(0.011, 0.012, 0.013, 0.04, 0.045, 0.05),
    discriminate_30 = (1:30) * 0.0331,
    discriminate_sqrt = sqrt((1:25) / 26),
    statsmodels_hommel_40 = c(
      0.00116, 0.00924, 0.01075, 0.01437, 0.01784, 0.01918, 0.02751, 0.02871,
      0.03054, 0.03246, 0.04259, 0.06879, 0.0691, 0.08081, 0.08593, 0.08993,
      0.09386, 0.09412, 0.09718, 0.09758, 0.09781, 0.09788, 0.13282, 0.20191,
      0.21757, 0.24031, 0.26061, 0.26762, 0.29474, 0.32901, 0.41386, 0.51479,
      0.52461, 0.53389, 0.56276, 0.62967, 0.72178, 0.73403, 0.87182, 0.95384
    ),
    by_n1007 = (1:1007) / 1009 / 1000,
    by_n1008 = (1:1008) / 1009 / 1000,
    with_na = c(0.01, NA, 0.03, 0.04, NA, 0.2)
  )
  cases <- lapply(names(vectors), function(name) {
    p <- vectors[[name]]
    adjusted <- lapply(methods, function(method) {
      a <- probe_call(paste("p.adjust", method, name), stats::p.adjust(p, method = method))
      if (length(a$warnings) > 0L) stop("p.adjust warned: ", a$warnings[[1L]])
      list(hex = I(hex(a$value)), value = I(encode_doubles(a$value)))
    })
    names(adjusted) <- methods
    list(name = name, p_hex = I(hex(p)), p = I(encode_doubles(p)), adjusted = adjusted)
  })
  probe_file(
    "probe-padjust",
    "stats::p.adjust, every method, on fixed vectors",
    list(
      methods = as.list(methods),
      platform = R.version$platform,
      longdouble_digits = .Machine$longdouble.digits,
      harmonic_sum_hex = as.list(hex(vapply(c(1007, 1008), function(n) sum(1 / seq_len(n)), 0))),
      cases = cases
    )
  )
}

# probe-ptukey: R's studentized range upper tail both ways (lower.tail = FALSE and
# 1 - cdf) on a grid, and PMCMRplus::gamesHowellTest's unadjusted p for every pair
# of every parametric case with three or more groups, next to ptukey at the pair's
# Welch df computed here.
ptukey_probe <- function(gh_cases) {
  grid <- expand.grid(
    q = c(1:8, 10, 12, 14, 17, 20, 25, 30),
    nmeans = c(2:6, 10, 20),
    df = c(1, 1.5, 2, 5, 10, 20, 50, 94.6978, 200, 1000, 25000, 1e5, Inf),
    KEEP.OUT.ATTRS = FALSE
  )
  tail_row <- function(q, nmeans, df) {
    upper <- probe_call("ptukey upper tail", stats::ptukey(q, nmeans, df, lower.tail = FALSE))
    cdf <- probe_call("ptukey", stats::ptukey(q, nmeans, df))
    list(
      q_hex = hex(q), nmeans = as.integer(nmeans), df_hex = hex(df),
      q = q, df = encode_doubles(df),
      upper_tail_hex = hex(upper$value), upper_tail = encode_doubles(upper$value),
      cdf_hex = hex(cdf$value),
      one_minus_cdf_hex = hex(1 - cdf$value), one_minus_cdf = encode_doubles(1 - cdf$value),
      warnings = as.list(unique(c(upper$warnings, cdf$warnings)))
    )
  }
  tails <- lapply(seq_len(nrow(grid)), function(i) {
    tail_row(grid$q[[i]], grid$nmeans[[i]], grid$df[[i]])
  })

  games_howell <- lapply(gh_cases, function(case) {
    dataset <- read_dataset(case)
    analysis <- between_analysis(dataset$data, case$args$x, case$args$y)
    rows <- statsexpressions_rows(analysis)
    levels <- levels(rows$x)
    k <- length(levels)
    out <- probe_call(
      paste("gamesHowellTest for", case$name),
      PMCMRplus::gamesHowellTest(x = rows$y, g = rows$x, p.adjust.method = "none")
    )
    test <- out$value
    pairs <- list()
    for (i in seq_len(k - 1L)) {
      for (j in (i + 1L):k) {
        a <- rows$y[rows$x == levels[[i]]]
        b <- rows$y[rows$x == levels[[j]]]
        va <- stats::var(a) / length(a)
        vb <- stats::var(b) / length(b)
        welch_df <- (va + vb)^2 / (va^2 / (length(a) - 1) + vb^2 / (length(b) - 1))
        statistic <- test$statistic[levels[[j]], levels[[i]]]
        p_value <- test$p.value[levels[[j]], levels[[i]]]
        tail <- tail_row(abs(statistic), k, welch_df)
        pairs[[length(pairs) + 1L]] <- list(
          group1 = levels[[i]], group2 = levels[[j]],
          statistic = statistic, statistic_hex = hex(statistic),
          p_value = p_value, p_value_hex = hex(p_value),
          welch_df = welch_df, welch_df_hex = hex(welch_df),
          ptukey_upper_tail_hex = tail$upper_tail_hex,
          ptukey_one_minus_cdf_hex = tail$one_minus_cdf_hex,
          ptukey_warnings = tail$warnings
        )
      }
    }
    list(
      case = case$name,
      dataset = list(file = basename(dataset$path), sha256 = file_sha256(dataset$path)),
      x = case$args$x, y = case$args$y,
      levels = as.list(levels),
      n_per_level = as.list(table(rows$x)),
      call = "PMCMRplus::gamesHowellTest(x = y, g = x, p.adjust.method = \"none\")",
      method = to_json_value(test$method),
      p_adjust_method = to_json_value(test$p.adjust.method),
      statistic_rows = as.list(rownames(test$statistic)),
      statistic_columns = as.list(colnames(test$statistic)),
      r_warnings = as.list(unique(out$warnings)),
      pairs = pairs
    )
  })
  datasets <- unique(lapply(games_howell, function(entry) entry$dataset))
  probe_file(
    "probe-ptukey",
    "stats::ptukey upper tail and 1 - cdf on a grid; PMCMRplus::gamesHowellTest unadjusted p per pair",
    list(tails = tails, games_howell = games_howell),
    datasets = datasets
  )
}

# probe-omega: effectsize::F_to_omega2 on a grid of F, df and CI settings; the
# F value F_w = omega2 df2 / ((1 - omega2) df1) its estimate corresponds to as a
# partial eta squared; effectsize::F_to_eta2 at F_w; the noncentral F probability
# at the noncentrality parameters behind each reported bound (lambda = bound df2 /
# (1 - bound)); and, for "greater", an accurate lower bound from uniroot with
# tol = 1e-12.
omega_probe <- function() {
  grid <- expand.grid(
    f = c(0.5, 1, 1.3, 2.5, 5.18, 10, 31.6, 500),
    df1 = 2:5,
    df2 = c(5.5, 13.23, 18.03, 97.4, 200, 1e4),
    ci = c(0.9, 0.95, 0.99),
    alternative = c("greater", "less", "two.sided"),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  # The case the Phase 1 omega squared investigation singled out.
  known <- data.frame(
    f = 10, df1 = 2L, df2 = 18.0318456017699, ci = 0.95,
    alternative = c("greater", "less", "two.sided"), stringsAsFactors = FALSE
  )
  grid <- rbind(grid, known)

  accurate_lower <- function(f_w, df1, df2, ci) {
    if (!(f_w > 0) || stats::pf(f_w, df1, df2, ncp = 0) <= ci) {
      return(list(lambda = 0, bound = 0, iterations = 0L, precision = 0))
    }
    hi <- 1
    while (stats::pf(f_w, df1, df2, ncp = hi) > ci) hi <- hi * 2
    root <- stats::uniroot(
      function(l) stats::pf(f_w, df1, df2, ncp = l) - ci, c(0, hi), tol = 1e-12
    )
    list(
      lambda = root$root, bound = root$root / (root$root + df2),
      iterations = root$iter, precision = root$estim.prec
    )
  }
  # P(F' <= f_w) at the noncentrality parameter behind an eta-squared-scale bound.
  at_bound <- function(f_w, df1, df2, bound) {
    if (!(is.finite(bound) && bound > 0 && bound < 1)) {
      return(NULL)
    }
    lambda <- bound * df2 / (1 - bound)
    list(lambda_hex = hex(lambda), cdf_hex = hex(stats::pf(f_w, df1, df2, ncp = lambda)))
  }
  number_and_hex <- function(x) {
    if (is.null(x)) {
      return(list(value = NULL, hex = NULL))
    }
    x <- as.numeric(x)
    list(value = encode_doubles(x), hex = hex(x))
  }

  rows <- lapply(seq_len(nrow(grid)), function(i) {
    f <- grid$f[[i]]
    df1 <- grid$df1[[i]]
    df2 <- grid$df2[[i]]
    ci <- grid$ci[[i]]
    alternative <- grid$alternative[[i]]
    omega <- probe_attempt(effectsize::F_to_omega2(
      f, df1, df2, ci = ci, alternative = alternative
    ))
    o <- omega$value
    estimate <- if (is.null(o)) NA_real_ else as.numeric(o[[1L]])
    f_w <- if (is.na(estimate)) NA_real_ else if (estimate > 0) {
      estimate * df2 / ((1 - estimate) * df1)
    } else {
      0
    }
    eta <- if (is.na(f_w)) NULL else probe_attempt(effectsize::F_to_eta2(
      f_w, df1, df2, ci = ci, alternative = alternative
    ))
    e <- eta$value
    accurate <- if (alternative == "greater" && !is.na(f_w)) {
      probe_attempt(accurate_lower(f_w, df1, df2, ci))
    }
    list(
      f = f, f_hex = hex(f), df1 = as.integer(df1), df2 = df2, df2_hex = hex(df2),
      ci = ci, ci_hex = hex(ci), alternative = alternative,
      omega2 = number_and_hex(if (is.null(o)) NULL else o[[1L]]),
      omega2_low = number_and_hex(o$CI_low),
      omega2_high = number_and_hex(o$CI_high),
      omega2_error = omega$error,
      omega2_warnings = omega$warnings,
      f_w = number_and_hex(if (is.na(f_w)) NULL else f_w),
      eta2_at_f_w = number_and_hex(if (is.null(e)) NULL else e[[1L]]),
      eta2_low = number_and_hex(e$CI_low),
      eta2_high = number_and_hex(e$CI_high),
      eta2_error = eta$error,
      eta2_warnings = eta$warnings,
      omega2_low_cdf = if (is.null(o) || is.na(f_w)) NULL else at_bound(f_w, df1, df2, o$CI_low),
      omega2_high_cdf = if (is.null(o) || is.na(f_w)) NULL else at_bound(f_w, df1, df2, o$CI_high),
      accurate_low = if (is.null(accurate)) NULL else list(
        lambda = number_and_hex(accurate$value$lambda),
        bound = number_and_hex(accurate$value$bound),
        iterations = accurate$value$iterations,
        precision = accurate$value$precision,
        error = accurate$error,
        warnings = accurate$warnings
      )
    )
  })
  probe_file(
    "probe-omega",
    "effectsize::F_to_omega2 and F_to_eta2 at the omega-implied F; accurate one-sided lower bounds",
    list(rows = rows)
  )
}

# Phase 2 probes ----------------------------------------------------------------------
#
# What the nonparametric, robust and Bayes types need beyond the cases: lincon's
# critical values (probe-smm), how WRS2 computes its p-values (probe-wrs2p), the
# AKP scaling constant (probe-akp), the MAP centrality (probe-map), and Mann-Whitney,
# rank-biserial, Kruskal-Wallis and Dunn on tied data (probe-ranks). Doubles are hex
# text as above. Where a probe draws random numbers it sets its own seed first.

# One oracle CSV, read as read_dataset() reads it, and its `datasets` entry.
probe_dataset <- function(name) {
  path <- file.path(dataset_dir, paste0(name, ".csv"))
  data <- utils::read.csv(
    path,
    na.strings = "", stringsAsFactors = FALSE, check.names = FALSE,
    fileEncoding = "UTF-8"
  )
  list(data = data, entry = list(file = basename(path), sha256 = file_sha256(path)))
}

# The complete values of `y` in group `level` of `x`, in file order.
group_values <- function(data, x, y, level) {
  keep <- !is.na(data[[x]]) & !is.na(data[[y]]) & as.character(data[[x]]) == level
  as.numeric(data[[y]][keep])
}

hex_list <- function(x) as.list(hex(as.numeric(x)))

# Normal scores: a smooth, deterministic sample of size n.
normal_scores <- function(n) stats::qnorm(stats::ppoints(n))

two_group_frame <- function(a, b, labels = c("a", "b")) {
  data.frame(
    g = factor(rep(labels, c(length(a), length(b))), levels = labels),
    y = c(a, b)
  )
}

# Run `run()` while every function in WRS2's namespace is replaced by a recorder, then
# restore the originals. Records how often each function was called and, for calls
# that return a single number, the arguments that evaluate to single numbers and the
# value (at most `keep` calls per function). Only names, argument values and return
# values are recorded: no function's code is read or printed.
record_wrs2_calls <- function(run, keep = 20L) {
  ns <- asNamespace("WRS2")
  names <- Filter(function(n) {
    f <- get(n, envir = ns, inherits = FALSE)
    is.function(f) && !is.primitive(f)
  }, ls(ns, all.names = TRUE))
  originals <- mget(names, envir = ns, inherits = FALSE)
  state <- new.env()
  state$counts <- list()
  state$calls <- list()
  make <- function(name, f) {
    force(name)
    force(f)
    wrapper <- function() {
      call <- sys.call()
      caller <- parent.frame()
      call[[1L]] <- f
      value <- eval(call, caller)
      count <- if (is.null(state$counts[[name]])) 1L else state$counts[[name]] + 1L
      state$counts[[name]] <- count
      if (is.numeric(value) && length(value) == 1L && count <= keep) {
        matched <- tryCatch(
          as.list(match.call(f, call, envir = caller))[-1L],
          error = function(e) list()
        )
        args <- list()
        for (arg in names(matched)) {
          if (!nzchar(arg)) next
          v <- tryCatch(eval(matched[[arg]], caller), error = function(e) NULL)
          # Single numbers, and short numeric vectors (such as a df per comparison).
          if (is.numeric(v) && length(v) >= 1L && length(v) <= 64L) {
            args[[arg]] <- as.numeric(v)
          }
        }
        state$calls[[length(state$calls) + 1L]] <- list(
          name = name, args = args, value = as.numeric(value)
        )
      }
      value
    }
    formals(wrapper) <- formals(f)
    wrapper
  }
  for (n in names) {
    unlockBinding(n, ns)
    assign(n, make(n, originals[[n]]), envir = ns)
  }
  on.exit({
    for (n in names) {
      assign(n, originals[[n]], envir = ns)
      lockBinding(n, ns)
    }
  })
  value <- tryCatch(run(), error = function(e) e)
  list(value = value, counts = state$counts, calls = state$calls)
}

# probe-smm: the Studentized maximum modulus critical values behind WRS2::lincon's
# pairwise CIs, estimate -/+ c x Yuen SE, with c for C comparisons (pairs) at the
# pair's Yuen df.
#
# 1. lincon on constructed data with 2 to 10 groups (C = 1 to 45), at several alphas,
#    with each pair's Yuen SE (|diff / t|) and df from WRS2::yuen on that pair: c is
#    lincon's half-width divided by the SE.
# 2. Which WRS2 functions lincon calls there (record_wrs2_calls), and which call
#    returns c: a function with one argument equal to the pair's df and one equal to C.
# 3. Each function found in 2, on a grid of df (every table row, values between rows,
#    below 2 and far beyond 200) and C = 1 to 50, its other arguments as lincon passed
#    them.
SMM_ALPHAS <- list(
  list(text = "0.05", value = 0.05), list(text = "1 - 0.95", value = 1 - 0.95),
  list(text = "0.01", value = 0.01), list(text = "1 - 0.99", value = 1 - 0.99),
  list(text = "0.1", value = 0.1), list(text = "1 - 0.9", value = 1 - 0.9)
)
SMM_DF_GRID <- sort(unique(c(
  1, 1.5, 2, 2.5, 3:5, 5.2, 6:7, 7.7, 8:11, 11.9, 12, 12.9, 13:20, 22, 24, 27, 30, 35,
  40, 50, 59.9, 60, 61, 80, 100, 110, 150, 199.5, 200, 201, 250, 500, 1000, 1e4, 1e6
)))
SMM_C_GRID <- 1:50

# k groups of 8 to 17 values with different spreads: every pair has a non-integer
# Yuen df.
smm_data <- function(k) {
  labels <- sprintf("g%02d", seq_len(k))
  y <- unlist(lapply(seq_len(k), function(j) {
    normal_scores(7L + j) * (1 + 0.17 * j) + 0.4 * j
  }))
  data.frame(g = factor(rep(labels, 7L + seq_len(k)), levels = labels), y = y)
}

lincon_pairs <- function(data, alpha, tr = 0.2) {
  set.seed(1L)
  out <- probe_attempt(WRS2::lincon(y ~ g, data = data, tr = tr, alpha = alpha))
  if (is.null(out$value)) {
    return(list(error = out$error, warnings = out$warnings, pairs = list()))
  }
  comp <- out$value$comp
  labels <- levels(data$g)
  pairs <- lapply(seq_len(nrow(comp)), function(i) {
    g1 <- labels[[comp[i, 1L]]]
    g2 <- labels[[comp[i, 2L]]]
    pair_data <- droplevels(data[data$g %in% c(g1, g2), , drop = FALSE])
    yu <- WRS2::yuen(y ~ g, data = pair_data, tr = tr)
    se <- abs(yu$diff / yu$test)
    critical <- (comp[i, "ci.upper"] - comp[i, "psihat"]) / se
    list(
      group1 = g1, group2 = g2,
      psihat_hex = hex(comp[i, "psihat"]), ci_lower_hex = hex(comp[i, "ci.lower"]),
      ci_upper_hex = hex(comp[i, "ci.upper"]), p_value_hex = hex(comp[i, "p.value"]),
      yuen_test_hex = hex(yu$test), yuen_df_hex = hex(yu$df), yuen_df = yu$df,
      yuen_diff_hex = hex(yu$diff), yuen_p_value_hex = hex(yu$p.value),
      critical_value = critical, critical_value_hex = hex(critical)
    )
  })
  list(
    error = NULL, warnings = out$warnings, alpha_reported = out$value$alpha,
    columns = as.list(colnames(comp)), pairs = pairs
  )
}

# Calls among `calls` that returned a pair's critical value and took its df and C.
critical_calls <- function(calls, pairs, n_comparisons) {
  found <- list()
  for (call in calls) {
    for (pair in pairs) {
      if (!isTRUE(abs(call$value - pair$critical_value) <= 1e-9 * abs(pair$critical_value))) next
      df_arg <- names(Filter(function(v) isTRUE(abs(v - pair$yuen_df) <= 1e-12 * pair$yuen_df), call$args))
      c_arg <- names(Filter(function(v) isTRUE(v == n_comparisons), call$args))
      if (length(df_arg) == 1L && length(c_arg) == 1L) {
        other <- call$args[setdiff(names(call$args), c(df_arg, c_arg))]
        found[[length(found) + 1L]] <- list(
          name = call$name, df_arg = df_arg, c_arg = c_arg, other_args = other
        )
        break
      }
    }
  }
  unique(found)
}

smm_grid <- function(spec) {
  f <- get(spec$name, envir = asNamespace("WRS2"), inherits = FALSE)
  value_at <- function(df, n_comparisons, seed = 1L) {
    args <- spec$other_args
    args[[spec$df_arg]] <- df
    args[[spec$c_arg]] <- n_comparisons
    set.seed(seed)
    probe_attempt(do.call(f, args))
  }
  # A simulated critical value differs between seeds, or (with a seed of its own)
  # takes long: then only a small grid.
  started <- proc.time()[["elapsed"]]
  first <- value_at(10, 6L, 1L)$value
  seconds <- proc.time()[["elapsed"]] - started
  second <- value_at(10, 6L, 2L)$value
  random <- !identical(first, second)
  small <- random || seconds > 0.05
  df_grid <- if (small) c(5, 10, 30) else SMM_DF_GRID
  c_grid <- if (small) c(1L, 3L, 6L, 10L) else SMM_C_GRID
  rows <- lapply(df_grid, function(df) {
    values <- character()
    errors <- list()
    warnings <- character()
    for (n_comparisons in c_grid) {
      out <- value_at(df, n_comparisons)
      v <- out$value
      values <- c(values, if (is.numeric(v) && length(v) == 1L) hex(as.numeric(v)) else "NA")
      if (!is.null(out$error)) {
        errors[[length(errors) + 1L]] <- list(C = n_comparisons, error = out$error)
      }
      warnings <- c(warnings, unlist(out$warnings))
    }
    list(
      df = encode_doubles(df), df_hex = hex(df), values_hex = as.list(values),
      errors = errors, warnings = as.list(unique(warnings))
    )
  })
  list(
    name = spec$name, df_arg = spec$df_arg, c_arg = spec$c_arg,
    other_args = lapply(spec$other_args, function(v) list(value = v, hex = hex(v))),
    random = random, small_grid = small, c_grid = as.list(c_grid), rows = rows
  )
}

smm_probe <- function() {
  ns_names <- ls(asNamespace("WRS2"), all.names = TRUE)
  backout <- list()
  census <- list()
  specs <- list()
  for (k in 2:10) {
    data <- smm_data(k)
    n_comparisons <- k * (k - 1L) / 2L
    for (alpha in SMM_ALPHAS) {
      result <- lincon_pairs(data, alpha$value)
      entry <- list(
        groups = k, comparisons = n_comparisons, alpha = alpha$text,
        alpha_hex = hex(alpha$value), result = result
      )
      # Which functions lincon calls, with 2, 4 and 10 groups.
      if (k %in% c(2L, 4L, 10L)) {
        recorded <- record_wrs2_calls(function() {
          set.seed(1L)
          WRS2::lincon(y ~ g, data = data, tr = 0.2, alpha = alpha$value)
        })
        found <- critical_calls(recorded$calls, result$pairs, n_comparisons)
        specs <- c(specs, found)
        census[[length(census) + 1L]] <- list(
          groups = k, comparisons = n_comparisons, alpha = alpha$text,
          error = if (inherits(recorded$value, "error")) conditionMessage(recorded$value),
          call_counts = recorded$counts,
          scalar_calls = lapply(recorded$calls, function(call) {
            list(
              name = call$name,
              args = lapply(call$args, function(v) list(value = v, hex = hex(v))),
              value_hex = hex(call$value)
            )
          }),
          critical_value_calls = lapply(found, function(spec) {
            list(name = spec$name, df_arg = spec$df_arg, c_arg = spec$c_arg)
          })
        )
      }
      backout[[length(backout) + 1L]] <- entry
    }
  }
  specs <- unique(specs)
  # Whether lincon's critical values off the table (an alpha other than 0.05 or 0.01,
  # or more comparisons than the table has) depend on the random seed.
  seeds <- list()
  for (setting in list(
    list(groups = 4L, alpha = SMM_ALPHAS[[2L]]), list(groups = 4L, alpha = SMM_ALPHAS[[5L]]),
    list(groups = 10L, alpha = SMM_ALPHAS[[1L]])
  )) {
    data <- smm_data(setting$groups)
    for (seed in 1:3) {
      set.seed(seed)
      out <- probe_attempt(WRS2::lincon(y ~ g, data = data, tr = 0.2, alpha = setting$alpha$value))
      comp <- if (is.null(out$value)) NULL else out$value$comp
      seeds[[length(seeds) + 1L]] <- list(
        groups = setting$groups, alpha = setting$alpha$text, seed = seed,
        error = out$error,
        ci_upper_hex = if (is.null(comp)) NULL else hex_list(comp[, "ci.upper"]),
        psihat_hex = if (is.null(comp)) NULL else hex_list(comp[, "psihat"])
      )
    }
  }
  probe_file(
    "probe-smm",
    "WRS2::lincon critical values backed out from its CIs, the WRS2 functions it calls, and those functions on a df x C grid",
    list(
      namespace_functions = as.list(grep("smm|crit", ns_names, value = TRUE)),
      df_grid_hex = as.list(hex(SMM_DF_GRID)),
      lincon = backout,
      calls = census,
      grids = lapply(specs, smm_grid),
      seed_dependence = seeds
    )
  )
}

# probe-wrs2p: WRS2's p-values next to the exact upper tail and 1 - cdf. yuen on two
# groups, and t1way and lincon on three, over shifts that take |t| from 0 to about
# 40 (p from 1 to 0) at small, moderate and large df. `shift` is in units of an
# approximate SE of the difference. lincon's pairs come with yuen's t and df on the
# same two groups.
WRS2P_SHIFTS <- c(
  0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11,
  12, 14, 16, 20, 25, 30, 40
)

wrs2p_probe <- function() {
  t_tails <- function(t, df) {
    list(
      one_minus_cdf_hex = hex(1 - stats::pt(abs(t), df)),
      upper_tail_hex = hex(stats::pt(abs(t), df, lower.tail = FALSE))
    )
  }
  two_designs <- list(
    list(n = c(6L, 7L), scale = c(1, 1.5)),
    list(n = c(20L, 20L), scale = c(1, 1.3)),
    list(n = c(400L, 350L), scale = c(1, 1.2))
  )
  yuen_rows <- list()
  for (design in two_designs) {
    unit <- 1.4 * sqrt(sum(design$scale^2 / design$n))
    for (shift in WRS2P_SHIFTS) {
      d <- two_group_frame(
        normal_scores(design$n[[1L]]) * design$scale[[1L]],
        normal_scores(design$n[[2L]]) * design$scale[[2L]] + shift * unit
      )
      r <- probe_call("yuen", WRS2::yuen(y ~ g, data = d))$value
      yuen_rows[[length(yuen_rows) + 1L]] <- c(
        list(
          n = as.list(design$n), shift = shift,
          test_hex = hex(r$test), df_hex = hex(r$df), p_value_hex = hex(r$p.value),
          p_value = r$p.value, diff_hex = hex(r$diff)
        ),
        t_tails(r$test, r$df)
      )
    }
  }
  three_designs <- list(
    list(n = c(5L, 6L, 7L), scale = c(1, 1.3, 0.8)),
    list(n = c(15L, 15L, 15L), scale = c(1, 1.2, 0.8)),
    list(n = c(300L, 250L, 280L), scale = c(1, 1.2, 0.8))
  )
  t1way_rows <- list()
  lincon_rows <- list()
  for (design in three_designs) {
    unit <- 1.4 * sqrt(sum(design$scale[1:2]^2 / design$n[1:2]))
    for (shift in WRS2P_SHIFTS) {
      labels <- c("a", "b", "c")
      d <- data.frame(
        g = factor(rep(labels, design$n), levels = labels),
        y = c(
          normal_scores(design$n[[1L]]) * design$scale[[1L]],
          normal_scores(design$n[[2L]]) * design$scale[[2L]] + shift * unit,
          normal_scores(design$n[[3L]]) * design$scale[[3L]] + 2 * shift * unit
        )
      )
      set.seed(1L)
      r <- probe_call("t1way", WRS2::t1way(y ~ g, data = d, nboot = 20L))$value
      t1way_rows[[length(t1way_rows) + 1L]] <- list(
        n = as.list(design$n), shift = shift,
        test_hex = hex(r$test), df1_hex = hex(r$df1), df2_hex = hex(r$df2),
        p_value_hex = hex(r$p.value), p_value = r$p.value,
        one_minus_cdf_hex = hex(1 - stats::pf(r$test, r$df1, r$df2)),
        upper_tail_hex = hex(stats::pf(r$test, r$df1, r$df2, lower.tail = FALSE))
      )
      # Unadjusted (method = "none"), and with lincon's default adjustment.
      comp <- probe_call("lincon", WRS2::lincon(y ~ g, data = d, method = "none"))$value$comp
      default <- probe_call("lincon", WRS2::lincon(y ~ g, data = d))$value$comp
      pairs <- lapply(seq_len(nrow(comp)), function(i) {
        g1 <- labels[[comp[i, 1L]]]
        g2 <- labels[[comp[i, 2L]]]
        yu <- WRS2::yuen(y ~ g, data = droplevels(d[d$g %in% c(g1, g2), , drop = FALSE]))
        c(
          list(
            group1 = g1, group2 = g2, p_value_hex = hex(comp[i, "p.value"]),
            p_value = comp[i, "p.value"],
            p_value_default_method_hex = hex(default[i, "p.value"]),
            psihat_hex = hex(comp[i, "psihat"]),
            yuen_test_hex = hex(yu$test), yuen_df_hex = hex(yu$df),
            yuen_p_value_hex = hex(yu$p.value)
          ),
          t_tails(yu$test, yu$df)
        )
      })
      lincon_rows[[length(lincon_rows) + 1L]] <- list(
        n = as.list(design$n), shift = shift, pairs = pairs
      )
    }
  }
  probe_file(
    "probe-wrs2p",
    "WRS2 yuen, t1way and lincon p-values next to 1 - pt / 1 - pf and the exact upper tail",
    list(yuen = yuen_rows, t1way = t1way_rows, lincon = lincon_rows)
  )
}

# probe-akp: WRS2::akp.effect on varied two-group data at several trimming levels,
# with both EQVAR settings, next to the trimmed means and winsorized SDs computed
# here, so the scaling constant WRS2 applies can be measured against the exact
# winsorized-normal value. The estimate does not depend on the resamples (nboot 10,
# seed 1).
AKP_TR <- c(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)

winsorized_sd <- function(x, tr) {
  n <- length(x)
  g <- floor(tr * n)
  s <- sort(x)
  stats::sd(pmin(pmax(x, s[[g + 1L]]), s[[n - g]]))
}

akp_probe <- function() {
  tooth <- probe_dataset("ToothGrowth")
  cars <- probe_dataset("mtcars")
  k2 <- probe_dataset("edge_k2")
  designs <- list(
    normal = list(normal_scores(20L), normal_scores(25L) * 1.4 + 0.8),
    skewed = list(stats::qexp(stats::ppoints(30L)), stats::qexp(stats::ppoints(24L)) * 1.5 + 0.3),
    ties = list(c(1, 2, 2, 3, 3, 3, 4, 4, 5, 6), c(2, 3, 3, 4, 4, 4, 5, 6, 6, 7, 8)),
    small = list(c(3.1, 4.5, 2.2, 5.9, 4.4), c(6.1, 5.2, 7.7, 6.6, 5.0, 8.1)),
    reversed = list(normal_scores(25L) * 1.4 + 0.8, normal_scores(20L)),
    ToothGrowth = list(
      group_values(tooth$data, "supp", "len", "OJ"), group_values(tooth$data, "supp", "len", "VC")
    ),
    mtcars_am = list(
      group_values(cars$data, "am", "mpg", "0"), group_values(cars$data, "am", "mpg", "1")
    ),
    edge_k2 = list(
      group_values(k2$data, "group", "value", "a"), group_values(k2$data, "group", "value", "b")
    )
  )
  rows <- list()
  for (name in names(designs)) {
    a <- designs[[name]][[1L]]
    b <- designs[[name]][[2L]]
    d <- two_group_frame(a, b)
    for (tr in AKP_TR) {
      wa <- winsorized_sd(a, tr)
      wb <- winsorized_sd(b, tr)
      difference <- mean(a, trim = tr) - mean(b, trim = tr)
      effects <- lapply(c(FALSE, TRUE), function(eqvar) {
        set.seed(1L)
        out <- probe_attempt(WRS2::akp.effect(
          y ~ g, data = d, EQVAR = eqvar, tr = tr, nboot = 10L
        ))
        effect <- if (is.null(out$value)) NULL else as.numeric(out$value$AKPeffect)
        list(
          EQVAR = eqvar,
          effect_hex = if (is.null(effect)) NULL else hex_list(effect),
          # The first effect over the trimmed-mean difference in units of group 1's
          # winsorized SD: the constant WRS2 multiplies by, if it scales that way.
          ratio_to_group1_hex = if (is.null(effect)) NULL else hex(effect[[1L]] / (difference / wa)),
          error = out$error, warnings = out$warnings
        )
      })
      rows[[length(rows) + 1L]] <- list(
        data = name, n = list(length(a), length(b)), tr = tr, tr_hex = hex(tr),
        trimmed_mean_difference_hex = hex(difference),
        winsorized_sd_hex = list(hex(wa), hex(wb)),
        akp = effects
      )
    }
  }
  # The constant alone on a fine grid of tr (normal design, EQVAR = FALSE).
  pair <- designs$normal
  d <- two_group_frame(pair[[1L]], pair[[2L]])
  constants <- lapply(seq(0.01, 0.49, by = 0.01), function(tr) {
    set.seed(1L)
    out <- probe_attempt(WRS2::akp.effect(y ~ g, data = d, EQVAR = FALSE, tr = tr, nboot = 10L))
    difference <- mean(pair[[1L]], trim = tr) - mean(pair[[2L]], trim = tr)
    list(
      tr = tr, tr_hex = hex(tr), error = out$error,
      ratio_to_group1_hex = if (is.null(out$value)) NULL else {
        hex(as.numeric(out$value$AKPeffect)[[1L]] / (difference / winsorized_sd(pair[[1L]], tr)))
      }
    )
  })
  probe_file(
    "probe-akp",
    "WRS2::akp.effect at several tr, both EQVAR settings, with trimmed-mean differences and winsorized SDs",
    list(
      constants = constants,
      tr_grid = as.list(AKP_TR),
      data = lapply(designs, function(pair) list(a_hex = hex_list(pair[[1L]]), b_hex = hex_list(pair[[2L]]))),
      rows = rows
    ),
    datasets = list(tooth$entry, cars$entry, k2$entry)
  )
}

# probe-map: the MAP centrality of type "bayes". For each vector: what
# statsExpressions::centrality_description(type = "bayes") reports for it as a single
# group; bayestestR::map_estimate() with its defaults; the Sheather-Jones and nrd0
# bandwidths; and stats::density with bw = "SJ" and n = 1024 on two grids (from
# min(x) to max(x), and density's default range), each with its grid ends, bandwidth,
# the index and location of its maximum and the densities around it, so a Python
# implementation can be pinned down. Six vectors also keep their full density values.
map_vectors <- function() {
  tooth <- probe_dataset("ToothGrowth")
  plant <- probe_dataset("PlantGrowth")
  cars <- probe_dataset("mtcars")
  iris_data <- probe_dataset("iris")
  ties <- probe_dataset("edge_ties")
  insects <- probe_dataset("InsectSprays")
  n2 <- probe_dataset("edge_n2")
  outliers <- probe_dataset("edge_outliers")
  large <- probe_dataset("edge_large_n")
  from_data <- function(set, x, y, levels, prefix) {
    out <- lapply(levels, function(level) group_values(set$data, x, y, level))
    names(out) <- paste(prefix, levels, sep = "_")
    out
  }
  vectors <- c(
    from_data(tooth, "supp", "len", c("OJ", "VC"), "ToothGrowth"),
    from_data(plant, "group", "weight", c("ctrl", "trt1", "trt2"), "PlantGrowth"),
    from_data(cars, "cyl", "mpg", c("4", "6", "8"), "mtcars_cyl"),
    from_data(iris_data, "Species", "Sepal.Width", c("setosa", "versicolor", "virginica"), "iris_sepal_width"),
    from_data(ties, "group", "score", c("a", "b", "c"), "edge_ties"),
    from_data(insects, "spray", "count", c("A", "B", "C", "D", "E", "F"), "InsectSprays"),
    from_data(n2, "group", "value", "a", "edge_n2"),
    from_data(outliers, "group", "value", "c", "edge_outliers"),
    from_data(large, "group", "value", "a", "edge_large_n"),
    list(
      normal_n3 = normal_scores(3L),
      normal_n5 = normal_scores(5L),
      normal_n10 = normal_scores(10L),
      normal_n50 = normal_scores(50L),
      normal_n1000 = normal_scores(1000L),
      normal_n5000 = normal_scores(5000L),
      exponential_40 = stats::qexp(stats::ppoints(40L)),
      lognormal_100 = stats::qlnorm(stats::ppoints(100L)),
      negative_skew_30 = -stats::qexp(stats::ppoints(30L)),
      uniform_100 = stats::ppoints(100L),
      t2_60 = stats::qt(stats::ppoints(60L), 2),
      bimodal_equal = c(normal_scores(30L) - 2.5, normal_scores(30L) + 2.5),
      bimodal_unequal = c(normal_scores(20L) - 2.5, normal_scores(40L) + 2.5),
      integers_with_ties = c(1, 1, 2, 2, 2, 3, 3, 4, 5),
      two_values = c(1, 2),
      two_tied_values = c(2, 2, 2, 5, 5),
      constant = rep(3, 5),
      zero_iqr = c(1, 1, 1, 1, 1, 1, 1, 5),
      scaled_small = 1e-8 * normal_scores(25L),
      shifted_large = 1e8 + normal_scores(25L),
      # |x| / (range / 1000) far above 2^31: timestamps with a spread of about 1.
      timestamps = 1.7e9 + 0.5 * normal_scores(40L),
      all_negative = -20 + stats::qexp(stats::ppoints(35L)) * 2,
      spanning_zero = stats::qlnorm(stats::ppoints(45L)) - 1.5,
      symmetric_grid = seq(-1, 1, length.out = 21L),
      normal_n501 = normal_scores(501L),
      normal_n1e5 = normal_scores(100000L),
      # Three tight clusters: bw.SJ's solution lies below a tenth of its
      # normal-reference starting bandwidth, so its search widens downward.
      tight_clusters = c(
        0.01 * normal_scores(30L), 1 + 0.01 * normal_scores(30L),
        5 + 0.01 * normal_scores(30L)
      ),
      tighter_clusters = c(
        0.003 * normal_scores(30L), 1 + 0.003 * normal_scores(30L),
        5 + 0.003 * normal_scores(30L)
      ),
      outlier_tail = c(0.1 * normal_scores(95L), 10, 12, 15, 20, 30)
    ),
    # Phase 3: where bw.SJ starts anchoring its pair-distance bins at min(x). One
    # base vector (range 12, bin width 0.01212) shifted by c: about 1e5 and 1e8 bin
    # widths, and just below and just above 2^31 bin widths (26027440 and
    # 26027507), where the two anchors' bandwidths differ by about 1e-3.
    stats::setNames(
      lapply(c(1200, 1211988, 26027440, 26027507), function(shift) {
        c(0, 1, 2, 2, 3, 3, 3, 4, 5, 7, 9, 12) + shift
      }),
      paste0("anchor_", c("1200", "1211988", "26027440", "26027507"))
    )
  )
  list(
    vectors = vectors,
    datasets = list(
      tooth$entry, plant$entry, cars$entry, iris_data$entry, ties$entry, insects$entry,
      n2$entry, outliers$entry, large$entry
    )
  )
}

MAP_FULL_DENSITY <- c(
  "ToothGrowth_OJ", "edge_ties_a", "InsectSprays_C", "normal_n10", "bimodal_unequal",
  "integers_with_ties", "timestamps", "tight_clusters"
)

density_summary <- function(x, full, ...) {
  out <- probe_attempt(stats::density(x, bw = "SJ", n = 1024L, ...))
  d <- out$value
  if (is.null(d)) {
    return(list(error = out$error, warnings = out$warnings))
  }
  top <- which.max(d$y)
  near <- max(1L, top - 2L):min(length(d$y), top + 2L)
  list(
    error = NULL, warnings = out$warnings,
    n = length(d$x), from_hex = hex(d$x[[1L]]), to_hex = hex(d$x[[length(d$x)]]),
    bw_hex = hex(d$bw), argmax = top, x_at_max_hex = hex(d$x[[top]]),
    x_at_max = d$x[[top]],
    y_near_max_index = as.list(near), y_near_max_hex = hex_list(d$y[near]),
    # Other grid points within 1e-9 (relative) of the maximum: near-ties.
    near_ties = as.list(which(d$y >= d$y[[top]] * (1 - 1e-9))),
    y_hex = if (full) hex_list(d$y) else NULL
  )
}

map_probe <- function() {
  collected <- map_vectors()
  rows <- lapply(names(collected$vectors), function(name) {
    x <- collected$vectors[[name]]
    full <- name %in% MAP_FULL_DENSITY
    map <- probe_attempt(as.numeric(bayestestR::map_estimate(x)))
    single <- data.frame(g = factor(rep("a", length(x))), y = x)
    centrality <- probe_attempt(statsExpressions::centrality_description(
      single, g, y, type = "bayes"
    ))
    estimate <- probe_attempt(bayestestR::estimate_density(x))
    density_of_estimate <- estimate$value
    list(
      name = name, n = length(x), x_hex = hex_list(x),
      map_estimate_hex = if (is.null(map$value)) NULL else hex(map$value),
      map_estimate_error = map$error, map_estimate_warnings = map$warnings,
      centrality_hex = if (is.null(centrality$value)) NULL else hex(as.numeric(centrality$value$y)),
      centrality_label = if (is.null(centrality$value)) NULL else to_json_value(centrality$value$expression),
      centrality_error = centrality$error, centrality_warnings = centrality$warnings,
      bw_sj = {
        out <- probe_attempt(stats::bw.SJ(x))
        list(hex = if (is.null(out$value)) NULL else hex(out$value), error = out$error, warnings = out$warnings)
      },
      bw_nrd0_hex = {
        out <- probe_attempt(stats::bw.nrd0(x))
        if (is.null(out$value)) NULL else hex(out$value)
      },
      density_data_range = density_summary(x, full, from = min(x), to = max(x)),
      density_default_range = density_summary(x, full),
      estimate_density = if (is.null(density_of_estimate)) {
        list(error = estimate$error, warnings = estimate$warnings)
      } else {
        list(
          error = NULL, warnings = estimate$warnings, n = nrow(density_of_estimate),
          from_hex = hex(density_of_estimate$x[[1L]]),
          to_hex = hex(density_of_estimate$x[[nrow(density_of_estimate)]]),
          x_at_max_hex = hex(density_of_estimate$x[[which.max(density_of_estimate$y)]])
        )
      }
    )
  })
  probe_file(
    "probe-map",
    "MAP centrality for type bayes: centrality_description, bayestestR::map_estimate, bw.SJ and stats::density summaries",
    list(full_density = as.list(MAP_FULL_DENSITY), rows = rows),
    datasets = collected$datasets
  )
}

# probe-ranks: Mann-Whitney (wilcox.test, exact = FALSE, with and without the
# continuity correction, every alternative) and effectsize::rank_biserial on tied and
# untied pairs; kruskal.test and PMCMRplus::kwAllPairsDunnTest (unadjusted) on tied
# data with three to six groups.
ranks_probe <- function() {
  ties <- probe_dataset("edge_ties")
  insects <- probe_dataset("InsectSprays")
  cars <- probe_dataset("mtcars")
  k2 <- probe_dataset("edge_k2")
  tv <- function(level) group_values(ties$data, "group", "score", level)
  iv <- function(level) group_values(insects$data, "spray", "count", level)
  pairs <- list(
    edge_ties_a_b = list(tv("a"), tv("b")),
    edge_ties_a_c = list(tv("a"), tv("c")),
    edge_ties_b_c = list(tv("b"), tv("c")),
    InsectSprays_C_D = list(iv("C"), iv("D")),
    InsectSprays_A_B = list(iv("A"), iv("B")),
    mtcars_am = list(
      group_values(cars$data, "am", "mpg", "0"), group_values(cars$data, "am", "mpg", "1")
    ),
    edge_k2 = list(
      group_values(k2$data, "group", "value", "a"), group_values(k2$data, "group", "value", "b")
    ),
    small_ties = list(c(1, 2, 2, 3), c(2, 3, 3, 4, 5)),
    mostly_tied = list(c(1, 1, 1, 2), c(1, 1, 2, 2, 2))
  )
  two_sample <- lapply(names(pairs), function(name) {
    a <- pairs[[name]][[1L]]
    b <- pairs[[name]][[2L]]
    tests <- list()
    for (alternative in c("two.sided", "less", "greater")) {
      for (correct in c(TRUE, FALSE)) {
        out <- probe_attempt(stats::wilcox.test(
          a, b, exact = FALSE, correct = correct, alternative = alternative
        ))
        tests[[length(tests) + 1L]] <- list(
          alternative = alternative, correct = correct,
          statistic_hex = if (is.null(out$value)) NULL else hex(unname(out$value$statistic)),
          p_value_hex = if (is.null(out$value)) NULL else hex(out$value$p.value),
          method = if (is.null(out$value)) NULL else out$value$method,
          error = out$error, warnings = out$warnings
        )
      }
    }
    biserial <- lapply(c("two.sided", "less", "greater"), function(alternative) {
      out <- probe_attempt(effectsize::rank_biserial(
        a, b, ci = 0.95, alternative = alternative, verbose = FALSE
      ))
      r <- out$value
      list(
        alternative = alternative,
        estimate_hex = if (is.null(r)) NULL else hex(r[[1L]]),
        conf_low_hex = if (is.null(r)) NULL else hex(r$CI_low),
        conf_high_hex = if (is.null(r)) NULL else hex(r$CI_high),
        error = out$error, warnings = out$warnings
      )
    })
    list(
      name = name, a_hex = hex_list(a), b_hex = hex_list(b), wilcox = tests,
      rank_biserial = biserial
    )
  })
  designs <- list(
    edge_ties = list(set = ties, x = "group", y = "score"),
    InsectSprays = list(set = insects, x = "spray", y = "count"),
    mtcars_cyl = list(set = cars, x = "cyl", y = "mpg")
  )
  k_sample <- lapply(names(designs), function(name) {
    design <- designs[[name]]
    analysis <- between_analysis(design$set$data, design$x, design$y)
    kw <- probe_call("kruskal.test", stats::kruskal.test(y ~ x, data = analysis))
    dunn <- probe_call(
      "kwAllPairsDunnTest",
      PMCMRplus::kwAllPairsDunnTest(x = analysis$y, g = analysis$x, p.adjust.method = "none")
    )
    levels <- levels(analysis$x)
    k <- length(levels)
    dunn_pairs <- list()
    for (i in seq_len(k - 1L)) {
      for (j in (i + 1L):k) {
        dunn_pairs[[length(dunn_pairs) + 1L]] <- list(
          group1 = levels[[i]], group2 = levels[[j]],
          statistic_hex = hex(dunn$value$statistic[levels[[j]], levels[[i]]]),
          p_value_hex = hex(dunn$value$p.value[levels[[j]], levels[[i]]])
        )
      }
    }
    list(
      name = name, levels = as.list(levels), n_per_level = as.list(table(analysis$x)),
      groups_hex = lapply(levels, function(level) hex_list(analysis$y[analysis$x == level])),
      kruskal_statistic_hex = hex(unname(kw$value$statistic)),
      kruskal_parameter = unname(kw$value$parameter),
      kruskal_p_value_hex = hex(kw$value$p.value),
      kruskal_warnings = as.list(unique(kw$warnings)),
      dunn_method = to_json_value(dunn$value$method),
      dunn_warnings = as.list(unique(dunn$warnings)),
      dunn_pairs = dunn_pairs
    )
  })
  probe_file(
    "probe-ranks",
    "wilcox.test (exact = FALSE), effectsize::rank_biserial, kruskal.test and kwAllPairsDunnTest on tied data",
    list(two_sample = two_sample, k_sample = k_sample),
    datasets = list(ties$entry, insects$entry, cars$entry, k2$entry)
  )
}

# Phase 3 probes ----------------------------------------------------------------------
#
# What withinstats needs beyond the cases: the paired Wilcoxon test and rank-biserial
# r on zeros and ties (probe-signrank), paired JZS Bayes factors at large |t|
# (probe-pairedbf), Friedman, Durbin-Conover and Kendall's W on ties
# (probe-friedman), afex's repeated-measures ANOVA with each sphericity correction
# and the omega squared behind it (probe-rmanova), and WRS2's yuend, rmanova, rmmcp,
# dep.effect and wmcpAKP (probe-wrs2within). Doubles are hex text as above.

# The within-subject datasets the probes read, as ggwithinstats analyzes them.
WITHIN_PROBE_DESIGNS <- list(
  list(name = "bugs_long", dataset = "bugs_long", x = "condition", y = "desire", subject = "subject"),
  list(name = "edge_within_missing", dataset = "edge_within_missing"),
  list(name = "edge_within_small", dataset = "edge_within_small"),
  list(name = "edge_within_ties", dataset = "edge_within_ties"),
  list(name = "edge_within_negative", dataset = "edge_within_negative"),
  list(name = "edge_within_outliers", dataset = "edge_within_outliers"),
  list(name = "edge_within_null", dataset = "edge_within_null"),
  list(name = "edge_within_k5", dataset = "edge_within_k5"),
  list(name = "edge_within_nonspherical", dataset = "edge_within_nonspherical")
)

# One design: the CSV's `datasets` entry; the complete subjects (within_analysis());
# the long rows with .rowid a factor, as oneway_anova() uses them; the wide table; and
# each level's values in subject order as hex.
within_probe_design <- function(design) {
  x <- design$x %||% "condition"
  y <- design$y %||% "value"
  subject <- design$subject %||% "subject"
  set <- probe_dataset(design$dataset)
  analysis <- within_analysis(set$data, x, y, subject)
  rows <- statsexpressions_paired_rows(analysis, x, y)
  long <- rows$long
  long$.rowid <- as.factor(long$.rowid)
  levels <- levels(analysis$x)
  list(
    name = design$name, entry = set$entry, analysis = analysis, long = long,
    paired_long = rows$long, wide = rows$wide, levels = levels, x = x, y = y,
    values_hex = stats::setNames(
      lapply(levels, function(level) hex_list(analysis$y[analysis$x == level])), levels
    ),
    subjects = as.list(as.character(unique(analysis$.rowid)))
  )
}

# A data frame's numeric columns as hex text next to the 15-digit values.
frame_with_hex <- function(df) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  names(df) <- make.unique(names(df))
  for (column in names(df)) {
    if (is.double(df[[column]])) df[[paste0(column, "_hex")]] <- hex(df[[column]])
  }
  frame_rows(df)
}

# probe-signrank: stats::wilcox.test(paired = TRUE) with exact NULL, TRUE and FALSE,
# with and without the continuity correction, every alternative; and
# effectsize::rank_biserial(paired = TRUE) at 95% and 90%, every alternative; on
# pairs with zero differences, tied absolute differences, all differences zero, n of
# 3 to 60, and untied pairs on either side of wilcox.test's exact threshold (n = 50).
signrank_probe <- function() {
  sleep <- probe_dataset("sleep")
  ties <- probe_dataset("edge_within2_ties")
  small <- probe_dataset("edge_within2_small")
  by_subject <- function(set, x, y, subject, level) {
    d <- set$data[!is.na(set$data[[y]]) & as.character(set$data[[x]]) == level, ]
    d[[y]][order(d[[subject]])]
  }
  shifted <- function(n, shift) {
    base <- seq_len(n) / n * 3
    list(base + normal_scores(n) + shift, base)
  }
  pairs <- list(
    sleep = list(
      by_subject(sleep, "group", "extra", "ID", "1"), by_subject(sleep, "group", "extra", "ID", "2")
    ),
    edge_within2_ties = list(
      by_subject(ties, "condition", "value", "subject", "post"),
      by_subject(ties, "condition", "value", "subject", "pre")
    ),
    edge_within2_small = list(
      by_subject(small, "condition", "value", "subject", "post"),
      by_subject(small, "condition", "value", "subject", "pre")
    ),
    untied_n8 = list(
      c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06),
      c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14)
    ),
    zeros_and_ties = list(
      c(3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 4, 6), c(3, 4, 4, 3, 5, 7, 6, 6, 10, 7, 5, 3)
    ),
    all_zero = list(1:6 + 0, 1:6 + 0),
    one_nonzero = list(c(1, 2, 3, 4), c(1, 2, 3, 5)),
    n3 = list(c(1.1, 2.5, 3.2), c(0.4, 1.0, 3.5)),
    ties_n60 = list(rep(1:6, 10L) + 0, rep(1:6, 10L) + rep(c(-2, -1, 0, 1, 1, 2), 10L)),
    untied_n49 = shifted(49L, 0.3),
    untied_n50 = shifted(50L, 0.3),
    negative_shift_n20 = shifted(20L, -0.5)
  )
  rows <- lapply(names(pairs), function(name) {
    a <- pairs[[name]][[1L]]
    b <- pairs[[name]][[2L]]
    tests <- list()
    for (exact in list(NULL, TRUE, FALSE)) {
      for (correct in c(TRUE, FALSE)) {
        for (alternative in c("two.sided", "less", "greater")) {
          out <- probe_attempt(stats::wilcox.test(
            a, b, paired = TRUE, exact = exact, correct = correct, alternative = alternative
          ))
          tests[[length(tests) + 1L]] <- list(
            exact = if (is.null(exact)) "NULL" else exact, correct = correct,
            alternative = alternative,
            statistic_hex = if (is.null(out$value)) NULL else hex(unname(out$value$statistic)),
            p_value_hex = if (is.null(out$value)) NULL else hex(out$value$p.value),
            method = if (is.null(out$value)) NULL else out$value$method,
            error = out$error, warnings = out$warnings
          )
        }
      }
    }
    biserial <- list()
    for (ci in c(0.95, 0.9)) {
      for (alternative in c("two.sided", "less", "greater")) {
        out <- probe_attempt(effectsize::rank_biserial(
          a, b, paired = TRUE, ci = ci, alternative = alternative, verbose = FALSE
        ))
        r <- out$value
        biserial[[length(biserial) + 1L]] <- list(
          ci = ci, alternative = alternative,
          estimate_hex = if (is.null(r)) NULL else hex(r[[1L]]),
          conf_low_hex = if (is.null(r)) NULL else hex(r$CI_low),
          conf_high_hex = if (is.null(r)) NULL else hex(r$CI_high),
          error = out$error, warnings = out$warnings
        )
      }
    }
    list(name = name, a_hex = hex_list(a), b_hex = hex_list(b), wilcox = tests, rank_biserial = biserial)
  })
  probe_file(
    "probe-signrank",
    "wilcox.test(paired = TRUE) by exact, correct and alternative; effectsize::rank_biserial(paired = TRUE)",
    list(pairs = rows),
    datasets = list(sleep$entry, ties$entry, small$entry)
  )
}

# probe-pairedbf: BayesFactor::ttestBF(x, y, paired = TRUE) at |t| from 0 to 1,000
# (n 5, 10, 30 and 200; rscale 0.707 and 0.8), with BayesFactor's own error estimate
# and the paired t from t.test: where its quadrature could miss (D50 found misses for
# two samples at |t| near 40 to 50).
PAIREDBF_T <- c(0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 1000)

pairedbf_probe <- function() {
  rows <- list()
  for (n in c(5L, 10L, 30L, 200L)) {
    noise <- normal_scores(n)
    base <- seq_len(n) / n * 3
    for (target in PAIREDBF_T) {
      b <- base
      a <- base + noise + target * stats::sd(noise) / sqrt(n)
      tt <- probe_call("paired t.test", stats::t.test(a, b, paired = TRUE))$value
      for (rscale in c(0.707, 0.8)) {
        out <- probe_attempt(BayesFactor::ttestBF(x = a, y = b, paired = TRUE, rscale = rscale))
        bf <- if (is.null(out$value)) NULL else BayesFactor::extractBF(out$value)
        log_bf <- if (is.null(out$value)) NULL else BayesFactor::extractBF(out$value, logbf = TRUE)
        rows[[length(rows) + 1L]] <- list(
          n = n, t_target = target, rscale = rscale, rscale_hex = hex(rscale),
          a_hex = hex_list(a), b_hex = hex_list(b),
          t_hex = hex(unname(tt$statistic)), df = unname(tt$parameter),
          bf10_hex = if (is.null(bf)) NULL else hex(bf$bf[[1L]]),
          log_bf10_hex = if (is.null(log_bf)) NULL else hex(log_bf$bf[[1L]]),
          error = if (is.null(bf)) NULL else encode_doubles(bf$error[[1L]]),
          call_error = out$error, warnings = out$warnings
        )
      }
    }
  }
  probe_file(
    "probe-pairedbf",
    "BayesFactor::ttestBF(paired = TRUE) and its error estimate at |t| from 0 to 1000",
    list(rows = rows)
  )
}

# probe-friedman: stats::friedman.test, PMCMRplus::durbinAllPairsTest (unadjusted) and
# effectsize::kendalls_w (no CI) on the within-subject datasets as ggwithinstats
# analyzes them, and on two constructed tables (every subject tied across conditions
# in some rows; no ties at all).
friedman_probe <- function() {
  designs <- lapply(WITHIN_PROBE_DESIGNS, within_probe_design)
  constructed <- list(
    tied_rows = matrix(c(
      1, 2, 3, 2, 2, 2, 3, 1, 2, 4, 4, 1, 1, 3, 2, 5, 5, 5, 2, 3, 4, 1, 1, 2
    ), ncol = 3L, byrow = TRUE),
    untied = matrix(c(
      1.1, 2.3, 3.7, 0.4, 2.2, 1.9, 3.3, 1.2, 2.8, 4.4, 4.9, 1.5, 0.7, 3.1, 2.6, 5.2, 5.8,
      5.5, 2.1, 3.6, 4.0, 1.3, 0.9, 2.4
    ), ncol = 3L, byrow = TRUE)
  )
  for (name in names(constructed)) {
    m <- constructed[[name]]
    long <- data.frame(
      .rowid = factor(rep(seq_len(nrow(m)), each = ncol(m))),
      x = factor(rep(c("a", "b", "c"), times = nrow(m))),
      y = as.vector(t(m))
    )
    designs[[length(designs) + 1L]] <- list(
      name = name, entry = NULL, long = long, x = "x", y = "y",
      paired_long = data.frame(.rowid = as.integer(long$.rowid), x = long$x, y = long$y),
      levels = c("a", "b", "c"),
      values_hex = stats::setNames(lapply(seq_len(ncol(m)), function(j) hex_list(m[, j])), c("a", "b", "c")),
      subjects = as.list(as.character(seq_len(nrow(m))))
    )
  }
  rows <- lapply(designs, function(design) {
    long <- design$long
    formula <- blocked_formula(design)
    friedman <- probe_attempt(stats::friedman.test(formula, data = long))
    paired <- design$paired_long
    durbin <- probe_attempt(PMCMRplus::durbinAllPairsTest(
      y = paired[[design$y]], groups = paired[[design$x]], blocks = paired$.rowid,
      p.adjust.method = "none"
    ))
    kendall <- probe_attempt(effectsize::kendalls_w(
      formula, data = long, ci = NULL, verbose = FALSE
    ))
    # With its bootstrap CI, as statsExpressions calls it (100 resamples, the cases'
    # seed).
    set.seed(CASE_SEED)
    kendall_ci <- probe_attempt(effectsize::kendalls_w(
      formula, data = long, ci = 0.95, iterations = 100L, verbose = FALSE
    ))
    levels <- design$levels
    k <- length(levels)
    durbin_pairs <- list()
    if (!is.null(durbin$value)) {
      for (i in seq_len(k - 1L)) {
        for (j in (i + 1L):k) {
          durbin_pairs[[length(durbin_pairs) + 1L]] <- list(
            group1 = levels[[i]], group2 = levels[[j]],
            statistic_hex = hex(durbin$value$statistic[levels[[j]], levels[[i]]]),
            p_value_hex = hex(durbin$value$p.value[levels[[j]], levels[[i]]])
          )
        }
      }
    }
    list(
      name = design$name, levels = as.list(levels), subjects = design$subjects,
      values_hex = design$values_hex,
      friedman_statistic_hex = if (is.null(friedman$value)) NULL else hex(unname(friedman$value$statistic)),
      friedman_parameter = if (is.null(friedman$value)) NULL else unname(friedman$value$parameter),
      friedman_p_value_hex = if (is.null(friedman$value)) NULL else hex(friedman$value$p.value),
      friedman_error = friedman$error, friedman_warnings = friedman$warnings,
      durbin_method = if (is.null(durbin$value)) NULL else to_json_value(durbin$value$method),
      durbin_dist = if (is.null(durbin$value)) NULL else to_json_value(durbin$value$dist),
      durbin_error = durbin$error, durbin_warnings = durbin$warnings,
      durbin_pairs = durbin_pairs,
      kendalls_w_hex = if (is.null(kendall$value)) NULL else hex(kendall$value[[1L]]),
      kendalls_w_error = kendall$error, kendalls_w_warnings = kendall$warnings,
      kendalls_w_ci = if (is.null(kendall_ci$value)) NULL else frame_with_hex(kendall_ci$value),
      kendalls_w_ci_error = kendall_ci$error, kendalls_w_ci_warnings = kendall_ci$warnings
    )
  })
  probe_file(
    "probe-friedman",
    "friedman.test, PMCMRplus::durbinAllPairsTest (unadjusted) and effectsize::kendalls_w on within-subject data with ties",
    list(designs = rows),
    datasets = Filter(Negate(is.null), lapply(designs, function(design) design$entry))
  )
}

# probe-rmanova: afex::aov_ez as statsExpressions calls it (include_aov = TRUE) on the
# within-subject datasets: the ANOVA table without correction and with the
# Greenhouse-Geisser and Huynh-Feldt corrections, the sphericity tests and epsilons
# (summary()), the error strata's sums of squares, what tidy_model_parameters()
# (ggstatsplot's subtitle) reads from the model, effectsize::omega_squared on the model
# at 95% and 90%, and effectsize::F_to_omega2 at the uncorrected and the corrected df.
rmanova_probe <- function() {
  designs <- lapply(WITHIN_PROBE_DESIGNS, within_probe_design)
  rows <- lapply(designs, function(design) {
    long <- design$long
    fit <- probe_call(
      paste("aov_ez for", design$name),
      afex::aov_ez(id = ".rowid", dv = design$y, data = long, within = design$x, include_aov = TRUE)
    )
    mod <- fit$value
    tables <- lapply(c("none", "GG", "HF"), function(correction) {
      out <- probe_attempt(as.data.frame(anova(mod, correction = correction, es = "none")))
      list(correction = correction, table = if (is.null(out$value)) NULL else frame_with_hex(out$value), error = out$error)
    })
    sphericity <- probe_attempt(summary(mod))
    s <- sphericity$value
    strata <- probe_attempt(lapply(summary(mod$aov), function(stratum) {
      table <- stratum[[1L]]
      frame_with_hex(data.frame(term = trimws(rownames(table)), unclass(table), check.names = FALSE))
    }))
    tidy <- probe_attempt(statsExpressions::tidy_model_parameters(mod))
    omega <- lapply(c(0.95, 0.9), function(ci) {
      out <- probe_attempt(effectsize::omega_squared(mod, ci = ci, verbose = FALSE))
      list(ci = ci, value = if (is.null(out$value)) NULL else frame_with_hex(out$value), error = out$error, warnings = out$warnings)
    })
    uncorrected <- tables[[1L]]$table
    gg <- tables[[2L]]$table
    from_f <- function(table) {
      if (is.null(table)) {
        return(NULL)
      }
      row <- table[[1L]]
      f <- row[["F"]]
      lapply(c(0.95, 0.9), function(ci) {
        out <- probe_attempt(effectsize::F_to_omega2(f, row[["num Df"]], row[["den Df"]], ci = ci))
        list(ci = ci, value = if (is.null(out$value)) NULL else frame_with_hex(out$value), error = out$error)
      })
    }
    list(
      name = design$name, levels = as.list(design$levels), subjects = design$subjects,
      values_hex = design$values_hex,
      aov_ez_warnings = as.list(unique(fit$warnings)), aov_ez_messages = as.list(unique(fit$messages)),
      anova = tables,
      sphericity_tests = if (is.null(s)) NULL else raw_matrix(s$sphericity.tests),
      pval_adjustments = if (is.null(s)) NULL else raw_matrix(s$pval.adjustments),
      summary_error = sphericity$error,
      strata = strata$value, strata_error = strata$error,
      tidy_model_parameters = if (is.null(tidy$value)) NULL else frame_with_hex(tidy$value),
      tidy_error = tidy$error,
      omega_squared = omega,
      f_to_omega2_uncorrected = from_f(uncorrected),
      f_to_omega2_gg = from_f(gg)
    )
  })
  probe_file(
    "probe-rmanova",
    "afex::aov_ez with no, GG and HF corrections, sphericity epsilons, error strata, omega squared",
    list(designs = rows),
    datasets = lapply(designs, function(design) design$entry)
  )
}

# probe-wrs2within: WRS2's within-subject functions next to exact tails.
# 1. yuend over shifts that take |t| from 0 to about 40, at n 6, 20 and 300:
#    t, df, p, the trimmed-mean difference and CI, with 1 - pt and the exact tail.
# 2. rmanova on three conditions over the same shifts (n 8, 20, 200): F, df, p, with
#    1 - pf and the exact tail.
# 3. rmmcp on those designs: its whole table (rmmcp reports a critical p per pair,
#    p.crit, from its own rule at alpha 0.05), with yuend on each pair.
# 4. dep.effect's whole table (AKP, QS, QStr, SIGN with CIs) and wmcpAKP's output on
#    the within-subject datasets at tr 0.2 and 0.1 (seed 1, 100 resamples).
wrs2within_probe <- function() {
  t_tails <- function(t, df) {
    list(
      one_minus_cdf_hex = hex(1 - stats::pt(abs(t), df)),
      upper_tail_hex = hex(stats::pt(abs(t), df, lower.tail = FALSE))
    )
  }
  # A fixed, scrambled order of n normal scores: the part of y not shared with x.
  scrambled <- function(n) normal_scores(n)[order(sin(seq_len(n) * 12.9898))]
  yuend_rows <- list()
  for (n in c(6L, 20L, 300L)) {
    x <- normal_scores(n)
    y0 <- 0.6 * x + 0.8 * scrambled(n)
    unit <- 1.4 * stats::sd(x - y0) / sqrt(n)
    for (shift in WRS2P_SHIFTS) {
      y <- y0 + shift * unit
      r <- probe_call("yuend", WRS2::yuend(x, y))$value
      yuend_rows[[length(yuend_rows) + 1L]] <- c(
        list(
          n = n, shift = shift, x_hex = hex_list(x), y_hex = hex_list(y),
          test_hex = hex(r$test), df_hex = hex(r$df), p_value_hex = hex(r$p.value),
          p_value = r$p.value, diff_hex = hex(r$diff), conf_int_hex = hex_list(r$conf.int)
        ),
        t_tails(r$test, r$df)
      )
    }
  }
  rm_rows <- list()
  for (n in c(8L, 20L, 200L)) {
    base <- normal_scores(n)
    noise <- cbind(scrambled(n), rev(scrambled(n)), scrambled(n)[c(n, seq_len(n - 1L))])
    unit <- 1.4 * stats::sd(noise[, 1L] - noise[, 2L]) / sqrt(n)
    for (shift in WRS2P_SHIFTS) {
      m <- base + 0.7 * noise + outer(rep(1, n), c(0, 1, 2) * shift * unit)
      y <- as.vector(t(m))
      groups <- factor(rep(c("a", "b", "c"), times = n))
      blocks <- rep(seq_len(n), each = 3L)
      r <- probe_call("rmanova", WRS2::rmanova(y, groups, blocks))$value
      mcp <- probe_call("rmmcp", WRS2::rmmcp(y, groups, blocks))$value
      pairs <- lapply(list(c(1L, 2L), c(1L, 3L), c(2L, 3L)), function(p) {
        yd <- WRS2::yuend(m[, p[[1L]]], m[, p[[2L]]])
        c(
          list(
            group1 = c("a", "b", "c")[[p[[1L]]]], group2 = c("a", "b", "c")[[p[[2L]]]],
            yuend_test_hex = hex(yd$test), yuend_df_hex = hex(yd$df),
            yuend_p_value_hex = hex(yd$p.value), yuend_diff_hex = hex(yd$diff)
          ),
          t_tails(yd$test, yd$df)
        )
      })
      rm_rows[[length(rm_rows) + 1L]] <- list(
        n = n, shift = shift,
        values_hex = lapply(1:3, function(j) hex_list(m[, j])),
        test_hex = hex(r$test), df1_hex = hex(r$df1), df2_hex = hex(r$df2),
        p_value_hex = hex(r$p.value), p_value = r$p.value,
        one_minus_cdf_hex = hex(1 - stats::pf(r$test, r$df1, r$df2)),
        upper_tail_hex = hex(stats::pf(r$test, r$df1, r$df2, lower.tail = FALSE)),
        rmmcp = frame_with_hex(data.frame(unclass(mcp$comp), check.names = FALSE)),
        rmmcp_fnames = to_json_value(mcp$fnames),
        rmmcp_tidy = frame_with_hex(statsExpressions::tidy_model_parameters(mcp)),
        pairs = pairs
      )
    }
  }
  two <- list(
    sleep = list(dataset = "sleep", x = "group", y = "extra", subject = "ID"),
    edge_within2_small = list(dataset = "edge_within2_small"),
    edge_within2_ties = list(dataset = "edge_within2_ties"),
    edge_within2_outliers = list(dataset = "edge_within2_outliers")
  )
  entries <- list()
  dep <- list()
  for (name in names(two)) {
    design <- c(list(name = name), two[[name]])
    d <- within_probe_design(design)
    entries[[length(entries) + 1L]] <- d$entry
    for (tr in c(0.2, 0.1)) {
      set.seed(1L)
      out <- probe_attempt(WRS2::dep.effect(x = d$wide[[2L]], y = d$wide[[3L]], tr = tr, nboot = 100L))
      dep[[length(dep) + 1L]] <- list(
        name = name, tr = tr, levels = as.list(d$levels), values_hex = d$values_hex,
        table = if (is.null(out$value)) NULL else raw_matrix(out$value),
        tidy = if (is.null(out$value)) NULL else frame_with_hex(statsExpressions::tidy_model_parameters(out$value, keep = "AKP")),
        error = out$error, warnings = out$warnings
      )
    }
  }
  akp <- list()
  for (design in WITHIN_PROBE_DESIGNS[c(1L, 3L, 4L, 8L)]) {
    d <- within_probe_design(design)
    entries[[length(entries) + 1L]] <- d$entry
    wide <- d$wide
    wide$.rowid <- NULL
    for (tr in c(0.2, 0.1)) {
      set.seed(1L)
      out <- probe_attempt(WRS2::wmcpAKP(wide, tr = tr, nboot = 100L))
      akp[[length(akp) + 1L]] <- list(
        name = d$name, tr = tr, levels = as.list(d$levels), values_hex = d$values_hex,
        value = if (is.null(out$value)) NULL else to_json_value(unclass(out$value)),
        value_hex = if (is.null(out$value)) NULL else lapply(Filter(is.numeric, unclass(out$value)), hex_list),
        error = out$error, warnings = out$warnings
      )
    }
  }
  probe_file(
    "probe-wrs2within",
    "WRS2 yuend, rmanova and rmmcp next to exact tails; dep.effect and wmcpAKP outputs",
    list(yuend = yuend_rows, rmanova = rm_rows, dep_effect = dep, wmcp_akp = akp),
    datasets = unique(entries)
  )
}

# Phase 4 probes ----------------------------------------------------------------------
#
# The percentile rules of every bootstrapped interval ggstatsplot uses, read from R's
# own replicate values (probe-bootrules); correlation::correlation's methods
# (probe-corr); one-sample JZS Bayes factors at large |t| (probe-onesamplebf).

# Tracing. While `run()` runs, base R's sort.int (behind sort()), sample.int and
# sample, stats' quantile.default and boot::boot are traced: for every sort.int or
# quantile call on a numeric vector whose length is in `lengths`, the vector it was
# given and what it returned; for sample.int and sample (`record_samples`), what they
# returned; for boot::boot, its t0 and t. This records the bootstrap replicate values
# a function sorts or takes quantiles of, and the resamples it draws, without reading
# its code. Tracing changes no value and draws no random number.
TRACE <- new.env()
TRACE$active <- FALSE

.trace_entry <- function(name, x, extra = NULL) {
  if (isTRUE(TRACE$active)) {
    TRACE$stack[[length(TRACE$stack) + 1L]] <- list(name = name, x = x, extra = extra)
  }
  invisible(NULL)
}

.trace_exit <- function(name, value) {
  if (isTRUE(TRACE$active) && length(TRACE$stack) > 0L) {
    n <- length(TRACE$stack)
    top <- TRACE$stack[[n]]
    TRACE$stack[[n]] <- NULL
    if (is.numeric(top$x) && length(top$x) %in% TRACE$lengths) {
      TRACE$calls[[length(TRACE$calls) + 1L]] <- list(
        name = name, input = as.numeric(top$x), extra = top$extra,
        output = if (is.numeric(value)) as.numeric(value) else NULL
      )
    }
  }
  invisible(NULL)
}

.trace_sample <- function(name, value) {
  if (isTRUE(TRACE$active) && isTRUE(TRACE$record_samples) && is.numeric(value)) {
    TRACE$samples[[length(TRACE$samples) + 1L]] <- list(name = name, value = as.numeric(value))
  }
  invisible(NULL)
}

.trace_boot <- function(value) {
  if (isTRUE(TRACE$active) && is.list(value)) {
    TRACE$boots[[length(TRACE$boots) + 1L]] <- list(
      t0 = as.numeric(value$t0), t = as.numeric(value$t), dim = dim(value$t)
    )
  }
  invisible(NULL)
}

TRACE_TARGETS <- list(
  list(
    what = "sort.int", where = "base",
    tracer = quote(.trace_entry("sort.int", x)),
    exit = quote(.trace_exit("sort.int", returnValue()))
  ),
  list(
    what = "quantile.default", where = "stats",
    tracer = quote(.trace_entry("quantile", x, list(probs = probs, type = type))),
    exit = quote(.trace_exit("quantile", returnValue()))
  ),
  list(
    what = "sample.int", where = "base", tracer = NULL,
    exit = quote(.trace_sample("sample.int", returnValue()))
  ),
  list(
    what = "sample", where = "base", tracer = NULL,
    exit = quote(.trace_sample("sample", returnValue()))
  ),
  list(
    what = "boot", where = "boot", tracer = NULL,
    exit = quote(.trace_boot(returnValue()))
  )
)

trace_where <- function(name) if (name == "base") baseenv() else asNamespace(name)

traced_run <- function(run, lengths, record_samples = FALSE) {
  loadNamespace("boot")
  TRACE$stack <- list()
  TRACE$calls <- list()
  TRACE$samples <- list()
  TRACE$boots <- list()
  TRACE$lengths <- lengths
  TRACE$record_samples <- record_samples
  on.exit({
    TRACE$active <- FALSE
    for (target in TRACE_TARGETS) {
      suppressMessages(untrace(target$what, where = trace_where(target$where)))
    }
  })
  for (target in TRACE_TARGETS) {
    args <- list(
      what = target$what, exit = target$exit, print = FALSE,
      where = trace_where(target$where)
    )
    if (!is.null(target$tracer)) args$tracer <- target$tracer
    # quote = TRUE: the tracer and exit expressions are passed unevaluated.
    suppressMessages(do.call(trace, args, quote = TRUE))
  }
  TRACE$active <- TRUE
  out <- probe_attempt(run())
  TRACE$active <- FALSE
  list(
    result = out,
    calls = lapply(TRACE$calls, function(call) {
      list(
        name = call$name, length = length(call$input),
        input_hex = hex_list(call$input),
        extra = if (is.null(call$extra)) NULL else lapply(call$extra, encode_doubles),
        output_hex = if (is.null(call$output)) NULL else hex_list(call$output)
      )
    }),
    samples = lapply(TRACE$samples, function(s) list(name = s$name, value_hex = hex_list(s$value))),
    boots = lapply(TRACE$boots, function(b) {
      list(t0_hex = hex_list(b$t0), t_hex = hex_list(b$t), dim = as.list(b$dim))
    }),
    unbalanced = length(TRACE$stack)
  )
}

# probe-bootrules (review T6 and T7): for each bootstrapped interval ggstatsplot uses
# (akp.effect, t1way's explanatory measure, rank_epsilon_squared, dep.effect,
# wmcpAKP, kendalls_w, and trimcibt's bootstrap-t), the bounds the function returns
# next to the replicate values it sorted (traced), at B = 39, 40, 99, 100, 199 and 200
# resamples and, where the function takes a level, 95% and 90%, so that the candidate
# percentile rules give different order statistics and a percentile helper applied
# to R's own replicates must return R's bounds exactly. At B = 100 and 95% the
# resamples drawn are recorded too. Then the T7 data sets with undefined resamples
# (edge_akp_undefined and 2 for AKP, edge_tied3_small for rank epsilon squared,
# edge_rm_some_tied for Kendall's W) at the resample counts ggstatsplot (100) and
# inferplot (2,000) use, three seeds each: what each function does with a 0/0
# resample.
BOOTRULE_B <- c(39L, 40L, 99L, 100L, 199L, 200L)
BOOTRULE_CONF <- c(0.95, 0.90)

bootrule_designs <- function() {
  scrambled <- function(n) normal_scores(n)[order(sin(seq_len(n) * 12.9898))]
  two <- data.frame(
    y = c(normal_scores(12L) * 2 + 10, normal_scores(15L) * 3 + 11.5),
    x = factor(rep(c("a", "b"), c(12L, 15L)))
  )
  three <- smm_data(3L)
  three <- data.frame(y = three$y, x = three$g)
  paired_x <- normal_scores(15L)
  paired_y <- 0.6 * paired_x + 0.8 * scrambled(15L) + 0.4
  base <- normal_scores(12L) * 2
  wide <- data.frame(
    a = base + 0.7 * scrambled(12L),
    b = base + 0.7 * rev(scrambled(12L)) + 0.5,
    c = base + 0.7 * scrambled(12L)[c(12L, 1:11)] + 0.9
  )
  long <- data.frame(
    .rowid = factor(rep(seq_len(12L), each = 3L)),
    x = factor(rep(c("a", "b", "c"), times = 12L)),
    y = as.vector(t(as.matrix(wide)))
  )
  one <- normal_scores(20L) * 2 + 0.7
  conf_alpha <- function(conf) list(text = paste("1 -", format(conf)), value = 1.0 - conf)
  list(
    akp.effect = list(
      call = "WRS2::akp.effect(y ~ x, two, EQVAR = FALSE, tr = 0.2, nboot = B, alpha = 1 - conf)",
      data = list(a_hex = hex_list(two$y[two$x == "a"]), b_hex = hex_list(two$y[two$x == "b"])),
      conf = BOOTRULE_CONF,
      run = function(B, conf) akp_candidate(two, FALSE, conf_alpha(conf), 0.2)$run(B)$values
    ),
    t1way = list(
      call = "WRS2::t1way(y ~ x, three, tr = 0.2, alpha = 1 - conf, nboot = B)",
      data = lapply(split(three$y, three$x), hex_list),
      conf = BOOTRULE_CONF,
      run = function(B, conf) {
        r <- WRS2::t1way(y ~ x, data = three, tr = 0.2, alpha = 1.0 - conf, nboot = B)
        c(r$effsize, r$effsize_ci)
      }
    ),
    rank_epsilon_squared = list(
      call = "effectsize::rank_epsilon_squared(y ~ x, three, ci = conf, alternative = \"greater\", iterations = B)",
      data = lapply(split(three$y, three$x), hex_list),
      conf = BOOTRULE_CONF,
      run = function(B, conf) {
        r <- effectsize::rank_epsilon_squared(
          y ~ x, data = three, ci = conf, alternative = "greater", iterations = B,
          verbose = FALSE
        )
        c(r[[1L]], r$CI_low, r$CI_high)
      }
    ),
    dep.effect = list(
      call = "WRS2::dep.effect(x = paired_x, y = paired_y, tr = 0.2, nboot = B)",
      data = list(x_hex = hex_list(paired_x), y_hex = hex_list(paired_y)),
      conf = NA_real_,
      run = function(B, conf) {
        r <- WRS2::dep.effect(x = paired_x, y = paired_y, tr = 0.2, nboot = B)
        tidy_effect(r, keep = "AKP")
      }
    ),
    wmcpAKP = list(
      call = "WRS2::wmcpAKP(wide, tr = 0.2, nboot = B)",
      data = lapply(wide, hex_list),
      conf = NA_real_,
      run = function(B, conf) tidy_effect(WRS2::wmcpAKP(wide, tr = 0.2, nboot = B))
    ),
    kendalls_w = list(
      call = "effectsize::kendalls_w(y ~ x | .rowid, long, ci = conf, iterations = B)",
      data = lapply(wide, hex_list),
      conf = BOOTRULE_CONF,
      run = function(B, conf) {
        r <- effectsize::kendalls_w(
          y ~ x | .rowid, data = long, ci = conf, iterations = B, verbose = FALSE
        )
        c(r[[1L]], r$CI_low, r$CI_high)
      }
    ),
    trimcibt = list(
      call = "WRS2::trimcibt(one, nv = 0.5, tr = 0.2, alpha = 1 - conf, nboot = B)",
      data = list(x_hex = hex_list(one)),
      conf = BOOTRULE_CONF,
      run = function(B, conf) {
        r <- WRS2::trimcibt(x = one, nv = 0.5, tr = 0.2, alpha = 1.0 - conf, nboot = B)
        c(r$estimate, as.numeric(r$ci)[1:2], r$p.value, r$test.stat)
      }
    )
  )
}

bootrule_row <- function(design, B, conf, seed, record_samples) {
  set.seed(seed)
  traced <- traced_run(function() design$run(B, conf), lengths = B, record_samples = record_samples)
  values <- traced$result$value
  list(
    B = B, conf = encode_doubles(conf), seed = seed,
    values_hex = if (is.null(values)) NULL else hex_list(values),
    error = traced$result$error, warnings = traced$result$warnings,
    traced = traced$calls, samples = traced$samples, boots = traced$boots,
    unbalanced_traces = traced$unbalanced
  )
}

bootrules_probe <- function() {
  designs <- bootrule_designs()
  rules <- lapply(names(designs), function(name) {
    design <- designs[[name]]
    rows <- list()
    for (B in BOOTRULE_B) {
      for (conf in design$conf) {
        rows[[length(rows) + 1L]] <- bootrule_row(
          design, B, conf, 1L, record_samples = B == 100L && (is.na(conf) || conf == 0.95)
        )
      }
    }
    list(name = name, call = design$call, data = design$data, rows = rows)
  })

  # T7: undefined resamples.
  akp1 <- probe_dataset("edge_akp_undefined")
  akp2 <- probe_dataset("edge_akp_undefined2")
  tied <- probe_dataset("edge_tied3_small")
  some <- probe_dataset("edge_rm_some_tied")
  between_frame <- function(set) between_analysis(set$data, "group", "value")
  akp_design <- function(set) {
    frame <- between_frame(set)
    list(
      call = "WRS2::akp.effect(y ~ x, data, EQVAR = FALSE, tr = 0.2, nboot = B, alpha = 1 - 0.95)",
      data = lapply(split(frame$y, frame$x), hex_list),
      run = function(B, conf) akp_candidate(frame, FALSE, conf_alpha95, 0.2)$run(B)$values
    )
  }
  conf_alpha95 <- list(text = "1 - 0.95", value = 1.0 - 0.95)
  tied_frame <- between_frame(tied)
  some_long <- within_analysis(some$data, "condition", "value", "subject")
  some_long$.rowid <- as.factor(some_long$.rowid)
  undefined_designs <- list(
    edge_akp_undefined = akp_design(akp1),
    edge_akp_undefined2 = akp_design(akp2),
    edge_tied3_small = list(
      call = "effectsize::rank_epsilon_squared(y ~ x, data, ci = 0.95, alternative = \"greater\", iterations = B)",
      data = lapply(split(tied_frame$y, tied_frame$x), hex_list),
      run = function(B, conf) {
        r <- effectsize::rank_epsilon_squared(
          y ~ x, data = tied_frame, ci = 0.95, alternative = "greater", iterations = B,
          verbose = FALSE
        )
        c(r[[1L]], r$CI_low, r$CI_high)
      }
    ),
    edge_rm_some_tied = list(
      call = "effectsize::kendalls_w(y ~ x | .rowid, data, ci = 0.95, iterations = B)",
      data = lapply(split(some_long$y, some_long$x), hex_list),
      run = function(B, conf) {
        r <- effectsize::kendalls_w(
          y ~ x | .rowid, data = some_long, ci = 0.95, iterations = B, verbose = FALSE
        )
        c(r[[1L]], r$CI_low, r$CI_high)
      }
    )
  )
  undefined <- lapply(names(undefined_designs), function(name) {
    design <- undefined_designs[[name]]
    rows <- list()
    for (B in c(GG_RESAMPLES, BOOT_REPLICATE_RESAMPLES)) {
      for (seed in c(CASE_SEED, 1L, 2L)) {
        rows[[length(rows) + 1L]] <- bootrule_row(design, B, 0.95, seed, record_samples = FALSE)
      }
    }
    list(name = name, call = design$call, data = design$data, rows = rows)
  })
  probe_file(
    "probe-bootrules",
    paste(
      "bootstrap bounds next to the replicate values each function sorted (traced sort.int,",
      "quantile and boot::boot), at several B and levels; T7 data with undefined resamples"
    ),
    list(rules = rules, undefined = undefined),
    datasets = list(akp1$entry, akp2$entry, tied$entry, some$entry)
  )
}

# probe-corr: what correlation::correlation computes for ggscatterstats, next to the
# base R and package functions that could be behind it: cor.test (Pearson at 95% and
# 90%; Spearman with exact NULL, TRUE and FALSE); correlation() with Pearson and
# Spearman at both levels; the winsorized correlation at tr 0.2 and 0.1 next to
# datawizard::winsorize's values and WRS2::wincor; and the Bayes factor at priors
# 0.707 and 0.8 next to BayesFactor::correlationBF at those rscales and at "medium".
corr_probe <- function() {
  cars <- probe_dataset("mtcars")
  iris_set <- probe_dataset("iris")
  sets <- list(
    small = probe_dataset("edge_corr_small"),
    ties = probe_dataset("edge_corr_ties"),
    null = probe_dataset("edge_corr_null"),
    outlier = probe_dataset("edge_corr_outlier"),
    missing = probe_dataset("edge_corr_missing"),
    perfect = probe_dataset("edge_corr_perfect")
  )
  complete <- function(set, x, y) {
    keep <- !is.na(set$data[[x]]) & !is.na(set$data[[y]])
    list(x = as.numeric(set$data[[x]][keep]), y = as.numeric(set$data[[y]][keep]))
  }
  designs <- c(
    list(
      mtcars = complete(cars, "wt", "mpg"),
      iris = complete(iris_set, "Sepal.Length", "Petal.Length")
    ),
    lapply(sets, function(set) complete(set, "x", "y")),
    list(perfect_neg = complete(sets$perfect, "x", "y_neg"))
  )
  rows <- lapply(names(designs), function(name) {
    x <- designs[[name]]$x
    y <- designs[[name]]$y
    data <- data.frame(x = x, y = y)
    row_of <- function(out) if (is.null(out$value)) NULL else frame_with_hex(out$value)
    pearson <- lapply(c(0.95, 0.9), function(cl) {
      out <- probe_attempt(stats::cor.test(x, y, conf.level = cl))
      r <- out$value
      list(
        conf = cl,
        estimate_hex = if (is.null(r)) NULL else hex(unname(r$estimate)),
        statistic_hex = if (is.null(r)) NULL else hex(unname(r$statistic)),
        parameter = if (is.null(r)) NULL else unname(r$parameter),
        p_value_hex = if (is.null(r)) NULL else hex(r$p.value),
        conf_int_hex = if (is.null(r)) NULL else hex_list(r$conf.int),
        error = out$error, warnings = out$warnings
      )
    })
    spearman <- lapply(list(NULL, TRUE, FALSE), function(exact) {
      out <- probe_attempt(stats::cor.test(x, y, method = "spearman", exact = exact))
      r <- out$value
      list(
        exact = if (is.null(exact)) "NULL" else exact,
        estimate_hex = if (is.null(r)) NULL else hex(unname(r$estimate)),
        statistic_hex = if (is.null(r)) NULL else hex(unname(r$statistic)),
        p_value_hex = if (is.null(r)) NULL else hex(r$p.value),
        error = out$error, warnings = out$warnings
      )
    })
    correlation_rows <- list()
    for (method in c("pearson", "spearman")) {
      for (cl in c(0.95, 0.9)) {
        out <- probe_attempt(correlation::correlation(data, method = method, ci = cl))
        correlation_rows[[length(correlation_rows) + 1L]] <- list(
          method = method, ci = cl, row = row_of(out), error = out$error, warnings = out$warnings
        )
      }
    }
    winsorized <- lapply(c(0.2, 0.1), function(tr) {
      out <- probe_attempt(correlation::correlation(data, method = "pearson", ci = 0.95, winsorize = tr))
      w <- probe_attempt(datawizard::winsorize(data, threshold = tr))
      wc <- probe_attempt(WRS2::wincor(x, y, tr = tr))
      list(
        tr = tr, correlation = row_of(out), error = out$error, warnings = out$warnings,
        winsorized_x_hex = if (is.null(w$value)) NULL else hex_list(w$value$x),
        winsorized_y_hex = if (is.null(w$value)) NULL else hex_list(w$value$y),
        winsorize_error = w$error, winsorize_warnings = w$warnings,
        wincor = if (is.null(wc$value)) NULL else lapply(Filter(is.numeric, unclass(wc$value)), hex_list),
        wincor_error = wc$error, wincor_warnings = wc$warnings
      )
    })
    bayes <- lapply(list(0.707, 0.8, "medium", 1 / 3), function(prior) {
      bf <- probe_attempt(BayesFactor::correlationBF(y = y, x = x, rscale = prior))
      extracted <- if (is.null(bf$value)) NULL else BayesFactor::extractBF(bf$value)
      log_bf <- if (is.null(bf$value)) NULL else BayesFactor::extractBF(bf$value, logbf = TRUE)
      corr <- if (is.numeric(prior)) {
        set.seed(1L)
        probe_attempt(correlation::correlation(
          data, method = "pearson", ci = 0.95, bayesian = TRUE, bayesian_prior = prior
        ))
      }
      list(
        prior = prior,
        bf10_hex = if (is.null(extracted)) NULL else hex(extracted$bf[[1L]]),
        log_bf10_hex = if (is.null(log_bf)) NULL else hex(log_bf$bf[[1L]]),
        bf_error = if (is.null(extracted)) NULL else encode_doubles(extracted$error[[1L]]),
        call_error = bf$error, warnings = bf$warnings,
        correlation = if (is.null(corr)) NULL else row_of(corr),
        correlation_error = if (is.null(corr)) NULL else corr$error,
        correlation_warnings = if (is.null(corr)) NULL else corr$warnings
      )
    })
    list(
      name = name, n = length(x), x_hex = hex_list(x), y_hex = hex_list(y),
      pearson = pearson, spearman = spearman, correlation = correlation_rows,
      winsorized = winsorized, bayes = bayes
    )
  })
  probe_file(
    "probe-corr",
    paste(
      "cor.test (Pearson; Spearman exact NULL/TRUE/FALSE), correlation::correlation, the",
      "winsorized correlation with datawizard::winsorize and WRS2::wincor, correlationBF"
    ),
    list(designs = rows),
    datasets = c(list(cars$entry, iris_set$entry), unname(lapply(sets, function(set) set$entry)))
  )
}

# probe-onesamplebf: BayesFactor::ttestBF(x, mu = mu) at |t| from 0 to 1,000 (n 5, 10,
# 30 and 200; mu 0 and 3; rscale 0.707 and 0.8), with its own error estimate and the
# one-sample t from t.test, as probe-pairedbf does for paired data: where its
# quadrature could miss the accurate one-sample Bayes factor.
onesamplebf_probe <- function() {
  rows <- list()
  for (n in c(5L, 10L, 30L, 200L)) {
    noise <- normal_scores(n)
    for (mu in c(0, 3)) {
      for (target in PAIREDBF_T) {
        x <- mu + noise + target * stats::sd(noise) / sqrt(n)
        tt <- probe_call("one-sample t.test", stats::t.test(x, mu = mu))$value
        for (rscale in c(0.707, 0.8)) {
          out <- probe_attempt(BayesFactor::ttestBF(x = x, mu = mu, rscale = rscale))
          bf <- if (is.null(out$value)) NULL else BayesFactor::extractBF(out$value)
          log_bf <- if (is.null(out$value)) NULL else BayesFactor::extractBF(out$value, logbf = TRUE)
          rows[[length(rows) + 1L]] <- list(
            n = n, mu = mu, t_target = target, rscale = rscale, rscale_hex = hex(rscale),
            x_hex = hex_list(x),
            t_hex = hex(unname(tt$statistic)), df = unname(tt$parameter),
            bf10_hex = if (is.null(bf)) NULL else hex(bf$bf[[1L]]),
            log_bf10_hex = if (is.null(log_bf)) NULL else hex(log_bf$bf[[1L]]),
            error = if (is.null(bf)) NULL else encode_doubles(bf$error[[1L]]),
            call_error = out$error, warnings = out$warnings
          )
        }
      }
    }
  }
  probe_file(
    "probe-onesamplebf",
    "BayesFactor::ttestBF(x, mu) and its error estimate at |t| from 0 to 1000",
    list(rows = rows)
  )
}

# probe-heldout: the HELD_OUT_CASES, each run exactly as a case is (run_case(),
# bootstrap reference included) and checked the same way (outcome_problems(), which
# must find nothing), its would-be fixture kept whole under `fixture`.
heldout_probe <- function() {
  entries <- lapply(HELD_OUT_CASES, function(case) {
    message("== (held out) ", case$name)
    fixture <- run_case(case)
    problems <- outcome_problems(case, fixture)
    if (length(problems) > 0L) {
      stop("held-out case ", case$name, ": ", paste(problems, collapse = "; "))
    }
    list(name = case$name, fixture = fixture)
  })
  datasets <- unique(lapply(entries, function(entry) {
    list(file = entry$fixture$dataset$file, sha256 = entry$fixture$dataset$sha256)
  }))
  probe_file(
    "probe-heldout",
    "cases whose R result differs from inferplot's by a rule not yet decided, run as cases are",
    list(cases = entries),
    datasets = datasets
  )
}

# probe-contingency: the R functions statsExpressions 2.2.1's contingency_table()
# and pairwise_contingency_table() call for ggbarstats, as black boxes on fixed
# grids: stats::chisq.test with and without the continuity correction (2 x 2 and
# larger tables, goodness of fit with given probabilities, valid and invalid),
# stats::mcnemar.test likewise (2 x 2, the McNemar-Bowker test on larger square tables,
# a table that is not square), effectsize::cramers_v (adjust TRUE and FALSE),
# effectsize::pearsons_c (goodness of fit) and effectsize::cohens_g at every
# alternative and at 95% and 90%, next to stats::prop.test's interval for the share
# of one kind of discordant pair (the documented basis of Cohen's g's interval), and
# stats::fisher.test at every alternative. Last, the percent labels ggbarstats writes
# inside the bars, paste0(round(perc, digits.perc), "%"), for counts k of n.
contingency_probe <- function() {
  attempt_row <- function(expr, fields) {
    out <- probe_attempt(expr)
    values <- if (is.null(out$value)) NULL else fields(out$value)
    c(values, list(error = out$error, warnings = out$warnings))
  }
  test_fields <- function(r) {
    list(
      statistic_hex = hex(unname(r$statistic)), parameter = encode_doubles(unname(r$parameter)),
      p_value_hex = hex(r$p.value), method = r$method,
      expected_hex = if (!is.null(r$expected)) hex_list(r$expected)
    )
  }
  es_fields <- function(r) {
    list(
      name = names(r)[[1L]], estimate_hex = hex(r[[1L]]), ci = r$CI,
      ci_low_hex = hex(r$CI_low), ci_high_hex = hex(r$CI_high)
    )
  }
  as_table <- function(v, nrow) matrix(as.numeric(v), nrow = nrow, byrow = TRUE)
  # R's own noncentral chi-square probability P(X <= chisq) at the noncentrality
  # parameter behind each reported bound in (0, 1) (`to_ncp` inverts the effect
  # size's conversion), to read how far effectsize's root search stopped from the
  # interval's tail probability.
  tails_at_bounds <- function(es, chisq, df, to_ncp) {
    lapply(list(low = es$CI_low, high = es$CI_high), function(bound) {
      if (is.null(bound) || !is.finite(bound) || bound <= 0 || bound >= 1 || !is.finite(chisq)) {
        return(NULL)
      }
      ncp <- to_ncp(bound)
      list(ncp_hex = hex(ncp), cdf_hex = hex(stats::pchisq(chisq, df, ncp = ncp)))
    })
  }
  # attempt_row() for an effect size, with tails_at_bounds() of its interval.
  es_row <- function(expr, chisq, df, to_ncp) {
    out <- probe_attempt(expr)
    es <- out$value
    c(
      if (is.null(es)) NULL else es_fields(es),
      list(
        error = out$error, warnings = out$warnings,
        tails_at_bounds = if (!is.null(es)) tails_at_bounds(es, chisq, df, to_ncp)
      )
    )
  }
  two_by_two <- list(
    c(1, 9, 8, 4), c(10, 2, 3, 15), c(0, 5, 5, 0), c(5, 5, 5, 5), c(12, 0, 0, 7),
    c(3, 1, 1, 3), c(30, 20, 25, 25), c(100, 80, 90, 110), c(1, 0, 0, 1), c(6, 1, 2, 7),
    c(50, 1, 1, 50), c(4, 4, 4, 5), c(0, 0, 3, 4), c(2, 7, 9, 3)
  )
  larger <- list(
    list(nrow = 2L, v = c(6, 0, 4, 3, 7, 5)),
    list(nrow = 3L, v = c(3, 1, 0, 0, 2, 1, 1, 0, 4)),
    list(nrow = 3L, v = c(8, 0, 0, 0, 6, 0, 0, 0, 7)),
    list(nrow = 2L, v = c(4, 8, 12, 6, 12, 18)),
    list(nrow = 3L, v = c(3, 8, 0, 4, 3, 12, 2, 2, 0)),
    list(nrow = 4L, v = c(122, 203, 167, 118, 528, 178, 673, 212)),
    list(nrow = 5L, v = unlist(lapply(1:5, function(i) {
      60 + 15 * i + 25 * (1:4) + 40 * ((i * (1:4)) %% 3)
    }))),
    list(nrow = 2L, v = c(1, 2, 3, 4, 40, 50, 60, 70))
  )
  tables <- c(
    lapply(two_by_two, function(v) as_table(v, 2L)),
    lapply(larger, function(t) as_table(t$v, t$nrow))
  )
  chisq <- lapply(tables, function(m) {
    list(
      table = lapply(seq_len(nrow(m)), function(i) as.list(m[i, ])),
      correct = attempt_row(stats::chisq.test(m, correct = TRUE), test_fields),
      uncorrected = attempt_row(stats::chisq.test(m, correct = FALSE), test_fields)
    )
  })
  gof_designs <- list(
    list(x = c(11, 7, 14), p = NULL), list(x = c(11, 7, 14), p = c(0.5, 0.25, 0.25)),
    list(x = c(11, 7, 14), p = c(0.25, 0.25, 0.5)), list(x = c(3, 0, 5), p = NULL),
    list(x = c(10, 10, 10), p = NULL), list(x = c(1, 2), p = NULL),
    list(x = c(50, 0), p = NULL), list(x = c(19, 13), p = c(0.6, 0.4)),
    list(x = c(11, 7, 14), p = c(2, 1, 1)), list(x = c(11, 7, 14), p = c(0.5, 0.5)),
    list(x = c(11, 7, 14), p = c(0.2, 0.3, 0.5)), list(x = 12, p = NULL),
    list(x = c(85, 104, 95, 106, 110), p = NULL)
  )
  gof <- lapply(gof_designs, function(d) {
    p <- d$p %||% rep(1 / length(d$x), length(d$x))
    test <- probe_attempt(stats::chisq.test(d$x, p = p, correct = FALSE))$value
    n <- sum(d$x)
    # C back to chi-square: n C^2 / (1 - C^2).
    to_ncp <- function(c) n * c^2 / (1 - c^2)
    alternatives <- lapply(c("two.sided", "greater", "less"), function(alternative) {
      lapply(c(0.95, 0.9), function(ci) {
        c(
          list(alternative = alternative, ci = ci),
          es_row(
            effectsize::pearsons_c(d$x, p = p, ci = ci, alternative = alternative),
            if (is.null(test)) NA_real_ else unname(test$statistic),
            if (is.null(test)) NA_real_ else unname(test$parameter), to_ncp
          )
        )
      })
    })
    list(
      x = as.list(d$x), p_hex = hex_list(p), p_given = !is.null(d$p),
      chisq = attempt_row(stats::chisq.test(d$x, p = p, correct = FALSE), test_fields),
      pearsons_c = unlist(alternatives, recursive = FALSE)
    )
  })
  cramers <- lapply(tables, function(m) {
    rows <- list()
    kept <- m[rowSums(m) > 0, colSums(m) > 0, drop = FALSE]
    n <- sum(kept)
    test <- probe_attempt(stats::chisq.test(kept, correct = FALSE))$value
    statistic <- if (is.null(test)) NA_real_ else unname(test$statistic)
    df <- if (is.null(test)) NA_real_ else unname(test$parameter)
    for (adjust in c(TRUE, FALSE)) {
      # V back to chi-square: n phi^2, with phi^2 = V^2 (min(r, c) - 1) or, adjusted,
      # V^2 (min(r~, c~) - 1) + (r - 1)(c - 1) / (n - 1) (Bergsma 2013).
      to_ncp <- function(v) {
        r <- nrow(kept)
        k <- ncol(kept)
        if (!adjust) {
          return(n * v^2 * (min(r, k) - 1))
        }
        shrunk <- min(r - (r - 1)^2 / (n - 1), k - (k - 1)^2 / (n - 1)) - 1
        n * (v^2 * shrunk + (r - 1) * (k - 1) / (n - 1))
      }
      for (alternative in c("two.sided", "greater", "less")) {
        for (ci in c(0.95, 0.9)) {
          rows[[length(rows) + 1L]] <- c(
            list(adjust = adjust, alternative = alternative, ci = ci),
            es_row(
              effectsize::cramers_v(m, adjust = adjust, ci = ci, alternative = alternative),
              statistic, df, to_ncp
            )
          )
        }
      }
    }
    list(table = lapply(seq_len(nrow(m)), function(i) as.list(m[i, ])), n = sum(m), rows = rows)
  })
  paired_tables <- c(
    lapply(
      list(
        c(14, 4, 12, 10), c(9, 0, 0, 11), c(5, 0, 1, 4), c(20, 30, 5, 45), c(1, 7, 0, 2),
        c(10, 6, 6, 10), c(0, 25, 3, 0)
      ),
      function(v) as_table(v, 2L)
    ),
    list(
      as_table(c(6, 5, 2, 1, 7, 6, 1, 3, 8), 3L),
      as_table(c(0, 0, 0, 3, 6, 0, 2, 5, 7), 3L),
      as_table(c(5, 0, 0, 0, 5, 2, 0, 1, 5), 3L),
      as_table(c(4, 1, 2, 3, 0, 5, 2, 1, 3, 3, 6, 2, 1, 0, 2, 7), 4L),
      as_table(c(6, 0, 3, 7, 5, 2), 2L)
    )
  )
  paired <- lapply(paired_tables, function(m) {
    g_rows <- list()
    for (alternative in c("two.sided", "greater", "less")) {
      for (ci in c(0.95, 0.9)) {
        g_rows[[length(g_rows) + 1L]] <- c(
          list(alternative = alternative, ci = ci),
          attempt_row(effectsize::cohens_g(m, ci = ci, alternative = alternative), es_fields)
        )
      }
    }
    # The documented basis of Cohen's g's interval: prop.test on one kind of
    # discordant pair (m[1, 2] of m[1, 2] + m[2, 1]), with and without the
    # continuity correction, and binom.test's exact interval.
    basis <- if (all(dim(m) == 2L) && m[1L, 2L] + m[2L, 1L] > 0) {
      b <- m[1L, 2L]
      total <- m[1L, 2L] + m[2L, 1L]
      lapply(c(0.95, 0.9), function(ci) {
        interval <- function(r) list(conf_int_hex = hex_list(r$conf.int))
        list(
          ci = ci, successes = b, trials = total,
          prop_test_correct = attempt_row(
            stats::prop.test(b, total, conf.level = ci, correct = TRUE), interval
          ),
          prop_test_uncorrected = attempt_row(
            stats::prop.test(b, total, conf.level = ci, correct = FALSE), interval
          ),
          binom_test = attempt_row(stats::binom.test(b, total, conf.level = ci), interval)
        )
      })
    }
    list(
      table = lapply(seq_len(nrow(m)), function(i) as.list(m[i, ])),
      correct = attempt_row(stats::mcnemar.test(m, correct = TRUE), test_fields),
      uncorrected = attempt_row(stats::mcnemar.test(m, correct = FALSE), test_fields),
      cohens_g = g_rows,
      interval_basis = basis
    )
  })
  fisher_tables <- c(
    lapply(two_by_two, function(v) as_table(v, 2L)),
    list(
      as_table(c(6, 0, 4, 3, 7, 5), 2L), as_table(c(1, 2, 3, 4, 4, 3, 2, 1), 2L),
      as_table(c(3, 1, 0, 0, 2, 1, 1, 0, 4), 3L), as_table(c(5, 0, 9, 2, 3, 4), 2L),
      as_table(c(10, 12, 9, 3, 2, 14), 2L)
    )
  )
  fisher <- lapply(fisher_tables, function(m) {
    list(
      table = lapply(seq_len(nrow(m)), function(i) as.list(m[i, ])),
      tests = lapply(c("two.sided", "greater", "less"), function(alternative) {
        c(
          list(alternative = alternative),
          attempt_row(
            stats::fisher.test(m, alternative = alternative),
            function(r) list(p_value_hex = hex(r$p.value), method = r$method)
          )
        )
      })
    )
  })
  label_rows <- list()
  for (n in c(3L, 7L, 8L, 11L, 16L, 32L, 40L, 200L)) {
    counts <- seq_len(n - 1L)
    perc <- (counts / n) * 100
    for (d in 0:2) {
      for (i in seq_along(counts)) {
        label_rows[[length(label_rows) + 1L]] <- list(
          k = counts[[i]], n = n, digits = d, perc_hex = hex(perc[[i]]),
          rounded_hex = hex(round(perc[[i]], d)), label = paste0(round(perc[[i]], d), "%")
        )
      }
    }
  }
  probe_file(
    "probe-contingency",
    paste(
      "chisq.test, mcnemar.test, cramers_v, pearsons_c, cohens_g with prop.test and",
      "binom.test, fisher.test on fixed tables; ggbarstats' percent labels"
    ),
    list(
      chisq = chisq, goodness_of_fit = gof, cramers_v = cramers, paired = paired,
      fisher = fisher, percent_labels = label_rows
    )
  )
}

# Probes of review round 3 ----------------------------------------------------------
#
# What R does where the reviews could only reason about it: a difference that is 0 as
# written but not in binary in the signed-rank test (probe-nearzero), the bootstrap-t
# test on the same ratings in two writings (probe-trimcibt), Cramer's V at large df and
# exact zeros in the contingency tests (probe-bartables), and R's round() on the
# percent labels at decimal ties (probe-round). Doubles are hex text as above.

# A row of `fields(value)` from probe_attempt(), with its error and warnings.
attempt_fields <- function(expr, fields) {
  out <- probe_attempt(expr)
  c(if (!is.null(out$value)) fields(out$value), list(error = out$error, warnings = out$warnings))
}

# The subtitle row of a ggstatsplot plot in hex, and the subtitle as text.
subtitle_hex <- function(plot) {
  row <- ggstatsplot::extract_stats(plot)$subtitle_data
  list(
    statistic_hex = hex(row$statistic), p_value_hex = hex(row$p.value),
    estimate_hex = hex(row$estimate), conf_low_hex = hex(row$conf.low),
    conf_high_hex = hex(row$conf.high), n_obs = row$n.obs, effectsize = row$effectsize,
    subtitle = deparse_text(ggstatsplot::extract_subtitle(plot))
  )
}

# probe-nearzero (review W9; D88, D105): stats::wilcox.test (exact = FALSE, as
# statsExpressions calls it; and with digits.rank = 7), effectsize::rank_biserial and
# ggstatsplot on data holding a value equal to the test value 0.3 as written but not in
# binary, with R's own differences x - mu. One sample: each column of
# edge_one_near_mu against 0.3, through gghistostats too. Paired: edge_paired_near_zero
# (subjects s01 to s10 in conditions a and b; a - b is 0.1 + 0.2 - 0.3 = 2^-54 for s01,
# exactly 0 for s02), through ggwithinstats too.
nearzero_probe <- function() {
  one <- probe_dataset("edge_one_near_mu")
  paired <- probe_dataset("edge_paired_near_zero")
  mu <- 0.3
  wilcox_fields <- function(r) {
    list(
      statistic_hex = hex(unname(r$statistic)), p_value_hex = hex(r$p.value), method = r$method
    )
  }
  biserial_fields <- function(r) {
    list(
      estimate_hex = hex(r[[1L]]), ci = r$CI, ci_low_hex = hex(r$CI_low),
      ci_high_hex = hex(r$CI_high)
    )
  }
  ggstatsplot_row <- function(make) {
    HIDDEN$log <- list()
    HIDDEN$values <- list()
    set.seed(CASE_SEED)
    attempt_fields(make(), subtitle_hex)
  }
  one_sample <- lapply(names(one$data), function(column) {
    x <- as.numeric(one$data[[column]])
    frame <- data.frame(value = x)
    list(
      column = column, x_hex = hex_list(x), mu_hex = hex(mu),
      differences_hex = hex_list(x - mu), exact_zeros = sum(x - mu == 0),
      wilcox = attempt_fields(stats::wilcox.test(x, mu = mu, exact = FALSE), wilcox_fields),
      wilcox_digits7 = attempt_fields(
        stats::wilcox.test(x, mu = mu, exact = FALSE, digits.rank = 7), wilcox_fields
      ),
      rank_biserial = attempt_fields(
        effectsize::rank_biserial(x, mu = mu, ci = 0.95), biserial_fields
      ),
      gghistostats = ggstatsplot_row(function() {
        ggstatsplot::gghistostats(
          data = frame, x = value, test.value = mu, type = "nonparametric"
        )
      })
    )
  })
  data <- paired$data
  a <- as.numeric(data$value[data$condition == "a"])
  b <- as.numeric(data$value[data$condition == "b"])
  subjects <- data$subject[data$condition == "a"]
  if (!identical(subjects, data$subject[data$condition == "b"])) {
    stop("edge_paired_near_zero: the subjects of a and b are not in the same order")
  }
  paired_row <- list(
    subjects = as.list(subjects), a_hex = hex_list(a), b_hex = hex_list(b),
    differences_hex = hex_list(a - b), exact_zeros = sum(a - b == 0),
    wilcox = attempt_fields(
      stats::wilcox.test(a, b, paired = TRUE, exact = FALSE), wilcox_fields
    ),
    wilcox_digits7 = attempt_fields(
      stats::wilcox.test(a, b, paired = TRUE, exact = FALSE, digits.rank = 7), wilcox_fields
    ),
    rank_biserial = attempt_fields(
      effectsize::rank_biserial(a, b, paired = TRUE, ci = 0.95), biserial_fields
    ),
    ggwithinstats = ggstatsplot_row(function() {
      ggstatsplot::ggwithinstats(
        data = data, x = condition, y = value, subject.id = subject, type = "nonparametric"
      )
    })
  )
  probe_file(
    "probe-nearzero",
    paste(
      "wilcox.test, rank_biserial, gghistostats and ggwithinstats on differences that are",
      "0 as written but 2^-54 in binary"
    ),
    list(mu_hex = hex(mu), one_sample = one_sample, paired = paired_row),
    datasets = list(one$entry, paired$entry)
  )
}

# probe-trimcibt (review Z1; D105): WRS2::trimcibt (tr 0.2, alpha = 1 - 0.95, as
# statsExpressions calls it) on edge_one_ratings in three writings of the same numbers:
# whole numbers against 2.5 (and 3), the CSV's tenths against 0.25 (and 0.3), and the
# whole numbers times 0.1 against 2.5 x 0.1 (and 3 x 0.1), at 200 (ggstatsplot's
# nboot), 2,000 and 20,000 resamples and three seeds. The bootstrap T* values R sorted
# are traced (as in probe-bootrules): how many are NaN (sort() drops them), infinite,
# finite, and, up to 2,000 resamples, all of them in the order R computed them.
TRIMCIBT_B <- c(200L, 2000L, 20000L)
trimcibt_probe <- function() {
  ratings <- probe_dataset("edge_one_ratings")
  d <- ratings$data
  sets <- list(
    ratings = list(
      whole = list(x = as.numeric(d$whole), nv = 2.5),
      tenths = list(x = as.numeric(d$tenths), nv = 0.25),
      times_tenth = list(x = as.numeric(d$whole) * 0.1, nv = 2.5 * 0.1)
    ),
    centred = list(
      whole = list(x = as.numeric(d$centred_whole), nv = 3),
      tenths = list(x = as.numeric(d$centred_tenths), nv = 0.3),
      times_tenth = list(x = as.numeric(d$centred_whole) * 0.1, nv = 3 * 0.1)
    )
  )
  rows <- list()
  for (set in names(sets)) {
    for (writing in names(sets[[set]])) {
      x <- sets[[set]][[writing]]$x
      nv <- sets[[set]][[writing]]$nv
      for (B in TRIMCIBT_B) {
        for (seed in c(CASE_SEED, 1L, 2L)) {
          set.seed(seed)
          traced <- traced_run(function() {
            r <- WRS2::trimcibt(x = x, nv = nv, tr = 0.2, alpha = 1.0 - 0.95, nboot = B)
            c(r$estimate, as.numeric(r$ci)[1:2], r$p.value, r$test.stat)
          }, lengths = B)
          # TRACE$calls still holds the traced vectors as doubles.
          sorts <- Filter(function(call) call$name == "sort.int", TRACE$calls)
          t_star <- if (length(sorts) > 0L) sorts[[1L]]$input
          values <- traced$result$value
          rows[[length(rows) + 1L]] <- list(
            set = set, writing = writing, B = B, seed = seed,
            x_hex = hex_list(x), nv_hex = hex(nv),
            values_hex = if (is.null(values)) NULL else hex_list(values),
            error = traced$result$error, warnings = traced$result$warnings,
            sorted_calls = length(sorts),
            nan = if (!is.null(t_star)) sum(is.nan(t_star)),
            infinite = if (!is.null(t_star)) sum(is.infinite(t_star)),
            finite = if (!is.null(t_star)) sum(is.finite(t_star)),
            t_star_hex = if (!is.null(t_star) && B <= 2000L) hex_list(t_star),
            unbalanced_traces = traced$unbalanced
          )
        }
      }
    }
  }
  probe_file(
    "probe-trimcibt",
    paste(
      "WRS2::trimcibt on the same ratings in three writings at 200, 2000 and 20000",
      "resamples: p, interval and the NaN and infinite T* it sorted"
    ),
    list(rows = rows),
    datasets = list(ratings$entry)
  )
}

# probe-bartables (reviews B5 and C3): stats::chisq.test (correct = FALSE) and
# effectsize::cramers_v (adjust TRUE and FALSE) or pearsons_c at 95%, two-sided as
# ggbarstats asks, with R's own noncentral chi-square probability at each bound in
# (0, 1): on the wide tables of the round-3 cases (2 x 20, 2 x 40 and two 2 x 61), on
# the exact zeros of those cases (a (0, 1), b (5, 4); 8, 2, 7, 8 against 0.32, 0.08,
# 0.28, 0.32; 49 counts of 2), and on two exactly independent tables of about 1e10 per
# cell given through counts, which ggbarstats is also asked to draw: a (1e10, 2e10), b
# (2e10, 4e10), and 1e9 r_i c_j with r = 3, 7, 11 and c = 2, 5, 11.
bartables_probe <- function() {
  as_matrix <- function(frame) {
    m <- stats::xtabs(n ~ x + y, data = frame)
    matrix(as.numeric(m), nrow = nrow(m), dimnames = dimnames(m))
  }
  rows_of <- function(m) lapply(seq_len(nrow(m)), function(i) as.list(unname(m[i, ])))
  test_fields <- function(r) {
    list(
      statistic_hex = hex(unname(r$statistic)), parameter = encode_doubles(unname(r$parameter)),
      p_value_hex = hex(r$p.value), expected_hex = hex_list(r$expected)
    )
  }
  es_fields <- function(r) {
    list(
      name = names(r)[[1L]], estimate_hex = hex(r[[1L]]), ci = r$CI,
      ci_low_hex = hex(r$CI_low), ci_high_hex = hex(r$CI_high)
    )
  }
  tails <- function(es, chisq, df, to_ncp) {
    lapply(list(low = es$CI_low, high = es$CI_high), function(bound) {
      if (!is.finite(bound) || bound <= 0 || bound >= 1) {
        return(NULL)
      }
      ncp <- to_ncp(bound)
      list(ncp_hex = hex(ncp), cdf_hex = hex(stats::pchisq(chisq, df, ncp = ncp)))
    })
  }
  two_way <- function(name, m) {
    n <- sum(m)
    test <- probe_call(name, stats::chisq.test(m, correct = FALSE))$value
    chisq <- unname(test$statistic)
    df <- unname(test$parameter)
    r <- nrow(m)
    k <- ncol(m)
    cramers <- lapply(c(TRUE, FALSE), function(adjust) {
      # V back to chi-square, as probe-contingency does.
      to_ncp <- function(v) {
        if (!adjust) {
          return(n * v^2 * (min(r, k) - 1))
        }
        shrunk <- min(r - (r - 1)^2 / (n - 1), k - (k - 1)^2 / (n - 1)) - 1
        n * (v^2 * shrunk + (r - 1) * (k - 1) / (n - 1))
      }
      out <- probe_attempt(effectsize::cramers_v(
        m, adjust = adjust, ci = 0.95, alternative = "two.sided"
      ))
      c(
        list(adjust = adjust),
        if (!is.null(out$value)) es_fields(out$value),
        list(
          error = out$error, warnings = out$warnings,
          tails_at_bounds = if (!is.null(out$value)) tails(out$value, chisq, df, to_ncp)
        )
      )
    })
    list(
      name = name, x_levels = as.list(rownames(m)), y_levels = as.list(colnames(m)),
      table = rows_of(m), n_hex = hex(n),
      chisq = attempt_fields(stats::chisq.test(m, correct = FALSE), test_fields),
      # Bergsma's correction as R's doubles give it: chisq / n - df / (n - 1).
      phi2_corrected_hex = hex(chisq / n - (r - 1) * (k - 1) / (n - 1)),
      cramers_v = cramers
    )
  }
  one_way <- function(name, x, p) {
    n <- sum(x)
    test <- probe_call(name, stats::chisq.test(x, p = p, correct = FALSE))$value
    chisq <- unname(test$statistic)
    df <- unname(test$parameter)
    out <- probe_attempt(effectsize::pearsons_c(x, p = p, ci = 0.95, alternative = "two.sided"))
    list(
      name = name, x = as.list(x), p_hex = hex_list(p), n_hex = hex(n),
      chisq = attempt_fields(stats::chisq.test(x, p = p, correct = FALSE), test_fields),
      pearsons_c = c(
        if (!is.null(out$value)) es_fields(out$value),
        list(
          error = out$error, warnings = out$warnings,
          tails_at_bounds = if (!is.null(out$value)) {
            tails(out$value, chisq, df, function(c) n * c^2 / (1 - c^2))
          }
        )
      )
    )
  }
  wide_sets <- lapply(
    c("edge_bar_wide20", "edge_bar_wide40", "edge_bar_wide61", "edge_bar_wide61r"), probe_dataset
  )
  names(wide_sets) <- c("edge_bar_wide20", "edge_bar_wide40", "edge_bar_wide61", "edge_bar_wide61r")
  wide <- lapply(names(wide_sets), function(name) two_way(name, as_matrix(wide_sets[[name]]$data)))
  zero_v <- probe_dataset("edge_bar_zero_v")
  exact_fit <- probe_dataset("edge_bar_exact_fit")
  equal49 <- probe_dataset("edge_bar_equal49")
  big_frames <- list(
    big_2x2 = data.frame(
      x = c("a", "a", "b", "b"), y = c("p", "q", "p", "q"), n = c(1e10, 2e10, 2e10, 4e10)
    ),
    big_3x3 = data.frame(
      x = rep(c("r1", "r2", "r3"), each = 3L), y = rep(c("c1", "c2", "c3"), times = 3L),
      n = as.vector(t(outer(c(3, 7, 11), c(2, 5, 11)))) * 1e9
    )
  )
  big <- lapply(names(big_frames), function(name) {
    frame <- big_frames[[name]]
    HIDDEN$log <- list()
    HIDDEN$values <- list()
    set.seed(CASE_SEED)
    c(
      two_way(name, as_matrix(frame)),
      list(ggbarstats = attempt_fields(
        ggstatsplot::ggbarstats(data = frame, x = x, y = y, counts = n, type = "parametric"),
        subtitle_hex
      ))
    )
  })
  # What ggbarstats prints: statsExpressions::contingency_table() as ggbarstats calls
  # it (its defaults are ggbarstats': two-sided, 95%, two digits), which also gives the
  # subtitle of edge_bar_equal49, where ggbarstats itself stops while drawing 49 levels
  # ("`n` must be a whole number between 1 and 24, not the number 49.").
  expression_row <- function(make) {
    attempt_fields(make(), function(r) {
      list(
        statistic_hex = hex(r$statistic), p_value_hex = hex(r$p.value),
        estimate_hex = hex(r$estimate), conf_low_hex = hex(r$conf.low),
        conf_high_hex = hex(r$conf.high), n_obs = r$n.obs, effectsize = r$effectsize,
        expression = deparse_text(r$expression[[1L]])
      )
    })
  }
  ratio <- c(0.32, 0.08, 0.28, 0.32)
  zeros <- c(
    list(c(
      two_way("edge_bar_zero_v", as_matrix(zero_v$data)),
      list(contingency_table = expression_row(function() {
        statsExpressions::contingency_table(data = zero_v$data, x = x, y = y, counts = n)
      }))
    )),
    list(c(
      one_way("edge_bar_exact_fit", as.numeric(exact_fit$data$n), ratio),
      list(contingency_table = expression_row(function() {
        statsExpressions::contingency_table(
          data = exact_fit$data, x = x, counts = n, ratio = ratio
        )
      }))
    )),
    list(c(
      one_way("edge_bar_equal49", as.numeric(equal49$data$n), rep(1 / 49, 49L)),
      list(
        contingency_table = expression_row(function() {
          statsExpressions::contingency_table(data = equal49$data, x = x, counts = n)
        }),
        ggbarstats = local({
          HIDDEN$log <- list()
          HIDDEN$values <- list()
          set.seed(CASE_SEED)
          attempt_fields(
            ggstatsplot::ggbarstats(data = equal49$data, x = x, counts = n, type = "parametric"),
            subtitle_hex
          )
        })
      )
    )),
    big
  )
  probe_file(
    "probe-bartables",
    paste(
      "chisq.test with cramers_v or pearsons_c and the tail at each bound: 2 x 20 to 2 x",
      "61 tables, and tables whose statistic or corrected effect size is exactly 0"
    ),
    list(wide = wide, zeros = zeros),
    datasets = c(
      unname(lapply(wide_sets, function(set) set$entry)),
      list(zero_v$entry, exact_fit$entry, equal49$entry)
    )
  )
}

# probe-round (D108): R's round(perc, d) at d = 0, 1 and 2 and the label
# paste0(round(perc, d), "%") for perc = (k / n) x 100, n = 40, 200, 1000 and 2000, at
# every k whose perc lies within 1e-9 of a rounding tie at one of those digits; then
# ggbarstats' own cell labels (extract_stats()'s descriptive_data, perc and .label) at
# digits.perc 0, 1 and 2 for one bar per such k <= n / 2 (a k, b n - k, through counts).
ROUND_N <- c(40L, 200L, 1000L, 2000L)
round_probe <- function() {
  near_tie <- function(perc, d) {
    scaled <- perc * 10^d
    abs((scaled - floor(scaled)) - 0.5) <= 1e-9
  }
  grid <- list()
  label_path <- list()
  for (n in ROUND_N) {
    k <- seq_len(n - 1L)
    perc <- (k / n) * 100
    ties <- lapply(0:2, function(d) near_tie(perc, d))
    keep <- which(Reduce(`|`, ties))
    for (i in keep) {
      grid[[length(grid) + 1L]] <- list(
        n = n, k = k[[i]], perc_hex = hex(perc[[i]]),
        tie_digits = as.list((0:2)[vapply(ties, function(t) t[[i]], logical(1L))]),
        rounded_hex = lapply(0:2, function(d) hex(round(perc[[i]], d))),
        labels = lapply(0:2, function(d) paste0(round(perc[[i]], d), "%"))
      )
    }
    bars <- k[keep][k[keep] <= n - k[keep]]
    frame <- data.frame(
      x = rep(c("a", "b"), times = length(bars)),
      y = rep(sprintf("n%04dk%04d", n, bars), each = 2L),
      n = as.vector(rbind(bars, n - bars))
    )
    for (d in 0:2) {
      HIDDEN$log <- list()
      HIDDEN$values <- list()
      set.seed(CASE_SEED)
      out <- probe_attempt(ggstatsplot::ggbarstats(
        data = frame, x = x, y = y, counts = n, type = "parametric", digits.perc = d,
        results.subtitle = FALSE, proportion.test = FALSE, bf.message = FALSE
      ))
      described <- if (!is.null(out$value)) ggstatsplot::extract_stats(out$value)$descriptive_data
      label_path[[length(label_path) + 1L]] <- list(
        n = n, digits_perc = d, bars = length(bars),
        error = out$error, warnings = out$warnings,
        cells = if (!is.null(described)) {
          lapply(seq_len(nrow(described)), function(j) {
            list(
              y = as.character(described$y[[j]]), x = as.character(described$x[[j]]),
              counts = as.integer(described$counts[[j]]), perc_hex = hex(described$perc[[j]]),
              label = described$.label[[j]]
            )
          })
        }
      )
    }
  }
  probe_file(
    "probe-round",
    paste(
      "round(perc, d) and paste0(round(perc, d), '%') at decimal ties of (k / n) x 100,",
      "and ggbarstats' own cell labels for the same k / n"
    ),
    list(grid = grid, label_path = label_path)
  )
}

# Main ------------------------------------------------------------------------------

# The probes first: they draw no random numbers that a case could see (every case
# sets its own seed), and a failing probe stops the run in minutes, not after the
# bootstrap references.
games_howell_cases <- Filter(function(case) {
  if (case$fun != "ggbetweenstats") {
    return(FALSE)
  }
  analysis <- between_analysis(read_dataset(case)$data, case$args$x, case$args$y)
  case$args$type == "parametric" && nlevels(analysis$x) > 2L
}, CASES)
PROBE_BUILDERS <- list(
  "probe-format" = format_probe,
  "probe-padjust" = padjust_probe,
  "probe-ptukey" = function() ptukey_probe(games_howell_cases),
  "probe-omega" = omega_probe,
  "probe-smm" = smm_probe,
  "probe-wrs2p" = wrs2p_probe,
  "probe-akp" = akp_probe,
  "probe-map" = map_probe,
  "probe-ranks" = ranks_probe,
  "probe-signrank" = signrank_probe,
  "probe-pairedbf" = pairedbf_probe,
  "probe-friedman" = friedman_probe,
  "probe-rmanova" = rmanova_probe,
  "probe-wrs2within" = wrs2within_probe,
  "probe-bootrules" = bootrules_probe,
  "probe-corr" = corr_probe,
  "probe-onesamplebf" = onesamplebf_probe,
  "probe-heldout" = heldout_probe,
  "probe-contingency" = contingency_probe,
  # Review round 3, last so that the probes above keep their shards.
  "probe-nearzero" = nearzero_probe,
  "probe-trimcibt" = trimcibt_probe,
  "probe-bartables" = bartables_probe,
  "probe-round" = round_probe
)

# Local debugging only: ORACLE_SUBSET, a regular expression, limits the run to the
# probes and cases whose names match it and writes no manifest, so the Python loader
# refuses the output. The oracle workflow never sets it.
SUBSET <- Sys.getenv("ORACLE_SUBSET")
if (nzchar(SUBSET)) {
  PROBE_BUILDERS <- PROBE_BUILDERS[grepl(SUBSET, names(PROBE_BUILDERS))]
  CASES <- Filter(function(case) grepl(SUBSET, case$name), CASES)
  case_names <- vapply(CASES, function(case) case$name, character(1L))
}

# Shards ---------------------------------------------------------------------------
#
# The bootstrap references take hours in one process, so oracle.yml runs the
# generator in parallel jobs: ORACLE_SHARD = "i/n" runs the i-th of n shares of the
# probes and cases and writes shard-i.json (what it ran, R's version and library)
# instead of a manifest. Cases that can share a bootstrap reference (same function,
# dataset, declared levels, type, conf.level and tr) stay in one shard, and the groups
# are spread by an estimate of their cost (one per case, 30 more for a bootstrap), the
# largest first, each to the least loaded shard. Every case and probe sets its own
# seeds, so a shard computes exactly what one process would. ORACLE_MERGE = "true"
# then runs no probe and no case: it reads the shard files the shards' outputs were
# merged with, checks that together they ran every probe and case exactly once, on
# this R and this library, removes them and writes the manifest.
SHARD <- Sys.getenv("ORACLE_SHARD")
MERGE <- identical(Sys.getenv("ORACLE_MERGE"), "true")

case_bootstraps <- function(case) {
  if (isFALSE(case$args$results.subtitle)) {
    return(FALSE)
  }
  type <- case$args$type
  if (case$fun == "ggscatterstats") {
    return(FALSE)
  }
  if (case$fun %in% BAR_FUNCTIONS) {
    return(FALSE)
  }
  if (type == "robust") {
    return(TRUE)
  }
  if (type != "nonparametric" || case$fun == "gghistostats") {
    return(FALSE)
  }
  nlevels(ANALYSES[[case$fun]](read_dataset(case)$data, case)$x) > 2L
}

# The cases of review round 3 (ROUND3_CASES) are placed after all the others, on the
# loads those leave, so that adding them moves no earlier case to another shard (the
# shard decides which case first loads a package and records its start-up message). A
# later case whose group already has a shard joins it.
assign_shards <- function(cases, n) {
  keys <- vapply(cases, function(case) {
    paste(
      case$fun, case$dataset, paste(unlist(case$factor_levels), collapse = ","),
      case$args$type, format(case_conf_level(case)), format(case_tr(case)),
      sep = "|"
    )
  }, character(1L))
  boots <- vapply(cases, case_bootstraps, logical(1L))
  later <- vapply(cases, function(case) case$name %in% ROUND3_CASE_NAMES, logical(1L))
  load <- numeric(n)
  out <- integer(length(cases))
  for (part in list(which(!later), which(later))) {
    groups <- split(part, keys[part])
    weights <- vapply(groups, function(ix) length(ix) + 30 * any(boots[ix]), numeric(1L))
    for (g in names(groups)[order(-weights, names(groups))]) {
      placed <- out[keys == g & out > 0L]
      shard <- if (length(placed) > 0L) placed[[1L]] else which.min(load)
      out[groups[[g]]] <- shard
      load[[shard]] <- load[[shard]] + weights[[g]]
    }
  }
  out
}

library_listing <- function() {
  packages <- utils::installed.packages(lib.loc = .libPaths())
  packages <- packages[!duplicated(packages[, "Package"]), , drop = FALSE]
  packages <- packages[order(packages[, "Package"]), , drop = FALSE]
  stats::setNames(as.list(unname(packages[, "Version"])), unname(packages[, "Package"]))
}

if (nzchar(SHARD) && MERGE) stop("set ORACLE_SHARD or ORACLE_MERGE, not both")
ALL_CASE_NAMES <- case_names
ALL_PROBE_FILES <- paste0(names(PROBE_BUILDERS), ".json")
if (nzchar(SHARD)) {
  parts <- as.integer(strsplit(SHARD, "/", fixed = TRUE)[[1L]])
  if (length(parts) != 2L || anyNA(parts) || parts[[1L]] < 1L || parts[[1L]] > parts[[2L]]) {
    stop("ORACLE_SHARD must be i/n with 1 <= i <= n, not ", SHARD)
  }
  shard_index <- parts[[1L]]
  shard_count <- parts[[2L]]
  CASES <- CASES[assign_shards(CASES, shard_count) == shard_index]
  case_names <- vapply(CASES, function(case) case$name, character(1L))
  PROBE_BUILDERS <- PROBE_BUILDERS[
    (seq_along(PROBE_BUILDERS) - 1L) %% shard_count + 1L == shard_index
  ]
}
if (MERGE) {
  PROBE_BUILDERS <- list()
  CASES <- list()
}

probe_files <- character()
for (probe_name in names(PROBE_BUILDERS)) {
  message("== ", probe_name)
  probe <- PROBE_BUILDERS[[probe_name]]()
  if (!identical(probe$name, probe_name)) stop(probe_name, " names itself ", probe$name)
  write_json_file(probe, file.path(out_dir, paste0(probe$name, ".json")))
  probe_files <- c(probe_files, paste0(probe$name, ".json"))
  rm(probe)
}

summary_rows <- character()
problems <- character()
for (case in CASES) {
  message("== ", case$name)
  started <- proc.time()[["elapsed"]]
  # A failure of the generator itself (not of ggstatsplot, which run_case records) is
  # reported with the other problems at the end, so one run shows every case.
  fixture <- tryCatch(run_case(case), error = function(e) e)
  if (inherits(fixture, "error")) {
    problems <- c(problems, paste0(case$name, ": generator failed: ", conditionMessage(fixture)))
    summary_rows <- c(summary_rows, sprintf(
      "| %s | generator failed: %s | | | | |",
      case$name, gsub("|", "/", conditionMessage(fixture), fixed = TRUE)
    ))
    next
  }
  write_json_file(fixture, file.path(out_dir, paste0(case$name, ".json")))
  case_problems <- outcome_problems(case, fixture)
  if (length(case_problems) > 0L) {
    problems <- c(problems, paste0(case$name, ": ", case_problems))
  }
  status <- if (!is.null(fixture$error)) paste("error:", fixture$error) else "ok"
  if (length(case_problems) > 0L) status <- paste(case_problems, collapse = "; ")
  hidden_conditions <- sum(vapply(fixture$suppressed, function(entry) {
    length(entry$warnings) + length(entry$messages) + !is.null(entry$error)
  }, numeric(1L)))
  boot <- fixture$bootstrap_reference
  summary_rows <- c(summary_rows, sprintf(
    "| %s | %s | %d | %d | %s | %.0f |",
    case$name, gsub("|", "/", status, fixed = TRUE), length(fixture$r_warnings),
    as.integer(hidden_conditions),
    if (is.null(boot)) "" else sprintf("replay matches ggstatsplot: %s", boot$replay$matches),
    proc.time()[["elapsed"]] - started
  ))
}

summary_text <- c(
  sprintf("ggstatsplot %s, R %s", PARITY_VERSION, VERSIONS$R),
  "",
  "| case | result | R warnings | conditions ggstatsplot hid | bootstrap reference | seconds |",
  "| --- | --- | --- | --- | --- | --- |",
  summary_rows
)
cat(summary_text, sep = "\n")
step_summary <- Sys.getenv("GITHUB_STEP_SUMMARY")
if (nzchar(step_summary)) cat(summary_text, sep = "\n", file = step_summary, append = TRUE)

if (length(problems) > 0L) {
  # No manifest: the Python fixture loader refuses a directory without one.
  message("Unexpected case outcomes:\n", paste(problems, collapse = "\n"))
  quit(status = 1L)
}
if (nzchar(SHARD)) {
  # What this shard ran, on which R and library; the merge checks it.
  write_json_file(
    list(
      shard = SHARD, cases = as.list(case_names), probes = as.list(probe_files),
      versions = VERSIONS, r_library = library_listing()
    ),
    file.path(out_dir, sprintf("shard-%d.json", shard_index))
  )
  message("ORACLE_SHARD is set: no manifest written")
  quit(status = 0L)
}
if (MERGE) {
  shard_paths <- list.files(out_dir, pattern = "^shard-[0-9]+\\.json$", full.names = TRUE)
  if (length(shard_paths) == 0L) stop("ORACLE_MERGE: no shard files in ", out_dir)
  shards <- lapply(shard_paths, jsonlite::fromJSON, simplifyVector = FALSE)
  merged_cases <- unlist(lapply(shards, function(shard) unlist(shard$cases)))
  merged_probes <- unlist(lapply(shards, function(shard) unlist(shard$probes)))
  counts <- unique(vapply(shards, function(shard) {
    as.integer(strsplit(shard$shard, "/", fixed = TRUE)[[1L]][[2L]])
  }, integer(1L)))
  if (length(counts) != 1L || counts != length(shards)) {
    stop("ORACLE_MERGE: ", length(shards), " shard files for shard counts ", paste(counts, collapse = ", "))
  }
  if (anyDuplicated(merged_cases) || !setequal(merged_cases, ALL_CASE_NAMES)) {
    stop("ORACLE_MERGE: the shards ran other cases than the generator lists")
  }
  if (anyDuplicated(merged_probes) || !setequal(merged_probes, ALL_PROBE_FILES)) {
    stop("ORACLE_MERGE: the shards ran other probes than the generator lists")
  }
  here <- jsonlite::fromJSON(
    jsonlite::toJSON(
      list(versions = VERSIONS, r_library = library_listing()),
      auto_unbox = TRUE, digits = NA, na = "null", null = "null"
    ),
    simplifyVector = FALSE
  )
  for (shard in shards) {
    if (!identical(shard$versions, here$versions) || !identical(shard$r_library, here$r_library)) {
      stop("ORACLE_MERGE: shard ", shard$shard, " ran on another R or library")
    }
  }
  unlink(shard_paths)
  case_names <- ALL_CASE_NAMES
  probe_files <- ALL_PROBE_FILES
}
if (nzchar(SUBSET)) {
  message("ORACLE_SUBSET is set: no manifest written")
  quit(status = 0L)
}

fixture_files <- sort(setdiff(
  list.files(out_dir, pattern = "\\.json$"), c("manifest.json", probe_files)
))
expected_files <- paste0(case_names, ".json")
if (!setequal(fixture_files, expected_files)) {
  stop(
    "output directory holds ", paste(setdiff(fixture_files, expected_files), collapse = ", "),
    "; missing ", paste(setdiff(expected_files, fixture_files), collapse = ", ")
  )
}
env_or_null <- function(name) {
  value <- Sys.getenv(name)
  if (nzchar(value)) value else NULL
}
manifest <- list(
  schema_version = SCHEMA_VERSION,
  generator = "oracle/generate.R",
  parity_target = paste("ggstatsplot", PARITY_VERSION),
  # The files that produced these fixtures; tests/oracle_fixtures.py compares them
  # with the committed ones.
  generator_files = stats::setNames(lapply(GENERATOR_FILES, file_sha256), GENERATOR_FILES),
  # The GitHub Actions run that produced the files. oracle.yml attests this manifest;
  # .github/workflows/fixtures.yml verifies the attestation and that the generator
  # at `commit` is the committed one.
  run = list(
    repository = env_or_null("GITHUB_REPOSITORY"),
    run_id = env_or_null("GITHUB_RUN_ID"),
    run_attempt = env_or_null("GITHUB_RUN_ATTEMPT"),
    commit = env_or_null("GITHUB_SHA")
  ),
  R = VERSIONS$R,
  rng_kind = VERSIONS$rng_kind,
  r_library = library_listing(),
  fixtures = stats::setNames(
    lapply(fixture_files, function(f) file_sha256(file.path(out_dir, f))),
    fixture_files
  ),
  # Probe files (see "Probes" above), hashed like the fixtures.
  probes = stats::setNames(
    lapply(sort(probe_files), function(f) file_sha256(file.path(out_dir, f))),
    sort(probe_files)
  )
)
write_json_file(manifest, file.path(out_dir, "manifest.json"))
message(sprintf(
  "wrote %d fixtures, %d probes and manifest.json to %s",
  length(fixture_files), length(probe_files), out_dir
))
