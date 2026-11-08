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
# Resamples ggbetweenstats uses for every bootstrap CI: statsExpressions 2.2.1's nboot
# default, which it passes as nboot (akp.effect, t1way) or iterations
# (rank_epsilon_squared).
GG_RESAMPLES <- 100L
# statsExpressions passes alpha = 1.0 - conf.level to WRS2. In floating point that is
# 0.050000000000000044, not 0.05, which can change which bootstrap order statistic a
# CI uses (alpha / 2 * 100 = 2.500000000000002). Candidates try both spellings.
CONF_LEVEL <- 0.95
ALPHAS <- list(
  list(text = "1 - 0.95", value = 1.0 - CONF_LEVEL),
  list(text = "0.05", value = 0.05)
)

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
# optionally declared factor levels (to create empty levels). `expect_error = TRUE`
# records a case where ggstatsplot itself is expected to fail; `expect_hidden_errors`
# names the components (subtitle, caption) whose error ggstatsplot is expected to
# swallow (see "Conditions ggstatsplot hides" below). Any other outcome stops the run.

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
      expect_error = FALSE,
      expect_hidden_errors = character()
    )
  })
}

# One parametric case, as a list of one case so that c() can combine it with the
# others. ggbetweenstats arguments other than x, y and type go in `extra_args`;
# `label` names the case.
parametric_case <- function(dataset, x, y, label = dataset, extra_args = list(),
                            factor_levels = NULL) {
  between_cases(
    dataset, x = x, y = y, label = label, types = "parametric",
    extra_args = extra_args, factor_levels = factor_levels
  )
}

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
  # Two values per group: Welch df between 1 and 2 per pair.
  parametric_case("edge_n2", x = "group", y = "value"),
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
  # Which pairs get brackets, and the axis label that says so.
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
  )
)

case_names <- vapply(CASES, function(case) case$name, character(1L))
if (anyDuplicated(case_names)) stop("duplicate case names: ", case_names[duplicated(case_names)])

# Arguments whose consequences the bookkeeping below knows (which hidden calls
# ggbetweenstats makes, which outputs it draws). Fail closed on any other.
KNOWN_ARGS <- c(
  "x", "y", "type", "alternative", "conf.level", "digits", "p.adjust.method",
  "pairwise.display", "results.subtitle", "bf.message", "bf.prior"
)
for (case in CASES) {
  unknown <- setdiff(names(case$args), KNOWN_ARGS)
  if (length(unknown) > 0L) {
    stop(case$name, ": generate.R has no bookkeeping for ", paste(unknown, collapse = ", "))
  }
  for (flag in c("results.subtitle", "bf.message")) {
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

log_hidden <- function(component, error, warnings, messages) {
  HIDDEN$log[[length(HIDDEN$log) + 1L]] <- list(
    component = component,
    error = error,
    warnings = as.list(warnings),
    messages = as.list(messages)
  )
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
  log_hidden(".eval_f", error, warnings, messages)
  value
}

# Records warnings and passes them on to ggstatsplot's own suppressWarnings(). The
# original call is evaluated as written, so non-standard evaluation is unchanged.
recording_wrapper <- function(component, f) {
  # Force both now: the caller's loop variable changes before the wrapper runs.
  force(component)
  force(f)
  wrapper <- function() {
    call <- sys.call()
    call[[1L]] <- f
    warnings <- character()
    value <- withCallingHandlers(
      eval(call, parent.frame()),
      warning = function(w) warnings <<- c(warnings, conditionMessage(w))
    )
    log_hidden(component, NULL, warnings, character())
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

# The `.eval_f` calls ggbetweenstats 1.1.4 makes, in order: with results.subtitle
# (default TRUE) it computes the subtitle, and then the Bayes caption when
# bf.condition (type "parametric") and bf.message (default TRUE) both hold; with
# results.subtitle = FALSE it computes neither (ggbetweenstats() and
# .subtitle_caption() in ggstatsplot's R/ggbetweenstats.R and R/0-utils.R).
expected_eval_f <- function(case) {
  if (isFALSE(case$args$results.subtitle)) {
    return(character())
  }
  c("subtitle", if (case$args$type == "parametric" && !isFALSE(case$args$bf.message)) "caption")
}

# Every hidden call ggbetweenstats makes, in order: the `.eval_f` calls above, then
# centrality_description() (always, also when the labels are not drawn), then
# pairwise_comparisons() for three or more groups unless pairwise.display = "none".
expected_hidden_calls <- function(case, levels) {
  display <- case$args$pairwise.display
  pairwise <- length(levels) > 2L && !identical(display, "none")
  c(expected_eval_f(case), "centrality_description", if (pairwise) "pairwise_comparisons")
}

# Name the `.eval_f` entries by their order. Any other number of calls than the
# arguments imply stops the run; a call that failed may have stopped early, so its
# entries are named only as far as they go (outcome_problems() then reports the
# error unless the case expects it).
name_hidden <- function(case, log, failed = FALSE) {
  eval_f <- which(vapply(log, function(entry) entry$component == ".eval_f", logical(1L)))
  expected <- expected_eval_f(case)
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
plot_layers <- function(plot) {
  # ggplot2 4.x plots are S7 objects; fall back to `$` for older S3 plots.
  layers <- tryCatch(plot@layers, error = function(e) plot$layers)
  layer_names <- names(layers)
  if (is.null(layer_names)) layer_names <- rep("", length(layers))
  unname(Map(function(layer, name, index) {
    data <- layer$data
    geom <- class(layer$geom)[[1L]]
    signif <- geom == "GeomSignif"
    list(
      name = name,
      geom = geom,
      data = if (is.data.frame(data)) frame_rows(data) else NULL,
      # Significance brackets keep the drawn pairs (`comparisons`) and their labels
      # (`annotations`) in the layer's parameters, not in its data; `built` is the
      # layer as ggplot2 computes it for drawing.
      params = if (signif) {
        list(
          stat = to_json_value(layer$stat_params),
          geom = to_json_value(layer$geom_params),
          aes = to_json_value(layer$aes_params)
        )
      } else {
        NULL
      },
      built = if (signif) frame_rows(ggplot2::layer_data(plot, index)) else NULL
    )
  }, layers, layer_names, seq_along(layers)))
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
# 1. Replay ggstatsplot's own computation. Inside ggbetweenstats() the effect-size
#    call is the first to draw random numbers (statsExpressions 2.2.1 source), so
#    set.seed(CASE_SEED) followed by the same call with GG_RESAMPLES resamples, on
#    the rows in statsExpressions' order, must reproduce ggstatsplot's estimate and
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

akp_candidate <- function(analysis, eqvar, alpha) {
  list(
    call = sprintf(
      "WRS2::akp.effect(y ~ x, data, EQVAR = %s, tr = 0.2, nboot = R, alpha = %s)",
      eqvar, alpha$text
    ),
    run = function(resamples) {
      r <- WRS2::akp.effect(
        y ~ x,
        data = analysis, EQVAR = eqvar, tr = 0.2, nboot = resamples, alpha = alpha$value
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
      list(
        values = c(effect[[1L]], as.numeric(pair)),
        raw = list(AKPeffect = r$AKPeffect, AKPci = r$AKPci, AKPci_dim = dim(r$AKPci))
      )
    }
  )
}

bootstrap_candidates <- function(case, analysis, subtitle_row) {
  # Fail closed: each function needs its own replay (paired row order, other effect
  # sizes) before its cases are added.
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
  if (length(case$expect_hidden_errors) > 0L) {
    # A swallowed subtitle leaves nothing to replay against (STATUS.md).
    stop(case$name, " declares a swallowed error; skip its bootstrap reference first")
  }
  for (argument in c("conf.level", "tr")) {
    if (!is.null(case$args[[argument]])) {
      stop(case$name, " sets ", argument, "; the replays assume its default")
    }
  }
  if (type == "robust" && k == 2L) {
    # statsExpressions' own call first: EQVAR = FALSE, alpha = 1 - conf.level.
    return(c(
      lapply(ALPHAS, function(alpha) akp_candidate(analysis, FALSE, alpha)),
      lapply(ALPHAS, function(alpha) akp_candidate(analysis, TRUE, alpha))
    ))
  }
  if (type == "robust") {
    return(lapply(ALPHAS, function(alpha) {
      list(
        call = sprintf(
          "WRS2::t1way(y ~ x, data, tr = 0.2, alpha = %s, nboot = R)", alpha$text
        ),
        run = function(resamples) {
          r <- WRS2::t1way(
            y ~ x,
            data = analysis, tr = 0.2, alpha = alpha$value, nboot = resamples
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
        "effectsize::rank_epsilon_squared(y ~ x, data, ci = 0.95, alternative = \"%s\", iterations = R)",
        alternative
      ),
      run = function(resamples) {
        r <- effectsize::rank_epsilon_squared(
          y ~ x,
          data = analysis, ci = 0.95, alternative = alternative,
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

bootstrap_reference <- function(case, analysis, stats) {
  subtitle_row <- stats$subtitle_data
  ordered <- statsexpressions_rows(analysis)
  candidates <- bootstrap_candidates(case, ordered, subtitle_row)
  if (length(candidates) == 0L) {
    return(NULL)
  }
  gg_values <- unname(c(subtitle_row$estimate, subtitle_row$conf.low, subtitle_row$conf.high))

  tried <- list()
  chosen <- NULL
  for (candidate in candidates) {
    replay <- run_or_stop(case, "ggstatsplot replay", {
      set.seed(CASE_SEED)
      candidate$run(GG_RESAMPLES)
    })
    values <- unname(replay$value$values)
    matches <- isTRUE(all.equal(values, gg_values, tolerance = 1e-10))
    tried[[length(tried) + 1L]] <- list(
      call = candidate$call, values = as.list(values), matches = matches
    )
    if (matches && is.null(chosen)) chosen <- candidate
  }
  replayed <- !is.null(chosen)
  if (!replayed) chosen <- candidates[[1L]]

  reference <- run_or_stop(case, "bootstrap reference", {
    set.seed(BOOT_SEED)
    chosen$run(BOOT_REFERENCE_RESAMPLES)
  })
  replicates <- run_or_stop(case, "bootstrap replicates", t(vapply(
    seq_len(BOOT_REPLICATES),
    function(i) {
      set.seed(BOOT_SEED + i)
      unname(chosen$run(BOOT_REPLICATE_RESAMPLES)$values)
    },
    numeric(3L)
  )))
  ref <- unname(reference$value$values)
  reps <- replicates$value
  list(
    quantity = "effect size confidence interval",
    call = chosen$call,
    data_order = "rows ordered by the levels of x, as statsExpressions passes them",
    conf_level = 0.95,
    ggstatsplot = list(
      estimate = gg_values[[1L]],
      conf_low = gg_values[[2L]],
      conf_high = gg_values[[3L]],
      resamples = GG_RESAMPLES,
      seed = CASE_SEED
    ),
    replay = list(matches = replayed, candidates = tried),
    seed = BOOT_SEED,
    resamples = BOOT_REFERENCE_RESAMPLES,
    estimate = ref[[1L]],
    conf_low = ref[[2L]],
    conf_high = ref[[3L]],
    raw = reference$value$raw,
    replicates = list(
      count = BOOT_REPLICATES,
      resamples = BOOT_REPLICATE_RESAMPLES,
      first_seed = BOOT_SEED + 1L,
      estimate_mean = mean(reps[, 1L]),
      estimate_sd = stats::sd(reps[, 1L]),
      conf_low_mean = mean(reps[, 2L]),
      conf_low_sd = stats::sd(reps[, 2L]),
      conf_high_mean = mean(reps[, 3L]),
      conf_high_sd = stats::sd(reps[, 3L])
    ),
    r_warnings = as.list(unique(c(reference$warnings, replicates$warnings)))
  )
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

# Which rows each function analyzes. Fail closed: within-subjects designs keep only
# complete subjects (statsExpressions::complete_subjects), one-sample designs have no
# x, and so on, so every other function needs its own entry before its cases.
ANALYSES <- list(ggbetweenstats = between_analysis)
for (case in CASES) {
  if (is.null(ANALYSES[[case$fun]])) {
    stop("generate.R has no analysis bookkeeping for ", case$fun, "; add it first")
  }
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
  fn <- call("::", as.name("ggstatsplot"), as.name(case$fun))
  cl <- as.call(c(list(fn, data = quote(data)), call_args))
  env <- environment()

  HIDDEN$log <- list()
  set.seed(CASE_SEED)
  outcome <- capture({
    plot <- eval(cl, envir = env)
    list(
      stats = ggstatsplot::extract_stats(plot),
      subtitle = ggstatsplot::extract_subtitle(plot),
      caption = ggstatsplot::extract_caption(plot),
      layers = plot_layers(plot)
    )
  })
  failed <- inherits(outcome$value, "error")
  hidden <- name_hidden(case, HIDDEN$log, failed)

  analysis <- ANALYSES[[case$fun]](data, case$args$x, case$args$y)
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
    analysis_levels = as.list(levels(analysis$x)),
    n_per_level = as.list(table(analysis$x)),
    error = if (failed) conditionMessage(outcome$value) else NULL,
    r_warnings = as.list(outcome$warnings),
    r_messages = as.list(outcome$messages),
    suppressed = hidden,
    expected_hidden_errors = as.list(case$expect_hidden_errors),
    subtitle = if (failed) NULL else to_json_value(outcome$value$subtitle),
    caption = if (failed) NULL else to_json_value(outcome$value$caption),
    stats = if (failed) NULL else to_json_value(outcome$value$stats),
    plot_layers = if (failed) NULL else outcome$value$layers,
    bootstrap_reference = if (failed) NULL else bootstrap_reference(case, analysis, outcome$value$stats),
    versions = VERSIONS
  )
  fixture
}

# Everything about a finished case that differs from what the case declares.
outcome_problems <- function(case, fixture) {
  errored <- !is.null(fixture$error)
  if (errored != isTRUE(case$expect_error)) {
    return(if (errored) paste("error:", fixture$error) else "expected an error, got none")
  }
  if (errored) {
    return(character())
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
  expected_calls <- expected_hidden_calls(case, unlist(fixture$analysis_levels))
  if (!identical(unname(calls), expected_calls)) {
    out <- c(out, sprintf(
      "hidden calls %s; expected %s",
      paste(calls, collapse = ", "), paste(expected_calls, collapse = ", ")
    ))
  }
  # ggbetweenstats draws the subtitle unless results.subtitle = FALSE, and the Bayes
  # caption only for type "parametric" with bf.message (ggstatsplot's bf.condition),
  # never without the subtitle; the tables behind them follow the same rule.
  evaluated <- expected_eval_f(case)
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
  # Pairwise comparisons exist exactly when ggbetweenstats computes them.
  wants_pairwise <- "pairwise_comparisons" %in% expected_calls
  has_pairwise <- length(fixture$stats$pairwise_comparisons_data) > 0L
  if (wants_pairwise != has_pairwise) {
    out <- c(out, if (wants_pairwise) "no pairwise comparisons" else "unexpected pairwise comparisons")
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

# Main ------------------------------------------------------------------------------

# The probes first: they draw no random numbers that a case could see (every case
# sets its own seed), and a failing probe stops the run in minutes, not after the
# bootstrap references.
games_howell_cases <- Filter(function(case) {
  analysis <- between_analysis(read_dataset(case)$data, case$args$x, case$args$y)
  case$args$type == "parametric" && nlevels(analysis$x) > 2L
}, CASES)
PROBES <- list(
  format_probe(), padjust_probe(), ptukey_probe(games_howell_cases), omega_probe()
)
for (probe in PROBES) {
  message("== ", probe$name)
  write_json_file(probe, file.path(out_dir, paste0(probe$name, ".json")))
}
probe_files <- vapply(PROBES, function(probe) paste0(probe$name, ".json"), character(1L))

summary_rows <- character()
problems <- character()
for (case in CASES) {
  message("== ", case$name)
  fixture <- run_case(case)
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
    "| %s | %s | %d | %d | %s |",
    case$name, gsub("|", "/", status, fixed = TRUE), length(fixture$r_warnings),
    as.integer(hidden_conditions),
    if (is.null(boot)) "" else sprintf("replay matches ggstatsplot: %s", boot$replay$matches)
  ))
}

summary_text <- c(
  sprintf("ggstatsplot %s, R %s", PARITY_VERSION, VERSIONS$R),
  "",
  "| case | result | R warnings | conditions ggstatsplot hid | bootstrap reference |",
  "| --- | --- | --- | --- | --- |",
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
library_packages <- utils::installed.packages(lib.loc = .libPaths())
library_packages <- library_packages[!duplicated(library_packages[, "Package"]), , drop = FALSE]
library_packages <- library_packages[order(library_packages[, "Package"]), , drop = FALSE]
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
  r_library = stats::setNames(
    as.list(unname(library_packages[, "Version"])),
    unname(library_packages[, "Package"])
  ),
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
