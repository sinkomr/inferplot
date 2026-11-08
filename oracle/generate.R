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
# holds the SHA-256 of the CSV it read. docs/design/SPEC.md section 9 describes the
# schema.
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
  between_cases("bugs_long", x = "condition", y = "desire")
)

case_names <- vapply(CASES, function(case) case$name, character(1L))
if (anyDuplicated(case_names)) stop("duplicate case names: ", case_names[duplicated(case_names)])

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

# ggbetweenstats calls `.eval_f` for the subtitle and then, for a parametric test with
# bf.message, for the Bayes caption; name those entries by their order.
name_hidden <- function(case, log) {
  eval_f <- which(vapply(log, function(entry) entry$component == ".eval_f", logical(1L)))
  if (length(eval_f) > 2L) {
    stop(case$name, ": ", length(eval_f), " `.eval_f` calls; expected subtitle and caption")
  }
  for (i in seq_along(eval_f)) log[[eval_f[[i]]]]$component <- c("subtitle", "caption")[[i]]
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
  for (argument in c("conf.level", "tr")) {
    if (!is.null(case$args[[argument]])) {
      stop(case$name, " sets ", argument, "; the replays assume its default")
    }
  }
  k <- nlevels(analysis$x)
  type <- case$args$type
  # statsExpressions 2.2.1 bootstraps the effect-size CI of exactly these
  # ggbetweenstats designs: AKP (robust, two groups), t1way's explanatory effect size
  # (robust, three or more) and rank epsilon squared (nonparametric, three or more).
  if (type %in% c("parametric", "bayes") || (type == "nonparametric" && k == 2L)) {
    return(list())
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
  hidden <- name_hidden(case, HIDDEN$log)

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
  # ggbetweenstats draws the subtitle unless results.subtitle = FALSE, and the Bayes
  # caption only for type "parametric" with bf.message (ggstatsplot's bf.condition).
  wants_subtitle <- !isFALSE(case$args$results.subtitle)
  wants_caption <- wants_subtitle && case$args$type == "parametric" &&
    !isFALSE(case$args$bf.message)
  has_subtitle <- !is.null(fixture$subtitle)
  has_caption <- !is.null(fixture$caption)
  if ((wants_subtitle != has_subtitle) && !("subtitle" %in% failed_parts)) {
    out <- c(out, if (wants_subtitle) "no subtitle" else "unexpected subtitle")
  }
  if ((wants_caption != has_caption) && !("caption" %in% failed_parts)) {
    out <- c(out, if (wants_caption) "no Bayes caption" else "unexpected caption")
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

# Main ------------------------------------------------------------------------------

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

fixture_files <- sort(setdiff(list.files(out_dir, pattern = "\\.json$"), "manifest.json"))
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
  )
)
write_json_file(manifest, file.path(out_dir, "manifest.json"))
message(sprintf("wrote %d fixtures and manifest.json to %s", length(fixture_files), out_dir))
