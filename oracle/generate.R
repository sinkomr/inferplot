#!/usr/bin/env Rscript
# inferplot R oracle: run ggstatsplot 1.1.4 and write one JSON fixture per case.
#
# Usage: Rscript oracle/generate.R <datasets-dir> <output-dir>
#
# Runs in GitHub Actions only (.github/workflows/oracle.yml), which pins R 4.5 and a
# dated Posit Package Manager CRAN snapshot and checks every installed version. The
# output is committed to tests/fixtures/oracle/ after review and never edited by hand:
# manifest.json holds the SHA-256 of every fixture, and each fixture holds the
# SHA-256 of the CSV it read. docs/design/SPEC.md section 9 describes the schema.
#
# This script only calls R packages; none of their code is copied into inferplot.

suppressPackageStartupMessages({
  library(ggstatsplot)
  library(jsonlite)
  library(digest)
})

PARITY_VERSION <- "1.1.4"
SCHEMA_VERSION <- 1L
CASE_SEED <- 20261107L # set before every ggstatsplot call
BOOT_SEED <- 1L # reference run; replicate i uses BOOT_SEED + i
BOOT_REFERENCE_RESAMPLES <- 20000L
BOOT_REPLICATE_RESAMPLES <- 2000L # inferplot's resample count (SPEC.md section 5)
BOOT_REPLICATES <- 50L
# Resamples ggbetweenstats uses for every bootstrap CI: statsExpressions 2.2.1's nboot
# default, which it passes as nboot (akp.effect, t1way) or iterations
# (rank_epsilon_squared).
GG_RESAMPLES <- 100L

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
# records a case where ggstatsplot itself is expected to fail.

TYPES <- c("parametric", "nonparametric", "robust", "bayes")

between_cases <- function(dataset, x, y, label = dataset, factor_levels = NULL) {
  lapply(TYPES, function(type) {
    list(
      name = paste("ggbetweenstats", label, type, sep = "-"),
      fun = "ggbetweenstats",
      dataset = dataset,
      factor_levels = factor_levels,
      args = list(x = x, y = y, type = type),
      columns = c("x", "y"),
      expect_error = FALSE
    )
  })
}

CASES <- c(
  between_cases("ToothGrowth", x = "supp", y = "len"), # k = 2, n = 30 + 30
  between_cases("PlantGrowth", x = "group", y = "weight"), # k = 3, n = 10 each
  between_cases("mtcars", x = "cyl", y = "mpg") # k = 3, n = 11/7/14, ties
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

akp_candidate <- function(analysis, eqvar) {
  list(
    call = sprintf(
      "WRS2::akp.effect(y ~ x, data, EQVAR = %s, tr = 0.2, nboot = R, alpha = 0.05)", eqvar
    ),
    run = function(resamples) {
      r <- WRS2::akp.effect(
        y ~ x,
        data = analysis, EQVAR = eqvar, tr = 0.2, nboot = resamples, alpha = 0.05
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
  if (case$fun != "ggbetweenstats") {
    return(list())
  }
  k <- nlevels(analysis$x)
  if (case$args$type == "robust" && k == 2L) {
    return(list(akp_candidate(analysis, TRUE), akp_candidate(analysis, FALSE)))
  }
  if (case$args$type == "robust" && k > 2L) {
    return(list(list(
      call = "WRS2::t1way(y ~ x, data, tr = 0.2, alpha = 0.05, nboot = R)",
      run = function(resamples) {
        r <- WRS2::t1way(y ~ x, data = analysis, tr = 0.2, alpha = 0.05, nboot = resamples)
        list(
          values = c(r$effsize, r$effsize_ci),
          raw = list(effsize = r$effsize, effsize_ci = r$effsize_ci)
        )
      }
    )))
  }
  if (case$args$type == "nonparametric" && k > 2L) {
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
  list()
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

run_case <- function(case) {
  dataset <- read_dataset(case)
  data <- dataset$data
  call_args <- case$args
  for (name in case$columns) call_args[[name]] <- as.name(call_args[[name]])
  fn <- call("::", as.name("ggstatsplot"), as.name(case$fun))
  cl <- as.call(c(list(fn, data = quote(data)), call_args))
  env <- environment()

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

  analysis <- between_analysis(data, case$args$x, case$args$y)
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
    subtitle = if (failed) NULL else to_json_value(outcome$value$subtitle),
    caption = if (failed) NULL else to_json_value(outcome$value$caption),
    stats = if (failed) NULL else to_json_value(outcome$value$stats),
    plot_layers = if (failed) NULL else outcome$value$layers,
    bootstrap_reference = if (failed) NULL else bootstrap_reference(case, analysis, outcome$value$stats),
    versions = VERSIONS
  )
  fixture
}

# Main ------------------------------------------------------------------------------

summary_rows <- character()
problems <- character()
for (case in CASES) {
  message("== ", case$name)
  fixture <- run_case(case)
  write_json_file(fixture, file.path(out_dir, paste0(case$name, ".json")))
  errored <- !is.null(fixture$error)
  status <- if (errored) paste("error:", fixture$error) else "ok"
  if (errored != isTRUE(case$expect_error)) {
    problems <- c(problems, sprintf("%s: %s", case$name, status))
  }
  boot <- fixture$bootstrap_reference
  summary_rows <- c(summary_rows, sprintf(
    "| %s | %s | %d | %s |",
    case$name, gsub("|", "/", status, fixed = TRUE), length(fixture$r_warnings),
    if (is.null(boot)) "" else sprintf("replay matches ggstatsplot: %s", boot$replay$matches)
  ))
}

summary_text <- c(
  sprintf("ggstatsplot %s, R %s", PARITY_VERSION, VERSIONS$R),
  "",
  "| case | result | R warnings | bootstrap reference |",
  "| --- | --- | --- | --- |",
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
  # The GitHub Actions run that produced the files; the fixture provenance workflow
  # downloads this run's artifact and compares it byte for byte with the commit.
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
