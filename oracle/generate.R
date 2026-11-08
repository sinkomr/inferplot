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

VERSIONS <- list(
  R = paste(R.version$major, R.version$minor, sep = "."),
  platform = R.version$platform,
  os = utils::sessionInfo()$running,
  collate = Sys.getlocale("LC_COLLATE"),
  repos = as.list(getOption("repos")),
  packages = stats::setNames(
    lapply(RECORDED_PACKAGES, function(p) as.character(packageVersion(p))),
    RECORDED_PACKAGES
  )
)

# Bootstrap references --------------------------------------------------------------
#
# For effect sizes whose CI ggstatsplot bootstraps, call the underlying function
# directly with BOOT_REFERENCE_RESAMPLES resamples for a precise reference, then
# BOOT_REPLICATES times with BOOT_REPLICATE_RESAMPLES resamples to measure the Monte
# Carlo spread of a CI computed the way inferplot computes it.

bootstrap_spec <- function(case, analysis, subtitle_row) {
  if (case$fun != "ggbetweenstats") {
    return(NULL)
  }
  k <- nlevels(analysis$x)
  if (case$args$type == "robust" && k == 2L) {
    return(list(
      call = "WRS2::akp.effect(y ~ x, data, EQVAR = TRUE, tr = 0.2, nboot = R, alpha = 0.05)",
      run = function(resamples) {
        r <- WRS2::akp.effect(
          y ~ x,
          data = analysis, EQVAR = TRUE, tr = 0.2, nboot = resamples, alpha = 0.05
        )
        c(r$AKPeffect, r$AKPci)
      }
    ))
  }
  if (case$args$type == "robust" && k > 2L) {
    return(list(
      call = "WRS2::t1way(y ~ x, data, tr = 0.2, alpha = 0.05, nboot = R)",
      run = function(resamples) {
        r <- WRS2::t1way(y ~ x, data = analysis, tr = 0.2, alpha = 0.05, nboot = resamples)
        c(r$effsize, r$effsize_ci)
      }
    ))
  }
  if (case$args$type == "nonparametric" && k > 2L) {
    # Match the sidedness ggstatsplot reported: an upper bound of exactly 1 means a
    # one-sided ("greater") interval.
    alternative <- if (isTRUE(subtitle_row$conf.high == 1)) "greater" else "two.sided"
    return(list(
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
        c(r[[1L]], r$CI_low, r$CI_high)
      }
    ))
  }
  NULL
}

bootstrap_reference <- function(case, analysis, stats) {
  subtitle_row <- stats$subtitle_data
  spec <- bootstrap_spec(case, analysis, subtitle_row)
  if (is.null(spec)) {
    return(NULL)
  }
  reference <- capture({
    set.seed(BOOT_SEED)
    spec$run(BOOT_REFERENCE_RESAMPLES)
  })
  if (inherits(reference$value, "error")) {
    stop("bootstrap reference failed for ", case$name, ": ", conditionMessage(reference$value))
  }
  replicates <- capture(t(vapply(seq_len(BOOT_REPLICATES), function(i) {
    set.seed(BOOT_SEED + i)
    spec$run(BOOT_REPLICATE_RESAMPLES)
  }, numeric(3L))))
  if (inherits(replicates$value, "error")) {
    stop("bootstrap replicates failed for ", case$name, ": ", conditionMessage(replicates$value))
  }
  ref <- unname(reference$value)
  reps <- replicates$value
  gg_estimate <- unname(subtitle_row$estimate)
  list(
    quantity = "effect size confidence interval",
    call = spec$call,
    conf_level = 0.95,
    seed = BOOT_SEED,
    resamples = BOOT_REFERENCE_RESAMPLES,
    estimate = ref[[1L]],
    conf_low = ref[[2L]],
    conf_high = ref[[3L]],
    ggstatsplot = list(
      estimate = gg_estimate,
      conf_low = unname(subtitle_row$conf.low),
      conf_high = unname(subtitle_row$conf.high)
    ),
    estimate_matches_ggstatsplot = isTRUE(all.equal(ref[[1L]], gg_estimate, tolerance = 1e-8)),
    replicates = list(
      count = BOOT_REPLICATES,
      resamples = BOOT_REPLICATE_RESAMPLES,
      first_seed = BOOT_SEED + 1L,
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
      caption = ggstatsplot::extract_caption(plot)
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
    if (is.null(boot)) "" else sprintf("estimate matches: %s", boot$estimate_matches_ggstatsplot)
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
manifest <- list(
  schema_version = SCHEMA_VERSION,
  generator = "oracle/generate.R",
  parity_target = paste("ggstatsplot", PARITY_VERSION),
  fixtures = stats::setNames(
    lapply(fixture_files, function(f) file_sha256(file.path(out_dir, f))),
    fixture_files
  )
)
write_json_file(manifest, file.path(out_dir, "manifest.json"))
message(sprintf("wrote %d fixtures and manifest.json to %s", length(fixture_files), out_dir))
