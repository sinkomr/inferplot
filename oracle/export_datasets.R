#!/usr/bin/env Rscript
# Export the R-provided oracle datasets to CSV.
#
# Usage: Rscript oracle/export_datasets.R <output-dir>
#
# The CSV files are committed under oracle/datasets/. The oracle workflow re-runs
# this script before generating fixtures and fails if a committed file changed, so
# the R generator and the Python tests always read identical bytes. Files named
# edge_*.csv are written for this project by hand and are not touched here, except
# the within-subject edge datasets edge_within*.csv, which this script writes from
# R's seeded random number generator (see "Within-subject edge datasets" below).
#
# Format: UTF-8, LF line endings, header row, strings quoted, missing values as
# empty fields, numbers as R's write.csv prints them (up to 15 significant digits).
# Factor columns are written as their labels; level order is not stored in the CSV.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1L) args[[1L]] else "oracle/datasets"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

package_data <- function(name, package) {
  env <- new.env()
  utils::data(list = name, package = package, envir = env)
  as.data.frame(get(name, envir = env), stringsAsFactors = FALSE)
}

write_dataset <- function(df, name) {
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  df[] <- lapply(df, function(col) if (is.factor(col)) as.character(col) else col)
  rownames(df) <- NULL
  path <- file.path(out_dir, paste0(name, ".csv"))
  utils::write.csv(df, path, row.names = FALSE, na = "", fileEncoding = "UTF-8")
  message(sprintf("wrote %s (%d rows, %d columns)", path, nrow(df), ncol(df)))
}

mtcars_df <- package_data("mtcars", "datasets")
mtcars_df <- cbind(model = rownames(mtcars_df), mtcars_df)
write_dataset(mtcars_df, "mtcars")

for (name in c("iris", "ToothGrowth", "PlantGrowth", "morley", "sleep", "InsectSprays")) {
  write_dataset(package_data(name, "datasets"), name)
}

for (name in c("bugs_long", "Titanic_full")) {
  write_dataset(package_data(name, "ggstatsplot"), name)
}

# Two groups (versicolor, virginica) of iris: Yuen p-values of about 1e-15 for the
# petal measurements, where 1 - pt and the exact upper tail differ visibly.
iris_df <- package_data("iris", "datasets")
write_dataset(iris_df[iris_df$Species != "setosa", ], "iris_no_setosa")

# Within-subject edge datasets (Phase 3) --------------------------------------------
#
# Long format, one row per subject and condition, with an explicit `subject` column:
# inferplot pairs observations by subject, never by row order. Each dataset sets its
# own seed, so every run writes the same bytes (R 4.5.3's default generators:
# Mersenne-Twister, Inversion, Rejection); values are rounded to two decimals.
# edge_within_missing.csv (Phase 0) is hand-written and is not touched here.

# A subjects x conditions matrix as long rows, subject by subject.
within_long <- function(values, subjects, conditions) {
  stopifnot(identical(dim(values), c(length(subjects), length(conditions))))
  data.frame(
    subject = rep(subjects, each = length(conditions)),
    condition = rep(conditions, times = length(subjects)),
    value = as.vector(t(round(values, 2L))),
    stringsAsFactors = FALSE
  )
}

subject_ids <- function(n) sprintf("s%02d", seq_len(n))

# Two conditions (pre, post) ---------------------------------------------------------

# Five subjects: paired t at 4 df, Wilcoxon on five differences, Yuen with one value
# trimmed from each end.
set.seed(3001L)
pre <- rnorm(5L, 20, 4)
write_dataset(
  within_long(cbind(pre, pre + rnorm(5L, 1.5, 1)), subject_ids(5L), c("pre", "post")),
  "edge_within2_small"
)

# Integer scores whose differences have two zeros and tied absolute values: Wilcoxon's
# zero handling and tie correction, and the paired rank-biserial on ties.
set.seed(3002L)
pre <- sample(2:6, 16L, replace = TRUE)
differences <- c(0, 0, 1, 1, 1, -1, -1, 2, 2, 2, -2, 3, 3, 1, 4, -3)
write_dataset(
  within_long(cbind(pre, pre + differences), subject_ids(16L), c("pre", "post")),
  "edge_within2_ties"
)

# Every value negative.
set.seed(3003L)
pre <- -abs(rnorm(14L, 30, 4))
write_dataset(
  within_long(cbind(pre, pre - abs(rnorm(14L, 1.5, 1.5))), subject_ids(14L), c("pre", "post")),
  "edge_within2_negative"
)

# Two subjects with extreme differences (+40 and -35).
set.seed(3004L)
pre <- rnorm(20L, 50, 5)
post <- pre + rnorm(20L, 1, 2)
post[c(4L, 15L)] <- post[c(4L, 15L)] + c(40, -35)
within2_outliers <- within_long(cbind(pre, post), subject_ids(20L), c("pre", "post"))
write_dataset(within2_outliers, "edge_within2_outliers")

# Differences centred near zero: paired t of about 0.3.
set.seed(3005L)
pre <- rnorm(20L, 10, 2)
noise <- rnorm(20L, 0, 1.5)
write_dataset(
  within_long(cbind(pre, pre + noise - mean(noise) + 0.1), subject_ids(20L), c("pre", "post")),
  "edge_within2_null"
)

# 1,500 subjects: n with a big mark, df above 1,000.
set.seed(3006L)
pre <- rnorm(1500L, 100, 15)
write_dataset(
  within_long(cbind(pre, pre + rnorm(1500L, 0.8, 6)), sprintf("s%04d", 1:1500), c("pre", "post")),
  "edge_within2_large_n"
)

# Incomplete subjects: s3 has no post row, s7's pre value is missing, a post row has
# no subject and an extra row of s14 has no condition. Subject ids are not padded,
# so text order (s1, s10, ..., s2) differs from numeric order.
set.seed(3007L)
pre <- rnorm(14L, 40, 6)
missing2 <- within_long(cbind(pre, pre + rnorm(14L, 2, 3)), paste0("s", 1:14), c("pre", "post"))
missing2 <- missing2[!(missing2$subject == "s3" & missing2$condition == "post"), ]
missing2$value[missing2$subject == "s7" & missing2$condition == "pre"] <- NA
missing2 <- rbind(
  missing2,
  data.frame(subject = NA, condition = "post", value = 41.5),
  data.frame(subject = "s14", condition = NA, value = 39.25)
)
write_dataset(missing2, "edge_within2_missing")

# edge_within2_outliers with its rows in a random order: subjects in a different
# order within each condition. The results must equal edge_within2_outliers'.
set.seed(3008L)
write_dataset(
  within2_outliers[sample(nrow(within2_outliers)), ],
  "edge_within2_shuffled"
)

# Three or more conditions ------------------------------------------------------------

# Four subjects.
set.seed(3101L)
base <- rnorm(4L, 20, 3)
write_dataset(
  within_long(
    outer(base, c(0, 1.5, 3), "+") + matrix(rnorm(12L, 0, 1), 4L),
    subject_ids(4L), c("a", "b", "c")
  ),
  "edge_within_small"
)

# Integer scores 1 to 5 with ties within subjects: Friedman's and Durbin-Conover's
# tie handling, Kendall's W with ties.
set.seed(3102L)
scores <- cbind(
  sample(1:5, 15L, replace = TRUE, prob = c(0.3, 0.3, 0.2, 0.1, 0.1)),
  sample(1:5, 15L, replace = TRUE, prob = c(0.2, 0.2, 0.2, 0.2, 0.2)),
  sample(1:5, 15L, replace = TRUE, prob = c(0.1, 0.1, 0.2, 0.3, 0.3))
)
write_dataset(within_long(scores, subject_ids(15L), c("a", "b", "c")), "edge_within_ties")

# Every value negative.
set.seed(3103L)
base <- rnorm(12L, -50, 6)
write_dataset(
  within_long(
    outer(base, c(0, -2, -4), "+") + matrix(rnorm(36L, 0, 1.5), 12L),
    subject_ids(12L), c("a", "b", "c")
  ),
  "edge_within_negative"
)

# Two outlying cells (+30 in b, -25 in c).
set.seed(3104L)
base <- rnorm(20L, 50, 5)
values <- outer(base, c(0, 1, 2), "+") + matrix(rnorm(60L, 0, 2), 20L)
values[3L, 2L] <- values[3L, 2L] + 30
values[11L, 3L] <- values[11L, 3L] - 25
within_outliers <- within_long(values, subject_ids(20L), c("a", "b", "c"))
write_dataset(within_outliers, "edge_within_outliers")

# Condition means equal up to rounding: F near 0, partial omega squared 0.
set.seed(3105L)
base <- rnorm(20L, 10, 2)
noise <- matrix(rnorm(60L, 0, 1.5), 20L)
noise <- sweep(noise, 2L, colMeans(noise))
write_dataset(
  within_long(base + noise, subject_ids(20L), c("a", "b", "c")),
  "edge_within_null"
)

# 1,100 subjects.
set.seed(3106L)
base <- rnorm(1100L, 100, 15)
write_dataset(
  within_long(
    outer(base, c(0, 0.5, 1), "+") + matrix(rnorm(3300L, 0, 6), 1100L),
    sprintf("s%04d", 1:1100), c("a", "b", "c")
  ),
  "edge_within_large_n"
)

# Five conditions, ten pairs, unequal spreads.
set.seed(3107L)
base <- rnorm(15L, 30, 5)
write_dataset(
  within_long(
    outer(base, c(0, 0.5, 1.5, 2, 4), "+") +
      sweep(matrix(rnorm(75L), 15L), 2L, c(1, 1.5, 2, 3, 4), "*"),
    subject_ids(15L), paste0("c", 1:5)
  ),
  "edge_within_k5"
)

# Strongly non-spherical: w, x and y move with the subject; z adds a large
# subject-specific term, so one direction dominates the differences and
# Greenhouse-Geisser's epsilon is near its lower bound 1 / (k - 1).
set.seed(3108L)
base <- rnorm(12L, 30, 5)
values <- outer(base, c(0, 0.4, 0.8, 1), "+") + matrix(rnorm(48L, 0, 0.3), 12L)
values[, 4L] <- values[, 4L] + 6 * rnorm(12L)
write_dataset(
  within_long(values, subject_ids(12L), c("w", "x", "y", "z")),
  "edge_within_nonspherical"
)

# edge_within_outliers with its rows in a random order.
set.seed(3109L)
write_dataset(within_outliers[sample(nrow(within_outliers)), ], "edge_within_shuffled")
