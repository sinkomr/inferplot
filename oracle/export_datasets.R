#!/usr/bin/env Rscript
# Export the R-provided oracle datasets to CSV.
#
# Usage: Rscript oracle/export_datasets.R <output-dir>
#
# The CSV files are committed under oracle/datasets/. The oracle workflow re-runs
# this script before generating fixtures and fails if a committed file changed, so
# the R generator and the Python tests always read identical bytes. Files named
# edge_*.csv are written for this project by hand and are not touched here.
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
