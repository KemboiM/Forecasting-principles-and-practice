# -----------------------------------------------------------------------------
# R/setup.R  -- sourced at the top of every chapter .Rmd
# Loads the packages used in "Forecasting: Principles and Practice" (2nd ed.)
# and connects R to Python (via reticulate) so each step can be shown in both.
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(fpp2)       # loads forecast, ggplot2, fma, expsmooth + book data
  library(knitr)
})

knitr::opts_chunk$set(
  echo = TRUE, message = FALSE, warning = FALSE,
  fig.width = 8, fig.height = 4.2, fig.align = "center",
  comment = "#>", dpi = 110
)

# ---- Python bridge ----------------------------------------------------------
# Preferred: a dedicated virtualenv called "fpp" created by R/setup_python.R.
# Otherwise reticulate falls back to whatever Python it finds (RETICULATE_PYTHON wins).
if (requireNamespace("reticulate", quietly = TRUE)) {
  if (Sys.getenv("RETICULATE_PYTHON") == "" &&
      "fpp" %in% tryCatch(reticulate::virtualenv_list(), error = function(e) character())) {
    reticulate::use_virtualenv("fpp", required = TRUE)
  }
}

# Project paths (chapter .Rmd files live one level below the project root)
proj_root <- normalizePath(file.path(getwd(), ".."), mustWork = FALSE)
if (!file.exists(file.path(proj_root, "data", "_metadata.csv"))) proj_root <- getwd()

# X-13ARIMA-SEATS binary (used by seasonal in R and by statsmodels in Python, Ch. 6)
if (requireNamespace("x13binary", quietly = TRUE)) {
  Sys.setenv(X13PATH = x13binary::x13path())
}
theme_set(theme_bw())
