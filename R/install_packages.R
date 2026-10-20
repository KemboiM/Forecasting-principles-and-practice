# -----------------------------------------------------------------------------
# R/install_packages.R -- run ONCE to install every R package used in the book.
# -----------------------------------------------------------------------------
pkgs <- c(
  "fpp2", "forecast", "fma", "expsmooth",       # book packages + data
  "ggplot2", "GGally", "gridExtra",              # graphics (Ch. 2, 5)
  "seasonal", "x13binary",                       # X11 / SEATS decomposition (Ch. 6)
  "urca", "tseries", "lmtest",                   # unit-root & residual tests (Ch. 8, 9)
  "vars",                                        # vector autoregressions (Ch. 11)
  "hts",                                         # hierarchical forecasting (Ch. 10)
  "thief",                                       # temporal hierarchies (Ch. 12)
  "readxl", "jsonlite",                          # reading exercise data / tests
  "rmarkdown", "knitr", "reticulate",            # notes in R + Python
  "testthat"                                     # model tests
)
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
message("All R packages installed: ", paste(pkgs, collapse = ", "))
