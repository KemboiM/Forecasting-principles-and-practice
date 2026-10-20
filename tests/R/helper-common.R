# Shared helpers for the testthat model tests (loaded automatically by testthat).
suppressPackageStartupMessages({ library(fpp2); library(testthat) })

# A forecast object is "valid" if it has h finite point forecasts and ordered intervals.
expect_valid_forecast <- function(fc, h) {
  expect_s3_class(fc, "forecast")
  expect_length(fc$mean, h)
  expect_true(all(is.finite(fc$mean)))
  if (!is.null(fc$lower) && length(fc$lower)) {
    lo <- as.matrix(fc$lower); hi <- as.matrix(fc$upper)
    expect_true(all(lo <= hi + 1e-8))
    if (ncol(lo) >= 2) {                      # 95% interval wider than 80%
      expect_true(all(lo[, 2] <= lo[, 1] + 1e-8))
      expect_true(all(hi[, 2] >= hi[, 1] - 1e-8))
    }
    expect_true(all(fc$mean >= lo[, 1] - 1e-8 & fc$mean <= hi[, 1] + 1e-8))
  }
}

# Test-set accuracy row as a named vector
test_acc <- function(fc, x) accuracy(fc, x)["Test set", c("RMSE", "MAE", "MAPE", "MASE")]
