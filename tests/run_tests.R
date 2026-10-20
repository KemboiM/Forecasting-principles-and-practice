# Run the model tests from the RStudio CONSOLE (uses the same R as RStudio).
#   source("tests/run_tests.R")      # defines run_tests()
#   run_tests(1)                     # Chapter 1 only
#   run_tests()                      # every chapter
run_tests <- function(chapter = NULL) {
  library(testthat)
  ch <- if (is.null(chapter)) NULL else sprintf("%02d", as.integer(chapter))
  cat("\n================ R model tests ================\n")
  if (is.null(ch)) {
    testthat::test_dir("tests/R", reporter = "summary", stop_on_failure = FALSE)
  } else {
    source("tests/R/helper-common.R")
    testthat::test_file(sprintf("tests/R/test-ch%s.R", ch), reporter = "summary")
  }
  cat("\n================ Python model tests ================\n")
  venv_py <- path.expand("~/.virtualenvs/fpp/bin/python")
  py <- if (file.exists(venv_py)) venv_py else Sys.which("python3")
  cat("Using Python:", py, "\n")
  target <- if (is.null(ch)) "tests/python" else sprintf("tests/python/test_ch%s.py", ch)
  status <- system2(py, c("-m", "pytest", "-q", target))
  if (status != 0) warning("Some Python tests failed (see output above).")
  invisible(status)
}
message("Ready. Run run_tests(1) for Chapter 1, or run_tests() for all chapters.")
