# Chapter 2 — time series graphics. The only model introduced is WHITE NOISE; every graphic
# and the white-noise checks used in the notes and exercises are exercised here.
context("Chapter 2: Time series graphics")

test_that("ts objects carry the right frequency and window() works", {
  y <- ts(c(123, 39, 78, 52, 110), start = 2012)
  expect_equal(frequency(y), 1); expect_equal(start(y), c(2012, 1))
  expect_equal(c(frequency(gold), frequency(woolyrnq), frequency(gas)), c(1, 4, 12))
  beer2 <- window(ausbeer, start = 1992)
  expect_equal(start(beer2), c(1992, 1))
  tute1 <- read.csv(file.path("..", "..", "data", "external", "tute1.csv"))
  mts <- ts(tute1[, -1], start = 1981, frequency = 4)
  expect_equal(dim(mts), c(100, 3)); expect_equal(end(mts), c(2005, 4))
})

test_that("all chapter graphics build without error", {
  beer2 <- window(ausbeer, start = 1992)
  plots <- list(autoplot(a10), ggseasonplot(a10, year.labels = TRUE), ggseasonplot(a10, polar = TRUE),
                ggsubseriesplot(a10), gglagplot(beer2), ggAcf(beer2), autoplot(visnights[, 1:5], facets = TRUE),
                GGally::ggpairs(as.data.frame(visnights[, 1:3])))
  for (p in plots) expect_true(inherits(p, "ggplot") || inherits(p, "ggmatrix"))
  for (p in plots[1:7]) expect_silent(ggplot2::ggplot_build(p))
})

test_that("gold outlier is observation 770", {
  expect_equal(which.max(gold), 770)
})

test_that("ACF of beer2: r4 largest, r2 most negative", {
  r <- Acf(window(ausbeer, start = 1992), lag.max = 9, plot = FALSE)$acf[-1]
  expect_equal(which.max(r), 4); expect_equal(which.min(r), 2)
})

test_that("white-noise model: simulated noise passes, structured series fail", {
  set.seed(30); y <- ts(rnorm(50))
  r <- Acf(y, lag.max = 16, plot = FALSE)$acf[-1]
  expect_lte(mean(abs(r) > 2 / sqrt(50)), 0.1)
  expect_gt(Box.test(y, lag = 10, type = "Ljung")$p.value, 0.05)
  expect_gt(Box.test(diff(goog), lag = 10, type = "Ljung")$p.value, 0.05)    # random walk changes
  expect_gt(Box.test(diff(dj), lag = 10, type = "Ljung")$p.value, 0.05)      # exercise 10
  expect_lt(Box.test(window(ausbeer, start = 1992), lag = 8, type = "Ljung")$p.value, 0.001)
  expect_lt(Box.test(a10, lag = 24, type = "Ljung")$p.value, 0.001)
  expect_lt(Box.test(window(pigs, start = 1990), lag = 24, type = "Ljung")$p.value, 0.05)  # exercise 9
})
