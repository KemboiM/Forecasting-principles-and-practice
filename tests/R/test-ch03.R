# Chapter 3 — every benchmark method, transformation, residual diagnostic, accuracy tool and
# prediction-interval method of the chapter (notes + exercises) is fitted and checked.
context("Chapter 3: The forecaster's toolbox")

beer_tr <- window(ausbeer, start = 1992, end = c(2007, 4)); beer_te <- window(ausbeer, start = 2008)
g_te <- window(goog, start = 201, end = 240)
a10_tr <- window(a10, end = c(2005, 12)); a10_te <- window(a10, start = 2006)
eggs_tr <- window(eggs, end = 1983); eggs_te <- window(eggs, start = 1984)

test_that("the four benchmark methods produce valid forecasts", {
  for (fc in list(meanf(beer_tr, h = 11), naive(beer_tr, h = 11), snaive(beer_tr, h = 11),
                  rwf(beer_tr, drift = TRUE, h = 11), rwf(goog200, h = 40), meanf(goog200, h = 40),
                  rwf(goog200, drift = TRUE, h = 40)))
    expect_valid_forecast(fc, length(fc$mean))
  expect_equal(as.numeric(snaive(beer_tr, h = 4)$mean), as.numeric(tail(beer_tr, 4)))
  expect_equal(as.numeric(naive(goog200, h = 3)$mean), rep(as.numeric(tail(goog200, 1)), 3))
})

test_that("benchmarks rank as the chapter says on the test sets", {
  b <- sapply(list(meanf(beer_tr, h = 10), rwf(beer_tr, h = 10), snaive(beer_tr, h = 10)),
              function(f) test_acc(f, beer_te)["MASE"])
  expect_equal(unname(which.min(b)), 3)                                   # seasonal naive best for beer
  g <- sapply(list(meanf(goog200, h = 40), rwf(goog200, h = 40), rwf(goog200, drift = TRUE, h = 40)),
              function(f) test_acc(f, g_te)["RMSE"])
  expect_equal(unname(which.min(g)), 3)                                   # drift best for google
})

test_that("Box-Cox transformations and bias adjustment", {
  lam <- BoxCox.lambda(elec)
  expect_true(lam > 0 && lam < 0.5)
  expect_equal(InvBoxCox(BoxCox(elec, lam), lam), elec, tolerance = 1e-8)
  for (fc in list(snaive(a10_tr, h = 31, lambda = 0), snaive(a10_tr, h = 31, lambda = BoxCox.lambda(a10_tr)),
                  rwf(a10_tr, drift = TRUE, h = 31, lambda = 0)))
    expect_valid_forecast(fc, 31)
  # snaive point forecasts are unchanged by a transformation
  expect_equal(as.numeric(snaive(a10_tr, h = 31, lambda = 0)$mean), as.numeric(snaive(a10_tr, h = 31)$mean))
  f1 <- rwf(eggs_tr, drift = TRUE, lambda = 0, h = 10)
  f2 <- rwf(eggs_tr, drift = TRUE, lambda = 0, h = 10, biasadj = TRUE)
  expect_valid_forecast(f1, 10); expect_true(all(f2$mean > f1$mean))
  expect_equal(as.numeric(milk / monthdays(milk))[1], as.numeric(milk[1]) / 31)
})

test_that("residual diagnostics: naive on goog200 is white noise, snaive on beer is not", {
  res <- residuals(naive(goog200))
  expect_gt(Box.test(res, lag = 10, fitdf = 0)$p.value, 0.05)
  expect_gt(Box.test(res, lag = 10, fitdf = 0, type = "Lj")$p.value, 0.05)
  expect_lt(abs(mean(res, na.rm = TRUE)), 1)
  expect_lt(checkresiduals(snaive(window(ausbeer, start = 1992)), plot = FALSE)$p.value, 0.001)
  expect_lt(checkresiduals(naive(WWWusage), plot = FALSE)$p.value, 0.001)
  expect_lt(checkresiduals(snaive(bricksq), plot = FALSE)$p.value, 0.001)
})

test_that("time series cross-validation", {
  e <- tsCV(goog200, rwf, drift = TRUE, h = 1)
  cv <- sqrt(mean(e^2, na.rm = TRUE))
  resid_rmse <- sqrt(mean(residuals(rwf(goog200, drift = TRUE))^2, na.rm = TRUE))
  expect_gt(cv, resid_rmse)
  e8 <- tsCV(goog200, naive, h = 8)
  mse <- colMeans(e8^2, na.rm = TRUE)
  expect_true(all(diff(mse) > 0))                                # error grows with horizon
})

test_that("prediction intervals: formulas, bootstrap, coverage", {
  fc <- naive(goog200, h = 10)
  s <- sqrt(mean(residuals(naive(goog200))^2, na.rm = TRUE))
  expect_equal(as.numeric(fc$upper[, 2] - fc$mean), qnorm(0.975) * s * sqrt(1:10), tolerance = 1e-8)
  set.seed(1)
  fb <- naive(goog200, h = 10, bootstrap = TRUE)
  expect_valid_forecast(fb, 10)
  fn <- naive(goog200, h = 40)
  cover <- mean(g_te >= fn$lower[, 2] & g_te <= fn$upper[, 2])
  expect_gte(cover, 0.8)
})

test_that("forecast() on a raw series uses ETS and beats seasonal naive on beer", {
  fc <- forecast(beer_tr, h = 10)
  expect_true(grepl("^ETS", fc$method))
  expect_valid_forecast(fc, 10)
  expect_lt(test_acc(fc, beer_te)["MASE"], test_acc(snaive(beer_tr, h = 10), beer_te)["MASE"])
})

test_that("exercise models: visnights, dowjones, ibmclose, hsales", {
  qld <- visnights[, "QLDMetro"]
  for (e in 2013:2015) expect_valid_forecast(snaive(window(qld, end = c(e, 4)), h = 4), 4)
  fd <- rwf(dowjones, drift = TRUE, h = 20)
  n <- length(dowjones)
  expect_equal(as.numeric(fd$mean), as.numeric(dowjones[n] + (dowjones[n] - dowjones[1]) / (n - 1) * (1:20)))
  ibm_tr <- subset(ibmclose, end = 300); ibm_te <- subset(ibmclose, start = 301)
  r <- sapply(list(meanf(ibm_tr, h = 69), naive(ibm_tr, h = 69), rwf(ibm_tr, drift = TRUE, h = 69)),
              function(f) test_acc(f, ibm_te)["RMSE"])
  expect_equal(unname(which.min(r)), 3)
  hs_tr <- window(hsales, end = c(1993, 11)); hs_te <- window(hsales, start = c(1993, 12))
  r <- sapply(list(meanf(hs_tr, h = 24), naive(hs_tr, h = 24), snaive(hs_tr, h = 24), rwf(hs_tr, drift = TRUE, h = 24)),
              function(f) test_acc(f, hs_te)["RMSE"])
  expect_equal(unname(which.min(r)), 3)
})
