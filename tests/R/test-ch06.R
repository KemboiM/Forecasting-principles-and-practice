# Chapter 6 — every decomposition and decomposition-based forecast is fitted and checked.
suppressPackageStartupMessages(library(seasonal))
context("Chapter 6: Time series decomposition")

train <- window(elecequip, end = c(2009, 12)); test <- window(elecequip, start = 2010)
h <- length(test)

test_that("moving averages: m-MA, 2xm-MA and their weights", {
  m5 <- ma(elecsales, 5)
  expect_equal(sum(is.na(m5)), 4)
  expect_equal(as.numeric(m5[3]), mean(elecsales[1:5]))
  beer2 <- window(ausbeer, start = 1992)
  expect_equal(as.numeric(ma(beer2, 4)[3]),
               sum(c(1/8, 1/4, 1/4, 1/4, 1/8) * beer2[1:5]))
  w <- convolve(rep(1/3, 3), rev(rep(1/5, 5)), type = "open")
  expect_equal(w, c(1, 2, 3, 3, 3, 2, 1) / 15)
})

test_that("classical decomposition (additive and multiplicative)", {
  dm <- decompose(elecequip, type = "multiplicative")
  expect_equal(sum(dm$figure), 12, tolerance = 1e-10)
  expect_equal(max(abs(dm$trend * dm$seasonal * dm$random - elecequip), na.rm = TRUE), 0, tolerance = 1e-8)
  da <- decompose(elecequip, type = "additive")
  expect_equal(sum(da$figure), 0, tolerance = 1e-10)
})

test_that("X11 and SEATS decompositions run", {
  for (fit in list(seas(elecequip, x11 = ""), seas(elecequip))) {
    expect_length(seasonal(fit), length(elecequip))
    expect_true(all(is.finite(seasadj(fit))))
    expect_equal(as.numeric(seasadj(fit)), as.numeric(elecequip / seasonal(fit)), tolerance = 1e-4)
  }
})

test_that("STL (periodic/robust and flexible) and mstl", {
  f1 <- stl(elecequip, t.window = 13, s.window = "periodic", robust = TRUE)
  expect_equal(max(abs(rowSums(f1$time.series) - elecequip)), 0, tolerance = 1e-8)
  s <- seasonal(f1); expect_equal(as.numeric(s[1:12]), as.numeric(s[13:24]))
  f2 <- stl(elecequip, s.window = 7)
  expect_false(isTRUE(all.equal(as.numeric(seasonal(f2)[1:12]), as.numeric(seasonal(f2)[181:192]))))
  ms <- mstl(elecequip)
  expect_equal(max(abs(ms[, "Trend"] + ms[, "Seasonal12"] + ms[, "Remainder"] - elecequip)), 0, tolerance = 1e-8)
})

test_that("strength of trend and seasonality lies in [0, 1]", {
  d <- mstl(elecequip); r <- d[, "Remainder"]
  ft <- max(0, 1 - var(r) / var(d[, "Trend"] + r)); fs <- max(0, 1 - var(r) / var(d[, "Seasonal12"] + r))
  expect_true(ft > 0.9 && ft <= 1); expect_true(fs > 0.8 && fs <= 1)
})

test_that("forecasting with decomposition: every method", {
  stlp <- stl(train, t.window = 13, s.window = "periodic", robust = TRUE)
  for (m in c("naive", "rwdrift", "ets", "arima")) {
    fc <- forecast(stlp, method = m, h = h)
    expect_valid_forecast(fc, h)
    expect_lt(accuracy(fc, test)["Test set", "RMSE"], accuracy(naive(train, h = h), test)["Test set", "RMSE"])
  }
  for (m in c("ets", "naive", "rwdrift")) expect_valid_forecast(stlf(train, h = h, method = m), h)
  expect_valid_forecast(stlf(fancy, method = "rwdrift", lambda = 0), 24)
  expect_valid_forecast(stlf(bricksq, method = "naive", s.window = 7, robust = TRUE), 8)
  # X11 / SEATS / classical seasonally adjusted series + naive
  for (sa in list(seasadj(seas(train, x11 = "")), seasadj(seas(train)),
                  train / decompose(train, "multiplicative")$seasonal)) {
    expect_valid_forecast(naive(sa, h = h), h)
  }
  expect_valid_forecast(snaive(train, h = h), h)
})
