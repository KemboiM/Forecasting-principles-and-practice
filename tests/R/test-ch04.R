# Chapter 4 — every model / computation used in the notes and practice exercises is checked.
context("Chapter 4: Judgmental forecasts")
tfc_file <- file.path("..", "..", "Chapter-04-Judgmental-Forecasts", "tfc_forecasts.csv")
if (!file.exists(tfc_file)) tfc_file <- "Chapter-04-Judgmental-Forecasts/tfc_forecasts.csv"
tfc <- read.csv(tfc_file)
vn <- ts(tfc$Data / 1000, start = 1998)
vn_tr <- window(vn, end = 2004); vn_te <- window(vn, start = 2005)

test_that("anchoring demo: naive, seasonal naive and ETS on a10", {
  tr <- window(a10, end = c(2005, 12)); te <- window(a10, start = 2006); h <- length(te)
  f_naive <- naive(tr, h = h); f_snaive <- snaive(tr, h = h); f_ets <- forecast(ets(tr), h = h)
  for (f in list(f_naive, f_snaive, f_ets)) expect_valid_forecast(f, h)
  # anchoring on the last value is much worse than ETS
  expect_gt(test_acc(f_naive, te)["RMSE"], test_acc(f_ets, te)["RMSE"])
})

test_that("forecast log on h02: snaive and ETS(M,A,M) one-step forecasts", {
  hist <- subset(h02, end = length(h02) - 1)
  expect_valid_forecast(snaive(hist, h = 2), 2)
  fit <- ets(hist, model = "MAM", damped = FALSE)
  expect_equal(fit$method, "ETS(M,A,M)")
  expect_valid_forecast(forecast(fit, h = 2), 2)
})

test_that("Delphi rounds shrink the spread", {
  x <- c(120, 135, 150, 95, 160, 210, 140, 130, 175, 60)
  move <- c(0.5, 0.3, 0.6, 0.4, 0.5, 0.2, 0.5, 0.6, 0.4, 0.3)
  x2 <- x + move * (median(x) - x); x3 <- x2 + move * (median(x2) - x2)
  expect_lt(IQR(x3), IQR(x2)); expect_lt(IQR(x2), IQR(x))
  expect_equal(mean(x3, trim = 0.1), 137.3375)
})

test_that("visitor nights models: naive, drift, ETS(A,N,N), linear trend", {
  hv <- length(vn_te)
  expect_valid_forecast(naive(vn_tr, h = hv), hv)
  expect_valid_forecast(rwf(vn_tr, drift = TRUE, h = hv), hv)
  expect_valid_forecast(forecast(ets(vn_tr, model = "ANN"), h = hv), hv)
  expect_valid_forecast(forecast(tslm(vn_tr ~ trend), h = hv), hv)
  # TFC judgmental forecasts were optimistic: mean error (actual - forecast) < book's ETS
  tfc_f <- ts(tfc$TFC[tfc$Year >= 2005] / 1000, start = 2005)
  ets_f <- ts(tfc$ETS[tfc$Year >= 2005] / 1000, start = 2005)
  expect_lt(accuracy(tfc_f, vn_te)[, "ME"], accuracy(ets_f, vn_te)[, "ME"])
})

test_that("practice 3: ETS + informed temperature adjustment beats unadjusted ETS", {
  train <- subset(elecdaily, end = 300); test <- subset(elecdaily, start = 301)
  fit <- ets(train[, "Demand"])
  fc <- forecast(fit, h = nrow(test))
  expect_valid_forecast(fc, nrow(test))
  err <- as.numeric(train[, "Demand"] - fitted(fit))
  adj <- lm(err ~ Temperature + I(Temperature^2) + WorkDay, data = as.data.frame(train))
  informed <- fc$mean + predict(adj, newdata = as.data.frame(test))
  expect_lt(accuracy(informed, test[, "Demand"])[, "RMSE"], accuracy(fc$mean, test[, "Demand"])[, "RMSE"])
})
