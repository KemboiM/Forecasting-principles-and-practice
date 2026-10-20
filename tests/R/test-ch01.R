# Chapter 1 — every model used in the notes is fitted, forecast and checked.
context("Chapter 1: Getting started")

train <- subset(elecdaily, end = 300)
test  <- subset(elecdaily, start = 301)
h <- nrow(test)

test_that("explanatory model (tslm) fits and forecasts", {
  fit <- tslm(Demand ~ Temperature + I(Temperature^2) + WorkDay, data = train)
  expect_equal(length(coef(fit)), 4)
  expect_gt(coef(fit)["I(Temperature^2)"], 0)           # U-shaped temperature effect
  expect_gt(coef(fit)["WorkDay"], 0)                     # more demand on working days
  fc <- forecast(fit, newdata = data.frame(Temperature = test[, "Temperature"],
                                           WorkDay = test[, "WorkDay"]))
  expect_valid_forecast(fc, h)
})

test_that("time series model (ets) fits and forecasts", {
  fit <- ets(train[, "Demand"])
  expect_true(grepl("^ETS", fit$method))
  expect_valid_forecast(forecast(fit, h = h), h)
})

test_that("mixed model (regression with ARIMA errors) fits and forecasts", {
  xr <- cbind(Temp = train[, "Temperature"], Temp2 = train[, "Temperature"]^2, WorkDay = train[, "WorkDay"])
  xt <- cbind(Temp = test[, "Temperature"],  Temp2 = test[, "Temperature"]^2,  WorkDay = test[, "WorkDay"])
  fit <- auto.arima(train[, "Demand"], xreg = xr)
  expect_equal(arimaorder(fit)[c("p", "d", "q", "P", "D", "Q")], c(p = 1, d = 0, q = 0, P = 2, D = 0, Q = 0))
  fc <- forecast(fit, xreg = xt)
  expect_valid_forecast(fc, h)
  # the mixed model should beat the seasonal naive benchmark on the test set
  expect_lt(test_acc(fc, test[, "Demand"])["RMSE"],
            test_acc(snaive(train[, "Demand"], h = h), test[, "Demand"])["RMSE"])
})

test_that("ETS(A,A,N) on austa (Section 1.7) and simulated futures", {
  fit <- ets(austa)
  expect_equal(fit$method, "ETS(A,A,N)")
  fc <- forecast(fit, h = 10)
  expect_valid_forecast(fc, 10)
  set.seed(1); sim <- simulate(fit, nsim = 10)
  expect_length(sim, 10); expect_true(all(is.finite(sim)))
  fit_tr <- ets(window(austa, end = 2005), model = "AAN", damped = FALSE)
  expect_valid_forecast(forecast(fit_tr, h = 10), 10)
})

test_that("ETS on beer (Figure 1.1) and seasonal naive benchmark", {
  beer2 <- window(ausbeer, start = 1992)
  expect_valid_forecast(forecast(ets(beer2), h = 8), 8)
  expect_valid_forecast(snaive(beer2, h = 8), 8)
})
