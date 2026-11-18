# Chapter 7 — every exponential smoothing method and ETS model is fitted, forecast and checked.
context("Chapter 7: Exponential smoothing")

oildata <- window(oil, start = 1996); air <- window(ausair, start = 1990)
aust <- window(austourists, start = 2005)

test_that("SES = ETS(A,N,N): flat forecasts and hand recursion", {
  fc <- ses(oildata, h = 5)
  expect_valid_forecast(fc, 5)
  expect_equal(length(unique(round(fc$mean, 8))), 1)
  p <- fc$model$par; l <- p["l"]
  for (y in oildata) l <- p["alpha"] * y + (1 - p["alpha"]) * l
  expect_equal(unname(l), fc$mean[1], tolerance = 1e-8)
})

test_that("Holt, damped Holt (fixed and estimated phi) and exponential trend", {
  expect_valid_forecast(holt(air, h = 5), 5)
  fd <- holt(air, damped = TRUE, phi = 0.9, h = 15)
  expect_valid_forecast(fd, 15); expect_equal(unname(fd$model$par["phi"]), 0.9)
  expect_lt(fd$mean[15], holt(air, h = 15)$mean[15])     # damping flattens the trend
  expect_valid_forecast(holt(livestock, damped = TRUE), 10)
  expect_valid_forecast(holt(air, exponential = TRUE, h = 5), 5)
  expect_valid_forecast(forecast(ets(livestock, model = "MMN", allow.multiplicative.trend = TRUE), h = 7), 7)
})

test_that("Holt-Winters additive, multiplicative and damped", {
  fa <- hw(aust, seasonal = "additive"); fm <- hw(aust, seasonal = "multiplicative")
  expect_valid_forecast(fa, 8); expect_valid_forecast(fm, 8)
  expect_equal(fa$model$components[1:3], c("A", "A", "A")); expect_equal(fm$model$components[1:3], c("M", "A", "M"))
  fd <- hw(subset(hyndsight, end = length(hyndsight) - 35), damped = TRUE, seasonal = "multiplicative", h = 35)
  expect_valid_forecast(fd, 35)
})

test_that("all 15 admissible ETS models fit and forecast; ets() picks the min AICc", {
  cands <- expand.grid(e = c("A", "M"), t = c("N", "A", "Ad"), s = c("N", "A", "M"), stringsAsFactors = FALSE)
  cands <- cands[!(cands$e == "A" & cands$s == "M"), ]
  expect_equal(nrow(cands), 15)
  aicc <- c()
  for (i in seq_len(nrow(cands))) {
    cc <- cands[i, ]
    f <- ets(aust, model = paste0(cc$e, substr(cc$t, 1, 1), cc$s), damped = cc$t == "Ad")
    expect_valid_forecast(forecast(f, h = 8), 8)
    aicc[f$method] <- f$aicc
  }
  auto <- ets(aust)
  expect_equal(auto$method, names(which.min(aicc)))
})

test_that("simulated and bootstrapped prediction intervals", {
  fit <- ets(aust); set.seed(1)
  expect_valid_forecast(forecast(fit, h = 8, simulate = TRUE, npaths = 1000), 8)
  expect_valid_forecast(forecast(fit, h = 8, simulate = TRUE, bootstrap = TRUE, npaths = 1000), 8)
})

test_that("ETS(A,N,N) variance formula (exercises 16-17)", {
  fit <- ets(oil, model = "ANN"); fc <- forecast(fit, h = 5, level = 95)
  lT <- as.numeric(tail(fit$states[, "l"], 1)); a <- unname(fit$par["alpha"]); h <- 1:5
  expect_equal(as.numeric(fc$upper), as.numeric(lT + qnorm(0.975) * sqrt(fit$sigma2 * (1 + a^2 * (h - 1)))),
               tolerance = 1e-8)
})

test_that("time series cross-validation of SES / Holt / damped", {
  for (f in list(function(y, h) ses(y, h = h), function(y, h) holt(y, h = h),
                 function(y, h) holt(y, damped = TRUE, h = h))) {
    e <- tsCV(livestock, f, h = 1)
    expect_true(is.finite(mean(e^2, na.rm = TRUE)))
  }
})
