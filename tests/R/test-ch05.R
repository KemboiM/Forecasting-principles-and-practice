# Chapter 5 — every regression model in the notes and exercises is fitted, forecast and checked.
context("Chapter 5: Time series regression models")

test_that("US consumption: simple, multiple, best regressions + CV + BG", {
  fit1 <- tslm(Consumption ~ Income, data = uschange)
  fit4 <- tslm(Consumption ~ Income + Production + Unemployment + Savings, data = uschange)
  fit3 <- tslm(Consumption ~ Income + Savings + Unemployment, data = uschange)
  expect_equal(unname(round(coef(fit1), 3)), c(0.545, 0.281))
  expect_equal(round(summary(fit4)$r.squared, 3), 0.754)
  expect_equal(round(unname(CV(fit4)["AICc"]), 2), -408.83)
  expect_gt(lmtest::bgtest(fit4, order = 8)$p.value, 0.05)
  nd <- data.frame(Income = rep(1, 4), Savings = rep(0.5, 4), Unemployment = rep(0, 4))
  expect_valid_forecast(forecast(fit3, newdata = nd), 4)
  expect_valid_forecast(forecast(fit1, newdata = data.frame(Income = rep(5, 4))), 4)
})

test_that("spurious regression has high R2 and autocorrelated residuals", {
  fit <- tslm(window(ausair, end = 2011) ~ guinearice)
  expect_gt(summary(fit)$r.squared, 0.95)
  expect_lt(lmtest::bgtest(fit, order = 8)$p.value, 0.01)
})

test_that("beer: trend + season equals trend + Fourier(K=2); forecasts valid", {
  beer2 <- window(ausbeer, start = 1992)
  f1 <- tslm(beer2 ~ trend + season); f2 <- tslm(beer2 ~ trend + fourier(beer2, K = 2))
  expect_equal(as.numeric(fitted(f1)), as.numeric(fitted(f2)))
  expect_valid_forecast(forecast(f1), 10)       # forecast.lm default h = 10
  expect_equal(sum(easter(ausbeer)), 55)
})

test_that("interventions, distributed lags", {
  d <- Seatbelts[, "drivers"]; law <- Seatbelts[, "law"]
  fit <- tslm(d ~ trend + season + law)
  expect_lt(coef(fit)["law"], -150)
  expect_valid_forecast(forecast(fit, newdata = data.frame(law = rep(1, 12))), 12)
  adv <- cbind(insurance[, "TV.advert"], stats::lag(insurance[, "TV.advert"], -1))[1:40, ]
  expect_equal(length(coef(lm(insurance[, "Quotes"] ~ adv))), 3)
})

test_that("marathon: linear, exponential, piecewise, cubic spline, splinef", {
  h <- 10
  expect_valid_forecast(forecast(tslm(marathon ~ trend), h = h), h)
  expect_valid_forecast(forecast(tslm(marathon ~ trend, lambda = 0), h = h), h)
  t <- time(marathon); tb1 <- pmax(0, t - 1940); tb2 <- pmax(0, t - 1980)
  nd <- data.frame(t = t[length(t)] + seq(h), tb1 = tb1[length(tb1)] + seq(h), tb2 = tb2[length(tb2)] + seq(h))
  expect_valid_forecast(forecast(tslm(marathon ~ t + tb1 + tb2), newdata = nd), h)
  t2 <- t^2; t3 <- t^3; b1 <- tb1^3; b2 <- tb2^3     # plain names: tslm() cannot see I() terms of local variables
  nd2 <- data.frame(t = nd$t, t2 = nd$t^2, t3 = nd$t^3, b1 = nd$tb1^3, b2 = nd$tb2^3)
  expect_valid_forecast(forecast(tslm(marathon ~ t + t2 + t3 + b1 + b2), newdata = nd2), h)
  expect_valid_forecast(splinef(marathon, lambda = 0, h = h), h)
})

test_that("exercise models: elecdaily, mens400, fancy, gasoline, huron", {
  d20 <- head(elecdaily, 20)
  expect_valid_forecast(forecast(tslm(Demand ~ Temperature, data = d20), newdata = data.frame(Temperature = c(15, 35))), 2)
  f400 <- tslm(mens400 ~ trend)
  expect_lt(coef(f400)["trend"], 0); expect_valid_forecast(forecast(f400, h = 1), 1)
  surf <- ts(as.numeric(cycle(fancy) == 3 & time(fancy) >= 1988), start = 1987, frequency = 12)
  ff <- tslm(fancy ~ trend + season + surf, lambda = 0)
  expect_valid_forecast(forecast(ff, newdata = data.frame(surf = rep(c(0, 0, 1, rep(0, 9)), 3))), 36)
  gas <- window(gasoline, end = 2005)
  aicc <- sapply(1:26, function(K) { X <- fourier(gas, K = K); CV(tslm(gas ~ trend + X))["AICc"] })
  expect_equal(unname(which.min(aicc)), 12)
  X12 <- fourier(gas, K = 12)
  expect_valid_forecast(forecast(tslm(gas ~ trend + X12), newdata = data.frame(X12 = I(fourier(gas, K = 12, h = 52)))), 52)
  th <- time(huron); t15 <- pmax(0, th - 1915)
  fl <- tslm(huron ~ trend); fp <- tslm(huron ~ th + t15)
  expect_lt(CV(fp)["AICc"], CV(fl)["AICc"])
  expect_valid_forecast(forecast(fp, newdata = data.frame(th = 1973:1980, t15 = 1973:1980 - 1915)), 8)
})
