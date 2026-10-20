# R reference values for the Python Chapter 1 tests (tests/python/test_ch01.py)
suppressPackageStartupMessages({library(fpp2); library(jsonlite)})
train <- subset(elecdaily, end = 300); test <- subset(elecdaily, start = 301)
fit_expl <- tslm(Demand ~ Temperature + I(Temperature^2) + WorkDay, data = train)
xr <- cbind(T = train[, "Temperature"], T2 = train[, "Temperature"]^2, W = train[, "WorkDay"])
fit_mixed <- auto.arima(train[, "Demand"], xreg = xr)
fc_mixed <- forecast(fit_mixed, xreg = cbind(T = test[, "Temperature"], T2 = test[, "Temperature"]^2, W = test[, "WorkDay"]))
fit_austa <- ets(austa)
ref <- list(
  tslm_coef = unname(coef(fit_expl)),
  mixed_coef = unname(coef(fit_mixed)),
  mixed_fc = as.numeric(fc_mixed$mean),
  austa_method = fit_austa$method,
  austa_fc = as.numeric(forecast(fit_austa, h = 10)$mean),
  snaive_fc = as.numeric(snaive(train[, "Demand"], h = 65)$mean)
)
write_json(ref, "tests/reference/ch01_reference.json", digits = 12, auto_unbox = TRUE, pretty = TRUE)
