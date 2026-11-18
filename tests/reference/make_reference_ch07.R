# R reference values for tests/python/test_ch07.py (Chapter 7: exponential smoothing)
suppressPackageStartupMessages({library(fpp2); library(jsonlite)})
oildata <- window(oil, start = 1996); air <- window(ausair, start = 1990)
aust <- window(austourists, start = 2005)
fc_ses <- ses(oildata, h = 5); fc_holt <- holt(air, h = 5)
fc_damp <- holt(air, damped = TRUE, phi = 0.9, h = 15)
fc_hwa <- hw(aust, seasonal = "additive"); fc_hwm <- hw(aust, seasonal = "multiplicative")
fit_aust <- ets(aust)
fit_ann <- ets(oil, model = "ANN")
num <- function(x) as.numeric(x)
ref <- list(
  ses_par = num(fc_ses$model$par), ses_mean = num(fc_ses$mean), ses_hi95 = num(fc_ses$upper[, 2]),
  ses_mse = fc_ses$model$mse,
  holt_par = num(fc_holt$model$par), holt_mean = num(fc_holt$mean), holt_hi95 = num(fc_holt$upper[, 2]),
  holt_mse = fc_holt$model$mse,
  damped_par = num(fc_damp$model$par), damped_mean = num(fc_damp$mean), damped_hi95 = num(fc_damp$upper[, 2]),
  hwa_mse = fc_hwa$model$mse, hwm_mse = fc_hwm$model$mse,
  hwa_mean = num(fc_hwa$mean), hwm_mean = num(fc_hwm$mean),
  aust_method = fit_aust$method, aust_mean = num(forecast(fit_aust, h = 8)$mean),
  ann_mean = num(forecast(fit_ann, h = 5)$mean), ann_hi95 = num(forecast(fit_ann, h = 5)$upper[, 2]),
  ann_sigma2 = fit_ann$sigma2,
  livestock_cv_ses_mse = mean(tsCV(livestock, ses, h = 1)^2, na.rm = TRUE)
)
write_json(ref, "tests/reference/ch07_reference.json", digits = 12, auto_unbox = TRUE, pretty = TRUE)
cat("wrote tests/reference/ch07_reference.json\n")
