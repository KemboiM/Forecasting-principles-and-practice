# R reference values for tests/python/test_ch04.py
suppressPackageStartupMessages({library(fpp2); library(jsonlite)})
tfc <- read.csv("Chapter-04-Judgmental-Forecasts/tfc_forecasts.csv")
vn <- ts(tfc$Data / 1000, start = 1998)
vn_tr <- window(vn, end = 2004)
tr <- window(a10, end = c(2005, 12))
ref <- list(
  a10_naive = as.numeric(naive(tr, h = 30)$mean),
  a10_snaive = as.numeric(snaive(tr, h = 30)$mean),
  a10_ets_method = ets(tr)$method,
  vn_drift = as.numeric(rwf(vn_tr, drift = TRUE, h = 7)$mean),
  vn_drift_lo95 = as.numeric(rwf(vn_tr, drift = TRUE, h = 7)$lower[, 2]),
  vn_lin = as.numeric(forecast(tslm(vn_tr ~ trend), h = 7)$mean),
  vn_lin_hi80 = as.numeric(forecast(tslm(vn_tr ~ trend), h = 7)$upper[, 1]),
  vn_ann = as.numeric(forecast(ets(vn_tr, model = "ANN"), h = 7)$mean),
  growth_q = as.numeric(quantile(diff(log(vn)), c(0.1, 0.5, 0.9)))
)
write_json(ref, "tests/reference/ch04_reference.json", digits = 12, auto_unbox = TRUE, pretty = TRUE)
