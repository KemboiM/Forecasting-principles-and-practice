# R reference values for tests/python/test_ch06.py (Chapter 6: decomposition)
suppressPackageStartupMessages({library(fpp2); library(seasonal); library(jsonlite)})
beer2 <- window(ausbeer, start = 1992)
dc <- decompose(elecequip, type = "multiplicative")
da <- decompose(elecequip, type = "additive")
stlp <- stl(elecequip, t.window = 13, s.window = "periodic", robust = TRUE)
stl7 <- stl(elecequip, s.window = 7)
stlb <- stl(bricksq, s.window = 7, robust = TRUE)
x11 <- seas(elecequip, x11 = ""); seats <- seas(elecequip)
ms <- mstl(elecequip)
fn <- forecast(stlp, method = "naive", h = 24)
fd <- forecast(stlp, method = "rwdrift", h = 24)
sf <- stlf(fancy, method = "rwdrift", lambda = 0)
sb <- stlf(bricksq, method = "naive", s.window = 7)
se <- stlf(elecequip)
num <- function(x) as.numeric(x)
ref <- list(
  ma5 = num(ma(elecsales, 5)), ma4 = num(ma(beer2, 4, centre = FALSE)), ma2x4 = num(ma(beer2, 4)),
  ma12 = num(ma(elecequip, 12)),
  dec_mult_seasonal = num(dc$figure), dec_mult_trend = num(dc$trend),
  dec_add_seasonal = num(da$figure),
  stlp_trend = num(stlp$time.series[, "trend"]), stlp_seasonal = num(stlp$time.series[, "seasonal"]),
  stl7_trend = num(stl7$time.series[, "trend"]), stl7_seasonal = num(stl7$time.series[, "seasonal"]),
  stlb_seasonal = num(stlb$time.series[, "seasonal"]),
  x11_seasonal = num(seasonal(x11)), x11_trend = num(trendcycle(x11)), x11_sa = num(seasadj(x11)),
  seats_seasonal = num(seasonal(seats)), seats_trend = num(trendcycle(seats)),
  mstl_seasonal = num(ms[, "Seasonal12"]), mstl_trend = num(ms[, "Trend"]),
  fc_naive_mean = num(fn$mean), fc_naive_hi95 = num(fn$upper[, 2]),
  fc_drift_mean = num(fd$mean), fc_drift_lo80 = num(fd$lower[, 1]),
  stlf_fancy_mean = num(sf$mean), stlf_fancy_hi95 = num(sf$upper[, 2]),
  stlf_bricks_mean = num(sb$mean),
  stlf_elec_method = se$method, stlf_elec_mean = num(se$mean)
)
write_json(ref, "tests/reference/ch06_reference.json", digits = 12, auto_unbox = TRUE, pretty = TRUE, na = "null")
cat("wrote tests/reference/ch06_reference.json\n")
