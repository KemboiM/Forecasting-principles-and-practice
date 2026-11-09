# R reference values for the Python Chapter 3 tests (tests/python/test_ch03.py)
# Run from the project root:  Rscript tests/reference/make_reference_ch03.R
suppressPackageStartupMessages({library(fpp2); library(jsonlite)})
fcl <- function(fc) list(mean = as.numeric(fc$mean), lower = unname(as.matrix(fc$lower)),
                         upper = unname(as.matrix(fc$upper)))
acc <- function(fc, x) as.list(accuracy(fc, x)["Test set", c("ME", "RMSE", "MAE", "MPE", "MAPE", "MASE")])
beer_tr <- window(ausbeer, start = 1992, end = c(2007, 4)); beer_te <- window(ausbeer, start = 2008)
g_te <- window(goog, start = 201, end = 240)
a10_tr <- window(a10, end = c(2005, 12)); a10_te <- window(a10, start = 2006)
eggs_tr <- window(eggs, end = 1983); eggs_te <- window(eggs, start = 1984)
lam_a10 <- BoxCox.lambda(a10_tr)
res <- residuals(naive(goog200))
bp <- Box.test(res, lag = 10, fitdf = 0)
qld <- visnights[, "QLDMetro"]
ibm_tr <- subset(ibmclose, end = 300); ibm_te <- subset(ibmclose, start = 301)
hs_tr <- window(hsales, end = c(1993, 11)); hs_te <- window(hsales, start = c(1993, 12))
cv <- function(f, ...) { e <- tsCV(goog200, f, h = 1, ...); sqrt(mean(e^2, na.rm = TRUE)) }
ref <- list(
  lambda = as.list(sapply(list(elec = elec, usnetelec = usnetelec, usgdp = usgdp, mcopper = mcopper,
                               enplanements = enplanements, cangas = cangas, dole = dole,
                               usdeaths = usdeaths, bricksq = bricksq, a10_tr = a10_tr), BoxCox.lambda)),
  milk_daily_head = as.numeric(head(milk / monthdays(milk), 12)),
  eggs_fc = fcl(rwf(eggs, drift = TRUE, lambda = 0, h = 50, level = 80)),
  eggs_fc_biasadj = as.numeric(rwf(eggs, drift = TRUE, lambda = 0, h = 50, level = 80, biasadj = TRUE)$mean),
  box_pierce = list(Q = unname(bp$statistic), p = bp$p.value),
  naive_goog = fcl(naive(goog200)),
  acc = list(
    beer_mean = acc(meanf(beer_tr, h = 11), beer_te), beer_naive = acc(naive(beer_tr, h = 11), beer_te),
    beer_snaive = acc(snaive(beer_tr, h = 11), beer_te), beer_drift = acc(rwf(beer_tr, drift = TRUE, h = 11), beer_te),
    goog_mean = acc(meanf(goog200, h = 40), g_te), goog_naive = acc(naive(goog200, h = 40), g_te),
    goog_drift = acc(rwf(goog200, drift = TRUE, h = 40), g_te),
    a10_snaive_bc = acc(snaive(a10_tr, h = 31, lambda = lam_a10), a10_te),
    a10_drift_log = acc(rwf(a10_tr, drift = TRUE, h = 31, lambda = 0), a10_te),
    a10_drift_bc = acc(rwf(a10_tr, drift = TRUE, h = 31, lambda = lam_a10), a10_te),
    eggs_log = acc(rwf(eggs_tr, drift = TRUE, lambda = 0, h = 10), eggs_te),
    eggs_log_biasadj = acc(rwf(eggs_tr, drift = TRUE, lambda = 0, h = 10, biasadj = TRUE), eggs_te),
    ibm_mean = acc(meanf(ibm_tr, h = 69), ibm_te), ibm_naive = acc(naive(ibm_tr, h = 69), ibm_te),
    ibm_drift = acc(rwf(ibm_tr, drift = TRUE, h = 69), ibm_te),
    hs_snaive = acc(snaive(hs_tr, h = 24), hs_te), hs_drift = acc(rwf(hs_tr, drift = TRUE, h = 24), hs_te)
  ),
  tscv = list(mean = cv(meanf), naive = cv(naive), drift = cv(rwf, drift = TRUE)),
  qld_mape = c(accuracy(snaive(window(qld, end = c(2015, 4)), h = 4), window(qld, start = 2016))["Test set", "MAPE"],
               accuracy(snaive(window(qld, end = c(2014, 4)), h = 4), window(qld, start = 2015, end = c(2015, 4)))["Test set", "MAPE"],
               accuracy(snaive(window(qld, end = c(2013, 4)), h = 4), window(qld, start = 2014, end = c(2014, 4)))["Test set", "MAPE"]),
  lb_beer_snaive = Box.test(residuals(snaive(window(ausbeer, start = 1992))), lag = 8, type = "Lj")$statistic[[1]],
  lb_www_naive = Box.test(residuals(naive(WWWusage)), lag = 10, type = "Lj")$statistic[[1]],
  dowjones_drift = as.numeric(rwf(dowjones, drift = TRUE, h = 20)$mean)
)
write_json(ref, "tests/reference/ch03_reference.json", digits = 12, auto_unbox = TRUE, pretty = TRUE)
cat("wrote tests/reference/ch03_reference.json\n")
