# R reference values for the Python Chapter 2 tests (tests/python/test_ch02.py)
# Run from the project root:  Rscript tests/reference/make_reference_ch02.R
suppressPackageStartupMessages({library(fpp2); library(jsonlite)})
lb <- function(x, lag) { b <- Box.test(na.omit(x), lag = lag, type = "Ljung-Box"); list(Q = unname(b$statistic), p = b$p.value) }
beer2 <- window(ausbeer, start = 1992)
aelec <- window(elec, start = 1980)
mypigs <- window(pigs, start = 1990)
tute1 <- read.csv("data/external/tute1.csv")
ref <- list(
  acf_beer2 = as.numeric(Acf(beer2, lag.max = 9, plot = FALSE)$acf[-1]),
  acf_aelec = as.numeric(Acf(aelec, lag.max = 48, plot = FALSE)$acf[-1]),
  cor_elecdemand = cor(elecdemand[, "Temperature"], elecdemand[, "Demand"]),
  freq = list(gold = frequency(gold), woolyrnq = frequency(woolyrnq), gas = frequency(gas),
              gasoline = frequency(gasoline)),
  gold_which_max = which.max(gold), gold_max = max(gold, na.rm = TRUE),
  lb_beer2 = lb(beer2, 8), lb_aelec = lb(aelec, 24), lb_a10 = lb(a10, 24),
  lb_dgoog = lb(diff(goog), 10), lb_ustreas = lb(ustreas, 10),
  lb_mypigs = lb(mypigs, 24), lb_ddj = lb(diff(dj), 10),
  tute1_colsums = as.numeric(colSums(tute1[, -1])),
  visnights_cor = unname(cor(visnights[, 1:5])[upper.tri(diag(5))])
)
write_json(ref, "tests/reference/ch02_reference.json", digits = 12, auto_unbox = TRUE, pretty = TRUE)
cat("wrote tests/reference/ch02_reference.json\n")
