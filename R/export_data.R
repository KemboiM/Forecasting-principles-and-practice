# -----------------------------------------------------------------------------
# export_data.R
# Exports every dataset used in "Forecasting: Principles and Practice" (2nd ed.,
# Hyndman & Athanasopoulos) from the R packages fpp2, fma, expsmooth, forecast
# (plus a few base-R datasets) to CSV so that the Python versions of every
# example and exercise use EXACTLY the same numbers as the R versions.
#
# Run from the project root (RStudio: open the .Rproj, then source this file).
# Output: data/<name>.csv  +  data/_metadata.csv
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(fpp2); library(fma); library(expsmooth); library(forecast)
})

out_dir <- "data"
dir.create(out_dir, showWarnings = FALSE)

pkgs <- c("fpp2", "fma", "expsmooth", "forecast")
base_sets <- c("AirPassengers", "USAccDeaths", "Nile", "LakeHuron", "sunspot.year",
               "nottem", "UKgas", "JohnsonJohnson", "co2", "austres",
               "UKDriverDeaths", "Seatbelts", "EuStockMarkets", "WWWusage")

items <- do.call(rbind, lapply(pkgs, function(p)
  data.frame(name = data(package = p)$results[, "Item"], package = p)))
items <- rbind(items, data.frame(name = base_sets, package = "datasets"))
items <- items[!duplicated(items$name), ]   # fpp2 takes priority over fma, etc.

meta <- list()
for (i in seq_len(nrow(items))) {
  nm <- items$name[i]; pk <- items$package[i]
  e <- new.env()
  data(list = nm, package = pk, envir = e)
  x <- get(nm, envir = e)
  if (is.ts(x)) {
    tt <- as.numeric(time(x))
    df <- data.frame(time = round(tt, 6),
                     year = floor(tt + 1e-8),
                     cycle = as.numeric(cycle(x)))
    vals <- as.data.frame(as.matrix(x))
    if (NCOL(x) == 1) names(vals) <- "value"
    df <- cbind(df, vals)
    msts_p <- if (inherits(x, "msts")) paste(attr(x, "msts"), collapse = ";") else ""
    meta[[nm]] <- data.frame(name = nm, package = pk, type = "ts",
                             start = round(tt[1], 6), frequency = frequency(x),
                             msts = msts_p, n = NROW(x), ncol = NCOL(x),
                             columns = paste(names(vals), collapse = ";"))
  } else if (is.data.frame(x)) {
    df <- x
    meta[[nm]] <- data.frame(name = nm, package = pk, type = "data.frame",
                             start = NA, frequency = NA, msts = "",
                             n = nrow(x), ncol = ncol(x),
                             columns = paste(names(x), collapse = ";"))
  } else next
  write.csv(df, file.path(out_dir, paste0(nm, ".csv")), row.names = FALSE)
}
meta <- do.call(rbind, meta)
write.csv(meta, file.path(out_dir, "_metadata.csv"), row.names = FALSE)
message("Exported ", nrow(meta), " datasets to ", normalizePath(out_dir))
