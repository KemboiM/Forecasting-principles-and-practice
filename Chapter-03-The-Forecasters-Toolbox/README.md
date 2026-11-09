# Chapter 3 — The forecaster's toolbox

Following Chapter 3 of *Forecasting: Principles and Practice* (2nd ed.) by
Rob J Hyndman & George Athanasopoulos — <https://otexts.com/fpp2/>.

| File | What it is |
|---|---|
| `ch03_notes.Rmd` | Section-by-section notes (3.1–3.6), every step in **R** and **Python** |
| `ch03_exercises.Rmd` | All 12 exercises of Section 3.7 with answers (R + Python) |
| `ch03_notes_python.ipynb`, `ch03_exercises_python.ipynb` | Python-only versions (auto-generated) |

## Sections
3.1 Simple forecasting methods · 3.2 Transformations and adjustments (calendar, population,
inflation, Box-Cox, bias adjustment) · 3.3 Residual diagnostics (Box-Pierce, Ljung-Box,
`checkresiduals`) · 3.4 Evaluating forecast accuracy (MAE, RMSE, MAPE, sMAPE, MASE, `tsCV`) ·
3.5 Prediction intervals (normal, benchmark formulas, bootstrap, with transformations) ·
3.6 The forecast package in R · 3.7 Exercises

## Models tested ("Model testing" section, `tests/R/test-ch03.R`, `tests/python/test_ch03.py`)
| Model / tool | Data | R | Python |
|---|---|---|---|
| Mean | beer, goog, ibm, hsales | `meanf()` | `fpp.meanf()` |
| Naïve | beer, goog, ibm, hsales, WWWusage | `naive()` / `rwf()` | `fpp.naive()` |
| Seasonal naïve (± Box-Cox) | beer, a10, hsales, visnights, bricksq | `snaive()` | `fpp.snaive()` |
| Drift (± log / Box-Cox, ± bias adjustment) | goog, a10, eggs, dowjones, ibm | `rwf(drift = TRUE)` | `fpp.rwf(drift=True)` |
| Bootstrapped naïve intervals | goog200 | `naive(bootstrap = TRUE)` | `fpp_ch03.naive_bootstrap()` |
| Automatic ETS (`forecast()` default) | ausbeer | `forecast()` | `statsforecast AutoETS` |
| Time series cross-validation | goog200 | `tsCV()` | `fpp.tsCV()` |

All benchmark forecasts, intervals, accuracy measures, Box-Cox λ values, Box-Pierce/Ljung-Box
statistics and tsCV results match R to ≥ 6 significant digits (`tests/python/test_ch03.py`).

## Data
`WWWusage` (base R) is stored in `data/WWWusage.csv` and loaded with `fpp_ch03.load_extra()`.
Exercises 3 and 8 use `retail.xlsx` (see Chapter 2).

## Run
Open `Forecasting-principles-and-practice.Rproj` in RStudio → open `ch03_notes.Rmd` → **Knit**.
Tests: `bash tests/run_tests.sh 3`.
