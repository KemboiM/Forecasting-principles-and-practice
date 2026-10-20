# Chapter 1 — Getting started

Following Chapter 1 of *Forecasting: Principles and Practice* (2nd ed.) by
Rob J Hyndman & George Athanasopoulos  <https://otexts.com/fpp2/>.

| File | What it is |
|---|---|
| `ch01_notes.Rmd` | Section-by-section notes (1.1–1.7), every step in R and Python |
| `ch01_exercises.Rmd` | Exercises 1.8 (both) with answers, plus a `melsyd` analysis in R and Python |
| `ch01_notes_python.ipynb` `ch01_exercises_python.ipynb` | Python-only versions |

## Sections
1.1 What can be forecast? · 1.2 Forecasting, planning and goals · 1.3 Determining what to forecast ·
1.4 Forecasting data and methods · 1.5 Some case studies · 1.6 The basic steps in a forecasting task ·
1.7 The statistical forecasting perspective · 1.8 Exercises

## Models tested
| Model | Data | R | Python |
|---|---|---|---|
| Explanatory regression | `elecdaily` | `tslm()` | `statsmodels ols` |
| Time series model (ETS) | `elecdaily`, `ausbeer`, `austa` | `ets()` | `statsforecast AutoETS` |
| Mixed model (regression + ARIMA errors) | `elecdaily` | `auto.arima(xreg=)` | `statsforecast AutoARIMA(X=)` |
| Seasonal naïve benchmark | `elecdaily`, `ausbeer` | `snaive()` | `fpp.snaive()` |
| Simulated futures | `austa` | `simulate(ets)` | `ETSModel.simulate()` |

## Run
Open `Forecasting-principles-and-practice.Rproj` in RStudio → open `ch01_notes.Rmd` → **Knit**
(or run chunk by chunk). Tests: `bash tests/run_tests.sh 1`.
