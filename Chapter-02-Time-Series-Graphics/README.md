# Chapter 2 — Time series graphics

Following Chapter 2 of *Forecasting: Principles and Practice* (2nd ed.) by
Rob J Hyndman & George Athanasopoulos — <https://otexts.com/fpp2/>.

| File | What it is |
|---|---|
| `ch02_notes.Rmd` | Section-by-section notes (2.1–2.9), every step in **R** and **Python** |
| `ch02_exercises.Rmd` | All 10 exercises of Section 2.10 with answers (R + Python) |
| `ch02_notes_python.ipynb`, `ch02_exercises_python.ipynb` | Python-only versions (auto-generated) |

## Sections
2.1 `ts` objects · 2.2 Time plots · 2.3 Time series patterns · 2.4 Seasonal plots ·
2.5 Seasonal subseries plots · 2.6 Scatterplots (correlation, Anscombe, scatterplot matrices) ·
2.7 Lag plots · 2.8 Autocorrelation · 2.9 White noise · 2.10 Exercises

## Graphics: R ↔ Python
| R (`forecast` / `ggplot2`) | Python |
|---|---|
| `autoplot()` | `fpp.autoplot()` |
| `ggseasonplot()` (incl. `polar = TRUE`) | `fpp.seasonplot()` |
| `ggsubseriesplot()` | `fpp.subseriesplot()` |
| `gglagplot()` | `fpp.lagplot()` |
| `ggAcf()` / `Acf()` | `fpp.acf_plot()` / `statsmodels acf()` |
| `GGally::ggpairs()` | matplotlib scatter matrix (in the notes) |

## Models tested ("Model testing" section, `tests/R/test-ch02.R`, `tests/python/test_ch02.py`)
Chapter 2 introduces one model — **white noise**. It is tested (ACF bounds ± 2/√T and Ljung-Box)
on simulated noise, `beer2`, `aelec`, `a10`, `diff(goog)`, `ustreas`, `mypigs` and `diff(dj)`.
Python ACF values, correlations and Ljung-Box statistics match R exactly.

## Data
`tute1.csv` (Exercise 2) is in `data/external/`. `retail.xlsx` (Exercise 3) is downloaded
automatically into `data/external/` from <https://otexts.com/fpp2/extrafiles/retail.xlsx>; if the
download fails, save it there yourself — the retail chunks are skipped while it is missing.

## Run
Open `Forecasting-principles-and-practice.Rproj` in RStudio → open `ch02_notes.Rmd` → **Knit**.
Tests: `bash tests/run_tests.sh 2`.
