# Chapter 6 — Time series decomposition

Following Chapter 6 of *Forecasting: Principles and Practice* (2nd ed.) by
Rob J Hyndman & George Athanasopoulos — <https://otexts.com/fpp2/>.

| File | What it is |
|---|---|
| `ch06_notes.Rmd` | Section-by-section notes (6.1–6.8), every step in **R** and **Python** |
| `ch06_exercises.Rmd` | All 8 exercises of Section 6.9 with answers in R and Python |
| `ch06_notes_python.ipynb`, `ch06_exercises_python.ipynb` | Python-only versions (auto-generated) |
| `../python/fpp_ch06.py` | Python versions of `ma()`, `decompose()`, `stl()`, `mstl()`, `seas()`, `stlf()` with R's defaults |

## Sections
6.1 Time series components · 6.2 Moving averages · 6.3 Classical decomposition ·
6.4 X11 decomposition · 6.5 SEATS decomposition · 6.6 STL decomposition ·
6.7 Measuring strength of trend and seasonality · 6.8 Forecasting with decomposition · 6.9 Exercises

## Models tested (notes → "Model testing" section, and `tests/*/test*ch06*`)
| Model / method | Data | R | Python |
|---|---|---|---|
| m-MA, 2×m-MA, weighted MA | `elecsales`, `ausbeer`, `elecequip` | `ma()` | `c6.ma()`, `c6.weighted_ma()` |
| Classical additive / multiplicative | `elecequip`, `plastics` | `decompose()` | `c6.classical()` (= statsmodels `seasonal_decompose`) |
| X11 | `elecequip`, `cangas`, retail | `seas(x11 = "")` | `c6.x13(method="x11")` (same X-13 binary) |
| SEATS | `elecequip`, `cangas` | `seas()` | `c6.x13(method="seats")` |
| STL (periodic / flexible / robust), mstl | `elecequip`, `cangas`, `bricksq` | `stl()`, `mstl()` | `c6.stl_r()`, `c6.mstl_r()` (statsmodels STL with R's settings) |
| Decomposition + naive / drift / ETS / ARIMA | `elecequip` | `forecast(stl, method=)` | `c6.forecast_stl()` |
| `stlf()` (incl. Box-Cox) | `elecequip`, `bricksq`, `writing`, `fancy` | `stlf()` | `c6.stlf()` |
| Benchmarks | `elecequip`, `bricksq` | `snaive()`, `naive()` | `fpp.snaive()`, `fpp.naive()` |

Classical, STL, X11 and SEATS components agree with R to ≥ 6 significant digits.

## Run
Open the `.Rproj` in RStudio → open `ch06_notes.Rmd` → **Knit**. Tests: `bash tests/run_tests.sh 6`.
X11/SEATS in Python use the X-13ARIMA-SEATS program installed by R's `x13binary` package
(found automatically via `X13PATH` or `Rscript`).
Exercise 3 needs `data/external/retail.xlsx` (downloaded automatically from the book's site if missing).
