# Chapter 7 — Exponential smoothing

Following Chapter 7 of *Forecasting: Principles and Practice* (2nd ed.) by
Rob J Hyndman & George Athanasopoulos — <https://otexts.com/fpp2/>.

| File | What it is |
|---|---|
| `ch07_notes.Rmd` | Section-by-section notes (7.1–7.7), every step in **R** and **Python** |
| `ch07_exercises.Rmd` | All 17 exercises of Section 7.8 with answers in R and Python |
| `ch07_notes_python.ipynb`, `ch07_exercises_python.ipynb` | Python-only versions (auto-generated) |
| `../python/fpp_ch07.py` | Python `ets()`, `ses()`, `holt()`, `hw()`, `simulate()` mirroring R's `forecast` package |

## Sections
7.1 Simple exponential smoothing · 7.2 Trend methods (Holt, damped) · 7.3 Holt-Winters' seasonal method ·
7.4 A taxonomy of exponential smoothing methods · 7.5 Innovations state space models ·
7.6 Estimation and model selection · 7.7 Forecasting with ETS models · 7.8 Exercises

## Models tested (notes → "Model testing" section, and `tests/*/test*ch07*`)
| Model / method | Data | R | Python |
|---|---|---|---|
| SES = ETS(A,N,N) | `oil`, `pigs`, `books`, `livestock` | `ses()` | `c7.ses()` |
| Holt's linear trend | `ausair`, `books`, `livestock`, `eggs` | `holt()` | `c7.holt()` |
| Damped Holt (fixed / estimated φ) | `ausair`, `livestock`, `eggs` | `holt(damped=TRUE)` | `c7.holt(damped=True)` |
| Exponential trend | `ausair`, `livestock`, `eggs` | `holt(exponential=TRUE)`, ETS(M,M,N) | `c7.holt(exponential=True)` |
| Holt-Winters additive / multiplicative / damped | `austourists`, `hyndsight`, `visitors`, retail | `hw()` | `c7.hw()` |
| All 15 admissible ETS models + automatic selection | `austourists` and many more | `ets()` | `c7.ets()` (statsforecast's port of `forecast::ets`) |
| Simulated / bootstrapped prediction intervals | `austourists` | `forecast(simulate=TRUE, bootstrap=TRUE)` | `c7.simulate()` |
| STL + ETS, Box-Cox ETS | `ukcars`, `visitors`, retail | `stlf()`, `ets(lambda=)` | `c6.stlf()`, `c7.ets(lambda_=)` |
| Benchmarks | all | `naive()`, `meanf()`, `snaive()` | `fpp.*` |

**R vs Python.** Fully specified non-seasonal models match R closely (parameters to ~3
digits, forecasts to ~4). For seasonal models the likelihood often has several local optima;
`fpp_ch07.ets()` polishes statsforecast's estimate with a multi-start search, which reaches an
MSE/likelihood at least as good as R's but sometimes at different parameter values — so
forecasts can differ by a few percent, and automatic selection can occasionally pick a
different model. The notes point these cases out.

## Run
Open the `.Rproj` in RStudio → open `ch07_notes.Rmd` → **Knit** (the exercises take a few
minutes because of time-series cross-validation). Tests: `bash tests/run_tests.sh 7`.
Exercises 8–9 need `data/external/retail.xlsx` (downloaded automatically from the book's site if missing).
