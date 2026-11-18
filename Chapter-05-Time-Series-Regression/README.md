# Chapter 5 — Time series regression models

Following Chapter 5 of *Forecasting: Principles and Practice* (2nd ed.) by
Rob J Hyndman & George Athanasopoulos — <https://otexts.com/fpp2/>.

| File | What it is |
|---|---|
| `ch05_notes.Rmd` | Section-by-section notes (5.1–5.9), every step in **R** and **Python** |
| `ch05_exercises.Rmd` | All exercises of Section 5.10 (1–8, every sub-part) in R and Python |
| `ch05_notes_python.ipynb`, `ch05_exercises_python.ipynb` | Python-only versions (auto-generated) |

Python uses `python/fpp_ch05.py`, which rebuilds the predictors `tslm()` creates (`trend`, `season`,
`fourier()`), plus `forecast.lm()`, `CV()`, `bgtest()`, `easter()` and `splinef()`. Coefficients,
forecasts and prediction intervals match R to machine precision (see `tests/python/test_ch05.py`).

## Sections
5.1 The linear model · 5.2 Least squares estimation · 5.3 Evaluating the regression model ·
5.4 Some useful predictors · 5.5 Selecting predictors · 5.6 Forecasting with regression ·
5.7 Matrix formulation · 5.8 Nonlinear regression · 5.9 Correlation, causation and forecasting · 5.10 Exercises

## Models tested (notes → "Model testing", and `tests/*/test*ch05*`)
| Model | Data | R | Python |
|---|---|---|---|
| Simple / multiple / best-subset regression, scenarios, lagged predictors | `uschange` | `tslm()`, `CV()` | `c5.tslm()`, `c5.CV()`, `smf.ols` |
| Spurious regression | `ausair` ~ `guinearice` | `tslm()`, `bgtest()` | `c5.tslm()`, `c5.bgtest()` |
| Trend + seasonal dummies, Fourier (harmonic) regression | `ausbeer`, `gasoline` | `tslm(~trend+season)`, `fourier()` | `c5.tslm(trend_, season, K)` |
| Step / spike interventions (my example) | `Seatbelts` | `tslm()` | `c5.tslm()` |
| Distributed lags | `insurance` | `lm()` | `smf.ols` |
| Linear, exponential, piecewise, cubic-spline trends, smoothing spline | `marathon`, `huron`, `mens400` | `tslm()`, `splinef()` | `c5.tslm()`, `c5.splinef()` |
| Log model with trend, season, surf dummy | `fancy` | `tslm(lambda=0)` | `c5.tslm(lambda_=0)` |
| Matrix formulation (by hand) | `uschange`, `huron` | `solve()` | `numpy.linalg` |

## Run
Open the `.Rproj` in RStudio → open `ch05_notes.Rmd` → **Knit**. Tests: `bash tests/run_tests.sh 5`.
