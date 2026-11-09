# Chapter 4 — Judgmental forecasts

Following Chapter 4 of *Forecasting: Principles and Practice* (2nd ed.) by
Rob J Hyndman & George Athanasopoulos — <https://otexts.com/fpp2/>.

| File | What it is |
|---|---|
| `ch04_notes.Rmd` | Section-by-section notes (4.1–4.7), every code step in **R** and **Python** |
| `ch04_exercises.Rmd` | The book has **no exercises** for this chapter — these are **my own practice exercises** (Delphi rounds, TFC bias by horizon, informed vs. uninformed adjustments, TFC discussion) |
| `ch04_notes_python.ipynb`, `ch04_exercises_python.ipynb` | Python-only versions (auto-generated) |
| `tfc_forecasts.csv` | Data behind the book's Figure 4.2 (domestic visitor nights, TFC / regression / ETS forecasts), from the authors' book source |

The chapter itself has no code; the code here illustrates each idea with real `fpp2` data where possible
and fixed (non-random) "expert" numbers where a panel is needed, so R and Python agree exactly.

## Sections
4.1 Beware of limitations · 4.2 Key principles · 4.3 The Delphi method · 4.4 Forecasting by analogy ·
4.5 Scenario forecasting · 4.6 New product forecasting · 4.7 Judgmental adjustments

## Models tested (notes → "Model testing", and `tests/*/test*ch04*`)
| Model / forecast | Data | R | Python |
|---|---|---|---|
| Naive ("anchoring"), seasonal naive | `a10` | `naive()`, `snaive()` | `fpp.naive()`, `fpp.snaive()` |
| ETS (automatic) | `a10`, `elecdaily` | `ets()` | `statsforecast AutoETS` |
| ETS(M,A,M) one-step forecast log | `h02` | `ets(model="MAM")` | `statsmodels ETSModel` |
| TFC judgmental vs. book's regression / ETS | visitor nights | given forecasts | given forecasts |
| Naive, drift, ETS(A,N,N), linear trend | visitor nights 1998–2004 | `naive`, `rwf`, `ets`, `tslm` | `fpp`, `AutoETS`, `statsmodels OLS` |
| Informed (temperature) adjustment of ETS | `elecdaily` | `lm()` on ETS errors | `smf.ols` on ETS errors |

## Run
Open the `.Rproj` in RStudio → open `ch04_notes.Rmd` → **Knit**. Tests: `bash tests/run_tests.sh 4`.
