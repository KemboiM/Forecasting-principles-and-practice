# Project conventions

How every chapter in this repository is built, so all chapters look and work the same.

## Folder layout per chapter

```
Chapter-NN-Short-Title/
├── README.md                    # what the chapter covers, section map, models tested, how to run
├── chNN_notes.Rmd               # the teaching, section by section (R + Python for every step)
├── chNN_exercises.Rmd           # every end-of-chapter exercise solved (R + Python)
├── chNN_notes_python.ipynb      # auto-generated Python-only version (python/rmd_to_ipynb.py)
└── chNN_exercises_python.ipynb  # auto-generated Python-only version
tests/R/test-chNN.R              # testthat: fits + forecasts EVERY model in the chapter
tests/python/test_chNN.py        # pytest: same models in Python, checked against R reference values
tests/reference/make_reference_chNN.R  # writes tests/reference/chNN_reference.json for the Python tests
```

## Inside each .Rmd

* YAML: `html_document`, `toc: true`, `toc_float: true`, `code_folding: show`, `df_print: kable`.
* First chunk `source("../R/setup.R")`; second chunk is a hidden Python setup chunk that puts
  `../python` on `sys.path` and imports `fpp`, numpy, pandas, matplotlib.
* An **Acknowledgement** block quoting the authors (Hyndman & Athanasopoulos, FPP2, OTexts 2018).
* `# N.k Section title` headings follow the book's sections **in order**.
* Explanations are short and in my own words (no copying of book text).
* Every code step is a heading with `{.tabset .tabset-pills}` and two sub-headings `#### R` and
  `#### Python`, each holding one chunk. Python chunk labels end in `-py`.
* Python plots end with `plt.show()`; assign unused plot return values to a named variable such as `ax_` (never `_`: under reticulate a `_ =` assignment silences the output of every later Python chunk).
* The notes end with **"Model testing — every model used in this chapter"**: every model the
  chapter teaches is fitted on a training set and its test-set accuracy reported in both languages,
  followed by a short interpretation, including any R-vs-Python differences.
* Exercises: one `# Exercise k` heading per book exercise, paraphrased question in italics,
  then R + Python tabs and a written answer/interpretation.

## Python engines used to mirror R

| R (forecast / fpp2) | Python |
|---|---|
| `meanf`, `naive`, `snaive`, `rwf`, `accuracy`, `tsCV`, `BoxCox`, `checkresiduals` | `python/fpp.py` (numerically identical, verified in `tests/python/test_fpp_core.py`) |
| `ets()` | `statsforecast.models.AutoETS` (selection) · `statsmodels ETSModel` (simulation, fixed models) |
| `ses`, `holt`, `hw` | `statsmodels.tsa.holtwinters` / `ETSModel` |
| `auto.arima`, `Arima` | `statsforecast.models.AutoARIMA` / `ARIMA` · `statsmodels SARIMAX` |
| `tslm`, `lm` | `statsmodels.formula.api.ols` |
| `stl`, `mstl`, `decompose`, `seas` | `statsmodels STL`, `MSTL`, `seasonal_decompose`, `x13_arima_analysis` |
| `ur.kpss`, `ndiffs`, `nsdiffs` | `statsmodels kpss`, `pmdarima ndiffs/nsdiffs` |
| `VAR` (vars) | `statsmodels.tsa.api.VAR` |
| `nnetar` | `sklearn MLPRegressor` on lagged inputs |
| `tbats` | `statsforecast AutoTBATS` / `tbats` |
| `hts`, `gts` | `hierarchicalforecast` |

Datasets come from `data/*.csv`, exported from the R packages by `R/export_data.R`, so both
languages use exactly the same numbers (`fpp.load("a10")` == `fpp2::a10`).
