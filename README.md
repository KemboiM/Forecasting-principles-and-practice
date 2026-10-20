# Forecasting: Principles and Practice — my study companion in R and Python

This repository holds my notes and exercise solutions, chapter by chapter, as I work through

> **Hyndman, R.J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.).
> OTexts: Melbourne, Australia.** <https://otexts.com/fpp2/>

## Acknowledgement

I am **following the authors, Rob J Hyndman and George Athanasopoulos**, and their book's teaching.
The structure, the methods, the datasets (R packages `fpp2`, `fma`, `expsmooth`, `forecast`) and
the exercises all come from them. The explanations here are my own summaries of their teaching. The
Python versions are my translations of their R code. The goal is to learn each method in both
languages and to check that the two agree. Please read and cite the original book, which is free
online. This repo does not contain the book itself.

## What's inside

Every chapter folder has:

| File | Contents |
|---|---|
| `chNN_notes.Rmd` | The chapter's teaching, section by section. **Every step has an R tab and a Python tab.** Ends with a *Model testing* section that fits **every model in the chapter** and compares their test-set accuracy. |
| `chNN_exercises.Rmd` | Every end-of-chapter exercise, solved in **R and Python**, with interpretation. |
| `chNN_*_python.ipynb` | Python-only notebooks generated from the `.Rmd` files, for Jupyter or VS Code. |
| `README.md` | Section map and the list of models tested. |

| Chapter | Topic |
|---|---|
| [1](Chapter-01-Getting-Started) | Getting started |
| [2](Chapter-02-Time-Series-Graphics) | Time series graphics |
| [3](Chapter-03-The-Forecasters-Toolbox) | The forecaster's toolbox |
| [4](Chapter-04-Judgmental-Forecasts) | Judgmental forecasts |
| [5](Chapter-05-Time-Series-Regression) | Time series regression models |
| [6](Chapter-06-Time-Series-Decomposition) | Time series decomposition |
| [7](Chapter-07-Exponential-Smoothing) | Exponential smoothing |
| [8](Chapter-08-ARIMA-Models) | ARIMA models |
| [9](Chapter-09-Dynamic-Regression) | Dynamic regression models |
| [10](Chapter-10-Hierarchical-Grouped-Series) | Forecasting hierarchical or grouped time series |
| [11](Chapter-11-Advanced-Forecasting-Methods) | Advanced forecasting methods |
| [12](Chapter-12-Practical-Forecasting-Issues) | Some practical forecasting issues |

Shared pieces:

```
R/setup.R             # sourced by every chapter (packages, knitr options, R<->Python bridge)
R/install_packages.R  # run once: installs every R package the book uses
R/setup_python.R      # run once: creates the "fpp" Python virtualenv used from RStudio
R/export_data.R       # exports all book datasets to data/*.csv so Python uses identical data
python/fpp.py         # Python versions of forecast-package tools (naive, snaive, rwf, accuracy, tsCV, BoxCox, ...)
python/rmd_to_ipynb.py# builds the Python-only notebooks from the .Rmd files
data/                 # every fpp2 / fma / expsmooth dataset as CSV (+ data/external for exercise files)
tests/                # automated tests: every model in every chapter, in R (testthat) and Python (pytest)
docs/CONVENTIONS.md   # how chapters are built
```

## Getting started (RStudio)

1. Open **`Forecasting-principles-and-practice.Rproj`** in RStudio.
2. Install the R packages once: `source("R/install_packages.R")`
3. Set up Python once: `source("R/setup_python.R")`, then *Session → Restart R*.
4. Open a chapter's `chNN_notes.Rmd` and **Knit**, or run it chunk by chunk. The `{python}` chunks run through `reticulate`.
5. (Optional) To rebuild the CSV data from the R packages: `source("R/export_data.R")`.

## Testing every model

```bash
bash tests/run_tests.sh        # all chapters, R then Python
bash tests/run_tests.sh 7      # one chapter
```

In RStudio: `source("tests/run_tests.R")`.

The Python tests also compare results with reference values produced by R (`tests/reference/`).
The comparison is strict where both languages use the same algorithm, such as benchmark methods,
regressions and fixed ARIMA models. Where the optimisers differ, the tests check structure instead
and the notes discuss the differences.

## Git workflow: one chapter at a time

I publish this repo one chapter at a time. Each chapter is one commit that contains its folder
plus its test files:

```bash
bash scripts/stage_chapter.sh 2      # stages Chapter-02-* and tests/*/*ch02* (nothing is committed or pushed)
git commit -m "Chapter 2: Time series graphics"
git push
```

In RStudio, open the **Terminal** tab and run the stage script, or tick the same files in the
**Git** tab. Then click *Commit* and *Push*. *Pull* before you start each session.

## Licence note

My code is free to reuse. The book's text, figures and data belong to the authors and their
publishers. See <https://otexts.com/fpp2/> for their terms.
