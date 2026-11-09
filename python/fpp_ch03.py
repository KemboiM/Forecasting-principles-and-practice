"""
fpp_ch03.py -- extra Python helpers used in Chapter 3 (The forecaster's toolbox).

* load_extra()       : base-R datasets saved as data/<name>.csv but not in _metadata.csv (e.g. WWWusage)
* monthdays()        : number of days in each month (forecast::monthdays)
* smape()            : symmetric MAPE (Section 3.4)
* box_pierce()       : Box-Pierce portmanteau test (Box.test(type = "Box-Pierce"))
* naive_bootstrap()  : naive forecasts with bootstrapped prediction intervals (naive(y, bootstrap=TRUE))
* plot_forecasts()   : overlay several forecasts on one plot (autoplot + autolayer)
* accuracy_table()   : test-set accuracy of several forecasts
* coverage()         : empirical coverage of a prediction interval on a test set
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fpp


def load_extra(name: str) -> pd.Series:
    """Load data/<name>.csv (annual-style index starting at its first time value)."""
    df = pd.read_csv(fpp.data_dir() / f"{name}.csv")
    return fpp.make_ts(df["value"].to_numpy(), start=float(df["time"].iloc[0]), frequency=1, name=name)


def monthdays(y: pd.Series) -> np.ndarray:
    """Days in each month (monthly PeriodIndex) -- like forecast::monthdays()."""
    return np.asarray(y.index.days_in_month, dtype=float)


def smape(actual, forecast) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    return float(np.nanmean(200 * np.abs(a - f) / (a + f)))


def box_pierce(resid, lag: int = 10, df: int = 0) -> pd.Series:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    e = np.asarray(resid, dtype=float)
    e = e[~np.isnan(e)]
    r = acorr_ljungbox(e, lags=[lag], model_df=df, boxpierce=True, return_df=True)
    return pd.Series({"Q": float(r["bp_stat"].iloc[0]), "df": lag - df, "p-value": float(r["bp_pvalue"].iloc[0])})


def naive_bootstrap(y: pd.Series, h: int = 10, npaths: int = 5000, level=(80, 95), seed: int = 0) -> fpp.Forecast:
    """Naive forecasts with bootstrapped PIs: y*_{T+k} = y*_{T+k-1} + e*, e* resampled from the residuals."""
    rng = np.random.default_rng(seed)
    base = fpp.naive(y, h=h, level=level)
    res = base.residuals.dropna().to_numpy()
    draws = rng.choice(res, size=(npaths, h), replace=True)
    paths = y.iloc[-1] + np.cumsum(draws, axis=1)
    idx = base.mean.index
    lower = pd.DataFrame({lv: np.quantile(paths, 0.5 - lv / 200, axis=0, method="median_unbiased") for lv in level}, index=idx)
    upper = pd.DataFrame({lv: np.quantile(paths, 0.5 + lv / 200, axis=0, method="median_unbiased") for lv in level}, index=idx)
    return fpp.Forecast("Naive method (bootstrap)", y, base.mean, lower, upper, base.fitted, base.residuals, tuple(level))




def plot_forecasts(y, forecasts: dict, title: str = "", xlab: str = "Time", ylab: str = "",
                   pi: bool = False, ax=None):
    """Overlay several Forecast objects on one plot (autoplot(y) + autolayer(fc, PI=FALSE) ...)."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(fpp.ts_time(y), y.to_numpy(), color="black", lw=1)
    for k, (name, fc) in enumerate(forecasts.items()):
        m = fc.mean.copy()
        m.attrs["frequency"] = fpp.frequency(fc.x)
        t = fpp.ts_time(m)
        if pi:
            for lv, a in zip(sorted(fc.level, reverse=True), [0.15, 0.3]):
                ax.fill_between(t, fc.lower[lv], fc.upper[lv], color=f"C{k}", alpha=a, lw=0)
        ax.plot(t, m.to_numpy(), color=f"C{k}", lw=1.6, label=name)
    ax.legend(title="Forecast")
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    return ax


def accuracy_table(forecasts: dict, test, cols=("RMSE", "MAE", "MAPE", "MASE")) -> pd.DataFrame:
    """Test-set accuracy table for several Forecast objects (rows = methods)."""
    return pd.DataFrame({k: fpp.accuracy(fc, test).loc["Test set", list(cols)] for k, fc in forecasts.items()}).T


def coverage(fc: "fpp.Forecast", test, level: int) -> float:
    """Share of test observations inside the level% prediction interval."""
    x = np.asarray(test, dtype=float)
    k = min(len(x), len(fc.mean))
    lo = fc.lower[level].to_numpy()[:k]
    hi = fc.upper[level].to_numpy()[:k]
    return float(np.mean((x[:k] >= lo) & (x[:k] <= hi)))
