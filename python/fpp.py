"""
fpp.py -- Python companion helpers for
"Forecasting: Principles and Practice" (2nd ed.) by Rob J Hyndman & George Athanasopoulos.

The book teaches with R (packages `forecast` and `fpp2`). This module provides the
small set of building blocks that are needed so every R example can be reproduced
in Python with *the same data and the same formulas*:

    load()                       -> read any fpp2 / fma / expsmooth dataset (exported by R/export_data.R)
    window(), ts_time()          -> R-style time windows on pandas objects
    meanf(), naive(), snaive(),
    rwf(drift=True)              -> the benchmark methods of Chapter 3 (with prediction intervals)
    accuracy()                   -> ME, RMSE, MAE, MPE, MAPE, MASE, ACF1  (same definitions as forecast::accuracy)
    tsCV()                       -> time-series cross-validation errors (same as forecast::tsCV)
    box_cox(), inv_box_cox(),
    box_cox_lambda()             -> Box-Cox transformation and Guerrero's lambda (same as forecast::BoxCox.lambda)
    checkresiduals()             -> residual diagnostics (time plot, ACF, histogram, Ljung-Box test)
    autoplot(), seasonplot(), subseriesplot(), lagplot(), acf_plot(), plot_forecast()

Everything else (ETS, ARIMA, regression, STL, VAR, ...) uses standard libraries
(statsmodels, statsforecast, pmdarima, scikit-learn) directly inside the chapter notes,
so that you learn the Python ecosystem rather than a wrapper.

This module is part of a personal study project that follows the book chapter by chapter.
All credit for the teaching material goes to the authors: https://otexts.com/fpp2/
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = [
    "data_dir", "load", "list_datasets", "frequency", "ts_time", "window", "make_ts",
    "Forecast", "meanf", "naive", "snaive", "rwf", "accuracy", "tsCV",
    "box_cox", "inv_box_cox", "box_cox_lambda", "ljung_box", "checkresiduals",
    "autoplot", "seasonplot", "subseriesplot", "lagplot", "acf_plot", "plot_forecast",
]

ArrayLike = Union[pd.Series, np.ndarray, Sequence[float]]


# =============================================================================
# Data loading
# =============================================================================
def data_dir() -> Path:
    """Locate the project's `data/` folder (env var FPP_DATA overrides)."""
    env = os.environ.get("FPP_DATA")
    if env:
        return Path(env)
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "_metadata.csv").exists():
            return p / "data"
    mod = Path(__file__).resolve().parent
    for p in [mod, *mod.parents]:
        if (p / "data" / "_metadata.csv").exists():
            return p / "data"
    raise FileNotFoundError(
        "Could not find data/_metadata.csv. Run R/export_data.R from the project root "
        "or set the FPP_DATA environment variable.")


_META: Optional[pd.DataFrame] = None


def _meta() -> pd.DataFrame:
    global _META
    if _META is None:
        _META = pd.read_csv(data_dir() / "_metadata.csv").set_index("name")
    return _META


def list_datasets(package: Optional[str] = None) -> pd.DataFrame:
    """Table of all available datasets (optionally only one R package)."""
    m = _meta().reset_index()
    return m if package is None else m[m["package"] == package]


def _make_index(df: pd.DataFrame, freq: float, start: float) -> pd.Index:
    years = df["year"].astype(int).to_numpy()
    cyc = df["cycle"].astype(int).to_numpy()
    if np.isclose(freq, 12):
        return pd.PeriodIndex([pd.Period(year=y, month=c, freq="M") for y, c in zip(years, cyc)], name="time")
    if np.isclose(freq, 4):
        return pd.PeriodIndex([pd.Period(year=y, quarter=c, freq="Q") for y, c in zip(years, cyc)], name="time")
    if np.isclose(freq, 1) and float(start).is_integer():
        return pd.PeriodIndex([pd.Period(year=y, freq="Y") for y in years], name="time")
    return pd.Index(df["time"].to_numpy(), name="time")


def make_ts(values: ArrayLike, start: Union[float, tuple] = 1, frequency: float = 1,
            name: Optional[str] = None) -> pd.Series:
    """Python analogue of R's ts(values, start=, frequency=).

    start can be a number (e.g. 1990) or (year, period) tuple (e.g. (1990, 3)).
    """
    values = np.asarray(values, dtype=float)
    if isinstance(start, tuple):
        start_t = start[0] + (start[1] - 1) / frequency
    else:
        start_t = float(start)
    t = start_t + np.arange(len(values)) / frequency
    year = np.floor(t + 1e-8).astype(int)
    cyc = np.rint((t - year) * frequency).astype(int) + 1
    df = pd.DataFrame({"time": t, "year": year, "cycle": cyc})
    idx = _make_index(df, frequency, start_t)
    s = pd.Series(values, index=idx, name=name or "value")
    s.attrs.update(frequency=frequency, start=start_t, name=name or "value")
    return s


def load(name: str) -> Union[pd.Series, pd.DataFrame]:
    """Load an fpp2/fma/expsmooth/forecast dataset exactly as R sees it.

    * univariate ts  -> pd.Series
    * multivariate ts-> pd.DataFrame
    * data.frame     -> pd.DataFrame
    Monthly / quarterly / annual series get a pandas PeriodIndex; any other
    frequency (weekly 52.18, daily 7, half-hourly 48 ...) gets the R decimal time.
    The seasonal period is stored in `obj.attrs["frequency"]`.
    """
    meta = _meta()
    if name not in meta.index:
        raise KeyError(f"Unknown dataset '{name}'. See fpp.list_datasets().")
    row = meta.loc[name]
    df = pd.read_csv(data_dir() / f"{name}.csv")
    if row["type"] != "ts":
        df.attrs.update(name=name)
        return df
    freq = float(row["frequency"])
    idx = _make_index(df, freq, float(row["start"]))
    cols = str(row["columns"]).split(";")
    out = df[cols].copy()
    out.index = idx
    if len(cols) == 1:
        out = out.iloc[:, 0].rename(name)
    msts = row.get("msts")
    msts = [float(v) for v in str(msts).split(";")] if isinstance(msts, str) and msts else None
    out.attrs.update(frequency=freq, start=float(row["start"]), name=name, msts=msts,
                     time=tuple(df["time"].to_numpy().tolist()), cycle=tuple(df["cycle"].to_numpy().tolist()))
    # (tuples, not numpy arrays, so pandas can compare attrs in concat/melt)
    return out


def frequency(x) -> float:
    """Seasonal period m of a series (R's frequency())."""
    f = getattr(x, "attrs", {}).get("frequency") if hasattr(x, "attrs") else None
    if f:
        return float(f)
    if isinstance(getattr(x, "index", None), pd.PeriodIndex):
        return {"M": 12.0, "Q": 4.0, "Y": 1.0, "A": 1.0}.get(x.index.freqstr[0], 1.0)
    return 1.0


def ts_time(x) -> np.ndarray:
    """Decimal time (R's time(x)) for a series loaded with load()/make_ts()."""
    idx = x.index
    if isinstance(idx, pd.PeriodIndex):
        f = frequency(x)
        if f == 12:
            return idx.year + (idx.month - 1) / 12
        if f == 4:
            return idx.year + (idx.quarter - 1) / 4
        return idx.year.astype(float).to_numpy()
    return np.asarray(idx, dtype=float)


def _time_of(t, f) -> float:
    if t is None:
        return None
    if isinstance(t, tuple):
        return t[0] + (t[1] - 1) / f
    return float(t)


def window(x, start=None, end=None):
    """R's window(x, start=, end=). start/end: number (e.g. 1995) or (year, period)."""
    f = frequency(x)
    t = ts_time(x)
    s = _time_of(start, f)
    e = _time_of(end, f)
    mask = np.ones(len(t), dtype=bool)
    if s is not None:
        mask &= t >= s - 1e-8
    if e is not None:
        mask &= t <= e + 1e-8
    out = x[mask]
    out.attrs = dict(x.attrs)
    out.attrs["start"] = float(t[mask][0]) if mask.any() else None
    for k in ("time", "cycle"):
        if k in x.attrs and x.attrs[k] is not None and len(x.attrs[k]) == len(t):
            out.attrs[k] = tuple(np.asarray(x.attrs[k])[mask].tolist())
    return out


def _future_index(x, h: int) -> pd.Index:
    idx = x.index
    if isinstance(idx, pd.PeriodIndex):
        return pd.period_range(idx[-1] + 1, periods=h, freq=idx.freq, name=idx.name)
    t = ts_time(x)
    f = frequency(x)
    return pd.Index(t[-1] + np.arange(1, h + 1) / f, name="time")


# =============================================================================
# Forecast objects and benchmark methods (Chapter 3)
# =============================================================================
@dataclass
class Forecast:
    """Mimics R's `forecast` object."""
    method: str
    x: pd.Series
    mean: pd.Series
    lower: pd.DataFrame
    upper: pd.DataFrame
    fitted: pd.Series
    residuals: pd.Series
    level: Sequence[int] = (80, 95)
    extra: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        out = pd.DataFrame({"Point Forecast": self.mean})
        for lv in self.level:
            out[f"Lo {lv}"] = self.lower[lv]
            out[f"Hi {lv}"] = self.upper[lv]
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"Forecasts from {self.method}\n{self.summary()}"


def _as_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.astype(float)
    return make_ts(np.asarray(y, dtype=float))


def _pi(mean: np.ndarray, se: np.ndarray, level, index, df: Optional[int] = None):
    lower, upper = {}, {}
    for lv in level:
        q = stats.t.ppf(0.5 + lv / 200, df) if df else stats.norm.ppf(0.5 + lv / 200)
        lower[lv] = mean - q * se
        upper[lv] = mean + q * se
    return pd.DataFrame(lower, index=index), pd.DataFrame(upper, index=index)


def _finish(method, y, mean, se, fitted, level, h, df=None, lam=None, biasadj=False):
    idx = _future_index(y, h)
    lo, hi = _pi(mean, se, level, idx, df)
    fitted = pd.Series(fitted, index=y.index)
    if lam is not None:
        yt = box_cox(y, lam)
        resid = yt - fitted
        if biasadj:
            mean_bt = inv_box_cox(mean, lam, biasadj=True, fvar=se ** 2)
        else:
            mean_bt = inv_box_cox(mean, lam)
        lo = lo.apply(lambda c: inv_box_cox(c.to_numpy(), lam))
        hi = hi.apply(lambda c: inv_box_cox(c.to_numpy(), lam))
        fitted = pd.Series(inv_box_cox(fitted.to_numpy(), lam), index=y.index)
        mean = mean_bt
    else:
        resid = y - fitted
    return Forecast(method, y, pd.Series(mean, index=idx, name="mean"), lo, hi,
                    fitted, pd.Series(resid, index=y.index), tuple(level))


def meanf(y, h: int = 10, level=(80, 95), lambda_=None, biasadj=False) -> Forecast:
    """Average method: all future values equal the historical mean (forecast::meanf)."""
    y = _as_series(y)
    z = box_cox(y, lambda_).to_numpy() if lambda_ is not None else y.to_numpy()
    n = np.sum(~np.isnan(z))
    mu = np.nanmean(z)
    s = np.nanstd(z, ddof=1)
    se = np.repeat(s * math.sqrt(1 + 1 / n), h)
    return _finish("Mean", y, np.repeat(mu, h), se, np.repeat(mu, len(z)), level, h,
                   df=n - 1, lam=lambda_, biasadj=biasadj)


def rwf(y, h: int = 10, drift: bool = False, level=(80, 95), lambda_=None, biasadj=False) -> Forecast:
    """Random walk (naive) forecasts, optionally with drift (forecast::rwf)."""
    y = _as_series(y)
    z = box_cox(y, lambda_).to_numpy() if lambda_ is not None else y.to_numpy()
    n = len(z)
    d = np.diff(z)
    steps = np.arange(1, h + 1)
    if drift:
        slope = np.nanmean(d)
        k = np.sum(~np.isnan(d))
        sigma = np.sqrt(np.nansum((d - slope) ** 2) / (k - 1))
        mean = z[-1] + slope * steps
        se = sigma * np.sqrt(steps * (1 + steps / k))
        fitted = np.r_[np.nan, z[:-1] + slope]
        method = "Random walk with drift"
    else:
        sigma = np.sqrt(np.nanmean(d ** 2))
        mean = np.repeat(z[-1], h)
        se = sigma * np.sqrt(steps)
        fitted = np.r_[np.nan, z[:-1]]
        method = "Naive method"
    return _finish(method, y, mean, se, fitted, level, h, lam=lambda_, biasadj=biasadj)


def naive(y, h: int = 10, level=(80, 95), lambda_=None, biasadj=False) -> Forecast:
    """Naive method (forecast::naive): last observation carried forward."""
    return rwf(y, h=h, drift=False, level=level, lambda_=lambda_, biasadj=biasadj)


def snaive(y, h: Optional[int] = None, level=(80, 95), lambda_=None, biasadj=False) -> Forecast:
    """Seasonal naive method (forecast::snaive): value from the same season last year."""
    y = _as_series(y)
    m = int(round(frequency(y)))
    if h is None:
        h = 2 * m
    z = box_cox(y, lambda_).to_numpy() if lambda_ is not None else y.to_numpy()
    n = len(z)
    last = z[n - m:]
    mean = np.array([last[i % m] for i in range(h)])
    fitted = np.r_[np.repeat(np.nan, m), z[:-m]]
    e = z - fitted
    sigma = np.sqrt(np.nanmean(e ** 2))
    k = (np.arange(h)) // m + 1
    se = sigma * np.sqrt(k)
    return _finish("Seasonal naive method", y, mean, se, fitted, level, h,
                   lam=lambda_, biasadj=biasadj)


# =============================================================================
# Accuracy and cross-validation
# =============================================================================
def _acf1(e) -> float:
    e = np.asarray(e, dtype=float)
    e = e[~np.isnan(e)]
    if len(e) < 3:
        return np.nan
    e0 = e - e.mean()
    return float(np.sum(e0[1:] * e0[:-1]) / np.sum(e0 ** 2))


def _mase_scale(train, m: int) -> float:
    train = np.asarray(train, dtype=float)
    d = train[m:] - train[:-m]
    return float(np.nanmean(np.abs(d)))


def accuracy(f, x=None, train=None, m: Optional[int] = None) -> pd.DataFrame:
    """Same measures as forecast::accuracy().

    accuracy(fc)              -> training-set accuracy (from fc.residuals)
    accuracy(fc, test)        -> training + test set rows
    accuracy(np.array, test, train=y_train)  -> test-set accuracy of a plain vector of forecasts
    MASE is scaled by the in-sample seasonal-naive MAE (m = frequency, as in R).
    """
    rows = {}
    if isinstance(f, Forecast):
        ytrain = f.x
        fit = f.fitted
        mm = m or max(1, int(round(frequency(ytrain))))
        scale = _mase_scale(ytrain.to_numpy(), mm)
        e = (ytrain - fit).to_numpy()
        pe = 100 * e / ytrain.to_numpy()
        rows["Training set"] = _measures(e, pe, scale)
        fc = f.mean.to_numpy()
    else:
        fc = np.asarray(f, dtype=float)
        ytrain = train
        scale = None
        if train is not None:
            mm = m or max(1, int(round(frequency(train))))
            scale = _mase_scale(np.asarray(train, dtype=float), mm)
    if x is not None:
        xv = np.asarray(x, dtype=float)
        k = min(len(xv), len(fc))
        e = xv[:k] - fc[:k]
        pe = 100 * e / xv[:k]
        rows["Test set"] = _measures(e, pe, scale)
    return pd.DataFrame(rows).T[["ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "ACF1"]]


def _measures(e, pe, scale) -> dict:
    e = np.asarray(e, dtype=float)
    pe = np.asarray(pe, dtype=float)
    return {
        "ME": np.nanmean(e),
        "RMSE": np.sqrt(np.nanmean(e ** 2)),
        "MAE": np.nanmean(np.abs(e)),
        "MPE": np.nanmean(pe),
        "MAPE": np.nanmean(np.abs(pe)),
        "MASE": np.nanmean(np.abs(e)) / scale if scale else np.nan,
        "ACF1": _acf1(e),
    }


def tsCV(y, forecastfunction: Callable, h: int = 1, initial: int = 0, window: Optional[int] = None,
         **kwargs) -> pd.DataFrame:
    """Time series cross-validation (forecast::tsCV).

    forecastfunction(y_train, h=h, **kwargs) must return a Forecast (or array of h means).
    Returns an n x h DataFrame of errors; row t holds e_{t+k|t} for k=1..h.
    """
    y = _as_series(y)
    n = len(y)
    errors = np.full((n, h), np.nan)
    first = (window + initial) if window is not None else (initial + 1)  # same origins as R
    for i in range(first, n):
        start = 0 if window is None else max(0, i - window)
        ytrain = y.iloc[start:i]
        ytrain.attrs = dict(y.attrs)
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = forecastfunction(ytrain, h=h, **kwargs)
            mean = fc.mean.to_numpy() if isinstance(fc, Forecast) else np.asarray(fc, dtype=float)
        except Exception:
            continue
        for k in range(h):
            if i + k < n:
                errors[i - 1, k] = y.iloc[i + k] - mean[k]
    return pd.DataFrame(errors, index=y.index, columns=[f"h={k + 1}" for k in range(h)])


# =============================================================================
# Box-Cox transformations
# =============================================================================
def box_cox(y, lam: float):
    """Box-Cox transform as in forecast::BoxCox (sign-preserving for lambda != 0)."""
    is_series = isinstance(y, pd.Series)
    z = y.to_numpy(dtype=float) if is_series else np.asarray(y, dtype=float)
    if lam == 0:
        out = np.log(z)
    else:
        out = (np.sign(z) * np.abs(z) ** lam - 1) / lam
    if is_series:
        s = pd.Series(out, index=y.index, name=y.name)
        s.attrs = dict(y.attrs)
        return s
    return out


def inv_box_cox(z, lam: float, biasadj: bool = False, fvar=None):
    """Inverse Box-Cox (forecast::InvBoxCox), with optional bias adjustment."""
    z = np.asarray(z, dtype=float)
    if lam == 0:
        out = np.exp(z)
    else:
        w = lam * z + 1
        out = np.sign(w) * np.abs(w) ** (1 / lam)
    if biasadj:
        if fvar is None:
            raise ValueError("fvar (forecast variance) needed for bias adjustment")
        fvar = np.asarray(fvar, dtype=float)
        if lam == 0:
            out = out * (1 + fvar / 2)
        else:
            out = out * (1 + fvar * (1 - lam) / (2 * (lam * z + 1) ** 2))
    return out


def box_cox_lambda(y, method: str = "guerrero", lower: float = -1, upper: float = 2) -> float:
    """Automatic Box-Cox lambda, replicating forecast::BoxCox.lambda (Guerrero 1993)."""
    x = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    m = int(round(frequency(y))) if hasattr(y, "attrs") else 1
    if method != "guerrero":
        raise NotImplementedError("only method='guerrero' is implemented")
    if np.any(x <= 0):
        lower = max(lower, 0)
    period = max(2, m)
    nobsf = len(x)
    nyr = nobsf // period
    x = x[nobsf - nyr * period:]
    mat = x.reshape(nyr, period).T  # columns = years (like R's matrix(..., nrow=period))

    def guer_cv(lam):
        avg = mat.mean(axis=0)
        sd = mat.std(axis=0, ddof=1)
        rat = sd / avg ** (1 - lam)
        return np.std(rat, ddof=1) / np.mean(rat)

    res = optimize.minimize_scalar(guer_cv, bounds=(lower, upper), method="bounded",
                                   options={"xatol": 1e-5})
    return float(res.x)


# =============================================================================
# Residual diagnostics
# =============================================================================
def ljung_box(resid, lag: Optional[int] = None, df: int = 0, m: int = 1) -> pd.Series:
    """Ljung-Box test as used by forecast::checkresiduals.

    Default lag = 10 for non-seasonal data, 2m for seasonal (capped at n/5), like R.
    df = number of estimated model parameters.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    e = np.asarray(resid, dtype=float)
    e = e[~np.isnan(e)]
    if lag is None:
        lag = 10 if m <= 1 else 2 * m
        lag = min(lag, int(round(len(e) / 5)))
        lag = max(df + 3, lag)
    res = acorr_ljungbox(e, lags=[lag], model_df=df, return_df=True)
    return pd.Series({"Q*": float(res["lb_stat"].iloc[0]), "df": lag - df,
                      "lag": lag, "p-value": float(res["lb_pvalue"].iloc[0])})


def checkresiduals(fc_or_resid, df: Optional[int] = None, lag: Optional[int] = None,
                   m: Optional[int] = None, plot: bool = True, title: str = "Residuals"):
    """Residual diagnostics like forecast::checkresiduals (plots + Ljung-Box test)."""
    if isinstance(fc_or_resid, Forecast):
        resid = fc_or_resid.residuals
        m = m or int(round(frequency(fc_or_resid.x)))
        if df is None:
            df = {"Naive method": 0, "Seasonal naive method": 0, "Mean": 1,
                  "Random walk with drift": 1}.get(fc_or_resid.method, 0)
        title = f"Residuals from {fc_or_resid.method}"
    else:
        resid = pd.Series(np.asarray(fc_or_resid, dtype=float)) if not isinstance(fc_or_resid, pd.Series) else fc_or_resid
        m = m or int(round(frequency(resid)))
        df = df or 0
    lb = ljung_box(resid, lag=lag, df=df, m=m)
    if plot:
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
        e = resid.dropna()
        fig = plt.figure(figsize=(9, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(ts_time(resid) if isinstance(resid.index, (pd.PeriodIndex,)) or resid.index.dtype.kind == "f"
                 else np.arange(len(resid)), resid.to_numpy(), marker=".", lw=1)
        ax1.axhline(0, color="grey", lw=0.8)
        ax1.set_title(title)
        ax2 = fig.add_subplot(2, 2, 3)
        plot_acf(e, ax=ax2, zero=False, lags=min(len(e) - 1, max(24, 2 * m)), title="ACF")
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.hist(e, bins="auto", density=True, alpha=0.7)
        xs = np.linspace(e.min(), e.max(), 200)
        ax3.plot(xs, stats.norm.pdf(xs, e.mean(), e.std()), color="C1")
        ax3.set_title("Histogram")
        fig.tight_layout()
        plt.show()
    print(f"\tLjung-Box test\n\nQ* = {lb['Q*']:.4f}, df = {int(lb['df'])}, p-value = {lb['p-value']:.4g}")
    return lb


# =============================================================================
# Plotting (ggplot-like helpers)
# =============================================================================
def _x(x):
    return ts_time(x)


def autoplot(*series, labels=None, title=None, xlab="Time", ylab=None, ax=None, facets=False):
    """Time plot like forecast::autoplot. Pass one or more Series (or a DataFrame)."""
    import matplotlib.pyplot as plt
    objs = []
    for s in series:
        if isinstance(s, pd.DataFrame):
            for c in s.columns:
                cs = s[c]
                cs.attrs = dict(s.attrs)
                objs.append((c, cs))
        else:
            objs.append((s.name, s))
    if labels:
        objs = [(lab, s) for lab, (_, s) in zip(labels, objs)]
    if facets and len(objs) > 1:
        fig, axes = plt.subplots(len(objs), 1, figsize=(9, 2.2 * len(objs)), sharex=True)
        for a, (lab, s) in zip(axes, objs):
            a.plot(_x(s), s.to_numpy())
            a.set_ylabel(lab)
        axes[-1].set_xlabel(xlab)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    for lab, s in objs:
        ax.plot(_x(s), s.to_numpy(), label=lab)
    if len(objs) > 1:
        ax.legend()
    ax.set_title(title or "")
    ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    return ax


def seasonplot(y, polar=False, year_labels=True, title=None, ylab=None, ax=None):
    """Seasonal plot (forecast::ggseasonplot): one line per year across the seasons."""
    import matplotlib.pyplot as plt
    m = int(round(frequency(y)))
    t = ts_time(y)
    yrs = np.floor(t + 1e-8).astype(int)
    cyc = np.rint((t - yrs) * m).astype(int) + 1
    df = pd.DataFrame({"year": yrs, "season": cyc, "y": y.to_numpy()})
    if ax is None:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111, projection="polar" if polar else None)
    cmap = plt.get_cmap("viridis")
    years = sorted(df["year"].unique())
    for i, (yr, g) in enumerate(df.groupby("year")):
        col = cmap(i / max(1, len(years) - 1))
        xs = g["season"].to_numpy()
        if polar:
            xs = 2 * np.pi * (xs - 1) / m
        ax.plot(xs, g["y"].to_numpy(), color=col, lw=1)
        if year_labels and not polar:
            ax.text(xs[-1] + 0.1, g["y"].iloc[-1], str(yr), color=col, fontsize=7)
    labels = ([pd.Period(year=2000, month=i, freq="M").strftime("%b") for i in range(1, 13)]
              if m == 12 else [f"Q{i}" for i in range(1, 5)] if m == 4 else [str(i) for i in range(1, m + 1)])
    if polar:
        ax.set_xticks(2 * np.pi * np.arange(m) / m)
    else:
        ax.set_xticks(range(1, m + 1))
    if m <= 12:
        ax.set_xticklabels(labels)
    ax.set_title(title or f"Seasonal plot: {y.name}")
    if ylab:
        ax.set_ylabel(ylab)
    return ax


def subseriesplot(y, title=None, ylab=None, ax=None):
    """Seasonal subseries plot (forecast::ggsubseriesplot) with season means in blue."""
    import matplotlib.pyplot as plt
    m = int(round(frequency(y)))
    t = ts_time(y)
    yrs = np.floor(t + 1e-8).astype(int)
    cyc = np.rint((t - yrs) * m).astype(int) + 1
    df = pd.DataFrame({"season": cyc, "y": y.to_numpy()})
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    for s in range(1, m + 1):
        v = df.loc[df["season"] == s, "y"].to_numpy()
        xs = s - 0.4 + 0.8 * np.arange(len(v)) / max(1, len(v) - 1)
        ax.plot(xs, v, color="black", lw=0.8)
        ax.hlines(np.nanmean(v), s - 0.4, s + 0.4, color="blue", lw=1.5)
    labels = ([pd.Period(year=2000, month=i, freq="M").strftime("%b") for i in range(1, 13)]
              if m == 12 else [f"Q{i}" for i in range(1, 5)] if m == 4 else [str(i) for i in range(1, m + 1)])
    ax.set_xticks(range(1, m + 1))
    ax.set_xticklabels(labels[:m])
    ax.set_title(title or f"Seasonal subseries plot: {y.name}")
    if ylab:
        ax.set_ylabel(ylab)
    return ax


def lagplot(y, lags: int = 9, ncols: int = 3, color_by_season: bool = True):
    """Lag plots (forecast::gglagplot): y_t against y_{t-k}, coloured by season."""
    import matplotlib.pyplot as plt
    v = y.to_numpy()
    m = int(round(frequency(y)))
    t = ts_time(y)
    cyc = np.rint((t - np.floor(t + 1e-8)) * m).astype(int) + 1
    nrows = int(np.ceil(lags / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for k in range(1, lags + 1):
        a = axes[k - 1]
        if color_by_season and m > 1:
            a.scatter(v[:-k], v[k:], c=cyc[k:], cmap="tab20" if m > 10 else "tab10", s=8)
        else:
            a.scatter(v[:-k], v[k:], s=8)
        lo, hi = np.nanmin(v), np.nanmax(v)
        a.plot([lo, hi], [lo, hi], color="grey", lw=0.6, ls="--")
        a.set_title(f"lag {k}", fontsize=9)
    for a in axes[lags:]:
        a.axis("off")
    fig.tight_layout()
    return axes


def acf_plot(y, lag_max: Optional[int] = None, pacf: bool = False, ax=None, title=None):
    """ACF / PACF bar plot like forecast::ggAcf / ggPacf (no lag 0)."""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    v = pd.Series(np.asarray(y, dtype=float)).dropna().to_numpy()
    m = int(round(frequency(y))) if hasattr(y, "attrs") else 1
    if lag_max is None:
        lag_max = int(min(max(10 * np.log10(len(v)), 2 * m), len(v) - 1))
        if pacf:
            lag_max = min(lag_max, len(v) // 2 - 1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
    fn = plot_pacf if pacf else plot_acf
    kw = dict(lags=lag_max, zero=False, ax=ax, title=title or ("PACF" if pacf else "ACF"), alpha=None)
    if pacf:
        kw["method"] = "ywm"
    fn(v, **kw)
    ci = 1.96 / np.sqrt(len(v))
    ax.axhline(ci, color="blue", ls="--", lw=0.8)
    ax.axhline(-ci, color="blue", ls="--", lw=0.8)
    return ax


def plot_forecast(fc: Forecast, include: Optional[int] = None, title=None, ax=None,
                  show_pi: bool = True, ylab=None, series_label=None):
    """Plot a Forecast object with shaded 80%/95% prediction intervals (autoplot(forecast))."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    x = fc.x if include is None else fc.x.iloc[-include:]
    if include is not None:
        x.attrs = dict(fc.x.attrs)
    ax.plot(_x(x), x.to_numpy(), color="black", lw=1, label=series_label)
    tf = ts_time(pd.Series(fc.mean.to_numpy(), index=fc.mean.index).pipe(
        lambda s: (s.attrs.update(frequency=frequency(fc.x)) or s)))
    if show_pi:
        for lv, alpha in sorted(zip(fc.level, [0.35, 0.2]), reverse=True):
            ax.fill_between(tf, fc.lower[lv].to_numpy(), fc.upper[lv].to_numpy(),
                            color="C0", alpha=alpha, lw=0, label=f"{lv}% PI")
    ax.plot(tf, fc.mean.to_numpy(), color="C0", lw=1.5, label="forecast")
    ax.set_title(title or f"Forecasts from {fc.method}")
    if ylab:
        ax.set_ylabel(ylab)
    return ax
