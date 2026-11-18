"""
fpp_ch06.py -- Chapter 6 (time series decomposition) helpers for the Python half of the notes.

Mirrors the R functions used in Chapter 6 of FPP2 (Hyndman & Athanasopoulos):

    ma(x, order, centre=True)        forecast::ma        (m-MA, 2xm-MA)
    classical(x, type)               stats::decompose     (classical additive / multiplicative)
    stl_r(x, s_window, ...)          stats::stl           (same defaults as R: s.degree=0, jumps, inner/outer)
    mstl_r(x, lambda_=None)          forecast::mstl       (single seasonality: s.window = 11)
    seasadj(dec)                     forecast::seasadj
    strength(dec)                    F_T and F_S of Section 6.7
    forecast_stl(dec, h, method)     forecast(stl_fit, method=...)   (naive / rwdrift / ets / arima)
    stlf(x, h, method, lambda_)      forecast::stlf
    x13(x, method="x11"|"seats")     seasonal::seas  -- runs the SAME X-13ARIMA-SEATS binary as R

The X-13 binary: R's x13binary package ships it as `x13ashtml` in
`x13binary::x13path()`. We look for it in $X13PATH (R/setup.R sets this when you knit in
RStudio), then ask R (`Rscript -e 'cat(x13binary::x13path())'`), then look for `x13as`/`x13ashtml`
on the PATH. This works on macOS, Linux and Windows.

Part of a personal study project following https://otexts.com/fpp2/ -- credit to the authors.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import fpp


# =============================================================================
# Moving averages
# =============================================================================
def ma(x, order: int, centre: bool = True) -> pd.Series:
    """forecast::ma -- m-MA; for even order with centre=True a 2 x m-MA."""
    v = np.asarray(x, dtype=float)
    n = len(v)
    if order % 2 == 1:
        w = np.repeat(1 / order, order)
    elif centre:
        w = np.r_[0.5, np.repeat(1.0, order - 1), 0.5] / order
    else:
        w = np.repeat(1 / order, order)
    k = len(w)
    out = np.full(n, np.nan)
    if order % 2 == 0 and not centre:
        # R: filter(..., sides=2) for even length puts the extra point in the future
        left = k // 2 - 1
    else:
        left = (k - 1) // 2
    for t in range(left, n - (k - 1 - left)):
        out[t] = np.dot(w, v[t - left:t - left + k])
    s = pd.Series(out, index=getattr(x, "index", None), name=f"{order}-MA")
    if hasattr(x, "attrs"):
        s.attrs = dict(x.attrs)
    return s


def weighted_ma(x, weights) -> pd.Series:
    """Centred weighted moving average with arbitrary symmetric weights."""
    v = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    k = len(w)
    half = (k - 1) // 2
    out = np.full(len(v), np.nan)
    for t in range(half, len(v) - half):
        out[t] = np.dot(w, v[t - half:t + half + 1])
    s = pd.Series(out, index=getattr(x, "index", None))
    if hasattr(x, "attrs"):
        s.attrs = dict(x.attrs)
    return s


# =============================================================================
# Decomposition containers
# =============================================================================
@dataclass
class Decomp:
    data: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    remainder: pd.Series
    type: str = "additive"
    method: str = ""

    def seasadj(self) -> pd.Series:
        return seasadj(self)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"data": self.data, "trend": self.trend,
                             "seasonal": self.seasonal, "remainder": self.remainder})

    def plot(self, title: Optional[str] = None):
        import matplotlib.pyplot as plt
        df = self.frame()
        t = fpp.ts_time(self.data)
        fig, axes = plt.subplots(4, 1, figsize=(8, 7), sharex=True)
        for ax, c in zip(axes, df.columns):
            if c == "remainder":
                base = 1.0 if self.type == "multiplicative" else 0.0
                ax.vlines(t, base, df[c].to_numpy(), lw=0.8)
                ax.axhline(base, color="grey", lw=0.5)
            else:
                ax.plot(t, df[c].to_numpy(), lw=1)
            ax.set_ylabel(c)
        axes[0].set_title(title or self.method)
        fig.tight_layout()
        return axes


def _wrap(x, values, name):
    s = pd.Series(np.asarray(values, dtype=float), index=x.index, name=name)
    s.attrs = dict(x.attrs)
    return s


def seasadj(dec: Decomp) -> pd.Series:
    """forecast::seasadj -- seasonally adjusted series."""
    if dec.type == "multiplicative":
        v = dec.data.to_numpy() / dec.seasonal.to_numpy()
    else:
        v = dec.data.to_numpy() - dec.seasonal.to_numpy()
    return _wrap(dec.data, v, "seasadj")


# =============================================================================
# Classical decomposition (stats::decompose)
# =============================================================================
def classical(x, type: str = "additive") -> Decomp:
    m = int(round(fpp.frequency(x)))
    v = x.to_numpy(dtype=float)
    n = len(v)
    trend = ma(x, m, centre=True).to_numpy()
    detr = v - trend if type == "additive" else v / trend
    # cycle position of each observation (R's cycle())
    t = fpp.ts_time(x)
    cyc = (np.rint((t - np.floor(t + 1e-8)) * m).astype(int)) % m
    figure = np.array([np.nanmean(detr[cyc == j]) for j in range(m)])
    figure = figure - figure.mean() if type == "additive" else figure / figure.mean()
    seasonal = figure[cyc]
    rem = v - seasonal - trend if type == "additive" else v / (seasonal * trend)
    return Decomp(x, _wrap(x, trend, "trend"), _wrap(x, seasonal, "seasonal"),
                  _wrap(x, rem, "remainder"), type, f"Classical {type} decomposition")


# =============================================================================
# STL with R's defaults
# =============================================================================
def _nextodd(v):
    v = int(round(v))
    return v + 1 if v % 2 == 0 else v


def stl_r(x, s_window="periodic", t_window: Optional[int] = None, robust: bool = False,
          s_degree: Optional[int] = None, lambda_: Optional[float] = None) -> Decomp:
    """stats::stl with exactly R's defaults (s.degree=0, jumps = ceil(window/10),
    inner = 1/2 and outer = 15/0 iterations for robust / non-robust fits).
    s_window: odd integer or "periodic"."""
    from statsmodels.tsa.seasonal import STL
    m = int(round(fpp.frequency(x)))
    y = fpp.box_cox(x, lambda_) if lambda_ is not None else x
    n = len(y)
    periodic = isinstance(s_window, str) and s_window.startswith("per")
    if periodic:
        sw = 10 * n + 1
        sdeg = 0
    else:
        sw = _nextodd(s_window)
        sdeg = 0 if s_degree is None else s_degree
    tw = _nextodd(math.ceil(1.5 * m / (1 - 1.5 / sw))) if t_window is None else _nextodd(t_window)
    lw = _nextodd(m)
    if lw <= m:        # odd periods: statsmodels needs low_pass > period (R uses = period)
        lw = m + 2
    if tw <= m:
        raise ValueError(f"statsmodels' STL needs t_window > period ({m}); R allows smaller values. "
                         f"Use t_window >= {_nextodd(m + 1)}.")
    endog = pd.Series(y.to_numpy(dtype=float))
    res = STL(endog, period=m, seasonal=sw, trend=tw, low_pass=lw, seasonal_deg=sdeg,
              trend_deg=1, low_pass_deg=1, robust=robust,
              seasonal_jump=math.ceil(sw / 10), trend_jump=math.ceil(tw / 10),
              low_pass_jump=math.ceil(lw / 10)).fit(inner_iter=1 if robust else 2,
                                                    outer_iter=15 if robust else 0)
    seas = res.seasonal.to_numpy()
    trend = res.trend.to_numpy()
    yv = y.to_numpy(dtype=float)
    if periodic:
        t = fpp.ts_time(y)
        cyc = (np.rint((t - np.floor(t + 1e-8)) * m).astype(int)) % m
        means = np.array([seas[cyc == j].mean() for j in range(m)])
        seas = means[cyc]
    rem = yv - seas - trend
    d = Decomp(_wrap(x, yv, "data"), _wrap(x, trend, "trend"), _wrap(x, seas, "seasonal"),
               _wrap(x, rem, "remainder"), "additive",
               f"STL (s.window={s_window}, t.window={tw}{', robust' if robust else ''})")
    d.lambda_ = lambda_
    return d


def mstl_r(x, lambda_: Optional[float] = None, s_window: int = 11, t_window=None,
           robust: bool = False) -> Decomp:
    """forecast::mstl for a single seasonal period (s.window = 7 + 4 = 11)."""
    d = stl_r(x, s_window=s_window, t_window=t_window, robust=robust, lambda_=lambda_)
    d.method = "MSTL"
    return d


# =============================================================================
# Strength of trend and seasonality (Section 6.7)
# =============================================================================
def strength(dec: Decomp) -> dict:
    r = dec.remainder.to_numpy()
    tr = dec.trend.to_numpy()
    s = dec.seasonal.to_numpy()
    ok = ~np.isnan(r)
    ft = max(0.0, 1 - np.var(r[ok], ddof=1) / np.var(tr[ok] + r[ok], ddof=1))
    fs = max(0.0, 1 - np.var(r[ok], ddof=1) / np.var(s[ok] + r[ok], ddof=1))
    return {"trend_strength": ft, "seasonal_strength": fs}


# =============================================================================
# Forecasting with decomposition
# =============================================================================
def _ets_nonseasonal(y, h, level, model="ZZN", damped=None):
    from statsforecast.models import AutoETS
    kw = dict(season_length=1, model=model)
    if damped is not None:
        kw["damped"] = damped
    m = AutoETS(**kw).fit(y.to_numpy(dtype=float))
    p = m.predict(h=h, level=list(level))
    idx = fpp._future_index(y, h)
    fc = fpp.Forecast(f"{m.model_['method']}", y, pd.Series(p["mean"], idx),
                      pd.DataFrame({lv: np.asarray(p[f"lo-{lv}"], dtype=float) for lv in level}, idx),
                      pd.DataFrame({lv: np.asarray(p[f"hi-{lv}"], dtype=float) for lv in level}, idx),
                      pd.Series(np.asarray(m.model_["fitted"]), y.index),
                      pd.Series(y.to_numpy() - np.asarray(m.model_["fitted"]), y.index), tuple(level))
    return fc


def _arima_nonseasonal(y, h, level):
    from statsforecast.models import AutoARIMA
    m = AutoARIMA(season_length=1, seasonal=False).fit(y.to_numpy(dtype=float))
    p = m.predict(h=h, level=list(level))
    idx = fpp._future_index(y, h)
    return fpp.Forecast("ARIMA", y, pd.Series(p["mean"], idx),
                        pd.DataFrame({lv: np.asarray(p[f"lo-{lv}"], dtype=float) for lv in level}, idx),
                        pd.DataFrame({lv: np.asarray(p[f"hi-{lv}"], dtype=float) for lv in level}, idx),
                        pd.Series(np.nan, y.index), pd.Series(np.nan, y.index), tuple(level))


def forecast_stl(dec: Decomp, h: Optional[int] = None, method: str = "ets", etsmodel: str = "ZZN",
                 damped=None, level=(80, 95), lambda_: Optional[float] = None) -> fpp.Forecast:
    """forecast(<stl fit>, method=...) -- forecast seasonally adjusted series with a
    non-seasonal method, then add back the last year of the seasonal component."""
    m = int(round(fpp.frequency(dec.data)))
    h = h or 2 * m
    sa = seasadj(dec)
    if method == "naive":
        f = fpp.rwf(sa, h=h, drift=False, level=level)
    elif method == "rwdrift":
        f = fpp.rwf(sa, h=h, drift=True, level=level)
    elif method == "ets":
        f = _ets_nonseasonal(sa, h, level, model=etsmodel, damped=damped)
    elif method == "arima":
        f = _arima_nonseasonal(sa, h, level)
    else:
        raise ValueError(method)
    seas = dec.seasonal.to_numpy()
    lastseas = np.array([seas[len(seas) - m + (i % m)] for i in range(h)])
    if dec.type == "multiplicative":
        mean = f.mean.to_numpy() * lastseas
        lo = f.lower.mul(lastseas, axis=0)
        hi = f.upper.mul(lastseas, axis=0)
        fitted = f.fitted * dec.seasonal
    else:
        mean = f.mean.to_numpy() + lastseas
        lo = f.lower.add(lastseas, axis=0)
        hi = f.upper.add(lastseas, axis=0)
        fitted = f.fitted + dec.seasonal
    lam = lambda_ if lambda_ is not None else getattr(dec, "lambda_", None)
    x = dec.data
    if lam is not None:
        mean = fpp.inv_box_cox(mean, lam)
        lo = lo.apply(lambda c: fpp.inv_box_cox(c.to_numpy(), lam))
        hi = hi.apply(lambda c: fpp.inv_box_cox(c.to_numpy(), lam))
        fitted = pd.Series(fpp.inv_box_cox(fitted.to_numpy(), lam), index=fitted.index)
        x = _wrap(dec.data, fpp.inv_box_cox(dec.data.to_numpy(), lam), "data")
    name = {"naive": "Random walk", "rwdrift": "Random walk with drift"}.get(method, f.method)
    return fpp.Forecast(f"STL + {name}", x, pd.Series(mean, index=f.mean.index), lo, hi,
                        fitted, x - fitted, tuple(level))


def stlf(x, h: Optional[int] = None, method: str = "ets", etsmodel: str = "ZZN", damped=None,
         s_window: int = 11, t_window=None, robust: bool = False, lambda_: Optional[float] = None,
         level=(80, 95)) -> fpp.Forecast:
    """forecast::stlf -- mstl() decomposition + non-seasonal forecast of the seasonally adjusted data."""
    dec = mstl_r(x, lambda_=lambda_, s_window=s_window, t_window=t_window, robust=robust)
    fc = forecast_stl(dec, h=h, method=method, etsmodel=etsmodel, damped=damped, level=level,
                      lambda_=lambda_)
    fc.x = x
    return fc


# =============================================================================
# X-13ARIMA-SEATS (X11 and SEATS) -- same binary and default spec as seasonal::seas()
# =============================================================================
def find_x13() -> Optional[Path]:
    """Locate the X-13 binary (x13ashtml from R's x13binary, or x13as)."""
    cands = []
    env = os.environ.get("X13PATH") or os.environ.get("X13_PATH")
    if env:
        cands.append(Path(env))
    try:
        out = subprocess.run(["Rscript", "-e", "cat(x13binary::x13path())"], capture_output=True,
                             text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            cands.append(Path(out.stdout.strip()))
    except Exception:
        pass
    for d in cands:
        if d.is_file():
            return d
        for name in ("x13ashtml", "x13as", "x13ashtml.exe", "x13as.exe"):
            if (d / name).exists():
                return d / name
    for name in ("x13as", "x13ashtml"):
        p = shutil.which(name)
        if p:
            return Path(p)
    return None


_TABLES = {"x11": {"seasonal": "d10", "seasadj": "d11", "trend": "d12", "irregular": "d13"},
           "seats": {"seasonal": "s10", "seasadj": "s11", "trend": "s12", "irregular": "s13"}}


def x13(x, method: str = "seats") -> Decomp:
    """Run X-13ARIMA-SEATS with seasonal::seas()'s default spec
    (automatic log/level, trading-day & Easter AIC tests, outliers, automdl) and
    either the SEATS (default) or X11 decomposition. Returns a multiplicative or
    additive Decomp depending on the transformation the program chose."""
    binary = find_x13()
    if binary is None:
        raise RuntimeError("X-13ARIMA-SEATS binary not found. In R run install.packages('x13binary'), "
                           "or set the X13PATH environment variable to its folder.")
    m = int(round(fpp.frequency(x)))
    t = fpp.ts_time(x)
    y0 = int(np.floor(t[0] + 1e-8))
    p0 = int(np.rint((t[0] - y0) * m)) + 1
    tab = _TABLES[method]
    vals = [f"{v:.10g}" for v in x.to_numpy(dtype=float)]
    data = "\n    ".join(" ".join(vals[i:i + 6]) for i in range(0, len(vals), 6))
    decomp_spec = ("x11{ save = (d10 d11 d12 d13) }" if method == "x11"
                   else "seats{ save = (s10 s11 s12 s13) }")
    spec = f"""series{{
  title = "fpp"
  start = {y0}.{p0}
  period = {m}
  data = ({data})
}}
transform{{ function = auto print = aictransform }}
regression{{ aictest = (td easter) }}
outlier{{ }}
automdl{{ print = bestfivemdl }}
{decomp_spec}
"""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "fpp"
        base.with_suffix(".spc").write_text(spec)
        proc = subprocess.run([str(binary), str(base), "-s"], capture_output=True, text=True,
                              cwd=tmp, timeout=300)
        out = {}
        for key, ext in tab.items():
            f = base.with_suffix("." + ext)
            if not f.exists():
                raise RuntimeError("X-13 failed:\n" + proc.stdout[-2000:] + proc.stderr[-2000:])
            lines = f.read_text().splitlines()
            vals = [float(ln.split()[1]) for ln in lines if ln.strip() and ln.split()[0].lstrip("-").isdigit()]
            out[key] = np.array(vals)
        # was a log transformation chosen? (multiplicative decomposition)
        log_text = ""
        for ext in (".log", ".html", ".out"):
            fp = base.with_suffix(ext)
            if fp.exists():
                log_text += fp.read_text(errors="ignore")
    seas = out["seasonal"]
    # X-13 reports multiplicative factors (centred on 1, all positive) when it chose a log
    # transformation, and additive effects (centred on 0) otherwise.
    mult = bool(np.all(seas > 0) and abs(np.nanmean(seas) - 1) < 0.3)
    typ = "multiplicative" if mult else "additive"
    d = Decomp(x, _wrap(x, out["trend"], "trend"), _wrap(x, seas, "seasonal"),
               _wrap(x, out["irregular"], "remainder"), typ, f"{method.upper()} decomposition")
    d.seasadj_x13 = _wrap(x, out["seasadj"], "seasadj")
    return d
