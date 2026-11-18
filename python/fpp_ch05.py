"""
fpp_ch05.py -- Python helpers for Chapter 5 (Time series regression models) of
"Forecasting: Principles and Practice" (2nd ed.) by Hyndman & Athanasopoulos.

They reproduce the R `forecast` functions used in the chapter so the Python results can be
compared number-for-number with R:

    trend(), season_dummies(), fourier()   -> the predictors tslm() creates (trend, season, fourier())
    tslm()                                 -> OLS on a design matrix (optionally Box-Cox transformed y)
    forecast_lm()                          -> forecast.lm(): point forecasts + t-based prediction intervals
    CV()                                   -> forecast::CV(): CV, AIC, AICc, BIC, AdjR2
    bgtest()                               -> lmtest::bgtest(): Breusch-Godfrey test
    checkresiduals_lm()                    -> checkresiduals() for a regression (plots + BG test)
    easter()                               -> forecast::easter(): Easter dummy (split across periods)
    splinef()                              -> forecast::splinef(): cubic smoothing spline forecasts
"""
from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.easter import easter as _easter_date
from scipy import optimize, stats

import fpp

__all__ = ["trend", "season_dummies", "fourier", "tslm", "forecast_lm", "CV", "bgtest",
           "checkresiduals_lm", "easter", "splinef", "LMFit"]


# -----------------------------------------------------------------------------
# Predictors created automatically by tslm()
# -----------------------------------------------------------------------------
def trend(y, h: Optional[int] = None) -> np.ndarray:
    """t = 1..T (or T+1..T+h for forecasts), as tslm's `trend`."""
    n = len(y)
    return np.arange(1, n + 1, dtype=float) if h is None else np.arange(n + 1, n + h + 1, dtype=float)


def _cycle(y, h: Optional[int] = None) -> np.ndarray:
    m = int(round(fpp.frequency(y)))
    t = fpp.ts_time(y)
    first = int(np.rint((t[0] - np.floor(t[0] + 1e-8)) * m)) + 1
    n = len(y)
    idx = np.arange(n) if h is None else np.arange(n, n + h)
    return (first - 1 + idx) % m + 1


def season_dummies(y, h: Optional[int] = None) -> pd.DataFrame:
    """Seasonal dummies season2..seasonm (season 1 is the baseline), as tslm's `season`."""
    m = int(round(fpp.frequency(y)))
    cyc = _cycle(y, h)
    return pd.DataFrame({f"season{j}": (cyc == j).astype(float) for j in range(2, m + 1)})


def fourier(y, K, h: Optional[int] = None, period=None) -> pd.DataFrame:
    """Exact port of forecast::fourier(x, K, h). Columns S1-m, C1-m, S2-m, ...

    K may be a list together with `period` (list) for multiple seasonalities.
    """
    if period is None:
        msts = getattr(y, "attrs", {}).get("msts")
        period = msts if msts else [fpp.frequency(y)]
    period = list(np.atleast_1d(period))
    K = list(np.atleast_1d(K))
    n = len(y)
    times = np.arange(1, n + 1) if h is None else n + np.arange(1, h + 1)
    p, labels = [], []
    for per, k in zip(period, K):
        for j in range(1, int(k) + 1):
            p.append(j / per)
            labels += [f"S{j}-{round(per)}", f"C{j}-{round(per)}"]
    cols = {}
    seen = []
    for j, pj in enumerate(p):
        if any(abs(pj - s) < 1e-12 for s in seen):
            continue
        seen.append(pj)
        if abs(2 * pj - round(2 * pj)) > np.finfo(float).eps:
            cols[labels[2 * j]] = np.sin(2 * np.pi * pj * times)
        cols[labels[2 * j + 1]] = np.cos(2 * np.pi * pj * times)
    return pd.DataFrame(cols)


# -----------------------------------------------------------------------------
# Fitting and forecasting
# -----------------------------------------------------------------------------
class LMFit:
    """A fitted regression: statsmodels results + what forecasting needs."""

    def __init__(self, y, X: pd.DataFrame, res, lambda_=None, biasadj=False):
        self.y = y
        self.X = X
        self.res = res
        self.lambda_ = lambda_
        self.biasadj = biasadj

    @property
    def params(self):
        return self.res.params

    def _full(self, v) -> pd.Series:
        out = np.full(len(self.y), np.nan)
        out[self.mask] = np.asarray(v, dtype=float)
        return pd.Series(out, index=self.y.index)

    @property
    def fitted(self) -> pd.Series:
        f = self.res.fittedvalues.to_numpy()
        if self.lambda_ is not None:
            f = fpp.inv_box_cox(f, self.lambda_)
        return self._full(f)

    @property
    def residuals(self) -> pd.Series:
        """Residuals on the (transformed) scale the model was fitted on, as residuals(tslm)."""
        return self._full(self.res.resid.to_numpy())

    def summary(self):
        return self.res.summary()


def tslm(y, X: Optional[pd.DataFrame] = None, trend_: bool = False, season: bool = False,
         K=None, lambda_=None, biasadj: bool = False, const: bool = True) -> LMFit:
    """Regression of y on predictors, like tslm(y ~ trend + season + fourier(y, K) + X).

    Column order follows R: intercept, trend, season2..m, fourier terms, then X.
    """
    parts = []
    if trend_:
        parts.append(pd.DataFrame({"trend": trend(y)}))
    if season:
        parts.append(season_dummies(y))
    if K is not None:
        parts.append(fourier(y, K))
    if X is not None:
        parts.append(pd.DataFrame(X).reset_index(drop=True))
    D = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(y)))
    if const:
        D.insert(0, "(Intercept)", 1.0)
    D.index = y.index
    z = fpp.box_cox(y, lambda_) if lambda_ is not None else y.astype(float)
    zv = np.asarray(z, dtype=float)
    mask = ~np.isnan(zv) & ~D.isna().any(axis=1).to_numpy()
    res = sm.OLS(zv[mask], D[mask]).fit(method="qr")   # QR like R's lm(): stable for ill-conditioned X
    fit = LMFit(y, D, res, lambda_, biasadj)
    fit.mask = mask
    fit.spec = dict(trend_=trend_, season=season, K=K, const=const,
                    xcols=list(pd.DataFrame(X).columns) if X is not None else [])
    return fit


def _future_design(fit: LMFit, h: int, newdata: Optional[pd.DataFrame]) -> pd.DataFrame:
    s = fit.spec
    parts = []
    if s["trend_"]:
        parts.append(pd.DataFrame({"trend": trend(fit.y, h)}))
    if s["season"]:
        parts.append(season_dummies(fit.y, h))
    if s["K"] is not None:
        parts.append(fourier(fit.y, s["K"], h))
    if s["xcols"]:
        parts.append(pd.DataFrame(newdata)[s["xcols"]].reset_index(drop=True))
    D = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(h))
    if s["const"]:
        D.insert(0, "(Intercept)", 1.0)
    return D


def forecast_lm(fit: LMFit, newdata: Optional[pd.DataFrame] = None, h: Optional[int] = None,
                level: Sequence[int] = (80, 95)) -> fpp.Forecast:
    """forecast.lm(): forecasts with prediction intervals using the t distribution."""
    if h is None:
        h = len(newdata) if newdata is not None else 10      # forecast.lm's default h = 10
    D = _future_design(fit, h, newdata)
    pred = fit.res.get_prediction(D)
    mean = pred.predicted_mean
    se_fit = pred.se_mean
    sigma2 = fit.res.scale
    se = np.sqrt(se_fit ** 2 + sigma2)
    dfree = fit.res.df_resid
    idx = fpp._future_index(fit.y, h)
    lower, upper = {}, {}
    for lv in level:
        q = stats.t.ppf(0.5 + lv / 200, dfree)
        lower[lv] = mean - q * se
        upper[lv] = mean + q * se
    if fit.lambda_ is not None:
        if fit.biasadj:
            mean_out = fpp.inv_box_cox(mean, fit.lambda_, biasadj=True, fvar=se ** 2)
        else:
            mean_out = fpp.inv_box_cox(mean, fit.lambda_)
        lower = {k: fpp.inv_box_cox(v, fit.lambda_) for k, v in lower.items()}
        upper = {k: fpp.inv_box_cox(v, fit.lambda_) for k, v in upper.items()}
    else:
        mean_out = mean
    fc = fpp.Forecast("Linear regression model", fit.y, pd.Series(mean_out, index=idx, name="mean"),
                      pd.DataFrame(lower, index=idx), pd.DataFrame(upper, index=idx),
                      fit.fitted, fit.residuals, tuple(level))
    fc.extra["se"] = se
    return fc


def CV(fit) -> pd.Series:
    """forecast::CV(): leave-one-out CV (via the hat matrix), AIC, AICc, BIC and adjusted R^2."""
    res = fit.res if isinstance(fit, LMFit) else fit
    e = np.asarray(res.resid, dtype=float)
    hii = res.get_influence().hat_matrix_diag
    n = len(e)
    k = int(res.df_model)                  # number of predictors (excluding intercept)
    sse = np.sum(e ** 2)
    aic = n * np.log(sse / n) + 2 * (k + 2)
    aicc = aic + 2 * (k + 2) * (k + 3) / (n - k - 3)
    bic = aic + (k + 2) * (np.log(n) - 2)
    cv = np.mean((e / (1 - hii)) ** 2)
    return pd.Series({"CV": cv, "AIC": aic, "AICc": aicc, "BIC": bic, "AdjR2": res.rsquared_adj})


def bgtest(fit, order: int) -> pd.Series:
    """Breusch-Godfrey LM test for serial correlation (lmtest::bgtest, fill = 0)."""
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    res = fit.res if isinstance(fit, LMFit) else fit
    lm, p, _, _ = acorr_breusch_godfrey(res, nlags=order, result_object=False)
    return pd.Series({"LM test": lm, "df": order, "p-value": p})


def checkresiduals_lm(fit: LMFit, lag: Optional[int] = None, plot: bool = True,
                      title: str = "Residuals from Linear regression model"):
    """checkresiduals() for a regression: time plot, ACF, histogram and the Breusch-Godfrey test."""
    e = fit.residuals
    m = fpp.frequency(fit.y)
    if lag is None:
        lag = int(2 * m) if m > 1 else 10
        lag = min(lag, int(round(len(e) / 5)))
        lag = max(len(fit.params) + 3, lag)   # R: max(df + 3, lag), df = no. of coefficients
    if plot:
        import matplotlib.pyplot as plt
        from statsmodels.graphics.tsaplots import plot_acf
        fig = plt.figure(figsize=(9, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(fpp.ts_time(fit.y), e.to_numpy(), marker=".", lw=1)  # NaN where obs. missing
        ax1.axhline(0, color="grey", lw=0.8)
        ax1.set_title(title)
        ax2 = fig.add_subplot(2, 2, 3)
        v = e.dropna().to_numpy()
        plot_acf(v, ax=ax2, zero=False, lags=min(len(v) - 1, max(24, int(2 * m))), title="ACF")
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.hist(v, bins="auto", density=True, alpha=0.7)
        xs = np.linspace(v.min(), v.max(), 200)
        ax3.plot(xs, stats.norm.pdf(xs, v.mean(), v.std()), color="C1")
        ax3.set_title("Histogram")
        fig.tight_layout()
        plt.show()
    bg = bgtest(fit, lag)
    print(f"\tBreusch-Godfrey test for serial correlation of order up to {lag}\n\n"
          f"LM test = {bg['LM test']:.4f}, df = {lag}, p-value = {bg['p-value']:.4g}")
    return bg


# -----------------------------------------------------------------------------
# Easter dummy  (port of forecast::easter, easter.mon = FALSE)
# -----------------------------------------------------------------------------
def easter(y) -> pd.Series:
    """Proportion of the Easter holidays (Good Friday .. Easter Sunday) falling in each period.

    Monthly or quarterly series only. When the 3-day holiday straddles 31 March the dummy is split
    between March/April (or Q1/Q2) and rounded to 2 decimals, exactly like R.
    """
    m = int(round(fpp.frequency(y)))
    if m not in (4, 12):
        raise ValueError("easter() needs monthly or quarterly data")
    t = fpp.ts_time(y)
    years = np.floor(t + 1e-8).astype(int)
    per = np.rint((t - years) * m).astype(int) + 1
    out = np.zeros(len(y))
    for i, (yr, p) in enumerate(zip(years, per)):
        es = _easter_date(int(yr))
        gf = es - dt.timedelta(days=2)
        days = [gf + dt.timedelta(days=d) for d in range(3)]
        if m == 12:
            in_p = [d for d in days if d.month == p]
        else:
            in_p = [d for d in days if (d.month - 1) // 3 + 1 == p]
        if in_p:
            out[i] = round(len(in_p) / 3, 2)
    return pd.Series(out, index=y.index, name="easter")


# -----------------------------------------------------------------------------
# Cubic smoothing spline forecasts (port of forecast::splinef)
# -----------------------------------------------------------------------------
def _make_sigma(n, n0=0):
    nn = n + n0
    i = np.arange(1, nn + 1)
    I, J = np.meshgrid(i, i, indexing="ij")
    a = np.minimum(I, J)
    b = np.maximum(I, J)
    return (a * a * (3 * b - a)) / 6 / n ** 3


def _spline_matrices(n, beta, cc=100, n0=0):
    nn = n + n0
    Sigma = _make_sigma(n, n0)
    s = np.column_stack([np.ones(nn), np.arange(1, nn + 1) / n])
    Omega = cc * s @ s.T + Sigma / beta + np.eye(nn)
    mx = Omega.max()
    inv_Omega = np.linalg.solve(Omega / mx, np.eye(nn)) / mx
    return dict(s=s, Sigma=Sigma, Omega=Omega, inv_Omega=inv_Omega)


def _spline_loglik(beta, y, cc=100):
    n = len(y)
    mat = _spline_matrices(n, beta / 1e6, cc)
    P = np.linalg.cholesky(mat["inv_Omega"]).T          # upper triangular, like R's chol()
    ystar = P @ y
    return -np.sum(np.log(np.diag(P))) + 0.5 * n * np.log(np.sum(ystar ** 2))


def splinef(y, h: int = 10, level=(80, 95), lambda_=None, method: str = "gcv") -> fpp.Forecast:
    """Cubic smoothing spline forecasts (forecast::splinef).

    method="mle": exact port of R (smoothing parameter by maximum likelihood on the last 100 obs).
    method="gcv": the smoothing parameter minimises generalised cross-validation of the equivalent
                  state-space smoother (R uses smooth.spline's GCV; values are close, not identical).
    The forecasts are a straight-line extrapolation of the trend at the end of the data.
    """
    x = fpp.box_cox(y, lambda_).to_numpy() if lambda_ is not None else np.asarray(y, dtype=float)
    n = len(x)
    if method == "mle":
        xx = x[-100:] if n > 100 else x
        r = optimize.minimize_scalar(_spline_loglik, bounds=(1e-6, 1e7), args=(xx,), method="bounded",
                                     options={"xatol": 1e-5})
        beta = r.x / 1e6
    else:
        def gcv(logb):
            mat = _spline_matrices(n, np.exp(logb))
            S = np.eye(n) - mat["inv_Omega"]           # smoother: signal posterior mean = (I - Omega^-1) x
            fit_ = S @ x
            return n * np.sum((x - fit_) ** 2) / (n - np.trace(S)) ** 2
        r = optimize.minimize_scalar(gcv, bounds=(np.log(1e-7), np.log(1e7)), method="bounded")
        beta = max(1e-7, float(np.exp(r.x)))
    mat = _spline_matrices(n, beta)
    newmat = _spline_matrices(n, beta, n0=h)
    Om = mat["Omega"]
    yfit = np.full(n, np.nan)
    e = np.full(n, np.nan)
    for i in range(1, n):
        U = Om[:i, i]
        Oinv = np.linalg.solve(Om[:i, :i] / 1e6, np.eye(i)) / 1e6
        yfit[i] = U @ Oinv @ x[:i]
        sd = np.sqrt(Om[i, i] - U @ Oinv @ U)
        e[i] = (x[i] - yfit[i]) / sd
    sigma2 = np.nanmean(e ** 2)
    U = newmat["Omega"][:n, n:n + h]
    Omega0 = newmat["Omega"][n:n + h, n:n + h]
    yhat = U.T @ mat["inv_Omega"] @ x
    sd = np.sqrt(sigma2 * np.diag(Omega0 - U.T @ mat["inv_Omega"] @ U))
    sfits = (np.eye(n) - mat["inv_Omega"]) @ x
    idx = fpp._future_index(y, h)
    lower, upper = {}, {}
    for lv in level:
        q = stats.norm.ppf(0.5 + lv / 200)
        lower[lv], upper[lv] = yhat - q * sd, yhat + q * sd
    resid = x - yfit
    if lambda_ is not None:
        yhat = fpp.inv_box_cox(yhat, lambda_)
        lower = {k: fpp.inv_box_cox(v, lambda_) for k, v in lower.items()}
        upper = {k: fpp.inv_box_cox(v, lambda_) for k, v in upper.items()}
        sfits = fpp.inv_box_cox(sfits, lambda_)
    fc = fpp.Forecast("Cubic Smoothing Spline", y, pd.Series(yhat, index=idx, name="mean"),
                      pd.DataFrame(lower, index=idx), pd.DataFrame(upper, index=idx),
                      pd.Series(sfits, index=y.index), pd.Series(resid, index=y.index), tuple(level))
    fc.extra["beta"] = beta * n ** 3
    return fc
