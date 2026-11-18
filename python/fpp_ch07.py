"""
fpp_ch07.py -- Chapter 7 (exponential smoothing) helpers for the Python half of the notes.

R's forecast package fits every exponential smoothing method with ets():
    ses(y)   = ets(y, "ANN", opt.crit="mse")
    holt(y)  = ets(y, "AAN", damped=, opt.crit="mse")          (exponential=TRUE -> "MMN")
    hw(y)    = ets(y, "AAA" / "MAM", damped=, opt.crit="mse")
    ets(y)   = automatic ETS model selection by AICc

statsforecast contains a direct port of forecast::ets (`statsforecast.ets.ets_f`), including the
`opt_crit` argument, so the functions below call it with exactly the same settings as R.
One small fix is applied: R estimates the residual variance as SSE / (n - #parameters),
statsforecast divides by one less degree of freedom; we use R's definition so the prediction
intervals match the book.

    ets(y, model="ZZZ", ...)  -> ETSFit  (fit.forecast(h) -> fpp.Forecast)
    ses(y, h), holt(y, h, damped, exponential), hw(y, h, seasonal, damped)   -> fpp.Forecast
    simulate(fit, nsim, npaths)  -> future sample paths (for simulated prediction intervals)

Part of a personal study project following https://otexts.com/fpp2/ -- credit to the authors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize
from statsforecast.ets import ets_f, forecast_ets, pegelsresid_C

import fpp

_PAR_NAMES = ["alpha", "beta", "gamma", "phi"]


@dataclass
class ETSFit:
    y: pd.Series
    model: dict
    lambda_: Optional[float] = None
    biasadj: bool = False

    # --- R-like accessors ---------------------------------------------------
    @property
    def method(self) -> str:
        return self.model["method"]

    @property
    def components(self):
        return self.model["components"]

    @property
    def par(self) -> dict:
        """Estimated parameters named as in R (alpha, beta, gamma, phi, l, b, s0, s1, ...)."""
        p = np.asarray(self.model["par"], dtype=float)
        out = {}
        for i, nm in enumerate(_PAR_NAMES):
            if i < len(p) and not np.isnan(p[i]):
                out[nm] = p[i]
        states = p[4:]
        comps = self.components
        names = ["l"]
        if comps[1] != "N":
            names.append("b")
        m = self.model["m"]
        if comps[2] != "N":
            names += [f"s{j}" for j in range(m - 1)]
        for nm, v in zip(names, states):
            out[nm] = v
        return out

    @property
    def sigma2(self) -> float:
        return self.model["sigma2"]

    @property
    def aic(self):
        return self.model["aic"]

    @property
    def aicc(self):
        return self.model["aicc"]

    @property
    def bic(self):
        return self.model["bic"]

    @property
    def loglik(self):
        return self.model["loglik"]

    @property
    def fitted(self) -> pd.Series:
        f = np.asarray(self.model["fitted"], dtype=float)
        if self.lambda_ is not None:
            f = fpp.inv_box_cox(f, self.lambda_)
        return pd.Series(f, index=self.y.index, name="fitted")

    @property
    def residuals(self) -> pd.Series:
        """Innovation residuals (relative errors for multiplicative-error models, as in R)."""
        return pd.Series(np.asarray(self.model["residuals"], dtype=float), index=self.y.index,
                         name="residuals")

    def response_residuals(self) -> pd.Series:
        """y - fitted on the original scale (what accuracy() uses)."""
        return self.y - self.fitted

    def states(self) -> pd.DataFrame:
        st = np.asarray(self.model["states"], dtype=float)
        comps = self.components
        cols = ["level"]
        if comps[1] != "N":
            cols.append("slope")
        if comps[2] != "N":
            cols += [f"s{j + 1}" for j in range(st.shape[1] - len(cols))]
        return pd.DataFrame(st[:, :len(cols)], columns=cols)

    def summary(self) -> str:
        p = ", ".join(f"{k} = {v:.4f}" for k, v in self.par.items())
        return (f"{self.method}\n  {p}\n  sigma^2 = {self.sigma2:.4g}  "
                f"AIC = {self.aic:.2f}  AICc = {self.aicc:.2f}  BIC = {self.bic:.2f}")

    def __repr__(self):  # pragma: no cover
        return self.summary()

    # --- forecasting ----------------------------------------------------------
    def forecast(self, h: int = None, level: Sequence[int] = (80, 95)) -> fpp.Forecast:
        m = int(round(fpp.frequency(self.y)))
        if h is None:
            h = 2 * m if m > 1 else 10
        p = forecast_ets(self.model, h=h, level=list(level))
        idx = fpp._future_index(self.y, h)
        comps = self.components
        class3 = (comps[0] == "M" and comps[2] == "M") or comps[1] == "M"
        if class3:
            # no simple analytic variance (R uses an approximation); use 5000 simulated paths
            mean_pt = _point_forecast(self.model, h)
            lam, self.lambda_ = self.lambda_, None
            paths = simulate(self, nsim=h, npaths=5000, seed=123)
            self.lambda_ = lam
            p = {"mean": mean_pt}
            for lv in level:
                p[f"lo-{lv}"] = np.quantile(paths, 0.5 - lv / 200, axis=0)
                p[f"hi-{lv}"] = np.quantile(paths, 0.5 + lv / 200, axis=0)
        mean0 = np.asarray(p["mean"], dtype=float)
        # statsforecast's compiled forecast mis-accumulates phi for damped trends;
        # recompute point forecasts with the book's formulas and shift the intervals.
        mean = _point_forecast(self.model, h)
        shift = mean - mean0
        lo = {lv: np.asarray(p[f"lo-{lv}"], dtype=float) + shift for lv in level}
        hi = {lv: np.asarray(p[f"hi-{lv}"], dtype=float) + shift for lv in level}
        if self.lambda_ is not None:
            lam = self.lambda_
            if self.biasadj:
                # variance on the transformed scale from the 95% (or first) interval width
                lv = level[-1]
                from scipy import stats
                se = (hi[lv] - lo[lv]) / (2 * stats.norm.ppf(0.5 + lv / 200))
                mean = fpp.inv_box_cox(mean, lam, biasadj=True, fvar=se ** 2)
            else:
                mean = fpp.inv_box_cox(mean, lam)
            lo = {k: fpp.inv_box_cox(v, lam) for k, v in lo.items()}
            hi = {k: fpp.inv_box_cox(v, lam) for k, v in hi.items()}
        fitted = self.fitted
        return fpp.Forecast(self.method, self.y, pd.Series(mean, idx, name="mean"),
                            pd.DataFrame(lo, idx), pd.DataFrame(hi, idx),
                            fitted, self.y - fitted, tuple(level), extra={"fit": self})


def _point_forecast(model: dict, h: int) -> np.ndarray:
    """ETS point forecasts (Table 7.8 of the book) from the final state vector."""
    comps = model["components"]
    trend, season, damped = comps[1], comps[2], comps[3] == "D"
    phi = float(model["par"][3]) if damped else 1.0
    x = np.asarray(model["states"], dtype=float)[-1]
    lvl = x[0]
    b = x[1] if trend != "N" else 0.0
    m = int(model["m"])
    s = x[(2 if trend != "N" else 1):][:m] if season != "N" else None
    out = np.empty(h)
    for i in range(1, h + 1):
        phistar = np.sum(phi ** np.arange(1, i + 1)) if damped else i
        if trend == "N":
            tr = lvl
        elif trend == "A":
            tr = lvl + phistar * b
        else:
            tr = lvl * b ** phistar
        if season == "N":
            out[i - 1] = tr
        else:
            sv = s[m - 1 - ((i - 1) % m)]
            out[i - 1] = tr + sv if season == "A" else tr * sv
    return out


def _r_sigma2(model: dict) -> float:
    """R: sigma2 = sum(e^2) / (n - length(par))."""
    e = np.asarray(model["residuals"], dtype=float)
    npar = int(np.sum(~np.isnan(np.asarray(model["par"], dtype=float))))
    if model["components"][2] != "N":
        npar -= 1          # R does not count the implied last seasonal state
    return float(np.nansum(e ** 2) / (np.sum(~np.isnan(e)) - npar))


def _refit_fixed(z, fit, fixed, opt_crit):
    comps = fit["components"]
    err, trend, season, damped = comps[0], comps[1], comps[2], comps[3] == "D"
    m = int(fit["m"])
    par0 = np.asarray(fit["par"], dtype=float).copy()
    names = ["alpha", "beta", "gamma", "phi"]
    for i, nm in enumerate(names):
        if nm in fixed:
            par0[i] = fixed[nm]
    use = [i for i, nm in enumerate(names) if not np.isnan(par0[i]) and nm not in fixed]
    nsm = 4
    nst = len(par0) - nsm
    nfree = nst - (1 if season != "N" else 0)     # last seasonal state is implied (sum constraint)
    lead = 1 + (trend != "N")
    x0 = np.r_[par0[use], par0[nsm:nsm + nfree]]

    def unpack(x):
        p = par0.copy()
        p[use] = x[:len(use)]
        p[nsm:nsm + nfree] = x[len(use):]
        if season != "N":
            p[-1] = m * (season == "M") - np.sum(p[nsm + lead:nsm + nfree])
        return p

    lo_, hi_ = 1e-4, 0.9999

    def sig(u):
        return 1 / (1 + np.exp(-np.clip(u, -500, 500)))

    def logit(v):
        v = min(max(v, 1e-6), 1 - 1e-6)
        return np.log(v / (1 - v))

    # smooth re-parameterisation enforcing the usual constraints
    #   alpha in (1e-4, .9999), beta in (1e-4, alpha), gamma in (1e-4, 1-alpha), phi in (.8, .98)
    def to_par(u):
        x = np.asarray(u, dtype=float).copy()
        a = fixed.get("alpha")
        k = 0
        vals = {}
        for i in use:
            if i == 0:
                vals[0] = lo_ + (hi_ - lo_) * sig(u[k])
            k += 1
        a = vals.get(0, par0[0])
        k = 0
        for i in use:
            if i == 1:
                vals[1] = lo_ + (a - lo_) * sig(u[k])
            elif i == 2:
                vals[2] = lo_ + (1 - a - lo_) * sig(u[k])
            elif i == 3:
                vals[3] = 0.8 + 0.18 * sig(u[k])
            k += 1
        for j, i in enumerate(use):
            x[j] = vals[i]
        return x

    def from_par(x):
        u = np.asarray(x, dtype=float).copy()
        p = unpack(x)
        a = p[0]
        for j, i in enumerate(use):
            v = x[j]
            if i == 0:
                u[j] = logit((v - lo_) / (hi_ - lo_))
            elif i == 1:
                u[j] = logit((v - lo_) / max(a - lo_, 1e-6))
            elif i == 2:
                u[j] = logit((v - lo_) / max(1 - a - lo_, 1e-6))
            elif i == 3:
                u[j] = logit((v - 0.8) / 0.18)
        return u

    def obj_u(u):
        p = unpack(to_par(u))
        a, b, g, ph = [(0.0 if np.isnan(v) else v) for v in p[:4]]
        amse, e, xs, lik = pegelsresid_C(z, m, p[nsm:], err, trend, season, damped,
                                         a, b, g, ph if damped else 1.0, 3)
        val = amse[0] if opt_crit == "mse" else lik
        return val if np.isfinite(val) else 1e12

    starts = [x0]
    for a0 in ((0.5,) if season != "N" else (0.2, 0.5, 0.8)):
        x = x0.copy()
        for j, i in enumerate(use):
            x[j] = {0: a0, 1: 0.1 * a0, 2: 0.1 * (1 - a0), 3: 0.95}[i]
        starts.append(x)
    best = None
    for start in starts:
        u0 = from_par(start)
        if obj_u(u0) >= 1e12:
            continue
        r = optimize.minimize(obj_u, u0, method="Nelder-Mead",
                              options={"maxiter": 4000, "maxfev": 6000, "xatol": 1e-8, "fatol": 1e-10, "adaptive": True})
        r = optimize.minimize(obj_u, r.x, method="Nelder-Mead",
                              options={"maxiter": 2000, "maxfev": 3000, "xatol": 1e-9, "fatol": 1e-11, "adaptive": True})
        if best is None or r.fun < best.fun:
            best = r
    if best is None or best.fun >= obj_u(from_par(x0)):
        return fit
    best.x = to_par(best.x)
    p = unpack(best.x)
    new = dict(fit)
    new["par"] = p
    new["n_params"] = len(use) + len(p) - nsm
    out = ets_f(z, m, model=new)
    out["method"] = fit["method"]
    return out


def ets(y, model: str = "ZZZ", damped: Optional[bool] = None, alpha=None, beta=None, gamma=None,
        phi=None, lambda_: Optional[float] = None, biasadj: bool = False, opt_crit: str = "lik",
        ic: str = "aicc", additive_only: bool = False, allow_multiplicative_trend: bool = False,
        restrict: bool = True, polish: Optional[bool] = None) -> ETSFit:
    """forecast::ets -- fit (and, with 'Z' letters, select) an ETS model.

    polish: refine the statsforecast estimate of a fully specified model (no 'Z') with a
    multi-start Nelder-Mead search. Default (None) = only for *seasonal* models, where
    statsforecast's optimiser sometimes gets stuck at the parameter bounds; for
    non-seasonal models its estimates already agree closely with R's."""
    y = y if isinstance(y, pd.Series) else fpp.make_ts(y)
    m = int(round(fpp.frequency(y)))
    if m > 24 and model[-1] in "ZAM":
        # R: ets() cannot handle seasonality above 24 -> non-seasonal model
        model = model[:2] + "N"
    z = y.to_numpy(dtype=float)
    if lambda_ is not None:
        z = fpp.box_cox(z, lambda_)
        additive_only = True
    fixed = {k: v for k, v in dict(alpha=alpha, beta=beta, gamma=gamma, phi=phi).items() if v is not None}
    if fixed and phi is not None and damped is None:
        damped = True
    fit = ets_f(z, max(m, 1), model=model, damped=damped, opt_crit=opt_crit, ic=ic,
                additive_only=additive_only,
                allow_multiplicative_trend=allow_multiplicative_trend, restrict=restrict)
    if polish is None:
        polish = len(model) == 3 and model[2] in "AM"
    if fixed or (polish and "Z" not in model):
        # statsforecast ignores user-fixed smoothing parameters, so re-optimise the
        # remaining parameters and initial states ourselves with the fixed values held.
        fit = _refit_fixed(z, fit, fixed, opt_crit)
    fit["sigma2"] = _r_sigma2(fit)
    return ETSFit(y, fit, lambda_, biasadj)


def ses(y, h: int = 10, level=(80, 95), alpha=None) -> fpp.Forecast:
    """forecast::ses -- simple exponential smoothing (ETS(A,N,N), optimised by MSE)."""
    fc = ets(y, "ANN", alpha=alpha, opt_crit="mse").forecast(h, level)
    fc.method = "Simple exponential smoothing"
    return fc


def holt(y, h: int = 10, damped: bool = False, exponential: bool = False, level=(80, 95),
         alpha=None, beta=None, phi=None) -> fpp.Forecast:
    """forecast::holt -- Holt's linear (or damped / exponential) trend method."""
    model = "MMN" if exponential else "AAN"
    fc = ets(y, model, damped=damped, alpha=alpha, beta=beta, phi=phi, opt_crit="mse",
             allow_multiplicative_trend=exponential).forecast(h, level)
    fc.method = ("Damped " if damped else "") + ("Holt's method" if not exponential else "Exponential trend method")
    return fc


def hw(y, h: Optional[int] = None, seasonal: str = "additive", damped: bool = False,
       exponential: bool = False, level=(80, 95)) -> fpp.Forecast:
    """forecast::hw -- Holt-Winters additive or multiplicative seasonal method."""
    m = int(round(fpp.frequency(y)))
    h = h or 2 * m
    if seasonal == "additive":
        model = "AAA"
    else:
        model = "MMM" if exponential else "MAM"
    fc = ets(y, model, damped=damped, opt_crit="mse",
             allow_multiplicative_trend=exponential).forecast(h, level)
    fc.method = ("Damped " if damped else "") + f"Holt-Winters' {seasonal} method"
    return fc


def simulate(fit: ETSFit, nsim: int, npaths: int = 1, seed: Optional[int] = None,
             bootstrap: bool = False) -> np.ndarray:
    """Simulate future sample paths from a fitted ETS model (forecast::simulate.ets).
    Returns an (npaths x nsim) array on the original scale."""
    rng = np.random.default_rng(seed)
    mod = fit.model
    comps = mod["components"]
    err, trend, season, damped = comps[0], comps[1], comps[2], comps[3] == "D"
    par = fit.par
    alpha = par.get("alpha", 0.0)
    beta = par.get("beta", 0.0)
    gamma = par.get("gamma", 0.0)
    phi = par.get("phi", 1.0) if damped else 1.0
    last = np.asarray(mod["states"], dtype=float)[-1]
    mlen = int(mod["m"])
    sigma = np.sqrt(fit.sigma2)
    resid = fit.residuals.dropna().to_numpy()
    # vectorised over paths: each array has one entry per sample path
    lvl = np.full(npaths, last[0])
    b = np.full(npaths, last[1]) if trend != "N" else np.zeros(npaths)
    if season != "N":
        s = np.tile(last[(2 if trend != "N" else 1):][:mlen], (npaths, 1))   # column 0 = most recent
    out = np.zeros((npaths, nsim))
    for t in range(nsim):
        e = rng.choice(resid, size=npaths) if bootstrap else rng.normal(0, sigma, size=npaths)
        if trend == "N":
            tr = lvl
        elif trend == "A":
            tr = lvl + phi * b
        else:
            tr = lvl * b ** phi
        if season == "N":
            sv = np.zeros(npaths) if True else None
            mu = tr
        else:
            sv = s[:, -1]
            mu = tr + sv if season == "A" else tr * sv
        out[:, t] = mu + e if err == "A" else mu * (1 + e)
        eps_l = e if err == "A" else mu * e          # innovation on the original scale
        if season == "M":
            lvl_new = tr + alpha * eps_l / sv
            adj = eps_l / sv
        else:
            lvl_new = tr + alpha * eps_l
            adj = eps_l
        if trend == "A":
            b = phi * b + beta * adj
        elif trend == "M":
            b = b ** phi + beta * adj / lvl
        if season == "A":
            s = np.column_stack([sv + gamma * eps_l, s[:, :-1]])
        elif season == "M":
            s = np.column_stack([sv + gamma * eps_l / tr, s[:, :-1]])
        lvl = lvl_new
    if fit.lambda_ is not None:
        out = fpp.inv_box_cox(out, fit.lambda_)
    return out
