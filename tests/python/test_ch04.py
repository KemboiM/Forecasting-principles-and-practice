"""Chapter 4 — every model / computation used in the notes and practice exercises, checked against R."""
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsforecast.models import AutoETS
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

import fpp
from conftest import REF, ROOT

ref = json.loads((REF / "ch04_reference.json").read_text())
tfc = pd.read_csv(ROOT / "Chapter-04-Judgmental-Forecasts" / "tfc_forecasts.csv")
vn = fpp.make_ts(tfc["Data"] / 1000, start=1998, frequency=1)
vn_tr, vn_te = fpp.window(vn, end=2004), fpp.window(vn, start=2005)
a10 = fpp.load("a10")
a10_tr, a10_te = fpp.window(a10, end=(2005, 12)), fpp.window(a10, start=2006)


def valid(mean, lo=None, hi=None, h=None):
    mean = np.asarray(mean, dtype=float)
    if h is not None:
        assert mean.shape == (h,)
    assert np.all(np.isfinite(mean))
    if lo is not None:
        assert np.all(np.asarray(lo) <= mean + 1e-8) and np.all(mean <= np.asarray(hi) + 1e-8)


def test_anchoring_models_match_R():
    np.testing.assert_allclose(fpp.naive(a10_tr, h=30).mean, ref["a10_naive"], rtol=1e-10)
    np.testing.assert_allclose(fpp.snaive(a10_tr, h=30).mean, ref["a10_snaive"], rtol=1e-10)
    m = AutoETS(season_length=12).fit(a10_tr.to_numpy())
    p = m.predict(h=30, level=[80])
    valid(p["mean"], p["lo-80"], p["hi-80"], h=30)
    rmse_ets = fpp.accuracy(p["mean"], a10_te).loc["Test set", "RMSE"]
    rmse_naive = fpp.accuracy(fpp.naive(a10_tr, h=30), a10_te).loc["Test set", "RMSE"]
    assert rmse_ets < rmse_naive


def test_h02_forecast_log_models():
    h02 = fpp.load("h02")
    hist = h02.iloc[:-1]
    hist.attrs = dict(h02.attrs)
    valid(fpp.snaive(hist, h=1).mean, h=1)
    m = ETSModel(hist.to_numpy(), error="mul", trend="add", seasonal="mul", seasonal_periods=12).fit(disp=False)
    valid(m.forecast(1), h=1)


def test_delphi_and_scenarios_match_R():
    x = np.array([120, 135, 150, 95, 160, 210, 140, 130, 175, 60], float)
    mv = np.array([0.5, 0.3, 0.6, 0.4, 0.5, 0.2, 0.5, 0.6, 0.4, 0.3])
    x2 = x + mv * (np.median(x) - x); x3 = x2 + mv * (np.median(x2) - x2)
    from scipy import stats
    assert stats.trim_mean(x3, 0.1) == pytest.approx(137.3375)
    g = np.diff(np.log(vn.to_numpy()))
    np.testing.assert_allclose(np.quantile(g, [0.1, 0.5, 0.9]), ref["growth_q"], rtol=1e-10)


def test_visitor_night_models_match_R():
    hv = len(vn_te)
    d = fpp.rwf(vn_tr, drift=True, h=hv)
    np.testing.assert_allclose(d.mean, ref["vn_drift"], rtol=1e-10)
    np.testing.assert_allclose(d.lower[95], ref["vn_drift_lo95"], rtol=1e-10)
    X = sm.add_constant(np.arange(1, 8.0))
    lin = sm.OLS(vn_tr.to_numpy(), X).fit()
    pr = lin.get_prediction(sm.add_constant(np.arange(8, 15.0))).summary_frame(alpha=0.2)
    np.testing.assert_allclose(pr["mean"], ref["vn_lin"], rtol=1e-10)
    np.testing.assert_allclose(pr["obs_ci_upper"], ref["vn_lin_hi80"], rtol=1e-10)
    ann = AutoETS(season_length=1, model="ANN").fit(vn_tr.to_numpy()).predict(h=hv)["mean"]
    np.testing.assert_allclose(ann, ref["vn_ann"], rtol=5e-3)
    tfc_f = tfc.loc[tfc.Year >= 2005, "TFC"].to_numpy() / 1000
    ets_f = tfc.loc[tfc.Year >= 2005, "ETS"].to_numpy() / 1000
    assert fpp.accuracy(tfc_f, vn_te).loc["Test set", "ME"] < fpp.accuracy(ets_f, vn_te).loc["Test set", "ME"]


def test_informed_adjustment_beats_unadjusted():
    elec = fpp.load("elecdaily")
    train, test = elec.iloc[:300], elec.iloc[300:]
    m = AutoETS(season_length=7).fit(train["Demand"].to_numpy())
    fc = m.predict(h=len(test))["mean"]
    err = train["Demand"].to_numpy() - m.predict_in_sample()["fitted"]
    adj = smf.ols("err ~ Temperature + I(Temperature**2) + WorkDay", data=train.assign(err=err)).fit()
    informed = fc + adj.predict(test).to_numpy()
    r = lambda f: fpp.accuracy(f, test["Demand"]).loc["Test set", "RMSE"]
    assert r(informed) < r(fc)
