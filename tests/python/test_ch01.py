"""Chapter 1 — every model used in the notes, in Python, checked against R."""
import json

import numpy as np
import pytest
import statsmodels.formula.api as smf
from statsforecast.models import ARIMA, AutoARIMA, AutoETS

import fpp
from conftest import REF

ref = json.loads((REF / "ch01_reference.json").read_text())
elec = fpp.load("elecdaily")
train, test = elec.iloc[:300], elec.iloc[300:]
H = len(test)


def xreg(d):
    return np.column_stack([d["Temperature"], d["Temperature"] ** 2, d["WorkDay"]])


def valid(mean, lo=None, hi=None, h=H):
    mean = np.asarray(mean)
    assert mean.shape == (h,) and np.all(np.isfinite(mean))
    if lo is not None:
        assert np.all(np.asarray(lo) <= mean + 1e-8) and np.all(mean <= np.asarray(hi) + 1e-8)


def test_explanatory_ols_matches_R_tslm():
    fit = smf.ols("Demand ~ Temperature + I(Temperature**2) + WorkDay", data=train).fit()
    np.testing.assert_allclose(fit.params.to_numpy(), ref["tslm_coef"], rtol=1e-8)
    valid(fit.predict(test).to_numpy())


def test_time_series_autoets_forecasts():
    m = AutoETS(season_length=7).fit(train["Demand"].to_numpy())
    assert m.model_["method"].startswith("ETS")
    p = m.predict(h=H, level=[80, 95])
    valid(p["mean"], p["lo-80"], p["hi-80"])
    assert np.all(p["lo-95"] <= p["lo-80"] + 1e-8)


def test_mixed_model_same_order_matches_R():
    """Regression with ARIMA(1,0,0)(2,0,0)[7] errors -- the model R's auto.arima picks.
    Fitting the SAME model in Python must give the same coefficients and forecasts."""
    m = ARIMA(order=(1, 0, 0), seasonal_order=(2, 0, 0), season_length=7,
              include_mean=True).fit(train["Demand"].to_numpy(), X=xreg(train))
    coef = np.array(list(m.model_["coef"].values()))
    np.testing.assert_allclose(coef, ref["mixed_coef"], rtol=2e-3, atol=2e-2)
    p = m.predict(h=H, X=xreg(test), level=[80, 95])
    valid(p["mean"], p["lo-80"], p["hi-80"])
    np.testing.assert_allclose(p["mean"], ref["mixed_fc"], rtol=1e-3)


def test_mixed_model_auto_arima_is_valid_and_beats_benchmark():
    """Python's own automatic search. Its chosen orders can differ from R's (and between
    computers/library versions), so we test that it forecasts sensibly, not that it matches."""
    m = AutoARIMA(season_length=7).fit(train["Demand"].to_numpy(), X=xreg(train))
    p = m.predict(h=H, X=xreg(test), level=[80, 95])
    valid(p["mean"], p["lo-80"], p["hi-80"])
    bench = fpp.snaive(train["Demand"], h=H)
    acc_m = fpp.accuracy(p["mean"], test["Demand"], train=train["Demand"]).loc["Test set", "RMSE"]
    acc_b = fpp.accuracy(bench, test["Demand"]).loc["Test set", "RMSE"]
    assert acc_m < acc_b


def test_austa_ets_matches_R():
    austa = fpp.load("austa")
    m = AutoETS(season_length=1).fit(austa.to_numpy())
    assert m.model_["method"] == ref["austa_method"]
    np.testing.assert_allclose(m.predict(h=10)["mean"], ref["austa_fc"], rtol=1e-3)


def test_seasonal_naive_benchmark_matches_R():
    fc = fpp.snaive(train["Demand"], h=H)
    np.testing.assert_allclose(fc.mean.to_numpy(), ref["snaive_fc"], rtol=1e-10)


def test_beer_ets_and_simulation():
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    beer2 = fpp.window(fpp.load("ausbeer"), start=1992)
    p = AutoETS(season_length=4).fit(beer2.to_numpy()).predict(h=8, level=[80])
    valid(p["mean"], p["lo-80"], p["hi-80"], h=8)
    austa = fpp.load("austa")
    fit = ETSModel(austa.astype(float).set_axis(austa.index.to_timestamp()), error="add", trend="add").fit(disp=False)
    np.random.seed(1)
    sims = fit.simulate(nsimulations=10, repetitions=10, anchor="end")
    assert sims.shape == (10, 10) and np.all(np.isfinite(sims.to_numpy()))
