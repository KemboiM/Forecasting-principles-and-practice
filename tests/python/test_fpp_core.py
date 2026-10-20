"""Proves python/fpp.py reproduces the R `forecast` package numbers (tolerance 1e-6)."""
import json

import numpy as np
import pytest

import fpp
from conftest import REF

ref = json.loads((REF / "core_reference.json").read_text())
beer = fpp.load("ausbeer")
beer2 = fpp.window(beer, start=1992, end=(2007, 4))
beer3 = fpp.window(beer, start=2008)
goog = fpp.load("goog")
goog200 = fpp.load("goog200")


def same_fc(fc, r, tol=1e-6):
    np.testing.assert_allclose(fc.mean.to_numpy(), r["mean"], rtol=tol)
    np.testing.assert_allclose(fc.lower[80].to_numpy(), r["lo80"], rtol=tol)
    np.testing.assert_allclose(fc.upper[80].to_numpy(), r["hi80"], rtol=tol)
    np.testing.assert_allclose(fc.lower[95].to_numpy(), r["lo95"], rtol=tol)
    np.testing.assert_allclose(fc.upper[95].to_numpy(), r["hi95"], rtol=tol)


@pytest.mark.parametrize("name,fn", [
    ("meanf_beer", lambda: fpp.meanf(beer2, h=10)),
    ("naive_beer", lambda: fpp.naive(beer2, h=10)),
    ("snaive_beer", lambda: fpp.snaive(beer2, h=10)),
    ("drift_goog", lambda: fpp.rwf(goog200, h=40, drift=True)),
    ("naive_goog", lambda: fpp.naive(goog200, h=40)),
    ("snaive_a10_lambda", lambda: fpp.snaive(fpp.load("a10"), h=24, lambda_=0)),
    ("drift_lambda_elec", lambda: fpp.rwf(fpp.load("elec"), h=12, drift=True, lambda_=0.3)),
])
def test_benchmark_forecasts_match_R(name, fn):
    same_fc(fn(), ref[name])


@pytest.mark.parametrize("name,fn,test", [
    ("acc_meanf", lambda: fpp.meanf(beer2, h=10), beer3),
    ("acc_naive", lambda: fpp.rwf(beer2, h=10), beer3),
    ("acc_snaive", lambda: fpp.snaive(beer2, h=10), beer3),
    ("acc_drift_goog", lambda: fpp.rwf(goog200, h=40, drift=True), goog.iloc[200:240]),
])
def test_accuracy_matches_R(name, fn, test):
    a = fpp.accuracy(fn(), test)
    r = ref[name]
    for col in ["ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "ACF1"]:
        exp = [v if v is not None else np.nan for v in r[col]]
        got = a[col].to_numpy()
        k = len(exp)
        np.testing.assert_allclose(got[:k], exp, rtol=1e-6, atol=1e-9, err_msg=f"{name}:{col}")


@pytest.mark.parametrize("ds", ["elec", "a10", "ausbeer", "usnetelec"])
def test_box_cox_lambda_matches_R(ds):
    assert fpp.box_cox_lambda(fpp.load(ds)) == pytest.approx(ref[f"lambda_{ds}"], abs=2e-3)


def test_tscv_matches_R():
    e = fpp.tsCV(goog200, fpp.rwf, drift=True, h=1)
    assert np.sqrt(np.nanmean(e.to_numpy() ** 2)) == pytest.approx(ref["tscv_rmse_drift"], rel=1e-8)
    e8 = fpp.tsCV(goog200, fpp.naive, h=8)
    np.testing.assert_allclose(np.nanmean(e8.to_numpy() ** 2, axis=0), ref["tscv_naive_h8_mse"], rtol=1e-8)


def test_tscv_initial_and_window_match_R():
    for key, kw in [("tscv_initial50", dict(initial=50)), ("tscv_window30", dict(window=30))]:
        e = fpp.tsCV(goog200, fpp.naive, h=1, **kw).to_numpy()
        assert np.sum(~np.isnan(e)) == ref[key][0]
        assert np.sqrt(np.nanmean(e ** 2)) == pytest.approx(ref[key][1], rel=1e-10)


def test_ljung_box_matches_R():
    lb = fpp.ljung_box(fpp.naive(goog200).residuals, lag=10, df=0)
    assert lb["Q*"] == pytest.approx(ref["ljung_naive_goog"]["stat"], rel=1e-8)
    assert lb["p-value"] == pytest.approx(ref["ljung_naive_goog"]["p"], rel=1e-6)


def test_load_shapes():
    assert len(fpp.load("a10")) == 204 and fpp.frequency(fpp.load("a10")) == 12
    us = fpp.load("uschange")
    assert list(us.columns) == ["Consumption", "Income", "Production", "Savings", "Unemployment"]
    assert fpp.frequency(fpp.load("gasoline")) == pytest.approx(52.17857, rel=1e-5)
