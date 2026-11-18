"""Chapter 6 — every decomposition / decomposition forecast in Python, checked against R."""
import json

import numpy as np
import pytest

import fpp
import fpp_ch06 as c6
from conftest import REF

ref = json.loads((REF / "ch06_reference.json").read_text())
elecequip = fpp.load("elecequip")
beer2 = fpp.window(fpp.load("ausbeer"), start=1992)


def arr(key):
    return np.array([np.nan if v is None else v for v in ref[key]], dtype=float)


def close(a, key, rtol=1e-6, atol=1e-8):
    np.testing.assert_allclose(np.asarray(a, dtype=float), arr(key), rtol=rtol, atol=atol, equal_nan=True)


def valid(fc, h):
    m = fc.mean.to_numpy()
    assert m.shape == (h,) and np.all(np.isfinite(m))
    assert np.all(fc.lower[95].to_numpy() <= fc.lower[80].to_numpy() + 1e-8)
    assert np.all(fc.lower[80].to_numpy() <= m + 1e-8) and np.all(m <= fc.upper[80].to_numpy() + 1e-8)


def test_moving_averages_match_R():
    close(c6.ma(fpp.load("elecsales"), 5), "ma5")
    close(c6.ma(beer2, 4, centre=False), "ma4")
    close(c6.ma(beer2, 4), "ma2x4")
    close(c6.ma(elecequip, 12), "ma12")


def test_classical_decomposition_matches_R():
    dm = c6.classical(elecequip, "multiplicative")
    close(dm.seasonal.iloc[:12], "dec_mult_seasonal")
    close(dm.trend, "dec_mult_trend")
    close(c6.classical(elecequip, "additive").seasonal.iloc[:12], "dec_add_seasonal")


def test_stl_matches_R():
    f = c6.stl_r(elecequip, "periodic", t_window=13, robust=True)
    close(f.trend, "stlp_trend", rtol=1e-6)
    close(f.seasonal, "stlp_seasonal", rtol=1e-6, atol=1e-6)
    f7 = c6.stl_r(elecequip, 7)
    close(f7.trend, "stl7_trend", rtol=1e-6)
    close(f7.seasonal, "stl7_seasonal", rtol=1e-6, atol=1e-6)
    close(c6.stl_r(fpp.load("bricksq"), 7, robust=True).seasonal, "stlb_seasonal", rtol=1e-5, atol=1e-5)
    ms = c6.mstl_r(elecequip)
    close(ms.seasonal, "mstl_seasonal", atol=1e-6)
    close(ms.trend, "mstl_trend", rtol=1e-6)


@pytest.mark.skipif(c6.find_x13() is None, reason="X-13ARIMA-SEATS binary not installed")
def test_x11_and_seats_match_R():
    x11 = c6.x13(elecequip, "x11")
    assert x11.type == "multiplicative"
    close(x11.seasonal, "x11_seasonal", rtol=1e-4)
    close(x11.trend, "x11_trend", rtol=1e-4)
    close(x11.seasadj_x13, "x11_sa", rtol=1e-4)
    seats = c6.x13(elecequip, "seats")
    close(seats.seasonal, "seats_seasonal", rtol=1e-4)
    close(seats.trend, "seats_trend", rtol=1e-4)


def test_strength_in_unit_interval():
    s = c6.strength(c6.mstl_r(elecequip))
    assert 0.9 < s["trend_strength"] <= 1 and 0.8 < s["seasonal_strength"] <= 1


def test_stl_forecasts_match_R():
    f = c6.stl_r(elecequip, "periodic", t_window=13, robust=True)
    fn = c6.forecast_stl(f, h=24, method="naive")
    valid(fn, 24)
    close(fn.mean, "fc_naive_mean", rtol=1e-6)
    close(fn.upper[95], "fc_naive_hi95", rtol=1e-6)
    fd = c6.forecast_stl(f, h=24, method="rwdrift")
    close(fd.mean, "fc_drift_mean", rtol=1e-6)
    close(fd.lower[80], "fc_drift_lo80", rtol=1e-6)
    sf = c6.stlf(fpp.load("fancy"), method="rwdrift", lambda_=0)
    close(sf.mean, "stlf_fancy_mean", rtol=1e-6)
    close(sf.upper[95], "stlf_fancy_hi95", rtol=1e-6)
    close(c6.stlf(fpp.load("bricksq"), method="naive", s_window=7).mean, "stlf_bricks_mean", rtol=1e-6)
    se = c6.stlf(elecequip)
    assert se.method.replace(" ", "") == ref["stlf_elec_method"].replace(" ", "")
    close(se.mean, "stlf_elec_mean", rtol=2e-2)       # ETS optimisers differ slightly


def test_every_decomposition_forecast_method_runs_and_beats_naive():
    train = fpp.window(elecequip, end=(2009, 12)); test = fpp.window(elecequip, start=2010)
    h = len(test)
    naive_rmse = fpp.accuracy(fpp.naive(train, h=h), test).loc["Test set", "RMSE"]
    decs = [c6.stl_r(train, "periodic", t_window=13, robust=True), c6.classical(train, "multiplicative")]
    if c6.find_x13() is not None:
        decs += [c6.x13(train, "x11"), c6.x13(train, "seats")]
    for d in decs:
        for m in ["naive", "rwdrift", "ets", "arima"]:
            fc = c6.forecast_stl(d, h=h, method=m)
            valid(fc, h)
            assert fpp.accuracy(fc.mean.to_numpy(), test, train=train).loc["Test set", "RMSE"] < naive_rmse
    for m in ["ets", "naive", "rwdrift"]:
        valid(c6.stlf(train, h=h, method=m), h)
    valid(fpp.snaive(train, h=h), h)
