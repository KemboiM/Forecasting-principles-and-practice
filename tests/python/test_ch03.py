"""Chapter 3 — every benchmark method, transformation, diagnostic and accuracy tool,
in Python, checked against the R forecast package (tests/reference/ch03_reference.json)."""
import json

import numpy as np
import pytest

import fpp
import fpp_ch03 as ch3
from conftest import REF

ref = json.loads((REF / "ch03_reference.json").read_text())
ausbeer, goog, goog200 = fpp.load("ausbeer"), fpp.load("goog"), fpp.load("goog200")
a10, eggs = fpp.load("a10"), fpp.load("eggs")
beer_tr = fpp.window(ausbeer, start=1992, end=(2007, 4)); beer_te = fpp.window(ausbeer, start=2008)
g_te = fpp.window(goog, start=201, end=240)
a10_tr = fpp.window(a10, end=(2005, 12)); a10_te = fpp.window(a10, start=2006)
eggs_tr = fpp.window(eggs, end=1983); eggs_te = fpp.window(eggs, start=1984)
ibm = fpp.load("ibmclose")
ibm_tr = ibm.iloc[:300]; ibm_tr.attrs = dict(ibm.attrs); ibm_te = ibm.iloc[300:]
hsales = fpp.load("hsales")
hs_tr = fpp.window(hsales, end=(1993, 11)); hs_te = fpp.window(hsales, start=(1993, 12))
COLS = ["ME", "RMSE", "MAE", "MPE", "MAPE", "MASE"]


def valid(fc, h):
    m = fc.mean.to_numpy()
    assert m.shape == (h,) and np.all(np.isfinite(m))
    for lv in fc.level:
        assert np.all(fc.lower[lv].to_numpy() <= m + 1e-8) and np.all(m <= fc.upper[lv].to_numpy() + 1e-8)
    if len(fc.level) == 2:
        assert np.all(fc.lower[95].to_numpy() <= fc.lower[80].to_numpy() + 1e-8)


lam_a10 = fpp.box_cox_lambda(a10_tr)
MODELS = {
    "beer_mean": (lambda: fpp.meanf(beer_tr, h=11), beer_te),
    "beer_naive": (lambda: fpp.naive(beer_tr, h=11), beer_te),
    "beer_snaive": (lambda: fpp.snaive(beer_tr, h=11), beer_te),
    "beer_drift": (lambda: fpp.rwf(beer_tr, drift=True, h=11), beer_te),
    "goog_mean": (lambda: fpp.meanf(goog200, h=40), g_te),
    "goog_naive": (lambda: fpp.naive(goog200, h=40), g_te),
    "goog_drift": (lambda: fpp.rwf(goog200, drift=True, h=40), g_te),
    "a10_snaive_bc": (lambda: fpp.snaive(a10_tr, h=31, lambda_=lam_a10), a10_te),
    "a10_drift_log": (lambda: fpp.rwf(a10_tr, drift=True, h=31, lambda_=0), a10_te),
    "a10_drift_bc": (lambda: fpp.rwf(a10_tr, drift=True, h=31, lambda_=lam_a10), a10_te),
    "eggs_log": (lambda: fpp.rwf(eggs_tr, drift=True, lambda_=0, h=10), eggs_te),
    "eggs_log_biasadj": (lambda: fpp.rwf(eggs_tr, drift=True, lambda_=0, h=10, biasadj=True), eggs_te),
    "ibm_mean": (lambda: fpp.meanf(ibm_tr, h=69), ibm_te),
    "ibm_naive": (lambda: fpp.naive(ibm_tr, h=69), ibm_te),
    "ibm_drift": (lambda: fpp.rwf(ibm_tr, drift=True, h=69), ibm_te),
    "hs_snaive": (lambda: fpp.snaive(hs_tr, h=24), hs_te),
    "hs_drift": (lambda: fpp.rwf(hs_tr, drift=True, h=24), hs_te),
}


@pytest.mark.parametrize("key", list(MODELS))
def test_every_benchmark_model_forecasts_and_matches_R_accuracy(key):
    make, test = MODELS[key]
    fc = make()
    valid(fc, len(fc.mean))
    got = fpp.accuracy(fc, test).loc["Test set", COLS].to_numpy(dtype=float)
    exp = np.array([ref["acc"][key][c] for c in COLS], dtype=float)
    np.testing.assert_allclose(got, exp, rtol=1e-6, err_msg=key)


@pytest.mark.parametrize("name", ["elec", "usnetelec", "usgdp", "mcopper", "enplanements", "cangas",
                                  "dole", "usdeaths", "bricksq"])
def test_box_cox_lambda_matches_R(name):
    assert fpp.box_cox_lambda(fpp.load(name)) == pytest.approx(ref["lambda"][name], abs=2e-3)


def test_calendar_adjustment_milk():
    milk = fpp.load("milk")
    np.testing.assert_allclose((milk / ch3.monthdays(milk)).to_numpy()[:12], ref["milk_daily_head"], rtol=1e-10)


def test_bias_adjusted_eggs_forecasts_match_R():
    fc = fpp.rwf(eggs, drift=True, lambda_=0, h=50, level=(80,))
    np.testing.assert_allclose(fc.mean.to_numpy(), ref["eggs_fc"]["mean"], rtol=1e-8)
    np.testing.assert_allclose(fc.lower[80].to_numpy(), np.ravel(ref["eggs_fc"]["lower"]), rtol=1e-8)
    fc2 = fpp.rwf(eggs, drift=True, lambda_=0, h=50, level=(80,), biasadj=True)
    np.testing.assert_allclose(fc2.mean.to_numpy(), ref["eggs_fc_biasadj"], rtol=1e-8)
    assert np.all(fc2.mean.to_numpy() > fc.mean.to_numpy())       # mean > median


def test_residual_diagnostics_match_R():
    res = fpp.naive(goog200).residuals
    bp = ch3.box_pierce(res, lag=10)
    assert bp["Q"] == pytest.approx(ref["box_pierce"]["Q"], rel=1e-8)
    assert bp["p-value"] == pytest.approx(ref["box_pierce"]["p"], rel=1e-6)
    assert fpp.ljung_box(fpp.snaive(fpp.window(ausbeer, start=1992)).residuals, lag=8)["Q*"] == \
        pytest.approx(ref["lb_beer_snaive"], rel=1e-8)
    assert fpp.ljung_box(fpp.naive(ch3.load_extra("WWWusage")).residuals, lag=10)["Q*"] == \
        pytest.approx(ref["lb_www_naive"], rel=1e-8)


def test_naive_prediction_intervals_match_R():
    fc = fpp.naive(goog200)
    lo, hi = np.array(ref["naive_goog"]["lower"]), np.array(ref["naive_goog"]["upper"])
    np.testing.assert_allclose(fc.lower[80].to_numpy(), lo[:, 0], rtol=1e-10)
    np.testing.assert_allclose(fc.upper[95].to_numpy(), hi[:, 1], rtol=1e-10)


def test_bootstrap_intervals_are_sensible():
    fb = ch3.naive_bootstrap(goog200, h=10, seed=1)
    valid(fb, 10)
    fn = fpp.naive(goog200, h=10)
    width_b = (fb.upper[95] - fb.lower[95]).to_numpy()
    width_n = (fn.upper[95] - fn.lower[95]).to_numpy()
    assert np.all(np.diff(width_b) > -1e-8)                     # widen with horizon
    assert np.all(np.abs(width_b / width_n - 1) < 0.35)          # similar to normal intervals


def test_tscv_matches_R():
    cv = lambda f, **kw: float(np.sqrt(np.nanmean(fpp.tsCV(goog200, f, h=1, **kw).to_numpy() ** 2)))
    assert cv(fpp.meanf) == pytest.approx(ref["tscv"]["mean"], rel=1e-8)
    assert cv(fpp.naive) == pytest.approx(ref["tscv"]["naive"], rel=1e-8)
    assert cv(fpp.rwf, drift=True) == pytest.approx(ref["tscv"]["drift"], rel=1e-8)


def test_visnights_mape_by_split_matches_R():
    vis = fpp.load("visnights")
    qld = vis["QLDMetro"]; qld.attrs = dict(vis.attrs)
    got = []
    for end in [2015, 2014, 2013]:
        tr = fpp.window(qld, end=(end, 4)); te = fpp.window(qld, start=end + 1, end=(end + 1, 4))
        got.append(fpp.accuracy(fpp.snaive(tr, h=4), te).loc["Test set", "MAPE"])
    np.testing.assert_allclose(got, ref["qld_mape"], rtol=1e-8)


def test_drift_equals_first_to_last_line():
    dow = fpp.load("dowjones")
    fc = fpp.rwf(dow, drift=True, h=20)
    np.testing.assert_allclose(fc.mean.to_numpy(), ref["dowjones_drift"], rtol=1e-10)
    y = dow.to_numpy()
    np.testing.assert_allclose(fc.mean.to_numpy(), y[-1] + (y[-1] - y[0]) / (len(y) - 1) * np.arange(1, 21))


def test_forecast_default_is_ets():
    from statsforecast.models import AutoETS
    m = AutoETS(season_length=4).fit(beer_tr.to_numpy())
    p = m.predict(h=11, level=[80, 95])
    assert m.model_["method"].startswith("ETS")
    assert np.all(p["lo-95"] <= p["mean"]) and np.all(p["mean"] <= p["hi-95"])
    acc_ets = fpp.accuracy(p["mean"], beer_te, train=beer_tr).loc["Test set", "MASE"]
    acc_snaive = fpp.accuracy(fpp.snaive(beer_tr, h=11), beer_te).loc["Test set", "MASE"]
    assert acc_ets < acc_snaive
