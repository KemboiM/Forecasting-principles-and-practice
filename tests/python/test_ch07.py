"""Chapter 7 — every exponential smoothing method / ETS model in Python, checked against R."""
import json

import numpy as np
import pytest
from scipy import stats

import fpp
import fpp_ch07 as c7
from conftest import REF

ref = json.loads((REF / "ch07_reference.json").read_text())
oildata = fpp.window(fpp.load("oil"), start=1996)
air = fpp.window(fpp.load("ausair"), start=1990)
aust = fpp.window(fpp.load("austourists"), start=2005)


def valid(fc, h):
    m = fc.mean.to_numpy()
    assert m.shape == (h,) and np.all(np.isfinite(m))
    lo80, hi80 = fc.lower[80].to_numpy(), fc.upper[80].to_numpy()
    assert np.all(fc.lower[95].to_numpy() <= lo80 + 1e-8) and np.all(fc.upper[95].to_numpy() >= hi80 - 1e-8)
    assert np.all(lo80 <= m + 1e-8) and np.all(m <= hi80 + 1e-8)


def test_ses_matches_R():
    fc = c7.ses(oildata, h=5)
    valid(fc, 5)
    fit = fc.extra["fit"]
    np.testing.assert_allclose([fit.par["alpha"], fit.par["l"]], ref["ses_par"], rtol=1e-3)
    np.testing.assert_allclose(fc.mean, ref["ses_mean"], rtol=1e-4)
    np.testing.assert_allclose(fc.upper[95], ref["ses_hi95"], rtol=1e-4)
    assert fit.model["mse"] <= ref["ses_mse"] * (1 + 1e-6)


def test_holt_matches_R():
    fc = c7.holt(air, h=5)
    valid(fc, 5)
    np.testing.assert_allclose(fc.mean, ref["holt_mean"], rtol=2e-3)
    np.testing.assert_allclose(fc.upper[95], ref["holt_hi95"], rtol=2e-3)
    assert fc.extra["fit"].model["mse"] <= ref["holt_mse"] * (1 + 1e-6)


def test_damped_holt_fixed_phi_matches_R():
    fc = c7.holt(air, damped=True, phi=0.9, h=15)
    valid(fc, 15)
    p = fc.extra["fit"].par
    assert p["phi"] == pytest.approx(0.9)
    np.testing.assert_allclose([p["alpha"], p["beta"], p["l"], p["b"]],
                               [ref["damped_par"][i] for i in (0, 1, 2, 3)], rtol=1e-3)
    np.testing.assert_allclose(fc.mean, ref["damped_mean"], rtol=1e-4)
    np.testing.assert_allclose(fc.upper[95], ref["damped_hi95"], rtol=1e-3)
    assert fc.mean.iloc[-1] < c7.holt(air, h=15).mean.iloc[-1]


def test_exponential_trend_and_estimated_damping():
    valid(c7.holt(air, exponential=True, h=5), 5)
    valid(c7.holt(fpp.load("livestock"), damped=True), 10)


def test_holt_winters_at_least_as_good_as_R():
    fa = c7.hw(aust, seasonal="additive"); fm = c7.hw(aust, seasonal="multiplicative")
    valid(fa, 8); valid(fm, 8)
    assert fa.extra["fit"].method == "ETS(A,A,A)" and fm.extra["fit"].method == "ETS(M,A,M)"
    # Python's multi-start search reaches an MSE no worse than R's (within 1%)
    assert fa.extra["fit"].model["mse"] <= ref["hwa_mse"] * 1.01
    assert fm.extra["fit"].model["mse"] <= ref["hwm_mse"] * 1.01
    np.testing.assert_allclose(fa.mean, ref["hwa_mean"], rtol=0.06)
    np.testing.assert_allclose(fm.mean, ref["hwm_mean"], rtol=0.06)


def test_damped_hw_daily():
    hy = fpp.load("hyndsight")
    tr = hy.iloc[:-35]; tr.attrs = dict(hy.attrs)
    valid(c7.hw(tr, damped=True, seasonal="multiplicative", h=35), 35)


def test_all_15_ets_models_and_selection():
    aicc = {}
    for e in "AM":
        for tr in ["N", "A", "Ad"]:
            for s in "NAM":
                if e == "A" and s == "M":
                    continue
                f = c7.ets(aust, model=e + tr[0] + s, damped=(tr == "Ad"), polish=False)
                valid(f.forecast(8), 8)
                aicc[f.method] = f.aicc
    assert len(aicc) == 15
    auto = c7.ets(aust)
    assert auto.method == min(aicc, key=aicc.get) == ref["aust_method"]
    np.testing.assert_allclose(auto.forecast(8).mean, ref["aust_mean"], rtol=0.05)


def test_simulated_intervals_and_ann_variance_formula():
    fit = c7.ets(aust)
    paths = c7.simulate(fit, nsim=8, npaths=1000, seed=1)
    boot = c7.simulate(fit, nsim=8, npaths=1000, seed=1, bootstrap=True)
    assert paths.shape == boot.shape == (1000, 8) and np.all(np.isfinite(paths))
    ann = c7.ets(fpp.load("oil"), "ANN")
    fc = ann.forecast(5, level=(95,))
    np.testing.assert_allclose(fc.mean, ref["ann_mean"], rtol=1e-3)
    assert ann.sigma2 == pytest.approx(ref["ann_sigma2"], rel=1e-2)
    lT = np.asarray(ann.model["states"])[-1, 0]; a = ann.par["alpha"]; h = np.arange(1, 6)
    np.testing.assert_allclose(fc.upper[95], lT + stats.norm.ppf(0.975) * np.sqrt(ann.sigma2 * (1 + a ** 2 * (h - 1))),
                               rtol=1e-8)


def test_ets_mam_equals_hw_multiplicative_recursion():
    fit = c7.ets(aust, "MAM")
    x = np.asarray(fit.model["states"])[-1]
    l, b, s = x[0], x[1], x[2:6][::-1]
    hand = [(l + h * b) * s[(h - 1) % 4] for h in range(1, 9)]
    np.testing.assert_allclose(fit.forecast(8).mean, hand, rtol=1e-10)


def test_tscv_ses_matches_R():
    e = fpp.tsCV(fpp.load("livestock"), c7.ses, h=1).to_numpy()
    # Python cannot fit SES on the first few (1-7 point) windows that R uses, so compare loosely
    assert np.nanmean(e ** 2) == pytest.approx(ref["livestock_cv_ses_mse"], rel=0.15)
