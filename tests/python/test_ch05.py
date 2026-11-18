"""Chapter 5 — every regression model in the notes and exercises, in Python, checked against R."""
import itertools
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

import fpp
import fpp_ch05 as c5
from conftest import REF

ref = json.loads((REF / "ch05_reference.json").read_text())
us = fpp.load("uschange")
y_cons = us["Consumption"]; y_cons.attrs = dict(us.attrs)


def same_fc(fc, r, rtol=1e-7):
    np.testing.assert_allclose(fc.mean.to_numpy(), r["mean"], rtol=rtol)
    for lv, lo, hi in [(80, "lo80", "hi80"), (95, "lo95", "hi95")]:
        np.testing.assert_allclose(fc.lower[lv].to_numpy(), r[lo], rtol=rtol)
        np.testing.assert_allclose(fc.upper[lv].to_numpy(), r[hi], rtol=rtol)
    assert np.all(fc.lower[95].to_numpy() <= fc.lower[80].to_numpy() + 1e-9)


def test_consumption_regression_cv_bg():
    fit = c5.tslm(y_cons, X=us[["Income", "Production", "Unemployment", "Savings"]])
    np.testing.assert_allclose(fit.params, ref["consMR_coef"], rtol=1e-8)
    np.testing.assert_allclose(c5.CV(fit).to_numpy(), ref["consMR_CV"], rtol=1e-8)
    assert c5.bgtest(fit, 8)["LM test"] == pytest.approx(ref["consMR_bg8"], rel=1e-6)


def test_all_16_subsets_aicc():
    preds = ["Income", "Production", "Savings", "Unemployment"]
    got = []
    for g in itertools.product([0, 1], repeat=4):
        g = g[::-1]                      # R's expand.grid varies the first column fastest
        rhs = " + ".join(p for p, u in zip(preds, g) if u) or "1"
        got.append(c5.CV(smf.ols(f"Consumption ~ {rhs}", data=us).fit())["AICc"])
    np.testing.assert_allclose(got, ref["subsets_AICc"], rtol=1e-8)


def test_beer_trend_season_and_fourier():
    beer2 = fpp.window(fpp.load("ausbeer"), start=1992)
    fit = c5.tslm(beer2, trend_=True, season=True)
    np.testing.assert_allclose(fit.params, ref["beer_coef"], rtol=1e-8)
    same_fc(c5.forecast_lm(fit), ref["beer_fc"])
    np.testing.assert_allclose(c5.tslm(beer2, trend_=True, K=2).params, ref["beer_fourier_coef"], rtol=1e-8)


def test_easter_matches_R():
    np.testing.assert_allclose(c5.easter(fpp.load("ausbeer")), ref["easter_ausbeer"], atol=1e-12)
    np.testing.assert_allclose(c5.easter(fpp.load("a10")), ref["easter_a10"], atol=1e-12)


def test_scenarios_and_simple_regression_intervals():
    fit = c5.tslm(y_cons, X=us[["Income", "Savings", "Unemployment"]])
    same_fc(c5.forecast_lm(fit, newdata=pd.DataFrame({"Income": [1.0] * 4, "Savings": [0.5] * 4,
                                                       "Unemployment": [0.0] * 4})), ref["scen_up"])
    fit1 = c5.tslm(y_cons, X=us[["Income"]])
    same_fc(c5.forecast_lm(fit1, newdata=pd.DataFrame({"Income": [us["Income"].mean()] * 4})), ref["cons_ave"])


def test_marathon_trend_models():
    m = fpp.load("marathon"); h = 10
    tm = fpp.ts_time(m)
    same_fc(c5.forecast_lm(c5.tslm(m, trend_=True), h=h), ref["mar_lin"])
    same_fc(c5.forecast_lm(c5.tslm(m, trend_=True, lambda_=0), h=h), ref["mar_exp"])
    X = pd.DataFrame({"t": tm, "tb1": np.maximum(0, tm - 1940), "tb2": np.maximum(0, tm - 1980)})
    nd = pd.DataFrame({"t": tm[-1] + np.arange(1, h + 1), "tb1": X.tb1.iloc[-1] + np.arange(1, h + 1),
                       "tb2": X.tb2.iloc[-1] + np.arange(1, h + 1)})
    same_fc(c5.forecast_lm(c5.tslm(m, X=X), newdata=nd), ref["mar_pw"])
    cub = lambda d: pd.DataFrame({"t": d.t, "t2": d.t ** 2, "t3": d.t ** 3, "b1": d.tb1 ** 3, "b2": d.tb2 ** 3})
    same_fc(c5.forecast_lm(c5.tslm(m, X=cub(X)), newdata=cub(nd)), ref["mar_spline"], rtol=1e-3)   # t^3 for years ~2000: ill-conditioned


def test_smoothing_spline():
    m = fpp.load("marathon")
    same_fc(c5.splinef(m, lambda_=0, method="mle", h=10), ref["mar_splinef_mle"], rtol=1e-4)
    g = c5.splinef(m, lambda_=0, method="gcv", h=10)          # GCV choice differs from smooth.spline
    np.testing.assert_allclose(g.mean, ref["mar_splinef_gcv"]["mean"], rtol=0.01)


def test_interventions_spurious_and_lags():
    sb = fpp.load("Seatbelts")
    d = sb["drivers"]; d.attrs = dict(sb.attrs)
    np.testing.assert_allclose(c5.tslm(d, trend_=True, season=True, X=sb[["law"]]).params, ref["seatbelt_coef"], rtol=1e-8)
    aus = fpp.window(fpp.load("ausair"), end=2011)
    sp = c5.tslm(aus, X=pd.DataFrame({"rice": fpp.load("guinearice").to_numpy()}))
    np.testing.assert_allclose(sp.params, ref["spurious_coef"], rtol=1e-8)
    assert c5.bgtest(sp, 8)["LM test"] == pytest.approx(ref["spurious_bg"], rel=1e-6)
    ins = fpp.load("insurance")
    dl = pd.DataFrame({"Q": ins["Quotes"], "a0": ins["TV.advert"], "a1": ins["TV.advert"].shift(1),
                       "a2": ins["TV.advert"].shift(2)})
    np.testing.assert_allclose(smf.ols("Q ~ a0 + a1 + a2", dl).fit().params, ref["lags_coef"], rtol=1e-8)


def test_exercise_models():
    elec = fpp.load("elecdaily")
    d20 = elec.iloc[:20]; y20 = d20["Demand"]; y20.attrs = dict(elec.attrs)
    same_fc(c5.forecast_lm(c5.tslm(y20, X=d20[["Temperature"]]),
                           newdata=pd.DataFrame({"Temperature": [15.0, 35.0]})), ref["ex1"])
    same_fc(c5.forecast_lm(c5.tslm(fpp.load("mens400"), trend_=True), h=1), ref["ex2"])
    fancy = fpp.load("fancy")
    surf = ((fancy.index.month == 3) & (fancy.index.year >= 1988)).astype(float)
    lf = np.log(fancy); lf.attrs = dict(fancy.attrs)
    ff = c5.tslm(lf, trend_=True, season=True, X=pd.DataFrame({"surf": surf}))
    np.testing.assert_allclose(ff.params, ref["ex5_coef"], rtol=1e-8)
    same_fc(c5.forecast_lm(ff, newdata=pd.DataFrame({"surf": ([0, 0, 1] + [0] * 9) * 3})), ref["ex5_fc"])
    assert c5.bgtest(ff, 24)["LM test"] == pytest.approx(ref["ex5_bg24"], rel=1e-6)
    gas = fpp.window(fpp.load("gasoline"), end=2005)
    aicc = [c5.CV(c5.tslm(gas, trend_=True, K=K))["AICc"] for K in range(1, 27)]
    np.testing.assert_allclose(aicc, ref["ex6_AICc"], rtol=1e-8)
    assert int(np.argmin(aicc)) + 1 == 12
    same_fc(c5.forecast_lm(c5.tslm(gas, trend_=True, K=12), h=52), ref["ex6_fc"])
    hu = fpp.load("huron"); th = fpp.ts_time(hu)
    same_fc(c5.forecast_lm(c5.tslm(hu, trend_=True), h=8), ref["ex7_lin"])
    same_fc(c5.forecast_lm(c5.tslm(hu, X=pd.DataFrame({"th": th, "t15": np.maximum(0, th - 1915)})),
                           newdata=pd.DataFrame({"th": np.arange(1973, 1981.0), "t15": np.arange(1973, 1981.0) - 1915})),
            ref["ex7_pw"])


def test_matrix_formula_ex8():
    y = fpp.load("huron").to_numpy(); T = len(y); tt = np.arange(1, T + 1)
    X = np.column_stack([np.ones(T), tt])
    np.testing.assert_allclose(np.linalg.inv(X.T @ X),
                               2 / (T * (T - 1)) * np.array([[2 * T + 1, -3], [-3, 6 / (T + 1)]]), rtol=1e-10)
