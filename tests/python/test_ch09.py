"""Chapter 9 — every dynamic regression model in Python, checked against R reference values."""
import json
import warnings

import numpy as np
import pytest
import statsmodels.formula.api as smf
from statsforecast.models import ARIMA, AutoARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import fpp
import fpp_ch09 as c9
from conftest import REF

warnings.filterwarnings("ignore")
ref = json.loads((REF / "ch09_reference.json").read_text())
us = fpp.load("uschange")
austa = fpp.load("austa")
cafe04 = fpp.window(fpp.load("auscafe"), start=2004)
ins = fpp.load("insurance")


def valid_fc(fc, h):
    m = fc.mean.to_numpy()
    assert m.shape == (h,) and np.all(np.isfinite(m))
    assert np.all(fc.lower[95].to_numpy() <= fc.lower[80].to_numpy() + 1e-9)
    assert np.all(fc.lower[80].to_numpy() <= m + 1e-9) and np.all(m <= fc.upper[80].to_numpy() + 1e-9)
    assert np.all(fc.upper[80].to_numpy() <= fc.upper[95].to_numpy() + 1e-9)


# ---- Fourier terms --------------------------------------------------------
def test_fourier_matches_R():
    np.testing.assert_allclose(c9.fourier(len(cafe04), 12, 2).to_numpy()[:24], ref["fourier_K2"], atol=1e-10)
    np.testing.assert_allclose(c9.fourier(len(cafe04), 12, 6, h=24).to_numpy(), ref["fourier_K6_h"], atol=1e-10)
    taylor = fpp.load("taylor")
    np.testing.assert_allclose(c9.fourier(len(taylor), [48, 336], [3, 5]).to_numpy()[:10], ref["fourier_msts"], atol=1e-10)
    assert c9.fourier(10, 12, 6).shape[1] == 11        # S6-12 is identically zero and dropped, as in R


# ---- 9.1-9.3 regression with ARIMA errors --------------------------------
def test_uschange_regression_with_arima_errors():
    # Python's automatic search must run and give a valid model (its orders may differ from R's
    # on another computer) ...
    auto = AutoARIMA().fit(us["Consumption"].to_numpy(), X=us[["Income"]].to_numpy(dtype=float))
    valid_fc(c9.sf_forecast(auto, us["Consumption"], 8, X_future=np.full((8, 1), us["Income"].mean())), 8)
    # ... and R's chosen model, fitted in Python, must match R's numbers.
    p_, d_, q_ = ref["us_order"]
    fit = ARIMA(order=(p_, d_, q_), include_mean=True).fit(us["Consumption"].to_numpy(),
                                                            X=us[["Income"]].to_numpy(dtype=float))
    np.testing.assert_allclose(c9.coefs(fit).to_numpy(), ref["us_coef"], atol=2e-3)
    fc = c9.sf_forecast(fit, us["Consumption"], 8, X_future=np.full((8, 1), us["Income"].mean()))
    valid_fc(fc, 8)
    np.testing.assert_allclose(fc.mean.to_numpy(), ref["us_fc"], atol=2e-3)
    eta = c9.reg_errors(fit, us["Consumption"], us[["Income"]])
    np.testing.assert_allclose(eta, ref["us_reg_errors"], atol=2e-3)
    lb = fpp.ljung_box(fit.model_["residuals"], df=c9.arma_df(fit), m=4)
    assert lb["Q*"] == pytest.approx(ref["us_lb"][0], rel=1e-3)


def test_levels_vs_differences_equivalence():
    y = us["Consumption"].to_numpy(); x = us[["Income"]].to_numpy()
    a = SARIMAX(y, exog=x, order=(1, 1, 0)).fit(disp=False)
    b = SARIMAX(np.diff(y), exog=np.diff(x, axis=0), order=(1, 0, 0)).fit(disp=False)
    np.testing.assert_allclose(a.params[:2], b.params[:2], rtol=1e-4)
    np.testing.assert_allclose(a.params[[1, 0]], ref["diff_coef"], rtol=1e-3)


def test_ols_ignoring_autocorrelation_has_wrong_inference():
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    ols = smf.ols("Consumption ~ Income", data=us).fit()
    assert acorr_breusch_godfrey(ols, nlags=8, result_object=False)[1] < 0.01


def test_elecdaily_dynamic_regression():
    e = fpp.load("elecdaily")
    X = np.column_stack([e["Temperature"], e["Temperature"] ** 2, e["WorkDay"]])
    fit = ARIMA(order=(2, 1, 2), seasonal_order=(2, 0, 0), season_length=7, include_drift=True).fit(e["Demand"].to_numpy(), X=X)
    c = c9.coefs(fit)
    assert c["ex_1"] < 0 < c["ex_2"] and c["ex_3"] > 20       # U-shape in temperature, workday effect
    Xf = np.column_stack([np.full(14, 26.0), np.full(14, 676.0), [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]])
    valid_fc(c9.sf_forecast(fit, e["Demand"], 14, X_future=Xf), 14)
    auto = AutoARIMA(season_length=7).fit(e["Demand"].to_numpy(), X=X)
    assert np.isfinite(auto.model_["aicc"])


# ---- 9.4 trends -----------------------------------------------------------
def test_deterministic_and_stochastic_trends():
    tr = np.arange(1, len(austa) + 1, dtype=float)[:, None]
    f1 = ARIMA(order=(2, 0, 0), include_mean=True).fit(austa.to_numpy(), X=tr)
    np.testing.assert_allclose(c9.coefs(f1).to_numpy(), ref["austa_det_coef"], atol=2e-3)
    fc1 = c9.sf_forecast(f1, austa, 10, X_future=len(austa) + np.arange(1, 11, dtype=float)[:, None])
    np.testing.assert_allclose(fc1.mean.to_numpy(), ref["austa_det_fc"], rtol=1e-3)
    f2 = AutoARIMA(d=1).fit(austa.to_numpy())
    np.testing.assert_allclose(c9.coefs(f2).to_numpy(), ref["austa_sto_coef"], atol=1e-2)
    fc2 = c9.sf_forecast(f2, austa, 10)
    valid_fc(fc1, 10); valid_fc(fc2, 10)
    np.testing.assert_allclose(fc2.mean.to_numpy(), ref["austa_sto_fc"], rtol=5e-3)
    # the stochastic trend has wider long-horizon intervals (both languages)
    assert fc2.upper[95].iloc[-1] - fc2.mean.iloc[-1] > fc1.upper[95].iloc[-1] - fc1.mean.iloc[-1]
    assert ref["austa_sto_hi95"][-1] - ref["austa_sto_fc"][-1] > ref["austa_det_hi95"][-1] - ref["austa_det_fc"][-1]


# ---- 9.5 dynamic harmonic regression -------------------------------------
@pytest.mark.parametrize("K", range(1, 7))
def test_cafe_fourier_models(K):
    logy = np.log(cafe04.to_numpy())
    fit = AutoARIMA(seasonal=False).fit(logy, X=c9.fourier(len(cafe04), 12, K).to_numpy())
    assert fit.model_["aicc"] == pytest.approx(ref["cafe_aicc"][K - 1], abs=3.0)
    fc = c9.sf_forecast(fit, cafe04, 24, X_future=c9.fourier(len(cafe04), 12, K, h=24), lambda_=0)
    valid_fc(fc, 24)
    if K == 5:
        np.testing.assert_allclose(fc.mean.to_numpy(), ref["cafe5_fc"], rtol=2e-3)


def test_cafe_best_K_is_5_like_R():
    logy = np.log(cafe04.to_numpy())
    aicc = [AutoARIMA(seasonal=False).fit(logy, X=c9.fourier(len(cafe04), 12, K).to_numpy()).model_["aicc"] for K in range(1, 7)]
    assert int(np.argmin(aicc)) + 1 == int(np.argmin(ref["cafe_aicc"])) + 1 == 5


# ---- 9.6 lagged predictors -----------------------------------------------
def test_insurance_lagged_predictors():
    A = c9.lag_matrix(ins["TV.advert"], [0, 1, 2, 3])
    q = ins["Quotes"].to_numpy(dtype=float)
    aicc = [AutoARIMA(stationary=True).fit(q[3:40], X=A.iloc[3:40, :k].to_numpy(dtype=float)).model_["aicc"] for k in range(1, 5)]
    assert int(np.argmin(aicc)) == int(np.argmin(ref["ins_aicc"])) == 1      # two predictors win
    np.testing.assert_allclose(aicc[:2], ref["ins_aicc"][:2], atol=0.05)
    fit = ARIMA(order=(3, 0, 0), include_mean=True).fit(q[1:], X=A.iloc[1:, :2].to_numpy(dtype=float))
    np.testing.assert_allclose(c9.coefs(fit).to_numpy(), ref["ins_coef"], atol=5e-3)
    Xf = np.column_stack([np.full(20, 8.0), np.r_[A.iloc[39, 0], np.full(19, 8.0)]])
    y = ins["Quotes"].iloc[1:]; y.attrs = dict(ins.attrs)
    fc = c9.sf_forecast(fit, y, 20, X_future=Xf)
    valid_fc(fc, 20)
    np.testing.assert_allclose(fc.mean.to_numpy(), ref["ins_fc"], rtol=2e-3)


# ---- exercise models ------------------------------------------------------
def test_exercise_models():
    adv = fpp.load("advert")
    np.testing.assert_allclose(smf.ols("sales ~ advert", data=adv).fit().params.to_numpy(), ref["advert_lm"], rtol=1e-8)
    assert np.isfinite(AutoARIMA().fit(adv["sales"].to_numpy(dtype=float),
                                       X=adv[["advert"]].to_numpy(dtype=float)).model_["aicc"])
    a = ARIMA(order=(0, 1, 0)).fit(adv["sales"].to_numpy(dtype=float),     # R's auto.arima choice
                                   X=adv[["advert"]].to_numpy(dtype=float))
    np.testing.assert_allclose(c9.coefs(a).to_numpy(), ref["advert_auto_coef"], atol=2e-3)
    hur = fpp.load("huron"); t = fpp.ts_time(hur).astype(float)
    fh = ARIMA(order=(2, 0, 0), include_mean=True).fit(hur.to_numpy(), X=np.column_stack([t, np.maximum(t - 1920, 0)]))
    got, exp = c9.coefs(fh).to_numpy(), np.array(ref["huron_coef"])
    np.testing.assert_allclose(got[[0, 1, 3, 4]], exp[[0, 1, 3, 4]], atol=3e-3)   # AR terms and slopes
    assert got[2] == pytest.approx(exp[2], rel=0.03)                             # weakly identified intercept


def test_cv_lm_matches_R_on_gasoline_harmonic_regression():
    import statsmodels.api as sm
    gas = fpp.load("gasoline"); tg = fpp.ts_time(gas); n = len(gas)
    X = np.column_stack([np.arange(1, n + 1), np.maximum(tg - 2007, 0), np.maximum(tg - 2013, 0),
                         c9.fourier(n, fpp.frequency(gas), 12).to_numpy()])
    cv = c9.cv_lm(sm.OLS(gas.to_numpy(), sm.add_constant(X)).fit())
    np.testing.assert_allclose(cv.to_numpy(), ref["gas_cv"], rtol=1e-6)
