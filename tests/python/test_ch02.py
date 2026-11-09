"""Chapter 2 — time series graphics + the white-noise model, in Python, checked against R."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

import fpp
from conftest import REF, ROOT

ref = json.loads((REF / "ch02_reference.json").read_text())
beer2 = fpp.window(fpp.load("ausbeer"), start=1992)
aelec = fpp.window(fpp.load("elec"), start=1980)


def lb(x, lag):
    r = acorr_ljungbox(pd.Series(np.asarray(x, dtype=float)).dropna().to_numpy(), lags=[lag])
    return float(r["lb_stat"].iloc[0]), float(r["lb_pvalue"].iloc[0])


def test_acf_values_match_R():
    np.testing.assert_allclose(acf(beer2.to_numpy(), nlags=9, fft=False)[1:], ref["acf_beer2"], rtol=1e-10)
    np.testing.assert_allclose(acf(aelec.to_numpy(), nlags=48, fft=False)[1:], ref["acf_aelec"], rtol=1e-10)


def test_correlations_match_R():
    ed = fpp.load("elecdemand")
    assert np.corrcoef(ed["Temperature"], ed["Demand"])[0, 1] == pytest.approx(ref["cor_elecdemand"], rel=1e-10)
    c = fpp.load("visnights").iloc[:, :5].corr().to_numpy()
    upper = c.T[np.triu_indices(5, 1)]          # R's column-major upper triangle
    np.testing.assert_allclose(np.sort(upper), np.sort(ref["visnights_cor"]), rtol=1e-10)


def test_frequencies_and_gold_outlier():
    for name, f in ref["freq"].items():
        assert fpp.frequency(fpp.load(name)) == pytest.approx(f)
    gold = fpp.load("gold")
    assert int(np.nanargmax(gold.to_numpy())) + 1 == ref["gold_which_max"]
    assert np.nanmax(gold.to_numpy()) == pytest.approx(ref["gold_max"])


def test_tute1_as_quarterly_series():
    tute1 = pd.read_csv(ROOT / "data" / "external" / "tute1.csv", index_col=0)
    np.testing.assert_allclose(tute1.sum().to_numpy(), ref["tute1_colsums"])
    idx = pd.period_range("1981Q1", periods=len(tute1), freq="Q")
    assert str(idx[-1]) == "2005Q4"


@pytest.mark.parametrize("key,series,lag", [
    ("lb_beer2", lambda: beer2, 8),
    ("lb_aelec", lambda: aelec, 24),
    ("lb_a10", lambda: fpp.load("a10"), 24),
    ("lb_dgoog", lambda: np.diff(fpp.load("goog").to_numpy()), 10),
    ("lb_ustreas", lambda: fpp.load("ustreas"), 10),
    ("lb_mypigs", lambda: fpp.window(fpp.load("pigs"), start=1990), 24),
    ("lb_ddj", lambda: np.diff(fpp.load("dj").to_numpy()), 10),
])
def test_white_noise_tests_match_R(key, series, lag):
    q, p = lb(series(), lag)
    assert q == pytest.approx(ref[key]["Q"], rel=1e-8)
    assert p == pytest.approx(ref[key]["p"], rel=1e-6, abs=1e-12)


def test_white_noise_model_decisions():
    wn = np.random.default_rng(30).normal(size=50)
    assert lb(wn, 10)[1] > 0.05
    assert lb(np.diff(fpp.load("goog").to_numpy()), 10)[1] > 0.05
    assert lb(beer2, 8)[1] < 0.001


def test_all_python_graphics_run():
    a10 = fpp.load("a10")
    fpp.autoplot(a10); fpp.seasonplot(a10); fpp.subseriesplot(a10)
    fpp.lagplot(beer2, lags=9); fpp.acf_plot(aelec, lag_max=48)
    fpp.autoplot(fpp.load("visnights").iloc[:, :5], facets=True)
    plt.close("all")
