"""Kalite-islemi: IC istatistik testleri (Newey-West + bagimsiz ornekleme).

Sorun (beyanli): label 5 gunluk ileri getiri; ardisik gunlerin IC'leri cakisir,
klasik t-stat siskin olabilir. Bu dosya NW duzeltmesinin durumu dogru
yonde raporladigini test eder (t_nw < ham t) ve stride ozetinin sayacini.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from training.evaluate import daily_ic, ic_summary, newey_west_t, stride_ic_summary


def _ic_series(rng: np.random.Generator, mean: float, phi: float, n: int = 400) -> pd.Series:
    """AR(1) IC serisi: x_t = mean + phi * (x_{t-1} - mean) + eps.

    phi > 0: pozitif otokorelasyon (horizon cakismasini taklif eder).
    """
    eps = rng.normal(0, 0.05, size=n)
    values = np.empty(n)
    anchor = mean
    for i in range(n):
        anchor = mean + phi * (anchor - mean) + eps[i]
        values[i] = anchor
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=n))


def test_nw_t_pozitif_otokorelasyonda_ham_tden_kucuk():
    """Pozitif otokorelasyonda NW standart hatasi buyudugunden |t_nw| ham t'den kucuk kalir."""

    rng = np.random.default_rng(11)
    positive = _ic_series(rng, mean=0.01, phi=0.6)
    negative = _ic_series(rng, mean=-0.01, phi=0.6)
    for series in (positive, negative):
        raw = float(series.mean() / series.std(ddof=1) * np.sqrt(len(series)))
        nw = newey_west_t(series, lag=4)
        assert abs(nw) < abs(raw)  # duzeltme MUHAFAZAKAR yonde olmali (pozitif otok)
        assert not math.isnan(nw)


def test_nw_t_beyaz_gurultude_ham_tye_yakin():
    """iid seride otokovaryanslar ~0: t_nw ham t'ye yaklasmali (ayni sinif)."""
    rng = np.random.default_rng(12)
    white = pd.Series(rng.normal(0.01, 0.05, size=500))
    raw = float(white.mean() / white.std(ddof=1) * np.sqrt(len(white)))
    nw = newey_west_t(white, lag=4)
    assert math.isclose(nw, raw, rel_tol=0.25)  # ayni buyukluk; samata yok


def test_stride_sayisi_dogru():
    """Son gunden geriye her 5 gunde bir orneklem: n=120 -> 24 adet."""
    rng = np.random.default_rng(13)
    series = pd.Series(rng.normal(size=120), index=pd.bdate_range("2024-01-01", periods=120))
    summary = stride_ic_summary(series, stride=5)
    assert summary["stride_n"] == 24
    assert math.isfinite(summary["stride_mean_ic"])
    # n stride sayisina tam bolunmuyorsa tavan alinir: n=117 -> 24 (117/5 = 23.4)
    summary117 = stride_ic_summary(pd.Series(rng.normal(size=117)), stride=5)
    assert summary117["stride_n"] == 24


def test_stride_ufuk_cevrimici_bagimsiz():
    """stride=5: secilen indeksler arasindaki fark 5; son gunden geriye."""
    values = np.arange(10.0, 110.0)  # 100 adet
    series = pd.Series(values)
    summary = stride_ic_summary(series, stride=5)
    assert summary["stride_n"] == 20
    # i~ (~(10..110)) son gunden geri: ilk orneklem son gun (109.0), sonra 104
    expected_subset = values[::-5]
    assert summary["stride_mean_ic"] == pytest.approx(float(expected_subset.mean()))


def test_ic_summary_t_nw_anahtarini_tasir():
    rng = np.random.default_rng(14)
    series = pd.Series(rng.normal(0.005, 0.05, size=80))
    summary = ic_summary(series, nw_lag=4)
    assert "t_nw" in summary
    assert summary["t_nw"] == pytest.approx(newey_west_t(series, lag=4))


def test_daily_ic_ve_summary_uzun_seride_t_nw_gerceklesir():
    """Sentetik skor-label panelinde gunluk IC uzerinden tam akis (entegrasyon)."""
    rng = np.random.default_rng(15)
    dates = pd.bdate_range("2024-01-01", periods=120)
    rows = []
    for date in dates:
        for ticker in ("AAA", "BBB", "CCC", "DDD", "EEE"):
            score = float(rng.normal())
            rows.append({"date": date, "ticker": ticker, "score": score, "label": score + 0.1 * float(rng.normal())})
    frame = pd.DataFrame(rows)
    ic = daily_ic(frame, "score", "label", min_n=2)
    summary = ic_summary(ic)
    assert summary["n_days"] == 120
    assert math.isfinite(summary["t_nw"])


def test_newey_west_t_kisa_seriler_nan():
    assert math.isnan(newey_west_t(pd.Series([0.5, 0.6]), lag=4))  # n<3
    assert math.isnan(newey_west_t(pd.Series([0.5, 0.5, 0.5, 0.5]), lag=2))  # degiskenlik yok
