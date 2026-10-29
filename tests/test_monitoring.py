"""Adim 7 (FAZ 5): monitoring testleri - sentetik panel uzerinden."""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import ks_statistic, psi
from monitoring.performance import coverage, latest_rolling_ic, overall_ic
from monitoring.run import feature_drift_report, performance_report, run_monitoring


def _synthetic_features(n_days: int = 120, drift_from_day: int | None = None) -> pd.DataFrame:
    """(date, ticker, f0, f1) panel; drift_from_day gununden itibaren f1 kayar."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    tickers = ("AAA", "BBB", "CCC")
    rows = []
    for position, date in enumerate(dates):
        for ticker in tickers:
            f0 = float(rng.normal())
            f1 = float(rng.normal()) if (drift_from_day is None or position < drift_from_day) else float(
                rng.normal() + 5.0
            )
            rows.append({"date": date, "ticker": ticker, "f0": f0, "f1": f1})
    return pd.DataFrame(rows)


def test_psi_ayni_dagilim_neredeyse_sifir():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=5000)
    current = rng.normal(size=5000)  # ayni dagilim
    value = psi(reference, current)
    assert 0.0 <= value < 0.1


def test_psi_kaymis_dagilim_buyuk():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=5000)
    current = rng.normal(size=5000) + 5.0  # buyuk kayma
    value = psi(reference, current)
    assert value > 1.0
    assert value < 20.0  # eps korumasi ile makul ust sinir


def test_psi_kisa_orneklem_nan():
    assert math.isnan(psi([1.0] * 5, [2.0] * 5))  # 10'dan az orneklem


def test_ks_kayma_belli():
    rng = np.random.default_rng(3)
    stat, p_value = ks_statistic(rng.normal(size=800), rng.normal(size=800) + 1.0)
    assert stat > 0.2
    assert p_value < 0.001
    stat_same, _ = ks_statistic(rng.normal(size=800), rng.normal(size=800))
    assert stat_same < 0.2  # ayni dagilim: istatistik kucuk


def test_rolling_ic_bilgili_skor_pozitif():
    """label = score + kucuk gurultu: sira bozulmaz; IC ~ 1.0 beklenir."""
    rng = np.random.default_rng(4)
    rows = []
    dates = pd.bdate_range("2025-06-01", periods=40)
    for date in dates:
        for ticker in ("AAA", "BBB", "CCC", "DDD"):
            score = float(rng.normal())
            rows.append({"date": date, "ticker": ticker, "score": score, "label": score + 0.2 * float(rng.normal())})
    predictions = pd.DataFrame(rows)
    value = latest_rolling_ic(predictions, window=20)
    assert not math.isnan(value)
    assert value > 0.5
    summary = overall_ic(predictions)
    assert summary["n_days"] == 40
    assert summary["positive_share"] == pytest.approx(1.0, abs=0.05)


def test_rolling_ic_yetersiz_veri_nan():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2025-06-01", periods=10)
    predictions = pd.DataFrame([
        {"date": date, "ticker": "AAA", "score": float(rng.normal()), "label": float(rng.normal())}
        for date in dates
    ])
    assert math.isnan(latest_rolling_ic(predictions, window=20))


def test_feature_drift_kaymayi_yakalar():
    features = _synthetic_features(n_days=120, drift_from_day=60)
    boundary = features["date"].iloc[180]  # 60. gunun satiri (gun basi 3 satir)
    report = feature_drift_report(features, boundary, ["f0", "f1"], psi_bins=10, psi_alert=0.2)
    by_feature = {row["feature"]: row for row in report}
    assert by_feature["f0"]["alert"] is False
    assert by_feature["f1"]["alert"] is True
    assert by_feature["f1"]["psi"] > 0.2


def test_performance_report_anahtarlar():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2025-06-01", periods=60)
    predictions = pd.DataFrame([
        {"date": date, "ticker": ticker, "score": float(rng.normal()), "label": float(rng.normal())}
        for date in dates for ticker in ("AAA", "BBB")
    ])
    summary = performance_report(predictions, window=20, alert_ic=0.0)
    assert "overall" in summary and "rolling_ic_20d" in summary
    assert summary["coverage_n_days_covered"] == 60
    assert isinstance(summary["alert"], bool)


def test_run_monitoring_rapor_yazar(tmp_path: pathlib.Path):
    features = _synthetic_features(n_days=120, drift_from_day=60)
    rng = np.random.default_rng(8)
    last_dates = sorted(features["date"].unique())[-20:]  # tahmin donemi: son 20 gun (hepsi drifli)
    prediction_rows = []
    for date in last_dates:
        for ticker in ("AAA", "BBB", "CCC"):
            score = float(rng.normal())
            prediction_rows.append({"date": date, "ticker": ticker, "score": score, "label": score})
    predictions = pd.DataFrame(prediction_rows)
    features_path = tmp_path / "features.parquet"
    predictions_path = tmp_path / "holdout_predictions.parquet"
    features.to_parquet(features_path, index=False)
    predictions.to_parquet(predictions_path, index=False)

    config_path = tmp_path / "monitoring.yaml"
    config_path.write_text(
        "\n".join([
            f"features_path: {features_path.as_posix()}",
            f"predictions_path: {predictions_path.as_posix()}",
            f"report_path: {(tmp_path / 'report.json').as_posix()}",
            "roll_ic_window: 5",
            "psi_bins: 10",
            "psi_alert: 0.2",
            "alert_ic: 0.0",
        ]) + "\n",
        encoding="utf-8",
    )
    report = run_monitoring(config_path)
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved == report  # dosya ile donen rapor ayni icerik
    assert {row["feature"]: row["alert"] for row in report["feature_drift"]}["f1"] is True
    assert report["performance"]["rolling_ic_5d"] is not None  # label = score: IC = 1.0
    assert report["performance"]["coverage_n_days_covered"] == 20


def test_coverage_alanlari():
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2025-06-01", periods=12)  # 2025-06-01 pazar; ilk is gunu 06-02
    predictions = pd.DataFrame([
        {"date": date, "ticker": ticker, "score": float(rng.normal()), "label": float(rng.normal())}
        for date in dates for ticker in ("AAA", "BBB")
    ])
    cov = coverage(predictions)
    assert cov["n_days_covered"] == 12
    assert cov["avg_tickers_per_day"] == pytest.approx(2.0)
    assert cov["first_date"] == "2025-06-02"
    assert cov["last_date"] == str(dates[-1].date())
