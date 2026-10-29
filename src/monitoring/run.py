"""Adim 7 (FAZ 5): monitoring raporlari - drift + performans + uyari listesi.

Akis:
1) features.parquet yukler; predictions parquet'inin baslangic gununu
   sinir alarak iki pencere kurar:
     - reference : tahmin doneminden ONCEKI dev (dagilim referansi)
     - current   : tahmin donemi (canli pencere)
2) Her feature icin PSI ve KS raporu; psi >= psi_alert ise uyari.
3) Tahmin akisi (skor/label) uzerinden kayan IC ve genel IC; kayan IC
   alert_ic altindaysa uyari.
4) reports/monitoring_report.json yazar ve ozeti loglar.

Kullanim: .venv\\Scripts\\python -m monitoring.run
"""

from __future__ import annotations

import json
import logging
import math
import pathlib

import pandas as pd
import yaml

from monitoring.drift import ks_statistic, psi
from monitoring.performance import (
    DEFAULT_NW_LAG,
    coverage,
    latest_rolling_ic,
    overall_ic,
)

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = DEFAULT_PROJECT_ROOT / "config" / "monitoring.yaml"

# DAGILIM olcumune dahil OLMAYAN sabit kolonlar (meta/etiket)
NON_FEATURE_COLS = {"date", "ticker", "forward_return", "label"}


def load_monitoring_config(path: str | pathlib.Path = DEFAULT_CONFIG_PATH) -> dict:
    """config yolu: verilmezse proje koku varsayilani; env secimli."""
    with pathlib.Path(path).open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    if path.is_absolute():
        return path
    return DEFAULT_PROJECT_ROOT / path


def feature_drift_report(
    features: pd.DataFrame,
    boundary: pd.Timestamp,
    feature_cols: list[str],
    psi_bins: int,
    psi_alert: float,
) -> list[dict]:
    """Referans (< boundary) vs guncel (>= boundary) pencerede feature drift'i."""
    reference = features[features["date"] < boundary]
    current = features[features["date"] >= boundary]
    rows: list[dict] = []
    for column in feature_cols:
        psi_value = psi(reference[column].to_numpy(), current[column].to_numpy(), bins=psi_bins)
        ks_stat, ks_p = ks_statistic(reference[column].to_numpy(), current[column].to_numpy())
        alert = bool(not math.isnan(psi_value) and psi_value >= psi_alert)
        rows.append({
            "feature": column,
            "psi": round(psi_value, 4) if not math.isnan(psi_value) else None,
            "ks_stat": round(ks_stat, 4) if not math.isnan(ks_stat) else None,
            "ks_p": round(ks_p, 6) if not math.isnan(ks_p) else None,
            "alert": alert,
        })
    return rows


def performance_report(
    predictions: pd.DataFrame, window: int, alert_ic: float, nw_lag: int = DEFAULT_NW_LAG
) -> dict:
    """Tahmin akisinin IC karnesi (ham t + Newey-West t) + kayan pencere uyari kontrolu."""
    summary = overall_ic(predictions, nw_lag=nw_lag)
    rolling = latest_rolling_ic(predictions, window=window)
    return {
        "overall": {
            key: (round(float(value), 4) if not math.isnan(float(value)) else None)
            for key, value in summary.items()
        },
        f"rolling_ic_{window}d": (round(rolling, 4) if not math.isnan(rolling) else None),
        "alert": bool(not math.isnan(rolling) and rolling < alert_ic),
        **{f"coverage_{key}": value for key, value in coverage(predictions).items()},
    }


def run_monitoring(config_path: str | pathlib.Path = DEFAULT_CONFIG_PATH) -> dict:
    """Config'e gore raporu uretir; JSON yazar ve ozeti loglar."""
    cfg = load_monitoring_config(config_path)
    features_path = _resolve(str(cfg.get("features_path", "data/processed/features.parquet")))
    predictions_path = _resolve(str(cfg.get("predictions_path", "data/processed/holdout_predictions.parquet")))
    report_path = _resolve(str(cfg.get("report_path", "reports/monitoring_report.json")))
    window = int(cfg.get("roll_ic_window", 20))
    psi_bins = int(cfg.get("psi_bins", 10))
    psi_alert = float(cfg.get("psi_alert", 0.2))
    alert_ic = float(cfg.get("alert_ic", 0.0))
    nw_lag = int(cfg.get("nw_lag", DEFAULT_NW_LAG))  # IC otokorelasyonu (horizon-1)

    features = pd.read_parquet(features_path)
    features["date"] = pd.to_datetime(features["date"])
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"])

    boundary = pd.Timestamp(predictions["date"].min())
    feature_cols = [column for column in features.columns if column not in NON_FEATURE_COLS]
    drift_rows = feature_drift_report(features, boundary, feature_cols, psi_bins, psi_alert)
    performance = performance_report(predictions, window, alert_ic, nw_lag=nw_lag)

    alerts = [row["feature"] for row in drift_rows if row["alert"]]
    if performance["alert"]:
        alerts.append(f"rolling_ic_{window}d<{alert_ic}")

    report = {
        "reference_end": str((boundary - pd.DateOffset(days=1)).date()),
        "current_window": [str(boundary.date()), str(features["date"].max().date())],
        "psi_alert_threshold": psi_alert,
        "feature_drift": drift_rows,
        "performance": performance,
        "alerts": alerts,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Monitoring raporu yazildi: %s | uyari: %s | kayan_ic: %s",
        report_path,
        alerts if alerts else "yok",
        performance[f"rolling_ic_{window}d"],
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    run_monitoring()
