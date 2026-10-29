"""Adim 9 (FAZ 7): dashboard hazirlik katmani (Streamlit'siz test edilebilir)."""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import pandas as pd

DEFAULT_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

EQUITY_PATH = DEFAULT_PROJECT_ROOT / "reports" / "equity_curve.csv"
SUMMARY_PATH = DEFAULT_PROJECT_ROOT / "reports" / "backtest_summary.json"
MONITORING_PATH = DEFAULT_PROJECT_ROOT / "reports" / "monitoring_report.json"
LATEST_SCORES_PATH = DEFAULT_PROJECT_ROOT / "data" / "processed" / "latest_scores.parquet"
FREEZE_PATH = DEFAULT_PROJECT_ROOT / "reports" / "forward_test" / "freeze.json"
FORWARD_PATH = DEFAULT_PROJECT_ROOT / "reports" / "forward_test" / "forward_report.json"
MLFLOW_DB = DEFAULT_PROJECT_ROOT / "mlflow.db"
DEFAULT_EXPERIMENT = "finans-training"
DEFAULT_REGISTRY = "finans-ranker"


def load_scores() -> pd.DataFrame:
    """data/processed/latest_scores.parquet yoksa bos frame (kolonlar sabit)."""
    if not LATEST_SCORES_PATH.is_file():
        return pd.DataFrame(columns=["date", "ticker", "score"])
    frame = pd.read_parquet(LATEST_SCORES_PATH)
    return frame.sort_values("score", ascending=False, ignore_index=True)


def top_bottom(frame: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sira garantisi ile en yuksek/dusuk n skor; eksik veride (bos, bos)."""
    ordered = frame.sort_values("score", ascending=False, ignore_index=True)
    top = ordered.head(n)
    bottom = ordered.tail(n).sort_values("score", ignore_index=True)
    return top, bottom


def load_summary() -> dict[str, Any]:
    """reports/backtest_summary.json; yoksa bos sozluk (uyari icin)."""
    if not SUMMARY_PATH.is_file():
        return {}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def load_equity() -> pd.DataFrame:
    """reports/equity_curve.csv; ilk kolonu tarih'e cevirir; yoksa bos frame."""
    if not EQUITY_PATH.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(EQUITY_PATH, index_col=0)
    frame.index.name = "date"
    return frame.reset_index()


def load_monitoring() -> dict[str, Any]:
    """reports/monitoring_report.json; yoksa bos sozluk."""
    if not MONITORING_PATH.is_file():
        return {}
    return json.loads(MONITORING_PATH.read_text(encoding="utf-8"))


def load_freeze() -> dict[str, Any]:
    """reports/forward_test/freeze.json; yoksa bos sozluk."""
    if not FREEZE_PATH.is_file():
        return {}
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def load_forward() -> dict[str, Any]:
    """reports/forward_test/forward_report.json; yoksa bos sozluk."""
    if not FORWARD_PATH.is_file():
        return {}
    return json.loads(FORWARD_PATH.read_text(encoding="utf-8"))


def forward_quantile_frame(forward: dict[str, Any]) -> pd.DataFrame:
    """Ileri test raporunun quantiles listesi -> (quantile, mean_return, vs_benchmark)."""
    rows = forward.get("quantiles") or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["quantile", "mean_return", "vs_benchmark"])
    for column in ("mean_return", "vs_benchmark"):
        frame[column] = frame[column].apply(
            lambda value: float(value) if value is not None else math.nan
        )
    return frame[["quantile", "mean_return", "vs_benchmark"]]


def psi_table(monitoring: dict[str, Any]) -> pd.DataFrame:
    """feature_drift listesi -> (feature, psi, alert) frame; psi None -> NaN."""
    rows = monitoring.get("feature_drift", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["feature", "psi", "alert"])
    frame["psi"] = frame["psi"].apply(lambda value: float(value) if value is not None else math.nan)
    frame["alert"] = frame["alert"].astype(bool)
    return frame[["feature", "psi", "alert"]]


def mlflow_runs_table(
    tracking_uri: str | None = None,
    experiment: str = DEFAULT_EXPERIMENT,
) -> pd.DataFrame:
    """mlflow.db'deki run'larin hafif tablosu (run adi + temel metrikler).

    - mlflow yok/bulunamazsa bos frame (uyari dizer aranmaz).
    - sqlite tracking: rd0? buyuk tablo degil: 4-5 run beklenir.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return pd.DataFrame(columns=["run", "model", "mean_ic", "holdout_mean_ic"])
    try:
        client = MlflowClient(tracking_uri=tracking_uri or f"sqlite:///{MLFLOW_DB.as_posix()}")
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            return pd.DataFrame(columns=["run", "model", "mean_ic", "holdout_mean_ic"])
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"])
    except Exception:  # noqa: BLE001 - dashboard veri kaynaklari kesin gerekmez
        return pd.DataFrame(columns=["run", "model", "mean_ic", "holdout_mean_ic"])
    rows = []
    for run in runs:
        rows.append({
            "run": run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            "model": run.data.params.get("model", ""),
            "mean_ic": run.data.metrics.get("mean_ic"),
            "holdout_mean_ic": run.data.metrics.get("holdout_mean_ic"),
        })
    return pd.DataFrame(rows)


def production_alias_info() -> str:
    """registry 'production' alias metni (yoksa aciklama)."""
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return "mlflow kurulu degil"
    try:
        client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB.as_posix()}")
        version = client.get_model_version_by_alias(DEFAULT_REGISTRY, "production")
        return f"finans-ranker v{version.version} (run {version.run_id[:8]})"
    except Exception as error:  # noqa: BLE001
        return f"alias okunamadi: {error}"
