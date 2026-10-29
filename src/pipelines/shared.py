"""Adim 8 (FAZ 6): gunluk akis yardimcilari (Prefect'siz test edilebilir katman)."""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SERVING_CONFIG = "config/serving.yaml"


def _load_serving_config(project_root: pathlib.Path, serving_config: str | pathlib.Path | None = None) -> dict[str, Any]:
    """config/serving.yaml okur; yerel dosya yoksa bos sozluk doner."""
    import yaml

    path = pathlib.Path(serving_config) if serving_config else pathlib.Path(project_root / DEFAULT_SERVING_CONFIG)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        return yaml.safe_load(handle) or {}


def resolve_features_path(project_root: pathlib.Path, serving_config: str | pathlib.Path | None = None) -> pathlib.Path:
    """config/serving.yaml'daki features_path'i proje kokune cozumler."""
    cfg = _load_serving_config(project_root, serving_config)
    configured = str(cfg.get("features_path", "data/processed/features.parquet"))
    candidate = pathlib.Path(configured)
    if candidate.is_absolute():
        return candidate
    return pathlib.Path(project_root) / candidate


def resolve_prices_path(project_root: pathlib.Path, serving_config: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """Canli feature hammadde yolu (config prices_path); tanimsizsa None."""
    cfg = _load_serving_config(project_root, serving_config)
    configured = cfg.get("prices_path")
    if configured is None:
        return None
    candidate = pathlib.Path(str(configured))
    return candidate if candidate.is_absolute() else pathlib.Path(project_root) / candidate


def resolve_features_config_path(project_root: pathlib.Path, serving_config: str | pathlib.Path | None = None) -> pathlib.Path | None:
    """Feature taniminin yolu (config features_config); tanimsizsa None."""
    cfg = _load_serving_config(project_root, serving_config)
    configured = cfg.get("features_config")
    if configured is None:
        return None
    candidate = pathlib.Path(str(configured))
    return candidate if candidate.is_absolute() else pathlib.Path(project_root) / candidate


def _default_model_loader() -> Callable[[], tuple[object, dict[str, Any]]]:
    from serving.model_store import load_production_model

    return load_production_model


def scores_for_day(
    source: Any,
    model_loader: Callable[[], tuple[object, dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    """Kaynagin en yeni gununun tum ticker skorlari; (date, ticker, score, model_name).

    - model_loader verilmemisse serving.model_store.load_production_model.
    - Sequence modellerde FeatureSource.get_window ticker bazli kullanilir;
      tabular modellerde tek satir (app.score_rows ile AYNI skor motoru).
    - Text source nesnesi FeatureSource.from_prices/from_frame ya da yedek
      FeatureSource(path) olabilir; hesap yolu aynidir.
    """
    if model_loader is None:
        model_loader = _default_model_loader()
    model, meta = model_loader()
    feature_cols = list(meta["feature_cols"])
    source.ensure_features(feature_cols)
    date = source.latest_date()
    rows: list[dict[str, Any]] = []
    from serving.app import score_rows

    for ticker in sorted(source._subframes):
        if meta.get("model_name") in {"lstm", "gru"}:
            matrix = source.get_window(ticker, date, feature_cols, int(meta.get("seq_len", 20)))
        else:
            matrix = source.get_row(ticker, date, feature_cols).to_frame().T
        scores = score_rows(model, meta, matrix)
        rows.append({
            "date": date, "ticker": ticker, "score": float(scores.ravel()[0]),
            "model_name": str(meta.get("model_name")),
        })
    return pd.DataFrame(rows)


def latest_scores_frame(
    project_root: pathlib.Path,
    features_path: str | pathlib.Path | None = None,
    model_loader: Callable[[], tuple[object, dict[str, Any]]] | None = None,
    source: object | None = None,
) -> pd.DataFrame:
    """Guncel gunun tum ticker skorlari (score_for_day'in features-parquet kullanimi).

    - source verilmemisse FeatureSource(features_path) kurulur; features_path
      verilmezse serving.yaml'dan cozumlenir.
    """
    model_loader = model_loader or _default_model_loader()
    if source is None:
        from serving.app import FeatureSource

        resolved = features_path or resolve_features_path(project_root)
        source = FeatureSource(resolved)
    return scores_for_day(source, model_loader)