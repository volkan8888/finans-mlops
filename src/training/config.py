"""Egitim yapilandirmasi (config/training.yaml).

Ingestion/features konvansiyonunu izler:
- goreli yollar proje kokune gore; utf-8-sig; logging; alan dogrulamasi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_TRAINING_CONFIG_PATH = "config/training.yaml"

_KNOWN_DEVICES = {"auto", "cpu", "cuda"}
_KNOWN_MODELS = {"ridge", "mlp", "lstm", "gru"}


@dataclass(frozen=True)
class TrainingConfig:
    """training.yaml icerigi; yollar proje kokune cozulmustur."""

    seed: int
    device: str
    holdout_years: int
    purge_days: int          # = label_horizon; fold sonu labe'in val'e tasmasini engeller
    embargo_days: int
    wf_min_train_years: int
    wf_val_months: int
    wf_step_months: int
    seq_len: int
    loss: str
    label_horizon: int
    defaults: dict[str, Any]
    models: tuple[dict[str, Any], ...]
    features_path: Path
    models_dir: Path
    holdout_predictions_path: Path
    project_root: Path = Path(".")
    mlflow: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Yapilandirma bulunamadi: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Yapilandirma dict bekliyordu, gelen: {type(data).__name__}")
    return data


def _positive_int(data: dict[str, Any], key: str, default: int, allow_zero: bool = False) -> int:
    value = int(data.get(key, default))
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{key} { '>0' if not allow_zero else '>=0'} olmali, gelen: {value}")
    return value


def load_training_config(path: str | Path = DEFAULT_TRAINING_CONFIG_PATH) -> TrainingConfig:
    """training.yaml yukler ve alanlari dogrular; yollar proje kokune cozulur."""
    config_path = Path(path)
    data = _load_yaml(config_path)
    project_root = config_path.resolve().parent.parent

    device = str(data.get("device", "auto"))
    if device not in _KNOWN_DEVICES:
        raise ValueError(f"device bilinmeyen: {device} (olasi {sorted(_KNOWN_DEVICES)})")

    loss = str(data.get("loss", "mse"))
    if loss not in {"mse", "listnet"}:
        raise ValueError(f"loss 'mse' veya 'listnet' olmali, gelen: {loss}")

    label_horizon = _positive_int(data, "label_horizon", 5)
    purge_days = _positive_int(data, "purge_days", label_horizon, allow_zero=True)
    # purge_days default'u label_horizon'a esit; config'e 0 yazilmis ve 0 verilmemisse ayni
    ayni = data.get("purge_days") is None
    if ayni:
        purge_days = label_horizon

    embargo_days = _positive_int(data, "embargo_days", 5, allow_zero=True)

    models_raw = data.get("models") or []
    if not models_raw:
        raise ValueError("models listesi bos olamaz (en az bir model tanimlayin)")
    models: list[dict[str, Any]] = []
    for entry in models_raw:
        name = str(entry.get("name", "")).lower()
        if name not in _KNOWN_MODELS:
            raise ValueError(f"Bilinmeyen model adi '{name}' ({_KNOWN_MODELS})")
        models.append({"name": name, "params": dict(entry.get("params") or {})})

    paths = data.get("paths") or {}
    features_path = Path(str(paths.get("features_path") or "data/processed/features.parquet"))
    models_dir = Path(str(paths.get("models_dir") or "models"))
    holdout_predictions_path = Path(str(
        paths.get("holdout_predictions_path") or "data/processed/holdout_predictions.parquet"
    ))
    features_path = features_path if features_path.is_absolute() else project_root / features_path
    models_dir = models_dir if models_dir.is_absolute() else project_root / models_dir
    holdout_predictions_path = (
        holdout_predictions_path
        if holdout_predictions_path.is_absolute()
        else project_root / holdout_predictions_path
    )

    cfg = TrainingConfig(
        seed=int(data.get("seed", 42)),
        device=device,
        holdout_years=_positive_int(data, "holdout_years", 2),
        purge_days=purge_days,
        embargo_days=embargo_days,
        wf_min_train_years=_positive_int(data, "wf_min_train_years", 3),
        wf_val_months=_positive_int(data, "wf_val_months", 6),
        wf_step_months=_positive_int(data, "wf_step_months", 6),
        seq_len=_positive_int(data, "seq_len", 20),
        loss=loss,
        label_horizon=label_horizon,
        defaults=dict(data.get("defaults") or {}),
        models=tuple(models),
        features_path=features_path,
        models_dir=models_dir,
        holdout_predictions_path=holdout_predictions_path,
        project_root=project_root,
        mlflow=dict(data.get("mlflow") or {}),
    )
    logger.info(
        "TrainingConfig yuklendi: seed=%d | holdout=%dy | wf=%dy/%dm/%dm | "
        "purge=%d+embargo=%d | seq_len=%d | loss=%s | modeller=%s | features=%s",
        cfg.seed, cfg.holdout_years, cfg.wf_min_train_years, cfg.wf_val_months,
        cfg.wf_step_months, cfg.purge_days, cfg.embargo_days, cfg.seq_len, cfg.loss,
        [m["name"] for m in cfg.models], cfg.features_path,
    )
    return cfg
