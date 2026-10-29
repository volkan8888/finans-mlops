"""Production model + config dondurme kaydi (ileri test butunlugu).

Ne yapar (MODEL EGITMEZ; HER SEYI OKUYUP KAYDEDER):
- production modeli cozumler: MLflow registry 'production' alias -> yerel
  models/<isim>/ fallback (serving.model_store ile AYNI tercih sirasi).
- model dosyalari (model.pt|model.joblib + meta.json) SHA-256.
- meta.json'dan: model_name, feature_cols, seed, label_horizon, seq_len,
  model_params, train_dates, holdout_range.
- config dosyalarinin SHA-256 (training/features/serving/data/backtest/
  monitoring/universe): ileri test doneminde config DEGISIMININ fark
  edilmesini saglar.
- Cikti: reports/forward_test/freeze.json (gecmise donuk tek kayit;
  yeniden kosulursa uzerine yazilir - kast edilen: mevcut durumun
  dondurulmus imzasi).

Kullanim:
    .venv\\Scripts\\python scripts/freeze_model.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from serving import model_store

logger = logging.getLogger(__name__)

DEFAULT_FREEZE_PATH = PROJECT_ROOT / "reports" / "forward_test" / "freeze.json"
CONFIG_FILES = (
    "config/training.yaml",
    "config/features.yaml",
    "config/serving.yaml",
    "config/data.yaml",
    "config/backtest.yaml",
    "config/monitoring.yaml",
    "config/universe.txt",
)
MODEL_FILES = ("model.pt", "model.joblib")


def sha256_file(path: pathlib.Path) -> str:
    """Dosya iceriginin SHA-256; dosya yoksa bos string (belirsizlik YOK)."""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_production(project_root: pathlib.Path) -> dict:
    """model_store tercih sirasi ile modelin YERINI ve kaynagini bulur.

    Donus: {"base_dir", "kind", "version"}; kind: mlflow_registry | local.
    MLflow alias YALNIZCA proje kokunde mlflow.db VARSA denenir (bos/temiz
    makinede gereksiz mlflow acilimi ve yanlis GOLD yola dusulmesin).
    """
    kind, version = "local", None
    tracking_db = project_root / "mlflow.db"
    if tracking_db.is_file():
        try:
            with (project_root / "config" / "training.yaml").open("r", encoding="utf-8-sig") as handle:
                import yaml

                cfg = (yaml.safe_load(handle) or {}).get("mlflow") or {}
            uri = model_store._mlflow_tracking_uri(cfg)
            name = str(cfg.get("registry_name", "finans-ranker"))
            import mlflow

            mlflow.set_tracking_uri(uri)
            client = mlflow.MlflowClient(tracking_uri=uri)
            resolved = model_store._resolve_from_registry(client, name, "production")
            mlflow_version = client.get_model_version_by_alias(name, "production")
            kind, version = "mlflow_registry", str(mlflow_version.version)
            return {"base_dir": pathlib.Path(resolved), "kind": kind, "version": version}
        except Exception as error:  # noqa: BLE001 - fallback izlenir: yerel dosya
            logger.info("MLflow 'production' alias bulunamadi (%s); yerel fallback devrede", error)
    # yerel fallback: en yeni models/*/meta.json
    models_dir = project_root / "models"
    best = None
    if models_dir.is_dir():
        for candidate in sorted(models_dir.glob("*/meta.json")):
            if best is None or candidate.stat().st_mtime > best.stat().st_mtime:
                best = candidate
    if best is None:
        raise FileNotFoundError("Production model bulunamadi (meta.json yok)")
    return {"base_dir": best.parent, "kind": "local", "version": None}


def collect_freeze(project_root: pathlib.Path = PROJECT_ROOT) -> dict:
    """Donmus imzayi toplar: model + dosya hash + config hash + meta ozet."""
    source = _resolve_production(project_root)
    base_dir = source["base_dir"]
    if not (base_dir / "meta.json").is_file():
        raise FileNotFoundError(f"meta.json yok: {base_dir}")
    with (base_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    model_hashes = {
        name: sha256_file(base_dir / name)
        for name in MODEL_FILES
        if sha256_file(base_dir / name) != ""
    }
    if not model_hashes:
        raise FileNotFoundError(f"model dosyasi yok (model.pt / model.joblib): {base_dir}")
    config_hashes = {
        relative: sha256_file(project_root / relative)
        for relative in CONFIG_FILES
    }
    return {
        "frozen_at": str(pd.Timestamp.now()),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "model": {
            "name": meta.get("model_name"),
            "source_kind": source["kind"],
            "registry_version": source["version"],
            "base_dir": str(base_dir),
            "sha256": {"model_files": model_hashes, "meta.json": sha256_file(base_dir / "meta.json")},
        },
        "meta_summary": {
            key: meta.get(key)
            for key in (
                "feature_cols",
                "seed",
                "label_horizon",
                "seq_len",
                "model_params",
                "train_dates",
                "holdout_range",
            )
        },
        "config_sha256": config_hashes,
        "note": (
            "Ileri test bu imzayla sinirlidir. Model dosyalari, config'ler veya "
            "hiperparametreler degisirse bu kayit gecerliligini yitirir "
            "(yeniden freeze edilmesi icin acikca karara baglanmalidir)."
        ),
    }


def main() -> int:
    """freeze.json yazar; ozet loglar."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        frozen = collect_freeze()
    except FileNotFoundError as error:
        logger.error("Freeze basarisiz: %s", error)
        return 1
    DEFAULT_FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_FREEZE_PATH.write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "freeze.json yazildi: %s | model=%s (%s) | model_sha8=%s",
        DEFAULT_FREEZE_PATH,
        frozen["model"]["name"],
        frozen["model"]["source_kind"],
        next(iter(frozen["model"]["sha256"]["model_files"].values()))[:8],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
