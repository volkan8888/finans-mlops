"""Production modeli yukleme: MLflow registry'den, olmazsa yerel models/ dir.

Tercih sirasi:
1) MLflow: tracking URI (training.yaml'dan ya da parametreden) ile registry'de
   'production' alias'li versiyon aranir.
2) MLflow olamazsa/basarisizsa yerel models/*/ (meta.json + model.pt|joblib)
   en yeni klasor.

Serving, model dosyasinin nerede oldugunu bilmek ZORUNDA degil: tek
giris noktasi load_production_model().
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_CONFIG = DEFAULT_PROJECT_ROOT / "config" / "training.yaml"
DEFAULT_MODELS_DIR = DEFAULT_PROJECT_ROOT / "models"


def _mlflow_tracking_uri(cfg: dict[str, Any]) -> str:
    from training.mlflow_track import resolve_tracking_uri

    return resolve_tracking_uri(str(cfg.get("tracking_uri", "sqlite:///mlflow.db")), DEFAULT_PROJECT_ROOT)


def _reconstruct_model(base_dir: pathlib.Path):
    """meta.json + model.pt|joblib dosyalarindan model nesnesini kurar.

    Training ve serving'in AYNI insa yolunu kullanmasi icin model
    build_model ile yeniden kurulur ve kaydedilen agirliklar geri yuklenir
    (training-serving skew onlemenin ana adimi budur).
    """
    meta_path = base_dir / "meta.json"
    torch_path = base_dir / "model.pt"
    joblib_path = base_dir / "model.joblib"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json yok: {base_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    from training.models import build_model

    if torch_path.is_file():
        import torch

        model = build_model(str(meta["model_name"]), len(meta["feature_cols"]), dict(meta.get("model_params") or {}))
        state = torch.load(torch_path, map_location="cpu")
        model.load_state_dict(state)
        if hasattr(model, "eval"):
            model.eval()
        return model, meta
    if joblib_path.is_file():
        import joblib

        return joblib.load(joblib_path), meta
    raise FileNotFoundError(f"model dosyasi yok (model.pt / model.joblib): {base_dir}")


def _locate_model_dir(root: str | pathlib.Path) -> pathlib.Path:
    """Artifact kokunde meta.json iceren klasoru bulur.

    Production run'da dosyalar artifacts/model/ altina yazilir; registry
    source'u ise run'in artifact KOKUNU gosterir: kok/model tercih edilir,
    sonra kokun kendisi, sonra ilk recursive meta.json (sirali).
    """
    root_path = pathlib.Path(root)
    candidate = root_path / "model"
    if (candidate / "meta.json").is_file():
        return candidate
    if (root_path / "meta.json").is_file():
        return root_path
    for meta_path in sorted(root_path.rglob("meta.json")):
        return meta_path.parent
    raise FileNotFoundError(f"meta.json bulunamadi: {root_path}")


def _resolve_from_registry(client, registry_name: str, alias: str) -> pathlib.Path:
    """Yeni MLflow'un alias API: get_model_version_by_alias; dosya yolu dondurur."""
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    version = client.get_model_version_by_alias(registry_name, alias)
    source = str(version.source)
    if source.startswith("file:"):
        # file:///C:/... -> C:/... (Windows ve POSIX uyumlu cozumleme)
        parsed = urlparse(source)
        path = pathlib.Path(url2pathname(unquote(parsed.path)))
        if path.exists():
            return _locate_model_dir(path)  # run kokunden model/ alt klasorune iner
        raise FileNotFoundError(f"registry kaynak yolu bulunamadi: {source}")
    # uzaktaki URI: artifaktlari yerel olarak indir
    local_dir = client.download_artifacts(version.run_id, "model", dst_path=None)
    return pathlib.Path(local_dir)


def load_production_model(
    tracking_uri: str | None = None,
    registry_name: str | None = None,
    alias: str = "production",
    local_models_dir: str | pathlib.Path | None = None,
):
    """Production model + meta bilgisi; once MLflow, sonra yerel fallback.

    Donum: (model, meta). MLflow'dan (artifacts/model/ klasorune yazilmis)
    veya yerel models/<isim>/ klasorunden kurulur. Her sefer MODEL VE META
    UZERINDEN reconstruct edilir: serving ile ayni feature seti garantisi.
    """
    base_dir = None
    try:
        import mlflow

        cfg = {}
        if tracking_uri is None or registry_name is None:
            import yaml

            with DEFAULT_TRAINING_CONFIG.open("r", encoding="utf-8-sig") as handle:
                cfg = yaml.safe_load(handle).get("mlflow") or {}
        uri = tracking_uri or _mlflow_tracking_uri(cfg)
        name = registry_name or str(cfg.get("registry_name", "finans-ranker"))
        mlflow.set_tracking_uri(uri)
        client = mlflow.MlflowClient(tracking_uri=uri)
        base_dir = _resolve_from_registry(client, name, alias)
        logger.info("Production model MLflow registry'den: %s (alias=%s)", base_dir, alias)
    except Exception as error:  # noqa: BLE001 - her MLflow hata turu fallback'tir
        logger.warning("MLflow'dan production model alinamadi (%s); yerel fallback devrede", error)

    if base_dir is None:
        # yerel fallback: meta.json iceren en yeni model klasoru (veya dir/model/meta.json)
        models_dir = pathlib.Path(local_models_dir or DEFAULT_MODELS_DIR)
        candidates = sorted(models_dir.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"Production model bulunamadi: {models_dir} bos ya da yok")
        base_dir = candidates[0].parent
        logger.info("Production model yerel dosyadan: %s", base_dir)
    return _reconstruct_model(base_dir)
