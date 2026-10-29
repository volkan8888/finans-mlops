"""MLflow deney takibi yardimcilari (FAZ 3).

- Yerel backend: sqlite (mlflow.db) + mlruns/ klasoru (ikisi de gitignored).
- Her model varyanti icin BIR run; fold IC metrikleri o run'a yazilir.
- Secilen model "finans-ranker" registry'sine yazilir ve alias 'production'
  atanir (MLflow >=2.9 alias API'si; kurulu 3.17'de dogru yoldur, eski
  'stages' API'si kullanilmaz).
- Registry erisilemezse hata loglanir; dosya artifaktlari calismaya devam.
"""

from __future__ import annotations

import logging
import math
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


def resolve_tracking_uri(tracking_uri: str, project_root: pathlib.Path) -> str:
    """'sqlite:///mlflow.db' gibi goreli URI'yi proje kokune baglar."""
    prefix = "sqlite:///"
    if tracking_uri.startswith(prefix):
        db_path = pathlib.Path(tracking_uri[len(prefix):])
        if not db_path.is_absolute():
            return f"sqlite:///{(project_root / db_path).as_posix()}"
    return tracking_uri


def setup_tracking(mlflow_cfg: dict[str, Any], project_root: pathlib.Path):
    """MLflow backend kurar; (mlflow, client, tracking_uri, experiment_id) dondurur.

    mlflow kapaliysa None doner (hata degil). sqlite yolu proje kokune baglanir.
    """
    if not mlflow_cfg.get("enabled", False):
        logger.info("MLflow kapali (mlflow.enabled=false); deney takibi atlandi")
        return None
    import mlflow  # agir import; yalnizca gerektiginde

    tracking_uri = resolve_tracking_uri(str(mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")), project_root)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment_name = str(mlflow_cfg.get("experiment", "finans-training"))
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    logger.info("MLflow: uri=%s | experiment=%s (id=%s)", tracking_uri, experiment_name, experiment_id)
    return mlflow, client, tracking_uri, experiment_id


def log_variant_run(
    mlflow,
    experiment_id: str,
    model_name: str,
    params: dict[str, Any],
    fold_reports: list[dict[str, Any]],
    combined_summary: dict[str, float],
) -> str:
    """Bir model varyantinin walk-forward run'ini kaydeder ve run_id dondurur."""
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"variant-{model_name}") as run:
        # MLflow param degerleri skaler ister; list/dict/tuple degerler stringlenir
        safe_params = {
            f"param_{key}": (str(value) if isinstance(value, (list, dict, tuple)) else value)
            for key, value in params.items()
        }
        mlflow.log_params({"model": model_name, **safe_params})
        for report in fold_reports:
            value = float(report["val_mean_ic"])
            if not math.isnan(value):  # NaN metrik yazilmaz (log dogrulugu)
                mlflow.log_metric(f"fold{report['fold']:02d}_val_ic", value)
        for key, value in combined_summary.items():
            try:
                mlflow.log_metric(key, float(value))
            except (TypeError, ValueError):
                continue
        logger.info("MLflow variant run: %s (id=%s)", model_name, run.info.run_id)
        return run.info.run_id


def finalize_production_run(
    mlflow,
    client,
    model_name: str,
    experiment_id: str,
    variant_run_id: str | None,
    metrics: dict[str, float],
    artifact_paths: dict[str, str],
    registry_name: str,
) -> str | None:
    """Secilen modelin FINAL run'ini kaydeder; registry + production alias dener.

    metrics: final holdout IC/ICIR + backtest metrikleri (girer: NaN yazilmaz).
    artifact_paths: {kucuk ad: dosya yolu} (model.pt, meta.json, ...).
    Donus: registry version id (olursa) veya None.
    """
    tags = {"variant_run": variant_run_id} if variant_run_id else None
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"production-{model_name}", tags=tags) as run:
        run_id = run.info.run_id
        mlflow.log_param("model", model_name)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if not math.isnan(float(v))})
        for path in artifact_paths.values():
            mlflow.log_artifact(path, artifact_path="model")
        version = None
        try:
            # model registry girdisi yoksa once kayitli modeli ac
            try:
                client.get_registered_model(registry_name)
            except Exception:  # noqa: BLE001 - "yok" hatasi: modeli olustur
                client.create_registered_model(registry_name)
            version = client.create_model_version(
                name=registry_name,
                source=run.info.artifact_uri,  # yerel dosya URI'si: models'lerin mumkun
                run_id=run_id,
            )
            client.set_registered_model_alias(registry_name, "production", version.version)
            logger.info("MLflow registry: '%s' v%s --> alias 'production' (run_id=%s)",
                        registry_name, version.version, run_id)
        except Exception as error:  # noqa: BLE001 - registry hatasi akisi bloklamaz
            logger.warning("MLflow registry kaydi basarisiz (%s); dosya artifaktlari yerinde kalir", error)
    return version.version if version is not None else None
