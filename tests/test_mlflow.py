"""Adim 5 (FAZ 3): MLflow deney takibi + model registry testleri (tmp dizinde)."""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

mlflow = pytest.importorskip("mlflow")  # mlflow kurulu degilse atla

from serving.model_store import _reconstruct_model, load_production_model
from training.mlflow_track import resolve_tracking_uri, setup_tracking


def _make_artifact(base: pathlib.Path) -> None:
    """model.pt + meta.json iceren kucuk bir yapay model dizini yazar.

    MLPRanker(hidden=[4], dropout=0.0) yapisi: Linear(2,4) [net.0], ReLU,
    Linear(4,1) [net.2]; agirliklar bunlarla uyumlu anahtar isimleriyle.
    """
    state = {"net.0.weight": torch.ones(4, 2), "net.0.bias": torch.zeros(4),
             "net.2.weight": torch.zeros(1, 4), "net.2.bias": torch.zeros(1)}
    torch.save(state, base / "model.pt")
    meta = {
        "model_name": "mlp",
        "feature_cols": ["f0", "f1"],
        "model_params": {"hidden": [4], "dropout": 0.0, "activation": "relu"},
    }
    with (base / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle)


def test_reconstruct_model_boyut_dogru(tmp_path: pathlib.Path):
    base = tmp_path / "mlp"
    base.mkdir()
    _make_artifact(base)
    model, meta = _reconstruct_model(base)
    x = torch.randn(3, 2)
    out = model(x)
    assert out.shape == (3,)
    assert meta["feature_cols"] == ["f0", "f1"]
    # agirliklar basar: net.0.weight ones
    assert float(model.net[0].weight.mean().item()) == pytest.approx(1.0)


def test_load_production_model_registry_then_fallback(tmp_path: pathlib.Path):
    registry_name = "test-ranker"
    artifact_dir = tmp_path / "artifacts"
    # gercek production run duzeni: dosyalar artifacts/model/ altindadir;
    # registry source'u run kokunu (artifacts/) gosterir -> _locate_model_dir iner
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)
    _make_artifact(model_dir)

    tracking_uri = resolve_tracking_uri(f"sqlite:///{(tmp_path / 'ml.db').as_posix()}", tmp_path)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.create_experiment("test-exp")
    with mlflow.start_run(experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
    client.create_registered_model(registry_name)  # mlflow3: once gerekli
    version = client.create_model_version(
        name=registry_name, source=artifact_dir.as_uri(), run_id=run_id
    )
    client.set_registered_model_alias(registry_name, "production", version.version)

    model, meta = load_production_model(
        tracking_uri=tracking_uri, registry_name=registry_name,
        local_models_dir=tmp_path / "bos-bir-yerde-yok",
    )
    assert meta["model_name"] == "mlp"
    assert model(torch.randn(5, 2)).shape == (5,)


def test_load_production_model_fallback_yerel(tmp_path: pathlib.Path):
    # registry yok/logsuz: yerel dirs/*
    models_dir = tmp_path / "models"
    base = models_dir / "mlp"
    base.mkdir(parents=True)
    _make_artifact(base)
    model, meta = load_production_model(
        tracking_uri=f"sqlite:///{(tmp_path / 'yok.db').as_posix()}",  # adrestek yok: fallback
        registry_name="yok-ranker",
        local_models_dir=models_dir,
    )
    assert meta["model_name"] == "mlp"
    assert model(torch.randn(7, 2)).shape == (7,)


def test_setup_tracking_kapalirken_none(tmp_path: pathlib.Path):
    assert setup_tracking({"enabled": False}, tmp_path) is None


def test_resolve_tracking_uri_goreli_sqlite(tmp_path):
    uri = resolve_tracking_uri("sqlite:///mlflow.db", tmp_path)
    assert uri == f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    # mutlak kaydet: degistmez
    absolute = f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
    assert resolve_tracking_uri(absolute, tmp_path) == absolute
