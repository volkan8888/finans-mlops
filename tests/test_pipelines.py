"""Adim 8 (FAZ 6) + gunluk akis yenilemesi: Prefect akislari + yardimcilar.

Prefect kurulu degilse importorskip ile atlanir. Akislarin agir adimlari
(egitim gibi) testte KOSULMAZ: ingestion adimi komut enjeksiyonu ile kucuk
python komutlarina baglanir; skor ve monitoring sentetik parquet'ler uzerinden.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys

# Prefect'in pytest teardown'undaki 'I/O operation on closed file' log
# gurultusunu susturur (server kapanisi logu, dogrudan hata degil).
os.environ.setdefault("PREFECT_LOGGING_LEVEL", "ERROR")

import pandas as pd
import pytest

prefect = pytest.importorskip("prefect")


def _synthetic_features_parquet(tmp_path: pathlib.Path, n_days: int = 30) -> pd.DataFrame:
    """(date, ticker, f0, f1, label) sentetik panel; parquet yazar, frame doner."""
    frame = pd.DataFrame([
        {"date": date, "ticker": ticker, "f0": float(i) * 0.01, "f1": 1.0, "label": 0.5}
        for i, date in enumerate(pd.bdate_range("2025-01-01", periods=n_days))
        for ticker in ("AAA", "BBB", "CCC")
    ])
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    return frame


def _synthetic_predictions_parquet(tmp_path: pathlib.Path) -> pathlib.Path:
    """(date, ticker, score, label) panel: skor = -f0 tarz, label = skor (IC=1)."""
    features = _synthetic_features_parquet(tmp_path, n_days=20)
    predictions = features[["date", "ticker"]].copy()
    predictions["score"] = -features["f0"]
    predictions["label"] = -features["f0"]
    path = tmp_path / "holdout_predictions.parquet"
    predictions.to_parquet(path, index=False)
    return path


def _synthetic_source_and_loader(tmp_path: pathlib.Path):
    """(FeatureSource, fake_loader, features_path) uclusu: sentetik + 'mlp'."""
    from serving.app import FeatureSource
    from training.models import build_model

    _synthetic_features_parquet(tmp_path)
    source = FeatureSource(tmp_path / "features.parquet")
    model = build_model("mlp", 2, {"hidden": [4], "dropout": 0.0})

    def loader():
        return model, {
            "model_name": "mlp", "feature_cols": ["f0", "f1"],
            "model_params": {"hidden": [4], "dropout": 0.0},
        }

    return source, loader, (tmp_path / "features.parquet")


def test_latest_scores_frame_tum_tickerlar(tmp_path: pathlib.Path):
    from pipelines.shared import latest_scores_frame

    source, loader, _ = _synthetic_source_and_loader(tmp_path)
    frame = latest_scores_frame(project_root=tmp_path, model_loader=loader, source=source)
    assert len(frame) == 3  # AAA/BBB/CCC
    assert set(frame["ticker"]) == {"AAA", "BBB", "CCC"}
    assert frame["date"].nunique() == 1
    assert frame["score"].between(-5, 5).all()  # kucuk model: makul aralik
    assert set(frame["model_name"]) == {"mlp"}  # yeni kolon: model adi


def test_latest_scores_frame_sequence_model_pencere(tmp_path: pathlib.Path):
    from pipelines.shared import latest_scores_frame
    from serving.app import FeatureSource
    from training.models import build_model

    frame = pd.DataFrame([
        {"date": date, "ticker": ticker, "f0": float(i) * 0.01, "f1": 1.0}
        for i, date in enumerate(pd.bdate_range("2025-01-01", periods=30))
        for ticker in ("AAA", "BBB")
    ])
    path = tmp_path / "features.parquet"
    frame.to_parquet(path, index=False)
    source = FeatureSource(path)
    model = build_model("gru", 2, {"hidden": 4, "layers": 1, "dropout": 0.0})

    def loader():
        return model, {
            "model_name": "gru", "feature_cols": ["f0", "f1"],
            "model_params": {"hidden": 4, "layers": 1, "dropout": 0.0}, "seq_len": 5,
        }

    scores = latest_scores_frame(project_root=tmp_path, model_loader=loader, source=source)
    assert len(scores) == 2  # iki ticker; pencere motoru hatasiz candir
    assert scores["score"].notna().all()


def test_resolve_paths_konfigurasyonu(tmp_path: pathlib.Path):
    from pipelines.shared import (
        resolve_features_config_path,
        resolve_features_path,
        resolve_prices_path,
    )

    config = tmp_path / "serving.yaml"
    config.write_text(
        "features_path: data/processed/features.parquet\n"
        "prices_path: data/raw/prices.parquet\n"
        "features_config: config/features.yaml\n",
        encoding="utf-8",
    )
    assert resolve_features_path(tmp_path, config) == tmp_path / "data" / "processed" / "features.parquet"
    assert resolve_prices_path(tmp_path, config) == tmp_path / "data" / "raw" / "prices.parquet"
    assert resolve_features_config_path(tmp_path, config) == tmp_path / "config" / "features.yaml"


def test_resolve_paths_tanimli_degil(tmp_path: pathlib.Path):
    from pipelines.shared import resolve_features_config_path, resolve_prices_path

    config = tmp_path / "serving.yaml"
    config.write_text("features_path: a.parquet\n", encoding="utf-8")
    assert resolve_prices_path(tmp_path, config) is None
    assert resolve_features_config_path(tmp_path, config) is None


# ---------------------------------------------------------------------------
# gunluk akis: task sirasi ingestion -> dogrulama -> features -> skor ->
# tahmin kaydi -> monitoring. Ingestion, panel'e 1 is gunu (BDay) ekleyen
# sahte komutla temsil edilir: eger task sirasi dogruysa skor gunu, ingestion
# SONRASI panelin son gunudur.
# ---------------------------------------------------------------------------

CONFIG_FEATURES_YAML = (
    "momentum_windows: [2, 3]\n"
    "momentum_skip: {window: 4, skip: 1}\n"
    "volatility_window: 3\n"
    "rsi_window: 3\n"
    "volume_window: 3\n"
    "reversal_window: 2\n"
    "label_horizon: 5\n"
    "winsorize: {lower: 0.1, upper: 0.9}\n"
    "min_tickers_per_date: 3\n"
    "benchmarks: []\n"
    "paths:\n"
    "  prices_path: data/raw/prices.parquet\n"
    "  output_path: data/processed/features_wo.parquet\n"
)
CONFIG_SERVING_YAML = (
    "source: prices\n"
    "prices_path: data/raw/prices.parquet\n"
    "features_config: config/features.yaml\n"
    "features_path: data/processed/features.parquet\n"
)
INGEST_APPEND_CODE = (
    "import sys, pandas as pd; path = sys.argv[1]; frame = pd.read_parquet(path);"
    "last = frame[frame['date'] == frame['date'].max()].copy();"
    "last['date'] = last['date'] + pd.tseries.offsets.BDay(1);"
    "pd.concat([frame, last], ignore_index=True).to_parquet(path, index=False)"
)

FAKE_FRAME = pd.DataFrame({
    "date": [pd.Timestamp("2025-03-03"), pd.Timestamp("2025-03-03")],
    "ticker": ["AAA", "BBB"], "score": [0.4, -0.2], "model_name": ["mlp", "mlp"],
})


def _synthetic_prices_parquet(tmp_path: pathlib.Path, n_days: int = 40) -> pathlib.Path:
    """(date, ticker, fiyat/hacim kolonlari tam) panel; pandere sizi geirin."""
    import numpy as np

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    rows = []
    for ticker in ("AAA", "BBB", "CCC", "DDD"):
        level = 100.0
        for date in dates:
            level *= 1.0 + float(rng.normal(0.0005, 0.008))
            rows.append({
                "date": date, "ticker": ticker,
                "open": level, "high": level * 1.001, "low": level * 0.999,
                "close": level, "adj_close": level, "volume": 10_000.0,
            })
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(raw_dir / "prices.parquet", index=False)
    return raw_dir / "prices.parquet"


def _setup_daily(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, n_days: int = 40) -> pathlib.Path:
    """Gunluk akisin sentetik altyapisini kurar; prices parquet yolu doner."""
    from pipelines import daily_flow

    prices_path = _synthetic_prices_parquet(tmp_path, n_days=n_days)
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "features.yaml").write_text(CONFIG_FEATURES_YAML, encoding="utf-8")
    (config_dir / "serving.yaml").write_text(CONFIG_SERVING_YAML, encoding="utf-8")

    monkeypatch.setattr(daily_flow, "PROJECT_ROOT", tmp_path)

    from pipelines import shared

    def fake_scores(source, model_loader=None):
        frame = FAKE_FRAME.copy()
        frame["date"] = pd.Timestamp(source.latest_date())
        return frame

    monkeypatch.setattr(shared, "scores_for_day", fake_scores)

    predictions_path = _synthetic_predictions_parquet(tmp_path)
    monkeypatch.setattr(
        "monitoring.run.load_monitoring_config",
        lambda path=None: {
            "features_path": str(tmp_path / "data" / "processed" / "features_wo.parquet"),
            "predictions_path": str(predictions_path),
            "report_path": str(tmp_path / "reports" / "monitoring.json"),
            "roll_ic_window": 5, "psi_bins": 10, "psi_alert": 0.2, "alert_ic": 0.0,
        },
    )
    # monitoring, feature panelini okur; cikti hazirdadir
    base_frame = pd.DataFrame([
        {"date": pd.Timestamp(date), "ticker": ticker, "f0": 0.1, "f1": 1.0}
        for date in pd.bdate_range("2025-01-02", periods=n_days)
        for ticker in ("AAA", "BBB", "CCC")
    ])
    out_features = tmp_path / "data" / "processed" / "features_wo.parquet"
    out_features.parent.mkdir(parents=True, exist_ok=True)
    base_frame.to_parquet(out_features, index=False)
    return prices_path


def test_daily_pipeline_ingestion_ve_sira(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Task sirasi okunur: ingestion sirada ONCE (panel buyur) ve skor gunu onun sonrasi."""
    from pipelines import daily_flow

    prices_path = _setup_daily(tmp_path, monkeypatch)
    before = pd.read_parquet(prices_path)
    ingest_command = ("python", "-c", INGEST_APPEND_CODE, str(prices_path))

    summary = daily_flow.daily_pipeline(ingestion_command=ingest_command, with_ingestion=True)
    after = pd.read_parquet(prices_path)

    assert len(after) > len(before)  # ingestion paneli BUYUTMUS (sira: ingestion once)
    assert (tmp_path / "data" / "processed" / "latest_scores.parquet").is_file()
    # n_days=40 son bday = 2025-02-26; sahte ingestion +1 bday = 2025-02-27
    assert summary["scores_date"] == "2025-02-27"
    assert summary["ingestion"].startswith("ok:")
    assert summary["predictions_created"] is True
    assert (tmp_path / "data" / "predictions" / "predictions_2025-02-27.parquet").is_file()


def test_daily_pipeline_ingestion_hata_keser(tmp_path, monkeypatch):
    """Basarisiz ingestion -> RuntimeError; tahmin kaydi OLMAMALI."""
    from pipelines import daily_flow

    _setup_daily(tmp_path, monkeypatch)
    bad = ("python", "-c", "import sys; sys.exit(3)")
    with pytest.raises(RuntimeError):
        daily_flow.daily_pipeline(ingestion_command=bad, with_ingestion=True)
    assert not (tmp_path / "data" / "predictions").exists()


def test_tahmin_kaydi_varsa_uzerine_yazmaz(tmp_path, monkeypatch):
    """Ayni gun iki kos: ayni dosya, uzerine YAZILMAZ, cogaltma yok."""
    from pipelines import daily_flow

    _setup_daily(tmp_path, monkeypatch)
    directory = tmp_path / "data" / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    existing = directory / "predictions_2025-03-03.parquet"
    existing_frame = pd.DataFrame({"date": [pd.Timestamp("2025-03-03")], "ticker": ["ZZZ"], "score": [9.9]})
    existing_frame.to_parquet(existing, index=False)
    before_bytes = existing.read_bytes()

    result = daily_flow.task_record_predictions.fn(FAKE_FRAME)
    assert result == {"path": str(existing), "created": False}
    assert existing.read_bytes() == before_bytes  # DEGISTIRMEZ
    assert len(list(directory.glob("predictions_*.parquet"))) == 1  # cogaltma yok


def test_tahmin_kaydi_yeni_gunu_yazar(tmp_path, monkeypatch):
    from pipelines import daily_flow

    _setup_daily(tmp_path, monkeypatch)
    directory = tmp_path / "data" / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "predictions_2025-03-03.parquet"
    result = daily_flow.task_record_predictions.fn(FAKE_FRAME)
    assert result["created"] is True
    saved = pd.read_parquet(target)
    assert set(saved["ticker"]) == {"AAA", "BBB"}


def test_tahmin_kaydi_bos_frame_isadi_durusur(tmp_path, monkeypatch):
    from pipelines import daily_flow

    _setup_daily(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError):
        daily_flow.task_record_predictions.fn(pd.DataFrame())


def test_retrain_pipeline_komutlar_hafif():
    """Komut enjeksiyonu: cok kucuk python komutlari ile zincir kosmasi."""
    from pipelines.retrain_flow import retrain_pipeline

    light = (sys.executable, "-c", "print('merhaba')")
    summary = retrain_pipeline(commands=(light, light), with_monitoring=False)
    assert len(summary["commands_ok"]) == 2
    assert all(item.startswith("ok:") for item in summary["commands_ok"])


def test_retrain_pipeline_basarisiz_komut_runtime_error():
    """Basarisiz komut -> RuntimeError (flow kesilir; check davranisi elle)."""
    from pipelines.retrain_flow import retrain_pipeline

    bad = (sys.executable, "-c", "import sys; sys.exit(3)")
    with pytest.raises(RuntimeError):
        retrain_pipeline(commands=(bad,), with_monitoring=False)


def test_run_daily_parser():
    """scripts/run_daily CLI parser: modlar gecerli, default daily."""
    scripts_dir = pathlib.Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    module = importlib.import_module("run_daily")
    assert module.build_parser().parse_args([]).mode == "daily"
    assert module.build_parser().parse_args(["--mode", "retrain"]).mode == "retrain"
