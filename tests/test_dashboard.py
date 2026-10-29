"""Adim 9 (FAZ 7): dashboard hazirlik katmani testleri (+ app import smoke)."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

st = pytest.importorskip("streamlit")  # streamlit kurulu degilse atla

from dashboard.prepare import (
    load_equity,
    load_scores,
    load_summary,
    mlflow_runs_table,
    production_alias_info,
    psi_table,
    top_bottom,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_load_scores_gercek_veya_bos():
    """Skor parquet varsa 44 satir guncel gun; yoksa temiz bos frame."""
    frame = load_scores()
    if (PROJECT_ROOT / "data" / "processed" / "latest_scores.parquet").is_file():
        assert len(frame) > 0
        assert {"date", "ticker", "score"} <= set(frame.columns)
        assert frame["score"].is_monotonic_decreasing  # sirali
    else:
        assert frame.empty


def test_top_bottom_sirasi():
    frame = pd.DataFrame({"ticker": ["A", "B", "C", "D", "E", "F"], "score": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]})
    top, bottom = top_bottom(frame, n=3)
    assert top["ticker"].tolist() == ["C", "B", "A"]  # en buyukten kucuge
    assert top["score"].tolist() == [3.0, 2.0, 1.0]
    assert bottom["score"].tolist() == [-3.0, -2.0, -1.0]


def test_load_summary_gercek_dosya():
    summary = load_summary()
    path = PROJECT_ROOT / "reports" / "backtest_summary.json"
    if path.is_file():
        assert "long_only" in summary and "long_short" in summary
        assert summary["long_only"]["metrics_net"]["total_return"] > 0.3  # bilinen deger
    else:
        assert summary == {}


def test_load_equity_gercek_dosya():
    equity = load_equity()
    path = PROJECT_ROOT / "reports" / "equity_curve.csv"
    if path.is_file():
        assert {"date", "long_short_net", "long_only_net", "spy"} <= set(equity.columns)
        assert len(equity) > 400
    else:
        assert equity.empty


def test_psi_table_none_guvenli():
    report = {"feature_drift": [{"feature": "f0", "psi": None, "alert": False}, {"feature": "f1", "psi": 0.3, "alert": True}]}
    table = psi_table(report)
    assert len(table) == 2
    assert table["psi"].iloc[0] != table["psi"].iloc[0]  # None -> NaN
    assert bool(table["alert"].iloc[1]) is True


def test_psi_table_bos():
    table = psi_table({"feature_drift": []})
    assert table.empty


def test_mlflow_runs_table_gercek_db():
    """mlflow.db varsa 4 run beklenir (3 variant + production); yoksa atla."""
    db = PROJECT_ROOT / "mlflow.db"
    if not db.is_file():
        pytest.skip("mlflow.db yok (CI ortami), gercek DB testi atlandi")
    table = mlflow_runs_table(experiment="finans-training")
    assert len(table) == 4
    assert "production-gru" in set(table["run"])
    # production run'in holdout metrigi dolu olmali
    prod_rows = table[table["run"] == "production-gru"]
    assert prod_rows["holdout_mean_ic"].notna().all()


def test_production_alias_info_bilgi():
    db_vardi = (PROJECT_ROOT / "mlflow.db").is_file()  # cagrimdan ONCE bak
    text = production_alias_info()
    if db_vardi:
        assert "finans-ranker" in text
    else:
        assert "okunamadi" in text or "kurulu" in text


def test_app_import_smoke():
    """app modulu import edilebilmeli (Streamli kaynagi olmadan)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("demo_app", PROJECT_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # sayfa fonksiyonlari cagrilmaz; sema kurulmaz
    assert hasattr(module, "page_scores")
    assert hasattr(module, "main")


def test_fmt_percent_nan_guvenli():
    import importlib.util

    spec = importlib.util.spec_from_file_location("demo_app", PROJECT_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.fmt_percent(0.1234) == "12.3%"
    assert module.fmt_percent(None) == "-"
    assert module.fmt_percent(float("nan")) == "-"
