"""Adim 6 (FAZ 4) serving testleri — sentetik feature + sahte model yukleyici."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from fastapi.testclient import TestClient

from serving.app import FeatureSource, create_app
from training.models import RidgeBaseline, build_model

# canli feature konfigurasyonuyla uyumlu kolonlar (build_features ciktisindan)
LIVE_FEATURES = [
    "momentum_2", "momentum_3", "momentum_skip_4_1", "volatility_3",
    "rsi_3", "volume_change_3", "reversal_2",
]


def _live_mlp_loader() -> tuple[object, dict]:
    """Canli tablonun GERCEK kolonlarini bilgik malik 'mlp'."""
    model = build_model("mlp", len(LIVE_FEATURES), {"hidden": [4], "dropout": 0.0})
    return model, {
        "model_name": "mlp", "feature_cols": list(LIVE_FEATURES),
        "model_params": {"hidden": [4], "dropout": 0.0},
    }


def make_features_parquet(tmp_path: pathlib.Path, n_days: int = 40) -> pd.DataFrame:
    """(date, ticker, f0, f1, forward_return, label) sentetik panel."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    tickers = ("AAA", "BBB", "CCC", "DDD")
    rows = []
    base = {t: rng.normal(0, 1) for t in tickers}
    for date in dates:
        for ticker in tickers:
            f0 = float(base[ticker] + 0.1 * rng.normal())
            f1 = float(rng.normal())
            rows.append({
                "date": date, "ticker": ticker, "f0": f0, "f1": f1,
                "forward_return": f0 * 0.01, "label": abs(rng.uniform()),
            })
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "features.parquet", index=False)
    return frame


@pytest.fixture()
def source(tmp_path: pathlib.Path) -> FeatureSource:
    make_features_parquet(tmp_path)
    return FeatureSource(tmp_path / "features.parquet")


def _mlp_loader() -> tuple[object, dict]:
    model = build_model("mlp", 2, {"hidden": [4], "dropout": 0.0})
    return model, {
        "model_name": "mlp", "feature_cols": ["f0", "f1"],
        "model_params": {"hidden": [4], "dropout": 0.0},
    }


def _gru_loader() -> tuple[object, dict]:
    model = build_model("gru", 2, {"hidden": 4, "layers": 1, "dropout": 0.0})
    return model, {
        "model_name": "gru", "feature_cols": ["f0", "f1"],
        "model_params": {"hidden": 4, "layers": 1, "dropout": 0.0}, "seq_len": 5,
    }


def test_health_ok(tmp_path: pathlib.Path, source: FeatureSource):
    app = create_app(model_loader=_mlp_loader, features_path=tmp_path / "features.parquet")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "mlp"
    assert body["n_features"] == 2
    assert body["feature_source_ok"] is True
    assert body["latest_feature_date"] == str(source.latest_date().date())


def test_predict_tabular_son_gun(tmp_path: pathlib.Path, source: FeatureSource):
    app = create_app(model_loader=_mlp_loader, features_path=tmp_path / "features.parquet")
    response = TestClient(app).post("/predict", json={"ticker": "AAA"})
    assert response.status_code == 200
    body = response.json()
    assert body["date"] == str(source.latest_date().date())
    assert isinstance(body["score"], float)
    assert 0.0 <= body["cross_sectional_pct"] <= 1.0  # tabular: yuzde hesaplanir


def test_predict_gecmis_tarih_cozulur(tmp_path: pathlib.Path, source: FeatureSource):
    app = create_app(model_loader=_mlp_loader, features_path=tmp_path / "features.parquet")
    # 2024-01-06 cumartesi: onceki islem gunu 2024-01-05 (cuma) secilmeli
    response = TestClient(app).post("/predict", json={"ticker": "AAA", "date": "2024-01-06"})
    assert response.status_code == 200
    assert response.json()["date"] == "2024-01-05"


def test_predict_ticker_yok_404(tmp_path: pathlib.Path, source: FeatureSource):
    app = create_app(model_loader=_mlp_loader, features_path=tmp_path / "features.parquet")
    response = TestClient(app).post("/predict", json={"ticker": "ZZZ"})
    assert response.status_code == 404


def test_predict_gru_pencere_ve_yuzde_yok(tmp_path: pathlib.Path, source: FeatureSource):
    app = create_app(model_loader=_gru_loader, features_path=tmp_path / "features.parquet")
    response = TestClient(app).post("/predict", json={"ticker": "AAA"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["score"], float)
    assert "cross_sectional_pct" not in body  # sequence modellerde yuzde yok (belgeli sinir)


def test_batch_karman_katman(tmp_path: pathlib.Path, source: FeatureSource):
    app = create_app(model_loader=_mlp_loader, features_path=tmp_path / "features.parquet")
    response = TestClient(app).post(
        "/predict/batch", json={"tickers": ["AAA", "ZZZ"], "date": None}
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert "score" in results[0]
    assert "error" in results[1]  # bos ticker'a hata raporu, genel crack yok


def test_feature_source_pencere_tam_degil_hata(source: FeatureSource):
    dates = source.dates_of("AAA")
    with pytest.raises(KeyError):
        source.get_window("AAA", dates[2], ["f0", "f1"], 5)  # sadece 3 gun var


def test_feature_source_kolon_eksik(source: FeatureSource):
    dates = source.dates_of("AAA")
    with pytest.raises(ValueError):
        source.get_row("AAA", dates[-1], ["f0", "olmayan_kolon"])


def test_model_yuklenemeyince_degraded(tmp_path: pathlib.Path, source: FeatureSource):
    def bad_loader():
        raise RuntimeError("yapay hata")

    app = create_app(model_loader=bad_loader, features_path=tmp_path / "features.parquet")
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_ridge_model_serve_edilebilir(tmp_path: pathlib.Path, source: FeatureSource):
    frame = source.frame
    ridge = RidgeBaseline(alpha=1.0)
    ridge.fit(frame[["f0", "f1"]].to_numpy(), frame["label"].to_numpy())

    def ridge_loader():
        return ridge, {"model_name": "ridge", "feature_cols": ["f0", "f1"]}

    app = create_app(model_loader=ridge_loader, features_path=tmp_path / "features.parquet")
    body = TestClient(app).post("/predict", json={"ticker": "BBB"}).json()
    assert body["model_name"] == "ridge"
    assert isinstance(body["score"], float)
    # ridge lineer: score ≈ a*f0 + b*f1 + c; deterministik kontrol icin elle tahmin
    manual = float(ridge.predict(source.get_row("BBB", source.latest_date(), ["f0", "f1"]).to_frame().T.to_numpy())[0])
    assert body["score"] == pytest.approx(manual, abs=1e-6)


# ---------------------------------------------------------------------------
# CANLI kaynak (prices -> build_features; bayat-skor sorununun cozuMundeki
# kalite-islemi): label'siz son gun skorlanir; feature degerleri egitimle
# birebir ayni (AYNI build_features kodu); /health bayatligi raporlar.
# ---------------------------------------------------------------------------

from features.build import build_dataset
from features.config import FeaturesConfig


def _live_cfg() -> FeaturesConfig:
    """Kucuk pencereli cfg: sin maintainable warm-up; horizon 5 (bayatlik sayabilasi)."""
    return FeaturesConfig(
        momentum_windows=(2, 3),
        skip_window=4,
        skip=1,
        volatility_window=3,
        rsi_window=3,
        volume_window=3,
        reversal_window=2,
        label_horizon=5,
        winsorize_lower=0.1,
        winsorize_upper=0.9,
        min_tickers_per_date=2,
        prices_path=None,
        output_path=None,
        benchmarks=(),
    )


def _synthetic_prices_parquet(tmp_path: pathlib.Path, n_days: int = 70, seed: int = 5) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """(date, ticker, adj_close, volume) basic panel; validate URL parquet yazar."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    tickers = ("AAA", "BBB", "CCC", "DDD")
    rows = []
    for ticker in tickers:
        level = float(rng.uniform(50.0, 200.0))
        for date in dates:
            level = max(level * (1.0 + float(rng.normal(0.0005, 0.01))), 1.0)
            rows.append({
                "date": date, "ticker": ticker,
                "open": level, "high": level * 1.01, "low": level * 0.99,
                "close": level, "adj_close": level,
                "volume": float(rng.uniform(1_000.0, 50_000.0)),
            })
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "prices.parquet", index=False)
    return frame, dates


def test_from_prices_label_siz_son_gun_skorlaniyor(tmp_path: pathlib.Path):
    """Fiyat deposunun SON gunku bar'i feature'larla gelir (bayat-skor yok).

    Egitim tablosunda (build_dataset) pero bir ticker'in son 5 gunu
    label'siz OLDUGUNDEN dusulur; canli kaynak (build_features) bunlari
    KORUR -> skorlanabilir son gun = fiyat son gunu.
    """
    prices, dates = _synthetic_prices_parquet(tmp_path)
    source = FeatureSource.from_prices(tmp_path / "prices.parquet", _live_cfg())

    assert source.source_kind == "prices"
    assert source.latest_date() == dates[-1]
    assert source.stale_price_dates == 0

    # egitim dataset'i ayni son 5 gunu dusmediginden (label NaN) onlardan kisa
    dataset = build_dataset(prices, _live_cfg())
    assert dataset["date"].max() < source.latest_date()
    dataset_dates = set(dataset["date"].unique())
    feature_dates = set(source.frame["date"].unique())
    assert feature_dates - dataset_dates  # label'li olmayan kisim canli tabloda VAR


def test_canli_feature_degerleri_egitimle_birebir(tmp_path: pathlib.Path):
    """AYNI build_features kodu: (date, ticker) ortak kesimde kolonlar essizlenir."""
    prices, _ = _synthetic_prices_parquet(tmp_path)
    source = FeatureSource.from_prices(tmp_path / "prices.parquet", _live_cfg())

    live = source.frame
    dataset = build_dataset(prices, _live_cfg())
    feature_cols = [c for c in live.columns if c not in ("date", "ticker")]
    assert feature_cols  # kolon tarafinda adi egitimle ayni olsun
    assert list(dataset.columns)[: len(feature_cols) + 2] == ["date", "ticker"] + feature_cols

    key = ["date", "ticker"]
    joined = live.set_index(key).join(dataset.drop(columns=["forward_return", "label"]).set_index(key), how="inner", rsuffix="_ds")
    assert len(joined) > 0
    for column in feature_cols:
        pd.testing.assert_series_equal(joined[column], joined[column + "_ds"], check_names=False)


def test_from_prices_min_tickers_kurali(tmp_path: pathlib.Path):
    """min_tickers_per_date kurali canli kaynakta da uygulanir."""
    prices, dates = _synthetic_prices_parquet(tmp_path)
    cfg = _live_cfg()
    # bir gunde yalnizca AAA kalsin (1 < min_tickers=2): o gun atilir
    partial = prices.loc[~((prices["ticker"].isin(["BBB", "CCC", "DDD"])) & (prices["date"] == dates[40]))]
    partial.to_parquet(tmp_path / "prices.parquet", index=False)
    source = FeatureSource.from_prices(tmp_path / "prices.parquet", cfg)
    assert not (source.frame["date"] == dates[40]).any()


def test_health_canli_kaynak_bayatlik_raporlar(tmp_path: pathlib.Path):
    """score_date + data_end_date + stale_price_dates /health'te gorunur."""
    _synthetic_prices_parquet(tmp_path)
    features_yaml = tmp_path / "config" / "features.yaml"
    features_yaml.parent.mkdir(parents=True)
    features_yaml.write_text(
        "momentum_windows: [2, 3]\n"
        "momentum_skip: {window: 4, skip: 1}\n"
        "volatility_window: 3\n"
        "rsi_window: 3\n"
        "volume_window: 3\n"
        "reversal_window: 2\n"
        "label_horizon: 5\n"
        "winsorize: {lower: 0.1, upper: 0.9}\n"
        "min_tickers_per_date: 2\n"
        "benchmarks: []\n",
        encoding="utf-8",
    )
    app = create_app(
        model_loader=_live_mlp_loader,
        prices_path=tmp_path / "prices.parquet",   # canli kaynak tercihi
        features_path=tmp_path / "features.parquet",  # /health'in gosterdigi yedek yolu
        features_config_path=features_yaml,
    )
    body = TestClient(app).get("/health").json()
    assert body["source_kind"] == "prices"
    assert body["stale_price_dates"] == 0
    assert pd.Timestamp(body["data_end_date"]) == pd.Timestamp(body["score_date"])  # canli -> bayatlik 0
    assert body["incomplete_session_dropped"] is None

    response = TestClient(app).post("/predict", json={"ticker": "AAA"}).json()
    assert response["date"] == body["score_date"]
    assert isinstance(response["score"], float)


def test_kapanis_oncesi_kismi_oturum_atlanir(tmp_path: pathlib.Path):
    """Kapanis (16:00 America/New_York) oncesi: bugunun bar'i yok sayilir."""
    _prices, dates = _synthetic_prices_parquet(tmp_path)
    # en yeni bar bugun gibi davranilir; now parametresiyle inject edilir
    session_day = dates[-1]
    morning = pd.Timestamp(session_day, tz="America/New_York").replace(hour=10)
    evening = pd.Timestamp(session_day, tz="America/New_York").replace(hour=20)

    before = FeatureSource.from_prices(tmp_path / "prices.parquet", _live_cfg(), now=morning)
    assert before.latest_date() == dates[-2]            # kismi olaca tahmin edilir; atlandi
    assert before.incomplete_session_dropped == session_day
    assert before.stale_price_dates == 1

    after = FeatureSource.from_prices(tmp_path / "prices.parquet", _live_cfg(), now=evening)
    assert after.latest_date() == session_day            # kapanis sonrasi tam dagi
    assert after.incomplete_session_dropped is None
    assert after.stale_price_dates == 0
