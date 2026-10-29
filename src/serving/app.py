"""Adim 6 (FAZ 4): production skor servisi (FastAPI).

Giris noktalari:
- GET  /health           : model + feature kaynak durumu
- POST /predict          : tek (ticker, date) skor
- POST /predict/batch    : ayni gun birden cok ticker

Motor:
- Model, serving/model_store.load_production_model ile yuklenir
  (MLflow registry -> yerel fallback); feature kolonlari meta.json'dan gelir.
- Feature girdisi: istemci feature yazmak zorunda DEGILDIR. Varsayilan
  kaynak fiyat deposudur (data/raw/prices.parquet -> build_features):
  training'deki AYNI feature kodu label'siz calisir ve canli skor,
  feature'lari tam hesaplanabilen EN SON fiyat gununu skorlar (bayat
  skor yok). Egitimde uretilen features.parquet YALNIZCA yedek
  kaynaktir (fiyat deposu okunamazsa); o kaynak-labellarda son horizon
  gunu dusuldugunden skorlar Earth-tagim 5 is gunu geride kalir; bu
  yuzden yedek kaynak ve bayatlik /health'te raporlanir.
- Skor: tabular model tek satir, sequence model pencere uzerinden;
  cross_sectional_pct: ayni gundeki tum ticker'lerin skorlarinda istenen
  ticker'in yuzdelik sirasi (0..1). Not: bu yuzde ilk surumde yalnizca
  tabular modellerde hesaplanir (sequence pencere maliyeti; belgeli sinir).
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from features.config import FeaturesConfig

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SERVING_CONFIG = DEFAULT_PROJECT_ROOT / "config" / "serving.yaml"
DEFAULT_FEATURES_CONFIG = DEFAULT_PROJECT_ROOT / "config" / "features.yaml"

SEQUENCE_NAMES = {"lstm", "gru"}
EASTERN_TZ = "America/New_York"
MARKET_CLOSE_HOUR = 16  # ABD oturumu kapanisi: 16:00 America/New_York


def eastern_now(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Verilen (veya guncel) zamani ABD pazari saatine (Eastern) getirir.

    Saat dilimi bilgisi YOKSA: girisin Eastern oldugu kabul edilir (yanilma
    riski agirlikli olarak yalnizca testlerde).
    """
    stamp = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=EASTERN_TZ)
    if stamp.tzinfo is None:
        return stamp.tz_localize(EASTERN_TZ)
    return stamp.tz_convert(EASTERN_TZ)


def price_session_limit(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Skorlamada kullanilabilecek EN YENI fiyat gununun ust siniri (bu gun HARIC).

    Kural (gunluk akis: docs/step8_orchestration.md):
    - Kapanis (16:00 Eastern) ONCESI: bugunun bar'i kismidir; ust sinir bugun
      (bugun skora girmez).
    - Kapanis SONRASI: bugunun bar'i tamamlanmis kabul edilir; ust sinir yarin
      (bugun dahil).
    """
    stamp = eastern_now(now)
    today = pd.Timestamp(stamp.date())
    if (stamp.hour, stamp.minute) < (MARKET_CLOSE_HOUR, 0):
        return today
    return today + pd.Timedelta(days=1)


def drop_incomplete_session(
    prices: pd.DataFrame, now: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Kismi oturum barini fiyat panelinden atar; (temiz_frame, atilan_gun) dondurur.

    Atilan gun LOGLANIR: kapanis oncesi cekilen son gun eksik olur; onu
    skorlamak yanlis fiyat demektir. ABD kapanisi sonrasi calisilirsa
    hicbir gun atilmaz (o gunun bar'i tam olmustur).
    """
    if prices is None or prices.empty:
        return prices, None
    limit = price_session_limit(now)
    last = pd.Timestamp(prices["date"].max())
    if last < limit:
        return prices, None
    removed_rows = int(prices["date"].ge(limit).sum())
    pruned = prices.loc[prices["date"] < limit]
    logger.warning(
        "Tamamlanmamis oturum atlandi: %s (ust sinir=%s; %d satir)",
        last.date(), limit.date(), removed_rows,
    )
    return pruned, last


class FeatureSource:
    """Feature kaynagi: (ticker, date) satir/pencere okuyucu.

    Iki kurulum yolu:
    - FeatureSource(path): features.parquet okur (eski davranis; test/yedek).
    - FeatureSource.from_prices(...): fiyat deposundan build_features
      (features/build.py) ile label'siz canli tablo kurulur; training'deki
      AYNI kod nedeniyle kolonlar ve sirasi egitimle birebir ayni olur.

    Ortak alanlar:
    - load'da date kolonu datetime'a cevrilir ve (ticker, date) siralanir.
    - ensure_features: meta.json'daki feature_cols kaynakta var mi kontrolu.
    - get_row: tek satir (Series), get_window: son seq_len gun matrisi (DataFrame).
    - latest_date: skorlanabilecek en yeni gun (= tablonun en yeni gunu).
    - source_kind: 'features_parquet' | 'prices' (yedek vs canli kaynak).
    - prices_end_date: ham fiyat deposunun en son gunu (yedek modda None).
    - stale_price_dates: skor gununun arkasindaki fiyat gunu sayisi (bayatlik).
    - incomplete_session_dropped: kapanis oncesi atilan kismi oturum gunu.
    """

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)
        self.source_kind = "features_parquet"
        self.prices_end_date: pd.Timestamp | None = None
        self.stale_price_dates = 0
        self.incomplete_session_dropped: pd.Timestamp | None = None
        if not self.path.is_file():
            raise FileNotFoundError(f"Feature kaynagi bulunamadi: {self.path}")
        self._install(pd.read_parquet(self.path))

    def _install(self, frame: pd.DataFrame) -> None:
        """Frame'i kurar: datetime + (date, ticker) siralama + per-ticker alt tablo.

        ticker bazli alt cerceveler: bir ticker'in ardisik gunleri burada
        art arda satirlarda gelir ((date, ticker) sirali ana tabloda ayni
        ticker'in gunleri birbirinden uzaktadir; pencere kurgusu icin
        per-ticker dilim sart).
        """
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
        self.frame = frame
        self._subframes: dict[str, pd.DataFrame] = {}
        self._positions: dict[str, dict[pd.Timestamp, int]] = {}
        for ticker, group in frame.groupby("ticker", sort=False):
            sub = group.reset_index(drop=True)
            self._subframes[ticker] = sub
            self._positions[ticker] = {date: pos for pos, date in enumerate(sub["date"])}

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        source_kind: str = "frame",
        prices_end_date: pd.Timestamp | None = None,
        stale_price_dates: int = 0,
        incomplete_session_dropped: pd.Timestamp | None = None,
    ) -> FeatureSource:
        """Hazir feature tablosundan kaynak kurar (gunluk akis ve testler icin)."""
        source = cls.__new__(cls)
        source.path = None
        source.source_kind = source_kind
        source.prices_end_date = prices_end_date
        source.stale_price_dates = int(stale_price_dates)
        source.incomplete_session_dropped = incomplete_session_dropped
        source._install(frame)
        return source

    @classmethod
    def from_prices(
        cls,
        prices_path: str | pathlib.Path,
        features_config: FeaturesConfig | None = None,
        features_config_path: str | pathlib.Path | None = None,
        now: pd.Timestamp | None = None,
        drop_incomplete: bool = True,
    ) -> FeatureSource:
        """Fiyat deposundan CANLI feature tablosu (label'siz; son horizon dahil).

        - TRAINING-SERVING SKEW ONLEMI: features.build.build_features
          cagrilir (training ile AYNI fonksiyon; kod kopyasi YOK).
          Egitimin features.parquet dosyasina DOKUNULMAZ.
        - Skorlanan gun: feature'larinin TAMAMI (warm-up dahil) hesaplanabilen
          en son gun; min_tickers_per_date kurali build_features'in icinde
          uygulanir.
        - Kapanis-oncesi kismi oturum (drop_incomplete=True) atilir.
        """
        from features.build import build_features

        prices = pd.read_parquet(prices_path)
        if prices.empty:
            raise ValueError(f"Fiyat paneli bos: {prices_path}")
        prices["date"] = pd.to_datetime(prices["date"])
        raw_dates = pd.DatetimeIndex(pd.unique(prices["date"]))
        if drop_incomplete:
            prices, dropped = drop_incomplete_session(prices, now)
        else:
            dropped = None
        if prices.empty:
            raise ValueError("Kismi oturum tek gundu ve bu yuzden atlandi; skor tablosu bos")
        cfg = features_config or cls._load_config(features_config_path)
        frame = build_features(prices, cfg)
        if frame.empty:
            raise ValueError(f"Feature uretimi bos: {prices_path} (warm-up / min_tickers)")
        score_end = pd.Timestamp(frame["date"].max())
        stale = int((raw_dates > score_end).sum())
        return cls.from_frame(
            frame,
            source_kind="prices",
            prices_end_date=raw_dates.max(),
            stale_price_dates=stale,
            incomplete_session_dropped=dropped,
        )

    @staticmethod
    def _load_config(path: str | pathlib.Path | None) -> FeaturesConfig:
        """Feature tanimini yukler (config/features.yaml varsayilan)."""
        from features.config import load_features_config

        if path is None:
            return load_features_config(DEFAULT_FEATURES_CONFIG)
        candidate = pathlib.Path(path)
        return load_features_config(candidate if candidate.is_absolute() else DEFAULT_PROJECT_ROOT / candidate)

    def _sub(self, ticker: str) -> pd.DataFrame:
        sub = self._subframes.get(ticker)
        if sub is None:
            raise KeyError(ticker)
        return sub

    def _require_columns(self, feature_cols: list[str]) -> None:
        missing = [column for column in feature_cols if column not in self.frame.columns]
        if missing:
            raise ValueError(f"Feature kaynaginda kolon yok: {missing}")

    def ensure_features(self, feature_cols: list[str]) -> None:
        """meta.json'dan gelen feature kolonlarinin kaynakta bulunup bulunmadigini test eder."""
        self._require_columns(feature_cols)

    def latest_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame["date"].max())

    def dates_of(self, ticker: str) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._sub(ticker)["date"])

    def resolve_date(self, ticker: str, date: str | None) -> pd.Timestamp:
        """Istenen gun yoksa o gunden ONCEKI en yakin islem gunu secilir."""
        target = pd.Timestamp(date) if date else self.latest_date()
        dates = self.dates_of(ticker)
        eligible = dates[dates <= target]
        if len(eligible) == 0:
            raise KeyError(f"{ticker} icin ilk islem gunu {dates[0].date()}; oncesi yok")
        return pd.Timestamp(eligible[-1])

    def get_row(self, ticker: str, date: pd.Timestamp, feature_cols: list[str]) -> pd.Series:
        self._require_columns(feature_cols)
        sub = self._sub(ticker)
        position = self._positions[ticker].get(date)
        if position is None:
            raise KeyError(f"({ticker}, {date.date()}) satiri feature kaynakta yok")
        return sub.loc[position, feature_cols]

    def get_window(
        self, ticker: str, date: pd.Timestamp, feature_cols: list[str], seq_len: int
    ) -> pd.DataFrame:
        """Son seq_len gunun (ticker'a ait) feature matrisi; tam pencere yoksa hata."""
        self._require_columns(feature_cols)
        sub = self._sub(ticker)
        position = self._positions[ticker].get(date)
        if position is None:
            raise KeyError(f"({ticker}, {date.date()}) satiri feature kaynakta yok")
        start = position - seq_len + 1
        if start < 0:
            raise KeyError(f"{ticker} @ {date.date()} icin {seq_len} gunluk pencere yok")
        window = sub.loc[start:position]
        if len(window) < seq_len:
            raise KeyError(f"{ticker} @ {date.date()} icin pencere tam degil")
        return window[feature_cols]


def score_rows(model, meta: dict[str, Any], matrix: pd.DataFrame) -> np.ndarray:
    """Modeli (N,F) tabanli matrise uygular; 1-Boyutlu skor dizisi dondurur.

    - torch tabular: (N, F) tensor
    - torch sequence: cagiran taraf (1, T, F) pencere gecirmelidir; bu fonksiyon
      sequence icin (T, F) matris alir ve batch boyutu 1 kabul eder.
    - ridge (sklearn): numpy predict.
    """
    value = matrix.to_numpy(dtype="float32")
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            if meta.get("model_name") in SEQUENCE_NAMES:
                output = model(torch.tensor(value, dtype=torch.float32).unsqueeze(0))  # (1, T, F) -> (1,)
            else:
                output = model(torch.tensor(value, dtype=torch.float32))  # (N, F) -> (N,)
        array = output.detach().cpu().numpy()
    else:
        array = model.predict(value)
    return np.asarray(array).ravel()


class PredictRequest(BaseModel):
    """/predict istek govdesi: ticker zorunlu; date yoksa en guncel gun."""

    ticker: str = Field(min_length=1, max_length=16)
    date: str | None = None  # "YYYY-MM-DD"


class BatchPredictRequest(BaseModel):
    """/predict/batch istek govdesi: ayni gun icin birden cok ticker."""

    tickers: list[str] = Field(min_length=1)
    date: str | None = None


def _score_one(
    model, meta: dict[str, Any], source: FeatureSource, ticker: str, date: str | None
) -> tuple[float, pd.Timestamp, np.ndarray | None]:
    """Tek (ticker, date) icin skor; ayni gunun cross-section skorlariyla.

    Donus: (score, islem gunu, cross-section skor dizisi veya None).
    """
    feature_cols = meta["feature_cols"]
    source.ensure_features(feature_cols)
    resolved = source.resolve_date(ticker, date)
    if meta.get("model_name") in SEQUENCE_NAMES:
        matrix = source.get_window(ticker, resolved, feature_cols, int(meta.get("seq_len", 20)))
        cross_scores: np.ndarray | None = None  # belgeli sinir: ilk surumde tabular icin
    else:
        matrix = source.get_row(ticker, resolved, feature_cols).to_frame().T
        same_day = source.frame[source.frame["date"] == resolved]
        cross_scores = score_rows(model, meta, same_day[feature_cols])
    scores = score_rows(model, meta, matrix)
    score = float(np.asarray(scores).ravel()[0])
    return score, resolved, cross_scores


def _absolute(root: pathlib.Path, configured: str | None) -> pathlib.Path | None:
    """Config'teki goreli yolu proje kokune uygular; yoksa None dondurur."""
    if not configured:
        return None
    path = pathlib.Path(str(configured))
    return path if path.is_absolute() else root / path


def create_app(
    model_loader: Callable[[], tuple[object, dict[str, Any]]] | None = None,
    features_path: str | pathlib.Path | None = None,
    prices_path: str | pathlib.Path | None = None,
    features_config_path: str | pathlib.Path | None = None,
    now: pd.Timestamp | None = None,
) -> FastAPI:
    """FastAPI uygulamasini kurar; model ve feature kaynagini HEMEN yukler.

    - model_loader verilmezse model_store.load_production_model kullanilir.
    - Kaynak sirasi (config/serving.yaml'daki 'source' anahtarina gore):
        source: prices           -> fiyat deposundan build_features (canli)
        source: features_parquet -> egitimin features.parquet'i (yedek)
      'prices' seciliyse ve fiyat deposu okunamazsa features.parquet'e
      duser; /health bayatligi raporlar (yedek kaynak bilincinde dusus).
    - features_path/prices_path/features_config_path parametrelere verilirse
      config yerine o kullanilir (test enjeksiyonu).
    - Yukleme hatalari servisi kesmez; /health 'degraded' doner.
    """
    application = FastAPI(title="finans-ranker API", version="0.3.1")
    state: dict[str, Any] = {
        "model": None, "meta": None, "source": None,
        "features_path": None, "prices_path": None,
    }

    with DEFAULT_SERVING_CONFIG.open("r", encoding="utf-8-sig") as handle:
        serving_cfg = yaml.safe_load(handle) or {}
    source_mode = str(serving_cfg.get("source", "prices")).lower()
    configured_features = features_path or _absolute(
        DEFAULT_PROJECT_ROOT, serving_cfg.get("features_path", "data/processed/features.parquet")
    )
    resolved_features = pathlib.Path(configured_features)
    resolved_prices = (
        prices_path
        if prices_path is not None
        else _absolute(DEFAULT_PROJECT_ROOT, serving_cfg.get("prices_path"))
    )
    resolved_features_config = (
        features_config_path
        if features_config_path is not None
        else _absolute(DEFAULT_PROJECT_ROOT, serving_cfg.get("features_config"))
    )
    state["features_path"] = str(resolved_features)
    state["prices_path"] = str(resolved_prices) if resolved_prices else None

    try:
        loader = model_loader
        if loader is None:
            from serving.model_store import load_production_model

            loader = load_production_model
        model, meta = loader()
        state["model"], state["meta"] = model, meta
        logger.info("Model yuklendi: %s | feature=%d", meta.get("model_name"), len(meta.get("feature_cols", [])))
    except Exception as error:  # noqa: BLE001 - servis ayakta kalsin; /health rapor eder
        logger.warning("Model yuklenemedi (%s); /health 'degraded' raporlayacak", error)

    try:
        if prices_path is not None:
            # acik canli tercihi: cagiran fiyat deposunu enjekte etti
            state["source"] = FeatureSource.from_prices(
                prices_path,
                features_config_path=resolved_features_config,
                now=now,
            )
        elif features_path is not None:
            # acik yedek tercihi: features-parquet kaynagi (testler/eski davranis)
            state["source"] = FeatureSource(resolved_features)
        elif source_mode == "prices" and resolved_prices is not None and resolved_prices.is_file():
            state["source"] = FeatureSource.from_prices(
                resolved_prices,
                features_config_path=resolved_features_config,
                now=now,
            )
        else:
            if source_mode == "prices":
                logger.warning("Fiyat deposu bulunamadi; features.parquet yedegi kullanilacak")
            state["source"] = FeatureSource(resolved_features)
    except Exception as error:  # noqa: BLE001
        logger.warning("Feature kaynagi yuklenemedi: %s", error)
        state["source"] = None
        if prices_path is not None:
            try:  # canli kaynak kurulamadiysa yedek kaynagi dene (bayat olabilir)
                state["source"] = FeatureSource(resolved_features)
                logger.warning("Canli feature kaynagi kurulamadi; features.parquet yedegine dusuldu (skorlar bayat olabilir)")
            except Exception:  # noqa: BLE001
                state["source"] = None

    def _require_ready() -> tuple[object, dict[str, Any], FeatureSource]:
        if state["model"] is None or state["meta"] is None:
            raise HTTPException(status_code=503, detail="Model yuklenmemis")
        if state["source"] is None:
            raise HTTPException(status_code=503, detail="Feature kaynagi yuklenmemis")
        return state["model"], state["meta"], state["source"]

    @application.get("/health")
    def health() -> dict[str, Any]:
        meta = state["meta"]
        source: FeatureSource | None = state["source"]
        has_source = source is not None
        latest = source.latest_date() if has_source else None
        scores_end = source.prices_end_date if has_source else None
        return {
            "status": "ok" if (meta is not None and has_source) else "degraded",
            "model_name": meta.get("model_name") if meta else None,
            "n_features": len(meta.get("feature_cols", [])) if meta else 0,
            "feature_source": state.get("features_path"),
            "feature_source_ok": has_source,
            "source_kind": source.source_kind if has_source else None,
            "latest_feature_date": str(latest.date()) if latest is not None else None,
            "score_date": str(latest.date()) if latest is not None else None,
            "data_end_date": str(scores_end.date()) if scores_end is not None else None,
            # bayatlik: skor gununun gerisinde kalan fiyat gunu sayisi
            # (canli kaynakta 0; features.parquet yedeginde >= label_horizon)
            "stale_price_dates": source.stale_price_dates if has_source else None,
            "incomplete_session_dropped": (
                str(source.incomplete_session_dropped.date())
                if has_source and source.incomplete_session_dropped is not None
                else None
            ),
        }

    @application.post("/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        model, meta, source = _require_ready()
        try:
            score, resolved, cross_scores = _score_one(model, meta, source, request.ticker, request.date)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        response: dict[str, Any] = {
            "ticker": request.ticker,
            "date": str(resolved.date()),
            "score": score,
            "model_name": meta.get("model_name"),
        }
        if cross_scores is not None:
            response["cross_sectional_pct"] = float((cross_scores <= score).mean())
        return response

    @application.post("/predict/batch")
    def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
        model, meta, source = _require_ready()
        results: list[dict[str, Any]] = []
        for ticker in request.tickers:
            try:
                score, resolved, cross_scores = _score_one(model, meta, source, ticker, request.date)
            except (KeyError, ValueError) as error:
                results.append({"ticker": ticker, "error": str(error)})
                continue
            item: dict[str, Any] = {"ticker": ticker, "date": str(resolved.date()), "score": score}
            if cross_scores is not None:
                item["cross_sectional_pct"] = float((cross_scores <= score).mean())
            results.append(item)
        return {"results": results}

    return application


def score_ticker(
    model, meta: dict[str, Any], source: FeatureSource, ticker: str, date: str | None = None
) -> dict[str, Any]:
    """API olmadan skor uretmek icin yardimci (test/CLI kullanilan)."""
    score, resolved, cross_scores = _score_one(model, meta, source, ticker, date)
    response: dict[str, Any] = {"ticker": ticker, "date": str(resolved.date()), "score": score}
    if cross_scores is not None:
        response["cross_sectional_pct"] = float((cross_scores <= score).mean())
    return response


app = create_app()


if __name__ == "__main__":
    import uvicorn

    with DEFAULT_SERVING_CONFIG.open("r", encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle) or {}
    uvicorn.run(app, host=str(cfg.get("host", "127.0.0.1")), port=int(cfg.get("port", 8000)))
