"""Feature tablosu kurulumu: ham feature -> warm-up -> min_tickers -> winsorize -> zscore.

SIZINTI GARANTILERI:
- shift(-n) YALNIZCA labels.py'de: her ticker'in akisi technical.py'deki
  gecmise bakan fonksiyonlardan gecer; gelecege bakan tek islem, label icin
  kullanilan forward_return'dur.
- Ticker karismaz: tum rolling/shift islemleri groupby("ticker") gruplarinin
  icinde, kronolojik sirali seriler uzerinde yapilir.
- Cross-sectional islemler ayni gunun verisiyle groupby("date") icinde yapilir;
  SIRA: ham feature -> winsorize -> zscore.

TRAINING-SERVING SKEW ONLEMI:
    build_features (serving) ve build_dataset (egitim) ayni `_build_core`
    akisini cagirir; tek fark, dataset'te kolonlarin eklenmesi ve label'i NaN
    olan satirlarin dusrulmesidir. Kod kopyasi yok, hesap sirasi aynidir.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from features.config import FeaturesConfig
from features.cross_section import winsorize_by_date, zscore_by_date
from features.labels import forward_return, rank_label_by_date
from features.technical import (
    compute_returns,
    momentum,
    momentum_skip,
    rsi,
    short_term_reversal,
    volatility,
    volume_change,
)

logger = logging.getLogger(__name__)

GROUP_DATE = "date"
ID_COLUMNS = ["date", "ticker"]
LABEL_COLUMNS = ["forward_return", "label"]


def _feature_columns(cfg: FeaturesConfig) -> list[str]:
    """cfg'den feature kolon adlarini uretir (SIRA SABITTIR; testlere dayanir)."""
    cols: list[str] = [f"momentum_{w}" for w in cfg.momentum_windows]
    cols.append(f"momentum_skip_{cfg.skip_window}_{cfg.skip}")
    cols.append(f"volatility_{cfg.volatility_window}")
    cols.append(f"rsi_{cfg.rsi_window}")
    cols.append(f"volume_change_{cfg.volume_window}")
    cols.append(f"reversal_{cfg.reversal_window}")
    return cols


def _validate_input(prices: pd.DataFrame) -> None:
    """Gerekli kolonlarin var oldugunu ve panelin bos olmadigini kontrol eder."""
    required = ["date", "ticker", "adj_close", "volume"]
    if prices.empty:
        raise ValueError("Fiyat paneli bos; feature uretilemez.")
    missing = [c for c in required if c not in prices.columns]
    if missing:
        raise ValueError(f"Fiyat panelinde kolon yok: {missing} (gerekli: {required})")


def _empty_output(cfg: FeaturesConfig, with_labels: bool) -> pd.DataFrame:
    """Bos ama semaya uygun cikti tablosu (kolon sirasi dahil)."""
    columns = ID_COLUMNS + _feature_columns(cfg)
    if with_labels:
        columns = columns + LABEL_COLUMNS
    return pd.DataFrame(columns=columns)


def _per_ticker_raw(
    stocks: pd.DataFrame, cfg: FeaturesConfig, with_labels: bool
) -> pd.DataFrame:
    """Her ticker'in serisini tek tek technical.py fonksiyonlarindan gecirir.

    Kurallar:
    - `stocks` zaten (ticker, date) sirali olmali (kronolojik gereklilik).
    - Hisseler KARISMAZ: her teknik islem TEK BIR serinin uzerinde yapilir;
      gruplara ayirma burada yapilir, fonksiyonlara tek tek seri verilir.
    """
    pieces: list[pd.DataFrame] = []
    for ticker, group in stocks.groupby("ticker", sort=False):
        adj_close = group["adj_close"]
        returns = compute_returns(adj_close)
        extra: dict[str, pd.Series] = {}
        # Her ham feature technical.py'den gelir; ticker izolasyonu
        # groupby("ticker") + kronolojik siralamayla saglanir.
        for window in cfg.momentum_windows:
            extra[f"momentum_{window}"] = momentum(adj_close, window)
        extra[f"momentum_skip_{cfg.skip_window}_{cfg.skip}"] = momentum_skip(
            adj_close, cfg.skip_window, cfg.skip
        )
        extra[f"volatility_{cfg.volatility_window}"] = volatility(
            returns, cfg.volatility_window
        )
        extra[f"rsi_{cfg.rsi_window}"] = rsi(adj_close, cfg.rsi_window)
        extra[f"volume_change_{cfg.volume_window}"] = volume_change(
            group["volume"], cfg.volume_window
        )
        extra[f"reversal_{cfg.reversal_window}"] = short_term_reversal(
            returns, cfg.reversal_window
        )
        if with_labels:
            # shift(-horizon) SADECE labels.py'de; feature'larda sızıntı yok.
            extra["forward_return"] = forward_return(adj_close, cfg.label_horizon)
        pieces.append(group.assign(**extra))
    if not pieces:
        return _empty_output(cfg, with_labels)
    return pd.concat(pieces, ignore_index=True)


def _build_core(
    prices: pd.DataFrame,
    cfg: FeaturesConfig,
    with_labels: bool,
    stats: dict[str, Any] | None,
) -> pd.DataFrame:
    """Tek boru hatti; build_features/build_dataset'i ayiran tek fark with_labels.

    Adimlar:
        1. benchmark filtre (cfg.benchmarks feature evrenine girmez)
        2. (ticker, date) kronolojik siralama
        3. ticker bazli ham feature
        4. warm-up: feature kolonlarinda NaN olan satirlar dusurulur (loglanir)
        5. gunde az hisse varsa gun atilir (loglanir)
        6. winsorize (gun icinde)  -> 7. zscore (gun icinde)
        8. dataset modunda: label = rank(forward_return); NaN label dusulur
    `stats` verildiyse olcum sayilari icine doldurulur (run.py ozeti icin).
    """
    counters: dict[str, Any] = {}
    _validate_input(prices)
    feature_cols = _feature_columns(cfg)

    # 1) benchmark filtre
    stocks = prices.loc[~prices["ticker"].isin(set(cfg.benchmarks))].copy()
    bench_rows = len(prices) - len(stocks)
    counters["benchmark_rows_dropped"] = bench_rows
    if bench_rows:
        seen_bench = sorted(set(prices["ticker"]) & set(cfg.benchmarks))
        logger.info("Benchmark satirlari elendi: %d satir (%s)", bench_rows, seen_bench)

    # 2) kronolojik siralama: her ticker kendi icinde date artan olmali
    stocks = stocks.sort_values(["ticker", "date"], kind="mergesort")

    # 3) ham ticker-bazli feature'lar (+ opsiyonel ham forward_return)
    frame = _per_ticker_raw(stocks, cfg, with_labels)
    if frame.empty:
        logger.warning("Stok satiri yok; bos feature cikti donduruluyor.")
        empty = _empty_output(cfg, with_labels)
        counters.update(
            warmup_dropped_rows=0,
            days_dropped=0,
            min_day_tickers=0,
            mean_day_tickers=0.0,
            label_dropped_rows=0,
        )
        if stats is not None:
            stats.update(counters)
        return empty

    # 4) warm-up ve eksik veri: NaN doldurma YOK; dusur + logla.
    # Neden: ffill/bfill yapay veri uretir; sahte degerler, modele gecmis
    # gunlerde olmayan sinyalleri hatirlatir. Dusurme sayisi loglanir
    # (sessiz kayip olmasin).
    before_warmup = len(frame)
    frame = frame.dropna(subset=feature_cols, how="any")
    warmup_dropped = before_warmup - len(frame)
    counters["warmup_dropped_rows"] = warmup_dropped
    if warmup_dropped:
        logger.info(
            "Warm-up/eksik feature satirlari dusuruldu: %d "
            "(pencere donemi + sifir hacim gunler)",
            warmup_dropped,
        )

    # 5) gun basina minimum hisse: az hisseli gun cross-section bilgi icin
    # istatistiksel olarak zayiftir; bu gunler atilir ve loglanir.
    sizes = frame.groupby(GROUP_DATE, sort=False)["ticker"].size()
    keep_days = sizes[sizes >= cfg.min_tickers_per_date].index
    days_dropped = int(len(sizes) - len(keep_days))
    counters["days_dropped"] = days_dropped
    if days_dropped:
        logger.info(
            "Az hisseli gunler atildi: %d gun (esik=%d)",
            days_dropped,
            cfg.min_tickers_per_date,
        )
    frame = frame.loc[frame[GROUP_DATE].isin(keep_days)]

    # 6-7) cross-sectional: winsorize -> zscore (sira ONEMLI: winsorize once
    # gelir, aksi halde gundeki tek abarti deger tum gunun z-score'unu bozar)
    frame = winsorize_by_date(
        frame, feature_cols, cfg.winsorize_lower, cfg.winsorize_upper
    )
    frame = zscore_by_date(frame, feature_cols)

    # 8) dataset modu: cross-sectional rank label + label'siz satirlar dusulur
    label_dropped = 0
    if with_labels:
        frame["label"] = rank_label_by_date(frame, "forward_return")
        before_label = len(frame)
        frame = frame.dropna(subset=["label"])  # son horizon gunleri
        label_dropped = before_label - len(frame)
        counters["label_dropped_rows"] = label_dropped
        if label_dropped:
            logger.info(
                "Label'siz satirlar dusuruldu: %d (her ticker'in son %d gunu)",
                label_dropped, cfg.label_horizon,
            )

    # 9) nihai siralama ve kolon secimi
    out = frame.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    columns = ID_COLUMNS + feature_cols
    if with_labels:
        columns = columns + LABEL_COLUMNS
    out = out[columns]

    final_sizes = out.groupby(GROUP_DATE, sort=False)["ticker"].size()
    counters["active_days"] = len(final_sizes)
    counters["min_day_tickers"] = int(final_sizes.min()) if len(final_sizes) else 0
    counters["mean_day_tickers"] = float(final_sizes.mean()) if len(final_sizes) else 0.0

    if stats is not None:
        stats.update(counters)
    return out


def build_features(prices: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Feature tablosu — LABEL ICERMEZ (serving yolu).

    serving bugunu tahmin eder: feature'lar yalnizca gecmise bakan
    hesaplardan gelir; her ticker'in son horizon satirlari DAHILDIR (bu
    satirlarda feature dolu ama label yoktur).
    """
    return _build_core(prices, cfg, with_labels=False, stats=None)


def build_dataset(
    prices: pd.DataFrame,
    cfg: FeaturesConfig,
    stats: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Egitim kumesi: build_features ile AYNI feature akisi + label.

    - "forward_return": ham (rank olmayan) ileri getiri — backtest icin.
    - "label": forward_return'in gun ici cross-sectional rank'i (0-1).
    Label'i NaN olan satirlar (her ticker'in son horizon gunu) dusurulur ve
    bu satir sayisi loglanir.
    """
    return _build_core(prices, cfg, with_labels=True, stats=stats)
