"""Adim 2 feature boru hattini calistiran giris noktasi.

Calistirma:
    `.venv/Scripts/python src/features/run.py`   (dogrudan script)
    veya  `.venv/Scripts/python -m src.features.run`  gibi.

Akis:
    1. config/features.yaml + benchmark listesi yukle
    2. data/raw/prices.parquet oku
    3. build_dataset (feature + label; sızıntı-guvenli akış)
    4. data/processed/features.parquet'e yaz (klasor yoksa olusturulur)
    5. OZET logging: satir/ticker sayisi, tarih araligi, warm-up'tan dusen
       satir, atilan gun sayisi, gun basina min/ortalama hisse, feature
       basina NaN orani.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):  # dogrudan calistirma
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.build import _feature_columns, build_dataset
from features.config import DEFAULT_FEATURES_CONFIG_PATH, load_features_config
from ingestion.storage import load_prices, save_prices

logger = logging.getLogger(__name__)


def run(config_path: str = DEFAULT_FEATURES_CONFIG_PATH) -> pd.DataFrame:
    """Feature boru hattini calistirir; yazdigi DataFrame'i dondurur."""
    cfg = load_features_config(config_path)
    logger.info("Feature boru hatti basliyor | config=%s", config_path)

    prices = load_prices(cfg.prices_path)
    if prices.empty:
        logger.error("Fiyat deposu bos; once ingestion calistirilmali: %s", cfg.prices_path)
        return pd.DataFrame()

    stats: dict[str, Any] = {}
    dataset = build_dataset(prices, cfg, stats=stats)

    if dataset.empty:
        logger.error("Feature cikti bos; cikti DOSYASI YAZILMADI.")
        return dataset

    # cikti klasoru: yoksa olustur
    output_path = cfg.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_prices(dataset, output_path)

    # --- ozet loglari -------------------------------------------------------
    tickers = dataset["ticker"].nunique()
    date_min = dataset["date"].min()
    date_max = dataset["date"].max()
    logger.info(
        "OZET: %d satir | %d ticker | %s .. %s | benchmark_elenen=%d satir | "
        "warmup_dusen=%d | atilan_gun=%d | gun_basina_min=%.0f hisse | "
        "gun_basina_ort=%.1f hisse | label_dusen=%d_satir",
        len(dataset),
        tickers,
        date_min.date(),
        date_max.date(),
        int(stats.get("benchmark_rows_dropped", 0)),
        int(stats.get("warmup_dropped_rows", 0)),
        int(stats.get("days_dropped", 0)),
        float(stats.get("min_day_tickers", 0)),
        float(stats.get("mean_day_tickers", 0)),
        int(stats.get("label_dropped_rows", 0)),
    )

    # feature basina NaN orani (0 olmali: warm-up + std==0 maskesi sonrasi)
    feature_cols = _feature_columns(cfg)
    nan_ratio = dataset[feature_cols].isna().mean()
    for feature, ratio in nan_ratio.items():
        logger.info("FEATURE %s | satir=%d | NaN orani=%.4f", feature, len(dataset), ratio)

    return dataset


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run()
