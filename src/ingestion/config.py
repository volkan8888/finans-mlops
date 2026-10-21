"""Yapilandirma yukleme ve universe temizligi.

Konsol: her fonksiyon `logging` modulunu kullanir, `print` kullanilmaz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/data.yaml"
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_PRICES_PATH = "data/raw/prices.parquet"
DEFAULT_BATCH_SIZE = 25


@dataclass(frozen=True)
class Config:
    """data.yaml icerigi; goreli yollar proje kokune gore cozulmustur."""

    universe: list[str]
    start_date: str
    prices_path: Path
    batch_size: int


def load_universe(raw: list[str] | None) -> list[str]:
    """Universe'u temizler: boslugu kirpar, buyuk harfe cevirir, bos
    girdileri ve tekrarlari atar (sira korunur).

    Tempiz hususlarin tamaminda veri icinde en az bir gecerli ticker kalmali.
    """
    universe: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        ticker = str(item).strip().upper()
        if not ticker:
            continue
        if ticker in seen:
            continue
        seen.add(ticker)
        universe.append(ticker)
    if not universe:
        raise ValueError("Universe bos: config'de en az bir gecerli ticker gerekiyor.")
    return universe


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """data.yaml'i yukler.

    Goreli yollar (ör. data/raw/prices.parquet) config dosyasinin bir ust
    dizini (proje koku) esas alinarak mutlak yola cevrilir.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config bulunamadi: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    project_root = config_path.resolve().parent.parent

    universe = load_universe(data.get("universe", []))

    start_date = str(data.get("start_date") or DEFAULT_START_DATE)
    try:
        date_type.fromisoformat(start_date)
    except ValueError as exc:
        raise ValueError(f"Geçersiz start_date: {start_date!r}") from exc

    storage = data.get("storage") or {}
    ingestion = data.get("ingestion") or {}

    prices_path_raw = Path(str(storage.get("prices_path") or DEFAULT_PRICES_PATH))
    prices_path = (
        prices_path_raw if prices_path_raw.is_absolute() else project_root / prices_path_raw
    )

    batch_size = int(ingestion.get("batch_size") or DEFAULT_BATCH_SIZE)
    if batch_size <= 0:
        raise ValueError(f"batch_size pozitif olmali, gelen deger: {batch_size}")

    logger.info(
        "Config yukleniyor: %s | universe=%s | start=%s | prices=%s | batch=%d",
        config_path,
        len(universe),
        start_date,
        prices_path,
        batch_size,
    )
    return Config(
        universe=universe,
        start_date=start_date,
        prices_path=prices_path,
        batch_size=batch_size,
    )
