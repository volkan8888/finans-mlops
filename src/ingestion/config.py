"""Yapilandirma yukleme (data.yaml) ve universe temizligi (universe.txt).

Adim 2 degisiklikleri:
- universe: `universe_path` ile universe.txt'ten okunur (geriye uyum: yaml'da
  inline `universe` listesi hala calisir).
- benchmarks: depoya yazilir ama feature/siralama evrenine girmez; cakisma
  varsa benchmarklar evrenden dusurulur.
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
# Son tarihte eksik (NaN close/adj_close) bar icin esik: evrenin bu oranindan
# fazla hisse eksikse o tarih KOMPLE atilir (ingestion/run.py temizligi).
DEFAULT_MAX_INCOMPLETE_LATEST_SHARE = 0.10


@dataclass(frozen=True)
class Config:
    """data.yaml + universe.txt icerigi; goreli yollar proje kokune gore cozulmustur.

    universe:   feature/siralama evrenindeki hisseler (benchmark degil).
    benchmarks: depoya yazilan, feature evrenine girmeyen referans veriler
                (orn. ^GSPC, SPY).
    max_incomplete_latest_share: son tarihte NaN bar istisnasi esigi
                (evren yuzdesi asilinca tarih komple atilir).
    """

    universe: list[str]
    benchmarks: list[str]
    start_date: str
    prices_path: Path
    batch_size: int
    max_incomplete_latest_share: float


def _clean_tickers(raw: list[str] | None) -> list[str]:
    """Trim, buyuk harf, bos/tekrar girdileri at (sira korunur)."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        ticker = str(item).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        cleaned.append(ticker)
    return cleaned


def load_universe(raw: list[str] | None) -> list[str]:
    """Temizlenmis universe; hicbir gecerli ticker kalmazsa ValueError."""
    universe = _clean_tickers(raw)
    if not universe:
        raise ValueError("Universe bos: en az bir gecerli ticker gerekiyor.")
    return universe


def load_universe_file(path: str | Path) -> list[str]:
    """universe.txt: her satira bir ticker; '#' yorumlu ve bos satirlar yok sayilir.

    Temizleme load_universe ile ayni yerde yapilir (trim/upper/dedupe).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Universe dosyasi bulunamadi: {file_path}")

    lines: list[str] = []
    with file_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            content = line.split("#", 1)[0].strip()
            if content:
                lines.append(content)
    return load_universe(lines)


def _resolve(project_root: Path, raw: str | Path) -> Path:
    """Goreli yolu proje kokune (config'in bir ust dizini) uyar."""
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """data.yaml + universe.txt'i yukler.

    - universe: `universe_path` (txt) birinci kaynak; txt yoksa yaml'daki
      inline `universe` listesi kullanilir (test/geriye uyum).
    - benchmarks: yaml'dan okunur; evrene girmesine izin verilmez.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config bulunamadi: {config_path}")

    with config_path.open("r", encoding="utf-8-sig") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}

    project_root = config_path.resolve().parent.parent

    universe_path_raw = data.get("universe_path")
    if universe_path_raw:
        universe_file = _resolve(project_root, universe_path_raw)
        universe = load_universe_file(universe_file)
        universe_source = str(universe_path_raw)
    elif data.get("universe"):
        universe = load_universe(data.get("universe"))
        universe_source = "data.yaml (inline)"
        logger.info("universe_path yok; data.yaml icindeki universe listesi kullaniliyor.")
    else:
        raise ValueError(
            "Evren gerekli: 'universe_path' (universe.txt tanimli olmali) "
            "veya data.yaml'da inline 'universe'."
        )

    benchmarks = _clean_tickers(data.get("benchmarks", []))
    overlap = set(universe) & set(benchmarks)
    if overlap:
        universe = [t for t in universe if t not in overlap]
        logger.warning(
            "Benchmarklar evrene sizmis; evrenden dusuruldu: %s", sorted(overlap)
        )

    start_date = str(data.get("start_date") or DEFAULT_START_DATE)
    try:
        date_type.fromisoformat(start_date)
    except ValueError as exc:
        raise ValueError(f"Gecersiz start_date: {start_date!r}") from exc

    storage = data.get("storage") or {}
    ingestion = data.get("ingestion") or {}

    prices_path = _resolve(project_root, storage.get("prices_path") or DEFAULT_PRICES_PATH)

    batch_size = int(ingestion.get("batch_size") or DEFAULT_BATCH_SIZE)
    if batch_size <= 0:
        raise ValueError(f"batch_size pozitif olmali, gelen deger: {batch_size}")

    share_raw = ingestion.get("max_incomplete_latest_share")
    max_share = float(share_raw) if share_raw is not None else DEFAULT_MAX_INCOMPLETE_LATEST_SHARE
    if not 0.0 <= max_share <= 1.0:
        raise ValueError(
            f"max_incomplete_latest_share [0,1] araliginda olmali, gelen deger: {max_share}"
        )

    logger.info(
        "Config yukleniyor: %s | universe=%d (%s) | benchmark=%d | start=%s | prices=%s | "
        "batch=%d | son-bar-esik=%.2f",
        config_path,
        len(universe),
        universe_source,
        len(benchmarks),
        start_date,
        prices_path,
        batch_size,
        max_share,
    )
    return Config(
        universe=universe,
        benchmarks=benchmarks,
        start_date=start_date,
        prices_path=prices_path,
        batch_size=batch_size,
        max_incomplete_latest_share=max_share,
    )
