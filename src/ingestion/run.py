"""Ingestion boru hattini calistiran giris noktasi.

Akis:
    1. config, universe (universe.txt) ve benchmarks yukle
    2. depodaki son tarihe bak, indirme penceresini belirle
       (en eski son tarih + 1 gun; hic veri yoksa config start_date)
    3. universe ve benchmarks'i ayri bloklarda indir; IKISI DE depoya yazilir,
       feature/siralama evrenine yalnizca universe girer
    4. validate_prices ile dogrula (bozuksa HICBIR SEY kaydetmeden dur)
    5. merge_prices + save_prices (merge idempotent)
    6. find_anomalies raporunu logla
    Ozet log: stok/benchmark sayisi, yeni satir sayisi, anomali sayisi.

Idempotentlik: ust uste kosulan ikinci gerceklemede yeni satir eklenmez;
veri birebir ayniysa dosya fiziksel olarak da yeniden yazilmaz.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # dogrudan calistirma: python src/ingestion/run.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.config import DEFAULT_CONFIG_PATH, load_config
from ingestion.download import download_in_batches
from ingestion.storage import (
    empty_prices_frame,
    get_last_date,
    load_prices,
    merge_prices,
    save_prices,
)
from ingestion.validation import find_anomalies, validate_prices

logger = logging.getLogger(__name__)

ANOMALY_LOG_LIMIT = 20


def _next_day(ts: pd.Timestamp) -> str:
    """Depodaki gunden sonraki gun: yeni pencerenin baslangici."""
    return (pd.Timestamp(ts) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def run(config_path: str = DEFAULT_CONFIG_PATH, end: str | None = None) -> None:
    started = time.perf_counter()

    cfg = load_config(config_path)
    universe = cfg.universe      # feature/siralama evreni (stoklar)
    benchmarks = cfg.benchmarks  # depoya yazilan, evrene girmeyen referanslar
    symbols_all = universe + benchmarks

    logger.info(
        "Ingestion basliyor: %d stok + %d benchmark | config=%s",
        len(universe), len(benchmarks), config_path,
    )

    existing = load_prices(cfg.prices_path)
    last_dates = {ticker: get_last_date(existing, ticker) for ticker in symbols_all}
    stored_dates = [ts for ts in last_dates.values() if ts is not None]
    never_stored = [t for t, d in last_dates.items() if d is None]
    stored_tickers = [t for t, d in last_dates.items() if d is not None]
    today = pd.Timestamp.now().normalize()

    def _dl(label: str, tickers: list[str], start: str, window_end: str | None) -> pd.DataFrame:
        """Grup halinde indirir; bos grup icin bos frame doner."""
        if not tickers:
            return empty_prices_frame()
        logger.info(
            "Indirme [%s]: %d sembol | %s .. %s",
            label, len(tickers), start, window_end or "bugun",
        )
        return download_in_batches(
            tickers, start=start, end=window_end, batch_size=cfg.batch_size
        )

    parts: list[pd.DataFrame] = []

    # 1) depoda hic veri olmayan semboller (yeni evren vs.): start_date'ten TAM tarihli
    if never_stored:
        logger.info(
            "Yeni semboller (%d): start_date'ten tam tarihli cekilecek: %s",
            len(never_stored),
            cfg.start_date,
        )
        parts.append(_dl("tam-tarih/yeni", never_stored, cfg.start_date, end))

    # 2) kayitli semboller: artimci pencere (son tarih + 1) veya ayni-gun tazeleme
    if stored_dates:
        anchor = min(stored_dates)
        logger.info(
            "Depo son tarihleri: min=%s max=%s (kayitli %d/%d symbol)",
            anchor.date(),
            max(stored_dates).date(),
            len(stored_dates),
            len(symbols_all),
        )
        if anchor > today:
            logger.info("Depo bugundan ileri tarihte; artimli indirme atlandi.")
        else:
            if anchor >= today:
                # bugunun satirlari tazelenir; degismisler merge keep=last ile guncellenir
                start = anchor.strftime("%Y-%m-%d")
                logger.info("Ayni-gun tazeleme penceresi: %s ..", start)
            else:
                start = _next_day(anchor)
                logger.info("Artimci indirme penceresi: %s .. %s", start, end or "bugun")
            parts.append(_dl("artimci", stored_tickers, start, end))

    if not parts:
        raw = empty_prices_frame()
    elif len(parts) == 1:
        raw = parts[0]
    else:
        raw = pd.concat(parts, ignore_index=True)

    if raw.empty:
        logger.info("Indirme bos dondu (yeni gun yok / kaynak yok); depo degismedi.")
        validated = empty_prices_frame()
    else:
        # bozuk veri gelirse ValueError firlar -> kaydetme adimina hic gelinmez
        validated = validate_prices(raw)

    merged = merge_prices(existing, validated)
    new_rows = len(merged) - len(existing)

    if new_rows > 0:
        save_prices(merged, cfg.prices_path)
    elif merged.equals(existing):
        logger.info("Yeni satir ve degisiklik yok; dosya yazilmadi (idempotent).")
    else:
        save_prices(merged, cfg.prices_path)
        logger.info("Satir sayisi ayni ama degerler guncellendi; dosya yeniden yazildi.")

    anomalies = find_anomalies(merged)
    if len(anomalies):
        logger.warning("Anomali raporu: %d supheli gun", len(anomalies))
        for _, row in anomalies.head(ANOMALY_LOG_LIMIT).iterrows():
            logger.warning("  ANOMALI %s | %s | %s", row["ticker"], row["date"].date(), row["reason"])
        if len(anomalies) > ANOMALY_LOG_LIMIT:
            logger.warning("  ...ve %d tane daha (rapor kisaltildi)", len(anomalies) - ANOMALY_LOG_LIMIT)
    else:
        logger.info("Anomali raporu: temiz, 0 supheli gun.")

    # hic veri gelmeyen semboller (kesif uyarisi)
    if not validated.empty:
        returned = set(validated["ticker"])
        missing = [
            ticker
            for ticker in symbols_all
            if ticker not in returned and get_last_date(existing, ticker) is None
        ]
        if missing:
            logger.warning("Hic veri gelmedi: %s", missing)

    elapsed = time.perf_counter() - started
    logger.info(
        "OZET: %d stok | %d benchmark | %d yeni satir | %d anomali | %.1f sn",
        len(universe),
        len(benchmarks),
        new_rows,
        len(anomalies),
        elapsed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    run()
