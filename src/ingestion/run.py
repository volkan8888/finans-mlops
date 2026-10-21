"""Ingestion boru hattini calistiran giris noktasi.

Akis:
    1. config ve universe yukle
    2. depodaki son tarihe bak, indirme penceresini belirle
       (en eski son tarih + 1 gun; hic veri yoksa config start_date)
    3. download_in_batches ile indir
    4. validate_prices ile dogrula (bozuksa HICBIR SEY kaydetmeden dur)
    5. merge_prices + save_prices (merge idempotent; ayni (date,ticker) birden
       iki kez eklenmez)
    6. find_anomalies raporunu logla
    Ozet log: ticker sayisi, yeni satir sayisi, anomali sayisi.

Idempotentlik: arka arkaya iki kez calistirildiginda ikinci koşuda
yeni satir eklenmez (merge cakismada yeni veriyi tutar ama tip olarak
ayni gundar gelirse satir sayisi degismez).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # dogrudan calistirma: python src/ingestion/run.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.config import DEFAULT_CONFIG_PATH, load_config, load_universe  # noqa: E402
from ingestion.download import download_in_batches  # noqa: E402
from ingestion.storage import (  # noqa: E402
    empty_prices_frame,
    get_last_date,
    load_prices,
    merge_prices,
    save_prices,
)
from ingestion.validation import find_anomalies, validate_prices  # noqa: E402

logger = logging.getLogger(__name__)

ANOMALY_LOG_LIMIT = 20


def _next_day(ts: pd.Timestamp) -> str:
    """Depodaki gunden sonraki gun: yeni pencerenin baslangici."""
    return (pd.Timestamp(ts) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def run(config_path: str = DEFAULT_CONFIG_PATH, end: str | None = None) -> None:
    started = time.perf_counter()

    cfg = load_config(config_path)
    universe = load_universe(cfg.universe)
    logger.info("Ingestion basliyor: %d ticker | config=%s", len(universe), config_path)

    existing = load_prices(cfg.prices_path)
    last_dates = {ticker: get_last_date(existing, ticker) for ticker in universe}
    stored = [ts for ts in last_dates.values() if ts is not None]

    if stored:
        anchor = min(stored)
        today = pd.Timestamp.now().normalize()
        logger.info(
            "Depodaki son tarihler: %s",
            {t: str(d.date()) for t, d in last_dates.items() if d is not None},
        )
        if anchor > today:
            logger.info("Depo bugundan ileri tarihte; indirme atlandi.")
            raw = empty_prices_frame()
        elif anchor >= today:
            # bugunun satirlari yeniden cekilir; gun ici parcal bar'lar, EOD
            # degerleriyle degistiyse merge keep=last ile guncellenir.
            start = anchor.strftime("%Y-%m-%d")
            logger.info("Ayni-gun tazeleme penceresi: %s ..", start)
            raw = download_in_batches(
                universe, start=start, end=end, batch_size=cfg.batch_size
            )
        else:
            start = _next_day(anchor)
            logger.info("Indirme penceresi: %s .. %s", start, end or "bugun")
            raw = download_in_batches(
                universe, start=start, end=end, batch_size=cfg.batch_size
            )
    else:
        start = cfg.start_date
        logger.info("Depo bos; config start_date kullaniliyor: %s ..", start)
        raw = download_in_batches(
            universe, start=start, end=end, batch_size=cfg.batch_size
        )

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

    # kaynakta hic veri gelmeyen ticker'lerde imza (kesif etiket)
    if not validated.empty:
        returned = set(validated["ticker"])
        missing = [
            ticker
            for ticker in universe
            if ticker not in returned and get_last_date(existing, ticker) is None
        ]
        if missing:
            logger.warning("Hic veri gelmedi: %s", missing)

    elapsed = time.perf_counter() - started
    logger.info(
        "OZET: %d ticker | %d yeni satir | %d anomali | %.1f sn",
        len(universe),
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
