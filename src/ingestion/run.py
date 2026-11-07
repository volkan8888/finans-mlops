"""Ingestion boru hattini calistiran giris noktasi.

Akis:
    1. config, universe (universe.txt) ve benchmarks yukle
    2. depodaki son tarihe bak, indirme penceresini belirle
       (en eski son tarih + 1 gun; hic veri yoksa config start_date)
    3. universe ve benchmarks'i ayri bloklarda indir; IKISI DE depoya yazilir,
       feature/siralama evrenine yalnizca universe girer
    4. son-bar temizligi: en yeni tarihteki NaN close/adj_close satirlari
       atilir (istisna yalnizca son tarih; ayrinti drop_incomplete_latest_rows)
    5. validate_prices ile dogrula (gecmis NaN dahil bozuksa HICBIR SEY
       kaydetmeden dur)
    6. merge_prices + save_prices (merge idempotent)
    7. find_anomalies raporunu logla
    Ozet log: stok/benchmark sayisi, yeni satir sayisi, anomali sayisi.

Idempotentlik: ust uste kosulan ikinci gerceklemede yeni satir eklenmez;
veri birebir ayniysa dosya fiziksel olarak da yeniden yazilmaz.

Cikis kodlari (`python -m ingestion.run`):
    0 = tamam | 1 = ag/indirme/calisma hatasi (tekrar denenebilir)
    | 2 = dogrulama/veri/config hatasi (tekrar deneme anlamsiz).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # dogrudan calistirma: python src/ingestion/run.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MAX_INCOMPLETE_LATEST_SHARE,
    load_config,
)
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


def drop_incomplete_latest_rows(
    panel: pd.DataFrame,
    universe: list[str],
    benchmarks: list[str],
    max_share: float = DEFAULT_MAX_INCOMPLETE_LATEST_SHARE,
) -> pd.DataFrame:
    """En yeni tarihteki (son bar) NaN close/adj_close satirlarini temizler.

    Istisna YALNIZCA son tarih icindir: yfinance bazen son barin
    close/adj_close'unu gecici olarak NaN verebilir (kapanis verisi gec
    yayiliyor). Gecmis tarihlerden NaN'li satir degistirilmez; validate_prices
    onlar icin aynen SERT hata verir (veri uydurulmaz).

    Kurallar:
    - Hisse (benchmark olmayan) satirinin close/adj_close'u NaN ise o satir
      atilir + WARNING (ticker listesi ile).
    - Eksik hisse sayisi evrenin max_share oranini ASARSA o tarih KOMPLE
      atilir (hisseler cogunlukla NaN ise gun fazla bozuk kabul edilir).
    - Benchmark satiri NaN ise yalnizca o satir atilir; eksik sayaca
      katilmaz, tarih komple atma tetiklenmez.

    Panel DataFrame (date,ticker,...) kolonlari; donecek de temiz panel.
    """
    if panel is None or panel.empty or "date" not in panel.columns:
        return panel
    son_date = panel["date"].max()
    if pd.isna(son_date):
        return panel

    son_mask = panel["date"] == son_date
    nan_mask = son_mask & (panel["close"].isna() | panel["adj_close"].isna())
    if not nan_mask.any():
        return panel

    benchmark_set = {str(item).upper() for item in benchmarks}
    eksik = panel.loc[nan_mask]
    is_benchmark = eksik["ticker"].isin(benchmark_set)
    hisse_sen = list(eksik.loc[~is_benchmark, "ticker"])
    benchmark_sen = list(eksik.loc[is_benchmark, "ticker"])
    universe_toplam = len(universe)
    share = len(hisse_sen) / universe_toplam if universe_toplam else 0.0
    tarih_str = str(pd.Timestamp(son_date).date())

    logger.warning(
        "Son bar %s: %d satirda close/adj_close NaN | hisse=%d | benchmark=%d",
        tarih_str,
        len(eksik),
        len(hisse_sen),
        len(benchmark_sen),
    )
    if share > max_share:
        logger.warning(
            "Son bar %s KOMPLE atiliyor: eksik hisse %d/%d (oran=%.3f > esik=%.3f) "
            "| benchmark atilan: %s | hisse atilan tam liste: %s",
            tarih_str,
            len(hisse_sen),
            universe_toplam,
            share,
            max_share,
            benchmark_sen,
            hisse_sen,
        )
        return panel.loc[~son_mask].reset_index(drop=True)

    logger.warning(
        "Son bar %s icin %d NaN satir atildi (hisse=%s | benchmark=%s); "
        "guncel barin eksik hisseleri sonraki kosularda tazeleme ile kendiliginden duzelir.",
        tarih_str,
        len(eksik),
        hisse_sen,
        benchmark_sen,
    )
    return panel.loc[~nan_mask].reset_index(drop=True)


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

    if not raw.empty:
        # istisna yalnizca SON tarih: eksik (NaN) son bar satirlari temizlenir;
        # gecmis tarihler validate_prices'a aynen sorgulamaya girer.
        raw = drop_incomplete_latest_rows(
            raw, universe, benchmarks, cfg.max_incomplete_latest_share
        )

    if raw.empty:
        logger.info("Indirme bos dondu (yeni gun yok / kaynak yok / son tarih atildi); depo degismedi.")
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


def main() -> int:
    """CLI girisi: cikis kodlari eylem ipucu tasir.

    0 = tamam | 1 = ag/indirme/beklenmeyen calisma hatasi (tekrar deneme
    mantikli) | 2 = dogrulama/veri/config hatasi (ayni hata tekrar uretilir;
    tekrar deneme anlamsiz).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run()
    except ValueError as exc:
        logger.error("Kalici veri/dogrulama hatasi (tekrar deneme anlamsiz): %s", exc)
        return 2
    except Exception:
        logger.exception("Beklenmeyen calisma hatasi (tekrar denenebilir)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
