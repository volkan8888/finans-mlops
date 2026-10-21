"""Fiyat deposu: tek parquet dosyasi; idempotent merge ile yukleme/kaydetme.

Depodaki kanonik sekil (buyuk/kucuk harf dahil):
    date (datetime64[ns]), ticker (str), open, high, low, close, adj_close
    (float), volume (float).

`merge_prices` idempotenttir: ayni veriyi iki kez birlestirince sonuc
degismez; ayni (date, ticker) cifti geldiginde YENI veri kazanir
(duzeltmis/kaynak guncel kabul edilir).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
KEY_COLUMNS = ["date", "ticker"]
SORT_COLUMNS = ["ticker", "date"]


def empty_prices_frame() -> pd.DataFrame:
    """Bos ama semaya uygun frame."""
    return pd.DataFrame(columns=PRICE_COLUMNS)


def load_prices(path: str | Path) -> pd.DataFrame:
    """Parquet'ten fiyat okur; dosya yoksa semaya uygun bos frame doner."""
    path = Path(path)
    if not path.is_file():
        logger.info("Depoda fiyat yok, bos frame donuyor: %s", path)
        return empty_prices_frame()

    df = pd.read_parquet(path)
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    logger.info("Yuklendi: %d satir <- %s", len(df), path)
    return df


def save_prices(df: pd.DataFrame, path: str | Path) -> Path:
    """Parquet olarak kaydeder; ust klasorler yoksa olusturur."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Kaydedildi: %d satir -> %s", len(df), path)
    return path


def merge_prices(existing: pd.DataFrame | None, new: pd.DataFrame | None) -> pd.DataFrame:
    """Iki fiyat framesini (date, ticker) anahtariyla birlestirir.

    - Cakisma durumunda yeni veri kazanir (keep='last', new sonda).
    - Sonuc (ticker, date) cinsinden siralidir, index 0..n-1.
    - Girdiler oldugu gibi korunur (kopya uzerinde calisir).
    - Idempotenttir: merge(x, y) sonrasi merge(x', y) ayni sonucu verir.
    """
    before = empty_prices_frame() if existing is None else existing
    after = empty_prices_frame() if new is None else new

    parts = [part.copy() for part in (before, after) if not part.empty]
    if not parts:
        return empty_prices_frame()
    combined = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    combined = combined.sort_values(SORT_COLUMNS, kind="mergesort").reset_index(drop=True)

    dropped = len(before) + len(after) - len(combined)
    if dropped:
        logger.debug("Merge: %d tekerrur satir dusturuldu", dropped)
    return combined


def get_last_date(df: pd.DataFrame, ticker: str) -> pd.Timestamp | None:
    """Depoda bu ticker icin kayitli en son tarih; hic veri yoksa None."""
    if df is None or df.empty:
        return None
    dates = df.loc[df["ticker"] == str(ticker), "date"]
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())
