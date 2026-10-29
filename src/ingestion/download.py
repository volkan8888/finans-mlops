"""yfinance tabanli toplu indirme + stabil kolon normalizasyonu.

Cikti (long format) her zaman su kolonlari tasir:
    date, ticker, open, high, low, close, adj_close, volume

- `Adj Close` yoksa (orn. eski kaynaklar) `close` kopyalanir.
- Tarihler tz'siz datetime64'e cevrilir.
- Bir ticker'in bir gunu icinde tek satir kalir (date uzerinden tekrar dusurulur).

Not: bu moduldeki fonksiyonlar ag kullaniyor; testler cagirmaz.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import yfinance as yf

from ingestion.storage import PRICE_COLUMNS

logger = logging.getLogger(__name__)

# Kolonlari kanonik isimlere cevirir ("Adj Close" -> "adj_close").
_COLUMN_RENAME: dict[str, str] = {
    "Date": "date",
    "Datetime": "date",
    "index": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

CHUNK_SLEEP_SECONDS = 0.5  # Yahoo rate-limit nazikligi icin blok arasi bekleme


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Coklu-ticker yfinance ciktisindan bu ticker'in kesiti.

    group_by='column' cikti MultiIndex(field, ticker) icin level=-1;
    bazi surumler ters sirada oldugundan level=0 da denenir.
    Ticker bulunamazsa None.
    """
    if isinstance(df.columns, pd.MultiIndex):
        for level in (-1, 0):
            try:
                slice_df = df.xs(ticker, axis=1, level=level)
            except KeyError:
                continue
            if slice_df is not None and not slice_df.empty:
                return slice_df
        return None
    return df


def _normalize(df: pd.DataFrame | None, ticker: str) -> pd.DataFrame:
    """Tek ticker'in ham ciktisini kanonik kolonlara getirir."""
    if df is None or df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    slice_df = _flatten_columns(df, ticker)
    if slice_df is None:
        logger.warning("yfinance ciktisinda %s yok, bos donuyor", ticker)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    out = slice_df.copy().rename(columns=_COLUMN_RENAME)
    out = out.reset_index()
    # reset_index'ten sonra ilk kolon index'ten gelir; adini 'date' yap
    out = out.rename(columns={out.columns[0]: "date"})

    if "adj_close" not in out.columns:
        if "close" not in out.columns:
            logger.warning("%s: close/adj_close yok, bos donuyor", ticker)
            return pd.DataFrame(columns=PRICE_COLUMNS)
        out["adj_close"] = out["close"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if isinstance(out["date"].dtype, pd.DatetimeTZDtype):
        out["date"] = out["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            logger.warning("%s: beklenen kolon yok, NaN dolduruluyor: %s", ticker, col)
            out[col] = float("nan")

    out = out.dropna(subset=["date"])                      # tarih bozuksa satirsiz
    out = out.dropna(subset=numeric_cols, how="all")       # tum fiyatlar bos gün
    out = out.drop_duplicates(subset=["date"], keep="last")

    out.insert(1, "ticker", ticker)
    out = out.sort_values("date", kind="mergesort").reset_index(drop=True)
    return out[PRICE_COLUMNS]


def fetch_symbol(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Tek ticker indirir; hata/veri yoksa bos frame doner (loglar)."""
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001 - ag/kaynak hatalari akisi oldurmuyor
        logger.error("Indirme hatasi %s (start=%s): %s", ticker, start, exc)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    normalized = _normalize(raw, ticker)
    if normalized.empty:
        logger.warning("Veri yok: %s (%s..%s)", ticker, start, end or "bugun")
        return normalized
    logger.info(
        "Indirildi: %-6s %4d satir (ilk=%s son=%s)",
        ticker,
        len(normalized),
        normalized["date"].iloc[0].date(),
        normalized["date"].iloc[-1].date(),
    )
    return normalized


def download_in_batches(
    tickers: list[str],
    start: str,
    end: str | None = None,
    batch_size: int = 25,
) -> pd.DataFrame:
    """Ticker'ari blok halinde indirir; tek long frame dondurur.

    Blok cagrisi tamamen boş dönerse/hata verirse blok tickers tek tek
    `fetch_symbol` ile tekrar denenir.
    """
    tickers = [str(t) for t in tickers]
    batch_size = max(1, int(batch_size))
    parts: list[pd.DataFrame] = []

    for offset in range(0, len(tickers), batch_size):
        chunk = tickers[offset : offset + batch_size]
        logger.info("Blok %02d: %d ticker... %s", offset // batch_size + 1, len(chunk), chunk)
        try:
            raw = yf.download(
                chunk,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception as exc:  # noqa: BLE001 - blok hatasi tek-tekleme yedegine duser
            logger.error("Blok indirme hatasi %s: %s - tek tek deneniyor", chunk, exc)
            raw = None

        if raw is not None and not raw.empty:
            for ticker in chunk:
                parts.append(_normalize(raw, ticker))
        else:
            # yfinance hatayi bazen sessizce bos frame olarak dondurur
            for ticker in chunk:
                parts.append(fetch_symbol(ticker, start=start, end=end))

        if offset + batch_size < len(tickers):
            time.sleep(CHUNK_SLEEP_SECONDS)

    if not parts:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)
    per_ticker = combined.groupby("ticker").size().to_dict()
    logger.info("Toplam indirme: %d satir | ticker basina: %s", len(combined), per_ticker)
    return combined
