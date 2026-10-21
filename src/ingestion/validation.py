"""Fiyat dogrulama (pandera) + anomali (supheli durum) taramasi.

- `validate_prices` SERTTIR: veri bozuksa ValueError firlatir; bozuk veri
  depoya KAYDEDILMEZ. Bozuk satirlar loglanir.
- `find_anomalies` YUMUSAKTIR: hata degil, supheli durumlari rapor verir
  (split/plug/gun koskusu olabilir); veriyi hicbir sekilde degistirmez/sevmez.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandera.pandas as pa

logger = logging.getLogger(__name__)

DEFAULT_MAX_ABS_RETURN = 0.40  # gunluk %40'i asan getiri supheli
DEFAULT_MAX_GAP_BUSINESS_DAYS = 5  # 5 is gununden uzun ticari bosluk supheli

REPORT_COLUMNS = ["ticker", "date", "reason"]
LOG_PREVIEW_LIMIT = 20


def get_price_schema() -> pa.DataFrameSchema:
    """Kanonik fiyat semasi: kolonlar, tipler, araliklar, benzersizlik.

    - open/high/low/close/adj_close > 0
    - volume >= 0
    - high >= low (gun icin)
    - (date, ticker) cifti benzersiz
    """
    positive = pa.Check.greater_than(0)
    return pa.DataFrameSchema(
        columns={
            "date": pa.Column(pa.DateTime),
            "ticker": pa.Column(str),
            "open": pa.Column(float, positive),
            "high": pa.Column(float, positive),
            "low": pa.Column(float, positive),
            "close": pa.Column(float, positive),
            "adj_close": pa.Column(float, positive),
            "volume": pa.Column(float, pa.Check.ge(0)),
        },
        checks=[
            # kolonlar arasi kontrol: gun icinde high en az low kadar
            pa.Check(lambda df: df["high"] >= df["low"], error="high < low"),
        ],
        unique=["date", "ticker"],
        strict=True,
        coerce=True,
    )


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Semayi uygular; gecerliyse (coerce edilmis) frame dondurur.

    Hata varsa nerede bozuk oldugu loglanir ve ValueError firlatilir;
    bozuk veri asla depoya kaydedilmez.
    """
    if df is None or df.empty:
        raise ValueError("Dogrulama icin veri bos.")

    try:
        validated = get_price_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases
        logger.error("Dogrulama basarisiz: %d hata (toplam %d satir)", len(failures), len(df))
        for _, row in failures.head(LOG_PREVIEW_LIMIT).iterrows():
            logger.error(
                "  | %s | kolon=%s | kontrol=%s | deger=%r",
                row.get("failure_case"),
                row.get("column"),
                row.get("check"),
                row.get("failure_case"),
            )
        raise ValueError(
            f"Fiyat verisi semayi gecemedi: {len(failures)} hata. Ayrinti loglarda."
        ) from exc

    return validated


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(columns=REPORT_COLUMNS)


def find_anomalies(
    df: pd.DataFrame,
    max_abs_return: float = DEFAULT_MAX_ABS_RETURN,
    max_gap_business_days: int = DEFAULT_MAX_GAP_BUSINESS_DAYS,
) -> pd.DataFrame:
    """Supheli gunleri raporlar (veriyi degistirmez, SILMEZ).

    Raporlar (kolonlar: ticker, date, reason):
    - gunluk |getiri| > max_abs_return  (split/veri hatasi olabilir)
    - volume == 0 olan gunler
    - bir ticker'da max_gap_business_days'ten uzun ticari gun boslugu
    """
    if df is None or df.empty:
        return _empty_report()

    data = df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    reports: list[pd.DataFrame] = []

    # 1) |getiri| eşiği: close[t] / close[t-1] - 1 (ayni ticker icinde)
    prev_close = data.groupby("ticker", sort=False)["close"].shift(1)
    ret = data["close"] / prev_close - 1.0
    jump_mask = ret.abs() > max_abs_return
    if jump_mask.any():
        jump = data.loc[jump_mask, ["ticker", "date"]].copy()
        jump["reason"] = [
            f"large_abs_return({value:+.1%})" for value in ret.loc[jump_mask]
        ]
        reports.append(jump)

    # 2) sifir hacim
    zero_mask = data["volume"] == 0
    if zero_mask.any():
        zero = data.loc[zero_mask, ["ticker", "date"]].copy()
        zero["reason"] = "zero_volume"
        reports.append(zero)

    # 3) uzun bosluk: bu kayit, gectigi kayitla arasinda cok fazla is gunu sayisi
    # Rapordaki tarih = bosluktan SONRAKI gun (boslugu gozeten satir).
    prev_date = data.groupby("ticker", sort=False)["date"].shift(1)
    has_prev = prev_date.notna() & data["date"].notna()
    if has_prev.any():
        start_days = prev_date.loc[has_prev].to_numpy().astype("M8[D]")
        end_days = data.loc[has_prev, "date"].to_numpy().astype("M8[D]")
        # busday_count [start, end) sayar; d2 dahil degil -> aradaki = sayim - 1
        missing_bdays = np.busday_count(start_days, end_days) - 1
        gap_mask = missing_bdays > max_gap_business_days
        if gap_mask.any():
            gap = data.loc[has_prev, ["ticker", "date"]].copy()
            gap = gap.loc[gap_mask].copy()
            gap["reason"] = [
                f"gap_{int(days)}_business_days" for days in missing_bdays[gap_mask]
            ]
            reports.append(gap)

    if not reports:
        return _empty_report()
    out = pd.concat(reports, ignore_index=True)
    out = out.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    logger.info("Anomali tarama: %d supheli gun bulundu", len(out))
    return out
