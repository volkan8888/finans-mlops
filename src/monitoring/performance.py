"""Adim 7 (FAZ 5): performans takibi - gunluk ve kayan IC.

- ic_series: training.evaluate.daily_ic ile gun bazinda spearman.
- latest_rolling_ic: son `window` gunde tamamlanan kayan ortalama IC.
  Not: label, horizon yapisinin gecikmesiyle gerceklesir; bu yuzden IC
  yalnizca label'i hazir olan gunler uzerinden hesaplanir (look-ahead yok).
"""

from __future__ import annotations

import pandas as pd

from training.evaluate import DEFAULT_NW_LAG, daily_ic


def ic_series(predictions: pd.DataFrame, score_col: str = "score", label_col: str = "label") -> pd.Series:
    """Gun bazinda IC serisi (date index'li)."""
    return daily_ic(predictions, score_col, label_col, min_n=2)


def latest_rolling_ic(predictions: pd.DataFrame, window: int = 20) -> float:
    """Tamamlanan son `window` gunlik kayan ortalama IC; yetersiz veri NaN."""
    series = ic_series(predictions)
    if series.empty:
        return float("nan")
    rolling = series.rolling(window=window, min_periods=window).mean()
    if rolling.notna().sum() == 0:
        return float("nan")
    return float(rolling.iloc[-1])


def overall_ic(predictions: pd.DataFrame, nw_lag: int = DEFAULT_NW_LAG) -> dict[str, float]:
    """Tum serinin IC ozeti (mean/icir/t_stat/t_nw/positive_share/n_days)."""
    from training.evaluate import ic_summary

    series = ic_series(predictions)
    return ic_summary(series, nw_lag=nw_lag)


def coverage(predictions: pd.DataFrame) -> dict[str, float]:
    """Skor akisinin kac gun kapsadigi ve gun basi ortalama ticker sayisi."""
    grouped = predictions.groupby("date")["ticker"].count()
    return {
        "n_days_covered": len(grouped),
        "avg_tickers_per_day": float(grouped.mean()) if len(grouped) else float("nan"),
        "first_date": str(predictions["date"].min().date()),
        "last_date": str(predictions["date"].max().date()),
    }
