"""Zamana gore veri bolme: holdout ve walk-forward fold'lar.

SIZINTI ONLEME KURALLARI:
- Asla karistirma yok: girdi tarih listesi kronolojik sirali kabul edilir.
- PURGE: fold sinirinda, label ufkunun (purge_days ~ label_horizon) val
  bolgesine TASMASINI engeller: train'in son satirlarinin label penceresi
  val'e sirirsa, model dogrudan gelecegi uses ve val fren. Purge: train ile
  val arasina "islem gunu turuna" bosluk koyar.
- EMBARGO: purge'dan SONRA ekstra bos islem gunu (config'ten). Iki bolme
  arasindaki bilgi tekrar/late haber akisini biraz daha ayrilar.
- Iki fonksiyon da SAF'tir: tarih girdi, tarih cikti; veriyle baska hic isim yapmaz.
"""

from __future__ import annotations

import pandas as pd


def _unique_dates(dates: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    """Girilen tarihleri unique + artan sirali DatetimeIndex'e cevirir."""
    index = pd.DatetimeIndex(dates)
    return index.dropna().unique().sort_values()


def holdout_split(
    dates: pd.DatetimeIndex | pd.Series,
    holdout_years: int,
    purge_days: int,
    embargo_days: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Tarihlere holdout ayirir: son `holdout_years` yil holdout, oncesi dev.

    dev, holdout'tan tam (purge_days + embargo_days) islem gunu ONCE bitirilir:
    dev satirlarinin label ufkunun (purge_days ~ label_horizon) holdout
    bolgesine tasmasi engellenir; uzerine embargo gunu eklenir.
    """
    dates = _unique_dates(dates)
    if len(dates) < 10:
        raise ValueError(f"Tarih paneli cok kucuk: {len(dates)} gun")

    cutoff = dates.max() - pd.DateOffset(years=holdout_years)
    holdout_index = dates[dates > cutoff]
    if len(holdout_index) == 0:
        raise ValueError(f"Tarih panelinde {holdout_years} yillik holdout bolgesi yok.")

    holdout_start_pos = int(dates.searchsorted(holdout_index.min()))
    gap = purge_days + embargo_days
    dev_index = dates[: max(holdout_start_pos - gap, 1)]
    return dev_index, holdout_index


def walk_forward_splits(
    dates: pd.DatetimeIndex | pd.Series,
    min_train_years: int,
    val_months: int,
    step_months: int,
    purge_days: int,
    embargo_days: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Kronolojik, buyuyen pencere ile kaydirilan walk-forward fold listesi.

    Her fold: (train_dates, val_dates).
    - val: uzunluk `val_months`; her adimda pencere `step_months` kayar.
    - train: EXPANDING window; ilk fold'da en az `min_train_years` olur,
      fold'lar ilerledikce buyur.
    - train ile val arasinda PURGE+EMBARGO boslugu: toplam (purge_days +
      embargo_days) islem gunu. Train, val'in ilk gununden bu kadar once
      biter; boylece train satirlarinin label ufkunun val'e sipari kismi
      olmaz.
    """
    dates = _unique_dates(dates)
    buffer = purge_days + embargo_days
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []

    val_start = dates.min() + pd.DateOffset(years=min_train_years)
    final_day = dates.max()
    while val_start <= final_day:
        val_end = val_start + pd.DateOffset(months=val_months)
        val_dates = dates[(dates >= val_start) & (dates < val_end)]
        if val_dates.empty:
            break
        val_start_pos = int(dates.searchsorted(val_dates.min()))
        train_dates = dates[: max(val_start_pos - buffer, 1)]
        if len(train_dates) == 0:
            break
        folds.append((train_dates, val_dates))
        val_start = val_start + pd.DateOffset(months=step_months)
    if not folds:
        raise ValueError(
            "Walk-forward fold uretilemedi: tarih paneli, min egitim yilina gore kucuk."
        )
    return folds
