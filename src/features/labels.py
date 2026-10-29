"""Etiket (label) olusturma: ileriye bakan fonksiyonlar buradadir.

SIZINTI UYARISI (projedeki en kritik kural):
    Bu MODUL, projede shift(-n) (yani GELECEGE bakan islem) kullanmaya izin
    verilen TEK YERDIR. Feature'lar (technical.py, build.py) hicbir yerde
    gelecege bakan islem icermemelidir; aksi halde model gelecegi gormus olup
    testlerde olmus gibi gorunur, gercek ticarette para kaybeder.

Etiket tasarimi: ham forward_return -> cross-sectional RANK (0-1).
    Neden rank: mutlak getiri gunden gune cok degisir (piyasa yuksekliginde
    herkes pozitif olabilir); oysa rank o gunun "en iyi getirenleri"ni sira
    olarak yakalar ve abartili tekil getirilerden daha az etkilenir. Ham
    forward_return de kolon olarak saklanir (backtest'te P&L hesaplamak icin).
"""

from __future__ import annotations

import pandas as pd

GROUP_COLUMN = "date"


def forward_return(adj_close: pd.Series, horizon: int) -> pd.Series:
    """t gununden t+horizon gunune getiri: P_{t+horizon} / P_t - 1.

    DIKKAT: bu fonksiyondaki shift(-horizon) GELECEGE bakar ve yalnizca
    LABEL icin kullanilir; feature pipeline'in diger hicbir yerinde shift(-n)
    yoktur. Son `horizon` satir gelecek verisi olmadigi icin NaN doner;
    build_dataset bu satirlari dusurur (build_features'ta KALIR: serving
    orada bugunu tahmin edebilir).
    """
    if horizon <= 0:
        raise ValueError(f"horizon pozitif olmali, gelen: {horizon}")
    future = adj_close.shift(-horizon)
    return future / adj_close - 1.0


def rank_label_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """Her gun cross-sectional rank (0-1), pct=True.

    - O gun getiri en yuksek olan hisse 1'e yakin, en dusuk olan 1/n'ye yakin
      rank alir.
    - NaN degerler NaN rank alir; build_dataset bunlari label-NaN dusurmesinde
      ayirir (feature-NaN'den ayri sayilir).
    - Esit degerlerde 'average' yontemi: esitlilere ayni rank payi verilir.
    """
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Girdide '{GROUP_COLUMN}' kolonu gerekli.")
    return df.groupby(GROUP_COLUMN, sort=False)[col].rank(pct=True)
