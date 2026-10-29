"""Cross-sectional islemler: ayni gundeki hisseler arasinda kirpma/standartlastirma.

Onemli kurallar:
- Tum islem `groupby("date")` icinde yapilir: her gun kendi quantile/ort/std
  dagilimini kullanir. Gunler birbirinden bagimsizdir (bir gunun degerleri
  diger gunun sinirlarini etkilemez).
- Cross-sectional SIRA: ham feature -> winsorize -> zscore. Winsorize once
  gelir: aksi halde gunun tek bir isi degeri (cok abartili hisse) o gunun
  ort/std'sini bozarak butun hisselerin z-score'unu calar.
- Girdi DataFrame DEGISTIRILMEZ; kopya uzerinde calisir, yeni frame dondurur.
- shift(-n) burada YOKTUR: bu modul gecmise/gelecege bakmaz, yalnizca ayni
  gunun verisi ile calisir.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

GROUP_COLUMN = "date"


def _validate(df: pd.DataFrame, cols: list[str]) -> None:
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Girdide '{GROUP_COLUMN}' kolonu gerekli.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Girdide kolon yok: {missing}")


def winsorize_by_date(
    df: pd.DataFrame, cols: list[str], lower: float, upper: float
) -> pd.DataFrame:
    """Her gunde kolon degerlerini gunun kendi quantile sinirlarina kirpar.

    Neden: gunun tek bir abarti degeri (orn. 4x yukselen hisse) o gunun ort/std
    dagilimini bozar; winsorize uclari sinira yaslar ama bilgiyi tamamen silmez.
    - lower/upper: 0-1 arasi quantile orani (orn. 0.01/0.99).
    - Sinirlar HER GUN kendi verisinden hesaplanir; baska gunun dagilimini
      kullanmak yapilmaz (gunler bagimsiz).
    - Bu asamada feature kolonlarinda NaN yoktur (warm-up once dusulmustur).
    """
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError(
            f"winsorize sinirlari gecersiz: 0 <= lower({lower}) < upper({upper}) <= 1"
        )
    _validate(df, cols)
    out = df.copy()
    for col in cols:
        out[col] = out.groupby(GROUP_COLUMN, sort=False)[col].transform(
            lambda s: s.clip(s.quantile(lower), s.quantile(upper))
        )
    return out


def zscore_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Her gunde cross-sectional z-score: (x - gun_ort) / gun_std (ddof=1).

    Neden: model ayni gunun hisselerini ARASINDA siralar; standartlastirma
    genel piyasa duzeyini siler (piyasa yuksekken herkesin zaten yuksek
    degeri olur) ve o gunun icindeki GORELI konumu one cikarir.

    std == 0 (o gunun tum degerleri ayni) veya std NaN (n=1) durumunda NaN
    yerine 0 doner: degiskenlik yoktur — z=0 notr degerdir; bolme hatasi
    yerini notr degere birakir.

    ddof=0 (populasyon std) secildi: gun ici std'si tam 1 olur; tam panel
    tek bakista describe() ile incelendiginde de std ~ 1.000 cikar. ddof=1
    (ornekleme std) kullansaydik her gun (n-1) bazli standartlastirma yapilir
    ve panel toplaminin std'si sqrt((n-1)/n) (44 hisse icin sqrt(43/44))
    kadardi ~ 0.9886 olurdu: kosmetik bir artefakt. ddof secimi IC gibi
    korelasyon olcumlerini hic degistirmez.
    """
    _validate(df, cols)
    out = df.copy()
    for col in cols:
        grouped = out.groupby(GROUP_COLUMN, sort=False)[col]
        mean = grouped.transform("mean")
        std = grouped.transform(lambda s: s.std(ddof=0))  # populasyon std: gun ici std tam 1
        z = (out[col] - mean) / std
        # degiskenligi olmayan (veya n=1) gunde: z = 0 (NaN degil, notr)
        z = z.mask(std == 0, 0.0)
        z = z.mask(std.isna(), 0.0)
        out[col] = z
    return out
