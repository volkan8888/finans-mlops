"""Gunluk degerlendirme araclari: daily IC, IC ozeti, kantil getirileri.

- IC: bir gunun skorlari ile rank hedefi (label) arasindaki Spearman korelasyonu.
  Neden Spearman: gorev siralamadir; mutlak getiri seviyesi degil, "siralamayi
  acikliyor mu" sorulur. Kucuk gun baglari (min_n alti) NaN sayilir.
- Hedef olarak label (rank), getiri olarak ham forward_return kullanilir:
  ikisi KARISTIRILMAZ (IC = siralama kalitesi; kantil getiri = para olcegi).
- Otokorelasyon notu: label, ufuk gunluk ileri getiri oldugundan ardisik
  gunlerin IC'leri cakisir (bagimsiz orneklem sayisi < gun sayisi); klasik
  t-stat siskin olabilir. ic_summary bu yuzden Newey-West (Bartlett)
  duzeltmeli t'yi (t_nw) de raporlar; bagimsiz ornekleme ozeti icin
  stride_ic_summary var (her horizon gunde bir IC).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_NW_LAG = 4  # label_horizon(5) - 1: ardisik label'larin cakisma uzunlugu


def daily_ic(df: pd.DataFrame, pred_col: str, target_col: str, min_n: int = 3) -> pd.Series:
    """Her gun icin Spearman IC serisi dondurur (index=date).

    Gun icinde en az min_n gecerli cift yoksa veya skor/etiket std=0 ise o gun
    NaN'dir (0 SAYILMAZ; yanlis pozitif IC urememek icin).
    """
    if df.empty:
        return pd.Series(dtype=float)

    def _spearman(group: pd.DataFrame) -> float:
        scores = group[pred_col].to_numpy(dtype=float)
        labels = group[target_col].to_numpy(dtype=float)
        if len(scores) < min_n or np.std(scores) == 0 or np.std(labels) == 0:
            return float("nan")
        return float(pd.Series(scores).corr(pd.Series(labels), method="spearman"))

    grouped = df[["date", pred_col, target_col]].groupby("date", sort=True)
    # include_groups=False: grubun "date" kolonunu fonksiyona gecirmez (pandas 2.2+ uyumu)
    return grouped.apply(_spearman, include_groups=False)


def newey_west_t(ic: pd.Series, lag: int = DEFAULT_NW_LAG) -> float:
    """Newey-West (Bartlett) duzeltmeli t-istatigi: otokorelasyonlu IC serileri.

    Var(mean) = (g0 + 2 * sum_l w_l * gl) / n   w_l = 1 - l/(lag+1)
    (gl = l adim otokovaryans; standart Newey-West sonsuz-dagilim halini
    kullanir). Pozitif otokorelasyon SE'yi buyutup |t|'yi dusurur; ardisik
    horizon cakismasinda ham t yalanci buyuk olabilir.
    n < 3 veya degiskenlik yoksa / sonuc non-finite ise NaN doner.
    """
    valid = pd.Series(ic).dropna().astype(float).to_numpy()
    n = len(valid)
    if n < 3:
        return float("nan")
    lag = int(min(lag, n - 1))
    if lag < 0:
        return float("nan")
    centered = valid - valid.mean()

    def _gamma(offset: int) -> float:
        return float(np.sum(centered[:-offset] * centered[offset:]) / n)

    variance = float(np.dot(centered, centered) / n)
    for step in range(1, lag + 1):
        weight = 1.0 - step / (lag + 1)  # Bartlett cekirdegi
        variance += 2.0 * weight * _gamma(step)
    if not math.isfinite(variance) or variance <= 0.0:
        return float("nan")
    standard_error = math.sqrt(max(variance, 0.0) / n)
    if standard_error <= 0.0:
        return float("nan")
    return float(valid.mean()) / standard_error


def stride_ic_summary(ic: pd.Series, stride: int) -> dict[str, float | int]:
    """'Bagimsiz ornekleme' ozeti: her stride. gunde bir IC alir.

    Ufuk cakismasini azaltmak icin: en YENI gunden geriye dogru her stride
    gun kaydirilarak alt-orneklem kurulur (5-gunluk label'da stride=5 ile
    ardisik uc orneklemin label pencereleri cakismaz). Alt-orneklemin
    ortalamasi ve t-istatigi raporlanir; n veya degiskenlik yetersizse NaN.
    """
    valid = pd.Series(ic).dropna().astype(float)
    count = len(valid)
    if stride <= 0 or count == 0:
        return {"stride_n": 0, "stride_mean_ic": float("nan"), "stride_t_stat": float("nan")}
    subset = valid.iloc[::-1].iloc[::stride]  # son gunden geriye yurur
    m = len(subset)
    mean = float(subset.mean())
    std = float(subset.std(ddof=1)) if m > 1 else 0.0
    t_stat = mean / std * math.sqrt(m) if std > 0 else float("nan")
    return {"stride_n": int(m), "stride_mean_ic": mean, "stride_t_stat": float(t_stat)}


def ic_summary(ic_series: pd.Series, nw_lag: int = DEFAULT_NW_LAG) -> dict[str, float | int]:
    """IC serisi ozeti: mean, std, ICIR, ham t + Newey-West t (t_nw) + pozitif oran.

    - ham t: iid varsayimla t = mean/std*sqrt(n); ardisik cakismada (5-gunluk
      label) siskin olabilir; t_nw daha muhafazakar bir olcek.
    - ICIR buyuk ve pozitif = sinyal tutarli; |t|>2 civari istatistik ikna
      edici gorulur.
    """
    valid = ic_series.dropna().astype(float)
    n = len(valid)
    if n == 0:
        return {
            "mean_ic": float("nan"), "std_ic": float("nan"), "icir": float("nan"),
            "t_stat": float("nan"), "t_nw": float("nan"),
            "positive_share": float("nan"), "n_days": 0,
        }
    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if n > 1 else 0.0
    icir = mean / std if std > 0 else float("nan")
    t_stat = mean / std * np.sqrt(n) if std > 0 else float("nan")
    return {
        "mean_ic": mean,
        "std_ic": std,
        "icir": float(icir),
        "t_stat": float(t_stat),
        "t_nw": float(newey_west_t(valid, lag=nw_lag)),
        "positive_share": float((valid > 0).mean()),
        "n_days": n,
    }


def quantile_returns(
    df: pd.DataFrame, pred_col: str, ret_col: str, n_quantiles: int = 5
) -> pd.DataFrame:
    """Skorlarin kantil bazli HAM getiri ortalamalari.

    Yontem: her gunde skor yuzdesi (pct rank) uzerinden kantil numarasi 0..n-1
    verilir; her kantildaki gunlerin ORT forward_return degeri raporlanir.
    Monotonluk: q0'dan q4'e getiri artiyorsa siralama gercekten ogrenilmis demektir.
    """
    if df.empty:
        return pd.DataFrame({"quantile": [], "mean_return": [], "n": []})
    data = df.copy()
    data["_rank_pct"] = data.groupby("date")[pred_col].rank(pct=True)
    data["_q"] = (data["_rank_pct"] * n_quantiles).clip(upper=n_quantiles - 1).astype(int)
    grouped = data.groupby("_q", sort=True)[ret_col]
    means = grouped.mean()
    counts = grouped.size()
    return pd.DataFrame({
        "quantile": range(n_quantiles),
        "mean_return": [float(means.get(q, np.nan)) for q in range(n_quantiles)],
        "n": [int(counts.get(q, 0)) for q in range(n_quantiles)],
    })