"""Ticker-bazli teknik feature'lar: saf fonksiyonlar (pd.Series girer/cikar).

SIZINTI KURALI:
    Bu moduldeki tum fonksiyonlar GECMISE bakar: her deger yalnizca t gunu VE
    ONCESININ verisiyle hesaplanir (shift(-n) burada YOKTUR; o yalnizca
    labels.py'deki forward_return icin kullanilir).
    Hisselerin birbirine karismamasi icin grup ayrimini cagiran taraf yapar:
    build.py her ticker'in tek serisini bu fonksiyonlara verir
    (groupby("ticker") disarida).

Kucuk girdi kurallari:
    - Girdi serisi kronolojik sirali olmali (build.py (ticker, date) siralayip
      gruplara ayirir).
    - Fonksiyonlar girdiyi degistirmez; yeni Series dondurur.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(adj_close: pd.Series) -> pd.Series:
    """Gunluk log getiri: ln(P_t / P_{t-1}).

    Neden log: gunluk getirilerin (volatilite ve reversal icin) TOPLANABILIR
    olmasini saglar; aritmetik getiri toplanamaz.
    Ilk satir shift(1) yuzunden NaN'dir; warm-up dusurmesinde sayilir.
    """
    previous = adj_close.shift(1)
    return np.log(adj_close / previous)


def momentum(adj_close: pd.Series, window: int) -> pd.Series:
    """`window` gunluk gecmis getiri: P_t / P_{t-window} - 1.

    Aritmetik getiri secildi: momentum agirlikli uzun donem getirisidir; dusuk
    momentum hisseleri "nasil kaybettigine" bagli olmaksizin siralanabilir.
    Ayrica aritmetik getiri, portfoy P&L formuna birebir uyar.
    t-window degeri yoksa NaN (warm-up).
    """
    past = adj_close.shift(window)
    return adj_close / past - 1.0


def momentum_skip(adj_close: pd.Series, window: int, skip: int) -> pd.Series:
    """Buyuk momentum'dan en yakin `skip` gunu atlar:
    P_{t-skip} / P_{t-window} - 1.

    Ornek (60, 5): 60 gunluk getiri ama SON 5 gun haric — "12 ay - 1 ay"
    momentum'unun (uluslararasi literaturde en bilinen anomali) gunluk karsiligi.
    Neden: son birkac gunde cok yukselen hisseler kisa vadede duzeltme
    yapabildigi icin, son gunleri cikarmak daha istikrarli bir momentum
    olcumu verir.
    """
    if not (0 <= skip < window):
        raise ValueError(f"skip({skip}) 0..window-1 araliginda olmali (window={window})")
    past_long = adj_close.shift(window)
    past_short = adj_close.shift(skip)
    return past_short / past_long - 1.0


def volatility(returns: pd.Series, window: int) -> pd.Series:
    """Gunluk log getiri serisinin rolling std'i (ornekleme std, ddof=1).

    Anlam: hissenin fiyat oynakligi (risk). Diger sartlar esken yuksek
    volatilite daha riskli kabul edilir.
    Ilk `window` gun ve returns'in ilk NaN'i yuzunden NaN doner; warm-up
    dusurmesinde sayilir.
    """
    return returns.rolling(window=window).std()


def rsi(adj_close: pd.Series, window: int = 14) -> pd.Series:
    """Klasik RSI, Wilder duzlestirilmesi (ewm alpha=1/window, adjust=False).

    RSI = 100 - 100 / (1 + RS),  RS = ort. kazanc / ort. kayip.
    Wilder: ayni 1/window yumusataligi, ewm(alpha=1/w, adjust=False) ile bire-
    bir uygulanir.

    Kose durumlari (bölme hatasi olmasin):
    - ort. kayip == 0 (seri surekli yukseliyor) -> RSI = 100
    - ort. kazanc == 0 (seri surekli dusuyor)  -> RSI = 0
    - ikisi de 0 (seri duz)                    -> RSI = 50 (notr)
    NaN'ler yalnizca min_periods tamamlanmamis (warm-up) konumlarinda kalir;
    oradaki NaN'lari doldurma YAPILMAYACAK (warm-up zaten dusurur).
    """
    if window <= 0:
        raise ValueError(f"window pozitif olmali, gelen: {window}")
    delta = adj_close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    # ort. kayip 0 olabilir -> paydayi NaN'a cevirip bolelim, sonra maskelerle kapat
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_series = 100.0 - 100.0 / (1.0 + rs)

    rsi_series = rsi_series.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi_series = rsi_series.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi_series = rsi_series.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi_series


def volume_change(volume: pd.Series, window: int) -> pd.Series:
    """Hacim degisimi: ln(volume_t / rolling_mean(volume, window)).

    Anlam : bu gunun hacmi, pencere ortalamasindan ne kadar sapmistir.
    Haber/ilgi (yuksek hacim) gostergesi.

    Guvenli durumlar:
    - volume == 0 gun  -> ln(0) = -inf yerine NaN (eksik bilgi sayiliyor;
      warm-up asamasinda satir dusurulur, DOLDURMA YAPILMAZ)
    - rolling mean == 0 (tum pencere sifir hacim) -> NaN
    """
    if window <= 0:
        raise ValueError(f"window pozitif olmali, gelen: {window}")
    rolling_mean = volume.rolling(window=window).mean()
    ratio = volume / rolling_mean.replace(0.0, np.nan)
    out = np.log(ratio)
    out = out.mask(volume <= 0)          # log(0) -> -inf yerine NaN
    out = out.mask(rolling_mean <= 0)    # tum pencere sifir hacim -> anlamsiz
    return out


def short_term_reversal(returns: pd.Series, window: int = 5) -> pd.Series:
    """Kisa vadeli reversal: son `window` gunun log getirisi TOPLAMI (ham girdi).

    Finansal anlam: son 5 gunde cok yukselen hisseler bir sonraki gunlerde
    duzeltme egilimi gosterebilir (kisa vade tersi hareket); tam tersi de.
    Feature'in ISARETI CEVRILMEZ: ham hali teslim edilir; model isaretin
    yönünü kendi ogrensin (onyargi/dolgu eklenmez).
    """
    if window <= 0:
        raise ValueError(f"window pozitif olmali, gelen: {window}")
    return returns.rolling(window=window).sum()
