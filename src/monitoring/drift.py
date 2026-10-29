"""Adim 7 (FAZ 5): drift olcumleri - PSI ve KS.

PSI (Population Stability Index): referans ve guncel dagilimin kutu
yogunluklarini karsilastirir (0 ~ ayni; >0.2 klasik uyari esigi).
Kutular REFERANS quantile'lerinden kurulur; boylece esik sabit kaliyor.

KS (Kolmogorov-Smirnov): scipy.stats ile iki orneklem karsilastirmasi;
istatistik ve p-degeri raporlanir (p < 0.05 dagilim degisimi ipucu).

Tum fonksiyonlar NaN degerleri filtreleyip calisir; yetersiz orneklemde
NaN dondurur (uyari uretmez - bilincli davranis, belgeli).
"""

from __future__ import annotations

import math

import numpy as np


def psi(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Referans quantile kutularina gore PSI skoru.

    - kutular: referansin ic quantile ayrimlari (linspace 0..1 arasi).
    - bos kutu payi eps ile korunur (log(0) engeli).
    - orneklem az ise (her taraf 10'dan az) NaN dondurur.
    """
    ref = np.asarray([float(value) for value in reference if not math.isnan(float(value))], dtype=float)
    cur = np.asarray([float(value) for value in current if not math.isnan(float(value))], dtype=float)
    if ref.size < 10 or cur.size < 10:
        return float("nan")
    edges = np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    ref_ids = np.searchsorted(edges, ref, side="right")
    cur_ids = np.searchsorted(edges, cur, side="right")
    ref_pct = np.bincount(ref_ids, minlength=bins + 1)[:bins] / ref.size
    cur_pct = np.bincount(cur_ids, minlength=bins + 1)[:bins] / cur.size
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(
    reference: np.ndarray | list[float],
    current: np.ndarray | list[float],
) -> tuple[float, float]:
    """KS istatistigi ve p-degeri (scipy varsa; yoksa (NaN, NaN))."""
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return float("nan"), float("nan")
    ref = np.asarray([float(value) for value in reference if not math.isnan(float(value))], dtype=float)
    cur = np.asarray([float(value) for value in current if not math.isnan(float(value))], dtype=float)
    if ref.size < 2 or cur.size < 2:
        return float("nan"), float("nan")
    result = ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)
