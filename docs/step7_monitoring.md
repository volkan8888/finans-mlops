# Adim 7 (FAZ 5) - Monitoring

## Neler takip edilir

1) **Feature drift**: referans dev (tahmin doneminden ONCEKI bolum) vs
   canli pencere (tahmin donemi) arasinda her feature icin:
   - `psi` : Population Stability Index (referans quantile kutulari, 10 kutu)
   - `ks`  : Kolmogorov-Smirnov istatistik + p-degeri (scipy)
2) **Performans**: skor/label akisi uzerinden gunluk IC serisi,
   genel ozet (mean/icir/t_stat/...) ve son tamamlanan `roll_ic_window`
   gunluk kayan IC.
3) **Kapsama (coverage)**: gun sayisi ve gun basi ortalama ticker.

## Esikler ve uyari mantigi (config/monitoring.yaml)

| Anahtar | Varsayilan | Anlami |
|---|---|---|
| psi_alert | 0.2 | PSI 0.2 ve ustu = feature uyari (klasik esik) |
| alert_ic | 0.0 | kayan IC 0'in altina duserse uyari |
| roll_ic_window | 20 gun | kayan IC penceresi |
| psi_bins | 10 | PSI kutu sayisi |

Uyarilar `alerts` listesinde toplanir (uyari yoksa bos liste).

## Sinir/kavram notlari

- **KS p-degeri buyuk veride asiri duyarli**: 55k satirda kucuk kaymalar
  bile p~0 uretir; bu yuzden uyari karari PSI'ya bagli (etki buyuklugu
  bazli), KS degerleri betimsel raporlanir. (Gercek raporda tum PSI < 0.02:
  dev -> holdout arasinda dagilim kaymasi yok.)
- **Label gecikmesi**: `label` (forward_return) horizon gun sonunda
  gerceklesir; IC hesabi yalnizca label'i hazir gunlerde calisir
  (look-ahead yok).
- **Referans pencere secimi**: tahmin parquenin ilk gunu (2024-10-15)
  sinira geriye kalan tum dev referans alisir; canli pencere tahmin
  donemi. Model degisirse (yeni production alias) reference sabit
  kalir, surekli ayni isaretciyle karsilastirma dogurur.

## Kullanim

```
.venv\\Scripts\\python -m monitoring.run
# -> reports/monitoring_report.json
```

Rapor yapiasi:
```json
{
  "reference_end": "2024-10-14",
  "current_window": ["2024-10-15", "2026-10-14"],
  "psi_alert_threshold": 0.2,
  "feature_drift": [{"feature": "...", "psi": 0.015, "ks_stat": ..., "ks_p": ..., "alert": false}, ...],
  "performance": {"overall": {...}, "rolling_ic_20d": 0.118, "alert": false,
                   "coverage_n_days_covered": 501, "coverage_avg_tickers_per_day": 44, ...},
  "alerts": []
}
```

## Ileri test raporu (src/monitoring/forward.py) - canli tahminlerin gerceklesmesi

Gunluk akisin TARIH DAMGALI tahmin kayitlari
(`data/predictions/predictions_YYYY-MM-DD.parquet`) zamanla gerceklesir.
`forward_test_report` her (gun, ticker) tahminini label AYNI tanimla
(islem-gunu takviminde D+horizon) gerceklesmis getiriyle eslestirir:

- Pencere dolduysa: `realized = P(D+h) / P(D) - 1` (h = label_horizon, 5).
- Dolmadiysa / fiyat yoksa: kayit atlanir ve `n_unrealized` SAYILIR (uydurma yok).
- Benchmark: SPY ayni pencerede; ust kuantilin farki `vs_benchmark`.
- IC: gunluk spearman(score, realized); t ham + Newey-West t (nw_lag).
- Kuantil tablo: gunluk cross-sectional skor kuantilinin ort. gerceklesmis getiri.

Rapor JSON, durum anahtari uc farkli olabilir:
- `ok`      : yeterli sayida kayit gerceklesti (overall + quantiles dolu)
- `tum_beklemede`: tahmin var, hicbir pencere henuz dolmadi (yeni tahmin biriktikce dolacak)
- `bos`     : tahmin/fiyat deposu yok

Kullanim:
```
.venv\\Scripts\\python -m monitoring.forward
# -> reports/forward_test/forward_report.json
```

Dondurme imzasi ayridir: `python scripts/freeze_model.py` ->
`reports/forward_test/freeze.json` (model dosyalari + config SHA-256;
ileri test donemi boyunca model/config degismezliginin kaniti).

## Modul haritasi

| Dosya | Gorev |
|---|---|
| src/monitoring/drift.py | psi(), ks_statistic() |
| src/monitoring/performance.py | ic_series(), latest_rolling_ic(), overall_ic(), coverage() |
| src/monitoring/run.py | config yukle -> iki pencere -> rapor + JSON |
| src/monitoring/forward.py | forward_test_report (canli gerceklesme + unrealized) |
| config/monitoring.yaml | yollar + esikler |
| tests/test_monitoring.py | 10 test (ayni/kaymis dagilim, IC, rapor) |
| tests/test_forward.py | 9 test (unrealized, kuantil+SPY, freeze imza) |

## Gercek veri sonucu (2026-10-23 calisirmasi)

- Feature drift: tum PSI <= 0.0156 (volatility_20 en yuksek) - uyari YOK.
- Performans: overall mean_ic 0.0082 (t=0.78, zayif - bilinen), son
  20-gun kayan IC 0.118 (son pencere pozitif seyrine girmis), uyari yok.
- Kapsama: 501 gun, gun basi 44 ticker (tam universum).
