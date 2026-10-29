# Adim 9 (FAZ 7) - Dashboard (Streamlit)

## Calistirma

```
.venv\\Scripts\\streamlit run app.py
# tarayicida http://localhost:8501
```

## Sekmeler

| Sekme | Kaynak dosya | Icerik |
|---|---|---|
| Skorlar | data/processed/latest_scores.parquet | top/bottom 5 tablo + bar; yoksa talimat |
| Backtest | reports/backtest_summary.json + equity_curve.csv | metrik kartlari (net/brut/alfa-beta) + line chart (stratejiler vs SPY) |
| Monitoring | reports/monitoring_report.json | PSI bar + tablo, uyari listesi, kayan IC |
| Ileri Test | reports/forward_test/freeze.json + forward_report.json | donmus imza karti + gerceklesen IC/t_nw + kuantil getiri bar; veri/yok durumunda net bos-durum mesaji |
| MLflow Runs | mlflow.db (sqlite) | run tablosu (variant/production) + 'production' alias versiyonu |

Panonun rolu SADECE OKUMAKTIR: veriler CLI akmalarindan gelir
(scripts/run_daily.py --mode daily, evaluation.run, monitoring.run);
dashboard bunlari asla yenilemez. Boylece panoda gorunen her sayi
dosyadaki raporla birebir aynidir (ikinci yazim yok).

## Veri yardimci katmani

- src/dashboard/prepare.py: tum deger okuma/temizleme fonksiyonlari
  (load_scores, top_bottom, load_summary, load_equity, load_monitoring,
  psi_table, mlflow_runs_table, production_alias_info). HIC streamlit
  bagimliligi YOK testlerde direk kullanilabilir.
- app.py: sayfa cizimi + fmt_percent (None/NaN guvenli formatlanma).

## Smoke dogrulama (gercek veri)

- tests/test_dashboard.py (10 test): prepare fonksiyonlari GERCEK rapor
  dosyalarina karsi test edilir (dosya varsa deger, yoksa bos davranis);
  mlflow runs tablosunda 4 run; app.py modul import smoke.
- Headless canli kosu: `streamlit run app.py` -> /_stcore/health = "ok".

## Bilinen sinirlar

- mlflow runs tablosu yalniz 'finans-training' experiment'inin son 20
  run'inin adini/metriklerini listeler (detayli arayuz icin MLflow UI:
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`).
- Dashboard veri guncellenmez (revizyon: otomatiklesen tasks tercih
  edilir; su an CLI akisi referans).
