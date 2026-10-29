# Adim 8 (FAZ 6) - Orkestrasyon (Prefect)

## Iki akis

| Akis | Adimlar | Urunler |
|---|---|---|
| daily (hizli) | 1) veri-ingestion 2) fiyat-dogrulama 3) feature-uretim (label'siz) 4) skor-uret 5) tahmin-kaydi 6) monitoring-raporu | data/raw/prices.parquet (taze) + data/processed/latest_scores.parquet + data/predictions/predictions_YYYY-MM-DD.parquet + reports/monitoring_report.json |
| retrain (agir) | 1) ingestion.run 2) features.run 3) training.run 4) evaluation.run 5) monitoring | yeni production model + mlflow versiyonu + raporlar |

Prefect 3.x ile kuruldu; akislar dogrudan cagirilinca yerel (ephemeral)
modda kosar: sunucu/kurulum GEREKMEZ. Deployment plani proje sahibine
birakilir: crontab (Unix) ya da Task Scheduler (Windows) ile
scripts/run_daily.py gundelik tetiklenebilir.

## Gunluk akis detay (kalite guncellemesi: veri akisi dahil)

Task sirasi sabit: ingestion -> dogrulama -> features -> skor ->
tahmin kaydi -> monitoring. Ingestion eksikse akis bayat fiyatla skor
uretir; bu yuzden ingestion ARTIK akisin birinci adimidir.

- `task_run_ingestion`: `python -m ingestion.run` alt yordami (kapanis
  penceresini, anomalileri, SERT validasyonu ingestion.run kendi icinde
  yonetir). Idempotent: ayni gun ikinci kosu merge keep=last ile ayni
  depoyu verir; degisiklik yoksa dosya fiziksel olarak yeniden yazilmaz.
- `task_validate_prices`: depodaki dosya pandera semasindan gecer;
  bozuksa akis RuntimeError ile DURUR (kaydetme yok). Anomali raporu
  YUMUSAK loglanir (akisi kesmez).
- `task_build_features`: fiyat deposundan `features.build.build_features`
  (LABEL'SIZ) ile o anki feature tablosu kurulur; egitimin
  features.parquet dosyasina DOKUNULMAZ. Skor kaynaginin tamami training
  ile AYNI koddur = training-serving skew yok (bkz. docs/step6_serving.md).
- `task_compute_scores`: production modelin (registry -> yerel)
  `data/processed/latest_scores.parquet` yenileyip skor tablosunu doner.
- `task_record_predictions`: her gunun tahmini TARIH DAMGALI dosyaya
  yazilir (`data/predictions/predictions_YYYY-MM-DD.parquet`); ayni gun
  ikinci kosulursa uzerine YAZILMAZ (uyari loglanir) ve dosya cogalmaz:
  tahmin kaydi sonradan degismez gibi davranir (ileri test butunlugu).
- `task_run_monitoring`: monitoring.run.run_monitoring (PSI/KS + kayan
  IC + uyari listesi).
- Donus ozeti: {"ingestion", "validation", "scores_path", "scores_date",
  "predictions_path", "predictions_created", "alerts": [...]}.

## Kapanis kurali (eksik bar korumasi)

ABD pazari 16:00 America/New_York'ta kapanir. Kapanis ONCESI calisirsan
bugunun bar'i kismi veridir; skorlama TAMAMLANMIS son oturuma dayanir:

- `price_session_limit(now)`: kapanis oncesi ust sinir = bugun; sonrasinda
  bugun dahil. Testlerde `now` parametresiyle enjekte edilir.
- `drop_incomplete_session`: ust sinirin uzerindeki barlar atilir ve
  atilan gun LOGLANIR. Bu kural serving kaynak kurulumunda
  (FeatureSource.from_prices) da aynidir; skor gununun arkasinda kac
  fiyat gunu kaldigi /health'te raporlanir.
- Skorlanan gun: feature'lari (warm-up dahil) tam hesaplanabilen EN SON
  gun; min_tickers_per_date kurali gecerli.

Tatil/eksik veri durumunda: akis son TAMAMLANMIS gunu skorlar; skor
tarihi ile fiyat deposunun son tarihi /health'te yan yana gosterilir
(bayatlik buradan okunur).

## Retrain akis detay

- Komutlar alt yordamdan (subprocess, cwd = proje koku) kosar;
  cikti tamponlanir (stdout/stderr yakalanir). Hata durumunda RuntimeError (flow
  kesilir): bozuk feature ile egitim istenmez.
- Sonda monitoring raporu (ayni surec) uretilir.
- Uzunlugu (egitim ~10 dakika CPU) nedeniyle gundelik tetikler icin
  `--mode daily` onerilir; retrain haftalik/aylik planlanir.


## Kullanim

```
.venv\\Scripts\\python scripts/run_daily.py --mode daily
.venv\\Scripts\\python scripts/run_daily.py --mode retrain
```

## Modul haritasi

| Dosya | Gorev |
|---|---|
| src/pipelines/shared.py | scores_for_day / latest_scores_frame + yol cozuculer (Prefect'siz test edilebilir) |
| src/pipelines/daily_flow.py | 6 task (ingestion/dogrulama/features/skor/tahmin kaydi/monitoring) + daily_pipeline flow |
| src/pipelines/retrain_flow.py | task_run_command + retrain_pipeline flow |
| scripts/run_daily.py | CLI: --mode daily|retrain |
| tests/test_pipelines.py | 12 test (komut enjeksiyonu + idempotency + sira) |

## Test stratejisi

- Akislarin agir adimlari testte kosulmaz: komutlar inject edilir
  (python -c "..."); skor katmani sentetik kaynakla doldurulur;
  monitoring config tmp yollara yonlendirilir.
- Idempotency: prediction dosyasina dokunmama davranisi icin ayni gun
  iki kos + mevcut dosyanin bayt-before/after kontrolleri.
- Prefect'in kapanis surecinde "I/O operation on closed file" log
  gurultusunun engellenmesi icin testlerde PREFECT_LOGGING_LEVEL=ERROR
  ayarlanir.

## Gercek kosu sonucu (2026-10-23)

- daily: 44 skor -> data/processed/latest_scores.parquet (2026-10-14);
  monitoring raporu yenilendi (uyari yok); cikis kodu 0.

## Gercek kosu sonucu (2026-10-29, kalite guncellemesi sonrasi)

- daily exit 0; ingestion yeni barlari cekti (fiyat son gun 2026-10-29);
  skor gunu 2026-10-29 (bayat-skor sorunu giderildi: onceki kosuda skor
  2026-10-14'te kalmisti); validation 136804 satir / anomali 3 (bilinen);
  predictions_2026-10-29.parquet yazildi. Ayni gun IKINCI kosu: exit 0,
  dosya uzerine yazilmadi (uyari), tek predictions dosyasi.
