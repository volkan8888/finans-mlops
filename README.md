# finans-mlops

Fiyat verisinde cross-sectional siralanma (rank) icin uc-uca bir
ML Ops projesi: ingestion -> features -> egitim (walk-forward) ->
backtest -> MLflow takip -> serving -> monitoring -> orkestrasyon ->
dashboard -> Docker/CI.

> Durust not: tek hiperparametre seti config'te
> secildi (varsayilanlar) ve sonuc durust raporlandi:
> holdout IC istatistiksel olarak anlamli degil (asagida). Holdout'a
> ya da backtest'e bakarak herhangi bir ayar YAPILMADI.

## Cevre ve kurulum

- Python 3.12 + sanal cevre (`.venv` onerilir)
- Ana bagimliliklar: pandas (<3), numpy, pyarrow, pandera, yfinance,
  scipy, scikit-learn, torch (CPU kurulumu onerilir), mlflow, fastapi,
  httpx, uvicorn, prefect, streamlit, plotly
- Kurulum: `pip install -e ".[dev]"`  (testler: pytest; lint: ruff)

## Proje semasi

```
config/          veri/egitim/backtest/serving/monitoring yapilandirmalari
data/            raw (yfinance) + processed (panel, tahminler)
docs/            adim adim calisma defteri (PROGRESS.md + step*_*.md)
models/          egitim cikislari (meta.json + model.pt | model.joblib)
mlruns/ + mlflow.db   MLflow deney takibi + model registry
reports/         backtest/monitoring rapor cikislari
scripts/         orkestrasyon CLI (run_daily.py, smoke_e2e.py)
src/
  ingestion/     fiyat indirme + dogrulama + publish
  features/      teknik feature paneli + etiketler
  training/      egitim altyapisi + MLflow entegrasyonu
  evaluation/    backtest motoru + benchmark karsilastirmasi
  serving/       model yukleme + FastAPI skor servisi
  monitoring/    PSI/KS drift + IC performans raporlari
  pipelines/     Prefect akislari (gunluk + retrain)
  dashboard/     Streamlit panosu hazirligi
tests/           122 test (ag-bagimsiz, sentetik)
app.py           Streamlit panosu giris dosyasi
```

## Metodoloji kurallari (net kurallar; izin verilmeyen hatalar)

- Shuffle HICBIR yerde yok (DataLoader dahil) - zaman serisi akisi korunur.
- Dev ve HOLDOUT ayridir: son 2 takvim yili holdout; SADECE BIR KEZ
  okunur (tek atis). Model/hiperparam secimi yalnizca walk-forward
  validasyon IC ile yapilir.
- Foldlar arasi PURGE (5) + EMBARGO (5) gun; label horizonlu tasma engellenir.
- Feature z-score: cross-sectional, ddof=0 (panel uretiminde sabit).
- Paket dagilimi pyproject.toml; adim adim belgeler docs/ altinda.

## Kullanim dizisi

```bash
# 1) veri indirme + dogrulama (yfinance; internet gereklidir)
python -m ingestion.run
# 2) feature paneli uretimi
python -m features.run
# 3) egitim (walk-forward x ridge/mlp/gru -> secim -> final + holdout + MLflow)
python -m training.run
# 4) holdout backtest (long_short/long_only + SPY karsilastirma)
python -m evaluation.run
# 5) skor servisi (production model: MLflow registry -> yerel fallback)
python -m serving.app        # http://127.0.0.1:8000 (GET /health, POST /predict)
# 6) monitoring raporu (drift + kayan IC)
python -m monitoring.run
# 7) gunluk akis (skor parquet + monitoring)
python scripts/run_daily.py --mode daily
# 8) retrain zinciri (ingestion -> features -> training -> evaluation -> monitoring)
python scripts/run_daily.py --mode retrain
# 9) Streamlit panosu
streamlit run app.py
# 10) Docker: skor servisi; MLflow UI (ayri profil)
docker compose up
docker compose --profile ui up
```

## Durust model raporu (gercek veri, 2026-10-23)

- Panel: 127,732 satir | 2,903 gun | 44 ticker | 8 feature; 13 fold.

### Siralama kalitesi (IC tablosu)

| Olcek | n gun | mean IC | t (ham) | t (Newey-West, lag=4) | t (bagimsiz, stride=5) |
|---|---|---|---|---|---|
| walk-forward val - ridge | 1635 | 0.0105 | 1.57 | yok (seri kaydedilmemisti) | yok |
| walk-forward val - mlp | 1635 | 0.0140 | 2.75 | yok (seri kaydedilmemisti) | yok |
| walk-forward val - gru | 1635 | 0.0175 | 2.75 | yok (seri kaydedilmemisti) | yok |
| HOLDOUT (tek atis) - gru | 501 | 0.0082 | 0.78 | 0.45 | 0.26 (101 gun) |

- NW/bagimsiz t yalnizca holdout icin hesaplanabildi
  (scripts/ic_report.py: kayitli tahminlerden yeniden hesap; model
  egitilmedi). Val tahmin serisi kaydedilmedigi icin NW geriye donuk
  hesaplanamaz; sonraki retrain'de seriyi diske yazmak bu acigi kapatir.
- Kantil getiri (gunluk ort., holdout): q0 0.47%, q1 0.19%, q2 0.29%,
  q3 0.41%, q4 0.51%. Alt kuantilin gunluk getirisi ust kuantil ile
  ayni seviyede: kisa bacak ters tepti (5-gunluk reversal),
  long_short dengesi bozuldu.

### Portfoy geri testi (10 bp tek-yon maliyet, 5 gun rebalance, 501 gun)

| Mod | Brut getiri / sharpe | Net getiri / sharpe | Beta (SPY'a) | Alfa* |
|---|---|---|---|---|
| long_short | -0.7% / 0.03 | -8.7% / -0.33 | ~0.0 | - |
| long_only | +46.7% / 1.06 | +37.2% / 0.89 | 0.85 | ~+7.5%/yil (t yetersiz) |
| SPY | +34.7% / 0.99 | - | 1.0 | - |

*Alfa/beta, backtest motorunun SPY'a karsilastirmasindan gelir; 
t-stat istatistiksel olarak yetersiz. long_short zararda; long_only
getirisi buyuk olcude beta.

## Durust sinirlar (yatirim oncesi okuyun)

- Sinyal holdout'ta istatistiksel anlamli DEGIL (t_nw 0.45, bagimsiz
  t 0.26): kalici pozitif IC kaniti yok.
- long_only getirisi cogunlukla beta (beta ~0.85): piyasa rallisinde
  her portfoy iyi gorunur.
- Survivorship bias: evren bugunun 44 buyuk hissesi (delist olanlar
  panelde yok); SPY karsilastirmasi da bu yuzden adil degildir.
- Veri yfinance'den (fiyat revizyonu / hata payi mumkun); kucuk evren
  (44 hisse); maliyet modeli basit (tek-yon 10 bp; slippage/komisyon yok).
- Label horizon 5 gun; IC cakismasi (Newey-West duzeltmesi yukarida).
- BU PROJE YATIRIM TAVSIYESI DEGILDIR: MLOps/uretim ornegidir.
- Karar: sinyal ZAYIF; urun yatirimi icin elimde guvenli kanit yok.
  Bu rapor sonuc haliyle durust; holdout'a bakarak degisiklik yok.

## Testler ve lint

- `pytest -q`: 122 test (ingestion 20, features 19, training 15,
  serving 15, pipelines 12, monitoring 10, dashboard 10, ic_stats 7,
  backtest 9, mlflow 5) - ag-bagimsiz (sentetik panel + adaptif rapor
  dogrulamalari).
- `ruff check .`: temiz (0 hata).
- Sonraki adimlar icin docs/PROGRESS.md guncel karnedir.

## Belgeler

- docs/PROGRESS.md: tek bakista durum (tum fazlar + gorunen sonuc).
- docs/step3_training.md ... step11_final_report.md: adim detaylari.
