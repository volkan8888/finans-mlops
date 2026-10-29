# Adim 6 (FAZ 4) - Serving

## Akis (ust duzey)

```
uvicorn src/serving/app.py              (veya .venv/Scripts/python -m serving.app)
                |
   [create_app] HEMEN yukler:
        1) model: model_store.load_production_model()
             a) MLflow registry 'finans-ranker' + alias 'production'
             b) yoksa yerel models/<isim>/ (meta.json + model.pt|joblib)
        2) feature kaynagi (config/serving.yaml'daki 'source' anahtarina gore):
             source: prices           -> data/raw/prices.parquet + build_features
                                         (training ile AYNI kod; LABEL'SIZ;
                                         kapanis-oncesi kismi bar atlanir)
             source: features_parquet -> data/processed/features.parquet (yedek)
             (prices seciliyken depo okunamazsa yedek kaynaga dusilir ve
              /health bayatligi raporlar)
                |
   Endpoints:
        GET  /health            -> {status, model_name, n_features, score_date,
                                    data_end_date, stale_price_dates,
                                    incomplete_session_dropped, ...}
        POST /predict           -> {ticker, date?, ...}
        POST /predict/batch     -> {tickers: [...], date?, ...}
```

## Canli kaynak (bayat-skor sorununun cozumu)

Egitim tablosunda (features.parquet) bir gunun label'i yoksa (son
horizon=5 gun) O SATIRLAR DUSULUR; serving o dosyayi kullanirsa canli
skor fiyattan 5 is gunu geride kalirdi (2026-10-29 fiyat, 2026-10-14
skor). Cozum: serving VE gunluk akis artik fiyat deposundan
`features.build.build_features` (LABEL'SIZ) ile anlik feature tablosu
kurar; feature kolonlari ve sirasi training ile BIREBIR aynidir (tek
kod yolu, kopya yok). Skorlanan gun her zaman feature'lari tam
hesaplanabilen EN SON gun; min_tickers_per_date kurali gecerlidir.
`/health` alanlari skor tarihi (score_date) ile veri son tarihi
(data_end_date) ve bayatligi (stale_price_dates) yanyana raporlar.

## Dosyalar ve gorevleri

| Dosya | Gorev |
|---|---|
| config/serving.yaml | kaynak secimi (source: prices) + yollar + host/port |
| src/serving/model_store.py | registry -> yerel fallback; model yeniden insa |
| src/serving/app.py | FeatureSource (parquet ya da fiyat deposu), FastAPI, scoring |
| tests/test_serving.py | 15 test (yedek + canli kaynak + kapanis kurali) |

## Istek/Cevap sekilsleri

### POST /predict
```json
{"ticker": "AAPL", "date": "2026-05-01"}   // date yoksa en guncel gun
```
```json
{
  "ticker": "AAPL", "date": "2026-05-01",
  "score": 0.4818,
  "cross_sectional_pct": 0.63,
  "model_name": "gru"
}
```

- date'a birebir karilik yoksa >> o gunden ONCEKI en yakin islem gunu kullanilir.
- bilinmeyen ticker / pencere yok => 404; feature kolonu eksik => 422;
  model/kaynak yuklenememis => 503.
- POST /predict/batch: ayni sekil; bazi tickers hata verirse that item 'error'
  sahasiyla cekilir (genel crack yok).

## Skor motoru (model turleri)

- ridge  : (N, F) numpy -> predict.
- mlp    : torch tabular; (N, F) tensor.
- gru/lstm : pencere: meta.json'daki seq_len kadar gun (ticker'a ozel dilim);
  (1, T, F) tensor -> tek skor.

## Training-serving skew onlemi

- Feature kolonlari meta.json'dan; FeatureSource.get_row baski kontrolu
  yapar (meta'da listalenen feature kaynakta yoksa 422).
- Model, build_model(model_name, input_dim, model_params) + load_state_dict
  yoluyla yeniden kurulur: params, config defaultlarinla uyumlu; kodla
  uyum (ayni build_model).
- Feature kopyasi olusturulmaz: feature kaynaginin AYNI parquet dosyasi
  (data/processed/features.parquet) hem egitimde hem serving'de tek
  hakikat kaynagi.

## Bilinen sinirlar

- cross_sectional_pct yalnizca tabular modellerde hesaplanir (sequence
  pencere maliyeti; belgeli sinir - istenirse kolayca genisletilir).
- Skorlama, feature parquet hangi kolonlarla uzetmis ise oyle calisir;
  yeni feature listesi meta.json'un feature_cols listesiyle eslesmezse
  422 doner: bilincli davranis (sessiz sapma yerine gorunur hata).
- /predict/batch her ogeyi ayri skorlar; tek process Uvicorn yeterli
  (model kucuk; pencere skoru milisaniye mertebesinde).
