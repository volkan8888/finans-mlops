# Adim 5 (FAZ 3) - MLflow Deney Takibi + Model Registry

## Backend ve konumlar

- Backend: `sqlite:///mlflow.db` (mutlak yol: proje koku; yardimci fonksiyon
  `resolve_tracking_uri` goreli URI'yi kokune baglar).
- Artifaktlar: `mlruns/` (yerel klasor; gitignored).
- Config: `config/training.yaml` -> `mlflow:` bolumu
  (enabled / tracking_uri / experiment / registry_name).

```
config/training.yaml
  mlflow:
    enabled: true
    tracking_uri: sqlite:///mlflow.db
    experiment: finans-training
    registry_name: finans-ranker
    artifact_dir: mlruns
```

## Calistirildiginda ne yazilir

```
experiment: finans-training
  run: variant-ridge | variant-mlp | variant-gru     (her walk-forward varyanti)
        param_model, param_hidden, param_alpha, ...
        fold01_val_ic ... fold13_val_ic
        mean_ic, std_ic, icir, t_stat, positive_share, n_days
  run: production-gru                                (secilen model; FINAL)
        param_model
        final_val_ic, holdout_mean_ic, holdout_icir, holdout_t_stat,
        holdout_positive_share
        bt_long_short_net_total_return, bt_long_short_net_sharpe,
        bt_long_short_net_max_drawdown, bt_long_only_* (FAZ 2 motoru)
        artifacts/model/{meta.json, model.pt, holdout_predictions.parquet}
```

- Secim kriteri metrikleri sadece walk-forward VAL IC'den gelir; holdout
  degerler bilgilendirme icin loglanir ama modele secim kararina girmez
  (akis siralamasi sabit).
- NaN metrikler yazilmaz (log dogrulugu).
- Liste/dict param degerleri string'e cevirilir (MLflow skaler ister).

## Model registry + production alias

- Eger registry'de model yoksa once `create_registered_model` acilir.
- `create_model_version(source=<production-run artifact URI>, run_id=...)`
- `set_registered_model_alias('finans-ranker', 'production', version)`:
  eski 'stages' API'si kullanilmaz (mlflow>=2.9 onerisi alias'tir).
- Registry erisilemezse: hata loglanir, dosya artifaktlari yerinde kalir;
  egitim akisi bozulmaz.

## Serving tarafindan tuketim (src/serving/model_store.py)

`load_production_model()` -> (model, meta):

1. MLflow registry'den alias 'production' versiyonunu bulur
   (`get_model_version_by_alias`).
2. Kaynak yerel file:// URI ise dosya yolu; degilse artifaktlari indirir.
3. `meta.json` + model dosyasindan modeli yeniden KURAR:
   - torch: `build_model(model_name, input_dim, model_params)` +
     `load_state_dict` (ayni insa kadar yol yani training-serving skew yok).
   - ridge: joblib.
4. MLflow olmazsa yerel fallback: `models/*/meta.json` en-yeni dizin.

Inspect GUI: proje kokunden `mlflow ui --backend-store-uri sqlite:///mlflow.db`
sonra tarayicida http://127.0.0.1:5000.

## Testler (tests/test_mlflow.py, 5 adet)

- `test_reconstruct_model_boyut_dogru`: meta.json+model.pt -> model kurulur
  (bilinen 0/1 agirlardan dogrulanir).
- `test_load_production_model_registry_then_fallback`: regsiter + alias ->
  model yukleme (tmp sqlite backend).
- `test_load_production_model_fallback_yerel`: bozuk registry URI ->
  yerel models/*/ klusorunden yukler.
- `test_setup_tracking_kapalirken_none`: enabled=false -> None (aksama degil).
- `test_resolve_tracking_uri_goreli_sqlite`: goreli -> mutlak; mutlak -> ayni.
