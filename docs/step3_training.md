# Adim 3 (FAZ 1) - Egitim Altyapisi

## Akis (ust duzey)

```
features.parquet
      |
      v
[holdout_split]  -->  dev (kalan)             +  holdout (son N yil, config'ten)
      |                            |           (tuning'de KULLANILMAZ)
      v                            v
[walk_forward_splits]   fold fold: egitim + val IC
      |                       |
      |                       v
      |                 [model secimi: SADECE fold val IC ortalamasi]
      v
[final egitim: dev'in son val dilimiyle early-stop]
      |
      v
[HOLDOUT tahmini: BIR KEZ] --> data/processed/holdout_predictions.parquet
      |                        models/<model>/model.pt|joblib + meta.json
      v
 raporlar (log) + MLflow (FAZ 3)
```

## Dosyalar ve gorevleri

| Dosya | Gorev |
|---|---|
| config/training.yaml | seed, bolme parametreleri, model listesi, mlflow ayarlari |
| src/training/config.py | TrainingConfig + dogrulamali loader (features config tarzi) |
| src/training/splits.py | holdout_split, walk_forward_splits (SAF: tarih girer, tarih cikar) |
| src/training/dataset.py | TabularDataset (MLP/ridge), SequenceDataset (GRU/LSTM) |
| src/training/models.py | MLPRanker, RecurrentRanker (+ LSTM/GRU varyanti), RidgeBaseline |
| src/training/train.py | Trainer (el-yazimi dongu), train_model giris noktasi |
| src/training/evaluate.py | daily_ic (Spearman), ic_summary, quantile_returns |
| src/training/run.py | ana akis + artefakt kaydetme |
| tests/test_training.py | 15 test (ag yok, sentetik, hizli) |

## Sizinti onlemleri (nasil uygulandi)

1. **Purge**: label ufkunu t+5 islem gunune yaydigi icin fold sinirinda
   train'in son satirlarinin label'i val'e sirar. Cozum: train, val'in ilk
   gununden `purge_days + embargo_days` islem gunu ONCE biter
   (walk_forward_splits: `train = dates[:val_pos - buffer]`).
2. **Embargo**: purge ustune ekstra 5 islem gunu (config'ten); bilgi tasmasi
   payini daha da kucultur.
3. **Holdout**: son 2 takvim yil (`holdout_years`). Walk-forward VE model
   secimi bunu hicbir formda kullanmaz. Holdout en sonda BIR KERE
   degerlendirilir. dev-vs-holdout sinirinda da purge+embargo uygulanir.
4. **Shuffle yok**: DataLoader'lar shuffle=False; veri zamana gore bolunur.
   (Mini-batch sirasi bile kronolojik; determinism ve kural uyumu.)
5. **Ek olceklemesi yok**: feature'lar zaten gunluk cross-sectional
   z-score'lu (Adim 2). Ayni transformun train'de fit edilmesi de gundeme
   gelmedi; fold'lar arasinda grup istatistigi kullanmadik (model girdisi
   gun icinde zaten standart). Ridge de dogrudan z-score'lu kolonlarla
   besleniyor; ek bir StandardScaler YOK (gereksiz: kolonlar ayni olcekde).
6. **SequenceDataset pencere disiplini**: pencere [t-seq_len+1 .. t] - SADECE
   t ve oncesi; gelecegi icermez. Ticker karismaz. Pencere tam degilse
   ornek atlanir (doldurma yok).
7. **Seed**: random/numpy/torch set_seed(42) ile sabit; ayni komut ayni sonuc.

## Walk-forward semasi (metin)

```
dev: 2015-03-31 ................. 2024-10-01   holdout: 2024-10-15 .. 2026-10-14
     |-- train (expanding) ---|-- val 6 ay --|
fold 01: train 2015..2018-02  | val 2018-03..2018-08
fold 02: train 2015..2018-08  | val 2018-09..2019-02   (6 ay kaydirilir)
...
fold 13: train 2015..2024-02  | val 2024-03..2024-08
```

- val pencereleri cakismaz (step = uzunluk = 6 ay).
- purge/embargo: her fold'da train'in ucu, val baslangicindan tam 10 islem
  gunu once biter.
- Cok kisa kalan son val penceresi (<10 gun) atlanir: gurultu olacakti.

## Metrikler: IC / ICIR

- **daily IC**: bir gunun skorlari ile label'lar arasindaki Spearman
  korelasyonu. Gun bazinda siralama kalitesinin dogrudan olcusu.
- **mean IC**: IC serisinin ortalamasi. Basit derece: 0.02-0.05 arasi
  gercek dunyada olasilikli kabul edilir; 0.05+ guclu sayilir.
- **ICIR**: mean(IC)/std(IC) - sinyalin tutarlilik orani.
- **t-stat**: mean/std*sqrt(n) - gun sayisina gore basit anlamlilik.
- **positive_share**: IC>0 gun orani.
- **kantil getiri**: skor kantisinin ortalamasi, ham forward_return uzerinden.
  Monoton olmali; olmuyorsa siralama ogrenilmamis sayilir.

## Modeller

- **ridge**: lineer baseline; 8 z-score'lu feature'in lineer kombinasyonu.
  Basit ama cok zorlayici bir baz.
- **mlp**: Tabular girdi, [64,32] hidden, ReLU + dropout.
- **gru**: son 20 gunun feature sekansini alir; GRU son gizli durumdan tek
  skor uretir. (LSTMRanker/GRURanker siniflari models.py'de var; config'e
  lstm girilebilir.)
- Hepsi TEK SKOR uretir: cross-sectional siralama icin.

## Bilinen risk ve notlar

- Sinyal zayif olabilir: gercek veride mean IC ~ 0.02 civari beklenmelidir;
  maliyetli backtest SPY'i gecmeyebilir. Bu durumda sonuc durust sekilde
  raporlanir (cok calisip test setine bakmadan ayar yapma maddenin ruhudur).
- Final egitimde son val diliminden DAHA YAKIN dev tarihleri training'de
  kullanilmaz (tutarli davranis; kucuk bir veri kaybi, belgelendirildi).
- Kosu butcesi: gercek veride fold basina her model; epoch sayilari config
  defaultlariyla kucuk tutuldu; toplam calisma ~30-40 dakika CPU.
