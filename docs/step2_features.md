# Adım 2 — Feature Pipeline (belgelendirme)

Amaç: ham fiyat verisini (adj_close, volume) geleceğe bakan bilgi içermeyen,
gün içinde karşılaştırılabilir özelliklere (feature) dönüştürmek; eğitim sürümünde
(build_dataset) bir adım ileri getiri etiketi (label) eklemek.

## 1. Akış (metin diyagramı)

```
data/raw/prices.parquet               (ingestion çıktısı; 44 hisse + 2 benchmark)
        |
        v
[run.py]  config/features.yaml yükle -> fiyatları parquet'ten oku
        |
        v
[build.py :: _build_core]    (TEK akış; serving ve eğitim AYNI kodu çağırır)
   1) benchmark filtrele         (^GSPC, SPY evrene girmez)
   2) (ticker, date) sırala      -> her hisse kendi kronolojik serisine ayrılır
   3) technical.py fonksiyonları -> ticker içi HAM feature hesabı
      (rolling/shift YALNIZCA GEÇMİŞE bakar)
        |
        |  (+ labels.forward_return: shift(-n) SADECE burada)
        v
   4) warm-up satırlarını düşür -> pencere döneminin NaN'leri ve sıfır
      hacimli günler çıkar (sayısı loglanır; fillna YASAK)
   5) min_tickers_per_date      -> az hisseli günler atılır (loglanır)
   6) cross_section.winsorize_by_date   (o günün kendi quantile'ları)
   7) cross_section.zscore_by_date      (o günün kendi ort./std'si; std=0 -> 0)
        |
        +--> build_features  (serving): [date, ticker, feature'lar]  (label YOK)
        +--> build_dataset  (eğitim):  aynısı + "forward_return" + "label"
        |
        v
data/processed/features.parquet
```

## 2. Dosyaların görevi

| Dosya | Görev |
|---|---|
| `config/features.yaml` | Pencere ve zaman ufku (horizon) değerleri, eşikler, yollar |
| `src/features/config.py` | `FeaturesConfig` + doğrulamalı loader; benchmark listesi önce features.yaml'dan, yoksa data.yaml'dan okunur |
| `src/features/technical.py` | Ticker-bazlı saf fonksiyonlar (pd.Series girer/çıkar). Hepsi geçmişi görür; `shift(-n)` yok |
| `src/features/cross_section.py` | Aynı günün hisseleri arasında winsorize + z-score (groupby("date")); girdi DataFrame'i değiştirmez |
| `src/features/labels.py` | forward_return + gün içi cross-sectional rank (0-1). `shift(-n)` SADECE burada |
| `src/features/build.py` | `build_features` (serving) ve `build_dataset` (eğitim); ikisi de aynı `_build_core` akışını çağırır |
| `src/features/run.py` | fiyat oku -> build_dataset -> `data/processed/features.parquet` yazar + özet log |

## 3. Feature'ların finansal anlamı

| Kod adı | Hesap | Ne ölçer | Not |
|---|---|---|---|
| `momentum_5/20/60` | P_t / P_{t-w} - 1 | Trend momentum'u: hissenin son w gündeki net getirisi | Literatürdeki klasik momentum sinyaline benzer; w büyüdükçe daha uzun vadeli eğimi yakalar |
| `momentum_skip_60_5` | P_{t-5} / P_{t-60} - 1 | Son 5 gün hariç 60 günlük getiri | "12 ay eksi 1 ay" mantığının günlük hali: son günlerdeki aşırı sıçrama kaynaklı düzeltme riskini azaltır |
| `volatility_20` | Günlük (log) getiri std'si (20 gün, ddof=1) | Risk/oynaklık | Yüksek volatilite daha riskli kabul edilir; düşük volatilite de bilgi taşır |
| `rsi_14` | Wilder yumuşatmalı RSI (ewm, alpha=1/14) | Aşırı alım/satım | Sürekli yükselen seri -> 100; sürekli düşen -> 0; düz seri -> 50 (koşe durumları maskelenir) |
| `volume_change_20` | ln(hacim / 20 gün ort. hacim) | İlgi/haber yoğunluğu (anormal hacim) | Hacim 0 günlerde -inf yerine NaN verilir ve satır düşürülür (doldurma yasak) |
| `reversal_5` | Son 5 günün log getiri toplamı | Kısa vadeli tersine dönüş sinyali | İşaret bilinçli olarak TERS çevrilmez; ham hali verilir ve yönü model kendi öğrenir |

## 4. Sızıntının (lookahead) önlendiği yerler

1. **Geleceğe bakan tek işlev: `labels.py::forward_return`** — projede
   `shift(-horizon)` sadece bu fonksiyonda var. Feature fonksiyonlarının hepsi
   geçmişi gören hesaplardır.
2. **Cross-sectional işlemler sadece aynı gün içindir** — winsorize, z-score ve
   rank her gün yalnızca o günün verisiyle hesaplanır. Yarın bir hissenin
   değerini değiştirmek bugünkü z-score sonuçlarını değiştirmez.
3. **Ticker izolasyonu** — rolling/shift her zaman groupby("ticker") parçası
   içindedir; hisse fiyat serileri birbirine karışmaz. (Not: cross-sectional
   z-score, aynı günün hisselerinin birbirini etkilemesi BİLİNÇLİ bir
   tasarımdır; bu ticker karışması değildir.)
4. **Warm-up doldurma yok** — pencere dönemindeki NaN'ler ffill/bfill
   yerine DÜŞÜRÜLÜR ve düşürme sayısı loglanır. Yapay doldurma sahte sinyal
   üretir.
5. **Training-serving skew (eğitim/sunum sapması) engellenir** —
   `build_features` ve `build_dataset` AYNI `_build_core` fonksiyonunu
   çağırır; tek fark `with_labels` parametresi. Kod tekrar yok; serving
   anında bugünün feature'ları hesaplanabilir.

## 5. Bilinen sınırlar ve riskler

- **z-score ddof secimi**: z-score artik DUZENLI (ddof=0, populasyon) std
  kullanir: gun icindeki hisselerin std'si tam 1 olur; tam paneli
  `describe()` ile bakinca da std ~ 1.000 cikar.
  Ilk surumde ddof=1 (ornekleme std) kullanilmisti: her gun (n-1) bazli
  standartlastirma yapilir, panel toplamının std'si ~0.9886 cikar
  (sqrt(43/44) artefakti). Bu yalnizca hesaplar arasi ddof farkidir
  (gundeki n=44, panelde n=127732); IC gibi korelasyon olcumlerini
  etkilemez.
- **Sıfır hacimli günler**: `volume_change` NaN olur ve satır
  "warm-up / eksik veri" aşamasında düşürülür. Adım-1'deki anomali raporundaki
  AMD 2015-01-02 (sıfır hacim) ürneği bu kapsamdadır.
- **Survivorship bias**: `universe.txt` bugünkü hisseler listesinden türetildi;
  2015'ten bu yana borsadan çıkan (delisted) hisseler panelde yok. İleride
  düzeltilmeli.
- **Veri revize riski**: yfinance geçmiş kapanış değerlerini sonradan
  gözden geçirebilir (split/dividend düzeltme); ingestion'daki same-day refresh
  bunu yakalar (merge keep=last).
- **min_tickers_per_date eşik sabitliği**: 10 (tek değer) kullanılır; küçük
  evrenlerde yeniden düşünülmeli.
- **Benchmark'lar çıktıda yok** — ancak ham fiyat depoda durur; ileride
  backtest'te piyasa karşılaştırması (referans) olarak kullanılabilir.
