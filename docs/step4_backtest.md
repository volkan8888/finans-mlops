# Adim 4 (FAZ 2) - Backtest

## Akis (ust duzey)

```
holdout_predictions.parquet (training FAZ 1 ciktisi: date,ticker,score,label,forward_return)
                |
                v
        [build_signals]  -- rebalance gunlerinde (her 5 islem gunu) cross-sectional
                |             ust %20 long, alt %20 short; equial agirlik
                v
        [run_backtest]   -- gunluk getiri serisi (gross), turnover, maliyet (net)
                |            look-ahead yok: r(T) agirliklari T-1 kapanisindan
                v
        [performance_metrics]  [compare_to_benchmark]  (SPY buy&hold, beta/alfa)
                |
                v
   reports/backtest_summary.json + reports/equity_curve.csv (gross/net/SPY)
```

## Dosyalar ve gorevleri

| Dosya | Gorev |
|---|---|
| config/backtest.yaml | kantil oranlar, rebalance adimi, maliyet, mod listesi, benchmark |
| src/evaluation/backtest.py | build_signals / run_backtest / apply_costs / metrikler / benchmark |
| src/evaluation/run.py | config loader + akis; JSON + CSV raporlari |
| tests/test_backtest.py | 9 test (sentetik, kapanis-disi hizli) |

## Look-ahead nasil engellendi (en kritik nokta)

- getirimantigi: r(T) = P(T)/P(T-1) - 1 (kapanis-kapanis).
- Getiri gunu T'de kullanilan agirliklar, T-1 kapanisindaki SON rebalance
  sinyalinden gelir. Rebalance gunu p'de uretilen sinyalin etkisi p+1
  getiri gununde BASLAR.
- Rebalance gunu arastirmasi pozisyon (index) bazlindir: `i % every == 0`.
- Sinyal ayni gun kendi getirisine katki VERSINADI degil: motor, onceki
  kapanisin agirliklarini getiri gunune uygular; bu motor testil
  sinyal-gecikme testinin nesnesidir (asagidaki test 2).

Test 2 (gecikme testi) bunu kanitlar: perfect sinyal t'de (t+1)'i biliyorsa
DOGRU motor her getiri gununu +%1 dolar; ayni veri "bir gun gecikmus"
sinyalini kullaninca -%1 (her gun kaybeder). Look-ahead yontemli bir motor ayni veride sabit -1 uretir; gecikme
kavgali/lookahead burada acikca ayirt edilir.

## Maliyeti modeli

- Islem maliyeti: rebalance gununde degisen agirlik toplami |dw| buyukligidir.
- Her birim hacim icin `cost_bps/10000` x 1 (buy OR sell); "tek yon" bedelsize.
- Bedel resmi olarak rebalance'in SONRAKI getiri gununde portfoyden dusulur.
- turnover=0 Hidrebalance: maliyet sifir (test 3 kapsar); cost_bps arttikca
  net getiri monoton dusus (test 3 kapsar).

## Drift ve basitlestirmeler

- Rebalance araliklarinda agirliklar SABIT kaydettilir (his fiyari rotasyonu
  absanmezzet): equal-weigh kucuk portfoyumuz icin yeterli basitlik; oyuncu
  degeri gocucte degism units.
- Askida/eksik fiyat verisi olan gun-katki: 0 katki (sayac loglanir).
- Long-short = dolar-notr (leg basina +/- 0.5); long-only = tam +1.
- Automatik survivorship: UNIVERSE = bugunun 44 hissesi; bu bias tam
  modelde yoktur (Adasi Nots: "Survivorship bias" a bak).

## Metrik anlamlari

| Metrik | anlam | not |
|---|---|---|
| total_return | kumulatif kazanaran | (1+r).prod()-1 |
| cagr | yilliklesmis getiri | 252 islem gunu bazli |
| ann_volatility | yilliklesmis std | std*sqrt(252) |
| sharpe | getiri/volatilite | mean/std*sqrt(252) (risksiz faiz 0) |
| sortino | getiri/kotunluk-vol | sadece negatif gunler |
| max_drawdown | tepe-dip | (1+r).cumprod tepeye gore |
| calmar | cagr / |maxDD| | risk-ayarli |
| win_rate | pozitif gun orani | |
| avg_turnover_per_rebalance | rebalance basina | maliyet olcegi |
| beta / alpha | SPY'a gore | gunluk alfa yillilastirilir |

## Bilinen sinirlar

- Survivorship bias: bugunun 44 ticker'indan yaptigimz universum; gecen
  hisseler ELEMENTS.
- Slippage/komisyon/vergi yok: yalnizca yuvarlak bir bps maliyet.
- Sparse universe+20 yil: kantil basina ~9 hisse; taban dar.
- Holdout 2 yillik; tek donem — istatistiksel kesinlik icin kisa bir pencere; dikkatli yorumlanmali.

## Gercek veri sonucu (FAZ 2 dogrulamasi; 2024-10-15..2026-10-14, 501 gun)

GRU (secilen model) holdout tahminleri, top/bottom %20, 5 gun rebalance,
10 bp tek-yon maliyet (reports/backtest_summary.json):

| Mod | Brut toplam | Net toplam | Net sharpe | Net maxDD | Ort. turnover/rebal. |
|---|---|---|---|---|---|
| long_short | -0.7% | -8.8% | -0.33 | -15.7% | ~0.85 |
| long_only  | +46.7% | +37.2% | 0.89 | -21.5% | ~0.67 |

Benchmark SPY ayni donem: +34.7% total, sharpe 0.99, maxDD -18.8%.
long_only, beta ~0.85 ile brut alfa ~+7.5%/yil; net alfa ~+5-6%/yil. AMA
holdout IC t-stat 0.78 (anlamli degil): bu ayristirmanin istatistiksel
guvence yok; sonuc buyuk olcude beta yorum.

Maliyet dengelemesi: ~100 rebalance x ~0.85 turnover x 10bp = ~8puan
long_short'ta yedigi maliyet (brut ~0'dan net -8.8'e). Rebalance adimo
artirtmak ya da maliyet disi modellerme surekli asma calismasi.

DURUST KARAR: sinyal zayif; long_shortumuz zararda (alt %20 ayni zamanda
en iyi getiren grup'ta: 5-gunluk reversal etkisi; bu modelde kullanilan
momentum-tarzi feature'larla uyumlu). long_only SPY'a fibires yakin;
net alfa pozitif gorunse de t-stat yetersiz. Test-setine bakarak ayar
yapmadik; rapor bu haliyle sonuctur.
- Sinyal zayif cikarsa SPY gercekte yenilebilir; bu icerik lamaitan icin.
