# Adim 11 (FAZ 9) - Son rapor

## Faz kayidi (bir bakis)

| Faz | Kapsam | Sonuc / dogrulama |
|---|---|---|
| 0 | temizlik, bagimliliklar, metodoloji kurallari | z-score ddof=0; purge+embargo 5/5; holdout son 2 yil tek atis; shuffle yok |
| 1 | egitim altyapisi (config/splits/dataset/models/train/evaluate/run) | 15 test; gercek kosu: GRU secildi |
| 2 | backtest motoru (lag-guvenli, maliyetli) | 9 test; gercek kosu: reports/ altinda raporlar |
| 3 | MLflow tracking + registry + alias | 5 test; mlflow.db 4 run; registry'de 'production' alias |
| 4 | serving (model_store + FastAPI) | 10 test; registry'den GRU yukleme + skorlar |
| 5 | monitoring (PSI/KS + kayan IC) | 10 test; rapor: uyari yok, tum PSI < 0.02 |
| 6 | Prefect orkestrasyon | 7 test; gunluk akis canli kosu (exit 0) |
| 7 | Streamlit dashboard | 10 test; headless smoke health = ok |
| 8 | Docker + CI | Dockerfile/compose/CI yaml; tum repo ruff temiz |
| 9 | README + uc-uca smoke + son rapor | 105 test; smoke_e2e gecti (exit 0) |

## Gercek veri karnesi (durust)

- Secim (walk-forward val IC, 1635 gun): GRU 0.0175 (t 2.75), MLP
  0.0140 (t 2.75), Ridge 0.0105 (t 1.57 - anlamsiz).
- HOLDOUT (tek atis, 501 gun): mean_ic 0.0082, t 0.78 - ANLAMSIZ.
  Kantil getiri tablosu (gunluk ort.): q0 0.47%, q1 0.19%, q2 0.29%,
  q3 0.41%, q4 0.51%. Alt kuantil ile ust kuantil ayni seviyede:
  kisa bacak ters tepti (5-gunluk reversal).
- Backtest (10 bp tek-yon maliyet, 5 gun rebalance): long_short net
  -8.8% (sharpe -0.33) - zararda. long_only net +37.2% (sharpe 0.89)
  vs SPY +34.7% (0.99): buyuk olcude beta (beta ~0.85; brut alfa
  ~+7.5%/yil ama t-stat istatistiksel olarak yetersiz).
- Serving: registry versiyonu + 'production' alias; yerel fallback ile
  ayni skor (reconstruct dogru; AAPL 0.4911).
- Monitoring: dev->holdout feature kaymasi yok (tum PSI < 0.0156);
  son 20-gunluk kayan IC 0.118 (son pencere pozitif); uyari listesi bos.
- Ops: gunluk Prefect akisi canli; retrain zinciri tanimli (elle tetik).

## Sinirlar (belgelenmis)

- Universum hayatta kalan 44 hisse: survivorship bias var.
- Label horizon 5 gun: sinyal-getiri eslesmesi gunluk getiri konvansiyonunda.
- Backtest'te agirliklar rebalance aralarinda sabit tutulur (drift yok);
  maliyet modeli yuvarlak 10 bp tek-yonundan ibaret.
- Tek atis holdout: 2 yil penceresi istatistiksel guc icin dar.

## Sonraki is fikirleri (onerilen; yapilmadi)

- Feature set genisletme: sektor dengeli gruplar, hacim dinamikleri, makro.
- Rebalance araligi ve maliyet varyanti denemesi - YENI bir gelistirme
  devri ile (holdout'a ikinci kez bakilmaz; bu kural sabittir).
- listnet tipi siralama kaybini tam devreye alma (config'de alternatif var).
- Prefect deployment + zamanlama (cron) ve eski run arsivleme.

## Kalite-islemi ekleri (2026-10-29)

### IC istatistikleri: ham t vs Newey-West t vs bagimsiz ornekleme t

Arka plan: label 5 is gunluk ileri getiri; ardisik gunlerin IC'leri
cakisir (bagimsiz ornek sayisi < gun sayisi). Klasik t (iid varsayimi)
bu cakismada SISKIN olabilir. Newey-West (Bartlett, lag = horizon-1 = 4)
otokorelasyonu cezasina katar; "bagimsiz ornekleme" adim olarak her
5. gunde bir IC alir (label'lar cakismaz).

Holdout (GRU, 501 gun; `scripts/ic_report.py`, kayitli tahminlerden
yeniden hesap - MODEL EGITILMEDI):

| seri            | n   | mean_ic  | t (ham) | t (NW, lag=4) | n bagimsiz | mean_ic (bagimsiz) | t (bagimsiz) |
|-----------------|-----|----------|---------|---------------|------------|--------------------|--------------|
| holdout (yeni)  | 501 | +0.0082  | +0.78   | +0.45         | 101        | +0.0057            | +0.26        |
| egitim-kaydi    | 501 | +0.0082  | +0.78   | - (seri yok)  | -          | -                  | -            |

Walk-forward VAL (1635 gun; ham t: GRU 2.75, MLP 2.75, Ridge 1.57):
val tahmin serisi DISKE YAZILMAMISTI; NW/bagimsiz t GERIYE DONUK
hesaplanamaz (durust kayit). Bu yuzden val'in NW t'si raporlanamaz.

Yorum:
1. Holdout'un karari degismez: ham t zaten 0.78 idi (anlamsiz); NW
   duzeltmesi 0.45'e, bagimsiz ornekleme 0.26'ya indirir. Sinyalin
   holdout'ta anlamsiz oldugu yaklasimi daha da guclendi.
2. Val t=2.75 cakisma riski tasir (purge/embargo fold'lar arasindadir,
   ancak val gunlerinin label'lari kendi icinde 5 gun cakisir). Ham
   t'ye gore NW t muhtemelen daha kucuk olacak; model secimi zaten
   fold'larin siralamasina (mean_ic) dayandigi icin secim etkilenmez.
   Bu riski ileride ortadan kaldirmak icin: egitim zamani val
   tahminlerini diske yaz (sonraki retrain'de NW t otomatik hesaplanir).
3. Monitoring/dashboard 't_nw' kolonu ekledi (lag config: nw_lag=4).
4. Ileri test altyapisi hazir (2026-10-29): freeze imzasi alindi
   (gru v1, model sha8 50241b34) - reports/forward_test/freeze.json +
   criteria.md sablonu; monitoring.forward canli tahminleri D+5
   pencerede gerceklestirir (unrealized sayar). Ilk kosuda 44 tahminin
   hepsi 'tum_beklemede' (pencere dolmadi); tahmin kayitlari biriktikce
   rapor dolacak. Dashboard'a "Ileri Test" sekmesi eklendi.

## Yonlendirme

- Adim belgeleri: docs/step3_training.md ... step10_docker_cicd.md.
- Durum karnesi: docs/PROGRESS.md (tek kaynak).
- CI: .github/workflows/ci.yml (lint-test + docker-build).
