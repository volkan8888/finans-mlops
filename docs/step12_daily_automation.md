# Adim 12 - Gunluk ileri test otomasyonu (GitHub Actions, bilgisayar kapali)

Amac: is gunlerinde tahmin uretimi bilgisayarsiz otomatik aksin.
Model, egitim, features, backtest ve ingestion KODUNA dokunulmadir.
Akis mevcut `scripts/run_daily.py --mode daily` zincirini CI'da kosar
ve urunleri `forward-data` dalinda biriktirir.

## Akis semasi

```
schedule (Pzt-Cum 22:30 UTC) / workflow_dispatch
        |
        v
[1] checkout master          -> kod + config + models/gru/meta.json
[2] python 3.12 + torch CPU + pip install -e ".[dev]"   (ci.yml ile ayni)
[3] forward-data dali kontrol (varsa forward-store/ klasorune checkout)
[4] state geri yukle: tahmin parquet'leri (NO-CLOBBER) +
      data/processed/features.parquet +
      data/processed/holdout_predictions.parquet + son raporlar
[5] model indir (release): gh release download model-v1 -> models/gru/model.pt
[6] hash dogrulama: scripts/verify_model_hash.py (freeze.json ile)
      | uyusmazsa kosu BASARISIZ (yanlis modelle tahmin uretilmez)
      v
[7] ingestion (3 deneme): python -m ingestion.run; yfinance gecici
      | hata -> 3 deneme (15 sn arayla); yine olmazsa BASARISIZ
      v
[8] yeni-bar kilidi: scripts/check_new_bar.py
      | bar=none (tatil / tahmin zaten var) -> exit 0, TEMIZ CIKIS
      | bar=new -> devam
      v
[9] gunluk akis: scripts/run_daily.py --mode daily
      | (validasyon -> features -> skor -> tahmin kaydi -> monitoring)
[10] ileri test raporu: python -m monitoring.forward
      | reports/forward_test/forward_report.json
      v
[11] forward-data dalina kayit: yeni tahmin parquet (append-only) +
      guncel forward_report.json + monitoring_report.json; push.
      Eski bir tahmin dosyasi degisir ise adim BASARISIZ.
```

MLflow notu: CI'da `mlflow.db` YOK. `serving.model_store` bunu normal
akisin PARCASI olarak loglar: "MLflow'dan production model alinamadi ...
yerel fallback devrede" (logger.warning). Sonra modeli `models/gru/`
klasorunden (meta.json + model.pt) kurar. Bu uyari BEKLENIR; hata degil.

## Cron saati ve DST

- Cron: `30 22 * * 1-5` (Pzt-Cum 22:30 UTC). GitHub cron UTC'dir; deger
  DEGISMEZ.
- ABD borsa kapanisi 16:00 America/New_York:

```
yaz (EDT): 16:00 ET = 20:00 UTC -> 22:30 UTC = 18:30 ET (kapanis + 2,5 saat)
kis (EST): 16:00 ET = 21:00 UTC -> 22:30 UTC = 17:30 ET (kapanis + 1,5 saat)
```

- Sonuc: 22:30 UTC, yaz/kis ne olursa olsun kapanis SONRASIDIR. Gunun
  bar'i tamamlanmis; kismi oturum riski yok. (Kapanis-oncesi kismi-bar
  korumasi ayrica kodda mevcut: docs/step8_orchestration.md.)
- `check_new_bar.py` ikinci guvence: tatil gunlerinde (yeni bar yok)
  kosu exit 0 ile temiz cikar; eski tarihli skor uretmez.

## Model: GitHub Release'den

- Asset: **model-v1** release icinde `model.pt` (tek asset).
- Workflow `gh release download model-v1 --pattern "model.pt"` ile indirir.
  `GITHUB_TOKEN` yeterli: EK SECRET YOK.
- Indirme sonrasi sha256, `reports/forward_test/freeze.json` ->
  `model.sha256.model_files["model.pt"]` ile karsilastirilir. Uyusmazsa
  kosu BASARISIZ. Freeze'de tam hash yoksa script sha8 (ilk 8 hex) ile
  karsilastirir ve "ZAYIF karsilastirma" uyarisi loglar.
- Referans hash (freeze 2026-10-30, gru v1):
  `50241b342bf7a9cc1b67366811a1f7b2a036a75275796e93cd28358271f17350`
  Yerel `models/gru/model.pt` ile birebir aynidir. Release'e koydugun
  dosya freeze'tekiyle AYNI dosya olmali; hash kontrolu bunu garantiler.

## forward-data dali (append-only tahmin deposu)

- Master'da `data/` ve `models/*/model.pt` gitignore'ludur. Tahminler
  dale `git add -f` ile eklenir; master degismedik.
- Her kosu master'a DEGIL, sadece `forward-data` dalina push. Dal yoksa
  ilk kosu orphan dal acarak devam eder.
- Tahmin dosyasi adi: `predictions_YYYY-MM-DD.parquet`. Dosyalar ASLA
  ezilmez:
  - kod seviyesi: daily_flow `task_record_predictions` no-overwrite;
  - workflow seviyesi: geri yuklemede `cp -n`; kayit oncesi
    `git diff origin/forward-data -- data/predictions` M/D kontrolu.
    Eski tahmin degisir veya silinirse kosu BASARISIZ.
- Raporlar (forward_report.json, monitoring_report.json) gunluk
  yenilenen OZETLER. Bunlar her kosuda yeni surumle guncellenir
  (ezilme tasarim icin SADECE raporlar icin; tahmin parquet icin YOK).
- Concurrency: `group: daily-forward`, `cancel-in-progress: false`.
  Iki kosunun ayni dala ayni anda push etmesi engellenir.
- `timeout-minutes: 30`.

## Ilk kurulum (elle yapilacaklar)

1) Release olustur (local, repo kokunden):
```
gh release create model-v1 models/gru/model.pt ^
  --title "gru model.pt (freeze 2026-10-30)" ^
  --notes "sha256 50241b342bf7a9cc1b67366811a1f7b2a036a75275796e93cd28358271f17350"
```
2) State dal kur (bir kere; dosyalar hala local diskte):
```
git fetch origin
git switch --orphan forward-data
git rm -r --cached . -q
git add -f data/processed/features.parquet data/processed/holdout_predictions.parquet
git commit -m "forward-data: referans state (features + holdout tahminleri)"
git push origin forward-data
git switch master
```
Dosyalar yoksa:
- `data/processed/features.parquet`: `python -m features.run` uretir.
- `data/processed/holdout_predictions.parquet`: son egitimin urunudur;
  local kopi var ise dale eklenir.
3) Ilk kosu: Actions -> daily-forward -> Run workflow (workflow_dispatch).
4) Repo Settings > Actions > Workflow permissions: "Read and write
   permissions" secili olmali (GITHUB_TOKEN push yetkisi icin).

## Sik hatalar ve cozumleri

| Belirti | Neden | Cozum |
|---|---|---|
| "forward-data dali yok" (state adimi) | ilk kurulum yapilmadi | "Ilk kurulum" adim 2 |
| "features.parquet / holdout_predictions.parquet yok" | referans dosyalar dala eklenmemis | adim 2 komutlari |
| "Model hash dogrulama BASARISIZ" | release assette yanlis/eski model.pt | freeze hashli dosyayi tekrar release et |
| "ingestion 3 denemede basarisiz" (kod=1) | yfinance kapasitesi ya da gecici ag hatasi | 15-30 dk sonra rerun; tekrar yapilir |
| "ingestion kod=2" (dogrulama/veri hatasi) | kalici veri sorunu; tekrar deneme anlamsiz | logda ticker/tarih satirini inc; cozulmeden rerun yarar etmez |
| "HATA: eski tahmin dosyasi degisti" | elle mudahale veya bozuk kopya | dosyayi forward-data dalinda eski haliyle onar |
| push reddedildi | Workflow permission ayari | Settings > Actions > Workflow permissions |
| hafta sonunda kosu yok | cron Pzt-Cum | tasarim geregi; sorun yok |

## Eksik son bar toleransi (6 Kasim 2026 olayi sonrasi)

Neden: yfinance bazi sembollerde son barin close/adj_close degerini
gecici olarak NaN verebilir (kapanis verisi gec yayilir). Eski davranis
bu satirlari validate_prices'ta SERT hata yapar, kosu 3 kez ayni hatayla
tekrarlanirdi (6 Kas 2026 koysunda yasandi: 2 ticker'in son satiri NaN,
logda ticker/tarih bos cikiyordu).

Yeni davranis (`src/ingestion/run.py`, dogrulamadan once):

- Yalnizca EN SON tarihte: NaN close/adj_close satirlari atilir ve
  WARNING loglanir (ticker listesi dahil).
- Eksik hisse sayisi evrenin %10'unu ASARSA o tarih KOMPLE atilir
  (`config/data.yaml` -> `ingestion.max_incomplete_latest_share`, 0.10).
  Veri UYDURULMAZ; akis yoksa bir sonraki tarihe kadar o gunun bar'ini
  kullanmaz.
- Benchmark (^GSPC, SPY) satiri NaN ise yalnizca o satir atilir; eksik
  sayaca katilmaz, komple-atma tetiklenmez.
- Gecmis tarihlerdeki NaN hala SERT hata (validate_prices degismedi).

Kendinden duzelme (self-healing): atilan satirlar bir sonraki kosuda
"ayni-gun tazeleme" penceresiyle yeniden cekilir; kaynak duzeldiginde
depo eksiksiz dolus. Son tarih komple atilirsa depo bir gun geride
kalir; `check_new_bar` "none" der ve o gun icin tahmin uretilmez, kosu
TEMIZ biter (bos/uydurma tahmin yok).

Cikis kodlari ve retry iliskisi:

| kod | anlam | daily.yml davranisi |
|---|---|---|
| 0 | ingestion tamam | devam |
| 1 | ag/indirme/beklenmeyen calisma hatasi | 3 deneme (15 sn arayla), sonra BASARISIZ |
| 2 | dogrulama/veri/config hatasi | dongu aninda bitirilir; BASARISIZ (tekrar yok) |

Dogrulama hata logu artik satir basina TICKER ve TARIH yazar
(`satir=NN ticker=XX tarih=YYYY-MM-DD | kolon=... | kontrol=...`); daha
once yalnizca deger/kolon/kontrol yaziyordu, ticker/tarih cikmiyordu.

## Zamanlayici guvenilirligi (onemli not)

GitHub Actions zamanlayicisi gecikebilir; bazen bir gunun kosusunu KOMPLE
ATLAYABILIR (yogun Actions saatlerinde gorulur). Bu durumda:

- Atlanan gun GERIYE DONUK URETILMEZ. Sonradan eski tarihli tahmin
  dosyasi uretmek ileri test butunlugunu bozar; YASAK.
- Bosluk raporda gorunur: `forward_report.json` -> `coverage` (tahmin
  gunleri listesi); iki ardisik tarih arasinda bosluk varsa ilgili gun
  atlanmistir. criteria.md donem sonu olcumu bundan okunabilir.
- Manuel tetik (workflow_dispatch) VAR. Gecikmis gun icin AYNI gunden
  kalinca kosu yapilabilir; ayni tarihin tahmini zaten varsa akis onu
  korur (no-overwrite), yeni tahmin uretilemez.

## Kontroller (ne yapildi / ne yapilmaz)

- Yeni: `.github/workflows/daily.yml`, `scripts/verify_model_hash.py`,
  `scripts/check_new_bar.py`, `tests/test_daily_automation.py` (10 test).
- Degismedi: model, egitim, features, backtest, ingestion kodu;
  criteria.md; freeze.json; ci.yml; .gitignore; run_daily.py; daily_flow.py.
- Dogrulama: pytest + ruff yesil; YAML `yaml.safe_load` ile gecerli;
  persist mantigi (orphan dal + no-clobber + eski-dosya kilidi) yerel
  sandbox'ta simule edildi. GERCEK workflow kosusu yerelde CALISTIRILAMADI
  (act/runner yok): ilk gercek kosu push sonrasi Actions'ta gorulur.
