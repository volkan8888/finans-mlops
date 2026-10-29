# PROGRESS — finans-mlops

Bu dosya proje ilerlemesinin tek kaynagidir; oturum arada kesilirse buradan devam edilir.

## Calisma kurallari (her fazda gecerli)
- Komutlar: `.venv\Scripts\python ...` ; test: `.venv\Scripts\python -m pytest -q`
- Commit ATMA (kullanici atar). Sizintisiz kod: shift(-n) yalnizca `src/features/labels.py`.
- Train/val/test zamana gore; shuffle yok. Label ufkuna gore purge+embargo var.
- Sadece train'de fit edilen transformlar; hiperparametre secimi SADECE val fold'larda.
- Son 2 takvim yili HOLDOUT: tuning'de kullanilamaz, tek kez raporlanir.
- Tum rastgelelik seed'li.

## Durum

### Tamamlandi
- [x] Adim 1 (ingestion) — commit 15fc097 (20 test)
- [x] Adim 2 Asama 0: universe.txt + data.yaml duzenlemesi (untracked, commit bekliyor)
- [x] Adim 2 Asama 1: feature pipeline (src/features/*, 19 test; toplam 39 yesil; untracked, commit bekliyor)

### Simdi aktif
- [x] FAZ 0 temizlik: z-score ddof=0 + docs duzeltme; forward_return uc deger inceleme; bagimliliklar kuruldu (torch 2.14 CPU, mlflow 3.17.2, vb; pandas<3 korundu, pyarrow 26->25)
- [x] FAZ 1 egitim altyapisi (config/training.yaml, src/training/*, tests 15, docs/step3) + GERCEK KOSU: 13 fold x 3 varyant; secim=gru (wf mean_ic 0.0175, mlp 0.0140, ridge 0.0105; t_stat 2.75 / 2.75 / 1.57); HOLDOUT mean_ic 0.0082 (t 0.78, ZAYIF — durust rapor)
- [x] FAZ 2 backtest (config/backtest.yaml, src/evaluation/*, tests 9, docs/step4) + GERCEK KOSU: long_short net -8.8% (sharpe -0.33), long_only net +37.2% (sharpe 0.89; SPY +34.7% / 0.99; brut alfa ~+7.5%/y ama t-stat dusuk); reports/backtest_summary.json + equity_curve.csv
- [x] FAZ 3 MLflow: src/training/mlflow_track.py + run.py entegrasyonu; src/serving/model_store.py (registry -> yerel fallback); tests/test_mlflow.py 5 test; docs/step5_mlflow.md
- [x] FAZ 3 GERCEK KOSU tamam: mlflow.db = experiment finans-training, 4 run (variant-ridge/mlp/gru + production-gru); production-gru metrikleri final_val_ic + holdout_* + bt_* (FAZ 2 motorundan); registry 'finans-ranker' v1 + alias 'production' (run f2935107). Serving registry'den yuklendi (skor yerel fallback ile ayni -> reconstruct dogru). Not: run kokunden model/ alt klasoru cozumlemesi eklendi (_locate_model_dir).
- [x] FAZ 4 serving: config/serving.yaml, src/serving/app.py (FeatureSource + /health + /predict + /predict/batch), tests/test_serving.py 10 test, docs/step6_serving.md; GERCEK model smoke: gru yuklendi (registry), AAPL/MSFT skorlari uretildi
- [x] FAZ 5 monitoring: config/monitoring.yaml, src/monitoring/{__init__,drift,performance,run}.py (PSI + KS + rolling IC + coverage), tests/test_monitoring.py 10 test, docs/step7_monitoring.md; GERCEK rapor: PSI tum feature'larda < 0.02 (uyari yok), overall IC 0.0082 (t=0.78 zayif), son 20-gun kayan IC 0.118
- [x] FAZ 6 orkestrasyon: src/pipelines/{__init__,shared,daily_flow,retrain_flow}.py (Prefect 3.9 flows: gunluk skor+monitoring; retrain = ingestion->features->training->evaluation zinciri), scripts/run_daily.py CLI (--mode daily|retrain), tests/test_pipelines.py 7 test, docs/step8_orchestration.md; GERCEK daily kosu ok (44 skor + rapor, exit 0)
- [x] FAZ 7 dashboard: src/dashboard/{__init__,prepare}.py (saf hazirlik katmani) + app.py Streamlit 4 sekme (Skorlar/Backtest/Monitoring/MLflow Runs), tests/test_dashboard.py 10 test, docs/step9_dashboard.md; headless canli smoke: /_stcore/health ok
- [ ] FAZ 8 Docker + CI/CD (Dockerfile, .dockerignore, docker-compose.yml, .github/workflows/*, ruff temizle, docs/step10_docker_cicd.md)
- [x] FAZ 8 Docker + CI: Dockerfile (python 3.12-slim, torch CPU wheel on-kurulumu, CMD=skor servisi), .dockerignore, docker-compose.yml (api + mlflow-ui profili), .github/workflows/ci.yml (ruff + pytest + docker build smoke), docs/step10_docker_cicd.md; TUM REPO ruff temiz
- [x] FAZ 9 README + uc-uca smoke + son rapor: README.md (kurulum/kullanim/durust rapor), scripts/smoke_e2e.py (Gercek kosu gecti: features->evaluation->daily flow + cikti kontrolu, exit 0), docs/step11_final_report.md

### Kalite-islemi (post-FAZ)
- [x] GOREV 1 bayat-skor: serving VE gunluk akis artik data/raw/prices.parquet + build_features (LABEL'SIZ, training ile AYNI kod) ile canli gun skorlar; features.parquet yedek kaynagi. Skor tarihi 2026-10-14 (5 gun bayat) -> 2026-10-29 (fiyatla ayni gun). /health: score_date + data_end_date + stale_price_dates + incomplete_session_dropped. tests/test_serving.py 10->15.
- [x] GOREV 2 gunluk akis veri akisi: daily task sirasi = ingestion -> dogrulama -> features -> skor -> tahmin kaydi -> monitoring. Ingestion alt yordami akisin 1. adimi; tahminler data/predictions/predictions_YYYY-MM-DD parquet (ayni gun uzerine YAZILMAZ). Kapanis kurali (16:00 America/New_York; kapanis oncesi kismi bar atlanir, loglanir). tests/test_pipelines.py 7->12. Gercek kosu 2x exit 0 (idempotent dokumentasyon: docs/step8).
- [x] GOREV 3 IC istatistik Newey-West: evaluate.py'ye newey_west_t (Bartlett, lag=horizon-1) ve stride_ic_summary (bagimsiz ornekleme, stride=5) eklendi; ic_summary artik t_nw dondurur; monitoring nw_lag=4 + dashboard metrigi; scripts/ic_report.py holdout kayitli tahminlerden yeniden hesap YAPTI (model egitmedi): mean_ic 0.0082 -> t ham 0.78 / t_nw 0.45 / bagimsiz t 0.26 (n_bagimsiz=101). VAL icin NW hesaplanamadi: walk-forward tahmin serisi diske yazilmamisti (docs/step11'de acikca). Rapor: reports/ic_stats_report.json. tests/test_ic_stats.py 7.
- [x] GOREV 4 bozuk metin temizligi: tum .py/.md/.yaml/Dockerfile taranarak 12 konum duzeltildi (app.py: botunu/HINGE/kostrun; config/training.yaml + src/training/train.py: siziint; README: kuintil; src/evaluation/backtest.py: ellesin + szintidir + sehir; tests/test_backtest.py docstring'i; docs/step11 tablo birlesik satiri; src/evaluation/run.py; src/features/technical.py; docs/step4_backtest.md; bu oturumun yeni dosyalarindaki benzerler dahil). Kalan ASCII-disi metinler DOGRU UTF-8 Turkce olanlar (docs/step2_features.md vb.) - dokunulmadi; docs/PROGRESS.md korundu.
- [x] GOREV 5 README sonuclar + gitignore istisna: README'de IC tablosu (t_ham + t_nw + bagimsiz t), kantil tablo, portfoy geri-test tablosu ve "Durust sinirlar" bolumu (holdout anlamsiz, long_only beta, survivorship + SPY karsilastirmasi adil degil, yfinance, kucuk evren, basit maliyet, tavsiye DEGIL). .gitignore: reports/* + kucuk JSON istisnalar (backtest/monitoring/ic_stats + forward_test/freeze.json + criteria.md); equity_curve.csv agir oldugundan haric.
- [x] GOREV 6 Docker/CI dogrulama (docker bu makinede kurulu degil): "not recognized" ciktisi AYNEN docs/step10'a yazildi; build/run/compose KOSULAMADI (uydurulan sonuc yok). Statik dogrulama yeterliydi: Dockerfile (torch CPU once; CUDA derlemesi cekilmez; data/models imaja girmez), compose mount yollari ayni, ci.yml komutlari lokal ruff/pytest ile HARFI HARFINE ayni, tzdata pyproject'e eklendi (zoneinfo America/New_York icin; lokalde 2026.5 dogrulandi).
- [x] GOREV 7 ileri test hazirligi: scripts/freeze_model.py -> reports/forward_test/freeze.json GERCEK KOSULDU (gru; mlflow_registry v1; model sha8 50241b34; config SHA-256'lar; seed 42; train/holdout araliklari). reports/forward_test/criteria.md SABLON (esikler proje sahibine). src/monitoring/forward.py forward_test_report: D->D+horizon islem-gunu tanimi; unrealized SKIP+SAY (uydurma yok); IC + Newey-West t; kuantil tablo; SPY farki. GERCEK KOŞU: status 'tum_beklemede' (44 unrealized: tahmin penceresi dolmadi, biriktikce rapor dolacak). dashboard "Ileri Test" sekmesi + prepare.load_freeze/load_forward. tests/test_forward.py 9.

### Test sayisi
- Baslangic: 39 (20 ingestion + 19 feature)
- FAZ 1: +15 (tests/test_training.py) => 54
- FAZ 2: +9 (tests/test_backtest.py) => 63
- FAZ 3: +5 (tests/test_mlflow.py) => 68
- FAZ 4: +10 (tests/test_serving.py) => 78
- FAZ 5: +10 (tests/test_monitoring.py) => 88
- FAZ 6: +7 (tests/test_pipelines.py) => 95
- FAZ 7: +10 (tests/test_dashboard.py) => 105

### Gercek veri egitim ozeti (FAZ 1, 2026-10-22)
- Panel: 127,732 satir | 2,903 gun | 44 ticker | 8 feature; 13 walk-forward fold (>=10 gun guard).
- Walk-forward combined val IC (1635 gun): ridge 0.0105 (t 1.57) / mlp 0.0140 (t 2.75) / gru 0.0175 (t 2.75) -> SECIM: gru.
- Final retrain (son val dilimi, early-stop): best val_ic -0.0494 (son pencere isabeti zayifti).
- HOLDOUT (501 gun): mean_ic 0.0082 | ICIR 0.035 | t 0.78 | positive 53% | kantiller: q0 0.47%, q1 0.19%, q2 0.29%, q3 0.41%, q4 0.51% (q0!), yani alt basinayi iyi demirleme: reversal.
- Sonuc zayif: durust raporlandi; hiperparam degisikligi icin holdout'a bakilmadi.

### Tasarim kararlar (kumulatif)
1. z-score ddof=0 (populasyon std): gun ici std=1; pooled ~1.000. Sebebi: ddof=1'de pooled std
   sqrt(43/44) artefakti olurdu; ddof secimi IC korelasyonunu degistirmez.
2. forward_return uc degerleri GERCEK olaylar: BA Mart 2020 (COVID), TSLA 2020-01 squeeze ve
   2024-11 secim rallisi, ORCL 2025-09-03 (AI bulut rehberi), AMD 2016-04 (x86 JV donemi),
   INTC 2026-04, AMD 2025-10 (OpenAI ortakligi donemi). Veri hatasi DEGIL, dokunulmadi.
3. Egitimde shuffle YOK (DataLoader shuffle=False) — determinism + "asla shuffle yok" kurali.
4. SequenceDataset penceresi tam panel uzerinden kurulur (gemi gecmise uzanabilir);
   label yalnizca hedef satirdan. full_df parametresi ek train_model imzasina eklendi.
5. LSTMRanker/GRURanker tek RecurrentRanker sinifinin cell_type varyanti (kod tekrarsiz).
6. Loss: label rank uzerinde MSE; listnet (gunluk pairwise softmax CE) config'ten secilebilir.
7. Final egitim: dev'in SON val dilimi early-stop icin; dilim oncesi dev'e kadar olan kisim
   train edilir. Daha yakin tarihler kullanilmaz (belgeli kayip, tutarlilik).
8. Rebalance arasinda agirliklar SABIT (drift ihmal, belgeli); askida ticker katkisi 0 (log).
9. Koşu bütcesi icin: default epochs 20/patience 3; gru epochs 10/patience 2.

### Bilinen riskler
- Adim 2 Asama 0+1 henuz commit edilmedi (kullanici atacak).
- Gercek holdout IC kucuk cikabilir (zayif sinyal); sonuc durust raporlanir.
