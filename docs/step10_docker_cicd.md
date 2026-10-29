# Adim 10 (FAZ 8) - Docker + CI/CD

## Dockerfile ozet

- Temel imaj: `python:3.12-slim`; `WORKDIR /app`.
- `torch` ONCE CPU wheel'inden kurulur
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`);
  boylece varsayilan index'in GB boyutlu CUDA derlemesi kaynaklanmaz.
- Sonra `pip install -e ".[dev]"` (pyproject'tan diger bagimliliklar).
- Veri/artifaktlar (data/, models/, mlflow.db, mlruns/, reports/)
  imaja KOPYALANMAZ; gerektiginde mount edilir.
- Varsayilan `CMD`: skor servisi (`python -m serving.app`, port 8000).

komutlar:
```
docker build -t finans-mlops:latest .
docker run --rm -p 8000:8000 finans-mlops:latest
# curl http://localhost:8000/health
```

Esas local kosuda Windows'ta Docker bu boyutta (torch CPU ~200MB
katman + deps) derlenir; CI'da otomatik dogrulanir (asagiya bakiniz).

## docker-compose.yml

- `api`: finans-mlops imaji; 8000:8000; mlflow.db/mlruns/models/data/reports
  read-only mount; healthcheck /health.
- `mlflow-ui`: `_ui` profili ile (`docker compose --profile ui up`);
  MLflow sunucusu (port 5000) ayni sqlite backend + mlruns klasoru uzerinden.

```
docker compose up          # sadece api
docker compose --profile ui up  # api + MLflow UI
```

## .github/workflows/ci.yml

- `lint-tests` job: python 3.12 -> torch CPU -> `pip install -e ".[dev]"`
  -> `ruff check .` (tum repo; 0 hata) -> `pytest -q` (122 test;
  ag-bagimsiz, sentetik veriyle calisir; true-artifact testleri dosya
  varsa derin dogrulama, yoksa kucuk varsayim onayi).
- `docker-build` job: imaji derler (push YOK; kimlik bilgisi gerekmez);
  lint-tests gectikten sonra.

## Uretime/makineye gore dogrulama durumu (2026-10-29, durust kayit)

Bu makinede Docker KURULU DEGIL. Komutun tam ciktisi (aynen):

```
PS> docker --version
docker : The term 'docker' is not recognized as the name of a cmdlet,
function, script file, or operable program. Check the spelling of the
name, or if a path is included, verify that the path is correct and try
again.
PS> docker compose version
(ayni hata)
```

Bu yuzden ASAGIDAKILER BU MAKINEDE KOSULAMADI (uydurma sonuc YOK):

- `docker build -t finans-mlops:latest .`
- `docker run --rm -p 8000:8000 finans-mlops:latest`
- `docker compose up` / `docker compose --profile ui up`

Kullanici Docker Desktop kurduktan sonra bu komutlari calistirip
sonucu buraya eklemelidir.

### Yapilabilen STATIK dogrulama (dosya okumasi; 2026-10-29)

- Dockerfile: `FROM python:3.12-slim`; torch CPU wheel'i `pip install
  -e ".[dev]"`'den ONCE (GPU derlemesi cekilmez); sadece src/config/
  scripts/tests imaja KOPYALANIR; data/, models/, mlflow.db, mlruns/
  gecmez.
- docker-compose.yml: data/models/mlflow.db/mlruns/reports host'tan
  mount edilir (ro). Yani model/veri dosyalari imaja dahil degildir ve
  mount yollari degismemistir (makine degisiminde dosya korumasi).
- ci.yml: `lint-tests` adimlari lokalde calistirdigimiz komutlarla
  HARFI HARFINE aynidir (`ruff check .` + `pytest -q`); `docker-build`
  job'i `needs: lint-tests` ile baglidir. Repo'da `daily.yml` YOK
  (yalnizca ci.yml); gunluk tetikleme (Task Scheduler/cron) dokumanda
  scripts/run_daily.py ile anlatilir, CI kapsaminda degil.
- tzdata: serving'in kapanis kurali `zoneinfo("America/New_York")`
  kullanir. Windows'ta Python pip tzdata paketine yaslanir; slim imaj
  (Debian) OS tzdata tasir. Risk bariz olsun diye `tzdata` pyproject
  bagimlilik listesine eklendi (imaj/OS farketmeksizin ayni davranis).
  Lokal dogrulama: tzdata 2026.5 kurulu ve America/New_York cozuluyor.

### CI/temiz makinede servis davranisi (statik not)

CI'da data/ klasoru yoktur. `create_app` kaynak kurulumunu
try/except ILE SARAR: source=prices okunamazsa features yedegi
denenir; o da yoksa `source_kind=None` ile /health donebilir ve
app importu KIRILMAZ (tests/test_serving.py'de degrade yolu
kapsamli test edildi).
## Konvansiyon notlari

- Testler ag-bagimsizdir (ingestion/features sentetik DataFrame'lerle);
  bu yuzden CI'da yfinance/veri indirmesi YOKTUR (deterministik ve hizli).
- Gercek-veri dogrulamasi (ingestion + training + backtest + monitoring
  + serving + daily akis) makinede elle FAZ 0-7 adimlarinda yapildi;
  raporlar docs/ sayfalarinda.
