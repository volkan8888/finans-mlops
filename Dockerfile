# Adim 10 (FAZ 8): finans-mlops imaji
# Temel: python 3.12 slim; kaynaklar (src/ config/ scripts/ tests/) imaja gecer.
# Build: docker build -t finans-mlops:latest .
# Run:   docker run --rm -p 8000:8000 finans-mlops:latest  (skor servisi FAZ 4)

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Katman cache dostu: once paket tanimi, sonra bagimlilik kurulumu
RUN mkdir -p src scripts config data reports models mlruns
COPY pyproject.toml ./
# torch'u CPU wheel'inden kurar: varsayilan index GPU derlemesini (GB'larca) cekmez
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e ".[dev]"

# Kaynaklar (veri dosyalari imaja KOPYALANMAZ; runtime mount edilir)
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Varsayilan komut: skor servisi
CMD ["python", "-m", "serving.app"]
