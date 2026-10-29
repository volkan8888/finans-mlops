"""Feature pipeline yapilandirmasi (config/features.yaml).

Ingestion konvansiyonunu aynen izler:
- Goreli yollar proje kokune (config dosyasinin bir ust dizinine) gore cozulur.
- logging kullanilir; print yok.
- Tum sayisal alanlar yuklemede dogrulanir.

Benchmark kaynak zinciri (tek kaynak prensibi + override esnekligi):
    1. features.yaml icinde 'benchmarks' anahtari varsa o kullanilir;
    2. yoksa ayni klasordeki config/data.yaml'in benchmarks listesi okunur;
    3. o da yoksa bos liste (benchmark filtrelemesi devre disi kalir).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_CONFIG_PATH = "config/features.yaml"
DEFAULT_DATA_YAML_PATH = "config/data.yaml"
DEFAULT_PRICES_PATH = "data/raw/prices.parquet"
DEFAULT_OUTPUT_PATH = "data/processed/features.parquet"


@dataclass(frozen=True)
class FeaturesConfig:
    """features.yaml icerigi; goreli yollar proje kokune gore cozulmustur.

    Feature tanimlari:
        momentum_windows  : momentum pencereleri (gun)
        skip_window/skip  : momentum_skip(window, skip) = t-window'dan t-skip'e getiri
        volatility_window : gunluk log getirilerin rolling std penceresi
        rsi_window        : RSI penceresi (Wilder)
        volume_window     : hacim degisimi icin rolling ortalama penceresi
        reversal_window   : kisa vadeli toplam getiri penceresi
    Etiket:
        label_horizon     : forward horizon (gun); son horizon gunde label NaN
    Kirpma/filtre:
        winsorize_lower/upper : gun icin quantile sinirlari
        min_tickers_per_date  : bir gune en az bu kadar hisse olmali; az ise gun atilir
    Yollar:
        prices_path : ham fiyat parquet
        output_path : yazilacak feature parquet
    benchmarks: depodan okunan ama feature evrenine girmeyen sembol listesi.
    """

    momentum_windows: tuple[int, ...]
    skip_window: int
    skip: int
    volatility_window: int
    rsi_window: int
    volume_window: int
    reversal_window: int
    label_horizon: int
    winsorize_lower: float
    winsorize_upper: float
    min_tickers_per_date: int
    prices_path: Path
    output_path: Path
    benchmarks: tuple[str, ...] = field(default=())


def _clean_tickers(raw: list[str] | None) -> list[str]:
    """Trim, buyuk harf, bos/tekrar girdileri at (ingestion ile ayni kural)."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        ticker = str(item).strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        cleaned.append(ticker)
    return cleaned


def _resolve(project_root: Path, raw: str | Path) -> Path:
    """Goreli yolu proje kokune uyar; mutlak ise aynen doner."""
    path = Path(str(raw))
    return path if path.is_absolute() else project_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Yapilandirma bulunamadi: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Yapilandirma dict bekliyordu, geldi: {type(data).__name__}")
    return data


def _benchmarks_from_features_or_data(
    data: dict[str, Any], project_root: Path
) -> list[str]:
    """Benchmark listesini features.yaml -> data.yaml zincirinden cozer."""
    if "benchmarks" in data:
        return _clean_tickers(data["benchmarks"])
    data_yaml_path = project_root / DEFAULT_DATA_YAML_PATH
    if data_yaml_path.is_file():
        benchmarks = _clean_tickers(_load_yaml(data_yaml_path).get("benchmarks", []))
        logger.info("Benchmarks data.yaml'dan alindi: %s", benchmarks)
        return benchmarks
    logger.warning("Benchmark kaynagi bulunamadi; benchmark filtreleme yapilmayacak.")
    return []


def load_features_config(
    path: str | Path = DEFAULT_FEATURES_CONFIG_PATH,
) -> FeaturesConfig:
    """features.yaml'i yukler ve alanlari dogrular.

    Dogrulama:
    - pencere/horizon sayisallari pozitif tam sayi olmali.
    - skip: 0 <= skip < skip_window (aradaki getiri tanimli olsun).
    - winsorize: 0 < lower < upper < 1.
    """
    config_path = Path(path)
    data = _load_yaml(config_path)
    project_root = config_path.resolve().parent.parent

    def _positive_int(key: str, default: int) -> int:
        value = int(data.get(key, default))
        if value <= 0:
            raise ValueError(f"{key} pozitif olmali, gelen deger: {value}")
        return value

    windows_raw = data.get("momentum_windows") or []
    momentum_windows = tuple(int(w) for w in windows_raw)
    if not momentum_windows or any(w <= 0 for w in momentum_windows):
        raise ValueError(f"momentum_windows pozitif tam sayi listesi olmali: {windows_raw}")

    skip_map = data.get("momentum_skip") or {}
    skip_window = int(skip_map.get("window", 60))
    skip = int(skip_map.get("skip", 5))
    if skip_window <= 0 or skip < 0 or skip >= skip_window:
        raise ValueError(
            f"momentum_skip gecersiz: skip({skip}) 0..{skip_window - 1} araliginda olmali "
            f"(window={skip_window})"
        )

    winsorize_map = data.get("winsorize") or {}
    lower = float(winsorize_map.get("lower", 0.01))
    upper = float(winsorize_map.get("upper", 0.99))
    if not (0.0 < lower < upper < 1.0):
        raise ValueError(
            f"winsorize gecersiz: 0 < lower({lower}) < upper({upper}) < 1 olmali"
        )

    min_tickers = _positive_int("min_tickers_per_date", 10)

    paths = data.get("paths") or {}
    prices_path = _resolve(project_root, paths.get("prices_path") or DEFAULT_PRICES_PATH)
    output_path = _resolve(project_root, paths.get("output_path") or DEFAULT_OUTPUT_PATH)

    benchmarks = _benchmarks_from_features_or_data(data, project_root)

    logger.info(
        "FeaturesConfig yukleniyor: %s | momentum=%s | skip=%d/%d | vol=%d | rsi=%d | "
        "volume=%d | reversal=%d | horizon=%d | winsorize=%.2f/%.2f | min_tickers=%d | "
        "benchmarks=%d | prices=%s | output=%s",
        config_path,
        list(momentum_windows),
        skip_window,
        skip,
        int(data.get("volatility_window", 20)),
        int(data.get("rsi_window", 14)),
        int(data.get("volume_window", 20)),
        int(data.get("reversal_window", 5)),
        int(data.get("label_horizon", 5)),
        lower,
        upper,
        min_tickers,
        len(benchmarks),
        prices_path,
        output_path,
    )
    return FeaturesConfig(
        momentum_windows=momentum_windows,
        skip_window=skip_window,
        skip=skip,
        volatility_window=_positive_int("volatility_window", 20),
        rsi_window=_positive_int("rsi_window", 14),
        volume_window=_positive_int("volume_window", 20),
        reversal_window=_positive_int("reversal_window", 5),
        label_horizon=_positive_int("label_horizon", 5),
        winsorize_lower=lower,
        winsorize_upper=upper,
        min_tickers_per_date=min_tickers,
        prices_path=prices_path,
        output_path=output_path,
        benchmarks=tuple(benchmarks),
    )
