"""Adim 8 (FAZ 6) + kalite gunluk akisi: gunluk is listesi.

Task sirasi (ONEM: ingestion -> dogrulama -> features -> skor -> tahmin
kaydi -> monitoring):
1) veri-ingestion   : `python -m ingestion.run` alt yordami (yeni barlari
                      ceker; ayni gun tekrar kosulsa da idempotent:
                      merge keep=last, degisiklik yoksa dosya yazilmaz).
2) fiyat-dogrulama  : depodaki fiyatlar semayi gecer mi (pandera SERT);
                      bozuksa akis DURUR, kaydetme yok.
3) feature-uretim   : fiyat deposundan build_features (LABEL'SIZ; training
                      ile AYNI fonksiyon) - egitim dosyasina dokunulmaz.
4) skor-uret        : production model en yeni gunun tum ticker skorlari;
                      data/processed/latest_scores.parquet yenilenir.
5) tahmin-kaydi     : data/predictions/predictions_YYYY-MM-DD.parquet;
                      ayni gun tekrar kosulsa DOSYA USTUNE YAZILMAZ
                      (tahmin kaydi sonradan degistirilemez mantigindadir;
                      ileri test butunlugu).
6) monitoring-raporu: PSI/KS + kayan IC raporu.

Kapanis-kurali (detay: docs/step8_orchestration.md):
- ABD pazari 16:00 America/New_York'ta kapanir. Kapanis ONCESI calisirsa
  bugunun bar'i kismidir => atlanir (loglanir) ve skorlar TAMAMLANMIS son
  oturuma dayanir. Kapanis SONRASI calisirsa bugunun bar'i tam sayilir.
- Kural serving kaynak kurulumunda (FeatureSource.from_prices) uygulanir;
  akis `now` parametresiyle test edilebilir.

Bu akis modeli YENIDEN EGITMEZ. Retrain icin pipelines.retrain_flow.

Kullanim:
    .venv\\Scripts\\python scripts/run_daily.py --mode daily
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys

import pandas as pd
from prefect import flow, task

logger = logging.getLogger(__name__)
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
LATEST_SCORES_RELATIVE = pathlib.Path("data/processed/latest_scores.parquet")
PREDICTIONS_DIR_RELATIVE = pathlib.Path("data/predictions")

DEFAULT_INGESTION_COMMAND: tuple[str, ...] = (sys.executable, "-m", "ingestion.run")


@task(name="veri-ingestion")
def task_run_ingestion(command: tuple[str, ...] | None = None) -> str:
    """Fiyat deposunu tazeler (alt yordam; idempotent); hata olursa durur.

    Alt yordam tercih edilir: ingestion kendi CLI akisini (config,
    anomali raporu, kaydetmeden-once-dogrulama) zaten uygular; retrain
    zinciri ile AYNI komut kullanilir.
    """
    effective = tuple(command) if command else DEFAULT_INGESTION_COMMAND
    logger.info("Ingestion komutu: %s", " ".join(effective))
    completed = subprocess.run(
        list(effective),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,  # kod elle kontrol: hatada okunur hata mesaji
    )
    if completed.returncode != 0:
        tail = (completed.stdout or "") + (completed.stderr or "")
        raise RuntimeError(f"Ingestion basarisiz (kod={completed.returncode}): {' '.join(effective)}\n{tail[-1200:]}")
    return f"ok:{' '.join(effective)}"


@task(name="fiyat-dogrulama")
def task_validate_prices(prices_path: str | pathlib.Path) -> dict:
    """Depodaki fiyatlar semayi SERT dogrular (pandera); bozukluk akisi keser.

    - ingestion tarafla zaten kaydetmeden once dogruluyordu; bu task son
      kapacidir: depoyu baskalari bozmus mu diye okur.
    - Anomali bulgulari YUMUSAKTIR (raporlanir; akisi kesmez).
    """
    from ingestion.validation import find_anomalies, validate_prices

    prices = pd.read_parquet(prices_path)
    validated = validate_prices(prices)  # ValueError -> akis basarisiz
    anomalies = find_anomalies(validated)
    logger.info(
        "Fiyat dogrulama: %d satir | %s .. %s | anomali=%d",
        len(validated), validated["date"].min().date(), validated["date"].max().date(), len(anomalies),
    )
    return {
        "rows": len(validated),
        "first_date": str(validated["date"].min().date()),
        "last_date": str(validated["date"].max().date()),
        "n_anomalies": len(anomalies),
    }


@task(name="feature-uretim")
def task_build_features(
    prices_path: str | pathlib.Path,
    features_config_path: str | pathlib.Path | None = None,
    now: pd.Timestamp | None = None,
) -> object:
    """Fiyat hammadde -> build_features (label'siz canli tablo + FeatureSource).

    - egitimin features.parquet dosyasi DOKUNULMAZ. Label'siz tabloda son
      horizon gunleri dusulmez (label NaN olsa da feature doludur); bu
      yuzden skorlar bayat olmaz ve en yeni tamamlanmis gune ait olur.
    - now verilir ve kapanis oncesi ise kismi oturum atlanir.
    """
    from serving.app import FeatureSource

    source = FeatureSource.from_prices(
        prices_path, features_config_path=features_config_path, now=now
    )
    logger.info(
        "Canli feature tablo: %d satir | skor gunu=%s | fiyat_sonu=%s | bayat_gun=%d | atlanan_oturum=%s",
        len(source.frame),
        source.latest_date().date(),
        source.prices_end_date.date() if source.prices_end_date is not None else None,
        source.stale_price_dates,
        source.incomplete_session_dropped.date() if source.incomplete_session_dropped is not None else None,
    )
    if source.stale_price_dates:
        logger.warning("Dikkat: skor gununun arkasinda %d fiyat gunu bayat", source.stale_price_dates)
    return source


@task(name="skor-uret")
def task_compute_scores(source: object) -> pd.DataFrame:
    """Production modelin canli tablo uzerinde en yeni gun skoru (44 satir beklenir)."""
    from pipelines.shared import scores_for_day

    scores = scores_for_day(source)
    out_path = PROJECT_ROOT / LATEST_SCORES_RELATIVE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_path, index=False)
    logger.info(
        "Guncel skorlar yazildi: %s (%d satir) | tarih=%s",
        out_path, len(scores), scores["date"].max().date(),
    )
    return scores


@task(name="tahmin-kaydi")
def task_record_predictions(scores: pd.DataFrame) -> dict:
    """Gunun tahminlerini TARIH DAMGALI dosyaya kaydeder; VARSA DEGISTIRMEZ.

    Idempotent + degismez kayit: ayni gun tekrar kosulursa ayni dosya
    adiyla bulunur; icerik korunur, uzerine YAZILMAZ ve UYARI loglanir.
    Boylece ileri test verisinin butunlugu korunur (replicate yok, sonraden
    degisim yok).
    """
    if scores is None or scores.empty:
        raise RuntimeError("Skor frame'i bos; tahmin kaydi yazilmaz")
    score_date = pd.Timestamp(scores["date"].max()).date()
    directory = PROJECT_ROOT / PREDICTIONS_DIR_RELATIVE
    path = directory / f"predictions_{score_date}.parquet"
    if path.is_file():
        logger.warning("Tahmin kaydi ZATEN VAR, uzerine yazilmaz: %s", path)
        return {"path": str(path), "created": False}
    directory.mkdir(parents=True, exist_ok=True)
    frame = scores.sort_values("ticker", kind="mergesort").reset_index(drop=True)
    frame.to_parquet(path, index=False)
    logger.info("Tahmin kayit yazildi: %s (%d satir)", path, len(frame))
    return {"path": str(path), "created": True}


@task(name="monitoring-raporu")
def task_run_monitoring() -> dict:
    """Monitoring raporunu uretir; uyari listesini loglar."""
    from monitoring.run import DEFAULT_CONFIG_PATH as MONITORING_CONFIG
    from monitoring.run import run_monitoring

    report = run_monitoring(MONITORING_CONFIG)
    logger.info("Monitoring uyari listesi: %s", report["alerts"] if report["alerts"] else "yok")
    return report


@flow(name="finans-daily")
def daily_pipeline(
    ingestion_command: tuple[str, ...] | None = None,
    now: pd.Timestamp | None = None,
    with_ingestion: bool = True,
) -> dict:
    """Gunluk akis; task sirasi ingestion'dan monitoring'e kadar korunur."""
    pipelines_root = PROJECT_ROOT
    from pipelines.shared import (
        resolve_features_config_path,
        resolve_prices_path,
    )

    ingestion_ok = task_run_ingestion(ingestion_command) if with_ingestion else None
    prices_path = resolve_prices_path(pipelines_root)
    if prices_path is None:
        raise RuntimeError("serving.yaml'da prices_path tanimsiz; gunluk akis canli fiyat bekler")
    if not prices_path.is_file():
        raise RuntimeError(f"Fiyat deposu bulunamadi: {prices_path} (ingestion calismali)")
    validation = task_validate_prices(prices_path)
    feature_config = resolve_features_config_path(pipelines_root)
    source = task_build_features(prices_path, feature_config, now=now)
    scores = task_compute_scores(source)
    record = task_record_predictions(scores)
    report = task_run_monitoring()
    return {
        "ingestion": ingestion_ok,
        "validation": validation,
        "scores_path": str(pipelines_root / LATEST_SCORES_RELATIVE),
        "scores_date": str(pd.Timestamp(scores["date"].max()).date()),
        "predictions_path": record["path"],
        "predictions_created": record["created"],
        "alerts": report["alerts"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    daily_pipeline()
