"""Adim 8 (FAZ 6): retrain akisi - tam yeniden egitim zinciri (Prefect).

Zincir (her biri ayri alt yordam, cwd = proje koku):
1) python -m ingestion.run    : fiyat indirme + dogrulama + publish
2) python -m features.run     : feature paneli yenile
3) python -m training.run     : walk-forward egitim + secim + holdout + MLflow
4) python -m evaluation.run   : holdout backtest raporlari
5) monitoring.run_monitoring  : drift + performans raporu (ayni surecte)

Komutlar uzerinden enjekte edilebilir (testlerde) veya command listesi
parametre ile degistirilebilir. Her komut basarisiz olursa flow durur
(check=True): hataya ragmen devam NADIR dogru davranis; burada duruyoruz
(bozuk feature ile egitim istemiyoruz).

Kullanim:
    .venv\\Scripts\\python scripts/run_daily.py --mode retrain
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys

from prefect import flow, task

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_COMMANDS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "-m", "ingestion.run"),
    (sys.executable, "-m", "features.run"),
    (sys.executable, "-m", "training.run"),
    (sys.executable, "-m", "evaluation.run"),
)


@task(name="komut-calistir")
def task_run_command(command: tuple[str, ...]) -> str:
    """Alt yordam komutu (cwd = proje koku); hata olursa exception. Uyari: cikti tamponlanir."""

    logger.info("Komut: %s", " ".join(command))
    completed = subprocess.run(
        list(command),
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,  # kod elle kontrol edilir: hatayi okunur mesaja ceviriyoruz
    )
    if completed.returncode != 0:
        tail = (completed.stdout or "")[-1500:]
        raise RuntimeError(f"Komut basarisiz (kod={completed.returncode}): {' '.join(command)}\n{tail}")
    if completed.stdout:
        logger.info("Cikti (kisa kesim): %s", completed.stdout[-800:])
    return f"ok:{' '.join(command)}"


@task(name="monitoring-raporu")
def task_run_monitoring() -> dict:
    """Egitim sonrasi monitoring raporu; uyari listesini dondurur."""
    from monitoring.run import DEFAULT_CONFIG_PATH as MONITORING_CONFIG
    from monitoring.run import run_monitoring

    report = run_monitoring(MONITORING_CONFIG)
    logger.info("Monitoring uyari listesi: %s", report["alerts"] if report["alerts"] else "yok")
    return report


@flow(name="finans-retrain")
def retrain_pipeline(commands: tuple[tuple[str, ...], ...] | None = None, with_monitoring: bool = True) -> dict:
    """Tam retrain zinciri; sonucta hangi adimlarin calistigi dondurulur."""
    effective = commands or DEFAULT_COMMANDS
    ran = []
    for command in effective:
        ran.append(task_run_command(command))
    report = task_run_monitoring() if with_monitoring else None
    return {"commands_ok": ran, "monitoring_alerts": (report or {}).get("alerts", [])}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    retrain_pipeline()
