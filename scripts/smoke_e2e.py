"""Adim 11 (FAZ 9): uc-uca smoke (hafif zincir, egitim yok).

Adimlar:
1) python -m features.run     : raw fiyatlar -> feature paneli yenileme
2) python -m evaluation.run   : MEVCUT holdout tahminleri -> backtest raporu
3) prefect daily_pipeline     : gunluk skor parquet + monitoring raporu
4) cikti kontrolu             : beklenen dosyalar var mi

NOTLAR:
- ingestion (yfinance indirme) ve training (walk-forward, ~10 dakika)
  bu smoke'ta YOK; once elle calistirilmis olmasi beklenir
  (production model + holdout tahminleri hazirda). Tam zincir:
  `python scripts/run_daily.py --mode retrain`.
- Proje koku path'leri modul tarafindan cozulur; CWD bagliligi yok.

Kullanim:
    .venv\\Scripts\\python scripts/smoke_e2e.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("features.run", (sys.executable, "-m", "features.run")),
    ("evaluation.run", (sys.executable, "-m", "evaluation.run")),
)

EXPECTED: tuple[str, ...] = (
    "data/processed/features.parquet",
    "reports/backtest_summary.json",
    "reports/equity_curve.csv",
    "reports/monitoring_report.json",
    "data/processed/latest_scores.parquet",
)


def main() -> int:
    """Zinciri kosar; beklenen cikti dosyalari eksikse 1 dondurur.

    veri tazeleme (ingestion) bu smoke'ta calismaz (network yok): daily
    adimi with_ingestion=False ile kosar.
    """
    for name, command in STEPS:
        print(f"[smoke] basliyor: {name}")
        completed = subprocess.run(list(command), cwd=str(PROJECT_ROOT), check=False)
        if completed.returncode != 0:
            print(f"[smoke] BASARISIZ: {name} (kod {completed.returncode})")
            return 1
        print(f"[smoke] tamam: {name}")

    print("[smoke] basliyor: prefect daily_pipeline (ingestion'suz; ag bagimsiz)")
    from pipelines.daily_flow import daily_pipeline

    summary = daily_pipeline(with_ingestion=False)
    print("[smoke] daily ozet:", summary)

    missing = [relative for relative in EXPECTED if not (PROJECT_ROOT / relative).is_file()]
    if missing:
        print(f"[smoke] BASARISIZ: eksik cikti dosyalari: {missing}")
        return 1
    print("[smoke] TUM CIKTILAR HAZIR - uc-uca smoke gecti")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
