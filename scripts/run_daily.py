"""Adim 8 (FAZ 6): orkestrasyon komut satiri - gunluk/retrain modu.

Kullanim:
    .venv\\Scripts\\python scripts/run_daily.py --mode daily
    .venv\\Scripts\\python scripts/run_daily.py --mode retrain

- daily   : ingestion (veri tazeleme) + dogrulama + canli skor + tahmin kaydi
  + monitoring (model egitmez; yfinance'le internet baglantisi gerekir).
- retrain : ingestion -> features -> training -> evaluation -> monitoring
  (zaman alan tam zincir; uyari listesini rapor eder).

Skriptin yol kaygisi yok: moduller proje kokunu kendileri cozumler
(CWD'e bagli varsayim yok).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    """Arguman cozucu: --mode daily|retrain; varsayilan daily."""
    parser = argparse.ArgumentParser(description="Gunluk/retrain orkestrasyon akislari")
    parser.add_argument("--mode", choices=["daily", "retrain"], default="daily")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Mod secimine gore ilgili Prefect akisini cagirir."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if args.mode == "daily":
        from pipelines.daily_flow import daily_pipeline

        summary = daily_pipeline()
        print("daily pipeline ozet:", summary)
    else:
        from pipelines.retrain_flow import retrain_pipeline

        summary = retrain_pipeline()
        print("retrain pipeline ozet:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
