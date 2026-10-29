"""IC istatistik raporu: ham t vs Newey-West t vs bagimsiz ornekleme t.

Sorun: label 5 is gunluk ileri getiri; ardisik gunlerin IC'leri cakisir
(bagimsiz orneklem < gun sayisi) ve klasik t-stat siskin olabilir.

Bu betik KAYITLI tahminlerden yeniden hesaplar (MODEL EGITMEZ):
- data/processed/holdout_predictions.parquet  -> holdout IC serisi
- models/<secilen>/meta.json                  -> saklanan ham ozet (karsilastirma)

Walk-forward val tahminleri diske YAZILMAMISTI (yalnizca holdout
parquet'i var): val IC icin NW/bagimsiz hesap YAPILAMAZ; bu durum raporda
'aciklama' olarak belirtilir (durust kurali).

Cikti: reports/ic_stats_report.json + konsol tablosu.

Kullanim:
    .venv\\Scripts\\python scripts/ic_report.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from training.evaluate import ic_summary, stride_ic_summary

logger = logging.getLogger(__name__)

DEFAULT_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "holdout_predictions.parquet"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "ic_stats_report.json"


def _latest_model_meta() -> tuple[str, dict] | tuple[None, None]:
    """models/ altinda en yeni meta.json'u bulur ve okur."""
    candidates = sorted((p for p in DEFAULT_MODELS_DIR.glob("*/meta.json")), key=lambda p: p.parent.name)
    if not candidates:
        return None, None
    path = candidates[-1]
    with path.open("r", encoding="utf-8") as handle:
        return path.parent.name, json.load(handle)


def main() -> int:
    """Holdout IC serisini yeniden hesaplar; ham/NW/bagimsiz tablo uretir."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not DEFAULT_PREDICTIONS.is_file():
        logger.error("Holdout tahminleri bulunamadi: %s", DEFAULT_PREDICTIONS)
        return 1
    predictions = pd.read_parquet(DEFAULT_PREDICTIONS)
    predictions["date"] = pd.to_datetime(predictions["date"])

    from training.evaluate import daily_ic

    ic = daily_ic(predictions, "score", "label", min_n=2)
    summary = ic_summary(ic, nw_lag=4)
    stride = stride_ic_summary(ic, stride=5)

    model_name, meta = _latest_model_meta()
    stored_holdout = {}
    if meta is not None:
        stored_holdout = meta.get("holdout_ic_summary") or {}

    rows = []
    rows.append({
        "seri": f"holdout ({model_name or '?'}) - yeniden hesap",
        "n_gun": summary["n_days"],
        "mean_ic": summary["mean_ic"],
        "t_ham": summary["t_stat"],
        "t_nw (lag=4)": summary["t_nw"],
        "n_bagimsiz": stride["stride_n"],
        "mean_ic (bagimsiz)": stride["stride_mean_ic"],
        "t_bagimsiz": stride["stride_t_stat"],
    })
    if stored_holdout:
        rows.append({
            "seri": "holdout - egitim zamani kaydi",
            "n_gun": stored_holdout.get("n_days"),
            "mean_ic": stored_holdout.get("mean_ic"),
            "t_ham": stored_holdout.get("t_stat"),
            "t_nw (lag=4)": None,  # o kosuda NW yoktu; seri kaydedilmedigi icin sonradan hesaplanamaz
            "n_bagimsiz": None,
            "mean_ic (bagimsiz)": None,
            "t_bagimsiz": None,
        })

    report = {
        "generated_at": str(pd.Timestamp.now()),
        "label_horizon": 5,
        "nw_lag": 4,
        "stride": 5,
        "holdout": {
            "n_days": summary["n_days"],
            "mean_ic": summary["mean_ic"],
            "std_ic": summary["std_ic"],
            "icir": summary["icir"],
            "t_raw": summary["t_stat"],
            "t_nw": summary["t_nw"],
            "positive_share": summary["positive_share"],
            "stride_n": stride["stride_n"],
            "stride_mean_ic": stride["stride_mean_ic"],
            "stride_t_stat": stride["stride_t_stat"],
        },
        "stored_holdout_raw": stored_holdout,
        "walk_forward_val": {
            "note": "walk-forward val tahminleri diske kaydedilmemisti; NW/bagimsiz t sonradan hesaplanamaz (yalnizca egitim kaydindaki ham t mevcut)",
        },
        "rows": rows,
    }

    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_REPORT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print(f"\nHoldout IC istatistikleri (n={summary['n_days']} gun):")
    print(f"  mean_ic        : {summary['mean_ic']:+.4f}")
    print(f"  t (ham)        : {summary['t_stat']:+.3f}")
    print(f"  t (NW, lag=4)  : {summary['t_nw']:+.3f}")
    print(f"  bagimsiz ornek : {stride['stride_n']} gun | mean {stride['stride_mean_ic']:+.4f} | t {stride['stride_t_stat']:+.3f}")
    print("\nRapor yazildi: reports/ic_stats_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
