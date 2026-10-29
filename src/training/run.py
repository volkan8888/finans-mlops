"""Egitim ana akisi: holdout ayir -> walk-forward -> model sec -> holdout'ta tek degerlendirme.

SIZINTI GARANTILERI:
- Holdout son N yil; HIPERPARAMETRE/model seciminde KULLANILMAZ; en sonda
  BIR KERE raporlanir.
- Walk-forward fold'lari kronolojik; train ile val arasinda PURGE+EMBARGO
  var (splits.py).
- Ek olcekleme YOK: feature'lar cross-sectional z-score'lu;
  model girdisi zaten standart (gerekce: docs/step3_training.md).
- Model secimi SADECE walk-forward VAL IC'sine gore yapilir.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch

from training.config import (
    DEFAULT_TRAINING_CONFIG_PATH,
    TrainingConfig,
    load_training_config,
)
from training.dataset import Dataset, SequenceDataset, TabularDataset
from training.evaluate import daily_ic, ic_summary, quantile_returns
from training.splits import holdout_split, walk_forward_splits
from training.train import META_COLUMNS, SEQUENCE_MODELS, predict_scores, train_model

logger = logging.getLogger(__name__)


def load_features(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """features.parquet okur, (date, ticker) siralanmis dondurur; feature kolonlarini tespit eder."""
    if not path.is_file():
        raise FileNotFoundError(f"features.parquet bulunamadi: {path} (once feature akisini calistirin)")
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValueError("features.parquet bos; once ingestion + features akisini calistirin")
    missing_meta = [c for c in ("date", "ticker", "forward_return", "label") if c not in frame.columns]
    if missing_meta:
        raise ValueError(f"features.parquet'te temel kolonlar yok: {missing_meta}")
    feature_cols = [c for c in frame.columns if c not in META_COLUMNS]
    if not feature_cols:
        raise ValueError("features.parquet'te feature kolonu bulunamadi")
    nan_counts = frame[feature_cols].isna().sum()
    if int(nan_counts.sum()) > 0:
        logger.warning("Feature kolonlarinda NaN mevcut: %s", nan_counts[nan_counts > 0].to_dict())
    return frame, feature_cols


def _rows_for_dates(panel: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    return panel[panel["date"].isin(dates)].reset_index(drop=True)


def _predict(
    model: object,
    model_name: str,
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    feature_cols: list[str],
    cfg: TrainingConfig,
    device: str,
) -> pd.DataFrame:
    """Verilen tarihler icin skor tablosu (date, ticker, score, label) dondurur."""
    rows = _rows_for_dates(panel, dates)
    if model_name == "ridge":
        scores = model.predict(rows[feature_cols].to_numpy())
        return rows.assign(score=scores)[["date", "ticker", "score", "label", "forward_return"]]
    if model_name in SEQUENCE_MODELS:
        dataset: Dataset = SequenceDataset(panel, dates=dates, feature_cols=feature_cols, seq_len=cfg.seq_len)
    else:
        dataset = TabularDataset(rows, feature_cols)
    predictions = predict_scores(model, dataset, device=device)
    # torch verilerinden label donar; forward_return'i panel geri bagla
    predictions = predictions.drop(columns=["label"])
    predictions = predictions.merge(
        panel[["date", "ticker", "label", "forward_return"]], on=["date", "ticker"], how="left"
    )
    return predictions


def _fold_report(fold_index: int, train_dates: pd.DatetimeIndex, val_dates: pd.DatetimeIndex, predictions: pd.DataFrame) -> dict[str, Any]:
    """Bir fold'un val tahminleri uzerinden ozet dondurur (gunluk IC bazli)."""
    ic = daily_ic(predictions, "score", "label", min_n=2)
    summary = ic_summary(ic)
    return {
        "fold": fold_index,
        "train_start": str(train_dates.min().date()),
        "train_end": str(train_dates.max().date()),
        "val_start": str(val_dates.min().date()),
        "val_end": str(val_dates.max().date()),
        "val_days": summary["n_days"],
        "val_mean_ic": summary["mean_ic"],
        "val_icir": summary["icir"],
    }


def _select_best(
    variant_combined: dict[str, pd.DataFrame],
) -> tuple[str, dict[str, Any]]:
    """Tum walk-forward val tahminlerinin toplam IC ozetine gore en iyi varyanti secer.

    Kural: en yuksek mean_ic; esitlikte daha yuksek ICIR. HOLDOUT bu karara
    girmez (sizinti kurali: secim sadece val fold'larindan).
    """
    ranked: list[tuple[str, dict[str, float]]] = []
    for name, combined in variant_combined.items():
        ic = daily_ic(combined, "score", "label", min_n=2)
        summary = ic_summary(ic)
        ranked.append((name, summary))
    ranked.sort(key=lambda item: (float(item[1]["mean_ic"]), float(item[1]["icir"])), reverse=True)
    best_name, best_summary = ranked[0]
    return best_name, best_summary


def _save_torch_model(model: torch.nn.Module, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")


def run(config_path: str | Path = DEFAULT_TRAINING_CONFIG_PATH) -> dict[str, Any]:
    """Egitim akisini uc-uc uygular; holdout metrik + artefakt yazar; ozet dondurur."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = load_training_config(config_path)
    panel, feature_cols = load_features(cfg.features_path)
    all_dates = pd.DatetimeIndex(panel["date"].unique()).sort_values()
    logger.info("Panel: %d satir | %d gun | %d ticker | %d feature", len(panel), len(all_dates),
                panel["ticker"].nunique(), len(feature_cols))

    dev_dates, holdout_dates = holdout_split(
        all_dates, cfg.holdout_years, cfg.purge_days, cfg.embargo_days
    )
    logger.info("Dev: %s .. %s (%d gun) | Holdout: %s .. %s (%d gun)",
                dev_dates.min().date(), dev_dates.max().date(), len(dev_dates),
                holdout_dates.min().date(), holdout_dates.max().date(), len(holdout_dates))

    folds = walk_forward_splits(
        dev_dates, cfg.wf_min_train_years, cfg.wf_val_months, cfg.wf_step_months,
        cfg.purge_days, cfg.embargo_days,
    )
    # son val penceresi dev sonuna sigmiyorsa 2-3 gunluk kalir; gurultu olur,
    # bu tur kisa val'lari at (guvenli ve belgelenmis bir ince ayar)
    folds = [(train, val) for train, val in folds if len(val) >= 10]
    if not folds:
        raise ValueError("Gecerli walk-forward fold kalmadi (val >= 10 gun)")
    logger.info("Walk-forward fold sayisi: %d", len(folds))

    device = cfg.device if cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Cihaz: %s (cfg.device=%s)", device, cfg.device)

    variant_fold_rows: dict[str, list[dict[str, Any]]] = {}
    variant_combined: dict[str, pd.DataFrame] = {}
    for entry in cfg.models:
        model_name = entry["name"]
        logger.info("=== VARIYANT: %s | params=%s", model_name, entry["params"])
        fold_reports: list[dict[str, Any]] = []
        combined_parts: list[pd.DataFrame] = []
        for fold_index, (fold_train_dates, fold_val_dates) in enumerate(folds, start=1):
            fold_train = _rows_for_dates(panel, fold_train_dates)
            fold_val = _rows_for_dates(panel, fold_val_dates)
            model, _history = train_model(
                model_name, fold_train, fold_val, cfg,
                feature_cols=feature_cols, full_df=panel,
            )
            predictions = _predict(model, model_name, panel, fold_val_dates, feature_cols, cfg, device)
            report = _fold_report(fold_index, fold_train_dates, fold_val_dates, predictions)
            logger.info(
                "fold %02d | train %s..%s | val %s..%s | val_mean_ic=%.4f (%d gun)",
                fold_index, report["train_start"], report["train_end"],
                report["val_start"], report["val_end"],
                report["val_mean_ic"], report["val_days"],
            )
            fold_reports.append(report)
            combined_parts.append(predictions)
        variant_fold_rows[model_name] = fold_reports
        variant_combined[model_name] = pd.concat(combined_parts, ignore_index=True)

    best_name, best_summary = _select_best(variant_combined)
    logger.info("SECIM (yalnizca walk-forward val IC): best=%s | summary=%s", best_name, best_summary)
    for name, combined_predictions in variant_combined.items():
        order_ic = daily_ic(combined_predictions, "score", "label", min_n=2)
        logger.info("Karne | %-6s | %s", name, ic_summary(order_ic))

    # FINAL egitim: son val dilimi ile early-stop; tum dev'i KULLANMAZ (purge kurallari)
    final_train_dates, final_val_dates = folds[-1]
    final_train = _rows_for_dates(panel, final_train_dates)
    final_val = _rows_for_dates(panel, final_val_dates)
    final_model, final_history = train_model(
        best_name, final_train, final_val, cfg, feature_cols=feature_cols, full_df=panel
    )
    logger.info("Final egitim %s: %d epoch | history=%s", best_name,
                len(final_history.get("epoch", [])) if isinstance(final_history, dict) else "?",
                {k: (v[-1] if isinstance(v, list) and v else v) for k, v in final_history.items()})

    # HOLDOUT: yalnizca BIR KERE degerlendirilir
    holdout_predictions = _predict(final_model, best_name, panel, holdout_dates, feature_cols, cfg, device)
    holdout_ic = daily_ic(holdout_predictions, "score", "label", min_n=2)
    holdout_summary = ic_summary(holdout_ic, nw_lag=cfg.label_horizon - 1)
    holdout_quantiles = quantile_returns(holdout_predictions, "score", "forward_return", n_quantiles=5)
    logger.info("HOLDOUT IC ozeti: %s", holdout_summary)
    logger.info("HOLDOUT kantil getiri (ham, ortalama):\n%s", holdout_quantiles.to_string(index=False))

    cfg.holdout_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_predictions.to_parquet(cfg.holdout_predictions_path, index=False)
    logger.info("Holdout tahminleri yazildi: %s (%d satir)", cfg.holdout_predictions_path, len(holdout_predictions))

    model_dir = Path(cfg.models_dir) / best_name
    meta = {
        "model_name": best_name,
        "feature_cols": feature_cols,
        "seed": cfg.seed,
        "device": device,
        "loss": cfg.loss,
        "label_horizon": cfg.label_horizon,
        "purge_days": cfg.purge_days,
        "embargo_days": cfg.embargo_days,
        "seq_len": cfg.seq_len,
        "model_params": entry_params(cfg, best_name),
        "train_dates": [str(final_train_dates.min().date()), str(final_train_dates.max().date())],
        "early_stop_val": [str(final_val_dates.min().date()), str(final_val_dates.max().date())],
        "holdout_range": [str(holdout_dates.min().date()), str(holdout_dates.max().date())],
        "final_val_ic": final_history.get("best_val_ic") if isinstance(final_history, dict) else None,
        "holdout_ic": holdout_summary,
        "fold_reports": variant_fold_rows[best_name],
        "model_comparison": {
            name: ic_summary(daily_ic(variant_combined[name], "score", "label", min_n=2))
            for name in variant_combined
        },
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(final_model, torch.nn.Module):
        torch.save(final_model.state_dict(), model_dir / "model.pt")
    else:
        joblib.dump(final_model, model_dir / "model.joblib")
    with (model_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
    logger.info("Model kaydedildi: %s", model_dir)

    # ---- deney takibi (FAZ 3): varyant run'lari + final + registry/alias
    variant_run_ids = {}
    context = None
    if cfg.mlflow.get("enabled", False):
        try:
            from training.mlflow_track import (
                finalize_production_run,
                log_variant_run,
                setup_tracking,
            )

            context = setup_tracking(cfg.mlflow, cfg.models_dir.parent)
        except Exception as error:  # noqa: BLE001 - takip araci akisi bloklayamaz
            logger.warning("MLflow backend kurulamadi (%s); deney takibi atlanir", error)
            context = None
    summary = {
        "best_model": best_name,
        "best_summary": best_summary,
        "holdout_summary": holdout_summary,
        "holdout_quantiles": holdout_quantiles,
        "fold_reports": variant_fold_rows,
        "model_dir": str(model_dir),
        "holdout_predictions_path": str(cfg.holdout_predictions_path),
    }
    if context is not None:
        mlflow, client, _tracking_uri, experiment_id = context
        try:
            for name, combined_predictions in variant_combined.items():
                combined_ic = daily_ic(combined_predictions, "score", "label", min_n=2)
                variant_run_ids[name] = log_variant_run(
                    mlflow, experiment_id, name,
                    entry_params(cfg, name) | {"defaults_loss": cfg.loss},
                    variant_fold_rows[name], ic_summary(combined_ic),
                )
            # backtest metrikleri (FAZ 2 motoru, holdout tahminleri uzerinden)
            backtest_summary = _holdout_backtest_metrics(holdout_predictions, cfg)
            final_val_ic = final_history.get("best_val_ic") if isinstance(final_history, dict) else None
            mlflow_metrics = {
                "final_val_ic": float(final_val_ic) if final_val_ic is not None else float("nan"),
                "holdout_mean_ic": holdout_summary["mean_ic"],
                "holdout_icir": holdout_summary["icir"],
                "holdout_t_stat": holdout_summary["t_stat"],
                "holdout_t_stat_nw": holdout_summary["t_nw"],
                "holdout_positive_share": holdout_summary["positive_share"],
                **backtest_summary,
            }
            artifact_map = {"meta.json": str(model_dir / "meta.json")}
            model_file = model_dir / ("model.pt" if isinstance(final_model, torch.nn.Module) else "model.joblib")
            artifact_map[model_file.name] = str(model_file)
            artifact_map["holdout_predictions.parquet"] = str(cfg.holdout_predictions_path)
            version = finalize_production_run(
                mlflow, client, best_name, experiment_id,
                variant_run_id=variant_run_ids.get(best_name),
                metrics=mlflow_metrics,
                artifact_paths=artifact_map,
                registry_name=str(cfg.mlflow.get("registry_name", "finans-ranker")),
            )
            summary["mlflow"] = {
                "variant_runs": variant_run_ids, "registry_version": version,
                "backtest_metrics_logged": backtest_summary,
            }
        except Exception as error:  # noqa: BLE001 - takip araci akisi bloklayamaz
            logger.warning("MLflow kayit hatasi (%s); egitim sonuclarina etki etmez", error)
    return summary


def entry_params(cfg: TrainingConfig, name: str) -> dict[str, Any]:
    """Config'ten, secilen modelin params sozlugunu dondurur."""
    for entry in cfg.models:
        if entry["name"] == name:
            return dict(entry["params"])
    return {}


def _holdout_backtest_metrics(holdout_predictions: pd.DataFrame, cfg: TrainingConfig) -> dict[str, float]:
    """Holdout tahminleri uzerinde hizli backtest; MLflow'a yazilacak metrikler.

    Parazit baglilik: backtest.yaml bulundugunda kullanilir; bulunmazsa bos
    doner (egitim akisi bu yuzden bloklanmaz).
    """
    try:
        from evaluation.backtest import performance_metrics, run_backtest
        from evaluation.run import load_backtest_config

        bt_cfg = load_backtest_config()
        prices = pd.read_parquet(bt_cfg.prices_path)
        prices["date"] = pd.to_datetime(prices["date"])
        metrics: dict[str, float] = {}
        for mode in bt_cfg.modes:
            result = run_backtest(holdout_predictions, prices, bt_cfg, mode=mode)
            net = performance_metrics(result["net"])
            metrics[f"bt_{mode}_net_total_return"] = float(net["total_return"])
            metrics[f"bt_{mode}_net_sharpe"] = float(net["sharpe"])
            metrics[f"bt_{mode}_net_max_drawdown"] = float(net["max_drawdown"])
        return metrics
    except Exception as error:  # noqa: BLE001 - rapor parcasi eksik kalabilir
        logger.warning("Holdout backtest metrikleri uretilemedi (%s); MLflow bu kisim eksik", error)
        return {}


if __name__ == "__main__":
    run()
