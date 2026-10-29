"""Backtest ana akisi: holdout tahminleri + prices -> metrikler + raporlar.

Ciktilar:
- reports/backtest_summary.json  (mod bazli metrikler + benchmark karsilastirmasi)
- reports/equity_curve.csv       (gross/net/SPY kumulatif serisi)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_BACKTEST_CONFIG_PATH = "config/backtest.yaml"

METRIC_KEYS = (
    "n_days", "total_return", "cagr", "ann_volatility", "sharpe",
    "sortino", "max_drawdown", "calmar", "win_rate",
)


def load_backtest_config(path: str | Path = DEFAULT_BACKTEST_CONFIG_PATH):
    """backtest.yaml yukler ve dogrular (veri yollari proje kokune cozulur)."""
    from evaluation.backtest import BacktestConfig

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Yapilandirma bulunamadi: {config_path}")
    with config_path.open("r", encoding="utf-8-sig") as handle:
        data = yaml.safe_load(handle) or {}
    paths = data.get("paths") or {}
    project_root = config_path.resolve().parent.parent
    predictions_path = str(project_root / paths.get("predictions_path", "data/processed/holdout_predictions.parquet"))
    prices_path = str(project_root / paths.get("prices_path", "data/raw/prices.parquet"))
    reports_dir = str(project_root / paths.get("reports_dir", "reports"))
    cfg = BacktestConfig(
        long_quantile=float(data["long_quantile"]),
        short_quantile=float(data.get("short_quantile", 0.2)),
        rebalance_every=int(data.get("rebalance_every", 5)),
        cost_bps=float(data.get("cost_bps", 10.0)),
        weighting=str(data.get("weighting", "equal")),
        modes=tuple(data.get("modes", ["long_short", "long_only"])),
        benchmark=str(data.get("benchmark", "SPY")),
        predictions_path=predictions_path,
        prices_path=prices_path,
        reports_dir=reports_dir,
    )
    logger.info(
        "BacktestConfig yuklendi: long=%.2f short=%.2f | rebalance=%d | cost=%.0f bp | modes=%s | bench=%s",
        cfg.long_quantile, cfg.short_quantile, cfg.rebalance_every, cfg.cost_bps,
        list(cfg.modes), cfg.benchmark,
    )
    return cfg


def run(config_path: str | Path = DEFAULT_BACKTEST_CONFIG_PATH) -> dict:
    """Backtest akisini uygular; ozet dondurur; rapor dosyalarini yazar."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    from evaluation.backtest import (
        benchmark_returns,
        compare_to_benchmark,
        performance_metrics,
        run_backtest,
    )

    cfg = load_backtest_config(config_path)
    if not Path(cfg.predictions_path).is_file():
        raise FileNotFoundError(
            f"Holdout tahminleri bulunamadi: {cfg.predictions_path} (once training.run calisir)"
        )
    if not Path(cfg.prices_path).is_file():
        raise FileNotFoundError(f"Fiyat verisi bulunamadi: {cfg.prices_path}")

    predictions = pd.read_parquet(cfg.predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"])
    prices = pd.read_parquet(cfg.prices_path)
    prices["date"] = pd.to_datetime(prices["date"])
    logger.info("Tahminler: %d satir | %s .. %s | fiyatlar: %d satir",
                len(predictions), predictions["date"].min().date(), predictions["date"].max().date(), len(prices))

    summary: dict = {"config": {
        "long_quantile": cfg.long_quantile, "short_quantile": cfg.short_quantile,
        "rebalance_every": cfg.rebalance_every, "cost_bps": cfg.cost_bps,
        "modes": list(cfg.modes), "benchmark": cfg.benchmark,
    }}
    equity_frames: dict[str, pd.Series] = {}

    for mode in cfg.modes:
        result = run_backtest(predictions, prices, cfg, mode=mode)
        metrics_gross = performance_metrics(result["gross"])
        metrics_net = performance_metrics(result["net"])
        comparison_gross = compare_to_benchmark(result["gross"], prices, cfg.benchmark)
        turnover_mean = float(result["turnover"].mean()) if len(result["turnover"]) else 0.0
        summary[mode] = {
            "metrics_gross": metrics_gross,
            "metrics_net": metrics_net,
            "avg_turnover_per_rebalance": turnover_mean,
            "missing_returns": result["missing_returns"],
            "vs_benchmark_gross": comparison_gross,
        }
        equity_frames[f"{mode}_gross"] = (1.0 + result["gross"]).cumprod()
        equity_frames[f"{mode}_net"] = (1.0 + result["net"]).cumprod()
        logger.info(
            "%s | gross: total=%.2f%% sharpe=%.2f maxDD=%.2f%% | net: total=%.2f%% sharpe=%.2f maxDD=%.2f%%",
            mode,
            100 * metrics_gross["total_return"], metrics_gross["sharpe"], 100 * metrics_gross["max_drawdown"],
            100 * metrics_net["total_return"], metrics_net["sharpe"], 100 * metrics_net["max_drawdown"],
        )
        bench = benchmark_returns(prices, cfg.benchmark, result["gross"].index)
        equity_frames[f"{cfg.benchmark.lower()}"] = (1.0 + bench).cumprod()

    reports_dir = Path(cfg.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "backtest_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    logger.info("Backtest ozeti yazildi: %s", summary_path)

    equity = pd.DataFrame(equity_frames)
    equity_path = reports_dir / "equity_curve.csv"
    equity.to_csv(equity_path, encoding="utf-8-sig")
    logger.info("Equity curve yazildi: %s", equity_path)
    return {"summary_path": str(summary_path), "equity_path": str(equity_path), "summary": summary}


if __name__ == "__main__":
    run()
