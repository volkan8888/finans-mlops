"""Ileri test raporu: canli tahminlerin gerceklesmesi (label'siz skor -> gercek getiri).

Veri kaynaklari:
- data/predictions/predictions_YYYY-MM-DD.parquet  (daily flow'un degismez kaydi)
- data/raw/prices.parquet                          (adj_close; SPY dahil)

Yontem: her (date, ticker) tahmini icin labels.forward_return ile AYNI
tanim kullanilir: panel islem-gunu takviminde D+horizon. gunun fiyatina
bakilir: P_{D+h} / P_D - 1. H penceresi henuz doldurulmadiysa (yada fiyat
yoksa) kayit ATLANIR ve SAYILIR (unrealized). SPY ayni pencerede
hesaplanir; ust kuantilin farki raporlanir.

IC: gun bazinda spearman(score, realized); t (ham) + t (Newey-West).
Rapor: reports/forward_test/forward_report.json (her kosuda guncellenen kucuk ozet).

Kullanim:
    .venv\\Scripts\\python -m monitoring.forward
"""

from __future__ import annotations

import json
import logging
import math
import pathlib

import pandas as pd

from training.evaluate import DEFAULT_NW_LAG, daily_ic, ic_summary

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS_DIR = DEFAULT_PROJECT_ROOT / "data" / "predictions"
DEFAULT_PRICES_PATH = DEFAULT_PROJECT_ROOT / "data" / "raw" / "prices.parquet"
DEFAULT_REPORT_PATH = DEFAULT_PROJECT_ROOT / "reports" / "forward_test" / "forward_report.json"
DEFAULT_BENCHMARK = "SPY"
DEFAULT_HORIZON = 5
DEFAULT_QUANTILES = 5


def load_prediction_frames(
    directory: str | pathlib.Path = DEFAULT_PREDICTIONS_DIR,
) -> tuple[pd.DataFrame, list[str]]:
    """predictions_*.parquet dosyalari tek frame'de toplar; bozuk dosya atlanir+SAYILIR.

    Donus: (tahminler [date,ticker,score,(model?)], atlanan_dosya_adlari).
    Dosya yoksa bos frame doner.
    """
    directory = pathlib.Path(directory)
    skipped: list[str] = []
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("predictions_*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception as error:  # noqa: BLE001 - bozuk dosya akisi kesmez
            logger.warning("Tahmin dosyasi okunamadi, atlandi: %s (%s)", path.name, error)
            skipped.append(path.name)
            continue
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "score"]), skipped
    predictions = pd.concat(frames, ignore_index=True)
    predictions = predictions.sort_values(["date", "ticker"], ignore_index=True)
    return predictions, skipped


def _close_pivot(prices: pd.DataFrame, benchmark: str) -> tuple[pd.DataFrame, list[str]]:
    """(date,ticker,adj_close) -> date x ticker fiyat matrisi; tarihler sirali."""
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    return panel, list(panel.index)


def realized_forward_returns(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    benchmark: str = DEFAULT_BENCHMARK,
) -> tuple[pd.DataFrame, int, pd.Series]:
    """Tahminlere karsi gerceklesmis D->D+h getiri; unrealized ATLANIP SAYILIR.

    Donus: (gerceklesenler [date, ticker, score, realized, realized_date],
    unrealized_sayisi, benchmark-serisi [date index; D'de D+h getiri]).
    Etiket semantigi (labels.forward_return ile ayni): P_{D+h}/P_D - 1
    (islem-gunu takvimi: panelin benzersiz tarihleri).
    """
    if predictions.empty:
        return pd.DataFrame(columns=["date", "ticker", "score", "realized", "realized_date"]), 0, pd.Series(dtype=float)
    if horizon <= 0:
        raise ValueError(f"horizon pozitif olmali, gelen: {horizon}")
    panel, dates = _close_pivot(prices, benchmark)
    position = {date: idx for idx, date in enumerate(dates)}

    realized_rows: list[dict] = []
    unrealized = 0
    for prediction_date, tickers, scores in zip(
        predictions["date"], predictions["ticker"], predictions["score"], strict=True
    ):
        date_index = position.get(pd.Timestamp(prediction_date))
        if date_index is None or date_index + horizon >= len(dates):
            unrealized += 1  # pencere henuz dolmadi
            continue
        future_date = dates[date_index + horizon]
        series = panel.iloc[date_index]
        future_series = panel.iloc[date_index + horizon]
        base_price = series.get(tickers)
        future_price = future_series.get(tickers)
        if pd.isna(base_price) or pd.isna(future_price):
            unrealized += 1  # ticker o gunde/5 gun sonra yok
            continue
        realized_rows.append({
            "date": pd.Timestamp(prediction_date),
            "ticker": tickers,
            "score": float(scores),
            "realized": float(future_price / base_price - 1.0),
            "realized_date": future_date,
        })

    benchmark_series = pd.Series(dtype=float)
    if benchmark in panel.columns:
        ratios: list[tuple[pd.Timestamp, float]] = []
        values = panel[benchmark]
        for idx in range(len(dates)):
            if idx + horizon >= len(dates):
                break
            start_price, end_price = values.iloc[idx], values.iloc[idx + horizon]
            if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
                continue
            ratios.append((dates[idx], float(end_price / start_price - 1.0)))
        benchmark_series = pd.Series(dict(ratios))
    realized = pd.DataFrame(
        realized_rows, columns=["date", "ticker", "score", "realized", "realized_date"]
    )
    return realized, unrealized, benchmark_series


def quantile_table(
    realized: pd.DataFrame,
    benchmark: pd.Series,
    n_quantiles: int = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Gunluk cross-sectional skor kuantillerine gore gerceklesen getiri ort.

    q0 = en dusuk skor grubu ... q{n-1} = en yuksek skor grubu.
    Her kuantil satiri icin: gun sayisi, ort. getiri, benchmark farki.
    """
    if realized.empty:
        return pd.DataFrame(columns=["quantile", "n_days", "mean_return", "vs_benchmark"])
    frame = realized.copy()
    frame["rank"] = frame.groupby("date")["score"].rank(pct=True)
    # n ticker + n kuantil altinda sinir degerleri (rank=0.25n ...) dogru gruba
    # dusmeli: ceil(rank*n) tabanli gruplama (floor tabanda sonuncu = tasac).
    frame["bin"] = (
        ((frame["rank"] * n_quantiles).apply(math.ceil) - 1)
        .clip(lower=0, upper=n_quantiles - 1)
        .astype(int)
    )
    benchmark_mean = float(benchmark.mean()) if len(benchmark) else float("nan")
    rows: list[dict] = []
    grouped = frame.groupby(["bin"])["realized"]
    for bin_index, mean_value in grouped.mean().items():
        day_count = int(frame[frame["bin"] == bin_index]["date"].nunique())
        difference = mean_value - benchmark_mean if not math.isnan(benchmark_mean) else None
        rows.append({
            "quantile": f"q{bin_index}",
            "n_days": day_count,
            "mean_return": float(mean_value),
            "vs_benchmark": float(difference) if difference is not None else None,
        })
    rows.sort(key=lambda item: item["quantile"])
    return pd.DataFrame(rows)


def forward_test_report(
    predictions_dir: str | pathlib.Path = DEFAULT_PREDICTIONS_DIR,
    prices_path: str | pathlib.Path = DEFAULT_PRICES_PATH,
    horizon: int = DEFAULT_HORIZON,
    nw_lag: int = DEFAULT_NW_LAG,
    n_quantiles: int = DEFAULT_QUANTILES,
    benchmark: str = DEFAULT_BENCHMARK,
    output_path: str | pathlib.Path | None = DEFAULT_REPORT_PATH,
) -> dict:
    """Ileri test raporu: gerceklesen IC + kuantil tablo + unrealized sayimi.

    Veri yoksa status='bos'; pencere henuz dolmadiysa status='tum_beklemede';
    diger durumda 'ok'. Rapor JSON'a yazilir (output_path verilirse).
    """
    predictions, skipped_files = load_prediction_frames(predictions_dir)
    coverage = {
        "n_prediction_days": int(predictions["date"].nunique()) if not predictions.empty else 0,
        "first_date": str(predictions["date"].min().date()) if not predictions.empty else None,
        "last_date": str(predictions["date"].max().date()) if not predictions.empty else None,
        "n_files_skipped": len(skipped_files),
        "skipped_files": skipped_files,
    }
    if predictions.empty:
        report = {"status": "bos", "horizon": horizon, "coverage": coverage, "alerts": ["tahmin dosyasi yok"]}
        logger.info("Ileri test: tahmin dosyasi yok (%s)", predictions_dir)
        return report
    if not pathlib.Path(prices_path).is_file():
        report = {"status": "bos", "horizon": horizon, "coverage": coverage, "alerts": ["fiyat deposu yok"]}
        logger.info("Ileri test: fiyat deposu yok (%s)", prices_path)
        return report

    prices = pd.read_parquet(prices_path)
    realized, unrealized, series = realized_forward_returns(
        predictions, prices, horizon=horizon, benchmark=benchmark
    )
    report: dict = {
        "horizon": horizon,
        "nw_lag": nw_lag,
        "benchmark": benchmark,
        "coverage": coverage,
        "n_realized": len(realized),
        "n_unrealized": int(unrealized),
        "quantiles": quantile_table(realized, series, n_quantiles=n_quantiles).to_dict(orient="records"),
    }
    if realized.empty:
        report["status"] = "tum_beklemede"
        report["alerts"] = [f"5-gunluk pencere henuz dolmadi; {unrealized} tahmin bekletiliyor (atlanip sayildi)"]
        logger.info(
            "Ileri test: pencere dolmadi (%d unrealized; tahmin gunu %s)",
            unrealized, coverage["first_date"],
        )
        return report
    summary = ic_summary(daily_ic(realized, "score", "realized", min_n=2), nw_lag=nw_lag)
    report["status"] = "ok"
    report["overall"] = {
        key: (round(float(value), 4) if not math.isnan(float(value)) else None)
        for key, value in summary.items()
    }
    alerts: list[str] = []
    mean_ic = summary.get("mean_ic")
    if mean_ic is not None and not math.isnan(float(mean_ic)) and float(mean_ic) <= 0.0:
        alerts.append("gerceklesen mean_ic <= 0")
    report["alerts"] = alerts
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "Ileri test raporu yazildi: %s | gerceklesen gun=%d | unrealized=%d | mean_ic=%s | t_nw=%s",
            output_path,
            summary.get("n_days", 0),
            unrealized,
            report["overall"].get("mean_ic"),
            report["overall"].get("t_nw"),
        )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    forward_test_report()
