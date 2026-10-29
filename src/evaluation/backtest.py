"""Backtest motoru: sinyal -> pozisyon -> gunluk getiri -> metrikler.

LOOK-AHEAD ONLEMI (en kritik kural):
- Sinyal, t kapanisindaki skorlardan uretilir.
- Sinyal, t KAPANISINDE isleme cevrilir: pozisyon t kapanista kurulur ve
  getiri t+1 gununden itibaren o agirliklarla isletilir. ONEMLI kural:
  r(T) = P(T)/P(T-1) - 1 getirisini yakalayan pozisyon, T-1 KAPANISINDA
  kurulmus olmalidir. Bu motorda: i pozisyondaki getiri gunu T icin
  agirliklar, T-1 (= bir onceki islem gunu) kapanisina kadarki SON
  REBALANCE gununun sinyalinden gelir. Rebalance gunu p'nin etkisi,
  p+1 getiri gununden baslar. Ayni gunun skoruna ayni gun getiri
  yazan her sey sizintidir; burada yasak.

DRIFT BASITLESTIRMESI (belgeli):
- Rebalance gunleri arasinda agirliklar SABIT tutulur; hisselerin
  degisen olcekleri (fiyat arttarsa agirligi buyur) MODELLENMEZ. Basit
  ama ucuz bir yontem; small equal-weight portfoylerde etkisi kucuk.
- Hisse getiri verisi yoksa (askida/eksik bar) O GUN katki 0 sayilir ve
  sayaci loglanir (doldurma yapilmaz, fark gizlenmez).

MALIYET: turnover = rebalance gununde degisen agirlik toplami
(|delta w| buy+seller birim); her birim icin cost_bps/10000 bedel, temerrut
rebalance'in SONRAKI getiri gununde dusulur (islem kapanista yapilir,
getiri bir sonraki gunden akar).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    """backtest.yaml icerigi (dogrulanmis)."""

    long_quantile: float
    short_quantile: float
    rebalance_every: int
    cost_bps: float
    weighting: str
    modes: tuple[str, ...]
    benchmark: str
    predictions_path: str
    prices_path: str
    reports_dir: str

    def __post_init__(self):
        if not 0.0 < self.long_quantile <= 1.0:
            raise ValueError(f"long_quantile (0,1] araliginda olmali: {self.long_quantile}")
        if not 0.0 <= self.short_quantile <= 1.0:
            raise ValueError(f"short_quantile [0,1] araliginda olmali: {self.short_quantile}")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every >= 1 olmali")
        if self.cost_bps < 0:
            raise ValueError("cost_bps negatif olamaz")
        if self.weighting != "equal":
            raise ValueError("yalnizca 'equal' agirlik vardir")
        for mode in self.modes:
            if mode not in {"long_short", "long_only"}:
                raise ValueError(f"bilinmeyen mod: {mode}")


def close_pivot(prices: pd.DataFrame, column: str = "adj_close") -> pd.DataFrame:
    """Uzun formatli fiyat panelinden (date x ticker) kapanis matrisi uretir."""
    missing = [c for c in ("date", "ticker", column) if c not in prices.columns]
    if missing:
        raise ValueError(f"Fiyat tablosunda kolon yok: {missing}")
    pivot = prices.pivot_table(index="date", columns="ticker", values=column, aggfunc="last")
    return pivot.sort_index()


def build_signals(pred_df: pd.DataFrame, cfg: BacktestConfig, mode: str) -> pd.DataFrame:
    """Rebalance gunlerinde hedef agirliklar; donus: (date, ticker, weight).

    Kural: her rebalance gununde skora gore cross-sectional siralama yapilir;
    ust `long_quantile` oran long, alt `short_quantile` oran short olur.
    Agirliklar leg icinde esit: long_only'de toplam +1, long_short'ta her
    leg +0.5/-0.5 (dolar-notr). mode='long_only' ise short leg zaten yoktur.
    """
    if mode not in {"long_short", "long_only"}:
        raise ValueError(f"bilinmeyen mod: {mode}")
    if pred_df.empty:
        raise ValueError("tahmin tablosu bos")
    required = {"date", "ticker", "score"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"tahmin tablosunda kolon yok: {missing}")

    days = pd.DatetimeIndex(sorted(pred_df["date"].unique()))
    rebalance_days = set(days[:: max(1, int(cfg.rebalance_every))])

    rows: list[tuple[pd.Timestamp, str, float]] = []
    for day in days:
        if day not in rebalance_days:
            continue
        scores = pred_df[pred_df["date"] == day].dropna(subset=["score"])
        n = len(scores)
        if n == 0:
            continue
        n_long = max(1, round(cfg.long_quantile * n))
        ordered = scores.sort_values("score", ascending=False)
        long_leg = ordered.head(n_long)
        leg_total = 1.0 if mode == "long_only" else 0.5
        for _, row in long_leg.iterrows():
            rows.append((day, row["ticker"], leg_total / len(long_leg)))
        if mode == "long_short" and cfg.short_quantile > 0:
            n_short = max(1, round(cfg.short_quantile * n))
            short_leg = ordered.tail(n_short)
            for _, row in short_leg.iterrows():
                rows.append((day, row["ticker"], -leg_total / len(short_leg)))
    return pd.DataFrame(rows, columns=["date", "ticker", "weight"])


def run_backtest(
    pred_df: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: BacktestConfig,
    mode: str,
) -> dict:
    """Gunluk portfoy getirisi serisi olusturur (maliyetsiz ve maliyetli).

    Donum: {"gross": Series, "net": Series, "turnover": Series,
            "weights": DataFrame, "missing_returns": int}
    - gross/net indexi GETIRI gunudur (T); agirliklar T-1 kapanisindan gelir.
    - missing_returns: getiri verisi eksik gun-katki sayisi (0 katki + log).
    """
    if mode not in cfg.modes:
        raise ValueError(f"mode '{mode}' config'te tanimli degil")
    days = pd.DatetimeIndex(sorted(pred_df["date"].unique()))
    universe = sorted(pred_df["ticker"].unique())
    pivot = close_pivot(prices)
    missing_columns = [ticker for ticker in universe if ticker not in pivot.columns]
    if missing_columns:
        raise ValueError(f"Fiyat pivotunda ticker yok: {missing_columns}")

    # gunluk getiri matrisi (satir indexi = getiri gunu T; T-1 kapanisina gore)
    ret_matrix = pivot.reindex(columns=universe).pct_change(fill_method=None)
    rets = ret_matrix.loc[ret_matrix.index.isin(days)]

    weights_table = build_signals(pred_df, cfg, mode)
    weights_by_day = {
        day: group.set_index("ticker")["weight"].to_dict()
        for day, group in weights_table.groupby("date", sort=True)
    }

    gross_series: list[float] = []
    net_series: list[float] = []
    days_processed: list[pd.Timestamp] = []
    turnover_pairs: list[tuple[pd.Timestamp, float]] = []
    current_weights: dict[str, float] = {}
    pending_cost = 0.0
    missing_count = 0
    rebalance_positions = {
        i for i, _day in enumerate(days) if i % max(1, int(cfg.rebalance_every)) == 0
    }

    for i, day in enumerate(days):
        if day not in rets.index:
            # bu gun icin fiyat yok: sinyal uygulanmaz, getiri yok (loglanir)
            missing_count += 1
            continue
        day_rets = rets.loc[day]
        # 1) bu getiri gunu: ONCEKI kapanista kurulmus agirliklar is gostergesi
        gross = 0.0
        for ticker, weight in current_weights.items():
            ret_value = day_rets.get(ticker, np.nan)
            if pd.isna(ret_value):
                missing_count += 1
                continue
            gross += weight * float(ret_value)
        net = gross - pending_cost
        days_processed.append(day)
        gross_series.append(gross)
        net_series.append(net)
        # 2) bugun rebalance gunu mu? (sinyal bugunun kapanista -> etkisi YARIN)
        if i in rebalance_positions:
            new_weights = weights_by_day.get(day, {})
            turnover = sum(
                abs(new_weights.get(ticker, 0.0) - weight)
                for ticker, weight in current_weights.items()
            ) + sum(
                abs(weight)
                for ticker, weight in new_weights.items() if ticker not in current_weights
            )
            current_weights = dict(new_weights)
            pending_cost = turnover * float(cfg.cost_bps) / 10000.0
            turnover_pairs.append((day, turnover))
        else:
            pending_cost = 0.0

    if missing_count > 0:
        logger.info("%s: %d eksik getiri katkisi (0 kabul edildi)", mode, missing_count)
    index = pd.DatetimeIndex(days_processed)
    return {
        "gross": pd.Series(gross_series, index=index, name=mode),
        "net": pd.Series(net_series, index=index, name=mode),
        "turnover": pd.Series(
            [t for _d, t in turnover_pairs], index=pd.DatetimeIndex([d for d, _t in turnover_pairs]),
            name="turnover",
        ),
        "weights": weights_table,
        "missing_returns": missing_count,
    }


def apply_costs(returns: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Getiri serisinden islem maliyetini dusen iki fonksiyon.

    turnover serisi, o getiri gunune ATANAN islem hacmidir (rebalance'in
    sonraki gune gecmis hali). cost = turnover * cost_bps/10000.
    Tipik kullanim: run_backtest icinde; baskan haliyle de disaridan cagrilabilir.
    """
    return returns - turnover * float(cost_bps) / 10000.0


def performance_metrics(returns: pd.Series, periods: int = 252) -> dict[str, float]:
    """Klasik metrikler; kucuk serilerde bile deterministik calisir.

    - CAGR: (1+toplam) ** (252/n) - 1  (islem gunu bazli)
    - Sortino: downside std = std(min(r,0), ddof=1) tabani (bos ise NaN)
    - max drawdown: (1+r).cumprod() serisinin tepeye gore en kotu kuyrugu
    """
    clean = returns.dropna().astype(float)
    n = len(clean)
    if n <= 1:
        result = {"n_days": n}
        for key in ("total_return", "cagr", "ann_volatility", "sharpe",
                    "sortino", "max_drawdown", "calmar", "win_rate"):
            result[key] = float("nan") if n == 0 else (0.0 if key == "total_return" else float("nan"))
        return result
    total = float((1.0 + clean).prod() - 1.0)
    cagr = float((1.0 + total) ** (periods / n) - 1.0)
    std = float(clean.std(ddof=1))
    ann_vol = std * np.sqrt(periods)
    mean = float(clean.mean())
    sharpe = mean / std * np.sqrt(periods) if std > 0 else float("nan")
    downside = clean[clean < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    sortino = mean / downside_std * np.sqrt(periods) if downside_std and downside_std > 0 else float("nan")
    equity = (1.0 + clean).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "n_days": n,
        "total_return": total,
        "cagr": cagr,
        "ann_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "win_rate": float((clean > 0).mean()),
    }


def benchmark_returns(prices: pd.DataFrame, symbol: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Benchmark (SPY) gunluk getirisi; strategy getiri gunleriyle hizali."""
    pivot = close_pivot(prices)
    if symbol not in pivot.columns:
        raise ValueError(f"benchmark '{symbol}' fiyat panelinde yok")
    bench = pivot[symbol].pct_change(fill_method=None).reindex(dates)
    name = f"{symbol}_buy_hold"
    return bench.rename(name)


def compare_to_benchmark(
    strategy_returns: pd.Series, benchmark_prices: pd.DataFrame, symbol: str
) -> dict:
    """Strateji ile benchmark'u yan yana koyar; ayrica beta/alfa raporlar.

    beta = cov(r_p, r_b)/var(r_b); alfa = mean(r_p - beta*r_b)  (gunluk,
    yillastirilmis alfa raporlanir).
    """
    bench = benchmark_returns(benchmark_prices, symbol, strategy_returns.index)
    joined = pd.concat([
        strategy_returns.rename("strategy"),
        bench.rename(symbol),
    ], axis=1).dropna()
    strategy_metrics = performance_metrics(joined["strategy"])
    benchmark_metrics = performance_metrics(joined[symbol])
    if len(joined) > 2:
        var_b = float(joined[symbol].var(ddof=1))
        beta = float(joined.cov().iloc[0, 1] / var_b) if var_b > 0 else float("nan")
        alpha_daily = float((joined["strategy"] - beta * joined[symbol]).mean()) if np.isfinite(beta) else float("nan")
    else:
        beta = float("nan")
        alpha_daily = float("nan")
    return {
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "beta": beta,
        "alpha_annualized": float(alpha_daily * 252) if np.isfinite(alpha_daily) else float("nan"),
        "alpha_daily": alpha_daily,
    }
