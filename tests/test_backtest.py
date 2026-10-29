"""Adim 4 (FAZ 2) backtest testleri — sentetik fiyat/tahmin, ag takipsiz."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.backtest import (
    BacktestConfig,
    apply_costs,
    benchmark_returns,
    build_signals,
    close_pivot,
    performance_metrics,
    run_backtest,
)

CFG = BacktestConfig(
    long_quantile=0.5, short_quantile=0.5, rebalance_every=5, cost_bps=10.0,
    weighting="equal", modes=("long_short", "long_only"), benchmark="SPY",
    predictions_path="x", prices_path="y", reports_dir="z",
)


def make_prices(days: int = 40, tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"), seed: int = 0) -> pd.DataFrame:
    """Uzun formatli sentetik fiyat tablosu (date, ticker, adj_close)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    level = {t: 100.0 for t in tickers}
    for date in dates:
        for ticker in tickers:
            level[ticker] *= float(1 + rng.normal(0.0003, 0.005))
            rows.append({"date": date, "ticker": ticker, "adj_close": level[ticker]})
    return pd.DataFrame(rows)


# build_signals ---------------------------------------------------------------


def test_long_short_toplam_agirligi_sifir_long_only_bir():
    dates = list(pd.bdate_range("2024-01-01", periods=10))
    predictions = pd.DataFrame({
        "date": dates * 4,
        "ticker": [t for _ in range(10) for t in ("AAA", "BBB", "CCC", "DDD")],
        "score": np.tile(np.array([0.4, 0.2, -0.1, -0.3]), 10),
    })
    ls = build_signals(predictions, CFG, mode="long_short")
    for _day, group in ls.groupby("date"):
        assert float(group["weight"].sum()) == pytest.approx(0.0, abs=1e-9)  # dolar-notr
        assert float(group.loc[group["weight"] > 0, "weight"].sum()) == pytest.approx(0.5, abs=1e-9)
    lo = build_signals(predictions, CFG, mode="long_only")
    for _day, group in lo.groupby("date"):
        assert float(group["weight"].sum()) == pytest.approx(1.0, abs=1e-9)  # tam mevzi


def test_build_signals_rebalance_gunlerinde_uretilir():
    dates = list(pd.bdate_range("2024-01-01", periods=10))
    predictions = pd.DataFrame({
        "date": dates * 2,
        "ticker": [t for _ in range(10) for t in ("AAA", "BBB")],
        "score": np.tile(np.array([0.9, -0.9]), 10),
    })
    signals = build_signals(predictions, CFG, mode="long_only")
    all_days = sorted(predictions["date"].unique())
    assert set(signals["date"].unique()) == set(all_days[::5])  # 0 ve 5. pozisyonlar


# run_backtest ----------------------------------------------------------------


def test_mukemmel_sinyal_kar_ters_sinyal_zarar():
    """AAA her gun +1%, DDD her gun -1% bilesik; perfect sinyal AAA'yi secer."""
    dates = pd.bdate_range("2024-01-01", periods=8)
    tickers = ("AAA", "BBB", "CCC", "DDD")
    growth = {"AAA": 1.01, "BBB": 1.0, "CCC": 1.0, "DDD": 0.99}
    rows = []
    levels = {t: 100.0 for t in tickers}
    for date in dates:
        for ticker in tickers:
            levels[ticker] *= growth[ticker]
            rows.append({"date": date, "ticker": ticker, "adj_close": levels[ticker]})
    prices = pd.DataFrame(rows)

    pred_rows = []
    for date in dates[:-1]:
        for rank, ticker in enumerate(tickers):
            pred_rows.append({"date": date, "ticker": ticker, "score": 3 - rank})
    predictions = pd.DataFrame(pred_rows)

    result = run_backtest(predictions, prices, CFG, mode="long_only")
    # ilk getiri gunu pozisyonsuz (0), sonra: 0.5x(AAA %1 + BBB %0) = %0.5
    assert float(result["gross"].iloc[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(result["gross"].iloc[1:].mean()) == pytest.approx(0.005, abs=1e-12)

    reversed_predictions = predictions.copy()
    reversed_predictions["score"] = -reversed_predictions["score"]
    result_rev = run_backtest(reversed_predictions, prices, CFG, mode="long_only")
    # ters sinyal: DDD(-1%) + CCC(0%) -> gunluk %-0.5
    assert float(result_rev["gross"].iloc[1:].mean()) == pytest.approx(-0.005, abs=1e-12)


def test_gecikme_testi_look_ahead_yok():
    """r(T) getirisini yakalayan pozisyon T-1 kapanisinda kurulmus olmali.

    Yapi: A'nin T indexli getirisi tekinde %+1, ciftinde %-1; B her gun
    ters yonde. Mukemmel sinyal t gununde (t+1)'in kazananini bilir.
    DOGRU motor: getiri gunu T -> T-1 kapanis sinyali -> hep kazanan (+1%).
    Sinyali BIR GUN GERIYE kaydirirsak: sinyal artik (t+2)'yi biliyor;
    (t+1) kazananini kacirir -> her gun kayip.
    """
    dates = pd.bdate_range("2024-01-01", periods=12)
    rows = []
    level_a = 100.0
    level_b = 100.0
    for index, date in enumerate(dates):
        if index > 0:
            rate_a = +0.01 if index % 2 == 1 else -0.01
            level_a *= 1 + rate_a
            level_b *= 1 - rate_a
        rows.append({"date": date, "ticker": "AAA", "adj_close": level_a})
        rows.append({"date": date, "ticker": "BBB", "adj_close": level_b})
    prices = pd.DataFrame(rows)

    winner = {index: ("AAA" if index % 2 == 1 else "BBB") for index in range(len(dates))}
    pred_rows = []
    for t in range(len(dates) - 1):  # sinyallerin ilk 11 gunu; t+1 <= 11 sinirli
        for ticker in ("AAA", "BBB"):
            score = 1.0 if ticker == winner[t + 1] else -1.0
            pred_rows.append({"date": dates[t], "ticker": ticker, "score": score})
    predictions = pd.DataFrame(pred_rows)

    cfg = BacktestConfig(
        long_quantile=0.5, short_quantile=0.0, rebalance_every=1, cost_bps=0.0,
        weighting="equal", modes=("long_only",), benchmark="SPY",
        predictions_path="x", prices_path="y", reports_dir="z",
    )
    correct = run_backtest(predictions, prices, cfg, mode="long_only")
    assert float(correct["gross"].iloc[0]) == pytest.approx(0.0, abs=1e-12)  # ilk gun pozisyon yok
    assert float(correct["gross"].iloc[1:].mean()) == pytest.approx(0.01, abs=1e-9)
    assert float(correct["gross"].iloc[1:].min()) == pytest.approx(0.01, abs=1e-9)

    # sinyali BIR GUN GERIYE OTELE: score(t) artik winner(t)'i biliyor (gec bilgi).
    # Bu sinyal r(t+1) icin kullanilir; winner(t+1) = winner(t)'in tersi (alternasyon)
    # -> her gun kaybeder: gross = -1% (dogru motor +1% idi: fark net gozuksun).
    late_rows = []
    for t in range(1, len(dates) - 1):
        for ticker in ("AAA", "BBB"):
            score = 1.0 if ticker == winner[t] else -1.0  # BILGISI TAZE DEGIL: winner(t)
            late_rows.append({"date": dates[t], "ticker": ticker, "score": score})
    late_predictions = pd.DataFrame(late_rows)
    late = run_backtest(late_predictions, prices, cfg, mode="long_only")
    mean_late = float(late["gross"].iloc[1:].mean())
    assert mean_late == pytest.approx(-0.01, abs=1e-9)  # gec sinyal her gun kaybeder


def test_cost_bps_arttikca_net_getiri_monoton_dusuyor():
    prices = make_prices(days=30, seed=4)
    rng = np.random.default_rng(6)
    dates = sorted(prices["date"].unique())
    rows = [{"date": day, "ticker": ticker, "score": float(rng.normal())}
            for day in dates for ticker in ("AAA", "BBB", "CCC")]
    predictions = pd.DataFrame(rows)

    nets = {}
    for cost in (0.0, 10.0, 30.0):
        cfg = BacktestConfig(
            long_quantile=0.33, short_quantile=0.33, rebalance_every=5, cost_bps=cost,
            weighting="equal", modes=("long_short",), benchmark="SPY",
            predictions_path="x", prices_path="y", reports_dir="z",
        )
        result = run_backtest(predictions, prices, cfg, mode="long_short")
        nets[cost] = float(result["net"].mean())
    assert nets[0.0] > nets[10.0] > nets[30.0]  # maliyet arttikca katmarartir deme


def test_sifir_turnover_sifir_maliyet():
    """Ranklar her gun ayni ise ilk giris haric turnover=0 -> net==gross."""
    dates = pd.bdate_range("2024-01-01", periods=11)
    prices = make_prices(days=11, seed=8)
    rows = [{"date": day, "ticker": t, "score": {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}[t]}
            for day in dates for t in ("AAA", "BBB", "CCC")]
    predictions = pd.DataFrame(rows)
    cfg = BacktestConfig(
        long_quantile=0.33, short_quantile=0.33, rebalance_every=5, cost_bps=50.0,
        weighting="equal", modes=("long_short",), benchmark="SPY",
        predictions_path="x", prices_path="y", reports_dir="z",
    )
    result = run_backtest(predictions, prices, cfg, mode="long_short")
    turnover = result["turnover"].to_list()
    assert turnover[0] > 0  # giriste jutulma
    assert all(t == pytest.approx(0.0, abs=1e-9) for t in turnover[1:])
    # net[1] = gross[1] - giris maliyeti; ndan sonra net == gross
    entry_cost = turnover[0] * 50.0 / 10000.0
    assert float(result["net"].iloc[1]) == pytest.approx(float(result["gross"].iloc[1]) - entry_cost, abs=1e-12)
    assert np.allclose(result["net"].to_numpy()[2:], result["gross"].to_numpy()[2:])


def test_performance_metrics_elle_hesaplanan_degerler():
    returns = pd.Series([0.01, -0.02, 0.02])
    metrics = performance_metrics(returns)
    expected_total = (1.01 * 0.98 * 1.02) - 1.0
    assert metrics["total_return"] == pytest.approx(expected_total, abs=1e-12)
    assert metrics["n_days"] == 3
    expected_ann_vol = float(np.std([0.01, -0.02, 0.02], ddof=1)) * np.sqrt(252)
    assert metrics["ann_volatility"] == pytest.approx(expected_ann_vol, abs=1e-12)
    assert metrics["max_drawdown"] == pytest.approx((1.01 * 0.98) / 1.01 - 1.0, abs=1e-12)
    assert metrics["win_rate"] == pytest.approx(2 / 3, abs=1e-12)


def test_benchmark_returns_hizali():
    prices = make_prices(days=20, seed=2, tickers=("AAA", "SPY"))
    pivot = close_pivot(prices)
    dates = pd.DatetimeIndex(pivot.index[1:10])
    bench = benchmark_returns(prices, "SPY", dates)
    expected = pivot["SPY"].pct_change(fill_method=None).reindex(dates)
    assert np.allclose(bench.to_numpy(), expected.to_numpy())


def test_apply_costs_noktasal():
    returns = pd.Series([0.01, 0.01, 0.01], index=pd.bdate_range("2024-01-01", periods=3))
    turnover = pd.Series([0.0, 1.0, 0.0], index=returns.index)
    net = apply_costs(returns, turnover, cost_bps=25.0)
    assert float(net.iloc[0]) == pytest.approx(0.01)
    assert float(net.iloc[1]) == pytest.approx(0.01 - 1.0 * 25 / 10000)
    assert float(net.iloc[2]) == pytest.approx(0.01)
