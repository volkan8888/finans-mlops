"""Adim 2 (asama 1) feature testleri — agsiz, kucuk sahte DataFrame'lerle.

Not: ingestion testleri (test_ingestion.py) bu dosyayla birlikte kosulur ve
yesil kalmak zorundadir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from features.build import build_dataset, build_features
from features.config import FeaturesConfig
from features.cross_section import winsorize_by_date, zscore_by_date
from features.labels import forward_return, rank_label_by_date
from features.technical import (
    compute_returns,
    momentum,
    momentum_skip,
    rsi,
    short_term_reversal,
    volatility,
    volume_change,
)

# ---------------------------------------------------------------------------
# ortak yardimcilar
# ---------------------------------------------------------------------------


def _dates(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _panel(
    paths: dict[str, list[float]],
    volumes: dict[str, list[float]] | None = None,
    dates: int | None = None,
) -> pd.DataFrame:
    """date, ticker, adj_close, volume kolonlu kucuk fiyat paneli kurar."""
    days = _dates(dates or max(len(v) for v in paths.values()))
    rows: list[dict] = []
    for ticker, closes in paths.items():
        vols = (volumes or {}).get(ticker, [1_000_000.0] * len(closes))
        for date, close, vol in zip(days, closes, vols):
            rows.append(
                {"date": date, "ticker": ticker, "adj_close": close, "volume": vol}
            )
    return pd.DataFrame(rows)


def _tiny_cfg(**overrides) -> FeaturesConfig:
    """Kucuk pencereli cfg: testlerde warm-up hizli kapanir."""
    defaults = {
        "momentum_windows": (2, 3),
        "skip_window": 4,
        "skip": 1,
        "volatility_window": 3,
        "rsi_window": 3,
        "volume_window": 3,
        "reversal_window": 2,
        "label_horizon": 2,
        "winsorize_lower": 0.1,
        "winsorize_upper": 0.9,
        "min_tickers_per_date": 2,
        "prices_path": None,
        "output_path": None,
        "benchmarks": ("^GSPC",),
    }
    defaults.update(overrides)
    return FeaturesConfig(**defaults)


# ---------------------------------------------------------------------------
# technical.py
# ---------------------------------------------------------------------------


def test_compute_returns_log_getirisi():
    s = pd.Series([100.0, 110.0, 121.0])
    out = compute_returns(s)
    assert np.isnan(out.iloc[0])                       # gecmis yok
    assert out.iloc[1] == pytest.approx(np.log(1.10))
    # log getiri toplanabilirlik: gunluklerin toplami = kumulatif
    # ln(110/100) + ln(121/110) = ln(121/100) = 2*ln(1.10)
    assert out.iloc[1] + out.iloc[2] == pytest.approx(2 * np.log(1.10), rel=1e-9)


def test_momentum_pencere_manuel():
    s = pd.Series([100.0, 101.0, 102.0, 103.0, 105.0])
    mom = momentum(s, 2)
    assert mom.iloc[:2].isna().all()                   # ilk `window` gun warm-up
    assert mom.iloc[2] == pytest.approx(102.0 / 100.0 - 1.0)
    assert mom.iloc[3] == pytest.approx(103.0 / 101.0 - 1.0)
    assert mom.iloc[4] == pytest.approx(105.0 / 102.0 - 1.0)


def test_momentum_skip_manuel_ve_nan_sayisi():
    s = pd.Series([100.0, 101.0, 102.0, 103.0, 105.0])
    sk = momentum_skip(s, window=3, skip=1)
    assert sk.iloc[:3].isna().all()                    # window=3 -> ilk 3 gun NaN
    assert sk.iloc[3] == pytest.approx(102.0 / 100.0 - 1.0)  # t-1 / t-3
    assert sk.iloc[4] == pytest.approx(103.0 / 101.0 - 1.0)

    long_s = pd.Series(np.arange(1.0, 71.0))           # 70 gunluk seri
    sk60 = momentum_skip(long_s, window=60, skip=5)
    assert sk60.isna().sum() == 60
    # son deger: t-skip (=65) / t-window (=10) - 1
    assert sk60.iloc[-1] == pytest.approx(long_s.iloc[-6] / long_s.iloc[-61] - 1.0)
    # ilk gecerli deger index 60'ta: shift(5)=56, shift(60)=1
    assert sk60.iloc[60] == pytest.approx(long_s.iloc[55] / long_s.iloc[0] - 1.0)


def test_volatility_rolling_std_ddof1():
    returns = pd.Series([0.01, -0.02, 0.03])
    out = volatility(returns, window=3)
    assert out.iloc[:2].isna().all()
    expected = np.std([0.01, -0.02, 0.03], ddof=1)
    assert out.iloc[2] == pytest.approx(expected)


def test_rsi_yukselir_100_duser_0_duz_50():
    up = pd.Series(np.linspace(100.0, 160.0, 40))      # her adim sabit pozitif
    down = pd.Series(np.linspace(160.0, 100.0, 40))    # her adim sabit negatif
    flat = pd.Series(np.full(40, 100.0))               # degisimsiz

    rsi_up = rsi(up, window=14)
    rsi_down = rsi(down, window=14)
    rsi_flat = rsi(flat, window=14)

    assert rsi_up.iloc[-1] == pytest.approx(100.0)     # ort. kayip 0
    assert rsi_down.iloc[-1] == pytest.approx(0.0)     # ort. kazanc 0
    assert rsi_flat.iloc[-1] == pytest.approx(50.0)    # ikisi de 0 -> notr
    assert rsi_up.iloc[:13].isna().all()               # warm-up NaN


def test_volume_change_manuel_ve_sifir_hacim_guvenli():
    s = pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0])
    out = volume_change(s, window=3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(np.log(3000.0 / 2000.0))
    assert out.iloc[4] == pytest.approx(np.log(5000.0 / 4000.0))

    # sifir hacim gunu: ln(0) = -inf yerine NaN donmeli
    zero = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 0.0, 1000.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        out0 = volume_change(zero, window=4)
    assert np.isnan(out0.iloc[4])
    assert np.isfinite(out0.dropna()).all()

    # tum pencere sifir hacim: ortalama 0 -> NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        allzero = volume_change(pd.Series(np.zeros(30)), window=4)
    assert allzero.isna().all()


def test_short_term_reversal_rolling_toplam():
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    out = short_term_reversal(returns, window=2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.01 + (-0.02))
    assert out.iloc[3] == pytest.approx(0.03 + 0.01)
    assert out.iloc[4] == pytest.approx(0.01 + (-0.01))


def test_ticker_karisma_yok_ham_seviye():
    # Not: cross-sectional adimdan SONRA z-score KASITLI olarak tum hisseleri
    # ayni gunde etkiler (bu dogru davranis). Bu yuzden karismazlik testi
    # BILINCLI olarak ham ticker-bazli technical seviyesinde yapilir:
    # A'nin fiyatini degistirmek B'nin SADECE ham feature'larini degistirmemeli.
    dates = _dates(40)
    a = pd.Series(np.linspace(100.0, 140.0, 40), index=dates)
    b = pd.Series(np.linspace(200.0, 260.0, 40), index=dates)

    b_before = {
        "returns": compute_returns(b),
        "momentum": momentum(b, 5),
        "momentum_skip": momentum_skip(b, 10, 2),
        "volatility": volatility(compute_returns(b), 5),
        "rsi": rsi(b, window=5),
        "volume": volume_change(pd.Series(np.full(40, 1000.0), index=dates), 5),
        "reversal": short_term_reversal(compute_returns(b), 3),
    }

    a_mutated = a.copy()
    a_mutated.iloc[-10:] = a_mutated.iloc[-10:] * 1.5   # sadece A'nin sonu bozulur
    _ = momentum(a_mutated, 5)                          # A'nin bozulmus serisinden feature uretir; sonuc kasitli olarak kullanilmaz

    b_after = {
        name: fn(b)
        for name, fn in {
            "returns": compute_returns,
            "momentum": lambda s: momentum(s, 5),
            "momentum_skip": lambda s: momentum_skip(s, 10, 2),
            "volatility": lambda s: volatility(compute_returns(b), 5),
            "rsi": lambda s: rsi(s, window=5),
            "volume": lambda s: volume_change(pd.Series(np.full(40, 1000.0), index=b.index), 5),
            "reversal": lambda s: short_term_reversal(compute_returns(b), 3),
        }.items()
    }

    for name, series_before in b_before.items():
        assert_series_equal_ignoring_nan(series_before, b_after[name], name=name)


def assert_series_equal_ignoring_nan(a: pd.Series, b: pd.Series, name: str) -> None:
    pd.testing.assert_series_equal(a, b, check_names=False)
    assert np.isnan(a).equals(np.isnan(b)), f"{name}: NaN desen degisti"


# ---------------------------------------------------------------------------
# cross_section.py
# ---------------------------------------------------------------------------


def test_winsorize_uc_deger_kirpar_ve_girileni_bozmaz():
    previous_day = _dates(1)[0]
    next_day = _dates(2)[1]
    df = pd.DataFrame(
        {
            "date": [previous_day] * 3 + [next_day] * 3,
            "value": [1.0, 2.0, 100.0, 10.0, 11.0, 12.0],
        }
    )
    snapshot = df.copy()

    out = winsorize_by_date(df, ["value"], lower=0.1, upper=0.9)

    assert_frame_equal(df, snapshot, check_dtype=False)   # girdi degismedi
    day1 = df[df["date"] == previous_day]["value"]
    day2 = df[df["date"] == next_day]["value"]
    o1 = out[out["date"] == previous_day]["value"]
    o2 = out[out["date"] == next_day]["value"]

    assert o1.max() < day1.max()                          # uc deger kirpildi
    assert o1.max() == pytest.approx(day1.quantile(0.9))  # gunun kendi siniri
    assert o1.min() == pytest.approx(day1.quantile(0.1))
    # gunler bagimsiz: her gun kendi sinirlarini kullanir
    assert o2.max() == pytest.approx(day2.quantile(0.9))
    assert o2.max() != pytest.approx(o1.max())
    # ortalama degerler kirpilmaz: 2. gunun orta degeri ayni kalir
    assert out["value"].iloc[4] == pytest.approx(11.0)


def test_zscore_by_date_ort_sifir_std_bir_ve_std_sifir_gunu():
    previous_day = _dates(1)[0]
    next_day = _dates(2)[1]
    df = pd.DataFrame(
        {
            "date": [previous_day] * 4 + [next_day] * 4,
            "x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 50.0],
        }
    )
    snapshot = df.copy()

    out = zscore_by_date(df, ["x"])["x"]

    assert_frame_equal(df, snapshot, check_dtype=False)   # girdi degismedi
    for day in (previous_day, next_day):
        vals = df[df["date"] == day]["x"].to_numpy()
        # z-score artik POPULASYON std (ddof=0) ile standartlastiriyor
        expected = (vals - vals.mean()) / vals.std(ddof=0)
        got = out[df["date"] == day].to_numpy()
        assert np.allclose(np.sort(got), np.sort(expected))
        assert np.allclose(got.mean(), 0.0, atol=1e-12)
        assert np.allclose(got.std(ddof=0), 1.0, atol=1e-9)

    # std == 0 gunu: tum hisseler ayni deger -> her z = 0 (NaN DEGIL)
    flat = pd.DataFrame(
        {
            "date": _dates(1).repeat(3),
            "c": [7.0, 7.0, 7.0],
        }
    )
    z = zscore_by_date(flat, ["c"])["c"]
    assert z.eq(0.0).all()
    assert not z.isna().any()


# ---------------------------------------------------------------------------
# labels.py
# ---------------------------------------------------------------------------


def test_forward_return_manuel_horizon_ve_son_nan():
    s = pd.Series([100.0, 110.0, 121.0])
    fh1 = forward_return(s, horizon=1)
    assert fh1.iloc[0] == pytest.approx(0.10)
    assert fh1.iloc[1] == pytest.approx(0.10)
    assert np.isnan(fh1.iloc[-1])

    fh2 = forward_return(s, horizon=2)
    assert fh2.iloc[0] == pytest.approx(0.21)
    assert fh2.iloc[1:].isna().all()


def test_forward_return_horizon_gecersiz():
    with pytest.raises(ValueError):
        forward_return(pd.Series([100.0, 110.0]), horizon=0)


def test_rank_label_by_date_pct():
    df = pd.DataFrame(
        {
            "date": _dates(1).repeat(3),
            "ret": [0.05, -0.10, 0.20],
        }
    )
    ranks = rank_label_by_date(df, "ret")
    assert ranks.iloc[0] == pytest.approx(2 / 3)   # ortanca getiri
    assert ranks.iloc[1] == pytest.approx(1 / 3)   # en dusuk
    assert ranks.iloc[2] == pytest.approx(1.0)     # en yuksek

    df_nan = df.assign(ret=df["ret"].astype(float))
    df_nan.loc[1, "ret"] = np.nan
    ranks_nan = rank_label_by_date(df_nan, "ret")
    assert np.isnan(ranks_nan.iloc[1])             # NaN label -> NaN rank


# ---------------------------------------------------------------------------
# build.py — sızıntı, kolon sözleşmesi, min_tickers, benchmark, label
# ---------------------------------------------------------------------------


def test_sizinti_mutasyon_geriye_tasamaz():
    """EN ÖNEMLI TEST.

    Bir ticker'ın son 10 günlük fiyatı değiştirildiğinde:
    - değişen günlerden ÖNCEKİ tüm feature değerleri birebir aynı kalmalı
      (sızıntı yok: feature'lar yalnızca t ve öncesini görür),
    - değişen günlerde feature'lar GERÇEKTEN değişmeli (aksi halde testin
      kendisi her zaman geçiyor olurdu; yanlış-pozitif koruması).
    """
    days = 60
    dates = _dates(days)
    aapl = list(100.0 * np.cumprod(1.0 + np.linspace(0.0005, 0.002, days)))
    msft = list(200.0 * np.cumprod(1.0 + np.linspace(-0.001, 0.0006, days)))
    prices = _panel({"AAPL": aapl, "MSFT": msft})

    cfg = _tiny_cfg(min_tickers_per_date=2)
    f_before = build_features(prices, cfg)

    mutated = prices.copy()
    cut_date = dates[days - 10]
    affected = (mutated["ticker"] == "AAPL") & (mutated["date"] >= cut_date)
    mutated.loc[affected, "adj_close"] = mutated.loc[affected, "adj_close"] * 1.3
    f_after = build_features(mutated, cfg)

    feature_cols = [c for c in f_before.columns if c not in ("date", "ticker")]

    # 1) değişen günden önceki her şey birebir aynı (SIZINTI YOK)
    old_before = f_before[f_before["date"] < cut_date].reset_index(drop=True)
    old_after = f_after[f_after["date"] < cut_date].reset_index(drop=True)
    assert_frame_equal(old_before, old_after)

    # 2) değişen günlerde AAPL satırları gerçekten değişmeli
    a_before = f_before[(f_before["ticker"] == "AAPL") & (f_before["date"] >= cut_date)][
        feature_cols
    ].reset_index(drop=True)
    a_after = f_after[(f_after["ticker"] == "AAPL") & (f_after["date"] >= cut_date)][
        feature_cols
    ].reset_index(drop=True)
    assert not a_before.equals(a_after)


def test_build_features_ve_dataset_kolonlari_ayni_sirada_ve_degerde():
    prices = _panel(
        {
            "AAPL": list(100.0 * np.cumprod(1.0 + np.linspace(0.0004, 0.002, 40))),
            "MSFT": list(200.0 * np.cumprod(1.0 + np.linspace(-0.0008, 0.0009, 40))),
        }
    )
    cfg = _tiny_cfg(min_tickers_per_date=2)

    f = build_features(prices, cfg)
    d = build_dataset(prices, cfg)

    feature_cols = [c for c in f.columns if c not in ("date", "ticker")]
    assert list(f.columns) == ["date", "ticker"] + feature_cols
    # dataset: ayni feature kolonlari + ayni sirada; sonda label ikilisi
    assert list(d.columns) == ["date", "ticker"] + feature_cols + [
        "forward_return",
        "label",
    ]

    # ayni (date, ticker) satirlarinda feature degerleri birebir ayni
    key = ["date", "ticker"]
    joined = f.set_index(key).join(
        d.drop(columns=["forward_return", "label"]).set_index(key), how="inner", rsuffix="_ds"
    )
    for col in feature_cols:
        pd.testing.assert_series_equal(joined[col], joined[col + "_ds"], check_names=False)


def test_min_tickers_per_date_az_hisseli_gun_duser():
    tickers = ["A", "B", "C", "D"]
    prices = _panel({t: [100.0 + i] * 20 for i, t in enumerate(tickers)})
    dates = _dates(20)
    cfg = _tiny_cfg(min_tickers_per_date=4)

    # tum gunler 4 hisseli: hicbir gun atilmaz (warm-up sonrasi 16 gun var)
    f_full = build_features(prices, cfg)
    assert set(f_full["date"]) == set(dates[4:])

    # tek gunden 2 hisse cekilirse o gun atilir
    partial = prices.copy()
    lonely = partial["date"] == dates[10]
    remove = partial["ticker"].isin(["C", "D"]) & lonely
    partial = partial.loc[~remove]
    f_part = build_features(partial, cfg)

    assert not (f_part["date"] == dates[10]).any()
    assert set(f_part["date"]) == set(dates[4:]) - {dates[10]}


def test_benchmarks_ciktida_yok():
    prices = _panel(
        {
            "AAPL": list(100.0 * np.cumprod(1.0 + np.linspace(0.0005, 0.002, 40))),
            "MSFT": list(200.0 * np.cumprod(1.0 + np.linspace(-0.001, 0.001, 40))),
            "^GSPC": list(4000.0 * np.cumprod(1.0 + np.linspace(0.0001, 0.0008, 40))),
        }
    )
    cfg = _tiny_cfg(min_tickers_per_date=2)  # tiny cfg'de benchmark ^GSPC

    f = build_features(prices, cfg)

    assert "^GSPC" not in set(f["ticker"])
    assert set(f["ticker"]) == {"AAPL", "MSFT"}


def test_build_dataset_label_gecerli_ve_feature_nanlari_temiz():
    prices = _panel(
        {
            "AAPL": list(100.0 * np.cumprod(1.0 + np.linspace(0.0005, 0.002, 40))),
            "MSFT": list(200.0 * np.cumprod(1.0 + np.linspace(-0.001, 0.001, 40))),
        }
    )
    cfg = _tiny_cfg(min_tickers_per_date=2)

    d = build_dataset(prices, cfg)

    feature_cols = [c for c in d.columns if c not in ("date", "ticker", "forward_return", "label")]
    assert not d["label"].isna().any()          # label NaN yok
    assert d["label"].between(0.0, 1.0).all()   # rank 0-1 araligi
    assert not d["forward_return"].isna().any()
    assert not d[feature_cols].isna().any().any()  # z-score sonrasi hicbir NaN kalmamali
    # 2 hisseli gunlerde pct rank degerleri: {0.5, 1.0}
    assert set(d["label"].unique()) == {0.5, 1.0}


def test_build_dataset_stats_raporu_dogru_sayilar():
    """Warm-up ve label dususleri, stats sozlugune dogru sayilarla yansimali."""
    prices = _panel(
        {
            "AAPL": list(100.0 * np.cumprod(1.0 + np.linspace(0.0005, 0.002, 20))),
            "MSFT": list(200.0 * np.cumprod(1.0 + np.linspace(-0.001, 0.001, 20))),
        }
    )
    cfg = _tiny_cfg(min_tickers_per_date=2)

    stats: dict = {}
    d = build_dataset(prices, cfg, stats=stats)

    # 40 panel satiri: her ticker'dan ilk `skip_window`(4) satır warm-up,
    # her ticker'ın son `label_horizon`(2) satırı label'siz
    assert stats["warmup_dropped_rows"] == 2 * 4
    assert stats["label_dropped_rows"] == 2 * 2
    assert stats["days_dropped"] == 0
    assert stats["min_day_tickers"] == 2
    assert len(d) == 2 * 20 - 8 - 4
