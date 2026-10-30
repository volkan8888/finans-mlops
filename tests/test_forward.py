"""Ileri test (forward) modulu testleri + freeze imza testi.

Sentetik panel: 4 ticker + SPY; deterministik gunluk getiri oranlari.
Etiket semantigi (labels.forward_return): D->D+horizon islem-gunu.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib

import pandas as pd
import pytest

from monitoring import forward

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _synthetic_prices(n_days: int = 14) -> pd.DataFrame:
    """4 ticker + SPY: her biri sabit gunluk oran (AAA en hizli, DDD en yavas)."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    rates = {"AAA": 0.004, "BBB": 0.003, "CCC": 0.002, "DDD": 0.001, "SPY": 0.0004}
    rows: list[dict] = []
    for ticker, rate in rates.items():
        level = 100.0
        for index, date in enumerate(dates):
            if index > 0:
                level *= 1.0 + rate
            rows.append({"date": date, "ticker": ticker, "adj_close": round(level, 6)})
    return pd.DataFrame(rows)


def _synthetic_predictions(day_indexes: list[int], n_days: int = 14, invert_day: int | None = None) -> pd.DataFrame:
    """Verilen gun indeksleri icin AAA>BBB>CCC>DDD skorlari (invert_day'de degerler ters)."""
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    base = {"AAA": 0.40, "BBB": 0.30, "CCC": 0.20, "DDD": 0.10}
    order = ["AAA", "BBB", "CCC", "DDD"]
    rows: list[dict] = []
    for day_index in day_indexes:
        if day_index == invert_day:
            mapping = {ticker: base[order[len(order) - 1 - order.index(ticker)]] for ticker in order}
        else:
            mapping = base
        for ticker in order:
            rows.append({
                "date": dates[day_index],
                "ticker": ticker,
                "score": mapping[ticker],
            })
    return pd.DataFrame(rows)


def test_load_prediction_frames_toplar_ve_bozugu_atlar(tmp_path: pathlib.Path):
    """Iki gun dosyasi toplanir; bozuk parquet atlanir ve SAYILIR."""
    directory = tmp_path / "data" / "predictions"
    directory.mkdir(parents=True)
    frame = _synthetic_predictions([0])
    frame.to_parquet(directory / "predictions_2025-01-01.parquet", index=False)
    frame2 = _synthetic_predictions([1])
    frame2.to_parquet(directory / "predictions_2025-01-02.parquet", index=False)
    (directory / "predictions_bozuk.parquet").write_bytes(b"bu-parquet-degil")

    merged, skipped = forward.load_prediction_frames(directory)
    assert skipped == ["predictions_bozuk.parquet"]
    assert sorted(merged["date"].dt.date.astype(str).unique()) == ["2025-01-01", "2025-01-02"]
    assert len(merged) == 8


def test_realized_forward_returns_horizon_doldugunda(tmp_path: pathlib.Path):
    """Horizon doldugunda getiri P_{D+h}/P_D - 1; unrealized 0; IC yuksek."""
    prices = _synthetic_prices(n_days=14)
    predictions = _synthetic_predictions([0, 1, 2, 3, 4, 5, 6], invert_day=3)
    realized, unrealized, spy = forward.realized_forward_returns(predictions, prices, horizon=3)
    assert unrealized == 0
    assert len(realized) == 7 * 4
    # A ticker: depolanan fiyatlardan ayni formulle beklenti (yuvarlamadan bagimsiz)
    aaa = realized[realized["ticker"] == "AAA"].iloc[0]
    aaa_prices = prices[prices["ticker"] == "AAA"]["adj_close"].tolist()
    aaa_dates = sorted(prices["date"].unique())
    expected = aaa_prices[3] / aaa_prices[0] - 1.0
    assert math.isclose(aaa["realized"], expected, rel_tol=1e-9)
    assert str(pd.Timestamp(aaa["realized_date"]).date()) == str(pd.Timestamp(aaa_dates[3]).date())
    # SPY da ayni pencere tanimiyla
    assert len(spy) == 11  # D+3 sonu son tarihe kadar dolduran 11 gun
    spy_prices = prices[prices["ticker"] == "SPY"]["adj_close"].tolist()
    assert math.isclose(spy.iloc[0], spy_prices[3] / spy_prices[0] - 1.0, rel_tol=1e-9)

    summary = forward.ic_summary(forward.daily_ic(realized, "score", "realized", min_n=2), nw_lag=4)
    # 6/7 gun kusursuz siralama + 1 gun ters: anlamlisi pozitif
    assert summary["mean_ic"] > 0.5
    assert math.isfinite(float(summary["t_stat"])) and math.isfinite(float(summary["t_nw"]))


def test_realized_unrealized_atlanir_sayilir():
    """Pencere dolmayan son gunler: kayit atlanir ve unrealized sayilir."""
    prices = _synthetic_prices(n_days=14)
    predictions = _synthetic_predictions([11, 12, 13])
    realized, unrealized, _ = forward.realized_forward_returns(predictions, prices, horizon=5)
    assert realized.empty
    assert unrealized == 3 * 4


def test_quantile_table_spay_farki():
    """Kuantil tablosu: ust kuantil SPY'dan yuksek; satirlar q0..q{n-1}."""
    prices = _synthetic_prices(n_days=14)
    predictions = _synthetic_predictions([0, 1, 2, 3, 4, 5, 6])
    realized, _, spy = forward.realized_forward_returns(predictions, prices, horizon=3)
    table = forward.quantile_table(realized, spy, n_quantiles=4)
    assert list(table["quantile"]) == ["q0", "q1", "q2", "q3"]
    top = table[table["quantile"] == "q3"].iloc[0]
    bottom = table[table["quantile"] == "q0"].iloc[0]
    assert top["mean_return"] > bottom["mean_return"]
    assert top["vs_benchmark"] > 0
    spy_mean = float(spy.mean())
    assert math.isclose(top["mean_return"] - spy_mean, top["vs_benchmark"], rel_tol=1e-9)


def test_forward_test_report_end_to_end(tmp_path: pathlib.Path):
    """Bos kosullar 'bos'; gercek veriyle 'ok' + JSON yazilir."""
    # bos durum: tahmin klasoru yok
    report = forward.forward_test_report(
        predictions_dir=tmp_path / "yok",
        prices_path=tmp_path / "prices.parquet",
        output_path=None,
    )
    assert report["status"] == "bos"

    # fiyat var, tahmin yok
    prices_path = tmp_path / "data" / "raw" / "prices.parquet"
    prices_path.parent.mkdir(parents=True)
    _synthetic_prices(n_days=14).to_parquet(prices_path, index=False)
    report = forward.forward_test_report(
        predictions_dir=tmp_path / "data" / "predictions",
        prices_path=prices_path,
        horizon=3,
        output_path=None,
    )
    assert report["status"] == "bos"

    # tahmin de var: 7 gun x 4 ticker; horizon 3; unrealized yok
    # (3. gunde sirayi ters cevir: IC serisi sabit kalmasin; NW t degisken olsun)
    preds_dir = tmp_path / "data" / "predictions"
    preds_dir.mkdir(parents=True)
    _synthetic_predictions([0, 1, 2, 3, 4, 5, 6], invert_day=3).to_parquet(
        preds_dir / "predictions_2025-01-01.parquet", index=False
    )
    output = tmp_path / "reports" / "forward_test" / "forward_report.json"
    report = forward.forward_test_report(
        predictions_dir=preds_dir,
        prices_path=prices_path,
        horizon=3,
        n_quantiles=4,
        output_path=output,
    )
    assert report["status"] == "ok"
    assert report["n_unrealized"] == 0
    assert report["coverage"]["n_prediction_days"] == 7
    overall = report["overall"]
    assert overall["mean_ic"] > 0.5 and overall["t_nw"] is not None
    assert [row["quantile"] for row in report["quantiles"]] == ["q0", "q1", "q2", "q3"]
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_forward_test_report_pencere_dolmadi(tmp_path: pathlib.Path):
    """Tahminler yeni: pencere dolmadi -> 'tum_beklemede' + unrealized sayi."""
    prices_path = tmp_path / "data" / "raw" / "prices.parquet"
    prices_path.parent.mkdir(parents=True)
    _synthetic_prices(n_days=14).to_parquet(prices_path, index=False)
    preds_dir = tmp_path / "data" / "predictions"
    preds_dir.mkdir(parents=True)
    _synthetic_predictions([11, 12, 13]).to_parquet(
        preds_dir / "predictions_2025-01-01.parquet", index=False
    )
    report = forward.forward_test_report(
        predictions_dir=preds_dir,
        prices_path=prices_path,
        horizon=5,
        output_path=None,
    )
    assert report["status"] == "tum_beklemede"
    assert report["n_unrealized"] == 12
    assert report["alerts"]


def test_freeze_local_model_imzalar(tmp_path: pathlib.Path):
    """freeze: yerel fallback modeli bulur; sha256 + config hash dogru."""
    spec = importlib.util.spec_from_file_location("freeze_model", PROJECT_ROOT / "scripts" / "freeze_model.py")
    freeze_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze_module)

    meta = {
        "model_name": "gru",
        "feature_cols": ["f0", "f1"],
        "seed": 42,
        "label_horizon": 5,
        "seq_len": 20,
        "model_params": {"hidden": 4},
        "train_dates": ["2025-01-01", "2025-01-02"],
        "holdout_range": ["2025-01-03", "2025-01-04"],
    }
    model_dir = tmp_path / "models" / "gru"
    model_dir.mkdir(parents=True)
    (model_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (model_dir / "model.pt").write_bytes(b"fake-weights")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "training.yaml").write_text("seed: 42\n", encoding="utf-8")

    frozen = freeze_module.collect_freeze(tmp_path)
    assert frozen["model"]["source_kind"] == "local"
    assert frozen["model"]["name"] == "gru"
    model_sha = frozen["model"]["sha256"]["model_files"]["model.pt"]
    expected = hashlib.sha256((model_dir / "model.pt").read_bytes()).hexdigest()
    assert model_sha == expected
    training_bytes = (config_dir / "training.yaml").read_bytes()
    assert frozen["config_sha256"]["config/training.yaml"] == hashlib.sha256(training_bytes).hexdigest()
    assert frozen["config_sha256"]["config/features.yaml"] == ""
    assert frozen["meta_summary"]["seed"] == 42
    assert "degisirse" in frozen["note"]


def test_freeze_model_yoksa_hata():
    """Bos klasorde collect_freeze FileNotFoundError verir."""
    spec = importlib.util.spec_from_file_location("freeze_model", PROJECT_ROOT / "scripts" / "freeze_model.py")
    freeze_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(freeze_module)

    with pytest.raises(FileNotFoundError):
        freeze_module.collect_freeze(pathlib.Path(__file__).resolve().parent)  # tests/ bos modeller


def test_dashboard_prepare_forward_helpers():
    """prepare.load_freeze/load_forward yoksa bostur; quantile frame None->NaN."""
    from dashboard import prepare

    forward_payload = {
        "status": "ok",
        "quantiles": [
            {"quantile": "q0", "n_days": 3, "mean_return": 0.001, "vs_benchmark": None},
            {"quantile": "q1", "n_days": 3, "mean_return": 0.002, "vs_benchmark": 0.0005},
        ],
    }
    table = prepare.forward_quantile_frame(forward_payload)
    assert list(table["quantile"]) == ["q0", "q1"]
    vs = table["vs_benchmark"].iloc[0]
    assert table["vs_benchmark"].iloc[1] == 0.0005
    assert math.isnan(vs)  # None -> NaN

    assert prepare.forward_quantile_frame({})["quantile"].tolist() == []  # bos
    assert prepare.load_forward() == {} or prepare.load_forward().get("status")
    assert prepare.load_freeze() == {} or prepare.load_freeze().get("model")


# ---------------------------------------------------------------------------
# Yeni olcumler: ust_kantil_eksi_evren / ust_eksi_alt_kantil / unrealized_payi
# ---------------------------------------------------------------------------

def _manual_realized(with_benchmarks: bool = False) -> pd.DataFrame:
    """Elle hesapli kucuk ornek: 3 gun x 10 ticker (+Istege bagli benchmark).

    Skor sira: T1 en dusuk (1.0) ... T10 en yuksek (10.0).
    Getiriler: gun 0 ve 2 -> r(i) = (i-1)/1000 (T10 en iyi); gun 1 -> TERS
    (r(i) = (10-i)/1000). n_quantiles=5 ile her kanta 2 ticker dusar:
    q4 = {T9, T10}, q0 = {T1, T2}.
    Elle degerler (n=5):
      gun 0/2: ust 0.0085, alt 0.0005, evren 0.0045 -> ust-evren +0.0040, ust-alt +0.0080
      gun 1  : ust 0.0005, alt 0.0085, evren 0.0045 -> ust-evren -0.0040, ust-alt -0.0080
    """
    dates = pd.bdate_range("2025-01-01", periods=3)
    tickers = [f"T{i}" for i in range(1, 11)]
    rows: list[dict] = []
    for day_index, date in enumerate(dates):
        for i, ticker in enumerate(tickers, start=1):
            ticker_return = (10 - i) / 1000.0 if day_index == 1 else (i - 1) / 1000.0
            rows.append({
                "date": date,
                "ticker": ticker,
                "score": float(i),
                "realized": ticker_return,
            })
    if with_benchmarks:
        for day_index, date in enumerate(dates):
            rows.append({"date": date, "ticker": "SPY", "score": 99.0, "realized": 0.05})
            rows.append({"date": date, "ticker": "^GSPC", "score": 98.0, "realized": 0.04})
    return pd.DataFrame(rows)


def test_ust_kantil_spread_elle_ornek_birebir():
    """Elle hesap: gunluk ust-evren / ust-alt degerleri birebir; seri + ort + NW."""
    realized = _manual_realized()
    spreads = forward.daily_spread_series(realized, n_quantiles=5)
    assert list(spreads["date"].dt.date.astype(str)) == ["2025-01-01", "2025-01-02", "2025-01-03"]

    # gun 0 ham kolonlar
    first = spreads.iloc[0]
    assert math.isclose(first["ust"], 0.0085, abs_tol=1e-12)
    assert math.isclose(first["alt"], 0.0005, abs_tol=1e-12)
    assert math.isclose(first["evren"], 0.0045, abs_tol=1e-12)
    # gunluk farklar (gun 1 ters sinyal)
    assert [round(value, 6) for value in spreads["ust_eksi_evren"]] == [0.004, -0.004, 0.004]
    assert [round(value, 6) for value in spreads["ust_eksi_alt"]] == [0.008, -0.008, 0.008]

    summary = forward._spread_summary(spreads, "ust_eksi_evren", nw_lag=4)
    assert summary["n_gun"] == 3
    assert summary["ortalama"] == 0.001333  # round(0.004/3, 6)
    assert summary["seri"][0] == {"date": "2025-01-01", "deger": 0.004}
    assert summary["t_nw"] is None or math.isfinite(float(summary["t_nw"]))
    summary_alt = forward._spread_summary(spreads, "ust_eksi_alt", nw_lag=4)
    assert summary_alt["ortalama"] == 0.002667  # round(0.008/3, 6)


def test_mukemmel_sinyal_ust_alt_pozitif():
    """skor = gelecek getiri: ust-alt farki pozitif ve buyuk."""
    rng = __import__("numpy").random.default_rng(42)
    rows: list[dict] = []
    dates = pd.bdate_range("2025-02-03", periods=6)
    for date in dates:
        returns = rng.uniform(-0.02, 0.02, 40)
        for ticker_index in range(40):
            rows.append({
                "date": date,
                "ticker": f"C{ticker_index:03d}",
                "score": float(returns[ticker_index]),
                "realized": float(returns[ticker_index]),
            })
    realized = pd.DataFrame(rows)
    spreads = forward.daily_spread_series(realized, n_quantiles=5)
    mean_spread = float(spreads["ust_eksi_alt"].mean())
    assert len(spreads) == 6
    assert mean_spread > 0.01  # mukemmel sinyalde genis pozitif gap


def test_ters_sinyal_ust_alt_negatif():
    """skor = -getiri: ust-alt farki negatif (buyuklugunde buyuk)."""
    rng = __import__("numpy").random.default_rng(7)
    rows: list[dict] = []
    dates = pd.bdate_range("2025-02-03", periods=6)
    for date in dates:
        returns = rng.uniform(-0.02, 0.02, 40)
        for ticker_index in range(40):
            rows.append({
                "date": date,
                "ticker": f"C{ticker_index:03d}",
                "score": float(-returns[ticker_index]),
                "realized": float(returns[ticker_index]),
            })
    realized = pd.DataFrame(rows)
    spreads = forward.daily_spread_series(realized, n_quantiles=5)
    assert float(spreads["ust_eksi_alt"].mean()) < -0.01


def test_rastgele_sinyal_ust_alt_sifir_yakini():
    """Bagimsiz skor/getiri: fark 0'a yakin (genis tolerans, sabit seed)."""
    rng = __import__("numpy").random.default_rng(1234)
    rows: list[dict] = []
    dates = pd.bdate_range("2025-02-03", periods=12)
    for date in dates:
        returns = rng.uniform(-0.02, 0.02, 40)
        scores = rng.uniform(-0.02, 0.02, 40)  # bagimsiz cizim
        for ticker_index in range(40):
            rows.append({
                "date": date,
                "ticker": f"C{ticker_index:03d}",
                "score": float(scores[ticker_index]),
                "realized": float(returns[ticker_index]),
            })
    realized = pd.DataFrame(rows)
    spreads = forward.daily_spread_series(realized, n_quantiles=5)
    assert abs(float(spreads["ust_eksi_alt"].mean())) < 0.01
    assert float(spreads["ust_eksi_alt"].abs().max()) < 0.06


def test_benchmark_tickerlari_evrende_yok():
    """SPY / ^GSPC evren ortalamasina girmiyor; haric tutulmazsa deger degisir."""
    realized = _manual_realized(with_benchmarks=True)
    spreads_default = forward.daily_spread_series(realized, n_quantiles=5)  # default: SPY/^GSPC haric
    spreads_none = forward.daily_spread_series(realized, n_quantiles=5, excluded=set())
    # default: SPY + ^GSPC sira VE evren ortalamasi disinda -> elle deger ayni
    assert math.isclose(spreads_default.iloc[0]["evren"], 0.0045, abs_tol=1e-9)
    assert math.isclose(spreads_none.iloc[0]["evren"], 0.01125, abs_tol=1e-9)  # (0.045+0.05+0.04)/12
    assert spreads_default.iloc[0]["ust_eksi_evren"] != spreads_none.iloc[0]["ust_eksi_evren"]


def test_report_hepsi_unrealized_oran_1(tmp_path: pathlib.Path):
    """Hic gerceklesen yok: cokme yok; oran 1.0; spreads None; status korunur."""
    prices_path = tmp_path / "prices.parquet"
    _synthetic_prices(n_days=14).to_parquet(prices_path, index=False)
    preds_dir = tmp_path / "pred"
    preds_dir.mkdir()
    _synthetic_predictions([11, 12, 13]).to_parquet(
        preds_dir / "predictions_2025-01-01.parquet", index=False
    )
    report = forward.forward_test_report(
        predictions_dir=preds_dir,
        prices_path=prices_path,
        horizon=5,
        output_path=None,
    )
    assert report["status"] == "tum_beklemede"
    assert report["unrealized_payi"] == {"toplam": 12, "gerceklesen": 0, "bekleyen": 12, "oran": 1.0}
    assert report["ust_kantil_eksi_evren"] is None
    assert report["ust_eksi_alt_kantil"] is None
    # mevcut alanlar aynen duruyor
    assert report["n_unrealized"] == 12 and report["n_realized"] == 0
    assert report["alerts"]


def test_report_karisik_unrealized_orani(tmp_path: pathlib.Path):
    """Kismi gerceklesen: oran beklenen; spreads yalniz gerceklesen gunlerden."""
    prices_path = tmp_path / "prices.parquet"
    _synthetic_prices(n_days=14).to_parquet(prices_path, index=False)
    preds_dir = tmp_path / "pred"
    preds_dir.mkdir()
    _synthetic_predictions([0, 1]).to_parquet(
        preds_dir / "predictions_2025-01-01.parquet", index=False
    )
    _synthetic_predictions([13]).to_parquet(
        preds_dir / "predictions_2025-01-20.parquet", index=False
    )
    report = forward.forward_test_report(
        predictions_dir=preds_dir,
        prices_path=prices_path,
        horizon=3,
        n_quantiles=4,
        output_path=None,
    )
    assert report["status"] == "ok"
    # toplam 12 satir (3 gun x 4 ticker); gun 0-1 gerceklesti (8), gun 13 bekler (4)
    assert report["unrealized_payi"] == {"toplam": 12, "gerceklesen": 8, "bekleyen": 4, "oran": 0.3333}
    spread = report["ust_kantil_eksi_evren"]
    assert spread["n_gun"] == 2
    assert spread["ortalama"] > 0  # AAA>BBB>CCC>DDD deterministik siralama
    assert report["ust_eksi_alt_kantil"]["ortalama"] > 0


def test_report_json_yeni_alanlar_retro_uyumlu(tmp_path: pathlib.Path):
    """JSON'da yeni alanlar var; mevcut alanlar yine ayni yerlerde."""
    prices_path = tmp_path / "prices.parquet"
    _synthetic_prices(n_days=14).to_parquet(prices_path, index=False)
    preds_dir = tmp_path / "pred"
    preds_dir.mkdir()
    _synthetic_predictions([0, 1]).to_parquet(
        preds_dir / "predictions_2025-01-01.parquet", index=False
    )
    output = tmp_path / "forward_report.json"
    report = forward.forward_test_report(
        predictions_dir=preds_dir,
        prices_path=prices_path,
        horizon=3,
        n_quantiles=4,
        output_path=output,
    )
    required = {
        "status", "horizon", "nw_lag", "benchmark", "coverage",
        "n_realized", "n_unrealized", "quantiles", "overall", "alerts",
        "ust_kantil_eksi_evren", "ust_eksi_alt_kantil", "unrealized_payi",
    }
    assert required <= set(report)
    assert report["unrealized_payi"]["oran"] == 0.0
    assert report["unrealized_payi"]["gerceklesen"] == 8
    from dashboard import prepare

    frame = prepare.forward_quantile_frame(report)
    assert not frame.empty  # eski dashboard yardimcisi calisir
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["ust_kantil_eksi_evren"]["n_gun"] == 2
