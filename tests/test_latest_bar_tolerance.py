"""Son-bar toleransi testleri (ag yok, sentetik, tmp_path gerektirmeyen).

- drop_incomplete_latest_rows: son tarihte NaN hisse atilir; esik asilirsa
  tarih komple atilir; benchmark satiri ise sadece satir atilir; gecmis
  tarihler dokunulmaz.
- validate_prices: temizlenmis paneli GECIRIR; gecmis NaN'i hala SERT hata
  yapar (davranis degismedi).
- main(): cikis kodlari (0 ok / 1 tekrar edilebilir / 2 dogrulama-config).
- config: max_incomplete_latest_share okuma + varsayilan.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from ingestion.config import (
    DEFAULT_MAX_INCOMPLETE_LATEST_SHARE,
    load_config,
)
from ingestion.run import drop_incomplete_latest_rows
from ingestion.validation import validate_prices

BENCHMARKS = ["^GSPC", "SPY"]
UNIVERSE = [f"A{index:02d}" for index in range(10)]


def _satir(tarih: str, ticker: str, *, close: float = 10.0, adj_close: float = 10.0) -> dict:
    """Tek fiyat satiri; NaN istenirse acikca float('nan') gecilir."""
    kayit = {
        "date": pd.Timestamp(tarih),
        "ticker": ticker,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": close,
        "adj_close": adj_close,
        "volume": 1000.0,
    }
    return kayit


def _panel(satirlar: list[dict]) -> pd.DataFrame:
    """Satir listesinden fiyat paneli (kolon sirasi ingestion semasinda)."""
    return pd.DataFrame(
        satirlar,
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
    )


def _tam_panel(tarihler: list[str], tickers: list[str]) -> pd.DataFrame:
    """Verilen her (tarih, ticker) cifti icin temiz satir."""
    satirlar = [_satir(tarih, ticker) for tarih in tarihler for ticker in tickers]
    return _panel(satirlar)


def test_son_tarih_tek_hisse_nan_atilir_ve_ingestion_temiz(caplog):
    """Son tarihteki tek NaN hisse satiri atilir; kalan panel validasyonu gecer."""
    tarihler = ["2026-11-03", "2026-11-04", "2026-11-05"]
    panel = _tam_panel(tarihler, UNIVERSE)
    son = pd.Timestamp("2026-11-05")
    eksik_index = panel.index[(panel["ticker"] == "A02") & (panel["date"] == son)][0]
    panel.loc[eksik_index, "close"] = np.nan

    with caplog.at_level(logging.WARNING, logger="ingestion.run"):
        temiz = drop_incomplete_latest_rows(panel, UNIVERSE, BENCHMARKS)

    assert len(temiz) == len(panel) - 1
    kalan = temiz[(temiz["ticker"] == "A02") & (temiz["date"] == son)]
    assert len(kalan) == 0
    assert temiz["date"].max() == son  # tarih korunur, sadece satir dustu
    assert validate_prices(temiz) is not None  # akis temiz ilerler (ingestion gibi)
    assert "A02" in caplog.text and "NaN" in caplog.text


def test_son_tarih_esik_asi_nan_tarih_komple_atilir():
    """Eksik hisse evren esiginin ustundeyse son tarih tamamen dusurulur."""
    tarihler = ["2026-11-03", "2026-11-04", "2026-11-05"]
    panel = _tam_panel(tarihler, UNIVERSE + BENCHMARKS)
    son = pd.Timestamp("2026-11-05")
    for bitmis in ("A00", "A01", "A02"):  # 3/10 = 0.30 > 0.10 esik
        eksik_index = panel.index[(panel["ticker"] == bitmis) & (panel["date"] == son)][0]
        panel.loc[eksik_index, "adj_close"] = np.nan

    temiz = drop_incomplete_latest_rows(panel, UNIVERSE, BENCHMARKS, max_share=0.10)

    assert son not in set(temiz["date"])  # tarih komple gitti (benchmark dahil)
    assert temiz["date"].max() == pd.Timestamp("2026-11-04")
    assert len(temiz) == 2 * (len(UNIVERSE) + len(BENCHMARKS))


def test_gecmis_tarih_nan_dokunulmaz_ve_sert_hata():
    """Gecmis tarihteki NaN degistirilmez; validate_prices yine de hata verir."""
    tarihler = ["2026-11-03", "2026-11-04", "2026-11-05"]
    panel = _tam_panel(tarihler, UNIVERSE[:4])
    eski = pd.Timestamp("2026-11-03")
    panel.loc[panel.index[(panel["ticker"] == "A01") & (panel["date"] == eski)][0], "close"] = np.nan

    temiz = drop_incomplete_latest_rows(panel, UNIVERSE[:4], BENCHMARKS)

    kalan = temiz[(temiz["ticker"] == "A01") & (temiz["date"] == eski)]
    assert len(kalan) == 1  # gecmis satira dokunulmadi
    with pytest.raises(ValueError):
        validate_prices(temiz)  # SERT hata korunur


def test_benchmark_nan_sadece_satir_atilir_tarih_korunur():
    """Benchmark NaN'i yalnizca o satiri dusturur; esige katilmaz, tarih korunur."""
    tarihler = ["2026-11-03", "2026-11-04", "2026-11-05"]
    panel = _tam_panel(tarihler, UNIVERSE + BENCHMARKS)
    son = pd.Timestamp("2026-11-05")
    spy_index = panel.index[(panel["ticker"] == "SPY") & (panel["date"] == son)][0]
    gspc_index = panel.index[(panel["ticker"] == "^GSPC") & (panel["date"] == son)][0]
    panel.loc[spy_index, "close"] = np.nan
    panel.loc[gspc_index, "adj_close"] = np.nan

    temiz = drop_incomplete_latest_rows(panel, UNIVERSE, BENCHMARKS, max_share=0.10)

    assert len(temiz) == len(panel) - 2  # sadece iki benchmark satiri dustu
    kalan_spy = temiz[(temiz["ticker"] == "SPY") & (temiz["date"] == son)]
    kalan_hisse = temiz[(temiz["ticker"] == "A00") & (temiz["date"] == son)]
    assert len(kalan_spy) == 0 and len(kalan_hisse) == 1
    assert temiz["date"].max() == son  # tarih komple atilmadi


def test_esik_siniri_esit_kenar_parcali_temizlik_yap():
    """Oran esige ESIT oldugunda tarih dokunulmaz; yalnizca satirlar atilir."""
    tarihler = ["2026-11-03", "2026-11-05"]
    panel = _tam_panel(tarihler, UNIVERSE)
    son = pd.Timestamp("2026-11-05")
    eksik_index = panel.index[(panel["ticker"] == "A03") & (panel["date"] == son)][0]
    panel.loc[eksik_index, "close"] = np.nan  # 1/10 = 0.10 == esik

    temiz = drop_incomplete_latest_rows(panel, UNIVERSE, BENCHMARKS, max_share=0.10)

    assert temiz["date"].max() == son  # tarih korundu (esik "ASARSA" anlami)
    kalan = temiz[(temiz["ticker"] == "A03") & (temiz["date"] == son)]
    assert len(kalan) == 0


def test_main_cikis_kodlari(monkeypatch):
    """main() cikis kodlari: 0 ok / 2 dogrulama / 1 diger hata."""
    import ingestion.run as mod

    def _yap(hata: Exception):
        def _call():
            raise hata

        return _call

    monkeypatch.setattr(mod, "run", lambda: None)
    assert mod.main() == 0
    monkeypatch.setattr(mod, "run", _yap(ValueError("Fiyat verisi semayi gecemedi: 2 hata")))
    assert mod.main() == 2
    monkeypatch.setattr(mod, "run", _yap(RuntimeError("yfinance baglanti hatasi")))
    assert mod.main() == 1


def test_config_share_oku_veya_varsayilan(tmp_path):
    """max_incomplete_latest_share yaml'dan okunur; yoksa varsayilan 0.10."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config_dir.joinpath("data.yaml").write_text(
        "universe:\n  - aapl\nstorage:\n  prices_path: data/raw/prices.parquet\n",
        encoding="utf-8",
    )
    cfg = load_config(config_dir / "data.yaml")
    assert cfg.max_incomplete_latest_share == pytest.approx(DEFAULT_MAX_INCOMPLETE_LATEST_SHARE)

    config_dir.joinpath("data.yaml").write_text(
        "universe:\n  - aapl\nstorage:\n  prices_path: data/raw/prices.parquet\n"
        "ingestion:\n  batch_size: 5\n  max_incomplete_latest_share: 0.25\n",
        encoding="utf-8",
    )
    cfg = load_config(config_dir / "data.yaml")
    assert cfg.max_incomplete_latest_share == pytest.approx(0.25)
