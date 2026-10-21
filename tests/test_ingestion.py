"""Adim 1 (ingestion) testleri — aga baglanmaz, kucuk sahte DataFrame'lerle calisir."""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ingestion.config import load_config, load_universe
from ingestion.validation import find_anomalies, validate_prices
from ingestion.storage import (
    PRICE_COLUMNS,
    get_last_date,
    load_prices,
    merge_prices,
    save_prices,
)


# ---------------------------------------------------------------------------
# config / load_universe
# ---------------------------------------------------------------------------


def test_load_universe_temizler():
    raw = ["  aapl ", "msft", "AAPL", "", "   ", " Goog \n"]
    assert load_universe(raw) == ["AAPL", "MSFT", "GOOG"]


def test_load_universe_bos_sadece_bosluk():
    with pytest.raises(ValueError):
        load_universe(["", "  "])


def test_load_universe_none():
    with pytest.raises(ValueError):
        load_universe(None)


def test_load_config_goreli_yol_proje_kokune_cozulur(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "data.yaml"
    config_file.write_text(
        "universe:\n"
        "  - aapl\n"
        "  - MSFT\n"
        "start_date: '2020-01-02'\n"
        "storage:\n"
        "  prices_path: data/raw/prices.parquet\n"
        "ingestion:\n"
        "  batch_size: 5\n",
        encoding="utf-8",
    )

    cfg = load_config(config_file)

    assert cfg.universe == ["AAPL", "MSFT"]
    assert cfg.start_date == "2020-01-02"
    assert cfg.batch_size == 5
    assert cfg.prices_path == tmp_path / "data" / "raw" / "prices.parquet"


# ---------------------------------------------------------------------------
# ortak yardimcilar
# ---------------------------------------------------------------------------


def _prices(rows: list[tuple], columns: list[str] | None = None) -> pd.DataFrame:
    """Kolon sirasina (PRICE_COLUMNS) uyan kucuk frame kurar; date datetime olur.

    rows: her satir, sirayla date, ticker, open, high, low, close,
    adj_close, volume degerleri (eksik kolon belirtilirse isimler uyarlanir).
    """
    cols = columns or PRICE_COLUMNS
    df = pd.DataFrame(rows, columns=cols)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])  # uretim verisiyle ayni tip
    return df


# ---------------------------------------------------------------------------
# storage.merge_prices
# ---------------------------------------------------------------------------


def test_merge_prices_yeni_veri_kazanir():
    existing = _prices(
        [
            ("2024-01-02", "AAPL", 100.0, 101.0, 99.0, 100.5, 100.5, 1_000),
            ("2024-01-03", "AAPL", 100.6, 102.0, 100.0, 101.0, 101.0, 1_100),
        ]
    )
    new = _prices(
        [
            # ayni anahtar + duzeltilmis deger -> yeni kazaniyor
            ("2024-01-03", "AAPL", 100.6, 102.0, 100.0, 105.0, 105.0, 2_000),
            # bambaşka yeni satir
            ("2024-01-01", "AAPL", 98.0, 99.0, 97.0, 98.5, 98.5, 900),
        ]
    )

    merged = merge_prices(existing, new)

    assert len(merged) == 3  # 2 eski + 1 yeni gercekten yeni
    row = merged.loc[merged["date"] == "2024-01-03"].iloc[0]
    assert row["close"] == 105.0 and row["volume"] == 2_000  # yeni deger
    row = merged.loc[merged["date"] == "2024-01-02"].iloc[0]
    assert row["close"] == 100.5  # cakismayan eski satir duruyor


def test_merge_prices_siralar_ve_girdiyi_bozmaz():
    existing = _prices(
        [("2024-01-03", "AAPL", 100.0, 101.0, 99.0, 100.5, 100.5, 1_000)]
    )
    new = _prices(
        [
            ("2024-01-02", "MSFT", 300.0, 301.0, 299.0, 300.5, 300.5, 500),
            ("2024-01-02", "AAPL", 100.0, 101.0, 99.0, 100.5, 100.5, 1_000),
        ]
    )
    existing_snapshot = existing.copy()
    new_snapshot = new.copy()

    merged = merge_prices(existing, new)

    # (ticker, date) sirali
    order = list(zip(merged["ticker"], merged["date"]))
    assert order == sorted(order)
    # girdiler degismedi
    assert_frame_equal(existing, existing_snapshot)
    assert_frame_equal(new, new_snapshot)
    # giris nesnelerinin kendisi dondurulmedi
    assert merged is not existing and merged is not new


def test_merge_prices_idempotent():
    existing = _prices(
        [("2024-01-02", "AAPL", 100.0, 101.0, 99.0, 100.5, 100.5, 1_000)]
    )
    new = _prices(
        [
            ("2024-01-03", "AAPL", 100.6, 102.0, 100.0, 101.0, 101.0, 1_100),
            ("2024-01-04", "MSFT", 300.0, 301.0, 299.0, 300.5, 300.5, 500),
        ]
    )

    once = merge_prices(existing, new)
    twice = merge_prices(once, new)

    assert_frame_equal(once, twice)


def test_get_last_date():
    df = _prices(
        [
            ("2024-01-02", "AAPL", 100.0, 101.0, 99.0, 100.5, 100.5, 1_000),
            ("2024-01-03", "AAPL", 100.6, 102.0, 100.0, 101.0, 101.0, 1_100),
            ("2024-01-02", "MSFT", 300.0, 301.0, 299.0, 300.5, 300.5, 500),
        ]
    )
    assert get_last_date(df, "AAPL") == pd.Timestamp("2024-01-03")
    assert get_last_date(df, "MSFT") == pd.Timestamp("2024-01-02")
    assert get_last_date(df, "YOK") is None
    assert get_last_date(None, "AAPL") is None
    assert get_last_date(pd.DataFrame(), "AAPL") is None


# ---------------------------------------------------------------------------
# validation.validate_prices
# ---------------------------------------------------------------------------


def _good_frame() -> pd.DataFrame:
    return _prices(
        [
            ("2024-01-02", "AAPL", 100.0, 101.0, 99.0, 100.5, 100.5, 1_000),
            ("2024-01-03", "AAPL", 100.6, 102.0, 100.0, 101.0, 101.0, 1_100),
            ("2024-01-03", "MSFT", 300.0, 301.0, 299.0, 300.5, 300.5, 500),
        ]
    )


def test_validate_prices_gecerli_veri_hata_vermez():
    out = validate_prices(_good_frame())
    assert len(out) == 3
    # tarih coerce oldu
    assert out["date"].dtype == "datetime64[ns]"


def test_validate_prices_negatif_fiyat():
    df = _good_frame()
    df.loc[0, "open"] = -1.0
    with pytest.raises(ValueError):
        validate_prices(df)


def test_validate_prices_high_lowden_kucuk():
    df = _good_frame()
    df.loc[0, "high"] = 90.0  # low 99, high 90
    with pytest.raises(ValueError):
        validate_prices(df)


def test_validate_prices_tekrarli_satir():
    df = _good_frame()
    df = pd.concat([df, df.head(1)], ignore_index=True)  # ayni (date, ticker)
    with pytest.raises(ValueError):
        validate_prices(df)


def test_validate_prices_sifir_fiyat():
    df = _good_frame()
    df.loc[1, "close"] = 0.0  # > 0 zorunlu
    with pytest.raises(ValueError):
        validate_prices(df)


# ---------------------------------------------------------------------------
# validation.find_anomalies
# ---------------------------------------------------------------------------


def _slim(rows: list[tuple]) -> pd.DataFrame:
    return _prices(rows, columns=["date", "ticker", "close", "volume"])


def test_find_anomalies_sicrama_ve_sifir_hacim():
    df = _slim(
        [
            ("2024-01-02", "AAPL", 100.0, 1_000),
            ("2024-01-03", "AAPL", 101.0, 1_100),   # normal (%1)
            ("2024-01-04", "AAPL", 102.0, 1_050),   # normal
            ("2024-01-05", "AAPL", 103.0, 1_000),   # normal
            ("2024-01-08", "AAPL", 165.0, 1_200),   # %60 sicrama!
            ("2024-01-02", "MSFT", 300.0, 0),       # sifir hacim
        ]
    )

    out = find_anomalies(df)

    assert list(out.columns) == ["ticker", "date", "reason"]
    assert len(out) == 2
    aapl_row = out[(out["ticker"] == "AAPL")].iloc[0]
    assert aapl_row["date"] == pd.Timestamp("2024-01-08")
    assert "large_abs_return" in aapl_row["reason"]  # %60.0
    msft_row = out[(out["ticker"] == "MSFT")].iloc[0]
    assert msft_row["reason"] == "zero_volume"


def test_find_anomalies_uzun_bosluk():
    # 2024-01-02 (sali)'dan 8 is gunu sonra: 2024-01-12 (cuma)
    # aradaki is gunu: 7 -> supheli
    df = _slim(
        [
            ("2024-01-02", "AAPL", 100.0, 1_000),
            ("2024-01-12", "AAPL", 101.0, 1_000),
            ("2024-01-15", "MSFT", 300.0, 500),
            ("2024-01-16", "MSFT", 301.0, 500),  # 1 is gunu aradi -> normal
        ]
    )

    out = find_anomalies(df)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["date"] == pd.Timestamp("2024-01-12")
    assert row["reason"] == "gap_7_business_days"


def test_find_anomalies_normal_gunlerde_bos_rapor():
    df = _slim(
        [
            ("2024-01-02", "AAPL", 100.0, 1_000),
            ("2024-01-03", "AAPL", 101.0, 1_100),
            ("2024-01-04", "AAPL", 102.0, 1_050),
            ("2024-01-05", "AAPL", 103.0, 1_000),
            ("2024-01-08", "AAPL", 102.8, 1_100),
        ]
    )

    out = find_anomalies(df)

    assert len(out) == 0
    assert list(out.columns) == ["ticker", "date", "reason"]


def test_find_anomalies_bos_girdi():
    out = find_anomalies(pd.DataFrame())
    assert list(out.columns) == ["ticker", "date", "reason"]
    assert len(out) == 0


# ---------------------------------------------------------------------------
# storage.save_prices + load_prices (roundtrip)
# ---------------------------------------------------------------------------


def test_save_prices_load_prices_roundtrip(tmp_path):
    target = tmp_path / "data" / "raw" / "prices.parquet"

    df = _prices(
        [
            ("2024-01-02", "AAPL", 100.0, 101.0, 99.0, 100.5, 100.5, 1_000),
            ("2024-01-03", "AAPL", 100.6, 102.0, 100.0, 101.0, 101.0, 1_100),
            ("2024-01-02", "MSFT", 300.0, 301.0, 299.0, 300.5, 300.5, 500),
        ]
    )
    df["volume"] = df["volume"].astype("int64")  # parquet'te tipler sadi kalmali

    returned_path = save_prices(df, target)
    assert returned_path == target and target.is_file()

    loaded = load_prices(target)
    assert_frame_equal(loaded, df)  # ayni satirlar + ayni tipler

    # var olmayan dosya: hata vermaz, semaya uygun bos frame gelir
    empty = load_prices(tmp_path / "yok.boyle.dosya.parquet")
    assert len(empty) == 0
