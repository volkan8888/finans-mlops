"""Adim 12 yardimci scriptleri testleri (ag yok, sentetik, tmp_path).

- scripts/verify_model_hash.py: dogru hash gecer, yanlis hash basarisiz,
  sha8 yedegi zayif-uyariyla gecer, dosya/alan yoksa basarisiz.
- scripts/check_new_bar.py: yeni gun -> new; ayni/bayat -> none; fiyat
  deposu yoksa FileNotFoundError.
Proje kokune dosya yazmaz (her sey tmp_path'te).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import pathlib
import sys

import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_script(filename: str):
    """scripts/<dosya> modulunu dosyadan yukler (paket degildir).

    exec_module ONCESI sys.modules'a eklenir: dataclass/typing modul
    aramalarini yapar (sys.modules'ta yoksa dataclass NoneType hatasidir).
    """
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_mod = _load_script("verify_model_hash.py")
bar_mod = _load_script("check_new_bar.py")


def _write_model(tmp_path: pathlib.Path, content: bytes = b"sentetik-model-icerigi") -> pathlib.Path:
    """Kucuk sentetik model dosyasi (gercek torch dosyasi degil)."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(content)
    return model_path


def _freeze_payload(model_path: pathlib.Path, *, full: str | None = None, sha8: str | None = None) -> dict:
    """Freeze semasiyla senkron payload: model.sha256.model_files["model.pt"]."""
    model_section: dict = {
        "name": "gru",
        "sha256": {"model_files": {"model.pt": full if full is not None else verify_mod.sha256_file(model_path)}},
    }
    if sha8 is not None:
        model_section["sha8"] = sha8
    if full == "YOK":
        model_section.pop("sha256")  # tam hash yok simulasyonu (sadece sha8)
    return {"model": model_section}


def test_sha256_file_bilinen_ozet(tmp_path: pathlib.Path):
    """Ozet fonksiyonu standart hashlib ile ayni degeri verir."""
    path = tmp_path / "veri.bin"
    path.write_bytes(b"finans-mlops-hash-check")
    expected = hashlib.sha256(b"finans-mlops-hash-check").hexdigest()
    assert verify_mod.sha256_file(path) == expected


def test_verify_model_hash_dogru_hash_gecer(tmp_path: pathlib.Path):
    """Tam sha256 uyumlu -> ok=True ve mode='sha256'."""
    model_path = _write_model(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(_freeze_payload(model_path)), encoding="utf-8")
    result = verify_mod.verify(model_path, freeze_path)
    assert result.ok is True
    assert result.mode == "sha256"


def test_verify_model_hash_yanlis_basarisiz(tmp_path: pathlib.Path):
    """Uyumsuz tam hash -> ok=False; iki hash de mesaj icinde."""
    model_path = _write_model(tmp_path)
    baska = "f" * 64
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(_freeze_payload(model_path, full=baska)), encoding="utf-8")
    result = verify_mod.verify(model_path, freeze_path)
    assert result.ok is False
    assert baska in result.message
    assert verify_mod.sha256_file(model_path) in result.message


def test_verify_model_hash_sha8_yedegi_zayif_uyari(tmp_path: pathlib.Path, caplog):
    """Tam hash yok, sha8 var -> ok=True; mode='sha8' + ZAYIF uyarisi."""
    model_path = _write_model(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(
        json.dumps(_freeze_payload(model_path, full="YOK", sha8=verify_mod.sha256_file(model_path)[:8])),
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        result = verify_mod.verify(model_path, freeze_path)
    assert result.ok is True
    assert result.mode == "sha8"
    assert "ZAYIF" in result.message


def test_verify_model_hash_eksik_dosya_veya_alan(tmp_path: pathlib.Path):
    """Model/freeze dosyasi ya da hash alani yoksa ok=False + acik mesaj."""
    yok = tmp_path / "yok.pt"
    result = verify_mod.verify(yok, tmp_path / "freeze.json")
    assert result.ok is False
    assert "model dosyasi yok" in result.message

    model_path = _write_model(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(_freeze_payload(model_path, full="YOK")), encoding="utf-8")
    result = verify_mod.verify(model_path, freeze_path)
    assert result.ok is False
    assert "hash'i bulunamadi" in result.message


# ---------------------------------------------------------------------------
# check_new_bar: gunluk workflow'un tatil kilidi
# ---------------------------------------------------------------------------

def _dataset(tmp_path: pathlib.Path, price_last: str, pred_last: str | None) -> tuple[pathlib.Path, pathlib.Path]:
    """Sentetik fiyat + tahmin dosyasi; tarihler ISO string.

    Fiyat deposu price_last'a KADAR gun icerir (sonraki gun yok) -> depo
    sonrasi max = price_last garanti.
    """
    prices_path = tmp_path / "prices.parquet"
    prices = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=5)})
    prices = prices[prices["date"] <= pd.Timestamp(price_last)]
    prices.to_parquet(prices_path, index=False)
    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    if pred_last is not None:
        empty = pd.DataFrame({"date": pd.to_datetime([]), "ticker": [], "score": []})
        empty.to_parquet(predictions_dir / f"predictions_{pred_last}.parquet", index=False)
    return prices_path, predictions_dir


def test_new_bar_fiyat_ileri_gun(tmp_path: pathlib.Path):
    """Son fiyat gunu son tahmin gununden ileri -> new."""
    prices_path, predictions_dir = _dataset(tmp_path, price_last="2025-01-06", pred_last="2025-01-03")
    state, price_last, pred_last = bar_mod.evaluate(prices_path, predictions_dir)
    assert state == "new"
    assert price_last == "2025-01-06" and pred_last == "2025-01-03"


def test_new_bar_tahmin_klasoru_bos_ilk_kosu(tmp_path: pathlib.Path):
    """Tahmin klasoru bos/yok -> new (ilk kosu)."""
    prices_path, predictions_dir = _dataset(tmp_path, price_last="2025-01-05", pred_last=None)
    state, _, pred_last = bar_mod.evaluate(prices_path, predictions_dir)
    assert state == "new" and pred_last is None


def test_none_tatil_ayni_gun(tmp_path: pathlib.Path):
    """Son tahmin gunu == son fiyat gunu -> none (tatil / zaten skorlandi)."""
    prices_path, predictions_dir = _dataset(tmp_path, price_last="2025-01-06", pred_last="2025-01-06")
    state, price_last, pred_last = bar_mod.evaluate(prices_path, predictions_dir)
    assert state == "none"
    assert price_last == pred_last == "2025-01-06"


def test_none_fiyat_geride_uyari(tmp_path: pathlib.Path, caplog):
    """Son fiyat gunu tahminlerden geride -> none (uyari loglanir; kesme yok)."""
    prices_path, predictions_dir = _dataset(tmp_path, price_last="2025-01-03", pred_last="2025-01-06")
    with caplog.at_level(logging.WARNING):
        state, _, _ = bar_mod.evaluate(prices_path, predictions_dir)
    assert state == "none"
    assert "geride" in caplog.text  # caplog.text formatli kayitlarin birlesimi


def test_fiyat_deposu_yok_hata(tmp_path: pathlib.Path):
    """Fiyat deposu yok -> FileNotFoundError (workflow hata ile uyarir)."""
    with pytest.raises(FileNotFoundError):
        bar_mod.evaluate(tmp_path / "yok.parquet", tmp_path)
