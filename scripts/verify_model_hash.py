"""Freeze imzasinin model dosyasi hash uyumunu dogrular.

Amac: gunluk ileri test akisi YANLIS modelle tahmin uretmesin.
Kaynak: reports/forward_test/freeze.json -> model.sha256.model_files["model.pt"]
Tam hash varsa birebir karsilastirilir. Tam hash yoksa (eski imza) sha8
(ilk 8 hex) ile karsilastirilir ve ZAYIF karsilastirma uyarisi loglanir.

Kullanim (workflow + yerel):
    .venv\\Scripts\\python scripts/verify_model_hash.py ^
        --model models/gru/model.pt --freeze reports/forward_test/freeze.json
Donus kodu: 0 uyum | 1 uyumsuzluk / dosya eksikligi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# sha8 (kisa imza) icin olasi yaerlesik anahtarlar (eski/modifiye freeze'ler icin).
SHA8_KEYS = ("sha8", "model_sha8", "sha256_sha8", "sha256_8")


@dataclass
class HashCheck:
    """Hash karsilastirma sonucu: ok + karsilastirma modu + mesaj."""

    ok: bool
    mode: str  # "sha256" (tam) | "sha8" (zayif)
    message: str


def sha256_file(path: str | pathlib.Path) -> str:
    """Dosyayi parca parca okuyarak sha256 hex ozetini dondurur (buyuk dosya-guvenli)."""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_from_freeze(freeze: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """freeze sozlukten beklenen hash'i cikarir: (tam-hash|None, sha8|None, kaynak).

    Onemli: kaynak hangi dosyadan geldi "full" veya "sha8" donmez; cagiran
    mesaj kurar. model_files["model.pt"] 64 hex ise tam; yoksa SHA8_KEYS
    sirasiyla denenir.
    """
    model_section = freeze.get("model") or {}
    sha_section = model_section.get("sha256") or {}
    model_files = sha_section.get("model_files") or {}
    full = model_files.get("model.pt")
    if isinstance(full, str) and len(full) == 64:
        return full, None, "full"
    for key in SHA8_KEYS:
        value = model_section.get(key)
        if isinstance(value, str) and value:
            return None, value, key
    return None, None, "yok"


def verify(model_path: str | pathlib.Path, freeze_path: str | pathlib.Path) -> HashCheck:
    """model.pt hash'ini freeze.json ile karsilastirir.

    Donus HashCheck(ok, mode, message):
    - ok=True ve mode="sha256": tam imza uyumu.
    - ok=True ve mode="sha8": ZAYIF imza uyumu (tam hash freeze'de yok; uyari
      mesajinda bildirilir; cagiran loglamadiginda gorunmez).
    - ok=False: uyumsuzluk ya da dosya/alan eksikligi; mesaj aciklar.
    """
    model_file = pathlib.Path(model_path)
    freeze_file = pathlib.Path(freeze_path)
    if not model_file.is_file():
        return HashCheck(False, "sha256", f"model dosyasi yok: {model_file}")
    if not freeze_file.is_file():
        return HashCheck(False, "sha256", f"freeze dosyasi yok: {freeze_file}")
    try:
        freeze = json.loads(freeze_file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        return HashCheck(False, "sha256", f"freeze JSON bozuk: {error}")

    full, sha8, source = _expected_from_freeze(freeze)
    actual = sha256_file(model_file)
    if full is not None:
        if actual == full.lower():
            return HashCheck(True, "sha256", f"tam sha256 uyumlu: {actual[:12]}...")
        return HashCheck(False, "sha256", f"hash UYUSMUYOR: beklenen={full} bulunan={actual}")
    if sha8 is not None:
        if actual[:8] == sha8.lower() or actual[: len(sha8)] == sha8.lower():
            return HashCheck(True, "sha8", f"ZAYIF karsilastirma (tam hash yok): sha8={actual[:8]}")
        return HashCheck(
            False,
            "sha8",
            f"hash UYUSMUYOR (sha8): beklenen={sha8} bulunan={actual[:8]} (tam={actual})",
        )
    return HashCheck(
        False,
        "sha256",
        f"freeze.json'da model.pt hash'i bulunamadi (model.sha256.model_files / {SHA8_KEYS}) [kaynak={source}]",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI girisi: sonucu loglar; uyumsuzlukta 1, uyumda 0 doner."""
    parser = argparse.ArgumentParser(description="freeze.json icindeki model.pt hash dogrulamasi")
    parser.add_argument("--model", required=True, help="model.pt dosya yolu")
    parser.add_argument("--freeze", required=True, help="reports/forward_test/freeze.json yolu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    result = verify(args.model, args.freeze)
    if result.ok and result.mode == "sha256":
        logger.info("Model hash DOGRULANDI (tam sha256): %s", result.message)
    elif result.ok:
        # Zayif imza: sessiz gecme, cagirana haber ver.
        logger.warning("Model hash sha8 ile KARSILANDIRILDI (tam hash freeze'de yok): %s", result.message)
    else:
        logger.error("Model hash dogrulamasi BASARISIZ: %s", result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
