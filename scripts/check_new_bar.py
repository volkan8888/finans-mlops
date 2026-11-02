"""Yeni tamamlanmis is gunu kontrolu: Gunluk workflow'un tatil kilidi.

Mantik: ingestion sonrasi fiyat deposunun SON gunu vs (forward-data dalindan
geri yuklenen) tahmin dosyalari. Son fiyat gunu icin tahmin ZATEN varsa
"none" (tatil / yeni bar yok) => workflow skorlamadan temiz cikar (exit 0).
BAYAT/terse mahal verilirse de none + uyari.

Kullanim:
    .venv\\Scripts\\python scripts/check_new_bar.py [--prices ...] [--predictions-dir ...]
CI'da script GITHUB_OUTPUT'a "bar=new|none" yazar (workflow adimi okumak icin).
Donus kodu: fiyat deposu yoksa 1 (veri hatasi); diger her durum 0.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PRICES_PATH = pathlib.Path("data/raw/prices.parquet")
DEFAULT_PREDICTIONS_DIR = pathlib.Path("data/predictions")
NAME_PATTERN = re.compile(r"^predictions_(\d{4}-\d{2}-\d{2})\.parquet$")


def parse_prediction_date(filename: str) -> str | None:
    """predictions_YYYY-MM-DD.parquet adindan ISO tarih dondurur; uyumsuzsa None."""
    match = NAME_PATTERN.match(pathlib.Path(filename).name)
    return match.group(1) if match else None


def latest_prediction_date(predictions_dir: str | pathlib.Path) -> str | None:
    """Tahmin klasorundeki en buyuk tarih (dosya adindan; parquet okumaz)."""
    directory = pathlib.Path(predictions_dir)
    if not directory.is_dir():
        return None
    dates = [parse_prediction_date(item.name) for item in directory.glob("predictions_*.parquet")]
    valid = sorted(item for item in dates if item is not None)
    return valid[-1] if valid else None


def latest_price_date(prices_path: str | pathlib.Path) -> str:
    """Fiyat deposunun (ingestion sonrasi) son tarihini ISO string dondurur."""
    path = pathlib.Path(prices_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fiyat deposu yok (ingestion calismali?): {path}")
    try:
        prices = pd.read_parquet(path, columns=["date"])  # hizli: sadece tarih
    except (KeyError, ValueError):
        prices = pd.read_parquet(path)
    return str(pd.Timestamp(prices["date"].max()).date())


def evaluate(
    prices_path: str | pathlib.Path = DEFAULT_PRICES_PATH,
    predictions_dir: str | pathlib.Path = DEFAULT_PREDICTIONS_DIR,
) -> tuple[str, str | None, str | None]:
    """Karar: ("new" | "none", son fiyattan tarih, son tahmin tarih).

    - none: son tahmin gunu >= son fiyat gunu (tatil / zaten skorlandi) -
      akis temiz cikar.
    - new : son fiyat gunu > son tahmin gunu (normal islem gunu akis ilerler).
    """
    price_last = latest_price_date(prices_path)
    pred_last = latest_prediction_date(predictions_dir)
    if pred_last is None or price_last > pred_last:
        return "new", price_last, pred_last
    if price_last < pred_last:
        logger.warning(
            "Fiyat son tarihi tahminlerden geride (%s < %s); yeni skorlanabilecek gun yok => none",
            price_last, pred_last,
        )
    else:
        logger.info(
            "Tatil / yeni bar yok: fiyat son gunu %s icin tahmin zaten var",
            price_last,
        )
    return "none", price_last, pred_last


def main(argv: list[str] | None = None) -> int:
    """Loglar + CI output; fiyat deposu yoksa 1, degilse 0 doner."""
    parser = argparse.ArgumentParser(description="Yeni tamamlanmis is gunu kontrolu")
    parser.add_argument("--prices", default=str(DEFAULT_PRICES_PATH), help="fiyat deposu parquet")
    parser.add_argument("--predictions-dir", default=str(DEFAULT_PREDICTIONS_DIR), help="tahmin klasoru")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    try:
        state, price_last, pred_last = evaluate(args.prices, args.predictions_dir)
    except FileNotFoundError as error:
        logger.error("Yeni bar kontrolu BASARISIZ: %s", error)
        return 1
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with pathlib.Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"bar={state}\n")
            handle.write(f"price_last={price_last or ''}\n")
    logger.info("Karar: bar=%s (fiyat son gunu=%s, son tahmin=%s)", state, price_last, pred_last)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
