"""Egitim veri kumeleri: TabularDataset (MLP/ridge) ve SequenceDataset (GRU/LSTM).

SIZINTI KURALLARI:
- Dataset'ler TARIH LISTESI alir; hangi orneklerin donecegini bolme
  (splits.py) belirler; dataset bolmeden habersizdir.
- SequenceDataset penceresi: (ticker, t) ornegi icin SON `seq_len` feature
  satiri [t-seq_len+1 .. t]; pencere SADECE t ve ONCESINE bakar; gelecek
  girmaz. Pencere tek ticker'in kendi serisinden kurulur (ticker karismaz).
- Pencere tam degilse (ticker'in t'de seq_len adet gecmis feature satiri
  yoksa) ornek ATLANIR (doldurma yapilmaz).
- Label pencerede DEGILDIR; sadece hedef satirin etiketi olarak dondurulur.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


def _subset(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Belirtilen tarihlere ait satirlari alip (date, ticker) siralar."""
    out = df[df["date"].isin(dates)].copy()
    return out.sort_values(["date", "ticker"], kind="mergesort")


def tabular_arrays(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Satir bazli egitim icin (X, y, date_id) tensor dondurur.

    date_id: gunluk groupby/ListNet/daily-IC icin kullanilan TAM SAYI grup
    anahtari (0..T-1); sayisal anlami yoktur.
    """
    unique_days = pd.DatetimeIndex(df["date"].unique()).sort_values()
    date_to_id = {day: idx for idx, day in enumerate(unique_days)}
    x = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.float32)
    d = torch.tensor(df["date"].map(date_to_id).to_numpy(), dtype=torch.long)
    return x, y, d


class TabularDataset(Dataset):
    """MLP ve ridge icin satir-bazli ornek: (feature vektoru, label, date_id)."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature kolonlari yok: {missing}")
        self.x, self.y, self.d = tabular_arrays(df, feature_cols)
        # tahminleri satirlara geri iliskilendirmek icin
        self.sample_dates = pd.Series(df["date"].to_numpy())
        self.sample_tickers = pd.Series(df["ticker"].to_numpy())

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.d[index]


class SequenceDataset(Dataset):
    """GRU/LSTM icin sekans ornekleri: her (ticker, t) icin son `seq_len` feature gunu.

    Veri paneli TAMAMI yuklenir (tum tarihlere); sadece HEDEF tarih listede
    olan (ticker, date) satirlar ornek olur. Pencere, t'den gecmise dogru
    uzanabilir; dogru davranis budur: val'de de model, o t'den ONCEKI gercek
    featurelara erisebilir (serving'de de aynen boylesidir).
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        dates: pd.DatetimeIndex,
        feature_cols: list[str],
        seq_len: int,
    ):
        if seq_len <= 0:
            raise ValueError(f"seq_len pozitif olmali, gelen: {seq_len}")
        self.seq_len = seq_len
        self.feature_cols = list(feature_cols)
        missing = [c for c in self.feature_cols if c not in features_df.columns]
        if missing:
            raise ValueError(f"Feature kolonlari yok: {missing}")

        # pencere ve etiket tensorleri her ticker icin ayri tutulur (karismaz)
        self._ticker_windows: dict[str, torch.Tensor] = {}
        self._labels: dict[str, torch.Tensor] = {}
        self._dates: dict[str, pd.DatetimeIndex] = {}
        self._samples: list[tuple[str, int]] = []  # (ticker, pos)
        date_ids: set[pd.Timestamp] = set()

        target_days = set(pd.DatetimeIndex(dates))
        panel = features_df.sort_values(["ticker", "date"], kind="mergesort")
        for ticker, group in panel.groupby("ticker", sort=False):
            group = group.reset_index(drop=True)
            days = pd.DatetimeIndex(group["date"])
            self._dates[ticker] = days
            self._ticker_windows[ticker] = torch.tensor(
                group[self.feature_cols].to_numpy(), dtype=torch.float32
            )
            self._labels[ticker] = torch.tensor(
                group["label"].to_numpy(), dtype=torch.float32
            )

        for ticker, group in panel.groupby("ticker", sort=False):
            ticker_dates = self._dates[ticker]
            for pos, day in enumerate(ticker_dates):
                if day not in target_days:
                    continue
                if torch.isnan(self._labels[ticker][pos]):
                    continue  # label'siz satir; features.parquet'te zaten silinmis olur
                if pos < self.seq_len - 1:
                    continue  # pencere tam degil -> ornek atlanir (doldurma yok)
                self._samples.append((ticker, pos))
                date_ids.add(day)

        # tahminleri satirlara geri iliskilendirmek icin
        self.sample_dates = pd.Series([self._dates[t][p] for t, p in self._samples])
        self.sample_tickers = pd.Series([t for t, _ in self._samples])
        self.date_ids = {day: idx for idx, day in enumerate(sorted(date_ids))}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        ticker, pos = self._samples[index]
        window = self._ticker_windows[ticker][pos - self.seq_len + 1 : pos + 1]
        label_value = self._labels[ticker][pos]
        day = self._dates[ticker][pos]
        date_id = self.date_ids[day]
        return window, label_value, torch.tensor(date_id, dtype=torch.long)
