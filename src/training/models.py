"""Modeller: Ridge baseline + PyTorch MLP/LSTM/GRU siralayici.

HEDEF: her (ticker, t) ornegi icin TEK SKOR; skorlar gun icinde cross-sectional
siralanir (daily IC ile degerlendirilir). Model getirinin mutlak seviyesini
degil, gunun icinde "kim iyi sirada" bilgisini ogrenir.
"""

from __future__ import annotations

import torch
from sklearn.linear_model import Ridge
from torch import nn


class MLPRanker(nn.Module):
    """Siralama icin ileri-geribaglamali ag; katman/dropout/aktivasyon config'ten gelir."""

    def __init__(self, input_dim: int, hidden: list[int], dropout: float, activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        layers: list[nn.Module] = []
        prev = input_dim
        for unit_count in hidden:
            layers.append(nn.Linear(prev, int(unit_count)))
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = int(unit_count)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, F]; [B, T, F] gelirse duzlestirir
        if x.dim() == 3:
            return self.net(x.flatten(start_dim=1)).squeeze(-1)
        return self.net(x).squeeze(-1)


class RecurrentRanker(nn.Module):
    """LSTM veya GRU = ayni sinif, cell_type parametresiyle (kod tekrari yok).

    Neden recurrent: son seq_len gunun feature akisini tek tek gordugu icin
    kisa serilerdeki ivme/hazin yapisi yakalanir. 1-2 katman yeterlidir:
    veri kucuktur (gun = ~44 hisse), buyuk modeller ezberlemeye kayar.
    """

    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float, cell_type: str = "gru"):
        super().__init__()
        cell_type = cell_type.lower()
        if cell_type not in {"lstm", "gru"}:
            raise ValueError(f"cell_type 'lstm' veya 'gru' olmali, gelen: {cell_type}")
        dropout_value = float(dropout) if int(layers) > 1 else 0.0
        common = {"input_size": input_dim, "hidden_size": int(hidden), "num_layers": int(layers), "batch_first": True}
        self.cell = nn.LSTM(dropout=dropout_value, **common) if cell_type == "lstm" \
            else nn.GRU(dropout=dropout_value, **common)
        self.head = nn.Linear(int(hidden), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        output, _ = self.cell(x)     # output: [B, T, H]
        last_hidden = output[:, -1, :]  # sadece t anina kadarki bilgi
        return self.head(last_hidden).squeeze(-1)


class LSTMRanker(RecurrentRanker):
    """RecurrentRanker'in LSTM varyanti (spec adlariyla uyum icin)."""

    def __init__(self, input_dim: int, hidden: int, layers: int = 1, dropout: float = 0.0):
        super().__init__(input_dim=input_dim, hidden=hidden, layers=layers, dropout=dropout, cell_type="lstm")


class GRURanker(RecurrentRanker):
    """RecurrentRanker'in GRU varyanti (spec adlariyla uyum icin)."""

    def __init__(self, input_dim: int, hidden: int, layers: int = 1, dropout: float = 0.0):
        super().__init__(input_dim=input_dim, hidden=hidden, layers=layers, dropout=dropout, cell_type="gru")


class RidgeBaseline:
    """scikit-learn Ridge sarmalayici; torch modellerle ayni fit/predict arayuzu.

    Neden ridge baseline: z-score'lu feature'larin linear kombinasyonu cok guclu
    bir baseline'dir. Torch model bunu ve SPY'i gecemiyorsa bunu DURUST sekilde
    raporlariz. Ek olcekleme YOK: feature'lar ustup gunluk z-score'lu zaten.
    """

    def __init__(self, alpha: float, **kwargs):
        self.model = Ridge(alpha=float(alpha), **kwargs)

    def fit(self, x, y):
        """X ([N, F] numpy), y ([N] numpy) ile fit eder; self dondurur."""
        self.model.fit(x, y)
        return self

    def predict(self, x):
        """X ile skor ([N] numpy) dondurur."""
        return self.model.predict(x)


def build_model(name: str, input_dim: int, params: dict, seq_len: int = 0) -> nn.Module | RidgeBaseline:
    """Model adina gore instance uretir (isimler config'ten gelir).

    BU FONKSIYON SEED KURMAZ: cagridan ONCE set_seed(cfg.seed) yapilmis olmali;
    yoksa config seed'i buradaki varsayilan ile ezilir.
    """
    if name == "ridge":
        return RidgeBaseline(alpha=float(params.get("alpha", 10.0)))
    if name == "mlp":
        return MLPRanker(
            input_dim=input_dim,
            hidden=list(params.get("hidden", [64, 32])),
            dropout=float(params.get("dropout", 0.1)),
            activation=str(params.get("activation", "relu")),
        )
    cell_type = "lstm" if name == "lstm" else "gru"
    return RecurrentRanker(
        input_dim=input_dim,
        hidden=int(params.get("hidden", 32)),
        layers=int(params.get("layers", 1)),
        dropout=float(params.get("dropout", 0.1)),
        cell_type=cell_type,
    )
