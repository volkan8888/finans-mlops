"""El-yazimi PyTorch egitim dongusu + train_model giris noktasi.

- Cihaz: cfg.device ("auto" -> cuda varsa cuda, yoksa cpu).
- Seed: random/numpy/torch cekirdekleri set_seed ile sabitlenir.
- Shuffle YOK: DataLoader'lar shuffle=False; veri zamana bolunurken
  karistirmanin hicbir yeri yok; mini-batch sirasi bile kronolojik kalir
  (determinism + "asla shuffle yok" kurali).
- Kayip: label (rank 0-1) uzerinde MSE; alternatif listnet (gunluk pairwise
  softmax CE; config: loss).
- Early stopping: VAL IC (gunluk Spearman ortalamasi) maksimize edilir;
  iyilesme patience epoch kadar durunca durur; EN IYI agirliklar geri
  yuklenir.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training.dataset import SequenceDataset, TabularDataset
from training.evaluate import daily_ic
from training.models import RidgeBaseline

logger = logging.getLogger(__name__)

SEQUENCE_MODELS = {"lstm", "gru"}
META_COLUMNS = {"date", "ticker", "forward_return", "label"}


def set_seed(seed: int) -> None:
    """random/numpy/torch cekirdeklerini sabitler (cuda dahil)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    """'auto' -> cuda varsa 'cuda', yoksa 'cpu'; aksi halde degeri dogrular."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"device bilinmeyen: {device}")
    return device


def _listnet_loss(scores: torch.Tensor, labels: torch.Tensor, date_ids: torch.Tensor) -> torch.Tensor:
    """Gunluk pairwise softmax kaybi (ListNet tarzi).

    Her date_id grubunda: hedef dagilim softmax(labels), tahmin dagilimi
    log_softmax(scores); ikisi arasinda cross-entropy. Tek ornekli gunler
    katki vermez (iclerinde siralama yok).
    """
    total = scores.new_zeros(())
    group_count = 0
    for day in date_ids.unique():
        mask = date_ids == day
        if int(mask.sum()) < 2:
            continue
        p_target = F.softmax(labels[mask], dim=0)
        log_pred = F.log_softmax(scores[mask], dim=0)
        total = total + F.kl_div(log_pred, p_target, log_target=False, reduction="sum")
        group_count += 1
    if group_count == 0:
        return scores.new_zeros(())
    return total / group_count


class Trainer:
    """El-yazimi dongu: forward -> loss -> backward -> optimizer.step.

    LR scheduler (ReduceLROnPlateau, mode=max) VAL IC'yi izler; iyilesme
    durunca ogrenme oranini yarilar. Early stopping yine val IC'ye gore.
    """

    def __init__(self, model, params: dict[str, Any], device: str, loss_kind: str = "mse"):
        self.model = model
        self.params = params
        self.device = device
        self.loss_kind = loss_kind
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params.get("lr", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-5
        )

    def _loss(self, scores: torch.Tensor, y: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "listnet":
            return _listnet_loss(scores, y, d)
        return F.mse_loss(scores, y)

    def _run_epoch(self, loader: DataLoader, train_mode: bool):
        """Bir epoch gezer: (ortalama kayip, (date_id, score, label) ciktilari)."""
        self.model.train() if train_mode else self.model.eval()
        loss_total = 0.0
        sample_total = 0
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for x, y, d in loader:
            x, y, d = x.to(self.device), y.to(self.device), d.to(self.device)
            scores = self.model(x)
            loss = self._loss(scores, y, d)
            if train_mode:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.params.get("grad_clip", 1.0))
                )
                self.optimizer.step()
            outputs.append((d.detach().cpu(), scores.detach().cpu(), y.detach().cpu()))
            loss_total += float(loss.item()) * len(x)
            sample_total += len(x)
        return loss_total / max(1, sample_total), outputs

    @staticmethod
    def _outputs_to_ic(outputs: list) -> pd.Series:
        """(date_id, score, label) ciktilarini gunluk IC serisine cevirir."""
        frames = [
            pd.DataFrame({"date": d.numpy(), "score": s.numpy(), "label": y.numpy()})
            for d, s, y in outputs
        ]
        panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            {"date": [], "score": [], "label": []}
        )
        return daily_ic(panel, "score", "label", min_n=2)

    def fit(self, train_ds: Dataset, val_ds: Dataset, epochs: int, patience: int) -> dict[str, Any]:
        """Egitim dongusu; history dict dondurur; en iyi agirliklari yukler."""
        train_loader = DataLoader(
            train_ds, batch_size=int(self.params.get("batch_size", 256)), shuffle=False
        )
        val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)
        best_state = None
        best_ic = -float("inf")
        best_epoch = -1
        stall = 0
        history: dict[str, Any] = {"epoch": [], "train_loss": [], "val_loss": [], "val_ic": [], "lr": []}
        for epoch in range(1, int(epochs) + 1):
            train_loss, _ = self._run_epoch(train_loader, train_mode=True)
            val_loss, val_outputs = self._run_epoch(val_loader, train_mode=False)
            ic_series = self._outputs_to_ic(val_outputs)
            val_ic = float(ic_series.mean()) if len(ic_series) > 0 else float("nan")
            self.scheduler.step(val_ic)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_ic"].append(val_ic)
            history["lr"].append(self.optimizer.param_groups[0]["lr"])
            logger.info(
                "epoch=%02d | train_loss=%.5f | val_loss=%.5f | val_ic=%.4f | lr=%.6f",
                epoch, train_loss, val_loss, val_ic, self.optimizer.param_groups[0]["lr"],
            )
            if np.isnan(val_ic):
                stall += 1
            elif float(val_ic) > best_ic:
                best_ic = float(val_ic)
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                stall = 0
            else:
                stall += 1
            if stall >= int(patience):
                logger.info(
                    "Early stopping: epoch=%d | en iyi val_ic=%.4f (epoch=%d)",
                    epoch, best_ic, best_epoch,
                )
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        history["best_epoch"] = best_epoch
        history["best_val_ic"] = None if best_epoch < 0 else float(best_ic)
        return history


def predict_scores(model, dataset: Dataset, device: str, batch_size: int = 4096) -> pd.DataFrame:
    """Dataset uzerinden skor uretir; (date, ticker, score, label) tablosu dondurur.

    Sadece torch modeller icindir; ridge'in sklearn yolu run.py'de ayridir.
    model.eval() cagrilir ve no_grad ile ileri gecis yapilir (gradif takipsiz).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    is_torch = isinstance(model, torch.nn.Module)
    if is_torch:
        model.eval()
    scores_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    for x, y, _d in loader:
        if is_torch:
            x = x.to(device)
            with torch.no_grad():
                scores = model(x).cpu().numpy()
        else:  # sklearn benzeri arayuz (yedek yol)
            scores = np.asarray(model.predict(x.numpy())).ravel()
        scores_out.append(np.asarray(scores).ravel())
        labels_out.append(y.numpy())
    scores = np.concatenate(scores_out)
    labels = np.concatenate(labels_out)
    return pd.DataFrame({
        "date": pd.to_datetime(dataset.sample_dates.to_numpy()),
        "ticker": dataset.sample_tickers.to_numpy(),
        "score": scores,
        "label": labels,
    })


def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Any,
    feature_cols: list[str] | None = None,
    full_df: pd.DataFrame | None = None,
) -> tuple[object, dict]:
    """Model egitir ve (model, history) dondurur.

    - ridge: sklearn fit; torch yok (hizli baseline).
    - mlp: TabularDataset + Trainer.
    - lstm/gru: SequenceDataset + Trainer; PENCERE GECMISE UZANABILIR diye
      full_df (TAM panel) beklenir: val penceresindeki orneklerin de kendi
      gecmis feature'larina erisebilmesi icin. Pencere yalnizca gecmise bakar;
      tam panel gecmek sizinti DEGILDIR (label sadece hedef satirdan okunur).
    """
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c not in META_COLUMNS]
    set_seed(cfg.seed)
    params = {**cfg.defaults}
    for entry in cfg.models:
        if entry["name"] == model_name:
            params.update({k: v for k, v in entry["params"].items()})

    if model_name == "ridge":
        model = RidgeBaseline(alpha=float(params.get("alpha", 10.0)))
        model.fit(train_df[feature_cols].to_numpy(), train_df["label"].to_numpy())
        preds = model.predict(val_df[feature_cols].to_numpy())
        val_frame = val_df.assign(score=preds)
        ic = daily_ic(val_frame, "score", "label", min_n=2)
        history = {"val_ic": [float(np.nanmean(ic)) if len(ic) > 0 else float("nan")]}
        return model, history

    from training.models import build_model  # dongusel import'u yerelde tut

    device = resolve_device(cfg.device)
    set_seed(cfg.seed)  # ayni cihaz/model insasindan once tekrar sabitle
    model = build_model(model_name, len(feature_cols), params).to(device)
    if model_name in SEQUENCE_MODELS:
        panel = full_df if full_df is not None else pd.concat([train_df, val_df], ignore_index=True)
        train_ds: Dataset = SequenceDataset(panel, dates=train_df["date"], feature_cols=feature_cols, seq_len=cfg.seq_len)
        val_ds: Dataset = SequenceDataset(panel, dates=val_df["date"], feature_cols=feature_cols, seq_len=cfg.seq_len)
    else:
        train_ds = TabularDataset(train_df, feature_cols)
        val_ds = TabularDataset(val_df, feature_cols)
    trainer = Trainer(model, params, device, loss_kind=cfg.loss)
    history = trainer.fit(
        train_ds, val_ds,
        epochs=int(params.get("epochs", 30)),
        patience=int(params.get("patience", 4)),
    )
    return model, history
