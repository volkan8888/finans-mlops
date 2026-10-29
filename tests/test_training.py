"""Adim 3 (FAZ 1) egitim testleri — ag yok, sentetik veri, hizli."""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # torch yoksa bu modul atlanir (kurulu olmali)

from training.config import load_training_config
from training.dataset import SequenceDataset, TabularDataset
from training.evaluate import daily_ic, quantile_returns
from training.models import build_model
from training.splits import holdout_split, walk_forward_splits
from training.train import TabularDataset as TabDS
from training.train import Trainer, set_seed, train_model

FEATURES = ["f0", "f1"]


def make_dates(
    n_days: int = 1200, start: str = "2018-01-01"
) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n_days)


def make_panel(
    n_days: int = 60, tickers: tuple[str, ...] = ("AAA", "BBB", "CCC"), seed: int = 0
) -> pd.DataFrame:
    """Sintetik panel: her (ticker, date) icin f0/f1 ve bunlara bagli label.

    label = gun icinde f0 - 0.5*f1'in rank'i (0..1); modelin OGRENEBILECEGI
    bir iliski kurar (train.py'daki ogrenme testinin temeli).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    base = {t: rng.normal(0.0, 1.0) for t in tickers}
    for date in dates:
        day = {}
        for ticker in tickers:
            day[ticker] = (
                base[ticker] + 0.2 * rng.normal(),
                rng.normal(),
            )
        combo = {t: day[t][0] - 0.5 * day[t][1] for t in tickers}
        ranks = pd.Series(combo).rank()
        for ticker in tickers:
            rows.append({
                "date": date,
                "ticker": ticker,
                "f0": day[ticker][0],
                "f1": day[ticker][1],
                "forward_return": 0.001 * ranks[ticker],
                "label": ranks[ticker] / len(tickers),
            })
    return pd.DataFrame(rows)


class DummyCfg:
    """train_model icin hafif config kopyasi (dosya okumadan)."""

    seed = 42
    device = "cpu"
    loss = "mse"
    seq_len = 5
    defaults: ClassVar[dict] = {"lr": 0.005, "weight_decay": 0.0, "batch_size": 16, "epochs": 40, "patience": 3, "grad_clip": 1.0}
    models = ({"name": "mlp", "params": {"hidden": [16, 8], "dropout": 0.0}},)


# ---------------------------------------------------------------- splits


def test_holdout_split_dev_holdout_cekismiyor_ve_gap_dogru():
    dates = make_dates(1300)  # ~5 yil
    dev, holdout = holdout_split(dates, holdout_years=2, purge_days=5, embargo_days=5)
    assert len(dev) > 0 and len(holdout) > 0
    # dev'in son gunu holdout'un ilk gununden ONCE (purge+embargo gunu kadar)
    holdout_start_pos = int(dates.searchsorted(holdout.min()))
    dev_end_pos = int(dates.searchsorted(dev.max()))
    assert holdout_start_pos - dev_end_pos == 5 + 5 + 1  # aradaki islem gunu sayisi
    assert set(dev) & set(holdout) == set()
    assert dev.max() < holdout.min()


def test_walk_forward_foldlar_kronolojik_expanding_ve_gap_dogru():
    dates = make_dates(800, start="2020-01-02")
    folds = walk_forward_splits(dates, min_train_years=1, val_months=3, step_months=3, purge_days=5, embargo_days=5)
    assert len(folds) >= 3
    for (train_prev, _), (train_cur, _) in pairwise(folds):
        # buyuyen pencere: sonraki fold'un train'i, oncekiniatisfi KAPSIYOR
        assert set(train_prev).issubset(set(train_cur))
    for train, val in folds:
        assert train.max() < val.min()
        val_start_pos = int(dates.searchsorted(val.min()))
        train_end_pos = int(dates.searchsorted(train.max()))
        assert val_start_pos - train_end_pos == 5 + 5 + 1
    # fold'lar val'lerine gore artan sirada
    val_starts = [val.min() for _, val in folds]
    assert val_starts == sorted(val_starts)
    assert len(set(val_starts)) == len(val_starts)  # cakismayan val'ler


# ---------------------------------------------------------------- datasets


def test_sequence_dataset_pencere_sadece_gecmis_ve_degisim_yerinde():
    rows = []
    n = 40
    dates = pd.bdate_range("2024-01-01", periods=n)
    for i, date in enumerate(dates):
        rows.append({"date": date, "ticker": "AAA", "f0": float(i), "f1": 1.0,
                     "forward_return": 0.0, "label": float(i) / n})
    panel = pd.DataFrame(rows)
    target = pd.DatetimeIndex([dates[29], dates[39]])  # pencere t<=29 ve t<=39
    ds = SequenceDataset(panel, dates=target, feature_cols=["f0", "f1"], seq_len=5)
    assert len(ds) == 2
    win_29, _, _ = ds[0]
    assert win_29.shape == (5, 2)
    # pencere SON satiri t gunudur: f0 = 29.0 (40 gunde 39uncu satir degil!)
    assert float(win_29[-1, 0]) == 29.0
    # gelecek yok: pencere max deger <= t
    assert float(win_29.max()) <= 29.0
    win_39, _, _ = ds[1]
    assert float(win_39[-1, 0]) == 39.0
    # sondaki 4 satir ayni (26..29 vs 36..39 degil — 29.bitir satiri degismedi)
    assert torch.allclose(win_39[1:], win_29[1:]) is False  # farkli gunler farkli pencere
    # simdi gelecegi (30+) BOZ: eski ornek (t=29) penceresi DEGISMEZ
    panel_changed = panel.copy()
    panel_changed.loc[panel_changed["date"] > dates[29], "f0"] = 999.0
    ds2 = SequenceDataset(panel_changed, dates=target, feature_cols=["f0", "f1"], seq_len=5)
    win_29_changed, _, _ = ds2[0]
    assert torch.allclose(win_29_changed, win_29)


def test_sequence_dataset_ticker_karismiyor():
    dates = pd.bdate_range("2024-01-01", periods=30)
    rows = []
    for ticker, sign in (("AAA", 1.0), ("BBB", -1.0)):
        for i, date in enumerate(dates):
            rows.append({"date": date, "ticker": ticker, "f0": sign * (i + 1), "f1": sign,
                         "forward_return": 0.0, "label": float(i)})
    panel = pd.DataFrame(rows)
    target = pd.DatetimeIndex(dates[10:])
    ds = SequenceDataset(panel, dates=target, feature_cols=["f0", "f1"], seq_len=5)
    for i in range(len(ds)):
        win, _, _ = ds[i]
        if ds.sample_tickers.iloc[i] == "AAA":
            assert float(win.min()) > 0  # AAA hep pozitif
        else:
            assert float(win.max()) < 0  # BBB hep negatif


def test_sequence_dataset_pencere_tam_degilse_ornek_atlanir():
    dates = pd.bdate_range("2024-01-01", periods=7)
    rows = [{"date": d, "ticker": "AAA", "f0": float(i), "f1": 0.0,
             "forward_return": 0.0, "label": 0.5} for i, d in enumerate(dates)]
    panel = pd.DataFrame(rows)
    ds = SequenceDataset(panel, dates=pd.DatetimeIndex(dates), feature_cols=["f0", "f1"], seq_len=5)
    # ilk 4 gunde pencere tam degil -> sadece index 4..6 ornek olur
    assert len(ds) == 3
    assert ds.sample_dates.iloc[0] == dates[4]


def test_tabular_dataset_satirlar_uyumlu():
    frame = make_panel(n_days=5)
    ds = TabularDataset(frame, FEATURES)
    assert len(ds) == len(frame)
    assert list(ds.sample_tickers.iloc[:3]) == list(frame["ticker"].iloc[:3])
    x, _y, d = ds[0]
    assert x.shape == (2,) and d.dtype == torch.long


# ---------------------------------------------------------------- evaluate


def test_daily_ic_mukemmel_ters_ve_rastgele():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows = []
    for date in dates:
        scores = rng.normal(size=20)
        labels = pd.Series(scores).rank(pct=True).to_numpy()
        for k in range(20):
            rows.append({"date": date, "ticker": f"T{k:02d}", "pred": scores[k],
                         "target": labels[k], "ret": 0.0})
    frame = pd.DataFrame(rows)
    ic = daily_ic(frame, "pred", "target", min_n=2)
    assert (ic.dropna() > 0.99).all()

    frame_rev = frame.copy()
    frame_rev["pred"] = -frame_rev["pred"]
    ic_rev = daily_ic(frame_rev, "pred", "target", min_n=2)
    assert (ic_rev.dropna() < -0.99).all()

    dates = pd.bdate_range("2024-03-01", periods=60)
    rows = []
    for date in dates:
        scores = rng.normal(size=15)
        labels = rng.normal(size=15)  # bagimsiz
        for k in range(15):
            rows.append({"date": date, "ticker": f"T{k:02d}", "pred": scores[k], "target": labels[k], "ret": 0.0})
    frame_rand = pd.DataFrame(rows)
    ic_rand = daily_ic(frame_rand, "pred", "target", min_n=2)
    assert abs(float(np.nanmean(ic_rand))) < 0.15


def test_quantile_returns_monoton():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=60)
    rows = []
    for date in dates:
        vals = rng.normal(size=25)
        labels = pd.Series(vals).rank(pct=True).to_numpy()
        for k in range(25):
            rows.append({"date": date, "ticker": f"T{k:02d}", "pred": vals[k],
                         "target": labels[k], "ret": labels[k]})
    frame = pd.DataFrame(rows)
    out = quantile_returns(frame, "pred", "ret", n_quantiles=5)
    means = out["mean_return"].to_list()
    assert all(a < b for a, b in pairwise(means))  # kutularca artan


# ---------------------------------------------------------------- models + trainer


def test_model_forward_sekilleri_dogru():
    m = build_model("mlp", 4, {"hidden": [8, 4], "dropout": 0.0})
    assert m(torch.randn(7, 4)).shape == (7,)
    g = build_model("gru", 4, {"hidden": 8, "layers": 1, "dropout": 0.0})
    assert g(torch.randn(7, 6, 4)).shape == (7,)
    l = build_model("lstm", 4, {"hidden": 8, "layers": 2, "dropout": 0.1})
    assert l(torch.randn(7, 6, 4)).shape == (7,)
    r = build_model("ridge", 4, {"alpha": 1.0})
    import numpy as np
    r.fit(np.random.normal(size=(20, 4)), np.random.normal(size=20))
    assert r.predict(np.random.normal(size=(5, 4))).shape == (5,)


def test_seed_ile_tekrarlanabilirlik_ayni_kayip():
    frame = make_panel(n_days=20, seed=1)
    losses = []
    for _ in range(2):
        set_seed(123)
        model = build_model("mlp", 2, {"hidden": [8], "dropout": 0.0})
        x = torch.tensor(frame[FEATURES].to_numpy()[:8], dtype=torch.float32)
        y = torch.tensor(frame[FEATURES].sum(axis=1).to_numpy()[:8], dtype=torch.float32)
        loss = torch.nn.functional.mse_loss(model(x), y)
        losses.append(float(loss.item()))
    assert losses[0] == pytest.approx(losses[1], abs=1e-9)


def test_trainer_ogreniyor_ve_val_ic_pozitif():
    frame = make_panel(n_days=80, tickers=("AAA", "BBB", "CCC", "DDD"), seed=5)
    split = frame["date"].unique()[60]
    train_df = frame[frame["date"] < split]
    val_df = frame[frame["date"] >= split]

    class LocalCfg(DummyCfg):
        seed = 42
        seq_len = 5

    _model, history = train_model("mlp", train_df, val_df, LocalCfg(), feature_cols=["f0", "f1"], full_df=frame)
    best = history["best_val_ic"]
    assert best is not None and best > 0.2  # ogrenilebilir iliski -> pozitif val IC
    assert history["train_loss"][-1] < history["train_loss"][0]  # kayip dustu


def test_trainer_early_stopping_tetiklaniyor():
    # val etiketleri gun icinde sabit -> std=0 -> daily_ic NaN -> stall -> erken durur
    rng = np.random.default_rng(11)
    rows = []
    for i, date in enumerate(pd.bdate_range("2024-01-01", periods=60)):
        for k in range(4):
            label_value = 0.5 if date >= pd.Timestamp("2024-03-18") else rng.uniform()
            rows.append({"date": date, "ticker": f"T{k}", "f0": rng.normal(), "f1": rng.normal(),
                         "forward_return": 0.0, "label": label_value})
    frame = pd.DataFrame(rows)

    class ShortCfg(DummyCfg):
        defaults: ClassVar[dict] = {**DummyCfg.defaults, "epochs": 20, "patience": 2}

    split = frame["date"].unique()[45]
    _model, history = train_model(
        "mlp", frame[frame["date"] < split], frame[frame["date"] >= split],
        ShortCfg(), feature_cols=["f0", "f1"], full_df=frame,
    )
    assert len(history["epoch"]) < 20  # erken durdu
    assert history["best_epoch"] is not None


def test_train_model_ridge_ogrenebiliyor():
    frame = make_panel(n_days=200, tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF"), seed=9)
    split = frame["date"].unique()[150]

    class LocalCfg(DummyCfg):
        models = ({"name": "ridge", "params": {"alpha": 1.0}},)

    _model, history = train_model("ridge", frame[frame["date"] < split], frame[frame["date"] >= split], LocalCfg(),
                                  feature_cols=["f0", "f1"], full_df=frame)
    assert history["val_ic"][0] > 0.5  # label-feature iliskisi linear: ridge guclu


def test_config_loader_gecerli_degerleri_aliyor():
    cfg = load_training_config()
    assert cfg.seed == 42
    assert cfg.holdout_years == 2
    assert cfg.purge_days == cfg.label_horizon
    assert {"ridge", "mlp", "gru"} <= {m["name"] for m in cfg.models}
    assert cfg.mlflow["experiment"] == "finans-training"
    # training semasina uymayan bir yaml (data.yaml) hataya dusurur (models bolumu bos)
    with pytest.raises(ValueError):
        load_training_config("config/data.yaml")


def test_trainer_class_fit_dongusu_kimin():
    frame = make_panel(n_days=30, seed=2)
    ds = TabDS(frame, FEATURES)
    val_ds = TabDS(frame, FEATURES)
    model = build_model("mlp", 2, {"hidden": [8], "dropout": 0.0})
    trainer = Trainer(model, {"lr": 0.01, "epochs": 3, "patience": 2, "batch_size": 32}, "cpu")
    history = trainer.fit(ds, val_ds, epochs=3, patience=2)
    assert len(history["epoch"]) == 3
    assert all(np.isfinite(loss) for loss in history["train_loss"])
