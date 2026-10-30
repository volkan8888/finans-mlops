"""Adim 9 (FAZ 7): Streamlit dashboard - finans-mlops panosu.

Calistirilmasi:
    .venv\\Scripts\\streamlit run app.py

Sayfalar:
1) Skorlar     : guncel gun skorlari (top/bottom 5); yoksa kosma talimati gosterilir.
2) Backtest    : equity_curve.csv (long_short_net / long_only_net / SPY) + metrik kartlari.
3) Monitoring  : PSI tablo+bar, uyari listesi, IC ozeti.
4) Ileri Test  : canli tahminlerin gerceklesmesi (freeze.json + forward_report.json).
5) MLflow Runs : experiment run'lari + production alias bilgi karti.

Girdiler: reports/* ve data/processed/* uretimleri; pano bunlari okur,
KENDI BASTINA hicbir sey uretmeZ (saf okuma; bayat veri uyari ile belirtilir).
"""

from __future__ import annotations

import math
import pathlib

import pandas as pd
import streamlit as st

from dashboard.prepare import (
    forward_quantile_frame,
    load_equity,
    load_forward,
    load_freeze,
    load_monitoring,
    load_scores,
    load_summary,
    mlflow_runs_table,
    production_alias_info,
    psi_table,
    top_bottom,
)

st.set_page_config(page_title="finans-mlops", page_icon=":chart:", layout="wide")

PROJECT_ROOT = pathlib.Path(__file__).parent


def fmt_percent(value) -> str:
    """0.1234 -> '12.3%' (None/NaN/pasat guvenli)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isnan(number):
        return "-"
    return f"{100 * number:.1f}%"


def page_scores() -> None:
    """Guncel skorlar: top/bottom tablolar; yoksa cikarma talimati."""
    st.header("Guncel skorlar")
    frame = load_scores()
    if frame.empty:
        st.warning("latest_scores.parquet yok. Gunluk aksini calistirin:\n"
                   "`python scripts/run_daily.py --mode daily`")
        return
    date = pd.Timestamp(frame["date"].max()).date()
    st.caption(f"Skor gunu: {date} | {len(frame)} ticker")
    top, bottom = top_bottom(frame, n=5)
    left, right = st.columns(2)
    with left:
        st.subheader("Ust 5 (uzun aday)")
        st.dataframe(top[["ticker", "score"]], use_container_width=True)
    with right:
        st.subheader("Alt 5 (kisa aday)")
        st.dataframe(bottom[["ticker", "score"]], use_container_width=True)
    st.bar_chart(frame.head(10).set_index("ticker")["score"])


def page_backtest() -> None:
    """Backtest raporu: metrik kartlari + equity grafigi."""
    st.header("Backtest (holdout)")
    summary = load_summary()
    if not summary:
        st.warning("backtest_summary.json yok. `python -m evaluation.run` calistirin.")
        return
    for mode in ("long_short", "long_only"):
        block = summary.get(mode, {})
        net = block.get("metrics_net", {})
        gross = block.get("metrics_gross", {})
        compare = block.get("vs_benchmark_gross", {})
        beta = compare.get("beta")
        left, mid, right = st.columns(3)
        left.metric(f"{mode} | net toplam", fmt_percent(net.get("total_return")), fmt_percent(net.get("sharpe")))
        mid.metric(f"{mode} | brut toplam", fmt_percent(gross.get("total_return")), fmt_percent(gross.get("sharpe")))
        right.metric(
            f"{mode} | SPY'a karsi (brut) alfa/yil",
            fmt_percent(compare.get("alpha_annualized")),
            f"beta {float(beta):.2f}" if beta is not None else "-",
        )
    st.caption("Maliyet 10 bp tek-yon; turnover ve lag konvansiyonlari icin docs/step4_backtest.md bakiniz.")
    equity = load_equity()
    if equity.empty:
        st.info("equity_curve.csv yok.")
        return
    st.line_chart(equity.set_index("date"))


def page_monitoring() -> None:
    """Monitoring raporu: PSI bar + uyari listesi + IC ozeti."""
    st.header("Monitoring")
    report = load_monitoring()
    if not report:
        st.warning("monitoring_report.json yok. `python -m monitoring.run` calistirin.")
        return
    table = psi_table(report)
    st.subheader("Feature drift (PSI)")
    st.bar_chart(table.set_index("feature")["psi"])
    st.dataframe(table, use_container_width=True)
    alerts = report.get("alerts", [])
    if alerts:
        st.error("Uyarilar: " + ", ".join(alerts))
    else:
        st.success("Uyari yok.")
    performance = report.get("performance", {})
    left, mid, right = st.columns(3)
    left.metric("20-gun kayan IC", performance.get("rolling_ic_20d", "-"))
    overall = performance.get("overall", {})
    mid.metric("Overall mean IC", overall.get("mean_ic", "-"), f"t={overall.get('t_stat', '-')}")
    right.metric("t (Newey-West lag=4)", overall.get("t_nw", "-"))


def page_forward() -> None:
    """Ileri test: freeze imzasi + gerceklesen IC/kuantiller (veri yoksa bos durum)."""
    st.header("Ileri test")
    freeze = load_freeze()
    forward = load_forward()

    if freeze:
        model = freeze.get("model", {})
        sha_files = model.get("sha256", {}).get("model_files", {})
        sha_short = next(iter(sha_files.values()), "")[:8] or "-"
        st.caption(
            "Dondurulmus imza: "
            f"model={model.get('name')} | kaynak={model.get('source_kind')} "
            f"| versiyon={model.get('registry_version') or '-'} | sha8={sha_short} "
            f"| freeze={str(freeze.get('frozen_at'))[:20]}"
        )
    else:
        st.info(
            "freeze.json yok. Once `python scripts/freeze_model.py` ile prod modeli "
            "imzala (ileri test butunlugu icin); sonra gunluk akis tahmin biriktirir."
        )

    if not forward:
        st.info(
            "Ileri test raporu yok. `python -m monitoring.forward` calistirin "
            "(veri: data/predictions + data/raw/prices.parquet)."
        )
        return
    status = forward.get("status")
    if status == "bos":
        st.warning("Ileri test verisi yok: tahmin dosyasi veya fiyat deposu bulunamadi.")
        return
    if status == "tum_beklemede":
        st.info(
            f"Tahminler var ama {forward.get('horizon')}-gunluk pencere henuz dolmadi "
            f"({forward.get('n_unrealized')} tahmin bekliyor; atlanip sayildi). "
            "Yeni tahmin biriktikce bu sayfa kendiliginden dolacak."
        )
        return

    overall = forward.get("overall", {})
    left, mid, right, far = st.columns(4)
    left.metric("Gerceklesen gun", overall.get("n_days", "-"))
    mid.metric("Mean IC (gerceklesen)", overall.get("mean_ic", "-"))
    right.metric("t (Newey-West)", overall.get("t_nw", "-"), f"ham t={overall.get('t_stat', '-')}")
    far.metric(
        "Unrealized (bekleyen)",
        forward.get("n_unrealized", 0),
        f"{forward.get('n_realized', 0)} gerceklesti",
    )
    spread_evs = forward.get("ust_kantil_eksi_evren") or {}
    spread_ust_alt = forward.get("ust_eksi_alt_kantil") or {}
    payi = forward.get("unrealized_payi") or {}
    row_left, row_mid, row_right = st.columns(3)
    row_left.metric(
        "Ust kantil - evren (gunluk ort)",
        fmt_percent(spread_evs.get("ortalama")),
        f"t_nw={spread_evs.get('t_nw') if spread_evs.get('t_nw') is not None else '-'}",
    )
    row_mid.metric(
        "Ust kantil - alt kantil (gunluk ort)",
        fmt_percent(spread_ust_alt.get("ortalama")),
        f"t_nw={spread_ust_alt.get('t_nw') if spread_ust_alt.get('t_nw') is not None else '-'}",
    )
    row_right.metric(
        "Unrealized payi",
        fmt_percent(payi.get("oran")),
        f"{payi.get('bekleyen', 0)} / {payi.get('toplam', 0)} tahmin bekliyor",
    )
    table = forward_quantile_frame(forward)
    if not table.empty:
        st.subheader("Kuantil gerceklesmesi (gunluk ort. getiri)")
        st.bar_chart(table.set_index("quantile")[["mean_return"]])
        st.dataframe(table, use_container_width=True)
        st.caption(
            "vs_benchmark kolonu, ayni pencerede SPY getirisine gore fark "
            "(ust kuantil satiri sinyalin benchmark'a ustunlugunu gostermesi beklenen yer)."
        )
    alerts = forward.get("alerts", [])
    if alerts:
        st.warning("Uyarilar: " + ", ".join(alerts))
    else:
        st.success("Uyari yok.")


def page_mlflow() -> None:
    """MLflow experiment ozeti + production alias."""
    st.header("MLflow runs")
    st.caption("Production alias: " + production_alias_info())
    table = mlflow_runs_table()
    if table.empty:
        st.info("mlflow.db/experiment bulunamadi (once bir egitim kosusu MLflow'a kayit olmali).")
        return
    st.dataframe(table, use_container_width=True)


def main() -> None:
    """Panoyu sekmelerle kurar."""
    st.sidebar.title("finans-mlops")
    page = st.sidebar.radio("Sayfa", ["Skorlar", "Backtest", "Monitoring", "Ileri Test", "MLflow Runs"])
    if page == "Skorlar":
        page_scores()
    elif page == "Backtest":
        page_backtest()
    elif page == "Monitoring":
        page_monitoring()
    elif page == "Ileri Test":
        page_forward()
    else:
        page_mlflow()


if __name__ == "__main__":
    main()
