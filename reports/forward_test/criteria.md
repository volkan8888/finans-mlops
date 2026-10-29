# Ileri test kabul kriterleri (SABLON)

Dondurulen model imzasi: `reports/forward_test/freeze.json`
(dondurme komutu: `python scripts/freeze_model.py`)

Bu dosya SABLONDIR: esikleri proje sahibi doldurur. Ileri test donemi
bitince "olcum" kolonu `reports/forward_test/forward_report.json`
icerigiyle doldurulur.

## Kriter tablosu (esikler doldurulacak)

| # | Kriter | Esik (doldurulacak) | Donem sonu olcum | Durum |
|---|---|---|---|---|
| 1 | Gerceklesen mean IC | ? (ornek: >= 0.005) | ? | ? |
| 2 | t (Newey-West, lag = label_horizon-1) | ? (ornek: >= 1.0) | ? | ? |
| 3 | Ust kuantil gunluk getiri - SPY farki | ? (ornek: > 0) | ? | ? |
| 4 | Unrealized tahmin payi | ? (ornek: < %25) | ? | ? |
| 5 | Monitoring uyari sayisi (drift + rolling IC) | ? | ? | ? |

## Kurallar

- Esikler donem ICINDE DEGISTIRILMEZ; degisiklik daima yeni donemin
  basindan itibaren gecerli olur ve oz gerek buraya not edilir.
- Uretim karari icin tum kriterlerin birlikte gecerli olmasi aranir;
  istisna durumunda proje sahibinin acik karari gerekir.
- Yatirim tavsiyesi degildir.
