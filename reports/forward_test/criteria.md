# Ileri test kabul kriterleri

Dondurulen model imzasi: `reports/forward_test/freeze.json`
Bu esikler ilk gerceklesen tahminden ONCE yazilmistir (commit tarihi kanittir).

## Degerlendirme tarihleri

- Erken kontrol (yaklasik 21 islem gunu): SADECE boru hatti sagligi
  (her gun tahmin uretildi mi, veri bozuk mu). Performans hakkinda yargi YOK.
- Ilk degerlendirme (63 islem gunu): asagidaki esikler uygulanir, karar
  "devam / durdur" dir, "basarili" ilan edilmez.
- Asil degerlendirme (126+ islem gunu): son karar.

## Kriter tablosu

| # | Kriter | Esik | Donem sonu olcum | Durum |
|---|---|---|---|---|
| 1 | Gerceklesen mean IC | >= 0.005 | ? | ? |
| 2 | t (Newey-West, lag = label_horizon-1) | >= 1.0 (63 gun), >= 1.5 (126 gun) | ? | ? |
| 3 | Ust kantil getirisi - evren esit agirlikli ortalama | > 0 | ? | ? |
| 4 | Ust kantil - alt kantil getiri farki | > 0 | ? | ? |
| 5 | Unrealized tahmin payi | < %25 | ? | ? |
| 6 | Monitoring: PSI alarm sayisi (PSI >= 0.2) | 0 | ? | ? |
| 7 | Monitoring: rolling IC alarm gunu orani | < %50 | ? | ? |

SPY farki ek bilgi olarak raporlanir, kabul kriteri degildir
(beta ve survivorship bias nedeniyle adil kiyas degil).

## Karar kurali

- Kriter 1-4'un TAMAMI saglanirsa: ileri test devam eder, model "umut verici"
  sayilir (kanitlanmis degil).
- Kriter 1 veya 2 basarisizsa: model uretim icin KULLANILMAZ, sonuc README'ye
  durustce yazilir.
- Gecici ayarlama yok: kotu giderse modeli ayarlayip "duzeldi" denmez; yeni
  model yeni bir ileri test donemi baslatir.

## Kurallar

- Esikler donem icinde DEGISTIRILMEZ; degisiklik yalnizca yeni donemin
  basindan gecerlidir ve gerekcesi buraya not edilir.
- Yatirim tavsiyesi degildir.