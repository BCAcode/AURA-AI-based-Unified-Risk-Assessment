# AURA - AI-Based Unified Risk Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-BETA-orange.svg)]()

AURA; yapay zeka, görüntü işleme ve uydu verilerini birleştiren, teknolojik ve yenilikçi bir orman yangını risk değerlendirme yazılımıdır.

### Katkıda Bulunanlar (Contributors)
- **@abkburak34-star** (Collaborator)
- **@bogacyalcin34123** (Collaborator)

---

## TÜRKÇE

[Tam dokümantasyon için aşağı kaydırın](#installation-tr)

---

## ENGLISH  

[Scroll down for full documentation](#installation-en)

---

<a name="installation-tr"></a>

## Kurulum (Türkçe)

### Sistem Gereksinimleri

**AURA 2.0:**
- İşletim Sistemi: Apple Silicon ile macOS (M1/M2/M3/M4/M5 - Pro/Max/Ultra)
- Python: 3.8+
- RAM: 4GB min (8GB önerili)

### Bağımlılıklar

```bash
pip install mlx mlx-nn numpy matplotlib scipy pillow requests urllib3 --break-system-packages
```

### Model Dosyası

`AURA_Ignition_5_Model.npz` (2.15 MB) dosyasını Desktop, Documents veya Downloads klasörüne koyun.

### Çalıştırma

```bash
python3 "AURA 2.0-ML | AI-based Unified Risk Assessment.py"
```

Detaylı dokümantasyon: Aşağıya kaydırın

---

<a name="installation-en"></a>

## Installation (English)

### System Requirements

**AURA 2.0:**
- OS: macOS with Apple Silicon (M1/M2/M3/M4/M5 - Pro/Max/Ultra)  
- Python: 3.8+
- RAM: 4GB min (8GB recommended)

### Dependencies

```bash
pip install mlx mlx-nn numpy matplotlib scipy pillow requests urllib3 --break-system-packages
```

### Model File

Place `AURA_Ignition_5_Model.npz` (2.15 MB) in Desktop, Documents, or Downloads folder.

### Run

```bash
python3 "AURA 2.0-ML | AI-based Unified Risk Assessment.py"
```

Full documentation: Scroll down

---
---

# TÜRKÇE DOKÜMANTASYON

## BETA Yazılım - Önemli Uyarı

**BU YAZILIM BETA AŞAMASINDADIR**

**Garanti Yoktur:**
- Tahminler yanlış olabilir
- Kritik kararlar için KULLANILMAMALIDIR

**Sorumluluk Reddi:**
- Geliştiriciler hiçbir zarardan SORUMLU DEĞİLDİR
- Yangın tahminlerinden, kayıplardan veya hasarlardan sorumluluk kabul edilmez

**Kullanım:**
- Eğitim ve araştırma için
- Profesyonel sistemlere yardımcı araç olarak
- Acil durum sistemleri için DEĞİL

---

## Açık Kaynak Lisansı

**MIT Lisansı - Tamamen Özgür**

- Serbest Kullanım (ticari dahil)
- Serbest Değiştirme  
- Serbest Dağıtım

Tek gereklilik: Telif hakkı bildirimini koruyun.

---

## Özellikler

### Versiyon 2.0
- 6 katmanlı sinir ağı (564K parametre)
- Gerçek zamanlı uydu verisi (ArcGIS World Imagery)
- Canlı hava durumu (Open-Meteo API)
- Risk ısı haritaları
- 7 ağaç türü desteği
- Apple Silicon optimize (MLX)

---

## Model Eğitimi

**Veri Kaynakları:**
- NASA FIRMS VIIRS S-NPP (2024 verileri ile ana eğitim)
- NASA FIRMS (2022-2023 verileri ile fine-tuning)
- Tüm dünya NASA verileri
- AURA 1.0 simülasyon sonuçları

**Performans:**
- R²: 0.700 (test seti)
- RMSE: 9.66
- Kalite Skoru: 9/10

---

## Kullanım

1. Koordinatları girin (enlem/boylam).
2. Modu seçin (AUTO/MANUAL).
   - **AUTO Mod:** Tarih bilgisini girin (Gün/Ay/Yıl). Çevresel veriler (sıcaklık, rüzgar vb.) otomatik çekilir.
   - **MANUAL Mod:** Çevresel parametreleri slider ile elle ayarlayın.
3. SYNC butonuna basarak verileri çekin.
4. Risk haritasını görüntüleyin.

**Risk Seviyeleri:**
- Yeşil: Düşük (< 30%)
- Sarı: Orta (30-60%)
- Turuncu: Yüksek (60-85%)
- Kırmızı: Aşırı (> 85%)

---

## Lisans

MIT Lisansı - Tamamen açık kaynak

```
Copyright (c) 2026 BCAcode

İzin verilir: Kullanım, değiştirme, dağıtım, satış
Gereklilik: Telif hakkı bildirimini koruyun
Garanti: YOK
Sorumluluk: YOK
```

---

## Atıf

```text
BCAcode. (2026). AURA: Yangın Risk Tahmin Sistemi (Versiyon 2.0) [Bilgisayar Yazılımı]. https://github.com/BCAcode/AURA-AI-based-Unified-Risk-Assessment
```

---

## İletişim

**GitHub Issues:** Hatalar ve özellik istekleri  
**GitHub Discussions:** Soru-cevap

---
---

# ENGLISH DOCUMENTATION

## BETA Software - Important Notice

**THIS SOFTWARE IS IN BETA**

**No Warranty:**
- Predictions may be wrong
- DO NOT use for critical decisions

**No Liability:**
- Developers accept NO responsibility
- Not liable for fire predictions, losses, or damages

**Use:**
- For education and research
- As supplementary tool to professional systems  
- NOT for emergency response systems

---

## Open Source License

**MIT License - Completely Free**

- Free to Use (including commercial)
- Free to Modify
- Free to Distribute

Only requirement: Keep copyright notice.

---

## Features

### Version 2.0
- 6-layer neural network (564K parameters)
- Real-time satellite data (ArcGIS World Imagery)
- Live weather (Open-Meteo API)
- Risk heatmaps
- 7 tree species support
- Apple Silicon optimized (MLX)

---

## Model Training

**Data Sources:**
- NASA FIRMS VIIRS S-NPP (Main training with 2024 data)
- NASA FIRMS (Fine-tuning with 2022-2023 data)
- Global NASA data
- AURA 1.0 simulation results

**Performance:**
- R²: 0.700 (test set)
- RMSE: 9.66
- Quality Score: 9/10

---

## Usage

1. Enter coordinates (lat/lon).
2. Select mode (AUTO/MANUAL).
   - **AUTO Mode:** Enter the date (Day/Month/Year). Environmental data (temp, wind, etc.) is fetched automatically.
   - **MANUAL Mode:** Adjust environmental parameters manually using sliders.
3. Click SYNC to fetch data.
4. View risk heatmap.

**Risk Levels:**
- Green: Low (< 30%)
- Yellow: Moderate (30-60%)
- Orange: High (60-85%)
- Red: Extreme (> 85%)

---

## License

MIT License - Completely open source

```
Copyright (c) 2026 BCAcode

Permission: Use, modify, distribute, sell
Requirement: Keep copyright notice
Warranty: NONE
Liability: NONE
```

---

## Citation

```text
BCAcode. (2026). AURA: Wildfire Risk Prediction System (Version 2.0) [Computer Software]. https://github.com/BCAcode/AURA-AI-based-Unified-Risk-Assessment
```

---

## Contact

**GitHub Issues:** Bugs and feature requests
**GitHub Discussions:** Q&A

---

<p align="center">
<strong>Orman yangını önleme için / For wildfire prevention</strong>
<br>
Contributors: @abkburak34-star, @bogacyalcin34123
</p>
