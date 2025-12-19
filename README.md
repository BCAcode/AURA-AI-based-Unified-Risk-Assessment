# AURA | AI-based Unified Risk Assessment
**Yapay Zeka Destekli Yangın Simülasyon ve Tahmin Sistemi (BETA)**

---

## ⚠ Beta Durumu

**AURA şu anda BETA sürümündedir.**

- Algoritmalar aktif olarak geliştirilmektedir  
- Model parametreleri ve ağırlıklar değişebilir  
- Performans ve doğruluk nihai değildir  
- Kod yapısı kararlı (stable) API olarak kabul edilmemelidir  

Yazılım; **test, araştırma, eğitim ve deneysel kullanım** için uygundur.

---

## 1. Genel Tanım

**AURA (AI-based Unified Risk Assessment)**, orman yangınlarının yayılımını **fizik tabanlı modeller** ve **yapay zeka destekli risk değerlendirme yaklaşımları** ile simüle eden, analiz eden ve yangının kontrol edilebilirliğini tahmin eden gelişmiş bir simülasyon yazılımıdır.

Sistem aşağıdaki parametreleri eş zamanlı değerlendirir:

- Rüzgar (yön + hız)  
- Ortam sıcaklığı  
- Nem oranı  
- Kuraklık seviyesi  
- Ağaç türü ve yanıcılık katsayıları  
- Yangın geometrisinin fraktal karmaşıklığı  

**Dosya adı:**  
AURA | AI-based Unified Risk Assessment.py

---

## 2. Amaç ve Kapsam

- Orman yangınlarının yayılım davranışını modellemek  
- Çevresel parametrelerin yangın üzerindeki etkisini analiz etmek  
- Yangın karmaşıklığı ile kontrol edilebilirlik arasındaki ilişkiyi incelemek  
- Eğitim, araştırma ve senaryo bazlı risk değerlendirmelerine temel oluşturmak  

> BETA sürümü nedeniyle **operasyonel afet yönetimi** için önerilmez.

---

## 3. Temel Özellikler

- Fizik tabanlı **hücresel otomata** yangın modeli  
- Yapay zeka destekli **kontrol edilebilirlik ve risk skoru**  
- **Fraktal boyut analizi** (Box-Counting yöntemi)  
- Çoklu ağaç türleri ve tür bazlı yanıcılık katsayıları  
- Rüzgar yönü ve hızına duyarlı **vektörel yayılım**  
- Gerçek zamanlı etkileşimli grafik arayüz  
- Anlık parametre değişimi ve senaryo karşılaştırması  

---

## 4. Python Uyumluluğu

- **Python 3.8+** desteklenir  
- Python 2 **desteklenmez**

Çalıştırma:

- Linux / macOS  
  `python3 "AURA | AI-based Unified Risk Assessment.py"`

- Windows  
  `python "AURA | AI-based Unified Risk Assessment.py"`

---

## 5. Kullanılan Teknolojiler

- Python 3  
- NumPy  
- Matplotlib (GUI + görselleştirme)

Harici GUI framework’ü, oyun motoru veya ek yapay zeka kütüphanesi kullanılmamıştır.

---

## 6. Simülasyon Mimarisi

### 6.1 Hücresel Alan Modeli

| Kod | Durum   |
|----:|---------|
| 0   | EMPTY   |
| 1   | TREE    |
| 2   | BURNING |

---

### 6.2 Ağaç Türleri

- Çam  
- Meşe  
- Çınar  
- Kayın  
- Kavak  
- Ardıç  
- Zeytin  
- Akasya  
- Maki  
- Çalı  
- Karaışık  

Her tür için farklı **yangın yayılım katsayısı** tanımlıdır.

---

### 6.3 Yangın Yayılım Algoritması

Bir hücrenin yanma olasılığı şu bileşenlerle hesaplanır:

- Komşu yanan hücre sayısı  
- Rüzgar yönü ile hizalanma  
- Rüzgar şiddeti  
- Ortam sıcaklığı  
- Nem oranı  
- Kuraklık seviyesi  
- Yakıt miktarı  
- Ağaç türü yayılım katsayısı  
- Fraktal karmaşıklık  

Model **olasılıksal** ve **zamana bağlıdır**.

---

### 6.4 Fraktal Boyut Hesabı

Yangın geometrisi **Box-Counting** yöntemi ile analiz edilir.

Fraktal boyut:

- Yangının düzensizliğini  
- Yayılımın kaotik yapısını  

nicel olarak ifade eder.

---

### 6.5 Kontrol Edilebilirlik Skoru

- 0–1 aralığında normalize edilir  
- Yüksek skor → daha kontrol edilebilir  
- Düşük skor → hızlı ve karmaşık yayılım  

Skor çevresel faktörler ve yangın alanı büyüklüğü ile **dinamik** olarak değişir.

---

## 7. Grafiksel Kullanıcı Arayüzü

### Görselleştirme

- Siyah: Boş alan  
- Yeşil tonları: Ağaçlar  
- Turuncu / Kırmızı: Aktif yangın  

### Kontroller

- Rüzgar yönü (dairesel pusula)  
- Rüzgar hızı (km/h)  
- Nem (%)  
- Sıcaklık (°C)  
- Kuraklık (%)  

### Butonlar

- BAŞLA  
- DUR  
- SIFIRLA  
- ÇIKIŞ  

---

## 8. Minimum Gereklilikler

### Yazılım
- Python 3.8 veya üzeri  
- Python 2 desteklenmez  

### Python Kütüphaneleri
- NumPy  
- Matplotlib  

### Donanım
- En az **2 GB RAM**  
- Çok çekirdekli CPU önerilir  
- GPU gerekmez  

### Ekran
- Minimum **1280×720**  
- 16:9 önerilir  

### İşletim Sistemi
- Windows 10 / 11  
- Linux (Ubuntu 20.04+ önerilir)  
- macOS 11+ (Apple Silicon desteklenir)  

---

## 9. Kurulum

Python yüklüyse:

`pip install numpy matplotlib`

Python3 kullanılıyorsa:

`pip3 install numpy matplotlib`

---

## 10. Kullanım Alanları

- Orman yangını davranış analizi  
- Risk senaryosu simülasyonları  
- Akademik ve teknik araştırmalar  
- Eğitim ve görsel anlatım  
- Karar destek sistemleri için prototip  

---

## 11. Sınırlamalar

- BETA sürümüdür  
- Gerçek zamanlı afet yönetimi için uygun değildir  
- Gerçek dünyanın sadeleştirilmiş bir temsilidir  
- Sonuçlar rastlantısal süreçlere bağlıdır  

---

## 12. Lisans

Açık lisans:

- Serbest kullanım  
- Serbest değiştirme  
- Serbest dağıtım  

Kaynak gösterimi önerilir ancak zorunlu değildir.
---

## 3. Temel Özellikler

- Fizik tabanlı hücresel otomata yangın modeli  
- Yapay zeka destekli kontrol edilebilirlik ve risk skoru  
- Fraktal boyut analizi (Box-Counting yöntemi)  
- Çoklu ağaç türleri ve tür bazlı yanıcılık katsayıları  
- Rüzgar yönü ve hızına duyarlı vektörel yayılım  
- Gerçek zamanlı etkileşimli grafik arayüz  
- Anlık parametre değişimi ve senaryo karşılaştırması  

---

## 4. Python ve Python3 Uyumluluğu

AURA, **Python 3** için geliştirilmiştir.

### Desteklenen Sürümler
- Python 3.8  
- Python 3.9  
- Python 3.10+  

> Python 2 **desteklenmez**.

### Komut Farkı (Sisteme Göre)


python3 "AURA | AI-based Unified Risk Assessment.py"


python "AURA | AI-based Unified Risk Assessment.py"


⸻

5. Kullanılan Teknolojiler
	•	Python 3
	•	NumPy
	•	Matplotlib (GUI + görselleştirme)

Harici GUI framework’ü, oyun motoru veya yapay zeka kütüphanesi kullanılmamıştır.

⸻

6. Simülasyon Mimarisi

6.1 Hücresel Alan Modeli

Simülasyon ortamı iki boyutlu bir grid (ızgara) yapısıdır.

Kod	Durum
0	EMPTY
1	TREE
2	BURNING


⸻

6.2 Ağaç Türleri
	•	Çam
	•	Meşe
	•	Çınar
	•	Kayın
	•	Kavak
	•	Ardıç
	•	Zeytin
	•	Akasya
	•	Maki
	•	Çalı
	•	Karaışık

Her türün yangın yayılım katsayısı farklıdır.

⸻

6.3 Yangın Yayılım Algoritması

Bir hücrenin yanma olasılığı şu bileşenlerle hesaplanır:
	•	Komşu yanan hücre sayısı
	•	Rüzgar yönü ile hizalanma
	•	Rüzgar şiddeti
	•	Ortam sıcaklığı
	•	Nem oranı
	•	Kuraklık seviyesi
	•	Yakıt miktarı
	•	Ağaç türü yayılım katsayısı
	•	Fraktal karmaşıklık

Model olasılıksal ve zamana bağlıdır.

⸻

6.4 Fraktal Boyut Hesabı

Yangın geometrisi Box-Counting yöntemi ile analiz edilir.

Fraktal boyut:
	•	Yangının düzensizliğini
	•	Yayılımın kaotik yapısını

sayısal olarak ifade eder.

⸻

6.5 Kontrol Edilebilirlik Skoru

0–1 aralığında normalize edilmiş bir metrik üretilir.
	•	Yüksek skor → daha kontrol edilebilir
	•	Düşük skor → hızlı ve karmaşık yayılım

Skor; çevresel faktörler ve yangın alanı büyüklüğüyle dinamik olarak değişir.

⸻

7. Grafiksel Kullanıcı Arayüzü

Görselleştirme
	•	Siyah: Boş alan
	•	Yeşil tonları: Ağaçlar
	•	Turuncu / Kırmızı: Aktif yangın

Kontroller
	•	Rüzgar yönü (dairesel pusula)
	•	Rüzgar hızı (km/h)
	•	Nem (%)
	•	Sıcaklık (°C)
	•	Kuraklık (%)

Butonlar
	•	BAŞLA
	•	DUR
	•	SIFIRLA
	•	ÇIKIŞ

⸻

## Minimum Gereklilikler

### Yazılım
- **Python 3.8 veya üzeri**
- Python 2 **desteklenmez**

### Python Kütüphaneleri
- **NumPy** (zorunlu)
- **Matplotlib** (zorunlu)

> Not: Bu kütüphaneler Python ile birlikte gelmez, ayrıca kurulmalıdır.

### Donanım
- **En az 2 GB RAM**  
  - Gerçek kullanımda simülasyon sırasında ~300–400 MB RAM tüketimi gözlemlenebilir
- **Çok çekirdekli CPU önerilir**  
  - NumPy, hesaplamalar sırasında tüm çekirdekleri kullanabilir
- GPU gereksinimi yok (CPU yeterlidir)

### Ekran
- Minimum **1280×720** çözünürlük
- 16:9 en-boy oranı önerilir

### İşletim Sistemi
- Windows 10 / 11  
- Linux (Ubuntu 20.04+ önerilir)  
- macOS 11+ (Apple Silicon desteklenir)

### Performans Notu
AURA, gerçek zamanlı simülasyon, fraktal analiz ve grafik arayüz kullandığı için
yüksek CPU ve RAM kullanımı gösterebilir. Bu durum **beklenen ve normaldir**.

8. Kurulum

> python yüklü ise:

  pip install numpy matplotlib

> python3 yüklü ise:

  pip3 install numpy matplotlib

⸻

9. Kullanım Alanları
	•	Orman yangını davranış analizi
	•	Risk senaryosu simülasyonları
	•	Akademik ve teknik araştırmalar
	•	Eğitim ve görsel anlatım
	•	Karar destek sistemleri için prototip model

⸻

10. Sınırlamalar ve Uyarılar
	•	BETA sürümüdür
	•	Gerçek zamanlı afet yönetimi için uygun değildir
	•	Gerçek dünya verilerinin sadeleştirilmiş bir temsilidir
	•	Sonuçlar rastlantısal süreçlere bağlıdır

⸻

11. Lisans

Açık lisans – lisans kısıtlaması yoktur.
	•	Serbest kullanım
	•	Serbest değiştirme
	•	Serbest dağıtım
	•	Akademik ve bireysel kullanımda sınırlama yok

Kaynak gösterimi önerilir ancak zorunlu değildir.

⸻


