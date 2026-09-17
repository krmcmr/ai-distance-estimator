# AI Distance Estimator

Webcam görüntüsü üzerinden, bir **A4 kağıdını referans alarak** nesnelerin gerçek boyutlarını (cm) ölçen bir bilgisayarlı görü uygulaması. Kağıdın tespiti için özel olarak eğitilmiş bir **YOLOv8** modeli kullanılır.

## Nasıl çalışır?

1. **Tespit** — Eğitilmiş YOLOv8 modeli (`best.pt`), kamera görüntüsünde A4 kağıdını bulur.
2. **Köşe bulma** — Tespit edilen alanın içinde kağıdın dört köşesi klasik görüntü işleme yöntemleriyle (Otsu eşikleme + kontur analizi) netleştirilir.
3. **Perspektif düzeltme** — Kağıt, gerçek A4 oranlarına (210 × 297 mm) sahip düz bir görüntüye dönüştürülür. Böylece kamera açısı ne olursa olsun her piksel sabit bir mm değerine karşılık gelir.
4. **Ölçüm** — Kağıdın üzerine konan nesneler bu düzleştirilmiş görüntüde tespit edilir; piksel boyutları, bilinen mm/piksel oranı kullanılarak gerçek boyuta çevrilir.

## Kurulum

**Gereksinimler:** Python 3.10+, (opsiyonel ama önerilir) CUDA destekli bir NVIDIA ekran kartı.

```bash
git clone https://github.com/krmcmr/ai-distance-estimator.git
cd ai-distance-estimator

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
```

PyTorch'u ekran kartınıza uygun CUDA sürümüyle kurun (bkz. [requirements.txt](requirements.txt) içindeki notlar), ardından geri kalan paketleri kurun:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Kullanım

Eğitilmiş `best.pt` modelini proje kök dizinine yerleştirin, ardından:

```bash
python olcum.py
```

| Argüman | Açıklama | Varsayılan |
|---|---|---|
| `--model` | Kullanılacak model dosyası | `best.pt` |
| `--kamera` | Kamera indeksi | `0` |
| `--conf` | Tespit güven eşiği | `0.5` |

**Tuşlar:** `q` çıkış, `s` ekran görüntüsü kaydet.

Uygulama iki pencere açar: kamera görüntüsü üzerinde tespit ve ölçüm sonuçları, ve kağıdın perspektifi düzeltilmiş hâli.

## Proje yapısı

```
ai-distance-estimator/
├── olcum.py              # Ana uygulama: tespit, perspektif düzeltme, ölçüm
├── requirements.txt       # Python bağımlılıkları
├── best.pt                 # Eğitilmiş YOLOv8 modeli (A4 tespiti)
└── docs/
    └── model-egitimi.md   # Modelin nasıl eğitildiğine dair adımlar
```

## Model ve veri seti

- Mimari: **YOLOv8n**
- Veri seti: [A4 Detection – Roboflow Universe](https://universe.roboflow.com/greg-sun/a4-detection)
- Eğitim adımları için bkz. [docs/model-egitimi.md](docs/model-egitimi.md)

## Sınırlamalar

- Ölçülecek nesnenin **kağıdın üzerinde** olması gerekir; kağıdın dışına taşan kısımlar ölçüme dahil edilmez.
- Nesnenin, kağıttan belirgin şekilde **daha koyu veya renkli** olması gerekir; beyaza yakın nesneler kağıttan ayırt edilemeyebilir.
- Kağıdın dört köşesinin de görüntüde net görünmesi, en doğru ölçüm için kameranın kağıda mümkün olduğunca dik açıyla bakması gerekir.

## Katkıda Bulunanlar

- [Kerem Çamur](https://github.com/krmcmr)
- [Taylan Tuna Aktaş](https://github.com/TaylannAktas)

## Lisans

Bu proje [MIT lisansı](LICENSE) ile lisanslanmıştır.
