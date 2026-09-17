# Model Eğitimi

`best.pt` modeli, Google Colab üzerinde aşağıdaki adımlarla eğitilmiştir. Model, [A4 Detection](https://universe.roboflow.com/greg-sun/a4-detection) veri seti kullanılarak YOLOv8n mimarisi üzerine eğitilmiştir.

### 1. Kütüphane Kurulumu

```bash
!pip install ultralytics
```
```python
import ultralytics
ultralytics.checks()
```

### 2. Veri Setini Ortama Çekme

```bash
!pip install roboflow
```
```python
from roboflow import Roboflow
rf = Roboflow(api_key="SENIN_API_ANAHTARIN")
project = rf.workspace("greg-sun").project("a4-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

### 3. Model Eğitimi

```bash
!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=50 imgsz=640
```

### 4. Sonuçları Repoya Yükleme

Eğitim bittiğinde sol taraftaki dosya (klasör) ikonuna tıklayarak
`runs/detect/train/weights/` klasörü içindeki `best.pt` dosyasına sağ tıklayıp bilgisayara indirilir.
Bu dosya projenin kök dizinine (`olcum.py` ile aynı klasöre) konularak [olcum.py](../olcum.py) ile kullanılabilir hâle gelir.
