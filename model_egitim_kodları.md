### 1. Kütüphane Kurulumu

```
!pip install ultralytics
import ultralytics
ultralytics.checks()
```

### 2. Veri Setini Ortama Çekme

```
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="SENIN_API_ANAHTARIN")
project = rf.workspace("greg-sun").project("a4-detection")
version = project.version(1)
dataset = version.download("yolov8")
```


### 3. Model Eğitimi

```
!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=50 imgsz=640
```


### 4. Sonuçları Repoya Yükleme

Eğitim bittiğinde sol taraftaki dosya (klasör) ikonuna tıklayarak
runs/detect/train/weights/ klasörü içindeki best.pt dosyasına sağ tıklayıp bilgisayara indirilecek.
Sonra bu best.pt dosyasını GitHub repomuza yüklenecek.
