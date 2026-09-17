# Model Training

The `best.pt` model was trained on Google Colab following the steps below, using the [A4 Detection](https://universe.roboflow.com/greg-sun/a4-detection) dataset on top of the YOLOv8n architecture.

### 1. Install Libraries

```bash
!pip install ultralytics
```
```python
import ultralytics
ultralytics.checks()
```

### 2. Pull the Dataset

```bash
!pip install roboflow
```
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("greg-sun").project("a4-detection")
version = project.version(1)
dataset = version.download("yolov8")
```

### 3. Train the Model

```bash
!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=50 imgsz=640
```

### 4. Upload the Result to the Repo

Once training finishes, open the file browser panel on the left, navigate to
`runs/detect/train/weights/`, right-click `best.pt` and download it.
Place this file in the project root (next to [measure.py](../measure.py)) to use it.
