# AI Distance Estimator

A computer vision app that measures the real-world size (in cm) of objects from a webcam feed, using an **A4 sheet of paper as a reference**. A purpose-trained **YOLOv8** model detects the paper.

## How it works

1. **Detection** — The trained YOLOv8 model (`best.pt`) locates the A4 sheet in the camera feed.
2. **Corner detection** — Inside the detected region, the sheet's four corners are refined with classic image processing (Otsu thresholding + contour analysis).
3. **Perspective correction** — The sheet is warped into a flat image with real A4 proportions (210 × 297 mm), so every pixel maps to a fixed mm value regardless of the camera angle.
4. **Measurement** — Objects placed on the sheet are detected in this flattened image; their pixel size is converted to real size using the known mm/pixel ratio.

## Setup

**Requirements:** Python 3.10+, an NVIDIA GPU with CUDA support (optional but recommended).

```bash
git clone https://github.com/krmcmr/ai-distance-estimator.git
cd ai-distance-estimator

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
```

Install PyTorch matching your GPU's CUDA version (see the notes in [requirements.txt](requirements.txt)), then install the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

Place the trained `best.pt` model in the project root, then run:

```bash
python measure.py
```

| Argument | Description | Default |
|---|---|---|
| `--model` | Model file to use | `best.pt` |
| `--camera` | Camera index | `0` |
| `--conf` | Detection confidence threshold | `0.5` |

**Keys:** `q` to quit, `s` to save a screenshot.

The app opens two windows: the camera feed with detections and measurements overlaid, and the perspective-corrected view of the paper.

## Project structure

```
ai-distance-estimator/
├── measure.py             # Main app: detection, perspective correction, measurement
├── requirements.txt       # Python dependencies
├── best.pt                # Trained YOLOv8 model (A4 detection)
└── docs/
    └── model-training.md  # How the model was trained
```

## Model and dataset

- Architecture: **YOLOv8n**
- Dataset: [A4 Detection – Roboflow Universe](https://universe.roboflow.com/greg-sun/a4-detection)
- Training steps: see [docs/model-training.md](docs/model-training.md)

## Limitations

- The object being measured must be **on the sheet**; anything hanging off the edge is not measured.
- The object needs to be **noticeably darker or more saturated** than the paper; near-white objects may not be distinguishable from it.
- For the most accurate measurement, all four corners of the sheet should be visible and the camera should look at the sheet as close to straight-down as possible.

## Contributors

- [Kerem Çamur](https://github.com/krmcmr)
- [Taylan Tuna Aktaş](https://github.com/TaylannAktas)

## License

This project is licensed under the [MIT License](LICENSE).
