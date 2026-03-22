# Tech Stack

## Core ML Framework

**Ultralytics YOLO v8** (`ultralytics>=8.0.0`)
- Model architecture: YOLOv8m (traffic lights), YOLOv8n (crosswalks, vehicles)
- Training approach: fine-tuning pre-trained COCO weights
- Weight format: `.pt` (PyTorch) — tracked in git under `weigths/`
- Export formats: `.onnx`, `.torchscript` — NOT tracked (too large, generated on demand)

## Python Stack

| Package | Version | Purpose |
|---|---|---|
| Python | ≥3.8 | Runtime |
| ultralytics | ≥8.0.0 | YOLO model training and inference |
| scikit-learn | ≥1.0.0 | K-Fold cross-validation |
| PyYAML | ≥6.0 | Dataset config files (`.yaml`) |
| numpy | ≥1.21.0 | Numerical operations |
| opencv-python | ≥4.5.0 | Image processing |
| Pillow | ≥9.0.0 | Image I/O |

## Hardware Requirements

**Training:**
- GPU NVIDIA with CUDA (recommended: ≥8 GB VRAM)
- CPU fallback available but slow (`device='cpu'`)

**Inference:**
- CPU: feasible for single-image inference (<500ms typical)
- GPU: real-time capable

## Dataset Format

**Source annotations:** Pascal VOC XML (`<bndbox>` with xmin/xmax/ymin/ymax)

**YOLO annotations:** normalized `.txt` files
- Format: `class_id x_center y_center width height` (all 0–1 normalized)
- Conversion tool: `trafficlight_detection/trafficlight.py`

**Dataset structure:**
```
dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── annotations/xml/
```

**Dataset config:** `TrafficLight.yaml` (YOLO format, `nc: 3`)

## Project Package Structure

```
SafeCross/
├── safecross/          ← importable Python package (business logic)
│   ├── __init__.py
│   └── decide.py       ← can_cross(), decide_verbose()
├── trafficlight_detection/   ← ML training pipeline (not a package)
│   ├── train.py
│   ├── kfold.py
│   ├── test.py
│   ├── trafficlight.py
│   └── TrafficLight.yaml
├── weigths/            ← trained model weights (in git)
│   ├── crosswalks.pt
│   ├── lights.pt
│   └── persons_cars.pt
└── .agent-os/          ← Agent OS product + standards docs
```

## Key Design Decisions

**1. Three independent models, not one multi-task model**
Each model specializes in one detection task. This simplifies training, allows
independent fine-tuning, and makes the decision logic transparent.

**2. Weights tracked in git**
The `.pt` files (~50 MB each) are committed because they are the primary
artifact of the research project. Large export formats (`.onnx`, `.torchscript`)
are excluded via `.gitignore`.

**3. Relative paths everywhere**
All file paths use `os.path.dirname(os.path.abspath(__file__))` as the base —
never hardcoded absolute paths. This makes the project portable across machines.

**4. Lazy model loading in safecross/decide.py**
Models are loaded inside the function call, not at module import time. This
avoids loading all 3 models if only one check is needed (e.g., crosswalk check
fails immediately).
