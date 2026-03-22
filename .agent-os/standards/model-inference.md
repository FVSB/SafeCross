# Model Inference

Standards for running inference with trained YOLO models.

## Loading Models

Always load from `weigths/` relative to the repo root, resolved at runtime:

```python
import os

# From safecross/ module (2 levels up from __file__)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(REPO_ROOT, "weigths")

model = YOLO(os.path.join(WEIGHTS_DIR, "lights.pt"))
```

From a script in `trafficlight_detection/` (1 level up):
```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model = YOLO(os.path.join(SCRIPT_DIR, "best.pt"))
```

## Confidence Threshold

Default confidence: **0.75**. Do not lower below 0.5 — increases false positives.

```python
results = model(img_path, conf=0.75)
```

## Batch vs. Single Image Inference

For test scripts (batch evaluation):
```python
results = model.predict(
    source=TEST_DIR,
    save=True,      # save annotated images
    save_txt=True,  # save .txt detections
)
```

For the decision module (single image, programmatic):
```python
results = model(img_path, conf=conf)
boxes  = results[0].boxes.xyxy.tolist()   # list of [x1,y1,x2,y2]
labels = results[0].names                  # {id: class_name}
cls_id = int(results[0].boxes.cls[0])
```

## Model Weight Files

The three production models live in `weigths/` (note: intentional typo from
original project, kept for consistency):

| File | Task | Architecture |
|---|---|---|
| `weigths/lights.pt` | Traffic light color | YOLOv8m |
| `weigths/crosswalks.pt` | Crosswalk presence | YOLOv8n |
| `weigths/persons_cars.pt` | Vehicle detection | YOLOv8n |

Always validate existence before loading:
```python
if not os.path.exists(weight_path):
    raise FileNotFoundError(f"Weight not found: {weight_path}")
```

## Lazy Loading in the Decision Module

Do NOT load models at module import. Load inside the function:

```python
# Correct — lazy loading
def can_cross(img_path, conf=0.75):
    model = YOLO(LIGHTS_WEIGHT)
    ...

# Wrong — eager loading at import
model = YOLO(LIGHTS_WEIGHT)  # don't do this at module level
```

This allows early exit (e.g., no crosswalk detected) without loading all models.
