# ML Training Pipeline

Standards for training and fine-tuning YOLO models in SafeCross.

## Fine-Tuning Hyperparameters

Use these defaults for YOLOv8m fine-tuning (traffic light model):

```python
results = model.train(
    data=DATA_YAML,       # always relative path using os.path.dirname(__file__)
    epochs=80,
    imgsz=640,
    batch=16,
    name='yolov8m_traffic_light',
    patience=20,          # early stopping if no improvement for 20 epochs
    device=0,             # 0 = first GPU; use 'cpu' if no GPU
)
```

For lighter models (YOLOv8n, crosswalks/vehicles): `epochs=50, batch=32, imgsz=512`.

## Cross-Validation Setup

Always use 5-fold when validating model robustness:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

`random_state=42` is the project-wide seed. Never change it between experiments
so results are reproducible.

## Relative Paths — Always

All paths must be relative to the script file, never absolute or CWD-dependent:

```python
# Correct
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(SCRIPT_DIR, "yolov8m.pt")
data_yaml  = os.path.join(SCRIPT_DIR, "TrafficLight.yaml")

# Wrong — hardcoded absolute path
data = 'C:/Users/Diana Laura/Desktop/ML/TrafficLight.yaml'
```

## Saving Models

Always save with `model.save()` after training:

```python
model.save(os.path.join(SCRIPT_DIR, "yolov8m_TrafficLight.pt"))
```

The best checkpoint is automatically saved to `runs/detect/<name>/weights/best.pt`.
Copy `best.pt` to `weigths/` in the repo root after evaluating.

## Entry Point Guard

Always wrap training calls in `if __name__ == '__main__'` with `freeze_support()`:

```python
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    results = model.train(...)
```

## Dataset Config (YAML)

Dataset config files always live next to the training script:

```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: red light
  1: yellow light
  2: green light
```
