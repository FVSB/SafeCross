# Project Structure

Standards for how the SafeCross repository is organized.

## Top-Level Layout

```
SafeCross/
├── weigths/                    ← trained model weights (in git)
├── safecross/                  ← importable Python package (core logic)
├── trafficlight_detection/     ← ML training pipeline (scripts, not a package)
├── docs/                       ← supplementary documentation
├── .agent-os/                  ← Agent OS product + standards
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## `weigths/` — Model Weights

- **Included in git:** all `.pt` files (PyTorch weights, ~50 MB each)
- **Excluded from git:** `.onnx`, `.torchscript`, `:Zone.Identifier` files
- **Name:** keep the typo "weigths" (not "weights") for consistency with original code
- Do not rename — all code references this path

| File | Purpose |
|---|---|
| `crosswalks.pt` | Pedestrian crosswalk detection |
| `lights.pt` | Traffic light color detection |
| `persons_cars.pt` | Person and vehicle detection |

## `safecross/` — Core Package

The importable Python package containing business logic. Keeps decision logic
completely separate from training code.

```
safecross/
├── __init__.py     ← exports can_cross, decide_verbose
└── decide.py       ← all decision logic
```

**Rule:** `safecross/` must not import from `trafficlight_detection/`.
It only depends on `weigths/` and `ultralytics`.

## `trafficlight_detection/` — Training Pipeline

Scripts for preparing data, training, and evaluating the traffic light model.
Not an importable package — run scripts directly.

```
trafficlight_detection/
├── trafficlight.py              ← annotation converter (run once)
├── train.py                     ← fine-tuning
├── kfold.py                     ← cross-validation
├── test.py                      ← inference evaluation
├── TrafficLight.yaml            ← dataset config
├── traffic-light-detection.ipynb
└── dataset/                     ← NOT in git (too large)
```

## `.agent-os/` — Agent OS

Agent OS product and standards documentation. Always commit `product/` and
`standards/`. The `specs/` directory is for in-progress work specs and may
be gitignored or committed case-by-case.

```
.agent-os/
├── product/
│   ├── mission.md
│   ├── roadmap.md
│   └── tech-stack.md
├── standards/
│   ├── index.yml
│   └── *.md
└── specs/
```

## `.gitignore` Rules

```
# Exclude large export formats
*.onnx
*.tflite
*.torchscript
*:Zone.Identifier

# Exclude YOLO training outputs (generated at runtime)
runs/

# Exclude dataset (too large, provide separately)
trafficlight_detection/dataset/
trafficlight_detection/fold_*/

# Do NOT exclude *.pt — weights are tracked in git
```
