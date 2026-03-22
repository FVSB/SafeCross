# Roadmap

## Phase 1 — Traffic Light Detection ✅ COMPLETED

**Goal:** Detect the color of a pedestrian traffic light (red/yellow/green).

**Deliverables:**
- Dataset of annotated traffic light images (Pascal VOC XML format)
- `trafficlight.py` — XML → YOLO annotation converter (3 classes)
- `train.py` — YOLOv8m fine-tuning (80 epochs, 640px, batch 16)
- `kfold.py` — 5-fold cross-validation for robustness evaluation
- `test.py` — inference on test images (day + night)
- `TrafficLight.yaml` — dataset config
- **Trained model:** `weigths/lights.pt` (~50 MB)

**Results:** >90% precision on traffic light color detection.

---

## Phase 2 — Crosswalk Detection ✅ COMPLETED

**Goal:** Detect whether a pedestrian crosswalk is present in the scene.

**Deliverables:**
- Fine-tuned YOLOv8n model for crosswalk detection
- K-fold cross-validation (k=3 and k=5 tested)
- Testing against first and extended datasets
- **Trained model:** `weigths/crosswalks.pt` (~50 MB)
- Export formats: `.onnx`, `.torchscript` (not tracked in git)

**Results:** >90% precision on crosswalk detection from pedestrian perspective.

---

## Phase 3 — Decision System ✅ COMPLETED

**Goal:** Combine all models into a unified cross/don't-cross decision engine.

**Deliverables:**
- `safecross/decide.py` — `can_cross()` + `decide_verbose()` functions
- `safecross/__init__.py` — importable Python package
- `weigths/persons_cars.pt` (~6 MB) — vehicle detection (used in extended logic)
- Decision rules: crosswalk + green light = CROSS; any other = WAIT
- Evaluation on 64 test images
- **Trained models:** all 3 in `weigths/`

**Results:**
- 0 False Positives achieved
- 48% false negative rate (conservative — acceptable for safety)
- Precision (cross): 1.0 | Recall (cross): 0.515 | F1: 0.680

---

## Phase 4 — Mobile Integration 🔲 PLANNED

**Goal:** Deploy SafeCross as a usable assistive app.

**Potential deliverables:**
- REST API wrapper around `safecross/decide.py`
- Mobile app (iOS/Android) with camera capture + audio feedback
- Real-time streaming inference (video frame input)
- Accessibility features: audio announcements, haptic feedback
- Multi-language support

**Open questions:**
- Target platform: dedicated device vs. standard smartphone?
- Connectivity: on-device inference vs. cloud API?
- Battery and latency constraints?

---

## Phase 5 — Dataset Expansion & Model Improvement 🔲 PLANNED

**Goal:** Improve recall without sacrificing zero-FP guarantee.

**Potential deliverables:**
- Larger annotated dataset (more diverse locations, lighting, countries)
- Night-specific model fine-tuning
- Vehicle-in-crosswalk detection integration into decision logic
- Reduce false negative rate from 48% toward <20%
