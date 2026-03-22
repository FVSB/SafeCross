# Pesos del Modelo — SafeCross Traffic Light Detection

Los pesos (archivos `.pt`) **no se incluyen en el repositorio** por su tamaño
(varios cientos de MB). Este documento explica cómo obtenerlos o reproducirlos.

---

## Pesos Involucrados

| Archivo                   | Descripción                                    | Tamaño aprox. |
|---------------------------|------------------------------------------------|---------------|
| `yolov8m.pt`              | YOLOv8m pre-entrenado en COCO (Ultralytics)    | ~52 MB        |
| `yolov8m_TrafficLight.pt` | Fine-tuning final sobre el dataset del proyecto| ~52 MB        |
| `best.pt`                 | Mejor checkpoint del entrenamiento (val loss)  | ~52 MB        |

---

## Cómo Obtener `yolov8m.pt` (peso base)

El peso base se descarga automáticamente desde los servidores de Ultralytics
la primera vez que se ejecuta `train.py`:

```python
from ultralytics import YOLO
model = YOLO("yolov8m.pt")  # se descarga automáticamente si no existe
```

O manualmente:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

---

## Cómo Generar `yolov8m_TrafficLight.pt` (peso entrenado)

Para reproducir el entrenamiento desde cero:

### Paso 1 — Preparar el dataset

```bash
cd trafficlight_detection/
python trafficlight.py          # convierte XML → YOLO .txt
# Mover los .txt generados a dataset/labels/train/
```

### Paso 2 — Entrenar

```bash
python train.py
```

- Epochs: 80
- Batch: 16
- Image size: 640×640
- Patience (early stopping): 20
- Hardware recomendado: GPU NVIDIA con ≥8 GB VRAM

El entrenamiento guarda:
- `runs/detect/yolov8m_traffic_light/weights/best.pt` — mejor checkpoint
- `runs/detect/yolov8m_traffic_light/weights/last.pt` — último checkpoint
- `yolov8m_TrafficLight.pt` — copia del modelo final

### Paso 3 (opcional) — Validación cruzada

```bash
python kfold.py
```

Genera 5 modelos en `runs/detect/yolov8m_traffic_light_fold_N/`.

---

## Dónde Colocar los Pesos

```
trafficlight_detection/
├── yolov8m.pt              ← peso base (descargar de Ultralytics)
├── best.pt                 ← mejor checkpoint (generado por train.py)
├── yolov8m_TrafficLight.pt ← modelo final (generado por train.py)
└── ...
```

---

## `.gitignore` — Los pesos están excluidos

Los archivos `.pt` están en el `.gitignore` del proyecto por su tamaño.
Para compartirlos se recomienda usar:

- **Google Drive / OneDrive**: Subir y compartir enlace en el README.
- **Hugging Face Hub**: Repositorio de modelos gratuito para proyectos de ML.
- **GitHub Releases**: Para archivos ≤2 GB se pueden adjuntar como assets de release.

---

## Métricas del Modelo Entrenado

> Las métricas exactas se obtienen del informe completo del proyecto
> (`Informe.pdf` en el repo `Machine_Learning_Final_Project`).

El modelo fue entrenado con:
- **Dataset**: Imágenes de semáforos peatonales (diurnas y nocturnas)
- **Técnica**: Fine-tuning de YOLOv8m
- **Validación**: 5-Fold Cross Validation
- **Clases**: red light, yellow light, green light

Para ver las métricas de entrenamiento después de entrenar:

```python
from ultralytics import YOLO
model = YOLO("best.pt")
metrics = model.val()
print(metrics.box.map)   # mAP@50:95
print(metrics.box.map50) # mAP@50
```
