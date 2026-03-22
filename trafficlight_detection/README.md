# Módulo: Detección de Semáforos Peatonales

Módulo de visión por computadora para la detección de luces de semáforos
peatonales usando **YOLOv8m** con fine-tuning sobre un dataset personalizado.
Forma parte del proyecto **SafeCross** de asistencia para personas con
discapacidad visual.

---

## Clases Detectadas

| ID | Nombre        | Descripción                |
|----|---------------|----------------------------|
| 0  | `red light`   | Semáforo en rojo (STOP)    |
| 1  | `yellow light`| Semáforo en amarillo       |
| 2  | `green light` | Semáforo en verde (CRUZAR) |

---

## Archivos del Módulo

| Archivo                         | Descripción                                              |
|---------------------------------|----------------------------------------------------------|
| `trafficlight.py`               | Convierte anotaciones XML → formato YOLO (`.txt`)        |
| `train.py`                      | Fine-tuning del modelo YOLOv8m base                      |
| `kfold.py`                      | Validación cruzada de 5 folds                            |
| `test.py`                       | Inferencia/predicción con el modelo entrenado            |
| `TrafficLight.yaml`             | Configuración del dataset para YOLO                      |
| `traffic-light-detection.ipynb` | Notebook completo con todo el pipeline                   |

Para información sobre los pesos del modelo ver: [`../docs/model_weights.md`](../docs/model_weights.md)

---

## Estructura del Dataset

```
trafficlight_detection/
└── dataset/
    ├── annotations/
    │   ├── xml/        ← anotaciones originales en formato Pascal VOC (.xml)
    │   └── output/     ← anotaciones convertidas para YOLO (.txt) [generado]
    ├── images/
    │   ├── train/      ← imágenes de entrenamiento
    │   ├── val/        ← imágenes de validación
    │   └── test/       ← imágenes de prueba
    └── labels/
        ├── train/      ← etiquetas YOLO de entrenamiento
        ├── val/
        └── test/
```

---

## Pasos del Pipeline

### 1. Preparar las anotaciones

Si tienes anotaciones en formato XML (Pascal VOC), conviértelas al formato YOLO:

```bash
python trafficlight.py
```

Los archivos `.txt` se generarán en `dataset/annotations/output/`.
Muévelos a `dataset/labels/train/` (o val/test según corresponda).

### 2. Entrenar el modelo

```bash
python train.py
```

Requiere:
- `yolov8m.pt` en este directorio (se descarga automáticamente de Ultralytics)
- `TrafficLight.yaml` correctamente configurado

Resultado: `yolov8m_TrafficLight.pt` + métricas en `runs/detect/yolov8m_traffic_light/`

### 3. Validación cruzada (opcional)

Para evaluar la robustez del modelo con 5-fold cross validation:

```bash
python kfold.py
```

Requiere `best.pt` como checkpoint base.

### 4. Ejecutar predicción

```bash
python test.py
```

Guarda resultados en `runs/detect/predict/`.

---

## Hiperparámetros Principales

| Parámetro    | Valor      | Script      |
|--------------|------------|-------------|
| Epochs       | 80         | train.py    |
| Batch size   | 16         | train.py    |
| Image size   | 640×640    | train.py    |
| Patience     | 20         | train.py    |
| K-Folds      | 5          | kfold.py    |
| Epochs/fold  | 15         | kfold.py    |

---

## Dependencias

Ver `requirements.txt` en la raíz del proyecto.

```bash
pip install -r ../requirements.txt
```
