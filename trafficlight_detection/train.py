"""
train.py
--------
Realiza fine-tuning de un modelo YOLOv8m pre-entrenado para la detección
de luces de semáforos peatonales (rojo, amarillo, verde).

Requisitos previos:
    - Tener el peso base `yolov8m.pt` en el mismo directorio que este script
      (se descarga automáticamente de Ultralytics si no existe).
    - Tener el archivo `TrafficLight.yaml` en el mismo directorio.
    - Dataset preparado con la estructura indicada en TrafficLight.yaml.
    - GPU disponible (device=0). Para CPU usar device='cpu'.

Uso:
    python train.py

Salida:
    - Modelo entrenado guardado en: yolov8m_TrafficLight.pt
    - Resultados y métricas en: runs/detect/yolov8m_traffic_light/
"""

import os
from multiprocessing import freeze_support

from ultralytics import YOLO

# Directorio donde reside este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas relativas al script (portables entre sistemas operativos)
MODEL_BASE = os.path.join(SCRIPT_DIR, "yolov8m.pt")
DATA_YAML = os.path.join(SCRIPT_DIR, "TrafficLight.yaml")
OUTPUT_MODEL = os.path.join(SCRIPT_DIR, "yolov8m_TrafficLight.pt")

# Hiperparámetros de entrenamiento
EPOCHS = 80
IMG_SIZE = 640
BATCH = 16
PATIENCE = 20  # Early stopping: detiene si no mejora en N epochs
DEVICE = 0     # 0 = primera GPU; usar 'cpu' si no hay GPU


if __name__ == "__main__":
    freeze_support()

    print(f"Cargando modelo base: {MODEL_BASE}")
    model = YOLO(MODEL_BASE)

    print(f"Iniciando fine-tuning con datos: {DATA_YAML}")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="yolov8m_traffic_light",
        patience=PATIENCE,
        device=DEVICE,
    )

    print(f"Entrenamiento completado. Guardando modelo en: {OUTPUT_MODEL}")
    model.save(OUTPUT_MODEL)
    print("Listo.")
