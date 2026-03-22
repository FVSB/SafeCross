"""
test.py
-------
Ejecuta inferencia con el modelo entrenado sobre imágenes de prueba y
guarda los resultados (imágenes anotadas + archivos de texto con detecciones).

Requisitos previos:
    - Tener el peso entrenado `best.pt` en el mismo directorio que este script
      (generado por train.py o kfold.py como el mejor checkpoint).
    - Dataset de test disponible en: dataset/images/test/

Uso:
    python test.py

Salida:
    - Imágenes anotadas en: runs/detect/predict/
    - Archivos .txt con detecciones en: runs/detect/predict/labels/
"""

import os

from ultralytics import YOLO

# Directorio donde reside este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta al modelo entrenado (best checkpoint)
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

# Directorio con imágenes de prueba
TEST_SOURCE = os.path.join(SCRIPT_DIR, "dataset", "images", "test")

if __name__ == "__main__":
    print(f"Cargando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Ejecutando predicción sobre: {TEST_SOURCE}")
    results = model.predict(
        source=TEST_SOURCE,
        save=True,       # Guarda imágenes con bounding boxes dibujados
        save_txt=True,   # Guarda detecciones en formato .txt (YOLO)
    )

    print(f"Predicción completada. Resultados guardados en: runs/detect/predict/")
