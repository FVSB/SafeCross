"""
kfold.py
--------
Implementa validación cruzada de 5 folds (K-Fold Cross Validation) para
el entrenamiento del modelo YOLOv8m de detección de semáforos.

Proceso por fold:
    1. Divide las imágenes de entrenamiento en train/val (80%/20%).
    2. Copia las imágenes y etiquetas a directorios temporales del fold.
    3. Genera un archivo YAML de configuración del dataset para ese fold.
    4. Entrena el modelo YOLOv8 durante 15 epochs.
    5. Limpia las imágenes temporales (los resultados de runs/ se conservan).

Clases del modelo:
    0 - red light    (luz roja)
    1 - yellow light (luz amarilla)
    2 - green light  (luz verde)

Requisitos previos:
    - Tener el peso base `best.pt` en el mismo directorio (checkpoint previo
      de train.py o de Ultralytics).
    - Dataset preparado en: dataset/images/train/ y dataset/labels/train/

Uso:
    python kfold.py

Salida:
    - 5 modelos entrenados en: runs/detect/yolov8m_traffic_light_fold_N/
    - Archivos YAML de configuración: data_fold_N.yaml
    - Listas de imágenes por fold: train_images_fold_N.txt / val_images_fold_N.txt
"""

import os
import shutil
from multiprocessing import freeze_support

import numpy as np
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO

# Directorio donde reside este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Clases del modelo (deben coincidir con trafficlight.py y TrafficLight.yaml)
CLASSES = ["red light", "yellow light", "green light"]
NUM_CLASSES = len(CLASSES)

# Hiperparámetros de K-Fold
N_SPLITS = 5
RANDOM_STATE = 42
EPOCHS = 15
IMG_SIZE = 640
BATCH = 16
DEVICE = 0  # 0 = primera GPU; usar 'cpu' si no hay GPU

# Rutas del dataset
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "dataset", "images", "train")
LABEL_FOLDER = os.path.join(SCRIPT_DIR, "dataset", "labels", "train")

# Peso base para cada fold (checkpoint previo)
BASE_MODEL = os.path.join(SCRIPT_DIR, "best.pt")


def run_kfold():
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    X = np.array(image_files)
    print(f"Total de imágenes de entrenamiento: {len(X)}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")
        train_images = X[train_idx]
        val_images = X[val_idx]

        fold_dir = os.path.join(SCRIPT_DIR, f"fold_{fold}")

        # Crear estructura de directorios del fold
        for split in ("train", "val"):
            os.makedirs(os.path.join(fold_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(fold_dir, "labels", split), exist_ok=True)

        # Copiar imágenes y etiquetas
        for img, split, split_images in [
            ("train", "train", train_images),
            ("val", "val", val_images),
        ]:
            for img_name in split_images:
                shutil.copy2(
                    os.path.join(IMAGE_FOLDER, img_name),
                    os.path.join(fold_dir, "images", split, img_name),
                )
                label_name = os.path.splitext(img_name)[0] + ".txt"
                label_src = os.path.join(LABEL_FOLDER, label_name)
                if os.path.exists(label_src):
                    shutil.copy2(
                        label_src,
                        os.path.join(fold_dir, "labels", split, label_name),
                    )

        # Configuración YAML del dataset para este fold
        data_cfg = {
            "train": os.path.join(fold_dir, "images", "train"),
            "val": os.path.join(fold_dir, "images", "val"),
            "nc": NUM_CLASSES,
            "names": CLASSES,
        }
        yaml_path = os.path.join(SCRIPT_DIR, f"data_fold_{fold}.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data_cfg, f, default_flow_style=False, allow_unicode=True)

        # Guardar listas de imágenes del fold
        with open(os.path.join(SCRIPT_DIR, f"train_images_fold_{fold}.txt"), "w") as f:
            f.write("\n".join(train_images))
        with open(os.path.join(SCRIPT_DIR, f"val_images_fold_{fold}.txt"), "w") as f:
            f.write("\n".join(val_images))

        # Entrenar el modelo para este fold
        print(f"Entrenando fold {fold} con {len(train_images)} train / {len(val_images)} val imágenes...")
        model = YOLO(BASE_MODEL)
        model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            name=f"yolov8m_traffic_light_fold_{fold}",
            device=DEVICE,
        )

        # Limpiar archivos temporales del fold (conservar resultados en runs/)
        for split in ("train", "val"):
            for fname in os.listdir(os.path.join(fold_dir, "images", split)):
                os.remove(os.path.join(fold_dir, "images", split, fname))
            for fname in os.listdir(os.path.join(fold_dir, "labels", split)):
                os.remove(os.path.join(fold_dir, "labels", split, fname))

        print(f"Fold {fold} completado.")

    print("\nEntrenamiento con validación cruzada completado.")
    print(f"Resultados en: runs/detect/yolov8m_traffic_light_fold_N/")


if __name__ == "__main__":
    freeze_support()
    run_kfold()
