"""
scripts/export_tflite.py
------------------------
Exporta los tres modelos YOLO (.pt) a formato TFLite para
inferencia local en el dispositivo móvil (sin conexión a internet).

Uso:
    cd SafeCross/
    python scripts/export_tflite.py

Salida (copiar manualmente a safecross_app/assets/models/):
    weigths/crosswalks_saved_model/crosswalks_float32.tflite
    weigths/lights_saved_model/lights_float32.tflite
    weigths/persons_cars_saved_model/persons_cars_float32.tflite

Requisitos:
    pip install ultralytics tensorflow

Notas:
    - lights.pt es YOLOv8m (~50 MB) → el .tflite resultante puede ocupar ~100 MB.
      Considera cuantización int8 (imgsz=320) para reducir el tamaño.
    - crosswalks.pt y persons_cars.pt son YOLOv8n (~6 MB) → más ligeros.
    - Ultralytics guarda el .tflite dentro de una carpeta _saved_model/.
"""

import shutil
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = REPO_ROOT / "weigths"
OUT_DIR = REPO_ROOT / "safecross_app" / "assets" / "models"

MODELS = [
    ("crosswalks.pt", "crosswalks.tflite"),
    ("lights.pt",     "lights.tflite"),
    # persons_cars no se usa en la lógica final, pero se exporta por si acaso
    # ("persons_cars.pt", "persons_cars.tflite"),
]


def export_model(pt_name: str, out_name: str) -> None:
    pt_path = WEIGHTS_DIR / pt_name
    if not pt_path.exists():
        print(f"  [!] No encontrado: {pt_path} — omitiendo")
        return

    print(f"\n  Exportando {pt_name} → TFLite …")
    model = YOLO(str(pt_path))

    # export() devuelve la ruta a la carpeta _saved_model
    result = model.export(
        format="tflite",
        imgsz=640,       # resolución de entrada del modelo
        half=False,      # float32 (más compatible; usa half=True para float16)
        int8=False,      # sin cuantización int8 (mayor precisión)
        nms=True,        # incluir NMS en el grafo TFLite
    )

    # Ultralytics nombra el archivo: <stem>_float32.tflite dentro de _saved_model/
    stem = pt_path.stem
    candidates = list(Path(result).glob("*.tflite"))

    if not candidates:
        print(f"  [!] No se encontró .tflite en: {result}")
        return

    src = candidates[0]
    dst = OUT_DIR / out_name
    shutil.copy2(src, dst)
    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"  ✓  Guardado en: {dst}  ({size_mb:.1f} MB)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SafeCross — Exportación a TFLite")
    print("=" * 40)

    for pt_name, out_name in MODELS:
        export_model(pt_name, out_name)

    print("\n¡Listo! Copia los .tflite generados a:")
    print(f"  {OUT_DIR}")
    print("\nLuego reconstruye la app con:  flutter build apk")


if __name__ == "__main__":
    main()
