"""
safecross/decide.py
-------------------
Módulo principal de decisión de SafeCross.

Combina tres modelos YOLO para determinar si es seguro cruzar la calle:
    1. crosswalks.pt   — detecta la existencia de un paso de peatones
    2. lights.pt       — detecta el color de la luz del semáforo (verde/rojo)
    3. persons_cars.pt — detecta vehículos en el cruce (no usado aún en lógica final)

Lógica de decisión:
    SE ACONSEJA CRUZAR si:
        - Existe un paso de peatones en la imagen, Y
        - El semáforo muestra luz verde, Y
        - No hay vehículos sobre el paso de peatones.

    SE ACONSEJA ESPERAR (o pedir ayuda) si:
        - No se detecta paso de peatones, O
        - El semáforo muestra luz roja o amarilla, O
        - Hay un vehículo de grandes dimensiones en el cruce.

Pesos del modelo:
    Ubicados en: weigths/ (en la raíz del repositorio)
        - weigths/crosswalks.pt
        - weigths/lights.pt
        - weigths/persons_cars.pt

Uso:
    from safecross.decide import can_cross
    result = can_cross("path/to/image.jpg")
    if result:
        print("Puedes cruzar")
    else:
        print("Espera o pide ayuda")
"""

import os
from typing import Optional

from ultralytics import YOLO

# Directorio raíz del repositorio (dos niveles arriba de este archivo)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas a los pesos del modelo
_WEIGHTS_DIR = os.path.join(_REPO_ROOT, "weigths")
_CROSSWALK_WEIGHT = os.path.join(_WEIGHTS_DIR, "crosswalks.pt")
_LIGHTS_WEIGHT = os.path.join(_WEIGHTS_DIR, "lights.pt")
_PERSONS_CARS_WEIGHT = os.path.join(_WEIGHTS_DIR, "persons_cars.pt")

# Nombre de la clase "verde" en el modelo de luces
_GREEN_CLASS_NAME = "verde"

# Confianza mínima por defecto para las detecciones
DEFAULT_CONF = 0.75


def _load_models():
    """Carga los tres modelos YOLO. Lanza FileNotFoundError si faltan pesos."""
    for path, name in [
        (_CROSSWALK_WEIGHT, "crosswalks.pt"),
        (_LIGHTS_WEIGHT, "lights.pt"),
        (_PERSONS_CARS_WEIGHT, "persons_cars.pt"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Peso del modelo no encontrado: {path}\n"
                f"Asegúrate de que los archivos .pt estén en: {_WEIGHTS_DIR}"
            )
    return YOLO(_CROSSWALK_WEIGHT), YOLO(_LIGHTS_WEIGHT), YOLO(_PERSONS_CARS_WEIGHT)


def can_cross(img_path: str, conf: float = DEFAULT_CONF) -> bool:
    """
    Determina si es seguro cruzar la calle en la imagen dada.

    Args:
        img_path (str): Ruta a la imagen de entrada.
        conf     (float): Umbral de confianza para las detecciones (0.0–1.0).

    Returns:
        bool: True si se recomienda cruzar, False si se recomienda esperar.

    Raises:
        FileNotFoundError: Si alguno de los archivos de pesos no existe.
        ValueError: Si la ruta de imagen no existe.
    """
    if not os.path.exists(img_path):
        raise ValueError(f"Imagen no encontrada: {img_path}")

    crosswalk_model, lights_model, persons_cars_model = _load_models()

    # 1. Verificar existencia de paso de peatones
    crosswalk_results = crosswalk_model(img_path, conf=conf)
    crosswalks_detected = crosswalk_results[0].boxes.xyxy.tolist()
    if len(crosswalks_detected) < 1:
        return False  # No hay paso de peatones visible

    # 2. Verificar estado del semáforo
    lights_results = lights_model(img_path, conf=conf)
    lights_detected = lights_results[0].boxes.xyxy.tolist()
    if len(lights_detected) != 1:
        return False  # No se detecta exactamente un semáforo

    class_names = lights_results[0].names
    detected_class_id = int(lights_results[0].boxes.cls.tolist()[0])
    detected_class_name = class_names[detected_class_id]

    if detected_class_name != _GREEN_CLASS_NAME:
        return False  # El semáforo no está en verde

    # 3. (Opcional) Verificar ausencia de vehículos en el cruce
    # persons_cars_results = persons_cars_model(img_path, conf=conf)
    # vehicles = persons_cars_results[0].boxes.xyxy.tolist()
    # if any_vehicle_in_crosswalk(vehicles, crosswalks_detected):
    #     return False

    return True  # Cruce seguro


def decide_verbose(img_path: str, conf: float = DEFAULT_CONF) -> dict:
    """
    Versión detallada de can_cross() que devuelve información de diagnóstico.

    Args:
        img_path (str): Ruta a la imagen de entrada.
        conf     (float): Umbral de confianza para las detecciones.

    Returns:
        dict con claves:
            - can_cross (bool): Resultado de la decisión.
            - crosswalk_detected (bool): Si se detectó un paso de peatones.
            - n_crosswalks (int): Número de pasos de peatones detectados.
            - light_detected (bool): Si se detectó exactamente un semáforo.
            - light_color (str|None): Color del semáforo detectado.
            - reason (str): Explicación del resultado.
    """
    if not os.path.exists(img_path):
        raise ValueError(f"Imagen no encontrada: {img_path}")

    crosswalk_model, lights_model, _ = _load_models()

    # Paso de peatones
    crosswalk_results = crosswalk_model(img_path, conf=conf)
    n_crosswalks = len(crosswalk_results[0].boxes.xyxy.tolist())

    if n_crosswalks < 1:
        return {
            "can_cross": False,
            "crosswalk_detected": False,
            "n_crosswalks": 0,
            "light_detected": False,
            "light_color": None,
            "reason": "No se detectó un paso de peatones en la imagen.",
        }

    # Semáforo
    lights_results = lights_model(img_path, conf=conf)
    n_lights = len(lights_results[0].boxes.xyxy.tolist())

    if n_lights != 1:
        return {
            "can_cross": False,
            "crosswalk_detected": True,
            "n_crosswalks": n_crosswalks,
            "light_detected": False,
            "light_color": None,
            "reason": f"Se esperaba 1 semáforo, se detectaron {n_lights}.",
        }

    class_names = lights_results[0].names
    detected_class_id = int(lights_results[0].boxes.cls.tolist()[0])
    light_color = class_names[detected_class_id]

    if light_color != _GREEN_CLASS_NAME:
        return {
            "can_cross": False,
            "crosswalk_detected": True,
            "n_crosswalks": n_crosswalks,
            "light_detected": True,
            "light_color": light_color,
            "reason": f"El semáforo está en '{light_color}', no en verde.",
        }

    return {
        "can_cross": True,
        "crosswalk_detected": True,
        "n_crosswalks": n_crosswalks,
        "light_detected": True,
        "light_color": light_color,
        "reason": "Paso de peatones visible y semáforo en verde. Puede cruzar.",
    }
