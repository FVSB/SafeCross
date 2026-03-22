# SafeCross

> Sistema de asistencia para personas con discapacidad visual al cruzar calles
> mediante visión por computadora y detección de objetos en tiempo real.

---

## Descripción del Proyecto

**SafeCross** combina tres modelos de detección de objetos (YOLOv8) para
determinar de forma segura cuándo una persona con discapacidad visual puede
cruzar una calle. El sistema analiza una imagen de la escena y decide:

- **CRUZAR**: si hay paso de peatones visible, semáforo en verde y sin vehículos obstruyendo
- **ESPERAR / PEDIR AYUDA**: en cualquier otro caso

El sistema prioriza la **eliminación de Falsos Positivos** (nunca indica cruzar
cuando no es seguro) sobre la sensibilidad, logrando **0 Falsos Positivos** en
las pruebas realizadas.

### Recursos del proyecto

| Recurso | Enlace |
|---|---|
| Informe completo (Overleaf) | https://www.overleaf.com/read/yhkcxxmfddhp#5b39d8 |
| Estado del arte (Google Sheets) | https://docs.google.com/spreadsheets/d/1XK1lsZq_OCa5U1rkWthGJNdcwmJ4CpSdBDDm9qTKgU8/edit?usp=sharing |
| Repo anterior (ML Final Project) | https://github.com/FVSB/Machine_Learning_Final_Project |

---

## Arquitectura del Sistema

SafeCross usa **tres modelos YOLO independientes** en pipeline:

```
Imagen de entrada (cámara del dispositivo)
         │
         ▼
┌─────────────────────┐
│   crosswalks.pt     │──► ¿Hay paso de peatones?
│   (YOLOv8n)         │         No → ESPERAR / PEDIR AYUDA
└─────────────────────┘         │
                                Sí
                                │
                                ▼
                   ┌─────────────────────┐
                   │    lights.pt        │──► ¿Cuál es el color?
                   │    (YOLOv8m)        │
                   └─────────────────────┘
                        │            │
                   Rojo/Amarillo    Verde
                        │            │
                     ESPERAR      CRUZAR ✓
                                (con precaución)
```

### Modelos

| Modelo              | Archivo                  | Detecta                        |
|---------------------|--------------------------|--------------------------------|
| Cruces peatonales   | `weigths/crosswalks.pt`  | Pasos de peatones en la escena |
| Semáforos           | `weigths/lights.pt`      | Color del semáforo peatonal    |
| Personas/Vehículos  | `weigths/persons_cars.pt`| Vehículos en el cruce          |

---

## Estructura del Repositorio

```
SafeCross/
├── README.md                          ← Este archivo
├── LICENSE                            ← MIT License
├── requirements.txt                   ← Dependencias Python
├── .gitignore
│
├── weigths/                           ← Pesos entrenados de los modelos
│   ├── crosswalks.pt                  ← Detección de pasos de peatones (~50 MB)
│   ├── lights.pt                      ← Detección de semáforos (~50 MB)
│   └── persons_cars.pt                ← Detección de personas/vehículos (~6 MB)
│
├── safecross/                         ← Módulo principal de decisión
│   ├── __init__.py
│   └── decide.py                      ← Función can_cross() y decide_verbose()
│
├── trafficlight_detection/            ← Pipeline de entrenamiento del modelo de luces
│   ├── README.md
│   ├── trafficlight.py               ← Conversión XML → YOLO
│   ├── train.py                       ← Fine-tuning YOLOv8m
│   ├── kfold.py                       ← Validación cruzada 5-Fold
│   ├── test.py                        ← Inferencia
│   ├── TrafficLight.yaml              ← Config del dataset
│   ├── traffic-light-detection.ipynb  ← Notebook completo del pipeline
│   └── dataset/                       ← [NO en git] Imágenes y etiquetas
│
└── docs/
    └── model_weights.md               ← Documentación de pesos y métricas
```

---

## Instalación

### Prerrequisitos

- Python 3.8+
- GPU NVIDIA con CUDA (recomendado para entrenamiento; CPU funciona para inferencia)

### Instalar dependencias

```bash
git clone https://github.com/FVSB/SafeCross.git
cd SafeCross
pip install -r requirements.txt
```

---

## Uso Rápido

### Decisión de cruce (módulo principal)

```python
from safecross.decide import can_cross, decide_verbose

# Decisión simple
if can_cross("foto_calle.jpg"):
    print("Puedes cruzar")
else:
    print("Espera o pide ayuda")

# Con diagnóstico detallado
info = decide_verbose("foto_calle.jpg")
print(f"Resultado: {info['can_cross']}")
print(f"Semáforo: {info['light_color']}")
print(f"Razón: {info['reason']}")
```

### Entrenar el modelo de semáforos

```bash
cd trafficlight_detection/
python trafficlight.py   # 1. Convertir anotaciones XML → YOLO
python train.py          # 2. Fine-tuning YOLOv8m (80 epochs, GPU)
python kfold.py          # 3. (Opcional) Validación cruzada 5-Fold
python test.py           # 4. Inferencia sobre imágenes de prueba
```

---

## Métricas del Sistema

Evaluado sobre 64 imágenes de prueba con los tres modelos combinados:

| Clase      | Precisión | Recall | F1-Score |
|------------|-----------|--------|----------|
| No cruzar  | 0.636     | **1.00**  | 0.778 |
| Cruzar     | **1.00**  | 0.515  | 0.680 |

**Logro principal:** 0 Falsos Positivos — el sistema nunca indica cruzar cuando no es seguro.

---

## Pesos del Modelo

Los tres modelos entrenados están incluidos en `weigths/`. Ver [`docs/model_weights.md`](docs/model_weights.md)
para instrucciones de uso, re-entrenamiento y métricas detalladas.

---

## Clases del Modelo de Semáforos

| ID | Clase          | Descripción         |
|----|----------------|---------------------|
| 0  | `red light`    | Semáforo en rojo    |
| 1  | `yellow light` | Semáforo en amarillo|
| 2  | `green light`  | Semáforo en verde   |

---

## Tecnologías

| Tecnología        | Uso                                        |
|-------------------|--------------------------------------------|
| Ultralytics YOLO  | Detección de objetos (YOLOv8n / YOLOv8m)  |
| scikit-learn      | K-Fold Cross Validation                    |
| PyYAML            | Configuración de datasets                  |
| NumPy             | Operaciones numéricas                      |
| OpenCV / PIL      | Procesamiento de imágenes                  |

---

## Licencia

MIT License — ver [LICENSE](LICENSE).

## Autores

- **Francisco Vicente Suárez Bellón** — [@FVSB](https://github.com/FVSB)
- **Diana Laura** — Entrenamiento del modelo de semáforos

*Proyecto de investigación académica — 2024*
