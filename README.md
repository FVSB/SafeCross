# SafeCross

> Investigación y desarrollo de un sistema de asistencia para personas con
> discapacidad visual al cruzar calles mediante visión por computadora.

---

## Descripción del Proyecto

**SafeCross** es un sistema de apoyo a la movilidad que utiliza detección de
objetos en tiempo real para identificar el estado de semáforos peatonales
(rojo, amarillo, verde) y asistir a personas con discapacidad visual en el
cruce seguro de calles.

El núcleo del sistema es un modelo **YOLOv8m** entrenado mediante fine-tuning
sobre un dataset personalizado de semáforos peatonales, tanto en condiciones
diurnas como nocturnas.

### Recursos del proyecto

- **Informe completo (Overleaf):** https://www.overleaf.com/read/yhkcxxmfddhp#5b39d8
- **Estado del arte (Google Sheets):** https://docs.google.com/spreadsheets/d/1XK1lsZq_OCa5U1rkWthGJNdcwmJ4CpSdBDDm9qTKgU8/edit?usp=sharing
- **Repo anterior (ML Final Project):** https://github.com/FVSB/Machine_Learning_Final_Project

---

## Arquitectura del Sistema

```
Imagen de cámara
       │
       ▼
┌──────────────────────┐
│   YOLOv8m            │  ← fine-tuning con dataset de semáforos
│   Traffic Light Det. │
└──────────────────────┘
       │
       ▼
  Clase detectada
  ┌────────────┐
  │ red light  │ → Señal de STOP (no cruzar)
  │ yellow     │ → Señal de precaución
  │ green light│ → Señal de CRUZAR
  └────────────┘
       │
       ▼
  Asistencia al usuario
  (audio / haptic / visual)
```

---

## Estructura del Repositorio

```
SafeCross/
├── README.md                        ← Este archivo
├── LICENSE                          ← MIT License
├── requirements.txt                 ← Dependencias Python
├── .gitignore
│
├── trafficlight_detection/          ← Módulo principal de visión por computadora
│   ├── README.md                    ← Documentación del módulo
│   ├── trafficlight.py              ← Conversión de anotaciones XML → YOLO
│   ├── train.py                     ← Entrenamiento del modelo (fine-tuning)
│   ├── kfold.py                     ← Validación cruzada 5-Fold
│   ├── test.py                      ← Inferencia / predicción
│   ├── TrafficLight.yaml            ← Configuración del dataset
│   ├── traffic-light-detection.ipynb← Notebook completo del pipeline
│   └── dataset/                     ← [NO incluido en git] Dataset de imágenes
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── annotations/
│           └── xml/                 ← Anotaciones originales Pascal VOC
│
└── docs/
    └── model_weights.md             ← Documentación de pesos del modelo
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

### 1. Preparar anotaciones del dataset

```bash
cd trafficlight_detection/
python trafficlight.py
```

Convierte anotaciones en formato XML (Pascal VOC) a formato YOLO `.txt`.

### 2. Entrenar el modelo

```bash
python train.py
```

Realiza fine-tuning de YOLOv8m. El modelo base `yolov8m.pt` se descarga
automáticamente si no está presente.

### 3. Validación cruzada (opcional)

```bash
python kfold.py
```

Evalúa la robustez del modelo con 5-Fold Cross Validation.

### 4. Inferencia sobre imágenes de prueba

```bash
python test.py
```

Guarda imágenes anotadas y archivos de texto con detecciones en `runs/detect/predict/`.

---

## Pesos del Modelo

Los pesos `.pt` **no se incluyen en git** por su tamaño (~52 MB cada uno).

| Archivo                   | Cómo obtenerlo                                       |
|---------------------------|------------------------------------------------------|
| `yolov8m.pt`              | Descarga automática al ejecutar `train.py`           |
| `yolov8m_TrafficLight.pt` | Generado al completar `train.py`                     |
| `best.pt`                 | Mejor checkpoint en `runs/detect/.../weights/best.pt`|

Ver instrucciones detalladas en [`docs/model_weights.md`](docs/model_weights.md).

---

## Clases Detectadas

| ID | Clase          | Significado         |
|----|----------------|---------------------|
| 0  | `red light`    | Semáforo en rojo    |
| 1  | `yellow light` | Semáforo en amarillo|
| 2  | `green light`  | Semáforo en verde   |

---

## Tecnologías Utilizadas

| Tecnología        | Versión mínima | Uso                              |
|-------------------|----------------|----------------------------------|
| Python            | 3.8            | Lenguaje principal               |
| Ultralytics YOLO  | 8.0.0          | Modelo de detección de objetos   |
| scikit-learn      | 1.0.0          | K-Fold Cross Validation          |
| PyYAML            | 6.0            | Configuración del dataset        |
| NumPy             | 1.21.0         | Operaciones numéricas            |

---

## Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## Autores

- **Francisco Vicente Suárez Bellón** — [@FVSB](https://github.com/FVSB)

*Proyecto de investigación académica — 2024*
