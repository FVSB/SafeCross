# Pesos del Modelo — SafeCross

SafeCross utiliza **tres modelos YOLO independientes** que trabajan juntos para
tomar la decisión de cruce. Los pesos se encuentran en la carpeta `weigths/`
del repositorio (extraídos de la rama `Results`).

---

## Modelos Disponibles

| Archivo                 | Tarea                            | Arquitectura | Tamaño |
|-------------------------|----------------------------------|--------------|--------|
| `weigths/crosswalks.pt` | Detección de pasos de peatones   | YOLOv8n      | ~50 MB |
| `weigths/lights.pt`     | Detección de color de semáforo   | YOLOv8m      | ~50 MB |
| `weigths/persons_cars.pt`| Detección de personas y vehículos| YOLOv8n      | ~6 MB  |

> **Formatos adicionales disponibles** (no incluidos en git por tamaño):
> - `crosswalks.onnx` (~99 MB) — para inferencia con ONNX Runtime
> - `crosswalks.torchscript` (~100 MB) — para deployment con TorchScript

---

## Cómo Usar los Pesos

Los pesos se cargan automáticamente desde `weigths/` cuando se usa el módulo
principal de SafeCross:

```python
from safecross.decide import can_cross, decide_verbose

# Decisión simple (True = cruzar, False = esperar)
resultado = can_cross("imagen.jpg")

# Decisión con detalles de diagnóstico
detalles = decide_verbose("imagen.jpg")
print(detalles)
# {
#   "can_cross": True,
#   "crosswalk_detected": True,
#   "n_crosswalks": 1,
#   "light_detected": True,
#   "light_color": "verde",
#   "reason": "Paso de peatones visible y semáforo en verde. Puede cruzar."
# }
```

---

## Lógica de Decisión (3 modelos en conjunto)

```
Imagen de entrada
      │
      ▼
┌─────────────────┐
│ crosswalks.pt   │──► ¿Hay paso de peatones?
└─────────────────┘         │
    No → ESPERAR           Sí
                            │
                            ▼
                   ┌─────────────────┐
                   │  lights.pt      │──► ¿Qué color?
                   └─────────────────┘
                        │        │
                    Rojo/Amarillo  Verde
                        │            │
                     ESPERAR       CRUZAR ✓
```

**Reglas:**
- **CRUZAR** si: paso de peatones visible + semáforo verde + sin vehículos en el cruce
- **ESPERAR** (o pedir ayuda) si: sin paso de peatones, o semáforo rojo/amarillo, o vehículo en cruce

---

## Métricas del Sistema Completo

Evaluado sobre 64 imágenes de prueba:

| Clase      | Precisión | Recall | F1-Score |
|------------|-----------|--------|----------|
| No cruzar  | 0.636     | **1.00**  | 0.778   |
| Cruzar     | **1.00**  | 0.515  | 0.680   |

**Puntos clave:**
- **0 Falsos Positivos**: El sistema nunca indica "cruzar" cuando no es seguro
- Tasa de 48% de Falsos Negativos: puede ser excesivamente conservador (indica esperar cuando sería seguro cruzar)
- El objetivo principal — eliminar FP — se cumple completamente

---

## Cómo Re-entrenar los Modelos

### Modelo de luces de semáforo (`lights.pt`)

Ver instrucciones detalladas en [`../trafficlight_detection/README.md`](../trafficlight_detection/README.md).

```bash
cd trafficlight_detection/
python train.py   # fine-tuning de YOLOv8m, 80 epochs
```

### Modelo de cruces peatonales (`crosswalks.pt`)

El modelo fue entrenado con fine-tuning de YOLOv8n sobre un dataset de
imágenes de cruces peatonales. Ver notebooks en la rama `Crosswalk`:
- `CrossWalk/First_Data_Set/Fine_tuning.ipynb`

### Modelo de personas y vehículos (`persons_cars.pt`)

Entrenado con YOLOv8n para detectar personas y vehículos en el contexto
de cruces peatonales.

---

## Notas sobre el `.gitignore`

Los archivos `.pt` de los modelos entrenados **SÍ están incluidos** en este
repositorio (en `weigths/`) ya que son el resultado final del proyecto.

Lo que está **excluido** por tamaño:
- Archivos `.onnx` y `.torchscript` (exportaciones grandes)
- El peso base `yolov8m.pt` (se descarga automáticamente de Ultralytics)
- Outputs de entrenamiento en `runs/`
