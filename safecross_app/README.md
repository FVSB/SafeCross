# SafeCross App — Flutter

App móvil que usa la cámara del teléfono para indicar si es seguro cruzar un paso de peatones.

## Arquitectura

```
Teléfono (Flutter)  ──── HTTP POST /decide ────▶  PC/Servidor (FastAPI + YOLO)
     │                        imagen                      │
     │◀─── JSON con decisión ────────────────────────────┘
     │
     ▼
  UI: PUEDES CRUZAR / ESPERA / CON CUIDADO
  TTS: Voz en español
```

## Pantallas

| Estado | Color | Texto |
|---|---|---|
| Semáforo verde + paso peatonal | 🟢 Verde | ¡PUEDES CRUZAR! |
| Semáforo rojo | 🔴 Rojo | ESPERA |
| Semáforo amarillo | 🟡 Amarillo | CON CUIDADO |
| Sin paso peatonal | ⚫ Gris | SIN PASO PEATONAL |
| Sin semáforo | ⚫ Gris azul | SEMÁFORO NO DETECTADO |

## Cómo usar

### 1. Arrancar el servidor (en el PC)

```bash
cd SafeCross/
pip install fastapi uvicorn ultralytics python-multipart
uvicorn safecross.api:app --host 0.0.0.0 --port 8000
```

### 2. Conectar la app al servidor

1. Abre la app en el teléfono
2. Pulsa el icono ⚙️ (arriba a la izquierda)
3. Introduce la IP local de tu PC:
   `http://192.168.X.X:8000`
   (PC y teléfono deben estar en la misma red Wi-Fi)

### 3. Apunta la cámara al cruce

La app analiza automáticamente un frame cada 3 segundos y anuncia el resultado en voz alta.

## Instalación (desarrollo)

```bash
cd safecross_app/
flutter pub get
flutter run
```

Requiere Flutter 3.x y Android SDK / Xcode instalados.
