"""
safecross/api.py
----------------
FastAPI backend para SafeCross.

Expone un endpoint POST /decide que recibe una imagen desde la app Flutter,
la procesa con los tres modelos YOLO y devuelve la decisión de cruce.

Uso:
    uvicorn safecross.api:app --host 0.0.0.0 --port 8000

    # O desde la raíz del repo:
    python -m uvicorn safecross.api:app --host 0.0.0.0 --port 8000
"""

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from safecross.decide import decide_verbose

app = FastAPI(
    title="SafeCross API",
    description="Determina si es seguro cruzar la calle a partir de una imagen.",
    version="1.0.0",
)

# Permitir peticiones desde la app Flutter (cualquier origen en desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Comprueba que el servidor está activo."""
    return {"status": "ok"}


@app.post("/decide")
async def decide(image: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la decisión de cruce.

    Body (multipart/form-data):
        image: archivo de imagen (.jpg, .png, etc.)

    Returns:
        {
            "can_cross": bool,
            "crosswalk_detected": bool,
            "n_crosswalks": int,
            "light_detected": bool,
            "light_color": str | null,
            "reason": str
        }
    """
    # Validar que sea una imagen
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    # Guardar temporalmente para pasarlo a los modelos
    suffix = os.path.splitext(image.filename or "frame.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = decide_verbose(tmp_path)
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {str(e)}")
    finally:
        os.unlink(tmp_path)
