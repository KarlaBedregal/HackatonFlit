"""
api_server.py — TerraGuard Arequipa REST API
FastAPI server que expone el backend Python al frontend Flutter.

Endpoints:
  GET  /health
  GET  /api/zones
  GET  /api/metrics
  GET  /api/map-samples?limit=150
  POST /api/analyze-image   (multipart, campo "file")
  POST /api/predict         (JSON)
  POST /api/report          (JSON)

Ejecutar:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_generator import (
    ETIQUETAS,
    ZONAS_RIESGO,
    dist_a_mina_mas_cercana,
    generar_dataset,
)
from src import model as M
from src.reporte import generar_reporte
from src.vision import analizar_hoja, leer_imagen_desde_bytes

# ---------------------------------------------------------------------------
# App y CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="TerraGuard Arequipa API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
COLOR_RIESGO = {"Bajo": "#2e7d32", "Medio": "#f9a825", "Alto": "#c62828"}

METAL_ZONA = {
    "Cerro Verde (Cu/As)": "Cobre",
    "La Joya (As agua)":   "Arsenico",
    "Valle del Tambo (As)": "Arsenico",
}

# ---------------------------------------------------------------------------
# Carga única del modelo al arrancar (tarda ~2 s)
# ---------------------------------------------------------------------------
_df = generar_dataset()
_modelo, _metricas, _ = M.entrenar(_df)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/zones")
def get_zones():
    zones = [
        {
            "nombre": nombre,
            "lat": float(lat),
            "lon": float(lon),
            "metal": METAL_ZONA.get(nombre, "Arsenico"),
            "dist_km": 0.0,
        }
        for nombre, (lat, lon) in ZONAS_RIESGO.items()
    ]
    return {"zones": zones}


@app.get("/api/metrics")
def get_metrics():
    return {"metricas": _metricas}


@app.get("/api/map-samples")
def get_map_samples(limit: int = 150):
    focos = [
        {"nombre": n, "lat": float(la), "lon": float(lo)}
        for n, (la, lo) in ZONAS_RIESGO.items()
    ]
    muestra = _df.sample(min(limit, len(_df)), random_state=1)
    puntos = [
        {
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "riesgo": ETIQUETAS[int(r["riesgo"])],
            "color": COLOR_RIESGO[ETIQUETAS[int(r["riesgo"])]],
        }
        for _, r in muestra.iterrows()
    ]
    return {"focos": focos, "puntos": puntos}


@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = leer_imagen_desde_bytes(contents)
    if img is None:
        return {
            "clorosis_pct": 42.0,
            "necrosis_pct": 18.0,
            "ok": False,
            "msg": "No se pudo decodificar la imagen. Usando estimacion de respaldo.",
        }
    return analizar_hoja(img)


class PredictRequest(BaseModel):
    zona: str
    ph: float
    clorosis_pct: float
    necrosis_pct: float
    lat: Optional[float] = None
    lon: Optional[float] = None


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.zona not in ZONAS_RIESGO:
        raise HTTPException(status_code=400, detail=f"Zona desconocida: {req.zona}")

    zlat, zlon = ZONAS_RIESGO[req.zona]
    lat = req.lat if req.lat is not None else zlat
    lon = req.lon if req.lon is not None else zlon

    _, dist = dist_a_mina_mas_cercana(lat, lon)
    pred = M.predecir_riesgo(_modelo, req.clorosis_pct, req.necrosis_pct, dist, req.ph)
    metal = METAL_ZONA.get(req.zona, "Arsenico")

    return {
        **pred,
        "zona": req.zona,
        "metal": metal,
        "dist_km": float(dist),
        "ph": float(req.ph),
        "clorosis_pct": float(req.clorosis_pct),
        "necrosis_pct": float(req.necrosis_pct),
        "lat": float(lat),
        "lon": float(lon),
        "color": COLOR_RIESGO[pred["riesgo_txt"]],
    }


class ReportRequest(BaseModel):
    zona: str
    metal: str
    riesgo: str
    dist: float
    ph: float
    clorosis: float
    necrosis: float
    api_key: Optional[str] = None


@app.post("/api/report")
def report(req: ReportRequest):
    texto = generar_reporte(
        req.zona, req.metal, req.riesgo,
        req.dist, req.ph, req.clorosis, req.necrosis,
        api_key=req.api_key,
    )
    return {"reporte": texto}
