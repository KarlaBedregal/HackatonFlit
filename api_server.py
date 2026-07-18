"""
api_server.py — TerraGuard Arequipa REST API v3.0

Auto-detección de zona minera desde GPS — el usuario no necesita seleccionar zona.
Sin dependencia de API key externa — todos los reportes se generan localmente.

Endpoints:
  GET  /health
  GET  /api/zones
  GET  /api/metrics
  GET  /api/eca
  GET  /api/map-samples?limit=150

  POST /api/analyze-image          (multipart, campo "file")
  POST /api/predict                (JSON — zona opcional, auto-detect por GPS)
  POST /api/full-assessment        (JSON — análisis completo en una sola llamada)
  POST /api/report                 (JSON — reporte Markdown, sin API key)

  GET  /api/health-risk            (query params: zona, riesgo, dist_km)
  GET  /api/economic-impact        (query params: zona, riesgo, dist_km)
  GET  /api/indicators             (query params: riesgo, score, dist_km, metal, ph)
  GET  /api/compliance-dashboard   (resumen de compliance por zona para empresa minera)

Lógica de auto-detección de zona:
  1. Si se proporciona lat/lon → encuentra la mina más cercana automáticamente
  2. Si se proporciona zona → usa esa zona (compatibilidad con frontend existente)
  3. Si no hay ni GPS ni zona → error 422 indicando qué falta

Ejecutar:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

from typing import Optional
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

from src.data_generator import (
    ETIQUETAS, FOCOS_MINEROS, ZONAS_RIESGO, ZONAS_AGRICOLAS,
    METAL_POR_ZONA, METAL_IDX_POR_ZONA,
    dist_a_mina_mas_cercana, generar_dataset,
    ECA_SUELO_AGRICOLA, ECA_AGUA_RIEGO,
)
from src import model as M
from src.reporte import generar_reporte
from src.vision import analizar_hoja, leer_imagen_desde_bytes
from src.health_risk import calcular_riesgo_salud
from src.indicators import estimar_indicadores, calcular_impacto_economico

# ─── App y CORS ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TerraGuard Arequipa API",
    version="3.0.0",
    description=(
        "Sistema predictivo de riesgo de contaminación por metales pesados en Arequipa. "
        "Auto-detección de zona minera desde GPS. Sin dependencia de API key externa."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLOR_RIESGO = {"Bajo": "#2e7d32", "Medio": "#f9a825", "Alto": "#c62828"}

# ─── Carga única al arrancar ───────────────────────────────────────────────────
_df = generar_dataset()
_modelo, _metricas, _ = M.entrenar(_df)


# ─── Helpers ───────────────────────────────────────────────────────────────────
def _resolver_zona_y_coords(zona: Optional[str], lat: Optional[float],
                              lon: Optional[float]) -> tuple:
    """
    Resuelve zona + coordenadas con la siguiente prioridad:
      1. GPS (lat/lon) → auto-detecta la zona más cercana
      2. Zona manual → usa centroide del área agrícola de esa zona
    Devuelve (zona_nombre, foco_dict, lat, lon, dist_km, origen_str).
    """
    if lat is not None and lon is not None:
        zona_nombre, dist = dist_a_mina_mas_cercana(lat, lon)
        foco = FOCOS_MINEROS[zona_nombre]
        return zona_nombre, foco, lat, lon, dist, "GPS"

    if zona:
        if zona not in FOCOS_MINEROS:
            raise HTTPException(400, f"Zona desconocida: '{zona}'. "
                                f"Zonas válidas: {list(FOCOS_MINEROS)}")
        foco = FOCOS_MINEROS[zona]
        agr_lat, agr_lon = foco["coords_agricola"]
        _, dist = dist_a_mina_mas_cercana(agr_lat, agr_lon)
        return zona, foco, agr_lat, agr_lon, dist, "area_agricola"

    raise HTTPException(
        422,
        "Se requiere al menos uno de: 'lat'+'lon' (GPS) o 'zona'. "
        "Activa el GPS en la app o selecciona una zona manualmente."
    )


# ─── Endpoints base ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0", "zonas": len(FOCOS_MINEROS)}


@app.get("/api/zones")
def get_zones():
    """Zonas mineras reales de Arequipa con metadatos completos."""
    from src.data_generator import _haversine_km
    zones = []
    for nombre, foco in FOCOS_MINEROS.items():
        ml, mn = foco["coords_mina"]
        al, an = foco["coords_agricola"]
        zones.append({
            "nombre":          nombre,
            "lat":             float(ml),
            "lon":             float(mn),
            "lat_agricola":    float(al),
            "lon_agricola":    float(an),
            "metal":           foco["metal_principal"],
            "metal_idx":       foco["metal_idx"],
            "contaminantes":   foco["contaminantes"],
            "score_base":      foco["score_base"],
            "dist_agr_mina_km": round(_haversine_km(al, an, ml, mn), 1),
            "descripcion":     _descripcion_zona(nombre, foco),
        })
    return {"zones": zones}


def _descripcion_zona(nombre: str, foco: dict) -> str:
    descs = {
        "Cerro Verde (Cu/Mo)":         "Mina de cobre/molibdeno más grande del Perú. Sancionada por OEFA en 2012, 2014 y 2015.",
        "La Joya (As)":                "Arsénico en agua subterránea 4.9× sobre ECA potable. 2,200 usuarios afectados.",
        "Valle del Tambo (As/Cd)":     "Crisis nacional 2021-2024: As 2000% sobre ECA. 13,000 ha de cultivos afectadas.",
        "Caylloma (Pb/Zn)":            "Minera Bateas (Hochschild). Drenaje ácido de mina a afluentes del Colca.",
        "Arcata (Ag/Au - Condesuyos)": "Mina subterránea con relaves. Riesgo de filtración a cuenca del Ocoña.",
    }
    return descs.get(nombre, f"Foco minero — {foco['metal_principal']}")


@app.get("/api/metrics")
def get_metrics():
    return {"metricas": _metricas}


@app.get("/api/eca")
def get_eca():
    return {
        "suelo_agricola_mg_kg": ECA_SUELO_AGRICOLA,
        "agua_riego_mg_l":      ECA_AGUA_RIEGO,
        "norma_suelo":          "DS 011-2017-MINAM (Tabla 1 — Uso Agrícola)",
        "norma_agua":           "DS 004-2017-MINAM Cat. 3 (Riego de vegetales)",
        "norma_aire_pm10_24h":  "100 μg/m³ (DS 003-2017-MINAM)",
    }


@app.get("/api/map-samples")
def get_map_samples(limit: int = 150):
    focos = [
        {"nombre": n, "lat": float(f["coords_mina"][0]), "lon": float(f["coords_mina"][1]),
         "metal": f["metal_principal"]}
        for n, f in FOCOS_MINEROS.items()
    ]
    muestra = _df.sample(min(limit, len(_df)), random_state=1)
    puntos = [
        {
            "lat":    float(r["lat"]),
            "lon":    float(r["lon"]),
            "riesgo": ETIQUETAS[int(r["riesgo"])],
            "color":  COLOR_RIESGO[ETIQUETAS[int(r["riesgo"])]],
        }
        for _, r in muestra.iterrows()
    ]
    return {"focos": focos, "puntos": puntos}


# ─── Visión ────────────────────────────────────────────────────────────────────
@app.post("/api/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analiza foto de hoja con pipeline ExG + HSV + CIELAB.
    Devuelve clorosis_pct y necrosis_pct reales (píxel a píxel).
    """
    contents = await file.read()
    img = leer_imagen_desde_bytes(contents)
    if img is None:
        return {
            "clorosis_pct": 42.0, "necrosis_pct": 18.0,
            "ok": False,
            "msg": "No se pudo decodificar la imagen. Usando estimación de respaldo.",
        }
    return analizar_hoja(img)


# ─── Predicción ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    ph: float
    clorosis_pct: float
    necrosis_pct: float
    # Zona y GPS son ahora ambos opcionales:
    # Si hay GPS → auto-detect; si hay zona → usar zona; si hay ambos → GPS gana
    zona: Optional[str] = None
    lat:  Optional[float] = None
    lon:  Optional[float] = None

    @model_validator(mode="after")
    def check_location(self):
        if self.zona is None and (self.lat is None or self.lon is None):
            raise ValueError(
                "Proporciona 'lat'+'lon' para auto-detección GPS, "
                "o 'zona' para selección manual."
            )
        return self


@app.post("/api/predict")
def predict(req: PredictRequest):
    """
    Predice el riesgo de contaminación.
    La zona se auto-detecta desde GPS si se proporcionan lat/lon.
    """
    zona_nombre, foco, lat, lon, dist, origen = _resolver_zona_y_coords(
        req.zona, req.lat, req.lon
    )
    metal     = foco["metal_principal"]
    metal_idx = foco["metal_idx"]
    pred = M.predecir_riesgo(_modelo, req.clorosis_pct, req.necrosis_pct,
                              dist, req.ph, metal_idx)
    return {
        **pred,
        "zona":             zona_nombre,
        "metal":            metal,
        "contaminantes":    foco["contaminantes"],
        "dist_km":          float(dist),
        "ph":               float(req.ph),
        "clorosis_pct":     float(req.clorosis_pct),
        "necrosis_pct":     float(req.necrosis_pct),
        "lat":              float(lat),
        "lon":              float(lon),
        "color":            COLOR_RIESGO[pred["riesgo_txt"]],
        "ubicacion_origen": origen,
        "score_base_zona":  foco["score_base"],
    }


# ─── Evaluación completa (una sola llamada, todo integrado) ───────────────────
class FullAssessmentRequest(BaseModel):
    ph: float
    clorosis_pct: float
    necrosis_pct: float
    zona: Optional[str] = None
    lat:  Optional[float] = None
    lon:  Optional[float] = None

    @model_validator(mode="after")
    def check_location(self):
        if self.zona is None and (self.lat is None or self.lon is None):
            raise ValueError(
                "Proporciona 'lat'+'lon' para auto-detección GPS, "
                "o 'zona' para selección manual."
            )
        return self


@app.post("/api/full-assessment")
def full_assessment(req: FullAssessmentRequest):
    """
    Evaluación completa en una sola llamada:
    predicción + indicadores + salud + económico + reporte Markdown.
    Ideal para el dashboard de empresa minera.
    """
    zona_nombre, foco, lat, lon, dist, origen = _resolver_zona_y_coords(
        req.zona, req.lat, req.lon
    )
    metal     = foco["metal_principal"]
    metal_idx = foco["metal_idx"]

    # Predicción
    pred = M.predecir_riesgo(_modelo, req.clorosis_pct, req.necrosis_pct,
                              dist, req.ph, metal_idx)
    riesgo = pred["riesgo_txt"]
    score  = pred["confianza"] / 100.0

    # Indicadores ambientales
    ind = estimar_indicadores(riesgo, foco["score_base"], dist, metal, req.ph)

    # Salud humana
    hr = calcular_riesgo_salud(foco["contaminantes"], riesgo, dist)

    # Impacto económico
    ie = calcular_impacto_economico(riesgo, foco["score_base"], dist, zona_nombre)

    # Reporte Markdown integrado
    reporte = generar_reporte(
        zona=zona_nombre, metal=metal, riesgo=riesgo,
        dist=dist, ph=req.ph, clorosis=req.clorosis_pct, necrosis=req.necrosis_pct,
        health_risk=hr, indicadores=ind, impacto_economico=ie,
        confianza=pred["confianza"], probabilidades=pred["probabilidades"],
        lat=lat, lon=lon, ubicacion_origen=origen,
    )

    return {
        "prediccion": {
            **pred,
            "zona":             zona_nombre,
            "metal":            metal,
            "contaminantes":    foco["contaminantes"],
            "dist_km":          float(dist),
            "lat":              float(lat),
            "lon":              float(lon),
            "color":            COLOR_RIESGO[riesgo],
            "ubicacion_origen": origen,
        },
        "indicadores_ambientales": ind,
        "riesgo_salud":            hr,
        "impacto_economico":       ie,
        "reporte_markdown":        reporte,
    }


# ─── Endpoint de reporte independiente ────────────────────────────────────────
class ReportRequest(BaseModel):
    zona: str
    metal: str
    riesgo: str
    dist: float
    ph: float
    clorosis: float
    necrosis: float
    lat:  Optional[float] = None
    lon:  Optional[float] = None
    ubicacion_origen: Optional[str] = None
    api_key: Optional[str] = None   # ignorado — mantenido por compatibilidad


@app.post("/api/report")
def report(req: ReportRequest):
    """
    Genera reporte técnico Markdown. No requiere API key.
    """
    if req.zona not in FOCOS_MINEROS:
        raise HTTPException(400, f"Zona desconocida: {req.zona}")
    foco = FOCOS_MINEROS[req.zona]

    ind = estimar_indicadores(req.riesgo, foco["score_base"], req.dist, req.metal, req.ph)
    hr  = calcular_riesgo_salud(foco["contaminantes"], req.riesgo, req.dist)
    ie  = calcular_impacto_economico(req.riesgo, foco["score_base"], req.dist, req.zona)

    texto = generar_reporte(
        zona=req.zona, metal=req.metal, riesgo=req.riesgo,
        dist=req.dist, ph=req.ph, clorosis=req.clorosis, necrosis=req.necrosis,
        health_risk=hr, indicadores=ind, impacto_economico=ie,
        lat=req.lat, lon=req.lon, ubicacion_origen=req.ubicacion_origen,
    )
    return {"reporte": texto}


# ─── Endpoints especializados para equipo ambiental minero ────────────────────
@app.get("/api/health-risk")
def get_health_risk(
    zona: str = Query(..., description="Nombre exacto de la zona"),
    riesgo: str = Query("Medio", description="Bajo | Medio | Alto"),
    dist_km: float = Query(5.0, description="Distancia al foco minero (km)"),
):
    """Calcula HRI, HQ y riesgo de cáncer para los contaminantes de una zona."""
    if zona not in FOCOS_MINEROS:
        raise HTTPException(400, f"Zona desconocida: {zona}")
    foco = FOCOS_MINEROS[zona]
    return calcular_riesgo_salud(foco["contaminantes"], riesgo, dist_km)


@app.get("/api/economic-impact")
def get_economic_impact(
    zona: str = Query(...),
    riesgo: str = Query("Medio"),
    dist_km: float = Query(5.0),
):
    """Cuantifica impacto económico y pasivos regulatorios para empresa minera."""
    if zona not in FOCOS_MINEROS:
        raise HTTPException(400, f"Zona desconocida: {zona}")
    foco = FOCOS_MINEROS[zona]
    return calcular_impacto_economico(riesgo, foco["score_base"], dist_km, zona)


@app.get("/api/indicators")
def get_indicators(
    zona: str = Query(...),
    riesgo: str = Query("Medio"),
    dist_km: float = Query(5.0),
    ph: float = Query(7.0),
):
    """Indicadores ambientales estimados (CE, MO, NTU, PM10) para una zona."""
    if zona not in FOCOS_MINEROS:
        raise HTTPException(400, f"Zona desconocida: {zona}")
    foco = FOCOS_MINEROS[zona]
    return estimar_indicadores(riesgo, foco["score_base"], dist_km,
                               foco["metal_principal"], ph)


@app.get("/api/compliance-dashboard")
def compliance_dashboard():
    """
    Dashboard de compliance normativo por zona para el equipo ambiental minero.
    Resume el estado estimado de cumplimiento de ECA DS 011, DS 004 y DS 003.
    """
    dashboard = []
    for nombre, foco in FOCOS_MINEROS.items():
        # Calcular para el área agrícola de la zona (distancia base)
        from src.data_generator import _haversine_km
        ml, mn = foco["coords_mina"]
        al, an = foco["coords_agricola"]
        dist_ref = _haversine_km(al, an, ml, mn)

        # Riesgo base de la zona
        sb = foco["score_base"]
        riesgo_ref = "Alto" if sb >= 0.55 else ("Medio" if sb >= 0.25 else "Bajo")

        hr = calcular_riesgo_salud(foco["contaminantes"], riesgo_ref, dist_ref)
        ie = calcular_impacto_economico(riesgo_ref, sb, dist_ref, nombre)

        dashboard.append({
            "zona":            nombre,
            "metal_principal": foco["metal_principal"],
            "contaminantes":   foco["contaminantes"],
            "riesgo_referencia": riesgo_ref,
            "score_base":      sb,
            "dist_agricola_km": round(dist_ref, 1),
            "hri":             hr["hri"],
            "hri_nivel":       hr["hri_nivel"],
            "cancer_risk":     hr["cancer_risk_total"],
            "cancer_nivel":    hr["cancer_risk_nivel"],
            "perdida_economica_usd": ie["impacto_estimado"]["perdida_produccion_usd"],
            "multa_max_oefa_usd":    ie["exposicion_regulatoria"]["multa_max_oefa_usd"],
            "prioridad_accion":      ie["prioridad_accion"],
            "alertas_eca": _alertas_eca(nombre, foco),
        })

    # Ordenar por HRI descendente (mayor riesgo primero)
    dashboard.sort(key=lambda z: -z["hri"])
    return {"zonas": dashboard, "total": len(dashboard)}


def _alertas_eca(nombre: str, foco: dict) -> list:
    alertas = []
    contaminantes = foco["contaminantes"]
    if "As" in contaminantes and nombre == "Valle del Tambo (As/Cd)":
        alertas.append("⚠️ As documentado 2000% sobre ECA agua (GORE AQP 2024)")
    if "As" in contaminantes and nombre == "La Joya (As)":
        alertas.append("⚠️ As agua subterránea 4.9× ECA potable (0.049 mg/L vs 0.01)")
    if "Cu" in contaminantes:
        alertas.append("📋 Cu en efluentes — verificar ECA DS 004-2017 Cat.3 (0.20 mg/L)")
    if "Pb" in contaminantes:
        alertas.append("📋 Pb — verificar ECA suelo DS 011-2017 (70 mg/kg)")
    if not alertas:
        alertas.append("✅ Sin excedencias confirmadas documentadas públicamente")
    return alertas
