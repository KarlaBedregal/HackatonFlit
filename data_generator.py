"""
data_generator.py  (Dev 1 — Datos & Modelo)
--------------------------------------------
Genera el dataset de riesgo de contaminacion por metales pesados en zonas
agricolas de Arequipa.

Modos de dataset:
  1) generar_dataset()             -> fallback 100% sintetico (calibrado con rangos reales)
  2) generar_dataset_hibrido()     -> RECOMENDADO: anclas ICP-MS reales + bootstrap
  3) generar_dataset_entrenamiento() -> alias principal usado por model.py

Fuente principal (muestras_reales_arequipa.csv):
  Zegarra-Aymara, L. et al. (2025). Acumulacion de metales pesados en zonas
  vitivinicolas de la region de Arequipa. Rev. Chapingo Ser. Horticultura, 31.
  https://doi.org/10.5154/r.rchsh.2024.11.016
  4 sitios 2023: La Joya, Majes Tradicion, CIEPA-Majes, San Isidro.
  Coordenadas, pH y 8 metales (ICP-MS, media +/- DE) por sitio.

pH por zona (lookup sin API):
  Valores representativos de la misma region / estudios citados en README del
  hackaton (La Joya, Tambo/Islay, suelos agricolas cerca de Cerro Verde).
  Dev 4 puede llamar obtener_ph_por_zona() al elegir zona en la UI.

Limitacion honesta:
  Solo hay 4 sitios con quimica de suelo completa. Clorosis/necrosis siguen
  siendo proxy fisiologico (no hay fotos pareadas). El bootstrap amplia N para
  el Random Forest sin inventar sitios nuevos: solo varia dentro de la DE real.
"""

import os

import numpy as np
import pandas as pd

# --- Coordenadas de focos de riesgo minero/hidrico en Arequipa ---
ZONAS_RIESGO = {
    "Cerro Verde (Cu/As)": (-16.5389, -71.6017),
    "La Joya (As agua)": (-16.5847, -71.9186),
    "Valle del Tambo (As)": (-17.0500, -71.7500),  # Cocachacra / Islay
}

CENTRO_AREQUIPA = (-16.409, -71.537)

# Proyectos mineros fijos — distancia en app via geopy
PROYECTOS_MINEROS = {
    "Cerro Verde": {
        "lat": -16.5389,
        "lon": -71.6017,
        "tipo": "Mina operativa (Cu/As)",
        "metal": "Cobre",
    },
    "Tia Maria": {
        "lat": -16.4353,
        "lon": -71.8125,
        "tipo": "Proyecto minero (Cu) — Cocachacra, Islay",
        "metal": "Cobre",
    },
}

# Zonas agricolas para dropdown de ubicacion de parcela
ZONAS_PARCELA = {
    "Campina Yura / Arequipa": {
        "lat": -16.4090,
        "lon": -71.5370,
        "zona_ph": "Cerro Verde (Cu/As)",
        "fuente": "Centro agricola Arequipa",
    },
    "La Joya (vinedos)": {
        "lat": -16.4575,
        "lon": -71.8046,
        "zona_ph": "La Joya (As agua)",
        "fuente": "Zegarra-Aymara et al. 2025 — ICP-MS",
    },
    "San Isidro (campo)": {
        "lat": -16.5705,
        "lon": -71.9294,
        "zona_ph": "La Joya (As agua)",
        "fuente": "Zegarra-Aymara et al. 2025 — ICP-MS",
    },
    "Valle de Majes (CIEPA)": {
        "lat": -16.3310,
        "lon": -72.2229,
        "zona_ph": "Valle del Tambo (As)",
        "fuente": "Zegarra-Aymara et al. 2025 — ICP-MS",
    },
    "Majes Tradicion": {
        "lat": -16.2259,
        "lon": -72.4434,
        "zona_ph": "Valle del Tambo (As)",
        "fuente": "Zegarra-Aymara et al. 2025 — ICP-MS",
    },
}

# Parcela agricola representativa por zona (compatibilidad entrenamiento)
COORDS_PARCELA_POR_ZONA = {
    "Cerro Verde (Cu/As)": {
        "lat": -16.4090,
        "lon": -71.5370,
        "sitio": "Campina de Yura (referencia agricola)",
        "fuente": "Centro agricola Arequipa; parcela al NE de Cerro Verde",
    },
    "La Joya (As agua)": {
        "lat": -16.4575,
        "lon": -71.8046,
        "sitio": "La Joya (vinedo)",
        "fuente": "Zegarra-Aymara et al. 2025, muestra real ICP-MS",
    },
    "Valle del Tambo (As)": {
        "lat": -16.3310,
        "lon": -72.2229,
        "sitio": "CIEPA-Majes",
        "fuente": "Zegarra-Aymara et al. 2025, muestra real ICP-MS",
    },
}

FEATURES = ["clorosis_pct", "necrosis_pct", "dist_mina_km", "ph_suelo"]
ETIQUETAS = {0: "Bajo", 1: "Medio", 2: "Alto"}

# pH representativo por zona — lookup local (sin API externa)
# Fuentes: medias del paper Zegarra-Aymara 2025 (La Joya/Majes) + rangos
# regionales citados en pitch (suelos alcalinos costa sur, OEFA mina Cerro Verde).
PH_POR_ZONA = {
    "Cerro Verde (Cu/As)": {
        "ph_representativo": 7.52,
        "ph_min": 7.10,
        "ph_max": 7.90,
        "de_medicion": 0.12,
        "metal_principal": "Cobre",
        "fuente": (
            "Suelos agricolas alcalinos Yura-Arequipa; campiña irrigada "
            "(rango 7.1-7.9). Contexto minero Cerro Verde — OEFA."
        ),
    },
    "La Joya (As agua)": {
        "ph_representativo": 7.39,
        "ph_min": 7.30,
        "ph_max": 7.58,
        "de_medicion": 0.05,
        "metal_principal": "Arsenico",
        "fuente": (
            "Zegarra-Aymara et al. 2025: La Joya (vinedo) pH=7.39, "
            "San Isidro pH=7.58; agua La Joya ~0.021 mg/L As (pitch)."
        ),
    },
    "Valle del Tambo (As)": {
        "ph_representativo": 7.40,
        "ph_min": 7.36,
        "ph_max": 7.42,
        "de_medicion": 0.05,
        "metal_principal": "Arsenico",
        "fuente": (
            "Zegarra-Aymara et al. 2025: CIEPA-Majes pH=7.36, "
            "Majes Tradicion pH=7.42; zona Islay/Cocachacra (As en agua)."
        ),
    },
}

# Mapeo sitio de campo -> foco de riesgo del MVP (para trazabilidad en pitch)
MAPEO_SITIO_A_ZONA = {
    "La Joya (vinedo)": "La Joya (As agua)",
    "San Isidro": "La Joya (As agua)",
    "Majes Tradicion": "Valle del Tambo (As)",
    "CIEPA-Majes": "Valle del Tambo (As)",
}

FUENTES_DATOS = {
    "muestras_campo": {
        "archivo": "muestras_reales_arequipa.csv",
        "n_sitios": 4,
        "metodo": "ICP-MS suelo (media +/- DE, 3 repeticiones)",
        "referencia": "Zegarra-Aymara et al. 2025, DOI 10.5154/r.rchsh.2024.11.016",
    },
    "ph_por_zona": {
        "metodo": "Lookup local PH_POR_ZONA (sin API)",
        "referencia": "Paper UNSA 2025 + rangos regionales del pitch",
    },
    "riesgo": {
        "metodo": "Indice REP (Hakanson 1980) sobre metales reales",
        "referencia": "Mismos umbrales que el paper (95 / 190 / 380)",
    },
    "clorosis_necrosis": {
        "metodo": "Proxy fisiologico anclado a As_mgkg y REP real",
        "referencia": "Pendiente: fotos pareadas en campo (Dev 2)",
    },
}


def _buscar_csv_real():
    aqui = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(aqui, "muestras_reales_arequipa.csv"),
        os.path.join(aqui, "data", "muestras_reales_arequipa.csv"),
        os.path.join(os.path.dirname(aqui), "muestras_reales_arequipa.csv"),
        os.path.join(os.path.dirname(aqui), "data", "muestras_reales_arequipa.csv"),
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    return candidatos[0]


RUTA_CSV_REAL = _buscar_csv_real()

# REP — Hakanson 1980
C_REF = {"as": 1.5, "b": 15, "cd": 0.098, "cu": 25, "fe": 35000, "hg": 0.06, "pb": 20, "zn": 71}
TR = {"as": 10, "b": 2, "cd": 30, "cu": 5, "fe": 1, "hg": 40, "pb": 5, "zn": 1}

# Techo As observado en los 4 sitios reales (para normalizar estres foliar)
AS_MAX_REAL = 5.016


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def dist_a_mina_mas_cercana(lat, lon):
    """Devuelve (nombre_zona, distancia_km) del foco minero/hidrico mas cercano."""
    mejor_nombre, mejor_dist = None, np.inf
    for nombre, (mlat, mlon) in ZONAS_RIESGO.items():
        d = _haversine_km(lat, lon, mlat, mlon)
        if d < mejor_dist:
            mejor_nombre, mejor_dist = nombre, d
    return mejor_nombre, round(mejor_dist, 2)


def calcular_distancia_proyecto(lat: float, lon: float) -> dict:
    """
    Distancia geodesica (geopy) desde la parcela al proyecto minero fijo
    mas cercano: Cerro Verde o Tia Maria.
    """
    from geopy.distance import geodesic

    origen = (lat, lon)
    distancias = {}
    for nombre, info in PROYECTOS_MINEROS.items():
        distancias[nombre] = round(geodesic(origen, (info["lat"], info["lon"])).kilometers, 2)

    proyecto_cercano = min(distancias, key=distancias.get)
    info_proj = PROYECTOS_MINEROS[proyecto_cercano]
    return {
        "proyecto_cercano": proyecto_cercano,
        "dist_km": distancias[proyecto_cercano],
        "metal_proyecto": info_proj["metal"],
        "tipo_proyecto": info_proj["tipo"],
        "distancias_todos": distancias,
        "lat_parcela": lat,
        "lon_parcela": lon,
    }


def zona_parcela_desde_coordenadas(lat: float, lon: float) -> dict:
    """Zona agricola mas cercana (para lookup de pH tras clic en mapa)."""
    mejor_nombre, mejor_dist = None, np.inf
    for nombre, info in ZONAS_PARCELA.items():
        d = _haversine_km(lat, lon, info["lat"], info["lon"])
        if d < mejor_dist:
            mejor_nombre, mejor_dist = nombre, d
    info = ZONAS_PARCELA[mejor_nombre]
    return {
        "nombre_zona": mejor_nombre,
        "zona_ph": info["zona_ph"],
        "dist_a_referencia_km": round(float(mejor_dist), 2),
        "fuente": info["fuente"],
    }


def coords_parcela_desde_zona(nombre_zona: str) -> dict:
    """Coordenadas y metadata de una zona del dropdown."""
    if nombre_zona not in ZONAS_PARCELA:
        raise KeyError(
            f"Zona '{nombre_zona}' no reconocida. Opciones: {list(ZONAS_PARCELA.keys())}"
        )
    info = ZONAS_PARCELA[nombre_zona]
    return {
        "lat": info["lat"],
        "lon": info["lon"],
        "nombre_zona": nombre_zona,
        "zona_ph": info["zona_ph"],
        "fuente": info["fuente"],
    }


def info_parcela_por_zona(zona: str) -> dict:
    """Compatibilidad: usa calcular_distancia_proyecto (geopy) en lugar de focos viejos."""
    if zona in ZONAS_PARCELA:
        p = coords_parcela_desde_zona(zona)
        dist_info = calcular_distancia_proyecto(p["lat"], p["lon"])
        return {
            "lat": p["lat"],
            "lon": p["lon"],
            "sitio_referencia": p["nombre_zona"],
            "fuente_coords": p["fuente"],
            "foco_mas_cercano": dist_info["proyecto_cercano"],
            "dist_mina_km": dist_info["dist_km"],
        }
    if zona not in COORDS_PARCELA_POR_ZONA:
        raise KeyError(f"Zona '{zona}' no reconocida.")
    info = COORDS_PARCELA_POR_ZONA[zona]
    dist_info = calcular_distancia_proyecto(info["lat"], info["lon"])
    return {
        "lat": info["lat"],
        "lon": info["lon"],
        "sitio_referencia": info["sitio"],
        "fuente_coords": info["fuente"],
        "foco_mas_cercano": dist_info["proyecto_cercano"],
        "dist_mina_km": dist_info["dist_km"],
    }


def obtener_ph_por_zona(zona: str, variacion: bool = False, seed=None) -> float:
    """
    Lookup de pH por zona geografica del MVP (sin API externa).

    Parametros
    ----------
    zona : clave de ZONAS_RIESGO o PH_POR_ZONA
    variacion : si True, muestrea dentro del rango documentado (bootstrap suave)
    seed : semilla opcional para reproducibilidad
    """
    if zona not in PH_POR_ZONA:
        raise KeyError(
            f"Zona '{zona}' no reconocida. Opciones: {list(PH_POR_ZONA.keys())}"
        )
    info = PH_POR_ZONA[zona]
    ph_base = info["ph_representativo"]
    if not variacion:
        return round(ph_base, 2)

    rng = np.random.default_rng(seed)
    ph = rng.normal(ph_base, info["de_medicion"])
    ph = np.clip(ph, info["ph_min"], info["ph_max"])
    return round(float(ph), 2)


def obtener_ph_por_coordenadas(lat: float, lon: float, variacion: bool = False) -> dict:
    """
    Simula 'pH por geolocalizacion': elige el foco mas cercano y devuelve pH lookup.
    """
    zona, dist = dist_a_mina_mas_cercana(lat, lon)
    ph = obtener_ph_por_zona(zona, variacion=variacion)
    return {
        "ph_suelo": ph,
        "zona": zona,
        "dist_mina_km": dist,
        "metal_principal": PH_POR_ZONA[zona]["metal_principal"],
        "fuente_ph": PH_POR_ZONA[zona]["fuente"],
    }


def resumen_fuentes_datos() -> dict:
    """Trazabilidad para pitch / metricas del modelo."""
    return {
        **FUENTES_DATOS,
        "ph_por_zona_valores": {
            z: v["ph_representativo"] for z, v in PH_POR_ZONA.items()
        },
        "csv_real_ruta": RUTA_CSV_REAL,
        "csv_real_existe": os.path.exists(RUTA_CSV_REAL),
    }


def _rep_a_riesgo(rep: float) -> int:
    if rep < 95:
        return 0
    if rep < 190:
        return 1
    return 2


def _calcular_rep(fila: dict) -> float:
    rep = 0.0
    for metal, cref in C_REF.items():
        c = fila.get(f"{metal}_mgkg")
        if c is None:
            continue
        rep += TR[metal] * (c / cref)
    return rep


def _estres_foliar_desde_quimica(as_mgkg: float, rep: float) -> float:
    """
    Proxy 0-1 de estres foliar anclado a quimica REAL del suelo:
    - 60% peso en As medido (metal clave del pitch en La Joya/Tambo)
    - 40% peso en REP (indice ecologico del paper)
    """
    estres_as = np.clip((as_mgkg - C_REF["as"]) / (AS_MAX_REAL - C_REF["as"]), 0, 1)
    estres_rep = np.clip(rep / 380.0, 0, 1)
    return float(0.6 * estres_as + 0.4 * estres_rep)


def _simular_sintomas_foliares(estres: float, rng) -> tuple[float, float]:
    """Clorosis/necrosis como proxy hasta que Dev 2 paree fotos reales."""
    clorosis = np.clip(rng.normal(12 + 58 * estres, 7), 0, 100)
    necrosis = np.clip(rng.normal(6 + 42 * estres, 6), 0, 100)
    return round(float(clorosis), 2), round(float(necrosis), 2)


def _fila_real_a_features(fila: pd.Series, rng=None) -> dict:
    """Convierte una fila ICP-MS real al formato de entrenamiento del RF."""
    rng = rng or np.random.default_rng(0)
    metales = {f"{m}_mgkg": fila[f"{m}_mgkg"] for m in C_REF}
    rep = _calcular_rep(metales)
    riesgo = _rep_a_riesgo(rep)
    estres = _estres_foliar_desde_quimica(fila["as_mgkg"], rep)
    clorosis, necrosis = _simular_sintomas_foliares(estres, rng)
    zona_foco = MAPEO_SITIO_A_ZONA.get(fila["zona"], dist_a_mina_mas_cercana(fila["lat"], fila["lon"])[0])

    return {
        "clorosis_pct": clorosis,
        "necrosis_pct": necrosis,
        "dist_mina_km": dist_a_mina_mas_cercana(fila["lat"], fila["lon"])[1],
        "ph_suelo": round(float(fila["ph_suelo"]), 2),
        "lat": round(float(fila["lat"]), 5),
        "lon": round(float(fila["lon"]), 5),
        "zona_cercana": zona_foco,
        "sitio_campo": fila["zona"],
        "rep_indice": round(float(rep), 2),
        "as_mgkg": round(float(fila["as_mgkg"]), 4),
        "riesgo": int(riesgo),
        "tipo_muestra": "real",
        "fuente": fila.get("fuente", "Zegarra-Aymara_et_al_2025_UNSA"),
    }


# ---------------------------------------------------------------------------
# MODO 1: sintetico calibrado (fallback)
# ---------------------------------------------------------------------------
def generar_dataset(n=600, seed=42):
    """
    Fallback 100% sintetico. pH y geo calibrados con PH_POR_ZONA y ZONAS_RIESGO.
    Usar solo si falta el CSV real.
    """
    rng = np.random.default_rng(seed)
    filas = []

    for _ in range(n):
        zona = rng.choice(list(ZONAS_RIESGO.keys()))
        base_lat, base_lon = ZONAS_RIESGO[zona]
        lat = base_lat + rng.normal(0, 0.08)
        lon = base_lon + rng.normal(0, 0.08)
        _, dist = dist_a_mina_mas_cercana(lat, lon)
        ph = obtener_ph_por_zona(zona, variacion=True, seed=rng.integers(1e9))

        cercania = np.clip(1 - dist / 45, 0, 1)
        info = PH_POR_ZONA[zona]
        as_proxy = C_REF["as"] + cercania * (AS_MAX_REAL - C_REF["as"])
        rep_proxy = _calcular_rep({"as_mgkg": as_proxy, "cu_mgkg": 20 + 15 * cercania})
        estres = _estres_foliar_desde_quimica(as_proxy, rep_proxy)
        clorosis, necrosis = _simular_sintomas_foliares(estres, rng)
        riesgo = _rep_a_riesgo(rep_proxy)

        filas.append({
            "clorosis_pct": clorosis,
            "necrosis_pct": necrosis,
            "dist_mina_km": dist,
            "ph_suelo": ph,
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "zona_cercana": zona,
            "sitio_campo": "sintetico",
            "rep_indice": round(rep_proxy, 2),
            "as_mgkg": round(as_proxy, 4),
            "riesgo": riesgo,
            "tipo_muestra": "sintetico",
            "fuente": f"sintetico_calibrado_{info['metal_principal']}",
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# MODO 2: hibrido anclado a ICP-MS real (RECOMENDADO)
# ---------------------------------------------------------------------------
def cargar_muestras_reales(ruta=RUTA_CSV_REAL) -> pd.DataFrame:
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No encuentro 'muestras_reales_arequipa.csv'. Ruta esperada: {ruta}"
        )
    df = pd.read_csv(ruta)
    df["dist_mina_km"] = df.apply(lambda r: dist_a_mina_mas_cercana(r["lat"], r["lon"])[1], axis=1)
    df["zona_foco"] = df["zona"].map(MAPEO_SITIO_A_ZONA).fillna(
        df.apply(lambda r: dist_a_mina_mas_cercana(r["lat"], r["lon"])[0], axis=1)
    )
    df["rep_indice"] = df.apply(
        lambda r: _calcular_rep({f"{m}_mgkg": r[f"{m}_mgkg"] for m in C_REF}), axis=1
    )
    df["riesgo"] = df["rep_indice"].apply(_rep_a_riesgo)
    return df


def generar_filas_reales(seed=42) -> pd.DataFrame:
    """Las 4 filas reales convertidas al schema de entrenamiento (verificables)."""
    rng = np.random.default_rng(seed)
    reales = cargar_muestras_reales()
    return pd.DataFrame([_fila_real_a_features(row, rng) for _, row in reales.iterrows()])


def generar_dataset_hibrido(n_por_zona: int = 60, seed: int = 42,
                            ruta_csv: str = RUTA_CSV_REAL,
                            incluir_reales: bool = True) -> pd.DataFrame:
    """
    Dataset semi-real para entrenar el RF:
      - incluye las 4 muestras ICP-MS reales (tipo_muestra='real')
      - genera n_por_zona variantes bootstrap dentro de la DE reportada
      - pH: valor medido del sitio (no lookup) en filas reales/bootstrap
      - riesgo: REP recalculado por fila (Hakanson 1980)
    """
    reales = cargar_muestras_reales(ruta_csv)
    rng = np.random.default_rng(seed)
    filas = []

    if incluir_reales:
        for _, fila in reales.iterrows():
            filas.append(_fila_real_a_features(fila, rng))

    for _, fila in reales.iterrows():
        ph_sd = PH_POR_ZONA.get(
            MAPEO_SITIO_A_ZONA.get(fila["zona"], ""),
            {"de_medicion": 0.05},
        ).get("de_medicion", 0.05)

        for _ in range(n_por_zona):
            lat = fila["lat"] + rng.normal(0, 0.0005)
            lon = fila["lon"] + rng.normal(0, 0.0005)
            ph = np.clip(fila["ph_suelo"] + rng.normal(0, ph_sd), 4.5, 9.0)

            metales_boot = {}
            for m in C_REF:
                metales_boot[f"{m}_mgkg"] = max(0.0, rng.normal(fila[f"{m}_mgkg"], fila[f"{m}_sd"]))

            rep = _calcular_rep(metales_boot)
            riesgo = _rep_a_riesgo(rep)
            estres = _estres_foliar_desde_quimica(metales_boot["as_mgkg"], rep)
            clorosis, necrosis = _simular_sintomas_foliares(estres, rng)
            zona_foco = MAPEO_SITIO_A_ZONA.get(
                fila["zona"], dist_a_mina_mas_cercana(lat, lon)[0]
            )

            filas.append({
                "clorosis_pct": clorosis,
                "necrosis_pct": necrosis,
                "dist_mina_km": dist_a_mina_mas_cercana(lat, lon)[1],
                "ph_suelo": round(float(ph), 2),
                "lat": round(float(lat), 5),
                "lon": round(float(lon), 5),
                "zona_cercana": zona_foco,
                "sitio_campo": fila["zona"],
                "rep_indice": round(float(rep), 2),
                "as_mgkg": round(float(metales_boot["as_mgkg"]), 4),
                "riesgo": int(riesgo),
                "tipo_muestra": "bootstrap",
                "fuente": fila.get("fuente", "Zegarra-Aymara_et_al_2025_UNSA"),
                **{k: round(float(v), 4) for k, v in metales_boot.items()},
            })

    return pd.DataFrame(filas)


def generar_dataset_entrenamiento(n_por_zona: int = 60, seed: int = 42,
                                  ruta_csv: str = RUTA_CSV_REAL) -> pd.DataFrame:
    """Punto de entrada principal para model.py (modo hibrido recomendado)."""
    return generar_dataset_hibrido(
        n_por_zona=n_por_zona, seed=seed, ruta_csv=ruta_csv, incluir_reales=True
    )


def exportar_dataset(ruta="dataset_arequipa_hibrido.csv", **kwargs) -> str:
    """Genera y guarda el CSV con metadatos de trazabilidad."""
    df = generar_dataset_entrenamiento(**kwargs)
    df.to_csv(ruta, index=False)
    return ruta


if __name__ == "__main__":
    print("=== Fuentes de datos ===")
    for k, v in resumen_fuentes_datos().items():
        print(f"  {k}: {v}")

    print("\n=== pH lookup por zona (sin API) ===")
    for zona in ZONAS_RIESGO:
        info = obtener_ph_por_coordenadas(*ZONAS_RIESGO[zona])
        print(f"  {zona}: pH={info['ph_suelo']} | metal={info['metal_principal']}")

    print("\n=== Muestras reales ICP-MS (anclas verificables) ===")
    print(cargar_muestras_reales()[["zona", "ph_suelo", "as_mgkg", "dist_mina_km", "rep_indice", "riesgo"]])

    print("\n=== Dataset de entrenamiento ===")
    df = generar_dataset_entrenamiento()
    print(df[FEATURES + ["riesgo", "tipo_muestra"]].head(8))
    print("\nConteo por tipo:")
    print(df["tipo_muestra"].value_counts())
    print("\nDistribucion de clases:")
    print(df["riesgo"].value_counts().sort_index())

    ruta = exportar_dataset()
    print(f"\nGuardado {ruta} ({len(df)} filas, {df['tipo_muestra'].eq('real').sum()} reales)")
