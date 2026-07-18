"""
data_generator.py
------------------
Genera un dataset simulado REALISTA de riesgo de contaminacion por metales
pesados (arsenico / cobre) en zonas agricolas de Arequipa.

Los rangos estan "sembrados" a partir de estudios reales:
- La Joya: arsenico en agua subterranea ~0.021 mg/L (duplica el ECA de 0.01)
- Valle del Tambo / Islay: arsenico superficial documentado
- Cerro Verde: cobre + arsenico, monitoreo observado por OEFA

IMPORTANTE PARA EL PITCH: son datos sinteticos calibrados con rangos de
estudios publicados. En produccion se reemplazan por muestreo primario.
"""

import numpy as np
import pandas as pd

# --- Coordenadas reales (lat, lon) de focos de riesgo en Arequipa ---
ZONAS_RIESGO = {
    "Cerro Verde (Cu/As)":  (-16.5389, -71.6017),
    "La Joya (As agua)":    (-16.5847, -71.9186),
    "Valle del Tambo (As)": (-17.0500, -71.7500),  # Cocachacra / Islay
}

# Centro aproximado de la campiña agricola de Arequipa
CENTRO_AREQUIPA = (-16.409, -71.537)


def _haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en km entre dos puntos geograficos."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def dist_a_mina_mas_cercana(lat, lon):
    """Devuelve (nombre_zona, distancia_km) de la mina/foco mas cercano."""
    mejor_nombre, mejor_dist = None, np.inf
    for nombre, (mlat, mlon) in ZONAS_RIESGO.items():
        d = _haversine_km(lat, lon, mlat, mlon)
        if d < mejor_dist:
            mejor_nombre, mejor_dist = nombre, d
    return mejor_nombre, round(mejor_dist, 2)


def generar_dataset(n=600, seed=42):
    """
    Genera un DataFrame con features y la etiqueta de riesgo.

    Features:
      - clorosis_pct   : % de amarillamiento en la hoja (vision)
      - necrosis_pct   : % de manchas/necrosis en la hoja (vision)
      - dist_mina_km   : distancia al foco minero mas cercano
      - ph_suelo       : pH del suelo (acido favorece movilidad del As)
      - lat, lon       : ubicacion (para el mapa)
    Target:
      - riesgo : 0=Bajo, 1=Medio, 2=Alto
    """
    rng = np.random.default_rng(seed)
    filas = []

    for _ in range(n):
        # Elegimos un foco base y dispersamos alrededor
        zona = rng.choice(list(ZONAS_RIESGO.keys()))
        base_lat, base_lon = ZONAS_RIESGO[zona]
        lat = base_lat + rng.normal(0, 0.15)
        lon = base_lon + rng.normal(0, 0.15)

        _, dist = dist_a_mina_mas_cercana(lat, lon)

        # pH: suelos de la zona tienden a ser neutro-alcalino, con variacion
        ph = np.clip(rng.normal(7.0, 0.9), 4.5, 9.0)

        # Sintomas visuales de estres (correlacionados con cercania a la mina)
        cercania_factor = np.clip(1 - dist / 40, 0, 1)  # 0 lejos, 1 cerca
        clorosis = np.clip(rng.normal(15 + 45 * cercania_factor, 12), 0, 100)
        necrosis = np.clip(rng.normal(8 + 35 * cercania_factor, 10), 0, 100)

        # --- Regla latente para generar la etiqueta (no la ve el modelo) ---
        # Riesgo sube con: cercania, sintomas visuales y pH acido
        score = (
            0.45 * cercania_factor
            + 0.0045 * clorosis
            + 0.0045 * necrosis
            + 0.10 * (ph < 6.0)
        )
        score += rng.normal(0, 0.06)  # ruido para que no sea trivial

        if score < 0.35:
            riesgo = 0  # Bajo
        elif score < 0.60:
            riesgo = 1  # Medio
        else:
            riesgo = 2  # Alto

        filas.append({
            "clorosis_pct": round(clorosis, 2),
            "necrosis_pct": round(necrosis, 2),
            "dist_mina_km": dist,
            "ph_suelo": round(ph, 2),
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "zona_cercana": _,  # placeholder, se recalcula abajo
            "riesgo": riesgo,
        })

    df = pd.DataFrame(filas)
    # Recalcular nombre de zona cercana de forma limpia
    df["zona_cercana"] = df.apply(
        lambda r: dist_a_mina_mas_cercana(r["lat"], r["lon"])[0], axis=1
    )
    return df


FEATURES = ["clorosis_pct", "necrosis_pct", "dist_mina_km", "ph_suelo"]
ETIQUETAS = {0: "Bajo", 1: "Medio", 2: "Alto"}


if __name__ == "__main__":
    df = generar_dataset()
    print(df.head())
    print("\nDistribucion de clases:")
    print(df["riesgo"].value_counts().sort_index())
    df.to_csv("dataset_arequipa.csv", index=False)
    print("\nGuardado dataset_arequipa.csv")
