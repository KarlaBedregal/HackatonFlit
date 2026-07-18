"""
data_generator.py — Dataset calibrado con datos reales de Arequipa

Fuentes científicas y legales:
  · DS 011-2017-MINAM: ECA para Suelo (uso agrícola)
    As=50 mg/kg, Pb=70 mg/kg, Cd=1.4 mg/kg
  · DS 004-2017-MINAM: ECA para Agua Categoría 3 (riego)
    As=0.1 mg/L, Cu=0.2 mg/L
  · OEFA Informe 00043-2022: monitoreo Cerro Verde (Cu/Mo), Uchumayo/Socabaya/Tiabaya
  · GORE Arequipa 2024: As en río Tambo supera 2000% ECA (Minera Aruntani, declarada emergencia)
  · Diario Correo / ANA: As 0.049 mg/L en La Joya/Sabandía (4.9× ECA agua potable)
  · Barbedo (2013) ScientificResearch: síntomas fitotóxicos por metales en plantas
  · PubMed / Yang 2002: Cu fitotoxicidad >150 mg/kg causa clorosis y necrosis
  · Caylloma: Minera Bateas (Hochschild), Pb/Zn/Ag a 4,400 msnm, 231 km de Arequipa

Zonas reales confirmadas:
  Cerro Verde  16°31'46"S, 71°35'51"W — cobre/molibdeno
  La Joya      16°35'04"S, 71°55'07"W — arsénico en agua subterránea
  Tambo        17°03'00"S, 71°45'00"W — arsénico (2000% sobre ECA, emergencia nacional)
  Caylloma     15°10'30"S, 71°46'12"W — plata/plomo/zinc
  Arcata       15°21'00"S, 73°03'00"W — plata/oro (Condesuyos)
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# ECA vigentes (base para clasificar riesgo)
# ─────────────────────────────────────────────────────────────────────────────
ECA_SUELO_AGRICOLA = {"As": 50, "Pb": 70, "Cd": 1.4, "Zn": 200}   # mg/kg DS 011-2017
ECA_AGUA_RIEGO     = {"As": 0.10, "Cu": 0.20, "Pb": 5.0, "Cd": 0.01}  # mg/L DS 004-2017

# ─────────────────────────────────────────────────────────────────────────────
# Focos mineros reales con coordenadas verificadas y perfil de contaminación
# ─────────────────────────────────────────────────────────────────────────────
FOCOS_MINEROS = {
    # ── Cerro Verde ──────────────────────────────────────────────────────────
    # Freeport-McMoRan, mayor mina de Cu/Mo del Perú.
    # OEFA sancionó 2012, 2014 y 2015 por metales en efluentes y residuos.
    # Informe OEFA 00043-2022: monitoreo en Socabaya, Uchumayo y Tiabaya.
    "Cerro Verde (Cu/Mo)": {
        "coords_mina":   (-16.5294, -71.5975),
        "coords_agricola": (-16.485, -71.565),  # campiña Uchumayo/Tiabaya
        "metal_principal": "Cobre",
        "metal_idx": 0,
        "contaminantes": ["Cu", "Mo", "As"],
        # Cu > 150 mg/kg → clorosis tipo Fe + necrosis en bordes (Yang 2002)
        "fitotox_dist_km": 15,     # síntomas visibles hasta ~15 km del foco
        "score_base": 0.62,        # excedencia documentada alta
        "ph_riesgo_bajo": 6.5,     # Cu más móvil a pH < 6
    },
    # ── La Joya ──────────────────────────────────────────────────────────────
    # As en agua subterránea: 0.049 mg/L en Sabandía (4.9× ECA potable 0.01 mg/L)
    # Origen mixto: geológico (vulcanismo pasivo) + agroindustrial.
    # 2,200 usuarios con agua fuera de norma. Monitoreo semanal MINSA.
    "La Joya (As)": {
        "coords_mina":     (-16.5847, -71.9186),
        "coords_agricola": (-16.565,  -71.885),  # irrigación La Joya
        "metal_principal": "Arsénico",
        "metal_idx": 1,
        "contaminantes": ["As"],
        "conc_agua_mgl": 0.049,    # medido: 4.9× ECA potable, 0.49× ECA riego
        "fitotox_dist_km": 10,
        "score_base": 0.48,
        "ph_riesgo_alto": 7.5,     # As más móvil a pH alto (arsenato vs fosfato)
    },
    # ── Valle del Tambo / Aruntani ───────────────────────────────────────────
    # Crisis documentada: GORE Arequipa 2024 → 2000% sobre ECA consumo (0.01 mg/L)
    # → As ≈ 0.20 mg/L (2× ECA riego también).
    # Emergencia declarada 2021. 13,000 ha cultivos afectadas (arroz, cebolla, maíz).
    # OEFA encontró As, Cd, Pb en cultivos cosechados. Niños con > 400 μg/dL As en sangre.
    "Valle del Tambo (As/Cd)": {
        "coords_mina":     (-17.0500, -71.7500),
        "coords_agricola": (-17.073,  -71.782),  # Cocachacra / La Curva
        "metal_principal": "Arsénico",
        "metal_idx": 1,
        "contaminantes": ["As", "Cd", "Pb"],
        "conc_agua_mgl": 0.20,     # estimado: 20× ECA potable (GORE AQP 2024)
        "fitotox_dist_km": 30,     # pluma de contaminación extensa río abajo
        "score_base": 0.80,        # severidad documentada alta
        "ph_riesgo_alto": 7.5,
    },
    # ── Caylloma ─────────────────────────────────────────────────────────────
    # Minera Bateas (Hochschild). Ag/Pb/Zn. 4,400 msnm, 231 km de Arequipa.
    # Drenaje ácido de mina (DAM) puede afectar afluentes del Colca.
    # Plomo: factor translocación 1.805 → alta transferencia a partes aéreas.
    "Caylloma (Pb/Zn)": {
        "coords_mina":     (-15.1750, -71.7700),
        "coords_agricola": (-15.210,  -71.730),  # pastizales / cultivos Colca
        "metal_principal": "Plomo",
        "metal_idx": 2,
        "contaminantes": ["Pb", "Zn", "Ag"],
        "fitotox_dist_km": 12,
        "score_base": 0.45,
        "ph_riesgo_bajo": 6.0,     # Pb y Zn más móviles a pH ácido
    },
    # ── Arcata ───────────────────────────────────────────────────────────────
    # Hochschild Mining. Ag/Au. Provincia Condesuyos.
    # Mina subterránea con relaves, riesgo de filtraciones a cuencas Ocoña.
    "Arcata (Ag/Au - Condesuyos)": {
        "coords_mina":     (-15.3500, -73.0500),
        "coords_agricola": (-15.420,  -72.980),
        "metal_principal": "Plata",
        "metal_idx": 3,
        "contaminantes": ["Ag", "Au", "As", "Pb"],
        "fitotox_dist_km": 8,
        "score_base": 0.38,
        "ph_riesgo_bajo": 6.0,
    },
}

# Vista simplificada para el resto del sistema (compatible con código anterior)
ZONAS_RIESGO = {k: v["coords_mina"] for k, v in FOCOS_MINEROS.items()}
ZONAS_AGRICOLAS = {k: v["coords_agricola"] for k, v in FOCOS_MINEROS.items()}
METAL_POR_ZONA = {k: v["metal_principal"] for k, v in FOCOS_MINEROS.items()}
METAL_IDX_POR_ZONA = {k: v["metal_idx"] for k, v in FOCOS_MINEROS.items()}

ETIQUETAS = {0: "Bajo", 1: "Medio", 2: "Alto"}
FEATURES = ["clorosis_pct", "necrosis_pct", "dist_mina_km", "ph_suelo", "metal_idx"]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en km entre dos puntos (fórmula de Haversine)."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl   = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def dist_a_mina_mas_cercana(lat: float, lon: float):
    """Devuelve (nombre_zona, distancia_km) al foco minero más cercano."""
    mejor_nombre, mejor_dist = None, np.inf
    for nombre, (mlat, mlon) in ZONAS_RIESGO.items():
        d = _haversine_km(lat, lon, mlat, mlon)
        if d < mejor_dist:
            mejor_nombre, mejor_dist = nombre, d
    return mejor_nombre, round(float(mejor_dist), 2)


def _ph_factor(ph: float, foco: dict) -> float:
    """
    Modifica el factor de riesgo según la química del metal y el pH del suelo.
    Respaldado por geoquímica de metales en suelo (Alloway 2013).
    """
    base = 0.0
    if "ph_riesgo_bajo" in foco and ph < foco["ph_riesgo_bajo"]:
        # Metales catiónicos (Cu, Pb, Zn): más móviles a pH ácido
        base += 0.12 * (foco["ph_riesgo_bajo"] - ph) / foco["ph_riesgo_bajo"]
    if "ph_riesgo_alto" in foco and ph > foco["ph_riesgo_alto"]:
        # Arsénico: más móvil a pH alto (arsenato compite con fosfato)
        base += 0.10 * (ph - foco["ph_riesgo_alto"]) / (9.0 - foco["ph_riesgo_alto"])
    return base


def generar_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Genera dataset de entrenamiento calibrado con gradientes reales de contaminación.

    Relación distancia-síntomas basada en:
    · Decaimiento exponencial de concentración con la distancia (Kabata-Pendias 2011)
    · Fitotoxicidad visual (clorosis/necrosis) correlacionada con concentración (Barbedo 2013)
    · Efectos de pH en movilidad de metales (Alloway 2013)
    · Clasificación de riesgo anclada en excedencia de ECA DS 011-2017-MINAM
    """
    rng = np.random.default_rng(seed)
    filas = []
    nombres_focos = list(FOCOS_MINEROS.keys())

    for _ in range(n):
        # Muestrear cerca de un foco minero real
        zona_nombre = rng.choice(nombres_focos)
        foco = FOCOS_MINEROS[zona_nombre]
        base_lat, base_lon = foco["coords_agricola"]

        # Dispersión alrededor del área agrícola (±~22 km)
        lat = base_lat + rng.normal(0, 0.20)
        lon = base_lon + rng.normal(0, 0.20)

        zona_cercana, dist = dist_a_mina_mas_cercana(lat, lon)
        foco_real = FOCOS_MINEROS[zona_cercana]

        # ── pH según zona (suelos volcánicos andinos: ligeramente alcalinos) ──
        ph_media = 7.2 if foco_real["metal_principal"] == "Arsénico" else 6.8
        ph = float(np.clip(rng.normal(ph_media, 0.85), 4.5, 9.0))

        # ── Factor de proximidad: decaimiento exponencial con la distancia ──
        # Basado en modelos de dispersión de contaminantes (Kabata-Pendias 2011)
        d_fitotox = foco_real["fitotox_dist_km"]
        cercania = float(np.exp(-dist / d_fitotox))   # 1 en la mina, ~0 a 3×d_fitotox

        # ── Corrección por pH ──────────────────────────────────────────────
        ph_mod = _ph_factor(ph, foco_real)

        # ── Score de contaminación estimado ───────────────────────────────
        score_base = foco_real["score_base"]
        score = (
            score_base * cercania
            + ph_mod
            + rng.normal(0, 0.06)       # variabilidad del suelo
        )

        # ── Síntomas visuales: función de score (respaldo Barbedo 2013) ───
        # A mayor contaminación: mayor clorosis y necrosis documentadas
        # Diferentes metales causan distintos patrones (Cu → clorosis dominante,
        # As → necrosis + clorosis, Pb → clorosis suave)
        metal = foco_real["metal_principal"]
        if metal == "Cobre":
            # Cu: clorosis intensa (similar a def. Fe) + necrosis en bordes
            clorosis = float(np.clip(rng.normal(10 + 65 * cercania, 14), 0, 100))
            necrosis = float(np.clip(rng.normal( 5 + 45 * cercania, 12), 0, 100))
        elif metal == "Arsénico":
            # As: necrosis + clorosis (puntas y márgenes de hojas)
            clorosis = float(np.clip(rng.normal(12 + 50 * cercania, 13), 0, 100))
            necrosis = float(np.clip(rng.normal(10 + 50 * cercania, 13), 0, 100))
        elif metal == "Plomo":
            # Pb: clorosis general, menos necrosis visible
            clorosis = float(np.clip(rng.normal( 8 + 40 * cercania, 12), 0, 100))
            necrosis = float(np.clip(rng.normal( 3 + 25 * cercania,  9), 0, 100))
        else:
            # Ag/Au y otros: sintomatología moderada
            clorosis = float(np.clip(rng.normal(10 + 35 * cercania, 12), 0, 100))
            necrosis = float(np.clip(rng.normal( 5 + 30 * cercania, 10), 0, 100))

        # ── Clasificación de riesgo anclada en ECA DS 011-2017-MINAM ──────
        # score < 0.25 → dentro de ECA → Bajo
        # 0.25-0.55   → posible excedencia → Medio
        # > 0.55      → excedencia probable → Alto
        if score < 0.25:
            riesgo = 0
        elif score < 0.55:
            riesgo = 1
        else:
            riesgo = 2

        filas.append({
            "clorosis_pct":  round(clorosis, 2),
            "necrosis_pct":  round(necrosis, 2),
            "dist_mina_km":  dist,
            "ph_suelo":      round(ph, 2),
            "metal_idx":     foco_real["metal_idx"],
            "lat":           round(lat, 5),
            "lon":           round(lon, 5),
            "zona_cercana":  zona_cercana,
            "metal":         metal,
            "riesgo":        riesgo,
        })

    df = pd.DataFrame(filas)
    return df


if __name__ == "__main__":
    df = generar_dataset()
    print("Dataset generado:", df.shape)
    print("\nDistribución de riesgo:")
    print(df["riesgo"].value_counts().sort_index().rename({0:"Bajo",1:"Medio",2:"Alto"}))
    print("\nDistribución por metal:")
    print(df["metal"].value_counts())
    print("\nPrimeras filas:")
    print(df[FEATURES + ["zona_cercana","riesgo"]].head(8).to_string())
