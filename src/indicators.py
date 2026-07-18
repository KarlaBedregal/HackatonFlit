"""
indicators.py — Indicadores ambientales y evaluación de impacto económico

Indicadores derivados del modelo de proximidad para complementar el diagnóstico.
Lo que un equipo ambiental minero monitorea rutinariamente:
  · CE  — conductividad eléctrica del suelo (proxy de concentración total de sales/metales)
  · MO  — materia orgánica (reducida por toxicidad microbiana del metal)
  · NTU — turbidez del agua de riego (partículas con metales adsorbidos)
  · PM10 — material particulado de voladuras y relaves al viento

Impacto económico:
  · Valor en riesgo de la producción agrícola en zona de influencia
  · Costo de muestreo ICP-MS priorizado
  · Estimación de pasivo de fitorremediación
  · Exposición a multas OEFA (hasta 10,000 UIT = ~USD 14M)

Fuentes de rangos documentados:
  OEFA monitoreos 2019-2022 (Cerro Verde, Tambo)
  ANA Boletín Hidroquímico 2023
  MIDAGRI Estadísticas Arequipa 2023
  SUNASS/DS 004-2017-MINAM Cat.3
  DS 003-2017-MINAM ECA Aire
"""

import numpy as np


def estimar_indicadores(riesgo: str, score: float, dist_km: float,
                         metal: str, ph: float) -> dict:
    """
    Estima indicadores de campo a partir del nivel de riesgo y distancia al foco.

    Returns:
        dict con valores estimados, rangos normales, alertas y ECA referencia.
    """
    # Factor combinado: intensidad de contaminación × decaimiento con distancia
    factor = float(np.clip(score * np.exp(-dist_km / 25.0), 0, 1))

    # ── Conductividad eléctrica (CE, dS/m) ───────────────────────────────────
    # Normal suelo agrícola andino: 0.2–0.8 dS/m
    # Contaminado cerca de mina activa: 3–9 dS/m (OEFA Cerro Verde 2022)
    ce = round(float(0.5 + 7.5 * factor), 2)
    ce_alerta = ce > 2.0

    # ── Materia orgánica (MO, %) ──────────────────────────────────────────────
    # Normal: 1.5–4.0%; la toxicidad microbiana del metal la degrada
    mo = round(float(3.2 - 2.8 * factor), 2)
    mo_alerta = mo < 1.0

    # ── Turbidez agua de riego (NTU) ─────────────────────────────────────────
    # Normal: 5–25 NTU; aguas afectadas por DAM o efluentes: 150–600 NTU
    turb = round(float(15 + 535 * factor), 1)
    turb_alerta = turb > 100   # ECA Cat.3: 100 NTU

    # ── PM10 de actividad minera (μg/m³) ─────────────────────────────────────
    # Open-pit (Cerro Verde, Tambo): 150–450 μg/m³ en sitio
    pm10_base = 1.2 if metal in ["Cobre", "Arsénico"] else 1.0  # mayor polvo minas Cu/As
    pm10 = round(float(30 + 390 * factor * pm10_base), 1)
    pm10_alerta = pm10 > 100   # ECA 24h DS 003-2017

    # ── pH: impacto específico según metal ───────────────────────────────────
    ph_alertas = []
    if metal == "Arsénico" and ph > 7.5:
        ph_alertas.append(
            f"pH={ph:.1f} alcalino — arsenato más móvil (compite con PO₄³⁻); "
            "considerar enmienda ácida controlada"
        )
    if metal in ["Cobre", "Plomo", "Zinc"] and ph < 6.0:
        ph_alertas.append(
            f"pH={ph:.1f} ácido — mayor movilidad de {metal} en solución del suelo; "
            "encalar para estabilizar (pH objetivo 6.5–7.0)"
        )

    alertas = []
    if ce_alerta:
        alertas.append(f"CE={ce} dS/m > 2.0 — toxicidad salina posible en cultivos sensibles")
    if mo_alerta:
        alertas.append(f"MO={mo}% < 1.0 — actividad microbiana severamente reducida")
    if turb_alerta:
        alertas.append(f"Turbidez={turb} NTU supera ECA Cat.3 (100 NTU, DS 004-2017)")
    if pm10_alerta:
        alertas.append(f"PM10={pm10} μg/m³ supera ECA 24h (100 μg/m³, DS 003-2017)")
    alertas.extend(ph_alertas)

    return {
        "conductividad_electrica_ds_m": ce,
        "materia_organica_pct":         mo,
        "turbidez_agua_ntu":            turb,
        "pm10_ug_m3":                   pm10,
        "alertas":                      alertas,
        "rangos_normales": {
            "ce_ds_m":      "0.2–0.8",
            "mo_pct":       "1.5–4.0",
            "turbidez_ntu": "5–25",
            "pm10_ug_m3":   "20–50",
        },
        "eca_referencia": {
            "turbidez":   "100 NTU (DS 004-2017-MINAM Cat.3)",
            "pm10_24h":   "100 μg/m³ (DS 003-2017-MINAM)",
            "pm10_anual": "50 μg/m³ (DS 003-2017-MINAM)",
        },
        "nota": (
            "Valores estimados por modelo de proximidad (Kabata-Pendias 2011). "
            "Confirmar con: conductivímetro de suelo, turbidímetro óptico, "
            "monitor PM10 Grimm/TSI."
        ),
    }


def calcular_impacto_economico(riesgo: str, score: float,
                                 dist_km: float, zona_nombre: str) -> dict:
    """
    Cuantifica el impacto económico para la empresa minera responsable.
    Relevante para: ESG reporting, pasivos ambientales, negociación comunitaria,
    y preparación de defensa ante multas OEFA.

    Fuentes: MIDAGRI Arequipa 2023 (rendimientos y precios), costos
    Latinlab/SGS Perú 2024, escala de multas OEFA vigente.
    """
    # Cultivos representativos de las zonas de influencia de las 5 minas
    CULTIVOS = {
        "Arroz":   {"rendimiento_t_ha": 8.0,  "precio_t_usd": 420, "ha": 4_500},
        "Cebolla": {"rendimiento_t_ha": 25.0, "precio_t_usd": 260, "ha": 3_800},
        "Maíz":    {"rendimiento_t_ha": 10.0, "precio_t_usd": 210, "ha": 2_100},
        "Alfalfa": {"rendimiento_t_ha": 40.0, "precio_t_usd":  95, "ha": 1_800},
        "Papa":    {"rendimiento_t_ha": 22.0, "precio_t_usd": 180, "ha":   900},
        "Ajo":     {"rendimiento_t_ha":  8.5, "precio_t_usd": 600, "ha":   420},
    }

    loss_base = {"Alto": 0.38, "Medio": 0.16, "Bajo": 0.03}.get(riesgo, 0.12)
    dist_factor = float(np.exp(-dist_km / 18.0))
    eff_loss = loss_base * dist_factor

    total_ha    = sum(c["ha"]                                     for c in CULTIVOS.values())
    valor_total = sum(c["rendimiento_t_ha"] * c["precio_t_usd"] * c["ha"]
                      for c in CULTIVOS.values())
    perdida_usd = valor_total * eff_loss

    # Costo de muestreo ICP-MS (SGS/Latinlab Arequipa, mayo 2024)
    n_muestras = {"Alto": 18, "Medio": 7, "Bajo": 2}.get(riesgo, 5)
    costo_lab  = n_muestras * 175   # USD/muestra (As, Pb, Cd, Cu, Zn, Zn — EPA 6020B)

    # Costo de fitorremediación activa
    area_rem_ha = round(total_ha * eff_loss * 0.28, 1)
    costo_fito  = round(area_rem_ha * 3_800)   # USD/ha fitorremediación intensiva

    # Exposición regulatoria OEFA
    uit_2024       = 5_150          # UIT Perú 2024 (soles)
    tipo_cambio    = 3.78
    multa_max_s    = 10_000 * uit_2024
    multa_max_usd  = round(multa_max_s / tipo_cambio)

    prioridad = {
        "Alto":  "INMEDIATA — muestras ICP-MS en 72h, notificar OEFA en 24h",
        "Medio": "30 DÍAS — muestreo antes del siguiente ciclo de cultivo",
        "Bajo":  "90 DÍAS — monitoreo rutinario, ciclo semestral",
    }.get(riesgo, "INDETERMINADA")

    return {
        "zona": zona_nombre,
        "area_agricola_influencia": {
            "total_ha": total_ha,
            "cultivos": {c: {"ha": d["ha"], "precio_t_usd": d["precio_t_usd"]}
                         for c, d in CULTIVOS.items()},
            "valor_produccion_anual_usd": round(valor_total),
        },
        "impacto_estimado": {
            "fraccion_perdida_pct":     round(eff_loss * 100, 1),
            "perdida_produccion_usd":   round(perdida_usd),
            "perdida_produccion_pen":   round(perdida_usd * tipo_cambio),
        },
        "costos_accion": {
            "muestras_icp_ms":          n_muestras,
            "costo_laboratorio_usd":    costo_lab,
            "area_fitorremediacion_ha": area_rem_ha,
            "costo_fitorremediacion_usd": costo_fito,
            "costo_total_usd":          costo_lab + costo_fito,
        },
        "exposicion_regulatoria": {
            "multa_max_oefa_soles": multa_max_s,
            "multa_max_oefa_usd":   multa_max_usd,
            "base_legal": (
                "Art. 135 Ley 28611 (LGA) + Escala de Multas OEFA 2023: "
                "hasta 10,000 UIT por infracción muy grave"
            ),
            "historial_cerro_verde": (
                "OEFA aplicó sanciones en 2012, 2014 y 2015 por incumplimiento "
                "de EMA — referencia para estimación de riesgo"
            ),
        },
        "prioridad_accion": prioridad,
        "fuentes": (
            "MIDAGRI Arequipa 2023 (rendimientos/precios), Latinlab/SGS Perú 2024 "
            "(costos ICP-MS), Escala de Multas OEFA 2023."
        ),
    }
