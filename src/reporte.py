"""
reporte.py — Reporte técnico profesional (sin dependencia de API externa)

Genera un reporte completo en Markdown integrado con todos los módulos:
  · Diagnóstico visual (clorosis / necrosis OpenCV)
  · Indicadores ambientales estimados (CE, MO, NTU, PM10)
  · Riesgo para la salud humana (EPA RAGS — HQ, HRI, CR)
  · Impacto económico y pasivos mineros (valor en riesgo, multas OEFA)
  · Cumplimiento normativo (DS 011, DS 004, DS 003)
  · Plan de biorremediación con especies nativas altoandinas
  · Acciones inmediatas priorizadas por nivel de riesgo

Destinatarios:
  · Equipo de medio ambiente de la empresa minera (ESG/compliance)
  · Autoridades: OEFA, ANA, DIGESA, GORE Arequipa
  · Comunidades afectadas e ingenieros agrónomos de campo

Nota: api_key se acepta por compatibilidad pero ya no es necesaria.
El reporte local es completo y usable en auditorías y comunicación comunitaria.
"""

from datetime import datetime
from typing import Optional


_PLANTAS = {
    "Alto": {
        "especies": [
            ("Baccharis salicifolia",      "Chilca",    "Fitoestabilización intensiva — inmoviliza metales en rizosfera"),
            ("Schoenoplectus californicus", "Totora",    "Acumula As, Pb y Cd en raíces; ideal en canales de riego"),
            ("Lolium perenne",             "Ryegrass",  "Alta biomasa radicular; efectivo para Pb y Cd"),
        ],
        "accion": (
            "Implantación urgente en linderos de parcela y bordes de canal de riego. "
            "Retiro y disposición de biomasa como residuo peligroso (RRSS-EP) cada 6 meses."
        ),
        "plazo": "Inicio en 30 días. Evaluación a los 6 meses.",
    },
    "Medio": {
        "especies": [
            ("Baccharis latifolia",    "Chilca grande", "Fitoestabilización preventiva en perímetro"),
            ("Festuca dolichophylla",  "Chilligua",     "Reduce erosión y dispersión de polvo contaminado"),
        ],
        "accion": (
            "Implantación en linderos de parcelas más próximas al foco minero. "
            "Monitoreo foliar semestral y muestreo de suelo en 30 días."
        ),
        "plazo": "Inicio en 60 días. Evaluación a los 12 meses.",
    },
    "Bajo": {
        "especies": [
            ("Distichia muscoides",      "Bofedal nativo", "Mantenimiento de humedales altoandinos"),
            ("cobertura vegetal nativa", "",               "Monitoreo rutinario sin intervención urgente"),
        ],
        "accion": "Mantener cobertura vegetal nativa. Muestreo confirmatorio en 90 días.",
        "plazo": "Ciclo de monitoreo anual.",
    },
}

_ACCIONES = {
    "Alto": [
        "🔴 SUSPENDER uso de agua de riego del canal posiblemente afectado hasta resultado ICP-MS",
        "🔴 NOTIFICAR a OEFA (oefa.gob.pe) y ANA dentro de 24 horas",
        "🔴 TOMAR muestras compuestas de suelo (0-30 cm y 30-60 cm) para ICP-MS certificado",
        "🔴 INFORMAR a trabajadores sobre riesgo dérmico e inhalación de polvo de suelo",
        "🔴 NO consumir productos de la parcela hasta confirmación de laboratorio",
        "🔬 ANALIZAR: As, Pb, Cd, Cu, Zn según DS 011-2017-MINAM (Tabla 1, uso agrícola)",
        "🔬 ANALIZAR agua de riego: As, Cu, Pb, Cd según DS 004-2017-MINAM Cat. 3",
        "📋 REGISTRAR GPS exacto de cada punto de muestreo para cadena de custodia legal",
    ],
    "Medio": [
        "🟡 PROGRAMAR muestreo de suelo y agua en los próximos 30 días",
        "🟡 REVISAR historial de análisis previos en la parcela (últimos 2 años)",
        "🟡 CONSULTAR con ingeniero agrónomo sobre rotación de cultivos y bioindicadores",
        "🟡 APLICAR enmiendas calcáreas (cal agrícola) si pH < 6.5 para inmovilizar metales",
        "🟡 INICIAR biorremediación preventiva en linderos",
        "📋 MONITOREAR síntomas visuales de nuevas plantas cada 2 semanas",
        "📋 REGISTRAR cambios en rendimiento agrícola como señal de alerta temprana",
    ],
    "Bajo": [
        "🟢 MONITOREO visual rutinario mensual de síntomas en cultivos",
        "🟢 MUESTREO de suelo en el próximo ciclo agrícola (6–12 meses)",
        "🟢 MANTENER registros de pH y conductividad eléctrica del suelo",
        "📋 PARTICIPAR en programa de vigilancia comunitaria si existe en la zona",
        "📋 ACTUALIZAR evaluación si la empresa minera reporta cambios en operaciones",
    ],
}


def generar_reporte(zona: str, metal: str, riesgo: str,
                    dist: float, ph: float, clorosis: float, necrosis: float,
                    health_risk: dict = None, indicadores: dict = None,
                    impacto_economico: dict = None,
                    confianza: float = None, probabilidades: dict = None,
                    lat: float = None, lon: float = None,
                    ubicacion_origen: str = None,
                    api_key: str = None) -> str:
    """
    Genera reporte técnico profesional completo. api_key ignorado.
    """
    fecha = datetime.now().strftime("%d/%m/%Y %H:%M")
    plantas = _PLANTAS.get(riesgo, _PLANTAS["Medio"])
    acciones = _ACCIONES.get(riesgo, _ACCIONES["Medio"])

    urgencia = {
        "Alto":  "🔴 RIESGO ALTO — ACCIÓN INMEDIATA REQUERIDA",
        "Medio": "🟡 RIESGO MEDIO — MUESTREO RECOMENDADO EN 30 DÍAS",
        "Bajo":  "🟢 RIESGO BAJO — MONITOREO PREVENTIVO",
    }.get(riesgo, "⚪ NIVEL INDETERMINADO")

    conf_txt = f" | Confianza: {confianza:.0f}%" if confianza else ""

    if lat is not None:
        ns = "S" if lat < 0 else "N"
        ew = "W" if lon < 0 else "E"
        gps_txt = f"{abs(lat):.5f}°{ns}, {abs(lon):.5f}°{ew} ({ubicacion_origen or 'GPS'})"
    else:
        gps_txt = "No proporcionado — se usó centroide del área agrícola de la zona"

    # ── Encabezado ─────────────────────────────────────────────────────────────
    rep = f"""# Reporte Técnico — TerraGuard Arequipa

**Fecha:** {fecha}
**Sistema:** TerraGuard Arequipa v2.0 · Random Forest + OpenCV (ExG+HSV+CIELAB)

---

## {urgencia}{conf_txt}

| Parámetro | Valor |
|---|---|
| Zona evaluada | {zona} |
| Contaminante principal | {metal} |
| **Nivel de riesgo** | **{riesgo}** |
| Distancia al foco minero | {dist:.2f} km |
| GPS | {gps_txt} |

"""
    if probabilidades:
        rep += "**Probabilidades del modelo:**  \n"
        for nivel, pct in sorted(probabilidades.items(), key=lambda x: -x[1]):
            bar = "█" * int(pct / 5)
            rep += f"- {nivel}: {pct:.1f}%  {bar}\n"
        rep += "\n"

    # ── Diagnóstico visual ─────────────────────────────────────────────────────
    clorosis_txt = "⚠️ Moderada-severa — estrés fitotóxico activo" if clorosis > 25 else "Leve o en rango normal"
    necrosis_txt = "⚠️ Daño celular irreversible presente" if necrosis > 10 else "Mínima o ausente"
    ph_txt = ("⚠️ Alcalino — arsenato más biodisponible" if ph > 7.5 else
              "⚠️ Ácido — Cu/Pb/Zn más móviles en solución" if ph < 6.2 else
              "✅ Neutro — movilidad moderada")
    rep += f"""---

## Diagnóstico Visual (OpenCV — ExG + HSV + CIELAB)

| Síntoma | Detectado | Interpretación |
|---|---|---|
| **Clorosis** (amarillamiento) | {clorosis:.1f}% | {clorosis_txt} |
| **Necrosis** (tejido muerto) | {necrosis:.1f}% | {necrosis_txt} |
| **pH del suelo** | {ph:.1f} | {ph_txt} |

**Algoritmo:** ExG/Otsu (Woebbecke 1995) + HSV dual-range + CIELAB calibrado con píxeles reales.
*Cu → clorosis tipo-Fe + necrosis en bordes · As → necrosis + clorosis en puntas · Pb → clorosis difusa*

"""

    # ── Indicadores ambientales ────────────────────────────────────────────────
    if indicadores:
        ind = indicadores
        alertas_txt = "\n".join(f"- ⚠️ {a}" for a in ind.get("alertas", [])) \
                      or "- ✅ Sin alertas en parámetros estimados"
        rep += f"""---

## Indicadores Ambientales Estimados

| Indicador | Valor estimado | Rango normal | ECA / Límite |
|---|---|---|---|
| Conductividad eléctrica (CE) | {ind.get("conductividad_electrica_ds_m")} dS/m | 0.2–0.8 dS/m | > 2.0: impacto en cultivos |
| Materia orgánica (MO) | {ind.get("materia_organica_pct")}% | 1.5–4.0% | < 1.0%: microbios inhibidos |
| Turbidez agua riego | {ind.get("turbidez_agua_ntu")} NTU | 5–25 NTU | 100 NTU (DS 004-2017 Cat.3) |
| PM10 polvo de mina | {ind.get("pm10_ug_m3")} μg/m³ | 20–50 μg/m³ | 100 μg/m³ 24h (DS 003-2017) |

**Alertas detectadas:**
{alertas_txt}

*{ind.get("nota", "")}*

"""

    # ── Salud humana ───────────────────────────────────────────────────────────
    if health_risk:
        hr = health_risk
        hri_flag = "🔴" if hr["hri"] >= 1.0 else ("🟡" if hr["hri"] >= 0.1 else "🟢")
        cr_flag  = "🔴" if hr["cancer_risk_total"] >= 1e-4 else (
                   "🟡" if hr["cancer_risk_total"] >= 1e-6 else "🟢")
        metales_rows = ""
        for m, d in hr.get("por_metal", {}).items():
            f_s = "⚠️" if d["alerta_suelo"] else "✅"
            f_a = "⚠️" if d["alerta_agua"]  else "✅"
            hq_fl = "🔴" if d["hq_total"] >= 1 else ("🟡" if d["hq_total"] >= 0.5 else "🟢")
            cr_s = f"{d['cancer_risk']:.2e}" if d["cancer_risk"] else "—"
            metales_rows += (
                f"| **{m}** | {d['cs_estimada_mg_kg']} mg/kg {f_s} | "
                f"{d['cw_estimada_mg_l']} mg/L {f_a} | "
                f"{d['hq_total']:.3f} {hq_fl} | {cr_s} |\n"
            )
        rep += f"""---

## Evaluación de Riesgo para la Salud Humana (EPA RAGS Part A)

**{hri_flag} HRI = {hr["hri"]:.4f} — {hr["hri_nivel"]}**
**{cr_flag} Riesgo carcinogénico = {hr["cancer_risk_total"]:.2e} — {hr["cancer_risk_nivel"]}**

> HRI < 1: aceptable · HRI 1–10: riesgo moderado · HRI > 10: acción inmediata
> CR < 1×10⁻⁶: aceptable · CR > 1×10⁻⁴: requiere intervención (criterio EPA)

| Metal | Cs estimada (suelo) | Cw estimada (agua) | HQ total | Cancer Risk |
|---|---|---|---|---|
{metales_rows}
**Vías modeladas:** ingestión de suelo + ingestión de agua de riego + contacto dérmico
*{hr.get("advertencia", "")}*

"""

    # ── Impacto económico ──────────────────────────────────────────────────────
    if impacto_economico:
        ie = impacto_economico
        area = ie.get("area_agricola_influencia", {})
        imp  = ie.get("impacto_estimado", {})
        cos  = ie.get("costos_accion", {})
        reg  = ie.get("exposicion_regulatoria", {})
        rep += f"""---

## Impacto Económico y Pasivos Ambientales

### Producción agrícola en zona de influencia — {zona}
- **Área total bajo influencia:** {area.get("total_ha", "?")} ha
- **Valor producción anual:** USD {area.get("valor_produccion_anual_usd", 0):,}
- **Cultivos principales:** {", ".join(area.get("cultivos", {}).keys())}

### Pérdida estimada — Riesgo {riesgo}
- **Fracción afectada:** {imp.get("fraccion_perdida_pct", "?")}% del valor total
- **Pérdida en producción:** USD {imp.get("perdida_produccion_usd", 0):,} (S/. {imp.get("perdida_produccion_pen", 0):,})

### Costos de acción correctiva
| Concepto | Detalle | Costo USD |
|---|---|---|
| Laboratorio ICP-MS | {cos.get("muestras_icp_ms", "?")} muestras × USD 175 | {cos.get("costo_laboratorio_usd", 0):,} |
| Fitorremediación activa | {cos.get("area_fitorremediacion_ha", "?")} ha × USD 3,800 | {cos.get("costo_fitorremediacion_usd", 0):,} |
| **Total acción correctiva** | | **USD {cos.get("costo_total_usd", 0):,}** |

### Exposición regulatoria
- **Multa máxima OEFA:** USD {reg.get("multa_max_oefa_usd", 0):,} (S/. {reg.get("multa_max_oefa_soles", 0):,})
- **Base legal:** {reg.get("base_legal", "—")}
- **Referencia histórica:** {reg.get("historial_cerro_verde", "—")}

**Prioridad de acción:** {ie.get("prioridad_accion", "—")}

*{ie.get("fuentes", "")}*

"""

    # ── Biorremediación ────────────────────────────────────────────────────────
    esp_rows = "\n".join(
        f"| *{s[0]}* | {s[1]} | {s[2]} |"
        for s in plantas["especies"] if s[0]
    )
    rep += f"""---

## Plan de Biorremediación con Plantas Nativas Altoandinas

| Especie | Nombre común | Mecanismo |
|---|---|---|
{esp_rows}

**Acción recomendada:** {plantas["accion"]}
**Plazo:** {plantas["plazo"]}

"""

    # ── Acciones inmediatas ────────────────────────────────────────────────────
    rep += f"""---

## Acciones Inmediatas — {riesgo.upper()}

{"".join(chr(10) + a for a in acciones)}

"""

    # ── Cumplimiento normativo ─────────────────────────────────────────────────
    ds011 = "⚠️ Posible excedencia — muestreo urgente" if riesgo == "Alto" else (
            "🟡 Verificar — muestreo recomendado" if riesgo == "Medio" else "✅ Dentro de rango estimado")
    ds004 = "⚠️ Verificar agua de canal prioritariamente" if riesgo in ["Alto","Medio"] else "✅ Sin alerta"
    ds003 = "⚠️ PM10 posiblemente elevado por actividad minera" if riesgo == "Alto" else "📋 Monitorear"
    rep += f"""---

## Estado de Cumplimiento Normativo Estimado

| Norma | Objeto | Estado estimado |
|---|---|---|
| DS 011-2017-MINAM | ECA Suelo agrícola (As, Pb, Cd, Zn) | {ds011} |
| DS 004-2017-MINAM Cat.3 | ECA Agua de riego (As, Cu, Pb, Cd) | {ds004} |
| DS 003-2017-MINAM | ECA Aire PM10 | {ds003} |
| Ley 28611 Art. 135 | Responsabilidad ambiental | Evaluación en curso |
| DS 017-2012-EM | PAMA — Plan de Adecuación y Manejo Ambiental | Verificar vigencia |
| Ley 29785 | Consulta previa a comunidades | Aplicable si hay cambios operativos |

---

## Próximos Pasos para el Equipo Ambiental

1. **Laboratorio ICP-MS** — Tomar muestras según cadena de custodia EPA/ISO 5667
2. **Registrar en SIGFA** — Sistema de Gestión OEFA (sigfa.oefa.gob.pe)
3. **Actualizar PMA** — Plan de Manejo Ambiental con resultados del tamizaje
4. **Coordinación** — GORE Arequipa (Gerencia Recursos Naturales), ANA, DIGESA
5. **Comunicación comunitaria** — Informar a agricultores según Ley 29785

---

*Generado por TerraGuard Arequipa v2.0 · {fecha}*
*Tamizaje preliminar — no reemplaza análisis ICP-MS certificado (EPA Method 6020B / ISO 11885)*
*Modelo: Random Forest (Accuracy ~0.82, AUC ROC ~0.95) · Visión: ExG+HSV+CIELAB (Woebbecke 1995)*
"""
    return rep
