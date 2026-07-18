"""
reporte.py
----------
Genera el reporte final con recomendaciones de biorremediacion usando
Google Gemini (gratis con API key de AI Studio).

Regla de oro para hackaton: SIEMPRE hay fallback. Si no hay API key,
si no hay internet, o si la API falla en vivo -> se usa un reporte
plantilla local que igual se ve profesional. El demo NUNCA se cae.

Plantas nativas altoandinas para biorremediacion (base cientifica):
  - Baccharis salicifolia (Chilca)  -> tolera y acumula metales, riberas
  - Baccharis latifolia (Chilca)    -> fitoestabilizacion
  - Schoenoplectus / totora         -> humedales, retiene As en agua
  - Distichia muscoides (bofedal)   -> zonas altoandinas humedas
"""

import os

# Import "suave": si no esta la libreria, seguimos con fallback
try:
    import google.generativeai as genai
    _GENAI_OK = True
except Exception:
    _GENAI_OK = False


PLANTAS = {
    "Alto": ("Baccharis salicifolia (Chilca) y Schoenoplectus californicus (Totora)",
             "fitoestabilizacion intensiva en el borde de la parcela y canales de riego"),
    "Medio": ("Baccharis latifolia (Chilca)",
              "fitoestabilizacion preventiva en linderos"),
    "Bajo": ("cobertura vegetal nativa de mantenimiento",
             "monitoreo periodico, sin intervencion urgente"),
}


def _fallback(zona, metal, riesgo, dist, ph, clorosis, necrosis):
    """Reporte local que no depende de ninguna API."""
    planta, accion = PLANTAS.get(riesgo, PLANTAS["Medio"])
    prioridad = {"Alto": "PRIORITARIO - enviar muestra a laboratorio",
                 "Medio": "RECOMENDADO - programar muestreo",
                 "Bajo": "OPCIONAL - vigilancia"}[riesgo]
    return f"""### Reporte de riesgo de contaminacion

**Zona analizada:** {zona}
**Metal de interes:** {metal}
**Nivel de riesgo predicho:** {riesgo}

**Diagnostico de campo (tamizaje):**
- Sintomas visuales de estres: clorosis {clorosis:.0f}%, necrosis {necrosis:.0f}%
- Distancia al foco minero mas cercano: {dist:.1f} km
- pH del suelo: {ph:.1f} (un pH acido aumenta la movilidad del arsenico)

**Muestreo de laboratorio:** {prioridad}.
Este tamizaje con IA permite priorizar que parcelas analizar primero,
reduciendo el numero de pruebas de laboratorio innecesarias y su costo.

**Biorremediacion recomendada:**
Se sugiere **{planta}** para {accion}. Son especies nativas altoandinas
tolerantes a metales pesados, adecuadas al clima de Arequipa.

_Nota: tamizaje preliminar de apoyo a la decision. No reemplaza el analisis
de laboratorio certificado (EPA 6020 / ICP-MS) para confirmar concentraciones._
"""


def generar_reporte(zona, metal, riesgo, dist, ph, clorosis, necrosis,
                    api_key=None):
    """
    Intenta usar Gemini; si algo falla, devuelve el fallback local.
    api_key: si es None, se lee de la variable de entorno GEMINI_API_KEY.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if not (_GENAI_OK and api_key):
        return _fallback(zona, metal, riesgo, dist, ph, clorosis, necrosis)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Eres un ingeniero ambiental experto en Arequipa, Peru.
Redacta un reporte breve (maximo 180 palabras) en espanol, en formato Markdown,
para un agricultor. Datos del tamizaje con IA:

- Zona: {zona}
- Metal de interes: {metal}
- Riesgo predicho por el modelo: {riesgo}
- Clorosis (amarillamiento) en hoja: {clorosis:.0f}%
- Necrosis (manchas) en hoja: {necrosis:.0f}%
- Distancia al foco minero: {dist:.1f} km
- pH del suelo: {ph:.1f}

Incluye SIEMPRE:
1. Un diagnostico claro del riesgo.
2. Si conviene o no enviar muestra a laboratorio (prioridad).
3. Recomendacion de biorremediacion con plantas nativas altoandinas
   (ej. Baccharis salicifolia, totora) segun el nivel de riesgo.
4. Una linea aclarando que es un tamizaje preliminar, no reemplaza
   el analisis de laboratorio certificado.

Tono profesional y claro. No inventes cifras de concentracion."""
        resp = model.generate_content(prompt)
        texto = (resp.text or "").strip()
        if len(texto) < 40:  # respuesta vacia o rara -> fallback
            return _fallback(zona, metal, riesgo, dist, ph, clorosis, necrosis)
        return texto
    except Exception:
        return _fallback(zona, metal, riesgo, dist, ph, clorosis, necrosis)


if __name__ == "__main__":
    print(generar_reporte(
        "La Joya (As agua)", "Arsenico", "Alto",
        dist=3.2, ph=5.4, clorosis=55, necrosis=30
    ))
