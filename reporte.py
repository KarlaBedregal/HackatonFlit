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

# Carga .env local (no se sube a Git). Si falta python-dotenv, seguimos igual.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Modelo Gemini: SOLO Flash / Flash-Lite (tier gratuito). Nunca "-pro".
# Cambiar solo esta constante si el nombre exacto varia en AI Studio.
MODELO_GEMINI = "gemini-flash-latest"  # alias gratis; 2.5-flash ya no acepta usuarios nuevos

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
    """Reporte local que no depende de ninguna API. Markdown con headers ##."""
    planta, accion = PLANTAS.get(riesgo, PLANTAS["Medio"])
    prioridad = {"Alto": "PRIORITARIO - enviar muestra a laboratorio",
                 "Medio": "RECOMENDADO - programar muestreo",
                 "Bajo": "OPCIONAL - vigilancia"}[riesgo]
    return f"""## Resumen del riesgo

En la zona **{zona}**, el tamizaje indica riesgo **{riesgo}** asociado a **{metal}**.
Conviene prestar atencion a esta parcela y priorizar acciones segun el nivel
indicado, sin alarmarse ni ignorar las senales de campo.

## Por que este resultado

- Sintomas visuales de estres: clorosis {clorosis:.0f}%, necrosis {necrosis:.0f}%
- Distancia al foco minero mas cercano: {dist:.1f} km
- pH del suelo: {ph:.1f} (un pH acido aumenta la movilidad del arsenico)

Una parcela mas cerca de la mina, con mas amarillamiento o manchas en la hoja
y pH desfavorable, eleva el riesgo predicho por el modelo.

## Recomendacion de biorremediacion (preliminar)

Se sugiere **{planta}** para {accion}. Son especies nativas altoandinas
tolerantes a metales pesados, adecuadas al clima de Arequipa.

Esta es una sugerencia preliminar: no reemplaza la asesoria tecnica de campo.

## Muestreo de laboratorio

{prioridad}.
Este tamizaje con IA permite priorizar que parcelas analizar primero,
reduciendo el numero de pruebas de laboratorio innecesarias y su costo.

## Aclaracion

_Tamizaje preliminar de apoyo a la decision. No reemplaza el analisis
de laboratorio certificado (EPA 6020 / ICP-MS) para confirmar concentraciones._
"""


def generar_reporte(zona, metal, riesgo, dist, ph, clorosis, necrosis,
                    api_key=None):
    """
    Intenta usar Gemini; si algo falla, devuelve el fallback local.
    Nunca lanza excepcion ni devuelve None (siempre un string Markdown).

    api_key: si es None, se lee GOOGLE_API_KEY (o GEMINI_API_KEY por
    compatibilidad con el resto del equipo).
    """
    api_key = (
        api_key
        or os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )

    if not (_GENAI_OK and api_key):
        print("[ADVERTENCIA] Se usa reporte local (fallback). "
              "Sin API key o sin libreria google-generativeai.")
        return _fallback(zona, metal, riesgo, dist, ph, clorosis, necrosis)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODELO_GEMINI)
        prompt = f"""Eres un ingeniero ambiental experto en Arequipa, Peru.
Redacta un reporte breve (maximo 220 palabras) en espanol, en formato Markdown
con headers ## (no texto plano), para un agricultor no tecnico.
Datos del tamizaje con IA:

- Zona: {zona}
- Metal de interes: {metal}
- Riesgo predicho por el modelo: {riesgo}
- Clorosis (amarillamiento) en hoja: {clorosis:.0f}%
- Necrosis (manchas) en hoja: {necrosis:.0f}%
- Distancia al foco minero: {dist:.1f} km
- pH del suelo: {ph:.1f}

Incluye SIEMPRE estas secciones con headers ##:
1. ## Resumen del riesgo
   (2-3 lineas, tono claro para un agricultor no tecnico)
2. ## Por que este resultado
   (breve: cruza distancia a la mina, pH y % de estres visual)
3. ## Recomendacion de biorremediacion (preliminar)
   (al menos UNA especie nativa altoandina, ej. Baccharis salicifolia / Chilca,
   totora. Se honesto: es sugerencia preliminar, no reemplaza asesoria tecnica
   de campo)
4. ## Aclaracion
   (esto NO reemplaza un analisis de laboratorio confirmatorio)

Tono profesional y claro. No inventes cifras de concentracion."""
        resp = model.generate_content(prompt)
        texto = (resp.text or "").strip()
        if len(texto) < 40:  # respuesta vacia o rara -> fallback
            print("[ADVERTENCIA] Se usa reporte local (fallback). "
                  "Respuesta de Gemini vacia o demasiado corta.")
            return _fallback(zona, metal, riesgo, dist, ph, clorosis, necrosis)
        return texto
    except Exception as e:
        print(f"[ADVERTENCIA] Se usa reporte local (fallback). Motivo: {e}")
        return _fallback(zona, metal, riesgo, dist, ph, clorosis, necrosis)


if __name__ == "__main__":
    # Datos mock: simulan outputs de Dev 1 (modelo) y Dev 2 (vision)
    mock_input = {
        "zona": "la_joya",
        "metal": "Arsenico",
        "riesgo": "Alto",
        "dist": 3.2,
        "ph": 6.7,
        "clorosis": 42.5,
        "necrosis": 18.0,
    }
    print(generar_reporte(**mock_input))
