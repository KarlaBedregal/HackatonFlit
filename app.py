"""
app.py  -  TerraGuard Arequipa
------------------------------
MVP: sistema predictivo de riesgo de contaminacion por metales pesados
en la agricultura de Arequipa.

Flujo:
  1) Sube foto de hoja  -> OpenCV extrae clorosis/necrosis
  2) Elige zona + pH    -> se calcula distancia a la mina
  3) Random Forest      -> predice riesgo Bajo/Medio/Alto
  4) Gemini (o fallback)-> reporte con biorremediacion
  5) Folium             -> mapa de riesgo de Arequipa

Ejecutar:  streamlit run app.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_generator import (
    generar_dataset, ZONAS_RIESGO, dist_a_mina_mas_cercana, ETIQUETAS
)
from src.vision import analizar_hoja, leer_imagen_desde_bytes
from src import model as M
from reporte import generar_reporte

st.set_page_config(page_title="TerraGuard Arequipa", page_icon="🌱", layout="wide")

COLOR_RIESGO = {"Bajo": "#2e7d32", "Medio": "#f9a825", "Alto": "#c62828"}


# ---------- Carga cacheada de datos y modelo ----------
@st.cache_resource
def cargar_todo():
    df = generar_dataset()
    modelo, metricas, _ = M.entrenar(df)
    return df, modelo, metricas


df, modelo, metricas = cargar_todo()

# ---------- Encabezado ----------
st.title("🌱 TerraGuard Arequipa")
st.caption("Tamizaje predictivo de riesgo de contaminacion por metales pesados "
           "para priorizar muestreo de laboratorio y ahorrar costos a la mineria y agro.")

col_izq, col_der = st.columns([1, 1])

# =========================================================
#  COLUMNA IZQUIERDA: entrada de datos y prediccion
# =========================================================
with col_izq:
    st.subheader("1. Datos de la parcela")

    zona = st.selectbox("Zona / foco de riesgo cercano", list(ZONAS_RIESGO.keys()))
    metal = "Cobre" if "Cerro Verde" in zona else "Arsenico"
    st.info(f"Metal de interes en esta zona: **{metal}**")

    ph = st.slider("pH del suelo", 4.0, 9.0, 6.5, 0.1,
                   help="Un pH acido (<6) aumenta la movilidad del arsenico.")

    st.subheader("2. Foto de la hoja (opcional)")
    foto = st.file_uploader("Sube una foto de la hoja del cultivo",
                            type=["jpg", "jpeg", "png"])

    if foto is not None:
        img = leer_imagen_desde_bytes(foto.read())
        if img is not None:
            vis = analizar_hoja(img)
            st.image(foto, caption="Hoja analizada", width=250)
        else:
            vis = analizar_hoja(np.zeros((10, 10, 3), np.uint8))  # fuerza fallback
            st.warning("No se pudo leer la imagen, usando estimacion de respaldo.")
    else:
        # Sin foto: sliders manuales (util para el simulador del jurado)
        st.markdown("*Sin foto: ajusta manualmente los sintomas visuales*")
        c1, c2 = st.columns(2)
        vis = {
            "clorosis_pct": c1.slider("Clorosis %", 0, 100, 30),
            "necrosis_pct": c2.slider("Necrosis %", 0, 100, 15),
            "ok": True, "msg": "manual",
        }

    # Distancia a la mina (centro de la zona elegida)
    zlat, zlon = ZONAS_RIESGO[zona]
    _, dist = dist_a_mina_mas_cercana(zlat, zlon)

    st.metric("Distancia al foco minero", f"{dist} km")
    st.write(f"Clorosis detectada: **{vis['clorosis_pct']:.0f}%**  |  "
             f"Necrosis detectada: **{vis['necrosis_pct']:.0f}%**")

    # ---------- Prediccion ----------
    pred = M.predecir_riesgo(modelo, vis["clorosis_pct"], vis["necrosis_pct"], dist, ph)
    riesgo_txt = pred["riesgo_txt"]

    st.markdown(
        f"<h2 style='color:{COLOR_RIESGO[riesgo_txt]}'>Riesgo: {riesgo_txt} "
        f"({pred['confianza']}% conf.)</h2>", unsafe_allow_html=True)
    st.write("Probabilidades:", pred["probabilidades"])

    # ---------- Reporte con IA ----------
    st.subheader("3. Reporte y biorremediacion")
    api_key = st.text_input("API key de Gemini (opcional)", type="password",
                            help="Sin key, se genera un reporte local igual de valido.")
    if st.button("Generar reporte", type="primary"):
        with st.spinner("Generando reporte..."):
            rep = generar_reporte(zona, metal, riesgo_txt, dist, ph,
                                  vis["clorosis_pct"], vis["necrosis_pct"],
                                  api_key=api_key or None)
        st.markdown(rep)

# =========================================================
#  COLUMNA DERECHA: mapa y metricas del modelo
# =========================================================
with col_der:
    st.subheader("Mapa de riesgo - Arequipa")
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(location=[-16.6, -71.7], zoom_start=8, tiles="CartoDB positron")
        # Focos de riesgo conocidos
        for nombre, (la, lo) in ZONAS_RIESGO.items():
            folium.Marker([la, lo], tooltip=nombre,
                          icon=folium.Icon(color="red", icon="industry", prefix="fa")
                          ).add_to(m)
        # Muestras del dataset coloreadas por riesgo
        muestra = df.sample(min(150, len(df)), random_state=1)
        for _, r in muestra.iterrows():
            c = COLOR_RIESGO[ETIQUETAS[r["riesgo"]]]
            folium.CircleMarker([r["lat"], r["lon"]], radius=3,
                                color=c, fill=True, fill_opacity=0.6).add_to(m)
        st_folium(m, height=380, width=None)
    except Exception as e:
        st.warning(f"Mapa no disponible ({e}). Mostrando tabla de zonas.")
        st.dataframe(pd.DataFrame(
            [(n, la, lo) for n, (la, lo) in ZONAS_RIESGO.items()],
            columns=["Zona", "Lat", "Lon"]))

    st.subheader("Rendimiento del modelo (Random Forest)")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Accuracy", metricas["accuracy"])
    mc2.metric("F1 (macro)", metricas["f1_macro"])
    mc3.metric("AUC ROC", metricas["auc_roc"])
    mc1.metric("Precision", metricas["precision_macro"])
    mc2.metric("Recall", metricas["recall_macro"])

    st.write("**Matriz de confusion** (filas=real, col=predicho)")
    cm = pd.DataFrame(metricas["matriz_confusion"],
                      index=["Bajo", "Medio", "Alto"],
                      columns=["Bajo", "Medio", "Alto"])
    st.dataframe(cm)

    st.write("**Importancia de variables** (por que decide el modelo)")
    imp = pd.Series(metricas["importancias"]).sort_values(ascending=True)
    st.bar_chart(imp)

st.divider()
st.caption("Tamizaje preliminar de apoyo a la decision. No reemplaza analisis "
           "de laboratorio certificado (EPA 6020 / ICP-MS). Datos sinteticos "
           "calibrados con rangos de estudios reales de Arequipa.")
