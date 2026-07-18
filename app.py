"""
app.py  -  TerraGuard Arequipa
------------------------------
MVP: tamizaje predictivo de riesgo de contaminacion por metales pesados.

Flujo:
  1) Usuario indica ubicacion de parcela (dropdown o clic en mapa)
  2) Distancia geodesica (geopy) al proyecto minero mas cercano
  3) pH por lookup de zona (o manual si tiene dato de campo)
  4) Foto opcional -> OpenCV extrae clorosis/necrosis
  5) Random Forest -> riesgo Bajo/Medio/Alto
  6) Folium muestra parcela coloreada por riesgo

Ejecutar:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from data_generator import (
    generar_dataset_hibrido,
    ZONAS_PARCELA,
    PROYECTOS_MINEROS,
    PH_POR_ZONA,
    ETIQUETAS,
    coords_parcela_desde_zona,
    zona_parcela_desde_coordenadas,
    calcular_distancia_proyecto,
    obtener_ph_por_zona,
)
from vision import analizar_hoja, leer_imagen_desde_bytes
import model as M
from reporte import generar_reporte

st.set_page_config(page_title="TerraGuard Arequipa", page_icon="🌱", layout="wide")

COLOR_RIESGO = {"Bajo": "#2e7d32", "Medio": "#f9a825", "Alto": "#c62828"}
COLOR_RIESGO_FOLIUM = {"Bajo": "green", "Medio": "orange", "Alto": "red"}
DEFAULT_ZONA = "La Joya (vinedos)"


def _init_session():
    if "parcela_lat" not in st.session_state:
        p = coords_parcela_desde_zona(DEFAULT_ZONA)
        st.session_state.parcela_lat = p["lat"]
        st.session_state.parcela_lon = p["lon"]
        st.session_state.parcela_nombre = p["nombre_zona"]
        st.session_state.zona_ph = p["zona_ph"]
        st.session_state.ubicacion_fuente = "dropdown"
        st.session_state.last_click = None


def _actualizar_desde_dropdown(nombre_zona: str):
    p = coords_parcela_desde_zona(nombre_zona)
    st.session_state.parcela_lat = p["lat"]
    st.session_state.parcela_lon = p["lon"]
    st.session_state.parcela_nombre = p["nombre_zona"]
    st.session_state.zona_ph = p["zona_ph"]
    st.session_state.ubicacion_fuente = "dropdown"
    st.session_state.last_click = None


def _actualizar_desde_mapa(lat: float, lon: float):
    zp = zona_parcela_desde_coordenadas(lat, lon)
    st.session_state.parcela_lat = lat
    st.session_state.parcela_lon = lon
    st.session_state.parcela_nombre = f"Mapa — cerca de {zp['nombre_zona']}"
    st.session_state.zona_ph = zp["zona_ph"]
    st.session_state.ubicacion_fuente = "mapa"
    st.session_state.last_click = (lat, lon)


@st.cache_resource
def cargar_todo():
    modelo, metricas, _ = M.entrenar(modo="hibrido")
    df = generar_dataset_hibrido()
    return df, modelo, metricas


def _crear_mapa(lat, lon, riesgo_txt=None):
    import folium

    m = folium.Map(location=[lat, lon], zoom_start=9, tiles="CartoDB positron")

    for nombre, info in PROYECTOS_MINEROS.items():
        folium.Marker(
            [info["lat"], info["lon"]],
            tooltip=f"{nombre} — {info['tipo']}",
            popup=folium.Popup(f"<b>{nombre}</b><br>{info['tipo']}", max_width=220),
            icon=folium.Icon(color="darkred", icon="industry", prefix="fa"),
        ).add_to(m)

    color = COLOR_RIESGO.get(riesgo_txt, "#1976d2")
    folium_color = COLOR_RIESGO_FOLIUM.get(riesgo_txt, "blue")
    etiqueta = riesgo_txt or "Parcela"

    folium.CircleMarker(
        [lat, lon],
        radius=14,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=3,
        popup=folium.Popup(
            f"<b>Tu parcela</b><br>Riesgo: {etiqueta}<br>Lat: {lat:.4f}, Lon: {lon:.4f}",
            max_width=240,
        ),
        tooltip=f"Tu parcela — Riesgo {etiqueta}",
    ).add_to(m)

    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color=folium_color, icon="leaf", prefix="fa"),
    ).add_to(m)

    return m


_init_session()
df, modelo, metricas = cargar_todo()

st.title("🌱 TerraGuard Arequipa")
st.caption(
    "Tamizaje predictivo de riesgo de contaminacion por metales pesados "
    "para priorizar muestreo de laboratorio."
)

col_izq, col_der = st.columns([1, 1])

# --- PASO 1: UBICACION DE PARCELA (independiente de la foto) ---
with col_izq:
    st.subheader("1. Ubicacion de tu parcela")
    st.caption("Indica donde esta tu parcela. Este paso es independiente de la foto.")

    zonas_list = list(ZONAS_PARCELA.keys())
    idx = zonas_list.index(st.session_state.parcela_nombre) if st.session_state.parcela_nombre in zonas_list else 0

    zona_dropdown = st.selectbox(
        "Zona agricola (lista)",
        zonas_list,
        index=idx,
        help="Selecciona la zona mas cercana a tu parcela.",
    )

    if st.session_state.ubicacion_fuente == "dropdown":
        _actualizar_desde_dropdown(zona_dropdown)
    elif st.button("Volver a usar la zona del listado"):
        _actualizar_desde_dropdown(zona_dropdown)
        st.rerun()

    if st.session_state.ubicacion_fuente == "mapa":
        st.success(
            f"Ubicacion por **clic en mapa**: "
            f"{st.session_state.parcela_lat:.4f}, {st.session_state.parcela_lon:.4f}"
        )
    else:
        st.info(f"Ubicacion: **{st.session_state.parcela_nombre}**")

    dist_info = calcular_distancia_proyecto(
        st.session_state.parcela_lat, st.session_state.parcela_lon
    )
    dist = dist_info["dist_km"]
    metal = PH_POR_ZONA[st.session_state.zona_ph]["metal_principal"]

    st.metric("Distancia al proyecto minero mas cercano", f"{dist} km")
    st.caption(
        f"Proyecto: **{dist_info['proyecto_cercano']}** ({dist_info['tipo_proyecto']})"
    )
    with st.expander("Distancias a Cerro Verde y Tia Maria"):
        for proj, d in dist_info["distancias_todos"].items():
            st.write(f"- **{proj}**: {d} km")

    st.subheader("2. pH del suelo")
    ph_lookup = obtener_ph_por_zona(st.session_state.zona_ph)
    ph_manual = st.checkbox(
        "Tengo medicion de campo (usar pH manual)",
        help="Si tienes un dato de laboratorio o kit de campo, ingresalo aqui.",
    )
    if ph_manual:
        ph = st.number_input(
            "pH medido en tu parcela",
            min_value=4.0, max_value=9.0, value=float(ph_lookup), step=0.1,
        )
    else:
        ph = ph_lookup
        st.info(
            f"pH estimado por zona: **{ph}** "
            f"(lookup — {PH_POR_ZONA[st.session_state.zona_ph]['fuente'][:70]}...)"
        )

    st.subheader("3. Foto de la hoja (opcional)")
    st.caption("La foto solo analiza sintomas visuales; no define la ubicacion.")
    foto = st.file_uploader("Sube una foto de la hoja del cultivo", type=["jpg", "jpeg", "png"])

    if foto is not None:
        img = leer_imagen_desde_bytes(foto.read())
        if img is not None:
            vis = analizar_hoja(img)
            st.image(foto, caption="Hoja analizada", width=250)
        else:
            vis = analizar_hoja(np.zeros((10, 10, 3), np.uint8))
            st.warning("No se pudo leer la imagen, usando estimacion de respaldo.")
    else:
        st.markdown("*Sin foto: ajusta manualmente los sintomas visuales*")
        c1, c2 = st.columns(2)
        vis = {
            "clorosis_pct": c1.slider("Clorosis %", 0, 100, 30),
            "necrosis_pct": c2.slider("Necrosis %", 0, 100, 15),
            "ok": True,
            "msg": "manual",
        }

    st.write(
        f"Clorosis: **{vis['clorosis_pct']:.0f}%** | "
        f"Necrosis: **{vis['necrosis_pct']:.0f}%**"
    )

    pred = M.predecir_riesgo(modelo, vis["clorosis_pct"], vis["necrosis_pct"], dist, ph)
    riesgo_txt = pred["riesgo_txt"]

    st.subheader("4. Resultado")
    st.markdown(
        f"<h2 style='color:{COLOR_RIESGO[riesgo_txt]}'>Riesgo: {riesgo_txt} "
        f"({pred['confianza']}% conf.)</h2>",
        unsafe_allow_html=True,
    )
    st.write("Probabilidades:", pred["probabilidades"])

    st.subheader("5. Reporte y biorremediacion")
    api_key = st.text_input("API key de Gemini (opcional)", type="password")
    if st.button("Generar reporte", type="primary"):
        with st.spinner("Generando reporte..."):
            rep = generar_reporte(
                st.session_state.parcela_nombre, metal, riesgo_txt, dist, ph,
                vis["clorosis_pct"], vis["necrosis_pct"],
                api_key=api_key or None,
            )
        st.markdown(rep)

# --- MAPA INTERACTIVO ---
with col_der:
    st.subheader("Mapa — Arequipa")
    st.caption("Haz **clic** en el mapa para marcar tu parcela. Tambien puedes usar el listado.")

    try:
        import folium
        from streamlit_folium import st_folium

        m = _crear_mapa(
            st.session_state.parcela_lat,
            st.session_state.parcela_lon,
            riesgo_txt,
        )

        muestra = df.sample(min(80, len(df)), random_state=1)
        for _, r in muestra.iterrows():
            folium.CircleMarker(
                [r["lat"], r["lon"]], radius=2,
                color=COLOR_RIESGO[ETIQUETAS[r["riesgo"]]],
                fill=True, fill_opacity=0.35, weight=1,
            ).add_to(m)

        map_data = st_folium(
            m, height=420, width=None,
            returned_objects=["last_clicked"],
            key="mapa_parcela",
        )

        clicked = map_data.get("last_clicked") if map_data else None
        if clicked and clicked.get("lat") is not None:
            clat, clon = clicked["lat"], clicked["lng"]
            prev = st.session_state.last_click
            if prev is None or abs(prev[0] - clat) > 1e-5 or abs(prev[1] - clon) > 1e-5:
                _actualizar_desde_mapa(clat, clon)
                st.rerun()

    except Exception as e:
        st.warning(f"Mapa no disponible ({e}).")
        st.dataframe(pd.DataFrame(
            [(n, v["lat"], v["lon"]) for n, v in ZONAS_PARCELA.items()],
            columns=["Zona", "Lat", "Lon"],
        ))

    st.subheader("Rendimiento del modelo")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Accuracy", metricas["accuracy"])
    mc2.metric("F1 (macro)", metricas["f1_macro"])
    mc3.metric("AUC ROC", metricas["auc_roc"])

    imp = pd.Series(metricas["importancias"]).sort_values(ascending=True)
    st.bar_chart(imp)

st.divider()
st.caption(
    "Tamizaje preliminar. No reemplaza analisis de laboratorio certificado (ICP-MS). "
    "Distancias calculadas con geopy (geodesica). pH por lookup regional o dato manual."
)
