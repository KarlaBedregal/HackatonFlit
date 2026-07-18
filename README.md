# 🌱 TerraGuard Arequipa — Plan de ataque del hackatón

Sistema predictivo de **riesgo de contaminación por metales pesados** en la
agricultura de Arequipa. La cámara no ve el metal: ve el **estrés de la planta**,
y lo cruzamos con **geolocalización** (distancia a la mina) y **pH del suelo**.
Un Random Forest predice riesgo **Bajo / Medio / Alto**, y Gemini redacta un
reporte con recomendación de **biorremediación con plantas nativas**.

> **Todo el stack es gratis.** No requiere ninguna suscripción de pago.

---

## 1. División de roles (4 devs)

### 👤 Dev 1 — Datos & Modelo (ML)
- **Misión:** dataset simulado + entrenar el Random Forest + sacar métricas.
- **Librerías:** `scikit-learn`, `pandas`, `numpy`, `joblib`.
- **Archivos que posee:** `src/data_generator.py`, `src/model.py`, el Colab.
- **Entrega al equipo:** `modelo_rf.joblib` + el dict `metricas` (accuracy, F1,
  AUC, matriz de confusión, importancias). Es la base que todos consumen.

### 👤 Dev 2 — Visión (OpenCV)
- **Misión:** de una foto de hoja, sacar `% clorosis` y `% necrosis`.
- **Librerías:** `opencv-python-headless`, `numpy`.
- **Archivo que posee:** `src/vision.py`.
- **Recibe:** bytes de la foto (del uploader). **Entrega:** un dict
  `{clorosis_pct, necrosis_pct, ok, msg}` con **fallback** si la foto falla.

### 👤 Dev 3 — IA generativa & Reporte (Gemini)
- **Misión:** convertir la predicción en un reporte claro con biorremediación.
- **Librerías:** `google-generativeai` (+ fallback local sin API).
- **Archivo que posee:** `src/reporte.py`.
- **Recibe:** zona, metal, riesgo, dist, pH, clorosis, necrosis.
  **Entrega:** texto Markdown listo para mostrar. **Nunca se cae** (fallback).

### 👤 Dev 4 — UI, Mapa & Pitch (Streamlit + Folium)
- **Misión:** ensamblar todo en `app.py`, mapa interactivo y guion del pitch.
- **Librerías:** `streamlit`, `folium`, `streamlit-folium`.
- **Archivo que posee:** `app.py`.
- **Recibe:** los módulos de los otros 3. **Entrega:** la app corriendo + demo.

---

## 2. Plan de ataque de 3 horas

| Tiempo | Qué pasa |
|---|---|
| **0:00–0:20** | Setup común: clonar repo, `pip install -r requirements.txt`, repartir roles. Cada quien crea SU rama. |
| **0:20–2:00** | **Trabajo en paralelo.** Cada dev trabaja SOLO en su archivo (ver abajo por qué evita conflictos de Git). Dev 1 corre el Colab y sube `modelo_rf.joblib`. |
| **2:00–2:40** | **Integración.** Dev 4 hace merge de las 4 ramas a `main`. Como cada uno tocó archivos distintos, los merges son limpios. |
| **2:40–3:00** | **Pulido + ensayo.** Correr `streamlit run app.py`, probar el demo 2 veces, preparar el caso de La Joya para mostrar en vivo. |

### ¿Cómo trabajar en paralelo SIN conflictos de Git?
El proyecto está **modularizado por archivo = por persona**. La regla de oro:
**cada dev solo edita su propio archivo.**
- Dev 1 → `src/data_generator.py`, `src/model.py`
- Dev 2 → `src/vision.py`
- Dev 3 → `src/reporte.py`
- Dev 4 → `app.py`

Como Git hace merge sin conflictos cuando la gente toca archivos diferentes,
esto elimina el 90% de los choques. Flujo:
```bash
git checkout -b dev1-modelo     # cada quien su rama
# ...trabajar solo en tu archivo...
git add src/model.py && git commit -m "modelo listo"
git push origin dev1-modelo
# al final, Dev 4 mergea todas las ramas a main
```

### 💡 Usen Cursor / Lovable para acelerar
- **Cursor:** para completar/ajustar el código de cada módulo y escribir tests
  rápidos. Ya tienen la base aquí; úsenlo para pulir, no para reescribir.
- **Lovable:** si quieren una landing/pitch page bonita aparte del Streamlit.

---

## 3. Cómo se ensambla el MVP (última hora)

El contrato entre módulos ya está definido (los inputs/outputs de arriba).
`app.py` importa los 3 módulos y los orquesta:

```
foto ─▶ vision.analizar_hoja() ─▶ {clorosis, necrosis}
                                        │
zona + pH ─▶ dist_a_mina_mas_cercana() ─┤
                                        ▼
                        model.predecir_riesgo() ─▶ riesgo
                                        │
                                        ▼
                        reporte.generar_reporte() ─▶ Markdown
                                        │
                        folium ─▶ mapa de riesgo de Arequipa
```

Si un módulo falla, su fallback mantiene la app viva (buenas prácticas de hackatón).

---

## 4. Cómo correr

```bash
pip install -r requirements.txt
streamlit run app.py
```

El modelo se entrena solo al arrancar (está cacheado, tarda ~2s). El Colab
(`TerraGuard_Arequipa_Colab.ipynb`) es para mostrar las métricas al jurado
con gráficos: súbanlo a Google Colab y Runtime > Run all.

---

## 5. Métricas del modelo (ejemplo de corrida)

- **Accuracy:** ~0.89
- **Precision (macro):** ~0.92
- **Recall (macro):** ~0.88
- **F1 (macro):** ~0.89
- **AUC ROC:** ~0.96
- Variable más importante: **distancia a la mina** (tiene sentido físico → gran punto de pitch).

---

## 6. Notas para defender el proyecto (honestas y fuertes)

1. **No reemplazamos el laboratorio, lo priorizamos.** Somos un tamizaje que
   reduce cuántas muestras caras hay que analizar → ahorro real a la minera/agro.
2. **Datos sintéticos calibrados** con rangos de estudios reales de Arequipa
   (La Joya ~0.021 mg/L As en agua; Tambo; Cerro Verde con observaciones de OEFA).
   En producción → muestreo primario. Somos transparentes en esto.
3. **Random Forest es explicable:** mostramos qué variables pesan (no una caja negra).
4. **Biorremediación con plantas nativas** (Baccharis salicifolia, totora) cierra
   el ciclo: detectar → recomendar remediación.
