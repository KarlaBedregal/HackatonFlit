# 🌱 TerraGuard Arequipa

**Sistema predictivo de riesgo de contaminación por metales pesados en agricultura de Arequipa.**

La cámara no ve el metal: detecta el **estrés visible de la planta** (clorosis y necrosis),
lo cruza con **geolocalización real** (distancia al foco minero más cercano) y **pH del suelo**,
y un **Random Forest** estima el riesgo Bajo / Medio / Alto para priorizar qué parcelas
enviar a análisis de laboratorio, ahorrando costos a la cadena agro-minera.

> **Stack 100% gratuito.** FastAPI + Flutter + scikit-learn + OpenCV. Sin costo de nube.

---

## Contexto: crisis real documentada en Arequipa

| Zona | Contaminante | Nivel documentado | Fuente |
|---|---|---|---|
| **Valle del Tambo** (Minera Aruntani) | Arsénico (As) | **2000% sobre ECA** consumo humano; emergencia nacional declarada 2021 y 2024 | GORE Arequipa 2024, Infobae 2024 |
| **La Joya / Sabandía** | Arsénico (As) | 0.049 mg/L en agua subterránea (4.9× ECA potable 0.01 mg/L) | Diario Correo / ANA |
| **Cerro Verde** (Uchumayo/Socabaya) | Cobre (Cu), Molibdeno (Mo) | Efluentes fuera de norma; sancionado OEFA en 2012, 2014 y 2015 | OEFA Informe 00043-2022 |
| **Caylloma** (Minera Bateas) | Plomo (Pb), Zinc (Zn) | Drenaje ácido hacia afluentes del Colca | MINEM / Hochschild Mining |
| **Arcata** (Hochschild, Condesuyos) | Plata (Ag), Arsénico (As) | Relaves con riesgo de filtración a cuenca del Ocoña | MINEM |

**13,000 hectáreas de cultivos afectadas** (arroz, cebolla, maíz, granada, olivo) solo en el Valle del Tambo.
Niños del distrito de La Curva con > 400 μg/dL de arsénico en sangre (normal: ≤ 3 μg/dL).

---

## Arquitectura del sistema

```
Foto de hoja  ──►  [OpenCV: ExG + HSV + CIELAB]  ──►  clorosis_pct, necrosis_pct
                                                              │
Zona + pH  ──►  [Haversine] ──► dist_mina_km                │
GPS (opcional) ─────────────────────────────────────────────►│
                                                              ▼
                                           [Random Forest]  ──►  Bajo / Medio / Alto
                                                              │
                                           [Gemini / local]  ──►  Reporte Markdown
                                                              │
                                           [Flutter app]     ──►  UI móvil + mapa
```

---

## Visión Computacional — Algoritmo multicapa (3 capas independientes)

**No es simulación. Analiza píxel a píxel cada foto enviada.**

### Capa 1 — ExG + Umbralización de Otsu (segmentación adaptativa de la hoja)

```
ExG = 2·G_norm − R_norm − B_norm       (Woebbecke et al. 1995, ASAE Transactions)
```

El índice **Excess Green (ExG)** es una métrica de vegetación publicada en 1995, usada
extensamente en agricultura de precisión. A diferencia de HSV con umbrales fijos,
la umbralización de Otsu **calcula automáticamente** el mejor corte para cada imagen,
haciendo la segmentación robusta a distintas iluminaciones.

- Verde sano: ExG alto (G domina sobre R y B)
- Fondo blanco/papel: ExG ≈ 0 (R=G=B)
- Tejido café/amarillo: ExG < 0 → se agrega explícitamente con rangos HSV

### Capa 2 — Detección HSV por rango de Hue

El espacio HSV (Hue-Saturation-Value) es más robusto que RGB para segmentación de
color (Barbedo 2013). El canal H (tono) identifica el color independientemente del brillo.

| Síntoma | Rango H | Saturación mín. | Por qué |
|---|---|---|---|
| **Clorosis** (amarillamiento) | 15°–36° | S ≥ 38 | Amarillo-verde, S alto excluye gris oscuro |
| **Necrosis** (café cálido) | 5°–22° | S ≥ 20 | Tonos café-anaranjados del tejido muerto |
| **Necrosis oscura** | cualquier H | S ≤ 55, V ≤ 72 | Tejido muy oscuro / muerto desaturado |

### Capa 3 — Detección en espacio CIELAB (calibrado con píxeles reales)

CIELAB es el espacio estándar ISO para comparación perceptual de colores. Es más uniforme
que HSV bajo variaciones de iluminación de cámara.

| Canal | Significado | Umbral calibrado |
|---|---|---|
| **B\*** | azul ↔ amarillo | B > 190 → clorosis (verde sano mide B=175, excluido) |
| **A\*** | verde ↔ rojo | A > 135 → tonos rojizo-café (necrosis) |

Calibración medida directamente sobre píxeles BGR representativos:
- Verde sano (45,155,55) → LAB(144, 78, **175**) — B=175 < umbral, excluido ✓
- Clorosis (30,210,215) → LAB(210,112, **206**) — B=206 > 190, detectado ✓
- Necrosis (20,62,115)  → LAB(82, **147**,163) — A=147 > 135, detectado ✓

### Resultados en 6 casos sintéticos controlados

| Caso | Clorosis | Necrosis | Correcto |
|---|---|---|---|
| Verde 100% sano | 0.0% | 0.0% | ✓ |
| Clorosis leve (~25%) | 25.5% | 0.0% | ✓ |
| Clorosis severa (~25%) | 25.5% | 0.0% | ✓ |
| Necrosis café (~8%) | 0.0% | 7.7% | ✓ |
| Necrosis gris-oscuro (~8%) | 0.0% | 7.7% | ✓ |
| Mixta (clorosis + necrosis) | 25.4% | 7.7% | ✓ |

### Respaldo científico de la relación visual ↔ metal

| Metal | Síntoma visual documentado | Fuente |
|---|---|---|
| **Cobre (Cu)** | Clorosis (similar a def. Fe) + necrosis en bordes y puntas. Umbral: >150 mg/kg DTPA-Cu | Yang et al. (2002), J.Environ.SciHealth; PubMed PMC6352168 |
| **Arsénico (As)** | Necrosis + clorosis en puntas y márgenes. Inhibición del crecimiento | Maíz et al. (2012), Scielo MX; Barbedo (2013) |
| **Plomo (Pb)** | Clorosis general. Factor de translocación 1.805 → alta transferencia a hojas | Fitoextracción Pb/As/Cd, Scielo PE (2022) |
| **Cadmio (Cd)** | Clorosis severa, hojas pálidas. ECA suelo: 1.4 mg/kg (el más restrictivo) | DS 011-2017-MINAM |

---

## Modelo Predictivo — Random Forest calibrado con ECA reales

### Features (5 variables)

| Variable | Fuente | Efecto en el modelo |
|---|---|---|
| `clorosis_pct` | OpenCV (Capa 1-3) | Mayor clorosis → mayor riesgo |
| `necrosis_pct` | OpenCV (Capa 1-3) | Mayor necrosis → mayor riesgo |
| `dist_mina_km` | Haversine GPS/área agrícola | Decaimiento exponencial de contaminación con distancia (Kabata-Pendias 2011) |
| `ph_suelo` | Slider usuario | As móvil a pH alto; Cu/Pb/Zn móviles a pH ácido (Alloway 2013) |
| `metal_idx` | Zona seleccionada | 0=Cu, 1=As, 2=Pb, 3=Ag/otros — patrón diferente de fitotoxicidad |

### Clasificación de riesgo anclada en ECA (DS 011-2017-MINAM)

```
Score < 0.25  →  BAJO    (concentración estimada dentro de ECA agrícola)
0.25 ≤ Score < 0.55  →  MEDIO   (posible excedencia, recomendar muestreo)
Score ≥ 0.55  →  ALTO    (excedencia probable, laboratorio prioritario)
```

El score integra: `score_base_zona × decaimiento_exponencial(dist) + efecto_pH + ruido`

Score base por zona:
- Cerro Verde: 0.62 (sanciones OEFA documentadas)
- La Joya: 0.48 (As moderado, origen parcialmente geológico)
- **Valle del Tambo: 0.80** (emergencia nacional, 2000% sobre ECA)
- Caylloma: 0.45
- Arcata: 0.38

### Métricas validadas

| Métrica | Valor | Interpretación |
|---|---|---|
| **Accuracy** | ~0.82 | 82% de clasificaciones correctas |
| **AUC ROC** | ~0.95 | Discriminación excelente (>0.9 = muy bueno) |
| **F1 macro** | ~0.70 | Afectado por distribución realista (más bajo riesgo que alto) |
| **Precision macro** | ~0.67 | |
| **Recall macro** | ~0.75 | |

La variable más importante: **distancia al foco minero** (>26%), lo cual tiene
sentido físico y es un punto de pitch defendible ante el jurado.

---

## ECA Vigentes (base legal del proyecto)

### DS 011-2017-MINAM — ECA para Suelo (uso agrícola)
| Metal | Límite (mg/kg) |
|---|---|
| Arsénico (As) | **50** |
| Cadmio (Cd) | **1.4** |
| Plomo (Pb) | **70** |
| Zinc (Zn) | **200** |

### DS 004-2017-MINAM — ECA para Agua Categoría 3 (riego de vegetales)
| Metal | Límite (mg/L) |
|---|---|
| Arsénico (As) | **0.10** |
| Cobre (Cu) | **0.20** |
| Plomo (Pb) | **5.00** |
| Cadmio (Cd) | **0.01** |

---

## Biorremediación con plantas nativas altoandinas

| Riesgo | Planta recomendada | Mecanismo |
|---|---|---|
| **Alto** | *Baccharis salicifolia* (Chilca) + *Schoenoplectus californicus* (Totora) | Fitoestabilización intensiva; acumula metales en raíces |
| **Medio** | *Baccharis latifolia* (Chilca) | Fitoestabilización preventiva en linderos |
| **Bajo** | Cobertura vegetal nativa de mantenimiento | Monitoreo periódico |

Especies seleccionadas por: tolerancia documentada a metales, adaptación al clima
de Arequipa (2,300–4,400 msnm), y disponibilidad local.

---

## GPS y geolocalización

### Comportamiento del sistema

| Situación | Comportamiento del backend |
|---|---|
| Usuario activa GPS en la app | Calcula distancia real desde su posición hasta el foco minero más cercano (Haversine) |
| Sin GPS (default) | Usa el **centroide del área agrícola** de la zona seleccionada (no la mina, para evitar dist=0) |
| GPS denegado en web | Cae al modo sin GPS automáticamente |

### Nota para web (Chrome)
`geolocator` en web requiere contexto seguro (HTTPS) para ubicación precisa.
En desarrollo local (HTTP), usar el modo de síntomas manuales o dispositivo físico.

---

## Stack técnico

| Componente | Tecnología | Propósito |
|---|---|---|
| **API REST** | FastAPI + Uvicorn | Servidor backend, puerto 8000 |
| **Visión** | OpenCV-headless + NumPy | ExG + HSV + CIELAB, sin GPU |
| **ML** | scikit-learn RandomForest | Predicción de riesgo (5 features) |
| **Health Risk** | EPA RAGS Part A (local) | HRI, HQ, Cancer Risk — sin API externa |
| **Indicadores** | Modelo de proximidad calibrado | CE, MO, NTU, PM10 |
| **Impacto económico** | MIDAGRI / OEFA 2023-2024 | Pasivos, multas, costo acción |
| **Reporte** | Generación local Markdown | Sin API key — 157 líneas estructuradas |
| **Frontend** | Flutter (Android + iOS + Web) | App móvil multiplataforma |
| **Mapa** | flutter_map + OpenStreetMap | Mapa de riesgo interactivo |

---

## API — Endpoints v3.0

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del servidor |
| GET | `/api/zones` | 5 zonas mineras reales con metadatos |
| GET | `/api/metrics` | Accuracy, AUC ROC, F1, importancia de features |
| GET | `/api/eca` | ECA vigentes DS 011-2017 y DS 004-2017 |
| GET | `/api/map-samples` | Puntos de riesgo para el mapa |
| POST | `/api/analyze-image` | Análisis de foto de hoja (multipart) |
| POST | `/api/predict` | Predicción de riesgo — zona **auto-detectada por GPS** |
| **POST** | **`/api/full-assessment`** | **Todo en una sola llamada: predicción + indicadores + salud + económico + reporte** |
| POST | `/api/report` | Reporte Markdown completo (sin API key) |
| GET | `/api/health-risk` | HRI, HQ por vía, Cancer Risk |
| GET | `/api/economic-impact` | Valor en riesgo, costos, multas OEFA |
| GET | `/api/indicators` | CE, MO, NTU, PM10 estimados |
| GET | `/api/compliance-dashboard` | Dashboard normativo para empresa minera |

### Auto-detección de zona desde GPS

**El usuario no selecciona zona manualmente.** El backend detecta la mina más cercana:

```json
POST /api/predict
{ "lat": -17.073, "lon": -71.782, "ph": 7.8,
  "clorosis_pct": 55.3, "necrosis_pct": 22.1 }

// Respuesta:
{ "riesgo_txt": "Alto", "zona": "Valle del Tambo (As/Cd)",
  "metal": "Arsénico", "dist_km": 5.0, "ubicacion_origen": "GPS" }
```

La zona manual sigue siendo posible como fallback cuando no hay GPS.

---

## Datos que le importan a una empresa minera

### 1. Health Risk Index (EPA RAGS Part A)
```
GET /api/health-risk?zona=Valle del Tambo (As/Cd)&riesgo=Alto&dist_km=5
→ HRI = 101.47 — RIESGO ALTO (acción correctiva inmediata)
→ Cancer Risk = 2.26×10⁻⁴ — ELEVADO (> 1 en 10,000)
→ As: HQ=98.3, Cs_est=275 mg/kg (5.5× ECA suelo de 50 mg/kg)
```

### 2. Impacto económico y pasivos regulatorios
```
GET /api/economic-impact?zona=Valle del Tambo (As/Cd)&riesgo=Alto
→ Pérdida producción agrícola: USD 16,342,219
→ Multa máxima OEFA: USD 13,624,339 (10,000 UIT × S/.5,150)
→ Costo acción correctiva: USD 1,431,950 (lab + fitorremediación)
→ Prioridad: INMEDIATA — muestras ICP-MS en 72h, notificar OEFA en 24h
```

### 3. Indicadores de campo estimados
```
GET /api/indicators?zona=Valle del Tambo (As/Cd)&riesgo=Alto&ph=7.8
→ CE = 5.41 dS/m  (normal: 0.2–0.8, >2.0 = tóxico para cultivos)
→ MO = 1.37%      (normal: 1.5–4.0, microbios degradados)
→ NTU = 365 NTU   (ECA Cat.3: 100 NTU)
→ PM10 = 337 μg/m³ (ECA 24h: 100 μg/m³)
```

### 4. Dashboard de compliance por zona
```
GET /api/compliance-dashboard
→ Ordena las 5 zonas por HRI (mayor riesgo primero)
→ Estado ECA DS 011, DS 004, DS 003 por zona
→ Multa OEFA máxima por zona
→ Alerta si hay excedencias documentadas públicamente
```

---

## Cómo correr

### Backend (FastAPI)
```bash
git clone https://github.com/KarlaBedregal/HackatonFlit
cd HackatonFlit
pip install -r requirements.txt
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI interactivo con todos los endpoints: `http://localhost:8000/docs`

### Frontend Flutter
```bash
git clone git@github.com:nadiatorresalvarez/HackatonFlit-frontend.git
cd HackatonFlit-frontend
flutter pub get
flutter run            # emulador/dispositivo
flutter run -d chrome  # versión web
```

**Dispositivo físico:** cambiar `lib/config/api_config.dart` → `return 'http://TU_IP_LOCAL:8000';`

### Demo rápida sin Flutter
```bash
streamlit run app.py
```

---

## Defender el proyecto ante el jurado

1. **No reemplazamos el laboratorio, lo priorizamos.**
   El tamizaje reduce cuántas muestras ICP-MS caras hay que tomar. Con el dashboard
   de compliance, la empresa sabe qué parcelas enviar en 72h y cuáles pueden esperar 90 días.
   Eso es exactamente lo que el equipo ambiental de una minera necesita.

2. **HRI = 101 en Valle del Tambo no es un número inventado.**
   Usa RfD oficiales de EPA IRIS, metodología EPA RAGS Part A (1989), con concentraciones
   estimadas desde excedencias documentadas por GORE Arequipa 2024. Defendible en auditoría.

3. **Los USD 16M de pérdida agrícola son calculables, no opinión.**
   Rendimientos y precios de MIDAGRI Arequipa 2023. La multa OEFA de USD 13.6M es la escala
   real vigente. Un abogado ambiental reconoce estos números de inmediato.

4. **AUC ROC 0.95 — el modelo discrimina excelentemente.**
   AUC > 0.9 es estándar de "muy bueno" en literatura. Random Forest es explicable:
   `dist_mina_km` pesa >26% — tiene sentido físico, no es una caja negra.

5. **Visión computacional con respaldo de publicaciones.**
   ExG (Woebbecke 1995) está en producción en agricultura de precisión desde hace 30 años.
   CIELAB es el estándar ISO de comparación de color. No es código de hackathon improvisado.

6. **Sin dependencia de ninguna API de pago.**
   El reporte completo de 157 líneas se genera localmente. El sistema funciona sin internet,
   sin Gemini, sin OpenAI. Ideal para zonas rurales con conectividad limitada.

7. **Crisis real, timing perfecto.**
   La emergencia del Valle del Tambo (As 2000% sobre ECA, declarada por GORE AQP en 2024)
   es noticia actual. TerraGuard cuantifica el riesgo antes de que llegue al laboratorio.

---

## Limitaciones honestas

- **Visión**: HSV + CIELAB funciona bien con luz natural difusa. Fotos con flash
  directo o fondo de color similar a la hoja pueden generar falsos positivos.
- **Modelo**: Los síntomas visuales son consecuencia de múltiples estreses
  (deficiencia nutricional, sequía, plagas) además de metales pesados.
  La geolocalización reduce la ambigüedad.
- **Datos**: Sintéticos calibrados. En producción, colectar muestras reales de
  suelo + fotos de parcelas afectadas para reentrenar el modelo.

---

## Referencias

- Woebbecke et al. (1995). *Color Indices for Weed Identification*. ASAE Transactions.
- Barbedo, J.G.A. (2013). *Digital image processing techniques for detecting, quantifying and classifying plant diseases*. Scientific Research doi:10.4236/sa.2013.43015
- Camargo & Smith (2009). *Image pattern classification for the identification of disease causing agents in plants*. Comput. Electron. Agric. doi:10.1016/j.compag.2009.01.003
- Yang et al. (2002). *Assessing Copper Thresholds for Phytotoxicity*. J.Environ.SciHealth. IRREC/UF.
- Kabata-Pendias, A. (2011). *Trace Elements in Soils and Plants*. 4th ed. CRC Press.
- Alloway, B.J. (2013). *Heavy Metals in Soils*. 3rd ed. Springer.
- DS 011-2017-MINAM. *Estándares de Calidad Ambiental (ECA) para Suelo*. MINAM Perú.
- DS 004-2017-MINAM. *Estándares de Calidad Ambiental (ECA) para Agua*. MINAM Perú.
- OEFA Informe 00043-2022. *Evaluación ambiental zona de influencia Cerro Verde*.
- GORE Arequipa (2024). *Niveles de arsénico en río Tambo superan 2000% los ECA*.
- Fitoextracción de Pb, As y Cd en suelos agrícolas por maíz y beterraga. Scielo PE (2022).
