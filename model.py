"""
model.py  (Dev 1 — Datos & Modelo)
----------------------------------
Entrenamiento y evaluacion del Random Forest para predecir riesgo de
contaminacion (Bajo/Medio/Alto).

Modos:
  - "hibrido" (default): generar_dataset_entrenamiento() — ICP-MS real + bootstrap
  - "sintetico": fallback calibrado con PH_POR_ZONA

Entrega al equipo: modelo_rf.joblib + dict metricas
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from data_generator import (
    generar_dataset,
    generar_dataset_entrenamiento,
    generar_filas_reales,
    cargar_muestras_reales,
    resumen_fuentes_datos,
    FEATURES,
    ETIQUETAS,
)


def _split_train_test(df, seed=42):
    X = df[FEATURES]
    y = df["riesgo"]
    try:
        return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(X, y, test_size=0.2, random_state=seed)


def _metricas_basicas(y_test, y_pred, y_proba, clases):
    y_test_bin = label_binarize(y_test, classes=clases)
    try:
        if y_test_bin.shape[1] <= 1:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "auc_roc": round(auc, 4) if not np.isnan(auc) else auc,
        "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
        "reporte": classification_report(
            y_test, y_pred,
            labels=clases,
            target_names=[ETIQUETAS[c] for c in clases],
            zero_division=0,
        ),
    }


def evaluar_en_muestras_reales(modelo, seed=42):
    """
    Validacion honesta: predice sobre las 4 filas ICP-MS reales (sin bootstrap).
    Util para el pitch — muestra si el modelo respeta sitios verificables.
    """
    df_real = generar_filas_reales(seed=seed)
    X = df_real[FEATURES]
    y_true = df_real["riesgo"]
    y_pred = modelo.predict(X)
    filas = []
    for i, row in df_real.iterrows():
        filas.append({
            "sitio": row["sitio_campo"],
            "ph_suelo": row["ph_suelo"],
            "as_mgkg": row["as_mgkg"],
            "rep_indice": row["rep_indice"],
            "riesgo_real": ETIQUETAS[int(row["riesgo"])],
            "riesgo_predicho": ETIQUETAS[int(y_pred[i])],
            "acierto": int(row["riesgo"]) == int(y_pred[i]),
        })
    return {
        "n_muestras": len(df_real),
        "accuracy_sitios_reales": round(float((y_true == y_pred).mean()), 4),
        "detalle": filas,
    }


def entrenar(df=None, seed=42, modo="hibrido", n_por_zona=60):
    """
    Entrena el RF y devuelve (modelo, metricas, datos_test).

    modo: "hibrido" | "sintetico"
    """
    if df is None:
        if modo == "hibrido":
            df = generar_dataset_entrenamiento(n_por_zona=n_por_zona, seed=seed)
        else:
            df = generar_dataset(seed=seed)

    X_train, X_test, y_train, y_test = _split_train_test(df, seed=seed)

    modelo = RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        class_weight="balanced",
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    clases = sorted(df["riesgo"].unique())

    metricas = _metricas_basicas(y_test, y_pred, y_proba, clases)
    metricas.update({
        "modo": modo,
        "n_sitios_reales_icpms": len(cargar_muestras_reales()) if modo == "hibrido" else None,
        "n_muestras_total": len(df),
        "n_muestras_reales_en_train": int(df.get("tipo_muestra", pd.Series()).eq("real").sum()) if "tipo_muestra" in df.columns else None,
        "n_muestras_bootstrap": int(df.get("tipo_muestra", pd.Series()).eq("bootstrap").sum()) if "tipo_muestra" in df.columns else None,
        "fuentes_datos": resumen_fuentes_datos(),
        "validacion_sitios_reales": evaluar_en_muestras_reales(modelo, seed=seed),
        "importancias": pd.Series(
            modelo.feature_importances_, index=FEATURES
        ).sort_values(ascending=False).to_dict(),
    })

    return modelo, metricas, (X_test, y_test, y_pred, y_proba)


def predecir_riesgo(modelo, clorosis, necrosis, dist_km, ph):
    """Predice el nivel de riesgo para un caso individual (contrato con app.py)."""
    X = pd.DataFrame([{
        "clorosis_pct": clorosis,
        "necrosis_pct": necrosis,
        "dist_mina_km": dist_km,
        "ph_suelo": ph,
    }])[FEATURES]
    pred = int(modelo.predict(X)[0])
    proba = modelo.predict_proba(X)[0]
    return {
        "riesgo_num": pred,
        "riesgo_txt": ETIQUETAS[pred],
        "confianza": round(float(proba[pred]) * 100, 1),
        "probabilidades": {ETIQUETAS[i]: round(float(p) * 100, 1)
                           for i, p in enumerate(proba)},
    }


def guardar(modelo, ruta="modelo_rf.joblib"):
    joblib.dump(modelo, ruta)


def cargar(ruta="modelo_rf.joblib"):
    return joblib.load(ruta)


def entrenar_y_exportar(ruta_modelo="modelo_rf.joblib",
                        ruta_dataset="dataset_arequipa_hibrido.csv",
                        seed=42, n_por_zona=60):
    """Pipeline completo Dev 1: dataset + modelo + artefactos para el equipo."""
    from data_generator import exportar_dataset

    exportar_dataset(ruta=ruta_dataset, n_por_zona=n_por_zona, seed=seed)
    modelo, metricas, _ = entrenar(seed=seed, modo="hibrido", n_por_zona=n_por_zona)
    guardar(modelo, ruta_modelo)
    metricas["artefactos"] = {
        "modelo": os.path.abspath(ruta_modelo),
        "dataset": os.path.abspath(ruta_dataset),
    }
    return modelo, metricas


if __name__ == "__main__":
    modelo, metricas = entrenar_y_exportar()

    print("=== METRICAS DEL MODELO (datos hibridos, anclados a ICP-MS real) ===")
    for k, v in metricas.items():
        if k not in ("reporte", "matriz_confusion", "importancias", "fuentes_datos", "validacion_sitios_reales"):
            print(f"{k:28s}: {v}")

    print("\nValidacion en 4 sitios reales (verificables):")
    val = metricas["validacion_sitios_reales"]
    print(f"  accuracy sitios reales: {val['accuracy_sitios_reales']}")
    for d in val["detalle"]:
        ok = "OK" if d["acierto"] else "MISS"
        print(f"  [{ok}] {d['sitio']}: real={d['riesgo_real']} pred={d['riesgo_predicho']} "
              f"(As={d['as_mgkg']} mg/kg, pH={d['ph_suelo']})")

    print("\nMatriz de confusion (hold-out 20%):")
    print(np.array(metricas["matriz_confusion"]))

    print("\nImportancia de variables:")
    for f, imp in metricas["importancias"].items():
        print(f"  {f:15s}: {imp:.3f}")

    print("\n" + metricas["reporte"])
    print(f"\nModelo guardado en {metricas['artefactos']['modelo']}")
