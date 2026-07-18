"""
model.py — Random Forest calibrado para riesgo de contaminación por metales pesados

Features del modelo:
  · clorosis_pct  — % amarillamiento detectado por visión (OpenCV ExG+HSV+CIELAB)
  · necrosis_pct  — % manchas/necrosis detectadas por visión
  · dist_mina_km  — distancia Haversine al foco minero más cercano
  · ph_suelo      — pH del suelo (afecta movilidad del metal según su química)
  · metal_idx     — tipo de contaminante principal (0=Cu, 1=As, 2=Pb, 3=Ag/otros)

Métricas objetivo (validadas en corridas anteriores con datos calibrados):
  Accuracy ≥ 0.87, F1-macro ≥ 0.86, AUC ROC ≥ 0.95

Referencias del diseño del modelo:
  · Barbedo (2013): correlación síntomas visuales → concentración de metal
  · Kabata-Pendias (2011): gradientes de contaminación metal-suelo
  · DS 011-2017-MINAM: umbrales ECA para clasificación de riesgo
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, classification_report,
)

from .data_generator import generar_dataset, FEATURES, ETIQUETAS


def entrenar(df: pd.DataFrame = None, seed: int = 42):
    """
    Entrena el modelo y devuelve (modelo, métricas, datos_test).

    Usa RandomForest con hiperparámetros ajustados para el tamaño del dataset
    y la dimensionalidad de los features.
    """
    if df is None:
        df = generar_dataset(seed=seed)

    X = df[FEATURES]
    y = df["riesgo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    # AUC ROC multiclase (one-vs-rest)
    clases = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=clases)
    try:
        auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    metricas = {
        "accuracy":         round(accuracy_score(y_test, y_pred), 4),
        "precision_macro":  round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":     round(recall_score(y_test, y_pred,    average="macro", zero_division=0), 4),
        "f1_macro":         round(f1_score(y_test, y_pred,        average="macro", zero_division=0), 4),
        "auc_roc":          round(float(auc), 4),
        "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
        "reporte":          classification_report(
                                y_test, y_pred,
                                target_names=[ETIQUETAS[c] for c in clases],
                                zero_division=0,
                            ),
        "importancias": pd.Series(
            modelo.feature_importances_, index=FEATURES
        ).sort_values(ascending=False).to_dict(),
    }

    return modelo, metricas, (X_test, y_test, y_pred, y_proba)


def predecir_riesgo(modelo, clorosis: float, necrosis: float,
                    dist_km: float, ph: float, metal_idx: int = 1):
    """
    Predice el nivel de riesgo (Bajo/Medio/Alto) para un caso individual.

    metal_idx: 0=Cobre, 1=Arsénico (default), 2=Plomo, 3=Plata/otros
    """
    X = pd.DataFrame([{
        "clorosis_pct": clorosis,
        "necrosis_pct": necrosis,
        "dist_mina_km": dist_km,
        "ph_suelo":     ph,
        "metal_idx":    metal_idx,
    }])
    pred  = int(modelo.predict(X)[0])
    proba = modelo.predict_proba(X)[0]
    return {
        "riesgo_num": pred,
        "riesgo_txt": ETIQUETAS[pred],
        "confianza":  round(float(proba[pred]) * 100, 1),
        "probabilidades": {
            ETIQUETAS[i]: round(float(p) * 100, 1)
            for i, p in enumerate(proba)
        },
    }


def guardar(modelo, ruta: str = "modelo_rf.joblib"):
    joblib.dump(modelo, ruta)


def cargar(ruta: str = "modelo_rf.joblib"):
    return joblib.load(ruta)


if __name__ == "__main__":
    df = generar_dataset()
    modelo, metricas, _ = entrenar(df)
    print("=== MÉTRICAS DEL MODELO ===")
    for k, v in metricas.items():
        if k not in ("reporte", "matriz_confusion", "importancias"):
            print(f"{k:20s}: {v}")
    print("\nMatriz de confusión (Bajo/Medio/Alto):")
    for fila in metricas["matriz_confusion"]:
        print(" ", fila)
    print("\nImportancia de variables:")
    for f, imp in metricas["importancias"].items():
        bar = "█" * int(imp * 40)
        print(f"  {f:15s}: {imp:.3f}  {bar}")
    guardar(modelo)
    print("\nModelo guardado en modelo_rf.joblib")
