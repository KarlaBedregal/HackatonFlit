"""
model.py
--------
Entrenamiento y evaluacion del Random Forest para predecir riesgo de
contaminacion (Bajo/Medio/Alto).

Incluye TODAS las metricas pedidas:
  - Matriz de Confusion
  - Accuracy
  - Precision (macro)
  - Recall / Sensibilidad (macro)
  - F1-Score (macro)
  - AUC ROC (multiclase, one-vs-rest)

Guarda el modelo entrenado con joblib para que la app de Streamlit lo cargue.
"""

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

from data_generator import generar_dataset, FEATURES, ETIQUETAS


def entrenar(df=None, seed=42):
    """Entrena el RF y devuelve (modelo, metricas, datos_test)."""
    if df is None:
        df = generar_dataset(seed=seed)

    X = df[FEATURES]
    y = df["riesgo"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

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

    # --- AUC ROC multiclase (one-vs-rest) ---
    clases = sorted(y.unique())
    y_test_bin = label_binarize(y_test, classes=clases)
    try:
        auc = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")

    metricas = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "auc_roc": round(auc, 4),
        "matriz_confusion": confusion_matrix(y_test, y_pred).tolist(),
        "reporte": classification_report(
            y_test, y_pred,
            target_names=[ETIQUETAS[c] for c in clases],
            zero_division=0,
        ),
        "importancias": pd.Series(
            modelo.feature_importances_, index=FEATURES
        ).sort_values(ascending=False).to_dict(),
    }

    return modelo, metricas, (X_test, y_test, y_pred, y_proba)


def predecir_riesgo(modelo, clorosis, necrosis, dist_km, ph):
    """Predice el nivel de riesgo para un caso individual."""
    X = pd.DataFrame([{
        "clorosis_pct": clorosis,
        "necrosis_pct": necrosis,
        "dist_mina_km": dist_km,
        "ph_suelo": ph,
    }])
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


if __name__ == "__main__":
    modelo, metricas, _ = entrenar()
    print("=== METRICAS DEL MODELO ===")
    for k, v in metricas.items():
        if k not in ("reporte", "matriz_confusion", "importancias"):
            print(f"{k:20s}: {v}")
    print("\nMatriz de confusion:")
    print(np.array(metricas["matriz_confusion"]))
    print("\nImportancia de variables:")
    for f, imp in metricas["importancias"].items():
        print(f"  {f:15s}: {imp:.3f}")
    print("\n" + metricas["reporte"])
    guardar(modelo)
    print("Modelo guardado en modelo_rf.joblib")
