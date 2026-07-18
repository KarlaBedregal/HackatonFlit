"""
vision.py
---------
Extraccion de features visuales de una foto de hoja usando OpenCV clasico
(SIN deep learning -> rapido, sin GPU, sin entrenar nada).

- clorosis_pct : % de pixeles amarillos/palidos (perdida de verdor)
- necrosis_pct : % de pixeles marrones/oscuros (manchas / tejido muerto)

Trabaja en espacio HSV que es mucho mas robusto para segmentar color
que RGB. Incluye fallback: si la imagen no se puede leer, devuelve mock data.
"""

import cv2
import numpy as np


def analizar_hoja(imagen_bgr):
    """
    Recibe una imagen BGR (numpy array de OpenCV) y devuelve un dict con
    los porcentajes de clorosis y necrosis sobre el area de la hoja.
    """
    try:
        img = cv2.resize(imagen_bgr, (400, 400))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1) Mascara de la hoja (todo lo que NO es fondo blanco/gris muy claro)
        #    Saturacion > 25 y valor razonable -> hay color de planta
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        mask_hoja = ((s > 25) & (v > 30) & (v < 250)).astype(np.uint8)
        area_hoja = int(mask_hoja.sum())
        if area_hoja < 500:  # casi no hay hoja detectada
            return _mock_resultado("Poca hoja detectada, usando estimacion")

        # 2) Verde sano  (Hue ~ 35-85 en OpenCV)
        verde = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

        # 3) Clorosis / amarillo-palido (Hue ~ 20-35, alta luminosidad)
        amarillo = cv2.inRange(hsv, (20, 40, 60), (35, 255, 255))

        # 4) Necrosis / marron-oscuro (Hue bajo + baja saturacion/valor)
        marron = cv2.inRange(hsv, (5, 30, 20), (20, 200, 160))

        # Restringimos todo al area de la hoja
        verde = cv2.bitwise_and(verde, verde, mask=mask_hoja)
        amarillo = cv2.bitwise_and(amarillo, amarillo, mask=mask_hoja)
        marron = cv2.bitwise_and(marron, marron, mask=mask_hoja)

        clorosis_pct = 100.0 * (amarillo > 0).sum() / area_hoja
        necrosis_pct = 100.0 * (marron > 0).sum() / area_hoja

        return {
            "clorosis_pct": round(float(np.clip(clorosis_pct, 0, 100)), 2),
            "necrosis_pct": round(float(np.clip(necrosis_pct, 0, 100)), 2),
            "area_hoja_px": area_hoja,
            "ok": True,
            "msg": "Analisis de vision correcto",
        }

    except Exception as e:  # fallback total para no romper el demo en vivo
        return _mock_resultado(f"Fallback vision: {e}")


def _mock_resultado(msg):
    """Datos de respaldo si la foto falla en vivo."""
    return {
        "clorosis_pct": 42.0,
        "necrosis_pct": 18.0,
        "area_hoja_px": 0,
        "ok": False,
        "msg": msg,
    }


def leer_imagen_desde_bytes(file_bytes):
    """Convierte los bytes de un st.file_uploader en imagen BGR de OpenCV."""
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except Exception:
        return None


if __name__ == "__main__":
    # Prueba rapida con una imagen sintetica amarillenta
    demo = np.zeros((400, 400, 3), dtype=np.uint8)
    demo[:, :] = (40, 200, 220)  # BGR amarillento
    print(analizar_hoja(demo))
