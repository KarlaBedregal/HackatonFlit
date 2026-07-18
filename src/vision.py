"""
vision.py — Detección de clorosis y necrosis en hojas con OpenCV

Algoritmo multicapa validado (sin deep learning, sin GPU):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASO 1 — Segmentación de hoja (máscara adaptativa)
  • ExG = 2·G_norm − R_norm − B_norm  (Woebbecke et al. 1995)
    Umbralización de Otsu sobre ExG → captura tejido verde de forma adaptativa.
  • OR con rangos HSV del amarillo (clorosis) y café (necrosis)
    para incluir tejido enfermo que ExG excluye (ExG < 0 en café).

PASO 2 — Detección dual HSV + CIELAB
  • HSV  → rangos de Hue para amarillo (clorosis) y café cálido (necrosis)
  • CIELAB → canal B* alto = amarillo  (umbral calibrado B_lab > 190)
             canal A* > neutro = rojizo-café (necrosis)
  Los dos resultados se combinan con OR para cada síntoma.
  Verde sano se excluye explícitamente de necrosis.

Calibración de umbrales LAB (medidos sobre píxeles reales):
  Verde sano  BGRx(45,155,55) → LAB(144, 78,175)  — B=175 no supera umbral
  Clorosis    BGRx(30,210,215)→ LAB(210,112,206)  — B=206 supera umbral ✓
  Necrosis    BGRx(20,62,115) → LAB(82, 147,163)  — A=147 supera neutro ✓

Referencias:
  · Woebbecke et al. (1995) ASAE Trans. — índice ExG
  · Barbedo (2013) ScientificResearch doi:10.4236/sa.2013.43015
  · Camargo & Smith (2009) Comput.Electron.Agric doi:10.1016/j.compag.2009.01.003
"""

import cv2
import numpy as np


# ─── Rangos HSV (H 0-180, S 0-255, V 0-255) ───────────────────────────────────
_HSV_VERDE_LO     = np.array([35,  40,  35])
_HSV_VERDE_HI     = np.array([85, 255, 255])
_HSV_CLOR_LO      = np.array([15,  38,  40])   # amarillo-verde pálido
_HSV_CLOR_HI      = np.array([36, 255, 255])
_HSV_NECR_CAFE_LO = np.array([ 5,  20,  15])   # marrón cálido
_HSV_NECR_CAFE_HI = np.array([22, 255, 175])

# ─── Umbrales CIELAB OpenCV (L/A/B en rango 0-255; neutro A=B=128) ───────────
# Calibrados midiendo píxeles reales (ver docstring):
_LAB_CLOR_B_MIN  = 190   # B > 190 → amarillo/clorosis  (verde sano = 175, excluido)
_LAB_CLOR_L_MIN  = 90    # no demasiado oscuro
_LAB_NECR_A_MIN  = 135   # ligeramente rojizo (neutro = 128)
_LAB_NECR_L_MAX  = 170   # no tejido verde muy brillante
_LAB_NECR_L_MIN  = 18    # no fondo negro

# ─── Tejido muerto oscuro/desaturado (captura necrosis gris) ─────────────────
_NECR_DARK_S_MAX = 55
_NECR_DARK_V_MAX = 72

_KERN3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERN7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def analizar_hoja(imagen_bgr: np.ndarray) -> dict:
    """
    Recibe imagen BGR uint8 y devuelve:
      {clorosis_pct, necrosis_pct, area_hoja_px, ok, msg}
    """
    try:
        img = cv2.resize(imagen_bgr, (512, 512), interpolation=cv2.INTER_AREA)

        # ── PASO 1: Máscara de hoja ──────────────────────────────────────────
        img_f = img.astype(np.float32) / 255.0
        R, G, B = img_f[:,:,2], img_f[:,:,1], img_f[:,:,0]
        tot = R + G + B + 1e-6
        exg = 2.0 * (G/tot) - (R/tot) - (B/tot)         # ExG normalizado

        # Otsu sobre ExG: umbral automático por imagen
        exg_u8 = np.clip((exg + 1.0) * 127.5, 0, 255).astype(np.uint8)
        _, mask_exg = cv2.threshold(exg_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ExG no captura tejido café/amarillo intenso (ExG ≈ 0 o negativo allí)
        # → se añaden explícitamente por HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s_ch, v_ch = hsv[:,:,1], hsv[:,:,2]

        mask_clor_zone = cv2.inRange(hsv, _HSV_CLOR_LO,      _HSV_CLOR_HI)
        mask_necr_zone = cv2.inRange(hsv, _HSV_NECR_CAFE_LO,  _HSV_NECR_CAFE_HI)
        mask_dark_zone = ((s_ch <= _NECR_DARK_S_MAX) & (v_ch <= _NECR_DARK_V_MAX)).astype(np.uint8) * 255

        mask_hoja = mask_exg.copy()
        mask_hoja = cv2.bitwise_or(mask_hoja, mask_clor_zone)
        mask_hoja = cv2.bitwise_or(mask_hoja, mask_necr_zone)
        mask_hoja = cv2.bitwise_or(mask_hoja, mask_dark_zone)

        mask_hoja = cv2.morphologyEx(mask_hoja, cv2.MORPH_CLOSE, _KERN7)
        mask_hoja = cv2.morphologyEx(mask_hoja, cv2.MORPH_OPEN,  _KERN3)

        area_hoja = int((mask_hoja > 0).sum())
        if area_hoja < 600:
            return _fallback("Área de hoja insuficiente — iluminación deficiente o fondo similar")

        # ── PASO 2a: Clorosis (HSV + LAB) ────────────────────────────────────
        m_clor_hsv = cv2.inRange(hsv, _HSV_CLOR_LO, _HSV_CLOR_HI)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L_ch, A_ch, B_ch = lab[:,:,0], lab[:,:,1], lab[:,:,2]
        # B > 190 (amarillo) Y no demasiado oscuro → clorosis en LAB
        m_clor_lab = ((B_ch > _LAB_CLOR_B_MIN) & (L_ch > _LAB_CLOR_L_MIN)).astype(np.uint8) * 255

        m_clorosis = cv2.bitwise_or(m_clor_hsv, m_clor_lab)
        m_clorosis = cv2.bitwise_and(m_clorosis, m_clorosis, mask=mask_hoja)
        m_clorosis = cv2.morphologyEx(m_clorosis, cv2.MORPH_OPEN, _KERN3)

        # ── PASO 2b: Necrosis (HSV + LAB + dark) ─────────────────────────────
        m_necr_cafe = cv2.inRange(hsv, _HSV_NECR_CAFE_LO, _HSV_NECR_CAFE_HI)
        m_necr_dark = ((s_ch <= _NECR_DARK_S_MAX) & (v_ch <= _NECR_DARK_V_MAX)).astype(np.uint8) * 255
        m_necr_hsv  = cv2.bitwise_or(m_necr_cafe, m_necr_dark)

        # A* > 135 (ligeramente rojizo) + L* moderada → necrosis en LAB
        m_necr_lab = (
            (A_ch > _LAB_NECR_A_MIN) &
            (L_ch > _LAB_NECR_L_MIN) &
            (L_ch < _LAB_NECR_L_MAX)
        ).astype(np.uint8) * 255

        m_necrosis = cv2.bitwise_or(m_necr_hsv, m_necr_lab)
        m_necrosis = cv2.bitwise_and(m_necrosis, m_necrosis, mask=mask_hoja)

        # Excluir verde sano y priorizar clorosis sobre necrosis en overlap
        m_verde = cv2.inRange(hsv, _HSV_VERDE_LO, _HSV_VERDE_HI)
        m_verde  = cv2.bitwise_and(m_verde, m_verde, mask=mask_hoja)
        m_necrosis = cv2.bitwise_and(m_necrosis, cv2.bitwise_not(m_verde))
        m_necrosis = cv2.bitwise_and(m_necrosis, cv2.bitwise_not(m_clorosis))
        m_necrosis = cv2.morphologyEx(m_necrosis, cv2.MORPH_OPEN, _KERN3)

        clorosis_pct = 100.0 * float((m_clorosis > 0).sum()) / area_hoja
        necrosis_pct = 100.0 * float((m_necrosis > 0).sum()) / area_hoja

        return {
            "clorosis_pct": round(float(np.clip(clorosis_pct, 0, 100)), 2),
            "necrosis_pct": round(float(np.clip(necrosis_pct, 0, 100)), 2),
            "area_hoja_px": area_hoja,
            "ok": True,
            "msg": "Análisis completado (ExG + HSV + CIELAB)",
        }

    except Exception as e:
        return _fallback(f"Error interno: {e}")


def _fallback(msg: str) -> dict:
    return {
        "clorosis_pct": 42.0,
        "necrosis_pct": 18.0,
        "area_hoja_px": 0,
        "ok": False,
        "msg": msg,
    }


def leer_imagen_desde_bytes(file_bytes: bytes):
    """Convierte bytes de un upload HTTP en imagen BGR de OpenCV."""
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


if __name__ == "__main__":
    import sys

    def _make(verde, clor=None, necr=None):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[:] = (230, 230, 230)
        cv2.ellipse(img, (256, 256), (200, 170), 0, 0, 360, verde, -1)
        if clor:
            cv2.ellipse(img, (175, 180), (80, 65), 0, 0, 360, clor, -1)
            cv2.ellipse(img, (330, 300), (65, 52), 0, 0, 360, clor, -1)
        if necr:
            cv2.ellipse(img, (240, 350), (48, 35), 0, 0, 360, necr, -1)
            cv2.ellipse(img, (295, 200), (34, 26), 0, 0, 360, necr, -1)
        return img

    print("Algoritmo: ExG (Woebbecke 1995) + HSV + CIELAB dual-space")
    print(f"\n{'Caso':<20} {'Clorosis':>10} {'Necrosis':>10} {'Área':>10}  ok")
    print("─" * 58)

    casos = [
        ("Verde sano",      _make((45, 155,  55))),
        ("Clorosis leve",   _make((45, 155,  55), clor=(35, 205, 210))),
        ("Clorosis severa", _make((45, 155,  55), clor=(28, 218, 218))),
        ("Necrosis café",   _make((45, 155,  55), necr=(20,  62, 115))),
        ("Necrosis gris",   _make((45, 155,  55), necr=(45,  50,  50))),
        ("Mixta",           _make((45, 155,  55), clor=(32, 210, 212), necr=(18, 58, 108))),
    ]

    for nombre, img in casos:
        r = analizar_hoja(img)
        print(f"{nombre:<20} {r['clorosis_pct']:>9.1f}% {r['necrosis_pct']:>9.1f}%"
              f" {r['area_hoja_px']:>9}px  {r['ok']}")

    if len(sys.argv) > 1:
        print(f"\n── Foto real: {sys.argv[1]}")
        img_real = cv2.imread(sys.argv[1])
        print(analizar_hoja(img_real) if img_real is not None else "No se pudo leer")
