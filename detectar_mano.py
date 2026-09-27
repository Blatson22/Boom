import math
import os

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pyray as rl

from contratos import EstadoMano

UMBRAL_EXTENDIDO = 0.15
UMBRAL_PELLIZCO = 0.35
UMBRAL_CONFIANZA = 0.6

CONEXIONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gestos_mlp.joblib")


def _cargar_modelo(ruta=RUTA_MODELO):
    try:
        modelo = joblib.load(ruta)
    except Exception:
        return None
    if not isinstance(modelo, dict) or "pipeline" not in modelo or "clases" not in modelo:
        return None
    return modelo


modelo_gestos = _cargar_modelo()


def iniciar_captura(camara=0):
    captura = cv2.VideoCapture(camara)
    if not captura.isOpened():
        return None
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return captura


def crear_detector(max_manos=1, modelo="hand_landmarker.task"):
    base_options = python.BaseOptions(model_asset_path=modelo)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=max_manos,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def obtener_landmarks(captura, detector):
    ret, frame = captura.read()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect_for_video(imagen, int(captura.get(cv2.CAP_PROP_POS_MSEC)))
    if not resultado.hand_landmarks:
        return None
    return resultado.hand_landmarks[0]

def centro_palma(landmarks):
    x = sum(p.x for p in landmarks) / len(landmarks)
    y = sum(p.y for p in landmarks) / len(landmarks)
    return x, y


def _distancia(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _tamano_mano(landmarks):
    return _distancia(landmarks[0], landmarks[9])


def _ratio_extension(landmarks, punta_idx, base_idx, escala):
    d_punta = _distancia(landmarks[0], landmarks[punta_idx])
    d_base = _distancia(landmarks[0], landmarks[base_idx])
    return (d_punta - d_base) / escala


def _dedo_extendido(landmarks, punta_idx, base_idx, escala):
    return _ratio_extension(landmarks, punta_idx, base_idx, escala) > UMBRAL_EXTENDIDO


def _clasificar_gesto_heuristica(landmarks):
    escala = _tamano_mano(landmarks)
    if escala == 0:
        return "indefinido"

    pulgar_indice = _distancia(landmarks[4], landmarks[8]) / escala
    medio_ext = _dedo_extendido(landmarks, 12, 9, escala)
    anular_ext = _dedo_extendido(landmarks, 16, 13, escala)
    menique_ext = _dedo_extendido(landmarks, 20, 17, escala)

    pellizcando = pulgar_indice < UMBRAL_PELLIZCO

    if pellizcando and (medio_ext or anular_ext or menique_ext):
        return "pinza"
    elif pellizcando and not (medio_ext or anular_ext or menique_ext):
        return "puno"
    elif not pellizcando and medio_ext and anular_ext and menique_ext:
        return "mano_abierta"
    else:
        return "indefinido"


def _clasificar_gesto_mlp(vector):
    pipeline = modelo_gestos["pipeline"]
    clases = modelo_gestos["clases"]
    prediccion = pipeline.predict([vector])[0]
    if isinstance(prediccion, str):
        gesto = prediccion
    else:
        gesto = clases[int(prediccion)]
    try:
        confianza = float(max(pipeline.predict_proba([vector])[0]))
    except Exception:
        confianza = 1.0
    if confianza < UMBRAL_CONFIANZA:
        return "indefinido"
    return gesto


def clasificar_gesto(landmarks):
    if landmarks is None:
        return "indefinido"
    if modelo_gestos is not None:
        vector = normalizar_landmarks(landmarks)
        if vector is None:
            return "indefinido"
        return _clasificar_gesto_mlp(vector)
    return _clasificar_gesto_heuristica(landmarks)


def diagnostico_gesto(landmarks):
    escala = _tamano_mano(landmarks)
    if escala == 0:
        return {}
    return {
        "pulgar_indice": round(_distancia(landmarks[4], landmarks[8]) / escala, 3),
        "medio": round(_ratio_extension(landmarks, 12, 9, escala), 3),
        "anular": round(_ratio_extension(landmarks, 16, 13, escala), 3),
        "menique": round(_ratio_extension(landmarks, 20, 17, escala), 3),
    }


def normalizar_landmarks(landmarks):
    escala = _tamano_mano(landmarks)
    if escala == 0:
        return None
    ref_x = landmarks[0].x
    ref_y = landmarks[0].y
    vector = []
    for p in landmarks:
        vector.append((p.x - ref_x) / escala)
        vector.append((p.y - ref_y) / escala)
        vector.append(p.z / escala)
    return vector


def landmarks_a_estado(landmarks):
    x, y = centro_palma(landmarks)
    posicion = rl.Vector3((x - 0.5) * 8.0, (1.0 - y) * 5.0 + 0.5, 0.0)
    gesto = clasificar_gesto(landmarks)
    return EstadoMano(posicion=posicion, gesto=gesto)


def dibujar_manos(frame, resultado):
    alto, ancho, _ = frame.shape
    for mano in resultado.hand_landmarks:
        puntos = [(int(p.x * ancho), int(p.y * alto)) for p in mano]
        for a, b in CONEXIONES:
            cv2.line(frame, puntos[a], puntos[b], (0, 255, 0), 2)
        for punto in puntos:
            cv2.circle(frame, punto, 4, (0, 0, 255), -1)
    return frame


def main():
    captura = iniciar_captura()
    if captura is None:
        print("No se pudo abrir la camara")
        return
    detector = crear_detector()
    while True:
        ret, frame = captura.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        resultado = detector.detect_for_video(imagen, int(captura.get(cv2.CAP_PROP_POS_MSEC)))
        frame = dibujar_manos(frame, resultado)
        cv2.imshow("Deteccion de manos", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()