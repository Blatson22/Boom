import os

import cv2
import mediapipe as mp
import numpy as np

from detectar_mano import (
    crear_detector,
    dibujar_manos,
    iniciar_captura,
    normalizar_landmarks,
)

RUTA_DATASET = "dataset.npz"
ETIQUETAS = {
    ord("1"): "puno",
    ord("2"): "pinza",
    ord("3"): "mano_abierta",
}


def detectar_frame(captura, detector):
    ret, frame = captura.read()
    if not ret:
        return None, None, None
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = detector.detect_for_video(imagen, int(captura.get(cv2.CAP_PROP_POS_MSEC)))
    landmarks = resultado.hand_landmarks[0] if resultado.hand_landmarks else None
    return frame, landmarks, resultado


def main():
    captura = iniciar_captura()
    if captura is None:
        print("No se pudo abrir la camara")
        return

    detector = crear_detector()
    muestras = []
    contador = {"puno": 0, "pinza": 0, "mano_abierta": 0}
    ventana = "Captura de dataset"

    if os.path.exists(RUTA_DATASET):
        datos = np.load(RUTA_DATASET, allow_pickle=True)
        if "X" in datos and "y" in datos:
            for vector, etiqueta in zip(np.asarray(datos["X"]), np.asarray(datos["y"])):
                etiqueta = str(etiqueta)
                muestras.append((list(vector), etiqueta))
                contador[etiqueta] = contador.get(etiqueta, 0) + 1

    cv2.namedWindow(ventana)

    while True:
        frame, landmarks, resultado = detectar_frame(captura, detector)
        if frame is None:
            break

        frame = dibujar_manos(frame, resultado)
        texto = (
            f"1:puno({contador['puno']})  2:pinza({contador['pinza']})  "
            f"3:mano_abierta({contador['mano_abierta']})  n:no guardar  q:salir"
        )
        cv2.putText(frame, texto, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(ventana, frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27 or tecla == ord("q"):
            break
        if tecla == ord("n"):
            continue
        if tecla not in ETIQUETAS:
            continue
        if landmarks is None:
            print("No hay mano visible, no se guardo la muestra")
            continue
        vector = normalizar_landmarks(landmarks)
        if vector is None:
            print("Escala de mano invalida, no se guardo la muestra")
            continue
        etiqueta = ETIQUETAS[tecla]
        muestras.append((vector, etiqueta))
        contador[etiqueta] += 1

    captura.release()
    cv2.destroyAllWindows()

    if not muestras:
        print("No se guardo ninguna muestra")
        return

    X = np.array([muestra[0] for muestra in muestras], dtype=np.float32)
    y = np.array([muestra[1] for muestra in muestras])
    np.savez(RUTA_DATASET, X=X, y=y)

    print("Dataset guardado en", RUTA_DATASET)
    total = 0
    for etiqueta in ETIQUETAS.values():
        print(etiqueta + ":", contador[etiqueta])
        total += contador[etiqueta]
    print("Total:", total)


if __name__ == "__main__":
    main()