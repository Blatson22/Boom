import math
import os
import random
import threading
import time
from collections import Counter, deque

import numpy as np
import pyray as rl

from contratos import EstadoMano
from detectar_mano import (
    UMBRAL_EXTENDIDO,
    UMBRAL_PELLIZCO,
    crear_detector,
    diagnostico_gesto,
    iniciar_captura,
    landmarks_a_estado,
    normalizar_landmarks,
    obtener_landmarks,
)

K_RESORTE = 40.0
AMORTIGUACION = 8.0

TAMANO_BUFFER_GESTOS = 5
UMBRAL_MAYORIA = 3
FPS_DETECCION = 30

RUTA_CORRECCIONES = "correcciones.npz"

N_CUBOS = 8
RESTITUCION = 0.5
RADIO_INFLUENCIA = 2.0
K_RESORTE_SUAVE = 6.0
UMBRAL_IMPACTO = 4.0
PARTICULAS_MIN = 8
PARTICULAS_MAX = 16
MAX_PARTICULAS = 300
RADIO_PARTICULA = 0.08
VELOCIDAD_COLOR = 5.0
COLOR_LENTO = rl.SKYBLUE
COLOR_RAPIDO = rl.RED
COLOR_PARTICULA = rl.ORANGE


def gesto_mayoritario(historial, gesto_previo, umbral):
    conteo = Counter(historial)
    gesto, veces = conteo.most_common(1)[0]
    if veces >= umbral:
        return gesto
    return gesto_previo


def presencia_mayoritaria(historial, umbral):
    return sum(historial) >= umbral


class Cubo:
    def __init__(self, posicion, tamano, color):
        self.posicion = posicion
        self.velocidad = rl.Vector3(0.0, 0.0, 0.0)
        self.tamano = tamano
        self.tamano_base = tamano
        self.color = color


class Particula:
    def __init__(self, posicion, velocidad, vida):
        self.posicion = posicion
        self.velocidad = velocidad
        self.vida = vida
        self.vida_inicial = vida


def distancia_v3(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def generar_particulas(posicion, particulas):
    if len(particulas) >= MAX_PARTICULAS:
        return
    cantidad = random.randint(PARTICULAS_MIN, PARTICULAS_MAX)
    for _ in range(cantidad):
        if len(particulas) >= MAX_PARTICULAS:
            break
        origen = rl.Vector3(posicion.x, RADIO_PARTICULA, posicion.z)
        velocidad = rl.Vector3(
            random.uniform(-2.5, 2.5),
            random.uniform(1.0, 4.5),
            random.uniform(-2.5, 2.5),
        )
        particula = Particula(origen, velocidad, random.uniform(0.4, 0.9))
        particulas.append(particula)


class HandTracker:
    def __init__(self):
        self.captura = iniciar_captura()
        self.detector = crear_detector(max_manos=1)
        self._lock = threading.Lock()
        self._estado = None
        self._diag = None
        self._vector = None
        self._activo = self.captura is not None
        self._hilo = None
        if self._activo:
            self._hilo = threading.Thread(target=self._loop, daemon=True)
            self._hilo.start()

    def _loop(self):
        periodo = 1.0 / FPS_DETECCION
        while self._activo:
            inicio = time.monotonic()
            landmarks = obtener_landmarks(self.captura, self.detector)
            nuevo = landmarks_a_estado(landmarks) if landmarks is not None else None
            diag = diagnostico_gesto(landmarks) if landmarks is not None else None
            vector = normalizar_landmarks(landmarks) if landmarks is not None else None
            with self._lock:
                self._estado = nuevo
                self._diag = diag
                self._vector = vector
            restante = periodo - (time.monotonic() - inicio)
            if restante > 0:
                time.sleep(restante)

    def estado_actual(self) -> EstadoMano | None:
        with self._lock:
            return self._estado

    def diagnostico_actual(self):
        with self._lock:
            return self._diag

    def vector_actual(self):
        with self._lock:
            return self._vector

    def cerrar(self):
        self._activo = False
        if self._hilo is not None:
            self._hilo.join(timeout=1.0)
        if self.captura is not None:
            self.captura.release()
        self.detector.close()


def main():
    rl.init_window(800, 600, "Simulacion 3D")
    rl.set_target_fps(60)

    camara = rl.Camera3D(
        rl.Vector3(0.0, 6.0, 12.0),
        rl.Vector3(0.0, 2.0, 0.0),
        rl.Vector3(0.0, 1.0, 0.0),
        45.0,
        rl.CAMERA_PERSPECTIVE,
    )

    tracker = HandTracker()

    correcciones = []

    gravedad = rl.Vector3(0.0, -9.8, 0.0)
    dt = 1.0 / 60.0

    cubos = []
    for _ in range(N_CUBOS):
        tamano = random.uniform(0.6, 1.2)
        posicion = rl.Vector3(
            random.uniform(-4.0, 4.0),
            random.uniform(1.0, 5.0) + 0.5 * tamano,
            random.uniform(-4.0, 4.0),
        )
        cubos.append(Cubo(posicion, tamano, COLOR_LENTO))

    particulas = []

    historial_gestos = deque(maxlen=TAMANO_BUFFER_GESTOS)
    historial_presencia = deque(maxlen=TAMANO_BUFFER_GESTOS)
    gesto_efectivo = "indefinido"
    objetivo_estable = None
    hay_mano = False

    while not rl.window_should_close():
        estado = tracker.estado_actual()

        etiqueta_correccion = None
        if rl.is_key_pressed(rl.KEY_ONE):
            etiqueta_correccion = "puno"
        elif rl.is_key_pressed(rl.KEY_TWO):
            etiqueta_correccion = "pinza"
        elif rl.is_key_pressed(rl.KEY_THREE):
            etiqueta_correccion = "mano_abierta"
        if etiqueta_correccion is not None:
            vector = tracker.vector_actual()
            if vector is not None:
                correcciones.append((vector, etiqueta_correccion))

        gesto_crudo = estado.gesto if estado is not None else "indefinido"
        historial_gestos.append(gesto_crudo)
        historial_presencia.append(estado is not None)
        gesto_efectivo = gesto_mayoritario(historial_gestos, gesto_efectivo, UMBRAL_MAYORIA)
        hay_mano = presencia_mayoritaria(historial_presencia, UMBRAL_MAYORIA)

        if estado is not None:
            objetivo_estable = estado.posicion

        objetivo = objetivo_estable if (hay_mano and objetivo_estable is not None) else None
        cubo_cercano = None
        if objetivo is not None and cubos:
            cubo_cercano = min(cubos, key=lambda c: distancia_v3(c.posicion, objetivo))

        if hay_mano and cubo_cercano is not None:
            if gesto_efectivo == "pinza":
                cubo_cercano.tamano = max(0.3, min(2.5, cubo_cercano.tamano + 0.5 * dt))
            elif gesto_efectivo == "mano_abierta":
                cubo_cercano.tamano += (cubo_cercano.tamano_base - cubo_cercano.tamano) * 2.0 * dt

        for cubo in cubos:
            agarrado = hay_mano and objetivo is not None and cubo is cubo_cercano and gesto_efectivo == "puno"
            if agarrado:
                cubo.posicion = rl.Vector3(objetivo.x, objetivo.y, objetivo.z)
                cubo.velocidad = rl.Vector3(0.0, 0.0, 0.0)
                continue

            ax = 0.0
            ay = gravedad.y
            az = 0.0

            if hay_mano and objetivo is not None:
                dx = objetivo.x - cubo.posicion.x
                dy = objetivo.y - cubo.posicion.y
                dz = objetivo.z - cubo.posicion.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                rigidez = 0.0
                if cubo is cubo_cercano:
                    rigidez = K_RESORTE
                elif dist < RADIO_INFLUENCIA:
                    rigidez = K_RESORTE_SUAVE
                if rigidez > 0.0 and dist > 0.0001:
                    ax += dx * rigidez - cubo.velocidad.x * AMORTIGUACION
                    ay += dy * rigidez - cubo.velocidad.y * AMORTIGUACION
                    az += dz * rigidez - cubo.velocidad.z * AMORTIGUACION

            cubo.velocidad.x += ax * dt
            cubo.velocidad.y += ay * dt
            cubo.velocidad.z += az * dt

            cubo.posicion.x += cubo.velocidad.x * dt
            cubo.posicion.y += cubo.velocidad.y * dt
            cubo.posicion.z += cubo.velocidad.z * dt

            suelo = 0.5 * cubo.tamano
            if cubo.posicion.y < suelo:
                impacto = -cubo.velocidad.y
                cubo.posicion.y = suelo
                cubo.velocidad.y = -cubo.velocidad.y * RESTITUCION
                if impacto > UMBRAL_IMPACTO:
                    generar_particulas(cubo.posicion, particulas)

        for particula in particulas:
            particula.velocidad.y += gravedad.y * dt
            particula.posicion.x += particula.velocidad.x * dt
            particula.posicion.y += particula.velocidad.y * dt
            particula.posicion.z += particula.velocidad.z * dt
            if particula.posicion.y < RADIO_PARTICULA:
                particula.posicion.y = RADIO_PARTICULA
                particula.velocidad.y = -particula.velocidad.y * 0.4
            particula.vida -= dt
        particulas = [p for p in particulas if p.vida > 0.0]

        rl.update_camera(camara, rl.CAMERA_FREE)

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
        rl.begin_mode_3d(camara)
        rl.draw_grid(20, 1.0)
        for cubo in cubos:
            rapidez = math.sqrt(
                cubo.velocidad.x * cubo.velocidad.x
                + cubo.velocidad.y * cubo.velocidad.y
                + cubo.velocidad.z * cubo.velocidad.z
            )
            mezcla = max(0.0, min(1.0, rapidez / VELOCIDAD_COLOR))
            color = rl.color_lerp(COLOR_LENTO, COLOR_RAPIDO, mezcla)
            cubo.color = color
            rl.draw_cube(cubo.posicion, cubo.tamano, cubo.tamano, cubo.tamano, color)
            rl.draw_cube_wires(cubo.posicion, cubo.tamano, cubo.tamano, cubo.tamano, rl.fade(color, 0.5))
        for particula in particulas:
            alfa = max(0.0, min(1.0, particula.vida / particula.vida_inicial))
            rl.draw_sphere(particula.posicion, RADIO_PARTICULA, rl.fade(COLOR_PARTICULA, alfa))
        if hay_mano and objetivo is not None:
            rl.draw_sphere(objetivo, 0.2, rl.BLUE)
        rl.end_mode_3d()
        rl.draw_fps(10, 10)
        rl.draw_text(f"Gesto: {gesto_efectivo}", 10, 40, 20, rl.DARKGRAY)
        rl.draw_text(f"Crudo: {gesto_crudo}", 10, 70, 20, rl.DARKGRAY)
        rl.draw_text(f"Cubos: {len(cubos)}  Particulas: {len(particulas)}", 10, 160, 20, rl.DARKGRAY)
        rl.draw_text(f"Correcciones: {len(correcciones)}", 10, 190, 20, rl.DARKGRAY)
        diag = tracker.diagnostico_actual()
        if diag:
            rl.draw_text(f"pulgar-indice {diag['pulgar_indice']}  (pinza < {UMBRAL_PELLIZCO})", 10, 100, 20, rl.DARKGRAY)
            rl.draw_text(
                f"medio {diag['medio']}  anular {diag['anular']}  menique {diag['menique']}  (ext > {UMBRAL_EXTENDIDO})",
                10, 130, 20, rl.DARKGRAY,
            )
        rl.end_drawing()

    tracker.cerrar()

    if correcciones:
        X_nuevas = np.array([vector for vector, _ in correcciones], dtype=np.float32)
        y_nuevas = np.array([etiqueta for _, etiqueta in correcciones])
        if os.path.exists(RUTA_CORRECCIONES):
            datos = np.load(RUTA_CORRECCIONES, allow_pickle=True)
            X_prev = np.asarray(datos["X"], dtype=np.float32)
            y_prev = np.asarray(datos["y"])
            X_total = np.vstack([X_prev, X_nuevas])
            y_total = np.concatenate([y_prev, y_nuevas])
        else:
            X_total = X_nuevas
            y_total = y_nuevas
        np.savez(RUTA_CORRECCIONES, X=X_total, y=y_total)

    rl.close_window()


if __name__ == "__main__":
    main()