import math
import os
import random
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import pyray as rl

from contratos import EstadoMano
from detectar_mano import (
    crear_detector,
    diagnostico_gesto,
    iniciar_captura,
    landmarks_a_estado,
    normalizar_landmarks,
    obtener_landmarks,
)

TAMANO_BUFFER_GESTOS = 5
UMBRAL_MAYORIA = 3
FPS_DETECCION = 30

RUTA_CORRECCIONES = "correcciones.npz"

AREA_X_MIN = -4.0
AREA_X_MAX = 4.0
AREA_Y_MIN = 0.8
AREA_Y_MAX = 3.2
AREA_Z_MIN = -4.0
AREA_Z_MAX = 4.0
VELOCIDAD_CUBO_MIN = 1.0
VELOCIDAD_CUBO_MAX = 2.2
VELOCIDAD_RAPIDO_MIN = 1.8
VELOCIDAD_RAPIDO_MAX = 2.8
VELOCIDAD_TANQUE_MIN = 0.6
VELOCIDAD_TANQUE_MAX = 1.0
UMBRAL_WAYPOINT = 0.3

COLOR_NORMAL = rl.SKYBLUE
COLOR_RAPIDO = rl.YELLOW
COLOR_TANQUE = rl.BLUE
MAX_CUBOS = 12
MAX_FRAGMENTOS = 300
RADIO_EXPLOSION = 3.0
RADIO_CADENA = 2.5
SHAKE_DURACION = 0.35
SHAKE_MAGNITUD = 0.4
VOLUMEN_EXPLOSION = 0.8
VOLUMEN_POP = 0.5
VIDA_FRAGMENTO = 1.2
COOLDOWN_ACCION = 0.35
FLASH_DURACION = 0.25

TAMANO_RAPIDO = 0.5
TAMANO_TANQUE = 1.4
VIDA_TANQUE = 2
TOPE_ESCALA_VELOCIDAD = 2.2

PUNTOS_NORMAL = 10
PUNTOS_RAPIDO = 25
PUNTOS_TANQUE = 50

TIEMPO_OLEADA_BASE = 12.0
TIEMPO_EXTRA_POR_OLEADA = 1.5
TIEMPO_TRANSICION = 1.0
VENTANA_COMBO = 1.5
COMBO_MAX = 5

ESTADO_INICIO = "inicio"
ESTADO_JUGANDO = "jugando"
ESTADO_GAME_OVER = "game_over"


def gesto_mayoritario(historial, gesto_previo, umbral):
    conteo = Counter(historial)
    gesto, veces = conteo.most_common(1)[0]
    if veces >= umbral:
        return gesto
    return gesto_previo


def presencia_mayoritaria(historial, umbral):
    return sum(historial) >= umbral


def objetivo_aleatorio():
    return rl.Vector3(
        random.uniform(AREA_X_MIN, AREA_X_MAX),
        random.uniform(AREA_Y_MIN, AREA_Y_MAX),
        random.uniform(AREA_Z_MIN, AREA_Z_MAX),
    )


def puntuacion_por_tipo(tipo):
    if tipo == "rapido":
        return PUNTOS_RAPIDO
    if tipo == "tanque":
        return PUNTOS_TANQUE
    return PUNTOS_NORMAL


def numero_cubos_oleada(oleada):
    return min(4 + 3 * (oleada - 1), MAX_CUBOS)


def duracion_oleada(oleada):
    return TIEMPO_OLEADA_BASE + TIEMPO_EXTRA_POR_OLEADA * (oleada - 1)


def escala_velocidad_oleada(oleada):
    return min(1.0 + 0.12 * (oleada - 1), TOPE_ESCALA_VELOCIDAD)


def elegir_tipo_cubo(oleada):
    tipos = ["normal", "normal", "normal"]
    if oleada >= 1:
        tipos.append("rapido")
    if oleada >= 2:
        tipos.append("tanque")
    return random.choice(tipos)


class Cubo:
    def __init__(self, posicion, tamano, color, tipo="normal", vida=1, velocidad_max=None):
        self.posicion = posicion
        self.velocidad = rl.Vector3(0.0, 0.0, 0.0)
        self.tamano = tamano
        self.tamano_base = tamano
        self.color = color
        self.tipo = tipo
        self.vida = vida
        self.flash = 0.0
        self.velocidad_max = velocidad_max if velocidad_max is not None else random.uniform(VELOCIDAD_CUBO_MIN, VELOCIDAD_CUBO_MAX)
        self.objetivo_movimiento = objetivo_aleatorio()


@dataclass
class Fragmento:
    posicion: rl.Vector3
    velocidad: rl.Vector3
    vida: float
    color: rl.Color


def distancia_v3(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def explotar_cubo(posicion_origen, n_fragmentos=12):
    fragmentos = []
    for _ in range(n_fragmentos):
        theta = random.uniform(0.0, 2.0 * math.pi)
        phi = random.uniform(0.0, math.pi)
        radio = random.uniform(1.5, 4.5)
        velocidad = rl.Vector3(
            radio * math.sin(phi) * math.cos(theta),
            radio * math.cos(phi) + 2.0,
            radio * math.sin(phi) * math.sin(theta),
        )
        color = rl.color_from_hsv(random.uniform(0.0, 60.0), 0.8, 1.0)
        fragmento = Fragmento(
            rl.Vector3(posicion_origen.x, posicion_origen.y, posicion_origen.z),
            velocidad,
            random.uniform(0.9, VIDA_FRAGMENTO),
            color,
        )
        fragmentos.append(fragmento)
    return fragmentos


def actualizar_fragmentos(fragmentos, dt):
    for fragmento in fragmentos:
        fragmento.velocidad.y -= 9.8 * dt
        fragmento.posicion.x += fragmento.velocidad.x * dt
        fragmento.posicion.y += fragmento.velocidad.y * dt
        fragmento.posicion.z += fragmento.velocidad.z * dt
        fragmento.vida -= dt
    return [f for f in fragmentos if f.vida > 0.0 and f.posicion.y > -1.0]


def crear_cubo_de_tipo(tipo, oleada):
    escala = escala_velocidad_oleada(oleada)
    if tipo == "rapido":
        tamano = TAMANO_RAPIDO
        color = COLOR_RAPIDO
        velocidad_max = random.uniform(VELOCIDAD_RAPIDO_MIN, VELOCIDAD_RAPIDO_MAX) * escala
        vida = 1
    elif tipo == "tanque":
        tamano = TAMANO_TANQUE
        color = COLOR_TANQUE
        velocidad_max = random.uniform(VELOCIDAD_TANQUE_MIN, VELOCIDAD_TANQUE_MAX) * escala
        vida = VIDA_TANQUE
    else:
        tamano = random.uniform(0.6, 1.2)
        color = COLOR_NORMAL
        velocidad_max = random.uniform(VELOCIDAD_CUBO_MIN, VELOCIDAD_CUBO_MAX) * escala
        vida = 1
    posicion = rl.Vector3(
        random.uniform(AREA_X_MIN, AREA_X_MAX),
        random.uniform(1.0, 5.0) + 0.5 * tamano,
        random.uniform(AREA_Z_MIN, AREA_Z_MAX),
    )
    return Cubo(posicion, tamano, color, tipo, vida, velocidad_max)


def crear_oleada(oleada):
    cubos = []
    for _ in range(numero_cubos_oleada(oleada)):
        cubos.append(crear_cubo_de_tipo(elegir_tipo_cubo(oleada), oleada))
    return cubos


def explotar_cubo_de_lista(cubos, cubo, fragmentos):
    cubos.remove(cubo)
    nuevos = explotar_cubo(cubo.posicion)
    hueco = MAX_FRAGMENTOS - len(fragmentos)
    if hueco > 0:
        fragmentos.extend(nuevos[:hueco])
    return cubo.posicion


def intentar_explotar(cubos, posicion, fragmentos):
    cubos_en_rango = [c for c in cubos if distancia_v3(c.posicion, posicion) <= RADIO_EXPLOSION]
    if not cubos_en_rango:
        return 0, False, 0
    cubo = min(cubos_en_rango, key=lambda c: distancia_v3(c.posicion, posicion))
    pendientes = [cubo]
    destruidos = 0
    impacto_sin_destruir = False
    puntos_base = 0
    while pendientes:
        objetivo_cadena = pendientes.pop()
        if objetivo_cadena not in cubos:
            continue
        if objetivo_cadena.tipo == "tanque" and objetivo_cadena.vida > 1:
            objetivo_cadena.vida -= 1
            objetivo_cadena.flash = FLASH_DURACION
            impacto_sin_destruir = True
            continue
        posicion_explosiva = explotar_cubo_de_lista(cubos, objetivo_cadena, fragmentos)
        destruidos += 1
        puntos_base += puntuacion_por_tipo(objetivo_cadena.tipo)
        for vecino in list(cubos):
            if distancia_v3(vecino.posicion, posicion_explosiva) <= RADIO_CADENA:
                pendientes.append(vecino)
    return destruidos, impacto_sin_destruir, puntos_base


def dibujar_centrado(texto, y, tamano, color):
    ancho = rl.measure_text(texto, tamano)
    x = int((rl.get_screen_width() - ancho) / 2)
    rl.draw_text(texto, x, y, tamano, color)


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

    audio_inicializado = False
    sonido_explosion = None
    sonido_pop = None
    try:
        rl.init_audio_device()
        audio_inicializado = True
        sonido_explosion = rl.load_sound("explosion.wav")
        sonido_pop = rl.load_sound("pop.wav")
        rl.set_sound_volume(sonido_explosion, VOLUMEN_EXPLOSION)
        rl.set_sound_volume(sonido_pop, VOLUMEN_POP)
    except Exception:
        audio_inicializado = False
        sonido_explosion = None
        sonido_pop = None

    camara = rl.Camera3D(
        rl.Vector3(0.0, 6.0, 12.0),
        rl.Vector3(0.0, 2.0, 0.0),
        rl.Vector3(0.0, 1.0, 0.0),
        45.0,
        rl.CAMERA_PERSPECTIVE,
    )

    tracker = HandTracker()

    correcciones = []

    dt = 1.0 / 60.0

    cubos = []
    fragmentos = []
    mostrar_hud = True
    shake = 0.0

    estado_juego = ESTADO_INICIO
    oleada = 1
    score = 0
    combo_multiplicador = 1
    tiempo_ultimo_combo = 0.0
    temporizador_oleada = duracion_oleada(oleada)
    tiempo_transicion = 0.0

    historial_gestos = deque(maxlen=TAMANO_BUFFER_GESTOS)
    historial_presencia = deque(maxlen=TAMANO_BUFFER_GESTOS)
    gesto_efectivo = "indefinido"
    gesto_efectivo_previo = "indefinido"
    objetivo_estable = None
    hay_mano = False
    tiempo_ultima_accion = 0.0

    while not rl.window_should_close():
        estado = tracker.estado_actual()

        if rl.is_key_pressed(rl.KEY_M):
            mostrar_hud = not mostrar_hud

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

        ahora = time.monotonic()
        gesto_nuevo = gesto_efectivo != gesto_efectivo_previo

        if estado_juego == ESTADO_INICIO:
            if gesto_efectivo == "pinza" and gesto_nuevo:
                oleada = 1
                score = 0
                combo_multiplicador = 1
                tiempo_ultimo_combo = 0.0
                temporizador_oleada = duracion_oleada(oleada)
                tiempo_transicion = 0.0
                cubos = crear_oleada(oleada)
                fragmentos.clear()
                estado_juego = ESTADO_JUGANDO

        elif estado_juego == ESTADO_JUGANDO:
            if tiempo_transicion > 0.0:
                tiempo_transicion -= dt
                if tiempo_transicion <= 0.0:
                    oleada += 1
                    cubos = crear_oleada(oleada)
                    temporizador_oleada = duracion_oleada(oleada)
                    tiempo_transicion = 0.0
            else:
                temporizador_oleada -= dt
                if temporizador_oleada <= 0.0:
                    if cubos:
                        estado_juego = ESTADO_GAME_OVER
                        combo_multiplicador = 1
                else:
                    if (ahora - tiempo_ultima_accion) >= COOLDOWN_ACCION:
                        if gesto_efectivo == "puno" and gesto_nuevo and objetivo_estable is not None:
                            destruidos, impacto, puntos_base = intentar_explotar(cubos, objetivo_estable, fragmentos)
                            if destruidos > 0 or impacto:
                                tiempo_ultima_accion = ahora
                                if destruidos > 0:
                                    shake = SHAKE_DURACION
                                    if sonido_explosion is not None:
                                        rl.play_sound(sonido_explosion)
                                    if ahora - tiempo_ultimo_combo <= VENTANA_COMBO:
                                        combo_multiplicador = min(COMBO_MAX, combo_multiplicador + 1)
                                    else:
                                        combo_multiplicador = 1
                                    tiempo_ultimo_combo = ahora
                                    score += puntos_base * combo_multiplicador
                                    if not cubos:
                                        tiempo_transicion = TIEMPO_TRANSICION
                                else:
                                    shake = SHAKE_DURACION * 0.5
                                    if sonido_pop is not None:
                                        rl.play_sound(sonido_pop)
                                    combo_multiplicador = 1

        elif estado_juego == ESTADO_GAME_OVER:
            if gesto_efectivo == "pinza" and gesto_nuevo:
                oleada = 1
                score = 0
                combo_multiplicador = 1
                tiempo_ultimo_combo = 0.0
                temporizador_oleada = duracion_oleada(oleada)
                tiempo_transicion = 0.0
                cubos = crear_oleada(oleada)
                fragmentos.clear()
                estado_juego = ESTADO_JUGANDO

        gesto_efectivo_previo = gesto_efectivo

        objetivo = objetivo_estable if (hay_mano and objetivo_estable is not None) else None
        cubo_cercano = None
        if estado_juego == ESTADO_JUGANDO and objetivo is not None and cubos:
            cubo_cercano = min(cubos, key=lambda c: distancia_v3(c.posicion, objetivo))

        for cubo in cubos:
            if cubo.flash > 0.0:
                cubo.flash -= dt
                if cubo.flash < 0.0:
                    cubo.flash = 0.0
            if estado_juego != ESTADO_JUGANDO:
                continue

            agarrado = hay_mano and objetivo is not None and cubo is cubo_cercano and gesto_efectivo == "mano_abierta"
            if agarrado:
                cubo.posicion = rl.Vector3(objetivo.x, objetivo.y, objetivo.z)
                cubo.velocidad = rl.Vector3(0.0, 0.0, 0.0)
                continue

            dx = cubo.objetivo_movimiento.x - cubo.posicion.x
            dy = cubo.objetivo_movimiento.y - cubo.posicion.y
            dz = cubo.objetivo_movimiento.z - cubo.posicion.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < UMBRAL_WAYPOINT:
                cubo.objetivo_movimiento = objetivo_aleatorio()
            else:
                cubo.velocidad = rl.Vector3(
                    dx / dist * cubo.velocidad_max,
                    dy / dist * cubo.velocidad_max,
                    dz / dist * cubo.velocidad_max,
                )
                cubo.posicion.x += cubo.velocidad.x * dt
                cubo.posicion.y += cubo.velocidad.y * dt
                cubo.posicion.z += cubo.velocidad.z * dt

            cubo.posicion.x = max(AREA_X_MIN, min(AREA_X_MAX, cubo.posicion.x))
            cubo.posicion.y = max(AREA_Y_MIN, min(AREA_Y_MAX, cubo.posicion.y))
            cubo.posicion.z = max(AREA_Z_MIN, min(AREA_Z_MAX, cubo.posicion.z))

        fragmentos = actualizar_fragmentos(fragmentos, dt)

        rl.update_camera(camara, rl.CAMERA_FREE)

        camara_dibujo = camara
        if shake > 0.0:
            intensidad = SHAKE_MAGNITUD * (shake / SHAKE_DURACION)
            camara_dibujo = rl.Camera3D(
                camara.position,
                rl.Vector3(
                    camara.target.x + random.uniform(-1.0, 1.0) * intensidad,
                    camara.target.y + random.uniform(-1.0, 1.0) * intensidad,
                    camara.target.z + random.uniform(-1.0, 1.0) * intensidad,
                ),
                camara.up,
                camara.fovy,
                camara.projection,
            )
            shake -= dt
            if shake < 0.0:
                shake = 0.0

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
        rl.begin_mode_3d(camara_dibujo)
        rl.draw_grid(20, 1.0)
        for cubo in cubos:
            color = cubo.color
            if cubo.flash > 0.0:
                color = rl.color_lerp(color, rl.WHITE, min(1.0, cubo.flash / FLASH_DURACION))
            rl.draw_cube(cubo.posicion, cubo.tamano, cubo.tamano, cubo.tamano, color)
            rl.draw_cube_wires(cubo.posicion, cubo.tamano, cubo.tamano, cubo.tamano, rl.fade(color, 0.5))
        for fragmento in fragmentos:
            alfa = max(0.0, min(1.0, fragmento.vida / VIDA_FRAGMENTO))
            rl.draw_cube(fragmento.posicion, 0.2, 0.2, 0.2, rl.fade(fragmento.color, alfa))
        if hay_mano and objetivo is not None:
            rl.draw_sphere(objetivo, 0.2, rl.BLUE)
        rl.end_mode_3d()

        if estado_juego == ESTADO_INICIO:
            dibujar_centrado("ARCADE DE MANOS", 150, 60, rl.DARKBLUE)
            dibujar_centrado("puno = explotar", 280, 24, rl.DARKGRAY)
            dibujar_centrado("mano abierta = agarrar", 320, 24, rl.DARKGRAY)
            dibujar_centrado("pinza para empezar", 380, 24, rl.MAROON)

        elif estado_juego == ESTADO_JUGANDO and mostrar_hud:
            rl.draw_fps(10, 10)
            rl.draw_text(f"Oleada: {oleada}", 10, 40, 20, rl.DARKGRAY)
            rl.draw_text(f"Tiempo: {max(0.0, temporizador_oleada):.1f}s", 10, 70, 20, rl.DARKGRAY)
            rl.draw_text(f"Score: {score}", 10, 100, 20, rl.DARKGRAY)
            rl.draw_text(f"Cubos: {len(cubos)}", 10, 130, 20, rl.DARKGRAY)
            rl.draw_text(f"Gesto: {gesto_efectivo}", 10, 160, 20, rl.DARKGRAY)
            rl.draw_text(f"Correcciones: {len(correcciones)}", 10, 190, 20, rl.DARKGRAY)
            combo_activo = combo_multiplicador > 1 and (ahora - tiempo_ultimo_combo) <= VENTANA_COMBO
            if combo_activo:
                rl.draw_text(f"Combo x{combo_multiplicador}", 10, 220, 30, rl.ORANGE)

        elif estado_juego == ESTADO_GAME_OVER:
            dibujar_centrado("GAME OVER", 150, 60, rl.RED)
            dibujar_centrado(f"Score final: {score}", 250, 30, rl.DARKBLUE)
            dibujar_centrado(f"Oleada alcanzada: {oleada}", 300, 24, rl.DARKGRAY)
            dibujar_centrado("pinza para reintentar", 380, 24, rl.MAROON)

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

    if sonido_explosion is not None:
        rl.unload_sound(sonido_explosion)
    if sonido_pop is not None:
        rl.unload_sound(sonido_pop)
    if audio_inicializado:
        rl.close_audio_device()

    rl.close_window()


if __name__ == "__main__":
    main()