import threading

import pyray as rl

from contratos import EstadoMano
from detectar_mano import (
    UMBRAL_EXTENDIDO,
    UMBRAL_PELLIZCO,
    crear_detector,
    diagnostico_gesto,
    iniciar_captura,
    landmarks_a_estado,
    obtener_landmarks,
)

K_RESORTE = 40.0
AMORTIGUACION = 8.0


class HandTracker:
    def __init__(self):
        self.captura = iniciar_captura()
        self.detector = crear_detector(max_manos=1)
        self._lock = threading.Lock()
        self._estado = None
        self._diag = None
        self._activo = self.captura is not None
        self._hilo = None
        if self._activo:
            self._hilo = threading.Thread(target=self._loop, daemon=True)
            self._hilo.start()

    def _loop(self):
        while self._activo:
            landmarks = obtener_landmarks(self.captura, self.detector)
            nuevo = landmarks_a_estado(landmarks) if landmarks is not None else None
            diag = diagnostico_gesto(landmarks) if landmarks is not None else None
            with self._lock:
                self._estado = nuevo
                self._diag = diag

    def estado_actual(self) -> EstadoMano | None:
        with self._lock:
            return self._estado

    def diagnostico_actual(self):
        with self._lock:
            return self._diag

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

    posicion = rl.Vector3(0.0, 3.0, 0.0)
    velocidad = rl.Vector3(0.0, 0.0, 0.0)
    gravedad = rl.Vector3(0.0, -9.8, 0.0)
    escala_cubo = 1.0
    dt = 1.0 / 60.0

    while not rl.window_should_close():
        estado = tracker.estado_actual()

        if estado is not None:
            objetivo = estado.posicion

            if estado.gesto == "puno":
                posicion = rl.Vector3(objetivo.x, objetivo.y, objetivo.z)
                velocidad = rl.Vector3(0.0, 0.0, 0.0)
            else:
                fuerza = rl.Vector3(
                    (objetivo.x - posicion.x) * K_RESORTE,
                    (objetivo.y - posicion.y) * K_RESORTE,
                    (objetivo.z - posicion.z) * K_RESORTE,
                )
                velocidad.x += (fuerza.x - velocidad.x * AMORTIGUACION) * dt
                velocidad.y += (fuerza.y - velocidad.y * AMORTIGUACION + gravedad.y) * dt
                velocidad.z += (fuerza.z - velocidad.z * AMORTIGUACION) * dt

            if estado.gesto == "pinza":
                escala_cubo = max(0.3, min(2.5, escala_cubo + 0.5 * dt))
            elif estado.gesto == "mano_abierta":
                escala_cubo += (1.0 - escala_cubo) * 2.0 * dt
        else:
            velocidad.y += gravedad.y * dt

        posicion.x += velocidad.x * dt
        posicion.y += velocidad.y * dt
        posicion.z += velocidad.z * dt

        if posicion.y < 0.5 * escala_cubo:
            posicion.y = 0.5 * escala_cubo
            velocidad.y = -velocidad.y * 0.5

        rl.update_camera(camara, rl.CAMERA_ORBITAL)

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
        rl.begin_mode_3d(camara)
        rl.draw_grid(20, 1.0)
        rl.draw_cube(posicion, escala_cubo, escala_cubo, escala_cubo, rl.RED)
        rl.draw_cube_wires(posicion, escala_cubo, escala_cubo, escala_cubo, rl.MAROON)
        if estado is not None:
            rl.draw_sphere(estado.posicion, 0.2, rl.BLUE)
        rl.end_mode_3d()
        rl.draw_fps(10, 10)
        if estado is not None:
            rl.draw_text(f"Gesto: {estado.gesto}", 10, 40, 20, rl.DARKGRAY)
        diag = tracker.diagnostico_actual()
        if diag:
            rl.draw_text(f"pulgar-indice {diag['pulgar_indice']}  (pinza < {UMBRAL_PELLIZCO})", 10, 70, 20, rl.DARKGRAY)
            rl.draw_text(
                f"medio {diag['medio']}  anular {diag['anular']}  menique {diag['menique']}  (ext > {UMBRAL_EXTENDIDO})",
                10, 100, 20, rl.DARKGRAY,
            )
        rl.end_drawing()

    tracker.cerrar()
    rl.close_window()


if __name__ == "__main__":
    main()