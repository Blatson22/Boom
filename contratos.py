from dataclasses import dataclass

import pyray as rl


@dataclass
class EstadoMano:
    posicion: rl.Vector3
    gesto: str
    confianza: float = 1.0