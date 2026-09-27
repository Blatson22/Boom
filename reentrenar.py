import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATASET = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RUTA_BASE, "dataset.npz")
RUTA_CORRECCIONES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RUTA_BASE, "correcciones.npz")
RUTA_MODELO = sys.argv[3] if len(sys.argv) > 3 else os.path.join(RUTA_BASE, "gestos_mlp.joblib")
RUTA_ENTRENAR = os.path.join(RUTA_BASE, "entrenar_modelo.py")


def cargar_npz(ruta):
    if not os.path.exists(ruta):
        return None, None
    datos = np.load(ruta, allow_pickle=True)
    if "X" not in datos or "y" not in datos:
        return None, None
    X = np.asarray(datos["X"], dtype=np.float32)
    y = np.asarray(datos["y"])
    if X.ndim != 2 or X.shape[1] != 63 or len(y) == 0 or len(y) != X.shape[0]:
        return None, None
    return X, y


def combinar_dataset():
    X, y = cargar_npz(RUTA_DATASET)
    if X is None:
        print("Dataset base invalido o inexistente:", RUTA_DATASET)
        return None, None
    X_corr, y_corr = cargar_npz(RUTA_CORRECCIONES)
    if X_corr is not None:
        X = np.vstack([X, X_corr])
        y = np.concatenate([y, y_corr])
    return X, y


def main():
    X, y = combinar_dataset()
    if X is None:
        return

    total = len(y)
    conteo = Counter(str(e) for e in y)
    print("Total de muestras usadas:", total, flush=True)
    for clase in sorted(conteo):
        print(clase + ":", conteo[clase], flush=True)

    descriptor, ruta_temporal = tempfile.mkstemp(suffix=".npz")
    os.close(descriptor)
    np.savez(ruta_temporal, X=X, y=y)

    try:
        subprocess.run([sys.executable, RUTA_ENTRENAR, ruta_temporal, RUTA_MODELO], check=True)
    finally:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)

    if os.path.exists(RUTA_CORRECCIONES):
        np.savez(RUTA_DATASET, X=X, y=y)
        os.remove(RUTA_CORRECCIONES)
        print("Correcciones incorporadas a", RUTA_DATASET, "y limpiadas de", RUTA_CORRECCIONES)


if __name__ == "__main__":
    main()