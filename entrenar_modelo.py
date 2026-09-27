import os
import sys

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUTA_DATASET = sys.argv[1] if len(sys.argv) > 1 else "dataset.npz"
RUTA_MODELO = sys.argv[2] if len(sys.argv) > 2 else "gestos_mlp.joblib"


def main():
    if not os.path.exists(RUTA_DATASET):
        print("No se encontro el dataset:", RUTA_DATASET)
        return

    datos = np.load(RUTA_DATASET, allow_pickle=True)
    if "X" not in datos or "y" not in datos:
        print("El dataset no contiene los campos X e y")
        return

    X = np.asarray(datos["X"], dtype=np.float32)
    y = np.asarray(datos["y"])
    if X.ndim != 2 or X.shape[1] != 63 or len(y) == 0 or len(y) != X.shape[0]:
        print("Formato de dataset invalido: se esperan X (n, 63) e y (n,)")
        return

    clases = sorted({str(e) for e in y})

    X_entren, X_prueba, y_entren, y_prueba = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=42),
    )
    pipeline.fit(X_entren, y_entren)

    y_pred = pipeline.predict(X_prueba)
    exactitud = accuracy_score(y_prueba, y_pred)

    print("Accuracy:", round(exactitud, 4))
    print(classification_report(y_prueba, y_pred, target_names=clases))

    joblib.dump({"pipeline": pipeline, "clases": clases}, RUTA_MODELO)
    print("Modelo guardado en:", RUTA_MODELO)
    print("Clases (orden canonico):", clases)
    print(
        "Formato de entrada: vector de 63 floats resultado de normalizar_landmarks "
        "(21 puntos en orden [x,y,z], muñeca en el origen y dividido por tamano_mano)."
    )


if __name__ == "__main__":
    main()