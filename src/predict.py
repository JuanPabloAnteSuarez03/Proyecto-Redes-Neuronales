"""Predicción con el modelo entrenado y demostración por línea de comandos."""
from __future__ import annotations

import argparse
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from .config import MODELS_DIR
from .data import load_dataframe

DEFAULT_MODEL = MODELS_DIR / "mlp_1x256.keras"


def load_trained_model(path: str | Path = DEFAULT_MODEL) -> keras.Model:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo {path}. Ejecute primero `python -m src.experiments`."
        )
    return keras.models.load_model(path, compile=False)


def predict_texts(
    model: keras.Model,
    texts: list[str],
    label_names: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Devuelve las k clases más probables para cada texto."""
    batch = tf.constant(np.asarray(texts, dtype=object), dtype=tf.string)
    probabilities = model.predict(batch, verbose=0)
    results = []
    for text, probs in zip(texts, probabilities):
        order = np.argsort(probs)[::-1][:top_k]
        results.append(
            {
                "texto": text,
                "prediccion": label_names[int(order[0])],
                "confianza": float(probs[order[0]]),
                "top_k": [
                    {"clase": label_names[int(i)], "probabilidad": float(probs[i])}
                    for i in order
                ],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción con la MLP entrenada")
    parser.add_argument("texto", nargs="*", help="Texto a clasificar")
    parser.add_argument("--modelo", default=str(DEFAULT_MODEL))
    parser.add_argument("--ejemplos", type=int, default=0, help="Muestra ejemplos del conjunto de prueba")
    args = parser.parse_args()

    model = load_trained_model(args.modelo)
    _, test, label_names = load_dataframe()

    if args.texto:
        texts = [" ".join(args.texto)]
    elif args.ejemplos > 0:
        texts = test.sample(args.ejemplos, random_state=7)["text"].tolist()
    else:
        texts = [
            "I just installed the new graphics card and the X server crashes when I start it. "
            "Does anyone know which driver version works with this kernel?",
            "The hockey team won the game in overtime with a goal from the rookie player.",
            "NASA announced a new mission to study the atmosphere of Mars next year.",
        ]

    for result in predict_texts(model, texts, label_names):
        print("\nTexto:", result["texto"][:240].replace("\n", " "))
        print(f"Predicción: {result['prediccion']} ({result['confianza']:.2%})")
        print("Top 3:")
        for item in result["top_k"]:
            print(f"  - {item['clase']}: {item['probabilidad']:.2%}")


if __name__ == "__main__":
    main()
