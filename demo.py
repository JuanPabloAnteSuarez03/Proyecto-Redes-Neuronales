"""Demostración ejecutable para la sustentación de la Entrega 1.

Uso:
    .venv/bin/python demo.py
"""
from __future__ import annotations

from src.data import load_dataframe
from src.predict import load_trained_model, predict_texts


def main() -> None:
    model = load_trained_model()
    _, _, label_names = load_dataframe()

    ejemplos = [
        "My new graphics card works fine, but the X server crashes when I "
        "start it. Which driver version should I use with this kernel?",
        "The team won the hockey game in overtime thanks to a goal by the rookie.",
        "NASA is planning a new mission to study the atmosphere of Mars.",
        "For sale: mountain bike, 21 speeds, aluminum frame, barely used. "
        "Asking $250 or best offer.",
        "I think the government should explain its position on the new tax "
        "reform before the next election.",
    ]

    print("=" * 78)
    print("DEMO — Clasificador MLP sobre 20 Newsgroups")
    print("=" * 78)
    for texto in ejemplos:
        result = predict_texts(model, [texto], label_names, top_k=3)[0]
        print(f"\nTexto: {texto}")
        print(f"Predicción: {result['prediccion']} ({result['confianza']:.2%})")
        print("Top 3:")
        for item in result["top_k"]:
            print(f"  - {item['clase']:<30s} {item['probabilidad']:.2%}")


if __name__ == "__main__":
    main()
