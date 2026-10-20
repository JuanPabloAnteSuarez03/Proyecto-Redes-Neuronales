"""Genera el notebook de la Entrega 1 (se ejecuta con nbconvert)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "entrega1_mlp.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Entrega 1 — Perceptrón Multicapa para clasificación de texto

**Curso:** Redes Neuronales · Universidad del Valle · 2026-2

**Integrantes:** Nicolás Garcés (218066) · Juan Pablo Ante (2140132)

Este notebook presenta el desarrollo de la primera entrega: clasificación de
documentos del corpus **20 Newsgroups** en 20 categorías mediante una red
**Perceptrón Multicapa (MLP)** implementada en **Keras/TensorFlow**.

El flujo completo (descarga, preprocesamiento, entrenamiento, ajuste de
hiperparámetros y evaluación) está implementado en los módulos de `src/` y se
reproduce con `python -m src.experiments`. Aquí se cargan los artefactos ya
entrenados, se analizan los resultados y se ejecuta una demostración.
""")

code(r"""
# Si el notebook se ejecuta desde notebooks/, agrega la raíz del proyecto al path
from pathlib import Path
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

from src.config import PRIMARY_CONFIG, RESULTS_DIR, SEED
from src.data import load_dataframe, dataframe_summary
from src.predict import load_trained_model, predict_texts

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 160)
print("Raíz del proyecto:", ROOT)
print("Semilla:", SEED)
""")

md(r"""
## 1. Conjunto de datos

**20 Newsgroups** (Lang, 1995) es una colección de aproximadamente 20.000
mensajes publicados en 20 grupos de noticias de Usenet durante 1993. Es un
problema de clasificación multiclase (no multietiqueta) balanceado de forma
aproximada, ampliamente usado como referencia en clasificación de texto.

Se emplea la partición oficial de scikit-learn: 11.314 documentos de
entrenamiento y 7.532 de prueba. Para evitar fuga de información se eliminan
**cabeceras, pies de página y citas**, que contienen el nombre del grupo y
delatarían la etiqueta.
""")

code(r"""
train, test, label_names = load_dataframe()
summary = dataframe_summary(train, test)

stats = pd.DataFrame(
    {
        "Valor": [
            summary["n_train"], summary["n_test"], summary["n_classes"],
            summary["words_mean"], summary["words_median"],
            summary["words_p95"], summary["words_max"], summary["empty_docs"],
        ]
    },
    index=[
        "Documentos de entrenamiento", "Documentos de prueba", "Clases",
        "Palabras por documento (media)", "Palabras por documento (mediana)",
        "Percentil 95 de longitud", "Longitud máxima", "Documentos vacíos",
    ],
)
stats["Valor"] = stats["Valor"].apply(lambda v: f"{v:,.1f}" if isinstance(v, float) else f"{v:,}")
display(stats)

display(train[["label_name", "text"]].head(4))
""")

code(r"""
class_counts = train["label_name"].value_counts().rename("Documentos")
display(class_counts.to_frame().T)

display(Image(filename=str(RESULTS_DIR / "figures" / "eda_distribucion_clases.png")))
display(Image(filename=str(RESULTS_DIR / "figures" / "eda_longitud_documentos.png")))
""")

md(r"""
## 2. Preprocesamiento y arquitectura

La entrada del modelo son **textos crudos**. La primera capa,
`TextVectorization`, aplica minúsculas, elimina puntuación y construye un
vocabulario de 20.000 tokens. Luego calcula la representación **TF-IDF** de
cada documento (frecuencia del término ponderada por la frecuencia inversa en
el documento, normalizada). El vectorizador se adapta **solo con el conjunto
de entrenamiento** para no filtrar información del test.

Sobre esa representación se apila una capa densa con activación ReLU, dropout
y una capa de salida softmax:

`TF-IDF (20.000) → Dense 256 + ReLU → Dropout 0.3 → Dense 20 + Softmax`

El barrido de hiperparámetros mostró que una sola capa oculta generaliza mejor
que arquitecturas más profundas para este corpus.
""")

code(r"""
model = load_trained_model()
model.summary()
""")

md(r"""
## 3. Condiciones de entrenamiento

| Elemento | Valor |
|---|---|
| Partición | 90 % entrenamiento / 10 % validación (estratificada) |
| Prueba | 7.532 documentos, nunca usados en entrenamiento |
| Optimizador | Adam, tasa de aprendizaje 1e-3 |
| Pérdida | Entropía cruzada categórica dispersa |
| Batch | 128 |
| Épocas máximas | 20 con parada temprana (paciencia 3 sobre `val_loss`) |
| Regularización | Dropout 0.3 |
| Semilla | 42 (determinismo activado) |
""")

code(r"""
primary_dir = RESULTS_DIR / "runs" / PRIMARY_CONFIG.name
history = pd.read_csv(primary_dir / "history.csv")
result = json.loads((primary_dir / "result.json").read_text())
metrics = json.loads((primary_dir / "metrics.json").read_text())

display(history.tail(3))
print(f"Épocas ejecutadas: {result['epochs_run']} | mejor época: {result['best_epoch']} "
      f"| parámetros: {result['n_parameters']:,} | tiempo: {result['train_time_seconds']:.1f} s")

display(Image(filename=str(RESULTS_DIR / "figures" / "curvas_entrenamiento_primary.png")))
""")

md(r"""
## 4. Evaluación en el conjunto de prueba

Se reportan exactitud, precisión, recall y F1 macro (promedio no ponderado por
clase), F1 ponderado y exactitud top-3. El **F1 macro** es la métrica principal
porque trata todas las clases por igual, independientemente de su frecuencia.
""")

code(r"""
metric_table = pd.DataFrame(
    {
        "Métrica": ["Exactitud", "Precisión macro", "Recall macro", "F1 macro",
                    "F1 ponderado", "Exactitud top-3"],
        "Valor": [metrics["accuracy"], metrics["macro_precision"], metrics["macro_recall"],
                  metrics["macro_f1"], metrics["weighted_f1"], metrics["top3_accuracy"]],
    }
)
metric_table["Valor"] = metric_table["Valor"].map(lambda v: f"{v:.4f}")
display(metric_table)

print(metrics["report_text"])
""")

code(r"""
display(Image(filename=str(RESULTS_DIR / "figures" / "matriz_confusion_primary.png")))
display(Image(filename=str(RESULTS_DIR / "figures" / "f1_por_clase_primary.png")))
""")

md(r"""
### Análisis de errores

Las confusiones más frecuentes se concentran entre grupos temáticamente
cercanos: los cinco grupos de computación (`comp.*`), las categorías de
religión (`alt.atheism`, `soc.religion.christian`, `talk.religion.misc`) y los
debates políticos (`talk.politics.*`). Esto es esperable porque comparten
vocabulario y muchos mensajes responden a hilos de otros grupos.
""")

code(r"""
confusions = pd.read_json(RESULTS_DIR / "figures" / "confusiones_principales.json")
display(confusions)
""")

md(r"""
## 5. Comparación de configuraciones

Se entrenaron siete configuraciones con el mismo protocolo, variando
profundidad, dropout, tamaño de vocabulario y balanceo de clases. La tabla y la
figura se generan en `src/experiments.py`.
""")

code(r"""
experiments = pd.read_csv(RESULTS_DIR / "experimentos.csv")
display(experiments[["name", "hidden_units", "dropout_rates", "vocab_size",
                     "balanced_class_weight", "accuracy", "macro_f1", "top3_accuracy"]]
        .sort_values("macro_f1", ascending=False)
        .style.format({"accuracy": "{:.4f}", "macro_f1": "{:.4f}", "top3_accuracy": "{:.4f}"}))

display(Image(filename=str(RESULTS_DIR / "figures" / "comparacion_experimentos.png")))
""")

md(r"""
## 6. Demostración

Se clasifican tres textos nuevos (no pertenecientes al corpus) y se muestran
las tres clases más probables.
""")

code(r"""
ejemplos = [
    "My new graphics card works fine, but the X server crashes when I start it. "
    "Which driver version should I use with this kernel?",
    "The team won the hockey game in overtime thanks to a goal by the rookie.",
    "NASA is planning a new mission to study the atmosphere of Mars and its moons.",
]
for item in predict_texts(model, ejemplos, label_names):
    print("\nTexto:", item["texto"])
    print("Predicción:", item["prediccion"], f"({item['confianza']:.2%})")
    for top in item["top_k"]:
        print(f"   {top['clase']:<28s} {top['probabilidad']:.2%}")
""")

md(r"""
## 7. Reproducibilidad

```bash
# Crear entorno e instalar dependencias
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Descargar datos, entrenar las 7 configuraciones y generar figuras
.venv/bin/python -m src.experiments

# Predicción por línea de comandos
.venv/bin/python -m src.predict "texto a clasificar"
```

Para reentrenar el modelo principal desde este notebook, cambie `ENTRENAR` a
`True` en la siguiente celda (tarda unos minutos en CPU).
""")

code(r"""
ENTRENAR = False
if ENTRENAR:
    from src.train import run_experiment
    run_experiment(PRIMARY_CONFIG)
""")

md(r"""
## 8. Conclusiones

- La MLP con representación TF-IDF alcanza una exactitud cercana al 68 % y un
  F1 macro cercano a 0.67 en 20 clases, muy por encima del azar (5 %).
- La representación TF-IDF permite que una red puramente densa supere a los
  modelos lineales clásicos: el modelo lineal ya es un baseline fuerte
  (~66 %), y una capa oculta aporta una mejora consistente.
- El dropout y la parada temprana controlan el sobreajuste: la exactitud de
  entrenamiento supera el 95 % mientras la de validación se estabiliza
  alrededor del 75 %.
- Los errores se concentran en clases semánticamente solapadas, lo que sugiere
  que la siguiente etapa (RNN/Transformer) debe capturar mejor el contexto y
  el orden de las palabras.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Notebook escrito en {OUT}")
