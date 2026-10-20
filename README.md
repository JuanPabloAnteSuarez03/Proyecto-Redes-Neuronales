# Entrega 1 — Perceptrón Multicapa para clasificación de texto

**Curso:** Redes Neuronales · Universidad del Valle · 2026-2

Clasificación de documentos del corpus **20 Newsgroups** en 20 categorías con
una **MLP** implementada en **Keras/TensorFlow**. La entrada son textos crudos
transformados a TF-IDF mediante una capa `TextVectorization`; la red tiene una
capa densa de 256 neuronas, dropout 0.3 y salida softmax.

## Resultados principales

| Métrica (conjunto de prueba) | Valor |
|---|---|
| Exactitud | 0.6771 |
| F1 macro | 0.6663 |
| F1 ponderado | 0.6762 |
| Exactitud top-3 | 0.8452 |
| Parámetros | 5.13 M |
| Entrenamiento (CPU) | ~30 s |

> Los valores exactos se generan en `results/experimentos.csv` y se insertan
> automáticamente en el informe y las diapositivas.

## Estructura

```
Proyecto/
├── src/
│   ├── config.py        # hiperparámetros, rutas y lista de experimentos
│   ├── data.py          # descarga/cacheo, limpieza y partición
│   ├── model.py         # vectorizador TF-IDF + MLP
│   ├── train.py         # entrenamiento, parada temprana y evaluación
│   ├── experiments.py   # ejecuta las 9 configuraciones y genera figuras
│   ├── plots.py         # EDA, curvas, matriz de confusión, comparaciones
│   ├── predict.py       # demo de predicción por CLI
│   └── build_docs.py    # tablas y macros LaTeX a partir de los resultados
├── notebooks/
│   ├── entrega1_mlp.ipynb  # notebook ejecutado con el análisis
│   └── build_notebook.py   # generador del notebook
├── data/                # CSV cacheados de 20 Newsgroups
├── results/             # métricas, modelos, historiales y figuras
├── informe/             # informe técnico en LaTeX y PDF
├── presentacion/        # diapositivas Beamer y PDF
├── tectonic             # compilador LaTeX autocontenido (opcional)
├── requirements.txt
└── README.md
```

## Reproducción

Se recomienda Python 3.12 (TensorFlow 2.20 no soporta Python 3.14 al momento
de la entrega).

```bash
# 1. Crear el entorno e instalar dependencias
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Descargar datos, entrenar las 9 configuraciones y generar figuras
.venv/bin/python -m src.experiments

# 3. (Opcional) regenerar tablas y macros del informe
.venv/bin/python -m src.build_docs

# 4. (Opcional) recompilar informe y diapositivas.
#    Los PDF ya están incluidos; se requiere Tectonic instalado.
tectonic -X compile informe/informe.tex
cd presentacion && tectonic -X compile presentacion.tex
```

La primera ejecución descarga 20 Newsgroups desde el repositorio de
scikit-learn y lo cachea en `data/20newsgroups_train.csv` y
`data/20newsgroups_test.csv`. Las ejecuciones posteriores no requieren red.

## Notebook

`notebooks/entrega1_mlp.ipynb` está ejecutado y contiene:

1. Contexto y objetivos.
2. Análisis exploratorio del corpus.
3. Arquitectura y condiciones de entrenamiento.
4. Métricas, matriz de confusión y análisis de errores.
5. Comparación de configuraciones.
6. Demostración con textos nuevos.

Para regenerarlo y ejecutarlo:

```bash
.venv/bin/python notebooks/build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    notebooks/entrega1_mlp.ipynb
```

## Demo de predicción

```bash
# Demostración preparada para la sustentación (5 ejemplos)
.venv/bin/python demo.py

# Predicción libre desde la línea de comandos
.venv/bin/python -m src.predict "my X server crashes with the new driver"
.venv/bin/python -m src.predict --ejemplos 3
```

## Decisiones metodológicas

- **Sin fuga de información:** se eliminan cabeceras, pies de página y citas;
  el vocabulario TF-IDF se adapta solo con entrenamiento y el test se evalúa
  una única vez.
- **Partición:** 90 % train / 10 % validación estratificada sobre la partición
  oficial de entrenamiento.
- **Selección de modelo:** parada temprana sobre `val_accuracy`, restaurando
  los pesos de la mejor época. El F1 macro se usa como métrica principal.
- **Comparación justa:** las 9 configuraciones comparten semilla, partición,
  vocabulario y criterio de parada.
- **Clasificación multiclase:** aunque el enunciado menciona múltiples
  etiquetas, los tres datasets propuestos tienen una etiqueta por documento.

## Integrantes

- **Nicolás Garcés** — código 218066
- **Juan Pablo Ante** — código 2140132
