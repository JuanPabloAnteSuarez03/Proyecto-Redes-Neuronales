# LEEME - Entrega 1 (paquete ligero)

Este ZIP es la **versión ligera** del proyecto, preparada para la plataforma
de la universidad, que no acepta archivos de más de 10 MB.

## Qué incluye

- Código fuente completo (`src/`, `demo.py`)
- Notebook ejecutado (`notebooks/entrega1_mlp.ipynb`)
- Informe técnico (`informe/informe.pdf`)
- Diapositivas (`presentacion/presentacion.pdf`) y guion
- Resultados: métricas, reportes de clasificación, historiales y figuras
- Enunciado del proyecto

## Qué se excluyó por tamaño

| Archivo | Tamaño | Dónde obtenerlo |
|---|---|---|
| `results/models/mlp_1x256.keras` | 21 MB | Repositorio GitHub |
| `data/20newsgroups_train.csv` | 14 MB | Se descarga solo con `src/data.py` |
| `data/20newsgroups_test.csv` | 8 MB | Se descarga solo con `src/data.py` |
| `results/runs/*/test_probabilities.npy` | 5,4 MB | Se regeneran al entrenar |

## Repositorio con el paquete completo

    https://github.com/JuanPabloAnteSuarez03/Proyecto-Redes-Neuronales

    git clone https://github.com/JuanPabloAnteSuarez03/Proyecto-Redes-Neuronales.git

En el repositorio está el ZIP completo (`Entrega1_RedesNeuronales.zip`), el
modelo entrenado y los datos procesados.

## Cómo regenerar lo que falta

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

    # Descarga 20 Newsgroups, entrena y regenera modelo, datos, métricas y figuras
    python -m src.experiments

    # Predicción individual con el modelo entrenado
    python -m src.predict "my X server crashes with the new driver"

## Modelo final

- MLP 1x256, dropout 0.3, TF-IDF con 20.000 términos
- Exactitud: 0.6771
- F1 macro: 0.6663
- F1 ponderado: 0.6762
