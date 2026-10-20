# Resultados de la Entrega 1

Todos los archivos se regeneran con:

```bash
.venv/bin/python -m src.experiments
```

## Archivos principales

| Archivo | Contenido |
|---|---|
| `experimentos.csv` | Tabla comparativa de las 9 configuraciones (exactitud, F1 macro, top-3, etc.). |
| `experimentos.json` | Resultados completos, incluido el reporte por clase y la matriz de confusión. |
| `runs/<config>/history.csv` | Pérdida y exactitud por época de cada configuración. |
| `runs/<config>/metrics.json` | Métricas detalladas y matriz de confusión de cada configuración. |
| `runs/<config>/classification_report.txt` | Precisión, recall, F1 y soporte por clase. |
| `runs/<config>/test_probabilities.npy` | Probabilidades predichas para el conjunto de prueba. |
| `figures/` | Figuras usadas en el informe y las diapositivas. |
| `models/mlp_1x256.keras` | Modelo principal listo para inferencia (sin estado del optimizador). |

El modelo se carga con:

```python
from src.predict import load_trained_model
model = load_trained_model()
```
