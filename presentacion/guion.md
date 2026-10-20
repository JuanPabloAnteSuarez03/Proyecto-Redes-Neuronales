# Guion de sustentación — Entrega 1 (10 minutos)

**Diapositivas:** `presentacion.pdf` (13 diapositivas).
**Demo:** `demo.py` o notebook `entrega1_mlp.ipynb`.
**Tiempo objetivo:** 9 min 30 s, dejando 30 s para preguntas.

## Antes de empezar

- Tener abierta una terminal en la raíz del proyecto con el entorno activo.
- Probar una vez `.venv/bin/python demo.py` (tarda unos segundos).
- Tener el notebook ya ejecutado; no ejecutar celdas en vivo salvo la demo.

## Reparto sugerido (2 integrantes)

| Bloque | Diapositivas | Tiempo | Quién |
|---|---|---|---|
| Introducción, datos, arquitectura y entrenamiento | 1–7 | 5:20 | Nicolás Garcés |
| Resultados, conclusiones y demo | 8–13 | 4:10 | Juan Pablo Ante |

## Guion por diapositiva

### 1. Título (0:20)
“Buenos días. Presentamos la Entrega 1 de Redes Neuronales: un Perceptrón
Multicapa para clasificar 20 Newsgroups en 20 categorías, usando Keras y
TensorFlow. El código, el informe y la demo están en el repositorio.”

### 2. Problema y objetivos (0:50)
“El problema es clasificar mensajes de Usenet en 20 temas. Es difícil por el
texto disperso, las clases que se solapan y el riesgo de fuga de información
en las cabeceras. Nuestro objetivo es entrenar y evaluar una MLP, ajustar
hiperparámetros y analizar los errores. Aclaramos que, aunque el enunciado
menciona múltiples etiquetas, los datos tienen una etiqueta por documento, así
que lo tratamos como clasificación multiclase.”

### 3. Conjunto de datos (1:00)
“Usamos la partición oficial de scikit-learn: 11.314 documentos de
entrenamiento y 7.532 de prueba, 20 clases y un desbalance moderado de 1,59 a
1. La clase más frecuente es rec.sport.hockey y la menos frecuente
talk.religion.misc. Esta diferencia justifica mirar el F1 macro además de la
exactitud.”

### 4. Preprocesamiento y protocolo (0:50)
“Eliminamos cabeceras, pies y citas para evitar que el nombre del grupo revele
la etiqueta. Usamos el texto crudo, TF-IDF con 20.000 términos adaptado solo
con entrenamiento, y una partición 90/10 estratificada. El test se toca una
sola vez. Elegimos TF-IDF porque es una representación fuerte y estable para
una MLP pura sobre un corpus pequeño.”

### 5. Arquitectura (1:00)
“La entrada es texto. La capa TextVectorization aprende el vocabulario y
calcula TF-IDF. Luego una capa densa de 256 neuronas con ReLU y dropout 0.3, y
una salida softmax de 20 clases. Son 5,13 millones de parámetros; el 99,9 %
está en la capa densa. Usamos entropía cruzada categórica y Adam.”

### 6. Condiciones de entrenamiento (0:50)
“Batch 128, hasta 25 épocas, parada temprana sobre la exactitud de validación
con paciencia 5, restaurando los mejores pesos. Semilla 42 y determinismo.
Entrenamos en CPU, en alrededor de 30 segundos.”

### 7. Ajuste de hiperparámetros (1:10)
“Comparamos 9 configuraciones con la misma partición y semilla. La tabla está
ordenada por F1 macro: el mejor es una sola capa de 256 con 0,6771 de exactitud
y 0,6663 de F1 macro. Añadir una segunda capa baja a 0,6578 y una tercera a
0,6433. El vocabulario de 10.000 términos baja a 0,6358; el balanceo de clases
no mejora el F1 macro. La tasa 5e-4 queda cerca, pero no supera. Mensaje:
la capa oculta aporta la ganancia; más profundidad sobreajusta.”

### 8. Resultados del modelo principal (1:10)
“En test obtenemos 0,6771 de exactitud, 0,6663 de F1 macro, 0,6762 de F1
ponderado y 0,8452 de top-3. La exactitud top-3 es útil si un humano revisa las
sugerencias. Hay una brecha de 7,2 puntos con validación: la validación es
pequeña, 1.132 documentos, y la distribución de test no es idéntica. Las curvas
muestran sobreajuste: entrenamiento por encima del 95 %, validación entre 72 y
75 %.”

### 9. Matriz de confusión (0:50)
“Los errores no son aleatorios. Se concentran en bloques: los grupos comp.*
comparten hardware y sistemas operativos; alt.atheism, soc.religion.christian y
talk.religion.misc comparten argumentos; y talk.politics.* comparte entidades.
También fallan documentos cortos o respuestas que citan el mensaje anterior.”

### 10. Desempeño por clase (0:50)
“Las clases más fáciles son hockey, 0,874; forsale, 0,780; y comp.windows.x,
0,750. Las más difíciles son talk.religion.misc, 0,358; talk.politics.misc,
0,462; y alt.atheism, 0,502. El modelo lineal ya era fuerte, 0,6572, y la MLP
sube el F1 macro a 0,6663.”

### 11. Conclusiones (0:50)
“La MLP con TF-IDF supera claramente al azar y al baseline lineal. Una capa
oculta y dropout 0,3 son el mejor compromiso; más capas no ayudan. El error se
concentra en clases semánticamente cercanas y en documentos sin contenido tras
el filtrado. La limitación principal es que bag-of-words ignora orden y
contexto. Por eso la Entrega 2 compara una RNN sin memoria con LSTM/GRU o
Transformer.”

### 12. Demo ejecutable (0:30)
“Ejecutamos `python demo.py`: clasifica cinco textos nuevos. Por ejemplo, un
problema de driver lo asigna a comp.windows.x con 0,41, un partido de hockey a
rec.sport.hockey con 0,98 y una misión a Marte a sci.space con 0,77. El
notebook reproduce todo el análisis.”

### 13. Bibliografía (0:20)
“Usamos el corpus de Lang, la referencia de Joachims para TF-IDF, Adam,
dropout y scikit-learn. Con esto cerramos.”

## Preguntas previsibles y respuestas

**¿Por qué no usaron embeddings?**
“Para una MLP pura y un corpus pequeño, TF-IDF es más estable y entrena en
segundos. Los embeddings sin contexto requieren más datos y se exploran en la
Entrega 2 con arquitecturas secuenciales.”

**¿Por qué una sola capa oculta?**
“Porque el barrido mostró que dos y tres capas reducen el F1 macro. La
capacidad adicional memoriza el entrenamiento; la ganancia está en la primera
capa.”

**¿El 67,7 % no es bajo?**
“Es superior al baseline lineal y muy superior al azar. En 20 Newsgroups hay
clases que se solapan y un 2,7 % de documentos quedan vacíos tras eliminar
metadatos. Además, el test oficial es más difícil que la validación. El análisis
por clase muestra que el problema no es uniforme.”

**¿No hay fuga de información?**
“No: eliminamos cabeceras, pies y citas; el vocabulario se adapta solo con
entrenamiento; y el test se evalúa una única vez.”

**¿Cómo garantizan reproducibilidad?**
“Semilla 42, determinismo de TensorFlow, misma partición y configuración
registrada en `results/runs/*/config.json`. Cualquiera puede correr
`python -m src.experiments`.”

**¿Por qué la brecha validación–test?**
“La validación tiene 1.132 documentos y alta varianza; el test tiene otra
composición de fechas e hilos. También hay sobreajuste, visible en las curvas.
Para la entrega final conviene validación cruzada y varias semillas.”
