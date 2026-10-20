"""Construcción del modelo Perceptrón Multicapa (MLP) en Keras.

La entrada son textos crudos. La primera capa es una TextVectorization que
aprende el vocabulario y calcula representaciones TF-IDF; después se apilan
capas densas completamente conectadas con dropout.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers, regularizers

from .config import ExperimentConfig


def build_vectorizer(
    x_train,
    vocab_size: int,
) -> layers.TextVectorization:
    """Crea y adapta el vectorizador TF-IDF usando SOLO el conjunto de train."""
    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        ngrams=1,
        output_mode="tf-idf",
        pad_to_max_tokens=True,
        name="tfidf",
    )
    vectorizer.adapt(tf.constant(x_train.to_numpy(), dtype=tf.string))
    return vectorizer


def build_model(
    vectorizer: layers.TextVectorization,
    n_classes: int,
    config: ExperimentConfig,
) -> keras.Model:
    """MLP: TF-IDF -> [Dense + Dropout] * k -> softmax."""
    inputs = keras.Input(shape=(1,), dtype=tf.string, name="texto")
    x = vectorizer(inputs)

    regularizer = regularizers.l2(config.l2) if config.l2 > 0 else None
    for i, (units, dropout) in enumerate(
        zip(config.hidden_units, config.dropout_rates), start=1
    ):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizer,
            name=f"densa_{i}_{units}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="salida")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=config.name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )
    return model
