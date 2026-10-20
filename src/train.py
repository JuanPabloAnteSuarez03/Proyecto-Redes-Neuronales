"""Entrenamiento y evaluación de un experimento de la MLP."""
from __future__ import annotations

import json
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    ExperimentConfig,
    MODELS_DIR,
    PRIMARY_CONFIG,
    RUNS_DIR,
    SEED,
)
from .data import load_splits
from .model import build_model, build_vectorizer


def set_seed(seed: int = SEED) -> None:
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def _class_weights(y_train: pd.Series, classes: np.ndarray) -> dict[int, float] | None:
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train.to_numpy()
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


def evaluate_model(
    model: keras.Model,
    x_test: pd.Series,
    y_test: pd.Series,
    label_names: list[str],
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Calcula métricas y matrices de confusión sobre el conjunto de prueba."""
    probabilities = model.predict(
        tf.constant(x_test.to_numpy(), dtype=tf.string), batch_size=256, verbose=0
    )
    y_pred = probabilities.argmax(axis=1)
    y_true = y_test.to_numpy()

    labels = list(range(len(label_names)))
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        zero_division=0,
        digits=4,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    top3 = float(
        np.mean([y_true[i] in np.argsort(probabilities[i])[-3:] for i in range(len(y_true))])
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "top3_accuracy": top3,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "report_text": report_text,
    }
    return metrics, y_pred, probabilities


def run_experiment(
    config: ExperimentConfig,
    save_artifacts: bool = True,
    verbose: int = 2,
) -> dict:
    """Entrena una configuración y devuelve/guarda sus resultados."""
    set_seed(config.seed)
    x_train, x_val, x_test, y_train, y_val, y_test, classes, label_names = load_splits(
        validation_size=config.validation_size, seed=config.seed
    )

    vectorizer = build_vectorizer(x_train, config.vocab_size)
    model = build_model(vectorizer, n_classes=len(label_names), config=config)

    run_dir: Path = RUNS_DIR / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.keras"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.patience,
            min_delta=config.min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]

    class_weight = _class_weights(y_train, classes) if config.balanced_class_weight else None

    started = time.time()
    history = model.fit(
        tf.constant(x_train.to_numpy(), dtype=tf.string),
        y_train.to_numpy(),
        validation_data=(
            tf.constant(x_val.to_numpy(), dtype=tf.string),
            y_val.to_numpy(),
        ),
        epochs=config.max_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose,
    )
    elapsed = time.time() - started

    # Se recupera el mejor modelo guardado por exactitud de validación.
    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)
        # El checkpoint es temporal; solo se conserva el modelo principal.
        checkpoint_path.unlink()

    metrics, y_pred, probabilities = evaluate_model(model, x_test, y_test, label_names)
    history_df = pd.DataFrame(history.history)
    best_epoch = int(history_df["val_accuracy"].idxmax() + 1)

    result = {
        "name": config.name,
        "config": config.to_dict(),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "n_classes": len(label_names),
        "class_names": label_names,
        "epochs_run": int(len(history_df)),
        "best_epoch": best_epoch,
        "train_time_seconds": float(elapsed),
        "best_val_loss": float(history_df["val_loss"].min()),
        "best_val_accuracy": float(history_df["val_accuracy"].max()),
        "final_train_accuracy": float(history_df["accuracy"].iloc[-1]),
        "n_parameters": int(model.count_params()),
        **{
            k: v
            for k, v in metrics.items()
            if k not in {"classification_report", "confusion_matrix", "report_text"}
        },
        "evaluation": metrics,
    }

    if save_artifacts:
        history_df.to_csv(run_dir / "history.csv", index=False)
        (run_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "classification_report.txt").write_text(
            metrics["report_text"], encoding="utf-8"
        )
        np.save(run_dir / "test_probabilities.npy", probabilities)
        np.save(run_dir / "test_predictions.npy", y_pred)
        (run_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Solo se conserva el modelo final para la demo; los modelos de las
        # variantes ocuparían cientos de MB y no son necesarios para el informe.
        if config.name == PRIMARY_CONFIG.name:
            # Se descarta el estado del optimizador: solo se necesita inferencia.
            model.optimizer = None
            model.save(MODELS_DIR / f"{config.name}.keras")

    return result


def summarize_experiments(results: list[dict]) -> pd.DataFrame:
    columns = [
        "name",
        "vocab_size",
        "hidden_units",
        "dropout_rates",
        "balanced_class_weight",
        "n_parameters",
        "epochs_run",
        "best_epoch",
        "train_time_seconds",
        "best_val_accuracy",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top3_accuracy",
    ]
    rows = []
    for result in results:
        cfg = result["config"]
        rows.append(
            {
                "name": result["name"],
                "vocab_size": cfg["vocab_size"],
                "hidden_units": "-".join(map(str, cfg["hidden_units"])) or "sin capas ocultas",
                "dropout_rates": "-".join(map(str, cfg["dropout_rates"])) or "-",
                "balanced_class_weight": cfg["balanced_class_weight"],
                "n_parameters": result["n_parameters"],
                "epochs_run": result["epochs_run"],
                "best_epoch": result["best_epoch"],
                "train_time_seconds": round(result["train_time_seconds"], 1),
                "best_val_accuracy": result["best_val_accuracy"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "top3_accuracy": result["top3_accuracy"],
            }
        )
    return pd.DataFrame(rows, columns=columns)
