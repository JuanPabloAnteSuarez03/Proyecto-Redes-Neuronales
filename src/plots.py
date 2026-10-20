"""Gráficas del proyecto (EDA, curvas de entrenamiento y evaluación)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "colorblind"
COLORS = sns.color_palette(PALETTE)


def _save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(train: pd.DataFrame) -> Path:
    counts = train["label_name"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    counts.plot.barh(ax=ax, color=COLORS[0])
    ax.set_xlabel("Número de documentos")
    ax.set_ylabel("")
    ax.set_title("Distribución de clases en el conjunto de entrenamiento")
    for i, value in enumerate(counts):
        ax.text(value + 20, i, str(value), va="center", fontsize=8)
    ax.set_xlim(0, counts.max() * 1.12)
    return _save(fig, "eda_distribucion_clases.png")


def plot_text_lengths(train: pd.DataFrame) -> Path:
    lengths = train["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.histplot(lengths.clip(upper=1000), bins=50, ax=ax, color=COLORS[1])
    ax.axvline(lengths.median(), color=COLORS[3], linestyle="--", label=f"Mediana = {lengths.median():.0f}")
    ax.axvline(lengths.mean(), color=COLORS[2], linestyle=":", label=f"Media = {lengths.mean():.0f}")
    ax.set_xlabel("Palabras por documento (recortado a 1000)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Longitud de los documentos")
    ax.legend()
    return _save(fig, "eda_longitud_documentos.png")


def plot_training_curves(history_csv: Path, name: str = "primary") -> Path:
    history = pd.read_csv(history_csv)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(history["loss"], label="Entrenamiento", color=COLORS[0])
    axes[0].plot(history["val_loss"], label="Validación", color=COLORS[1])
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Pérdida (entropía cruzada)")
    axes[0].set_title("Pérdida")
    axes[1].plot(history["accuracy"], label="Entrenamiento", color=COLORS[0])
    axes[1].plot(history["val_accuracy"], label="Validación", color=COLORS[1])
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Exactitud")
    axes[1].set_title("Exactitud")
    for ax in axes:
        ax.legend()
    return _save(fig, f"curvas_entrenamiento_{name}.png")


def plot_accuracy_curve(history_csv: Path) -> Path:
    """Versión compacta de la curva de exactitud para diapositivas."""
    history = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    epochs = np.arange(1, len(history) + 1)
    ax.plot(epochs, history["accuracy"], marker="o", markersize=4, linewidth=2, label="Entrenamiento", color=COLORS[0])
    ax.plot(epochs, history["val_accuracy"], marker="o", markersize=4, linewidth=2, label="Validación", color=COLORS[1])
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud")
    ax.set_title("Exactitud por época")
    ax.set_ylim(0.5, 1.0)
    ax.legend()
    return _save(fig, "curvas_exactitud_primary.png")


def plot_confusion_matrix(
    cm: list[list[int]], label_names: list[str], name: str = "primary"
) -> Path:
    cm = np.asarray(cm)
    normalized = cm / cm.sum(axis=1, keepdims=True)
    short = [label.replace("soc.religion.", "soc.rel.").replace("talk.politics.", "talk.pol.") for label in label_names]
    fig, ax = plt.subplots(figsize=(10.5, 9))
    sns.heatmap(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=short,
        yticklabels=short,
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Proporción de la clase real"},
        ax=ax,
    )
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión normalizada (conjunto de prueba)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    return _save(fig, f"matriz_confusion_{name}.png")


def plot_per_class_f1(report: dict, label_names: list[str], name: str = "primary") -> Path:
    rows = [(label, report[label]["f1-score"]) for label in label_names]
    rows.sort(key=lambda item: item[1])
    labels, scores = zip(*rows)
    short = [label.replace("soc.religion.", "soc.rel.").replace("talk.politics.", "talk.pol.") for label in labels]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    bars = ax.barh(short, scores, color=COLORS[0])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1-score")
    ax.set_title("F1-score por clase (conjunto de prueba)")
    ax.axvline(np.mean(scores), color=COLORS[3], linestyle="--", label=f"Macro F1 = {np.mean(scores):.3f}")
    ax.legend(loc="lower right")
    return _save(fig, f"f1_por_clase_{name}.png")


def plot_experiments(summary: pd.DataFrame) -> Path:
    ordered = summary.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(ordered))
    height = 0.38
    ax.barh(y + height / 2, ordered["accuracy"], height=height, label="Exactitud", color=COLORS[0])
    ax.barh(y - height / 2, ordered["macro_f1"], height=height, label="Macro F1", color=COLORS[1])
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["name"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Puntaje")
    ax.set_title("Comparación de configuraciones en el conjunto de prueba")
    ax.legend(loc="lower right")
    for i, (acc, f1) in enumerate(zip(ordered["accuracy"], ordered["macro_f1"])):
        ax.text(acc + 0.01, i + height / 2, f"{acc:.3f}", va="center", fontsize=7)
        ax.text(f1 + 0.01, i - height / 2, f"{f1:.3f}", va="center", fontsize=7)
    return _save(fig, "comparacion_experimentos.png")


def plot_learning_curve_from_runs(summary: pd.DataFrame, runs_dir: Path) -> Path:
    """Superpone la exactitud de validación de todos los experimentos."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, name in enumerate(summary["name"]):
        path = runs_dir / name / "history.csv"
        if not path.exists():
            continue
        history = pd.read_csv(path)
        ax.plot(
            np.arange(1, len(history) + 1),
            history["val_accuracy"],
            marker="o",
            markersize=2.5,
            linewidth=1.4,
            color=COLORS[i % len(COLORS)],
            label=name,
        )
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud de validación")
    ax.set_title("Evolución de la exactitud de validación por configuración")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, "curvas_validacion_experimentos.png")


def top_confusions(report_cm: list[list[int]], label_names: list[str], k: int = 8) -> list[dict]:
    cm = np.asarray(report_cm)
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    flat = np.dstack(np.unravel_index(np.argsort(off_diagonal.ravel())[::-1][:k], cm.shape))[0]
    return [
        {
            "real": label_names[i],
            "predicha": label_names[j],
            "casos": int(cm[i, j]),
            "proporcion_clase_real": float(cm[i, j] / cm[i].sum()),
        }
        for i, j in flat
    ]


def generate_all_plots(
    train: pd.DataFrame,
    primary_result: dict,
    summary: pd.DataFrame,
    run_dir: Path,
) -> dict[str, str]:
    evaluation = primary_result.get("evaluation", primary_result)
    confusion = evaluation["confusion_matrix"]
    class_report = evaluation["classification_report"]
    paths = {
        "class_distribution": plot_class_distribution(train),
        "text_lengths": plot_text_lengths(train),
        "training_curves": plot_training_curves(run_dir / "history.csv"),
        "accuracy_curve": plot_accuracy_curve(run_dir / "history.csv"),
        "confusion_matrix": plot_confusion_matrix(
            confusion, primary_result["class_names"]
        ),
        "per_class_f1": plot_per_class_f1(
            class_report, primary_result["class_names"]
        ),
        "experiments": plot_experiments(summary),
        "validation_curves": plot_learning_curve_from_runs(summary, run_dir.parent),
    }
    confusions = top_confusions(confusion, primary_result["class_names"])
    (FIGURES_DIR / "confusiones_principales.json").write_text(
        json.dumps(confusions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {key: str(value) for key, value in paths.items()}
