"""Carga y preparación del conjunto de datos 20 Newsgroups.

Se usa la versión de scikit-learn, que descarga la colección original de UCI.
Para clasificar por el contenido del mensaje (y no por cabeceras que delatan
el grupo de noticias) se eliminan cabeceras, pies de página y citas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, SEED

TRAIN_CSV = DATA_DIR / "20newsgroups_train.csv"
TEST_CSV = DATA_DIR / "20newsgroups_test.csv"
REMOVE = ("headers", "footers", "quotes")


def _fetch_to_frame(subset: str) -> pd.DataFrame:
    bunch = fetch_20newsgroups(
        subset=subset,
        remove=REMOVE,
        download_if_missing=True,
        random_state=SEED,
    )
    return pd.DataFrame(
        {
            "text": bunch.data,
            "label": bunch.target,
            "label_name": [bunch.target_names[i] for i in bunch.target],
        }
    )


def load_dataframe() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Devuelve (train, test, nombres_de_clases) y cachea los CSV."""
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        # keep_default_na=False evita que textos como "NA" se conviertan en NaN.
        train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
        test = pd.read_csv(TEST_CSV, keep_default_na=False)
        label_names = sorted(train["label_name"].unique().tolist())
    else:
        train = _fetch_to_frame("train")
        test = _fetch_to_frame("test")
        train.to_csv(TRAIN_CSV, index=False)
        test.to_csv(TEST_CSV, index=False)
        label_names = sorted(train["label_name"].unique().tolist())

    train = train.dropna(subset=["text"]).reset_index(drop=True)
    test = test.dropna(subset=["text"]).reset_index(drop=True)
    return train, test, label_names


def load_splits(
    validation_size: float = 0.1,
    seed: int = SEED,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, np.ndarray, list[str]]:
    """Devuelve x_train, x_val, x_test, y_train, y_val, y_test, clases, nombres."""
    train, test, label_names = load_dataframe()

    x_train, x_val, y_train, y_val = train_test_split(
        train["text"],
        train["label"].to_numpy(),
        test_size=validation_size,
        random_state=seed,
        stratify=train["label"].to_numpy(),
    )
    x_test = test["text"]
    y_test = test["label"].to_numpy()
    classes = np.unique(y_train)
    return (
        x_train.reset_index(drop=True),
        x_val.reset_index(drop=True),
        x_test.reset_index(drop=True),
        pd.Series(y_train),
        pd.Series(y_val),
        pd.Series(y_test),
        classes,
        label_names,
    )


def dataframe_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    lengths = train["text"].str.split().str.len()
    return {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_classes": int(train["label"].nunique()),
        "n_words_total": int(lengths.sum()),
        "words_mean": float(lengths.mean()),
        "words_median": float(lengths.median()),
        "words_p95": float(lengths.quantile(0.95)),
        "words_max": int(lengths.max()),
        "empty_docs": int((lengths == 0).sum()),
        "class_counts": train["label_name"].value_counts().sort_index().to_dict(),
    }
