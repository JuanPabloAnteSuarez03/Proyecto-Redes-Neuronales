"""Genera tablas y macros LaTeX a partir de los resultados de los experimentos.

Uso: python -m src.build_docs
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR, ROOT
from .data import load_dataframe, dataframe_summary

REPORT_DIR = ROOT / "informe"
PRESENTATION_DIR = ROOT / "presentacion"
PRIMARY_NAME = "mlp_1x256"

DISPLAY_NAMES = {
    "lineal_softmax": "Lineal (sin capas ocultas)",
    "mlp_1x256": "MLP 1$\\times$256 (principal)",
    "mlp_2x512_256": "MLP 2$\\times$512--256",
    "mlp_2x512_256_dropout050": "MLP 2$\\times$512--256, dropout 0.5--0.4",
    "mlp_3x512_256_128": "MLP 3$\\times$512--256--128",
    "mlp_2x512_256_vocab10k": "MLP 2$\\times$512--256, vocab. 10k",
    "mlp_2x512_256_balanceado": "MLP 2$\\times$512--256, balanceado",
    "mlp_2x512_256_lr5e4": "MLP 2$\\times$512--256, lr 5e-4",
    "mlp_2x512_256_l2": "MLP 2$\\times$512--256, L2 1e-4",
}


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    experiments = pd.read_csv(RESULTS_DIR / "experimentos.csv")
    primary_dir = RESULTS_DIR / "runs" / PRIMARY_NAME
    result = json.loads((primary_dir / "result.json").read_text(encoding="utf-8"))
    metrics = json.loads((primary_dir / "metrics.json").read_text(encoding="utf-8"))

    train_df, test_df = load_dataframe()[:2]
    summary = dataframe_summary(train_df, test_df)

    best = experiments.sort_values("macro_f1", ascending=False).iloc[0]
    linear = experiments[experiments["name"] == "lineal_softmax"].iloc[0]
    one_hidden = experiments[experiments["name"] == "mlp_1x256"].iloc[0]

    def value(name: str, column: str) -> float:
        row = experiments[experiments["name"] == name]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    class_counts = summary["class_counts"]
    imbalance = max(class_counts.values()) / min(class_counts.values())
    max_class = max(class_counts, key=class_counts.get)
    min_class = min(class_counts, key=class_counts.get)

    macros = {
        "Accuracy": fmt(result["accuracy"]),
        "MacroPrecision": fmt(result["macro_precision"]),
        "MacroRecall": fmt(result["macro_recall"]),
        "MacroF": fmt(result["macro_f1"]),
        "WeightedF": fmt(result["weighted_f1"]),
        "TopThree": fmt(result["top3_accuracy"]),
        "BestValAcc": fmt(result["best_val_accuracy"]),
        "ValTestGap": fmt((result["best_val_accuracy"] - result["accuracy"]) * 100, 2),
        "EpochsRun": str(result["epochs_run"]),
        "BestEpoch": str(result["best_epoch"]),
        "NumParams": f"{result['n_parameters'] / 1e6:.2f}",
        "ParamsExact": f"{result['n_parameters']:,}".replace(",", "{,}"),
        "TrainTime": fmt(result["train_time_seconds"], 1),
        "TrainTimeMin": fmt(result["train_time_seconds"] / 60, 1),
        "Vocab": f"{result['config']['vocab_size']:,}".replace(",", "{,}"),
        "NTrain": f"{result['n_train']:,}".replace(",", "{,}"),
        "NVal": f"{result['n_val']:,}".replace(",", "{,}"),
        "NTest": f"{result['n_test']:,}".replace(",", "{,}"),
        "NTrainTotal": f"{summary['n_train']:,}".replace(",", "{,}"),
        "NTestTotal": f"{summary['n_test']:,}".replace(",", "{,}"),
        "WordsMean": fmt(summary["words_mean"], 1),
        "WordsMedian": fmt(summary["words_median"], 0),
        "WordsPNoventaCinco": fmt(summary["words_p95"], 0),
        "WordsMax": f"{summary['words_max']:,}".replace(",", "{,}"),
        "EmptyDocs": f"{summary['empty_docs']:,}".replace(",", "{,}"),
        "Imbalance": fmt(imbalance, 2),
        "MaxClassName": max_class.replace("_", r"\_"),
        "MaxClassCount": f"{class_counts[max_class]:,}".replace(",", "{,}"),
        "MinClassName": min_class.replace("_", r"\_"),
        "MinClassCount": f"{class_counts[min_class]:,}".replace(",", "{,}"),
        "NumExperiments": str(len(experiments)),
        "PrimaryName": DISPLAY_NAMES.get(PRIMARY_NAME, PRIMARY_NAME),
        "BestName": DISPLAY_NAMES.get(best["name"], best["name"]),
        "BestAcc": fmt(best["accuracy"]),
        "BestF": fmt(best["macro_f1"]),
        "LinearAcc": fmt(linear["accuracy"]),
        "LinearF": fmt(linear["macro_f1"]),
        "OneHiddenAcc": fmt(one_hidden["accuracy"]),
        "OneHiddenF": fmt(one_hidden["macro_f1"]),
    }
    for short, name in [
        ("Dropout", "mlp_2x512_256_dropout050"),
        ("Deep", "mlp_3x512_256_128"),
        ("VocabTen", "mlp_2x512_256_vocab10k"),
        ("Balanced", "mlp_2x512_256_balanceado"),
        ("Lr", "mlp_2x512_256_lr5e4"),
        ("Ltwo", "mlp_2x512_256_l2"),
    ]:
        row = experiments[experiments["name"] == name]
        macros[f"{short}Acc"] = fmt(value(name, "accuracy"))
        macros[f"{short}F"] = fmt(value(name, "macro_f1"))
        macros[f"{short}Epochs"] = str(int(row["epochs_run"].iloc[0])) if len(row) else "---"
        macros[f"{short}Params"] = (
            f"{row['n_parameters'].iloc[0] / 1e6:.1f}M" if len(row) else "---"
        )
    macros["LinearEpochs"] = str(int(linear["epochs_run"]))
    macros["LinearParams"] = f"{linear['n_parameters'] / 1e6:.1f}M"
    macros["OneHiddenEpochs"] = str(int(one_hidden["epochs_run"]))
    macros["OneHiddenParams"] = f"{one_hidden['n_parameters'] / 1e6:.1f}M"
    macros["PrimaryParams"] = f"{result['n_parameters'] / 1e6:.1f}M"
    two_layer = experiments[experiments["name"] == "mlp_2x512_256"]
    macros["TwoLayerAcc"] = fmt(value("mlp_2x512_256", "accuracy"))
    macros["TwoLayerF"] = fmt(value("mlp_2x512_256", "macro_f1"))
    macros["TwoLayerEpochs"] = str(int(two_layer["epochs_run"].iloc[0])) if len(two_layer) else "---"
    macros["TwoLayerParams"] = (
        f"{two_layer['n_parameters'].iloc[0] / 1e6:.1f}M" if len(two_layer) else "---"
    )

    macro_lines = [
        "% Archivo generado automáticamente por src/build_docs.py. No editar a mano.",
    ]
    for key, val in macros.items():
        macro_lines.append(f"\\newcommand{{\\{key}}}{{{val}}}")
    (REPORT_DIR / "resultados.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Tabla de experimentos
    # ------------------------------------------------------------------
    ordered = experiments.copy()
    ordered["display"] = ordered["name"].map(DISPLAY_NAMES).fillna(ordered["name"])
    ordered = ordered.sort_values("macro_f1", ascending=False)

    rows = []
    for _, row in ordered.iterrows():
        layers = row["hidden_units"].replace("sin capas ocultas", "---")
        name = row["display"]
        if row["name"] == best["name"]:
            name = f"\\textbf{{{name}}}"
        rows.append(
            " & ".join(
                [
                    name,
                    layers,
                    row["dropout_rates"],
                    f"{int(row['vocab_size']) // 1000}k",
                    "Sí" if bool(row["balanced_class_weight"]) else "No",
                    f"{row['n_parameters'] / 1e6:.1f}M",
                    str(int(row["epochs_run"])),
                    fmt(row["accuracy"]),
                    fmt(row["macro_f1"]),
                    fmt(row["top3_accuracy"]),
                ]
            )
            + r" \\"
        )
    table = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Resultados de las configuraciones evaluadas en el conjunto de prueba, ordenadas por F1 macro.}",
            r"\label{tab:experimentos}",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lccccccccc}",
            r"\toprule",
            r"Configuración & Capas & Drop. & Vocab. & Bal. & Parám. & Ép. & Exact. & F1 & Top-3 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (REPORT_DIR / "tabla_experimentos.tex").write_text(table + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Tabla por clase (modelo principal)
    # ------------------------------------------------------------------
    report = metrics["classification_report"]
    class_rows = []
    for label in result["class_names"]:
        info = report[label]
        pretty = label.replace("_", r"\_")
        class_rows.append(
            f"{pretty} & {fmt(info['precision'], 3)} & {fmt(info['recall'], 3)} & "
            f"{fmt(info['f1-score'], 3)} & {int(info['support'])} \\\\"
        )
    per_class = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Métricas por clase del modelo principal en el conjunto de prueba.}",
            r"\label{tab:por-clase}",
            r"\small",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Clase & Precisión & Recall & F1 & Soporte \\",
            r"\midrule",
            *class_rows,
            r"\midrule",
            f"Macro promedio & {fmt(metrics['macro_precision'], 3)} & {fmt(metrics['macro_recall'], 3)} & "
            f"{fmt(metrics['macro_f1'], 3)} & {result['n_test']:,} \\\\".replace(",", "{,}"),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (REPORT_DIR / "tabla_por_clase.tex").write_text(per_class + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Confusiones principales
    # ------------------------------------------------------------------
    confusions = json.loads(
        (RESULTS_DIR / "figures" / "confusiones_principales.json").read_text(encoding="utf-8")
    )
    confusion_rows = []
    for item in confusions:
        confusion_rows.append(
            f"{item['real'].replace('_', r'\_')} & {item['predicha'].replace('_', r'\_')} & "
            f"{item['casos']} & {item['proporcion_clase_real'] * 100:.1f}\\% \\\\"
        )
    confusion_table = "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Confusiones más frecuentes del modelo principal.}",
            r"\label{tab:confusiones}",
            r"\small",
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Clase real & Clase predicha & Casos & \% de la clase real \\",
            r"\midrule",
            *confusion_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    (REPORT_DIR / "tabla_confusiones.tex").write_text(confusion_table + "\n", encoding="utf-8")

    print("Archivos generados en informe/:")
    for name in ["resultados.tex", "tabla_experimentos.tex", "tabla_por_clase.tex", "tabla_confusiones.tex"]:
        print(" -", REPORT_DIR / name)
    print("\nMacros principales:")
    for key in ["Accuracy", "MacroF", "TopThree", "BestName", "BestF", "NumParams", "TrainTime"]:
        print(f"  {key} = {macros[key]}")


if __name__ == "__main__":
    main()
