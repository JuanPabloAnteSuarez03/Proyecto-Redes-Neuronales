"""Ejecuta todos los experimentos de ajuste de hiperparámetros."""
from __future__ import annotations

import argparse
import json
import time
import traceback

import pandas as pd

from .config import EXPERIMENTS, PRIMARY_CONFIG, RESULTS_DIR, ExperimentConfig
from .data import load_dataframe
from .plots import generate_all_plots
from .train import run_experiment, summarize_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimentos de la Entrega 1")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Nombres de experimentos a ejecutar (por defecto, todos).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reutiliza result.json de experimentos ya ejecutados.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs: list[ExperimentConfig] = [
        cfg for cfg in EXPERIMENTS if args.only is None or cfg.name in args.only
    ]
    if not configs:
        raise SystemExit("No se seleccionó ningún experimento válido.")

    print("=" * 78)
    print(f"Experimentos a ejecutar: {[cfg.name for cfg in configs]}")
    print("=" * 78)

    results = []
    for i, config in enumerate(configs, start=1):
        result_path = RESULTS_DIR / "runs" / config.name / "result.json"
        if args.skip_existing and result_path.exists():
            print(f"[{i}/{len(configs)}] Reutilizando {config.name}")
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        print(f"\n[{i}/{len(configs)}] Entrenando {config.name} ...")
        started = time.time()
        try:
            result = run_experiment(config)
            results.append(result)
            print(
                f"  accuracy={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}  "
                f"épocas={result['epochs_run']}  tiempo={time.time() - started:.1f}s"
            )
        except Exception:  # pragma: no cover - utilidad para ejecución por lotes
            print(f"  ERROR en {config.name}:")
            traceback.print_exc()

    summary = summarize_experiments(results)
    summary.to_csv(RESULTS_DIR / "experimentos.csv", index=False)
    (RESULTS_DIR / "experimentos.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\nResumen de experimentos:\n")
    print(summary.drop(columns=["n_parameters"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    primary = next(
        (r for r in results if r["name"] == PRIMARY_CONFIG.name),
        max(results, key=lambda r: r["macro_f1"]),
    )
    train_df, _, _ = load_dataframe()
    paths = generate_all_plots(
        train_df,
        primary,
        summary,
        RESULTS_DIR / "runs" / primary["name"],
    )
    print("\nFiguras generadas:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")
    print(f"\nResultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
