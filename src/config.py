"""Configuración central del proyecto de clasificación con 20 Newsgroups."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

for _directory in (DATA_DIR, RESULTS_DIR, RUNS_DIR, FIGURES_DIR, MODELS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

SEED = 42


# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Hiperparámetros de un experimento de la MLP."""

    name: str = "mlp_2x512_256"
    vocab_size: int = 20_000
    hidden_units: tuple[int, ...] = (512, 256)
    dropout_rates: tuple[float, ...] = (0.4, 0.3)
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 25
    patience: int = 5
    min_delta: float = 1e-4
    l2: float = 0.0
    balanced_class_weight: bool = False
    validation_size: float = 0.1
    seed: int = SEED

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.hidden_units and len(self.hidden_units) != len(self.dropout_rates):
            raise ValueError(
                "hidden_units y dropout_rates deben tener la misma longitud "
                f"(recibido {len(self.hidden_units)} y {len(self.dropout_rates)})."
            )


# La configuración con mejor F1 macro en el barrido fue una sola capa oculta
# de 256 neuronas (ver results/experimentos.csv). Es también la más simple.
PRIMARY_CONFIG = ExperimentConfig(
    name="mlp_1x256",
    hidden_units=(256,),
    dropout_rates=(0.3,),
)

# Experimentos de ajuste de hiperparámetros. Se entrenan con el mismo
# protocolo (misma partición, semilla y criterio de parada temprana) para que
# las comparaciones sean justas.
EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(name="lineal_softmax", hidden_units=(), dropout_rates=()),
    PRIMARY_CONFIG,
    ExperimentConfig(name="mlp_2x512_256", hidden_units=(512, 256), dropout_rates=(0.4, 0.3)),
    ExperimentConfig(
        name="mlp_2x512_256_dropout050",
        hidden_units=(512, 256),
        dropout_rates=(0.5, 0.4),
    ),
    ExperimentConfig(
        name="mlp_3x512_256_128",
        hidden_units=(512, 256, 128),
        dropout_rates=(0.4, 0.3, 0.2),
    ),
    ExperimentConfig(name="mlp_2x512_256_vocab10k", vocab_size=10_000),
    ExperimentConfig(
        name="mlp_2x512_256_balanceado",
        balanced_class_weight=True,
    ),
    ExperimentConfig(name="mlp_2x512_256_lr5e4", learning_rate=5e-4),
    ExperimentConfig(name="mlp_2x512_256_l2", l2=1e-4),
]
