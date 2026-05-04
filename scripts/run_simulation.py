from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.config import SimulationConfig
from simulation.data_generation import (
    flatten_for_sklearn,
    generate_panel,
    split_train_validation_test,
)


def main():
    config = SimulationConfig(
        n_assets=200,
        n_periods=180,
        n_characteristics=100,
        seed=123,
    )

    for case in ["a", "b", "c"]:
        panel = generate_panel(config, case=case)
        splits = split_train_validation_test(panel)

        x_train, y_train = flatten_for_sklearn(splits["train"])
        x_validation, y_validation = flatten_for_sklearn(splits["validation"])
        x_test, y_test = flatten_for_sklearn(splits["test"])

        print(f"case ({case})")
        print(f"  z panel shape: {panel['z'].shape}")
        print(f"  r panel shape: {panel['r'].shape}")
        print(f"  train X/y: {x_train.shape}, {y_train.shape}")
        print(f"  validation X/y: {x_validation.shape}, {y_validation.shape}")
        print(f"  test X/y: {x_test.shape}, {y_test.shape}")


if __name__ == "__main__":
    main()
