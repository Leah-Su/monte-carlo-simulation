from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.metrics import predictive_r2
from models.linear_models import (
    fit_ols3,
    fit_ols3_huber,
    predict_ols3,
    predict_ols3_huber,
    tune_elastic_net,
    tune_elastic_net_huber,
)
from models.neural_networks import tune_neural_network
from models.random_forest import tune_random_forest
from simulation.config import SimulationConfig
from simulation.data_generation import (
    flatten_for_sklearn,
    generate_panel,
    split_train_validation_test,
)


@dataclass(frozen=True)
class ExperimentConfig:
    cases: list[str]
    pc_values: list[int]
    repetitions: int
    models: list[str]
    seed: int = 123

    enet_alphas: tuple[float, ...] = tuple(np.logspace(-4, -1, 10))
    rf_depths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    rf_max_features: tuple[int, ...] = (3, 5, 10, 20, 30, 50, 100)
    rf_n_estimators: int = 300
    rf_tune_n_estimators: int | None = None

    nn_learning_rates: tuple[float, ...] = (0.001, 0.01)
    nn_l1_penalties: tuple[float, ...] = tuple(np.logspace(-5, -3, 5))
    nn_batch_size: int = 10_000
    nn_epochs: int = 100
    nn_patience: int = 5
    nn_ensemble_size: int = 10
    nn_tune_ensemble_size: int | None = None
    nn_device: str | None = None


def _evaluate_predictions(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray,
) -> tuple[float, float]:
    train_mean = float(y_train.mean())
    is_r2 = predictive_r2(y_train, y_train_pred, train_mean)
    oos_r2 = predictive_r2(y_test, y_test_pred, train_mean)
    return is_r2, oos_r2


def run_one_repetition(
    case: str,
    pc: int,
    repetition: int,
    experiment: ExperimentConfig,
) -> list[dict]:
    simulation_config = SimulationConfig(
        n_assets=200,
        n_periods=180,
        n_characteristics=pc,
        seed=experiment.seed + 10_000 * repetition + 100 * pc,
    )
    panel = generate_panel(simulation_config, case=case)
    splits = split_train_validation_test(panel)

    x_train, y_train = flatten_for_sklearn(splits["train"])
    x_validation, y_validation = flatten_for_sklearn(splits["validation"])
    x_test, y_test = flatten_for_sklearn(splits["test"])

    rows = []

    if "OLS-3" in experiment.models:
        started = time.time()
        model = fit_ols3(splits["train"], case)
        y_train_pred = predict_ols3(model, splits["train"], case)
        y_test_pred = predict_ols3(model, splits["test"], case)
        is_r2, oos_r2 = _evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred)
        rows.append(
            {
                "case": case,
                "pc": pc,
                "repetition": repetition,
                "model": "OLS-3",
                "is_r2": is_r2,
                "oos_r2": oos_r2,
                "best_params": "{}",
                "seconds": time.time() - started,
            }
        )

    if "OLS-3-H" in experiment.models:
        started = time.time()
        model = fit_ols3_huber(
            splits["train"],
            case,
            random_state=experiment.seed + repetition,
        )
        y_train_pred = predict_ols3_huber(model, splits["train"], case)
        y_test_pred = predict_ols3_huber(model, splits["test"], case)
        is_r2, oos_r2 = _evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred)
        rows.append(
            {
                "case": case,
                "pc": pc,
                "repetition": repetition,
                "model": "OLS-3-H",
                "is_r2": is_r2,
                "oos_r2": oos_r2,
                "best_params": "{'huber_quantile': 0.999}",
                "seconds": time.time() - started,
            }
        )

    if "ENet" in experiment.models:
        started = time.time()
        model, params = tune_elastic_net(
            x_train,
            y_train,
            x_validation,
            y_validation,
            alphas=list(experiment.enet_alphas),
            l1_ratio=0.5,
            random_state=experiment.seed + repetition,
        )
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        is_r2, oos_r2 = _evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred)
        rows.append(
            {
                "case": case,
                "pc": pc,
                "repetition": repetition,
                "model": "ENet",
                "is_r2": is_r2,
                "oos_r2": oos_r2,
                "best_params": str(params),
                "seconds": time.time() - started,
            }
        )

    if "ENet-H" in experiment.models:
        started = time.time()
        model, params = tune_elastic_net_huber(
            x_train,
            y_train,
            x_validation,
            y_validation,
            alphas=list(experiment.enet_alphas),
            l1_ratio=0.5,
            random_state=experiment.seed + repetition,
        )
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        is_r2, oos_r2 = _evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred)
        rows.append(
            {
                "case": case,
                "pc": pc,
                "repetition": repetition,
                "model": "ENet-H",
                "is_r2": is_r2,
                "oos_r2": oos_r2,
                "best_params": str(params),
                "seconds": time.time() - started,
            }
        )

    if "RF" in experiment.models:
        started = time.time()
        model, params = tune_random_forest(
            x_train,
            y_train,
            x_validation,
            y_validation,
            depths=list(experiment.rf_depths),
            max_features_grid=list(experiment.rf_max_features),
            n_estimators=experiment.rf_n_estimators,
            tune_n_estimators=experiment.rf_tune_n_estimators,
            random_state=experiment.seed + repetition,
        )
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        is_r2, oos_r2 = _evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred)
        rows.append(
            {
                "case": case,
                "pc": pc,
                "repetition": repetition,
                "model": "RF",
                "is_r2": is_r2,
                "oos_r2": oos_r2,
                "best_params": str(params),
                "seconds": time.time() - started,
            }
        )

    for model_name in ["NN1", "NN2", "NN3"]:
        if model_name not in experiment.models:
            continue

        started = time.time()
        model, params = tune_neural_network(
            model_name,
            x_train,
            y_train,
            x_validation,
            y_validation,
            learning_rates=list(experiment.nn_learning_rates),
            l1_penalties=list(experiment.nn_l1_penalties),
            batch_size=experiment.nn_batch_size,
            max_epochs=experiment.nn_epochs,
            patience=experiment.nn_patience,
            ensemble_size=experiment.nn_ensemble_size,
            tune_ensemble_size=experiment.nn_tune_ensemble_size,
            base_seed=experiment.seed + 1000 * repetition,
            device=experiment.nn_device,
        )
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        is_r2, oos_r2 = _evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred)
        rows.append(
            {
                "case": case,
                "pc": pc,
                "repetition": repetition,
                "model": model_name,
                "is_r2": is_r2,
                "oos_r2": oos_r2,
                "best_params": str(params),
                "seconds": time.time() - started,
            }
        )

    return rows


def _completed_repetitions(results: pd.DataFrame, experiment: ExperimentConfig) -> set[tuple[str, int, int]]:
    if results.empty:
        return set()

    required_models = set(experiment.models)
    completed = set()
    for (case, pc, repetition), group in results.groupby(["case", "pc", "repetition"]):
        if required_models.issubset(set(group["model"])):
            completed.add((case, int(pc), int(repetition)))
    return completed


def run_monte_carlo(
    experiment: ExperimentConfig,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        rows = existing.to_dict("records")
        completed = _completed_repetitions(existing, experiment)
        print(f"Resuming from {checkpoint_path}; found {len(completed)} completed repetitions.")
    else:
        rows = []
        completed = set()

    for case in experiment.cases:
        for pc in experiment.pc_values:
            for repetition in range(experiment.repetitions):
                key = (case, pc, repetition)
                if key in completed:
                    print(f"Skipping case={case}, Pc={pc}, repetition={repetition + 1}/{experiment.repetitions}")
                    continue

                print(f"Running case={case}, Pc={pc}, repetition={repetition + 1}/{experiment.repetitions}")
                rows.extend(run_one_repetition(case, pc, repetition, experiment))

                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["case", "pc", "model"], as_index=False)
        .agg(
            is_r2_mean=("is_r2", "mean"),
            oos_r2_mean=("oos_r2", "mean"),
            is_r2_std=("is_r2", "std"),
            oos_r2_std=("oos_r2", "std"),
            avg_seconds=("seconds", "mean"),
        )
        .sort_values(["case", "pc", "model"])
    )
    summary["is_r2_percent"] = 100.0 * summary["is_r2_mean"]
    summary["oos_r2_percent"] = 100.0 * summary["oos_r2_mean"]
    return summary
