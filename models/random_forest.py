import numpy as np
from sklearn.ensemble import RandomForestRegressor

from experiments.metrics import mean_squared_error


def tune_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    depths: list[int],
    max_features_grid: list[int],
    n_estimators: int = 300,
    tune_n_estimators: int | None = None,
    random_state: int = 0,
    n_jobs: int = -1,
) -> tuple[RandomForestRegressor, dict]:
    best_model = None
    best_score = np.inf
    best_params = {}
    tune_n_estimators = tune_n_estimators or n_estimators

    n_features = x_train.shape[1]
    valid_max_features = sorted({m for m in max_features_grid if m <= n_features})

    for depth in depths:
        for max_features in valid_max_features:
            model = RandomForestRegressor(
                n_estimators=tune_n_estimators,
                max_depth=depth,
                max_features=max_features,
                bootstrap=True,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            model.fit(x_train, y_train)
            score = mean_squared_error(y_validation, model.predict(x_validation))

            if score < best_score:
                best_score = score
                best_model = model
                best_params = {
                    "max_depth": depth,
                    "max_features": max_features,
                    "n_estimators": n_estimators,
                    "tune_n_estimators": tune_n_estimators,
                }

    if tune_n_estimators != n_estimators:
        best_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=best_params["max_depth"],
            max_features=best_params["max_features"],
            bootstrap=True,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        best_model.fit(x_train, y_train)

    return best_model, best_params

