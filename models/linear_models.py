import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from experiments.metrics import mean_squared_error


class ScaledTargetRegressor:
    """A tiny wrapper that standardizes y during training and unscales predictions."""

    def __init__(self, model, y_mean: float, y_std: float):
        self.model = model
        self.y_mean = y_mean
        self.y_std = y_std

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x) * self.y_std + self.y_mean


def _scaled_target(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std == 0.0:
        y_std = 1.0
    return ((y - y_mean) / y_std), y_mean, y_std


def _huber_epsilon(y_scaled: np.ndarray, quantile: float = 0.999) -> float:
    """
    Huber threshold based on the 99.9% quantile, matching Table A.5's rule.
    The lower bound avoids a degenerate threshold in very quiet samples.
    """
    return max(float(np.quantile(np.abs(y_scaled - y_scaled.mean()), quantile)), 1e-6)


def ols3_features(panel_sample: dict[str, np.ndarray], case: str) -> np.ndarray:
    """
    Build OLS-3 features.

    In the empirical paper, OLS-3 preselects size, book-to-market, and momentum
    as the only covariates. In this simulation, we map those three covariates to
    the first three raw characteristics: c1, c2, and c3.
    """
    c = panel_sample["c"]
    c1 = c[:, :, 0]
    c2 = c[:, :, 1]
    c3 = c[:, :, 2]

    if case not in {"a", "b", "c"}:
        raise ValueError("case must be one of {'a', 'b', 'c'}.")

    features = np.stack([c1, c2, c3], axis=2)
    return features.reshape(-1, features.shape[2])


def fit_ols3(train: dict[str, np.ndarray], case: str) -> LinearRegression:
    x_train = ols3_features(train, case)
    y_train = train["r"].reshape(-1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def predict_ols3(model: LinearRegression, sample: dict[str, np.ndarray], case: str) -> np.ndarray:
    return model.predict(ols3_features(sample, case))


def fit_ols3_huber(
    train: dict[str, np.ndarray],
    case: str,
    random_state: int = 0,
) -> ScaledTargetRegressor:
    x_train = ols3_features(train, case)
    y_train = train["r"].reshape(-1)
    y_train_scaled, y_mean, y_std = _scaled_target(y_train)
    epsilon = _huber_epsilon(y_train_scaled)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "huber",
                SGDRegressor(
                    loss="huber",
                    penalty=None,
                    epsilon=epsilon,
                    alpha=0.0,
                    max_iter=20_000,
                    tol=1e-5,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train_scaled)
    return ScaledTargetRegressor(model=model, y_mean=y_mean, y_std=y_std)


def predict_ols3_huber(
    model: ScaledTargetRegressor,
    sample: dict[str, np.ndarray],
    case: str,
) -> np.ndarray:
    return model.predict(ols3_features(sample, case))


def tune_elastic_net(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    alphas: list[float],
    l1_ratio: float = 0.5,
    random_state: int = 0,
) -> tuple[Pipeline, dict]:
    best_model = None
    best_score = np.inf
    best_params = {}

    for alpha in alphas:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "enet",
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=20_000,
                        tol=1e-5,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        score = mean_squared_error(y_validation, model.predict(x_validation))

        if score < best_score:
            best_score = score
            best_model = model
            best_params = {"alpha": alpha, "l1_ratio": l1_ratio}

    return best_model, best_params


def tune_elastic_net_huber(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    alphas: list[float],
    l1_ratio: float = 0.5,
    random_state: int = 0,
) -> tuple[ScaledTargetRegressor, dict]:
    y_train_scaled, y_mean, y_std = _scaled_target(y_train)
    y_validation_scaled = (y_validation - y_mean) / y_std
    epsilon = _huber_epsilon(y_train_scaled)

    best_model = None
    best_score = np.inf
    best_params = {}

    for alpha in alphas:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "enet_huber",
                    SGDRegressor(
                        loss="huber",
                        penalty="elasticnet",
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        epsilon=epsilon,
                        max_iter=20_000,
                        tol=1e-5,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train_scaled)
        score = mean_squared_error(y_validation_scaled, model.predict(x_validation))

        if score < best_score:
            best_score = score
            best_model = model
            best_params = {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "huber_epsilon": epsilon,
            }

    wrapped = ScaledTargetRegressor(model=best_model, y_mean=y_mean, y_std=y_std)
    return wrapped, best_params
