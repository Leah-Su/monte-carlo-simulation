import numpy as np

from models.linear_models import tune_elastic_net_huber
from models.neural_networks import NeuralNetEnsemble, tune_neural_network


RIN_TO_NN = {
    "ENet-RIN1": "NN1",
    "ENet-RIN2": "NN2",
    "ENet-RIN3": "NN3",
}


class ENetResidualNet:
    def __init__(self, enet_model, residual_model: NeuralNetEnsemble):
        self.enet_model = enet_model
        self.residual_model = residual_model

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.enet_model.predict(x) + self.residual_model.predict(x)


def tune_enet_residual_network(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    enet_alphas: list[float],
    nn_learning_rates: list[float],
    nn_l1_penalties: list[float],
    enet_l1_ratio: float = 0.5,
    nn_batch_size: int = 10_000,
    nn_epochs: int = 100,
    nn_patience: int = 5,
    nn_ensemble_size: int = 10,
    nn_tune_ensemble_size: int | None = None,
    random_state: int = 0,
    device: str | None = None,
) -> tuple[ENetResidualNet, dict]:
    if model_name not in RIN_TO_NN:
        raise ValueError(f"Unknown residual model: {model_name}")

    enet_model, enet_params = tune_elastic_net_huber(
        x_train,
        y_train,
        x_validation,
        y_validation,
        alphas=enet_alphas,
        l1_ratio=enet_l1_ratio,
        random_state=random_state,
    )
    enet_train_pred = enet_model.predict(x_train)
    enet_validation_pred = enet_model.predict(x_validation)
    residual_train = y_train - enet_train_pred
    residual_validation = y_validation - enet_validation_pred

    residual_model, residual_params = tune_neural_network(
        RIN_TO_NN[model_name],
        x_train,
        residual_train,
        x_validation,
        residual_validation,
        learning_rates=nn_learning_rates,
        l1_penalties=nn_l1_penalties,
        batch_size=nn_batch_size,
        max_epochs=nn_epochs,
        patience=nn_patience,
        ensemble_size=nn_ensemble_size,
        tune_ensemble_size=nn_tune_ensemble_size,
        base_seed=random_state,
        device=device,
        validation_prediction_offset=enet_validation_pred,
        validation_target_for_score=y_validation,
    )

    model = ENetResidualNet(enet_model=enet_model, residual_model=residual_model)
    params = {
        "enet": enet_params,
        "residual_nn": residual_params,
        "first_stage": "ENet-H",
    }
    return model, params
