from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from experiments.metrics import mean_squared_error


NN_ARCHITECTURES = {
    "NN1": [32],
    "NN2": [32, 16],
    "NN3": [32, 16, 8],
}


def default_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_torch_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if (
        device == "mps"
        and (getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available())
    ):
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return device


@dataclass(frozen=True)
class NeuralNetParams:
    learning_rate: float
    l1_penalty: float
    hidden_layers: list[int]


class FeedForwardNet(nn.Module):
    def __init__(self, n_features: int, hidden_layers: list[int]):
        super().__init__()
        layers = []
        in_features = n_features

        for width in hidden_layers:
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            in_features = width

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


class NeuralNetEnsemble:
    def __init__(
        self,
        scaler: StandardScaler,
        models: list[FeedForwardNet],
        device: str,
        y_mean: float,
        y_std: float,
    ):
        self.scaler = scaler
        self.models = models
        self.device = device
        self.y_mean = y_mean
        self.y_std = y_std

    def predict(self, x: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        x_scaled = self.scaler.transform(x).astype(np.float32)
        tensor = torch.from_numpy(x_scaled).to(self.device)

        predictions = []
        for model in self.models:
            model.eval()
            chunks = []
            with torch.no_grad():
                for start in range(0, tensor.shape[0], batch_size):
                    batch_x = tensor[start : start + batch_size]
                    chunks.append(model(batch_x).cpu().numpy())
            predictions.append(np.concatenate(chunks))

        scaled_prediction = np.mean(predictions, axis=0)
        return scaled_prediction * self.y_std + self.y_mean


def _l1_norm(model: nn.Module) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            penalty = penalty + module.weight.abs().sum()
    return penalty


def _fit_single_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    params: NeuralNetParams,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
) -> FeedForwardNet:
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = FeedForwardNet(x_train.shape[1], params.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = nn.MSELoss()

    x_train_tensor = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device)
    x_val_tensor = torch.from_numpy(x_validation.astype(np.float32)).to(device)
    y_val_tensor = torch.from_numpy(y_validation.astype(np.float32)).to(device)

    best_state = None
    best_validation = np.inf
    stale_epochs = 0
    n_train = x_train_tensor.shape[0]

    for _ in range(max_epochs):
        model.train()
        permutation = torch.randperm(n_train, device=device)

        for start in range(0, n_train, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_x = x_train_tensor[batch_idx]
            batch_y = y_train_tensor[batch_idx]
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y) + params.l1_penalty * _l1_norm(model)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(x_val_tensor), y_val_tensor).item()

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def tune_neural_network(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    learning_rates: list[float],
    l1_penalties: list[float],
    batch_size: int = 10_000,
    max_epochs: int = 100,
    patience: int = 5,
    ensemble_size: int = 10,
    tune_ensemble_size: int | None = None,
    base_seed: int = 0,
    device: str | None = None,
) -> tuple[NeuralNetEnsemble, dict]:
    if model_name not in NN_ARCHITECTURES:
        raise ValueError(f"Unknown neural network model: {model_name}")

    device = validate_torch_device(device or default_torch_device())
    tune_ensemble_size = tune_ensemble_size or ensemble_size
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    x_validation_scaled = scaler.transform(x_validation).astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0.0:
        y_std = 1.0
    y_train_scaled = ((y_train - y_mean) / y_std).astype(np.float32)
    y_validation_scaled = ((y_validation - y_mean) / y_std).astype(np.float32)

    best_model = None
    best_params_object = None
    best_score = np.inf
    best_params = {}

    for learning_rate in learning_rates:
        for l1_penalty in l1_penalties:
            params = NeuralNetParams(
                learning_rate=learning_rate,
                l1_penalty=l1_penalty,
                hidden_layers=NN_ARCHITECTURES[model_name],
            )

            models = []
            for ensemble_idx in range(tune_ensemble_size):
                model = _fit_single_network(
                    x_train_scaled,
                    y_train_scaled,
                    x_validation_scaled,
                    y_validation_scaled,
                    params=params,
                    seed=base_seed + ensemble_idx,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    device=device,
                )
                models.append(model)

            ensemble = NeuralNetEnsemble(
                scaler=scaler,
                models=models,
                device=device,
                y_mean=y_mean,
                y_std=y_std,
            )
            score = mean_squared_error(y_validation, ensemble.predict(x_validation))

            if score < best_score:
                best_score = score
                best_model = ensemble
                best_params_object = params
                best_params = {
                    "learning_rate": learning_rate,
                    "l1_penalty": l1_penalty,
                    "hidden_layers": NN_ARCHITECTURES[model_name],
                    "ensemble_size": ensemble_size,
                    "tune_ensemble_size": tune_ensemble_size,
                }

    if tune_ensemble_size == ensemble_size:
        return best_model, best_params

    final_models = []
    for ensemble_idx in range(ensemble_size):
        model = _fit_single_network(
            x_train_scaled,
            y_train_scaled,
            x_validation_scaled,
            y_validation_scaled,
            params=best_params_object,
            seed=base_seed + ensemble_idx,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
        )
        final_models.append(model)

    final_ensemble = NeuralNetEnsemble(
        scaler=scaler,
        models=final_models,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
    )

    return final_ensemble, best_params
