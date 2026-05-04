import numpy as np
import pandas as pd

from simulation.config import SimulationConfig


def cross_section_rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map cross-sectional ranks to [-1, 1]."""
    n = len(values)
    ranks = pd.Series(values).rank(method="first").to_numpy()
    return 2.0 * ranks / (n + 1.0) - 1.0


def generate_characteristics(config: SimulationConfig) -> np.ndarray:
    """
    Generate c_{i,j,t}.

    Returns
    -------
    c : ndarray, shape (T, N, Pc)
    """
    rng = np.random.default_rng(config.seed)
    n, t_periods, pc = (
        config.n_assets,
        config.n_periods,
        config.n_characteristics,
    )

    rho_j = rng.uniform(0.9, 1.0, size=pc)
    c_bar = np.zeros((t_periods, n, pc))
    c = np.zeros((t_periods, n, pc))

    c_bar[0] = rng.normal(0.0, 1.0, size=(n, pc))
    for j in range(pc):
        c[0, :, j] = cross_section_rank_normalize(c_bar[0, :, j])

    for t in range(1, t_periods):
        innovation = rng.normal(
            0.0,
            np.sqrt(1.0 - rho_j**2),
            size=(n, pc),
        )
        c_bar[t] = rho_j * c_bar[t - 1] + innovation

        for j in range(pc):
            c[t, :, j] = cross_section_rank_normalize(c_bar[t, :, j])

    return c


def generate_x(config: SimulationConfig) -> np.ndarray:
    """
    Generate x_t = rho x_{t-1} + u_t.

    Returns
    -------
    x : ndarray, shape (T,)
    """
    rng = np.random.default_rng(config.seed + 1)
    x = np.zeros(config.n_periods)
    x[0] = rng.normal(0.0, 1.0)

    for t in range(1, config.n_periods):
        innovation = rng.normal(0.0, np.sqrt(1.0 - config.x_rho**2))
        x[t] = config.x_rho * x[t - 1] + innovation

    return x


def compute_g_star(c: np.ndarray, x: np.ndarray, case: str) -> np.ndarray:
    """
    Compute the true expected return function g*(z_{i,t}).

    Parameters
    ----------
    c : ndarray, shape (T, N, Pc)
    x : ndarray, shape (T,)
    case : {"a", "b", "c"}

    Returns
    -------
    g : ndarray, shape (T, N)
    """
    c1 = c[:, :, 0]
    c2 = c[:, :, 1]
    c3 = c[:, :, 2]
    x_t = x[:, None]

    if case == "a":
        return 0.02 * c1 + 0.02 * c2 + 0.02 * c3 * x_t

    if case == "b":
        return 0.04 * c1**2 + 0.03 * c1 * c2 + 0.012 * np.sign(c3 * x_t)

    if case == "c":
        if c.shape[2] < 5:
            raise ValueError("case='c' requires n_characteristics >= 5.")

        c4 = c[:, :, 3]
        c5 = c[:, :, 4]

        g_linear = 0.025 * c1 + 0.020 * c2 + 0.015 * c3 * x_t
        g_interaction = (
            0.035 * c1 * c4
            - 0.030 * c2 * c5
            + 0.020 * np.tanh(c3 * x_t)
        )
        return g_linear + 0.35 * g_interaction

    raise ValueError("case must be one of {'a', 'b', 'c'}.")


def build_predictors(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Build z_{i,t} = (1, x_t)' tensor c_{i,t}.

    Returns
    -------
    z : ndarray, shape (T, N, 2 * Pc)
    """
    return np.concatenate([c, c * x[:, None, None]], axis=2)


def build_oracle_features(c: np.ndarray, x: np.ndarray, case: str) -> np.ndarray:
    """
    Diagnostic-only features that expose the true low-dimensional signal terms.
    """
    c1 = c[:, :, 0]
    c2 = c[:, :, 1]
    c3 = c[:, :, 2]
    x_t = x[:, None]

    if case == "a":
        features = [c1, c2, c3 * x_t]
    elif case == "b":
        features = [c1**2, c1 * c2, np.sign(c3 * x_t)]
    elif case == "c":
        if c.shape[2] < 5:
            raise ValueError("case='c' requires n_characteristics >= 5.")
        c4 = c[:, :, 3]
        c5 = c[:, :, 4]
        features = [c1, c2, c3 * x_t, c1 * c4, c2 * c5, np.tanh(c3 * x_t)]
    else:
        raise ValueError("case must be one of {'a', 'b', 'c'}.")

    return np.stack(features, axis=2)


def calibrate_signal_to_predictive_r2(
    g: np.ndarray,
    factor_error: np.ndarray,
    epsilon: np.ndarray,
    target_predictive_r2: float,
) -> tuple[np.ndarray, float]:
    """
    Rescale g so Var(g) / Var(g + error) approximately matches the target.

    This keeps the factor and idiosyncratic error draws fixed and changes only
    the signal strength. It is a practical finite-sample calibration for the
    Appendix A statement that predictive R^2 is set around 5%.
    """
    if not 0.0 < target_predictive_r2 < 1.0:
        raise ValueError("target_predictive_r2 must be between 0 and 1.")

    signal_variance = float(np.var(g))
    error_variance = float(np.var(factor_error + epsilon))

    if signal_variance == 0.0:
        return g, 1.0

    scale = np.sqrt(
        target_predictive_r2
        * error_variance
        / ((1.0 - target_predictive_r2) * signal_variance)
    )
    return scale * g, float(scale)


def panel_diagnostics(g: np.ndarray, r: np.ndarray) -> dict[str, float]:
    signal_variance = float(np.var(g))
    return_variance = float(np.var(r))
    predictive_share = signal_variance / return_variance if return_variance > 0 else np.nan
    annualized_volatility = float(np.std(r) * np.sqrt(12.0))

    return {
        "signal_variance": signal_variance,
        "return_variance": return_variance,
        "predictive_share": predictive_share,
        "annualized_volatility": annualized_volatility,
    }


def generate_panel(config: SimulationConfig, case: str = "a") -> dict[str, np.ndarray]:
    """
    Generate one Monte Carlo sample.

    Returns a dictionary containing r, g, z, c, x, v, epsilon.
    """
    rng = np.random.default_rng(config.seed + 2)

    c = generate_characteristics(config)
    x = generate_x(config)
    z = build_predictors(c, x)
    if config.include_oracle_features:
        z = np.concatenate([z, build_oracle_features(c, x, case)], axis=2)
    g = compute_g_star(c, x, case)

    v = rng.normal(
        0.0,
        config.factor_vol,
        size=(config.n_periods, 3),
    )
    beta = c[:, :, :3]
    factor_error = np.einsum("tnk,tk->tn", beta, v)

    # Standard t has variance df / (df - 2). Rescale so variance is epsilon_vol^2.
    t_variance = config.epsilon_df / (config.epsilon_df - 2.0)
    epsilon_scale = config.epsilon_vol / np.sqrt(t_variance)
    epsilon = rng.standard_t(
        df=config.epsilon_df,
        size=(config.n_periods, config.n_assets),
    ) * epsilon_scale

    signal_scale = 1.0
    if config.calibrate_dgp:
        g, signal_scale = calibrate_signal_to_predictive_r2(
            g,
            factor_error,
            epsilon,
            target_predictive_r2=config.target_predictive_r2,
        )

    r = g + factor_error + epsilon
    diagnostics = panel_diagnostics(g, r)

    return {
        "r": r,
        "g": g,
        "z": z,
        "c": c,
        "x": x,
        "v": v,
        "epsilon": epsilon,
        "signal_scale": np.array(signal_scale),
        "diagnostics": diagnostics,
        "include_oracle_features": np.array(config.include_oracle_features),
    }


def split_train_validation_test(panel: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Split T periods into three consecutive equal parts."""
    total_periods = panel["r"].shape[0]
    split = total_periods // 3

    slices = {
        "train": slice(0, split),
        "validation": slice(split, 2 * split),
        "test": slice(2 * split, total_periods),
    }

    output = {}
    for name, idx in slices.items():
        output[name] = {
            key: value[idx]
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_periods
            else value
            for key, value in panel.items()
        }

    return output


def flatten_for_sklearn(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert panel arrays to sklearn-style X, y.

    X shape: (T * N, P)
    y shape: (T * N,)
    """
    z = sample["z"]
    r = sample["r"]
    return z.reshape(-1, z.shape[2]), r.reshape(-1)
