from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Default settings from Appendix A."""

    n_assets: int = 200
    n_periods: int = 180
    n_characteristics: int = 100
    n_x_terms: int = 2
    x_rho: float = 0.95
    factor_vol: float = 0.05
    epsilon_vol: float = 0.05
    epsilon_df: int = 5
    seed: int = 123

    @property
    def n_features(self) -> int:
        return self.n_characteristics * self.n_x_terms


