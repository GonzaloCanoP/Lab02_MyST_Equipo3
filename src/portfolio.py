"""Risk Parity, agregación de señales y rebalanceo (P4).

La covarianza en t usa solo retornos hasta t (CLAUDE.md, sección 4).
"""

import pandas as pd


def estimate_cov(returns: pd.DataFrame, method: str) -> pd.DataFrame:
    """Matriz de covarianza de retornos; `method` ∈ {"sample", "ewma", "ledoit_wolf"}."""
    raise NotImplementedError


def risk_parity_weights(cov: pd.DataFrame, tol: float = 1e-10) -> pd.Series:
    """Pesos de Risk Parity con la formulación convexa de Spinu (2013); w_i > 0 y Σ w = 1."""
    raise NotImplementedError


def inverse_vol_weights(cov: pd.DataFrame) -> pd.Series:
    """Versión naive de Risk Parity: w_i = (1/σ_i) / Σ_j (1/σ_j)."""
    raise NotImplementedError


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Contribución total al riesgo: RC_i = w_i (Σw)_i / σ_p, con Σ RC_i = σ_p."""
    raise NotImplementedError


def equal_weights(tickers: list[str]) -> pd.Series:
    """Pesos iguales 1/n (benchmark, SPEC punto 5)."""
    raise NotImplementedError


def compose_target(
    base_weights: pd.Series, strength: pd.Series, regime_multiplier: float
) -> pd.Series:
    """Peso objetivo: m · w̃ / max(1, Σ|w̃|), con w̃ = w · s (SPEC punto 5)."""
    raise NotImplementedError


def sleeve_weights(
    prices: dict,
    signals: dict[str, pd.DataFrame],
    regimes: pd.Series,
    config: dict,
    method: str = "risk_parity",
) -> pd.DataFrame:
    """Panel causal de |w_target| con rebalanceo aplicado; `method` ∈ {"risk_parity", "equal"}."""
    raise NotImplementedError


def turnover(weights_drift: pd.DataFrame, weights_target: pd.DataFrame) -> pd.Series:
    """Turnover por fecha: ½ Σ |w_t − w_t−|, con w_t− el peso después del drift."""
    raise NotImplementedError


def rebalance_sweep(
    prices: dict,
    params_by_regime: dict,
    regimes: pd.Series,
    config: dict,
    bands: list[float],
    frequencies: list[str],
) -> pd.DataFrame:
    """Barrido de rebalanceo: retorno bruto, costo total, retorno neto y turnover."""
    raise NotImplementedError
