"""Risk Parity, agregación de señales y rebalanceo (P4).

La covarianza en t usa solo retornos hasta t (CLAUDE.md, sección 4).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# El contrato de `estimate_cov` no recibe `config`, así que λ vive aquí (SPEC_portafolio.md:
# T_eff = 1 / (1 − λ) = 100 días, comparable a la ventana de 126 días).
EWMA_LAMBDA = 0.99


def _validate_cov(cov: pd.DataFrame) -> None:
    """Exige una covarianza cuadrada, finita, simétrica y con índice igual a columnas."""
    if cov.shape[0] != cov.shape[1] or not cov.index.equals(cov.columns):
        raise ValueError("cov debe ser cuadrada, con el mismo índice en filas y columnas")
    values = cov.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cov contiene valores no finitos")
    if not np.allclose(values, values.T, atol=1e-12):
        raise ValueError("cov no es simétrica")
    if (np.diag(values) <= 0).any():
        raise ValueError("cov tiene varianzas no positivas")


def estimate_cov(returns: pd.DataFrame, method: str) -> pd.DataFrame:
    """Estima la matriz de covarianzas de retornos (SPEC_portafolio.md, Estimador de covarianza).

    Parameters
    ----------
    returns : pd.DataFrame
        Log-retornos diarios, fechas × tickers. Nunca precios (CLAUDE.md, sección 4 y Paso 2).
        No se rellenan huecos: si hay NaN, se lanza un error.
    method : str
        "sample" (muestral), "ewma" (λ = 0.99, media cero) o "ledoit_wolf" (encogimiento).

    Returns
    -------
    pd.DataFrame
        Σ tickers × tickers, simétrica.
    """
    if returns.isna().any().any():
        raise ValueError("returns contiene NaN; elimina la primera fila de la diferencia antes")
    tickers = returns.columns

    if method == "sample":
        values = returns.cov().to_numpy()
    elif method == "ewma":
        x = returns.to_numpy(dtype=float)
        # Peso λ^(T−1−k) a la observación k: la más reciente pesa más. Σ = Σ_k w_k r_k r_kᵀ.
        w = EWMA_LAMBDA ** np.arange(len(x) - 1, -1, -1)
        w = w / w.sum()
        values = (x * w[:, None]).T @ x
    elif method == "ledoit_wolf":
        values = LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_
    else:
        raise ValueError(f"método de covarianza desconocido: {method!r}")

    values = (values + values.T) / 2  # elimina asimetría numérica
    return pd.DataFrame(values, index=tickers, columns=tickers)


def _newton_polish(y: np.ndarray, S: np.ndarray, tol: float, max_iter: int = 50) -> np.ndarray:
    """Pule la solución con Newton amortiguado, manteniendo y > 0.

    Gradiente: g = S y − 1/(n y).  Hessiano: H = S + diag(1 / (n y²)).
    """
    n = len(y)
    for _ in range(max_iter):
        gradient = S @ y - 1.0 / (n * y)
        if np.max(np.abs(gradient)) < tol:
            return y
        hessian = S + np.diag(1.0 / (n * y**2))
        step = np.linalg.solve(hessian, gradient)
        t = 1.0
        while np.any(y - t * step <= 0):  # recorta el paso hasta que y siga positivo
            t /= 2
        y = y - t * step
    raise RuntimeError("risk_parity_weights no convergió dentro de la tolerancia")


def risk_parity_weights(cov: pd.DataFrame, tol: float = 1e-10) -> pd.Series:
    """Pesos Risk Parity long-only con la formulación convexa de Spinu (Paso 5).

    Resuelve  min_{y>0} ½ yᵀΣy − (1/n) Σ ln y_i  y normaliza w = y / Σ y_j.
    La solución es única porque el problema es convexo.

    Parameters
    ----------
    cov : pd.DataFrame
        Σ tickers × tickers (de `estimate_cov`).
    tol : float
        Tolerancia sobre el gradiente, que equivale a y_i (Σy)_i = 1/n para todo i.

    Returns
    -------
    pd.Series
        w_i > 0 con Σ w_i = 1, indexado como `cov`.
    """
    _validate_cov(cov)
    n = cov.shape[0]
    # Los pesos no cambian si se escala Σ; se normaliza para que el problema esté bien condicionado
    # (varianzas diarias del orden de 1e-4 dejan el término logarítmico dominando).
    S = cov.to_numpy(dtype=float)
    S = S / np.mean(np.diag(S))

    def objective(y: np.ndarray) -> float:
        return 0.5 * y @ S @ y - np.log(y).sum() / n

    def gradient(y: np.ndarray) -> np.ndarray:
        return S @ y - 1.0 / (n * y)

    # Punto de partida: volatilidad inversa, escalada para que yᵀSy = 1 (valor en el óptimo).
    y0 = 1.0 / np.sqrt(np.diag(S))
    y0 = y0 / np.sqrt(y0 @ S @ y0)

    result = minimize(
        objective, y0, jac=gradient, method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
    )
    y = _newton_polish(result.x, S, tol)
    return pd.Series(y / y.sum(), index=cov.index, name="rp_weights")


def inverse_vol_weights(cov: pd.DataFrame) -> pd.Series:
    """Versión naive de Risk Parity (Paso 4): w_i = (1/σ_i) / Σ_j (1/σ_j)."""
    _validate_cov(cov)
    inv_vol = 1.0 / np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    return pd.Series(inv_vol / inv_vol.sum(), index=cov.index, name="inverse_vol_weights")


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Contribución total al riesgo (Paso 3): RC_i = w_i (Σw)_i / σ_p, con Σ RC_i = σ_p exacto."""
    _validate_cov(cov)
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    if np.isnan(w).any():
        raise ValueError("weights no cubre todos los activos de cov")
    sigma_w = cov.to_numpy(dtype=float) @ w
    sigma_p = np.sqrt(w @ sigma_w)
    return pd.Series(w * sigma_w / sigma_p, index=cov.index, name="risk_contribution")


def equal_weights(tickers: list[str]) -> pd.Series:
    """Pesos iguales 1/n (benchmark, SPEC punto 5)."""
    return pd.Series(1.0 / len(tickers), index=list(tickers), name="equal_weights")


def compose_target(
    base_weights: pd.Series, strength: pd.Series, regime_multiplier: float
) -> pd.Series:
    """Peso objetivo de un día (Paso 7): m · w̃ / max(1, Σ|w̃|), con w̃ = w · s (SPEC punto 5).

    Parameters
    ----------
    base_weights : pd.Series
        w^RP (o 1/n en el benchmark), indexado por ticker.
    strength : pd.Series
        s_i ∈ [-1, 1] de `generate_signals`. Conserva el signo.
    regime_multiplier : float
        m(régimen) ∈ (0, 1].

    Returns
    -------
    pd.Series
        w_target con signo. El panel que recibe el motor es |w_target| (`sleeve_weights`).
    """
    if not 0 < regime_multiplier <= 1:
        raise ValueError("regime_multiplier debe estar en (0, 1]")
    s = strength.reindex(base_weights.index)
    if s.isna().any():
        raise ValueError("strength no cubre todos los activos de base_weights")
    if (s.abs() > 1 + 1e-12).any():
        raise ValueError("strength debe estar en [-1, 1]")

    scaled = base_weights * s
    gross = scaled.abs().sum()
    return regime_multiplier * scaled / max(1.0, gross)


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
    """Turnover por fecha (Paso 8): T_t = ½ Σ |w_t − w_t−|, con w_t− el peso después del drift.

    `weights_drift` es el peso justo antes de rebalancear y `weights_target` el peso al que se
    rebalancea. Comparar contra el objetivo viejo en vez del drift sobrestima el costo.
    """
    if not weights_drift.index.equals(weights_target.index) or \
            not weights_drift.columns.equals(weights_target.columns):
        raise ValueError("weights_drift y weights_target deben tener el mismo índice y columnas")
    return 0.5 * (weights_target - weights_drift).abs().sum(axis=1).rename("turnover")


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