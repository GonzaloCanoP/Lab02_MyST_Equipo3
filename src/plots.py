"""Figuras del proyecto: una función por figura, cada una devuelve `matplotlib.figure.Figure` (P3).

Reciben resultados ya calculados; aquí no hay cálculos de estrategia. Firmas propuestas en P0:
P3 puede ajustarlas avisando a P1 (main.py).
"""

import optuna
import pandas as pd
from matplotlib.figure import Figure


def plot_equity(curves: dict[str, pd.Series], blocks: dict[str, tuple]) -> Figure:
    """Valor del portafolio en train y test, con benchmark; `blocks` sombrea cada bloque."""
    raise NotImplementedError


def plot_drawdown(drawdowns: dict[str, pd.Series]) -> Figure:
    """Curvas de drawdown."""
    raise NotImplementedError


def plot_returns_table(tables: dict[str, pd.DataFrame]) -> Figure:
    """Retornos mensuales, trimestrales y anuales (salida de `returns_table`)."""
    raise NotImplementedError


def plot_sensitivity(sensitivity: pd.DataFrame) -> Figure:
    """Cambio en Calmar al variar cada parámetro ±20% (salida de `sensitivity`)."""
    raise NotImplementedError


def plot_cost_curve(cost_curve: pd.DataFrame, base_cost_bps: float) -> Figure:
    """Retorno neto contra costo de ida y vuelta, marcando el costo base (salida de `cost_sweep`)."""
    raise NotImplementedError


def plot_regime_timeline(price: pd.Series, filtered: pd.Series, viterbi: pd.Series) -> Figure:
    """Precio coloreado por régimen filtrado, con la versión Viterbi al lado."""
    raise NotImplementedError


def plot_regime_features(features: pd.DataFrame, labels: pd.Series) -> Figure:
    """Distribución de las variables de régimen por régimen."""
    raise NotImplementedError


def plot_equity_regimes(equity: pd.Series, regimes: pd.Series) -> Figure:
    """Curva de equity con los regímenes superpuestos."""
    raise NotImplementedError


def plot_risk_contributions(contributions: pd.DataFrame) -> Figure:
    """Contribuciones al riesgo por activo: Risk Parity contra pesos iguales."""
    raise NotImplementedError


def plot_signal_heatmap(strength: pd.DataFrame) -> Figure:
    """Mapa de calor fecha × activo de la fuerza de señal s_i."""
    raise NotImplementedError


def plot_corr_by_regime(corr_by_regime: dict[str, pd.DataFrame]) -> Figure:
    """Matriz de correlación de los activos en cada régimen."""
    raise NotImplementedError


def plot_rebalance_sweep(sweep: pd.DataFrame) -> Figure:
    """Retorno bruto, costo total, retorno neto y turnover contra frecuencia o banda δ."""
    raise NotImplementedError


def plot_surface_3d(grid: pd.DataFrame, x: str, y: str, z: str = "calmar") -> Figure:
    """Superficie 3D del Calmar sobre las dos dimensiones más influyentes del random search."""
    raise NotImplementedError


def plot_optimization_history(study: optuna.Study) -> Figure:
    """Historia de optimización del estudio TPE."""
    raise NotImplementedError


def plot_param_importance(study: optuna.Study) -> Figure:
    """Importancia de parámetros del estudio TPE."""
    raise NotImplementedError


def plot_slices(study: optuna.Study) -> Figure:
    """Slice plots del estudio TPE."""
    raise NotImplementedError
