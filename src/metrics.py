"""Métricas de desempeño, drawdown, tablas de retornos y exposición (P2)."""

import pandas as pd

from src.backtest import BacktestResult


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    rf: pd.Series | float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Métricas de desempeño de una curva de equity y sus operaciones.

    Calmar = retorno anualizado / |MDD|; Sharpe y Sortino anualizados con √periods_per_year.

    Returns
    -------
    dict
        ann_return, ann_vol, sharpe, sortino, calmar, max_drawdown, n_trades, win_rate, payoff_ratio.
    """
    raise NotImplementedError


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown en cada fecha: equity / máximo acumulado − 1."""
    raise NotImplementedError


def returns_table(equity: pd.Series) -> dict[str, pd.DataFrame]:
    """Tablas de retornos con llaves "mensual", "trimestral" y "anual"."""
    raise NotImplementedError


def exposure_metrics(result: BacktestResult) -> dict:
    """Tiempo en mercado, operaciones por mes y salidas por motivo (SPEC punto 8)."""
    raise NotImplementedError


def breakeven_winrate(
    trades: pd.DataFrame,
    k_stop: float,
    reward_ratio: float,
    atr_over_price: float,
    config: dict,
) -> dict:
    """Win rate de equilibrio p* teórico (SPEC punto 8) contra el empírico de las operaciones."""
    raise NotImplementedError
