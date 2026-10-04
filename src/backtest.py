"""Motor de backtesting event-driven (P1).

Único lugar del proyecto donde la señal de t se ejecuta en t+1 (CLAUDE.md, sección 4).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    """Resultado de `run_backtest`.

    Attributes
    ----------
    equity : pd.Series
        Valor del portafolio al cierre de cada fecha: cash + Σ shares · close.
    cash : pd.Series
        Efectivo al cierre de cada fecha.
    positions : pd.DataFrame
        Panel fecha × ticker de acciones con signo (positivo largo, negativo corto).
    exposure : pd.DataFrame
        Panel fecha × ticker de nocional con signo al cierre.
    fills : pd.DataFrame
        Una fila por ejecución: date, ticker, side, shares, price, notional, commission, slippage, reason.
    trades : pd.DataFrame
        Una fila por operación cerrada: trade_id, ticker, side, entry_date, entry_price, exit_date,
        exit_price, shares, exit_reason {signal, stop, target, max_holding}, regime_at_entry,
        pnl_gross, commission, slippage, borrow, pnl_net.
    costs : pd.DataFrame
        Panel fecha × {commission, slippage, borrow}.
    """

    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    exposure: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame


def run_backtest(
    prices: dict,
    signals: dict[str, pd.DataFrame],
    sleeve_weights: pd.DataFrame,
    trade_params: pd.DataFrame,
    config: dict,
    entry_mask: pd.Series | None = None,
) -> BacktestResult:
    """Simula la estrategia barra por barra con estado explícito de cash, posiciones y equity.

    SPEC puntos 4 a 7: salidas (SL, TP, señal opuesta, holding máximo, rearme), sizing fixed
    fractional con tope sin apalancamiento, costos y orden de eventos dentro de la barra t+1.
    Función pura: no lee archivos, no usa estado global y no modifica sus argumentos.

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        OHLCV por ticker con índice común.
    signals : dict[str, pd.DataFrame]
        Salida de `generate_signals`; se usan los paneles "state" y "atr".
    sleeve_weights : pd.DataFrame
        Panel de |w_target| ≥ 0 con Σ ≤ 1 por fecha. Activo individual: una columna de unos.
    trade_params : pd.DataFrame
        Fecha × {k_stop, reward_ratio, max_holding, risk_per_trade}, vigentes al cierre de cada fecha.
    config : dict
        `CONFIG` de main.py (capital, comisión, slippage, borrow, resize_on_rebalance).
    entry_mask : pd.Series or None
        True donde se permite abrir con la señal de esa fecha. None permite abrir siempre.

    Returns
    -------
    BacktestResult
    """
    raise NotImplementedError
