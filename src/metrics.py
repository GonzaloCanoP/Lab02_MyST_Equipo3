"""Métricas de desempeño, drawdown, tablas de retornos y exposición (P2)."""

import numpy as np
import pandas as pd

from src.backtest import BacktestResult


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    rf: pd.Series | float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Calcula las principales métricas de desempeño de la estrategia.

    Calmar = retorno anualizado / |MDD|.
    Sharpe y Sortino se anualizan con sqrt(periods_per_year).

    Parameters
    ----------
    equity : pd.Series
        Curva de equity.
    trades : pd.DataFrame
        Operaciones cerradas; usa la columna ``pnl_net``.
    rf : pd.Series or float
        Tasa libre de riesgo diaria.
    periods_per_year : int
        Número de periodos de negociación por año.

    Returns
    -------
    dict
        ann_return, ann_vol, sharpe, sortino, calmar,
        max_drawdown, n_trades, win_rate y payoff_ratio.
    """
    if len(equity) < 2:
        raise ValueError("equity necesita al menos dos observaciones.")
    equity = equity.astype(float)

    returns = equity.pct_change(fill_method=None).dropna()
    n_periods = len(returns)

    if equity.iloc[0] > 0 and equity.iloc[-1] > 0:
        ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n_periods) - 1
    else:
        ann_return = np.nan
    ann_vol = returns.std(ddof=1) * np.sqrt(periods_per_year)

    if isinstance(rf, pd.Series):
        aligned = pd.concat(
            [returns.rename("strategy"), rf.rename("rf")], axis=1, join="inner"
        ).dropna()
        excess_returns = aligned["strategy"] - aligned["rf"]
    else:
        excess_returns = returns - float(rf)
    excess_std = excess_returns.std(ddof=1)

    if len(excess_returns) >= 2 and np.isfinite(excess_std) and excess_std > 0:
        sharpe = excess_returns.mean() / excess_std * np.sqrt(periods_per_year)
    else:
        sharpe = np.nan
    downside = np.minimum(excess_returns.to_numpy(dtype=float), 0.0)

    if len(downside) > 0:
        downside_deviation = np.sqrt(np.mean(downside**2))
    else:
        downside_deviation = np.nan

    if np.isfinite(downside_deviation) and downside_deviation > 0:
        sortino = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
    else:
        sortino = np.nan
    drawdown = drawdown_series(equity)
    max_drawdown = drawdown.min()

    if np.isfinite(max_drawdown) and max_drawdown < 0:
        calmar = ann_return / abs(max_drawdown)
    else:
        # MDD = 0 todavía está marcado como decisión pendiente
        # en P2. Se mantiene transparente como NaN.
        calmar = np.nan
    n_trades = int(len(trades))

    if n_trades > 0:
        if "pnl_net" not in trades.columns:
            raise ValueError("trades necesita la columna pnl_net.")
        pnl = pd.to_numeric(trades["pnl_net"], errors="coerce").dropna()
        if len(pnl) > 0:
            win_rate = float((pnl > 0).mean())
        else:
            win_rate = np.nan
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if not wins.empty and not losses.empty:
            payoff_ratio = float(wins.mean() / abs(losses.mean()))
        else:
            payoff_ratio = np.nan
    else:
        win_rate = np.nan
        payoff_ratio = np.nan

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_drawdown),
        "n_trades": n_trades,
        "win_rate": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
    }


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown en cada fecha: equity / máximo acumulado - 1."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    return drawdown.rename("drawdown")


def returns_table(equity: pd.Series) -> dict[str, pd.DataFrame]:
    """Construye tablas de retornos mensuales, trimestrales y anuales.

    Los retornos de cada periodo se calculan componiendo los retornos
    diarios de la curva de equity.

    Parameters
    ----------
    equity : pd.Series
        Curva de equity con DatetimeIndex.

    Returns
    -------
    dict[str, pd.DataFrame]
        Tablas con llaves ``mensual``, ``trimestral`` y ``anual``.
    """
    if not isinstance(equity.index, pd.DatetimeIndex):
        raise TypeError("equity debe tener un DatetimeIndex.")

    returns = equity.pct_change(fill_method=None).dropna()
    monthly = (1.0 + returns).groupby(returns.index.to_period("M")).prod() - 1.0
    monthly_data = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "return": monthly.to_numpy()}
    )
    monthly_table = (
        monthly_data.pivot(index="year", columns="month", values="return")
        .reindex(columns=range(1, 13))
        .sort_index()
    )
    monthly_table.index.name = "Año"
    monthly_table.columns.name = "Mes"
    quarterly = (1.0 + returns).groupby(returns.index.to_period("Q")).prod() - 1.0
    quarterly_data = pd.DataFrame(
        {
            "year": quarterly.index.year,
            "quarter": quarterly.index.quarter,
            "return": quarterly.to_numpy(),
        }
    )
    quarterly_table = (
        quarterly_data.pivot(index="year", columns="quarter", values="return")
        .reindex(columns=range(1, 5))
        .sort_index()
    )
    quarterly_table.index.name = "Año"
    quarterly_table.columns.name = "Trimestre"
    annual = (1.0 + returns).groupby(returns.index.to_period("Y")).prod() - 1.0
    annual_table = pd.DataFrame({"Retorno": annual.to_numpy()}, index=annual.index.year)
    annual_table.index.name = "Año"
    annual_table.columns.name = "Periodo"

    return {"mensual": monthly_table, "trimestral": quarterly_table, "anual": annual_table}


def exposure_metrics(result: BacktestResult) -> dict:
    """Calcula métricas de exposición de la estrategia.

    Reporta porcentaje de días con posición por activo y para
    el portafolio, operaciones cerradas por mes y motivos de salida.

    Parameters
    ----------
    result : BacktestResult
        Resultado producido por ``run_backtest``.

    Returns
    -------
    dict
        Métricas de exposición y actividad.
    """
    positions = result.positions

    if len(positions) == 0:
        return {
            "time_in_market_by_asset": pd.Series(dtype=float),
            "time_in_market_portfolio": np.nan,
            "trades_per_month": np.nan,
            "exit_reasons": pd.Series(dtype=int, name="count"),
        }
    in_market = positions.abs() > 0
    time_by_asset = in_market.mean().rename("time_in_market")
    time_portfolio = float(in_market.any(axis=1).mean())
    n_months = result.equity.index.to_period("M").nunique()
    trades_per_month = len(result.trades) / n_months if n_months > 0 else np.nan
    reasons = ["signal", "stop", "target", "max_holding"]

    if len(result.trades) > 0 and "exit_reason" in result.trades.columns:
        exit_reasons = (
            result.trades["exit_reason"].value_counts().reindex(reasons, fill_value=0).astype(int)
        )
    else:
        exit_reasons = pd.Series(0, index=reasons, dtype=int)
    exit_reasons.name = "count"

    return {
        "time_in_market_by_asset": time_by_asset,
        "time_in_market_portfolio": time_portfolio,
        "trades_per_month": float(trades_per_month),
        "exit_reasons": exit_reasons,
    }


def breakeven_winrate(
    trades: pd.DataFrame, k_stop: float, reward_ratio: float, atr_over_price: float, config: dict
) -> dict:
    """Compara el win rate de equilibrio teórico y empírico.

    El teórico sigue SPEC punto 8:

        p* = (1 + c) / (1 + r)

        c = 2 * (commission + slippage)
            / (k_stop * ATR/P)

    El empírico usa:

        p* = 1 / (1 + payoff)

    Parameters
    ----------
    trades : pd.DataFrame
        Operaciones cerradas con columna ``pnl_net``.
    k_stop : float
        Multiplicador del stop.
    reward_ratio : float
        Relación recompensa/riesgo.
    atr_over_price : float
        ATR dividido entre precio.
    config : dict
        Configuración del proyecto.

    Returns
    -------
    dict
        Break-even teórico, empírico y win rate observado.
    """
    if k_stop <= 0:
        raise ValueError("k_stop debe ser positivo.")

    if reward_ratio <= 0:
        raise ValueError("reward_ratio debe ser positivo.")

    if atr_over_price <= 0:
        raise ValueError("atr_over_price debe ser positivo.")
    cost_in_r = 2.0 * (config["commission"] + config["slippage"]) / (k_stop * atr_over_price)
    theoretical = (1.0 + cost_in_r) / (1.0 + reward_ratio)

    if len(trades) > 0 and "pnl_net" in trades.columns:
        pnl = pd.to_numeric(trades["pnl_net"], errors="coerce").dropna()
        observed = float((pnl > 0).mean()) if len(pnl) > 0 else np.nan
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if not wins.empty and not losses.empty:
            payoff_ratio = float(wins.mean() / abs(losses.mean()))
            empirical = 1.0 / (1.0 + payoff_ratio)
        else:
            payoff_ratio = np.nan
            empirical = np.nan
    else:
        observed = np.nan
        payoff_ratio = np.nan
        empirical = np.nan

    return {
        "theoretical_win_rate": float(theoretical),
        "empirical_breakeven_win_rate": float(empirical),
        "observed_win_rate": float(observed),
        "payoff_ratio": float(payoff_ratio),
        "cost_in_r": float(cost_in_r),
    }


def block_equity(equity: pd.Series, start, end) -> pd.Series:
    """Equity de un bloque, desde el último cierre anterior a `start` (si existe) hasta `end`.

    Así el retorno del primer día del bloque cuenta: sin el cierre previo, `pct_change` lo pierde.
    """
    history = equity.loc[:end]
    before = history.index[history.index < pd.Timestamp(start)]
    return history.loc[before[-1] :] if len(before) else history.loc[start:]


def metrics_by_block(
    equity: pd.Series,
    trades: pd.DataFrame,
    blocks: dict[str, tuple],
    rf: pd.Series | float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """`compute_metrics` por bloque: train, validation y test por separado (P2, tareas 5 y 20).

    El equity de cada bloque arranca en el último cierre anterior al bloque (`block_equity`); las
    operaciones se asignan al bloque de su fecha de entrada. Un bloque con menos de dos
    observaciones se omite.

    Parameters
    ----------
    equity : pd.Series
        Curva de equity continua.
    trades : pd.DataFrame
        Operaciones cerradas (`BacktestResult.trades`).
    blocks : dict[str, tuple]
        Salida de `block_dates(config)`.
    rf, periods_per_year
        Como en `compute_metrics`.

    Returns
    -------
    pd.DataFrame
        Un renglón por bloque con las métricas de `compute_metrics`.
    """
    rows = {}
    for name, (start, end) in blocks.items():
        curve = block_equity(equity, start, end)
        if len(curve) < 2:  # bloque sin datos (por ejemplo, antes de la primera ventana OOS)
            continue
        if len(trades):
            entry = pd.to_datetime(trades["entry_date"])
            block_trades = trades[(entry >= start) & (entry <= end)]
        else:
            block_trades = trades
        rows[name] = compute_metrics(curve, block_trades, rf, periods_per_year)
    return pd.DataFrame.from_dict(rows, orient="index")


def buy_and_hold_equity(prices: dict, config: dict, start, end) -> pd.Series:
    """Benchmark buy & hold: 1/n del capital en cada activo, sin rebalanceo (P2, tarea 20).

    Compra al Open del primer día del periodo con los mismos costos de entrada de la estrategia
    (slippage en contra y comisión, SPEC punto 6) y valúa al Close; no se vende, así que no hay
    costo de salida (como las posiciones abiertas al fin de la muestra).

    Returns
    -------
    pd.Series
        Equity al cierre de cada fecha del periodo.
    """
    capital = float(config["initial_capital"])
    budget = capital / len(prices)
    cost_per_share = 1 + config["commission"]
    value = 0.0
    for ohlcv in prices.values():
        window = ohlcv.loc[start:end]
        entry = window["open"].iloc[0] * (1 + config["slippage"])
        shares = budget / (entry * cost_per_share)
        value = value + shares * window["close"]
    return value.rename("buy_and_hold")


def volatility_matched(equity: pd.Series, reference: pd.Series) -> pd.Series:
    """`equity` con sus retornos diarios escalados a la volatilidad de `reference` (ex post).

    Comparación de riesgo igual para el reporte (estrategia contra buy & hold): r' = r · σ_ref / σ,
    con las σ de toda la muestra mostrada. Usa información de todo el periodo, así que es una
    referencia ex post y no una estrategia operable.
    """
    returns = equity.pct_change(fill_method=None).dropna()
    target = reference.pct_change(fill_method=None).dropna()
    scaled = returns * target.std() / returns.std()
    curve = (1.0 + scaled).cumprod() * equity.iloc[0]
    return pd.concat([equity.iloc[:1], curve]).rename(f"{equity.name}_vol_igual")

