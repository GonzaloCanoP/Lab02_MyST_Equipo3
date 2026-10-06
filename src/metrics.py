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

    returns = equity.pct_change(
        fill_method=None
    ).dropna()

    n_periods = len(returns)

    if equity.iloc[0] > 0 and equity.iloc[-1] > 0:
        ann_return = (
            equity.iloc[-1] / equity.iloc[0]
        ) ** (
            periods_per_year / n_periods
        ) - 1
    else:
        ann_return = np.nan

    ann_vol = (
        returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    if isinstance(rf, pd.Series):
        aligned = pd.concat(
            [
                returns.rename("strategy"),
                rf.rename("rf"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        excess_returns = (
            aligned["strategy"]
            - aligned["rf"]
        )
    else:
        excess_returns = (
            returns
            - float(rf)
        )

    excess_std = excess_returns.std(ddof=1)

    if (
        len(excess_returns) >= 2
        and np.isfinite(excess_std)
        and excess_std > 0
    ):
        sharpe = (
            excess_returns.mean()
            / excess_std
            * np.sqrt(periods_per_year)
        )
    else:
        sharpe = np.nan

    downside = np.minimum(
        excess_returns.to_numpy(dtype=float),
        0.0,
    )

    if len(downside) > 0:
        downside_deviation = np.sqrt(
            np.mean(downside**2)
        )
    else:
        downside_deviation = np.nan

    if (
        np.isfinite(downside_deviation)
        and downside_deviation > 0
    ):
        sortino = (
            excess_returns.mean()
            / downside_deviation
            * np.sqrt(periods_per_year)
        )
    else:
        sortino = np.nan

    drawdown = drawdown_series(equity)
    max_drawdown = drawdown.min()

    if (
        np.isfinite(max_drawdown)
        and max_drawdown < 0
    ):
        calmar = (
            ann_return
            / abs(max_drawdown)
        )
    else:
        # MDD = 0 todavía está marcado como decisión pendiente
        # en P2. Se mantiene transparente como NaN.
        calmar = np.nan

    n_trades = int(len(trades))

    if n_trades > 0:
        if "pnl_net" not in trades.columns:
            raise ValueError(
                "trades necesita la columna pnl_net."
            )

        pnl = pd.to_numeric(
            trades["pnl_net"],
            errors="coerce",
        ).dropna()

        if len(pnl) > 0:
            win_rate = float(
                (pnl > 0).mean()
            )
        else:
            win_rate = np.nan

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        if (
            not wins.empty
            and not losses.empty
        ):
            payoff_ratio = float(
                wins.mean()
                / abs(losses.mean())
            )
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


def drawdown_series(
    equity: pd.Series,
) -> pd.Series:
    """Drawdown en cada fecha: equity / máximo acumulado - 1."""
    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1
    )

    return drawdown.rename(
        "drawdown"
    )


def returns_table(
    equity: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Tablas de retornos con llaves mensual, trimestral y anual."""
    raise NotImplementedError


def exposure_metrics(
    result: BacktestResult,
) -> dict:
    """Tiempo en mercado, operaciones por mes y salidas por motivo."""
    raise NotImplementedError


def breakeven_winrate(
    trades: pd.DataFrame,
    k_stop: float,
    reward_ratio: float,
    atr_over_price: float,
    config: dict,
) -> dict:
    """Win rate de equilibrio teórico contra el empírico."""
    raise NotImplementedError