"""Motor de backtesting event-driven (P1).

Único lugar del proyecto donde la señal de t se ejecuta en t+1 (CLAUDE.md, sección 4).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FILL_COLUMNS = [
    "date", "ticker", "side", "shares", "price", "notional", "commission", "slippage", "reason",
]
TRADE_COLUMNS = [
    "trade_id", "ticker", "side", "entry_date", "entry_price", "exit_date", "exit_price", "shares",
    "exit_reason", "regime_at_entry", "pnl_gross", "commission", "slippage", "borrow", "pnl_net",
]


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

    Notes
    -----
    Convenciones (SPEC punto 7 y CLAUDE.md, sección 5):

    - La barra t+1 ejecuta la señal de t en este orden: salidas por señal o por holding al Open,
      entradas al Open, gaps contra SL/TP, SL/TP intrabarra con High y Low, valuación al Close.
    - Toda entrada, incluida la reversa, se dimensiona con C = |w_t| · Equity_t al cierre de la barra
      de la señal t: acciones = ρ · C / (k · ATR₀), con tope acciones · E · (1 + comisión) ≤ C.
    - Holding máximo: la barra de entrada cuenta como la primera; con m barras cumplidas sale al Open
      de la siguiente.
    - Si `trade_params` trae la columna opcional "regime", se copia a `regime_at_entry`.
    """
    if config.get("resize_on_rebalance", False):
        raise ValueError("resize_on_rebalance=True no está implementado (SPEC_portafolio, P4)")

    tickers = list(prices)
    dates = prices[tickers[0]].index
    n_dates, n_assets = len(dates), len(tickers)

    def panel(df: pd.DataFrame) -> np.ndarray:
        return df.reindex(index=dates, columns=tickers).to_numpy(dtype=float)

    opens = np.column_stack([prices[t]["open"].to_numpy(dtype=float) for t in tickers])
    highs = np.column_stack([prices[t]["high"].to_numpy(dtype=float) for t in tickers])
    lows = np.column_stack([prices[t]["low"].to_numpy(dtype=float) for t in tickers])
    closes = np.column_stack([prices[t]["close"].to_numpy(dtype=float) for t in tickers])
    state = np.nan_to_num(panel(signals["state"]), nan=0.0)
    atr = panel(signals["atr"])
    sleeve = panel(sleeve_weights)
    params = trade_params.reindex(dates)
    k_stop = params["k_stop"].to_numpy(dtype=float)
    reward = params["reward_ratio"].to_numpy(dtype=float)
    max_hold = params["max_holding"].to_numpy(dtype=float)
    risk = params["risk_per_trade"].to_numpy(dtype=float)
    # Sin parámetros completos en la fecha de la señal no se abre (transparencia de NaN).
    params_ok = np.isfinite(np.column_stack([k_stop, reward, max_hold, risk])).all(axis=1)
    regime = params["regime"].to_numpy() if "regime" in params else np.full(n_dates, np.nan)
    if entry_mask is None:
        allowed = np.ones(n_dates, dtype=bool)
    else:
        allowed = entry_mask.reindex(dates).fillna(False).to_numpy(dtype=bool)

    commission = config["commission"]
    slippage = config["slippage"]
    daily_borrow = config["borrow_rate"] / config["periods_per_year"]

    cash = float(config["initial_capital"])
    # Una posición abierta por activo (sin pirámides); None si está plano.
    open_pos: list[dict | None] = [None] * n_assets
    # armed[j][lado]: lado ∈ {+1, −1}. Se desarma tras SL u holding máximo (SPEC punto 4).
    armed = [{1: True, -1: True} for _ in range(n_assets)]

    equity = np.empty(n_dates)
    cash_hist = np.empty(n_dates)
    shares_hist = np.zeros((n_dates, n_assets))
    cost_hist = np.zeros((n_dates, 3))  # commission, slippage, borrow
    fills: list[dict] = []
    trades: list[dict] = []

    def fill(i: int, j: int, signed_shares: float, ref_price: float, reason: str) -> float:
        """Ejecuta una orden con slippage en contra y comisión; devuelve el precio de llenado."""
        nonlocal cash
        direction = np.sign(signed_shares)
        price = ref_price * (1 + direction * slippage)
        notional = abs(signed_shares) * price
        fee = notional * commission
        slip = abs(signed_shares) * ref_price * slippage
        cash -= signed_shares * price + fee
        cost_hist[i, 0] += fee
        cost_hist[i, 1] += slip
        fills.append(
            {
                "date": dates[i],
                "ticker": tickers[j],
                "side": "buy" if direction > 0 else "sell",
                "shares": abs(signed_shares),
                "price": price,
                "notional": notional,
                "commission": fee,
                "slippage": slip,
                "reason": reason,
            }
        )
        return price

    def close_position(i: int, j: int, ref_price: float, reason: str) -> None:
        pos = open_pos[j]
        exit_price = fill(i, j, -pos["side"] * pos["shares"], ref_price, reason)
        last = fills[-1]
        commission_total = pos["commission"] + last["commission"]
        slippage_total = pos["slippage"] + last["slippage"]
        pnl_gross = pos["side"] * pos["shares"] * (ref_price - pos["entry_ref"])
        trades.append(
            {
                "trade_id": len(trades),
                "ticker": tickers[j],
                "side": "long" if pos["side"] > 0 else "short",
                "entry_date": dates[pos["entry_idx"]],
                "entry_price": pos["entry_price"],
                "exit_date": dates[i],
                "exit_price": exit_price,
                "shares": pos["shares"],
                "exit_reason": reason,
                "regime_at_entry": pos["regime"],
                "pnl_gross": pnl_gross,
                "commission": commission_total,
                "slippage": slippage_total,
                "borrow": pos["borrow"],
                "pnl_net": pnl_gross - commission_total - slippage_total - pos["borrow"],
            }
        )
        open_pos[j] = None
        if reason in ("stop", "max_holding"):
            armed[j][pos["side"]] = False

    for i in range(n_dates):
        if i > 0:
            s = i - 1  # barra de la señal

            # 1. Salidas por señal opuesta o por holding máximo, al Open.
            for j in range(n_assets):
                pos = open_pos[j]
                if pos is None:
                    continue
                if state[s, j] == -pos["side"]:
                    close_position(i, j, opens[i, j], "signal")
                elif i - pos["entry_idx"] >= pos["max_holding"]:
                    close_position(i, j, opens[i, j], "max_holding")

            # 2. Entradas al Open, dimensionadas con el equity al cierre de la barra de la señal.
            for j in range(n_assets):
                side = int(state[s, j])
                capital = sleeve[s, j] * equity[s]
                stop_dist = k_stop[s] * atr[s, j]
                if (
                    open_pos[j] is not None
                    or side == 0
                    or not armed[j][side]
                    or not allowed[s]
                    or not params_ok[s]
                    or not capital > 0
                    or not stop_dist > 0
                ):
                    continue
                entry_ref = opens[i, j]
                entry_est = entry_ref * (1 + side * slippage)
                n_shares = min(
                    risk[s] * capital / stop_dist,
                    capital / (entry_est * (1 + commission)),
                )
                entry_price = fill(i, j, side * n_shares, entry_ref, "entry")
                open_pos[j] = {
                    "side": side,
                    "shares": n_shares,
                    "entry_idx": i,
                    "entry_ref": entry_ref,
                    "entry_price": entry_price,
                    "stop": entry_price - side * stop_dist,
                    "target": entry_price + side * reward[s] * stop_dist,
                    "max_holding": max_hold[s],
                    "regime": regime[s],
                    "commission": fills[-1]["commission"],
                    "slippage": fills[-1]["slippage"],
                    "borrow": 0.0,
                }

        # 3-4. Gaps contra SL/TP (desde la barra siguiente a la entrada) y SL/TP intrabarra.
        for j in range(n_assets):
            pos = open_pos[j]
            if pos is None:
                continue
            side, stop, target = pos["side"], pos["stop"], pos["target"]
            if pos["entry_idx"] < i:
                if side * (opens[i, j] - stop) <= 0:
                    close_position(i, j, opens[i, j], "stop")
                    continue
                if side * (opens[i, j] - target) >= 0:
                    close_position(i, j, target, "target")
                    continue
            adverse = lows[i, j] if side > 0 else highs[i, j]
            favorable = highs[i, j] if side > 0 else lows[i, j]
            if side * (adverse - stop) <= 0:  # empate intrabarra: primero el stop
                close_position(i, j, stop, "stop")
            elif side * (favorable - target) >= 0:
                close_position(i, j, target, "target")

        # 5. Valuación al Close: borrow sobre el nocional en corto y rearme.
        for j in range(n_assets):
            pos = open_pos[j]
            if pos is not None:
                shares_hist[i, j] = pos["side"] * pos["shares"]
                if pos["side"] < 0:
                    fee = pos["shares"] * closes[i, j] * daily_borrow
                    cash -= fee
                    pos["borrow"] += fee
                    cost_hist[i, 2] += fee
            for side in (1, -1):
                if not armed[j][side] and state[i, j] != side:
                    armed[j][side] = True

        cash_hist[i] = cash
        equity[i] = cash + (shares_hist[i] * closes[i]).sum()

    exposure = shares_hist * closes
    return BacktestResult(
        equity=pd.Series(equity, index=dates, name="equity"),
        cash=pd.Series(cash_hist, index=dates, name="cash"),
        positions=pd.DataFrame(shares_hist, index=dates, columns=tickers),
        exposure=pd.DataFrame(exposure, index=dates, columns=tickers),
        fills=pd.DataFrame(fills, columns=FILL_COLUMNS),
        trades=pd.DataFrame(trades, columns=TRADE_COLUMNS),
        costs=pd.DataFrame(cost_hist, index=dates, columns=["commission", "slippage", "borrow"]),
    )


IMPACT_COLUMNS = [
    "trade_id", "ticker", "entry_date", "entry_notional", "entry_participation", "entry_bps",
    "exit_date", "exit_notional", "exit_participation", "exit_bps", "impact_cost",
]


def market_impact(trades: pd.DataFrame, prices: dict, config: dict) -> pd.DataFrame:
    """Impacto de mercado estimado ex post con el modelo de raíz cuadrada; no entra al motor.

    P1, tarea 10 (SPEC punto 6: el impacto se declara como limitación con su magnitud). Por cada
    llenado de una operación cerrada:

        impacto = σ_diaria · sqrt(Q / ADV)

    con Q el nocional del llenado, ADV la media del volumen en dólares (close · volume) y σ_diaria la
    desviación estándar de los retornos simples, ambas en una ventana de `config["adv_window"]` días
    al cierre de la barra de la señal (la anterior al llenado). Así la estimación es causal: un
    operador solo conoce esos valores al decidir la orden.

    Parameters
    ----------
    trades : pd.DataFrame
        `BacktestResult.trades`.
    prices : dict[str, pd.DataFrame]
        OHLCV por ticker con el mismo índice usado en el backtest.
    config : dict
        Usa `adv_window`.

    Returns
    -------
    pd.DataFrame
        Una fila por operación: nocional, participación (Q / ADV) e impacto en bps de la entrada y de
        la salida, e `impact_cost`, el costo total en USD de ambos llenados. NaN si la ventana aún no
        tiene datos suficientes.
    """
    window = config["adv_window"]
    rows = []
    for ticker, group in trades.groupby("ticker", sort=False):
        df = prices[ticker]
        sigma = df["close"].pct_change(fill_method=None).rolling(window, min_periods=window).std()
        adv = (df["close"] * df["volume"]).rolling(window, min_periods=window).mean()
        legs = {}
        for leg in ("entry", "exit"):
            signal_pos = df.index.get_indexer(group[f"{leg}_date"]) - 1
            notional = group["shares"].to_numpy() * group[f"{leg}_price"].to_numpy()
            participation = notional / adv.to_numpy()[signal_pos]
            legs[f"{leg}_date"] = group[f"{leg}_date"].to_numpy()
            legs[f"{leg}_notional"] = notional
            legs[f"{leg}_participation"] = participation
            legs[f"{leg}_bps"] = sigma.to_numpy()[signal_pos] * np.sqrt(participation) * 1e4
        legs["impact_cost"] = (
            legs["entry_notional"] * legs["entry_bps"] + legs["exit_notional"] * legs["exit_bps"]
        ) / 1e4
        rows.append(pd.DataFrame({"trade_id": group["trade_id"].to_numpy(), "ticker": ticker, **legs}))
    if not rows:
        return pd.DataFrame(columns=IMPACT_COLUMNS)
    return pd.concat(rows, ignore_index=True)[IMPACT_COLUMNS].sort_values("trade_id", ignore_index=True)
