import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestResult
from src.metrics import (
    breakeven_winrate,
    compute_metrics,
    drawdown_series,
    exposure_metrics,
    returns_table,
)


def test_drawdown_series_matches_manual_calculation():
    equity = pd.Series(
        [100.0, 105.0, 102.0, 110.0, 108.0, 115.0, 111.0, 118.0, 116.0, 120.0],
        index=pd.bdate_range("2020-01-01", periods=10),
    )
    expected = (equity / equity.cummax() - 1).rename("drawdown")
    result = drawdown_series(equity)
    pd.testing.assert_series_equal(result, expected)


def test_compute_metrics_matches_manual_calculation():
    equity = pd.Series(
        [100.0, 105.0, 102.0, 110.0, 108.0, 115.0, 111.0, 118.0, 116.0, 120.0],
        index=pd.bdate_range("2020-01-01", periods=10),
    )
    trades = pd.DataFrame({"pnl_net": [100.0, -50.0, 200.0, -100.0]})
    periods = 252

    returns = equity.pct_change(fill_method=None).dropna()
    expected_ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (periods / len(returns)) - 1
    expected_ann_vol = returns.std(ddof=1) * np.sqrt(periods)
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(periods)
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside_deviation = np.sqrt(np.mean(downside**2))
    expected_sortino = returns.mean() / downside_deviation * np.sqrt(periods)
    expected_mdd = (equity / equity.cummax() - 1).min()
    expected_calmar = expected_ann_return / abs(expected_mdd)
    result = compute_metrics(equity, trades, rf=0.0, periods_per_year=periods)
    assert result["ann_return"] == pytest.approx(expected_ann_return, abs=1e-9)
    assert result["ann_vol"] == pytest.approx(expected_ann_vol, abs=1e-9)
    assert result["sharpe"] == pytest.approx(expected_sharpe, abs=1e-9)
    assert result["sortino"] == pytest.approx(expected_sortino, abs=1e-9)
    assert result["max_drawdown"] == pytest.approx(expected_mdd, abs=1e-9)
    assert result["calmar"] == pytest.approx(expected_calmar, abs=1e-9)
    assert result["n_trades"] == 4
    assert result["win_rate"] == pytest.approx(0.5)
    assert result["payoff_ratio"] == pytest.approx(2.0)


def test_returns_table_compounds_returns():
    dates = pd.to_datetime(["2020-01-29", "2020-01-30", "2020-01-31", "2020-02-03", "2020-02-04"])
    equity = pd.Series([100.0, 105.0, 110.0, 115.5, 121.0], index=dates)
    tables = returns_table(equity)
    assert set(tables) == {"mensual", "trimestral", "anual"}
    assert tables["mensual"].loc[2020, 1] == pytest.approx(0.10)
    assert tables["mensual"].loc[2020, 2] == pytest.approx(0.10)
    assert tables["trimestral"].loc[2020, 1] == pytest.approx(0.21)
    assert tables["anual"].loc[2020, "Retorno"] == pytest.approx(0.21)


def test_exposure_metrics():
    index = pd.bdate_range("2020-01-01", periods=4)
    positions = pd.DataFrame({"A": [0.0, 1.0, 1.0, 0.0], "B": [0.0, 0.0, -1.0, -1.0]}, index=index)
    trades = pd.DataFrame({"exit_reason": ["target", "stop"], "pnl_net": [100.0, -50.0]})
    result = BacktestResult(
        equity=pd.Series([100.0, 101.0, 102.0, 103.0], index=index),
        cash=pd.Series(100.0, index=index),
        positions=positions,
        exposure=positions.copy(),
        fills=pd.DataFrame(),
        trades=trades,
        costs=pd.DataFrame(0.0, index=index, columns=["commission", "slippage", "borrow"]),
    )
    metrics = exposure_metrics(result)
    assert metrics["time_in_market_by_asset"]["A"] == pytest.approx(0.5)
    assert metrics["time_in_market_by_asset"]["B"] == pytest.approx(0.5)
    assert metrics["time_in_market_portfolio"] == pytest.approx(0.75)
    assert metrics["trades_per_month"] == pytest.approx(2.0)
    assert metrics["exit_reasons"]["target"] == 1
    assert metrics["exit_reasons"]["stop"] == 1


def test_breakeven_winrate_matches_spec_formula():
    trades = pd.DataFrame({"pnl_net": [100.0, -50.0, 200.0, -100.0]})
    config = {"commission": 0.00125, "slippage": 0.0002}
    k_stop = 2.0
    reward_ratio = 2.0
    atr_over_price = 0.017
    cost_in_r = 2.0 * (config["commission"] + config["slippage"]) / (k_stop * atr_over_price)
    expected_theoretical = (1.0 + cost_in_r) / (1.0 + reward_ratio)
    result = breakeven_winrate(trades, k_stop, reward_ratio, atr_over_price, config)
    assert result["theoretical_win_rate"] == pytest.approx(expected_theoretical)
    assert result["observed_win_rate"] == pytest.approx(0.5)
    assert result["payoff_ratio"] == pytest.approx(2.0)
    assert result["empirical_breakeven_win_rate"] == pytest.approx(1.0 / 3.0)


def test_metrics_by_block_starts_at_previous_close():
    """El bloque usa el cierre anterior como base y las operaciones por fecha de entrada."""
    from src.metrics import metrics_by_block

    index = pd.bdate_range("2020-01-01", periods=6)
    equity = pd.Series([100.0, 110.0, 121.0, 108.9, 119.79, 131.769], index=index)
    trades = pd.DataFrame({"entry_date": [index[1], index[4]], "pnl_net": [10.0, -5.0]})
    blocks = {"a": (index[0], index[2]), "b": (index[3], index[5])}
    table = metrics_by_block(equity, trades, blocks)
    # Bloque b: 121 → 108.9 → 119.79 → 131.769, tres retornos (−10%, +10%, +10%).
    expected_b = compute_metrics(equity.iloc[2:], trades.iloc[[1]])
    assert table.loc["b", "ann_return"] == pytest.approx(expected_b["ann_return"], abs=1e-12)
    assert table.loc["b", "max_drawdown"] == pytest.approx(-0.10, abs=1e-12)
    assert table.loc["a", "n_trades"] == 1
    assert table.loc["b", "win_rate"] == 0.0


def test_buy_and_hold_equity_by_hand(config_test):
    """Dos activos, 50/50: compra al Open con slippage y comisión, valúa al Close."""
    from src.metrics import buy_and_hold_equity

    index = pd.bdate_range("2020-01-01", periods=2)
    prices = {
        "X": pd.DataFrame({"open": [100.0, 0], "close": [110.0, 120.0]}, index=index),
        "Y": pd.DataFrame({"open": [50.0, 0], "close": [50.0, 40.0]}, index=index),
    }
    config = dict(config_test, initial_capital=1000.0, commission=0.001, slippage=0.0)
    equity = buy_and_hold_equity(prices, config, index[0], index[1])
    # X: 500 / (100 · 1.001) = 4.995005 acciones; Y: 500 / (50 · 1.001) = 9.990010 acciones.
    assert equity.iloc[0] == pytest.approx(4.995005 * 110 + 9.990010 * 50, abs=1e-4)
    assert equity.iloc[1] == pytest.approx(4.995005 * 120 + 9.990010 * 40, abs=1e-4)
