"""Pruebas del motor (P1): contabilidad (prueba 3 del lab) y golden-file tests.

Los golden-file tests usan caminos de un activo de pocas barras. El equity final de cada uno se
calculó a mano (la derivación está en el docstring) y se escribe como constante: nunca se obtiene
corriendo el código que se prueba.

Convención de los caminos: cada barra es (open, high, low, close) y `states[t]` es la señal al cierre
de t, que se ejecuta al Open de t+1. Salvo que se indique, capital 100,000, sin costos, ATR = 1,
k = 2, r = 2, ρ = 1% y m = 20; así la distancia al stop es 2, el TP está a 4 y el tamaño es
0.01 · 100,000 / 2 = 500 acciones.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest

BASE_PARAMS = {"k_stop": 2.0, "reward_ratio": 2.0, "max_holding": 20, "risk_per_trade": 0.01}


def flat(price: float) -> tuple[float, float, float, float]:
    """Barra sin rango: open = high = low = close."""
    return (price, price, price, price)


def run_path(config, bars, states, atr=1.0, params=None, costs=False, config_overrides=None):
    """Corre `run_backtest` sobre un camino de un activo construido a mano."""
    index = pd.bdate_range("2020-01-01", periods=len(bars), name="date")
    prices = {"X": pd.DataFrame(bars, index=index, columns=["open", "high", "low", "close"])}
    prices["X"]["volume"] = 1_000_000.0
    signals = {
        "state": pd.DataFrame({"X": states}, index=index, dtype=float),
        "atr": pd.DataFrame({"X": atr}, index=index, dtype=float),
    }
    sleeve = pd.DataFrame({"X": 1.0}, index=index)
    trade_params = pd.DataFrame({**BASE_PARAMS, **(params or {})}, index=index)
    cfg = copy.deepcopy(config)
    cfg["initial_capital"] = 100_000.0
    if not costs:
        cfg.update(commission=0.0, slippage=0.0, borrow_rate=0.0)
    cfg.update(config_overrides or {})
    return run_backtest(prices, signals, sleeve, trade_params, cfg)


# ---------------------------------------------------------------------------- contabilidad


def causal_signals(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Stub local de señales causal: cruce de medias 5/20 y ATR como media de 14 días de high − low."""
    state, atr = {}, {}
    for ticker, df in prices.items():
        diff = df["close"].rolling(5).mean() - df["close"].rolling(20).mean()
        state[ticker] = np.sign(diff).fillna(0.0)
        atr[ticker] = (df["high"] - df["low"]).rolling(14).mean()
    return {"state": pd.DataFrame(state), "atr": pd.DataFrame(atr)}


@pytest.fixture
def synthetic_run(synthetic_prices, config_test):
    """Backtest de 3 activos sintéticos con costos completos, largos y cortos."""
    index = synthetic_prices["A0"].index
    tickers = list(synthetic_prices)
    signals = causal_signals(synthetic_prices)
    sleeve = pd.DataFrame(1 / 3, index=index, columns=tickers)
    trade_params = pd.DataFrame(BASE_PARAMS, index=index)
    result = run_backtest(synthetic_prices, signals, sleeve, trade_params, config_test)
    return synthetic_prices, signals, sleeve, trade_params, result


def test_accounting_equity_equals_cash_plus_positions(synthetic_run):
    """Prueba 3: equity == cash + Σ shares · close en todas las fechas (tol 1e-6)."""
    prices, _, _, _, result = synthetic_run
    closes = pd.DataFrame({t: df["close"] for t, df in prices.items()})
    marked = result.cash + (result.positions * closes).sum(axis=1)
    np.testing.assert_allclose(result.equity, marked, rtol=0, atol=1e-6)
    assert (result.trades["side"] == "long").any() and (result.trades["side"] == "short").any()


def test_accounting_commission_matches_notional(synthetic_run, config_test):
    """Prueba 3: Σ commission == Σ |notional| · 0.00125 (tol 1e-6)."""
    *_, result = synthetic_run
    expected = result.fills["notional"].abs().sum() * config_test["commission"]
    assert result.fills["commission"].sum() == pytest.approx(expected, abs=1e-6)
    assert result.costs["commission"].sum() == pytest.approx(expected, abs=1e-6)


def test_run_backtest_does_not_modify_inputs(synthetic_prices, config_test):
    """`run_backtest` es pura: no modifica sus argumentos."""
    index = synthetic_prices["A0"].index
    signals = causal_signals(synthetic_prices)
    sleeve = pd.DataFrame(1 / 3, index=index, columns=list(synthetic_prices))
    trade_params = pd.DataFrame(BASE_PARAMS, index=index)
    before = copy.deepcopy((synthetic_prices, signals, sleeve, trade_params, config_test))
    run_backtest(synthetic_prices, signals, sleeve, trade_params, config_test)
    for ticker in synthetic_prices:
        pd.testing.assert_frame_equal(synthetic_prices[ticker], before[0][ticker])
    for key in signals:
        pd.testing.assert_frame_equal(signals[key], before[1][key])
    pd.testing.assert_frame_equal(sleeve, before[2])
    pd.testing.assert_frame_equal(trade_params, before[3])
    assert config_test == before[4]


# ---------------------------------------------------------------------------- golden files


def test_golden_1_sl_and_tp_same_bar_exits_at_stop(config_test):
    """SL y TP en la misma barra → sale por stop (SPEC punto 7, empate intrabarra).

    Entra largo al Open de la barra 1 en 100: SL = 98, TP = 104, 500 acciones. La barra 2 toca
    97 y 105: se ejecuta el stop en 98. Equity = 100,000 + 500 · (98 − 100) = 99,000.
    """
    bars = [flat(100), (100, 101, 99, 100), (100, 105, 97, 100), flat(100)]
    result = run_path(config_test, bars, [1, 0, 0, 0])
    assert result.equity.iloc[-1] == pytest.approx(99_000.0)
    assert result.trades["exit_reason"].tolist() == ["stop"]


def test_golden_2_gap_through_stop_fills_at_open(config_test):
    """Gap que abre más allá del SL → llena al Open (SPEC punto 7).

    Largo en 100 con SL = 98. La barra 2 abre en 95: sale en 95.
    Equity = 100,000 + 500 · (95 − 100) = 97,500.
    """
    bars = [flat(100), (100, 101, 99, 100), (95, 96, 94, 95), flat(95)]
    result = run_path(config_test, bars, [1, 0, 0, 0])
    assert result.equity.iloc[-1] == pytest.approx(97_500.0)
    assert result.trades["exit_price"].tolist() == [95.0]


def test_golden_3_gap_through_target_fills_at_target(config_test):
    """Gap que abre más allá del TP → llena en el nivel del TP, sin mejora (SPEC punto 7).

    Largo en 100 con TP = 104. La barra 2 abre en 110: sale en 104.
    Equity = 100,000 + 500 · (104 − 100) = 102,000.
    """
    bars = [flat(100), (100, 101, 99, 100), (110, 111, 109, 110), flat(110)]
    result = run_path(config_test, bars, [1, 0, 0, 0])
    assert result.equity.iloc[-1] == pytest.approx(102_000.0)
    assert result.trades["exit_reason"].tolist() == ["target"]


def test_golden_3b_target_gap_with_full_costs(config_test):
    """El mismo camino del golden 3 con comisión de 0.125% y slippage de 2 bps (SPEC punto 6).

    Entrada: E = 100 · 1.0002 = 100.02; SL = 98.02; TP = 104.02; 500 acciones (el tope
    100,000 / (100.02 · 1.00125) ≈ 998.5 no aplica). Nocional 50,010; comisión 62.5125.
    Cash = 100,000 − 50,010 − 62.5125 = 49,927.4875.
    Salida en el TP con slippage en contra: 104.02 · 0.9998 = 103.999196. Nocional 51,999.598;
    comisión 64.9994975. Equity = 49,927.4875 + 51,999.598 − 64.9994975 = 101,862.0860025.
    """
    bars = [flat(100), (100, 101, 99, 100), (110, 111, 109, 110), flat(110)]
    result = run_path(config_test, bars, [1, 0, 0, 0], costs=True)
    assert result.equity.iloc[-1] == pytest.approx(101_862.0860025, abs=1e-6)
    assert result.trades["commission"].iloc[0] == pytest.approx(127.5119975, abs=1e-6)


def test_golden_4a_rearm_after_target_reenters_next_bar(config_test):
    """Tras un TP el lado queda armado: reentra a la barra siguiente (SPEC punto 4, rearme).

    Largo en 100, TP en 104 durante la barra 2 → equity 102,000. La señal sigue en +1 al cierre de
    la barra 2, así que reentra al Open de la barra 3 en 104 con 0.01 · 102,000 / 2 = 510 acciones.
    La barra 4 cierra en 105: equity = 102,000 + 510 · (105 − 104) = 102,510.
    """
    bars = [flat(100), (100, 101, 99, 100), (100, 104.5, 99.5, 104), flat(104), (104, 105, 104, 105)]
    result = run_path(config_test, bars, [1, 1, 1, 0, 0])
    assert result.equity.iloc[-1] == pytest.approx(102_510.0)
    assert result.positions["X"].iloc[-1] == pytest.approx(510.0)


def test_golden_4b_no_reentry_after_stop_until_state_changes(config_test):
    """Tras un SL el lado se desarma hasta la primera barra con Estado ≠ L (SPEC punto 4, rearme).

    Largo en 100, stop en 98 durante la barra 2 → equity 99,000. La señal sigue en +1 en las
    barras 2 y 3 (no reentra), pasa a 0 en la barra 4 (se rearma) y vuelve a +1 en la barra 5.
    Reentra al Open de la barra 6 en 99 con 0.01 · 99,000 / 2 = 495 acciones; cierra en 100.
    Equity = 99,000 + 495 · (100 − 99) = 99,495. Si hubiera reentrado en la barra 3 o 4, el alza de
    la barra 4 cambiaría el resultado.
    """
    bars = [
        flat(100),
        (100, 101, 99, 100),
        (100, 100, 97, 98),
        flat(98),
        (98, 99, 98, 99),
        flat(99),
        (99, 100, 99, 100),
    ]
    result = run_path(config_test, bars, [1, 1, 1, 1, 0, 1, 0])
    assert result.equity.iloc[-1] == pytest.approx(99_495.0)
    assert len(result.fills) == 3


def test_golden_5_max_holding_exits_at_open_of_bar_m_plus_1(config_test):
    """Holding máximo m = 3 → cierra al Open de la barra m + 1 = 4 (SPEC punto 4).

    Largo en el Open de la barra 1 en 100. La barra 1 es la primera de las 3; tras el cierre de la
    barra 3 ya cumplió m y sale al Open de la barra 4 en 103. La señal sigue en +1 pero el lado
    queda desarmado. Equity = 100,000 + 500 · (103 − 100) = 101,500.
    """
    bars = [flat(100), flat(100), flat(101), flat(102), flat(103), flat(104)]
    result = run_path(config_test, bars, [1, 1, 1, 1, 1, 1], params={"max_holding": 3})
    assert result.equity.iloc[-1] == pytest.approx(101_500.0)
    assert result.trades["exit_reason"].tolist() == ["max_holding"]
    assert result.trades["exit_date"].iloc[0] == result.equity.index[4]


def test_golden_6_opposite_signal_reverses_at_same_open(config_test):
    """Señal opuesta → cierra y abre el lado contrario en el mismo Open (SPEC punto 4).

    Largo en 100 (500 acciones). Equity al cierre de la barra 2 = 100,000 + 500 · 1 = 100,500.
    La señal −1 de la barra 2 cierra el largo al Open de la barra 3 en 102 (+1,000) y abre un corto
    en 102 con 0.01 · 100,500 / 2 = 502.5 acciones. La barra 3 cierra en 101:
    equity = 101,000 + 502.5 · (102 − 101) = 101,502.5.
    """
    bars = [flat(100), (100, 100.5, 99.5, 100), (101, 101.5, 100.5, 101), (102, 102.5, 101.5, 101)]
    result = run_path(config_test, bars, [1, 0, -1, 0])
    assert result.equity.iloc[-1] == pytest.approx(101_502.5)
    assert result.positions["X"].iloc[-1] == pytest.approx(-502.5)
    assert result.trades["exit_reason"].tolist() == ["signal"]


def test_golden_7_no_leverage_cap_trims_size(config_test):
    """El tope sin apalancamiento recorta el tamaño (SPEC punto 5).

    Con ATR = 0.25 la distancia al stop es 0.5 y el riesgo pediría 0.01 · 100,000 / 0.5 = 2,000
    acciones (200,000 de nocional). El tope es 100,000 / 100 = 1,000 acciones.
    Equity = 100,000 + 1,000 · (100.5 − 100) = 100,500.
    """
    bars = [flat(100), (100, 100.2, 99.8, 100.2), (100.4, 100.5, 100.3, 100.5)]
    result = run_path(config_test, bars, [1, 0, 0], atr=0.25)
    assert result.equity.iloc[-1] == pytest.approx(100_500.0)
    assert result.positions["X"].iloc[-1] == pytest.approx(1_000.0)


def test_golden_8_borrow_accrues_only_on_short_days(config_test):
    """El borrow se devenga solo en los días en corto (SPEC punto 6).

    Tasa de borrow 25.2% anual → 0.1% diario, para que las cifras sean redondas. Corto de 500
    acciones en 100 en las barras 1 y 2 (m = 2, sale al Open de la barra 3): 2 · 500 · 100 · 0.001
    = 100. Después un largo plano en las barras 4 y 5, que no paga borrow.
    Equity = 100,000 − 100 = 99,900.
    """
    bars = [flat(100), (100, 100.5, 99.5, 100), flat(100), flat(100), flat(100), flat(100)]
    result = run_path(
        config_test,
        bars,
        [-1, -1, 0, 1, 0, 0],
        params={"max_holding": 2},
        config_overrides={"borrow_rate": 0.252},
    )
    assert result.equity.iloc[-1] == pytest.approx(99_900.0)
    assert result.costs["borrow"].sum() == pytest.approx(100.0)
    assert (result.costs["borrow"] > 0).tolist() == [False, True, True, False, False, False]
    assert result.positions["X"].iloc[-1] > 0


def test_golden_9_signal_at_t_executes_at_t_plus_1(config_test):
    """La señal de t se ejecuta en t+1 y nunca en t (SPEC punto 7).

    La señal +1 aparece al cierre de la barra 1 (que sube de 100 a 105). Si se ejecutara en t
    capturaría ese alza; se ejecuta al Open de la barra 2 en 105 con 500 acciones y cierra en 106.
    Equity = 100,000 + 500 · (106 − 105) = 100,500.
    """
    bars = [flat(100), (100, 105, 100, 105), (105, 106, 105, 106)]
    result = run_path(config_test, bars, [0, 1, 0])
    assert result.equity.iloc[-1] == pytest.approx(100_500.0)
    assert result.fills["date"].tolist() == [result.equity.index[2]]
    assert result.fills["price"].tolist() == [105.0]
