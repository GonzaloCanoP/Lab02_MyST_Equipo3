"""Prueba de truncamiento del pipeline completo (P1): indicadores → señales → combinación → backtest.

Mientras `signals.py` sea stub se usa un stub local causal (`causal_signals` de test_backtest); al
integrarse `signals.py` a `main`, se cambia por `compute_indicators` → `generate_signals`.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from tests.test_backtest import BASE_PARAMS, causal_signals


def run_pipeline(prices: dict[str, pd.DataFrame], config: dict) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Corre el pipeline completo y devuelve señales, pesos y equity."""
    index = next(iter(prices.values())).index
    signals = causal_signals(prices)
    sleeve = pd.DataFrame(1 / len(prices), index=index, columns=list(prices))
    trade_params = pd.DataFrame(BASE_PARAMS, index=index)
    equity = run_backtest(prices, signals, sleeve, trade_params, config).equity
    return signals, sleeve, equity


@pytest.mark.parametrize("t", [100, 300, 599])
def test_pipeline_truncation(synthetic_prices, config_test, t):
    """Señales, pesos y equity en t no cambian al recalcular sobre df.iloc[:t+1] (CLAUDE.md, sección 4).

    Se comparan también las señales en t porque el equity en t solo depende de las señales hasta
    t − 1: una fuga de una barra en las señales no se vería en el equity.
    """
    full_signals, full_sleeve, full_equity = run_pipeline(synthetic_prices, config_test)
    trunc_signals, trunc_sleeve, trunc_equity = run_pipeline(
        {k: df.iloc[: t + 1] for k, df in synthetic_prices.items()}, config_test
    )
    date = full_equity.index[t]
    assert trunc_equity.index[-1] == date
    for key in full_signals:
        pd.testing.assert_series_equal(trunc_signals[key].loc[date], full_signals[key].loc[date])
    pd.testing.assert_series_equal(trunc_sleeve.loc[date], full_sleeve.loc[date])
    np.testing.assert_allclose(trunc_equity.iloc[-1], full_equity.iloc[t], rtol=0, atol=1e-9)
