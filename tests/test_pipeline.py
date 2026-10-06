"""Prueba de truncamiento del pipeline completo.

Indicadores -> señales -> combinación -> backtest.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.signals import generate_signals


TRADE_PARAM_KEYS = [
    "k_stop",
    "reward_ratio",
    "max_holding",
]


def run_pipeline(
    prices: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.Series,
]:
    """Corre señales reales y backtest para la prueba de causalidad."""

    index = next(
        iter(prices.values())
    ).index

    tickers = list(prices)

    regimes = pd.Series(
        "tendencia",
        index=index,
        name="regime",
    )

    params_by_regime = {
        "tendencia":
            config["base_params"]
    }

    signals = generate_signals(
        prices,
        params_by_regime,
        regimes,
        config,
    )

    sleeve = pd.DataFrame(
        1 / len(tickers),
        index=index,
        columns=tickers,
    )

    trade_params = pd.DataFrame(
        {
            key:
                config[
                    "base_params"
                ][key]
            for key
            in TRADE_PARAM_KEYS
        },
        index=index,
    )

    equity = run_backtest(
        prices,
        signals,
        sleeve,
        trade_params,
        config,
    ).equity

    return (
        signals,
        sleeve,
        equity,
    )


@pytest.mark.parametrize(
    "t",
    [
        150,
        400,
        599,
    ],
)
def test_pipeline_truncation(
    synthetic_prices,
    config_test,
    t,
):
    """El pipeline en t no puede cambiar al agregar datos futuros (CLAUDE.md, sección 4).

    Se comparan también las señales en t porque el equity en t solo depende de las señales hasta
    t − 1: una fuga de una barra en las señales no se vería en el equity.
    """

    (
        full_signals,
        full_sleeve,
        full_equity,
    ) = run_pipeline(
        synthetic_prices,
        config_test,
    )

    truncated_prices = {
        ticker:
            df.iloc[: t + 1]
        for ticker, df
        in synthetic_prices.items()
    }

    (
        trunc_signals,
        trunc_sleeve,
        trunc_equity,
    ) = run_pipeline(
        truncated_prices,
        config_test,
    )

    date = full_equity.index[t]

    assert (
        trunc_equity.index[-1]
        == date
    )

    for key in full_signals:

        pd.testing.assert_series_equal(
            trunc_signals[
                key
            ].loc[date],
            full_signals[
                key
            ].loc[date],
        )

    pd.testing.assert_series_equal(
        trunc_sleeve.loc[date],
        full_sleeve.loc[date],
    )

    np.testing.assert_allclose(
        trunc_equity.iloc[-1],
        full_equity.iloc[t],
        rtol=0,
        atol=1e-9,
    )