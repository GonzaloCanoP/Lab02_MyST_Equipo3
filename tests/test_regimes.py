"""Pruebas de régimen (P3): truncamiento de variables y cálculo contra fórmula directa.

Datos sintéticos de `conftest.py`; ninguna prueba lee `data/` ni usa red (CLAUDE.md, sección 8).
"""

import numpy as np
import pytest

from src.regimes import FEATURE_COLUMNS, regime_features


@pytest.mark.parametrize("t", [100, 300, 599])
def test_features_truncation(synthetic_prices, config_test, t):
    """Cada variable en t no cambia al recalcular sobre df.iloc[:t+1] (prueba 4 del lab)."""
    full = regime_features(synthetic_prices, config_test)
    truncated = regime_features(
        {ticker: df.iloc[: t + 1] for ticker, df in synthetic_prices.items()}, config_test
    )
    assert truncated.index[-1] == full.index[t]
    for column in FEATURE_COLUMNS:
        np.testing.assert_allclose(
            truncated[column].iloc[-1], full[column].iloc[t], rtol=0, atol=1e-12, err_msg=column
        )


def test_features_match_formula(synthetic_prices, config_test):
    """En la última fecha, las variables coinciden con la fórmula del SPEC calculada con numpy."""
    window = config_test["regime_window"]
    close = np.column_stack([df["close"].to_numpy() for df in synthetic_prices.values()])
    log_ret = np.log1p((close[1:] / close[:-1] - 1).mean(axis=1))
    last = log_ret[-window:]
    lagged = log_ret[-window - 1 : -1]

    features = regime_features(synthetic_prices, config_test).iloc[-1]

    np.testing.assert_allclose(
        features["volatility"], last.std(ddof=1) * np.sqrt(252), rtol=1e-10
    )
    np.testing.assert_allclose(
        features["efficiency"], abs(last.sum()) / np.abs(last).sum(), rtol=1e-10
    )
    np.testing.assert_allclose(features["autocorr"], np.corrcoef(last, lagged)[0, 1], rtol=1e-10)


def test_features_nan_until_window(synthetic_prices, config_test):
    """Sin w observaciones la variable es NaN, sin rellenar; después ya no hay NaN."""
    window = config_test["regime_window"]
    features = regime_features(synthetic_prices, config_test)
    # El primer retorno existe en la fila 1; volatility y efficiency necesitan w retornos y
    # autocorr uno más, por el rezago.
    first_valid = {"volatility": window, "efficiency": window, "autocorr": window + 1}
    for column, first in first_valid.items():
        assert features[column].iloc[:first].isna().all(), column
        assert features[column].iloc[first:].notna().all(), column


def test_efficiency_in_unit_interval(synthetic_prices, config_test):
    """La razón de eficiencia está en [0, 1] por construcción (desigualdad del triángulo)."""
    efficiency = regime_features(synthetic_prices, config_test)["efficiency"].dropna()
    assert ((efficiency >= 0) & (efficiency <= 1)).all()
