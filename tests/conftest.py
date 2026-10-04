"""Fixtures sintéticos compartidos. Ninguna prueba lee `data/` ni usa red (CLAUDE.md, sección 8)."""

import copy

import numpy as np
import pandas as pd
import pytest

from main import CONFIG


def make_synthetic_prices(
    n_assets: int = 3, n_days: int = 600, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """OHLCV sintético coherente: low ≤ open, close ≤ high y volumen positivo.

    Close sigue un paseo aleatorio geométrico; el open sale de un gap sobre el close anterior y
    high/low se abren por encima y por debajo del rango open-close.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n_days, name="date")
    prices = {}
    for i in range(n_assets):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days)))
        prev_close = np.concatenate(([100.0], close[:-1]))
        open_ = prev_close * np.exp(rng.normal(0.0, 0.005, n_days))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.005, n_days)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.005, n_days)))
        volume = rng.integers(1_000_000, 5_000_000, n_days).astype(float)
        prices[f"A{i}"] = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
    return prices


@pytest.fixture
def synthetic_prices() -> dict[str, pd.DataFrame]:
    """3 activos sintéticos de 600 días con semilla 42."""
    return make_synthetic_prices(n_assets=3, n_days=600, seed=42)


@pytest.fixture
def config_test() -> dict:
    """Copia independiente de `CONFIG` para que una prueba no altere a otra."""
    return copy.deepcopy(CONFIG)
