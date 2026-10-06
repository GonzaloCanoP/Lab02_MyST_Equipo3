"""Detección de régimen de mercado: tendencia, reversión y crisis (P3).

La etiqueta que se opera es la FILTRADA (algoritmo forward). Viterbi solo se usa en la figura
comparativa (CLAUDE.md, sección 4).
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["volatility", "efficiency", "autocorr"]


def _market_log_returns(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """Log-retorno diario del índice equiponderado: ℓ_t = ln(1 + media_i r_i,t).

    SPEC_portafolio, Régimen, "Serie de mercado". El índice se rebalancea a diario, así que su
    retorno simple es la media de los retornos simples. `skipna=False`: si falta un activo en t, el
    retorno del índice queda NaN en vez de promediar con menos activos.
    """
    close = pd.DataFrame({ticker: df["close"] for ticker, df in prices.items()})
    simple = close.pct_change(fill_method=None).mean(axis=1, skipna=False)
    return np.log1p(simple).rename("market")


def regime_features(prices: dict, config: dict) -> pd.DataFrame:
    """Variables de régimen sobre la serie de mercado con ventana móvil hacia atrás.

    SPEC_portafolio, Régimen, "Variables". Con w = `config["regime_window"]` y k = t − w + 1, …, t:

    - volatility = std(ℓ_k) · √252
    - efficiency = |Σ ℓ_k| / Σ |ℓ_k|  (razón de eficiencia de Kaufman, en [0, 1])
    - autocorr   = corr(ℓ_k, ℓ_k−1)

    Todas las ventanas miran solo al pasado, así que el valor en t usa datos hasta el cierre de t.
    Mientras no hay w observaciones el valor es NaN; no se rellena.

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        Un DataFrame OHLCV por ticker con índice común (salida de `load_prices`).
    config : dict
        Usa `regime_window` y `periods_per_year`.

    Returns
    -------
    pd.DataFrame
        Columnas `volatility`, `efficiency` y `autocorr`, con el índice de `prices`.
    """
    window = config["regime_window"]
    log_ret = _market_log_returns(prices)
    rolling = log_ret.rolling(window, min_periods=window)
    volatility = rolling.std() * np.sqrt(config["periods_per_year"])
    efficiency = rolling.sum().abs() / log_ret.abs().rolling(window, min_periods=window).sum()
    # El rezago ℓ_k−1 mira al pasado: no es un desplazamiento para simular ejecución (CLAUDE.md §4).
    autocorr = rolling.corr(log_ret.shift(1))
    return pd.DataFrame(
        {"volatility": volatility, "efficiency": efficiency, "autocorr": autocorr},
        index=log_ret.index,
    )[FEATURE_COLUMNS]


def fit_regime_model(
    features_train: pd.DataFrame, method: str, seed: int, config: dict
) -> object:
    """Ajusta el clasificador con datos de entrenamiento; `method` ∈ {"rules", "kmeans", "hmm"}.

    `config` aporta los valores fijos del clasificador (cuantiles, `n_init`, covarianza del HMM).
    """
    raise NotImplementedError


def predict_regimes(model: object, features: pd.DataFrame) -> pd.Series:
    """Etiqueta FILTRADA por fecha (recursión forward; nunca `predict` ni `predict_proba`)."""
    raise NotImplementedError


def viterbi_path(model: object, features: pd.DataFrame) -> pd.Series:
    """Ruta de Viterbi. Usa el futuro: SOLO para la figura comparativa, nunca para operar."""
    raise NotImplementedError


def label_regimes(prices: dict, config: dict) -> pd.Series:
    """Etiqueta causal para todas las fechas, con reajuste mensual y datos hasta cada fecha."""
    raise NotImplementedError


def regime_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    blocks: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> dict:
    """Silhouette, duración media, transiciones por mes y % de tiempo por régimen y por bloque.

    `blocks` es la salida de `block_dates(config)`; sin él, el % de tiempo se reporta solo para
    toda la muestra.
    """
    raise NotImplementedError
