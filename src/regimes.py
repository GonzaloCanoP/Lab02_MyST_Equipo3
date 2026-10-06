"""Detección de régimen de mercado: tendencia, reversión y crisis (P3).

La etiqueta que se opera es la FILTRADA (algoritmo forward). Viterbi solo se usa en la figura
comparativa (CLAUDE.md, sección 4).
"""

import pandas as pd


def regime_features(prices: dict, config: dict) -> pd.DataFrame:
    """Variables de régimen sobre la serie de mercado con ventana móvil de 63 días (SPEC_portafolio)."""
    raise NotImplementedError


def fit_regime_model(features_train: pd.DataFrame, method: str, seed: int) -> object:
    """Ajusta el clasificador con datos de entrenamiento; `method` ∈ {"rules", "kmeans", "hmm"}."""
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
