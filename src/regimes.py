"""Detección de régimen de mercado: tendencia, reversión y crisis (P3).

La etiqueta que se opera es la FILTRADA (algoritmo forward). Viterbi solo se usa en la figura
comparativa (CLAUDE.md, sección 4).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

FEATURE_COLUMNS = ["volatility", "efficiency", "autocorr"]


@dataclass
class RegimeModel:
    """Clasificador de régimen ajustado (salida de `fit_regime_model`).

    Guarda todo lo necesario para etiquetar fechas nuevas sin volver a ver la muestra de ajuste:
    la estandarización, el estimador, el nombre de cada grupo y, en reglas, los umbrales.
    """

    method: str
    mean: pd.Series
    std: pd.Series
    names: dict[int, str] = field(default_factory=dict)
    estimator: KMeans | GaussianHMM | None = None
    thresholds: dict[str, float] = field(default_factory=dict)


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

    SPEC_portafolio, Régimen, "Métodos comparados". Solo se usan las filas sin NaN. La media y la
    desviación de esas filas estandarizan las variables de K-means y del HMM, y se guardan en el
    modelo para estandarizar después cualquier fecha con los mismos valores.

    Parameters
    ----------
    features_train : pd.DataFrame
        Salida de `regime_features` restringida a la muestra de ajuste.
    method : str
        "rules", "kmeans" o "hmm".
    seed : int
        `random_state` de K-means y del HMM.
    config : dict
        Usa `regime_n_states`, `regime_crisis_quantile`, `regime_trend_quantile`,
        `regime_kmeans_n_init`, `regime_hmm_covariance` y `regime_hmm_n_iter`.

    Returns
    -------
    RegimeModel
    """
    if config["regime_n_states"] != 3:
        raise ValueError("La regla de nombres está definida para 3 regímenes.")
    valid = features_train[FEATURE_COLUMNS].dropna()
    if valid.empty:
        raise ValueError("No hay observaciones válidas para ajustar el modelo.")
    mean, std = valid.mean(), valid.std()

    if method == "rules":
        thresholds = {
            "volatility": valid["volatility"].quantile(config["regime_crisis_quantile"]),
            "efficiency": valid["efficiency"].quantile(config["regime_trend_quantile"]),
        }
        return RegimeModel(method, mean, std, thresholds=thresholds)

    z = ((valid - mean) / std).to_numpy()
    if method == "kmeans":
        estimator = KMeans(
            n_clusters=config["regime_n_states"],
            n_init=config["regime_kmeans_n_init"],
            random_state=seed,
        ).fit(z)
        centers_z = estimator.cluster_centers_
    elif method == "hmm":
        estimator = GaussianHMM(
            n_components=config["regime_n_states"],
            covariance_type=config["regime_hmm_covariance"],
            n_iter=config["regime_hmm_n_iter"],
            random_state=seed,
        ).fit(z)
        centers_z = estimator.means_
    else:
        raise ValueError(f"Método desconocido: {method}")

    # Los centroides se llevan a unidades originales para nombrarlos (SPEC_portafolio, "Nombres").
    centers = pd.DataFrame(centers_z * std.to_numpy() + mean.to_numpy(), columns=FEATURE_COLUMNS)
    return RegimeModel(method, mean, std, names=_name_states(centers), estimator=estimator)


def _name_states(centers: pd.DataFrame) -> dict[int, str]:
    """Regla determinista de nombres sobre los centroides en unidades originales.

    SPEC_portafolio, Régimen, "Nombres de los regímenes": mayor volatilidad → crisis; de los otros
    dos, mayor eficiencia → tendencia; el restante → reversion. Evita que los nombres se
    intercambien entre reajustes, porque los números de grupo pueden cambiar en cada ajuste.
    """
    crisis = centers["volatility"].idxmax()
    rest = centers.drop(index=crisis)
    trend = rest["efficiency"].idxmax()
    reversion = rest.drop(index=trend).index[0]
    return {int(crisis): "crisis", int(trend): "tendencia", int(reversion): "reversion"}


def _filtered_log_probs(hmm: GaussianHMM, z: np.ndarray) -> np.ndarray:
    """Recursión forward: ln P(s_t = j | x_1, …, x_t) para cada t y cada estado j.

    SPEC_portafolio, Régimen, HMM. En logaritmos para no tener underflow:

        ln α_1(j) = ln π_j + ln b_j(x_1)
        ln α_t(j) = ln b_j(x_t) + logsumexp_i (ln α_t−1(i) + ln A_ij)

    y en cada t se normaliza para que Σ_j α_t(j) = 1. Solo usa x hasta t: a diferencia de
    `predict_proba` (forward-backward) y `predict` (Viterbi) de hmmlearn, no mira al futuro.
    """
    log_b = np.column_stack(
        [
            np.atleast_1d(multivariate_normal(hmm.means_[j], hmm.covars_[j]).logpdf(z))
            for j in range(hmm.n_components)
        ]
    )
    # Una transición imposible (probabilidad 0) da ln 0 = −inf, que logsumexp maneja bien.
    with np.errstate(divide="ignore"):
        log_start = np.log(hmm.startprob_)
        log_trans = np.log(hmm.transmat_)

    log_alpha = np.empty_like(log_b)
    current = log_start + log_b[0]
    log_alpha[0] = current - logsumexp(current)
    for t in range(1, len(z)):
        current = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
        log_alpha[t] = current - logsumexp(current)
    return log_alpha


def predict_regimes(model: object, features: pd.DataFrame) -> pd.Series:
    """Etiqueta FILTRADA por fecha (recursión forward; nunca `predict` ni `predict_proba`).

    SPEC_portafolio, Régimen, "Métodos comparados". La etiqueta en t usa variables hasta t:

    - Reglas: umbrales del ajuste sobre las variables de t.
    - K-means: centroide más cercano a las variables de t (lo mismo que hace `KMeans.predict`, escrito
      a mano para que esta función no llame a ningún método de predicción de una librería).
    - HMM: argmax de la probabilidad filtrada de `_filtered_log_probs`.

    Las filas con NaN quedan sin etiqueta. La recursión del HMM corre sobre las filas válidas en orden;
    con datos de `load_prices` solo hay NaN al inicio (calentamiento), así que no hay huecos.

    Parameters
    ----------
    model : RegimeModel
        Salida de `fit_regime_model`.
    features : pd.DataFrame
        Salida de `regime_features`.

    Returns
    -------
    pd.Series
        "tendencia", "reversion" o "crisis" por fecha, NaN sin datos; `name="regime"`.
    """
    labels = pd.Series(np.nan, index=features.index, dtype=object, name="regime")
    valid = features[FEATURE_COLUMNS].dropna()
    if valid.empty:
        return labels

    if model.method == "rules":
        names = np.where(
            valid["volatility"] > model.thresholds["volatility"],
            "crisis",
            np.where(valid["efficiency"] > model.thresholds["efficiency"], "tendencia", "reversion"),
        )
    else:
        z = ((valid - model.mean) / model.std).to_numpy()
        if model.method == "kmeans":
            centers = model.estimator.cluster_centers_
            states = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        else:
            states = _filtered_log_probs(model.estimator, z).argmax(axis=1)
        names = [model.names[int(s)] for s in states]

    labels.loc[valid.index] = names
    return labels


def viterbi_path(model: object, features: pd.DataFrame) -> pd.Series:
    """Ruta de Viterbi. Usa el futuro: SOLO para la figura comparativa, nunca para operar.

    Es la secuencia de estados más probable dada TODA la muestra de `features`: reetiqueta el pasado
    con información futura. Existe para mostrar en la figura de régimen la diferencia contra la
    etiqueta filtrada (P3, tarea 4).

    Returns
    -------
    pd.Series
        Mismos nombres y formato que `predict_regimes`.
    """
    if model.method != "hmm":
        raise ValueError("La ruta de Viterbi solo existe para el HMM.")
    labels = pd.Series(np.nan, index=features.index, dtype=object, name="regime")
    valid = features[FEATURE_COLUMNS].dropna()
    if valid.empty:
        return labels
    z = ((valid - model.mean) / model.std).to_numpy()
    labels.loc[valid.index] = [model.names[int(s)] for s in model.estimator.predict(z)]
    return labels


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
