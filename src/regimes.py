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
from sklearn.metrics import silhouette_score

from src.data import block_dates

FEATURE_COLUMNS = ["volatility", "efficiency", "autocorr"]
REGIME_NAMES = ["tendencia", "reversion", "crisis"]


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


def market_index(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """Nivel del índice equiponderado, base 1 en la primera fecha (para las figuras de régimen).

    Es la misma serie de mercado de la que salen las variables (SPEC_portafolio, Régimen):
    I_t = exp(Σ ℓ_k), con ℓ del primer día igual a 0 porque no hay retorno previo. Un hueco
    posterior queda NaN en esa fecha (no se rellena).
    """
    log_ret = _market_log_returns(prices)
    log_ret.iloc[0] = 0.0
    return np.exp(log_ret.cumsum()).rename("market_index")


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
        `regime_kmeans_n_init`, `regime_hmm_covariance`, `regime_hmm_n_iter` y `regime_hmm_n_init`.

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
        # Con un solo inicio, EM puede quedar en un óptimo local degenerado: dos estados con la
        # misma media que alternan cada día. Se reinicia con semillas seed, seed + 1, … y se queda
        # el de mayor verosimilitud sobre la muestra de ajuste (misma idea que n_init de K-means).
        candidates = [
            GaussianHMM(
                n_components=config["regime_n_states"],
                covariance_type=config["regime_hmm_covariance"],
                n_iter=config["regime_hmm_n_iter"],
                random_state=seed + k,
            ).fit(z)
            for k in range(config["regime_hmm_n_init"])
        ]
        estimator = max(candidates, key=lambda hmm: hmm.score(z))
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
    """Etiqueta causal para todas las fechas, con reajuste mensual y datos hasta cada fecha.

    SPEC_portafolio, Régimen, "Esquema de ajuste". En el primer día hábil τ de cada periodo de
    `regime_refit_freq`, desde `regime_first_fit`, el modelo de `regime_method` se ajusta con las
    variables válidas de fechas anteriores a τ (ventana expandible) y etiqueta los días del periodo
    sin reajustar. La etiqueta en t usa un modelo ajustado con datos anteriores a τ ≤ t y variables
    hasta t, así que no cambia al agregar datos posteriores (prueba 4 del lab).

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        Un DataFrame OHLCV por ticker con índice común (salida de `load_prices`).
    config : dict
        Usa `regime_method`, `regime_refit_freq`, `regime_first_fit`, `regime_min_fit_obs`, `seed`
        y lo que piden `regime_features` y `fit_regime_model`.

    Returns
    -------
    pd.Series
        "tendencia", "reversion" o "crisis" por fecha de `prices`; NaN antes del primer ajuste con
        al menos `regime_min_fit_obs` observaciones. `name="regime"`.
    """
    features = regime_features(prices, config)
    labels = pd.Series(np.nan, index=features.index, dtype=object, name="regime")
    operable = features.loc[pd.Timestamp(config["regime_first_fit"]) :]
    for _, period in operable.groupby(pd.Grouper(freq=config["regime_refit_freq"])):
        if period.empty:
            continue
        refit_date, period_end = period.index[0], period.index[-1]
        fit_sample = features.loc[features.index < refit_date].dropna()
        if len(fit_sample) < config["regime_min_fit_obs"]:
            continue
        model = fit_regime_model(fit_sample, config["regime_method"], config["seed"], config)
        # Se predice desde el inicio para que la recursión del HMM tenga toda su historia; las
        # reglas y K-means etiquetan cada día por separado, así que no les afecta.
        predicted = predict_regimes(model, features.loc[:period_end])
        labels.loc[refit_date:period_end] = predicted.loc[refit_date:period_end]
    return labels


def regime_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    blocks: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
) -> dict:
    """Silhouette, duración media, transiciones por mes y % de tiempo por régimen y por bloque.

    `blocks` es la salida de `block_dates(config)`; sin él, el % de tiempo se reporta solo para
    toda la muestra.

    SPEC_portafolio, Régimen, "Metas de validación". Se evalúan las fechas con variables y etiqueta:

    - silhouette: sobre las variables estandarizadas con la media y la desviación de esas fechas
      (métrica descriptiva, no se usa para operar). NaN si hay menos de dos regímenes.
    - duracion_media: días hábiles promedio de las rachas consecutivas con la misma etiqueta. Las
      rachas cortadas por el inicio o el fin de la muestra cuentan con su duración observada.
    - transiciones_por_mes: cambios de etiqueta entre el número de meses calendario de la muestra.

    Parameters
    ----------
    features : pd.DataFrame
        Salida de `regime_features`.
    labels : pd.Series
        Etiqueta por fecha (salida de `predict_regimes` o `label_regimes`).
    blocks : dict, optional
        Bloques {"train": (inicio, fin), …}, con fechas inclusivas.

    Returns
    -------
    dict
        silhouette, duracion_media, duracion_por_regimen (Series), transiciones_por_mes y
        pct_tiempo (DataFrame bloque × régimen, en % de los días etiquetados de cada bloque).
    """
    data = features[FEATURE_COLUMNS].assign(regime=labels).dropna()
    regime = data["regime"]

    z = (data[FEATURE_COLUMNS] - data[FEATURE_COLUMNS].mean()) / data[FEATURE_COLUMNS].std()
    silhouette = silhouette_score(z, regime) if regime.nunique() > 1 else np.nan

    # Cada cambio de etiqueta abre una racha nueva; la suma acumulada numera las rachas.
    run_id = (regime != regime.shift()).cumsum()
    runs = regime.groupby(run_id).agg(["first", "size"])
    n_months = data.index.to_period("M").nunique()

    periods = {"muestra": (data.index[0], data.index[-1])} if blocks is None else blocks
    pct = {
        name: regime.loc[start:end].value_counts(normalize=True) * 100
        for name, (start, end) in periods.items()
    }
    return {
        "silhouette": silhouette,
        "duracion_media": runs["size"].mean(),
        "duracion_por_regimen": runs.groupby("first")["size"].mean(),
        "transiciones_por_mes": (len(runs) - 1) / n_months,
        "pct_tiempo": pd.DataFrame(pct).T.reindex(columns=REGIME_NAMES).fillna(0.0),
    }


def compare_regime_methods(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Tabla comparativa de reglas, K-means y HMM (P3, tareas 2 y 3).

    SPEC_portafolio, Régimen, "Tabla comparativa": cada método se ajusta una vez con las fechas de
    train y etiqueta validation con el modelo congelado. Test no participa en la elección del
    método, así que aquí no se etiqueta.

    Parameters
    ----------
    features : pd.DataFrame
        Salida de `regime_features` sobre todos los datos.
    config : dict
        Usa `blocks`, `seed` y los valores del clasificador (ver `fit_regime_model`).

    Returns
    -------
    pd.DataFrame
        Índice (método, bloque) con silhouette, duracion_media, transiciones_por_mes y el % de
        tiempo en cada régimen.
    """
    blocks = block_dates(config)
    train_start, train_end = blocks["train"]
    rows = {}
    for method in ("rules", "kmeans", "hmm"):
        model = fit_regime_model(
            features.loc[train_start:train_end], method, config["seed"], config
        )
        # La recursión del HMM corre de train a validation sin cortes, como correría en vivo.
        labels = predict_regimes(model, features.loc[train_start : blocks["validation"][1]])
        for block in ("train", "validation"):
            start, end = blocks[block]
            result = regime_validation(features.loc[start:end], labels.loc[start:end])
            rows[(method, block)] = {
                "silhouette": result["silhouette"],
                "duracion_media": result["duracion_media"],
                "transiciones_por_mes": result["transiciones_por_mes"],
                **{f"pct_{name}": result["pct_tiempo"].loc["muestra", name] for name in REGIME_NAMES},
            }
    return pd.DataFrame(rows).T.rename_axis(["metodo", "bloque"]).astype(float)
