"""Pruebas de régimen (P3): variables, clasificadores y etiqueta filtrada.

Datos sintéticos de `conftest.py`; ninguna prueba lee `data/` ni usa red (CLAUDE.md, sección 8).
"""

import ast
import inspect
import textwrap

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

import src.regimes as regimes
from src.regimes import (
    FEATURE_COLUMNS,
    _filtered_log_probs,
    _name_states,
    fit_regime_model,
    predict_regimes,
    regime_features,
    viterbi_path,
)

METHODS = ["rules", "kmeans", "hmm"]


@pytest.fixture
def features(synthetic_prices, config_test) -> pd.DataFrame:
    """Variables de régimen sobre los precios sintéticos."""
    return regime_features(synthetic_prices, config_test)


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


@pytest.mark.parametrize("order", [[0, 1, 2], [2, 0, 1], [1, 2, 0]])
def test_name_rule_does_not_depend_on_state_number(order):
    """Los nombres salen de los centroides, no del número de grupo (sin intercambio entre reajustes)."""
    centers = pd.DataFrame(
        {
            "volatility": [0.40, 0.15, 0.12],
            "efficiency": [0.10, 0.30, 0.05],
            "autocorr": [-0.05, 0.02, -0.10],
        }
    )
    expected = {0: "crisis", 1: "tendencia", 2: "reversion"}
    permuted = centers.iloc[order].reset_index(drop=True)
    names = _name_states(permuted)
    assert {new: names[new] for new in range(3)} == {
        new: expected[old] for new, old in enumerate(order)
    }


def test_rules_follow_thresholds(features, config_test):
    """Reglas: crisis sobre el cuantil de volatilidad; si no, tendencia sobre el de eficiencia."""
    model = fit_regime_model(features, "rules", config_test["seed"], config_test)
    valid = features.dropna()
    vol_q = valid["volatility"].quantile(config_test["regime_crisis_quantile"])
    eff_q = valid["efficiency"].quantile(config_test["regime_trend_quantile"])
    expected = np.where(
        valid["volatility"] > vol_q,
        "crisis",
        np.where(valid["efficiency"] > eff_q, "tendencia", "reversion"),
    )
    labels = predict_regimes(model, features)
    assert (labels.loc[valid.index].to_numpy() == expected).all()


@pytest.mark.parametrize("t", [10, 200, -1])
def test_hmm_filtered_equals_smoothed_on_last_day(features, config_test, t):
    """Valida la recursión forward contra hmmlearn sin usar el futuro.

    En el último día de una muestra no hay futuro, así que la probabilidad filtrada y la suavizada
    son la misma. Por eso el forward sobre x[:t+1] debe coincidir con la última fila de
    `predict_proba(x[:t+1])`. Aquí `predict_proba` solo sirve de referencia en la prueba.
    """
    model = fit_regime_model(features, "hmm", config_test["seed"], config_test)
    valid = features.dropna()
    z = ((valid - model.mean) / model.std).to_numpy()
    t = t % len(z)
    filtered = np.exp(_filtered_log_probs(model.estimator, z[: t + 1]))[-1]
    smoothed = model.estimator.predict_proba(z[: t + 1])[-1]
    np.testing.assert_allclose(filtered, smoothed, rtol=0, atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("t", [100, 300, 599])
def test_label_truncation(features, config_test, method, t):
    """Con el modelo fijo, la etiqueta en t no cambia al agregar datos posteriores (prueba 4)."""
    model = fit_regime_model(features, method, config_test["seed"], config_test)
    full = predict_regimes(model, features)
    truncated = predict_regimes(model, features.iloc[: t + 1])
    assert truncated.index[-1] == full.index[t]
    assert truncated.iloc[-1] == full.iloc[t]


def test_labels_nan_without_features(features, config_test):
    """Sin variables válidas (calentamiento) no hay etiqueta."""
    model = fit_regime_model(features, "hmm", config_test["seed"], config_test)
    labels = predict_regimes(model, features)
    assert labels[features.isna().any(axis=1)].isna().all()
    assert set(labels.dropna().unique()) <= {"tendencia", "reversion", "crisis"}


@pytest.mark.parametrize("method", ["kmeans", "hmm"])
def test_same_seed_same_labels(features, config_test, method):
    """Misma semilla, mismas etiquetas (CLAUDE.md, sección 9)."""
    first = predict_regimes(fit_regime_model(features, method, 42, config_test), features)
    second = predict_regimes(fit_regime_model(features, method, 42, config_test), features)
    pd.testing.assert_series_equal(first, second)


def test_operable_label_never_calls_library_inference():
    """La etiqueta operable no llama a predict, predict_proba, score_samples ni decode (checklist P3).

    Se revisan las llamadas en el árbol sintáctico, no el texto, para que un docstring que mencione
    esos nombres no dé un falso positivo.
    """
    forbidden = {"predict", "predict_proba", "score_samples", "decode"}
    for function in (regimes.predict_regimes, regimes._filtered_log_probs):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not calls & forbidden, f"{function.__name__} llama a {calls & forbidden}"


def test_viterbi_only_for_hmm(features, config_test):
    """Viterbi solo existe para el HMM y devuelve los mismos nombres que la filtrada."""
    with pytest.raises(ValueError):
        viterbi_path(fit_regime_model(features, "kmeans", 42, config_test), features)
    path = viterbi_path(fit_regime_model(features, "hmm", 42, config_test), features)
    assert set(path.dropna().unique()) <= {"tendencia", "reversion", "crisis"}


def test_hmm_restarts_keep_best_likelihood(features, config_test):
    """Con reinicios, la verosimilitud nunca es menor que la del primer inicio (seed)."""
    single = dict(config_test, regime_hmm_n_init=1)
    z_model = fit_regime_model(features, "hmm", 42, config_test)
    z = ((features.dropna() - z_model.mean) / z_model.std).to_numpy()
    best = z_model.estimator.score(z)
    first = fit_regime_model(features, "hmm", 42, single).estimator.score(z)
    assert best >= first


def test_regime_validation_by_hand():
    """Duración, transiciones y % de tiempo contra un cálculo en papel.

    10 días hábiles (5 en enero y 5 en febrero de 2020): T T T R R C C C C T.
    Rachas: T3, R2, C4, T1 → duración media 10 / 4 = 2.5; tendencia (3 + 1) / 2 = 2, reversión 2,
    crisis 4. Transiciones: 3 en 2 meses → 1.5 por mes. Tiempo: T 40%, R 20%, C 40%.
    Bloque "a" (enero): T 60%, R 40%. Bloque "b" (febrero): T 20%, C 80%.
    """
    dates = pd.bdate_range("2020-01-27", periods=10)
    codes = list("TTTRRCCCCT")
    names = {"T": "tendencia", "R": "reversion", "C": "crisis"}
    labels = pd.Series([names[c] for c in codes], index=dates)
    # Variables separadas por régimen para que la silhouette esté definida.
    level = {"T": 0.0, "R": 1.0, "C": 5.0}
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {col: [level[c] + rng.normal(0, 0.1) for c in codes] for col in FEATURE_COLUMNS},
        index=dates,
    )
    blocks = {"a": (dates[0], dates[4]), "b": (dates[5], dates[9])}

    result = regimes.regime_validation(features, labels, blocks)

    assert result["duracion_media"] == pytest.approx(2.5)
    assert result["duracion_por_regimen"].to_dict() == pytest.approx(
        {"tendencia": 2.0, "reversion": 2.0, "crisis": 4.0}
    )
    assert result["transiciones_por_mes"] == pytest.approx(1.5)
    expected_pct = pd.DataFrame(
        {"tendencia": [60.0, 20.0], "reversion": [40.0, 0.0], "crisis": [0.0, 80.0]},
        index=["a", "b"],
    )
    pd.testing.assert_frame_equal(
        result["pct_tiempo"], expected_pct, check_names=False, check_dtype=False
    )
    z = (features - features.mean()) / features.std()
    assert result["silhouette"] == pytest.approx(silhouette_score(z, labels))


def test_regime_validation_without_blocks_uses_whole_sample(features, config_test):
    """Sin bloques, el % de tiempo se reporta para toda la muestra y suma 100."""
    model = fit_regime_model(features, "rules", 42, config_test)
    result = regimes.regime_validation(features, predict_regimes(model, features))
    assert list(result["pct_tiempo"].index) == ["muestra"]
    assert result["pct_tiempo"].loc["muestra"].sum() == pytest.approx(100.0)
