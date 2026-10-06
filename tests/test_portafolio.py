"""Pruebas de las funciones base de P4 (CLAUDE.md, sección 8; checklist de P4_portafolio.md).

Datos sintéticos con semilla fija; sin red y sin leer `data/`.
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (
    compose_target,
    turnover,
    equal_weights,
    estimate_cov,
    inverse_vol_weights,
    risk_contributions,
    risk_parity_weights,
)

SEED = 42
TICKERS = ["AAPL", "MSFT", "META", "AMD", "XOM", "SMH", "GLD", "COPX"]


def synthetic_cov(n: int = 8, seed: int = SEED) -> pd.DataFrame:
    """Σ de n activos con volatilidades y correlaciones distintas (todas por pares diferentes)."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n, 3))
    raw = factors @ factors.T + np.diag(rng.uniform(0.5, 1.5, n))
    d = np.sqrt(np.diag(raw))
    corr = raw / np.outer(d, d)
    vols = rng.uniform(0.10, 0.60, n)
    cov = corr * np.outer(vols, vols)
    return pd.DataFrame(cov, index=TICKERS[:n], columns=TICKERS[:n])


def equicorrelated_cov(vols: list[float], rho: float) -> pd.DataFrame:
    """Σ con la misma correlación ρ entre todos los pares y volatilidades distintas."""
    n = len(vols)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(vols, vols)
    names = TICKERS[:n]
    return pd.DataFrame(cov, index=names, columns=names)


def test_risk_parity_equalizes_risk_contributions():
    cov = synthetic_cov()
    weights = risk_parity_weights(cov)
    rc = risk_contributions(weights, cov)
    sigma_p = np.sqrt(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy())

    assert rc.max() - rc.min() < 1e-6
    assert abs(rc.sum() - sigma_p) < 1e-8  # Euler: Σ RC = σ_p
    assert abs(weights.sum() - 1.0) < 1e-12
    assert (weights > 0).all()


def test_risk_parity_differs_from_inverse_vol_with_unequal_correlations():
    cov = synthetic_cov()
    gap = (risk_parity_weights(cov) - inverse_vol_weights(cov)).abs().max()
    assert gap > 1e-3  # con correlaciones distintas la versión naive no es exacta


def test_risk_parity_equals_inverse_vol_with_equal_correlations():
    cov = equicorrelated_cov([0.05, 0.12, 0.20, 0.25, 0.31, 0.40, 0.18, 0.28], rho=0.3)
    gap = (risk_parity_weights(cov) - inverse_vol_weights(cov)).abs().max()
    assert gap < 1e-6


def test_risk_parity_is_invariant_to_covariance_scale():
    cov = synthetic_cov()
    daily = cov / 252  # varianzas diarias, como las que salen de retornos diarios
    assert (risk_parity_weights(cov) - risk_parity_weights(daily)).abs().max() < 1e-8


def test_two_asset_example_from_course_material():
    """Pasos 2 y 4: σ = (5%, 25%), ρ = 0, pesos iguales y volatilidad inversa."""
    cov = pd.DataFrame(np.diag([0.05**2, 0.25**2]), index=["A", "B"], columns=["A", "B"])
    equal = pd.Series([0.5, 0.5], index=["A", "B"])
    rc = risk_contributions(equal, cov)
    assert rc["B"] / rc.sum() == pytest.approx(0.9615, abs=1e-4)  # activo 2 aporta ~96%

    inv_vol = inverse_vol_weights(cov)
    assert inv_vol["A"] == pytest.approx(20 / 24)
    assert inv_vol["B"] == pytest.approx(4 / 24)


def test_equal_weights():
    w = equal_weights(TICKERS)
    assert len(w) == 8
    assert w.sum() == pytest.approx(1.0)
    assert (w == 1 / 8).all()


def synthetic_returns(n_days: int = 300, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(n_days, 8)) * rng.uniform(0.5, 2.0, 8)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=TICKERS)


@pytest.mark.parametrize("method", ["sample", "ewma", "ledoit_wolf"])
def test_estimate_cov_shape_symmetry_and_psd(method):
    cov = estimate_cov(synthetic_returns(), method)
    assert list(cov.index) == TICKERS and list(cov.columns) == TICKERS
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() > 0  # semidefinida positiva (aquí definida)
    assert risk_parity_weights(cov).sum() == pytest.approx(1.0)


def test_estimate_cov_sample_matches_numpy():
    returns = synthetic_returns()
    cov = estimate_cov(returns, "sample")
    assert np.allclose(cov.to_numpy(), np.cov(returns.to_numpy(), rowvar=False))


def test_ewma_weights_recent_observations_more():
    """Un choque reciente debe pesar más en EWMA que el mismo choque al inicio de la ventana."""
    returns = synthetic_returns(200)
    early, late = returns.copy(), returns.copy()
    early.iloc[0, 0] = 0.10
    late.iloc[-1, 0] = 0.10
    var_early = estimate_cov(early, "ewma").iloc[0, 0]
    var_late = estimate_cov(late, "ewma").iloc[0, 0]
    assert var_late > var_early


def test_estimate_cov_rejects_nan_and_unknown_method():
    returns = synthetic_returns()
    with_nan = returns.copy()
    with_nan.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        estimate_cov(with_nan, "sample")
    with pytest.raises(ValueError):
        estimate_cov(returns, "otro")


def test_functions_do_not_modify_inputs():
    cov = synthetic_cov()
    copy = cov.copy()
    weights = risk_parity_weights(cov)
    weights_copy = weights.copy()
    risk_contributions(weights, cov)
    inverse_vol_weights(cov)
    pd.testing.assert_frame_equal(cov, copy)
    pd.testing.assert_series_equal(weights, weights_copy)


def test_compose_target_integrating_example_from_course_material():
    """Paso 7: w^RP = (0.5, 0.3, 0.2), votos A=(1,1,1), B=(1,1,0), C=(1,-1,-1), m = 0.7."""
    base = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    strength = pd.Series([1.0, 2 / 3, 0.0], index=["A", "B", "C"])
    target = compose_target(base, strength, regime_multiplier=0.7)
    assert target["A"] == pytest.approx(0.35, abs=1e-12)
    assert target["B"] == pytest.approx(0.14, abs=1e-12)
    assert target["C"] == pytest.approx(0.0, abs=1e-12)


def test_compose_target_keeps_sign_and_gross_at_most_one():
    base = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    strength = pd.Series([-1.0, 1.0, -2 / 3], index=["A", "B", "C"])
    target = compose_target(base, strength, regime_multiplier=1.0)
    assert target["A"] < 0 and target["B"] > 0 and target["C"] < 0
    assert target.abs().sum() <= 1 + 1e-12


def test_compose_target_normalizes_when_gross_exceeds_one():
    base = pd.Series([0.8, 0.6], index=["A", "B"])  # no suma 1: fuerza el max(1, Σ|w̃|)
    strength = pd.Series([1.0, 1.0], index=["A", "B"])
    target = compose_target(base, strength, regime_multiplier=1.0)
    assert target.abs().sum() == pytest.approx(1.0)
    assert target["A"] == pytest.approx(0.8 / 1.4)


def test_compose_target_rejects_invalid_inputs():
    base = pd.Series([0.5, 0.5], index=["A", "B"])
    with pytest.raises(ValueError):
        compose_target(base, pd.Series([1.0, 1.0], index=["A", "B"]), regime_multiplier=1.5)
    with pytest.raises(ValueError):
        compose_target(base, pd.Series([1.0, 2.0], index=["A", "B"]), regime_multiplier=1.0)
    with pytest.raises(ValueError):
        compose_target(base, pd.Series([1.0], index=["A"]), regime_multiplier=1.0)


def test_turnover_example_from_course_material():
    """Paso 8: capital 100,000, objetivo viejo (0.40, 0.60), A +20% y B −10%, nuevo objetivo (0.5, 0.5)."""
    a, b = 40_000 * 1.20, 60_000 * 0.90
    drift = pd.DataFrame([[a / (a + b), b / (a + b)]], columns=["A", "B"], index=[pd.Timestamp("2020-01-31")])
    target = pd.DataFrame([[0.5, 0.5]], columns=["A", "B"], index=drift.index)
    old_target = pd.DataFrame([[0.4, 0.6]], columns=["A", "B"], index=drift.index)

    assert turnover(drift, target).iloc[0] == pytest.approx(0.0294, abs=1e-4)
    assert turnover(old_target, target).iloc[0] == pytest.approx(0.10)  # sobrestima el costo real


def test_turnover_is_zero_without_changes_and_validates_alignment():
    index = pd.bdate_range("2020-01-01", periods=3)
    w = pd.DataFrame(0.25, index=index, columns=["A", "B", "C", "D"])
    assert (turnover(w, w) == 0).all()
    with pytest.raises(ValueError):
        turnover(w, w.iloc[:2])