"""Pruebas de las funciones base de P4 (CLAUDE.md, sección 8; checklist de P4_portafolio.md).

Datos sintéticos con semilla fija; sin red y sin leer `data/`.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.signals import generate_signals
from src.portfolio import (
    compose_target,
    equal_weights,
    estimate_cov,
    inverse_vol_weights,
    resolve_signal_conflicts,
    performance_comparison,
    portfolio_results,
    risk_contribution_comparison,
    risk_contribution_plot_frame,
    risk_contributions,
    rebalance_sweep,
    sweep_plot_frame,
    risk_parity_weights,
    sleeve_weights,
    turnover,
    weight_stability,
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


# ---------------------------------------------------------------------------------------------
# sleeve_weights y política de conflictos. Las señales y los regímenes son stubs locales y
# causales, para no depender de signals.py (P2) ni de regimes.py (P3). Usan los fixtures de
# conftest.py: `synthetic_prices` (3 activos, 600 días) y `config_test` (copia de CONFIG).
# ---------------------------------------------------------------------------------------------


def causal_strength(prices: dict) -> pd.DataFrame:
    """Fuerza 2 de 3 con tres votos causales (SMA, momento de 5 y de 20 días)."""
    columns = {}
    for ticker, ohlcv in prices.items():
        close = ohlcv["close"]
        votes = pd.concat(
            [
                np.sign(close - close.rolling(10).mean()),
                np.sign(close.diff(5)),
                np.sign(close.diff(20)),
            ],
            axis=1,
        ).fillna(0.0)
        total = votes.sum(axis=1)
        columns[ticker] = (total / 3).where(total.abs() >= 2, 0.0)
    return pd.DataFrame(columns)


def cyclic_regimes(index: pd.DatetimeIndex) -> pd.Series:
    """Régimen que cambia cada 40 días, sin etiqueta en los primeros 30 (como `label_regimes`)."""
    names = ["tendencia", "reversion", "crisis"]
    labels = pd.Series(
        [names[(i // 40) % 3] for i in range(len(index))], index=index, dtype=object, name="regime"
    )
    labels.iloc[:30] = np.nan
    return labels


def corr_matrix(names: list[str], rho: float) -> pd.DataFrame:
    corr = np.full((len(names), len(names)), rho)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=names, columns=names)


def test_sleeve_weights_shape_nonnegative_and_gross_at_most_one(synthetic_prices, config_test):
    index = synthetic_prices["A0"].index
    signals = {"strength": causal_strength(synthetic_prices)}
    panel = sleeve_weights(synthetic_prices, signals, cyclic_regimes(index), config_test)

    window = config_test["cov_window"]
    assert panel.index.equals(index) and list(panel.columns) == list(synthetic_prices)
    defined = panel.dropna()
    assert not defined.empty
    assert (defined >= 0).all().all()
    assert (defined.sum(axis=1) <= 1 + 1e-12).all()
    assert panel.iloc[:window].isna().all().all()  # sin historia suficiente para Σ
    assert panel.iloc[window + 5 :].notna().all().all()


def test_sleeve_weights_truncation(synthetic_prices, config_test):
    """El panel en t no cambia al recalcular todo con datos hasta t (CLAUDE.md, sección 4)."""
    index = synthetic_prices["A0"].index
    regimes = cyclic_regimes(index)
    # Con revisión diaria y banda 0, w^RP se recalcula en cada fecha t: una fuga hacia adelante en
    # la ventana de Σ cambiaría el último peso. Con umbral −1 casi todo par con señales opuestas
    # entra en conflicto, y se ejercita esa rama.
    variants = [
        {},
        {"rebalance_frequency": "D", "rebalance_band": 0.0, "cov_window": 60},
    ]
    for variant in variants:
        for threshold in (0.7, -1.0):
            config = {**config_test, **variant, "conflict_corr_threshold": threshold}
            full = sleeve_weights(
                synthetic_prices, {"strength": causal_strength(synthetic_prices)}, regimes, config
            )
            for t in (200, 350, 500):
                cut = {ticker: ohlcv.iloc[: t + 1] for ticker, ohlcv in synthetic_prices.items()}
                truncated = sleeve_weights(
                    cut, {"strength": causal_strength(cut)}, regimes.iloc[: t + 1], config
                )
                pd.testing.assert_frame_equal(
                    truncated, full.iloc[: t + 1], atol=1e-12, rtol=0, check_freq=False
                )


def test_equal_method_uses_one_over_n(synthetic_prices, config_test):
    config_test["conflict_corr_threshold"] = 2.0  # sin conflictos: aísla la agregación
    index = synthetic_prices["A0"].index
    strength = causal_strength(synthetic_prices)
    regimes = cyclic_regimes(index)
    panel = sleeve_weights(
        synthetic_prices, {"strength": strength}, regimes, config_test, method="equal"
    )
    multiplier = regimes.map(config_test["regime_multiplier"]).astype(float)
    expected = strength.abs().mul(multiplier, axis=0) / len(synthetic_prices)  # Σ ≤ 1: max(1, ·) = 1
    defined = panel.dropna().index
    np.testing.assert_allclose(panel.loc[defined].to_numpy(), expected.loc[defined].to_numpy(), atol=1e-12)


def test_risk_parity_wiring_matches_manual_computation(synthetic_prices, config_test):
    """Con revisión diaria y banda 0, w^RP en t es Risk Parity sobre la ventana que termina en t."""
    config_test.update(
        rebalance_frequency="D", rebalance_band=0.0, conflict_corr_threshold=2.0, cov_window=60
    )
    index = synthetic_prices["A0"].index
    strength = causal_strength(synthetic_prices)
    regimes = cyclic_regimes(index)
    panel = sleeve_weights(synthetic_prices, {"strength": strength}, regimes, config_test)

    closes = pd.DataFrame({t: ohlcv["close"] for t, ohlcv in synthetic_prices.items()})
    returns = closes.pct_change(fill_method=None)
    for pos in (150, 300, 450):
        w_rp = risk_parity_weights(estimate_cov(returns.iloc[pos - 59 : pos + 1], config_test["cov_method"]))
        multiplier = config_test["regime_multiplier"][regimes.iloc[pos]]
        expected = compose_target(w_rp, strength.iloc[pos], multiplier).abs()
        np.testing.assert_allclose(panel.iloc[pos].to_numpy(), expected.to_numpy(), atol=1e-9)


def _rp_component(synthetic_prices, config_test):
    """w^RP vigente por fecha, aislado con fuerza 1 en todos los activos (w_target = m · w^RP)."""
    index = synthetic_prices["A0"].index
    regimes = cyclic_regimes(index)
    ones = pd.DataFrame(1.0, index=index, columns=list(synthetic_prices))
    panel = sleeve_weights(synthetic_prices, {"strength": ones}, regimes, config_test)
    multiplier = regimes.map(config_test["regime_multiplier"]).astype(float)
    return panel.div(multiplier, axis=0).dropna()


def test_rebalance_changes_weights_only_on_calendar_dates(synthetic_prices, config_test):
    config_test["rebalance_band"] = 0.0  # cualquier cambio se adopta, pero solo en fechas de revisión
    base = _rp_component(synthetic_prices, config_test)
    index = synthetic_prices["A0"].index
    is_first_of_month = pd.Series(~index.to_period("M").duplicated(), index=index)

    changed = base.diff().abs().sum(axis=1) > 1e-9
    assert changed.any()
    assert not changed[~is_first_of_month.loc[base.index]].any()


def test_rebalance_band_blocks_small_changes(synthetic_prices, config_test):
    config_test["rebalance_band"] = 10.0  # nunca se supera: w^RP queda en su primera adopción
    base = _rp_component(synthetic_prices, config_test)
    assert base.diff().abs().to_numpy()[1:].max() < 1e-9


def test_conflict_keeps_stronger_signal_and_zeroes_the_weaker():
    s = pd.Series([1.0, -2 / 3, 0.0], index=["A", "B", "C"])
    out = resolve_signal_conflicts(s, corr_matrix(["A", "B", "C"], 0.9), threshold=0.7)
    assert out.tolist() == [1.0, 0.0, 0.0]


def test_conflict_tie_zeroes_both():
    s = pd.Series([2 / 3, -2 / 3], index=["A", "B"])
    out = resolve_signal_conflicts(s, corr_matrix(["A", "B"], 0.9), threshold=0.7)
    assert out.tolist() == [0.0, 0.0]


def test_no_conflict_with_low_correlation_or_same_sign():
    names = ["A", "B"]
    opposite = pd.Series([1.0, -1.0], index=names)
    same = pd.Series([1.0, 2 / 3], index=names)
    pd.testing.assert_series_equal(resolve_signal_conflicts(opposite, corr_matrix(names, 0.3), 0.7), opposite)
    pd.testing.assert_series_equal(resolve_signal_conflicts(same, corr_matrix(names, 0.95), 0.7), same)


def test_conflict_resolution_does_not_depend_on_asset_order():
    s = pd.Series([1.0, -2 / 3, 1 / 3, -1.0], index=["A", "B", "C", "D"])
    corr = corr_matrix(list(s.index), 0.9)
    reference = resolve_signal_conflicts(s, corr, 0.7)
    for order in (["D", "C", "B", "A"], ["B", "D", "A", "C"]):
        shuffled = resolve_signal_conflicts(s[order], corr.loc[order, order], 0.7)
        pd.testing.assert_series_equal(shuffled.sort_index(), reference.sort_index())


def test_sleeve_weights_rejects_invalid_method_and_unknown_regime(synthetic_prices, config_test):
    index = synthetic_prices["A0"].index
    signals = {"strength": causal_strength(synthetic_prices)}
    with pytest.raises(ValueError):
        sleeve_weights(synthetic_prices, signals, cyclic_regimes(index), config_test, method="otro")
    bad = cyclic_regimes(index)
    bad.iloc[100] = "lateral"
    with pytest.raises(ValueError):
        sleeve_weights(synthetic_prices, signals, bad, config_test)


def test_sleeve_weights_does_not_modify_inputs(synthetic_prices, config_test):
    index = synthetic_prices["A0"].index
    signals = {"strength": causal_strength(synthetic_prices)}
    regimes = cyclic_regimes(index)
    strength_copy, regimes_copy = signals["strength"].copy(), regimes.copy()
    close_copy = synthetic_prices["A0"]["close"].copy()
    sleeve_weights(synthetic_prices, signals, regimes, config_test)
    pd.testing.assert_frame_equal(signals["strength"], strength_copy)
    pd.testing.assert_series_equal(regimes, regimes_copy)
    pd.testing.assert_series_equal(synthetic_prices["A0"]["close"], close_copy)


def test_panel_feeds_run_backtest_and_accounting_holds(synthetic_prices, config_test):
    """El panel real entra a `run_backtest` y la contabilidad cuadra: equity = cash + posiciones."""
    index = synthetic_prices["A0"].index
    strength = causal_strength(synthetic_prices)
    atr = pd.DataFrame(
        {t: (ohlcv["high"] - ohlcv["low"]).rolling(14).mean() for t, ohlcv in synthetic_prices.items()}
    )
    signals = {"state": np.sign(strength).astype(int), "strength": strength, "atr": atr}
    panel = sleeve_weights(synthetic_prices, signals, cyclic_regimes(index), config_test)
    trade_params = pd.DataFrame(
        {"k_stop": 2.0, "reward_ratio": 2.0, "max_holding": 20}, index=index
    )

    result = run_backtest(synthetic_prices, signals, panel, trade_params, config_test)

    closes = pd.DataFrame({t: ohlcv["close"] for t, ohlcv in synthetic_prices.items()})
    marked = result.cash + (result.positions * closes).sum(axis=1)
    np.testing.assert_allclose(result.equity.to_numpy(), marked.to_numpy(), atol=1e-6)
    assert len(result.trades) > 0
    assert panel.iloc[: config_test["cov_window"]].isna().all().all()
    assert result.fills["date"].min() > index[config_test["cov_window"]]  # sin panel no se opera


def test_conflict_correlation_uses_only_the_trailing_window_up_to_t(synthetic_prices, config_test):
    """La ρ que recibe la política de conflictos en t usa solo los últimos `cov_window` retornos hasta t."""
    import src.portfolio as module

    index = synthetic_prices["A0"].index
    strength = causal_strength(synthetic_prices)
    regimes = cyclic_regimes(index)
    seen = []
    original = module._conflict_losers

    def spy(s, rho, threshold):
        seen.append(rho.copy())
        return original(s, rho, threshold)

    module._conflict_losers = spy
    try:
        sleeve_weights(synthetic_prices, {"strength": strength}, regimes, config_test)
    finally:
        module._conflict_losers = original

    # Días en que se aplica la política: régimen y w^RP definidos y señales de signo opuesto.
    closes = pd.DataFrame({t: ohlcv["close"] for t, ohlcv in synthetic_prices.items()})
    returns = closes.pct_change(fill_method=None)
    base = module._base_weights(returns, config_test, "risk_parity")
    s = strength.reindex(columns=returns.columns)
    conflict_days = regimes.notna() & base.notna().all(axis=1) & (s > 0).any(axis=1) & (s < 0).any(axis=1)
    positions = np.flatnonzero(conflict_days.to_numpy())

    assert len(positions) > 0  # hubo días con señales opuestas
    assert len(seen) == len(positions)
    window = config_test["cov_window"]
    for pos, rho in zip(positions, seen):
        expected = returns.iloc[pos - window + 1 : pos + 1].corr().to_numpy()
        np.testing.assert_array_equal(rho, expected)


# ---------------------------------------------------------------------------------------------
# rebalance_sweep
# ---------------------------------------------------------------------------------------------


def _sweep_inputs(synthetic_prices, config_test):
    index = synthetic_prices["A0"].index
    params = dict.fromkeys(["tendencia", "reversion", "crisis"], config_test["base_params"])
    return params, cyclic_regimes(index)


def test_rebalance_sweep_shape_and_accounting(synthetic_prices, config_test):
    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    out = rebalance_sweep(synthetic_prices, params, regimes, config_test, [0.0, 0.05], ["M", "Q"])
    assert len(out) == 4
    assert list(out.columns) == [
        "frequency", "band", "n_rebalances", "turnover",
        "gross_return", "total_cost", "net_return", "n_trades",
    ]
    assert (out["total_cost"] >= 0).all() and (out["turnover"] >= 0).all()
    # bruto − neto = costos / capital inicial (costos ≥ 0 → bruto ≥ neto)
    gap = out["gross_return"] - out["net_return"]
    np.testing.assert_allclose(gap, out["total_cost"] / config_test["initial_capital"], rtol=1e-9)


def test_rebalance_sweep_wide_band_blocks_rotation_and_more_rebalances_when_frequent(
    synthetic_prices, config_test
):
    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    out = rebalance_sweep(synthetic_prices, params, regimes, config_test, [0.0, 10.0], ["M", "Q"])
    wide = out[out["band"] == 10.0]
    assert (wide["n_rebalances"] == 0).all() and (wide["turnover"] == 0).all()
    tight = out[out["band"] == 0.0].set_index("frequency")
    assert tight.loc["M", "n_rebalances"] >= tight.loc["Q", "n_rebalances"]


def test_rebalance_sweep_matches_manual_run_and_restricts_to_period(synthetic_prices, config_test):
    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    index = synthetic_prices["A0"].index
    full = rebalance_sweep(synthetic_prices, params, regimes, config_test, [0.05], ["M"])
    cfg = {**config_test, "rebalance_frequency": "M", "rebalance_band": 0.05}
    signals = generate_signals(synthetic_prices, params, regimes, cfg)
    trade_params = pd.DataFrame(
        {k: config_test["base_params"][k] for k in ["k_stop", "reward_ratio", "max_holding"]},
        index=index,
    )
    panel = sleeve_weights(synthetic_prices, signals, regimes, cfg)
    manual = run_backtest(synthetic_prices, signals, panel, trade_params, cfg)
    assert full["net_return"].iloc[0] == pytest.approx(manual.equity.iloc[-1] / manual.equity.iloc[0] - 1)
    assert full["n_trades"].iloc[0] == len(manual.trades)
    part = rebalance_sweep(
        synthetic_prices, params, regimes, config_test, [0.05], ["M"], period=(index[300], index[-1])
    )
    eq = manual.equity.loc[index[299]:]  # el bloque arranca en el cierre anterior (block_equity)
    assert part["net_return"].iloc[0] == pytest.approx(eq.iloc[-1] / eq.iloc[0] - 1)
    assert part["n_trades"].iloc[0] <= full["n_trades"].iloc[0]


def test_period_results_do_not_depend_on_data_after_the_period(synthetic_prices, config_test):
    """Con `period`, agregar datos posteriores a su fin no cambia el barrido ni el desempeño."""
    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    index = synthetic_prices["A0"].index
    period = (index[200], index[400])
    shorter = {t: ohlcv.iloc[:450] for t, ohlcv in synthetic_prices.items()}
    for function, kwargs in (
        (rebalance_sweep, {"bands": [0.05], "frequencies": ["M"]}),
        (performance_comparison, {"include_assets": False}),
    ):
        full = function(synthetic_prices, params, regimes, config_test, period=period, **kwargs)
        cut = function(shorter, params, regimes, config_test, period=period, **kwargs)
        pd.testing.assert_frame_equal(full, cut)


def test_rebalance_sweep_rejects_invalid_inputs_and_keeps_config(synthetic_prices, config_test):
    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    before = dict(config_test)
    with pytest.raises(ValueError):
        rebalance_sweep(synthetic_prices, params, regimes, config_test, [], ["M"])
    with pytest.raises(ValueError):
        rebalance_sweep(synthetic_prices, params, regimes, config_test, [-0.1], ["M"])
    rebalance_sweep(synthetic_prices, params, regimes, config_test, [0.05], ["M"])
    assert config_test == before


# ---------------------------------------------------------------------------------------------
# weight_stability
# ---------------------------------------------------------------------------------------------


def test_weight_stability_shape_and_manual_check(synthetic_prices, config_test):
    out = weight_stability(synthetic_prices, config_test)
    assert list(out.index) == ["sample", "ewma", "ledoit_wolf"]
    assert list(out.columns) == [*synthetic_prices, "mean_std", "n_reviews"]
    assert (out[list(synthetic_prices)] >= 0).all().all()
    np.testing.assert_allclose(out["mean_std"], out[list(synthetic_prices)].mean(axis=1))
    # comprobación manual con Ledoit-Wolf: pesos en cada primer día hábil del mes, sin banda
    returns = pd.DataFrame({t: o["close"] for t, o in synthetic_prices.items()}).pct_change(fill_method=None)
    window = config_test["cov_window"]
    reviews = ~returns.index.to_period("M").duplicated()
    rows = [
        risk_parity_weights(estimate_cov(returns.iloc[p - window + 1 : p + 1], "ledoit_wolf"))
        for p in range(window, len(returns))
        if reviews[p]
    ]
    expected = pd.DataFrame(rows).std(ddof=1)
    np.testing.assert_allclose(out.loc["ledoit_wolf", list(synthetic_prices)], expected, rtol=1e-6)
    assert out.loc["ledoit_wolf", "n_reviews"] == len(rows)


def test_weight_stability_period_restricts_reviews_and_does_not_modify_config(
    synthetic_prices, config_test
):
    index = synthetic_prices["A0"].index
    before = dict(config_test)
    full = weight_stability(synthetic_prices, config_test)
    part = weight_stability(synthetic_prices, config_test, period=(index[300], index[-1]))
    assert (part["n_reviews"] < full["n_reviews"]).all()
    assert config_test == before


# ---------------------------------------------------------------------------------------------
# risk_contribution_comparison
# ---------------------------------------------------------------------------------------------


def test_risk_contribution_comparison_parity_equalizes_and_equal_does_not(
    synthetic_prices, config_test
):
    out = risk_contribution_comparison(synthetic_prices, config_test)
    names = list(synthetic_prices)
    assert list(out.index) == ["risk_parity", "equal"]
    np.testing.assert_allclose(out.loc["risk_parity", names], 1 / len(names), atol=1e-6)
    np.testing.assert_allclose(out[names].sum(axis=1), 1.0, atol=1e-9)
    assert out.loc["risk_parity", "spread"] < 1e-6
    assert out.loc["equal", "spread"] > out.loc["risk_parity", "spread"]
    # Risk Parity con volatilidades distintas pesa menos al activo más volátil: menor ex ante
    # que 1/n solo si hay dispersión; aquí basta que sean positivos y finitos
    assert (out["ann_vol"] > 0).all() and np.isfinite(out["ann_vol"]).all()


def test_risk_contribution_comparison_period_and_manual_check(synthetic_prices, config_test):
    index = synthetic_prices["A0"].index
    period = (index[300], index[-1])
    part = risk_contribution_comparison(synthetic_prices, config_test, period=period)
    returns = pd.DataFrame({t: o["close"] for t, o in synthetic_prices.items()}).pct_change(fill_method=None)
    window, reviews = config_test["cov_window"], ~returns.index.to_period("M").duplicated()
    shares = []
    for pos in range(window, len(returns)):
        if reviews[pos] and returns.index[pos] >= period[0]:
            cov = estimate_cov(returns.iloc[pos - window + 1 : pos + 1], config_test["cov_method"])
            rc = risk_contributions(equal_weights(list(returns.columns)), cov)
            shares.append((rc / rc.sum()).to_numpy())
    names = list(synthetic_prices)
    np.testing.assert_allclose(part.loc["equal", names], np.vstack(shares).mean(axis=0), rtol=1e-9)
    with pytest.raises(ValueError):
        risk_contribution_comparison(
            synthetic_prices, config_test, period=(index[0], index[1])
        )


# ---------------------------------------------------------------------------------------------
# performance_comparison
# ---------------------------------------------------------------------------------------------


def test_performance_comparison_rows_columns_and_manual_check(synthetic_prices, config_test):
    from src.metrics import compute_metrics

    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    out = performance_comparison(synthetic_prices, params, regimes, config_test)
    assert list(out.index) == ["risk_parity", "equal", *synthetic_prices]
    assert {"calmar", "max_drawdown", "n_trades", "ann_return"} <= set(out.columns)
    assert (out["max_drawdown"] <= 0).all()

    index = synthetic_prices["A0"].index
    signals = generate_signals(synthetic_prices, params, regimes, config_test)
    trade_params = pd.DataFrame(
        {k: config_test["base_params"][k] for k in ["k_stop", "reward_ratio", "max_holding"]},
        index=index,
    )
    panel = sleeve_weights(synthetic_prices, signals, regimes, config_test)
    manual = run_backtest(synthetic_prices, signals, panel, trade_params, config_test)
    expected = compute_metrics(manual.equity, manual.trades, 0.0, config_test["periods_per_year"])
    assert out.loc["risk_parity", "ann_return"] == pytest.approx(expected["ann_return"])
    assert out.loc["risk_parity", "n_trades"] == expected["n_trades"]


def test_performance_comparison_period_and_options(synthetic_prices, config_test):
    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    index = synthetic_prices["A0"].index
    full = performance_comparison(synthetic_prices, params, regimes, config_test, include_assets=False)
    part = performance_comparison(
        synthetic_prices, params, regimes, config_test, period=(index[300], index[-1]), include_assets=False
    )
    assert list(full.index) == ["risk_parity", "equal"]
    assert (part["n_trades"] <= full["n_trades"]).all()
    assert not part["ann_return"].equals(full["ann_return"])


# ---------------------------------------------------------------------------------------------
# portfolio_results
# ---------------------------------------------------------------------------------------------


def test_portfolio_results_structure_and_matches_components(synthetic_prices, config_test):
    params, regimes = _sweep_inputs(synthetic_prices, config_test)
    index = synthetic_prices["A0"].index
    config = {**config_test, "rebalance_bands": [0.0, 0.05], "rebalance_frequencies": ["M", "Q"]}
    periods = {"train": (index[0], index[350]), "other": (index[351], index[-1])}
    out = portfolio_results(synthetic_prices, params, regimes, config, 0.0, periods)

    assert set(out) == {"weight_stability", "risk_contributions", "performance", "sweep"}
    assert set(out["performance"]) == set(out["sweep"]) == {"train", "other"}
    assert len(out["sweep"]["train"]) == 4
    pd.testing.assert_frame_equal(
        out["weight_stability"], weight_stability(synthetic_prices, config, period=periods["train"])
    )
    pd.testing.assert_frame_equal(
        out["sweep"]["other"],
        rebalance_sweep(synthetic_prices, params, regimes, config, [0.0, 0.05], ["M", "Q"], period=periods["other"]),
    )
    with pytest.raises(ValueError):
        portfolio_results(synthetic_prices, params, regimes, config, 0.0, {})


# ---------------------------------------------------------------------------------------------
# Adaptadores a las figuras de plots.py (P3)
# ---------------------------------------------------------------------------------------------


def test_risk_contribution_plot_frame_matches_plot_contract(synthetic_prices, config_test):
    out = risk_contribution_comparison(synthetic_prices, config_test)
    frame = risk_contribution_plot_frame(out)
    assert list(frame.index) == list(synthetic_prices)  # ticker × esquema
    assert list(frame.columns) == ["Risk Parity", "Pesos iguales"]
    np.testing.assert_allclose(frame.sum(), 1.0, atol=1e-9)
    assert frame.loc["A0", "Pesos iguales"] == out.loc["equal", "A0"]


def test_sweep_plot_frame_annualizes_and_defines_cost_as_gross_minus_net():
    sweep = pd.DataFrame(
        {
            "frequency": ["W", "M"],
            "band": [0.0, 0.05],
            "n_rebalances": [10, 5],
            "turnover": [0.6, 0.3],
            "gross_return": [0.21, 0.1],
            "total_cost": [1.0, 1.0],
            "net_return": [0.1, 0.0],
            "n_trades": [3, 3],
        }
    )
    period = ("2020-01-01", "2021-12-31")  # 730 días ≈ 1.9986 años
    out = sweep_plot_frame(sweep, period, {})
    years = 730 / 365.25
    assert list(out.columns) == ["gross_return", "cost", "net_return", "turnover"]
    assert list(out.index) == ["W · 0", "M · 0.05"]
    assert out.loc["W · 0", "gross_return"] == pytest.approx(1.21 ** (1 / years) - 1)
    assert out.loc["W · 0", "net_return"] == pytest.approx(1.1 ** (1 / years) - 1)
    np.testing.assert_allclose(out["cost"], out["gross_return"] - out["net_return"])
    assert out.loc["M · 0.05", "net_return"] == pytest.approx(0.0)
    np.testing.assert_allclose(out["turnover"], [0.6, 0.3])