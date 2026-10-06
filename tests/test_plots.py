"""Pruebas de figuras (P3): cada función devuelve una Figure con título, ejes y leyenda.

Checklist de P3: "cada figura tiene título, ejes y leyenda". Se usan resultados sintéticos con la
forma que documenta cada función; ninguna prueba lee `data/` ni usa red.
"""

import io
import math

import numpy as np
import optuna
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src import plots
from src.data import block_dates
from src.regimes import (
    fit_regime_model,
    market_index,
    predict_regimes,
    regime_features,
    viterbi_path,
)

TICKERS = ["A0", "A1", "A2"]


def assert_complete(fig: Figure) -> None:
    """Título, ejes etiquetados, leyenda (o barra de color) y que se pueda guardar como PNG."""
    assert isinstance(fig, Figure)
    suptitle = fig._suptitle.get_text() if fig._suptitle is not None else ""
    colorbars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert axes
    for ax in axes:
        assert ax.get_title() or suptitle, "falta título"
        assert ax.get_xlabel(), f"falta etiqueta x en '{ax.get_title()}'"
        assert ax.get_ylabel(), f"falta etiqueta y en '{ax.get_title()}'"
        if hasattr(ax, "get_zlabel"):
            assert ax.get_zlabel(), "falta etiqueta z"
    has_legend = bool(fig.legends) or any(ax.get_legend() for ax in axes) or bool(colorbars)
    assert has_legend, "falta leyenda o barra de color"
    fig.savefig(io.BytesIO(), format="png")


@pytest.fixture
def regime_data(synthetic_prices, config_test):
    """Variables, etiqueta filtrada y Viterbi de un HMM sobre los precios sintéticos."""
    features = regime_features(synthetic_prices, config_test)
    model = fit_regime_model(features, "hmm", 42, dict(config_test, regime_hmm_n_init=2))
    return features, predict_regimes(model, features), viterbi_path(model, features)


@pytest.fixture
def equity(synthetic_prices) -> pd.Series:
    """Una curva de equity sintética en USD."""
    return 1_000_000 * synthetic_prices["A0"]["close"] / synthetic_prices["A0"]["close"].iloc[0]


@pytest.fixture
def study() -> optuna.Study:
    """Estudio pequeño con pruebas infactibles (−inf), como los de P2."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        x = trial.suggest_float("k_stop", 1.0, 4.0)
        n = trial.suggest_int("sma_fast", 5, 30)
        return -math.inf if x > 3.5 else math.sin(x) * n

    result = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.RandomSampler(seed=42))
    result.optimize(objective, n_trials=40)
    return result


def test_plot_equity(equity, config_test):
    curves = {"Risk Parity": equity, "Pesos iguales": equity * 0.9, "Buy & hold": equity * 1.1}
    assert_complete(plots.plot_equity(curves, block_dates(config_test)))


def test_plot_equity_rejects_too_many_series(equity, config_test):
    """Más de 8 series no se distinguen por color: se pide agrupar en vez de inventar colores."""
    curves = {f"c{i}": equity for i in range(9)}
    with pytest.raises(ValueError):
        plots.plot_equity(curves, block_dates(config_test))


def test_plot_drawdown(equity):
    drawdown = equity / equity.cummax() - 1
    assert_complete(plots.plot_drawdown({"Risk Parity": drawdown, "Pesos iguales": drawdown}))


def test_plot_returns_table():
    rng = np.random.default_rng(0)
    years = pd.Index([2018, 2019, 2020], name="Año")
    tables = {
        "mensual": pd.DataFrame(rng.normal(0, 0.03, (3, 12)), index=years,
                                columns=pd.Index(range(1, 13), name="Mes")),
        "trimestral": pd.DataFrame(rng.normal(0, 0.05, (3, 4)), index=years,
                                   columns=pd.Index(["T1", "T2", "T3", "T4"], name="Trimestre")),
        "anual": pd.DataFrame({"Retorno": rng.normal(0, 0.1, 3)}, index=years),
    }
    assert_complete(plots.plot_returns_table(tables))


def test_plot_sensitivity():
    sensitivity = pd.DataFrame(
        {"−20%": [0.1, -0.3, 0.05], "+20%": [-0.2, 0.15, 0.0]},
        index=["sma_fast", "k_stop", "reward_ratio"],
    )
    assert_complete(plots.plot_sensitivity(sensitivity))


def test_plot_cost_curve():
    bps = pd.Index(range(0, 105, 5), name="bps")
    curve = pd.DataFrame({"net_return": 0.12 - 0.002 * bps.to_numpy()}, index=bps)
    assert_complete(plots.plot_cost_curve(curve, 29))


def test_plot_regime_timeline(synthetic_prices, regime_data):
    _, filtered, viterbi = regime_data
    assert_complete(plots.plot_regime_timeline(market_index(synthetic_prices), filtered, viterbi))


def test_plot_regime_features(regime_data):
    features, filtered, _ = regime_data
    assert_complete(plots.plot_regime_features(features, filtered))


def test_plot_equity_regimes(equity, regime_data):
    _, filtered, _ = regime_data
    assert_complete(plots.plot_equity_regimes(equity, filtered))


def test_plot_risk_contributions():
    contributions = pd.DataFrame(
        {"Risk Parity": [1 / 3] * 3, "Pesos iguales": [0.2, 0.3, 0.5]}, index=TICKERS
    )
    assert_complete(plots.plot_risk_contributions(contributions))


def test_plot_signal_heatmap(synthetic_prices):
    rng = np.random.default_rng(0)
    index = synthetic_prices["A0"].index
    strength = pd.DataFrame(rng.choice([-1, -2 / 3, 0, 2 / 3, 1], (len(index), 3)), index=index,
                            columns=TICKERS)
    assert_complete(plots.plot_signal_heatmap(strength))


def test_plot_corr_by_regime(synthetic_prices, regime_data):
    _, filtered, _ = regime_data
    returns = pd.DataFrame({t: df["close"].pct_change() for t, df in synthetic_prices.items()})
    corr = {name: returns[filtered == name].corr() for name in filtered.dropna().unique()}
    assert_complete(plots.plot_corr_by_regime(corr))


def test_plot_rebalance_sweep():
    index = pd.MultiIndex.from_product([["W", "M"], [0.02, 0.05]], names=["frecuencia", "banda"])
    sweep = pd.DataFrame(
        {"gross_return": [0.10, 0.11, 0.10, 0.09], "cost": [0.03, 0.02, 0.015, 0.01],
         "net_return": [0.07, 0.09, 0.085, 0.08], "turnover": [3.0, 2.0, 1.5, 1.0]},
        index=index,
    )
    assert_complete(plots.plot_rebalance_sweep(sweep))


def test_plot_surface_3d():
    rng = np.random.default_rng(0)
    grid = pd.DataFrame({"sma_fast": rng.uniform(5, 30, 60), "k_stop": rng.uniform(1, 4, 60)})
    grid["calmar"] = np.sin(grid["k_stop"]) + grid["sma_fast"] / 30
    grid.loc[:4, "calmar"] = -np.inf
    assert_complete(plots.plot_surface_3d(grid, "sma_fast", "k_stop"))


def test_plot_optimization_history(study):
    assert_complete(plots.plot_optimization_history(study))


def test_plot_param_importance(study):
    assert_complete(plots.plot_param_importance(study, seed=42))


def test_plot_param_importance_is_deterministic(study):
    """Misma semilla, misma importancia (fANOVA es aleatorio)."""
    first = plots.plot_param_importance(study, seed=42).axes[0].patches
    second = plots.plot_param_importance(study, seed=42).axes[0].patches
    assert [p.get_width() for p in first] == [p.get_width() for p in second]


def test_plot_slices(study):
    assert_complete(plots.plot_slices(study))


def test_all_figure_functions_are_tested():
    """Las 16 funciones de la tabla de P3 existen y tienen prueba en este archivo."""
    names = [n for n in dir(plots) if n.startswith("plot_")]
    assert len(names) == 16
    tested = {n.removeprefix("test_") for n in globals() if n.startswith("test_plot_")}
    assert set(names) <= tested
