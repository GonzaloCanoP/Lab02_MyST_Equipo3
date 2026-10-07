"""Ejecuta el proyecto completo sin intervención ni red: `python main.py`.

Carga → auditoría → régimen → portafolio y corrida base (train) → estudios de diagnóstico →
robustez (sensibilidad, costos, pregunta 1) → walk-forward → backtests finales (Risk Parity, pesos
iguales, activos individuales y buy & hold) → métricas por bloque → impacto de mercado y auditoría
de sesgos → `results/` y `docs/figuras/` (CLAUDE.md, sección 9).

Con `CONFIG["final_run"] = False` los datos se recortan al fin de validation: test no se toca. La
corrida única de test se hace con `final_run = True`, con los θ congelados y el hash del commit
registrado antes (SPEC punto 1).
"""

import pickle
import subprocess
import time
from pathlib import Path

import pandas as pd

from src.backtest import market_impact, run_backtest
from src.data import audit_prices, block_dates, download_prices, load_prices, load_risk_free
from src.metrics import (
    breakeven_winrate,
    buy_and_hold_equity,
    compute_metrics,
    drawdown_series,
    exposure_metrics,
    metrics_by_block,
    returns_table,
    volatility_matched,
)
from src.optimize import (
    cost_sweep,
    diagnostic_study,
    select_plateau,
    sensitivity,
    single_indicator_comparison,
    surface_grid,
    walk_forward,
    walk_forward_signals,
    wf_efficiency,
)
from src.plots import (
    plot_corr_by_regime,
    plot_cost_curve,
    plot_drawdown,
    plot_equity,
    plot_equity_regimes,
    plot_optimization_history,
    plot_param_importance,
    plot_rebalance_sweep,
    plot_regime_features,
    plot_regime_timeline,
    plot_returns_table,
    plot_risk_contributions,
    plot_sensitivity,
    plot_signal_heatmap,
    plot_slices,
    plot_strategy_vs_benchmark,
    plot_surface_3d,
)
from src.portfolio import (
    choose_estimator,
    choose_rebalance,
    portfolio_results,
    rebalance_sweep,
    risk_contribution_plot_frame,
    sleeve_weights,
    sweep_plot_frame,
    weight_stability,
)
from src.regimes import REGIME_NAMES, label_regimes, regime_performance, regime_results
from src.signals import compute_indicators, generate_signals, sma_macd_vote_correlation

SEED = 42

# Todos los valores fijos del SPEC. Ningún módulo de src/ los escribe a mano: los recibe en `config`.
CONFIG = {
    "seed": SEED,
    # True solo para la corrida única de test (θ congelados y hash registrado); con False los datos
    # se recortan al fin de validation y test no se toca.
    "final_run": True,
    # Datos (SPEC punto 1, CLAUDE.md sección 3)
    "tickers": ["AAPL", "MSFT", "META", "AMD", "XOM", "SMH", "GLD", "COPX"],
    "risk_free_ticker": "^IRX",
    "data_dir": "data",
    "results_dir": "results",
    "figures_dir": "docs/figuras",
    "start": "2017-01-02",
    "end": "2026-09-30",
    "warmup_end": "2017-12-31",  # 2017 solo calienta indicadores; no genera señales
    "blocks": {
        "train": ("2018-01-01", "2021-12-31"),
        "validation": ("2022-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2026-09-30"),
    },
    # Capital y costos (SPEC punto 6)
    "initial_capital": 1_000_000.0,
    "commission": 0.00125,  # por lado, sobre el nocional
    "slippage": 0.0002,  # 2 bps en contra en todo llenado
    "borrow_rate": 0.0025,  # anual sobre el nocional en corto, devengado /252
    "periods_per_year": 252,
    # Indicadores fijos (SPEC punto 2)
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_window": 14,
    "min_agree": 2,  # regla 2 de 3 (SPEC punto 3)
    # Valores base de θ (SPEC puntos 2, 4 y 5)
    "base_params": {
        "sma_fast": 20,
        "sma_slow": 50,
        "rsi_window": 14,
        "rsi_lo": 30,
        "rsi_hi": 70,
        "k_stop": 2.0,
        "reward_ratio": 2.0,
        "max_holding": 20,
    },
    # Rangos de búsqueda, 8 dimensiones (SPEC punto 7, v1.4)
    "search_ranges": {
        "sma_fast": (5, 30),
        "sma_slow": (40, 120),
        "rsi_window": (7, 21),
        "rsi_lo": (20, 40),
        "rsi_hi": (60, 80),
        "k_stop": (1.0, 4.0),
        "reward_ratio": (1.0, 4.0),
        "max_holding": (5, 40),
    },
    # Walk-forward: solo los parámetros de salida se optimizan por régimen y ventana; los de
    # indicadores quedan en el θ* del diagnóstico sobre train (SPEC punto 7, v1.4)
    "wf_search_params": ["k_stop", "reward_ratio", "max_holding"],
    # Optimización (SPEC punto 7, P2)
    "n_trials_diagnostic": 200,  # random y TPE sobre todo train
    "n_trials_wf": 150,  # por régimen y por ventana
    "plateau_top_frac": 0.10,
    "min_trades_per_window": 24,  # por cada 6 meses de ventana, sumando los 8 activos
    "min_regime_days": 21,  # por debajo se usa el θ único de la ventana
    "embargo_days": 5,
    "surface_points": 8,  # valores por dimensión de la superficie 3D (P2, tarea 13)
    # Walk-forward (SPEC punto 1)
    "wf_train_months": 6,
    "wf_test_months": 1,
    "wf_step_months": 1,
    "n_jobs": -1,  # paralelismo entre ventanas, nunca dentro de un estudio
    # Régimen (SPEC_portafolio, P3)
    "regime_window": 63,  # días hábiles de la ventana móvil de las variables
    "regime_method": "rules",  # elegido por P3 con train y validation (SPEC_portafolio, Régimen)
    "regime_n_states": 3,  # tendencia, reversion, crisis
    "regime_first_fit": "2018-01-01",  # primer ajuste; 2017 solo aporta historia de calentamiento
    "regime_min_fit_obs": 126,  # observaciones válidas mínimas para ajustar un modelo
    "regime_refit_freq": "MS",  # reajuste al inicio de cada mes con ventana expandible
    "regime_crisis_quantile": 0.80,  # reglas: volatilidad sobre este cuantil → crisis
    "regime_trend_quantile": 0.50,  # reglas: eficiencia sobre este cuantil → tendencia
    "regime_kmeans_n_init": 10,  # reinicios de K-means
    "regime_hmm_covariance": "full",  # covarianza del GaussianHMM
    "regime_hmm_n_iter": 200,  # iteraciones máximas de EM del HMM
    "regime_hmm_n_init": 10,  # reinicios del HMM; se queda el de mayor verosimilitud
    # Portafolio (SPEC_portafolio, P4)
    "regime_multiplier": {"tendencia": 1.0, "reversion": 0.7, "crisis": 0.3},
    "cov_method": "ledoit_wolf",
    "cov_window": 126,  # días hábiles, mayor que la SMA lenta máxima
    "rebalance_frequency": "M",  # revisión mensual de w^RP (disparador híbrido)
    "rebalance_band": 0.05,  # δ: se adopta el w^RP nuevo solo si ‖Δw‖₁ > δ
    "conflict_corr_threshold": 0.7,  # ρ sobre la que se resuelven señales opuestas
    "resize_on_rebalance": False,  # acordado por P1 y P4
    # Decisiones de validation (SPEC_portafolio): Ledoit-Wolf se reemplaza solo si otro estimador es
    # al menos 10% más estable; M y δ = 0.05 se mantienen salvo +1 pp anual de retorno neto
    "estimator_min_gain": 0.10,
    "rebalance_min_gain": 0.01,
    # Barridos y robustez
    "sensitivity_pct": 0.20,
    "cost_sweep_bps": list(range(0, 105, 5)),  # ida y vuelta, 0 a 100 bps
    "rebalance_bands": [0.0, 0.025, 0.05, 0.10, 0.20],
    "rebalance_frequencies": ["W", "M", "Q"],
    # Impacto de mercado ex post (P1)
    "adv_window": 20,  # días del volumen promedio en dólares y de la σ diaria
}

TRADE_PARAM_KEYS = ["k_stop", "reward_ratio", "max_holding"]


def log(message: str) -> None:
    """Avance de la corrida con la hora, para seguir las etapas largas."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_results(obj: object, name: str, config: dict) -> None:
    """Guarda un resultado en `results/<name>.pkl` para que los notebooks solo lo carguen."""
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def save_figure(fig, name: str, config: dict) -> None:
    """Guarda una figura de `src/plots.py` en `docs/figuras/<name>.png`."""
    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=150)


def active_blocks(prices: dict, config: dict) -> dict[str, tuple]:
    """Bloques con datos en esta corrida: sin `final_run`, test no aparece."""
    last = next(iter(prices.values())).index[-1]
    return {name: (start, end) for name, (start, end) in block_dates(config).items() if start <= last}


def stage_load(config: dict) -> tuple[dict, pd.Series, pd.DataFrame]:
    """Carga precios y tasa libre de riesgo y audita los datos completos.

    Solo descarga si falta algún archivo en `data/`. Sin `final_run`, precios y tasa se recortan al
    fin de validation después de la auditoría, así que ninguna etapa posterior ve test.
    """
    symbols = config["tickers"] + [config["risk_free_ticker"]]
    data_dir = Path(config["data_dir"])
    if not all((data_dir / f"{s.lstrip('^')}.csv").exists() for s in symbols):
        download_prices(symbols, config["start"], config["end"], out_dir=config["data_dir"])
    prices = load_prices(config["tickers"], data_dir=config["data_dir"])
    rf = load_risk_free(data_dir=config["data_dir"])
    audit = audit_prices(prices)
    if not config["final_run"]:
        cut = block_dates(config)["validation"][1]
        prices = {ticker: df.loc[:cut] for ticker, df in prices.items()}
        rf = rf.loc[:cut]
    return prices, rf, audit


def stage_regime_report(prices: dict, regimes: pd.Series, config: dict) -> dict:
    """Resultados y figuras de régimen (P3): comparación de métodos, validación y tarea 4."""
    results = regime_results(prices, regimes, config)
    save_results(results, "regimen", config)
    train_start, train_end = block_dates(config)["train"]
    save_figure(
        plot_regime_timeline(
            results["market_index"], results["hmm_filtered"], results["hmm_viterbi"]
        ),
        "regimen_filtrada_vs_viterbi",
        config,
    )
    save_figure(
        plot_regime_features(
            results["features"].loc[train_start:train_end], regimes.loc[train_start:train_end]
        ),
        "regimen_variables_train",
        config,
    )
    save_figure(plot_corr_by_regime(results["corr_by_regime"]), "regimen_correlacion_train", config)
    return results


def stage_portfolio(prices: dict, rf: pd.Series, regimes: pd.Series, config: dict) -> dict:
    """Resultados de P4 con θ base, solo en train. Validation se mide con los θ finales."""
    train = block_dates(config)["train"]
    params = dict.fromkeys(REGIME_NAMES, config["base_params"])
    results = portfolio_results(prices, params, regimes, config, rf, {"train": train})
    save_results(results, "portafolio", config)
    save_figure(
        plot_risk_contributions(risk_contribution_plot_frame(results["risk_contributions"])),
        "portafolio_contribuciones_riesgo",
        config,
    )
    sweep = sweep_plot_frame(results["sweep"]["train"], train, config)
    save_figure(plot_rebalance_sweep(sweep), "portafolio_barrido_rebalanceo", config)
    return results


def stage_base_run(prices: dict, config: dict) -> dict:
    """Corrida base sobre train: valores base del SPEC, θ único y régimen ignorado (P1, tarea 4).

    Dos versiones con las mismas señales: cada activo solo (C = Equity) y el portafolio de pesos
    iguales (1/8 constante). Los precios se recortan al fin de train para no tocar validation ni
    test; 2017 solo calienta indicadores, así que `entry_mask` no deja abrir antes de train.
    """
    train_start, train_end = block_dates(config)["train"]
    window = {ticker: df.loc[:train_end] for ticker, df in prices.items()}
    tickers = list(window)
    dates = window[tickers[0]].index
    base = config["base_params"]
    # Régimen ignorado: una sola etiqueta y el mismo θ en las tres llaves (CLAUDE.md, sección 7).
    signals = generate_signals(
        window, dict.fromkeys(REGIME_NAMES, base), pd.Series(REGIME_NAMES[0], index=dates), config
    )
    trade_params = pd.DataFrame({key: base[key] for key in TRADE_PARAM_KEYS}, index=dates)
    entry_mask = pd.Series(dates >= train_start, index=dates)

    results = {
        ticker: run_backtest(
            {ticker: window[ticker]},
            {name: panel[[ticker]] for name, panel in signals.items()},
            pd.DataFrame({ticker: 1.0}, index=dates),
            trade_params,
            config,
            entry_mask,
        )
        for ticker in tickers
    }
    results["Pesos iguales"] = run_backtest(
        window,
        signals,
        pd.DataFrame(1.0 / len(tickers), index=dates, columns=tickers),
        trade_params,
        config,
        entry_mask,
    )
    equity = {name: result.equity.loc[train_start:] for name, result in results.items()}
    drawdowns = {name: drawdown_series(curve) for name, curve in equity.items()}
    return {"results": results, "equity": equity, "drawdowns": drawdowns}


def stage_save_base_run(base_run: dict, config: dict) -> None:
    """Guarda la corrida base y sus figuras: activos individuales y pesos iguales por separado."""
    save_results(base_run, "corrida_base", config)
    train = {"train": block_dates(config)["train"]}
    individual = [name for name in base_run["equity"] if name != "Pesos iguales"]
    for suffix, names in (("activos", individual), ("pesos_iguales", ["Pesos iguales"])):
        save_figure(
            plot_equity({name: base_run["equity"][name] for name in names}, train),
            f"base_equity_{suffix}",
            config,
        )
        save_figure(
            plot_drawdown({name: base_run["drawdowns"][name] for name in names}),
            f"base_drawdown_{suffix}",
            config,
        )


def _study_summary(study) -> dict:
    """Lo que se reporta de un estudio de Optuna: N, factibles, tiempo y la tabla de pruebas."""
    return {
        "user_attrs": dict(study.user_attrs),
        "trials": study.trials_dataframe(attrs=("number", "value", "params", "user_attrs")),
    }


def stage_diagnostics(prices: dict, config: dict) -> dict:
    """Estudios de diagnóstico sobre train con θ único: random y TPE, meseta y superficie (P2)."""
    studies = {
        sampler: diagnostic_study(
            prices, config, sampler, config["n_trials_diagnostic"], config["seed"]
        )
        for sampler in ("random", "tpe")
    }
    plateau = select_plateau(studies["tpe"], config["plateau_top_frac"])
    surface = surface_grid(
        prices, config, studies["random"], studies["tpe"], config["seed"], config["surface_points"]
    )
    study = studies["tpe"]
    save_figure(plot_optimization_history(study), "diagnostico_historia_tpe", config)
    save_figure(
        plot_param_importance(study, config["seed"]), "diagnostico_importancia_tpe", config
    )
    save_figure(plot_slices(study), "diagnostico_slices_tpe", config)
    save_figure(
        plot_surface_3d(surface["grid"], surface["x"], surface["y"]),
        "diagnostico_superficie_random",
        config,
    )
    diagnostics = {
        "studies": {name: _study_summary(s) for name, s in studies.items()},
        "plateau": plateau,
        "surface": surface,
    }
    save_results(diagnostics, "diagnostico", config)
    return diagnostics


def stage_robustness(prices: dict, plateau: dict, config: dict) -> dict:
    """Sensibilidad ±20%, curva de costos y pregunta 1 con el θ* de la meseta, sobre train.

    θ* es el del diagnóstico (θ único, régimen ignorado), así que se evalúa igual: una sola
    etiqueta en todas las fechas. También se reporta la correlación SMA–MACD de SPEC punto 9.
    """
    train = block_dates(config)["train"]
    dates = next(iter(prices.values())).index
    params = dict.fromkeys(REGIME_NAMES, plateau)
    labels = pd.Series(REGIME_NAMES[0], index=dates)
    robustness = {
        "theta": plateau,
        "sensitivity": sensitivity(
            params, prices, labels, config, config["sensitivity_pct"], period=train
        ),
        "cost_curve": cost_sweep(params, prices, labels, config, config["cost_sweep_bps"], train),
        "single_indicator": single_indicator_comparison(params, prices, labels, config, train),
        "sma_macd_correlation": sma_macd_vote_correlation(
            prices, config["base_params"], config, train
        ),
    }
    save_results(robustness, "robustez", config)
    save_figure(plot_sensitivity(robustness["sensitivity"]), "robustez_sensibilidad", config)
    curve = robustness["cost_curve"]
    save_figure(plot_cost_curve(curve, curve["base_cost_bps"].iloc[0]), "robustez_costos", config)
    return robustness


def _fold_table(run: dict) -> pd.DataFrame:
    """Una fila por ventana: fechas, Calmar y retorno IS contra OOS, operaciones y efectivo."""
    rows = []
    for fold in run["folds"]:
        rows.append(
            {
                "fold": fold["fold"],
                "test_start": fold["test_start"],
                "is_calmar": fold["is_metrics"]["calmar"],
                "oos_calmar": fold["oos_metrics"]["calmar"],
                "is_ann_return": fold["is_metrics"]["ann_return"],
                "oos_ann_return": fold["oos_metrics"]["ann_return"],
                "oos_trades": len(fold["oos_trades"]),
                "cash_regimes": ", ".join(fold["cash_regimes"]),
                "n_trials": fold["n_trials_total"],
                "seconds": fold["elapsed_seconds"],
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def stage_walk_forward(prices: dict, regimes: pd.Series, plateau: dict, config: dict) -> dict:
    """Walk-forward rolling y anchored, por régimen y con θ único (P2).

    Los parámetros de indicadores quedan fijos en el θ* del diagnóstico (`plateau`) y por ventana
    solo se optimizan los de `wf_search_params` (SPEC punto 7, v1.4).
    """
    config = {**config, "wf_fixed_params": plateau}
    runs = {}
    for mode in ("rolling", "anchored"):
        for per_regime in (True, False):
            log(f"walk-forward {mode}, {'por régimen' if per_regime else 'θ único'}")
            runs[(mode, per_regime)] = walk_forward(
                prices, regimes, config, mode=mode, per_regime=per_regime
            )
    wf = {
        "runs": runs,
        "efficiency": {key: wf_efficiency(run) for key, run in runs.items()},
        "efficiency_calmar": {key: wf_efficiency(run, "calmar") for key, run in runs.items()},
        "folds": {key: _fold_table(run) for key, run in runs.items()},
        "n_trials_total": sum(run["n_trials_total"] for run in runs.values()),
        "elapsed_seconds": sum(run["elapsed_seconds"] for run in runs.values()),
    }
    save_results(wf, "walk_forward", config)
    return wf


def stage_validation_decisions(
    prices: dict, regimes: pd.Series, wf: dict, plateau: dict, config: dict
) -> dict:
    """Corrida única de validation para las decisiones discretas de P4 (SPEC punto 1).

    Estimador de Σ (estabilidad de los pesos) y frecuencia y δ del rebalanceo (barrido con el θ
    final: indicadores en el θ* de train y k, r, m del walk-forward rolling por régimen), cada uno
    con la regla fijada antes de verlos. `CONFIG` debe coincidir con lo elegido; si no, la corrida
    lo avisa y hay que actualizarlo y volver a correr.
    """
    validation = block_dates(config)["validation"]
    stability = weight_stability(prices, config, period=validation)
    sweep = rebalance_sweep(
        prices,
        dict.fromkeys(REGIME_NAMES, plateau),
        regimes,
        config,
        config["rebalance_bands"],
        config["rebalance_frequencies"],
        trade_params=wf["runs"][("rolling", True)]["trade_params"].reindex(
            next(iter(prices.values())).index
        ),
        period=validation,
    )
    annualized = sweep_plot_frame(sweep, validation, config)
    decisions = {
        "weight_stability": stability,
        "sweep": sweep,
        "sweep_annualized": annualized,
        "cov_method": choose_estimator(stability, "ledoit_wolf", config["estimator_min_gain"]),
        "rebalance": choose_rebalance(
            annualized,
            (config["rebalance_frequency"], config["rebalance_band"]),
            config["rebalance_min_gain"],
        ),
    }
    save_results(decisions, "decisiones_validation", config)
    save_figure(plot_rebalance_sweep(annualized), "validacion_barrido_rebalanceo", config)
    chosen = (decisions["cov_method"], *decisions["rebalance"])
    current = (config["cov_method"], config["rebalance_frequency"], config["rebalance_band"])
    if chosen != current:
        log(f"AVISO: validation elige {chosen} y CONFIG usa {current}; actualizar y volver a correr")
    return decisions


def stage_regime_performance(wf: dict, regimes: pd.Series, rf: pd.Series, config: dict) -> dict:
    """θ por régimen contra θ único y métricas por régimen, fuera de muestra (P3, pregunta 5)."""
    performance = regime_performance(wf["runs"], regimes, rf, config)
    save_results(performance, "regimen_desempeno", config)
    blocks = block_dates(config)
    curves = {
        "θ por régimen": wf["runs"][("rolling", True)]["oos_equity"],
        "θ único": wf["runs"][("rolling", False)]["oos_equity"],
    }
    save_figure(plot_equity(curves, blocks), "regimen_theta_por_regimen_vs_unico", config)
    save_figure(
        plot_equity_regimes(curves["θ por régimen"], regimes), "regimen_equity_oos", config
    )
    return performance


def stage_final_backtests(prices: dict, regimes: pd.Series, wf: dict, config: dict) -> dict:
    """Backtests finales continuos con los θ del walk-forward rolling por régimen.

    Cada mes fuera de muestra opera con el θ de su ventana (`walk_forward_signals`) y las
    posiciones pasan de un mes al siguiente. Risk Parity y pesos iguales usan las mismas señales,
    costos y rebalanceo; cada activo solo corre con C = Equity (SPEC punto 5).
    """
    run = wf["runs"][("rolling", True)]
    dates = next(iter(prices.values())).index
    signals = walk_forward_signals(prices, regimes, run, config)
    trade_params = run["trade_params"].reindex(dates)
    portfolios = {
        method: run_backtest(
            prices,
            signals,
            sleeve_weights(prices, signals, regimes, config, method=method),
            trade_params,
            config,
        )
        for method in ("risk_parity", "equal")
    }
    assets = {
        ticker: run_backtest(
            {ticker: prices[ticker]},
            {name: panel[[ticker]] for name, panel in signals.items()},
            pd.DataFrame({ticker: 1.0}, index=dates),
            trade_params,
            config,
        )
        for ticker in prices
    }
    return {"portfolios": portfolios, "assets": assets, "signals": signals,
            "trade_params": trade_params, "first_oos": run["folds"][0]["test_start"]}


def stage_report(final: dict, prices: dict, rf: pd.Series, regimes: pd.Series, config: dict) -> dict:
    """Métricas por bloque contra buy & hold y activos individuales, exposición y break-even.

    El bloque train de los backtests finales empieza en el primer mes fuera de muestra (antes no
    hay θ), y buy & hold se mide en las mismas fechas.
    """
    blocks = {
        name: (max(start, final["first_oos"]), end)
        for name, (start, end) in active_blocks(prices, config).items()
    }
    ppy = config["periods_per_year"]
    labels = {"risk_parity": "Risk Parity", "equal": "Pesos iguales"}
    strategies = {labels[k]: r for k, r in final["portfolios"].items()} | final["assets"]
    no_trades = pd.DataFrame(columns=["entry_date", "pnl_net"])
    tables = {
        name: metrics_by_block(result.equity, result.trades, blocks, rf, ppy)
        for name, result in strategies.items()
    }
    tables["Buy & hold"] = pd.DataFrame.from_dict(
        {
            name: compute_metrics(buy_and_hold_equity(prices, config, start, end), no_trades, rf, ppy)
            for name, (start, end) in blocks.items()
        },
        orient="index",
    )
    by_block = pd.concat(tables, names=["estrategia", "bloque"])

    # Break-even de SPEC punto 8 por bloque: teórico con la mediana de k y r operados y el ATR/P de
    # train, contra el empírico de las operaciones.
    train_start, train_end = block_dates(config)["train"]
    atr_over_price = pd.Series(
        {
            ticker: (
                compute_indicators(df.loc[:train_end], config["base_params"], config)["atr"]
                / df["close"].loc[:train_end]
            ).loc[train_start:].mean()
            for ticker, df in prices.items()
        }
    ).mean()
    tp = final["trade_params"].dropna(subset=["k_stop"])
    rp = final["portfolios"]["risk_parity"]
    entries = pd.to_datetime(rp.trades["entry_date"])
    breakeven = pd.DataFrame.from_dict(
        {
            name: breakeven_winrate(
                rp.trades[(entries >= start) & (entries <= end)],
                tp["k_stop"].median(),
                tp["reward_ratio"].median(),
                atr_over_price,
                config,
            )
            for name, (start, end) in blocks.items()
        },
        orient="index",
    )
    report = {
        "blocks": blocks,
        "metrics_by_block": by_block,
        "exposure": {labels[k]: exposure_metrics(r) for k, r in final["portfolios"].items()},
        "breakeven": breakeven,
        "returns_table": returns_table(rp.equity.loc[final["first_oos"] :]),
    }
    save_results({**report, "final": final}, "final_backtests", config)

    start = final["first_oos"]
    curves = {labels[k]: r.equity.loc[start:] for k, r in final["portfolios"].items()}
    bh_end = next(iter(prices.values())).index[-1]
    curves["Buy & hold"] = buy_and_hold_equity(prices, config, start, bh_end)
    save_figure(plot_equity(curves, blocks), "final_equity", config)
    save_figure(
        plot_drawdown({name: drawdown_series(curve) for name, curve in curves.items()}),
        "final_drawdown",
        config,
    )
    save_figure(plot_returns_table(report["returns_table"]), "final_retornos_risk_parity", config)
    save_figure(
        plot_equity({t: r.equity.loc[start:] for t, r in final["assets"].items()}, blocks),
        "final_equity_activos",
        config,
    )
    weekly = final["signals"]["strength"].loc[start:].resample("W").last()
    save_figure(plot_signal_heatmap(weekly), "final_mapa_senales", config)
    save_figure(plot_equity_regimes(rp.equity.loc[start:], regimes), "final_equity_regimen", config)
    return report


def stage_benchmark_comparison(
    final: dict, prices: dict, rf: pd.Series, blocks: dict, config: dict
) -> dict:
    """Estrategia (Risk Parity) contra buy & hold, con y sin el mismo nivel de riesgo.

    Buy & hold compra 1/8 en cada activo en el primer mes fuera de muestra y no rebalancea. La
    versión "a la volatilidad de la estrategia" escala sus retornos diarios ex post
    (`volatility_matched`): compara las dos a igual riesgo, pero no es operable.
    """
    start = final["first_oos"]
    end = next(iter(prices.values())).index[-1]
    strategy = final["portfolios"]["risk_parity"].equity.loc[start:]
    buy_hold = buy_and_hold_equity(prices, config, start, end)
    matched = volatility_matched(buy_hold, strategy)
    curves = {
        "Estrategia (Risk Parity)": strategy,
        "Buy & hold": buy_hold,
        "Buy & hold a la vol. de la estrategia (ex post)": matched,
    }
    no_trades = pd.DataFrame(columns=["entry_date", "pnl_net"])
    rp_trades = final["portfolios"]["risk_parity"].trades
    ppy = config["periods_per_year"]
    table = pd.concat(
        {
            "Estrategia (Risk Parity)": metrics_by_block(strategy, rp_trades, blocks, rf, ppy),
            "Buy & hold": metrics_by_block(buy_hold, no_trades, blocks, rf, ppy),
            "Buy & hold a vol. igual (ex post)": metrics_by_block(matched, no_trades, blocks, rf, ppy),
        },
        names=["estrategia", "bloque"],
    )
    comparison = {"curves": curves, "metrics": table}
    save_results(comparison, "comparacion_buy_hold", config)
    save_figure(plot_strategy_vs_benchmark(curves, blocks), "final_vs_buy_and_hold", config)
    return comparison


def stage_market_impact(result, prices: dict, block: tuple, config: dict) -> dict:
    """Impacto de mercado ex post sobre las operaciones de un bloque (P1, tarea 10).

    En la corrida final el bloque es test; antes, validation. Se reporta en bps por llenado y como
    fracción del equity inicial y del retorno del bloque. Si el retorno es negativo,
    `impact_pct_return` sale negativo: el impacto haría la pérdida más grande.
    """
    start, end = block
    trades = result.trades[result.trades["entry_date"].between(start, end)]
    detail = market_impact(trades, prices, config)
    bps = pd.concat([detail["entry_bps"], detail["exit_bps"]])
    equity = result.equity.loc[start:end]
    block_return = equity.iloc[-1] / equity.iloc[0] - 1
    impact_pct_equity = detail["impact_cost"].sum() / equity.iloc[0]
    summary = {
        "block": block,
        "n_trades": len(detail),
        "mean_bps": bps.mean(),
        "median_bps": bps.median(),
        "p95_bps": bps.quantile(0.95),
        "max_participation": detail[["entry_participation", "exit_participation"]].max().max(),
        "impact_cost": detail["impact_cost"].sum(),
        "impact_pct_equity": impact_pct_equity,
        "block_return": block_return,
        "impact_pct_return": impact_pct_equity / block_return if block_return else float("nan"),
        "by_ticker_bps": detail.groupby("ticker")[["entry_bps", "exit_bps"]].mean().mean(axis=1),
    }
    impact = {"detail": detail, "summary": summary}
    save_results(impact, "impacto_mercado", config)
    return impact


def stage_bias_audit(
    audit: pd.DataFrame, wf: dict, impact: dict, block_name: str, config: dict
) -> None:
    """Escribe `results/auditoria_sesgos.md`: un renglón por sesgo con su evidencia (P1, tarea 5).

    Las cifras salen de la corrida; el texto de cada renglón resume la decisión que lo controla.
    """
    impact_summary = impact["summary"]
    data_issues = int(audit.drop(columns=["n_obs", "start", "end", "dropped_dates"]).to_numpy().sum())
    efficiency = ", ".join(
        f"{mode} {'por régimen' if per_regime else 'θ único'} = {value:.2f}"
        for (mode, per_regime), value in wf["efficiency"].items()
    )
    rows = [
        (
            "Look-ahead",
            "La señal de t se ejecuta al Open de t+1 y ese desplazamiento solo existe en "
            "`run_backtest`. Pruebas de truncamiento en verde para indicadores y señales "
            "(`test_signals.py`), régimen (`test_regimes.py`), pesos (`test_portfolio.py`) y el "
            "pipeline completo (`test_pipeline.py`); golden test 9 (t → t+1). El régimen operado es "
            "la etiqueta filtrada (forward); Viterbi solo aparece en una figura.",
        ),
        (
            "Survivorship",
            f"Los {len(audit)} activos se eligieron en 2026 entre empresas y ETFs que sobrevivieron "
            f"hasta hoy, así que el resultado está sesgado al alza. Datos completos y traslapados: "
            f"{audit['n_obs'].iloc[0]:,} días comunes de {audit['start'].min():%Y-%m-%d} a "
            f"{audit['end'].max():%Y-%m-%d}, {int(audit['dropped_dates'].max())} fechas descartadas "
            f"y {data_issues} incidencias de calidad.",
        ),
        (
            "Overfitting / data snooping",
            f"Optuna TPE con {config['n_trials_wf']} pruebas por estudio y ventana del walk-forward "
            f"({wf['n_trials_total']:,} configuraciones evaluadas en total), más "
            f"{config['n_trials_diagnostic']} random y {config['n_trials_diagnostic']} TPE de "
            f"diagnóstico. Se elige el medoide del mejor {config['plateau_top_frac']:.0%} (meseta), "
            f"no el máximo. m(régimen) no entra al espacio de búsqueda. Eficiencia del "
            f"walk-forward: {efficiency}. Test se corre una sola vez con θ congelados.",
        ),
        (
            "Ejecución optimista",
            f"Ejecución al Open de t+1 con slippage de {config['slippage'] * 1e4:.0f} bps en contra "
            f"en todo llenado, incluidos SL y TP; comisión de {config['commission']:.3%} por lado y "
            f"borrow en cortos. Empate intrabarra: primero el stop. Gap más allá del TP llena en el "
            f"TP, sin mejora. Impacto de mercado no modelado; estimado ex post sobre "
            f"{impact_summary['n_trades']} operaciones de {block_name}: "
            f"{impact_summary['mean_bps']:.1f} bps promedio por llenado (p95 "
            f"{impact_summary['p95_bps']:.1f} bps), {impact_summary['impact_pct_equity']:.2%} del "
            f"equity inicial del bloque.",
        ),
        (
            "Selección de muestra / periodo",
            f"Rango {config['start']} → {config['end']} y partición train / validation / test "
            f"fijados en el SPEC antes de ver resultados; 2017 solo calienta indicadores. El "
            f"walk-forward corre continuo sobre los tres bloques. Validation se usa una vez para "
            f"decisiones discretas.",
        ),
    ]
    lines = ["# Auditoría de sesgos", "", "| Sesgo | Evidencia |", "|---|---|"]
    lines += [f"| {bias} | {evidence} |" for bias, evidence in rows]
    path = Path(config["results_dir"]) / "auditoria_sesgos.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage_metadata(config: dict, started: float) -> None:
    """Registra la corrida: commit, si fue la final (con test) y su duración (CLAUDE.md §9)."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout.strip()
    save_results(
        {
            "commit": commit,
            "uncommitted_changes": bool(dirty),
            "final_run": config["final_run"],
            "elapsed_seconds": time.time() - started,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "corrida",
        config,
    )


def main() -> None:
    """Corre todas las etapas en orden."""
    started = time.time()
    config = CONFIG
    log(f"carga de datos (final_run={config['final_run']})")
    prices, rf, audit = stage_load(config)
    regimes = label_regimes(prices, config)
    save_results({"audit": audit, "regimes": regimes}, "datos_regimen", config)
    log("régimen")
    stage_regime_report(prices, regimes, config)
    log("portafolio (train, θ base)")
    stage_portfolio(prices, rf, regimes, config)
    log("corrida base (train)")
    stage_save_base_run(stage_base_run(prices, config), config)
    log("estudios de diagnóstico")
    diagnostics = stage_diagnostics(prices, config)
    log("robustez con θ* de la meseta")
    stage_robustness(prices, diagnostics["plateau"], config)
    wf = stage_walk_forward(prices, regimes, diagnostics["plateau"], config)
    log("decisiones de validation (P4)")
    stage_validation_decisions(prices, regimes, wf, diagnostics["plateau"], config)
    log("desempeño por régimen")
    stage_regime_performance(wf, regimes, rf, config)
    log("backtests finales y reporte")
    final = stage_final_backtests(prices, regimes, wf, config)
    report = stage_report(final, prices, rf, regimes, config)
    stage_benchmark_comparison(final, prices, rf, report["blocks"], config)
    block_name = "test" if config["final_run"] else "validation"
    impact = stage_market_impact(
        final["portfolios"]["risk_parity"], prices, report["blocks"][block_name], config
    )
    stage_bias_audit(audit, wf, impact, block_name, config)
    stage_metadata(config, started)
    log(f"listo en {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
