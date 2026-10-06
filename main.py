"""Ejecuta el proyecto completo sin intervención ni red: `python main.py`.

Carga → auditoría → régimen → corrida base → estudios de diagnóstico → walk-forward → backtests
finales (Risk Parity y pesos iguales) → métricas → impacto de mercado y auditoría de sesgos →
`results/` y `docs/figuras/` (CLAUDE.md, sección 9).
"""

import pickle
from pathlib import Path

import pandas as pd

from src.backtest import market_impact, run_backtest
from src.data import audit_prices, block_dates, download_prices, load_prices, load_risk_free
from src.metrics import compute_metrics, drawdown_series
from src.optimize import diagnostic_study, select_plateau, walk_forward, wf_efficiency
from src.plots import plot_drawdown, plot_equity, plot_rebalance_sweep, plot_risk_contributions
from src.portfolio import (
       portfolio_results,
       risk_contribution_plot_frame,
       sleeve_weights,
       sweep_plot_frame,
   )
from src.regimes import REGIME_NAMES, label_regimes
from src.signals import generate_signals

SEED = 42

# Todos los valores fijos del SPEC. Ningún módulo de src/ los escribe a mano: los recibe en `config`.
CONFIG = {
    "seed": SEED,
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
        "risk_per_trade": 0.01,
    },
    # Rangos de búsqueda, 9 dimensiones por régimen (SPEC punto 7)
    "search_ranges": {
        "sma_fast": (5, 30),
        "sma_slow": (40, 120),
        "rsi_window": (7, 21),
        "rsi_lo": (20, 40),
        "rsi_hi": (60, 80),
        "k_stop": (1.0, 4.0),
        "reward_ratio": (1.0, 4.0),
        "max_holding": (5, 40),
        "risk_per_trade": (0.005, 0.02),
    },
    # Optimización (SPEC punto 7, P2)
    "n_trials_diagnostic": 200,  # random y TPE sobre todo train
    "n_trials_wf": 150,  # por régimen y por ventana
    "plateau_top_frac": 0.10,
    "min_trades_per_window": 24,  # por activo, en 6 meses; prorrateado por régimen
    "min_regime_days": 21,  # por debajo se usa el θ único de la ventana
    "embargo_days": 5,
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
    "regime_multiplier": {  # PENDIENTE: lo define P4 en SPEC_portafolio.md
        "tendencia": 1.0,
        "reversion": 0.7,
        "crisis": 0.3,
    },
    "cov_method": "ledoit_wolf",  # PENDIENTE: lo define P4 en SPEC_portafolio.md
    "cov_window": 126,  # PENDIENTE: lo define P4 en SPEC_portafolio.md
    "rebalance_frequency": "M",  # PENDIENTE: lo define P4 en SPEC_portafolio.md
    "rebalance_band": 0.05,  # PENDIENTE: lo define P4 en SPEC_portafolio.md
    "resize_on_rebalance": False,  # PENDIENTE: lo acuerdan P1 y P4
    # Barridos y robustez
    "sensitivity_pct": 0.20,
    "cost_sweep_bps": list(range(0, 105, 5)),  # ida y vuelta, 0 a 100 bps
    "rebalance_bands": [0.0, 0.025, 0.05, 0.10, 0.20],
    "rebalance_frequencies": ["W", "M", "Q"],
    # Impacto de mercado ex post (P1)
    "adv_window": 20,  # días del volumen promedio en dólares y de la σ diaria
}

TRADE_PARAM_KEYS = ["k_stop", "reward_ratio", "max_holding", "risk_per_trade"]


def save_results(obj: object, name: str, config: dict) -> None:
    """Guarda un resultado en `results/<name>.pkl` para que el notebook solo lo cargue."""
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def save_figure(fig, name: str, config: dict) -> None:
    """Guarda una figura de `src/plots.py` en `docs/figuras/<name>.png`."""
    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=150)


def stage_load(config: dict) -> tuple[dict, object, object]:
    """Carga precios y tasa libre de riesgo, y audita los datos.

    Solo descarga si falta algún archivo en `data/`; con los datos congelados no usa red.
    """
    symbols = config["tickers"] + [config["risk_free_ticker"]]
    data_dir = Path(config["data_dir"])
    if not all((data_dir / f"{s.lstrip('^')}.csv").exists() for s in symbols):
        download_prices(symbols, config["start"], config["end"], out_dir=config["data_dir"])
    prices = load_prices(config["tickers"], data_dir=config["data_dir"])
    rf = load_risk_free(data_dir=config["data_dir"])
    audit = audit_prices(prices)
    return prices, rf, audit


def stage_regimes(prices: dict, config: dict):
    """Etiqueta de régimen filtrada y causal para todas las fechas (P3)."""
    return label_regimes(prices, config)


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


def stage_diagnostics(prices: dict, config: dict) -> dict:
    """Estudios de diagnóstico sobre train con θ único: random search y TPE (P2)."""
    studies = {
        sampler: diagnostic_study(
            prices, config, sampler, config["n_trials_diagnostic"], config["seed"]
        )
        for sampler in ("random", "tpe")
    }
    plateau = select_plateau(studies["tpe"], config["plateau_top_frac"])
    return {"studies": studies, "plateau": plateau}


def stage_walk_forward(prices: dict, regimes, config: dict) -> dict:
    """Walk-forward rolling y anchored, por régimen y con θ único (P2)."""
    runs = {
        (mode, per_regime): walk_forward(prices, regimes, config, mode=mode, per_regime=per_regime)
        for mode in ("rolling", "anchored")
        for per_regime in (True, False)
    }
    efficiency = {key: wf_efficiency(run) for key, run in runs.items()}
    return {"runs": runs, "efficiency": efficiency}


def stage_final_backtests(prices: dict, regimes, wf: dict, config: dict) -> dict:
    """Backtests finales con θ congelados: Risk Parity contra pesos iguales.

    PENDIENTE: las llaves de la salida de `walk_forward` ("params_by_regime", "trade_params") se
    acuerdan con P2 al integrar `optimize.py`.
    """
    main_run = wf["runs"][("rolling", True)]
    signals = generate_signals(prices, main_run["params_by_regime"], regimes, config)
    return {
        method: run_backtest(
            prices,
            signals,
            sleeve_weights(prices, signals, regimes, config, method=method),
            main_run["trade_params"],
            config,
        )
        for method in ("risk_parity", "equal")
    }


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

def stage_portfolio(prices: dict, rf, regimes, config: dict) -> dict:
       """Resultados de P4 con θ base, solo en train. Validation se mide una vez con los θ finales."""
       blocks = block_dates(config)
       params = dict.fromkeys(REGIME_NAMES, config["base_params"])
       results = portfolio_results(prices, params, regimes, config, 0.0, {"train": blocks["train"]})
       save_results(results, "portafolio", config)
       save_figure(
           plot_risk_contributions(risk_contribution_plot_frame(results["risk_contributions"])),
           "portafolio_contribuciones_riesgo",
           config,
       )
       sweep = sweep_plot_frame(results["sweep"]["train"], blocks["train"], config)
       save_figure(plot_rebalance_sweep(sweep), "portafolio_barrido_rebalanceo", config)
       return results


def stage_report(backtests: dict, rf, config: dict) -> None:
    """Calcula métricas y guarda resultados en `results/`; las figuras van a `docs/figuras/`."""
    metrics = {
        name: compute_metrics(result.equity, result.trades, rf, config["periods_per_year"])
        for name, result in backtests.items()
    }
    save_results({"backtests": backtests, "metrics": metrics}, "final_backtests", config)
    labels = {"risk_parity": "Risk Parity", "equal": "Pesos iguales"}
    save_figure(
        plot_equity(
            {labels[name]: result.equity for name, result in backtests.items()}, block_dates(config)
        ),
        "final_equity",
        config,
    )
    save_figure(
        plot_drawdown(
            {labels[name]: drawdown_series(result.equity) for name, result in backtests.items()}
        ),
        "final_drawdown",
        config,
    )


def stage_market_impact(result, prices: dict, config: dict) -> dict:
    """Impacto de mercado ex post sobre las operaciones de test (P1, tarea 10).

    Se reporta en bps por llenado y como fracción del equity y del retorno de test. Si el retorno de
    test es negativo, `impact_pct_return` sale negativo: el impacto haría la pérdida más grande.
    """
    test_start, test_end = block_dates(config)["test"]
    trades = result.trades[result.trades["entry_date"].between(test_start, test_end)]
    detail = market_impact(trades, prices, config)
    bps = pd.concat([detail["entry_bps"], detail["exit_bps"]])
    equity = result.equity.loc[test_start:test_end]
    test_return = equity.iloc[-1] / equity.iloc[0] - 1
    impact_pct_equity = detail["impact_cost"].sum() / equity.iloc[0]
    summary = {
        "n_trades": len(detail),
        "mean_bps": bps.mean(),
        "median_bps": bps.median(),
        "p95_bps": bps.quantile(0.95),
        "max_participation": detail[["entry_participation", "exit_participation"]].max().max(),
        "impact_cost": detail["impact_cost"].sum(),
        "impact_pct_equity": impact_pct_equity,
        "test_return": test_return,
        "impact_pct_return": impact_pct_equity / test_return,
    }
    impact = {"detail": detail, "summary": summary}
    save_results(impact, "impacto_mercado", config)
    return impact


def stage_bias_audit(audit: pd.DataFrame, wf: dict, impact: dict, config: dict) -> None:
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
            "(`test_signals.py`), régimen (`test_regimes.py`) y el pipeline completo "
            "(`test_pipeline.py`); golden test 9 (t → t+1). El régimen operado es la etiqueta "
            "filtrada (forward); Viterbi solo aparece en una figura.",
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
            f"Optuna TPE con {config['n_trials_wf']} pruebas por estudio y ventana del walk-forward, "
            f"más {config['n_trials_diagnostic']} random y {config['n_trials_diagnostic']} TPE de "
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
            f"{impact_summary['n_trades']} operaciones de test: {impact_summary['mean_bps']:.1f} bps "
            f"promedio por llenado (p95 {impact_summary['p95_bps']:.1f} bps), "
            f"{impact_summary['impact_pct_equity']:.2%} del equity inicial de test.",
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


def main() -> None:
    """Corre todas las etapas en orden."""
    prices, rf, audit = stage_load(CONFIG)
    regimes = stage_regimes(prices, CONFIG)
    save_results({"audit": audit, "regimes": regimes}, "datos_regimen", CONFIG)
    stage_portfolio(prices, rf, regimes, CONFIG)
    stage_save_base_run(stage_base_run(prices, CONFIG), CONFIG)
    stage_diagnostics(prices, CONFIG)
    wf = stage_walk_forward(prices, regimes, CONFIG)
    backtests = stage_final_backtests(prices, regimes, wf, CONFIG)
    stage_report(backtests, rf, CONFIG)
    impact = stage_market_impact(backtests["risk_parity"], prices, CONFIG)
    stage_bias_audit(audit, wf, impact, CONFIG)


if __name__ == "__main__":
    main()
