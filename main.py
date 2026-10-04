"""Ejecuta el proyecto completo sin intervención ni red: `python main.py`.

Carga → auditoría → régimen → estudios de diagnóstico → walk-forward → backtests finales
(Risk Parity y pesos iguales) → métricas → `results/` y `docs/figuras/` (CLAUDE.md, sección 9).
"""

import pickle
from pathlib import Path

from src.backtest import run_backtest
from src.data import audit_prices, download_prices, load_prices, load_risk_free
from src.metrics import compute_metrics
from src.optimize import diagnostic_study, select_plateau, walk_forward, wf_efficiency
from src.portfolio import sleeve_weights
from src.regimes import label_regimes
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
    "regime_window": 63,
    "regime_method": "hmm",  # PENDIENTE: lo define P3 en SPEC_portafolio.md
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
    "adv_window": 20,  # días para el volumen promedio en dólares
}


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


def stage_report(backtests: dict, rf, config: dict) -> None:
    """Calcula métricas y guarda resultados en `results/`; las figuras van a `docs/figuras/`."""
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(config["figures_dir"]).mkdir(parents=True, exist_ok=True)
    metrics = {
        name: compute_metrics(result.equity, result.trades, rf, config["periods_per_year"])
        for name, result in backtests.items()
    }
    with open(results_dir / "final_backtests.pkl", "wb") as f:
        pickle.dump({"backtests": backtests, "metrics": metrics}, f)


def main() -> None:
    """Corre todas las etapas en orden."""
    prices, rf, _audit = stage_load(CONFIG)
    regimes = stage_regimes(prices, CONFIG)
    stage_diagnostics(prices, CONFIG)
    wf = stage_walk_forward(prices, regimes, CONFIG)
    backtests = stage_final_backtests(prices, regimes, wf, CONFIG)
    stage_report(backtests, rf, CONFIG)


if __name__ == "__main__":
    main()
