"""Optimización con Optuna, walk-forward y análisis de robustez (P2).

Firmas propuestas: P2 puede ajustarlas avisando a P1 (main.py). CLAUDE.md, sección 7.
"""

import math
import time

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from src.backtest import run_backtest
from src.metrics import compute_metrics
from src.portfolio import sleeve_weights
from src.signals import compute_indicators, generate_signals, indicator_votes

# Un estudio del walk-forward corre 150 pruebas y hay cientos de estudios: el log INFO de Optuna
# (un renglón por prueba) taparía cualquier mensaje útil. Se conservan advertencias y errores.
optuna.logging.set_verbosity(optuna.logging.WARNING)

_PARAMETER_KINDS = {
    "sma_fast": "int",
    "sma_slow": "int",
    "rsi_window": "int",
    "rsi_lo": "int",
    "rsi_hi": "int",
    "k_stop": "float",
    "reward_ratio": "float",
    "max_holding": "int",
}

_TRADE_PARAM_KEYS = ("k_stop", "reward_ratio", "max_holding")


def _trade_params_panel(
    regimes: pd.Series, params_by_regime: dict, index: pd.Index
) -> pd.DataFrame:
    """Construye el panel fecha × parámetros que recibe run_backtest.

    Cada fecha usa los parámetros correspondientes al régimen vigente.
    Las fechas sin régimen, o de un régimen en efectivo (parámetros None, SPEC punto 7),
    permanecen como NaN y no habilitan entradas.
    """
    labels = regimes.reindex(index)
    panel = pd.DataFrame(np.nan, index=index, columns=_TRADE_PARAM_KEYS, dtype=float)

    for regime_name, params in params_by_regime.items():
        if params is None:
            continue
        mask = labels == regime_name
        for key in _TRADE_PARAM_KEYS:
            panel.loc[mask, key] = params[key]
    panel["regime"] = labels

    return panel


def _signal_params(params_by_regime: dict, config: dict) -> dict:
    """θ para `generate_signals`: un régimen en efectivo (None) usa los valores base.

    Su señal no se opera, porque `_trade_params_panel` deja NaN en esas fechas y `run_backtest`
    no abre sin parámetros; solo hace falta para que `generate_signals` cubra todos los regímenes.
    """
    return {
        name: dict(config["base_params"]) if params is None else params
        for name, params in params_by_regime.items()
    }


def _minimum_trades(train_dates: pd.DatetimeIndex, n_regime_days: int, config: dict) -> int:
    """Actividad mínima de una ventana (SPEC punto 7).

    24 operaciones cerradas por cada `wf_train_months` meses de ventana (una por activo cada dos
    meses), escaladas al largo real de la ventana (diagnóstico sobre todo train y anchored) y
    prorrateadas por los días del régimen: N_min,g = ceil(N · D_g / D).
    """
    n_months = train_dates.to_period("M").nunique()
    per_window = config["min_trades_per_window"] * n_months / config["wf_train_months"]
    return int(math.ceil(per_window * n_regime_days / len(train_dates)))


def _evaluate_params(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    entry_mask: pd.Series | None = None,
    period: tuple | None = None,
    indicator: str | None = None,
) -> tuple:
    """Ejecuta señales, portafolio, backtest y métricas para un candidato.

    Sigue el flujo definido en P2: generate_signals → sleeve_weights → run_backtest → métricas.
    Con `indicator` ∈ {"sma", "macd", "rsi"} las señales salen de ese indicador solo (pregunta 1).
    Con `period`, las métricas usan solo el equity y las operaciones abiertas dentro del periodo.
    """
    tickers = list(prices)

    if not tickers:
        raise ValueError("prices no puede estar vacío.")
    index = prices[tickers[0]].index
    labels = regimes.reindex(index)
    signal_params = _signal_params(params_by_regime, config)
    if indicator is None:
        signals = generate_signals(prices, signal_params, labels, config)
    else:
        signals = _single_indicator_signals(prices, signal_params, labels, config, indicator)
    sleeve = sleeve_weights(prices, signals, labels, config, method="risk_parity")
    trade_params = _trade_params_panel(labels, params_by_regime, index)
    result = run_backtest(prices, signals, sleeve, trade_params, config, entry_mask=entry_mask)

    if period is None:
        equity_for_metrics = result.equity
        trades_for_metrics = result.trades
    else:
        start = pd.Timestamp(period[0])
        end = pd.Timestamp(period[1])
        equity_for_metrics = result.equity.loc[start:end]
        if len(equity_for_metrics) < 2:
            raise ValueError("La ventana debe contener al menos dos observaciones.")
        if len(result.trades) == 0:
            trades_for_metrics = result.trades.copy()
        else:
            entry_dates = pd.to_datetime(result.trades["entry_date"])
            trade_mask = (entry_dates >= start) & (entry_dates <= end)
            trades_for_metrics = result.trades.loc[trade_mask].copy()
    metrics = compute_metrics(
        equity_for_metrics, trades_for_metrics, rf=0.0, periods_per_year=config["periods_per_year"]
    )

    return result, metrics


def _single_indicator_signals(
    prices: dict, params_by_regime: dict, regimes: pd.Series, config: dict, indicator: str
) -> dict[str, pd.DataFrame]:
    """Genera señales usando únicamente SMA, MACD o RSI.

    Se utiliza para comparar cada indicador por separado contra
    la regla principal de confirmación 2 de 3 de P2.
    """
    vote_columns = {"sma": "v_sma", "macd": "v_macd", "rsi": "v_rsi"}

    if indicator not in vote_columns:
        raise ValueError("indicator debe ser 'sma', 'macd' o 'rsi'.")
    tickers = list(prices)

    if not tickers:
        raise ValueError("prices no puede estar vacío.")
    index = prices[tickers[0]].index
    labels = regimes.reindex(index)
    state = pd.DataFrame(0, index=index, columns=tickers, dtype=int)
    strength = pd.DataFrame(0.0, index=index, columns=tickers)
    atr = pd.DataFrame(np.nan, index=index, columns=tickers)
    used_regimes = set(labels.dropna().unique())
    missing = used_regimes - set(params_by_regime)

    if missing:
        raise ValueError(f"Faltan parámetros para los regímenes: {sorted(missing)}")
    vote_column = vote_columns[indicator]

    for ticker in tickers:
        price_data = prices[ticker]
        for regime_name in used_regimes:
            params = params_by_regime[regime_name]
            indicators = compute_indicators(price_data, params, config)
            votes = indicator_votes(indicators, params)
            vote = votes[vote_column].astype(int)
            mask = labels == regime_name
            state.loc[mask, ticker] = vote.loc[mask]
            strength.loc[mask, ticker] = vote.loc[mask].astype(float)
            atr.loc[mask, ticker] = indicators.loc[mask, "atr"]

    return {"state": state, "strength": strength, "atr": atr}


def _evaluate_on_period(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    period: tuple | None,
    indicator: str | None = None,
) -> tuple:
    """`_evaluate_params` con θ congelado sobre un periodo (sensibilidad, costos, pregunta 1).

    Los precios se recortan al fin del periodo (nada posterior entra al cálculo) y solo se abren
    posiciones dentro de él; la historia previa solo calienta indicadores y Σ. Sin `period` se usa
    toda la muestra.
    """
    if period is None:
        return _evaluate_params(params_by_regime, prices, regimes, config, indicator=indicator)
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    window_prices = {ticker: data.loc[:end] for ticker, data in prices.items()}
    index = next(iter(window_prices.values())).index
    entry_mask = pd.Series((index >= start) & (index <= end), index=index)
    return _evaluate_params(
        params_by_regime,
        window_prices,
        regimes,
        config,
        entry_mask=entry_mask,
        period=(start, end),
        indicator=indicator,
    )


def search_space() -> dict:
    """Describe las ocho dimensiones del espacio de búsqueda de P2 (SPEC punto 7, v1.4).

    Los límites numéricos se toman de ``config["search_ranges"]``
    cuando se ejecuta un estudio de Optuna. Aquí únicamente se
    define el tipo de cada hiperparámetro para evitar duplicar los
    valores del SPEC dentro de ``src/``.

    Returns
    -------
    dict
        Nombre de parámetro -> tipo de sugerencia de Optuna
        (``"int"`` o ``"float"``).
    """
    return dict(_PARAMETER_KINDS)


def _suggest_params(trial: optuna.Trial, config: dict, names: list[str] | None = None) -> dict:
    """Propone los parámetros `names` (todo el espacio si es None) con los rangos de CONFIG."""
    params = {}

    for name, kind in search_space().items():
        if names is not None and name not in names:
            continue
        low, high = config["search_ranges"][name]
        if kind == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))

    return params


def _diagnostic_setup(prices: dict, config: dict) -> dict:
    """Entradas fijas de los estudios de diagnóstico sobre todo train (P2, tareas 13 y 14).

    Precios recortados al fin de train, régimen ignorado (una sola etiqueta: la de mayor
    multiplicador, como en la corrida base), entradas solo dentro de train salvo los últimos
    `embargo_days` días (purga y embargo) y la actividad mínima escalada a 4 años.
    """
    train_start = pd.Timestamp(config["blocks"]["train"][0])
    train_end = pd.Timestamp(config["blocks"]["train"][1])
    window_prices = {ticker: data.loc[:train_end].copy() for ticker, data in prices.items()}
    index = next(iter(window_prices.values())).index
    train_dates = index[(index >= train_start) & (index <= train_end)]

    if len(train_dates) == 0:
        raise ValueError("No hay observaciones dentro de Train.")
    embargo_days = int(config["embargo_days"])

    if embargo_days < 0:
        raise ValueError("embargo_days no puede ser negativo.")

    if embargo_days >= len(train_dates):
        raise ValueError("El embargo consume toda la ventana de Train.")
    entry_mask = pd.Series(False, index=index, dtype=bool)
    entry_mask.loc[train_dates] = True
    if embargo_days > 0:
        entry_mask.loc[train_dates[-embargo_days:]] = False
    single_regime = max(config["regime_multiplier"], key=config["regime_multiplier"].get)
    return {
        "prices": window_prices,
        "labels": pd.Series(single_regime, index=index, name="regime"),
        "single_regime": single_regime,
        "entry_mask": entry_mask,
        "period": (train_start, train_end),
        "minimum_trades": _minimum_trades(train_dates, len(train_dates), config),
    }


def _diagnostic_score(params: dict, setup: dict, config: dict) -> tuple[float, int]:
    """Calmar de θ sobre train con costos; −inf si no cumple la actividad mínima (S08).

    Un Calmar no finito (MDD = 0) también se trata como infactible.
    """
    _, metrics = _evaluate_params(
        {setup["single_regime"]: params},
        setup["prices"],
        setup["labels"],
        config,
        entry_mask=setup["entry_mask"],
        period=setup["period"],
    )
    n_trades = int(metrics["n_trades"])
    calmar = float(metrics["calmar"])
    if n_trades < setup["minimum_trades"] or not np.isfinite(calmar):
        return -np.inf, n_trades
    return calmar, n_trades


def diagnostic_study(
    prices: dict, config: dict, sampler: str, n_trials: int, seed: int
) -> optuna.Study:
    """Estudio de diagnóstico sobre todo train con θ único (P2, tareas 13 y 14; S08).

    Cada prueba corre señales → Risk Parity → backtest con costos → Calmar; las que no alcanzan
    la actividad mínima o no tienen Calmar finito valen −inf (`_diagnostic_score`).

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        OHLCV completo; se recorta al fin de train.
    config : dict
        `CONFIG`.
    sampler : str
        "random" (fase 1) o "tpe" (fase 2).
    n_trials : int
        Número de pruebas (200 en `CONFIG`, tope del lab).
    seed : int
        Semilla del sampler.

    Returns
    -------
    optuna.Study
        Con user_attrs: sampler, seed, pruebas pedidas, evaluadas y factibles, tiempo, fechas de
        train, embargo y actividad mínima.
    """
    sampler_name = sampler.lower()

    if sampler_name not in {"random", "tpe"}:
        raise ValueError("sampler debe ser 'random' o 'tpe'.")

    if n_trials <= 0:
        raise ValueError("n_trials debe ser positivo.")

    if not prices:
        raise ValueError("prices no puede estar vacío.")

    if sampler_name == "random":
        optuna_sampler = optuna.samplers.RandomSampler(seed=seed)
    else:
        optuna_sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=optuna_sampler)
    setup = _diagnostic_setup(prices, config)
    train_start, train_end = setup["period"]

    def objective(trial: optuna.Trial) -> float:
        value, n_trades = _diagnostic_score(_suggest_params(trial, config), setup, config)
        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("minimum_trades", setup["minimum_trades"])
        trial.set_user_attr("feasible", bool(np.isfinite(value)))
        if np.isfinite(value):
            trial.set_user_attr("calmar", value)
        return value

    start_time = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    elapsed_seconds = time.perf_counter() - start_time
    feasible_trials = [
        trial
        for trial in study.trials
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(trial.value)
        )
    ]
    study.set_user_attr("sampler", sampler_name)
    study.set_user_attr("seed", int(seed))
    study.set_user_attr("n_trials_requested", int(n_trials))
    study.set_user_attr("n_trials_evaluated", len(study.trials))
    study.set_user_attr("n_trials_feasible", len(feasible_trials))
    study.set_user_attr("elapsed_seconds", float(elapsed_seconds))
    study.set_user_attr("train_start", str(train_start.date()))
    study.set_user_attr("train_end", str(train_end.date()))
    study.set_user_attr("embargo_days", int(config["embargo_days"]))
    study.set_user_attr("minimum_trades", setup["minimum_trades"])

    return study


def select_plateau(study: optuna.Study, top_frac: float = 0.10) -> dict:
    """Selecciona el medoide del mejor porcentaje de trials factibles.

    Los trials con valor no finito, como ``-inf`` por no cumplir
    el mínimo de operaciones, se excluyen. Los parámetros se
    normalizan a [0, 1] usando los límites de sus distribuciones
    de Optuna y se elige el trial con menor distancia total al
    resto del conjunto superior.

    Parameters
    ----------
    study : optuna.Study
        Estudio de Optuna ya terminado.
    top_frac : float
        Fracción de los mejores trials factibles que forman la
        meseta candidata.

    Returns
    -------
    dict
        Parámetros del trial que representa el centro de la meseta.
    """
    if not 0 < top_frac <= 1:
        raise ValueError("top_frac debe estar en el intervalo (0, 1].")
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    feasible = [
        trial for trial in complete_trials if (trial.value is not None and np.isfinite(trial.value))
    ]

    if not feasible:
        raise ValueError("El estudio no contiene trials factibles.")
    maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    ordered = sorted(feasible, key=lambda trial: trial.value, reverse=maximize)
    n_top = max(1, math.ceil(len(ordered) * top_frac))
    top_trials = ordered[:n_top]

    if len(top_trials) == 1:
        return dict(top_trials[0].params)
    parameter_names = sorted(top_trials[0].params)

    for trial in top_trials:
        if set(trial.params) != set(parameter_names):
            raise ValueError("Los trials de la meseta deben usar el mismo espacio de parámetros.")
    normalized_rows = []

    for trial in top_trials:
        row = []
        for name in parameter_names:
            distribution = trial.distributions[name]
            if not hasattr(distribution, "low") or not hasattr(distribution, "high"):
                raise TypeError("select_plateau solo admite parámetros numéricos.")
            low = float(distribution.low)
            high = float(distribution.high)
            value = float(trial.params[name])
            if high == low:
                scaled = 0.0
            else:
                scaled = (value - low) / (high - low)
            row.append(scaled)
        normalized_rows.append(row)
    matrix = np.asarray(normalized_rows, dtype=float)
    differences = matrix[:, None, :] - matrix[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=2))
    total_distance = distances.sum(axis=1)
    medoid_position = int(np.argmin(total_distance))

    return dict(top_trials[medoid_position].params)


def _feasible_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Pruebas completas con valor finito (las −inf no cumplen la actividad mínima)."""
    return [
        trial
        for trial in study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        if trial.value is not None and np.isfinite(trial.value)
    ]


def surface_grid(
    prices: dict,
    config: dict,
    random_study: optuna.Study,
    tpe_study: optuna.Study,
    seed: int,
    n_points: int = 8,
) -> dict:
    """Cuadrícula del Calmar sobre las dos dimensiones más influyentes (P2, tarea 13).

    Las dos dimensiones salen de la importancia fANOVA del estudio TPE (solo pruebas factibles,
    con la semilla del proyecto); el resto de θ queda fijo en la mejor prueba factible del random
    search. Cada punto se evalúa igual que una prueba del diagnóstico: train, θ único, costos y
    −inf si no cumple la actividad mínima. No existe una superficie completa en más de 2
    dimensiones: esta es un corte del espacio de 9.

    Parameters
    ----------
    prices, config
        Datos y `CONFIG`.
    random_study, tpe_study : optuna.Study
        Salidas de `diagnostic_study` con "random" y "tpe".
    seed : int
        Semilla de fANOVA.
    n_points : int
        Valores por dimensión (los enteros repetidos tras redondear se eliminan).

    Returns
    -------
    dict
        "grid" (DataFrame con una columna por dimensión y "calmar"), "x", "y" (nombres de las
        dimensiones) y "fixed" (θ del random search).
    """
    feasible_random, feasible_tpe = _feasible_trials(random_study), _feasible_trials(tpe_study)
    if not feasible_random or not feasible_tpe:
        raise ValueError("Los estudios no tienen pruebas factibles.")
    importance_study = optuna.create_study(directions=tpe_study.directions)
    importance_study.add_trials(feasible_tpe)
    importances = optuna.importance.get_param_importances(
        importance_study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed)
    )
    x, y = list(importances)[:2]  # ordenadas de mayor a menor
    fixed = dict(max(feasible_random, key=lambda trial: trial.value).params)
    axes = {}
    for name in (x, y):
        low, high = config["search_ranges"][name]
        values = np.linspace(low, high, n_points)
        axes[name] = (
            np.unique(np.round(values).astype(int)) if _PARAMETER_KINDS[name] == "int" else values
        )
    setup = _diagnostic_setup(prices, config)
    rows = []
    for x_value in axes[x]:
        for y_value in axes[y]:
            params = {**fixed, x: x_value.item(), y: y_value.item()}
            calmar, n_trades = _diagnostic_score(params, setup, config)
            rows.append(
                {x: x_value.item(), y: y_value.item(), "calmar": calmar, "n_trades": n_trades}
            )
    return {"grid": pd.DataFrame(rows), "x": x, "y": y, "fixed": fixed}


def _window_base(config: dict) -> dict:
    """θ completo del que parte cada estudio del walk-forward (SPEC punto 7, v1.4).

    `wf_fixed_params` es el θ* del diagnóstico sobre train, que `main.py` agrega a la
    configuración antes del walk-forward; sin él se usan los valores base. Las dimensiones de
    `wf_search_params` se optimizan por ventana y las demás quedan fijas en este θ.
    """
    return dict(config.get("wf_fixed_params") or config["base_params"])


def _candidate_params_by_regime(candidate: dict, regime: str | None, config: dict) -> dict:
    """Construye los parámetros usados por un trial.

    Con regime=None se usa el mismo theta en todos los regímenes.
    Al optimizar un régimen específico, ese régimen usa el
    candidato y los demás conservan el θ fijo de la ventana (`_window_base`); como en el estudio
    solo se abre en los días del régimen, esos valores no se operan.
    """
    regime_names = list(config["regime_multiplier"])

    if regime is None:
        return {name: dict(candidate) for name in regime_names}
    params = {name: dict(_window_base(config)) for name in regime_names}
    params[regime] = dict(candidate)

    return params


def optimize_regime(
    prices: dict, regimes: pd.Series, regime: str | None, window: tuple, config: dict, seed: int
) -> dict:
    """Estudio TPE de θ en una ventana de entrenamiento (SPEC punto 7).

    Con `regime=None` se busca un θ único para todos los regímenes. Con un régimen, solo se abren
    posiciones en sus días y la actividad mínima se prorratea (`_minimum_trades`). Si el régimen
    ocupa menos de `min_regime_days` días, no se optimiza y el fold usa el θ único.

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        OHLCV completo; se recorta al fin de la ventana (purga).
    regimes : pd.Series
        Etiqueta filtrada y causal.
    regime : str or None
        Régimen a optimizar, o None para θ único.
    window : tuple
        (inicio, fin) de entrenamiento; los últimos `embargo_days` días no abren (embargo).
    config : dict
        `CONFIG`: rangos, `n_trials_wf`, `plateau_top_frac`, actividad mínima y embargo.
    seed : int
        Semilla del TPESampler (`SEED + i` en la ventana i).

    Returns
    -------
    dict
        params (medoide de la meseta, o None), study, feasible, fallback_to_single, días de la
        ventana y del régimen, minimum_trades, métricas IS del θ elegido y la ventana.
    """
    if not prices:
        raise ValueError("prices no puede estar vacío.")

    if len(window) != 2:
        raise ValueError("window debe ser (inicio, fin).")
    start = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1])

    if start > end:
        raise ValueError("El inicio de la ventana no puede ser posterior al fin.")
    regime_names = list(config["regime_multiplier"])

    if regime is not None and regime not in regime_names:
        raise ValueError(f"Régimen desconocido: {regime!r}")
    window_prices = {ticker: data.loc[:end].copy() for ticker, data in prices.items()}
    first_ticker = next(iter(window_prices))
    index = window_prices[first_ticker].index
    train_dates = index[(index >= start) & (index <= end)]

    if len(train_dates) == 0:
        raise ValueError("La ventana no contiene observaciones.")
    labels = regimes.reindex(index)
    unknown_regimes = set(labels.dropna().unique()) - set(regime_names)

    if unknown_regimes:
        raise ValueError(f"Hay regímenes sin configuración: {sorted(unknown_regimes)}")
    n_window_days = len(train_dates)

    if regime is None:
        n_regime_days = n_window_days
    else:
        n_regime_days = int((labels.loc[train_dates] == regime).sum())
    minimum_trades = _minimum_trades(train_dates, n_regime_days, config)
    if regime is not None:
        if n_regime_days < config["min_regime_days"]:
            return {
                "regime": regime,
                "params": None,
                "study": None,
                "feasible": False,
                "fallback_to_single": True,
                "n_window_days": n_window_days,
                "n_regime_days": n_regime_days,
                "minimum_trades": minimum_trades,
                "is_ann_return": np.nan,
                "is_calmar": np.nan,
                "n_trades": 0,
                "window": (start, end),
            }
    embargo_days = int(config["embargo_days"])

    if embargo_days < 0:
        raise ValueError("embargo_days no puede ser negativo.")

    if embargo_days >= len(train_dates):
        raise ValueError("El embargo consume toda la ventana.")
    entry_mask = pd.Series(False, index=index, dtype=bool)
    entry_mask.loc[train_dates] = True

    if regime is not None:
        entry_mask &= labels == regime

    if embargo_days > 0:
        entry_mask.loc[train_dates[-embargo_days:]] = False
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        candidate = {
            **_window_base(config),
            **_suggest_params(trial, config, config.get("wf_search_params")),
        }
        params_by_regime = _candidate_params_by_regime(candidate, regime, config)
        _, metrics = _evaluate_params(
            params_by_regime,
            window_prices,
            labels,
            config,
            entry_mask=entry_mask,
            period=(start, end),
        )
        n_trades = int(metrics["n_trades"])
        trial.set_user_attr("n_trades", n_trades)
        trial.set_user_attr("minimum_trades", minimum_trades)
        if n_trades < minimum_trades:
            trial.set_user_attr("feasible", False)
            return -np.inf
        calmar = float(metrics["calmar"])
        if not np.isfinite(calmar):
            # Política provisional de P2 para MDD = 0:
            # el trial se considera infactible.
            trial.set_user_attr("feasible", False)
            return -np.inf
        trial.set_user_attr("feasible", True)
        return calmar

    start_time = time.perf_counter()
    study.optimize(objective, n_trials=int(config["n_trials_wf"]), n_jobs=1)
    elapsed_seconds = time.perf_counter() - start_time
    feasible_trials = [
        trial
        for trial in study.trials
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(trial.value)
        )
    ]
    study.set_user_attr("seed", int(seed))
    study.set_user_attr("regime", ("single" if regime is None else regime))
    study.set_user_attr("minimum_trades", minimum_trades)
    study.set_user_attr("n_regime_days", n_regime_days)
    study.set_user_attr("n_window_days", n_window_days)
    study.set_user_attr("n_trials_requested", int(config["n_trials_wf"]))
    study.set_user_attr("n_trials_evaluated", len(study.trials))
    study.set_user_attr("n_trials_feasible", len(feasible_trials))
    study.set_user_attr("elapsed_seconds", float(elapsed_seconds))

    if not feasible_trials:
        return {
            "regime": regime,
            "params": None,
            "study": study,
            "feasible": False,
            "fallback_to_single": False,
            "n_window_days": n_window_days,
            "n_regime_days": n_regime_days,
            "minimum_trades": minimum_trades,
            "is_ann_return": np.nan,
            "is_calmar": np.nan,
            "n_trades": 0,
            "window": (start, end),
        }
    params = {**_window_base(config), **select_plateau(study, config["plateau_top_frac"])}
    params_by_regime = _candidate_params_by_regime(params, regime, config)
    _, selected_metrics = _evaluate_params(
        params_by_regime, window_prices, labels, config, entry_mask=entry_mask, period=(start, end)
    )

    return {
        "regime": regime,
        "params": params,
        "study": study,
        "feasible": True,
        "fallback_to_single": False,
        "n_window_days": n_window_days,
        "n_regime_days": n_regime_days,
        "minimum_trades": minimum_trades,
        "is_ann_return": float(selected_metrics["ann_return"]),
        "is_calmar": float(selected_metrics["calmar"]),
        "n_trades": int(selected_metrics["n_trades"]),
        "window": (start, end),
    }


def _walk_forward_windows(prices: dict, config: dict, mode: str) -> list[dict]:
    """Construye las ventanas mensuales del walk-forward.

    Rolling conserva únicamente los últimos ``wf_train_months``.
    Anchored conserva fijo el inicio de Train y expande el final.
    """
    if mode not in {"rolling", "anchored"}:
        raise ValueError("mode debe ser 'rolling' o 'anchored'.")

    if not prices:
        raise ValueError("prices no puede estar vacío.")
    train_months = int(config["wf_train_months"])
    test_months = int(config["wf_test_months"])
    step_months = int(config["wf_step_months"])

    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("Los tamaños del walk-forward deben ser positivos.")
    overall_start = pd.Timestamp(config["blocks"]["train"][0])
    overall_end = pd.Timestamp(config["blocks"]["test"][1])
    first_ticker = next(iter(prices))
    dates = prices[first_ticker].index
    test_start = overall_start + pd.DateOffset(months=train_months)
    windows = []
    fold_number = 0

    while test_start <= overall_end:
        train_end = test_start - pd.Timedelta(days=1)
        if mode == "rolling":
            train_start = test_start - pd.DateOffset(months=train_months)
        else:
            train_start = overall_start
        test_end = min(
            test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), overall_end
        )
        train_dates = dates[(dates >= train_start) & (dates <= train_end)]
        test_dates = dates[(dates >= test_start) & (dates <= test_end)]
        if len(train_dates) > 0 and len(test_dates) > 0:
            windows.append(
                {
                    "fold": fold_number,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
            fold_number += 1
        test_start = test_start + pd.DateOffset(months=step_months)

    if not windows:
        raise ValueError("No se pudieron construir ventanas walk-forward.")

    return windows


def _optimization_summary(result: dict) -> dict:
    """Resume un resultado de optimize_regime sin guardar el Study completo."""
    study = result.get("study")

    if study is None:
        n_trials = 0
        elapsed_seconds = 0.0
    else:
        n_trials = len(study.trials)
        elapsed_seconds = float(study.user_attrs.get("elapsed_seconds", 0.0))

    return {
        "regime": result["regime"],
        "params": result["params"],
        "feasible": result["feasible"],
        "fallback_to_single": result["fallback_to_single"],
        "minimum_trades": result["minimum_trades"],
        "n_regime_days": result["n_regime_days"],
        "n_window_days": result["n_window_days"],
        "is_ann_return": result["is_ann_return"],
        "is_calmar": result["is_calmar"],
        "n_trades": result["n_trades"],
        "n_trials": n_trials,
        "elapsed_seconds": elapsed_seconds,
    }


def _run_walk_forward_fold(
    window: dict, prices: dict, regimes: pd.Series, config: dict, per_regime: bool, seed: int
) -> dict:
    """Optimiza Train y evalúa el mes OOS de un fold.

    Sin configuraciones factibles no se detiene el walk-forward (SPEC punto 7): un régimen sin θ
    factible usa el θ único de la ventana, y si tampoco hay θ único factible queda en efectivo
    (parámetros None) durante el mes de prueba. Esos regímenes se listan en `cash_regimes`.
    """
    started = time.perf_counter()
    train_start = window["train_start"]
    train_end = window["train_end"]
    test_start = window["test_start"]
    test_end = window["test_end"]
    regime_names = list(config["regime_multiplier"])
    single_result = optimize_regime(
        prices, regimes, regime=None, window=(train_start, train_end), config=config, seed=seed
    )

    single_params = single_result["params"]
    optimization = {"single": _optimization_summary(single_result)}

    if per_regime:
        params_by_regime = {}
        for regime_name in regime_names:
            regime_result = optimize_regime(
                prices,
                regimes,
                regime=regime_name,
                window=(train_start, train_end),
                config=config,
                seed=seed,
            )
            optimization[regime_name] = _optimization_summary(regime_result)
            if regime_result["feasible"]:
                params_by_regime[regime_name] = dict(regime_result["params"])
            else:  # menos de min_regime_days o sin pruebas factibles: θ único (o efectivo)
                params_by_regime[regime_name] = (
                    None if single_params is None else dict(single_params)
                )
    else:
        params_by_regime = {
            regime_name: None if single_params is None else dict(single_params)
            for regime_name in regime_names
        }
    cash_regimes = [name for name, params in params_by_regime.items() if params is None]
    train_prices = {ticker: data.loc[:train_end].copy() for ticker, data in prices.items()}
    first_ticker = next(iter(train_prices))
    train_index = train_prices[first_ticker].index
    train_dates = train_index[(train_index >= train_start) & (train_index <= train_end)]
    train_entry_mask = pd.Series(False, index=train_index, dtype=bool)
    train_entry_mask.loc[train_dates] = True
    embargo_days = int(config["embargo_days"])

    if embargo_days > 0:
        train_entry_mask.loc[train_dates[-embargo_days:]] = False
    _, is_metrics = _evaluate_params(
        params_by_regime,
        train_prices,
        regimes,
        config,
        entry_mask=train_entry_mask,
        period=(train_start, train_end),
    )
    test_prices = {ticker: data.loc[:test_end].copy() for ticker, data in prices.items()}
    test_index = test_prices[first_ticker].index
    test_dates = test_index[(test_index >= test_start) & (test_index <= test_end)]

    if len(test_dates) == 0:
        raise ValueError("El fold no contiene observaciones OOS.")
    test_entry_mask = pd.Series(False, index=test_index, dtype=bool)
    test_entry_mask.loc[test_dates] = True
    oos_result, oos_metrics = _evaluate_params(
        params_by_regime,
        test_prices,
        regimes,
        config,
        entry_mask=test_entry_mask,
        period=(test_start, test_end),
    )
    oos_equity = oos_result.equity.loc[test_start:test_end].copy()

    if len(oos_result.trades) == 0:
        oos_trades = oos_result.trades.copy()
    else:
        entry_dates = pd.to_datetime(oos_result.trades["entry_date"])
        trade_mask = (entry_dates >= test_start) & (entry_dates <= test_end)
        oos_trades = oos_result.trades.loc[trade_mask].copy()
    trade_params = (
        _trade_params_panel(regimes, params_by_regime, test_index).loc[test_start:test_end].copy()
    )
    n_trials_total = sum(item["n_trials"] for item in optimization.values())
    elapsed_seconds = time.perf_counter() - started

    return {
        "fold": window["fold"],
        "seed": seed,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "params_by_regime": params_by_regime,
        "cash_regimes": cash_regimes,
        "optimization": optimization,
        "is_metrics": is_metrics,
        "oos_metrics": oos_metrics,
        "oos_equity": oos_equity,
        "oos_trades": oos_trades,
        "trade_params": trade_params,
        "n_trials_total": n_trials_total,
        "elapsed_seconds": float(elapsed_seconds),
    }


def walk_forward(
    prices: dict, regimes: pd.Series, config: dict, mode: str = "rolling", per_regime: bool = True
) -> dict:
    """Walk-forward de 6 meses de entrenamiento, 1 de prueba y paso mensual (SPEC punto 1).

    Cada ventana optimiza solo con sus datos de entrenamiento y evalúa el mes siguiente fuera de
    muestra; las ventanas corren en paralelo (procesos) con semilla `SEED + i`. La equity OOS se
    encadena: cada mes arranca con el equity final del anterior.

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        OHLCV completo.
    regimes : pd.Series
        Etiqueta filtrada y causal.
    config : dict
        `CONFIG`.
    mode : str
        "rolling" (ventana fija de `wf_train_months`) o "anchored" (inicio fijo, ventana creciente).
    per_regime : bool
        True: un θ por régimen en cada ventana; False: θ único.

    Returns
    -------
    dict
        folds (detalle por ventana), params_by_fold, trade_params (panel OOS concatenado, con la
        columna "regime"), oos_equity, oos_trades, n_folds, n_trials_total, elapsed_seconds y
        periods_per_year.
    """
    if mode not in {"rolling", "anchored"}:
        raise ValueError("mode debe ser 'rolling' o 'anchored'.")

    if not isinstance(per_regime, bool):
        raise TypeError("per_regime debe ser booleano.")
    windows = _walk_forward_windows(prices, config, mode)
    started = time.perf_counter()
    jobs = [
        delayed(_run_walk_forward_fold)(
            window, prices, regimes, config, per_regime, int(config["seed"]) + i
        )
        for i, window in enumerate(windows)
    ]
    n_jobs = int(config.get("n_jobs", 1))

    if n_jobs == 0:
        raise ValueError("n_jobs no puede ser cero.")
    # Procesos y no threads: el loop del motor es Python puro y con threads el GIL lo serializa.
    folds = Parallel(n_jobs=n_jobs)(jobs)
    folds = sorted(folds, key=lambda item: item["fold"])
    equity_segments = []
    current_equity = float(config["initial_capital"])

    for fold in folds:
        segment = fold["oos_equity"].dropna().astype(float)
        if segment.empty:
            continue
        chained = segment / float(config["initial_capital"]) * current_equity
        current_equity = float(chained.iloc[-1])
        equity_segments.append(chained)

    if not equity_segments:
        raise ValueError("No se obtuvo equity OOS.")
    oos_equity = pd.concat(equity_segments).sort_index()

    if not (oos_equity.index.is_unique):
        raise ValueError("Las ventanas OOS se traslapan.")
    trade_param_frames = [fold["trade_params"] for fold in folds if len(fold["trade_params"]) > 0]

    if trade_param_frames:
        trade_params = pd.concat(trade_param_frames).sort_index()
    else:
        trade_params = pd.DataFrame()
    trade_frames = [fold["oos_trades"] for fold in folds if len(fold["oos_trades"]) > 0]

    if trade_frames:
        oos_trades = pd.concat(trade_frames, ignore_index=True)
    else:
        oos_trades = pd.DataFrame()
    params_by_fold = {fold["fold"]: fold["params_by_regime"] for fold in folds}
    elapsed_seconds = time.perf_counter() - started

    return {
        "mode": mode,
        "per_regime": per_regime,
        "folds": folds,
        "params_by_fold": params_by_fold,
        "trade_params": trade_params,
        "oos_equity": oos_equity,
        "oos_trades": oos_trades,
        "n_folds": len(folds),
        "n_trials_total": sum(fold["n_trials_total"] for fold in folds),
        "elapsed_seconds": float(elapsed_seconds),
        "periods_per_year": int(config["periods_per_year"]),
    }


def walk_forward_signals(prices: dict, regimes: pd.Series, wf_result: dict, config: dict) -> dict:
    """Señales del backtest final: cada mes fuera de muestra con el θ de su ventana.

    Para cada fold, `generate_signals` corre con sus θ sobre los precios hasta el fin de su mes de
    prueba (causal) y se toman solo las filas de ese mes. Fuera de los meses OOS el estado y la
    fuerza son 0 y el ATR es NaN, así que no se abre posición. Un régimen en efectivo (θ None) no
    se opera porque `trade_params` del walk-forward trae NaN en esas fechas.

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        OHLCV completo.
    regimes : pd.Series
        Etiqueta filtrada y causal.
    wf_result : dict
        Salida de `walk_forward`.
    config : dict
        `CONFIG`.

    Returns
    -------
    dict[str, pd.DataFrame]
        "state", "strength" y "atr", como `generate_signals`, con el índice de `prices`.
    """
    tickers = list(prices)
    dates = prices[tickers[0]].index
    signals = {
        "state": pd.DataFrame(0, index=dates, columns=tickers, dtype=int),
        "strength": pd.DataFrame(0.0, index=dates, columns=tickers),
        "atr": pd.DataFrame(np.nan, index=dates, columns=tickers),
    }
    for fold in wf_result["folds"]:
        start, end = fold["test_start"], fold["test_end"]
        history = {ticker: data.loc[:end] for ticker, data in prices.items()}
        fold_signals = generate_signals(
            history, _signal_params(fold["params_by_regime"], config), regimes, config
        )
        for name, panel in fold_signals.items():
            month = panel.loc[start:end]
            signals[name].loc[month.index] = month
    return signals


def wf_efficiency(wf_result: dict, metric: str = "ann_return") -> float:
    """Calcula la eficiencia walk-forward.

    La eficiencia se define como:

        desempeño OOS concatenado
        -------------------------
        promedio desempeño IS

    Por defecto se usa retorno anualizado, como indica P2.
    También puede calcularse con Calmar mediante
    ``metric="calmar"``.

    Parameters
    ----------
    wf_result : dict
        Resultado producido por ``walk_forward``.
    metric : str
        ``"ann_return"`` o ``"calmar"``.

    Returns
    -------
    float
        Walk-forward efficiency. Devuelve NaN cuando el
        cociente no está definido.
    """
    if metric not in {"ann_return", "calmar"}:
        raise ValueError("metric debe ser 'ann_return' o 'calmar'.")
    folds = wf_result.get("folds")

    if not folds:
        raise ValueError("wf_result no contiene folds.")
    oos_equity = wf_result.get("oos_equity")

    if not isinstance(oos_equity, pd.Series) or len(oos_equity) < 2:
        raise ValueError("wf_result necesita una equity OOS con al menos dos observaciones.")

    if "periods_per_year" not in wf_result:
        raise ValueError("wf_result necesita periods_per_year.")
    periods_per_year = int(wf_result["periods_per_year"])

    if periods_per_year <= 0:
        raise ValueError("periods_per_year debe ser positivo.")
    oos_trades = wf_result.get("oos_trades")

    if oos_trades is None:
        oos_trades = pd.DataFrame()
    oos_metrics = compute_metrics(oos_equity, oos_trades, rf=0.0, periods_per_year=periods_per_year)
    oos_value = float(oos_metrics[metric])
    is_values = []

    for fold in folds:
        is_metrics = fold.get("is_metrics", {})
        if metric not in is_metrics:
            return np.nan
        value = float(is_metrics[metric])
        # Una ventana completa en efectivo no tiene Calmar (MDD = 0); no aporta al promedio IS.
        if np.isfinite(value):
            is_values.append(value)

    if not is_values:
        return np.nan
    average_is = float(np.mean(is_values))

    if not np.isfinite(oos_value) or not np.isfinite(average_is) or np.isclose(average_is, 0.0):
        return np.nan

    return float(oos_value / average_is)


def sensitivity(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    pct: float = 0.20,
    period: tuple | None = None,
) -> pd.DataFrame:
    """Sensibilidad del Calmar a variar cada parámetro óptimo ±pct (P2, tarea 7).

    Cada parámetro se modifica solo, con los demás fijos; los enteros se redondean al entero más
    cercano. Si los tres regímenes comparten θ (θ único), se varía en los tres a la vez y el
    renglón se etiqueta "unico"; si no, se varía régimen por régimen.

    Parameters
    ----------
    params_by_regime : dict
        θ congelado por régimen.
    prices, regimes, config
        Datos, etiqueta causal y `CONFIG`.
    pct : float
        Variación relativa (SPEC: 0.20).
    period : tuple or None
        (inicio, fin) de la evaluación; ver `_evaluate_on_period`.

    Returns
    -------
    pd.DataFrame
        regime, parameter, factor, base_value, tested_value, baseline_calmar, calmar y
        delta_calmar.
    """
    if not 0 < pct < 1:
        raise ValueError("pct debe estar en el intervalo (0, 1).")
    _, baseline_metrics = _evaluate_on_period(params_by_regime, prices, regimes, config, period)
    baseline_calmar = baseline_metrics["calmar"]
    distinct = {tuple(sorted(params.items())) for params in params_by_regime.values()}
    if len(distinct) == 1:
        groups = {"unico": list(params_by_regime)}
    else:
        groups = {name: [name] for name in sorted(params_by_regime)}
    rows = []

    for group_name, members in groups.items():
        base_params = params_by_regime[members[0]]
        for parameter in search_space():
            base_value = base_params[parameter]
            for factor in (1.0 - pct, 1.0 + pct):
                candidate = {name: dict(values) for name, values in params_by_regime.items()}
                varied_value = base_value * factor
                if _PARAMETER_KINDS[parameter] == "int":
                    varied_value = int(round(varied_value))
                else:
                    varied_value = float(varied_value)
                for member in members:
                    candidate[member][parameter] = varied_value
                _, candidate_metrics = _evaluate_on_period(
                    candidate, prices, regimes, config, period
                )
                candidate_calmar = candidate_metrics["calmar"]
                if np.isfinite(baseline_calmar) and np.isfinite(candidate_calmar):
                    delta_calmar = candidate_calmar - baseline_calmar
                else:
                    delta_calmar = np.nan
                rows.append(
                    {
                        "regime": group_name,
                        "parameter": parameter,
                        "factor": factor,
                        "base_value": base_value,
                        "tested_value": varied_value,
                        "baseline_calmar": baseline_calmar,
                        "calmar": candidate_calmar,
                        "delta_calmar": delta_calmar,
                    }
                )

    return pd.DataFrame(rows)


def _breakeven_cost_bps(cost_curve: pd.DataFrame) -> float:
    """Estima el costo de equilibrio donde el retorno neto llega a cero.

    Si el cruce ocurre entre dos niveles evaluados, se usa
    interpolación lineal entre ambos puntos.
    """
    net_return = cost_curve["net_return"].dropna().sort_index()

    if net_return.empty:
        return np.nan
    values = net_return.to_numpy(dtype=float)
    costs = net_return.index.to_numpy(dtype=float)
    zero_mask = np.isclose(values, 0.0, atol=1e-12)

    if zero_mask.any():
        first_zero = np.flatnonzero(zero_mask)[0]
        return float(costs[first_zero])

    if values[0] < 0:
        return 0.0

    for position in range(1, len(values)):
        previous_return = values[position - 1]
        current_return = values[position]
        if previous_return > 0 and current_return < 0:
            previous_cost = costs[position - 1]
            current_cost = costs[position]
            crossing = previous_cost + (
                -previous_return
                * (current_cost - previous_cost)
                / (current_return - previous_return)
            )
            return float(crossing)

    return np.nan


def cost_sweep(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    round_trip_bps: list[float],
    period: tuple | None = None,
) -> pd.DataFrame:
    """Evalúa la estrategia frente a distintos costos de transacción (P2, tareas 8 y 11).

    El barrido usa costo total de ida y vuelta en basis points.
    Comisión y slippage se escalan manteniendo la proporción del
    caso base. Así, 29 bps reproduce exactamente la estructura de
    costos definida en CONFIG.

    Parameters
    ----------
    params_by_regime : dict
        Parámetros congelados por régimen.
    prices : dict
        Datos OHLCV por activo.
    regimes : pd.Series
        Régimen causal por fecha.
    config : dict
        Configuración del proyecto.
    round_trip_bps : list[float]
        Costos totales de ida y vuelta a evaluar.
    period : tuple or None
        (inicio, fin) de la evaluación; ver `_evaluate_on_period`.

    Returns
    -------
    pd.DataFrame
        Retorno neto anualizado, Calmar y número de trades por
        nivel de costo, junto con el break-even y el margen frente
        al costo base.
    """
    if not round_trip_bps:
        raise ValueError("round_trip_bps no puede estar vacío.")
    costs = sorted({float(value) for value in round_trip_bps})

    if any((not np.isfinite(value) or value < 0) for value in costs):
        raise ValueError("Los costos deben ser finitos y no negativos.")
    base_commission = float(config["commission"])
    base_slippage = float(config["slippage"])
    base_per_side = base_commission + base_slippage
    base_round_trip_bps = 2.0 * base_per_side * 10_000.0
    rows = []

    for cost_bps in costs:
        candidate_config = dict(config)
        target_per_side = cost_bps / 20_000.0
        if base_per_side > 0:
            scale = target_per_side / base_per_side
            candidate_config["commission"] = base_commission * scale
            candidate_config["slippage"] = base_slippage * scale
        else:
            candidate_config["commission"] = 0.0
            candidate_config["slippage"] = target_per_side
        _, metrics = _evaluate_on_period(
            params_by_regime, prices, regimes, candidate_config, period
        )
        rows.append(
            {
                "round_trip_bps": cost_bps,
                "net_return": metrics["ann_return"],
                "calmar": metrics["calmar"],
                "n_trades": int(metrics["n_trades"]),
            }
        )
    cost_curve = pd.DataFrame(rows).set_index("round_trip_bps").sort_index()
    cost_curve.index.name = "round_trip_bps"
    break_even_bps = _breakeven_cost_bps(cost_curve)

    if np.isfinite(break_even_bps):
        margin_bps = break_even_bps - base_round_trip_bps
    else:
        margin_bps = np.nan
    cost_curve["break_even_bps"] = break_even_bps
    cost_curve["base_cost_bps"] = base_round_trip_bps
    cost_curve["margin_bps"] = margin_bps

    return cost_curve


def single_indicator_comparison(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    period: tuple | None = None,
) -> pd.DataFrame:
    """Compara cada indicador solo contra la regla 2 de 3 (pregunta 1 del lab).

    Con un indicador solo, su voto es el estado y la fuerza (SPEC punto 3 sin compuerta).

    Parameters
    ----------
    params_by_regime, prices, regimes, config
        θ congelado, datos, etiqueta causal y `CONFIG`.
    period : tuple or None
        (inicio, fin) de la evaluación; ver `_evaluate_on_period`.

    Returns
    -------
    pd.DataFrame
        Índice `strategy` ∈ {2_de_3, sma, macd, rsi}; columnas n_trades y calmar.
    """
    rows = []
    for strategy, indicator in (("2_de_3", None), ("sma", "sma"), ("macd", "macd"), ("rsi", "rsi")):
        _, metrics = _evaluate_on_period(
            params_by_regime, prices, regimes, config, period, indicator=indicator
        )
        rows.append(
            {
                "strategy": strategy,
                "n_trades": int(metrics["n_trades"]),
                "calmar": metrics["calmar"],
            }
        )

    return pd.DataFrame(rows).set_index("strategy")
