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
from src.signals import (
    compute_indicators,
    generate_signals,
    indicator_votes,
)

_PARAMETER_KINDS = {
    "sma_fast": "int",
    "sma_slow": "int",
    "rsi_window": "int",
    "rsi_lo": "int",
    "rsi_hi": "int",
    "k_stop": "float",
    "reward_ratio": "float",
    "max_holding": "int",
    "risk_per_trade": "float",
}

_TRADE_PARAM_KEYS = (
    "k_stop",
    "reward_ratio",
    "max_holding",
    "risk_per_trade",
)

def _trade_params_panel(
    regimes: pd.Series,
    params_by_regime: dict,
    index: pd.Index,
) -> pd.DataFrame:
    """Construye el panel fecha × parámetros que recibe run_backtest.

    Cada fecha usa los parámetros correspondientes al régimen vigente.
    Las fechas sin régimen permanecen como NaN y no habilitan entradas.
    """
    labels = regimes.reindex(index)

    panel = pd.DataFrame(
        np.nan,
        index=index,
        columns=_TRADE_PARAM_KEYS,
        dtype=float,
    )

    for regime_name, params in params_by_regime.items():
        mask = labels == regime_name

        for key in _TRADE_PARAM_KEYS:
            panel.loc[
                mask,
                key,
            ] = params[key]

    panel["regime"] = labels

    return panel

def _evaluate_params(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    entry_mask: pd.Series | None = None,
    period: tuple | None = None,
) -> tuple:
    """Ejecuta señales, portafolio, backtest y métricas para un candidato.

    Sigue el flujo definido en P2:
    generate_signals -> sleeve_weights -> run_backtest -> métricas.
    """
    tickers = list(prices)

    if not tickers:
        raise ValueError(
            "prices no puede estar vacío."
        )

    index = prices[
        tickers[0]
    ].index

    labels = regimes.reindex(index)

    signals = generate_signals(
        prices,
        params_by_regime,
        labels,
        config,
    )

    sleeve = sleeve_weights(
        prices,
        signals,
        labels,
        config,
        method="risk_parity",
    )

    trade_params = _trade_params_panel(
        labels,
        params_by_regime,
        index,
    )

    result = run_backtest(
        prices,
        signals,
        sleeve,
        trade_params,
        config,
        entry_mask=entry_mask,
    )

    if period is None:
        equity_for_metrics = (
            result.equity
        )

        trades_for_metrics = (
            result.trades
        )

    else:
        start = pd.Timestamp(
            period[0]
        )

        end = pd.Timestamp(
            period[1]
        )

        equity_for_metrics = (
            result.equity.loc[
                start:end
            ]
        )

        if len(
            equity_for_metrics
        ) < 2:
            raise ValueError(
                "La ventana debe contener "
                "al menos dos observaciones."
            )

        if len(result.trades) == 0:
            trades_for_metrics = (
                result.trades.copy()
            )

        else:
            entry_dates = (
                pd.to_datetime(
                    result.trades[
                        "entry_date"
                    ]
                )
            )

            trade_mask = (
                (entry_dates >= start)
                & (entry_dates <= end)
            )

            trades_for_metrics = (
                result.trades.loc[
                    trade_mask
                ].copy()
            )

    metrics = compute_metrics(
        equity_for_metrics,
        trades_for_metrics,
        rf=0.0,
        periods_per_year=config[
            "periods_per_year"
        ],
    )

    return result, metrics

def _single_indicator_signals(
    prices: dict,
    params_by_regime: dict,
    regimes: pd.Series,
    config: dict,
    indicator: str,
) -> dict[str, pd.DataFrame]:
    """Genera señales usando únicamente SMA, MACD o RSI.

    Se utiliza para comparar cada indicador por separado contra
    la regla principal de confirmación 2 de 3 de P2.
    """
    vote_columns = {
        "sma": "v_sma",
        "macd": "v_macd",
        "rsi": "v_rsi",
    }

    if indicator not in vote_columns:
        raise ValueError(
            "indicator debe ser 'sma', 'macd' o 'rsi'."
        )

    tickers = list(prices)

    if not tickers:
        raise ValueError(
            "prices no puede estar vacío."
        )

    index = prices[
        tickers[0]
    ].index

    labels = regimes.reindex(index)

    state = pd.DataFrame(
        0,
        index=index,
        columns=tickers,
        dtype=int,
    )

    strength = pd.DataFrame(
        0.0,
        index=index,
        columns=tickers,
    )

    atr = pd.DataFrame(
        np.nan,
        index=index,
        columns=tickers,
    )

    used_regimes = set(
        labels.dropna().unique()
    )

    missing = (
        used_regimes
        - set(params_by_regime)
    )

    if missing:
        raise ValueError(
            "Faltan parámetros para "
            f"los regímenes: {sorted(missing)}"
        )

    vote_column = vote_columns[
        indicator
    ]

    for ticker in tickers:
        price_data = prices[
            ticker
        ]

        for regime_name in used_regimes:
            params = params_by_regime[
                regime_name
            ]

            indicators = (
                compute_indicators(
                    price_data,
                    params,
                    config,
                )
            )

            votes = indicator_votes(
                indicators,
                params,
            )

            vote = votes[
                vote_column
            ].astype(int)

            mask = (
                labels
                == regime_name
            )

            state.loc[
                mask,
                ticker,
            ] = vote.loc[
                mask
            ]

            strength.loc[
                mask,
                ticker,
            ] = vote.loc[
                mask
            ].astype(float)

            atr.loc[
                mask,
                ticker,
            ] = indicators.loc[
                mask,
                "atr",
            ]

    return {
        "state": state,
        "strength": strength,
        "atr": atr,
    }

def _evaluate_single_indicator(
    indicator: str,
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
) -> tuple:
    """Ejecuta el backtest usando únicamente un indicador."""
    tickers = list(prices)

    if not tickers:
        raise ValueError(
            "prices no puede estar vacío."
        )

    index = prices[
        tickers[0]
    ].index

    labels = regimes.reindex(
        index
    )

    signals = (
        _single_indicator_signals(
            prices,
            params_by_regime,
            labels,
            config,
            indicator,
        )
    )

    sleeve = sleeve_weights(
        prices,
        signals,
        labels,
        config,
        method="risk_parity",
    )

    trade_params = (
        _trade_params_panel(
            labels,
            params_by_regime,
            index,
        )
    )

    result = run_backtest(
        prices,
        signals,
        sleeve,
        trade_params,
        config,
    )

    metrics = compute_metrics(
        result.equity,
        result.trades,
        rf=0.0,
        periods_per_year=config[
            "periods_per_year"
        ],
    )

    return result, metrics

def search_space() -> dict:
    """Describe las nueve dimensiones del espacio de búsqueda de P2.

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

def _suggest_params(
    trial: optuna.Trial,
    config: dict,
) -> dict:
    """Propone los nueve parámetros de P2 usando los rangos de CONFIG."""
    params = {}

    for name, kind in search_space().items():
        low, high = config[
            "search_ranges"
        ][name]

        if kind == "int":
            params[name] = (
                trial.suggest_int(
                    name,
                    int(low),
                    int(high),
                )
            )
        else:
            params[name] = (
                trial.suggest_float(
                    name,
                    float(low),
                    float(high),
                )
            )

    return params

def diagnostic_study(
    prices: dict,
    config: dict,
    sampler: str,
    n_trials: int,
    seed: int,
) -> optuna.Study:
    """Corre el diagnóstico de Optuna sobre Train con un theta único.

    Se puede usar RandomSampler o TPESampler. Cada prueba ejecuta
    señales -> Risk Parity -> backtest con costos -> Calmar.

    Las configuraciones que no alcanzan la actividad mínima o cuyo
    Calmar no es finito reciben -inf.
    """
    sampler_name = sampler.lower()

    if sampler_name not in {
        "random",
        "tpe",
    }:
        raise ValueError(
            "sampler debe ser 'random' o 'tpe'."
        )

    if n_trials <= 0:
        raise ValueError(
            "n_trials debe ser positivo."
        )

    if not prices:
        raise ValueError(
            "prices no puede estar vacío."
        )

    if sampler_name == "random":
        optuna_sampler = (
            optuna.samplers.RandomSampler(
                seed=seed
            )
        )
    else:
        optuna_sampler = (
            optuna.samplers.TPESampler(
                seed=seed
            )
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna_sampler,
    )

    train_start = pd.Timestamp(
        config["blocks"]["train"][0]
    )
    train_end = pd.Timestamp(
        config["blocks"]["train"][1]
    )

    window_prices = {
        ticker: data.loc[
            :train_end
        ].copy()
        for ticker, data
        in prices.items()
    }

    first_ticker = next(
        iter(window_prices)
    )

    index = window_prices[
        first_ticker
    ].index

    train_dates = index[
        (index >= train_start)
        & (index <= train_end)
    ]

    if len(train_dates) == 0:
        raise ValueError(
            "No hay observaciones dentro de Train."
        )

    embargo_days = int(
        config["embargo_days"]
    )

    if embargo_days < 0:
        raise ValueError(
            "embargo_days no puede ser negativo."
        )

    if embargo_days >= len(
        train_dates
    ):
        raise ValueError(
            "El embargo consume toda la ventana de Train."
        )

    entry_mask = pd.Series(
        False,
        index=index,
        dtype=bool,
    )

    entry_mask.loc[
        train_dates
    ] = True

    if embargo_days > 0:
        embargo_dates = train_dates[
            -embargo_days:
        ]

        entry_mask.loc[
            embargo_dates
        ] = False

    # El diagnóstico usa theta único sin diferenciar regímenes.
    # Se usa el régimen con mayor multiplicador, equivalente al
    # caso neutral de la corrida base.
    single_regime = max(
        config["regime_multiplier"],
        key=config[
            "regime_multiplier"
        ].get,
    )

    regime_labels = pd.Series(
        single_regime,
        index=index,
        name="regime",
    )

    minimum_trades = int(
        config[
            "min_trades_per_window"
        ]
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        params = _suggest_params(
            trial,
            config,
        )

        params_by_regime = {
            single_regime: params
        }

        result, metrics = (
            _evaluate_params(
                params_by_regime,
                window_prices,
                regime_labels,
                config,
                entry_mask=entry_mask,
                period=(
                    train_start,
                    train_end,
                ),
            )
        )

        n_trades = int(
            metrics["n_trades"]
        )

        trial.set_user_attr(
            "n_trades",
            n_trades,
        )

        trial.set_user_attr(
            "minimum_trades",
            minimum_trades,
        )

        if n_trades < minimum_trades:
            trial.set_user_attr(
                "feasible",
                False,
            )

            return -np.inf

        calmar = float(
            metrics["calmar"]
        )

        if not np.isfinite(
            calmar
        ):
            trial.set_user_attr(
                "feasible",
                False,
            )

            return -np.inf

        trial.set_user_attr(
            "feasible",
            True,
        )

        trial.set_user_attr(
            "calmar",
            calmar,
        )

        return calmar

    start_time = (
        time.perf_counter()
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    feasible_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(
                trial.value
            )
        )
    ]

    study.set_user_attr(
        "sampler",
        sampler_name,
    )

    study.set_user_attr(
        "seed",
        int(seed),
    )

    study.set_user_attr(
        "n_trials_requested",
        int(n_trials),
    )

    study.set_user_attr(
        "n_trials_evaluated",
        len(study.trials),
    )

    study.set_user_attr(
        "n_trials_feasible",
        len(feasible_trials),
    )

    study.set_user_attr(
        "elapsed_seconds",
        float(elapsed_seconds),
    )

    study.set_user_attr(
        "train_start",
        str(train_start.date()),
    )

    study.set_user_attr(
        "train_end",
        str(train_end.date()),
    )

    study.set_user_attr(
        "embargo_days",
        embargo_days,
    )

    study.set_user_attr(
        "minimum_trades",
        minimum_trades,
    )

    return study

def select_plateau(
    study: optuna.Study,
    top_frac: float = 0.10,
) -> dict:
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
        raise ValueError(
            "top_frac debe estar en el intervalo (0, 1]."
        )

    complete_trials = study.get_trials(
        deepcopy=False,
        states=[
            optuna.trial.TrialState.COMPLETE
        ],
    )

    feasible = [
        trial
        for trial in complete_trials
        if (
            trial.value is not None
            and np.isfinite(trial.value)
        )
    ]

    if not feasible:
        raise ValueError(
            "El estudio no contiene trials factibles."
        )

    maximize = (
        study.direction
        == optuna.study.StudyDirection.MAXIMIZE
    )

    ordered = sorted(
        feasible,
        key=lambda trial: trial.value,
        reverse=maximize,
    )

    n_top = max(
        1,
        math.ceil(
            len(ordered)
            * top_frac
        ),
    )

    top_trials = ordered[:n_top]

    if len(top_trials) == 1:
        return dict(
            top_trials[0].params
        )

    parameter_names = sorted(
        top_trials[0].params
    )

    for trial in top_trials:
        if set(trial.params) != set(
            parameter_names
        ):
            raise ValueError(
                "Los trials de la meseta deben usar "
                "el mismo espacio de parámetros."
            )

    normalized_rows = []

    for trial in top_trials:
        row = []

        for name in parameter_names:
            distribution = (
                trial.distributions[name]
            )

            if not hasattr(
                distribution,
                "low",
            ) or not hasattr(
                distribution,
                "high",
            ):
                raise TypeError(
                    "select_plateau solo admite "
                    "parámetros numéricos."
                )

            low = float(
                distribution.low
            )

            high = float(
                distribution.high
            )

            value = float(
                trial.params[name]
            )

            if high == low:
                scaled = 0.0
            else:
                scaled = (
                    value - low
                ) / (
                    high - low
                )

            row.append(
                scaled
            )

        normalized_rows.append(
            row
        )

    matrix = np.asarray(
        normalized_rows,
        dtype=float,
    )

    differences = (
        matrix[:, None, :]
        - matrix[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            differences**2,
            axis=2,
        )
    )

    total_distance = (
        distances.sum(axis=1)
    )

    medoid_position = int(
        np.argmin(
            total_distance
        )
    )

    return dict(
        top_trials[
            medoid_position
        ].params
    )

def _candidate_params_by_regime(
    candidate: dict,
    regime: str | None,
    config: dict,
) -> dict:
    """Construye los parámetros usados por un trial.

    Con regime=None se usa el mismo theta en todos los regímenes.
    Al optimizar un régimen específico, ese régimen usa el
    candidato y los demás conservan los valores base.
    """
    regime_names = list(
        config[
            "regime_multiplier"
        ]
    )

    if regime is None:
        return {
            name: dict(candidate)
            for name in regime_names
        }

    params = {
        name: dict(
            config["base_params"]
        )
        for name in regime_names
    }

    params[regime] = dict(
        candidate
    )

    return params

def optimize_regime(
    prices: dict,
    regimes: pd.Series,
    regime: str | None,
    window: tuple,
    config: dict,
    seed: int,
) -> dict:
    """Optimiza theta dentro de una ventana de entrenamiento.

    Con ``regime=None`` se obtiene un theta único compartido por
    todos los regímenes. Para un régimen concreto se aplica la
    actividad mínima prorrateada definida en el SPEC.

    Si el régimen tiene menos de ``min_regime_days`` observaciones,
    no se optimiza y se indica que debe usarse el theta único de
    la ventana.
    """
    if not prices:
        raise ValueError(
            "prices no puede estar vacío."
        )

    if len(window) != 2:
        raise ValueError(
            "window debe ser (inicio, fin)."
        )

    start = pd.Timestamp(
        window[0]
    )

    end = pd.Timestamp(
        window[1]
    )

    if start > end:
        raise ValueError(
            "El inicio de la ventana "
            "no puede ser posterior al fin."
        )

    regime_names = list(
        config[
            "regime_multiplier"
        ]
    )

    if (
        regime is not None
        and regime not in regime_names
    ):
        raise ValueError(
            f"Régimen desconocido: {regime!r}"
        )

    window_prices = {
        ticker: data.loc[
            :end
        ].copy()
        for ticker, data
        in prices.items()
    }

    first_ticker = next(
        iter(window_prices)
    )

    index = window_prices[
        first_ticker
    ].index

    train_dates = index[
        (index >= start)
        & (index <= end)
    ]

    if len(train_dates) == 0:
        raise ValueError(
            "La ventana no contiene observaciones."
        )

    labels = regimes.reindex(
        index
    )

    unknown_regimes = (
        set(
            labels.dropna().unique()
        )
        - set(regime_names)
    )

    if unknown_regimes:
        raise ValueError(
            "Hay regímenes sin configuración: "
            f"{sorted(unknown_regimes)}"
        )

    n_window_days = len(
        train_dates
    )

    if regime is None:
        n_regime_days = (
            n_window_days
        )

        minimum_trades = int(
            config[
                "min_trades_per_window"
            ]
        )

    else:
        n_regime_days = int(
            (
                labels.loc[
                    train_dates
                ]
                == regime
            ).sum()
        )

        minimum_trades = int(
            math.ceil(
                config[
                    "min_trades_per_window"
                ]
                * n_regime_days
                / n_window_days
            )
        )

        if (
            n_regime_days
            < config[
                "min_regime_days"
            ]
        ):
            return {
                "regime": regime,
                "params": None,
                "study": None,
                "feasible": False,
                "fallback_to_single": True,
                "n_window_days":
                    n_window_days,
                "n_regime_days":
                    n_regime_days,
                "minimum_trades":
                    minimum_trades,
                "is_ann_return":
                    np.nan,
                "is_calmar":
                    np.nan,
                "n_trades":
                    0,
                "window":
                    (start, end),
            }

    embargo_days = int(
        config[
            "embargo_days"
        ]
    )

    if embargo_days < 0:
        raise ValueError(
            "embargo_days no puede ser negativo."
        )

    if embargo_days >= len(
        train_dates
    ):
        raise ValueError(
            "El embargo consume toda la ventana."
        )

    entry_mask = pd.Series(
        False,
        index=index,
        dtype=bool,
    )

    entry_mask.loc[
        train_dates
    ] = True

    if regime is not None:
        entry_mask &= (
            labels == regime
        )

    if embargo_days > 0:
        entry_mask.loc[
            train_dates[
                -embargo_days:
            ]
        ] = False

    sampler = (
        optuna.samplers.TPESampler(
            seed=seed
        )
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    def objective(
        trial: optuna.Trial,
    ) -> float:
        candidate = _suggest_params(
            trial,
            config,
        )

        params_by_regime = (
            _candidate_params_by_regime(
                candidate,
                regime,
                config,
            )
        )

        _, metrics = (
            _evaluate_params(
                params_by_regime,
                window_prices,
                labels,
                config,
                entry_mask=entry_mask,
                period=(
                    start,
                    end,
                ),
            )
        )

        n_trades = int(
            metrics["n_trades"]
        )

        trial.set_user_attr(
            "n_trades",
            n_trades,
        )

        trial.set_user_attr(
            "minimum_trades",
            minimum_trades,
        )

        if (
            n_trades
            < minimum_trades
        ):
            trial.set_user_attr(
                "feasible",
                False,
            )

            return -np.inf

        calmar = float(
            metrics["calmar"]
        )

        if not np.isfinite(
            calmar
        ):
            # Política provisional de P2 para MDD = 0:
            # el trial se considera infactible.
            trial.set_user_attr(
                "feasible",
                False,
            )

            return -np.inf

        trial.set_user_attr(
            "feasible",
            True,
        )

        return calmar

    start_time = (
        time.perf_counter()
    )

    study.optimize(
        objective,
        n_trials=int(
            config[
                "n_trials_wf"
            ]
        ),
        n_jobs=1,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    feasible_trials = [
        trial
        for trial in study.trials
        if (
            trial.state
            == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(
                trial.value
            )
        )
    ]

    study.set_user_attr(
        "seed",
        int(seed),
    )

    study.set_user_attr(
        "regime",
        (
            "single"
            if regime is None
            else regime
        ),
    )

    study.set_user_attr(
        "minimum_trades",
        minimum_trades,
    )

    study.set_user_attr(
        "n_regime_days",
        n_regime_days,
    )

    study.set_user_attr(
        "n_window_days",
        n_window_days,
    )

    study.set_user_attr(
        "n_trials_requested",
        int(
            config[
                "n_trials_wf"
            ]
        ),
    )

    study.set_user_attr(
        "n_trials_evaluated",
        len(study.trials),
    )

    study.set_user_attr(
        "n_trials_feasible",
        len(feasible_trials),
    )

    study.set_user_attr(
        "elapsed_seconds",
        float(elapsed_seconds),
    )

    if not feasible_trials:
        return {
            "regime": regime,
            "params": None,
            "study": study,
            "feasible": False,
            "fallback_to_single": False,
            "n_window_days":
                n_window_days,
            "n_regime_days":
                n_regime_days,
            "minimum_trades":
                minimum_trades,
            "is_ann_return":
                np.nan,
            "is_calmar":
                np.nan,
            "n_trades":
                0,
            "window":
                (start, end),
        }

    params = select_plateau(
        study,
        config[
            "plateau_top_frac"
        ],
    )

    params_by_regime = (
        _candidate_params_by_regime(
            params,
            regime,
            config,
        )
    )

    _, selected_metrics = (
        _evaluate_params(
            params_by_regime,
            window_prices,
            labels,
            config,
            entry_mask=entry_mask,
            period=(
                start,
                end,
            ),
        )
    )

    return {
        "regime": regime,
        "params": params,
        "study": study,
        "feasible": True,
        "fallback_to_single": False,
        "n_window_days":
            n_window_days,
        "n_regime_days":
            n_regime_days,
        "minimum_trades":
            minimum_trades,
        "is_ann_return":
            float(
                selected_metrics[
                    "ann_return"
                ]
            ),
        "is_calmar":
            float(
                selected_metrics[
                    "calmar"
                ]
            ),
        "n_trades":
            int(
                selected_metrics[
                    "n_trades"
                ]
            ),
        "window":
            (start, end),
    }

def _walk_forward_windows(
    prices: dict,
    config: dict,
    mode: str,
) -> list[dict]:
    """Construye las ventanas mensuales del walk-forward.

    Rolling conserva únicamente los últimos ``wf_train_months``.
    Anchored conserva fijo el inicio de Train y expande el final.
    """
    if mode not in {
        "rolling",
        "anchored",
    }:
        raise ValueError(
            "mode debe ser 'rolling' o 'anchored'."
        )

    if not prices:
        raise ValueError(
            "prices no puede estar vacío."
        )

    train_months = int(
        config["wf_train_months"]
    )

    test_months = int(
        config["wf_test_months"]
    )

    step_months = int(
        config["wf_step_months"]
    )

    if (
        train_months <= 0
        or test_months <= 0
        or step_months <= 0
    ):
        raise ValueError(
            "Los tamaños del walk-forward "
            "deben ser positivos."
        )

    overall_start = pd.Timestamp(
        config["blocks"]["train"][0]
    )

    overall_end = pd.Timestamp(
        config["blocks"]["test"][1]
    )

    first_ticker = next(
        iter(prices)
    )

    dates = prices[
        first_ticker
    ].index

    test_start = (
        overall_start
        + pd.DateOffset(
            months=train_months
        )
    )

    windows = []
    fold_number = 0

    while test_start <= overall_end:
        train_end = (
            test_start
            - pd.Timedelta(days=1)
        )

        if mode == "rolling":
            train_start = (
                test_start
                - pd.DateOffset(
                    months=train_months
                )
            )
        else:
            train_start = (
                overall_start
            )

        test_end = min(
            test_start
            + pd.DateOffset(
                months=test_months
            )
            - pd.Timedelta(days=1),
            overall_end,
        )

        train_dates = dates[
            (dates >= train_start)
            & (dates <= train_end)
        ]

        test_dates = dates[
            (dates >= test_start)
            & (dates <= test_end)
        ]

        if (
            len(train_dates) > 0
            and len(test_dates) > 0
        ):
            windows.append(
                {
                    "fold":
                        fold_number,
                    "train_start":
                        train_start,
                    "train_end":
                        train_end,
                    "test_start":
                        test_start,
                    "test_end":
                        test_end,
                }
            )

            fold_number += 1

        test_start = (
            test_start
            + pd.DateOffset(
                months=step_months
            )
        )

    if not windows:
        raise ValueError(
            "No se pudieron construir "
            "ventanas walk-forward."
        )

    return windows

def _optimization_summary(
    result: dict,
) -> dict:
    """Resume un resultado de optimize_regime sin guardar el Study completo."""
    study = result.get(
        "study"
    )

    if study is None:
        n_trials = 0
        elapsed_seconds = 0.0
    else:
        n_trials = len(
            study.trials
        )

        elapsed_seconds = float(
            study.user_attrs.get(
                "elapsed_seconds",
                0.0,
            )
        )

    return {
        "regime":
            result["regime"],
        "params":
            result["params"],
        "feasible":
            result["feasible"],
        "fallback_to_single":
            result[
                "fallback_to_single"
            ],
        "minimum_trades":
            result[
                "minimum_trades"
            ],
        "n_regime_days":
            result[
                "n_regime_days"
            ],
        "n_window_days":
            result[
                "n_window_days"
            ],
        "is_ann_return":
            result[
                "is_ann_return"
            ],
        "is_calmar":
            result[
                "is_calmar"
            ],
        "n_trades":
            result[
                "n_trades"
            ],
        "n_trials":
            n_trials,
        "elapsed_seconds":
            elapsed_seconds,
    }

def _run_walk_forward_fold(
    window: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    per_regime: bool,
    seed: int,
) -> dict:
    """Optimiza Train y evalúa el mes OOS de un fold."""
    started = time.perf_counter()

    train_start = window[
        "train_start"
    ]

    train_end = window[
        "train_end"
    ]

    test_start = window[
        "test_start"
    ]

    test_end = window[
        "test_end"
    ]

    regime_names = list(
        config[
            "regime_multiplier"
        ]
    )

    single_result = optimize_regime(
        prices,
        regimes,
        regime=None,
        window=(
            train_start,
            train_end,
        ),
        config=config,
        seed=seed,
    )

    if (
        not single_result[
            "feasible"
        ]
        or single_result[
            "params"
        ] is None
    ):
        raise RuntimeError(
            "No se encontró un theta único "
            f"factible en el fold {window['fold']}."
        )

    single_params = dict(
        single_result[
            "params"
        ]
    )

    optimization = {
        "single":
            _optimization_summary(
                single_result
            )
    }

    if per_regime:
        params_by_regime = {}

        for regime_name in regime_names:
            regime_result = (
                optimize_regime(
                    prices,
                    regimes,
                    regime=regime_name,
                    window=(
                        train_start,
                        train_end,
                    ),
                    config=config,
                    seed=seed,
                )
            )

            optimization[
                regime_name
            ] = (
                _optimization_summary(
                    regime_result
                )
            )

            if regime_result[
                "fallback_to_single"
            ]:
                params_by_regime[
                    regime_name
                ] = dict(
                    single_params
                )

            elif regime_result[
                "feasible"
            ]:
                params_by_regime[
                    regime_name
                ] = dict(
                    regime_result[
                        "params"
                    ]
                )

            else:
                raise RuntimeError(
                    "No se encontraron parámetros "
                    "factibles para el régimen "
                    f"{regime_name!r} en el "
                    f"fold {window['fold']}."
                )

    else:
        params_by_regime = {
            regime_name:
                dict(single_params)
            for regime_name
            in regime_names
        }

    train_prices = {
        ticker: data.loc[
            :train_end
        ].copy()
        for ticker, data
        in prices.items()
    }

    first_ticker = next(
        iter(train_prices)
    )

    train_index = (
        train_prices[
            first_ticker
        ].index
    )

    train_dates = train_index[
        (train_index >= train_start)
        & (train_index <= train_end)
    ]

    train_entry_mask = pd.Series(
        False,
        index=train_index,
        dtype=bool,
    )

    train_entry_mask.loc[
        train_dates
    ] = True

    embargo_days = int(
        config[
            "embargo_days"
        ]
    )

    if embargo_days > 0:
        train_entry_mask.loc[
            train_dates[
                -embargo_days:
            ]
        ] = False

    _, is_metrics = (
        _evaluate_params(
            params_by_regime,
            train_prices,
            regimes,
            config,
            entry_mask=
                train_entry_mask,
            period=(
                train_start,
                train_end,
            ),
        )
    )

    test_prices = {
        ticker: data.loc[
            :test_end
        ].copy()
        for ticker, data
        in prices.items()
    }

    test_index = (
        test_prices[
            first_ticker
        ].index
    )

    test_dates = test_index[
        (test_index >= test_start)
        & (test_index <= test_end)
    ]

    if len(test_dates) == 0:
        raise ValueError(
            "El fold no contiene "
            "observaciones OOS."
        )

    test_entry_mask = pd.Series(
        False,
        index=test_index,
        dtype=bool,
    )

    test_entry_mask.loc[
        test_dates
    ] = True

    oos_result, oos_metrics = (
        _evaluate_params(
            params_by_regime,
            test_prices,
            regimes,
            config,
            entry_mask=
                test_entry_mask,
            period=(
                test_start,
                test_end,
            ),
        )
    )

    oos_equity = (
        oos_result.equity.loc[
            test_start:test_end
        ].copy()
    )

    if len(
        oos_result.trades
    ) == 0:
        oos_trades = (
            oos_result.trades.copy()
        )
    else:
        entry_dates = (
            pd.to_datetime(
                oos_result.trades[
                    "entry_date"
                ]
            )
        )

        trade_mask = (
            (entry_dates >= test_start)
            & (entry_dates <= test_end)
        )

        oos_trades = (
            oos_result.trades.loc[
                trade_mask
            ].copy()
        )

    trade_params = (
        _trade_params_panel(
            regimes,
            params_by_regime,
            test_index,
        )
        .loc[
            test_start:test_end
        ]
        .copy()
    )

    n_trials_total = sum(
        item["n_trials"]
        for item
        in optimization.values()
    )

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    return {
        "fold":
            window["fold"],
        "seed":
            seed,
        "train_start":
            train_start,
        "train_end":
            train_end,
        "test_start":
            test_start,
        "test_end":
            test_end,
        "params_by_regime":
            params_by_regime,
        "optimization":
            optimization,
        "is_metrics":
            is_metrics,
        "oos_metrics":
            oos_metrics,
        "oos_equity":
            oos_equity,
        "oos_trades":
            oos_trades,
        "trade_params":
            trade_params,
        "n_trials_total":
            n_trials_total,
        "elapsed_seconds":
            float(
                elapsed_seconds
            ),
    }

def walk_forward(
    prices: dict,
    regimes: pd.Series,
    config: dict,
    mode: str = "rolling",
    per_regime: bool = True,
) -> dict:
    """Ejecuta walk-forward rolling o anchored.

    Cada fold optimiza únicamente con Train y evalúa el mes
    siguiente fuera de muestra. Las ventanas pueden usar parámetros
    específicos por régimen o un theta único compartido.
    """
    if mode not in {
        "rolling",
        "anchored",
    }:
        raise ValueError(
            "mode debe ser 'rolling' o 'anchored'."
        )

    if not isinstance(
        per_regime,
        bool,
    ):
        raise TypeError(
            "per_regime debe ser booleano."
        )

    windows = (
        _walk_forward_windows(
            prices,
            config,
            mode,
        )
    )

    started = time.perf_counter()

    jobs = [
        delayed(
            _run_walk_forward_fold
        )(
            window,
            prices,
            regimes,
            config,
            per_regime,
            int(
                config["seed"]
            )
            + i,
        )
        for i, window
        in enumerate(windows)
    ]

    n_jobs = int(
        config.get(
            "n_jobs",
            1,
        )
    )

    if n_jobs == 0:
        raise ValueError(
            "n_jobs no puede ser cero."
        )

    folds = Parallel(
        n_jobs=n_jobs,
        prefer="threads",
    )(jobs)

    folds = sorted(
        folds,
        key=lambda item:
            item["fold"],
    )

    equity_segments = []

    current_equity = float(
        config[
            "initial_capital"
        ]
    )

    for fold in folds:
        segment = (
            fold[
                "oos_equity"
            ]
            .dropna()
            .astype(float)
        )

        if segment.empty:
            continue

        chained = (
            segment
            / float(
                config[
                    "initial_capital"
                ]
            )
            * current_equity
        )

        current_equity = float(
            chained.iloc[-1]
        )

        equity_segments.append(
            chained
        )

    if not equity_segments:
        raise ValueError(
            "No se obtuvo equity OOS."
        )

    oos_equity = pd.concat(
        equity_segments
    ).sort_index()

    if not (
        oos_equity.index.is_unique
    ):
        raise ValueError(
            "Las ventanas OOS se traslapan."
        )

    trade_param_frames = [
        fold["trade_params"]
        for fold in folds
        if len(
            fold["trade_params"]
        ) > 0
    ]

    if trade_param_frames:
        trade_params = pd.concat(
            trade_param_frames
        ).sort_index()
    else:
        trade_params = (
            pd.DataFrame()
        )

    trade_frames = [
        fold["oos_trades"]
        for fold in folds
        if len(
            fold["oos_trades"]
        ) > 0
    ]

    if trade_frames:
        oos_trades = pd.concat(
            trade_frames,
            ignore_index=True,
        )
    else:
        oos_trades = (
            pd.DataFrame()
        )

    params_by_fold = {
        fold["fold"]:
            fold[
                "params_by_regime"
            ]
        for fold in folds
    }

    elapsed_seconds = (
        time.perf_counter()
        - started
    )

    return {
        "mode":
            mode,
        "per_regime":
            per_regime,
        "folds":
            folds,
        "params_by_fold":
            params_by_fold,
        "trade_params":
            trade_params,
        "oos_equity":
            oos_equity,
        "oos_trades":
            oos_trades,
        "n_folds":
            len(folds),
        "n_trials_total":
            sum(
                fold[
                    "n_trials_total"
                ]
                for fold in folds
            ),
        "elapsed_seconds":
            float(
                elapsed_seconds
            ),
    }

def wf_efficiency(wf_result: dict) -> float:
    """Desempeño fuera de muestra concatenado / promedio del desempeño dentro de muestra."""
    raise NotImplementedError


def sensitivity(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    pct: float = 0.20,
) -> pd.DataFrame:
    """Sensibilidad de cada parámetro óptimo a ±pct.

    Cada parámetro se modifica individualmente manteniendo los demás
    constantes. Los parámetros enteros se redondean al entero más
    cercano, como exige P2.

    Returns
    -------
    pd.DataFrame
        Régimen, parámetro, factor, valor base, valor probado,
        Calmar base, Calmar probado y delta de Calmar.
    """
    if not 0 < pct < 1:
        raise ValueError(
            "pct debe estar en el intervalo (0, 1)."
        )

    _, baseline_metrics = _evaluate_params(
        params_by_regime,
        prices,
        regimes,
        config,
    )

    baseline_calmar = baseline_metrics[
        "calmar"
    ]

    rows = []

    for regime_name in sorted(
        params_by_regime
    ):
        base_params = params_by_regime[
            regime_name
        ]

        for parameter in search_space():
            base_value = base_params[
                parameter
            ]

            for factor in (
                1.0 - pct,
                1.0 + pct,
            ):
                candidate = {
                    name: dict(values)
                    for name, values
                    in params_by_regime.items()
                }

                varied_value = (
                    base_value
                    * factor
                )

                if (
                    _PARAMETER_KINDS[
                        parameter
                    ]
                    == "int"
                ):
                    varied_value = int(
                        round(varied_value)
                    )
                else:
                    varied_value = float(
                        varied_value
                    )

                candidate[
                    regime_name
                ][
                    parameter
                ] = varied_value

                _, candidate_metrics = (
                    _evaluate_params(
                        candidate,
                        prices,
                        regimes,
                        config,
                    )
                )

                candidate_calmar = (
                    candidate_metrics[
                        "calmar"
                    ]
                )

                if (
                    np.isfinite(
                        baseline_calmar
                    )
                    and np.isfinite(
                        candidate_calmar
                    )
                ):
                    delta_calmar = (
                        candidate_calmar
                        - baseline_calmar
                    )
                else:
                    delta_calmar = np.nan

                rows.append(
                    {
                        "regime":
                            regime_name,
                        "parameter":
                            parameter,
                        "factor":
                            factor,
                        "base_value":
                            base_value,
                        "tested_value":
                            varied_value,
                        "baseline_calmar":
                            baseline_calmar,
                        "calmar":
                            candidate_calmar,
                        "delta_calmar":
                            delta_calmar,
                    }
                )

    return pd.DataFrame(rows)

def _breakeven_cost_bps(
    cost_curve: pd.DataFrame,
) -> float:
    """Estima el costo de equilibrio donde el retorno neto llega a cero.

    Si el cruce ocurre entre dos niveles evaluados, se usa
    interpolación lineal entre ambos puntos.
    """
    net_return = (
        cost_curve["net_return"]
        .dropna()
        .sort_index()
    )

    if net_return.empty:
        return np.nan

    values = net_return.to_numpy(
        dtype=float
    )

    costs = net_return.index.to_numpy(
        dtype=float
    )

    zero_mask = np.isclose(
        values,
        0.0,
        atol=1e-12,
    )

    if zero_mask.any():
        first_zero = np.flatnonzero(
            zero_mask
        )[0]

        return float(
            costs[first_zero]
        )

    if values[0] < 0:
        return 0.0

    for position in range(
        1,
        len(values),
    ):
        previous_return = values[
            position - 1
        ]

        current_return = values[
            position
        ]

        if (
            previous_return > 0
            and current_return < 0
        ):
            previous_cost = costs[
                position - 1
            ]

            current_cost = costs[
                position
            ]

            crossing = (
                previous_cost
                + (
                    -previous_return
                    * (
                        current_cost
                        - previous_cost
                    )
                    / (
                        current_return
                        - previous_return
                    )
                )
            )

            return float(
                crossing
            )

    return np.nan

def cost_sweep(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    round_trip_bps: list[float],
) -> pd.DataFrame:
    """Evalúa la estrategia frente a distintos costos de transacción.

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

    Returns
    -------
    pd.DataFrame
        Retorno neto anualizado, Calmar y número de trades por
        nivel de costo, junto con el break-even y el margen frente
        al costo base.
    """
    if not round_trip_bps:
        raise ValueError(
            "round_trip_bps no puede estar vacío."
        )

    costs = sorted(
        {
            float(value)
            for value
            in round_trip_bps
        }
    )

    if any(
        (
            not np.isfinite(value)
            or value < 0
        )
        for value in costs
    ):
        raise ValueError(
            "Los costos deben ser finitos y no negativos."
        )

    base_commission = float(
        config["commission"]
    )

    base_slippage = float(
        config["slippage"]
    )

    base_per_side = (
        base_commission
        + base_slippage
    )

    base_round_trip_bps = (
        2.0
        * base_per_side
        * 10_000.0
    )

    rows = []

    for cost_bps in costs:
        candidate_config = dict(
            config
        )

        target_per_side = (
            cost_bps
            / 20_000.0
        )

        if base_per_side > 0:
            scale = (
                target_per_side
                / base_per_side
            )

            candidate_config[
                "commission"
            ] = (
                base_commission
                * scale
            )

            candidate_config[
                "slippage"
            ] = (
                base_slippage
                * scale
            )

        else:
            candidate_config[
                "commission"
            ] = 0.0

            candidate_config[
                "slippage"
            ] = target_per_side

        result, metrics = (
            _evaluate_params(
                params_by_regime,
                prices,
                regimes,
                candidate_config,
            )
        )

        rows.append(
            {
                "round_trip_bps":
                    cost_bps,
                "net_return":
                    metrics[
                        "ann_return"
                    ],
                "calmar":
                    metrics[
                        "calmar"
                    ],
                "n_trades":
                    int(
                        len(
                            result.trades
                        )
                    ),
            }
        )

    cost_curve = (
        pd.DataFrame(rows)
        .set_index(
            "round_trip_bps"
        )
        .sort_index()
    )

    cost_curve.index.name = (
        "round_trip_bps"
    )

    break_even_bps = (
        _breakeven_cost_bps(
            cost_curve
        )
    )

    if np.isfinite(
        break_even_bps
    ):
        margin_bps = (
            break_even_bps
            - base_round_trip_bps
        )
    else:
        margin_bps = np.nan

    cost_curve[
        "break_even_bps"
    ] = break_even_bps

    cost_curve[
        "base_cost_bps"
    ] = base_round_trip_bps

    cost_curve[
        "margin_bps"
    ] = margin_bps

    return cost_curve

def single_indicator_comparison(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Compara indicadores individuales contra la regla 2 de 3.

    Reporta el número de operaciones cerradas y el Calmar para
    SMA, MACD, RSI y la estrategia principal de confirmación
    2 de 3.

    Returns
    -------
    pd.DataFrame
        Estrategia, número de operaciones y Calmar.
    """
    rows = []

    result_2of3, metrics_2of3 = (
        _evaluate_params(
            params_by_regime,
            prices,
            regimes,
            config,
        )
    )

    rows.append(
        {
            "strategy": "2_de_3",
            "n_trades": int(
                len(
                    result_2of3.trades
                )
            ),
            "calmar": metrics_2of3[
                "calmar"
            ],
        }
    )

    for indicator in (
        "sma",
        "macd",
        "rsi",
    ):
        result, metrics = (
            _evaluate_single_indicator(
                indicator,
                params_by_regime,
                prices,
                regimes,
                config,
            )
        )

        rows.append(
            {
                "strategy":
                    indicator,
                "n_trades": int(
                    len(
                        result.trades
                    )
                ),
                "calmar":
                    metrics["calmar"],
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("strategy")
    )
