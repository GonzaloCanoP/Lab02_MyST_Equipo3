"""Optimización con Optuna, walk-forward y análisis de robustez (P2).

Firmas propuestas: P2 puede ajustarlas avisando a P1 (main.py). CLAUDE.md, sección 7.
"""

import math

import numpy as np
import optuna
import pandas as pd

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

    metrics = compute_metrics(
        result.equity,
        result.trades,
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


def diagnostic_study(
    prices: dict, config: dict, sampler: str, n_trials: int, seed: int
) -> optuna.Study:
    """Estudio de diagnóstico sobre todo train con θ único; `sampler` ∈ {"random", "tpe"}."""
    raise NotImplementedError


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


def optimize_regime(
    prices: dict,
    regimes: pd.Series,
    regime: str | None,
    window: tuple,
    config: dict,
    seed: int,
) -> dict:
    """Optimiza θ para un régimen dentro de una ventana de entrenamiento; `regime=None` → θ único."""
    raise NotImplementedError


def walk_forward(
    prices: dict,
    regimes: pd.Series,
    config: dict,
    mode: str = "rolling",
    per_regime: bool = True,
) -> dict:
    """Walk-forward 6m/1m/mensual (SPEC punto 1); `mode` ∈ {"rolling", "anchored"}."""
    raise NotImplementedError


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
