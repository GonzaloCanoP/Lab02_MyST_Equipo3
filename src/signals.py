"""Indicadores técnicos, votos y señal confirmada 2 de 3 (P2).

Todo lo calculado en t usa información disponible hasta el cierre de t.
La ejecución t -> t+1 ocurre únicamente dentro de run_backtest.
"""

import numpy as np
import pandas as pd


def _wilder_average(series: pd.Series, window: int) -> pd.Series:
    """Promedio suavizado de Wilder usando una EWM causal."""
    return series.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()


def compute_indicators(
    ohlcv: pd.DataFrame,
    params: dict,
    config: dict,
) -> pd.DataFrame:
    """Calcula SMA, MACD, RSI de Wilder y ATR de Wilder."""

    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]

    sma_fast = close.rolling(
        window=params["sma_fast"],
        min_periods=params["sma_fast"],
    ).mean()

    sma_slow = close.rolling(
        window=params["sma_slow"],
        min_periods=params["sma_slow"],
    ).mean()

    ema_fast = close.ewm(
        span=config["macd_fast"],
        adjust=False,
        min_periods=config["macd_fast"],
    ).mean()

    ema_slow = close.ewm(
        span=config["macd_slow"],
        adjust=False,
        min_periods=config["macd_slow"],
    ).mean()

    macd = ema_fast - ema_slow

    macd_signal = macd.ewm(
        span=config["macd_signal"],
        adjust=False,
        min_periods=config["macd_signal"],
    ).mean()

    macd_hist = macd - macd_signal

    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = _wilder_average(
        gains,
        params["rsi_window"],
    )

    avg_loss = _wilder_average(
        losses,
        params["rsi_window"],
    )

    rs = avg_gain / avg_loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = _wilder_average(
        true_range,
        config["atr_window"],
    )

    return pd.DataFrame(
        {
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "macd_hist": macd_hist,
            "rsi": rsi,
            "atr": atr,
        },
        index=ohlcv.index,
    )


def indicator_votes(
    indicators: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Convierte SMA, MACD y RSI en votos -1, 0 o +1."""

    index = indicators.index

    sma_difference = (
        indicators["sma_fast"]
        - indicators["sma_slow"]
    )

    v_sma = np.sign(
        sma_difference
    ).fillna(0).astype(int)

    v_macd = np.sign(
        indicators["macd_hist"]
    ).fillna(0).astype(int)

    rsi = indicators["rsi"]

    v_rsi = pd.Series(
        0,
        index=index,
        dtype=int,
    )

    long_mask = (
        (rsi > 50)
        & (rsi < params["rsi_hi"])
    )

    short_mask = (
        (rsi > params["rsi_lo"])
        & (rsi <= 50)
    )

    v_rsi.loc[long_mask] = 1
    v_rsi.loc[short_mask] = -1

    return pd.DataFrame(
        {
            "v_sma": v_sma,
            "v_macd": v_macd,
            "v_rsi": v_rsi,
        },
        index=index,
    )


def confirm_signal(
    votes: pd.DataFrame,
    min_agree: int = 2,
) -> pd.Series:
    """Estado = signo de la suma si hay confirmación suficiente."""

    vote_sum = votes[
        ["v_sma", "v_macd", "v_rsi"]
    ].sum(axis=1)

    state = pd.Series(
        0,
        index=votes.index,
        dtype=int,
        name="state",
    )

    confirmed = (
        vote_sum.abs()
        >= min_agree
    )

    state.loc[confirmed] = np.sign(
        vote_sum.loc[confirmed]
    ).astype(int)

    return state


def signal_strength(
    votes: pd.DataFrame,
    min_agree: int = 2,
) -> pd.Series:
    """Fuerza = suma de votos / 3 si se cumple la compuerta."""

    vote_sum = votes[
        ["v_sma", "v_macd", "v_rsi"]
    ].sum(axis=1)

    strength = pd.Series(
        0.0,
        index=votes.index,
        name="strength",
    )

    confirmed = (
        vote_sum.abs()
        >= min_agree
    )

    strength.loc[confirmed] = (
        vote_sum.loc[confirmed]
        / 3
    )

    return strength


def generate_signals(
    prices: dict,
    params_by_regime: dict[str, dict],
    regimes: pd.Series,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Genera estado, fuerza y ATR para todos los activos."""

    tickers = list(prices)

    if not tickers:
        raise ValueError(
            "prices no puede estar vacío."
        )

    dates = prices[
        tickers[0]
    ].index

    regimes = regimes.reindex(
        dates
    )

    state = pd.DataFrame(
        0,
        index=dates,
        columns=tickers,
        dtype=int,
    )

    strength = pd.DataFrame(
        0.0,
        index=dates,
        columns=tickers,
    )

    atr = pd.DataFrame(
        np.nan,
        index=dates,
        columns=tickers,
    )

    used_regimes = set(
        regimes.dropna().unique()
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

    for ticker in tickers:

        price_data = prices[ticker]

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

            ticker_state = (
                confirm_signal(
                    votes,
                    config["min_agree"],
                )
            )

            ticker_strength = (
                signal_strength(
                    votes,
                    config["min_agree"],
                )
            )

            mask = (
                regimes
                == regime_name
            )

            state.loc[
                mask,
                ticker,
            ] = ticker_state.loc[
                mask
            ]

            strength.loc[
                mask,
                ticker,
            ] = ticker_strength.loc[
                mask
            ]

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

def sma_macd_vote_correlation(
    prices: dict,
    params: dict,
    config: dict,
    period: tuple,
) -> pd.DataFrame:
    """Mide la correlación entre los votos SMA y MACD.

    Los indicadores se calculan usando toda la historia disponible
    hasta el final del periodo para conservar el warm-up, pero la
    correlación se calcula únicamente dentro de ``period``.

    Se reporta una correlación por activo y una correlación agrupada
    con todas las observaciones disponibles.

    Parameters
    ----------
    prices : dict
        Datos OHLCV por activo.
    params : dict
        Parámetros de indicadores. Para la calibración inicial de
        P2 se usan ``config["base_params"]``.
    config : dict
        Configuración del proyecto.
    period : tuple
        Fecha inicial y final del periodo de calibración.

    Returns
    -------
    pd.DataFrame
        Índice por ticker más una fila ``pooled`` con columnas
        ``correlation`` y ``n_obs``.
    """
    if not prices:
        raise ValueError(
            "prices no puede estar vacío."
        )

    if len(period) != 2:
        raise ValueError(
            "period debe ser (inicio, fin)."
        )

    start = pd.Timestamp(
        period[0]
    )

    end = pd.Timestamp(
        period[1]
    )

    if start > end:
        raise ValueError(
            "El inicio del periodo no puede "
            "ser posterior al fin."
        )

    rows = []
    pooled_pairs = []

    for ticker, ohlcv in prices.items():
        history = ohlcv.loc[
            :end
        ]

        indicators = compute_indicators(
            history,
            params,
            config,
        )

        votes = indicator_votes(
            indicators,
            params,
        )

        available = (
            indicators[
                "sma_slow"
            ].notna()
            & indicators[
                "macd_hist"
            ].notna()
        )

        pair = (
            votes.loc[
                available,
                [
                    "v_sma",
                    "v_macd",
                ],
            ]
            .loc[
                start:end
            ]
            .copy()
        )

        n_obs = len(
            pair
        )

        if (
            n_obs >= 2
            and pair[
                "v_sma"
            ].nunique() > 1
            and pair[
                "v_macd"
            ].nunique() > 1
        ):
            correlation = float(
                pair[
                    "v_sma"
                ].corr(
                    pair[
                        "v_macd"
                    ]
                )
            )
        else:
            correlation = np.nan

        rows.append(
            {
                "ticker":
                    ticker,
                "correlation":
                    correlation,
                "n_obs":
                    n_obs,
            }
        )

        if n_obs > 0:
            pooled_pairs.append(
                pair
            )

    if pooled_pairs:
        pooled = pd.concat(
            pooled_pairs,
            ignore_index=True,
        )

        if (
            len(pooled) >= 2
            and pooled[
                "v_sma"
            ].nunique() > 1
            and pooled[
                "v_macd"
            ].nunique() > 1
        ):
            pooled_correlation = float(
                pooled[
                    "v_sma"
                ].corr(
                    pooled[
                        "v_macd"
                    ]
                )
            )
        else:
            pooled_correlation = np.nan

        pooled_n_obs = len(
            pooled
        )
    else:
        pooled_correlation = np.nan
        pooled_n_obs = 0

    rows.append(
        {
            "ticker":
                "pooled",
            "correlation":
                pooled_correlation,
            "n_obs":
                pooled_n_obs,
        }
    )

    return (
        pd.DataFrame(rows)
        .set_index("ticker")
    )