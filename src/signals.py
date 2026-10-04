"""Indicadores técnicos, votos y señal confirmada 2 de 3 (P2).

Todo lo calculado en t usa información hasta el cierre de t; aquí no se hace `shift` para simular
ejecución (CLAUDE.md, sección 4).
"""

import pandas as pd


def compute_indicators(ohlcv: pd.DataFrame, params: dict, config: dict) -> pd.DataFrame:
    """Calcula SMA rápida y lenta, histograma MACD, RSI de Wilder y ATR de Wilder.

    SPEC punto 2. MACD 12/26/9 y ATR 14 vienen de `config`; f, s y n vienen de `params`.

    Returns
    -------
    pd.DataFrame
        Columnas: sma_fast, sma_slow, macd_hist, rsi, atr. NaN mientras no hay historia suficiente.
    """
    raise NotImplementedError


def indicator_votes(indicators: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Convierte los indicadores en votos x_j ∈ {−1, 0, +1}.

    SPEC punto 2: x_sma = sgn(SMA_f − SMA_s); x_macd = sgn(MACD − señal); RSI vota +1 si
    50 < RSI < hi, −1 si lo < RSI ≤ 50 y 0 en otro caso. Sin historia suficiente, el voto es 0.

    Returns
    -------
    pd.DataFrame
        Columnas: v_sma, v_macd, v_rsi.
    """
    raise NotImplementedError


def confirm_signal(votes: pd.DataFrame, min_agree: int = 2) -> pd.Series:
    """Estado_t = sgn(Σx) si |Σx| ≥ min_agree; 0 en otro caso (SPEC punto 3, compuerta)."""
    raise NotImplementedError


def signal_strength(votes: pd.DataFrame, min_agree: int = 2) -> pd.Series:
    """s_t = Σx / 3 si |Σx| ≥ min_agree; 0 en otro caso (SPEC punto 3, fuerza)."""
    raise NotImplementedError


def generate_signals(
    prices: dict,
    params_by_regime: dict[str, dict],
    regimes: pd.Series,
    config: dict,
) -> dict[str, pd.DataFrame]:
    """Genera los paneles de señal para todos los activos.

    En cada fecha usa los parámetros del régimen vigente. Para un θ único, `params_by_regime`
    lleva el mismo dict en las tres llaves.

    Returns
    -------
    dict[str, pd.DataFrame]
        {"state", "strength", "atr"}: paneles fecha × ticker.
    """
    raise NotImplementedError
