"""Optimización con Optuna, walk-forward y análisis de robustez (P2).

Firmas propuestas: P2 puede ajustarlas avisando a P1 (main.py). CLAUDE.md, sección 7.
"""

import optuna
import pandas as pd


def search_space() -> dict:
    """Espacio de búsqueda de 9 dimensiones por régimen: f, s, n, lo, hi, k, r, m, ρ_g (SPEC punto 7)."""
    raise NotImplementedError


def diagnostic_study(
    prices: dict, config: dict, sampler: str, n_trials: int, seed: int
) -> optuna.Study:
    """Estudio de diagnóstico sobre todo train con θ único; `sampler` ∈ {"random", "tpe"}."""
    raise NotImplementedError


def select_plateau(study: optuna.Study, top_frac: float = 0.10) -> dict:
    """Medoide del mejor `top_frac` de pruebas factibles (centro de meseta, SPEC punto 7)."""
    raise NotImplementedError


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
    """Varía cada parámetro a ×(1 − pct) y ×(1 + pct) y reporta el cambio en Calmar."""
    raise NotImplementedError


def cost_sweep(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
    round_trip_bps: list[float],
) -> pd.DataFrame:
    """Retorno neto contra costo de ida y vuelta; identifica el punto de equilibrio."""
    raise NotImplementedError


def single_indicator_comparison(
    params_by_regime: dict,
    prices: dict,
    regimes: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """Operaciones y Calmar de cada indicador solo contra la regla 2 de 3 (pregunta 1)."""
    raise NotImplementedError
