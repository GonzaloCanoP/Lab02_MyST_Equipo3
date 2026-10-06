"""Risk Parity, agregación de señales y rebalanceo (P4).

La covarianza en t usa solo retornos hasta t (CLAUDE.md, sección 4).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.backtest import run_backtest
from src.metrics import compute_metrics
from src.signals import generate_signals

# El contrato de `estimate_cov` no recibe `config`, así que λ vive aquí (SPEC_portafolio.md:
# T_eff = 1 / (1 − λ) = 100 días, comparable a la ventana de 126 días).
_TRADE_PARAM_KEYS = ["k_stop", "reward_ratio", "max_holding", "risk_per_trade"]
EWMA_LAMBDA = 0.99
ESTIMATORS = ("sample", "ewma", "ledoit_wolf")


def _validate_cov(cov: pd.DataFrame) -> None:
    """Exige una covarianza cuadrada, finita, simétrica y con índice igual a columnas."""
    if cov.shape[0] != cov.shape[1] or not cov.index.equals(cov.columns):
        raise ValueError("cov debe ser cuadrada, con el mismo índice en filas y columnas")
    values = cov.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cov contiene valores no finitos")
    if not np.allclose(values, values.T, atol=1e-12):
        raise ValueError("cov no es simétrica")
    if (np.diag(values) <= 0).any():
        raise ValueError("cov tiene varianzas no positivas")


def estimate_cov(returns: pd.DataFrame, method: str) -> pd.DataFrame:
    """Estima la matriz de covarianzas de retornos (SPEC_portafolio.md, Estimador de covarianza).

    Parameters
    ----------
    returns : pd.DataFrame
        Retornos simples diarios, fechas × tickers. Nunca precios (CLAUDE.md, sección 4 y Paso 2).
        No se rellenan huecos: si hay NaN, se lanza un error.
    method : str
        "sample" (muestral), "ewma" (λ = 0.99, media cero) o "ledoit_wolf" (encogimiento).

    Returns
    -------
    pd.DataFrame
        Σ tickers × tickers, simétrica.
    """
    if returns.isna().any().any():
        raise ValueError("returns contiene NaN; elimina la primera fila de la diferencia antes")
    tickers = returns.columns

    if method == "sample":
        values = returns.cov().to_numpy()
    elif method == "ewma":
        x = returns.to_numpy(dtype=float)
        # Peso λ^(T−1−k) a la observación k: la más reciente pesa más. Σ = Σ_k w_k r_k r_kᵀ.
        w = EWMA_LAMBDA ** np.arange(len(x) - 1, -1, -1)
        w = w / w.sum()
        values = (x * w[:, None]).T @ x
    elif method == "ledoit_wolf":
        values = LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_
    else:
        raise ValueError(f"método de covarianza desconocido: {method!r}")

    values = (values + values.T) / 2  # elimina asimetría numérica
    return pd.DataFrame(values, index=tickers, columns=tickers)


def _newton_polish(y: np.ndarray, S: np.ndarray, tol: float, max_iter: int = 50) -> np.ndarray:
    """Pule la solución con Newton amortiguado, manteniendo y > 0.

    Gradiente: g = S y − 1/(n y).  Hessiano: H = S + diag(1 / (n y²)).
    """
    n = len(y)
    for _ in range(max_iter):
        gradient = S @ y - 1.0 / (n * y)
        if np.max(np.abs(gradient)) < tol:
            return y
        hessian = S + np.diag(1.0 / (n * y**2))
        step = np.linalg.solve(hessian, gradient)
        t = 1.0
        while np.any(y - t * step <= 0):  # recorta el paso hasta que y siga positivo
            t /= 2
        y = y - t * step
    raise RuntimeError("risk_parity_weights no convergió dentro de la tolerancia")


def risk_parity_weights(cov: pd.DataFrame, tol: float = 1e-10) -> pd.Series:
    """Pesos Risk Parity long-only con la formulación convexa de Spinu (Paso 5).

    Resuelve  min_{y>0} ½ yᵀΣy − (1/n) Σ ln y_i  y normaliza w = y / Σ y_j.
    La solución es única porque el problema es convexo.

    Parameters
    ----------
    cov : pd.DataFrame
        Σ tickers × tickers (de `estimate_cov`).
    tol : float
        Tolerancia sobre el gradiente, que equivale a y_i (Σy)_i = 1/n para todo i.

    Returns
    -------
    pd.Series
        w_i > 0 con Σ w_i = 1, indexado como `cov`.
    """
    _validate_cov(cov)
    n = cov.shape[0]
    # Los pesos no cambian si se escala Σ; se normaliza para que el problema esté bien condicionado
    # (varianzas diarias del orden de 1e-4 dejan el término logarítmico dominando).
    S = cov.to_numpy(dtype=float)
    S = S / np.mean(np.diag(S))

    def objective(y: np.ndarray) -> float:
        return 0.5 * y @ S @ y - np.log(y).sum() / n

    def gradient(y: np.ndarray) -> np.ndarray:
        return S @ y - 1.0 / (n * y)

    # Punto de partida: volatilidad inversa, escalada para que yᵀSy = 1 (valor en el óptimo).
    y0 = 1.0 / np.sqrt(np.diag(S))
    y0 = y0 / np.sqrt(y0 @ S @ y0)

    result = minimize(
        objective, y0, jac=gradient, method="L-BFGS-B",
        bounds=[(1e-12, None)] * n,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000},
    )
    y = _newton_polish(result.x, S, tol)
    return pd.Series(y / y.sum(), index=cov.index, name="rp_weights")


def inverse_vol_weights(cov: pd.DataFrame) -> pd.Series:
    """Versión naive de Risk Parity (Paso 4): w_i = (1/σ_i) / Σ_j (1/σ_j)."""
    _validate_cov(cov)
    inv_vol = 1.0 / np.sqrt(np.diag(cov.to_numpy(dtype=float)))
    return pd.Series(inv_vol / inv_vol.sum(), index=cov.index, name="inverse_vol_weights")


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Contribución total al riesgo (Paso 3): RC_i = w_i (Σw)_i / σ_p, con Σ RC_i = σ_p exacto."""
    _validate_cov(cov)
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    if np.isnan(w).any():
        raise ValueError("weights no cubre todos los activos de cov")
    sigma_w = cov.to_numpy(dtype=float) @ w
    sigma_p = np.sqrt(w @ sigma_w)
    return pd.Series(w * sigma_w / sigma_p, index=cov.index, name="risk_contribution")


def equal_weights(tickers: list[str]) -> pd.Series:
    """Pesos iguales 1/n (benchmark, SPEC punto 5)."""
    return pd.Series(1.0 / len(tickers), index=list(tickers), name="equal_weights")


def compose_target(
    base_weights: pd.Series, strength: pd.Series, regime_multiplier: float
) -> pd.Series:
    """Peso objetivo de un día (Paso 7): m · w̃ / max(1, Σ|w̃|), con w̃ = w · s (SPEC punto 5).

    Parameters
    ----------
    base_weights : pd.Series
        w^RP (o 1/n en el benchmark), indexado por ticker.
    strength : pd.Series
        s_i ∈ [-1, 1] de `generate_signals`. Conserva el signo.
    regime_multiplier : float
        m(régimen) ∈ (0, 1].

    Returns
    -------
    pd.Series
        w_target con signo. El panel que recibe el motor es |w_target| (`sleeve_weights`).
    """
    if not 0 < regime_multiplier <= 1:
        raise ValueError("regime_multiplier debe estar en (0, 1]")
    s = strength.reindex(base_weights.index)
    if s.isna().any():
        raise ValueError("strength no cubre todos los activos de base_weights")
    if (s.abs() > 1 + 1e-12).any():
        raise ValueError("strength debe estar en [-1, 1]")

    scaled = base_weights * s
    gross = scaled.abs().sum()
    return regime_multiplier * scaled / max(1.0, gross)


def resolve_signal_conflicts(
    strength: pd.Series, corr: pd.DataFrame, threshold: float = 0.7
) -> pd.Series:
    """Política ante señales opuestas en activos muy correlacionados (SPEC_portafolio, Agregación).

    Dos activos i, j están en conflicto si ρ_ij > `threshold` y s_i, s_j tienen signo opuesto.
    En cada conflicto se conserva la señal de mayor |s| y la otra pasa a 0; con |s_i| = |s_j|,
    ambas pasan a 0. Todas las decisiones se toman sobre la `strength` original, así que el
    resultado no depende del orden de los activos.

    Parameters
    ----------
    strength : pd.Series
        s_i del día, indexada por ticker.
    corr : pd.DataFrame
        Correlaciones entre los activos, calculadas solo con retornos hasta ese día.
    threshold : float
        Umbral de correlación (SPEC: 0.7).

    Returns
    -------
    pd.Series
        Copia de `strength` con las señales perdedoras en 0.
    """
    s = strength.to_numpy(dtype=float)
    rho = corr.reindex(index=strength.index, columns=strength.index).to_numpy(dtype=float)
    strength_abs = np.abs(s)
    conflict = (rho > threshold) & (np.outer(s, s) < 0)
    # i pierde si algún rival con el que choca tiene |s| mayor o igual al suyo.
    loses = (conflict & (strength_abs[None, :] >= strength_abs[:, None])).any(axis=1)
    resolved = strength.copy()
    resolved[loses] = 0.0
    return resolved


def _simple_returns(prices: dict) -> pd.DataFrame:
    """Retornos simples diarios del `close` de cada activo (nunca precios; Paso 2).

    Simples y no logarítmicos: el retorno del portafolio es lineal en ellos (r_p = Σ w_i r_i), así
    que σ_p² = wᵀΣw y las contribuciones al riesgo son exactas.
    """
    closes = pd.DataFrame({ticker: ohlcv["close"] for ticker, ohlcv in prices.items()})
    return closes.pct_change(fill_method=None)


def _base_weights(returns: pd.DataFrame, config: dict, method: str) -> pd.DataFrame:
    """Panel de w^RP vigente por fecha, con rebalanceo híbrido (SPEC_portafolio, Rebalanceo).

    En el primer día hábil de cada periodo de `rebalance_frequency` se reestima Σ con los últimos
    `cov_window` retornos (hasta ese día) y se calcula el w^RP candidato; se adopta solo si
    ‖w_cand − w_vigente‖₁ > `rebalance_band`. Entre revisiones w^RP se mantiene. La primera
    adopción ocurre en cuanto hay `cov_window` retornos. Antes de eso el panel es NaN.
    """
    tickers = list(returns.columns)
    window, band = config["cov_window"], config["rebalance_band"]
    # Primer día hábil de cada periodo; depende solo de fechas anteriores, así que es causal.
    is_review = ~returns.index.to_period(config["rebalance_frequency"]).duplicated()
    panel = np.full(returns.shape, np.nan)
    current = None
    for pos in range(len(returns)):
        if pos >= window and (current is None or is_review[pos]):
            if method == "risk_parity":
                sample = returns.iloc[pos - window + 1 : pos + 1]
                candidate = risk_parity_weights(estimate_cov(sample, config["cov_method"]))
            else:
                candidate = equal_weights(tickers)
            if current is None or (candidate - current).abs().sum() > band:
                current = candidate
        if current is not None:
            panel[pos] = current.to_numpy()
    return pd.DataFrame(panel, index=returns.index, columns=tickers)


def sleeve_weights(
    prices: dict,
    signals: dict[str, pd.DataFrame],
    regimes: pd.Series,
    config: dict,
    method: str = "risk_parity",
) -> pd.DataFrame:
    """Panel causal de |w_target| con rebalanceo aplicado; `method` ∈ {"risk_parity", "equal"}.

    Es la única puerta entre el portafolio y el motor (SPEC punto 5). Para cada fecha t:

    1. w^RP vigente (`_base_weights`), o 1/n con `method="equal"` y el mismo rebalanceo.
    2. s_i de `signals["strength"]`, con la política de conflictos entre activos correlacionados
       (`resolve_signal_conflicts`, ρ sobre los últimos `cov_window` retornos hasta t).
    3. `compose_target` con m(régimen) de `config["regime_multiplier"]`; se devuelve |w_target|.

    s_i y m(régimen) se actualizan cada día; solo w^RP se rebalancea. No hay `shift`: el
    desplazamiento t → t+1 lo hace `run_backtest` (CLAUDE.md, sección 4).

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        OHLCV por ticker con índice común.
    signals : dict[str, pd.DataFrame]
        Salida de `generate_signals`; aquí solo se usa el panel "strength".
    regimes : pd.Series
        Etiqueta filtrada por fecha ("tendencia", "reversion", "crisis"); NaN sin etiqueta.
    config : dict
        Usa `cov_method`, `cov_window`, `rebalance_frequency` (alias de periodo de pandas: "D",
        "W", "M", "Q"), `rebalance_band`, `regime_multiplier` y, opcional,
        `conflict_corr_threshold` (0.7 por defecto).
    method : str
        "risk_parity" o "equal" (benchmark de pesos iguales con las mismas señales y costos).

    Returns
    -------
    pd.DataFrame
        Fechas × tickers con |w_target| ≥ 0 y Σ ≤ 1. NaN donde no hay asignación definida (sin
        historia para Σ o sin régimen); `run_backtest` no abre posiciones con NaN.
    """
    if method not in ("risk_parity", "equal"):
        raise ValueError(f"method desconocido: {method!r}")
    multiplier = config["regime_multiplier"]
    returns = _simple_returns(prices)
    dates, tickers = returns.index, list(returns.columns)

    labels = regimes.reindex(dates)
    unknown = set(labels.dropna().unique()) - set(multiplier)
    if unknown:
        raise ValueError(f"regime_multiplier no define los regímenes: {sorted(unknown)}")

    strength = signals["strength"].reindex(index=dates, columns=tickers).fillna(0.0)
    base = _base_weights(returns, config, method)
    window = config["cov_window"]
    threshold = config.get("conflict_corr_threshold", 0.7)

    target = np.full(base.shape, np.nan)
    for pos in range(len(dates)):
        regime, w_rp = labels.iloc[pos], base.iloc[pos]
        if pd.isna(regime) or w_rp.isna().any():
            continue
        s = strength.iloc[pos]
        if (s > 0).any() and (s < 0).any():  # sin señales opuestas no hay conflicto posible
            corr = returns.iloc[pos - window + 1 : pos + 1].corr()
            s = resolve_signal_conflicts(s, corr, threshold)
        target[pos] = compose_target(w_rp, s, multiplier[regime]).abs().to_numpy()
    return pd.DataFrame(target, index=dates, columns=tickers)


def turnover(weights_drift: pd.DataFrame, weights_target: pd.DataFrame) -> pd.Series:
    """Turnover por fecha (Paso 8): T_t = ½ Σ |w_t − w_t−|, con w_t− el peso después del drift.

    `weights_drift` es el peso justo antes de rebalancear y `weights_target` el peso al que se
    rebalancea. Comparar contra el objetivo viejo en vez del drift sobrestima el costo.
    """
    if not weights_drift.index.equals(weights_target.index) or \
            not weights_drift.columns.equals(weights_target.columns):
        raise ValueError("weights_drift y weights_target deben tener el mismo índice y columnas")
    return 0.5 * (weights_target - weights_drift).abs().sum(axis=1).rename("turnover")


def rebalance_sweep(
    prices: dict,
    params_by_regime: dict,
    regimes: pd.Series,
    config: dict,
    bands: list[float],
    frequencies: list[str],
    trade_params: pd.DataFrame | None = None,
    period: tuple | None = None,
) -> pd.DataFrame:
    """Barrido de rebalanceo: retorno bruto, costo total, retorno neto y turnover.

    Corre Risk Parity completo (señales → `sleeve_weights` → `run_backtest`) para cada
    combinación de frecuencia × banda, con las mismas señales y θ en todas.

    Parameters
    ----------
    trade_params : pd.DataFrame, optional
        Parámetros de operación por fecha para `run_backtest`. Si falta, se usa
        `config["base_params"]` constante, como la corrida base.
    period : (inicio, fin), optional
        Recorta las métricas a un bloque (p. ej. train o validation); el backtest corre completo.

    Returns
    -------
    pd.DataFrame
        Una fila por (frequency, band): n_rebalances (adopciones de pesos nuevos), turnover
        (anualizado: media de ½Σ|Δw| de los pesos adoptados, sumada y dividida entre los años),
        gross_return = (equity final + costos) / equity inicial − 1, total_cost (comisión +
        slippage + borrow, en dinero), net_return = equity final / equity inicial − 1 y n_trades.
    """
    if not bands or not frequencies:
        raise ValueError("bands y frequencies no pueden estar vacías")
    if any(b < 0 for b in bands):
        raise ValueError("las bandas deben ser no negativas")

    signals = generate_signals(prices, params_by_regime, regimes, config)
    dates = next(iter(prices.values())).index
    if trade_params is None:
        trade_params = pd.DataFrame(
            {k: config["base_params"][k] for k in _TRADE_PARAM_KEYS}, index=dates
        )
    returns = _simple_returns(prices)
    start, end = (dates[0], dates[-1]) if period is None else (pd.Timestamp(period[0]), pd.Timestamp(period[1]))

    rows = []
    for frequency in frequencies:
        for band in bands:
            cfg = {**config, "rebalance_frequency": frequency, "rebalance_band": band}
            panel = sleeve_weights(prices, signals, regimes, cfg, method="risk_parity")
            result = run_backtest(prices, signals, panel, trade_params, cfg)

            base = _base_weights(returns, cfg, "risk_parity")
            change = 0.5 * base.diff().abs().sum(axis=1)  # NaN en la primera fila con pesos
            change = change.loc[start:end].fillna(0.0)
            first = base.dropna().index[0] if base.notna().any().any() else None
            if first is not None and first in change.index:
                change.loc[first] = 0.0  # la primera adopción parte de cero: no es rotación
            years = max((end - start).days / 365.25, 1e-9)

            equity = result.equity.loc[start:end]
            total_cost = float(result.costs.loc[start:end].to_numpy().sum())
            net = equity.iloc[-1] / equity.iloc[0] - 1
            gross = (equity.iloc[-1] + total_cost) / equity.iloc[0] - 1
            trades = result.trades
            n_trades = int(trades["entry_date"].between(start, end).sum()) if len(trades) else 0
            rows.append(
                {
                    "frequency": frequency,
                    "band": band,
                    "n_rebalances": int((change > 0).sum()),
                    "turnover": float(change.sum() / years),
                    "gross_return": float(gross),
                    "total_cost": total_cost,
                    "net_return": float(net),
                    "n_trades": n_trades,
                }
            )
    return pd.DataFrame(rows)


def weight_stability(prices: dict, config: dict, period: tuple | None = None) -> pd.DataFrame:
    """Estabilidad de w^RP por estimador: desviación estándar del peso de cada activo entre revisiones.

    Para cada estimador de `ESTIMATORS` calcula los pesos Risk Parity en cada fecha de revisión
    (primer día hábil de cada periodo de `rebalance_frequency`) sin banda de aceptación, para ver
    todo el movimiento que produce el estimador. Menor desviación = pesos más estables
    (SPEC_portafolio, Estimador de covarianza).

    Parameters
    ----------
    period : (inicio, fin), optional
        Recorta las revisiones a un bloque (p. ej. train).

    Returns
    -------
    pd.DataFrame
        Una fila por estimador; columnas: una por activo (desviación estándar de su peso),
        `mean_std` (promedio entre activos) y `n_reviews` (revisiones usadas).
    """
    returns = _simple_returns(prices)
    is_review = ~returns.index.to_period(config["rebalance_frequency"]).duplicated()
    rows = {}
    for method in ESTIMATORS:
        cfg = {**config, "cov_method": method, "rebalance_band": 0.0}
        panel = _base_weights(returns, cfg, "risk_parity")
        reviewed = panel[is_review].dropna()
        if period is not None:
            reviewed = reviewed.loc[pd.Timestamp(period[0]) : pd.Timestamp(period[1])]
        std = reviewed.std(ddof=1)
        rows[method] = {**std.to_dict(), "mean_std": float(std.mean()), "n_reviews": len(reviewed)}
    return pd.DataFrame.from_dict(rows, orient="index")


def risk_contribution_comparison(
    prices: dict, config: dict, period: tuple | None = None
) -> pd.DataFrame:
    """Contribución al riesgo de Risk Parity contra pesos iguales, medida con la Σ de cada revisión.

    En cada fecha de revisión (primer día hábil de cada periodo de `rebalance_frequency`, con al
    menos `cov_window` retornos) se estima Σ con `cov_method` y se calcula la parte del riesgo que
    aporta cada activo, RC_i / σ_p, con los pesos de cada método. Risk Parity iguala esas partes
    (1/n); pesos iguales no.

    Returns
    -------
    pd.DataFrame
        Filas "risk_parity" y "equal". Columnas: una por activo (parte media del riesgo),
        `spread` (media del máximo menos el mínimo de las partes entre fechas) y `ann_vol`
        (volatilidad anualizada media del portafolio ex ante).
    """
    returns = _simple_returns(prices)
    tickers = list(returns.columns)
    window = config["cov_window"]
    is_review = ~returns.index.to_period(config["rebalance_frequency"]).duplicated()
    shares = {"risk_parity": [], "equal": []}
    vols = {"risk_parity": [], "equal": []}
    for pos in range(window, len(returns)):
        date = returns.index[pos]
        if not is_review[pos]:
            continue
        if period is not None and not pd.Timestamp(period[0]) <= date <= pd.Timestamp(period[1]):
            continue
        cov = estimate_cov(returns.iloc[pos - window + 1 : pos + 1], config["cov_method"])
        weights = {"risk_parity": risk_parity_weights(cov), "equal": equal_weights(tickers)}
        for method, w in weights.items():
            rc = risk_contributions(w, cov)
            sigma_p = rc.sum()
            shares[method].append((rc / sigma_p).to_numpy())
            vols[method].append(sigma_p * np.sqrt(config["periods_per_year"]))
    rows = {}
    for method, history in shares.items():
        if not history:
            raise ValueError("no hay fechas de revisión en el periodo pedido")
        panel = np.vstack(history)
        rows[method] = {
            **dict(zip(tickers, panel.mean(axis=0))),
            "spread": float((panel.max(axis=1) - panel.min(axis=1)).mean()),
            "ann_vol": float(np.mean(vols[method])),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def performance_comparison(
    prices: dict,
    params_by_regime: dict,
    regimes: pd.Series,
    config: dict,
    rf: pd.Series | float = 0.0,
    trade_params: pd.DataFrame | None = None,
    period: tuple | None = None,
    include_assets: bool = True,
) -> pd.DataFrame:
    """Desempeño de Risk Parity, pesos iguales y (opcional) cada activo solo, con las mismas señales.

    Los dos portafolios usan `sleeve_weights`; cada activo solo corre con C = Equity (columna de
    unos), como la corrida base de P1. Las métricas son las de `compute_metrics` sobre el tramo
    `period` de la curva y las operaciones que abren dentro de él.

    Parameters
    ----------
    rf : pd.Series or float
        Tasa libre de riesgo diaria, como la espera `compute_metrics`.
    trade_params : pd.DataFrame, optional
        Si falta, se usa `config["base_params"]` constante.
    period : (inicio, fin), optional
        Bloque sobre el que se miden las métricas; el backtest corre completo.

    Returns
    -------
    pd.DataFrame
        Una fila por estrategia ("risk_parity", "equal" y los tickers) con las columnas de
        `compute_metrics`.
    """
    signals = generate_signals(prices, params_by_regime, regimes, config)
    dates = next(iter(prices.values())).index
    if trade_params is None:
        trade_params = pd.DataFrame(
            {k: config["base_params"][k] for k in _TRADE_PARAM_KEYS}, index=dates
        )
    start, end = (dates[0], dates[-1]) if period is None else (pd.Timestamp(period[0]), pd.Timestamp(period[1]))

    results = {
        method: run_backtest(
            prices, signals, sleeve_weights(prices, signals, regimes, config, method=method),
            trade_params, config,
        )
        for method in ("risk_parity", "equal")
    }
    if include_assets:
        for ticker in prices:
            results[ticker] = run_backtest(
                {ticker: prices[ticker]},
                {name: panel[[ticker]] for name, panel in signals.items()},
                pd.DataFrame({ticker: 1.0}, index=dates),
                trade_params,
                config,
            )

    rows = {}
    for name, result in results.items():
        trades = result.trades[result.trades["entry_date"].between(start, end)]
        rows[name] = compute_metrics(
            result.equity.loc[start:end], trades, rf, config["periods_per_year"]
        )
    return pd.DataFrame.from_dict(rows, orient="index")


def portfolio_results(
    prices: dict,
    params_by_regime: dict,
    regimes: pd.Series,
    config: dict,
    rf: pd.Series | float,
    periods: dict[str, tuple],
    trade_params: pd.DataFrame | None = None,
) -> dict:
    """Todos los resultados de P4 en un dict, listo para `results/portafolio.pkl`.

    Parameters
    ----------
    periods : dict[str, (inicio, fin)]
        Bloques a medir, p. ej. {"train": ...}. Cada bloque que se incluye se mira: el que decide
        (validation) se pasa una sola vez, con los θ finales (SPEC punto 1).

    Returns
    -------
    dict
        "weight_stability" y "risk_contributions" (solo en el primer bloque de `periods`, que es
        donde se reportan), y por bloque: "performance" (Risk Parity, pesos iguales y activos) y
        "sweep" (frecuencia × banda de `config["rebalance_frequencies"]` y `["rebalance_bands"]`).
    """
    if not periods:
        raise ValueError("periods no puede estar vacío")
    first = next(iter(periods.values()))
    return {
        "weight_stability": weight_stability(prices, config, period=first),
        "risk_contributions": risk_contribution_comparison(prices, config, period=first),
        "performance": {
            name: performance_comparison(
                prices, params_by_regime, regimes, config, rf, trade_params, period
            )
            for name, period in periods.items()
        },
        "sweep": {
            name: rebalance_sweep(
                prices, params_by_regime, regimes, config,
                config["rebalance_bands"], config["rebalance_frequencies"], trade_params, period,
            )
            for name, period in periods.items()
        },
    }


def risk_contribution_plot_frame(contributions: pd.DataFrame) -> pd.DataFrame:
    """Prepara la salida de `risk_contribution_comparison` para `plot_risk_contributions`.

    Devuelve ticker × esquema con las partes del riesgo (fracción), quitando `spread` y `ann_vol`
    y con los nombres de esquema que se leen en la figura.
    """
    shares = contributions.drop(columns=["spread", "ann_vol"]).T
    return shares.rename(columns={"risk_parity": "Risk Parity", "equal": "Pesos iguales"})


def sweep_plot_frame(sweep: pd.DataFrame, period: tuple, config: dict) -> pd.DataFrame:
    """Prepara la salida de `rebalance_sweep` para `plot_rebalance_sweep` (todo anualizado).

    Convierte los retornos acumulados del bloque `period` a retornos anuales compuestos y define el
    costo como la diferencia entre el retorno bruto y el neto anualizados, para que las tres
    series estén en la misma unidad. El índice es "frecuencia · banda".
    """
    years = max((pd.Timestamp(period[1]) - pd.Timestamp(period[0])).days / 365.25, 1e-9)
    gross = (1.0 + sweep["gross_return"]) ** (1.0 / years) - 1.0
    net = (1.0 + sweep["net_return"]) ** (1.0 / years) - 1.0
    frame = pd.DataFrame(
        {"gross_return": gross, "cost": gross - net, "net_return": net, "turnover": sweep["turnover"]}
    )
    frame.index = [f"{f} · {b:g}" for f, b in zip(sweep["frequency"], sweep["band"])]
    return frame