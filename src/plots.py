"""Figuras del proyecto: una función por figura, cada una devuelve `matplotlib.figure.Figure` (P3).

Reciben resultados ya calculados; aquí no hay cálculos de estrategia. Cada figura lleva título, ejes
etiquetados y leyenda (o barra de color en los mapas de calor), como pide el lab (sección 3.6).

Las figuras se crean con `Figure` y no con `pyplot`, así que no dependen de un estado global y se
guardan con `fig.savefig(...)` desde `main.py`.

Colores: paleta categórica validada para daltonismo (azul, naranja, aqua, …), asignada en orden fijo
y siempre por entidad: tendencia es azul, crisis naranja y reversión aqua en todas las figuras. Las
escalas divergentes van de rojo (negativo) a gris (cero) a azul (positivo).
"""

import math

import matplotlib.dates as mdates
import numpy as np
import optuna
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
REGIME_COLORS = {"tendencia": "#2a78d6", "crisis": "#eb6834", "reversion": "#1baf7a"}
REGIME_LABELS = {"tendencia": "Tendencia", "reversion": "Reversión", "crisis": "Crisis"}
FEATURE_LABELS = {
    "volatility": "Volatilidad anualizada",
    "efficiency": "Razón de eficiencia",
    "autocorr": "Autocorrelación de lag 1",
}
BLOCK_SHADES = ["#f0efec", "#e1e0d9", "#c3c2b7"]
INK, INK_SECONDARY, INK_MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("divergente", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "secuencial", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _style(ax) -> None:
    """Ejes y cuadrícula discretos para que los datos dominen."""
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _legend(ax, **kwargs) -> None:
    """Leyenda sin marco, con texto en tinta (nunca del color de la serie)."""
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, **kwargs)


def _figure_legend(fig, handles: list) -> None:
    """Leyenda debajo de los ejes, para que no tape curvas que recorren todo el periodo."""
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 6),
               frameon=False, fontsize=9, labelcolor=INK_SECONDARY)


def _series_colors(n: int) -> list[str]:
    """Colores categóricos en orden fijo; más de 8 series no se distinguen y se rechazan."""
    if n > len(SERIES_COLORS):
        raise ValueError(f"{n} series: agrúpalas o usa varias figuras (máximo {len(SERIES_COLORS)}).")
    return SERIES_COLORS[:n]


def _regime_spans(labels: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Rachas consecutivas de un mismo régimen como (inicio, fin, nombre); NaN no se sombrea.

    Cada racha termina donde empieza la fecha siguiente, para que el sombreado no deje huecos.
    """
    run_id = (labels != labels.shift()).cumsum()
    dates = labels.index
    spans = []
    for _, run in labels.groupby(run_id):
        name = run.iloc[0]
        if not isinstance(name, str):
            continue
        end_pos = dates.get_loc(run.index[-1])
        end = dates[end_pos + 1] if end_pos + 1 < len(dates) else run.index[-1]
        spans.append((run.index[0], end, name))
    return spans


def _shade_regimes(ax, labels: pd.Series) -> list[Patch]:
    """Sombrea el fondo según el régimen y devuelve las entradas de leyenda de los presentes."""
    present = []
    for start, end, name in _regime_spans(labels):
        ax.axvspan(start, end, color=REGIME_COLORS[name], alpha=0.22, linewidth=0)
        if name not in present:
            present.append(name)
    order = [name for name in REGIME_COLORS if name in present]
    return [Patch(color=REGIME_COLORS[n], alpha=0.5, label=REGIME_LABELS[n]) for n in order]


def _heatmap(fig, ax, data: pd.DataFrame, vmax: float, annotate_fmt: str | None):
    """Mapa de calor divergente centrado en cero; devuelve la imagen para la barra de color."""
    image = ax.imshow(data.to_numpy(dtype=float), cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                      aspect="auto", interpolation="nearest")
    ax.set_xticks(range(data.shape[1]), [str(c) for c in data.columns])
    ax.set_yticks(range(data.shape[0]), [str(i) for i in data.index])
    if annotate_fmt is not None:
        for (row, col), value in np.ndenumerate(data.to_numpy(dtype=float)):
            if np.isfinite(value):
                ax.text(col, row, format(value, annotate_fmt), ha="center", va="center",
                        fontsize=8, color=INK)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _finite_trials(study: optuna.Study) -> tuple[list[optuna.trial.FrozenTrial], int]:
    """Pruebas completas con valor finito y el número de infactibles (−inf u otros no finitos)."""
    complete = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    finite = [t for t in complete if math.isfinite(t.value)]
    return finite, len(complete) - len(finite)


def plot_equity(curves: dict[str, pd.Series], blocks: dict[str, tuple]) -> Figure:
    """Valor del portafolio en train y test, con benchmark; `blocks` sombrea cada bloque.

    Parameters
    ----------
    curves : dict[str, pd.Series]
        Nombre de la curva → equity en USD por fecha (por ejemplo, "Risk Parity", "Pesos iguales",
        "Buy & hold").
    blocks : dict[str, tuple]
        Salida de `block_dates(config)`: nombre → (inicio, fin).
    """
    fig = Figure(figsize=(10, 5), layout="constrained")
    ax = fig.add_subplot()
    for (name, (start, end)), shade in zip(blocks.items(), BLOCK_SHADES):
        ax.axvspan(start, end, color=shade, alpha=0.7, linewidth=0, label=f"Bloque {name}")
    for (name, curve), color in zip(curves.items(), _series_colors(len(curves))):
        ax.plot(curve.index, curve.to_numpy(), color=color, linewidth=1.5, label=name)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set(title="Valor del portafolio por bloque", xlabel="Fecha", ylabel="Valor (USD)")
    _style(ax)
    _figure_legend(fig, ax.get_legend_handles_labels()[0])
    return fig


def plot_strategy_vs_benchmark(curves: dict[str, pd.Series], blocks: dict[str, tuple]) -> Figure:
    """Estrategia contra buy & hold: crecimiento de 100 USD (escala log) y drawdown.

    Parameters
    ----------
    curves : dict[str, pd.Series]
        Nombre → equity por fecha; todas se rebasan a 100 en su primera fecha. Una curva cuyo
        nombre contiene "ex post" se dibuja punteada (referencia no operable).
    blocks : dict[str, tuple]
        Bloques a sombrear (salida de `block_dates`, recortada a las fechas mostradas).
    """
    fig = Figure(figsize=(11, 7), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for (name, (start, end)), shade in zip(blocks.items(), BLOCK_SHADES):
        for ax in (top, bottom):
            ax.axvspan(start, end, color=shade, alpha=0.7, linewidth=0,
                       label=f"Bloque {name}" if ax is top else None)
    for (name, curve), color in zip(curves.items(), _series_colors(len(curves))):
        style = "--" if "ex post" in name else "-"
        growth = 100 * curve / curve.iloc[0]
        top.plot(growth.index, growth.to_numpy(), color=color, linewidth=1.5, linestyle=style,
                 label=name)
        drawdown = (curve / curve.cummax() - 1) * 100
        bottom.plot(drawdown.index, drawdown.to_numpy(), color=color, linewidth=1.2,
                    linestyle=style)
    top.set_yscale("log")
    for formatter in (top.yaxis.set_major_formatter, top.yaxis.set_minor_formatter):
        formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    top.set(title="Crecimiento de 100 USD", xlabel="Fecha", ylabel="Valor (base 100, escala log)")
    bottom.axhline(0, color=AXIS, linewidth=1)
    bottom.set(title="Drawdown", xlabel="Fecha", ylabel="Drawdown (%)")
    for ax in (top, bottom):
        _style(ax)
    _figure_legend(fig, top.get_legend_handles_labels()[0])
    fig.suptitle("Estrategia contra buy & hold", color=INK)
    return fig


def plot_drawdown(drawdowns: dict[str, pd.Series]) -> Figure:
    """Curvas de drawdown.

    Parameters
    ----------
    drawdowns : dict[str, pd.Series]
        Nombre → drawdown por fecha como fracción ≤ 0 (salida de `drawdown_series`).
    """
    fig = Figure(figsize=(10, 4), layout="constrained")
    ax = fig.add_subplot()
    for (name, dd), color in zip(drawdowns.items(), _series_colors(len(drawdowns))):
        ax.plot(dd.index, dd.to_numpy() * 100, color=color, linewidth=1.5, label=name)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set(title="Drawdown", xlabel="Fecha", ylabel="Drawdown (%)")
    _style(ax)
    _legend(ax, loc="lower left")
    return fig


def plot_returns_table(tables: dict[str, pd.DataFrame]) -> Figure:
    """Retornos mensuales, trimestrales y anuales (salida de `returns_table`).

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        "mensual", "trimestral" y "anual". Cada tabla se dibuja tal cual: filas y columnas del
        DataFrame (por ejemplo, año × mes), con retornos como fracción. Todas comparten la escala
        de color, simétrica alrededor de cero.
    """
    heights = [max(1.6, 0.35 * len(t) + 0.9) for t in tables.values()]
    fig = Figure(figsize=(11, sum(heights) + 0.6), layout="constrained")
    axes = fig.subplots(len(tables), 1, gridspec_kw={"height_ratios": heights}, squeeze=False)[:, 0]
    vmax = max(np.nanmax(np.abs(t.to_numpy(dtype=float))) for t in tables.values()) * 100
    for ax, (name, table) in zip(axes, tables.items()):
        image = _heatmap(fig, ax, table.astype(float) * 100, vmax, ".1f")
        ax.set(
            title=f"Retornos: {name} (%)",
            xlabel=table.columns.name or "Periodo",
            ylabel=table.index.name or "Año",
        )
        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)
    fig.colorbar(image, ax=list(axes), label="Retorno (%)", shrink=0.8)
    fig.suptitle("Tabla de retornos", color=INK)
    return fig


def plot_sensitivity(sensitivity: pd.DataFrame) -> Figure:
    """Cambio en Calmar al variar cada parámetro ±20% (salida de `sensitivity`).

    Un panel por régimen; en cada uno, una barra por parámetro y factor (×0.8 y ×1.2).

    Parameters
    ----------
    sensitivity : pd.DataFrame
        Salida de `sensitivity`: una fila por régimen, parámetro y factor, con las columnas
        `regime`, `parameter`, `factor` y `delta_calmar` (Calmar probado menos Calmar base).
    """
    table = sensitivity.pivot_table(
        index=["regime", "parameter"], columns="factor", values="delta_calmar", aggfunc="first"
    )
    regimes = list(dict.fromkeys(sensitivity["regime"]))
    factors = list(table.columns)
    n_params = max(len(table.loc[r]) for r in regimes)
    fig = Figure(figsize=(4.6 * len(regimes), 0.42 * n_params + 2.2), layout="constrained")
    axes = fig.subplots(1, len(regimes), squeeze=False, sharex=True)[0]
    height = 0.8 / len(factors)
    colors = _series_colors(len(factors))
    for ax, regime in zip(axes, regimes):
        panel = table.loc[regime]
        positions = np.arange(len(panel))
        for k, (factor, color) in enumerate(zip(factors, colors)):
            ax.barh(positions + (k - (len(factors) - 1) / 2) * height, panel[factor].to_numpy(),
                    height=height * 0.9, color=color, label=f"Parámetro × {factor:g}")
        ax.set_yticks(positions, [str(p) for p in panel.index])
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set(title=REGIME_LABELS.get(regime, str(regime)), xlabel="Δ Calmar",
               ylabel="Parámetro")
        _style(ax)
        ax.grid(False, axis="y")
    _figure_legend(fig, axes[0].get_legend_handles_labels()[0])
    fig.suptitle("Sensibilidad de Calmar a ±20% en cada parámetro", color=INK)
    return fig


def plot_cost_curve(
    cost_curve: pd.DataFrame, base_cost_bps: float, column: str = "net_return"
) -> Figure:
    """Retorno neto contra costo de ida y vuelta, marcando el costo base (salida de `cost_sweep`).

    Parameters
    ----------
    cost_curve : pd.DataFrame
        Índice = costo de ida y vuelta en bps; `column` = retorno neto anualizado como fracción.
    base_cost_bps : float
        Costo de ida y vuelta del caso base (29 bps, SPEC punto 6).
    column : str
        Columna a graficar.
    """
    fig = Figure(figsize=(9, 4.5), layout="constrained")
    ax = fig.add_subplot()
    ax.plot(cost_curve.index, cost_curve[column].to_numpy() * 100, color=SERIES_COLORS[0],
            linewidth=1.5, marker="o", markersize=4, label="Retorno neto anualizado")
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(base_cost_bps, color=INK_MUTED, linewidth=1.2, linestyle="--",
               label=f"Costo base ({base_cost_bps:g} bps)")
    ax.set(title="Retorno neto contra costo de transacción",
           xlabel="Costo de ida y vuelta (bps)", ylabel="Retorno neto anualizado (%)")
    _style(ax)
    _legend(ax, loc="best")
    return fig


def plot_regime_timeline(price: pd.Series, filtered: pd.Series, viterbi: pd.Series) -> Figure:
    """Precio coloreado por régimen filtrado, con la versión Viterbi al lado.

    Parameters
    ----------
    price : pd.Series
        Serie de mercado (por ejemplo, `market_index` de `regimes.py`).
    filtered : pd.Series
        Etiqueta filtrada (operable), salida de `predict_regimes` o `label_regimes`.
    viterbi : pd.Series
        Ruta de Viterbi, salida de `viterbi_path`. Usa el futuro: solo se muestra para comparar.
    """
    fig = Figure(figsize=(11, 6.5), layout="constrained")
    axes = fig.subplots(2, 1, sharex=True)
    panels = [
        (filtered, "Etiqueta filtrada: solo datos hasta t (la que se opera)"),
        (viterbi, "Ruta de Viterbi: usa toda la muestra (solo para comparar)"),
    ]
    for ax, (labels, title) in zip(axes, panels):
        handles = _shade_regimes(ax, labels.reindex(price.index))
        (line,) = ax.plot(price.index, price.to_numpy(), color=INK, linewidth=1,
                          label="Índice de mercado")
        ax.set(title=title, xlabel="Fecha", ylabel="Índice (base 1)")
        _style(ax)
        ax.grid(False, axis="x")
        _legend(ax, handles=[line, *handles], loc="upper left", ncols=4)
    fig.suptitle("Régimen de mercado: etiqueta filtrada contra Viterbi", color=INK)
    return fig


def plot_regime_features(features: pd.DataFrame, labels: pd.Series) -> Figure:
    """Distribución de las variables de régimen por régimen.

    Un histograma por variable y régimen; la línea punteada marca la media de cada régimen (su
    centroide en esa variable), que es lo que justifica el nombre (SPEC_portafolio, "Nombres").

    Parameters
    ----------
    features : pd.DataFrame
        Salida de `regime_features`.
    labels : pd.Series
        Etiqueta por fecha.
    """
    data = features.assign(regime=labels).dropna()
    columns = list(features.columns)
    fig = Figure(figsize=(4.2 * len(columns), 4.2), layout="constrained")
    axes = fig.subplots(1, len(columns), squeeze=False)[0]
    regimes_present = [name for name in REGIME_COLORS if name in set(data["regime"])]
    for ax, column in zip(axes, columns):
        bins = np.histogram_bin_edges(data[column], bins=30)
        for name in regimes_present:
            values = data.loc[data["regime"] == name, column]
            color = REGIME_COLORS[name]
            ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.18,
                    color=color)
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5, color=color,
                    label=f"{REGIME_LABELS[name]} (n = {len(values)})")
            ax.axvline(values.mean(), color=color, linewidth=1.2, linestyle="--")
        label = FEATURE_LABELS.get(column, column)
        ax.set(title=label, xlabel=label, ylabel="Densidad")
        _style(ax)
    # Una sola leyenda fuera de los paneles: dentro tapaba las barras en alguno de ellos.
    _figure_legend(fig, axes[0].get_legend_handles_labels()[0])
    fig.suptitle("Distribución de las variables por régimen (línea punteada: media)", color=INK)
    return fig


def plot_equity_regimes(equity: pd.Series, regimes: pd.Series) -> Figure:
    """Curva de equity con los regímenes superpuestos.

    Parameters
    ----------
    equity : pd.Series
        Equity en USD por fecha.
    regimes : pd.Series
        Etiqueta de régimen vigente en cada fecha.
    """
    fig = Figure(figsize=(10, 5), layout="constrained")
    ax = fig.add_subplot()
    handles = _shade_regimes(ax, regimes.reindex(equity.index))
    (line,) = ax.plot(equity.index, equity.to_numpy(), color=INK, linewidth=1.5, label="Equity")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set(title="Equity con el régimen de mercado vigente", xlabel="Fecha", ylabel="Valor (USD)")
    _style(ax)
    ax.grid(False, axis="x")
    _figure_legend(fig, [line, *handles])
    return fig


def plot_risk_contributions(contributions: pd.DataFrame) -> Figure:
    """Contribuciones al riesgo por activo: Risk Parity contra pesos iguales.

    Parameters
    ----------
    contributions : pd.DataFrame
        Índice = ticker; una columna por esquema (por ejemplo, "Risk Parity" y "Pesos iguales") con
        RC_i / σ_p como fracción (cada columna suma 1).
    """
    fig = Figure(figsize=(10, 4.8), layout="constrained")
    ax = fig.add_subplot()
    n = contributions.shape[1]
    width = 0.8 / n
    positions = np.arange(len(contributions))
    for k, (column, color) in enumerate(zip(contributions.columns, _series_colors(n))):
        ax.bar(positions + (k - (n - 1) / 2) * width, contributions[column].to_numpy() * 100,
               width=width * 0.9, color=color, label=str(column))
    ax.axhline(100 / len(contributions), color=INK_MUTED, linewidth=1.2, linestyle="--",
               label=f"Contribución igual (1/{len(contributions)})")
    ax.set_xticks(positions, [str(t) for t in contributions.index])
    ax.set(title="Contribución al riesgo por activo", xlabel="Activo",
           ylabel="Contribución al riesgo (% de σ_p)")
    _style(ax)
    ax.grid(False, axis="x")
    _legend(ax, loc="upper right")
    return fig


def plot_signal_heatmap(strength: pd.DataFrame) -> Figure:
    """Mapa de calor fecha × activo de la fuerza de señal s_i.

    Parameters
    ----------
    strength : pd.DataFrame
        Panel fecha × ticker con s ∈ {−1, −2/3, 0, 2/3, 1} (`generate_signals(...)["strength"]`).
    """
    fig = Figure(figsize=(12, 0.45 * strength.shape[1] + 1.8), layout="constrained")
    ax = fig.add_subplot()
    start, end = mdates.date2num(strength.index[0]), mdates.date2num(strength.index[-1])
    image = ax.imshow(strength.T.to_numpy(dtype=float), cmap=DIVERGING, vmin=-1, vmax=1,
                      aspect="auto", interpolation="nearest",
                      extent=(start, end, strength.shape[1] - 0.5, -0.5))
    ax.xaxis_date()
    ax.set_yticks(range(strength.shape[1]), [str(c) for c in strength.columns])
    ax.set(title="Fuerza de la señal por activo", xlabel="Fecha", ylabel="Activo")
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    fig.colorbar(image, ax=ax, label="Fuerza s_i (−1 corto, 0 sin señal, +1 largo)")
    return fig


def plot_corr_by_regime(corr_by_regime: dict[str, pd.DataFrame]) -> Figure:
    """Matriz de correlación de los activos en cada régimen.

    Parameters
    ----------
    corr_by_regime : dict[str, pd.DataFrame]
        Régimen → matriz de correlación ticker × ticker de los retornos en las fechas de ese
        régimen.
    """
    names = [n for n in REGIME_COLORS if n in corr_by_regime] + [
        n for n in corr_by_regime if n not in REGIME_COLORS
    ]
    fig = Figure(figsize=(5 * len(names) + 1, 5), layout="constrained")
    axes = fig.subplots(1, len(names), squeeze=False)[0]
    for ax, name in zip(axes, names):
        image = _heatmap(fig, ax, corr_by_regime[name], 1.0, ".2f")
        ax.tick_params(axis="x", labelrotation=45)
        ax.set(title=REGIME_LABELS.get(name, name), xlabel="Activo", ylabel="Activo")
        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)
    fig.colorbar(image, ax=list(axes), label="Correlación", shrink=0.8)
    fig.suptitle("Correlación entre activos por régimen", color=INK)
    return fig


def plot_rebalance_sweep(sweep: pd.DataFrame) -> Figure:
    """Retorno bruto, costo total, retorno neto y turnover contra frecuencia o banda δ.

    Una sola figura con dos paneles que comparten el eje x: arriba los tres retornos (misma escala)
    y abajo el turnover, en vez de un segundo eje y.

    Parameters
    ----------
    sweep : pd.DataFrame
        Índice = configuración (frecuencia, δ o su combinación; un MultiIndex se une con " · ").
        Columnas `gross_return`, `cost` y `net_return` (anualizados, fracción) y `turnover` (anual,
        fracción).
    """
    labels = [" · ".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in sweep.index]
    positions = np.arange(len(sweep))
    fig = Figure(figsize=(max(8, 0.6 * len(sweep) + 3), 7), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    series = [("gross_return", "Retorno bruto"), ("cost", "Costo total"),
              ("net_return", "Retorno neto")]
    for (column, label), color in zip(series, _series_colors(len(series))):
        top.plot(positions, sweep[column].to_numpy() * 100, color=color, linewidth=1.5,
                 marker="o", markersize=4, label=label)
    top.axhline(0, color=AXIS, linewidth=1)
    top.set(title="Retornos y costo anualizados", xlabel="Configuración de rebalanceo",
            ylabel="% anual")
    bottom.bar(positions, sweep["turnover"].to_numpy() * 100, width=0.6,
               color=SERIES_COLORS[0], label="Turnover realizado")
    bottom.set(title="Turnover", xlabel="Configuración de rebalanceo", ylabel="Turnover anual (%)")
    bottom.set_xticks(positions, labels, rotation=45, ha="right")
    for ax in (top, bottom):
        _style(ax)
        _legend(ax, loc="best")
    bottom.grid(False, axis="x")
    fig.suptitle("Barrido de rebalanceo", color=INK)
    return fig


def plot_surface_3d(grid: pd.DataFrame, x: str, y: str, z: str = "calmar") -> Figure:
    """Superficie 3D del Calmar sobre las dos dimensiones más influyentes del random search.

    Las configuraciones infactibles (z no finito, como −inf) no entran a la superficie; se indica
    cuántas son en la leyenda. Solo existe una superficie en 2 dimensiones: el resto de θ queda fijo
    en el mejor valor del random search (P2).

    Parameters
    ----------
    grid : pd.DataFrame
        Una fila por configuración evaluada, con columnas `x`, `y` y `z`.
    """
    finite = grid[np.isfinite(grid[z].to_numpy(dtype=float))]
    n_infeasible = len(grid) - len(finite)
    fig = Figure(figsize=(9, 7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(finite[x], finite[y], finite[z], cmap=SEQUENTIAL, alpha=0.85, linewidth=0.2,
                    edgecolor="white")
    ax.scatter(finite[x], finite[y], finite[z], color=INK, s=8, depthshade=False)
    ax.set(title=f"Superficie de {z} sobre {x} y {y}", xlabel=x, ylabel=y, zlabel=z)
    handles = [
        Patch(color=SERIES_COLORS[0], alpha=0.85, label="Superficie interpolada"),
        Line2D([], [], color=INK, marker="o", linestyle="", markersize=4,
               label="Configuraciones evaluadas"),
    ]
    if n_infeasible:
        handles.append(Line2D([], [], linestyle="", label=f"Infactibles omitidas: {n_infeasible}"))
    _legend(ax, handles=handles, loc="upper left")
    return fig


def plot_optimization_history(study: optuna.Study) -> Figure:
    """Historia de optimización del estudio TPE.

    Puntos: valor de cada prueba factible. Línea: mejor valor acumulado. Las pruebas infactibles
    (objetivo −inf por no cumplir la actividad mínima) se cuentan en la leyenda.
    """
    finite, n_infeasible = _finite_trials(study)
    numbers = np.array([t.number for t in finite])
    values = np.array([t.value for t in finite])
    best = (np.maximum if study.direction == optuna.study.StudyDirection.MAXIMIZE
            else np.minimum).accumulate(values)
    fig = Figure(figsize=(10, 4.8), layout="constrained")
    ax = fig.add_subplot()
    ax.scatter(numbers, values, color=SERIES_COLORS[0], s=14, alpha=0.7, label="Prueba factible")
    ax.step(numbers, best, where="post", color=SERIES_COLORS[1], linewidth=1.5,
            label="Mejor hasta esa prueba")
    if n_infeasible:
        ax.plot([], [], linestyle="", label=f"Infactibles (−inf): {n_infeasible}")
    ax.set(title="Historia de optimización", xlabel="Número de prueba", ylabel="Valor objetivo")
    _style(ax)
    _legend(ax, loc="lower right")
    return fig


def plot_param_importance(study: optuna.Study, seed: int) -> Figure:
    """Importancia de parámetros del estudio TPE.

    fANOVA sobre las pruebas factibles: con las −inf adentro, Optuna no falla pero la importancia sale
    distorsionada. fANOVA es aleatorio, así que recibe la semilla del proyecto.
    """
    finite, n_infeasible = _finite_trials(study)
    feasible = optuna.create_study(directions=study.directions)
    feasible.add_trials(finite)
    importances = optuna.importance.get_param_importances(
        feasible, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed)
    )
    ordered = sorted(importances.items(), key=lambda kv: kv[1])
    fig = Figure(figsize=(8, 0.45 * len(ordered) + 1.8), layout="constrained")
    ax = fig.add_subplot()
    ax.barh([k for k, _ in ordered], [v for _, v in ordered], color=SERIES_COLORS[0], height=0.6,
            label=f"fANOVA ({len(finite)} pruebas factibles, {n_infeasible} infactibles omitidas)")
    ax.set(title="Importancia de los parámetros", xlabel="Importancia relativa",
           ylabel="Parámetro")
    _style(ax)
    ax.grid(False, axis="y")
    _legend(ax, loc="lower right")
    return fig


def plot_slices(study: optuna.Study) -> Figure:
    """Slice plots del estudio TPE: valor objetivo contra cada parámetro, color = número de prueba."""
    finite, n_infeasible = _finite_trials(study)
    params = sorted({p for t in finite for p in t.params})
    ncols = min(3, len(params))
    nrows = math.ceil(len(params) / ncols)
    fig = Figure(figsize=(4.3 * ncols, 3.4 * nrows + 0.6), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
    numbers = [t.number for t in finite]
    for ax, param in zip(axes, params):
        xs = [t.params.get(param, np.nan) for t in finite]
        points = ax.scatter(xs, [t.value for t in finite], c=numbers, cmap=SEQUENTIAL, s=12)
        ax.set(title=param, xlabel=param, ylabel="Valor objetivo")
        _style(ax)
    for ax in axes[len(params):]:
        fig.delaxes(ax)
    handles = [Line2D([], [], color=SEQUENTIAL(0.7), marker="o", linestyle="", markersize=4,
                      label="Prueba factible")]
    if n_infeasible:
        handles.append(Line2D([], [], linestyle="", label=f"Infactibles (−inf): {n_infeasible}"))
    # Leyenda fuera de los paneles para no tapar puntos.
    _figure_legend(fig, handles)
    fig.colorbar(points, ax=list(axes[: len(params)]), label="Número de prueba", shrink=0.8)
    fig.suptitle("Valor objetivo contra cada parámetro", color=INK)
    return fig
