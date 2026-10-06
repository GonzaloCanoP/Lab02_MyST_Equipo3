# Lab 02 — Estrategias de Trading con Análisis Técnico · Equipo 3 · Nivel C

Repositorio `Lab02_MyST_Equipo3` · Microestructuras y Sistemas de Trading (IT1731B) · ITESO, Otoño 2026

CLAUDE.md principal del proyecto. El repositorio es independiente del repo del curso, así que aquí
viven también las convenciones globales. Las reglas de la ESTRATEGIA están en `SPEC.md` y
`SPEC_portafolio.md`; este archivo define CÓMO se implementan. Los archivos `P*.md` heredan todo lo de
aquí y no lo repiten.

## 0. Orden de lectura obligatorio

1. Este archivo.
2. `instrucciones/SPEC.md`: estrategia por activo (entrada, salida, sizing, costos, convenciones).
3. `instrucciones/SPEC_portafolio.md`: régimen, Risk Parity, agregación de señales y rebalanceo.
4. El archivo de la parte en la que se trabaja (`instrucciones/P1_*.md`, etc.).

Si un SPEC contradice este archivo, o si una decisión no está en ninguno: detente y pregunta.
Nunca resuelvas una decisión abierta en silencio.

## 1. Equipo, partes y ramas

| Parte | Integrante | Rama | Archivos propios |
|---|---|---|---|
| P0 — Esqueleto | *[PENDIENTE]* | `main` (única excepción) | Estructura y stubs |
| P1 — Datos, motor y auditoría | *[PENDIENTE]* | `p1-<integrante>` | `src/data.py`, `src/backtest.py`, `tests/test_backtest.py`, `tests/test_pipeline.py`, `tests/conftest.py`, `notebooks/analisis_P1.ipynb`, `main.py`, `README.md`, `requirements.txt`, `.gitignore` |
| P2 — Señales, métricas y optimización | *[PENDIENTE]* | `p2-<integrante>` | `src/signals.py`, `src/metrics.py`, `src/optimize.py`, `tests/test_signals.py`, `tests/test_metrics.py` |
| P3 — Régimen y visualización | *[PENDIENTE]* | `p3-<integrante>` | `src/regimes.py`, `src/plots.py`, `tests/test_regimes.py`, sección "Régimen" de `SPEC_portafolio.md` |
| P4 — Portafolio y Risk Parity | *[PENDIENTE]* | `p4-<integrante>` | `src/portfolio.py`, `tests/test_portfolio.py`, `notebooks/analysis.ipynb`, sección "Portafolio" de `SPEC_portafolio.md` |

Integrantes: Gonzalo Cano Padilla, Juan Manuel Espinosa Cárdenas, Jerónimo Rojas Alvarado y Raúl
Zanatta Casas. La asignación de partes está PENDIENTE (sección 14).

Cada integrante aporta 2 de los 8 activos y escribe en el reporte la descripción de la estrategia de
sus 2 activos.

## 2. Estructura del repositorio

```
Lab02_MyST_Equipo3/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── .claude/
│   └── CLAUDE.md               # una línea: @../instrucciones/CLAUDE.md
├── instrucciones/
│   ├── CLAUDE.md               # este archivo
│   ├── SPEC.md
│   ├── SPEC_portafolio.md
│   └── P0_esqueleto.md · P1_datos_motor.md · P2_senales_optimizacion.md
│       P3_regimen_visualizacion.md · P4_portafolio.md
├── data/                       # datos crudos congelados del portafolio (8 tickers + ^IRX)
├── src/
│   ├── __init__.py
│   ├── data.py  backtest.py  signals.py  metrics.py
│   ├── optimize.py  regimes.py  portfolio.py  plots.py
├── tests/
│   ├── conftest.py             # fixtures sintéticos compartidos
│   └── test_*.py
├── notebooks/
│   ├── analysis.ipynb          # análisis del portafolio (P4)
│   └── analisis_P1.ipynb       # datos, motor y pruebas de P1
├── results/                    # salidas de main.py — en .gitignore
└── docs/
    ├── figuras/                # PNG generados por main.py
    ├── reporte.pdf
    └── presentacion.pdf
```

No se crean archivos sueltos en la raíz distintos a los listados, ni módulos nuevos en `src/` sin
acuerdo del equipo.

## 3. Datos y política de red

- Los datos del portafolio viven en `data/`: un archivo por ticker y uno para `^IRX` (T-Bill a 13
  semanas, tasa libre de riesgo). Todo el proyecto los lee con `load_prices` y `load_risk_free`;
  ningún otro módulo abre archivos de `data/` directamente.
- La única función con red es `download_prices` en `src/data.py`. Existe para que el proyecto sea
  reproducible desde cero; `main.py` no la llama si los archivos ya existen.
- `main.py`, `tests/` y `notebooks/` NUNCA usan red.
- Columnas en minúsculas: `open, high, low, close, volume`. `DatetimeIndex` diario, común a los 8
  activos (intersección de fechas; las fechas descartadas se reportan en la auditoría).
- Prohibido rellenar (`ffill`, `bfill`, interpolar) en silencio. Todo hueco se reporta en `audit_prices`.
- Formato: un CSV por activo, `data/<TICKER>.csv` (por ejemplo `data/AAPL.csv`), con columna `date`
  como índice y `open, high, low, close, volume`. `^IRX` se guarda como `data/IRX.csv` (sin `^`).
- Tickers: AAPL, MSFT, META, AMD, XOM, SMH, GLD, COPX (SPEC.md, punto 1).

## 4. Causalidad — reglas CRÍTICAS

Una violación invalida el backtest completo.

- Todo lo que se decide en t usa información hasta el cierre de t.
- El desplazamiento t → t+1 vive en UN solo lugar: `run_backtest`. Ningún otro módulo hace `shift`
  para simular ejecución.
- Prohibido: `rolling(center=True)`, `shift(-k)`, `bfill`, normalizar con estadísticas de la muestra
  completa, ajustar cualquier modelo con datos posteriores a su ventana.
- Régimen: la etiqueta que se opera es la FILTRADA (algoritmo forward). Viterbi y las probabilidades
  suavizadas (forward-backward, como `predict_proba` de hmmlearn) están prohibidos para operar; Viterbi
  solo aparece en la figura comparativa de P3.
- Covarianza de Risk Parity en t: solo retornos hasta t.
- Walk-forward: la ventana de prueba nunca participa en la optimización. Los indicadores pueden usar
  historia previa como calentamiento.
- Prueba de truncamiento: todo valor en t calculado sobre `df.iloc[:t+1]` debe ser idéntico al
  calculado sobre la serie completa. Aplica a indicadores, señales, régimen, pesos y al pipeline
  completo.

## 5. Convenciones de implementación del motor

Las reglas de trading están en SPEC.md, puntos 4 a 7. Aquí solo van detalles de implementación:

- `run_backtest` es una función pura: no lee archivos, no usa estado global y no modifica sus
  argumentos. Mismas entradas → mismo resultado.
- Loop explícito por fecha con estado `cash`, `positions` (acciones con signo por ticker) y
  `equity = cash + Σ shares · close`. Dentro del loop se usan arreglos de numpy, no `.loc` por fila.
- Los parámetros de la operación (k, r, m, ρ) y ATR₀ se congelan al abrir y se guardan en la posición.
- El capital asignado C_i,t y el equity usados para dimensionar son los del cierre de la barra de la
  señal.
- Slippage: se aplica en contra sobre todo precio de llenado, incluidos los niveles de SL y TP:
  compra a P·(1 + 0.0002), venta a P·(1 − 0.0002).
- Comisión: 0.125% del nocional en cada llenado.
- Borrow: se devenga cada día sobre el nocional en corto al cierre, con tasa 0.25% / 252.
- Reversa por señal opuesta: primero se cierra y después se abre el nuevo lado. Como toda entrada, se
  dimensiona con C = |w_t| · Equity_t al cierre de la barra de la señal (decidido 2026-10-04).
- Holding máximo: la barra de entrada cuenta como la primera; con m barras cumplidas se sale al Open
  de la barra m + 1 (decidido 2026-10-04).
- Gaps: se evalúan desde la barra siguiente a la entrada; en la barra de entrada el Open es el precio
  de entrada y solo se revisan SL y TP intrabarra.
- Fin de muestra o de ventana: las posiciones abiertas se valúan a mercado y NO cuentan como
  operación cerrada.
- Dos registros separados: `fills` (cada ejecución) y `trades` (cada operación cerrada).

## 6. Glosario SPEC → código

| SPEC | Código | | SPEC | Código |
|---|---|---|---|---|
| f | `sma_fast` | | k | `k_stop` |
| s | `sma_slow` | | r | `reward_ratio` |
| n | `rsi_window` | | m | `max_holding` |
| lo | `rsi_lo` | | ρ_g | `risk_per_trade` |
| hi | `rsi_hi` | | m(régimen) | `regime_multiplier` |
| Estado_t | `state` | | s_i (fuerza) | `strength` |
| w^RP | `rp_weights` | | w^target | `target_weights` |
| C_i,t | `sleeve` | | δ (banda) | `rebalance_band` |
| ATR₀ | `atr_entry` | | régimen | `{"tendencia", "reversion", "crisis"}` |

## 7. Contratos entre módulos — CONGELADOS en P0

Cambiar una firma requiere un PR que SOLO cambie el contrato, aprobado por los 4.

Tipos comunes:
- `prices: dict[str, pd.DataFrame]` — un DataFrame OHLCV por ticker, índice común.
- "Panel": `pd.DataFrame` de fechas × tickers, con el mismo índice que `prices`.
- `regimes: pd.Series` — una etiqueta de mercado por fecha, filtrada y causal.

```python
# ---------- src/data.py (P1)
download_prices(tickers: list[str], start: str, end: str, out_dir: str = "data") -> None
load_prices(tickers: list[str], data_dir: str = "data") -> dict[str, pd.DataFrame]
load_risk_free(data_dir: str = "data") -> pd.Series            # tasa diaria en decimal, desde ^IRX
audit_prices(prices: dict[str, pd.DataFrame]) -> pd.DataFrame
block_dates(config: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]   # train/validation/test

# ---------- src/signals.py (P2)
compute_indicators(ohlcv: pd.DataFrame, params: dict, config: dict) -> pd.DataFrame
    # columnas: sma_fast, sma_slow, macd_hist, rsi, atr
indicator_votes(indicators: pd.DataFrame, params: dict) -> pd.DataFrame # v_sma, v_macd, v_rsi en {-1, 0, 1}
confirm_signal(votes: pd.DataFrame, min_agree: int = 2) -> pd.Series    # state = sgn(Σx) si |Σx| ≥ 2, si no 0
signal_strength(votes: pd.DataFrame, min_agree: int = 2) -> pd.Series   # s = Σx / 3 si |Σx| ≥ 2, si no 0
generate_signals(prices: dict, params_by_regime: dict[str, dict],
                 regimes: pd.Series, config: dict) -> dict[str, pd.DataFrame]
    # {"state", "strength", "atr"}: paneles; en cada fecha usa los parámetros del régimen vigente.
    # Para un θ único: params_by_regime con el mismo dict en las tres llaves.

# ---------- src/backtest.py (P1)
@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame   # panel de acciones con signo
    exposure: pd.DataFrame    # panel de nocional con signo al cierre
    fills: pd.DataFrame       # date, ticker, side, shares, price, notional, commission, slippage, reason
    trades: pd.DataFrame      # trade_id, ticker, side, entry_date, entry_price, exit_date, exit_price,
                              # shares, exit_reason {signal, stop, target, max_holding},
                              # regime_at_entry, pnl_gross, commission, slippage, borrow, pnl_net
    costs: pd.DataFrame       # fecha × {commission, slippage, borrow}

run_backtest(prices: dict, signals: dict[str, pd.DataFrame], sleeve_weights: pd.DataFrame,
             trade_params: pd.DataFrame, config: dict,
             entry_mask: pd.Series | None = None) -> BacktestResult
    # signals: salida de generate_signals (usa "state" y "atr").
    # sleeve_weights: panel de |w_target| ≥ 0 con Σ ≤ 1 (activo individual: una columna de unos).
    # trade_params: fecha × {k_stop, reward_ratio, max_holding, risk_per_trade}, vigentes al cierre
    #   de cada fecha.
    # entry_mask: True donde se permite abrir (optimización por régimen y embargo).
    # trade_params puede traer la columna opcional "regime" (régimen vigente en la fecha); si existe,
    #   se copia a trades.regime_at_entry; si no, queda NaN (decidido 2026-10-04).

# ---------- src/metrics.py (P2)
compute_metrics(equity: pd.Series, trades: pd.DataFrame,
                rf: pd.Series | float = 0.0, periods_per_year: int = 252) -> dict
    # ann_return, ann_vol, sharpe, sortino, calmar, max_drawdown, n_trades, win_rate, payoff_ratio
drawdown_series(equity: pd.Series) -> pd.Series
returns_table(equity: pd.Series) -> dict[str, pd.DataFrame]    # "mensual", "trimestral", "anual"
exposure_metrics(result: BacktestResult) -> dict               # tiempo en mercado, ops/mes, salidas por motivo
breakeven_winrate(trades: pd.DataFrame, k_stop: float, reward_ratio: float,
                  atr_over_price: float, config: dict) -> dict  # p* teórico (SPEC 8) contra empírico

# ---------- src/regimes.py (P3)
regime_features(prices: dict, config: dict) -> pd.DataFrame      # ventana móvil de 63 días
fit_regime_model(features_train: pd.DataFrame, method: str, seed: int) -> object
    # method ∈ {"rules", "kmeans", "hmm"}
predict_regimes(model: object, features: pd.DataFrame) -> pd.Series   # etiqueta FILTRADA
viterbi_path(model: object, features: pd.DataFrame) -> pd.Series      # SOLO para la figura comparativa
label_regimes(prices: dict, config: dict) -> pd.Series         # etiqueta causal para todas las fechas
regime_validation(features: pd.DataFrame, labels: pd.Series,
                  blocks: dict[str, tuple] | None = None) -> dict
    # silhouette, duracion_media, transiciones_por_mes, pct_tiempo por régimen y por bloque
    # blocks: salida de block_dates(config); sin él, pct_tiempo solo de toda la muestra
    #   (agregado 2026-10-05: sin las fechas no se puede reportar train/validation/test)

# ---------- src/portfolio.py (P4)
estimate_cov(returns: pd.DataFrame, method: str) -> pd.DataFrame    # "sample", "ewma", "ledoit_wolf"
risk_parity_weights(cov: pd.DataFrame, tol: float = 1e-10) -> pd.Series   # Spinu; w_i > 0, Σ = 1
inverse_vol_weights(cov: pd.DataFrame) -> pd.Series
risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series    # RC_i = w_i (Σw)_i / σ_p
equal_weights(tickers: list[str]) -> pd.Series
compose_target(base_weights: pd.Series, strength: pd.Series,
               regime_multiplier: float) -> pd.Series          # m · w̃ / max(1, Σ|w̃|), w̃ = w · s
sleeve_weights(prices: dict, signals: dict[str, pd.DataFrame], regimes: pd.Series,
               config: dict, method: str = "risk_parity") -> pd.DataFrame
    # panel causal de |w_target| con rebalanceo aplicado; method ∈ {"risk_parity", "equal"}
turnover(weights_drift: pd.DataFrame, weights_target: pd.DataFrame) -> pd.Series   # ½ Σ|w_t − w_t−|
rebalance_sweep(prices: dict, params_by_regime: dict, regimes: pd.Series, config: dict,
                bands: list[float], frequencies: list[str]) -> pd.DataFrame

# ---------- src/optimize.py (P2) — firmas propuestas; P2 puede ajustarlas avisando a P1 (main.py)
search_space() -> dict
diagnostic_study(prices: dict, config: dict, sampler: str, n_trials: int, seed: int) -> "optuna.Study"
select_plateau(study: "optuna.Study", top_frac: float = 0.10) -> dict    # medoide del mejor 10% factible
optimize_regime(prices: dict, regimes: pd.Series, regime: str | None, window: tuple,
                config: dict, seed: int) -> dict               # regime=None → θ único
walk_forward(prices: dict, regimes: pd.Series, config: dict,
             mode: str = "rolling", per_regime: bool = True) -> dict   # mode ∈ {"rolling", "anchored"}
wf_efficiency(wf_result: dict) -> float
sensitivity(params_by_regime: dict, prices: dict, regimes: pd.Series, config: dict,
            pct: float = 0.20) -> pd.DataFrame
cost_sweep(params_by_regime: dict, prices: dict, regimes: pd.Series, config: dict,
           round_trip_bps: list[float]) -> pd.DataFrame
single_indicator_comparison(params_by_regime: dict, prices: dict, regimes: pd.Series,
                            config: dict) -> pd.DataFrame        # pregunta 1

# ---------- src/plots.py (P3) — una función por figura; cada una devuelve matplotlib.figure.Figure
```

Si dependes de una función que todavía es stub, prueba con los fixtures de `tests/conftest.py` o con un
stub local dentro de tu prueba. Nunca edites el archivo de otra parte.

## 8. Pruebas (pytest)

Todas usan datos sintéticos con semilla fija. Ninguna lee `data/` ni usa red. Se corren con
`python -m pytest` desde la raíz.

| Archivo | Prueba | Criterio | Parte |
|---|---|---|---|
| `test_signals.py` | Truncamiento de indicadores y señales | Valor en t sobre `df.iloc[:t+1]` igual al de la serie completa, para ≥ 3 valores de t, columna por columna | P2 |
| `test_signals.py` | Confirmación | (+1, 0, 0) → 0; (+1, +1, 0) → +1; (+1, +1, −1) → 0; (−1, −1, −1) → −1 | P2 |
| `test_pipeline.py` | Truncamiento del pipeline | indicadores → señales → combinación → backtest: el equity en t no cambia al truncar en t | P1 |
| `test_backtest.py` | Contabilidad | `equity == cash + Σ shares·close` en todas las fechas (tol 1e-6) y `Σ commission == Σ |notional| · 0.00125` (tol 1e-6) | P1 |
| `test_backtest.py` | Golden-file | Equity final calculado a mano en caminos cortos (ver P1) | P1 |
| `test_metrics.py` | Métricas | Sharpe, MDD y Calmar contra cálculo a mano en una serie corta | P2 |
| `test_portfolio.py` | Risk Parity | `max(RC) − min(RC) < 1e-6` y `Σ RC = σ_p` (tol 1e-8) | P4 |
| `test_regimes.py` | Truncamiento de régimen | Features columna por columna y etiqueta filtrada en t no cambian al agregar datos posteriores | P3 |

## 9. Reproducibilidad

- `python main.py` ejecuta todo sin intervención ni red: carga → auditoría → régimen → estudios de
  diagnóstico → walk-forward → backtests finales (Risk Parity y pesos iguales) → métricas →
  `results/` y `docs/figuras/`.
- `main.py` contiene un dict `CONFIG` con TODOS los valores fijos del SPEC: fechas, comisión, slippage,
  borrow, capital, MACD 12/26/9, ATR 14, ventanas del walk-forward, pruebas por estudio, tasa de
  actividad mínima, embargo, ventana de régimen, multiplicadores por régimen y tickers. Ningún módulo
  de `src/` escribe esos valores a mano; los recibe en `config`.
- `SEED = 42` vive en `CONFIG` y se pasa explícito a los samplers de Optuna, al `random_state` de los
  modelos y a `np.random.default_rng`. La ventana i del walk-forward usa la semilla `SEED + i`.
  Prohibido `np.random.seed` global.
- El paralelismo es entre ventanas del walk-forward (`joblib`), nunca dentro de un estudio de Optuna.
- `results/` guarda los resultados en pickle o parquet. El notebook solo los carga; nunca reoptimiza.
- Antes de la corrida final sobre test: θ congelados en `results/`, commit en `main`, y el hash del
  commit registrado en el README y en el reporte.

## 10. Estilo de código

- Nombres de funciones y variables en inglés; docstrings, comentarios y textos de figuras en español.
- Funciones de responsabilidad única, con type hints y docstring estilo NumPy que cite el punto del
  SPEC que implementan (por ejemplo, "SPEC punto 4, rearme").
- Indicadores vectorizados; motor con loop explícito.
- Sin código muerto, comentado ni duplicado. Un solo motor para activo individual y portafolio.
- Transparencia de NaN: lo indefinido se reporta como NaN.
- Figuras con título, ejes etiquetados y leyenda.
- Notebook: solo importa de `src/`, carga `results/` y grafica. Celdas cortas, una idea por celda,
  sin fórmulas.
- Cualquier integrante debe poder explicar cualquier línea: claridad sobre trucos, y comentarios
  sobre el porqué de las decisiones no obvias.

## 11. Flujo de Git

- P0 se hace directo en `main`. Después, `main` queda protegida en GitHub: solo recibe cambios por PR.
- Cada integrante trabaja en su rama (sección 1) y abre un PR por cada función o grupo de funciones
  terminado con su prueba. El PR lo revisa al menos otro integrante.
- Antes de abrir un PR: `git merge main` en tu rama y `python -m pytest` en verde.
- Commits pequeños, en español, con prefijo: `feat(backtest): ...`, `test(signals): ...`, `fix(...)`,
  `docs(...)`, `refactor(...)`.
- Cada integrante hace commits desde su propia cuenta. Claude Code nunca usa `--author` ni hace
  commits a nombre de otro.
- Claude Code no hace `push`, `merge`, `rebase` ni abre PRs sin instrucción explícita.
- Se evalúa el último commit en `main` antes de las 23:59 del día anterior a la exposición.

## 12. Reglas para Claude Code

- Lee la sección 0 completa antes de tocar código.
- Trabaja solo en los archivos de la parte de la sesión (sección 1).
- Respeta los contratos de la sección 7; no cambies firmas.
- Ante una decisión no documentada, márcala como `PENDIENTE DE CONFIRMAR` y pregunta antes de
  implementar.
- No hables de funciones que no existen como si existieran: verifica primero.
- Al terminar: corre `python -m pytest` y reporta el resultado.
- Agrega a tu subsección "Uso de IA" del README qué se hizo con asistencia.

## 13. Reporte y presentación

| Parte | Secciones del reporte | Preguntas |
|---|---|---|
| P1 | Datos y auditoría; motor y supuestos de ejecución; tabla de auditoría de sesgos; advertencia de impacto de mercado con magnitud estimada | 7 (coordina) |
| P2 | Estrategia y regla 2 de 3 como fórmula; optimización (tiempo total y configuraciones evaluadas); walk-forward rolling contra anchored y su eficiencia; sensibilidad ±20%; curva de costos | 1, 2, 3, 4 |
| P3 | Análisis de régimen: comparación de los tres métodos, elección, validación, métricas por régimen, θ por régimen contra θ único | 5 |
| P4 | Risk Parity, estimadores de covarianza, agregación, rebalanceo y turnover; Risk Parity contra pesos iguales y contra activos individuales | 6 |

Todos: descripción de la estrategia en sus 2 activos. Presentación: máximo 12 diapositivas, unas 3 por
integrante.

## 14. PENDIENTE DE CONFIRMAR

1. ~~Los 8 tickers y el formato de `data/`~~ RESUELTO (sección 3). Falta el reparto de 2 por integrante.
2. ~~Versión de Python, que se fija en P0.~~ RESUELTO: Python 3.13.9 (venv en `.venv/`).
3. Todo lo marcado como pendiente en `SPEC_portafolio.md`, que P3 y P4 completan como primera tarea.
4. Si el rebalanceo redimensiona posiciones abiertas (`resize_on_rebalance`). Lo deciden P4 y P1.
5. Asignación de cada parte (P0 a P4) a un integrante y nombre de las ramas.
