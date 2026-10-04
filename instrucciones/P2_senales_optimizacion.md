# P2 — Señales, métricas y optimización

Hereda todo de `instrucciones/CLAUDE.md`.

## Las tareas

### Del enunciado del Lab 02
1. "Implemente un mínimo de tres indicadores técnicos provenientes de al menos dos familias distintas."
2. "La regla de confirmación exige que al menos 2 de 3 indicadores coincidan en dirección para abrir una
   posición. Documente la regla como fórmula en el reporte."
3. "Los indicadores se calculan con información disponible hasta el cierre de la barra t y la posición
   se ejecuta en t+1."
4. "Optimice los hiperparámetros con un método sistemático. Limite la optimización a entre 100 y 200
   pruebas por ventana. Documente el tiempo total de optimización y el número total de configuraciones
   evaluadas. Imponga una restricción de número mínimo de operaciones por ventana."
5. "Reporte, para cada conjunto de datos por separado: Sharpe Ratio, Sortino Ratio, Calmar Ratio,
   Maximum Drawdown y Win Rate. Acompáñelas de la tabla de retornos mensuales, trimestrales y anuales."
6. Walk-forward de nivel C: entrenamiento de 6 meses, prueba de 1 mes, paso mensual. "Optimice los
   parámetros de cada régimen dentro de cada ventana de entrenamiento."
7. "Sensibilidad de parámetros: varíe cada parámetro óptimo en ±20% y reporte el impacto."
8. "Curva de retorno neto contra nivel de costo de transacción."
9. Pruebas 1 (causalidad) y 2 (regla de confirmación).

### De la actividad de backtest del curso
10. "Agrega en src/metrics.py las métricas de desempeño, drawdown máximo y Calmar. Súmales el turnover y
    el winrate, teórico (el p* de tu SPEC.md) contra empírico (el de tus operaciones)."
11. "Gráfico de sensibilidad a costos: Sharpe o equity contra el costo de ida y vuelta, de 0 a 50 bps."

### De la actividad de optimización del curso (S08)
12. Objetivo θ* = argmax_θ Calmar(backtest(train, θ)), sobre un backtest CON costos. "Si un set de
    parámetros produce menos de N_MIN trades, la función objetivo debe retornar −inf."
13. Fase 1 — Random Search con `optuna.samplers.RandomSampler`, con N reportado explícitamente.
    Superficie 3D del Calmar sobre las dos dimensiones más influyentes (según la importancia del TPE),
    con el resto de θ fijo en el mejor valor del random search. Declarar que no existe una superficie
    completa en más de 2 dimensiones.
14. Fase 2 — TPE con `TPESampler(seed=42)`, mismo espacio y misma restricción. Visualizaciones:
    historia de optimización, importancia de parámetros y slice plots. "Si es un pico aislado, el θ*
    final debe tomarse del centro de la mejor meseta visible, no del argmax literal."
15. Walk-forward anchored y rolling, comparados: "la diferencia entre ambas es en sí misma un
    diagnóstico de si el edge es estable o decae con el tiempo."
16. "Aplica purga y embargo en cada frontera de fold."
17. "Concatena la curva de equity out-of-sample de todos los folds."
18. "Walk-forward efficiency = performance out-of-sample / performance in-sample. Valores por debajo de
    ~0.5 son evidencia de que el resultado in-sample es mayormente ruido ajustado."
19. El bloque final "se toca UNA SOLA VEZ, al final, con θ* ya congelado": congelar θ*, commit, registrar
    el hash y reportar el resultado tal cual salga.
20. Tabla de métricas (retorno anualizado, volatilidad, Sharpe, Sortino, Calmar, max drawdown, número de
    trades, win rate, payoff ratio) por bloque, más benchmark buy & hold.

## Cómo encaja con el proyecto

La actividad S08 se hizo sobre un solo activo con folds K = 4 o 5 y un split 60/20/20. En el Lab 02
mandan el enunciado y `SPEC.md`:

| Actividad S08 | En este proyecto |
|---|---|
| Datos BTC 1m a 15m | 8 acciones diarias de `data/`, vía `load_prices` |
| Split 60/20/20 | Train / Validation / Test de SPEC.md, punto 1. El bloque "que se toca una vez" es **Test** |
| N_MIN = 30 | Actividad mínima de SPEC.md, punto 7 (24 por ventana de 6 meses, prorrateada por régimen) |
| Random 200 y TPE 300 trials | Estudios de diagnóstico sobre todo Train con θ único: 200 random y 200 TPE (tope del lab). Dentro del walk-forward: 150 TPE por régimen y por ventana |
| K folds anchored y rolling | Rolling 6m/1m/mensual (el del lab, el principal) y anchored con entrenamiento creciente y prueba de 1 mes, como diagnóstico |
| Purga y embargo | Al final de cada ventana de entrenamiento las posiciones se valúan a mercado (purga) y no se abren posiciones en sus últimos 5 días (embargo, `entry_mask`) |

- `signals.py`: SPEC.md, puntos 2 y 3. Compuerta |Σx| ≥ 2 y fuerza s = Σx / 3. MACD 12/26/9 y ATR 14
  vienen de `config`. RSI y ATR con suavizado de Wilder (`ewm(alpha=1/n, adjust=False)`).
- `metrics.py`: Calmar = retorno anualizado / |MDD|. Sharpe y Sortino anualizados con √252 y con la tasa
  diaria de `load_risk_free`. Win Rate = % de operaciones cerradas con `pnl_net > 0`. Payoff = ganancia
  media / pérdida media. `breakeven_winrate` compara el p* de SPEC.md, punto 8, contra el empírico. El
  turnover lo calcula P4 en `portfolio.turnover`; aquí solo se reporta.
- Cada prueba de Optuna: `generate_signals` → `sleeve_weights` (P4) → `run_backtest` (P1) con
  `entry_mask` = días del régimen y sin embargo → Calmar. Infactible si `n_trades < N_min,g`.
- `select_plateau`: medoide del mejor 10% de pruebas factibles, con distancia euclidiana sobre los
  parámetros normalizados a [0, 1]. Se usa tanto en el diagnóstico como en cada ventana.
- θ por régimen contra θ único: `walk_forward(per_regime=True)` y `walk_forward(per_regime=False)`.
  La comparación la interpreta P3 (pregunta 5).
- `wf_efficiency`: retorno anualizado OOS concatenado / promedio del retorno anualizado IS de las
  ventanas (se reporta también con Calmar).
- `cost_sweep`: costo de ida y vuelta de 0 a 100 bps (el curso pide 0 a 50; el caso base del lab ya es
  de 29 bps, así que se extiende para encontrar el punto de equilibrio).
- Benchmark buy & hold: portafolio equiponderado de los 8 activos, sin rebalanceo, con los mismos costos
  de entrada.
- Paralelismo con `joblib` entre ventanas, nunca dentro de un estudio. Semilla `SEED + i` por ventana.

## Entregable
- `src/signals.py`, `src/metrics.py` y `src/optimize.py`.
- `tests/test_signals.py` (pruebas 1 y 2) y `tests/test_metrics.py`.
- En `results/`, vía `main.py`: los dos estudios de diagnóstico (N de trials incluido), θ* de meseta,
  parámetros por ventana y régimen, Calmar IS contra OOS por ventana, eficiencia rolling y anchored,
  tiempo total y configuraciones evaluadas, sensibilidad, curva de costos, comparación contra
  indicadores individuales y tabla de métricas por bloque con buy & hold.

## Checklist
- [ ] Prueba de truncamiento en verde para t ∈ {150, 400, último}, columna por columna (indicadores, votos, estado y fuerza)
- [ ] Prueba de confirmación: (+1, 0, 0) → 0; (+1, +1, 0) → +1 con fuerza 2/3; (+1, +1, −1) → 0; (−1, −1, −1) → −1 con fuerza −1
- [ ] `test_metrics.py`: Sharpe, MDD y Calmar coinciden con un cálculo a mano en una serie de 10 valores (tol 1e-9)
- [ ] Toda prueba con `n_trades < N_min,g` devuelve −inf
- [ ] Estudios de diagnóstico: 200 random y 200 TPE, con N guardado en `results/`
- [ ] Las 4 figuras del diagnóstico (superficie 3D, historia, importancia, slice plots) se generan vía `plots.py`
- [ ] Cada estudio del walk-forward corre exactamente 150 pruebas con semilla `SEED + i`
- [ ] Dos corridas de `walk_forward` con la misma semilla dan parámetros idénticos
- [ ] Ninguna fecha de prueba aparece dentro de los datos de ajuste (prueba con fechas)
- [ ] Rolling y anchored corridos, cada uno con su eficiencia
- [ ] `sensitivity` varía cada parámetro a ×0.8 y ×1.2 (enteros redondeados) y reporta el ΔCalmar
- [ ] `cost_sweep` de 0 a 100 bps identifica el punto de equilibrio y el margen frente a 29 bps
- [ ] `single_indicator_comparison` reporta número de operaciones y Calmar de cada indicador solo contra la regla 2 de 3

## Puntos abiertos
- PENDIENTE DE CONFIRMAR: Calmar cuando MDD = 0 con operaciones. Propuesta: marcar la prueba como
  infactible.
- PENDIENTE DE CONFIRMAR: correlación entre los votos de SMA y MACD en train (SPEC.md, punto 9). Medir
  y reportar al equipo antes de congelar los indicadores.
- PENDIENTE DE CONFIRMAR con el profe: si los estudios de diagnóstico sobre todo Train pueden usar los
  300 trials de S08 o se respeta el tope de 200 del lab (se usa 200 mientras tanto).
