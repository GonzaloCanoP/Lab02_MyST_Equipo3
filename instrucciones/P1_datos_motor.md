# P1 — Datos, motor de backtesting y auditoría de sesgos

Hereda todo de `instrucciones/CLAUDE.md`. 

## Las tareas

### De la actividad de backtest del curso
1. "Implementa backtest() como una función pura en src/backtest.py, con el estado de cash, posición y
   equity explícito."
2. "Escribe el golden-file test: un camino corto de unas pocas barras con el equity final calculado a
   mano, 'en papel', no corriendo el código que se está probando, y compáralo contra lo que produce tu
   función."
3. "Escribe la prueba de truncamiento a nivel de todo el pipeline (indicadores → señales → combinación
   → backtest), no solo de un indicador suelto: recalcula sobre df.iloc[:t+1] y exige que el valor en
   t no cambie."
4. "Corre tu estrategia del spec y produce: curva de equity, curva de drawdown, lista de operaciones y
   el gráfico de sensibilidad a costos."
5. "Llena la tabla de auditoría de sesgos para tu configuración, con evidencia concreta para cada
   renglón: look-ahead, survivorship, overfitting/data snooping, ejecución optimista,
   selección/periodo de muestra."

### Del enunciado del Lab 02
6. "Descargue al menos 6 años de datos diarios históricos para cada activo. Todos los activos deben
   tener datos completos y traslapados en el periodo seleccionado."
7. "El motor debe ser event-driven, con estado explícito de efectivo, posiciones y valor del portafolio.
   Debe manejar stop-loss y take-profit, y declarar la convención usada cuando ambos caen dentro del
   rango de la misma barra. Los costos de transacción se aplican en cada apertura y cada cierre."
8. Prueba 3: "el valor final del portafolio es igual al efectivo más el valor de las posiciones
   abiertas, y la suma de los costos aplicados coincide con el número de operaciones por la comisión."
9. "main.py debe ejecutar el proyecto completo con un solo comando."
10. "El backtest asume ejecución completa al precio modelado y no incorpora impacto de mercado ni fallas
    de ejecución. Su reporte debe señalar esta limitación de manera explícita y estimar su magnitud."

## Cómo encaja
- `data.py`: los datos del portafolio están en `data/` con el formato de CLAUDE.md, sección 3. `load_risk_free` convierte `^IRX` (rendimiento
  anual en porcentaje) a tasa diaria decimal: ÷ 100 ÷ 252.
- `backtest.py`: `run_backtest` implementa SPEC.md, puntos 4 a 7, con las convenciones de CLAUDE.md,
  sección 5. Tamaño: `shares = ρ · C_i / |E − SL|`, con `C_i = sleeve_weights[i] · Equity` y tope
  `shares · E · (1 + comisión) ≤ C_i`.
- Rearme (SPEC punto 4): estado `armed[ticker][side]` que se desarma tras un SL o un holding máximo y
  se rearma en la primera barra con `state ≠ side`.
- `entry_mask`: si es False en t, no se abren posiciones con la señal de t. Las salidas no se afectan.
- Golden-file tests en `test_backtest.py`: caminos de 5 a 10 barras construidos a mano, con el equity
  final calculado en papel y escrito como constante en el test. Uno por caso:
  1. SL y TP en la misma barra → sale por stop.
  2. Gap que abre más allá del SL → llena al Open.
  3. Gap que abre más allá del TP → llena en el nivel del TP.
  4. Rearme: tras un TP reentra a la barra siguiente; tras un SL no reentra hasta que cambia el estado.
  5. Holding máximo → cierra al Open de la barra m+1.
  6. Señal opuesta → cierra y abre el lado contrario en el mismo Open.
  7. El tope sin apalancamiento recorta el tamaño.
  8. El borrow se devenga solo en los días en corto.
  9. La señal de t se ejecuta en t+1 y nunca en t.
- `test_pipeline.py`: sobre el fixture sintético, corre `compute_indicators` → `generate_signals` →
  `run_backtest` con `df.iloc[:t+1]` y con la serie completa, y exige el mismo equity en t para
  ≥ 3 valores de t. Mientras signals sea stub, usa un stub local de señales causal; al integrarse
  `signals.py` a `main`, cambia al real.
- Corrida base (tarea 4): valores base de SPEC.md, punto 2, θ único, régimen ignorado, sobre train.
  Dos versiones: cada activo individual (`sleeve_weights` = una columna de unos) y portafolio de pesos
  iguales (panel constante de 1/8). Produce equity, drawdown y lista de operaciones en `results/`.
  La curva de costos la implementa P2 en `cost_sweep`; aquí solo se corre.
- Auditoría de sesgos (tarea 5): tabla en `results/auditoria_sesgos.md` con un renglón por sesgo y
  evidencia concreta (prueba, cifra o decisión documentada). Survivorship es especialmente relevante:
  los 8 activos se eligen hoy, entre empresas que sobrevivieron hasta 2026.
- Impacto de mercado (tarea 10): modelo de raíz cuadrada, `impacto ≈ σ_diaria · sqrt(shares / ADV)`,
  aplicado ex post a las operaciones de test. Se reporta en bps y como porcentaje del retorno; no
  entra al motor.

## Entregable
- `src/data.py` y `src/backtest.py`, completos.
- `tests/test_backtest.py` (contabilidad y 9 golden-file tests) y `tests/test_pipeline.py`.
- `results/`: corrida base, auditoría de sesgos y estimación de impacto.
- `main.py` integrado de punta a punta, conforme los demás módulos se integren a `main`.
- README completo (excepto las subsecciones de IA de los demás) y `requirements.txt` actualizado.

## Checklist
- [ ] `audit_prices` reporta NaN, precios ≤ 0, `high < low`, `close` fuera de `[low, high]`, volumen cero y fechas descartadas
- [ ] Ninguna función de `data.py`, salvo `download_prices`, importa `yfinance`
- [ ] `run_backtest` no modifica sus argumentos (prueba con copias y `assert_frame_equal`)
- [ ] Prueba de contabilidad en verde con tolerancia 1e-6, sobre 3 activos sintéticos con largos y cortos
- [ ] Los 9 golden-file tests pasan con valores calculados a mano y escritos como constantes
- [ ] `test_pipeline.py` en verde para ≥ 3 valores de t
- [ ] Ningún `shift` para simular ejecución fuera de `run_backtest` (`grep` en `src/`)
- [ ] Corrida base en `results/`: equity, drawdown y lista de operaciones, por activo y de pesos iguales
- [ ] Tabla de auditoría con los 5 sesgos y evidencia concreta en cada renglón
- [ ] Estimación de impacto de mercado en bps en `results/`
- [ ] `python main.py` corre de punta a punta sin red en una copia limpia del repo

## Puntos abiertos
- ~~`resize_on_rebalance`~~ RESUELTO (SPEC_portafolio, Rebalanceo): `False`. Una posición abierta
  conserva su tamaño; el costo del rebalanceo se estima ex post con `turnover`.
- ~~ADV para el impacto~~ RESUELTO (2026-10-05): Q = nocional del llenado, ADV = media de 20 días del
  volumen en dólares (close · volume) y σ = desviación de 20 días de los retornos simples, ambos al
  cierre de la barra de la señal (`adv_window`). La función es `market_impact` en `src/backtest.py`.

## Estado y pendientes (2026-10-05)

### Hecho
- `src/data.py` completo. La auditoría de los datos reales sale limpia (0 NaN, 0 incoherencias OHLC,
  0 fechas descartadas; 2,449 días hábiles comunes).
- `src/backtest.py`: `run_backtest` completo, con las decisiones de CLAUDE.md, sección 5, y
  `market_impact` (modelo de raíz cuadrada ex post, P1 tarea 10).
- `tests/test_backtest.py`: prueba 3 (contabilidad), inmutabilidad de argumentos, los 9 golden-file
  tests, uno extra con costos completos y 3 pruebas de `market_impact` (golden a mano, causalidad y
  sin operaciones).
- `tests/test_pipeline.py`: truncamiento con `compute_indicators` → `generate_signals` reales.
- `notebooks/analisis_P1.ipynb`: datos, recorrido del motor sobre golden cases y contabilidad sintética.
- `main.py`:
  - `stage_base_run` + `stage_save_base_run` (tarea 4): valores base, θ único, régimen ignorado,
    sobre train; 8 activos individuales y pesos iguales. Guarda `results/corrida_base.pkl` y 4
    figuras `docs/figuras/base_*.png`.
  - `stage_report`: métricas, `results/final_backtests.pkl` y figuras `final_equity` y
    `final_drawdown`.
  - `stage_market_impact`: impacto sobre las operaciones de test de Risk Parity, en bps y como % del
    equity y del retorno de test → `results/impacto_mercado.pkl`.
  - `stage_bias_audit` (tarea 5): escribe `results/auditoria_sesgos.md` con las cifras de la corrida.
  - `results/datos_regimen.pkl`: auditoría de datos y etiqueta de régimen.

### Hallazgos de la corrida base (train, para la calibración de SPEC punto 9)
Corrida previa con un drawdown provisional (P2 aún no implementa `drawdown_series`):
- **Actividad mínima:** 380 operaciones en 4 años con pesos iguales, 47.5 por ventana de 6 meses. El
  mínimo de 24 es del portafolio (1 por activo cada dos meses × 8 activos), así que se cumple; la
  versión anterior de esta nota lo comparaba contra 5.9 operaciones por activo. Calibración
  registrada en SPEC punto 9 (v1.3).
- Contabilidad: |equity − (cash + Σ shares · close)| ≤ 1.2e-10 en la corrida real.
- Salidas en pesos iguales: 171 stop, 98 target, 95 holding máximo, 16 señal.
- Impacto ex post (pesos iguales, train): ~4 bps promedio por llenado; COPX domina (~27 bps,
  participación media 2.8% del ADV), el resto < 1.2 bps.

### PENDIENTE: requiere P2 (`metrics.py` y `optimize.py`)
- [x] `drawdown_series` y `compute_metrics` implementadas por P2.
- [ ] Curva de sensibilidad a costos: correr `cost_sweep` (con `period` = train) en `main.py`.
- [ ] Agregar la corrida base y la curva de costos a `analisis_P1.ipynb`.
- [ ] `stage_final_backtests`: `walk_forward` devuelve `params_by_fold` (un θ por mes), no un solo
      `params_by_regime`. Las señales se arman por ventana con su θ y se concatenan los meses de
      prueba; `trade_params` ya viene concatenado con la columna "regime".

### PENDIENTE: requiere P4 (`portfolio.py`)
- [x] `resize_on_rebalance = False` (SPEC_portafolio, Rebalanceo).
- [x] Una entrada con C_i = 0 no abre ni consume el armado (`capital > 0` es condición de entrada).
- [x] Panel real de `sleeve_weights` con valores base: Σ|w| ≤ 0.98 en todas las fechas.
- [x] `rebalance_frequency` ("M"), `rebalance_frequencies` (W/M/Q) y `rebalance_bands` alineados con
      SPEC_portafolio v0.3.

### PENDIENTE: al final (test congelado)
- [ ] `python main.py` de punta a punta en una copia limpia sin red.
- [ ] README: hash del commit de la corrida final.
