# SPEC — Lab 02 · Equipo 3 · Nivel C (acciones y ETFs diarios)

**Versión:** 1.3 · **Estado:** calibrado con train; validation y test sin observar

Estrategia direccional larga y corta, aplicada con las mismas reglas a 8 activos. Risk Parity asigna
el capital por activo y fixed fractional dimensiona cada operación. Este documento se versiona ANTES
de correr cualquier backtest; train, validation y test no se han observado. El detalle de régimen,
agregación y rebalanceo va en `SPEC_portafolio.md`.

## 1. Universe and frequency
- **Activos:** 8 activos líquidos de EE.UU. (2 por integrante), listados antes de 2016 y con historia
  completa y traslapada. Datos en `data/`.

| Ticker | Activo | Tipo | Cotiza desde |
|---|---|---|---|
| AAPL | Apple | Acción | 1980 |
| MSFT | Microsoft | Acción | 1986 |
| META | Meta Platforms | Acción | 2012 |
| AMD | Advanced Micro Devices | Acción | 1972 |
| XOM | Exxon Mobil | Acción | 1970 |
| SMH | VanEck Semiconductor ETF | ETF | 2000 |
| GLD | SPDR Gold Shares | ETF | 2004 |
| COPX | Global X Copper Miners ETF | ETF | 2010 |

  Reparto por integrante: *[PENDIENTE]*. Tasa libre de riesgo: `^IRX` (T-Bill a 13 semanas).
  Sesgo de supervivencia: la lista se eligió en 2026 entre activos que sobrevivieron hasta hoy; se
  declara en el reporte. Concentración declarada: AAPL, MSFT, META, AMD y SMH son tecnología
  (SMH contiene a AMD), así que se espera correlación alta entre ellos; Risk Parity la penaliza.
  COPX es el menos líquido del universo: el slippage de 2 bps puede subestimar su costo real.
- **Barras:** diarias OHLCV, ajustadas por splits y dividendos (`auto_adjust=True`).
- **Rango:** 2017-01-02 → 2026-09-30. El año 2017 es solo calentamiento: no genera señales.
  Quedan 8.75 años operables, por encima del mínimo de 6 que pide el lab.
- **Partición:**

| Bloque | Fechas | Uso |
|---|---|---|
| Train | 2018-01-01 → 2021-12-31 | Diseño de reglas, rangos de búsqueda, estudios de diagnóstico y calibraciones |
| Validation | 2022-01-01 → 2023-12-31 | Una corrida para decisiones discretas (método de régimen, estimador de covarianza, banda de rebalanceo) |
| Test | 2024-01-01 → 2026-09-30 | Una sola corrida con todo congelado y commit previo |

- **Walk-forward:** entrenamiento de 6 meses, prueba de 1 mes, paso mensual (rolling). Como diagnóstico
  se corre también la variante anchored (entrenamiento creciente). Corre de forma continua sobre los
  tres bloques; las métricas de cada bloque se calculan concatenando sus meses fuera de muestra.

## 2. Features
| Indicador | Familia | Voto | Rango de búsqueda | Valor base |
|---|---|---|---|---|
| Cruce de SMA | Tendencia | x_sma = sgn(SMA_f − SMA_s) | f ∈ [5, 30], s ∈ [40, 120] | 20 / 50 |
| Histograma MACD | Momento | x_macd = sgn(MACD − señal) | Fijo | 12 / 26 / 9 |
| RSI de Wilder | Momento | +1 si 50 < RSI < hi · −1 si lo < RSI ≤ 50 · 0 en otro caso | n ∈ [7, 21], lo ∈ [20, 40], hi ∈ [60, 80] | 14 / 30 / 70 |
| ATR de Wilder | Volatilidad | No vota; fija SL, TP y tamaño | Fijo | 14 |

Mientras un indicador no tiene observaciones suficientes, su voto es 0.

## 3. Entry rule
Con x_j ∈ {−1, 0, +1} los votos de los k = 3 indicadores y Σx = x_sma + x_macd + x_rsi:

    s_t = Σx / 3     si |Σx| ≥ 2
          0          en otro caso                    (umbral: 2 de 3)

    Estado_t = sgn(s_t) ∈ {−1, 0, +1}

La compuerta (¿hay posición?) y la fuerza (¿cuánto riesgo?) son preguntas separadas: Estado_t decide
la entrada y s_t ∈ {−1, −2/3, 0, 2/3, 1} escala la asignación del portafolio (`SPEC_portafolio.md`).
Con esta regla, dos votos a favor y uno en contra (Σx = 1) no abren posición.

Se abre una posición de lado L = Estado_t si el lado L está armado (punto 4).

## 4. Exit rule
Con ATR₀ = ATR14 al cierre de la barra de la señal, congelado al abrir, y L = ±1 el lado:

    SL = E − L · k · ATR₀        TP = E + L · r · k · ATR₀        k ∈ [1, 4], r ∈ [1, 4]   (base 2, 2)

- **Señal opuesta:** si Estado_t = −L, se cierra al Open de t+1 y se abre el lado contrario en esa misma
  apertura. Un Estado_t = 0 no cierra la posición.
- **Holding máximo:** m ∈ [5, 40] barras (base 20). Al cumplirse, se cierra al Open de la barra siguiente.
- **Rearme del lado L:**
  - Tras un **take-profit**, el lado queda armado de inmediato.
  - Tras un **stop-loss o un holding máximo**, el lado queda desarmado hasta la primera barra con
    Estado_t ≠ L.

  Un TP indica que la tendencia funcionó y sigue viva. Un SL indica que falló, y reentrar en ella
  produce cadenas de stops.
- **Cambio de ventana o de régimen:** una posición abierta conserva hasta su cierre los parámetros
  con los que abrió.

## 5. Sizing
Dos capas:

1. **Asignación por activo:** el capital asignado al activo i es

       C_i,t = |w_i,t^target| · Equity_t

   donde w^target combina Risk Parity, la fuerza de la señal y el multiplicador de régimen
   (`SPEC_portafolio.md`):  w̃_i = w_i^RP · s_i,  w^target = m(régimen) · w̃ / max(1, Σ|w̃_i|).

2. **Tamaño de la operación (fixed fractional):** con riesgo ρ_g por operación en el régimen g,

       acciones = ρ_g · C_i,t / |E − SL|        ρ_g ∈ [0.5%, 2%]   (base 1%)

   Se permiten acciones fraccionarias.

- **Sin apalancamiento:** acciones · E · (1 + comisión) ≤ C_i,t. Como Σ|w^target| ≤ 1, la exposición
  bruta total nunca excede el capital. Si el tope aplica, el riesgo efectivo queda por debajo de ρ_g.
- **Presupuesto de riesgo:** ρ_g del capital asignado al activo, medido hasta el stop. Es una
  intención, no una pérdida máxima garantizada: los gaps pueden rebasarlo.
- **Efecto declarado:** el riesgo por operación es ρ · |w^target| · Equity, así que los activos de menor
  volatilidad (mayor w^RP) arriesgan más en dinero por operación. Las contribuciones al riesgo
  realizadas se reportan junto a las objetivo.
- **Benchmark de pesos iguales:** w^RP se sustituye por 1/8, con las mismas señales, costos y rebalanceo.
- **Referencia de activo individual:** C = Equity.

## 6. Costs
| Concepto | Valor | Fuente / justificación |
|---|---|---|
| Comisión | 0.125% del nocional por lado | Parámetro fijo del lab |
| Slippage | 2 bps por lado | Spreads de alrededor de 1 bp en mega caps y 4.5 bps en promedio del S&P 500 (Equicurious, *Bid-ask spreads and liquidity in US equities*) |
| Borrow fee | 0.25% anual sobre el nocional en corto, devengado por día | Tasa ilustrativa de acciones "General Collateral" (Interactive Brokers, *The risks of shorting, part II*) |
| Dividendos | Implícitos en los precios ajustados | El largo los cobra y el corto los paga |

Capital inicial: USD 1,000,000. Costo de ida y vuelta: 2 × (0.125% + 0.02%) = 0.29% (29 bps). Comisión,
slippage y borrow se registran por separado. No se modela impacto de mercado; se declara como
limitación con una estimación de su magnitud.

## 7. Conventions
- **Cuándo se actúa:** la señal se calcula al cierre de t y se ejecuta al Open de t+1.
- **Orden dentro de la barra t+1:**
  1. Salidas por señal o por holding, al Open.
  2. Entradas al Open.
  3. Gap: si el Open rebasa el SL, se llena al Open; si rebasa el TP, se llena en el nivel del TP
     (sin mejora de precio, supuesto conservador).
  4. SL y TP intrabarra con High y Low, evaluados desde la barra de entrada.
  5. Valuación al Close.
- **Empate intrabarra:** si SL y TP caen en la misma barra, se ejecuta primero el stop-loss.
- **Posiciones:** una por activo, sin pirámides.
- **Optimización:**
  - Optuna (TPE), semilla 42, objetivo Calmar.
  - 150 pruebas por régimen y por ventana del walk-forward.
  - Un juego de parámetros por régimen, compartido por los 8 activos. Como referencia se optimiza
    también un θ único sin régimen.
  - Se elige el medoide del mejor 10% de pruebas factibles (centro de meseta), no el máximo.
  - Purga y embargo: en cada ventana de entrenamiento, las posiciones se valúan a mercado al final de
    la ventana y no se abren posiciones en sus últimos 5 días.
- **Espacio de búsqueda (9 dimensiones por régimen):** f, s, n, lo, hi, k, r, m, ρ_g.
- **Actividad mínima:** al menos una operación cerrada por activo cada dos meses en promedio,
  es decir 24 por ventana de 6 meses. En cada estudio de régimen se exige
  N_min,g = ceil(24 · D_g / D), con D_g los días del régimen g en la ventana y D los días totales.
  Las configuraciones que no cumplen se descartan.
  - Si un régimen ocupa menos de 21 días de la ventana, no se optimiza por separado: usa el θ único de
    la ventana.
  - Si ningún estudio tiene configuraciones factibles, el activo queda en efectivo en ese régimen
    durante el mes de prueba y se declara en el reporte. El mínimo no se relaja después de ver resultados.

## 8. Break-even win rate
Con E[R] = p·r − (1 − p) − c = 0:

    p* = (1 + c) / (1 + r),     c = 2 · (0.00125 + 0.0002) / (k · ATR₀ / P)

Con k = 2, r = 2 y el ATR/P medio medido en train (2.53%, promedio de los 8 activos; de 1.05% en GLD
a 4.42% en AMD): c ≈ 0.057 y p* ≈ 35.2%.

Las salidas por señal y por holding hacen que R deje de ser +r o −1. Por eso también se reporta, ex post
y por bloque, p* = 1 / (1 + payoff), con el payoff = ganancia media / pérdida media neta de costos.

**Métricas de exposición:** porcentaje de días con posición abierta (por activo y del portafolio),
operaciones por mes y salidas por motivo (señal, stop, take-profit, holding máximo).

## 9. Pendientes de calibrar con train
| Punto | Qué se mide | Criterio |
|---|---|---|
| ~~Lista de activos~~ | RESUELTO en v1.2 (punto 1) | Correlación entre ellos se mide en train y se reporta |
| ~~Actividad mínima (punto 7)~~ | RESUELTO: 46.9 operaciones por ventana de 6 meses | Se mantiene 24 |
| ~~Redundancia SMA–MACD (punto 2)~~ | RESUELTO: correlación −0.135 | Se mantiene MACD |
| ~~Break-even (punto 8)~~ | RESUELTO: ATR/P medio 2.53% | p* = 35.2% |

Toda calibración usa solo train y se registra abajo antes de correr validation.

### Calibraciones con train (2026-10-06)
Valores base del punto 2, θ único, régimen ignorado, Risk Parity y costos completos, sobre train
(2018–2021):

- **Actividad mínima:** 375 operaciones cerradas en 4 años, 46.9 por ventana de 6 meses; el mínimo
  de 24 (1 por activo cada dos meses, sumando los 8) se cumple con holgura y no se cambia. El mínimo
  escala con el largo de la ventana: 192 en los estudios de diagnóstico sobre los 4 años de train, y
  crece con la ventana anchored.
- **Redundancia SMA–MACD:** correlación de los votos de −0.135 con los 8 activos juntos (entre −0.191
  y −0.100 por activo), lejos de 0.7. El histograma MACD (MACD − señal) mide la aceleración del
  momento, no su dirección, así que no repite al cruce de SMA. Se mantiene MACD.
- **Break-even:** ATR/P medio de 2.53%, así que c ≈ 0.057 y p* ≈ 35.2% (punto 8). Ex post, con las
  salidas por señal y por holding, el payoff fue 1.30 y el p* empírico 43.4%, contra un win rate
  observado de 44.3%: la estrategia base queda justo por encima del equilibrio en train.

## Registro de cambios
| Versión | Fecha | Cambio | Justificación |
|---|---|---|---|
| 1.0 | 2026-10-04 | Primera versión, combinación de los cuatro specs del equipo | — (previo al backtest) |
| 1.1 | 2026-10-04 | Compuerta y fuerza según el material de Risk Parity del curso (|Σx| ≥ 2, s = Σx/3); C_i a partir de w^target; valores base; variante anchored; purga y embargo; θ único de referencia | Alinear con los materiales del profe, previo al backtest |
| 1.2 | 2026-10-04 | Lista de los 8 activos (acciones y ETFs) | Definida por el equipo antes de descargar datos |
| 1.3 | 2026-10-06 | Calibraciones del punto 9 con train: actividad mínima, SMA–MACD y break-even; el mínimo de operaciones escala con el largo de la ventana | Medidas solo con train, antes de correr validation |
