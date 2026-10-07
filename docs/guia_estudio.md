# Guía de estudio y defensa — Lab 02 · Equipo 3 · Nivel C

Guía para estudiar el proyecto y responder preguntas en la presentación. Todas las cifras salen de la
corrida final (`results/`, commit congelado `45bedfa`) y coinciden con `notebooks/resultados.ipynb`
y `docs/reporte.pdf`.

**Cómo usarla:** la sección 1 es el resumen que hay que saberse de memoria. Las secciones 2 a 4
explican qué hicimos y cómo funciona. La 5 explica los cambios y por qué los hicimos. La 6 explica
los resultados y a qué se deben. La 7 tiene las 7 preguntas del lab, y la 8 las preguntas que
probablemente haga el profe, con su respuesta. Al final hay una hoja de cifras clave.

---

## 1. El proyecto en un minuto

**Qué hicimos.** Una estrategia direccional larga y corta sobre 8 activos de EE.UU. con barras
diarias (2017 a septiembre de 2026).
- **Entrada:** se abre posición solo cuando al menos 2 de 3 indicadores coinciden (SMA, MACD y RSI).
- **Salidas:** el ATR fija el stop-loss y el take-profit; también se sale por señal opuesta o por
  holding máximo.
- **Asignación:** el capital se reparte con Risk Parity, escalado por la fuerza de la señal y por el
  régimen de mercado (tendencia, reversión o crisis).
- **Optimización:** Optuna en un walk-forward de 6 meses de entrenamiento y 1 mes de prueba.

**Cómo lo hicimos.** Con disciplina de backtesting:
- todo es causal y está verificado con pruebas de truncamiento;
- todas las decisiones se tomaron con train;
- validation se usó una sola vez, para decisiones con reglas fijadas antes de verla;
- test se corrió una sola vez, con el código congelado y el hash del commit registrado.

**Qué salió.** La estrategia **no tiene ventaja fuera de muestra**: −0.4% anual en validation y
+0.7% en test, contra 7.8% y 38.4% de buy & hold. A cambio, su drawdown es de 2 a 6 veces menor.

**Lo que sí funciona:**
- la regla 2 de 3 frente a un indicador solo;
- el θ por régimen, que le gana al θ único en los tres bloques;
- Risk Parity, que iguala el riesgo entre activos y reduce el drawdown.

**Lo que no funciona:** el edge in-sample. La eficiencia del walk-forward es 0.15: solo sobrevive el
15% de la ventaja.

**Mensaje para defender:** *el valor del proyecto está en el proceso. Construimos un sistema que no
se engaña a sí mismo, y por eso podemos afirmar con confianza que esta estrategia no tiene edge
después de costos en este universo. Un backtest con look-ahead o con validation "tuneado" habría
mostrado resultados bonitos y falsos.*

---

## 2. El SPEC: cómo está conformada la estrategia

Los documentos son `instrucciones/SPEC.md` (v1.4) y `instrucciones/SPEC_portafolio.md` (v0.6).
Todos los valores fijos viven en el diccionario `CONFIG` de `main.py`; ningún módulo de `src/` los
escribe a mano.

### 2.1 Universo y datos

| Ticker | Activo | Por qué está |
|---|---|---|
| AAPL, MSFT, META | Mega caps de tecnología | Muy líquidas, spreads mínimos |
| AMD | Semiconductores | La más volátil (ATR/P de 4.4%) |
| SMH | ETF de semiconductores | Diversificación sectorial; contiene a AMD |
| XOM | Energía | Sector distinto, ciclo distinto |
| GLD | Oro | Refugio, baja correlación con acciones |
| COPX | Mineras de cobre | Ciclo industrial; el menos líquido |

- **Selección:** los cuatro elegimos los 8 activos **en conjunto**, con criterios comunes: liquidez,
  historia completa desde antes de 2016 y sectores distintos. No hubo reparto por integrante.
- **Datos:** Yahoo Finance, diarios, ajustados por splits y dividendos (`auto_adjust=True`). La tasa
  libre de riesgo es `^IRX` (T-Bill de 13 semanas), convertida a tasa diaria con ÷ 100 ÷ 252.
- **Auditoría:** 2,449 días comunes, 0 NaN, 0 precios no positivos, 0 incoherencias OHLC y 0 fechas
  descartadas. Nunca se rellena un hueco (prohibido `ffill` y `bfill`).
- **Concentración declarada:** 5 de los 8 activos son tecnología o semiconductores.

### 2.2 Bloques

| Bloque | Fechas | Para qué |
|---|---|---|
| Calentamiento | 2017 | Solo calienta indicadores; no genera señales |
| Train | 2018-01-01 → 2021-12-31 | Diseño, diagnóstico, calibraciones y todas las decisiones de diseño |
| Validation | 2022-01-01 → 2023-12-31 | Una corrida para decisiones discretas: estimador de covarianza, rebalanceo y método de régimen |
| Test | 2024-01-01 → 2026-09-30 | Una sola corrida con todo congelado |

**Por qué ese orden:** test es lo más reciente y simula operar "en vivo". Validation ya está
contaminado, porque con él se eligió entre opciones. Test es la única medida honesta.

### 2.3 Indicadores y señales

| Indicador | Familia | Voto | Rango de búsqueda | Base | θ* (final) |
|---|---|---|---|---|---|
| Cruce de SMA | Tendencia | x_sma = sgn(SMA_f − SMA_s) | f ∈ [5, 30], s ∈ [40, 120] | 20 / 50 | **19 / 64** |
| Histograma MACD | Momento | x_macd = sgn(MACD − señal) | Fijo | 12 / 26 / 9 | 12 / 26 / 9 |
| RSI de Wilder | Momento | +1 si 50 < RSI < hi; −1 si lo < RSI ≤ 50; 0 si no | n ∈ [7, 21], lo ∈ [20, 40], hi ∈ [60, 80] | 14 / 30 / 70 | **20 / 39 / 65** |
| ATR de Wilder | Volatilidad | No vota; fija SL y TP | Fijo | 14 | 14 |

- Mientras un indicador no tiene datos suficientes, vota 0.
- RSI y ATR usan el suavizado de Wilder: `ewm(alpha=1/n, adjust=False)`.
- **Dos familias** (tendencia y momento), como pide el lab. El ATR es de volatilidad, pero no vota.
- **El RSI se usa como momento, no como reversión.** Vota +1 cuando el RSI está arriba de 50 y −1
  cuando está abajo, salvo en los extremos (sobrecompra arriba de hi, sobreventa abajo de lo), donde
  vota 0 para no entrar en movimientos agotados.

### 2.4 Regla de confirmación 2 de 3 (la fórmula)

Con Σx = x_sma + x_macd + x_rsi:

```
s_t = Σx / 3   si |Σx| ≥ 2        Estado_t = sgn(s_t) ∈ {−1, 0, +1}
      0        en otro caso
```

- **Estado** es la compuerta: decide si hay posición y de qué lado.
- **Fuerza** s ∈ {−1, −2/3, 0, 2/3, 1} es la convicción: escala el capital. 3 de 3 pesa 1; 2 de 3
  pesa 2/3.
- **Casos:** (+1, 0, 0) → 0; (+1, +1, 0) → +1 con fuerza 2/3; (+1, +1, −1) → 0 (Σx = 1 no abre);
  (−1, −1, −1) → −1 con fuerza −1. Estos casos están en `tests/test_signals.py`.

### 2.5 Salidas

Con ATR₀ congelado al abrir y L = ±1 el lado:

```
SL = E − L · k · ATR₀        TP = E + L · r · k · ATR₀
```

| Parámetro | Qué es | Rango | Base | θ* del diagnóstico |
|---|---|---|---|---|
| k (`k_stop`) | Distancia al stop en ATRs | [1, 4] | 2 | 3.81 |
| r (`reward_ratio`) | TP como múltiplo de la distancia al stop | [1, 4] | 2 | 3.64 |
| m (`max_holding`) | Holding máximo en barras | [5, 40] | 20 | 40 |

En el walk-forward, k, r y m se reoptimizan por régimen en cada ventana (ver 2.9).

- **Señal opuesta:** si el estado cambia al lado contrario, se cierra al Open de t+1 y se abre el
  otro lado en ese mismo Open. Un estado 0 **no** cierra la posición.
- **Holding máximo:** la barra de entrada cuenta como la primera; con m barras cumplidas se sale al
  Open de la barra m+1.
- **Rearme:** tras un TP, el lado queda armado (la tendencia funcionó y sigue). Tras un SL o un
  holding máximo, queda desarmado hasta que el estado cambie (evita cadenas de stops en una tendencia
  que falló).
- **Al abrir se congelan** los parámetros (k, r, m) y ATR₀. Si cambia el régimen o la ventana, la
  posición conserva los suyos.

### 2.6 Convenciones de ejecución (orden dentro de la barra t+1)

1. Salidas por señal o por holding, al Open.
2. Entradas al Open.
3. Gaps: si el Open ya rebasó el SL, se llena **al Open**. Si rebasó el TP, se llena **en el nivel
   del TP**, sin mejora (supuesto conservador).
4. SL y TP intrabarra con High y Low. **Si los dos caen en la misma barra, primero el stop**
   (conservador: no sabemos qué tocó primero).
5. Valuación al Close.

- **Una posición por activo**, sin pirámides.
- **Fin de muestra o de ventana:** las posiciones abiertas se valúan a mercado y no cuentan como
  operación cerrada.

### 2.7 Asignación y tamaño (SPEC punto 5, v1.4)

```
w̃_i = w_i^RP · s_i
w^target = m(régimen) · w̃ / max(1, Σ|w̃_i|)
C_i = |w_i^target| · Equity_t          (capital asignado al activo i, al cierre de la señal)
acciones = C_i / (E · (1 + comisión))   (cada entrada usa todo su capital asignado)
```

- **Sin apalancamiento:** Σ|w^target| ≤ 1, porque Σw^RP = 1, |s| ≤ 1 y m ≤ 1.
- **m(régimen):** tendencia 1.0, reversión 0.7 y crisis 0.3.
- **Benchmark de pesos iguales:** w^RP se sustituye por 1/8, con las mismas señales, costos y
  rebalanceo.
- **Activo individual:** C = Equity.

### 2.8 Costos

| Concepto | Valor | Justificación |
|---|---|---|
| Comisión | 0.125% del nocional por lado | Parámetro del lab |
| Slippage | 2 bps en contra en todo llenado (incluye SL y TP) | Spreads de ~1 bp en mega caps y ~4.5 bps en promedio del S&P 500 |
| Borrow | 0.25% anual sobre el nocional en corto, diario | Tasa "General Collateral" ilustrativa |
| Ida y vuelta | 2 × (0.125% + 0.02%) = **29 bps** | |

### 2.9 Optimización y walk-forward

- **Objetivo:** Calmar = retorno anualizado / |max drawdown|, calculado con costos, como en el
  curso.
- **Actividad mínima:** 24 operaciones cerradas por cada 6 meses de ventana (1 por activo cada dos
  meses, sumando los 8 activos). Escala con el largo de la ventana (192 en el diagnóstico de 4 años)
  y se prorratea por los días de cada régimen. Por debajo del mínimo, la prueba vale −inf.
- **Estudios de diagnóstico** sobre los 4 años de train:
  - 200 de random search y 200 de TPE, con semilla 42, en 8 dimensiones (f, s, n, lo, hi, k, r, m);
  - se elige **θ\*** como el medoide del mejor 10% de las pruebas factibles: el centro de la meseta,
    no el máximo.
- **Walk-forward rolling (el principal):**
  - 6 meses de entrenamiento, 1 de prueba y paso mensual (requisito del nivel C); 99 ventanas en
    total;
  - **los indicadores quedan fijos en θ\***; en cada ventana se optimizan **k, r y m por régimen**,
    con 150 pruebas TPE por estudio;
  - **purga:** al final de la ventana de entrenamiento, las posiciones se valúan a mercado;
  - **embargo:** no se abren posiciones en los últimos 5 días;
  - **semilla:** la ventana i usa 42 + i;
  - si un régimen tiene menos de 21 días en la ventana, usa el θ único. Si no hay ningún θ factible,
    ese régimen queda en efectivo (nunca pasó).
- **Variantes de diagnóstico:** anchored (ventana creciente) y θ único (sin régimen).
- **Backtest final:** continuo; cada mes fuera de muestra opera con el θ de su ventana, y las
  posiciones pasan de un mes al siguiente.

### 2.10 Régimen de mercado (SPEC_portafolio, sección Régimen)

- **Serie:** índice equiponderado de los 8 activos, con log-retornos ℓ_t = ln(1 + media de los
  retornos simples).
- **Variables** (ventana de 63 días, solo hacia atrás):
  - `volatility` = std(ℓ) · √252;
  - `efficiency` = |Σℓ| / Σ|ℓ| (razón de Kaufman: 1 es línea recta, 0 es ruido puro);
  - `autocorr` = correlación de lag 1.
- **Método elegido: reglas.**
  - Crisis si la volatilidad está arriba de su cuantil 0.80.
  - Si no, tendencia si la eficiencia está arriba de su cuantil 0.50.
  - Si no, reversión.
- **Ajuste:** se reajusta el primer día hábil de cada mes con ventana expandible (todos los datos
  anteriores). La etiqueta en t solo usa datos hasta t.
- **Por qué log-retornos aquí** (el único lugar del proyecto que los usa): la eficiencia necesita que
  la suma de la ventana sea el movimiento total, Σℓ = ln(P_t / P_t−63), y eso solo se cumple con
  logaritmos.

### 2.11 Risk Parity (SPEC_portafolio, sección Portafolio)

- **Formulación convexa de Spinu:** min_y ½ yᵀΣy − (1/n) Σ ln y_i, con w = y / Σy.
  - Es convexa porque Σ es semidefinida positiva y −ln es convexa, así que tiene solución única.
  - −ln(y) → ∞ cuando y → 0 y mantiene los pesos positivos sin restricciones.
  - Se resuelve con L-BFGS-B y se pule con Newton (tolerancia 1e-10).
- **Covarianza:** Ledoit-Wolf sobre 126 días de **retornos simples** hasta t (nunca precios). Son
  retornos simples porque el retorno del portafolio es lineal en ellos: σ²_p = wᵀΣw y las RC son
  exactas.
- **Contribución al riesgo:** RC_i = w_i (Σw)_i / σ_p, y Σ RC_i = σ_p exacto por el teorema de Euler.
- **Rebalanceo híbrido:** el primer día hábil de cada mes se calcula el w^RP candidato y se adopta
  solo si ‖Δw‖₁ > δ = 0.05. La fuerza s y m(régimen) se actualizan a diario.
- **Conflictos:** si ρ_ij > 0.7 (ventana de 126 días) y las señales son opuestas, se queda la de
  mayor |s|; en empate, las dos pasan a 0. Evita pagar costos dobles por posiciones que se anulan.
- **`resize_on_rebalance = False`:** el rebalanceo solo cambia el tamaño de las entradas nuevas; las
  posiciones abiertas no se redimensionan.

---

## 3. Cómo está construido (arquitectura)

| Módulo | Parte | Qué hace |
|---|---|---|
| `src/data.py` | P1 | `load_prices`, `load_risk_free`, `audit_prices`, `block_dates`; `download_prices` es la única función con red |
| `src/backtest.py` | P1 | `run_backtest`: motor event-driven con loop explícito, cash, posiciones y equity; `market_impact` ex post |
| `src/signals.py` | P2 | Indicadores, votos, compuerta 2 de 3, fuerza y señales por régimen |
| `src/metrics.py` | P2 | Sharpe, Sortino, Calmar, MDD, win rate, payoff, tablas de retornos, métricas por bloque, buy & hold |
| `src/optimize.py` | P2 | Estudios de Optuna, meseta, walk-forward, eficiencia, sensibilidad, costos, superficie |
| `src/regimes.py` | P3 | Variables, reglas, K-means y HMM, etiqueta filtrada y validación |
| `src/portfolio.py` | P4 | Covarianzas, Spinu, RC, agregación, rebalanceo, conflictos y decisiones de validation |
| `src/plots.py` | P3 | Una función por figura, todas con título, ejes y leyenda |
| `main.py` | P1 | `CONFIG` y el pipeline completo, sin red ni intervención |

**Flujo de cada backtest:**
1. `generate_signals`: estado, fuerza y ATR por activo.
2. `sleeve_weights`: panel |w^target|.
3. `run_backtest`: ejecuta en t+1.
4. Métricas.

---

## 4. Cómo garantizamos que no hay look-ahead (causalidad)

Es lo primero que revisa un profe de backtesting. Las reglas:

- **Un solo lugar desplaza t → t+1:** `run_backtest`. Ningún otro módulo hace `shift` para simular
  ejecución.
- **Prohibido:** `rolling(center=True)`, `shift(-k)`, `bfill`, normalizar con estadísticas de toda la
  muestra y ajustar modelos con datos posteriores a su ventana.
- **Régimen filtrado:** el HMM (que comparamos) usa la recursión *forward* escrita a mano. `predict`
  (Viterbi) y `predict_proba` (suavizada, forward-backward) usan el futuro; Viterbi solo aparece en
  una figura comparativa.
- **Covarianza en t:** solo retornos hasta t.
- **Walk-forward:** la ventana de prueba nunca participa en la optimización, con purga y embargo.

**Pruebas automatizadas** (173 con pytest, todas con datos sintéticos, sin red):

| Prueba | Qué demuestra |
|---|---|
| Truncamiento de indicadores y señales | El valor en t calculado con `df.iloc[:t+1]` es idéntico al de la serie completa |
| Truncamiento del pipeline completo | Indicadores → señales → pesos → backtest: el equity en t no cambia al agregar datos posteriores |
| Truncamiento del régimen | Variables y etiqueta filtrada en t no cambian con datos futuros |
| Truncamiento de los pesos | `sleeve_weights` en t no cambia con datos futuros |
| Contabilidad | equity = cash + Σ acciones · close en todas las fechas (tol. 1e-6); comisiones = Σ nocional × 0.125% |
| 9 golden tests | Equity final calculado a mano en caminos de pocas barras: empate SL/TP, gap en SL, gap en TP, rearme, holding máximo, reversa, tamaño, borrow, t → t+1 |
| Risk Parity | max(RC) − min(RC) < 1e-6 y Σ RC = σ_p |
| Confirmación | Los 4 casos de la regla 2 de 3 |

---

## 5. Cambios que hicimos y por qué

Todos quedaron registrados en el changelog de cada SPEC y en git. La regla general: **todo cambio se
justificó con train.** Validation solo se usó para decisiones con reglas fijadas y registradas en git
*antes* de verla.

### 5.1 Cambios técnicos (no cambian la estrategia)

| Cambio | Por qué |
|---|---|
| Retornos simples en la covarianza de Risk Parity y en las correlaciones (antes log) | El retorno del portafolio es lineal en retornos simples (r_p = Σ w_i r_i), así que σ²_p = wᵀΣw y las RC son exactas. El régimen se quedó en log por la aditividad de la eficiencia y para no cambiar variables después de ver validation |
| `sleeve_weights` vectorizado y caché de w^RP | Cada prueba de Optuna tardaba 0.67 s; ahora 0.13 s. Con unos 130 mil backtests, el walk-forward pasó de ~16 horas a ~38 minutos. El resultado es idéntico (verificado a 1e-15) |
| Paralelismo con procesos, no threads | El loop del motor es Python puro y con threads el GIL lo serializa |
| Ventana sin θ factible → efectivo (antes el programa tronaba) | El SPEC lo pedía así; perder horas de cómputo por una ventana era inaceptable |
| Actividad mínima escalada al largo de la ventana | El diagnóstico de 4 años exigía solo 24 operaciones (debían ser 192) y el anchored no escalaba |
| Sensibilidad, costos y pregunta 1 medidos sobre train (antes sobre toda la muestra) | Antes mezclaban validation y test, lo cual es una fuga de información |
| Bloques medidos desde el cierre anterior | Antes se perdía el retorno del primer día de cada bloque |
| `final_run` en `CONFIG` | Para correr todo sin tocar test (False) y luego la corrida única con test (True) |

### 5.2 Calibraciones de SPEC punto 9 (v1.3, con train)

| Qué | Medición en train | Decisión |
|---|---|---|
| Actividad mínima | 46.9 operaciones por ventana de 6 meses con los valores base | Se mantiene 24 |
| Redundancia SMA–MACD | Correlación de votos de **−0.135** (−0.19 a −0.10 por activo) | Se mantiene MACD; el umbral para cambiarlo era 0.7 |
| Punto de equilibrio | ATR/P medio de 2.53% → c ≈ 0.057 y p* = (1 + c)/(1 + r) ≈ 35.2% con k = r = 2 | Se registra |

**Por qué la correlación es negativa:** el histograma MACD mide la *aceleración* del momento
(MACD − señal), no su dirección. Cuando una tendencia madura, la SMA sigue en +1 y el histograma ya
voltea a −1. Por eso no repiten información.

### 5.3 Decisiones de P4 con train (SPEC_portafolio v0.5)

| Decisión | Evidencia en train | Resultado |
|---|---|---|
| Estimador de Σ | Desviación media del peso: Ledoit-Wolf 0.0230, muestral 0.0275, EWMA 0.0274 | Ledoit-Wolf; otro solo lo reemplaza si es ≥ 10% más estable en validation |
| m(régimen) | Con m = 1 y θ base: Sharpe 1.18 en tendencia, −0.83 en reversión, −0.23 en crisis (peor MDD) | Se confirman 1.0 / 0.7 / 0.3, los valores del curso, sin ajustarlos a la tabla |
| Umbral de conflictos | AAPL-MSFT 0.76, MSFT-SMH 0.76, AAPL-SMH 0.72 | Se confirma ρ > 0.7 (actúa sobre el trío casi idéntico) |
| Qué se rebalancea | — | Solo w^RP; s y m a diario, para que una señal entre en t+1 |

### 5.4 SPEC v1.4: los dos cambios grandes (con train)

**1. Sizing: de fixed fractional a nocional = C_i.**
- **Antes:** acciones = ρ · C_i / |E − SL|, con ρ = 1%.
- **Problema 1, doble escalamiento:** C_i ya es una fracción del equity (~1/8 × fuerza × m), y
  encima solo se arriesgaba el 1% de esa fracción. La exposición bruta media en train era de
  **10%**, con volatilidad anual de 1.3% y retorno casi 0. La estrategia estaba 90% en efectivo.
- **Problema 2, ρ no se puede optimizar con Calmar:** retorno y drawdown escalan igual con el tamaño,
  así que el Calmar no cambia. En la sensibilidad, ρ ±20% movió el Calmar 0.003.
- **Ahora:** acciones = C_i / (E · (1 + comisión)). El tamaño lo deciden Risk Parity, la fuerza y el
  régimen; el stop solo define la salida. La exposición media pasó a ~31–39%. Las 9 golden tests se
  recalcularon a mano.

**2. Walk-forward con menos dimensiones.**
- **Problema:** con 9 parámetros por régimen y solo 6 meses de datos, en las ventanas de train el
  Calmar IS mediano era **7.6** contra **0.8** OOS. Los parámetros de indicadores recorrían todo su
  rango de un mes al siguiente: puro sobreajuste.
- **Ahora:** los indicadores se fijan en θ\* (4 años de train) y por ventana solo se optimizan k, r y
  m por régimen. Los indicadores describen la señal y necesitan más datos; las salidas se adaptan al
  régimen.

**3. Lo que NO cambiamos, y por qué: los cortos.**
- En train los cortos perdían (−84k contra +116k de los largos) y long-only mejoraba el Sharpe.
- **No lo hicimos:** sabemos (estamos en 2026) que el universo siguió subiendo después de 2021.
  Quitar los cortos mezclaría sesgo de supervivencia con sesgo de retrospectiva.
- Se mantuvo el diseño original y se reporta como observación.

**Importante para defender:** validation ya se había visto en una corrida previa cuando se hizo v1.4,
pero los cambios se justificaron con cifras de train, y así quedó escrito en el changelog. Test no se
había tocado.

### 5.5 Decisiones de validation (corrida única, SPEC_portafolio v0.6)

Las reglas se escribieron y se subieron a git **antes** de correr validation:

| Decisión | Regla | Resultado |
|---|---|---|
| Estimador | Gana el más estable, pero Ledoit-Wolf solo se reemplaza si otro es ≥ 10% mejor | Ledoit-Wolf 0.0145 contra 0.0161 (muestral) y 0.0162 (EWMA): **Ledoit-Wolf** |
| Frecuencia y δ | Se mantiene mensual con 0.05 salvo +1 pp anual de retorno neto; si varias superan, gana la de menor turnover | Mejor alternativa: Q · 0.2, con +0.19 pp. **Se mantiene M · 0.05** |
| Método de régimen (P3) | Silhouette en validation, con desempate por menos transiciones | **Reglas** |

### 5.6 Protocolo de la corrida final

1. Merge de todo a `main`.
2. Commit de congelamiento `45bedfa`, que solo cambia `final_run = True`.
3. Corrida única con test desde ese commit (38.7 minutos).
4. `results/corrida.pkl` registra el mismo hash. Marca `uncommitted_changes = True` solo por las
   figuras que la propia corrida escribió; ningún archivo con seguimiento cambió.
5. Commits posteriores: solo resultados y documentación.

---

## 6. Resultados y a qué se deben

### 6.1 Backtest final fuera de muestra (walk-forward rolling por régimen)

| | Train (jul-2018 a 2021) | Validation (2022–2023) | Test (2024–sep 2026) |
|---|---|---|---|
| **Risk Parity**: retorno anual | 4.0% | −0.4% | 0.7% |
| Volatilidad | 5.1% | 5.8% | 7.2% |
| Sharpe | 0.59 | −0.65 | −0.46 |
| Calmar | 0.82 | −0.06 | 0.07 |
| Max drawdown | −4.8% | −6.2% | −10.7% |
| Operaciones · win rate | 280 · 47.9% | 145 · 46.2% | 224 · 45.5% |
| **Pesos iguales**: retorno · Calmar · MDD | 5.1% · 0.89 · −5.7% | −0.4% · −0.06 · −7.7% | 0.1% · 0.01 · −14.8% |
| **Buy & hold**: retorno · Calmar · MDD | 40.0% · 1.32 · −30.3% | 7.8% · 0.31 · −25.1% | 38.4% · 1.81 · −21.3% |

- **Capital:** 1,000,000 → 1,159,068 USD en 8.25 años. Buy & hold lo multiplicó por 10.
- **Por año (Risk Parity):** 2018 +2.2%, 2019 +6.7%, 2020 +3.5%, 2021 +1.5%, 2022 −2.8%, 2023
  +2.1%, 2024 +2.0%, 2025 +4.6% y 2026 −4.4% (hasta septiembre).

**A qué se debe:**
1. **Edge muy delgado frente a los costos.**
   - El win rate observado (45.5–47.9%) apenas supera el de equilibrio empírico (43.7–44.0%), que se
     calcula con el payoff real de ~1.28.
   - Los costos cuestan 1.4–2.2% del equity al año. Sin costos, el resultado fuera de muestra sería
     de ~1% anual en validation y ~3% en test.
2. **La optimización de 6 meses no generaliza:** eficiencia de 0.15 (pregunta 2).
3. **Universo alcista:** 2018–2026 fue un mercado muy alcista para tecnología. La estrategia tiene en
   promedio ~65% del capital en efectivo y tiene cortos, que restaron en los tres bloques (en test, cortos
   −145k contra largos +187k).
4. **Train está inflado:** los indicadores se fijaron con todo train, así que el tramo de train del
   walk-forward no es del todo fuera de muestra. **Los números que cuentan son los de validation y
   test.**

**Por qué el Sharpe es negativo aunque test tenga retorno positivo:** el Sharpe usa el exceso sobre
la tasa libre de riesgo. El T-Bill promedió 1.0% en train, 3.5% en validation y **4.3% en test**.
Ganar 0.7% cuando el T-Bill paga 4.3% da un Sharpe negativo.

### 6.2 Estrategia contra buy & hold a igual riesgo

Escalando los retornos de buy & hold a la volatilidad de la estrategia (es ex post, no operable):
Sharpe de 1.14 contra 0.59 en train y de 0.68 contra −0.46 en test. **Buy & hold gana aun a igual
riesgo**, así que la diferencia no es solo la menor exposición: la estrategia no captura la deriva
alcista y no tiene edge que la compense.

### 6.3 Salidas y comportamiento

| Salida | Operaciones | P&L neto |
|---|---|---|
| Holding máximo | 324 | +612k (+1,889 por operación) |
| Target | 88 | +792k |
| Stop | 202 | −1,115k |
| Señal opuesta | 35 | −128k |

- Hay alguna posición abierta 81% de los días, con 5.5 operaciones por mes y un holding medio de ~19
  días calendario.
- Domina la salida por holding máximo: el θ\* tiene k = 3.8 y r = 3.6, con stops lejanos y targets
  aún más lejos, así que muchas operaciones se cierran por tiempo antes de tocar cualquiera de los
  dos.

### 6.4 Activos individuales (C = Equity con las mismas señales)

- Ningún activo es consistente en los tres bloques.
- AAPL es el único positivo en todos (27.7%, 10.9% y 5.0%), pero con drawdowns de 17% a 35%.
- **El portafolio diversifica:** su drawdown (−4.8% a −10.7%) es mucho menor que el de cualquier
  activo solo.

### 6.5 Régimen

| | Train | Validation | Test |
|---|---|---|---|
| % tendencia / reversión / crisis | 40 / 28 / 32 | 30 / 19 / 51 | 49 / 40 / 11 |
| Duración media (días) | 12.4 | 19.3 | 12.5 |
| Transiciones por mes | 1.67 | 1.04 | 1.64 |
| Silhouette | 0.13 | 0.34 | 0.30 |

- **Persistencia:** cumple la meta de duración (≥ 10 días).
- **Silhouette:** no supera la meta de 0.4. Las variables vienen de ventanas traslapadas y cambian de
  forma continua, así que no forman grupos compactos. Se reporta tal cual.
- **2022 fue sobre todo crisis:** el bajista de 2022 superó el cuantil 0.80 de la volatilidad
  histórica.

---

## 7. Las 7 preguntas del lab (respuestas para la presentación)

### P1. ¿Qué aporta la regla 2 de 3 frente a un solo indicador?

En train, con el mismo θ\*:

| Estrategia | Operaciones | Calmar |
|---|---|---|
| **2 de 3** | **202** | **1.08** |
| SMA sola | 140 | 0.30 |
| RSI solo | 673 | 0.04 |
| MACD solo | 561 | −0.07 |

**Respuesta:** la confirmación filtra señales de baja calidad. Recorta 64% de las operaciones de MACD
y 70% de las de RSI, que cambian de voto con frecuencia y pagan 29 bps cada ida y vuelta. Además
triplica el Calmar del mejor indicador solo. SMA opera poco, pero sin confirmación de momento entra
tarde. Los votos no son redundantes (correlación SMA–MACD de −0.135).

### P2. ¿Cuánto se degrada entre entrenamiento y prueba? ¿Qué proporción sobrevive?

- **Eficiencia** = retorno anualizado OOS concatenado / promedio del retorno IS de las ventanas =
  **0.15** (0.03 con Calmar). **Sobrevive ~15% de la ventaja.** El curso dice que debajo de ~0.5 el
  resultado IS es mayormente ruido ajustado.
- **Calmar mediano por ventana:** en validation, 2.5 IS contra −0.8 OOS; en test, 3.4 IS contra
  −0.6 OOS.
- **Otras variantes:** las tres tienen eficiencia negativa.
- **Rolling contra anchored:** anchored no lo arregla (Calmar OOS mediano de −2.4 en test). Según el
  curso, la diferencia entre ambas es en sí un diagnóstico: aquí el edge **decae** en el tiempo, no
  es estable.
- **Esfuerzo:** se evaluaron 131,850 configuraciones en 38 minutos de optimización.

### P3. ¿Qué tan sensible es a ±20%? ¿Meseta o pico aislado?

**Meseta moderada, no un pico aislado**, con una excepción. Sobre un Calmar base de 1.08:
- k y r: ±20% mueve el Calmar entre −0.10 y +0.04;
- la mediana de las 16 variaciones conserva 91% del Calmar, y 14 de 16 conservan más de la mitad;
- lo frágil son las medias móviles: `sma_slow` ×1.2 da −0.65, `max_holding` ×0.8 y `rsi_hi` ×0.8
  dan −0.44, y `sma_fast` da −0.28 y −0.38.

Esto respalda elegir el centro de la meseta y no el máximo. Matiz para defender: la meseta es
in-sample, y la P2 muestra que aun así no se sostiene fuera de muestra.

### P4. ¿A qué costo deja de ser rentable? ¿Qué margen hay frente a 0.125%?

- **En train con θ\*:** es rentable en todo el barrido de 0 a 100 bps de ida y vuelta (13.1% anual
  sin costos, 11.2% con ~30 bps y 6.9% con 100 bps). El margen frente a los 29 bps base es **mayor a
  71 bps**.
- **Fuera de muestra el margen desaparece:** con costos base da −0.4% (validation) y +0.7% (test),
  mientras los costos son 1.4–2.2% anual. El equilibrio real queda **alrededor de los 29 bps**
  especificados: margen casi nulo.

### P5. ¿El desempeño difiere entre regímenes? ¿Qué aporta la capa?

Sharpe por régimen (walk-forward rolling por régimen):

| Régimen | Train | Validation | Test |
|---|---|---|---|
| Tendencia | 1.44 | −0.67 | 0.45 |
| Reversión | −0.76 | −2.62 | −1.29 |
| Crisis | −0.09 | −0.46 | −0.76 |

- **Sí difiere, y de forma consistente en reversión,** que pierde en los tres bloques: una estrategia
  de seguimiento de señales no funciona en un mercado sin dirección. Tendencia gana en train y test.
- **Matiz:** cada celda tiene entre 78 y 373 días, así que no es estadísticamente robusto.
- **θ por régimen contra θ único:** gana en los tres bloques (train 2.8% contra 1.2%; validation
  −1.4% contra −3.8%; test **+1.8% contra −3.5%**). Adaptar k, r y m al régimen es lo que mejor
  generaliza.
- **m(régimen):** reduce la exposición en reversión y crisis, justo donde se pierde.

### P6. ¿Risk Parity mejora el Calmar frente a pesos iguales? ¿A costa de qué?

| Bloque | Calmar RP / pesos iguales | MDD RP / pesos iguales | Vol RP / pesos iguales |
|---|---|---|---|
| Train | 0.82 / 0.89 | −4.8% / −5.7% | 5.1% / 6.5% |
| Validation | −0.06 / −0.06 | −6.2% / −7.7% | 5.8% / 7.4% |
| Test | **0.07 / 0.01** | −10.7% / −14.8% | 7.2% / 8.8% |

- **El Calmar solo mejora en test,** pero el riesgo baja en los tres bloques: el drawdown es 16% a
  28% menor.
- **Justificación por asignación de riesgo** (la que pide el lab): con Risk Parity cada activo aporta
  12.5% del riesgo (dispersión 0). Con pesos iguales, AMD aporta 23.7% y GLD 1.6% (dispersión 0.22),
  y la volatilidad ex ante es 22.2% contra 16.6%.
- **A costa de:** sobreponderar los activos de baja volatilidad (GLD, XOM) y subponderar AMD y la
  tecnología, que fueron los que más subieron. Por eso sacrifica retorno en train (4.0% contra 5.1%).

### P7. Tres limitaciones para capital real

1. **No hay edge fuera de muestra:** eficiencia de 0.15, retorno por debajo del T-Bill y muy lejos
   de buy & hold.
2. **Sesgos del universo:** supervivencia (activos elegidos en 2026), concentración en tecnología
   (5 de 8, y el régimen refleja sobre todo ese sector) y muestra alcista (los cortos restaron).
3. **Ejecución optimista:** llenado completo al Open, al SL y al TP; slippage fijo de 2 bps; borrow
   fijo sin recall; sin rechazos, ejecuciones parciales ni impacto.

### Advertencia obligatoria: impacto de mercado

- **El supuesto:** el backtest asume ejecución completa al precio modelado y no incorpora impacto de
  mercado ni fallas de ejecución.
- **El modelo de raíz cuadrada (ex post):** impacto ≈ σ_diaria · √(Q / ADV), con Q el nocional del
  llenado, y ADV y σ de 20 días al cierre de la señal (causal).
- **Sobre las 224 operaciones de test:**
  - **1.5 bps** promedio por llenado (mediana 0.8 y p95 6.7);
  - participación máxima de 0.5% del volumen diario;
  - en total, **0.58% del equity**, cerca de **23% del retorno del bloque**;
  - **COPX** concentra el impacto con 6.8 bps (el resto, menos de 1.4).
- **Escala:** crece con √Q. Con 100 veces más capital sería ~10 veces mayor y borraría la ganancia.

---

## 8. Preguntas probables del profe y cómo responderlas

**¿Por qué la estrategia no funciona?**
El edge que encuentra la optimización en 6 meses no se repite el mes siguiente (eficiencia de 0.15).
El poco edge que queda es del tamaño de los costos: el win rate supera al de equilibrio por apenas
1.5 a 4 puntos. Y en un universo tan alcista, tener ~65% del capital en efectivo y operar cortos juega en contra.

**¿No es un fracaso que pierda contra buy & hold?**
El objetivo del lab es construir y evaluar una estrategia con rigor, no ganarle al mercado. Un
resultado negativo bien medido vale más que uno positivo con fugas. Además, en riesgo sí hay
diferencia: el drawdown es de 2 a 6 veces menor. La conclusión honesta es que este conjunto de
señales técnicas no tiene edge después de costos en este universo.

**¿Por qué cambiaron el SPEC? ¿No es data snooping?**
- Los cambios de v1.4 se justificaron con cifras de train: la exposición de 10% y el sobreajuste
  (Calmar IS 7.6 contra OOS 0.8 en las ventanas de train).
- Test nunca se había visto.
- Validation se usó una vez, para decisiones con reglas escritas en git antes de verla; se puede
  demostrar con el orden de los commits.
- Lo que no hicimos fue iterar sobre validation, ni quitar los cortos con información posterior.

**¿Por qué no quitaron los cortos si restaban?**
Porque la decisión estaría contaminada por algo que sabemos fuera de los datos: el universo siguió
subiendo después de 2021. Sería sesgo de retrospectiva. Mantuvimos el diseño original y lo
reportamos.

**¿Por qué el Sharpe es negativo si en test ganan 0.7%?**
El Sharpe usa el exceso sobre la tasa libre de riesgo, y el T-Bill pagó 4.3% en 2024–2026. La
estrategia no le gana al efectivo sin riesgo.

**¿Por qué los resultados de train se ven bien?**
Porque están inflados: θ\* se eligió con los 4 años de train, y esas ventanas del walk-forward usan
indicadores ajustados con sus propios datos. Por eso hay que mirar validation y test.

**¿Por qué Calmar como objetivo?**
Es el del curso (actividad S08). Penaliza el drawdown, que es lo que más le importa a un operador. Su
debilidad: con 6 meses es ruidoso, porque un solo drawdown lo domina. Es una de las mejoras que
proponemos.

**¿Por qué el medoide del mejor 10% y no el máximo?**
El máximo suele ser un pico de ruido. El medoide (la prueba con menor distancia total a las demás del
grupo, con parámetros normalizados a [0, 1]) es el centro de la meseta: un θ rodeado de θ que
también funcionan, menos sensible a pequeños cambios. La P3 lo respalda.

**¿Qué es la purga y el embargo?**
- **Purga:** al final de cada ventana de entrenamiento, las posiciones se valúan a mercado; ninguna
  operación de entrenamiento "ve" el mes de prueba.
- **Embargo:** no se abren posiciones en los últimos 5 días de la ventana, para que las operaciones
  que terminarían en el periodo de prueba no contaminen la evaluación.

**¿Por qué 6 meses y 1 mes en el walk-forward?**
Es el requisito del nivel C. Reconocemos que 6 meses son pocos para estimar parámetros por régimen:
por eso fijamos los indicadores con 4 años y solo reoptimizamos k, r y m.

**¿Cómo eligieron el método de régimen?**
- La regla se fijó antes de ver validation:
  1. se descartan los métodos con duración media menor a 10 días;
  2. gana la mayor silhouette en validation;
  3. si dos quedan a menos de 0.05, gana el de menos transiciones por mes.
- **Resultado:** reglas (silhouette 0.34) y K-means (0.30) empataron dentro de 0.05, y reglas ganó
  por transiciones (0.79 contra 1.25 por mes).
- **Además:** reglas es el único método que marca 2022 como crisis (K-means y HMM no marcan ningún
  día de crisis en validation), y es el más transparente.
- **El HMM:** es demasiado lento para operarlo, con rachas de 167 días en validation.

**¿Qué diferencia hay entre la etiqueta filtrada y Viterbi?**
- **Filtrada:** P(s_t | x_1…x_t), solo datos hasta t, con la recursión forward. Es la única operable.
- **Viterbi:** la secuencia más probable dada toda la muestra; reetiqueta el pasado con el futuro.
- **Suavizada (`predict_proba`):** también usa toda la muestra.
- Si se operara Viterbi o la suavizada, el backtest tendría look-ahead.

**¿Por qué el HMM necesitó reinicios?**
Con un solo inicio, EM quedaba en un óptimo local degenerado: dos estados con la misma media que
alternaban cada día (672 transiciones en 4 años). Con 10 reinicios, quedándose con el de mayor
verosimilitud, bajó a 14 transiciones.

**¿Por qué Ledoit-Wolf?**
- Con 8 activos y 126 días (n/T ≈ 0.06), la covarianza muestral dispersa los eigenvalores de forma
  artificial.
- Ledoit-Wolf encoge hacia un objetivo estructurado con un δ\* analítico, sin hiperparámetros.
- Fue el más estable en train (0.0230) y en validation (0.0145).

**¿Por qué Spinu y no minimizar Σ(RC_i − RC_j)²?**
Esa función no es convexa: tiene regiones planas y depende del punto de partida. La de Spinu es
convexa, con solución única, y se puede correr sin supervisión en cada rebalanceo.

**¿Por qué Risk Parity no mejora el retorno?**
No pretende hacerlo: iguala el riesgo, no maximiza el retorno. Sobrepondera lo poco volátil (GLD) y
subpondera lo volátil (AMD), y en esta muestra lo volátil fue lo que más subió.

**¿Por qué el barrido de rebalanceo casi no cambia nada?**
Con `resize_on_rebalance = False`, cambiar w^RP solo afecta el tamaño de las entradas nuevas, no de
las posiciones abiertas. Las 15 combinaciones quedan en un rango de 0.6 pp. Es una limitación
declarada.

**¿Cómo saben que el motor está bien?**
Por los 9 golden tests con equity calculado a mano, la prueba de contabilidad
(equity = cash + posiciones con tolerancia 1e-6, y comisiones = Σ nocional × 0.125%), la prueba de
inmutabilidad de argumentos y el truncamiento del pipeline completo.

**¿Qué pasa si SL y TP caen en la misma barra?**
Se ejecuta primero el stop. Con barras diarias no sabemos cuál ocurrió primero, así que asumimos el
peor caso.

**¿Y los gaps?**
Un gap más allá del stop llena al Open (peor precio, realista). Un gap más allá del TP llena en el TP,
sin la mejora del gap (conservador).

**¿Por qué el slippage es de 2 bps?**
Los spreads en mega caps rondan 1 bp y el promedio del S&P 500 es ~4.5 bps. Para COPX probablemente
es poco; lo declaramos y el impacto estimado lo confirma (6.8 bps).

**¿Qué es la actividad mínima y por qué?**
Al menos 24 operaciones cada 6 meses (1 por activo cada 2 meses). Evita que la optimización elija
configuraciones que casi no operan: con pocas operaciones, un Calmar alto es suerte. Si no se cumple,
la prueba vale −inf.

**¿Qué mejorarían?**
Todo esto habría que validarlo con datos nuevos, no con este test:
- un universo más amplio y diverso;
- ventanas de entrenamiento más largas;
- un objetivo menos ruidoso que el Calmar de 6 meses (por ejemplo, Sharpe con penalización por
  rotación);
- un multiplicador de reversión menor, fijado de antemano;
- `resize_on_rebalance` para que Risk Parity actúe sobre las posiciones abiertas.

**¿Cómo usaron la IA?**
Claude Code y ChatGPT como asistentes de programación y revisión; el detalle por integrante está en
el README. Las decisiones de diseño las tomó el equipo con evidencia de train. Todo se probó con
pytest.

---

## 9. Hoja de cifras clave

| Concepto | Cifra |
|---|---|
| Activos · días · periodo | 8 · 2,449 · 2017-01 a 2026-09 |
| Costo de ida y vuelta | 29 bps (0.125% + 2 bps por lado) |
| θ\* (indicadores) | SMA 19/64 · RSI 20, 39/65 · MACD 12/26/9 · ATR 14 |
| θ\* (salidas, diagnóstico) | k = 3.81 · r = 3.64 · m = 40 |
| m(régimen) | 1.0 / 0.7 / 0.3 |
| Retorno anual RP (train / validation / test) | 4.0% / −0.4% / 0.7% |
| Max drawdown RP | −4.8% / −6.2% / −10.7% |
| Buy & hold anual | 40.0% / 7.8% / 38.4% |
| Capital final | 1.159 M USD (buy & hold ×10) |
| Eficiencia del walk-forward (retorno / Calmar) | 0.15 / 0.03 |
| Configuraciones evaluadas · tiempo | 131,850 · 38 min |
| P1: operaciones · Calmar (2 de 3 contra MACD) | 202 · 1.08 contra 561 · −0.07 |
| P3: Calmar conservado (mediana) | 91% |
| P4: margen en train · fuera de muestra | > 71 bps · ~0 |
| P5: θ por régimen contra único (test) | +1.8% contra −3.5% |
| P6: RC con RP · pesos iguales | 12.5% c/u · 1.6% a 23.7% |
| Impacto (test) | 1.5 bps por llenado · 0.58% del equity · COPX 6.8 bps |
| Exposición media · tiempo en mercado | 31–39% · 81% |
| Correlación SMA–MACD | −0.135 |
| Estabilidad de pesos (LW / muestral / EWMA, train) | 0.0230 / 0.0275 / 0.0274 |
| Pruebas | 173 en verde |
| Commit congelado | `45bedfa` |
