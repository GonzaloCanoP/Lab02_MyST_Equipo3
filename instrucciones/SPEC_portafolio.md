# SPEC de portafolio — Lab 02 · Equipo 3 · Nivel C

**Versión:** 0.1 · **Estado:** plantilla. P3 completa la sección "Régimen" y P4 la sección
"Portafolio", cada una por PR, ANTES de implementar su módulo. Toda decisión se toma con train.

Complementa a `SPEC.md`. Las fórmulas base vienen del material del curso "Fundamentos Matemáticos de
Risk Parity" y de la actividad de Market Regime Detection.

---

## Régimen (P3)

### Serie de mercado
Índice equiponderado de los 8 activos de `data/`, rebalanceado a diario:

    r_t = (1/8) Σ_i (close_i,t / close_i,t−1 − 1),        ℓ_t = ln(1 + r_t)

- **Por qué:** no requiere datos extra (`data/` está congelado y `main.py` no usa red) y mide el
  régimen del universo que se opera, no el de un índice externo.
- **Limitación declarada:** 5 de los 8 activos son tecnología (`SPEC.md` punto 1), así que el
  régimen refleja sobre todo a ese sector.

### Variables (ventana móvil de 63 días hábiles)
Con w = 63 (`regime_window`) y la suma o el estadístico sobre k = t − w + 1, …, t:

| Variable | Fórmula | Qué captura |
|---|---|---|
| `volatility` | std(ℓ_k) · √252 | Volatilidad: crisis |
| `efficiency` | \|Σ ℓ_k\| / Σ \|ℓ_k\|, en [0, 1] (razón de eficiencia de Kaufman) | Fuerza direccional: 1 es una línea recta, 0 es puro ruido |
| `autocorr` | corr(ℓ_k, ℓ_k−1) | Reversión: negativa indica que los movimientos se revierten |

- Ventanas solo hacia atrás: el valor en t usa ℓ hasta t. El rezago ℓ_k−1 de `autocorr` mira al
  pasado; no es un desplazamiento para simular ejecución (CLAUDE.md, sección 4).
- Mientras no hay w observaciones, la variable es NaN; no se rellena. La primera fecha válida cae en
  abril de 2017, dentro del calentamiento.
- `efficiency` no tiene dirección a propósito: la estrategia opera largo y corto, así que una
  tendencia bajista sirve igual que una alcista. Un retorno acumulado como medida de tendencia
  mezclaría la tendencia con la crisis, porque un crash también es un retorno grande.

### Métodos comparados
Los tres trabajan sobre las variables estandarizadas con la media y la desviación de la muestra de
ajuste: z = (x − μ_ajuste) / σ_ajuste.

- **Reglas:** crisis si `volatility` > su cuantil 0.80 de la muestra de ajuste
  (`regime_crisis_quantile`); si no, tendencia si `efficiency` > su cuantil 0.50
  (`regime_trend_quantile`); si no, reversión.
- **K-means:** k = 3 (`regime_n_states`), `n_init = 10`, `random_state = seed`.
- **HMM:** `GaussianHMM` de hmmlearn, 3 estados, covarianza completa, hasta 200 iteraciones de EM.
  Se ajusta 10 veces (`regime_hmm_n_init`) con `random_state = seed, seed + 1, …` y se queda el de
  mayor verosimilitud en la muestra de ajuste. Con un solo inicio, EM quedó en train en un óptimo
  local degenerado: dos estados con la misma media que alternan cada día (672 transiciones en 4 años
  y log-verosimilitud −3041, contra 14 transiciones y −2432 con los reinicios). La etiqueta operable
  es la FILTRADA, con la recursión forward escrita a mano
  con `startprob_`, `transmat_` y la densidad gaussiana de cada estado, en logaritmos:

      ln α_1(j) = ln π_j + ln b_j(x_1)
      ln α_t(j) = ln b_j(x_t) + logsumexp_i (ln α_t−1(i) + ln A_ij),   normalizado para sumar 1
      ŝ_t = argmax_j α_t(j)

  `predict` (Viterbi) y `predict_proba` (suavizada) usan el futuro: nunca se usan para etiquetar.
  Viterbi solo aparece en la figura comparativa.

**Tabla comparativa:** cada método se ajusta una vez con las fechas de train y etiqueta validation con
el modelo congelado. Se reportan silhouette (sobre z), duración media, transiciones por mes y % de
tiempo en cada régimen, en train y en validation.

**Regla de elección, fijada antes de ver resultados:**
1. Se descartan los métodos con duración media < 10 días hábiles en train o en validation.
2. Entre los que quedan, gana el de mayor silhouette en validation.
3. Si dos quedan a menos de 0.05 de silhouette, gana el de menos transiciones por mes en validation.
4. Si ninguno cumple el paso 1, gana el de mayor duración media en validation y se aplica el filtro de
   duración mínima (abajo).

Test no participa en la elección.

**Resultados (2026-10-05, `compare_regime_methods`; ajuste en train, validation con el modelo
congelado, test sin tocar):**

| Método | Bloque | Silhouette | Duración media (días) | Transiciones/mes | % tendencia | % reversión | % crisis |
|---|---|---|---|---|---|---|---|
| Reglas | train | 0.23 | 11.2 | 1.85 | 41.7 | 38.3 | 20.0 |
| Reglas | validation | 0.34 | 25.1 | 0.79 | 27.7 | 20.4 | 51.9 |
| K-means | train | 0.38 | 17.1 | 1.21 | 31.6 | 61.9 | 6.6 |
| K-means | validation | 0.30 | 16.2 | 1.25 | 19.4 | 80.6 | 0.0 |
| HMM | train | 0.20 | 67.2 | 0.29 | 61.1 | 26.4 | 12.5 |
| HMM | validation | 0.20 | 167.0 | 0.08 | 27.9 | 72.1 | 0.0 |

**Método elegido: reglas** (`regime_method = "rules"`).

**Justificación:**
- **Aplicación de la regla fijada antes de ver resultados:** los tres cumplen la duración mínima
  (paso 1). Reglas tiene la mayor silhouette en validation (0.34), pero K-means queda a 0.04, dentro
  del margen de 0.05 (paso 3). El desempate es por menos transiciones por mes en validation: reglas
  0.79 contra K-means 1.25.
- **Respaldo cualitativo, que no decidió la elección:**
  - K-means y el HMM no marcan ningún día de crisis en 2022–2023: su grupo de crisis quedó
    definido por el crash de marzo de 2020, que 2022 no alcanza. Reglas sí reconoce como crisis el
    mercado bajista de 2022.
  - El HMM es demasiado lento para operarlo: rachas de 167 días en validation, prácticamente un solo
    régimen en dos años.
  - Reglas es el método más transparente: dos umbrales explícitos.
- **Meta no cumplida:** ninguno supera silhouette 0.4. Se reporta tal cual: las variables de ventanas
  traslapadas cambian de forma continua y no forman grupos compactos.
- **% de crisis en validation (51.9%):** con el modelo congelado en train, el umbral de volatilidad
  (cuantil 0.80 de train) queda por debajo de la volatilidad de 2022. En la operación, `label_regimes`
  reajusta cada mes con ventana expandible, así que el umbral incorpora 2022 y ese porcentaje baja.

**Medias por régimen en train (reglas)**, umbrales `volatility` > 0.252 → crisis y `efficiency` >
0.141 → tendencia:

| Régimen | volatility | efficiency | autocorr | Duración media (días) |
|---|---|---|---|---|
| crisis | 0.386 | 0.132 | −0.204 | 28.9 |
| tendencia | 0.171 | 0.247 | −0.055 | 10.0 |
| reversion | 0.197 | 0.080 | −0.023 | 9.4 |

Crisis duplica la volatilidad de los otros dos y tendencia triplica la eficiencia de reversión, así
que esos nombres quedan respaldados. **Reversión no tiene la `autocorr` más negativa** (verificación
de la sección "Nombres"): la más negativa es crisis, por los rebotes fuertes de marzo de 2020. En
reglas, "reversión" describe un mercado sin dirección (rango), no autocorrelación negativa; así se
declara en el reporte.

### Esquema de ajuste
- **Reajuste mensual con ventana expandible** (`regime_refit_freq = "MS"`). En el primer día hábil τ
  de cada mes, desde 2018-01-01 (`regime_first_fit`), el modelo se ajusta con todas las variables
  válidas de fechas anteriores a τ, y con él se etiquetan los días de ese mes sin reajustar.
- **Mínimo de datos:** se ajusta solo con al menos 126 observaciones válidas (`regime_min_fit_obs`).
  Con datos reales, al 2018-01-01 hay 187 (del 2017-04-05 al cierre de 2017).
- **Antes del primer ajuste la etiqueta es NaN.** Con NaN no se abre posición (`generate_signals`
  da state = 0).
- **HMM:** en cada fecha t del mes, la recursión forward corre con los parámetros del ajuste vigente
  desde la primera variable válida hasta t. La etiqueta en t solo usa variables hasta t y un modelo
  ajustado con datos anteriores a τ ≤ t, así que no cambia al agregar datos posteriores (prueba 4 del
  lab).
- **Por qué expandible:** 126 días son pocos para estimar 3 estados con covarianza completa; la
  ventana expandible da más datos sin usar el futuro.
- **Por qué mensual:** coincide con el paso del walk-forward; reajustar a diario es caro e inestable.
- **Por qué empezar en 2018:** 2017 es calentamiento (`SPEC.md` punto 1), y esperar a tener 252
  observaciones dejaría sin etiqueta los primeros meses de train.

### Nombres de los regímenes
Los modelos producen grupos numerados, que pueden cambiar de número en cada reajuste. En cada ajuste
el nombre se asigna con una regla determinista sobre los centroides (K-means) o las medias por estado
(HMM), llevados a las unidades originales:

1. El grupo con mayor `volatility` media es **crisis**.
2. De los otros dos, el de mayor `efficiency` media es **tendencia**.
3. El restante es **reversion**.

En reglas, los nombres salen de la construcción. Después se verifica, sin usarlo para nombrar, que
reversión tenga la menor `autocorr` media; si no la tiene, se reporta.

Valores de la etiqueta: `"tendencia"`, `"reversion"` (sin acento) y `"crisis"`, iguales a las llaves
de `regime_multiplier`.

### Metas de validación (equivalentes diarios)
| Meta del lab (5 min) | Equivalente diario | Valor |
|---|---|---|
| Duración media > 12 horas | Duración media en días hábiles | ≥ 10 |
| Actualización cada 1 a 6 horas | Frecuencia de etiquetado | Diaria |
| Silhouette > 0.4 | Igual | > 0.4 |
| Estabilidad fuera de muestra | % de tiempo por régimen en train, validation y test | Se reporta |
| Diferenciación | Métricas de la estrategia por régimen | Se reporta (pregunta 5) |

- **Por qué 10 días:** la meta del lab pide que un régimen dure más que la mitad de una sesión de 5
  minutos. En barras diarias, dos semanas hábiles es el equivalente que permite operar el régimen:
  con menos, los parámetros cambian más rápido que la duración típica de una operación (holding base
  de 20 barras).
- Si el método elegido no llega a 10 días, se agrega un **filtro de duración mínima causal**: la
  etiqueta cambia solo después de d días consecutivos con la nueva etiqueta filtrada, con d fijado en
  train. No se aplica por adelantado.
- Si la silhouette no supera 0.4, se reporta tal cual y no se cambian las variables después de ver
  validation.

---

## Portafolio (P4)

### Pesos base: Risk Parity
Formulación convexa de Spinu (2013):

    min_{y > 0}  ½ yᵀΣy − (1/n) Σ ln y_i,        w^RP = y / Σ y_j

Comparación: volatilidad inversa (versión naive) y pesos iguales.

### Estimador de covarianza
Se reportan los pesos bajo al menos dos estimadores y se compara su estabilidad.
**Estimador elegido:** *[PENDIENTE. Candidatos: muestral, EWMA, Ledoit-Wolf]* · **Ventana:** *[PENDIENTE]*
· **Justificación:** *[PENDIENTE]*

### Agregación de señales

    w̃_i = w_i^RP · s_i,        w^target = m(régimen) · w̃ / max(1, Σ|w̃_i|)

con s_i la fuerza de `SPEC.md` punto 3.

| Régimen | m(régimen) | Justificación |
|---|---|---|
| tendencia | *[PENDIENTE, referencia 1.0]* | |
| reversion | *[PENDIENTE, referencia 0.7]* | |
| crisis | *[PENDIENTE, referencia 0.3]* | |

**Política ante señales en conflicto entre activos muy correlacionados:** *[PENDIENTE]*

### Rebalanceo
- Turnover: T_t = ½ Σ |w_i,t − w_i,t−|, con w_t− el peso después del drift.
- Disparador: *[PENDIENTE. Propuesta: híbrido, se revisa la banda δ solo en fechas de calendario.]*
- Frecuencia de revisión: *[PENDIENTE]* · Banda δ: *[PENDIENTE]*
- Barrido: retorno bruto, costo total, retorno neto y turnover contra la frecuencia o δ.
- `resize_on_rebalance` (¿se redimensionan las posiciones abiertas?): *[PENDIENTE, acordar con P1]*

---

## Registro de cambios
| Versión | Fecha | Cambio | Autor |
|---|---|---|---|
| 0.1 | 2026-10-04 | Plantilla | Equipo |
