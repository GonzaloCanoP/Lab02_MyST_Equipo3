# SPEC de portafolio — Lab 02 · Equipo 3 · Nivel C

**Versión:** 0.1 · **Estado:** plantilla. P3 completa la sección "Régimen" y P4 la sección
"Portafolio", cada una por PR, ANTES de implementar su módulo. Toda decisión se toma con train.

Complementa a `SPEC.md`. Las fórmulas base vienen del material del curso "Fundamentos Matemáticos de
Risk Parity" y de la actividad de Market Regime Detection.

---

## Régimen (P3)

### Serie de mercado
*[PENDIENTE. Propuesta: índice equiponderado de los 8 activos, sin datos extra.]*

### Variables (ventana móvil de 63 días hábiles)
| Variable | Fórmula | Qué captura |
|---|---|---|
| *[PENDIENTE]* | | Volatilidad |
| *[PENDIENTE]* | | Fuerza direccional / tendencia |
| *[PENDIENTE]* | | Reversión (autocorrelación, cruces de la media) |

### Métodos comparados
Reglas, K-means y HMM, ajustados en train. Tabla comparativa: silhouette, duración media, transiciones
por mes y participación de cada régimen.

**Método elegido:** *[PENDIENTE]* · **Justificación:** *[PENDIENTE]*

### Esquema de ajuste
*[PENDIENTE. Propuesta: ventana expandible, reajuste mensual; la etiqueta operable es la filtrada.]*

### Nombres de los regímenes
Regla para asignar {tendencia, reversion, crisis} a partir de los centroides: *[PENDIENTE]*

### Metas de validación (equivalentes diarios)
| Meta del lab (5 min) | Equivalente diario | Valor |
|---|---|---|
| Duración media > 12 horas | Duración media en días hábiles | *[PENDIENTE, propuesta ≥ 10]* |
| Actualización cada 1 a 6 horas | Frecuencia de etiquetado | Diaria |
| Silhouette > 0.4 | Igual | > 0.4 |

---

## Portafolio (P4)

### Pesos base: Risk Parity
Formulación convexa de Spinu (2013):

    min_{y > 0}  ½ yᵀΣy − (1/n) Σ ln y_i,        w^RP = y / Σ y_j

Σ se estima sobre log-retornos diarios del `close`, nunca sobre precios, y solo con datos hasta t.
Comparaciones: volatilidad inversa (versión naive) y pesos iguales (1/8).

### Estimador de covarianza
Se calculan los pesos con tres estimadores (muestral, EWMA y Ledoit-Wolf) y se reporta en train la
estabilidad de cada uno (desviación estándar del peso de cada activo entre rebalanceos).

- **Ventana:** 126 días hábiles, mayor que la SMA lenta máxima (120). EWMA con λ = 0.99
  (T_eff = 100 días, comparable a la ventana muestral; con λ = 0.94, T_eff ≈ 17 sería muy ruidoso con
  n = 8).
- **Candidato inicial:** Ledoit-Wolf. Con n/T ≈ 0.06 el estimador muestral dispersa los eigenvalores de
  forma artificial; el encogimiento lo corrige con un δ* analítico, sin hiperparámetros extra.
- **Decisión:** una sola corrida en validation (SPEC punto 1). Se elige el estimador con menor
  desviación estándar promedio de los pesos; si la diferencia con Ledoit-Wolf es menor a
  *[PENDIENTE: umbral, fijado antes de correr]*, se queda Ledoit-Wolf.

### Agregación de señales

    w̃_i = w_i^RP · s_i,        w^target = m(régimen) · w̃ / max(1, Σ|w̃_i|)

con s_i la fuerza de `SPEC.md` punto 3. Como Σ w^RP = 1, |s_i| ≤ 1 y m(régimen) ≤ 1, siempre se cumple
Σ|w^target| ≤ 1.

| Régimen | m(régimen) | Justificación |
|---|---|---|
| tendencia | 1.0 | Régimen donde la señal tiene ventaja; exposición completa |
| reversion | 0.7 | Las señales de tendencia pierden fiabilidad |
| crisis | 0.3 | Drawdowns más profundos; se limita el riesgo |

Valores de referencia del material del curso. m(régimen) es un hiperparámetro de diseño que **no entra
al espacio de búsqueda θ** de `SPEC.md` punto 7 (pocas observaciones de crisis en train; limita el data
snooping). Se respalda con la tabla de Sharpe y MDD por régimen en train. *[PENDIENTE: confirmar con esa
tabla; si no hay diferencia entre regímenes, se documenta y se usa 1.0 en los tres.]*

**Política ante señales en conflicto entre activos correlacionados:** si ρ_ij > 0.7 (ventana de 126
días, solo datos hasta t) y s_i, s_j tienen signo opuesto, se conserva la de mayor |s| y la otra se pone
en 0; con |s_i| = |s_j|, ambas pasan a 0. Se aplica a s antes de `compose_target`. Razón: posiciones
opuestas en activos casi equivalentes cancelan riesgo y pagan costos por duplicado.
*[PENDIENTE: confirmar el umbral 0.7 con la matriz de correlación de train.]*
Con s_i = 0 resulta C_i = 0 aunque Estado ≠ 0. *[PENDIENTE con P1: una entrada con C_i = 0 no abre
posición, no consume el armado y no cuenta como operación.]*

### Rebalanceo
- **Qué se rebalancea:** solo w^RP. Entre rebalanceos, w^RP se mantiene vigente; s_i y m(régimen) se
  actualizan cada día, para que una señal nueva se ejecute en t+1 sin esperar la siguiente revisión.
  *[PROPUESTA: confirmar con el equipo.]*
- **Disparador híbrido:** en fechas de calendario (cada f días hábiles) se reestima Σ y se calcula el
  w^RP candidato; se adopta solo si ‖w^RP_cand − w^RP_vigente‖₁ > δ.
- **Valores iniciales:** f = 5 días hábiles, δ = 0.05. Barrido: f ∈ {1, 5, 10, 21, 63} y
  δ ∈ {0.02, 0.05, 0.10, 0.20}, con retorno bruto, costo total, retorno neto y turnover realizado en
  una sola figura. f y δ se resuelven con este barrido y no con Optuna.
- **Decisión de f y δ:** una sola corrida en validation.
- **Turnover:** T_t = ½ Σ |w_i,t − w_i,t−|, con w_t− el peso realizado después del drift. Costo anual
  ≈ T̄ · f · 2c, con c = 0.125% + 0.02% por lado (SPEC punto 6).
- **`resize_on_rebalance`:** `False` *[PENDIENTE: confirmar con P1]*. Consistente con SPEC punto 4: una
  posición abierta conserva su tamaño y Estado = 0 no la cierra. El rebalanceo solo cambia el tamaño de
  las entradas nuevas; su costo no entra al equity y se estima ex post con `turnover` sobre la
  exposición realizada. Limitación que se declara en el reporte.

---

## Registro de cambios
| Versión | Fecha | Cambio | Autor |
|---|---|---|---|
| 0.1 | 2026-10-04 | Plantilla | Equipo |
| 0.2 | 2026-10-05 | Sección Portafolio (borrador P4) | Zanatta |