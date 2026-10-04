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
