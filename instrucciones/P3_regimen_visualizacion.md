# P3 — Régimen de mercado y visualización

Hereda todo de `instrucciones/CLAUDE.md`. 

Fuentes: actividad "Market Regime Detection" del curso y sección 3.4 del enunciado del Lab 02. Su
contenido está transcrito abajo; no hace falta consultar los originales.

## Contexto teórico

### Qué es un régimen
Un régimen es un estado no observable del mercado que cambia la forma en que se comportan los
precios. El lab pide tres:
- **Tendencia:** momento alto y movimiento direccional fuerte.
- **Reversión a la media:** comportamiento de rango y reversiones frecuentes.
- **Crisis:** volatilidad alta y movimientos extremos.

Los clasificadores no saben nombres: producen grupos numerados. El nombre se asigna después, leyendo
los centroides (o medias por estado) de las features en train.

### Tres clasificadores
- **Reglas:** umbrales explícitos sobre las features (por ejemplo, volatilidad por encima de su
  cuantil 80 → crisis). Transparente, sin estimación, pero los umbrales son arbitrarios.
- **K-means:** agrupa los vectores de features por distancia al centroide más cercano. Cada día se
  clasifica de forma independiente, sin memoria del día anterior; por eso tiende a cambiar de grupo
  con frecuencia.
- **HMM (modelo oculto de Markov):** supone un estado oculto que sigue una cadena de Markov con matriz
  de transición A, y features que se generan con una distribución distinta por estado. La matriz de
  transición le da persistencia: cambiar de estado tiene un costo de probabilidad.

### Filtrado contra Viterbi (la distinción crítica)
Para un HMM hay tres formas de etiquetar el día t:
- **Filtrada:** P(s_t = j | x_1, …, x_t). Usa solo información hasta t. Se calcula con la recursión
  forward:

      α_1(j) = π_j · b_j(x_1)
      α_t(j) = b_j(x_t) · Σ_i α_{t−1}(i) · A_ij        (normalizando α_t para que sume 1)
      ŝ_t = argmax_j α_t(j)

  donde π es la distribución inicial (`startprob_`), A la matriz de transición (`transmat_`) y b_j(x_t)
  la verosimilitud de x_t bajo el estado j. **Es la única que se puede operar.**
- **Suavizada:** P(s_t = j | x_1, …, x_T). Usa toda la muestra (forward-backward). Es lo que devuelve
  `predict_proba` de hmmlearn.
- **Viterbi:** la secuencia de estados más probable dada toda la muestra. Es lo que devuelve `predict`
  de hmmlearn. Reetiqueta el pasado usando el futuro.

La suavizada y Viterbi se ven más limpias y persistentes porque conocen el futuro; por eso operar con
ellas invalida el backtest. La figura de la tarea 4 pone lado a lado la filtrada y Viterbi justamente
para mostrar esa diferencia.

### Métricas de validación
- **Silhouette:** para cada observación, (b − a) / max(a, b), con a la distancia media a su propio
  grupo y b la distancia media al grupo vecino más cercano. Va de −1 a 1; la meta del lab es > 0.4.
- **Duración media:** días consecutivos promedio en un mismo régimen. Mide persistencia.
- **Transiciones por mes:** número de cambios de etiqueta por mes. Una clasificación que cambia
  demasiado no se puede operar: cada cambio puede alterar parámetros y exposición.
- **Participación:** % del tiempo en cada régimen, comparado entre train y test para juzgar la
  estabilidad fuera de muestra.

### Operar los regímenes
En vez de un θ* único para toda la muestra, se optimiza un θ por estado usando solo los días de ese
estado en train:

    θ*_j = argmax_θ J(backtest(train | s_t = j, θ)),   j = 1, …, k

y en cada barra se opera con θ*_{ŝ_t}, donde ŝ_t es la etiqueta FILTRADA. La capa de régimen se
justifica solo si mejora frente al θ* único, en train y en test. Si el desempeño no difiere entre
regímenes, hay que decir qué aporta entonces la capa (pregunta 5 del lab).

## Las tareas

### De la actividad de Market Regime Detection del curso
1. Implementar `regime_features()` y correr la prueba de truncamiento en cada columna.
2. Ajustar los tres clasificadores (reglas, K-means, HMM) en el tramo de entrenamiento.
3. Tabla comparativa: silhouette, duración media, transiciones por mes y participación de cada régimen.
4. Línea de tiempo del precio coloreada por la etiqueta **filtrada**, con la versión Viterbi al lado.
5. Distribución de las features por régimen y justificación de los nombres a partir de los centroides.
6. Elegir un método y defenderlo (justificar).
7. Operar los regímenes: optimizar un θ por estado, θ*_j = argmax_θ J(backtest(train | s_t = j, θ)),
   j = 1, …, k; operar en cada barra θ*_ŝt con la etiqueta filtrada y comparar contra el θ* único en
   Train y Test.

### Del enunciado del Lab 02
8. Tres regímenes: tendencia, reversión a la media y crisis. Nivel C: "ventana móvil de 3 meses para
   las variables."
9. "Si usa un HMM, la clasificación debe basarse en probabilidades filtradas, no en la ruta de Viterbi."
10. Validación: persistencia (duración media y frecuencia de transiciones), separación (silhouette > 0.4),
    diferenciación (métricas de la estrategia por régimen) y estabilidad fuera de muestra (proporción
    del tiempo de cada régimen en train y en test).
11. Prueba 4: "la etiqueta de régimen en t no cambia al agregar datos posteriores a t."
12. Figuras obligatorias, sección 3.6, puntos 1 a 7, todas con título, ejes etiquetados y leyenda.

## Cómo encaja
- PRIMERA TAREA, antes de programar: completar la sección "Régimen" de `SPEC_portafolio.md` (serie de
  mercado, variables, esquema de ajuste, metas diarias) e integrarla a `main` por PR. El método elegido
  y los nombres se completan después de las tareas 3 a 6, con datos de train.
- Las features se calculan sobre la serie de mercado (propuesta: índice equiponderado de los 8
  activos de `data/`, vía `load_prices`) con ventana móvil de 63 días hábiles.
- `fit_regime_model(features_train, method, seed)` con `method ∈ {"rules", "kmeans", "hmm"}`:
  - Reglas: umbrales fijados con cuantiles de train.
  - K-means: `random_state=seed`; features estandarizadas con media y desviación de train.
  - HMM: `GaussianHMM` de hmmlearn con `random_state=seed`.
- **Cuidado con hmmlearn:** `predict()` devuelve la ruta de Viterbi y `predict_proba()` devuelve
  probabilidades SUAVIZADAS (forward-backward); las dos usan el futuro. `predict_regimes` implementa la
  recursión forward a mano con `startprob_`, `transmat_` y la verosimilitud de cada estado, y etiqueta
  con el argmax de la probabilidad filtrada. `viterbi_path` existe solo para la figura de la tarea 4.
- `label_regimes`: reajuste mensual con datos hasta esa fecha; el mes siguiente se etiqueta sin
  reajustar. Produce una etiqueta causal para todas las fechas.
- Tarea 7: los estudios por estado y el θ único los corre P2 (`walk_forward` con `per_regime=True` y
  `False`). P3 arma la comparación por régimen y la interpreta (pregunta 5).
- `plots.py`: una función por figura, que recibe resultados ya calculados y devuelve `Figure`. Sin
  cálculos de estrategia dentro.

| Figura | Función |
|---|---|
| Valor del portafolio, train y test, con benchmark | `plot_equity` |
| Drawdown | `plot_drawdown` |
| Retornos mensuales, trimestrales y anuales | `plot_returns_table` |
| Sensibilidad ±20% | `plot_sensitivity` |
| Retorno neto contra costo de ida y vuelta | `plot_cost_curve` |
| Régimen filtrado sobre el precio, con Viterbi al lado | `plot_regime_timeline` |
| Distribución de features por régimen | `plot_regime_features` |
| Equity con regímenes superpuestos | `plot_equity_regimes` |
| Contribuciones al riesgo: Risk Parity contra pesos iguales | `plot_risk_contributions` |
| Mapa de calor de fuerza de señal | `plot_signal_heatmap` |
| Correlación por régimen | `plot_corr_by_regime` |
| Barrido de rebalanceo (bruto, costo, neto, turnover) | `plot_rebalance_sweep` |
| Superficie 3D del random search | `plot_surface_3d` |
| Historia de optimización | `plot_optimization_history` |
| Importancia de parámetros | `plot_param_importance` |
| Slice plots | `plot_slices` |

## Entregable
- Sección "Régimen" de `SPEC_portafolio.md`.
- `src/regimes.py` y `src/plots.py`.
- `tests/test_regimes.py`.
- En `results/`: tabla comparativa de los tres métodos, validación del método elegido, métricas por
  régimen y comparación θ por régimen contra θ único en train y test.

## Checklist
- [ ] Sección "Régimen" de `SPEC_portafolio.md` integrada a `main` antes de implementar
- [ ] Prueba de truncamiento en verde para cada columna de `regime_features` en ≥ 3 fechas
- [ ] Prueba de truncamiento en verde para la etiqueta filtrada en ≥ 3 fechas
- [ ] `predict_regimes` no llama a `.predict()` ni a `.predict_proba()` de hmmlearn (`grep`)
- [ ] Los tres métodos ajustados solo con train; tabla comparativa con silhouette, duración media, transiciones por mes y participación
- [ ] Método elegido y justificación escritos en `SPEC_portafolio.md`
- [ ] `regime_validation` reporta % de tiempo por régimen en train, validation y test
- [ ] Las 16 funciones de la tabla existen y cada figura tiene título, ejes y leyenda
- [ ] Las figuras se guardan en `docs/figuras/` desde `main.py`, no desde el notebook

## Puntos abiertos (se resuelven en SPEC_portafolio.md)
- Serie de mercado: índice equiponderado de los 8 activos o un índice externo (requeriría datos extra).
- Esquema de ajuste: propuesta de ventana expandible, porque 126 días por ventana son pocos para 3 estados.
- Equivalentes diarios de las metas de 5 minutos: duración media (propuesta ≥ 10 días hábiles) y
  actualización (diaria).
- Si hace falta un filtro de duración mínima para la persistencia, debe ser causal.
