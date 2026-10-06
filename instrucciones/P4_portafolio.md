# P4 — Portafolio y Risk Parity

Hereda todo de `instrucciones/CLAUDE.md`.

Fuente: "Fundamentos Matemáticos de Risk Parity — De Rp al Rebalanceo" (Prof. Luis Alvarado, ITESO,
Otoño 2026). Su contenido está transcrito abajo como contexto; no hace falta consultar el original.
Los "Pasos" citados en las tareas son los de esta sección.

## Contexto teórico

### El problema
Con un solo activo, la única decisión era cuánto invertir. Con n activos correlacionados y con
volatilidades distintas, la pregunta pasa a ser cómo repartir un presupuesto de riesgo fijo entre n
oportunidades: qué vector w hace que ningún activo domine el riesgo del portafolio.

Risk Parity responde que el portafolio adecuado es aquel donde cada activo contribuye por igual al
riesgo total, sin importar cuánto capital reciba:

    RC_i(w) = RC_j(w)  ∀ i, j,     sujeto a Σ w_i = 1 y, en la versión long-only, w_i > 0.

Por qué importa:
1. No necesita retornos esperados: se construye solo con Σ, el único insumo que se estima con error
   aceptable (a diferencia de Markowitz).
2. Pesos iguales no son riesgo igual: con volatilidades distintas, repartir capital en partes iguales
   reparte el riesgo de forma muy desigual.
3. Es exacto: por el Teorema de Euler, las contribuciones por activo suman exactamente el riesgo total.
4. No es el portafolio final: se combina con las señales técnicas y el régimen antes de volverse una
   orden ejecutable.

### Paso 1 — Retorno y varianza del portafolio
R_p = Σ w_i R_i. Para dos activos, por bilinealidad de la covarianza:

    Var(R_p) = w₁²σ₁² + w₂²σ₂² + 2 w₁ w₂ ρ σ₁ σ₂

ρ = 0 significa no correlación lineal, no independencia. La fórmula es una identidad algebraica: solo
requiere que existan los segundos momentos.

### Paso 2 — Forma matricial
Con Σ_ij = Cov(R_i, R_j):

    Var(R_p) = wᵀΣw,        σ_p(w) = √(wᵀΣw)

Cada entrada (Σw)_k es Cov(R_k, R_p). **Σ se estima siempre sobre retornos, nunca sobre precios**: los
precios no son estacionarios y producen correlaciones espurias cercanas a 1.

Ejemplo: σ₁ = 5%, σ₂ = 25%, ρ = 0, w = (0.5, 0.5) → σ_p ≈ 12.7%, y el activo 2 aporta el 96% de la
varianza. El portafolio "balanceado" en capital es, en riesgo, casi puro equity.

### Paso 3 — Contribución marginal y total
Por regla de la cadena, con u = wᵀΣw y ∂u/∂w_k = 2(Σw)_k (Σ simétrica):

    ∂σ_p / ∂w_k = (Σw)_k / σ_p                    (contribución marginal)
    RC_k = w_k (Σw)_k / σ_p                       (contribución total)
    Σ_k RC_k = wᵀΣw / σ_p = σ_p                   (exacto, por Euler: σ_p es homogénea de grado 1)

RC_i / σ_p es un porcentaje genuino del riesgo total.

### Paso 4 — Condición de Risk Parity

    RC_i = σ_p / n  ∀ i   ⇔   w_i (Σw)_i = w_j (Σw)_j  ∀ i, j,   Σ w_i = 1, w_i > 0

Penaliza a los activos volátiles y, además, a los muy correlacionados con el resto, porque su
contribución marginal incluye los términos de covarianza.

Versión naive, volatilidad inversa:

    w_i = (1/σ_i) / Σ_j (1/σ_j)

Es exacta solo si todas las correlaciones por pares son iguales. En otro caso ignora la estructura de
correlación: dos activos del mismo sector siguen dominando el riesgo conjunto. Ejemplo con σ₁ = 5% y
σ₂ = 25%: w = (0.833, 0.167).

### Paso 5 — Formulación convexa (Spinu, 2013)
Minimizar Σ(RC_i − RC_j)² directamente es no convexo: tiene regiones planas y el resultado depende del
punto de partida, inaceptable en un pipeline que corre sin supervisión en cada rebalanceo. En su lugar,
con una variable auxiliar y > 0 (no es el peso final):

    min_{y > 0}  ½ yᵀΣy − (1/n) Σ_i ln y_i,        w = y / Σ_j y_j

La forma cuadrática es convexa porque Σ es semidefinida positiva; −ln(y_i) es convexa y tiende a +∞
cuando y_i → 0⁺, así que mantiene y > 0 sin restricción explícita. Suma de convexas es convexa:
solución única.

### Paso 6 — Estimación de Σ
Tres defensas, de menor a mayor esfuerzo:
1. **Ventana más larga que la de la señal.** El error del estimador muestral escala con n/T; más T
   reduce la varianza a costa de suponer Σ estacionaria en la ventana.
2. **EWMA:** Σ_t = λ Σ_{t−1} + (1 − λ) r_t r_tᵀ, con muestra efectiva T_eff = 1/(1 − λ). Evita el salto
   discreto ("ghosting") cuando una observación extrema sale de una ventana rodante.
3. **Ledoit-Wolf:** Σ_shrink = δ* F + (1 − δ*) S, con δ* analítico que minimiza el error esperado en norma
   de Frobenius. Corrige la dispersión artificial de los eigenvalores de S.

Las tres mejoran solo el insumo Σ; no cambian la formulación del Paso 5. Reportar los pesos con al
menos dos estimadores y comparar su estabilidad es, en sí mismo, un resultado del análisis.

### Paso 7 — Agregación de señales
w^RP dice cuánto riesgo puede tomar cada activo; s_i ∈ [−1, 1] dice qué tan convencido está el sistema
de la dirección. Ninguno por separado es un portafolio ejecutable.

Con k indicadores que votan x_ij ∈ {−1, 0, 1}:

    s_i = (1/k) Σ_j x_ij   si |Σ_j x_ij| ≥ 2
          0                en otro caso

La compuerta (¿hay posición?) y la fuerza (¿cuánto riesgo?) son preguntas separadas: s_i no se colapsa
a {−1, 0, 1}, para no perder la diferencia entre 3 de 3 y 2 de 3.

    w̃_i = w_i^RP · s_i,        w^target = m(régimen) · w̃ / max(1, Σ_i |w̃_i|)

m(régimen) ∈ (0, 1] es un hiperparámetro de diseño, no se estima de los datos; por ejemplo 1.0 en
tendencia, 0.7 en reversión y 0.3 en crisis. Se justifica con la diferenciación de desempeño por
régimen (Sharpe y drawdown) o, con cautela por las pocas observaciones de crisis, se incluye en θ.

Ejemplo integrador: w^RP = (0.5, 0.3, 0.2); votos A = (1, 1, 1) → s = 1, w̃ = 0.50;
B = (1, 1, 0) → s = 2/3, w̃ = 0.20; C = (1, −1, −1) → |Σ| = 1 < 2, s = 0, w̃ = 0. Exposición bruta
0.70 ≤ 1, denominador 1. Con m = 0.7: w^target = (0.35, 0.14, 0).

### Paso 8 — Rebalanceo y turnover

    T_t = ½ Σ_i |w_i,t − w_i,t−|

w_t− es el peso justo antes de rebalancear, después del drift (lo que el movimiento de precios ya
cambió sin operar). El ½ cuenta un viaje redondo una sola vez.

    costo_anual ≈ T̄ · f · 2c        (turnover por rebalanceo × rebalanceos al año × comisión de ida y vuelta)

Ejemplo: capital 100,000 con objetivo (0.40, 0.60). Tras A +20% y B −10%: A = 48,000, B = 54,000,
total 102,000, drift w_t− = (0.4706, 0.5294). Nuevo objetivo (0.50, 0.50). Contra el objetivo viejo,
T = 0.10; contra el drift real, T = 0.0294. El mercado ya hizo parte del trabajo: comparar contra el
objetivo viejo sobreestima el costo.

Disparadores:
- **Calendario:** fecha fija; predecible, pero opera aunque la desviación sea mínima.
- **Banda:** rebalancea solo si ‖w_t − w^target‖₁ > δ; opera solo cuando importa, pero exige monitoreo
  continuo.
- **Híbrido (recomendado):** revisa la banda solo en fechas de calendario; a costa de un posible
  retraso entre que se cruza δ y la siguiente revisión.

δ, igual que m(régimen), es un hiperparámetro. Se barre f o δ y se grafica en una sola figura retorno
bruto, costo total, retorno neto y turnover: hay un óptimo interior entre nunca rebalancear (Risk
Parity deja de serlo) y rebalancear siempre (el costo domina el P&L).

## Las tareas

### Del material de Risk Parity del curso
1. Contribuciones al riesgo (Paso 3): RC_i = w_i (Σw)_i / σ_p, con Σ RC_i = σ_p exacto (Euler).
2. Condición de Risk Parity (Paso 4): RC_i = σ_p / n para todo i, con Σ w_i = 1 y w_i > 0. Comparar
   contra la versión naive de volatilidad inversa.
3. Resolverla con la formulación convexa de Spinu (Paso 5):
   min_{y>0} ½ yᵀΣy − (1/n) Σ ln y_i, y normalizar w = y / Σ y_j. No minimizar Σ(RC_i − RC_j)²
   directamente: es no convexo.
4. Estimar Σ sobre retornos, nunca sobre precios (Paso 6). "Reportar los pesos bajo al menos dos
   estimadores distintos y comparar su estabilidad es, en sí mismo, un resultado del análisis."
5. Agregación de señales (Paso 7): w̃_i = w_i^RP · s_i y w^target = m(régimen) · w̃ / max(1, Σ|w̃_i|).
   m(régimen) se justifica con la diferenciación de desempeño por régimen o entra al espacio θ.
6. Turnover (Paso 8): T_t = ½ Σ |w_i,t − w_i,t−|, con w_t− el peso después del drift. Costo anualizado
   ≈ T̄ · f · 2c.
7. Disparadores de rebalanceo: calendario, banda δ o híbrido. Barrer f o δ y graficar en una sola figura
   retorno bruto, costo total, retorno neto y turnover realizado.

### Del enunciado del Lab 02
8. "Verifique numéricamente que las contribuciones al riesgo resultantes son iguales entre sí. Incluya
   esa verificación como prueba automatizada."
9. "Declare y justifique el estimador de covarianza utilizado."
10. "El manejo de señales en conflicto entre activos, especialmente cuando los activos están altamente
    correlacionados. Declare la política elegida en el reporte."
11. "Compare su portafolio de Risk Parity contra un benchmark de pesos iguales, con las mismas señales,
    los mismos costos y el mismo rebalanceo. La justificación debe ser de asignación de riesgo, no de
    maximización de retorno."
12. "Comparación del desempeño del portafolio contra el desempeño de cada activo individual."
13. Entregables: contribuciones al riesgo (Risk Parity contra pesos iguales), mapa de calor de señales,
    evolución de la matriz de correlación entre regímenes, barrido de rebalanceo, métricas del
    portafolio por régimen.
14. `notebooks/analysis.ipynb`: "solo análisis y figuras, sin lógica."

## Cómo encaja
- PRIMERA TAREA, antes de programar: completar la sección "Portafolio" de `SPEC_portafolio.md`
  (estimador candidato, valores de m(régimen), política de conflictos, disparador de rebalanceo y
  `resize_on_rebalance` acordado con P1) e integrarla a `main` por PR.
- Retornos para Σ: retornos simples diarios del `close` de cada activo de `data/`, vía `load_prices`.
- `risk_parity_weights`: Spinu con `scipy.optimize.minimize` (L-BFGS-B con cota inferior positiva) o
  Newton; después normalizar.
- `estimate_cov`: implementar `"sample"`, `"ewma"` y `"ledoit_wolf"` (sklearn `LedoitWolf`). Comparar la
  estabilidad de los pesos (por ejemplo, desviación estándar del peso de cada activo entre rebalanceos)
  con al menos dos estimadores, en train.
- `compose_target` implementa el Paso 7 tal cual. La fuerza s_i viene de `generate_signals` (P2).
- `sleeve_weights` es la única puerta entre portafolio y motor: devuelve el panel |w^target| que recibe
  `run_backtest`. Con `method="equal"`, w^RP se sustituye por 1/8 con la misma agregación y el mismo
  rebalanceo.
- Validación del ejemplo del documento: con w^RP = (0.5, 0.3, 0.2), votos A = (1, 1, 1),
  B = (1, 1, 0), C = (1, −1, −1) y m = 0.7, `compose_target` debe devolver (0.35, 0.14, 0).
- Turnover contra drift: el ejemplo del Paso 8 (drift a (0.4706, 0.5294) y objetivo (0.5, 0.5)) debe
  dar T = 0.0294.
- Notebook: carga `results/`, llama funciones de `plots.py` (P3) y muestra tablas, incluidos los N de
  trials de los estudios de diagnóstico. Celdas cortas, una idea por celda.

## Entregable
- Sección "Portafolio" de `SPEC_portafolio.md`.
- `src/portfolio.py` y `tests/test_portfolio.py`.
- `notebooks/analysis.ipynb`.
- En `results/`: estabilidad de pesos por estimador, Risk Parity contra pesos iguales (Calmar, MDD,
  contribuciones objetivo y realizadas), portafolio contra cada activo individual, turnover y barrido
  de rebalanceo.

## Checklist
- [ ] Sección "Portafolio" de `SPEC_portafolio.md` integrada a `main` antes de implementar
- [ ] `test_portfolio.py`: `max(RC) − min(RC) < 1e-6` y `Σ RC = σ_p` (tol 1e-8) para una covarianza sintética de 8 activos con volatilidades y correlaciones distintas
- [ ] `test_portfolio.py`: con correlaciones iguales, Risk Parity coincide con volatilidad inversa (tol 1e-6)
- [ ] `test_portfolio.py`: el ejemplo integrador del Paso 7 da (0.35, 0.14, 0) y el del Paso 8 da T = 0.0294
- [ ] `sleeve_weights` pasa la prueba de truncamiento (el panel en t no cambia al agregar datos posteriores)
- [ ] Todo panel cumple `Σ |w| ≤ 1` y `|w| ≥ 0` en todas las fechas
- [ ] Pesos reportados con al menos dos estimadores y su estabilidad comparada
- [ ] Barrido de rebalanceo con retorno bruto, costo, neto y turnover en una sola figura
- [ ] El notebook no define funciones ni contiene fórmulas; `grep "def " notebooks/analysis.ipynb` no encuentra nada

## Puntos abiertos (se resuelven en SPEC_portafolio.md)
- Estimador elegido y su ventana (propuesta: Ledoit-Wolf sobre 126 días).
- m(régimen): valores de referencia del documento (1.0, 0.7, 0.3) o dentro del espacio θ.
- Política ante señales opuestas en activos con correlación alta.
- Disparador: propuesta híbrida (banda δ revisada en fechas de calendario); valores del barrido.
- `resize_on_rebalance`, a acordar con P1. Ojo: con SPEC.md, un Estado = 0 no cierra la posición, pero
  hace s_i = 0 y por tanto w^target_i = 0. Si se redimensiona al rebalancear, ese caso cerraría la
  posición; hay que decidir cuál regla manda.
