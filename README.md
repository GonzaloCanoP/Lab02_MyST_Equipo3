# Lab02_MyST_Equipo3 — Estrategias de Trading con Análisis Técnico

Microestructuras y Sistemas de Trading (IT1731B) · ITESO, Otoño 2026 · **Nivel de alcance: C**

## Integrantes

| Parte | Integrante | Responsabilidad |
|---|---|---|
| P0 y P1 | Gonzalo Cano Padilla | Esqueleto, datos, motor de backtesting, auditoría de sesgos y `main.py`; revisión final del proyecto, ajustes e integración de las cuatro partes |
| P2 | Jerónimo Rojas Alvarado | Señales, métricas, optimización y walk-forward |
| P3 | Juan Manuel Espinosa Cárdenas | Régimen de mercado y figuras |
| P4 | Raúl Zanatta Casas | Risk Parity, agregación de señales y rebalanceo |

## Descripción

Estrategia direccional larga y corta sobre 8 activos líquidos de EE.UU. (AAPL, MSFT, META, AMD, XOM,
SMH, GLD y COPX), con barras diarias de 2017 a septiembre de 2026.

- **Señal:** se abre posición solo si al menos 2 de 3 indicadores de dos familias coinciden en
  dirección: cruce de SMA (tendencia), histograma MACD y RSI de Wilder (momento).
- **Salidas:** stop-loss y take-profit en múltiplos del ATR de Wilder, señal opuesta y holding
  máximo, con reglas de rearme.
- **Asignación:** Risk Parity (Spinu, covarianza Ledoit-Wolf de 126 días) reparte el capital entre
  activos. La fuerza de la señal (Σ votos / 3) y un multiplicador por régimen la escalan: tendencia
  1.0, reversión 0.7 y crisis 0.3. Cada operación usa todo su capital asignado, sin apalancamiento.
- **Régimen:** tendencia, reversión o crisis, con reglas sobre volatilidad y razón de eficiencia del
  índice equiponderado de los 8 activos (ventana de 63 días). La etiqueta es filtrada y causal, con
  reajuste mensual.
- **Optimización:** Optuna (TPE) en un walk-forward de 6 meses de entrenamiento y 1 de prueba, con
  paso mensual. Los parámetros de indicadores se fijan con el estudio de diagnóstico sobre train, y
  por régimen y ventana se optimizan el stop, la razón recompensa/riesgo y el holding máximo. Se
  elige el centro de la meseta (medoide del mejor 10%), no el máximo.
- **Costos:** comisión de 0.125% por lado, slippage de 2 bps en todo llenado (incluidos SL y TP) y
  borrow de 0.25% anual en cortos.

Las reglas completas, con su justificación y su historial de cambios, están en
`instrucciones/SPEC.md` (v1.4) y `instrucciones/SPEC_portafolio.md` (v0.6). Toda decisión se tomó con
train, salvo las decisiones discretas que el SPEC asigna a la corrida única de validation (estimador
de covarianza y rebalanceo). Test se corre una sola vez, al final.

| Bloque | Fechas | Uso |
|---|---|---|
| Calentamiento | 2017 | Solo indicadores; no genera señales |
| Train | 2018-01-01 → 2021-12-31 | Diseño, diagnóstico y calibraciones |
| Validation | 2022-01-01 → 2023-12-31 | Una corrida para decisiones discretas |
| Test | 2024-01-01 → 2026-09-30 | Una sola corrida con todo congelado |

## Estructura

```
├── main.py                 # pipeline completo y CONFIG con todos los valores fijos
├── data/                   # un CSV por activo e IRX.csv (tasa libre de riesgo), congelados
├── src/                    # data, backtest, signals, metrics, optimize, regimes, portfolio, plots
├── tests/                  # pytest con datos sintéticos (sin red ni data/)
├── notebooks/
│   ├── resultados.ipynb    # todos los resultados y las respuestas a las preguntas del lab
│   ├── analisis_P1.ipynb   # datos, motor y golden tests
│   └── analysis.ipynb      # portafolio con θ base (P4)
├── docs/figuras/           # figuras que genera main.py
├── results/                # resultados de main.py (en .gitignore)
└── instrucciones/          # SPEC, SPEC_portafolio e instrucciones por parte
```

## Instalación

Requiere **Python 3.13.9**.

```bash
git clone https://github.com/GonzaloCanoP/Lab02_MyST_Equipo3.git
cd Lab02_MyST_Equipo3
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducir los resultados

```bash
python main.py          # pipeline completo, sin intervención ni red
python -m pytest        # 173 pruebas con datos sintéticos
```

`main.py` corre todo en orden:
1. carga y auditoría de datos;
2. régimen;
3. portafolio y corrida base sobre train;
4. estudios de diagnóstico (200 random y 200 TPE);
5. robustez (sensibilidad ±20%, curva de costos, comparación por indicador);
6. walk-forward rolling y anchored, por régimen y con θ único;
7. decisiones de validation;
8. backtests finales: Risk Parity, pesos iguales, cada activo y buy & hold;
9. impacto de mercado y auditoría de sesgos.

Los resultados van a `results/` y las figuras a `docs/figuras/`. Después, `notebooks/resultados.ipynb`
solo carga esos archivos.

- `CONFIG["final_run"] = False` (por defecto): recorta los datos al fin de validation; test no se
  toca. Tarda unos 26 minutos con 11 núcleos.
- `CONFIG["final_run"] = True`: la corrida única con test, después de congelar el código y registrar
  el commit abajo. Tarda alrededor de una hora.

Los datos están congelados en `data/`; solo si falta algún archivo se descargan de Yahoo Finance
(`download_prices`, la única función con red).

## Resultados

*Corrida previa (`final_run = False`): train y validation. Test se agrega en la corrida final.*

Backtest continuo con los parámetros del walk-forward rolling por régimen. El bloque train empieza
en julio de 2018, el primer mes fuera de muestra.

| Estrategia | Bloque | Retorno anual | Volatilidad | Sharpe | Calmar | Max drawdown | Operaciones |
|---|---|---|---|---|---|---|---|
| Risk Parity | train | 4.0% | 5.1% | 0.59 | 0.82 | −4.8% | 280 |
| Risk Parity | validation | −0.4% | 5.8% | −0.65 | −0.06 | −6.2% | 141 |
| Pesos iguales | train | 5.1% | 6.5% | 0.65 | 0.89 | −5.7% | 280 |
| Pesos iguales | validation | −0.4% | 7.4% | −0.50 | −0.06 | −7.7% | 141 |
| Buy & hold | train | 40.0% | 28.8% | 1.28 | 1.32 | −30.3% | — |
| Buy & hold | validation | 7.8% | 22.9% | 0.29 | 0.31 | −25.1% | — |

La estrategia tiene un drawdown mucho menor que buy & hold, pero no muestra ventaja fuera de muestra.
La eficiencia del walk-forward es 0.14: sobrevive ~14% de la ventaja in-sample. El detalle está en
`notebooks/resultados.ipynb` y en `docs/figuras/final_vs_buy_and_hold.png`.

## Respuestas a las preguntas de análisis

Resumen; la versión completa, con tablas y figuras, está al final de `notebooks/resultados.ipynb`.

1. **2 de 3 contra un indicador:** con el mismo θ* en train, la regla hace 202 operaciones con Calmar
   1.08. MACD solo hace 561 (Calmar −0.07), RSI solo 673 (0.04) y SMA solo 140 (0.30). La
   confirmación recorta entre 64% y 70% las operaciones de MACD y RSI, y mejora el Calmar.
2. **Degradación en el walk-forward:** la eficiencia es 0.14 en retorno y 0.02 en Calmar. En las
   ventanas de validation, el Calmar IS mediano es 2.5 y el OOS −0.8. El anchored se degrada menos:
   6 meses no bastan para estimar los parámetros por régimen.
3. **Sensibilidad ±20%:** es una meseta moderada, no un pico aislado. La mediana de las 16
   variaciones conserva 91% del Calmar. Lo más sensible son las medias móviles (`sma_slow` ×1.2:
   −0.65 sobre 1.08).
4. **Costo de equilibrio:** en train la estrategia sigue rentable hasta 100 bps de ida y vuelta, más
   del triple de los 29 bps del caso base (margen > 71 bps). Fuera de muestra el margen es nulo:
   validation ya pierde con los costos base.
5. **Regímenes:** en train hay diferencias claras (Sharpe 1.44 en tendencia, −0.76 en reversión),
   pero en validation los tres son negativos. La capa de régimen aporta en dos cosas: el θ por
   régimen le gana al θ único en ambos bloques, y m(crisis) = 0.3 limita la exposición en 2022.
6. **Risk Parity contra pesos iguales:** no mejora el Calmar (0.82 contra 0.89 en train), pero
   iguala las contribuciones al riesgo (12.5% cada activo, contra 1.6% a 23.7% con pesos iguales) y
   reduce el drawdown y la volatilidad. El costo es menor retorno: subpondera la tecnología, que fue
   lo que más subió.
7. **Limitaciones para capital real:**
   - no hay evidencia de edge fuera de muestra;
   - sesgos de selección del universo: supervivencia, concentración en tecnología y muestra alcista;
   - ejecución optimista: llenado completo, slippage fijo, sin impacto ni rechazos.

**Advertencia de ejecución:** el backtest asume ejecución completa al precio modelado y no incorpora
impacto de mercado ni fallas de ejecución. Estimado ex post con el modelo de raíz cuadrada sobre las
operaciones de validation:
- 2.3 bps promedio por llenado (p95 de 11.9 bps), equivalente a 0.41% del equity del bloque;
- COPX concentra el impacto, con 10.8 bps promedio.

El impacto crece con la raíz del tamaño: con 100 veces más capital sería ~10 veces mayor.

## Semilla

`SEED = 42`, definida en `CONFIG` dentro de `main.py`. Se pasa explícita a los samplers de Optuna, al
`random_state` de los modelos y a fANOVA. La ventana i del walk-forward usa la semilla `SEED + i`. No
se usa `np.random.seed` global. Con la misma semilla, `main.py` reproduce los mismos resultados.

## Commit de la corrida final

*[Pendiente: hash del commit en `main` con el código congelado, registrado antes de correr con
`final_run = True`. `main.py` también guarda el hash en `results/corrida.pkl`.]*

## Uso de IA

### Gonzalo Cano Padilla

- P0: con asistencia de Claude Code se revisó la estructura del repositorio, se fijaron las
  dependencias, se escribieron los stubs de `src/` con sus firmas, `main.py` con `CONFIG`,
  `tests/conftest.py` y este README. También se implementó `download_prices` y se descargaron los datos.
- P1, motor y auditoría: con asistencia de Claude Code se implementaron `run_backtest`, sus pruebas
  (contabilidad, inmutabilidad, 9 golden-file tests calculados a mano y truncamiento del pipeline) y
  `notebooks/analisis_P1.ipynb`.
- P1, cierre: con asistencia de Claude Code se agregaron `market_impact` (modelo de raíz cuadrada ex
  post, con prueba golden calculada a mano y prueba de causalidad) y, en `main.py`, la corrida base
  sobre train, el guardado de resultados y figuras, la estimación de impacto y la tabla de auditoría
  de sesgos generada en `results/auditoria_sesgos.md`. Las decisiones abiertas (unidades del ADV,
  ubicación de la función, dónde vive la auditoría) se confirmaron antes de implementarlas.
- Revisión final del proyecto e integración (2026-10-06): hice la revisión final de las cuatro partes
  contra los SPEC y los ajustes necesarios, con asistencia de Claude Code. Los cambios fueron estos.
  - Retornos simples en la covarianza de Risk Parity y en las correlaciones (el régimen sigue en
    log-retornos, por aditividad).
  - `sleeve_weights` vectorizado, con caché de w^RP. El resultado es idéntico al anterior y la
    optimización pasó de ~16 h a ~26 min.
  - Ventanas sin θ factible en efectivo, actividad mínima escalada con el largo de la ventana y
    paralelismo con procesos.
  - Periodo de evaluación en la sensibilidad, los costos y la pregunta 1; superficie 3D, métricas
    por bloque, buy & hold y la comparación a igual volatilidad.
  - `main.py` completo, con la corrida previa sin test (`final_run`), y `notebooks/resultados.ipynb`
    con las respuestas a las preguntas.
- Decisiones del equipo con evidencia de train (SPEC v1.3 y v1.4, SPEC_portafolio v0.5 y v0.6):
  - calibraciones del punto 9;
  - umbrales de P4;
  - sizing con nocional = C_i, porque la exposición era de 10%;
  - walk-forward solo sobre k, r y m, porque el rolling con 9 dimensiones sobreajustaba;
  - mantener largo/corto para no usar información posterior a 2021.

  Las reglas de las decisiones de validation se fijaron y registraron antes de verlas. Claude Code
  propuso las opciones con su evidencia y el equipo decidió.

### Juan Manuel Espinosa Cárdenas

- P3, régimen y visualización: con asistencia de Claude Code (Anthropic) se revisaron las
  instrucciones y las ramas del equipo, se redactó la sección "Régimen" de `SPEC_portafolio.md` y se
  implementaron `src/regimes.py`, `src/plots.py`, `tests/test_regimes.py` y `tests/test_plots.py`.
- Las decisiones de diseño (serie de mercado, variables, regla de elección del método, esquema de
  ajuste y regla de nombres) se discutieron y aprobaron antes de implementarlas; la regla de
  elección se fijó antes de ver resultados de validation y el método (reglas) salió de aplicarla.
- La IA detectó y documentó en el SPEC que el HMM con un solo inicio caía en un óptimo local
  degenerado, y se corrigió con reinicios. También corrigió en el SPEC una expectativa propia que los
  datos no confirmaron (el % de crisis en validation no baja con el reajuste mensual).
- Verificación: la recursión forward se validó contra hmmlearn en el último día de cada muestra, y se
  comprobó que las pruebas de truncamiento fallan si se introduce una fuga a propósito.
- Correcciones en `main` antes de P3 (llaves de régimen en `CONFIG`, docstring de
  `test_pipeline.py`, prueba de régimen NaN en `test_signals.py` y contrato de `regime_validation`),
  hechas con asistencia de Claude Code y avisadas al equipo.

### Jerónimo Rojas Alvarado

- P2, señales: Se utilizó ChatGPT como apoyo para revisar las especificaciones,
  estructurar las funciones de indicadores y diseñar las pruebas de causalidad
  y confirmación. El código fue revisado y ejecutado localmente antes del commit.

### Raúl Zanatta Casas

*[Pendiente]*
