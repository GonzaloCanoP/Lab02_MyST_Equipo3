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

- `CONFIG["final_run"] = True` (como quedó congelado): la corrida completa con test. Tarda unos 39
  minutos con 11 núcleos.
- `CONFIG["final_run"] = False`: recorta los datos al fin de validation y no toca test (unos 26
  minutos). Se usó antes de congelar el código, para decidir sin ver test.

Los datos están congelados en `data/`; solo si falta algún archivo se descargan de Yahoo Finance
(`download_prices`, la única función con red).

## Reporte

El reporte es **`notebooks/resultados.ipynb`**, exportado a PDF. Contiene:
- datos, estrategia (con la regla 2 de 3 como fórmula) y metodología;
- los resultados de cada parte, con su justificación;
- las respuestas a las 7 preguntas del lab, con la advertencia de impacto de mercado;
- conclusiones y uso de IA.

Para exportarlo, después de `python main.py`:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/resultados.ipynb
jupyter nbconvert --to pdf notebooks/resultados.ipynb   # o "Export as PDF" desde Jupyter/VS Code
```

## Resultados

Backtest continuo fuera de muestra: cada mes opera con los parámetros de su ventana del walk-forward
rolling por régimen, de julio de 2018 (primer mes fuera de muestra) a septiembre de 2026.

| Estrategia | Bloque | Retorno anual | Volatilidad | Sharpe | Calmar | Max drawdown | Operaciones |
|---|---|---|---|---|---|---|---|
| Risk Parity | train | 4.0% | 5.1% | 0.59 | 0.82 | −4.8% | 280 |
| Risk Parity | validation | −0.4% | 5.8% | −0.65 | −0.06 | −6.2% | 145 |
| Risk Parity | **test** | **0.7%** | **7.2%** | **−0.46** | **0.07** | **−10.7%** | **224** |
| Pesos iguales | train | 5.1% | 6.5% | 0.65 | 0.89 | −5.7% | 280 |
| Pesos iguales | validation | −0.4% | 7.4% | −0.50 | −0.06 | −7.7% | 145 |
| Pesos iguales | test | 0.1% | 8.8% | −0.43 | 0.01 | −14.8% | 224 |
| Buy & hold | train | 40.0% | 28.8% | 1.28 | 1.32 | −30.3% | — |
| Buy & hold | validation | 7.8% | 22.9% | 0.29 | 0.31 | −25.1% | — |
| Buy & hold | test | 38.4% | 23.1% | 1.34 | 1.81 | −21.3% | — |

- **Fuera de muestra la estrategia es casi plana:** el capital pasa de 1,000,000 a 1,159,068 USD en
  8.25 años, mientras buy & hold lo multiplica por 10.
- **El Sharpe es negativo en validation y test**, porque el retorno no supera al T-Bill (3.5% a 4.3%
  anual en esos años).
- **El drawdown es de 2 a 6 veces menor que el de buy & hold**, pero aun a igual volatilidad buy &
  hold gana en los tres bloques (`docs/figuras/final_vs_buy_and_hold.png`).

## Respuestas a las preguntas de análisis

Resumen; la versión completa, con tablas y figuras, está en `notebooks/resultados.ipynb`.

1. **2 de 3 contra un indicador:** con el mismo θ* en train, la regla hace 202 operaciones con Calmar
   1.08. MACD solo hace 561 (Calmar −0.07), RSI solo 673 (0.04) y SMA solo 140 (0.30). La
   confirmación recorta entre 64% y 70% las operaciones de MACD y RSI, y mejora el Calmar.
2. **Degradación en el walk-forward:** la eficiencia es 0.15 en retorno y 0.03 en Calmar: sobrevive
   ~15% de la ventaja in-sample. El Calmar mediano por ventana cae de 2.5 IS a −0.8 OOS en
   validation, y de 3.4 a −0.6 en test. Las otras tres variantes tienen eficiencia negativa.
3. **Sensibilidad ±20%:** es una meseta moderada, no un pico aislado. La mediana de las 16
   variaciones conserva 91% del Calmar. Lo más sensible son las medias móviles (`sma_slow` ×1.2:
   −0.65 sobre 1.08).
4. **Costo de equilibrio:** en train la estrategia sigue rentable hasta 100 bps de ida y vuelta (margen
   > 71 bps frente a los 29 bps del caso base). Fuera de muestra el equilibrio queda alrededor de los
   29 bps especificados: el margen es casi nulo.
5. **Regímenes:** sí difieren, y reversión pierde en los tres bloques (Sharpe −0.76, −2.62 y −1.29).
   Tendencia gana en train y test (1.44 y 0.45). El θ por régimen le gana al θ único en los tres
   bloques (en test, +1.8% contra −3.5% anual).
6. **Risk Parity contra pesos iguales:** mejora el Calmar solo en test (0.07 contra 0.01), pero en
   los tres bloques reduce el drawdown (16% a 28% menor) y la volatilidad, e iguala las
   contribuciones al riesgo (12.5% cada activo, contra 1.6% a 23.7% con pesos iguales). El costo es
   menor retorno en train: subpondera la tecnología, que fue lo que más subió.
7. **Limitaciones para capital real:**
   - no hay evidencia de edge fuera de muestra;
   - sesgos del universo: supervivencia, concentración en tecnología y muestra alcista (los cortos
     restaron en los tres bloques);
   - ejecución optimista: llenado completo, slippage fijo, sin impacto ni rechazos.

**Advertencia de ejecución:** el backtest asume ejecución completa al precio modelado y no incorpora
impacto de mercado ni fallas de ejecución. Estimado ex post con el modelo de raíz cuadrada sobre las
224 operaciones de test:
- 1.5 bps promedio por llenado (p95 de 6.7 bps);
- 0.58% del equity del bloque, cerca de 23% de su retorno;
- COPX concentra el impacto, con 6.8 bps promedio.

El impacto crece con la raíz del tamaño: con 100 veces más capital sería ~10 veces mayor y borraría
la ganancia.

## Semilla

`SEED = 42`, definida en `CONFIG` dentro de `main.py`. Se pasa explícita a los samplers de Optuna, al
`random_state` de los modelos y a fANOVA. La ventana i del walk-forward usa la semilla `SEED + i`. No
se usa `np.random.seed` global. Con la misma semilla, `main.py` reproduce los mismos resultados.

## Commit de la corrida final

**`45bedfaaa0377322fbeffa18387666ccc5b362d6`** (`45bedfa`, en `main`): código congelado con
`final_run = True`, registrado antes de correr test. La corrida única con test terminó el 2026-10-06 a
las 19:05 (38.7 min) y `results/corrida.pkl` guarda ese mismo hash. Ese archivo marca
`uncommitted_changes = True` solo porque las figuras que la propia corrida escribió en
`docs/figuras/` todavía no estaban en git; ningún archivo con seguimiento cambió. Los commits
posteriores solo agregan figuras, notebooks ejecutados y documentación.

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
