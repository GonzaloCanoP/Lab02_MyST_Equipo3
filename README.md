# Lab02_MyST_Equipo3 — Estrategias de Trading con Análisis Técnico

Microestructuras y Sistemas de Trading (IT1731B) · ITESO, Otoño 2026

## Integrantes

- Gonzalo Cano Padilla
- Juan Manuel Espinosa Cárdenas
- Jerónimo Rojas Alvarado
- Raúl Zanatta Casas

*[Pendiente: asignación de partes P1 a P4.]*

**Nivel de alcance:** C

## Descripción

Estrategia direccional larga y corta sobre 8 activos líquidos de EE.UU. (AAPL, MSFT, META, AMD, XOM,
SMH, GLD y COPX) con barras diarias de 2017 a 2026. La entrada exige que al menos 2 de 3 indicadores
técnicos (cruce de SMA, histograma MACD y RSI) coincidan en dirección; el ATR fija el stop-loss, el
take-profit y el tamaño de cada operación (fixed fractional). El capital se reparte entre activos con
Risk Parity, escalado por la fuerza de la señal y por un multiplicador según el régimen de mercado
(tendencia, reversión o crisis), detectado con una etiqueta filtrada y causal. Los parámetros se
optimizan con Optuna en un walk-forward de 6 meses de entrenamiento y 1 mes de prueba, por régimen, y
el resultado se compara contra un portafolio de pesos iguales. Las reglas completas están en
`instrucciones/SPEC.md` y `instrucciones/SPEC_portafolio.md`.

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
python main.py
```

Corre todo sin intervención ni red: carga de datos, auditoría, régimen, optimización, walk-forward,
backtests finales y métricas. Los resultados se guardan en `results/` y las figuras en `docs/figuras/`.
Los datos están congelados en `data/` (un CSV por activo y `IRX.csv` para la tasa libre de riesgo);
solo si falta algún archivo se descargan de Yahoo Finance.

Pruebas:

```bash
python -m pytest
```

## Semilla

`SEED = 42`, definida en `CONFIG` dentro de `main.py`. Se pasa explícita a los samplers de Optuna, al
`random_state` de los modelos y a `np.random.default_rng`. La ventana i del walk-forward usa la
semilla `SEED + i`. No se usa `np.random.seed` global.

## Commit de la corrida final

*[Pendiente: hash del commit con los θ congelados antes de correr test.]*

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
  sobre train, el guardado de resultados y figuras, la estimación de impacto sobre test y la tabla de
  auditoría de sesgos generada en `results/auditoria_sesgos.md`. Las decisiones abiertas (unidades del
  ADV, ubicación de la función, dónde vive la auditoría) se confirmaron antes de implementarlas.

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
