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

### Juan Manuel Espinosa Cárdenas

*[Pendiente]*

### Jerónimo Rojas Alvarado

*[Pendiente]*

### Raúl Zanatta Casas

*[Pendiente]*
