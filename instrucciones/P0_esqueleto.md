# P0 — Esqueleto del proyecto

Hereda todo de `instrucciones/CLAUDE.md`. Se hace directo en `main`, antes de
abrir las ramas de los demás. NO se implementa lógica.

## Las tareas (del enunciado, sección 2.3)
1. "El repositorio debe contener esta estructura. No se aceptan archivos sueltos en la raíz distintos a
   los listados."
2. "README.md debe contener: nombre de los integrantes, nivel de alcance (A, B o C), descripción del
   proyecto en un párrafo, instrucciones de instalación y el comando exacto para reproducir todos los
   resultados."
3. "main.py debe ejecutar el proyecto completo con un solo comando: python main.py"
4. "La semilla aleatoria debe estar fijada y documentada en el README."

## Cómo encaja
- Estructura: CLAUDE.md, sección 2. Los datos del portafolio ya están en `data/`; no se mueven ni se
  renombran.
- Stubs: cada función de CLAUDE.md, sección 7, con su firma exacta, type hints, docstring (qué recibe,
  qué devuelve, qué punto del SPEC implementa) y cuerpo `raise NotImplementedError`. `BacktestResult`
  queda definido completo, porque es un dataclass sin lógica.
- `main.py`: el dict `CONFIG` con los valores de `SPEC.md` y una función por etapa (`stage_load`,
  `stage_regimes`, `stage_diagnostics`, `stage_walk_forward`, `stage_final_backtests`, `stage_report`)
  que llama a los stubs en orden.
- `tests/conftest.py`: fixture `synthetic_prices(n_assets=3, n_days=600, seed=42)` con OHLCV sintético
  coherente (`low ≤ open, close ≤ high`, volumen positivo) y fixture `config_test` con los valores de
  `CONFIG`.
- `README.md`: secciones vacías, con una subsección "Uso de IA" por integrante.
- `.gitignore`: `results/`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `.DS_Store`.
- `requirements.txt`: numpy, pandas, yfinance, scipy, scikit-learn, hmmlearn, optuna, matplotlib,
  joblib, pytest, jupyter, con versiones fijas tomadas de `pip freeze`.
- `.claude/CLAUDE.md` con una sola línea: `@../instrucciones/CLAUDE.md`.
- Versión de Python: 3.13.9 (confirmada).

## Entregable
- Estructura completa en `main`, en uno o varios commits `chore(esqueleto): ...`.
- Una rama por parte (`p1-<integrante>` a `p4-<integrante>`), creadas desde ese `main` una vez asignadas
  las partes (CLAUDE.md, sección 14).
- Regla de protección de `main` activa en GitHub (requiere PR).

## Checklist
- [ ] El árbol coincide con CLAUDE.md, sección 2; no hay archivos sueltos extra en la raíz
- [ ] `python -c "import src.data, src.signals, src.backtest, src.metrics, src.optimize, src.regimes, src.portfolio, src.plots"` termina sin error
- [ ] Toda función de la sección 7 existe con su firma exacta y lanza `NotImplementedError`
- [ ] `python -m pytest` termina sin errores de recolección
- [ ] Los valores de `CONFIG` coinciden uno a uno con `SPEC.md`
- [ ] `/memory` dentro de Claude Code muestra cargado `instrucciones/CLAUDE.md`
- [ ] Las 4 ramas existen y `main` está protegida
