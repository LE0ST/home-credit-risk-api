# Fase 4B: experimentos de validación

Estos scripts y resultados pertenecen a `exp/feature-engineering`. No sustituyen
`models/xgb_home_credit.joblib`, `predict.py`, `Dockerfile` ni el umbral funcional.

## Reproducir

Con Python 3.11 y el CSV autorizado `data/raw/application_train.csv`:

```bash
uv sync --group dev
uv run python experiments/feature_engineering.py --cv-folds 3
uv run python experiments/cross_validate_xgb.py
uv run python experiments/feature_ablation.py
uv run python experiments/feature_isolation.py
```

LightGBM 4.7.0 está fijado en el grupo de dependencias `dev`; el Dockerfile
instala dependencias de runtime con `--no-dev`. Los tres scripts usan semilla 42,
dos hilos y el split estratificado ya definido: train 60%, validation 20%, test
20%. Para evaluación, materializan únicamente train y validation. CV y ablación
operan solo sobre train o validation según se indica; nunca calculan métricas para
test.

## Transformaciones

`HomeCreditFeatures` es un transformador scikit-learn con `fit` y `transform`.
Convierte denominadores no positivos (o cero para edad) en faltantes, elimina el
sentinela `DAYS_EMPLOYED=365243` del valor continuo y agrega su indicador. Calcula
las cinco razones, así como media y desviación estándar fila a fila de EXT_SOURCE.
Los límites de recorte de razones son los cuantiles 0.5% y 99.5% aprendidos solo
con las filas de `fit`; en CV cada fold aprende sus propios límites. El resto de
faltantes se imputa en el pipeline de preprocesamiento.

## Resultados

- `validation_results.json`: métricas, parámetros, versiones, tiempos, RSS,
  configuración de semillas y CV estratificada de tres folds.
- `feature_ablation.json`: comparación XGBoost sin nuevas features y ablación de
  una feature cada vez respecto al conjunto completo.
- `feature_isolation.json`: prueba de `CREDIT_ANNUITY_RATIO` como única feature
  derivada.

Las métricas de validation orientan la investigación y no son una evaluación
independiente final. La variante sin `CODE_GENDER` es una sensibilidad de entrada;
no es una evaluación de equidad.
