# Home Credit Default Risk Prediction

Proyecto educativo de clasificación para estudiar si una solicitud de crédito presenta dificultad de pago (`TARGET`). Se utiliza exclusivamente `application_train.csv` de la [competición oficial Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk). Kaggle puede exigir cuenta y aceptación de las reglas para obtener el archivo. No se descargan tablas auxiliares. Este modelo no es apto para decisiones crediticias reales.

## Preparación

1. Descarga `application_train.csv` desde la competición oficial y colócalo en `data/raw/application_train.csv`.
2. Instala uv y las dependencias con `uv sync` (Python 3.11).
3. Ejecuta el notebook desde la raíz: `uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`.

`data/` está excluida de Git para evitar versionar los datos.

## Fase 1

El notebook presenta dimensiones, tipos, faltantes, duplicados, distribuciones y balance de `TARGET`. Se excluye `SK_ID_CURR`; el objetivo se mantiene separado. La partición estratificada es 60/20/20 (`random_state=42`). Imputación, escalado y codificación se ajustan dentro de un pipeline entrenado solo en train. La regresión logística se evalúa en validation con ROC-AUC, Average Precision, precision, recall y matriz de confusión. Test queda reservado.

## Fase 2: comparación en validation

Se conserva el split de Fase 1 y no se consulta test. El notebook compara la regresión logística de referencia con dos configuraciones pequeñas de Random Forest y XGBoost, priorizando Average Precision y usando ROC-AUC como métrica complementaria. El candidato con mayor Average Precision fue XGBoost (180 árboles, profundidad 4, tasa de aprendizaje 0,06). Se documentan curva Precision–Recall, umbrales, importancias y recursos aproximados. Los resultados sirven para selección en validation; test sigue reservado.

## Fase 3: entrenamiento y API local

El contrato completo de entrada (120 columnas y tipos observados) está en `feature_schema.json`. Cada solicitud a `/predict` debe enviar un objeto JSON con todas esas features; `TARGET`, `SK_ID_CURR`, campos ausentes o columnas extra se rechazan. Los faltantes se representan como `null` y se imputan dentro del pipeline. La respuesta contiene `default_probability`, `predicted_default` y el umbral congelado `0.161378`.

Para entrenar desde cero, coloca el archivo oficial autorizado en `data/raw/application_train.csv` y ejecuta:

```bash
uv sync
uv run python train.py
```

El script reproduce el split 60/20/20 con semilla 42, entrena con train+validation, guarda el pipeline completo en `models/xgb_home_credit.joblib` y registra la única evaluación de test en `reports/test_metrics.json`. El CSV y el reporte local están excluidos de Git; el artefacto aprobado `models/xgb_home_credit.joblib` se conserva en Git.

Inicia el servicio en una terminal:

```bash
uv run uvicorn predict:app --host 127.0.0.1 --port 8000
```

El cliente HTTP usa por defecto `examples/predict_request.json`, un ejemplo sintético sin `TARGET` ni `SK_ID_CURR`. Si se desea probar con una fila local del CSV, puede pasarse `--data` (el cliente elimina esas dos columnas):

```bash
uv run python test_predict.py
```

Comprueba la API con `GET /health` y `POST /predict`. Las pruebas de consistencia se ejecutan con `uv run pytest`.

Este proyecto y sus métricas son exclusivamente educativos; no deben utilizarse para decisiones crediticias reales.

## Resultado Fase 3

El entrenamiento final (180 árboles, profundidad 4, `learning_rate=0.06`, umbral `0.161378`) obtuvo en la única evaluación test: ROC-AUC **0.7607**, Average Precision **0.2476**, precision **0.2620**, recall **0.3768**. Matriz de confusión `[[51269, 5269], [3094, 1871]]`. La variante sin `CODE_GENDER` en validation tuvo ROC-AUC 0.753782 y Average Precision 0.242827 frente a 0.755089 y 0.243390 con la variable; no cambió la selección.

La prueba automatizada confirma igualdad exacta entre la predicción directa de `predict_record` y la respuesta de FastAPI para la misma fila de train. El notebook de Fase 2 evalúa el candidato entrenado en train; el modelo final se vuelve a ajustar en train+validation, por lo que sus probabilidades no son una comparación fila-a-fila equivalente aunque el preprocesamiento sea el mismo.


## Fase 4: Docker local

### Requisitos

- Docker Engine con daemon activo.
- El artefacto `models/xgb_home_credit.joblib` y `feature_schema.json` incluidos en el clon.
- No se necesita el CSV original para construir ni ejecutar el contenedor.

El modelo aprobado es la única excepción de Git bajo `models/`; los CSV, entornos, cachés y demás archivos de modelo permanecen excluidos. La imagen instala las dependencias de runtime desde `uv.lock` con `uv sync --locked --no-dev`; usa `xgboost-cpu==3.2.0`, compatible con el artefacto, escucha en `0.0.0.0:8000` y ejecuta como usuario no privilegiado.

### Construir e iniciar

```bash
docker build -t home-credit-risk-api .
docker run --rm -d --name home-credit-risk-api -p 8000:8000 home-credit-risk-api
```

### Comprobar health y predecir

```bash
curl http://localhost:8000/health
uv run python test_predict.py --url http://127.0.0.1:8000
```

La respuesta incluye `default_probability`, `predicted_default` y `threshold`. El contenedor tiene un `HEALTHCHECK` que consulta `/health`; se puede inspeccionar con:

```bash
docker inspect --format='{{.State.Health.Status}}' home-credit-risk-api
```

Detener y eliminar el contenedor:

```bash
docker stop home-credit-risk-api
```

La imagen contiene el modelo y el esquema, nunca `data/`. Un clon limpio puede servir predicciones con el artefacto incluido sin descargar ni montar el dataset.
