# Informe Fase 4B

## Protocolo

- Checkpoint funcional: `af6fce6` (`feat: establish reproducible home credit ML baseline`).
- Rama original `master` conserva ese commit. La exploración está en
  `exp/feature-engineering`.
- Semilla 42; split estratificado 60/20/20, igual al del proyecto; dos hilos.
- El CSV de desarrollo se usó para construir train y validation. No se calcularon
  métricas, umbrales ni comparaciones en la partición test. Los scripts no guardan
  modelos candidatos.
- Versiones: scikit-learn 1.9.1, XGBoost 3.2.0 y LightGBM 4.7.0.
- El preprocesamiento y los límites de recorte se ajustan dentro de cada pipeline;
  en CV se vuelven a ajustar desde cero en cada fold.

## Resultados en validation

Las métricas del baseline son las comunicadas antes de esta fase. Los tiempos son
fit más predicción de validation. RSS es el máximo observado del proceso durante
cada ajuste; incluye los datos y el preprocesamiento.

| Modelo | ROC-AUC | Average Precision | Segundos | RSS máximo |
|---|---:|---:|---:|---:|
| XGBoost referencia comunicada | 0.755089 | 0.243390 | — | — |
| XGBoost base + 8 features derivadas | 0.759695 | 0.247879 | 10.03 | 1,526 MiB |
| XGBoost ajustado + features | 0.760463 | 0.248969 | 11.04 | 1,736 MiB |
| XGBoost base + features, sin `CODE_GENDER` | 0.758950 | 0.247849 | 9.85 | 1,624 MiB |
| LightGBM inicial + features | 0.766480 | 0.254957 | 20.46 | 1,596 MiB |
| LightGBM ajustado + features | **0.767261** | **0.256644** | 26.25 | 1,780 MiB |

Parámetros de XGBoost base: 180 árboles, profundidad 4, tasa 0.06, `min_child_weight=10`,
`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5`, `max_bin=128`.
El ajuste pequeño usa 220 árboles, tasa 0.05 y `min_child_weight=8`; conserva el
resto. LightGBM inicial usa 500 árboles, tasa 0.05, 31 hojas, mínimo 50 filas por
hoja y `reg_lambda=5`. El ajuste usa 700 árboles, tasa 0.035, 31 hojas, mínimo 100
filas por hoja y `reg_lambda=8`; ambos usan muestreo de filas/columnas 0.8 y dos
hilos.

La variante ajustada de LightGBM lidera el split fijo por +0.012172 ROC-AUC y
+0.013254 AP frente a la referencia comunicada. XGBoost con las nuevas features
mejora +0.004606 ROC-AUC y +0.004489 AP frente a esa referencia. La ablación
reentrenada sin ingeniería dio 0.755400 / 0.244557, mientras el XGBoost base con
todas las features dio 0.759695 / 0.247879; ese control local difiere ligeramente
de las métricas históricas.

## Contribución de las features

Se probaron `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`,
`CREDIT_ANNUITY_RATIO`, `EMPLOYED_TO_AGE_RATIO`, `INCOME_PER_PERSON`,
`EXT_SOURCE_MEAN`, `EXT_SOURCE_STD` e indicador de `DAYS_EMPLOYED=365243`.
Denominadores no positivos se convierten en faltantes (cero también para la edad),
se reemplaza el sentinela y se agrega su indicador, y las razones se recortan en
cuantiles 0.5%/99.5% aprendidos solo en train.

En la ablación XGBoost sobre el mismo split, el conjunto de features subió AP de
0.244557 a 0.247879. `CREDIT_ANNUITY_RATIO` fue la feature con señal marginal más
clara: al quitarla de las ocho, AP cayó a 0.244489; como única feature derivada,
obtuvo 0.250715 (ROC-AUC 0.761262). Quitar cualquiera de las otras siete, una a una,
produjo un AP mayor que el conjunto completo en este split. Esto sugiere probar un
conjunto más pequeño; no prueba que esas variables sean universalmente inútiles.

Sin `CODE_GENDER`, XGBoost obtuvo 0.758950 / 0.247849, frente a
0.759695 / 0.247879 con la columna. La diferencia en AP es −0.000030. Es una
comparación predictiva de validation y no una evaluación de equidad.

## Validación cruzada

CV estratificada de tres folds, únicamente sobre las 184,506 filas del train (la
validation de 61,502 filas quedó fuera). Los valores son media ± desviación
estándar entre folds:

| Modelo | ROC-AUC | Average Precision | Tiempo total CV |
|---|---:|---:|---:|
| LightGBM ajustado + features | 0.762720 ± 0.001538 | **0.245235 ± 0.005388** | 54.44 s |
| LightGBM inicial + features | 0.760665 ± 0.001728 | 0.242435 ± 0.005484 | 42.20 s |
| XGBoost ajustado + features | 0.757857 ± 0.001854 | 0.240012 ± 0.006688 | 22.50 s |

La CV respalda a LightGBM en promedio, aunque su AP varía entre folds. Sus folds
entrenan con menos filas que el ajuste usado para validation, así que sus valores
absolutos no son directamente equivalentes a las métricas del split fijo.

## Candidato y límites

Propongo **LightGBM ajustado con features** para una siguiente ronda de confirmación:
lidera ROC-AUC y AP en validation, y también las medias CV entre los candidatos
probados. La variante XGBoost que usa solo `CREDIT_ANNUITY_RATIO` es otra hipótesis
prometedora y más sencilla, pero solo se midió en el split fijo y requiere CV antes
de considerarla con más peso. No se propone ningún cambio de producción ni umbral.

Las ablaciones múltiples se hicieron sobre una misma validation, por lo que pueden
favorecer accidentalmente algunas combinaciones. No se evaluó test, no se creó un
nuevo umbral y no se entrenó con datos privados de GCI World. Docker/API no se
modificaron ni se volvieron a ejecutar; sus pruebas y documentación se conservan en
el commit baseline. El entorno local recibió LightGBM solo como dependencia `dev`.
