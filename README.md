# Home Credit Default Risk Prediction

This educational classification project estimates whether a loan applicant may have difficulty repaying a loan (`TARGET`). It uses only `application_train.csv` from the official [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk). Kaggle may require an account and acceptance of the competition rules to download the file. No auxiliary tables are used. This model is not suitable for real credit decisions.

## Setup

1. Download the authorized `application_train.csv` from the competition and place it at `data/raw/application_train.csv`.
2. Install uv and project dependencies with `uv sync` (Python 3.11).
3. To execute the notebook from the repository root, run `uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`.

`data/` is excluded from Git to keep the dataset out of version control.

## Phase 1: exploratory data analysis and baseline

The notebook reviews dimensions, data types, missing values, duplicates, distributions, and the target balance. `SK_ID_CURR` is excluded from the features, and `TARGET` remains separate as the label. The data is split into stratified 60/20/20 training, validation, and test sets (`random_state=42`). Imputation, scaling, and encoding are fitted inside a pipeline using only the training set. Logistic regression is evaluated on validation with ROC-AUC, Average Precision, precision, recall, and a confusion matrix. The test set is reserved.

## Phase 2: model comparison on validation

The Phase 1 split is preserved, and the test set is not used for model selection. The notebook compares the logistic regression baseline with two small Random Forest and XGBoost configurations. Average Precision is the primary metric because the classes are imbalanced; ROC-AUC is a complementary metric. XGBoost (180 trees, depth 4, learning rate 0.06) achieved the highest Average Precision. The notebook also records a Precision–Recall curve, threshold comparisons, feature importances, and approximate resource use. These results support validation-based model selection; the test set remains reserved.

## Phase 3: training and local API

The complete input contract (120 columns and observed types) is in `feature_schema.json`. Each request to `/predict` must include a JSON object with all expected features. `TARGET`, `SK_ID_CURR`, missing fields, and extra fields are rejected. Represent missing values as `null`; the pipeline imputes them. The response contains `default_probability`, `predicted_default`, and the frozen decision threshold `0.161378`.

To train from scratch, place the authorized competition file at `data/raw/application_train.csv` and run:

```bash
uv sync
uv run python train.py
```

The script recreates the stratified 60/20/20 split with seed 42, trains on training plus validation, saves the full pipeline to `models/xgb_home_credit.joblib`, and records the single final test evaluation in `reports/test_metrics.json`. The CSV and local report are excluded from Git; the approved `models/xgb_home_credit.joblib` artifact is included.

Start the service in a terminal:

```bash
uv run uvicorn predict:app --host 127.0.0.1 --port 8000
```

The HTTP client uses `examples/predict_request.json` by default. It is synthetic and contains neither `TARGET` nor `SK_ID_CURR`. To create a request from one local CSV row, pass `--data`; the client removes those two columns:

```bash
uv run python test_predict.py
```

Check the API with `GET /health` and `POST /predict`. Run the consistency tests with `uv run pytest`.

This project and its metrics are for educational use only and must not be used for real credit decisions.

## Phase 3 results

The final training run (180 trees, depth 4, `learning_rate=0.06`, threshold `0.161378`) achieved the following on the single test evaluation: ROC-AUC **0.7607**, Average Precision **0.2476**, precision **0.2620**, and recall **0.3768**. The confusion matrix was `[[51269, 5269], [3094, 1871]]`. On validation, the variant without `CODE_GENDER` achieved ROC-AUC 0.753782 and Average Precision 0.242827, compared with 0.755089 and 0.243390 with the feature; this did not change model selection.

The automated test confirms that `predict_record` and FastAPI return exactly the same prediction for a training row. The Phase 2 notebook evaluates a candidate trained on the training set. The final model is refit on training plus validation, so its probabilities are not a row-by-row equivalent comparison, even though the preprocessing is the same.

## Phase 4: local Docker deployment

### Requirements

- Docker Engine with an active daemon.
- The `models/xgb_home_credit.joblib` artifact and `feature_schema.json` included in the clone.
- The original CSV is not required to build or run the container.

The approved model is the only Git exception under `models/`; CSV files, environments, caches, and other model artifacts remain excluded. The image installs runtime dependencies from `uv.lock` with `uv sync --locked --no-dev`, uses `xgboost-cpu==3.2.0` compatible with the artifact, listens on `0.0.0.0:8000`, and runs as a non-root user.

### Build and start

```bash
docker build -t home-credit-risk-api .
docker run --rm -d --name home-credit-risk-api -p 8000:8000 home-credit-risk-api
```

### Check health and make a prediction

```bash
curl http://localhost:8000/health
uv run python test_predict.py --url http://127.0.0.1:8000
```

The response includes `default_probability`, `predicted_default`, and `threshold`. The container's `HEALTHCHECK` queries `/health`; inspect it with:

```bash
docker inspect --format='{{.State.Health.Status}}' home-credit-risk-api
```

Stop and remove the container:

```bash
docker stop home-credit-risk-api
```

The image contains the model and schema, never `data/`. A clean clone can serve predictions using the included artifact without downloading or mounting the dataset.

## Phase 4B: controlled feature engineering experiments

The `exp/feature-engineering` branch contains the recorded validation and cross-validation experiments. See [`experiments/README.md`](experiments/README.md) for reproduction steps and [`experiments/REPORT.md`](experiments/REPORT.md) for results and limitations. These experiments do not replace the production model or alter its threshold.
