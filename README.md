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

The complete input contract (120 columns and observed types) is in `feature_schema.json`. Each request to `/predict` must include a JSON object with all expected features. `TARGET`, `SK_ID_CURR`, missing fields, and extra fields are rejected. Represent missing values as `null`; the pipeline imputes them. The response contains `default_probability`, `predicted_default`, and the selected model’s decision threshold. The XGBoost baseline uses `0.161378`; LightGBM uses `0.1566716424524132`.

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

## Production Docker image

The image serves the included LightGBM pipeline by default and also includes the
preserved XGBoost baseline. It needs Docker, both artifacts in `models/`, and
no competition CSV. The build context is restricted to the production inputs;
the runtime contains neither training data nor development tools.

The multi-stage build uses Python 3.11, uv 0.12.23 and `uv.lock` to install only
runtime dependencies. It uses CPU-only XGBoost, LightGBM, and `libgomp1` for
OpenMP, and runs as an unprivileged user. `credit_features.py` supplies the
feature transformer required to load the LightGBM artifact. The prediction
pipeline computes all engineered features from the original 120 fields.

Build and run LightGBM on localhost:

```bash
docker build -t home-credit-risk-api:lightgbm .
docker run --rm -d --name home-credit-lightgbm -p 127.0.0.1:8000:8000 home-credit-risk-api:lightgbm
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data-binary @examples/predict_request.json
docker inspect --format='{{.State.Health.Status}}' home-credit-lightgbm
docker stop home-credit-lightgbm
```

To serve the original XGBoost artifact with its own threshold, use a separate
container and port:

```bash
docker run --rm -d --name home-credit-xgboost -e HOME_CREDIT_MODEL=xgboost -p 127.0.0.1:8001:8000 home-credit-risk-api:lightgbm
curl http://127.0.0.1:8001/health
curl -X POST http://127.0.0.1:8001/predict -H 'Content-Type: application/json' --data-binary @examples/predict_request.json
docker stop home-credit-xgboost
```

Use `uv sync --locked` and `uv run pytest` to reproduce local verification.
To reproduce the LightGBM artifact, obtain the authorized CSV as described above
and run `uv run python train_lightgbm.py`. It selects the threshold by maximum
F1 on the fixed validation split, then fits on train plus validation; it does
not evaluate the held-out test set. The recorded LightGBM validation precision,
recall and F1 are 0.2516, 0.4226 and 0.3154 respectively. These validation
results must not be compared as if they were the XGBoost test results reported
in Phase 3. They are educational measurements, not evidence for real lending
or credit decisions.

## Phase 4B: controlled feature engineering experiments

The `exp/feature-engineering` branch contains the recorded validation and cross-validation experiments. See [`experiments/README.md`](experiments/README.md) for reproduction steps and [`experiments/REPORT.md`](experiments/REPORT.md) for results and limitations. The selected tuned LightGBM configuration is now integrated as the default model; the XGBoost baseline and its threshold remain available separately.

### Model selection

The default API model is the tuned LightGBM pipeline, which computes the approved
features internally. Set `HOME_CREDIT_MODEL=xgboost` to serve the preserved
XGBoost baseline and its own threshold. The LightGBM threshold is selected by
maximum F1 on the fixed validation split; see `reports/lightgbm_validation.json`.
Run `uv run python train_lightgbm.py` to reproduce the development-only fit.
The held-out test set is not used by this script.

## Interpreting this educational workflow

The positive class represents repayment difficulty and accounts for about 8.07%
of applications. False negatives miss applicants who experience difficulty;
false positives flag applicants who would repay. Changing a threshold trades
these errors against each other. ROC-AUC describes ranking across thresholds;
Average Precision emphasizes positive-class retrieval under this imbalance.
Neither establishes calibration or suitability for lending.

This Kaggle Home Credit Default Risk dataset is a separate competition dataset
from the course Module 6 CreditScoring dataset; their schemas, populations and
metrics are not interchangeable. Obtain the official Kaggle training CSV after
accepting its access rules and place it in `data/raw/application_train.csv`.
The established seed-42 stratified split is 60/20/20. Candidate preprocessing is
fit on training rows; fixed validation and training-only three-fold CV guide
selection. The final LightGBM fit uses train plus validation. Its reported
validation metrics and F1-optimized threshold are selection estimates and can
be optimistic. They are not independent test results. The historical XGBoost
test evaluation is reported separately; do not rerun it for further tuning.

The notebook contains aggregate EDA, validation comparisons and diagnostic plots,
not applicant records. Associations and feature importance are not causal claims.
Removing gender alone is not a fairness audit: proxies, subgroup performance,
calibration, consent and deployment impacts have not been adequately assessed.
This project is educational only and must not determine real credit decisions.

`examples/predict_request.json` demonstrates all-null missing-value handling.
`examples/predict_request_synthetic.json` is a hand-authored hypothetical request
with correctly typed numerical values; unspecified fields remain null. Neither
request is copied from a real applicant. The test suite uses these committed
examples and requires no Kaggle dataset.

To reproduce the aggregate notebook outputs from a clean kernel, run from the
repository root with the development dependencies installed:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebook.ipynb
```

This reruns historical training/validation candidate comparisons, not the final
artifact training scripts. Approved artifacts are only read for descriptive
importance summaries. Notebook Phase 4B tables explicitly load the recorded
experiment JSON files; they are historical results, not newly rerun experiments.

The remediation notebook rerun produced XGBoost 2 validation ROC-AUC 0.755400,
Average Precision 0.244557, and a diagnostic maximum-F1 threshold of 0.151809.
These newly executed results differ from historical recorded measurements. They
do not replace the frozen XGBoost production threshold (0.161378), the LightGBM
threshold, either approved artifact, or the historical experiment JSON records.

See the metric provenance review in `experiments/REPORT.md`: the notebook agrees
with the saved no-engineering control at displayed precision. The older reference
lacks enough provenance to identify its discrepancy conclusively.
