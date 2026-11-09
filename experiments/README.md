# Phase 4B: validation experiments

These scripts and results are on `exp/feature-engineering`. They do not replace
`models/xgb_home_credit.joblib`, `predict.py`, `Dockerfile`, or the production
decision threshold.

## Reproduce

With Python 3.11 and the authorized CSV at `data/raw/application_train.csv`:

```bash
uv sync --group dev
uv run python experiments/feature_engineering.py --cv-folds 3
uv run python experiments/cross_validate_xgb.py
uv run python experiments/feature_ablation.py
uv run python experiments/feature_isolation.py
uv run python -m experiments.cv_confirmation
```

LightGBM 4.7.0 is pinned in the `dev` dependency group; the Dockerfile installs
runtime dependencies with `--no-dev`. The scripts use seed 42, two threads, and
the established stratified split: 60% training, 20% validation, and 20% test.
Only training and validation feature rows are materialized for the comparisons.
Cross-validation and ablation use the partitions described below; no test metrics
are calculated.

## Transformations

`HomeCreditFeatures` is a scikit-learn transformer with `fit` and `transform`.
Non-positive denominators (and zero for age) become missing values. The
`DAYS_EMPLOYED=365243` sentinel is removed from the continuous value and represented
by an indicator. The transformer calculates five ratios and row-wise mean and
standard deviation for EXT_SOURCE. Ratio clipping limits are the 0.5th and 99.5th
percentiles fitted only on rows passed to `fit`; each cross-validation fold learns
its own limits. The preprocessing pipeline imputes remaining missing values.

## Results

- `validation_results.json`: metrics, parameters, versions, fit times, RSS, seeds,
  and three-fold stratified cross-validation.
- `feature_ablation.json`: XGBoost comparison without engineered features and
  one-at-a-time feature ablations against the full feature set.
- `feature_isolation.json`: test of `CREDIT_ANNUITY_RATIO` as the only engineered
  input feature.
- `cv_confirmation_results.json`: four-candidate confirmation on one shared set
  of three stratified training folds, with paired fold differences and a local
  FastAPI/joblib compatibility smoke check.

Validation metrics guide the experiment and are not an independent final
evaluation. The `CODE_GENDER` variant is a predictive sensitivity check, not a
fairness evaluation.
