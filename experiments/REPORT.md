# Phase 4B report

## Protocol

- Functional checkpoint: `af6fce6` (`feat: establish reproducible home credit ML baseline`).
- The original `master` branch retains that commit. Experiments are on
  `exp/feature-engineering`.
- Seed 42; the established stratified 60/20/20 split; two threads.
- The development CSV was used to construct the training and validation sets. No
  metrics, thresholds, or model comparisons were computed on the test partition.
  The scripts do not save candidate models.
- Versions: scikit-learn 1.9.1, XGBoost 3.2.0, and LightGBM 4.7.0.
- Preprocessing and clipping limits are fitted within each pipeline. Each CV fold
  fits them again using that fold's training rows only.

## Validation results

Baseline metrics below are those reported before this phase. Times include fitting
and validation prediction. RSS is the maximum process resident memory observed
during each fit, including data and preprocessing.

| Model | ROC-AUC | Average Precision | Seconds | Peak RSS |
|---|---:|---:|---:|---:|
| Reported XGBoost reference | 0.755089 | 0.243390 | — | — |
| Base XGBoost + 8 engineered features | 0.759695 | 0.247879 | 10.03 | 1,526 MiB |
| Tuned XGBoost + features | 0.760463 | 0.248969 | 11.04 | 1,736 MiB |
| Base XGBoost + features, without `CODE_GENDER` | 0.758950 | 0.247849 | 9.85 | 1,624 MiB |
| Initial LightGBM + features | 0.766480 | 0.254957 | 20.46 | 1,596 MiB |
| Tuned LightGBM + features | **0.767261** | **0.256644** | 26.25 | 1,780 MiB |

Base XGBoost parameters: 180 trees, depth 4, learning rate 0.06,
`min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5`,
and `max_bin=128`. The small adjustment uses 220 trees, learning rate 0.05, and
`min_child_weight=8`; the remaining parameters are unchanged. Initial LightGBM
uses 500 trees, learning rate 0.05, 31 leaves, 50 minimum rows per leaf, and
`reg_lambda=5`. The adjustment uses 700 trees, learning rate 0.035, 31 leaves,
100 minimum rows per leaf, and `reg_lambda=8`; both use row/column sampling of 0.8
and two threads.

Tuned LightGBM leads on the fixed validation split by +0.012172 ROC-AUC and
+0.013254 AP against the reported reference. XGBoost with all engineered features
improves by +0.004606 ROC-AUC and +0.004489 AP. A local refit without feature
engineering scored 0.755400 / 0.244557, while base XGBoost with all engineered
features scored 0.759695 / 0.247879; the local control differs slightly from the
historical metrics.

## Feature contribution

The experiment evaluated `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`,
`CREDIT_ANNUITY_RATIO`, `EMPLOYED_TO_AGE_RATIO`, `INCOME_PER_PERSON`,
`EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`, and the `DAYS_EMPLOYED=365243` indicator.
Non-positive denominators become missing values (zero is also invalid for age),
the sentinel is replaced and flagged, and ratios are clipped to the 0.5th and
99.5th percentiles fitted on training rows only.

In the XGBoost ablation on the same split, the full feature set raised AP from
0.244557 to 0.247879. `CREDIT_ANNUITY_RATIO` showed the clearest marginal signal:
removing it from the eight features reduced AP to 0.244489; as the only engineered
input feature, it achieved 0.250715 (ROC-AUC 0.761262). Removing any of the other
seven features one at a time produced higher AP than the full set on this split.
This suggests testing a smaller set; it does not show that these features are
universally unhelpful.

Without `CODE_GENDER`, XGBoost scored 0.758950 / 0.247849, compared with
0.759695 / 0.247879 with the feature. The AP difference is −0.000030. This is a
validation prediction comparison, not a fairness evaluation.

## Cross-validation

Three-fold stratified CV used only the 184,506 training rows; the 61,502 validation
rows were excluded. Values are mean ± standard deviation across folds:

| Model | ROC-AUC | Average Precision | Total CV time |
|---|---:|---:|---:|
| Tuned LightGBM + features | 0.762720 ± 0.001538 | **0.245235 ± 0.005388** | 54.44 s |
| Initial LightGBM + features | 0.760665 ± 0.001728 | 0.242435 ± 0.005484 | 42.20 s |
| Tuned XGBoost + features | 0.757857 ± 0.001854 | 0.240012 ± 0.006688 | 22.50 s |

CV supports LightGBM on average, although its AP varies across folds. Each fold is
trained on fewer rows than the model evaluated on the fixed validation split, so
the absolute values are not directly comparable.

## Candidate and limitations

Tuned **LightGBM with the engineered features** is the candidate proposed for a
further confirmation round: it leads ROC-AUC and AP on validation and also has the
best CV means among the tested candidates. XGBoost with only `CREDIT_ANNUITY_RATIO`
is another promising, simpler hypothesis, but it was measured only on the fixed
split and needs CV before it should carry more weight. No production change or
threshold change is proposed.

Multiple ablations used the same validation split and may favor some combinations
by chance. The test set was not evaluated, no new threshold was selected, and no
private GCI World data was used for training. Docker and the API were not changed
or rerun; their tests and documentation remain in the baseline commit. LightGBM
was added only to the local development dependency group.
