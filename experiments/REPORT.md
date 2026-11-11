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

## Historical candidate recommendation and limitations

At the time of this experiment, tuned **LightGBM with the engineered features** was proposed for a
further confirmation round: it leads ROC-AUC and AP on validation and also has the
best CV means among the tested candidates. XGBoost with only `CREDIT_ANNUITY_RATIO`
is another promising, simpler hypothesis, but it was measured only on the fixed
split and needs CV before it should carry more weight. That experiment itself did not change production or its threshold; the subsequent
approved integration made tuned LightGBM the default.

Multiple ablations used the same validation split and may favor some combinations
by chance. The test set was not evaluated, no new threshold was selected, and no
private GCI World data was used for training. Docker and the API were not changed
or rerun; their tests and documentation remain in the baseline commit. LightGBM
was added only to the local development dependency group.

## Controlled four-candidate CV confirmation

All four candidates were evaluated on the same three folds from the established
184,506-row training partition. Fold indices were generated once with
`StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` and reused for each
candidate. Preprocessing and ratio clipping were fitted inside each fold. The
held-out validation and test feature rows were not parsed for this comparison.
The target column was read to reconstruct the established stratified split; held-out labels and features were not used for fitting or metric computation.

| Candidate | ROC-AUC, mean ± SD | Average Precision, mean ± SD | Total fit + predict | Peak RSS |
|---|---:|---:|---:|---:|
| Tuned LightGBM, complete features | **0.762720 ± 0.001538** | **0.245235 ± 0.005388** | 50.72 s | 1,313 MiB |
| Initial LightGBM, complete features | 0.760665 ± 0.001728 | 0.242435 ± 0.005484 | 38.93 s | 1,269 MiB |
| Tuned XGBoost, complete features | 0.757857 ± 0.001854 | 0.240012 ± 0.006688 | 21.87 s | 1,428 MiB |
| XGBoost, `CREDIT_ANNUITY_RATIO` only | 0.756807 ± 0.003201 | 0.238098 ± 0.007919 | 18.48 s | 1,384 MiB |

Per-fold ROC-AUC / Average Precision:

| Fold | Tuned LightGBM | Initial LightGBM | Tuned XGBoost | Ratio-only XGBoost |
|---:|---:|---:|---:|---:|
| 1 | 0.763761 / 0.241709 | 0.760233 / 0.238412 | 0.757277 / 0.235318 | 0.756404 / 0.232681 |
| 2 | 0.763446 / 0.251438 | 0.762568 / 0.248681 | 0.759931 / 0.247670 | 0.760191 / 0.247187 |
| 3 | 0.760953 / 0.242560 | 0.759195 / 0.240211 | 0.756361 / 0.237049 | 0.753826 / 0.234428 |

Paired differences against tuned LightGBM on each identical fold (candidate minus
tuned LightGBM; ROC-AUC delta / AP delta):

| Candidate | Fold 1 | Fold 2 | Fold 3 |
|---|---:|---:|---:|
| Initial LightGBM | −0.003528 / −0.003297 | −0.000877 / −0.002756 | −0.001758 / −0.002349 |
| Tuned XGBoost | −0.006483 / −0.006390 | −0.003514 / −0.003768 | −0.004591 / −0.005511 |
| Ratio-only XGBoost | −0.007356 / −0.009028 | −0.003255 / −0.004251 | −0.007127 / −0.008132 |

Tuned LightGBM led both mean metrics and scored higher on both metrics in every
fold. This is consistent evidence across these three folds, not a broad statistical
guarantee. The single-ratio XGBoost result is lower in CV than its earlier fixed
validation result, which shows why that single split was not enough to select it.

## Feature and inference compatibility checks

Synthetic checks covered zero and missing denominators, positive and negative
infinity inputs, the employment sentinel, all-missing EXT_SOURCE values, stable
feature order, and repeat-transform equality. Both feature transformers are
scikit-learn estimators with explicit `fit`/`transform`; clipping bounds are learned
on the training fold only. `CreditAnnuityRatio` preserves the original columns and
appends one derived feature.

For both the tuned LightGBM pipeline and the ratio-only XGBoost pipeline, the
first-fold fitted bundle was serialized to a temporary directory and loaded in a
fresh Python process. The existing FastAPI `/predict` route returned HTTP 200 with
the unchanged `default_probability`, `predicted_default`, and `threshold` keys for
the synthetic example request. Temporary candidate bundles were removed after the
smoke check.


## Production integration status

Tuned LightGBM is now the default API model and is included in Docker with the
stable `credit_features` transformer. XGBoost remains available through
`HOME_CREDIT_MODEL=xgboost`. Historical experiment results are validation/CV
selection evidence, not independent LightGBM test evaluation.

## Metric provenance review

The historical Phase 2 training-only XGBoost validation reference is ROC-AUC
0.755089 and AP 0.243390 (six decimals recorded). Its frozen production threshold
is 0.161378. These are not validation measurements of the final artifact refitted
on train plus validation. That artifact's historical test ROC-AUC/AP are
0.7607282727652057 / 0.247628209742408; no test evaluation was rerun.

The current notebook displays validation ROC-AUC/AP 0.7554 / 0.2446 in the
comparison table and 0.755400 / 0.244557 in the sensitivity table. Its diagnostic
maximum-F1 threshold is 0.151809 (six decimals printed). More precise notebook
values were not saved. The earlier no-engineering control in
`feature_ablation.json` records 0.7553995021235814 / 0.24455713089777548,
which agrees at the notebook's displayed precision; these full-precision control
values must not be attributed as exact saved notebook measurements.

Inspection of baseline commit af6fce6 and current sources finds the same 180
trees, depth 4, learning rate 0.06, min_child_weight 10, row/column sampling 0.8,
reg_lambda 5, max_bin 128, hist method, two threads, seed 42 and binary logistic
objective. Both use 120 original inputs (104 numeric, 16 categorical), median
imputation with indicators and scaling, plus most-frequent categorical imputation
and unknown-tolerant one-hot encoding fitted on training only. Two stratified
seed-42 splits reserve 40%, then divide it equally; the index-based experimental
split construction preserves the same ordering and stratification. Engineered
feature and tuned-XGBoost candidates are different models, not baseline replicas.

The XGBoost artifact and Phase 4B records report scikit-learn 1.9.1 and XGBoost
3.2.0, matching the current environment (NumPy 2.4.6, pandas 3.1.1). The original
validation run has no complete environment/data fingerprint, validation score
vector or committed notebook outputs. Therefore its actual kernel execution
cannot be reconstructed from the unchanged source alone. Stale state, unrecorded
execution differences, environment/data drift and nondeterminism are possible,
but none is demonstrated. The confirmed translation errors prevent execution;
they do not explain the older numeric reference.

Both documented threshold procedures maximize F1 on validation PR thresholds.
The notebook rounds the displayed thresholds to six decimals; rounding cannot
explain the approximately 0.009569 difference. Different fitted scores can change
the optimum, but the missing historical score vector prevents a conclusive
reconstruction. The discrepancy is documented, not causally resolved. Historical
metrics, approved artifacts and thresholds are preserved. No additional fitting
was performed for this review. Future experimental runs should record complete
versions, data/split fingerprints and full-precision aggregate metrics.
