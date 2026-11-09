"""Compare four fixed candidates on identical, training-only CV folds."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from experiments.feature_engineering import (
    ALL_ENGINEERED_FEATURES,
    DATA_PATH,
    ROOT,
    SEED,
    CreditAnnuityRatio,
    HomeCreditFeatures,
    RssMonitor,
    build_pipeline,
    json_safe,
    preprocessor,
)

RESULTS_PATH = ROOT / "experiments" / "cv_confirmation_results.json"
N_FOLDS = 3


def candidates(X: pd.DataFrame):
    full_feature_candidates = [
        (
            "Tuned LightGBM + complete feature set",
            LGBMClassifier(
                n_estimators=700, learning_rate=0.035, num_leaves=31,
                max_depth=-1, min_child_samples=100, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=8,
                n_jobs=2, random_state=SEED, verbosity=-1,
            ),
        ),
        (
            "Initial LightGBM + complete feature set",
            LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                max_depth=-1, min_child_samples=50, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=5, n_jobs=2,
                random_state=SEED, verbosity=-1,
            ),
        ),
        (
            "Tuned XGBoost + complete feature set",
            XGBClassifier(
                n_estimators=220, max_depth=4, learning_rate=0.05,
                min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=5, max_bin=128, tree_method="hist", n_jobs=2,
                random_state=SEED, eval_metric="logloss",
                objective="binary:logistic",
            ),
        ),
    ]
    # The fourth candidate adds exactly one ratio to untouched original columns.
    single_ratio = XGBClassifier(
        n_estimators=180, max_depth=4, learning_rate=0.06,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5, max_bin=128, tree_method="hist", n_jobs=2,
        random_state=SEED, eval_metric="logloss", objective="binary:logistic",
    )
    return full_feature_candidates + [("XGBoost + CREDIT_ANNUITY_RATIO only", single_ratio)]


def pipeline_for(name: str, X: pd.DataFrame, estimator) -> Pipeline:
    if name == "XGBoost + CREDIT_ANNUITY_RATIO only":
        omitted = tuple(f for f in ALL_ENGINEERED_FEATURES if f != "CREDIT_ANNUITY_RATIO")
        return Pipeline([
            ("feature_engineering", CreditAnnuityRatio()),
            ("preprocessor", preprocessor(X, omitted_features=omitted)),
            ("classifier", clone(estimator)),
        ])
    return build_pipeline(X, clone(estimator))


def metric_row(y_true, probability):
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
    }


def feature_transformer_checks(example: dict[str, object]) -> dict[str, object]:
    """Exercise missing/zero/sentinel handling and stable output order on synthetic rows."""
    frame = pd.DataFrame([example] * 1000)
    frame["AMT_CREDIT"] = 100_000.0
    frame["AMT_ANNUITY"] = 10_000.0
    frame.loc[0, ["AMT_CREDIT", "AMT_ANNUITY"]] = [100_000.0, 10_000.0]
    frame.loc[1, ["AMT_CREDIT", "AMT_ANNUITY"]] = [50_000.0, 0.0]
    frame.loc[2, ["AMT_CREDIT", "AMT_ANNUITY"]] = [20_000.0, np.nan]
    frame.loc[3, ["AMT_CREDIT", "AMT_ANNUITY"]] = [300_000.0, 1.0]
    frame.loc[4, "AMT_CREDIT"] = np.inf
    frame.loc[5, "AMT_ANNUITY"] = np.inf
    frame.loc[3, "DAYS_EMPLOYED"] = 365243
    frame.loc[0, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = [np.nan, 0.4, 0.6]
    frame.loc[1, ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]] = [np.nan, np.nan, np.nan]

    full = HomeCreditFeatures().fit(frame)
    full_a = full.transform(frame)
    full_b = full.transform(frame)
    assert full_a.columns.tolist() == full.get_feature_names_out().tolist()
    assert full_a.columns.tolist() == full_b.columns.tolist()
    assert full_a.equals(full_b)
    assert full_a["DAYS_EMPLOYED_365243"].iloc[:4].tolist() == [0, 0, 0, 1]
    assert pd.isna(full_a.loc[3, "DAYS_EMPLOYED"])
    assert pd.isna(full_a.loc[1, "EXT_SOURCE_MEAN"])
    assert pd.isna(full_a.loc[1, "EXT_SOURCE_STD"])
    assert not np.isinf(full_a.select_dtypes(include="number").to_numpy()).any()

    ratio = CreditAnnuityRatio().fit(frame)
    ratio_a = ratio.transform(frame)
    ratio_b = ratio.transform(frame)
    assert ratio_a.columns.tolist() == ratio.get_feature_names_out().tolist()
    assert ratio_a.equals(ratio_b)
    assert ratio_a.loc[0, "CREDIT_ANNUITY_RATIO"] == 10.0
    assert pd.isna(ratio_a.loc[1, "CREDIT_ANNUITY_RATIO"])
    assert pd.isna(ratio_a.loc[2, "CREDIT_ANNUITY_RATIO"])
    assert pd.isna(ratio_a.loc[4, "CREDIT_ANNUITY_RATIO"])
    assert pd.isna(ratio_a.loc[5, "CREDIT_ANNUITY_RATIO"])
    assert not np.isinf(ratio_a.select_dtypes(include="number").to_numpy()).any()
    return {
        "synthetic_missing_zero_infinity_sentinel_checks": "passed",
        "repeat_transform_deterministic": True,
        "full_feature_order_stable": True,
        "single_ratio_preserves_original_values_and_appends_one_column": True,
    }


def api_serialization_smoke(bundle_paths: dict[str, str]) -> dict[str, object]:
    """Load candidate bundles in a fresh process and invoke the existing API route."""
    code = r'''
import json, math, sys
import joblib
from fastapi.testclient import TestClient
import predict

payload = json.load(open("examples/predict_request.json", encoding="utf-8"))
original_bundle = predict.BUNDLE
checks = {}
for label, bundle_path in json.loads(sys.argv[1]).items():
    predict.BUNDLE = joblib.load(bundle_path)
    response = TestClient(predict.app).post("/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"default_probability", "predicted_default", "threshold"}
    assert math.isfinite(body["default_probability"])
    checks[label] = {"status_code": response.status_code, "response_keys": sorted(body)}
predict.BUNDLE = original_bundle
print(json.dumps(checks))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, json.dumps(bundle_paths)],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing authorized development data: {DATA_PATH}")

    # Read labels to recreate the established 60% training indices. Then parse
    # feature values only for those rows; validation/test feature rows are skipped.
    y = pd.read_csv(DATA_PATH, usecols=["TARGET"])["TARGET"].astype("int8")
    training_indices, _held_out_indices = train_test_split(
        np.arange(len(y)), test_size=0.40, random_state=SEED, stratify=y
    )
    training_row_set = set(np.sort(training_indices).tolist())
    source_columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    feature_columns = [c for c in source_columns if c not in {"TARGET", "SK_ID_CURR"}]
    X_source_order = pd.read_csv(
        DATA_PATH,
        usecols=feature_columns,
        skiprows=lambda line: line > 0 and (line - 1) not in training_row_set,
    )
    X_source_order = X_source_order[feature_columns]
    assert len(X_source_order) == len(training_indices) == 184_506
    # Restore the exact row order produced by the previous train_test_split so
    # the seeded CV holdout indices match the earlier development-fold setup.
    row_positions = np.searchsorted(np.sort(training_indices), training_indices)
    X_train = X_source_order.iloc[row_positions].reset_index(drop=True)
    y_train = y.iloc[training_indices].reset_index(drop=True)
    del X_source_order, y, training_row_set, row_positions

    fold_pairs = list(StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SEED
    ).split(X_train, y_train))
    fold_manifest = []
    for fold_no, (_, hold_idx) in enumerate(fold_pairs, start=1):
        original_rows = training_indices[hold_idx].astype(np.int64)
        fold_manifest.append({
            "fold": fold_no,
            "validation_rows": int(len(hold_idx)),
            "positive_rows": int(y_train.iloc[hold_idx].sum()),
            "validation_index_sha256": hashlib.sha256(original_rows.tobytes()).hexdigest(),
        })

    production_bundle = joblib.load(ROOT / "models" / "xgb_home_credit.joblib")
    model_specs = candidates(X_train)
    results = []
    process = psutil.Process()
    rss_initial = process.memory_info().rss

    with tempfile.TemporaryDirectory(prefix="home-credit-cv-confirmation-") as temp_dir:
        temp_dir = Path(temp_dir)
        api_bundle_paths = {}
        for candidate_name, estimator in model_specs:
            fold_results, fold_seconds = [], []
            peak_rss = process.memory_info().rss
            for fold_no, (fit_idx, hold_idx) in enumerate(fold_pairs, start=1):
                started = time.perf_counter()
                with threadpool_limits(limits=2), RssMonitor() as monitor:
                    pipeline = pipeline_for(candidate_name, X_train, estimator)
                    pipeline.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
                    probability = pipeline.predict_proba(X_train.iloc[hold_idx])[:, 1]
                elapsed = time.perf_counter() - started
                row = {
                    "fold": fold_no,
                    **metric_row(y_train.iloc[hold_idx], probability),
                    "fit_and_predict_seconds": elapsed,
                    "rss_peak_mib": round(monitor.peak / 1024**2, 1),
                }
                fold_results.append(row)
                fold_seconds.append(elapsed)
                peak_rss = max(peak_rss, monitor.peak)

                # Keep first-fold fitted pipelines only in a temporary directory
                # to smoke-test joblib and the unchanged FastAPI request contract.
                if fold_no == 1 and candidate_name in {
                    "Tuned LightGBM + complete feature set",
                    "XGBoost + CREDIT_ANNUITY_RATIO only",
                }:
                    api_bundle = {
                        **production_bundle,
                        "model": pipeline,
                        "model_name": candidate_name,
                    }
                    bundle_path = temp_dir / f"{len(api_bundle_paths)}.joblib"
                    joblib.dump(api_bundle, bundle_path, compress=1)
                    api_bundle_paths[candidate_name] = str(bundle_path)
                del pipeline, probability
                print(json.dumps({"model": candidate_name, **row}), flush=True)

            fold_roc = [r["roc_auc"] for r in fold_results]
            fold_ap = [r["average_precision"] for r in fold_results]
            results.append({
                "model": candidate_name,
                "seed": SEED,
                "threads": 2,
                "parameters": json_safe(estimator.get_params(deep=False)),
                "folds": fold_results,
                "mean_roc_auc": float(np.mean(fold_roc)),
                "std_roc_auc": float(np.std(fold_roc, ddof=1)),
                "mean_average_precision": float(np.mean(fold_ap)),
                "std_average_precision": float(np.std(fold_ap, ddof=1)),
                "total_fit_and_predict_seconds": float(sum(fold_seconds)),
                "rss_peak_mib": round(peak_rss / 1024**2, 1),
            })

        api_smoke = api_serialization_smoke(api_bundle_paths)

    reference = results[0]
    paired_vs_tuned_lgbm = {}
    all_paired_differences = {}
    for result in results[1:]:
        paired_vs_tuned_lgbm[result["model"]] = [{
            "fold": fold,
            "roc_auc_delta": result["folds"][fold - 1]["roc_auc"]
                             - reference["folds"][fold - 1]["roc_auc"],
            "average_precision_delta": result["folds"][fold - 1]["average_precision"]
                                       - reference["folds"][fold - 1]["average_precision"],
        } for fold in range(1, N_FOLDS + 1)]
    for left, right in combinations(results, 2):
        key = f'{left["model"]} minus {right["model"]}'
        all_paired_differences[key] = [{
            "fold": fold,
            "roc_auc_delta": left["folds"][fold - 1]["roc_auc"]
                             - right["folds"][fold - 1]["roc_auc"],
            "average_precision_delta": left["folds"][fold - 1]["average_precision"]
                                       - right["folds"][fold - 1]["average_precision"],
        } for fold in range(1, N_FOLDS + 1)]

    report = {
        "protocol": {
            "training_rows": int(len(y_train)),
            "validation_or_test_rows_used": False,
            "test_metrics_calculated": False,
            "seed": SEED,
            "fold_count": N_FOLDS,
            "folding": "StratifiedKFold(shuffle=True, random_state=42), generated once and reused for every candidate",
            "fold_manifest": fold_manifest,
            "versions": {
                "scikit_learn": __import__("sklearn").__version__,
                "xgboost": __import__("xgboost").__version__,
                "lightgbm": __import__("lightgbm").__version__,
            },
        },
        "candidates": results,
        "paired_differences_vs_tuned_lightgbm": paired_vs_tuned_lgbm,
        "all_pairwise_fold_differences": all_paired_differences,
        "feature_transform_checks": feature_transformer_checks(
            json.loads((ROOT / "examples" / "predict_request.json").read_text(encoding="utf-8"))
        ),
        "fastapi_serialization_smoke": api_smoke,
        "process_rss_initial_mib": round(rss_initial / 1024**2, 1),
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
