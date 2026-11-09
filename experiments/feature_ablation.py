"""Single-feature ablation for the engineered XGBoost validation candidate."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from feature_engineering import (
    ALL_ENGINEERED_FEATURES,
    DATA_PATH,
    ROOT,
    RssMonitor,
    SEED,
    build_pipeline,
    scores,
)

RESULTS_PATH = ROOT / "experiments" / "feature_ablation.json"
VALIDATION_PATH = ROOT / "experiments" / "validation_results.json"
PARAMS = {
    "n_estimators": 180, "max_depth": 4, "learning_rate": 0.06,
    "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 5, "max_bin": 128, "tree_method": "hist",
    "n_jobs": 2, "random_state": SEED, "eval_metric": "logloss",
    "objective": "binary:logistic",
}


def main():
    df = pd.read_csv(DATA_PATH)
    y = df.pop("TARGET").astype("int8")
    X = df.drop(columns=["SK_ID_CURR"])
    train_idx, remaining_idx = train_test_split(
        np.arange(len(y)), test_size=0.40, random_state=SEED, stratify=y
    )
    val_relative_idx, _discarded_test_relative_idx = train_test_split(
        np.arange(len(remaining_idx)), test_size=0.50,
        random_state=SEED, stratify=y.iloc[remaining_idx]
    )
    val_idx = remaining_idx[val_relative_idx]
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    del df, X, y, train_idx, remaining_idx, val_relative_idx, val_idx

    cases = [("No engineered features", tuple(ALL_ENGINEERED_FEATURES), False)]
    cases.extend((f"All except {feature}", (feature,), True) for feature in ALL_ENGINEERED_FEATURES)
    rows = []
    process = psutil.Process()
    rss_start = process.memory_info().rss
    for name, omitted, use_feature_engineering in cases:
        started = time.perf_counter()
        with threadpool_limits(limits=2), RssMonitor() as monitor:
            pipeline = build_pipeline(
                X_train, XGBClassifier(**PARAMS), omitted_features=omitted,
                use_feature_engineering=use_feature_engineering,
            )
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_val)[:, 1]
        row = {
            "model": name,
            "omitted_features": list(omitted),
            **scores(y_val, probability),
            "fit_and_validation_predict_seconds": time.perf_counter() - started,
            "rss_peak_mib": round(monitor.peak / 1024**2, 1),
            "rss_delta_peak_mib": round((monitor.peak - rss_start) / 1024**2, 1),
            "threads": 2,
            "seed": SEED,
            "parameters": PARAMS,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        del pipeline, probability

    baseline_ap = next(row["average_precision"] for row in rows if row["model"] == "No engineered features")
    all_row = next(
        row for row in json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))["validation_results"]
        if row["model"] == "XGBoost base + features"
    )
    report = {
        "test_partition_evaluated_or_used_for_selection": False,
        "feature_engineering_scope": ALL_ENGINEERED_FEATURES,
        "clip_quantiles_fit_on_training_partition_only": [0.005, 0.995],
        "reference_validation": {"roc_auc": 0.755089, "average_precision": 0.243390},
        "results": rows,
    }
    report["no_engineered_features_validation"] = next(
        {"roc_auc": row["roc_auc"], "average_precision": row["average_precision"]}
        for row in rows if row["model"] == "No engineered features"
    )
    report["all_engineered_features_validation"] = {
        "roc_auc": all_row["roc_auc"],
        "average_precision": all_row["average_precision"],
    }
    for row in rows:
        row["delta_average_precision_vs_no_engineered_features"] = (
            row["average_precision"] - baseline_ap
        )
        row["delta_average_precision_vs_all_engineered_features"] = (
            row["average_precision"] - all_row["average_precision"]
        )
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
