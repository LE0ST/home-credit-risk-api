"""Measure CREDIT_ANNUITY_RATIO as the only engineered input feature."""
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

    omitted = tuple(feature for feature in ALL_ENGINEERED_FEATURES
                    if feature != "CREDIT_ANNUITY_RATIO")
    started = time.perf_counter()
    with threadpool_limits(limits=2), RssMonitor() as monitor:
        pipeline = build_pipeline(
            X_train, XGBClassifier(**PARAMS), omitted_features=omitted
        )
        pipeline.fit(X_train, y_train)
        probability = pipeline.predict_proba(X_val)[:, 1]
    result = {
        "model": "XGBoost base + CREDIT_ANNUITY_RATIO only",
        "omitted_features": list(omitted),
        **scores(y_val, probability),
        "seconds": time.perf_counter() - started,
        "rss_peak_mib": round(monitor.peak / 1024**2, 1),
        "seed": SEED,
        "threads": 2,
        "parameters": PARAMS,
        "test_partition_evaluated_or_used_for_selection": False,
    }
    report = {
        **result,
        "baseline_reference": {"roc_auc": 0.755089, "average_precision": 0.243390},
        "full_engineered_xgb": {"roc_auc": 0.7596951962640084,
                                 "average_precision": 0.2478794816759051},
    }
    path = ROOT / "experiments" / "feature_isolation.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
