"""Three-fold stratified CV for the leading XGBoost validation candidate."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from feature_engineering import (
    DATA_PATH,
    RESULTS_PATH,
    SEED,
    RssMonitor,
    build_pipeline,
    scores,
)


def main():
    df = pd.read_csv(DATA_PATH)
    y = df.pop("TARGET").astype("int8")
    X = df.drop(columns=["SK_ID_CURR"])
    train_idx, _remaining_idx = train_test_split(
        np.arange(len(y)), test_size=0.40, random_state=SEED, stratify=y
    )
    X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    del X, y, train_idx, _remaining_idx, df

    estimator = XGBClassifier(
        n_estimators=220, max_depth=4, learning_rate=0.05,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=5, max_bin=128, tree_method="hist", n_jobs=2,
        random_state=SEED, eval_metric="logloss", objective="binary:logistic",
    )
    fold_results, fold_seconds = [], []
    with RssMonitor() as monitor:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        for fold, (fit_i, hold_i) in enumerate(splitter.split(X_train, y_train), start=1):
            started = time.perf_counter()
            with threadpool_limits(limits=2):
                pipeline = build_pipeline(X_train, clone(estimator))
                pipeline.fit(X_train.iloc[fit_i], y_train.iloc[fit_i])
                probability = pipeline.predict_proba(X_train.iloc[hold_i])[:, 1]
            elapsed = time.perf_counter() - started
            fold_seconds.append(elapsed)
            fold_results.append(scores(y_train.iloc[hold_i], probability))
            print(json.dumps({"cv_model": "XGBoost tuned + features", "fold": fold,
                              **fold_results[-1], "seconds": elapsed}), flush=True)
            del pipeline, probability

    report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    report["cross_validation"].append({
        "model": "XGBoost tuned + features",
        "folds": 3,
        "seed": SEED,
        "mean_roc_auc": float(np.mean([r["roc_auc"] for r in fold_results])),
        "std_roc_auc": float(np.std([r["roc_auc"] for r in fold_results], ddof=1)),
        "mean_average_precision": float(np.mean([r["average_precision"] for r in fold_results])),
        "std_average_precision": float(np.std([r["average_precision"] for r in fold_results], ddof=1)),
        "fold_seconds": fold_seconds,
        "rss_peak_mib": round(monitor.peak / 1024**2, 1),
    })
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
