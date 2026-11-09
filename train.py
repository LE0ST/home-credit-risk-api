"""Train the frozen Home Credit XGBoost model and evaluate it once on test."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "application_train.csv"
MODEL_PATH = ROOT / "models" / "xgb_home_credit.joblib"
METRICS_PATH = ROOT / "reports" / "test_metrics.json"
RANDOM_STATE = 42
# Frozen from the Phase 2 validation analysis before touching test.
THRESHOLD = 0.161378
XGB_PARAMS = {
    "n_estimators": 180,
    "max_depth": 4,
    "learning_rate": 0.06,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5,
    "max_bin": 128,
    "tree_method": "hist",
    "n_jobs": 2,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "objective": "binary:logistic",
}


def make_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the same train-fitted imputation/scaling/one-hot pipeline as the notebook."""
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X_train.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
    return preprocessor, numeric, categorical


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Download application_train.csv from the official "
            "Kaggle competition after accepting its rules."
        )

    df = pd.read_csv(DATA_PATH)
    if "TARGET" not in df or "SK_ID_CURR" not in df:
        raise ValueError("The CSV does not match the expected training schema.")
    y = df["TARGET"].astype("int8")
    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    if "TARGET" in X or "SK_ID_CURR" in X:
        raise AssertionError("Target o identificador filtrado en features.")

    # Recreate the fixed 60/20/20 split; test is kept untouched until final evaluation.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp
    )
    del X_tmp, y_tmp
    assert len(X_train) == 184_506 and len(X_val) == 61_502 and len(X_test) == 61_503

    # Validation results fixed the model parameters and threshold in Phase 2.
    # Refit preprocessing and estimator on all development rows (train + validation).
    X_dev = pd.concat([X_train, X_val], axis=0)
    y_dev = pd.concat([y_train, y_val], axis=0)
    preprocessor, numeric_features, categorical_features = make_preprocessor(X_dev)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**XGB_PARAMS)),
        ]
    )

    started = time.perf_counter()
    pipeline.fit(X_dev, y_dev)
    fit_seconds = time.perf_counter() - started

    feature_types = {
        name: ("numeric" if name in numeric_features else "categorical")
        for name in X.columns
    }
    bundle = {
        "model": pipeline,
        "threshold": THRESHOLD,
        "feature_columns": X.columns.tolist(),
        "feature_types": feature_types,
        "model_name": "XGBoost 2",
        "model_params": XGB_PARAMS,
        "random_state": RANDOM_STATE,
        "split": {"train": 0.60, "validation": 0.20, "test": 0.20},
        "preprocessing": "median + missing indicators + StandardScaler for numeric; most_frequent + OneHotEncoder(handle_unknown='ignore') for categorical",
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)

    # The sole test evaluation happens after hyperparameters and threshold are frozen.
    test_probability = pipeline.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= THRESHOLD).astype("int8")
    metrics = {
        "model": "XGBoost 2 refit on train+validation",
        "threshold": THRESHOLD,
        "test_rows": int(len(y_test)),
        "test_positive_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, test_probability)),
        "average_precision": float(average_precision_score(y_test, test_probability)),
        "precision": float(precision_score(y_test, test_prediction, zero_division=0)),
        "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, test_prediction, labels=[0, 1]).tolist(),
        "fit_seconds_train_validation": fit_seconds,
        "xgboost_params": XGB_PARAMS,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
