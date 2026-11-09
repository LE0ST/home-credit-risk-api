"""Controlled validation-only feature/model experiment; never reads a test split."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "application_train.csv"
RESULTS_PATH = Path(__file__).with_name("validation_results.json")
SEED = 42
RATIO_FEATURES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "EMPLOYED_TO_AGE_RATIO",
    "INCOME_PER_PERSON",
]
ALL_ENGINEERED_FEATURES = [
    "DAYS_EMPLOYED_365243", *RATIO_FEATURES, "EXT_SOURCE_MEAN", "EXT_SOURCE_STD",
]


class HomeCreditFeatures(BaseEstimator, TransformerMixin):
    """Add safe ratios and external-source aggregates with train-fitted clipping."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        generated = self._generate(X)
        self.clip_bounds_ = {}
        for name in RATIO_FEATURES:
            values = generated[name].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                lo, hi = values.quantile([0.005, 0.995]).tolist()
                self.clip_bounds_[name] = (float(lo), float(hi))
            else:
                self.clip_bounds_[name] = (float("nan"), float("nan"))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @staticmethod
    def _divide(
        numerator: pd.Series, denominator: pd.Series, *, positive_denominator: bool = True
    ) -> pd.Series:
        valid_denominator = denominator > 0 if positive_denominator else denominator.ne(0)
        denominator = denominator.where(valid_denominator)
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _generate(cls, X: pd.DataFrame) -> pd.DataFrame:
        # Prevent infinities in raw numeric inputs from reaching SimpleImputer.
        out = X.replace([np.inf, -np.inf], np.nan).copy()
        employed = out["DAYS_EMPLOYED"]
        out["DAYS_EMPLOYED_365243"] = employed.eq(365243).astype("int8")
        # 365243 is Home Credit's sentinel for unavailable employment duration.
        # Other nonnegative employment-day values are also invalid in this schema.
        out["DAYS_EMPLOYED"] = employed.mask(employed >= 0)
        out["CREDIT_INCOME_RATIO"] = cls._divide(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL"])
        out["ANNUITY_INCOME_RATIO"] = cls._divide(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL"])
        out["CREDIT_ANNUITY_RATIO"] = cls._divide(out["AMT_CREDIT"], out["AMT_ANNUITY"])
        out["EMPLOYED_TO_AGE_RATIO"] = cls._divide(
            out["DAYS_EMPLOYED"], out["DAYS_BIRTH"], positive_denominator=False
        )
        out["INCOME_PER_PERSON"] = cls._divide(out["AMT_INCOME_TOTAL"], out["CNT_FAM_MEMBERS"])
        ext = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        out["EXT_SOURCE_MEAN"] = ext.mean(axis=1, skipna=True)
        out["EXT_SOURCE_STD"] = ext.std(axis=1, skipna=True, ddof=0)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._generate(X)
        for name, (lo, hi) in self.clip_bounds_.items():
            out[name] = out[name].clip(lower=lo, upper=hi)
        return out

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        names.extend([
            "DAYS_EMPLOYED_365243", *RATIO_FEATURES,
            "EXT_SOURCE_MEAN", "EXT_SOURCE_STD",
        ])
        return np.asarray(names, dtype=object)


class CreditAnnuityRatio(BaseEstimator, TransformerMixin):
    """Append a train-clipped credit-to-annuity ratio without altering raw inputs."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        ratio = self._calculate(X)
        finite = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite):
            self.clip_bounds_ = tuple(float(v) for v in finite.quantile([0.005, 0.995]))
        else:
            self.clip_bounds_ = (float("nan"), float("nan"))
        return self

    @staticmethod
    def _calculate(X: pd.DataFrame) -> pd.Series:
        numerator = X["AMT_CREDIT"].replace([np.inf, -np.inf], np.nan)
        annuity = X["AMT_ANNUITY"].replace([np.inf, -np.inf], np.nan)
        denominator = annuity.where(annuity > 0)
        return (numerator / denominator).replace([np.inf, -np.inf], np.nan)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        expected = list(self.feature_names_in_)
        if list(X.columns) != expected:
            raise ValueError("Input columns or their order differ from the fitted schema.")
        if "CREDIT_ANNUITY_RATIO" in X.columns:
            raise ValueError("CREDIT_ANNUITY_RATIO is already present in the input.")
        # Keep raw inputs unchanged except for converting invalid infinities to
        # missing values, which the fitted preprocessing pipeline can impute.
        out = X.replace([np.inf, -np.inf], np.nan).copy()
        out["CREDIT_ANNUITY_RATIO"] = self._calculate(X).clip(
            lower=self.clip_bounds_[0], upper=self.clip_bounds_[1]
        )
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([*self.feature_names_in_, "CREDIT_ANNUITY_RATIO"], dtype=object)


class RssMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.running = False
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while self.running:
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.2)

    def __enter__(self):
        self.running = True
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.running = False
        self.thread.join()


def preprocessor(
    X: pd.DataFrame, omitted_features: tuple[str, ...] = ()
) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    # HomeCreditFeatures adds only numeric columns; tell ColumnTransformer their
    # names up front so the engineered values are actually passed to the model.
    numeric.extend([
        name for name in ALL_ENGINEERED_FEATURES
        if name not in numeric and name not in omitted_features
    ])
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def build_pipeline(
    X: pd.DataFrame, estimator, *, omitted_features: tuple[str, ...] = (),
    use_feature_engineering: bool = True,
) -> Pipeline:
    return Pipeline([
        ("feature_engineering", HomeCreditFeatures() if use_feature_engineering else "passthrough"),
        ("preprocessor", preprocessor(X, omitted_features)),
        ("classifier", estimator),
    ])


def model_specs():
    common_xgb = dict(
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5, max_bin=128,
        tree_method="hist", n_jobs=2, random_state=SEED,
        eval_metric="logloss", objective="binary:logistic",
    )
    return [
        ("XGBoost base + features", XGBClassifier(
            **common_xgb, n_estimators=180, max_depth=4,
            learning_rate=0.06, min_child_weight=10)),
        ("XGBoost tuned + features", XGBClassifier(
            **common_xgb, n_estimators=220, max_depth=4,
            learning_rate=0.05, min_child_weight=8)),
        ("LightGBM initial + features", LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=-1, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=5, n_jobs=2,
            random_state=SEED, verbosity=-1)),
        ("LightGBM tuned + features", LGBMClassifier(
            n_estimators=700, learning_rate=0.035, num_leaves=31,
            max_depth=-1, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=8,
            n_jobs=2, random_state=SEED, verbosity=-1)),
    ]


def scores(y, probability):
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
    }


def json_safe(value):
    """Normalize estimator parameters to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing authorized development data: {DATA_PATH}")

    # Reconstruct the established 60/20/20 seed-42 split. Only train and validation
    # rows are materialized below; the test index array is discarded unused.
    df = pd.read_csv(DATA_PATH)
    y = df.pop("TARGET").astype("int8")
    df = df.drop(columns=["SK_ID_CURR"])
    train_idx, remaining_idx = train_test_split(
        np.arange(len(y)), test_size=0.40, random_state=SEED, stratify=y
    )
    val_relative_idx, _discarded_test_relative_idx = train_test_split(
        np.arange(len(remaining_idx)), test_size=0.50,
        random_state=SEED, stratify=y.iloc[remaining_idx]
    )
    val_idx = remaining_idx[val_relative_idx]
    X_train, y_train = df.iloc[train_idx].copy(), y.iloc[train_idx].copy()
    X_val, y_val = df.iloc[val_idx].copy(), y.iloc[val_idx].copy()
    del df, y, train_idx, remaining_idx, val_relative_idx, val_idx

    results = []
    process = psutil.Process()
    rss_start = process.memory_info().rss
    for name, estimator in model_specs():
        started = time.perf_counter()
        with threadpool_limits(limits=2), RssMonitor() as monitor:
            pipe = build_pipeline(X_train, clone(estimator))
            pipe.fit(X_train, y_train)
            probability = pipe.predict_proba(X_val)[:, 1]
        elapsed = time.perf_counter() - started
        row = {
            "model": name,
            **scores(y_val, probability),
            "fit_and_validation_predict_seconds": elapsed,
            "rss_peak_mib": round(monitor.peak / 1024**2, 1),
            "rss_delta_peak_mib": round((monitor.peak - rss_start) / 1024**2, 1),
            "threads": 2,
            "seed": SEED,
            "parameters": json_safe(pipe.named_steps["classifier"].get_params(deep=False)),
        }
        results.append(row)
        print(json.dumps(row, default=str), flush=True)
        del pipe, probability

    # Repeat the base XGBoost comparison without CODE_GENDER.
    no_gender_name = "XGBoost base + features without CODE_GENDER"
    no_gender_estimator = model_specs()[0][1]
    Xtr = X_train.drop(columns=["CODE_GENDER"])
    Xva = X_val.drop(columns=["CODE_GENDER"])
    started = time.perf_counter()
    with threadpool_limits(limits=2), RssMonitor() as monitor:
        no_gender_pipe = build_pipeline(Xtr, clone(no_gender_estimator))
        no_gender_pipe.fit(Xtr, y_train)
        probability = no_gender_pipe.predict_proba(Xva)[:, 1]
    gender_row = {
        "model": no_gender_name,
        **scores(y_val, probability),
        "fit_and_validation_predict_seconds": time.perf_counter() - started,
        "rss_peak_mib": round(monitor.peak / 1024**2, 1),
        "rss_delta_peak_mib": round((monitor.peak - rss_start) / 1024**2, 1),
        "threads": 2,
        "seed": SEED,
        "parameters": json_safe(no_gender_pipe.named_steps["classifier"].get_params(deep=False)),
    }
    results.append(gender_row)
    print(json.dumps(gender_row, default=str), flush=True)
    del no_gender_pipe, probability, Xtr, Xva

    cv_results = []
    if not args.skip_cv and args.cv_folds >= 2:
        # CV is performed only on the training partition. The fixed validation
        # partition stays outside CV and is not used for fitting fold transforms.
        best = sorted(results, key=lambda r: r["average_precision"], reverse=True)[:2]
        import gc
        gc.collect()
        by_name = dict(model_specs())
        by_name[no_gender_name] = no_gender_estimator
        for result in best:
            name = result["model"]
            cv_X = X_train.drop(columns=["CODE_GENDER"]) if "without CODE_GENDER" in name else X_train
            estimator = clone(by_name[name])
            fold_scores, fold_seconds = [], []
            splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)
            for fold, (fit_i, hold_i) in enumerate(splitter.split(cv_X, y_train), start=1):
                started = time.perf_counter()
                with threadpool_limits(limits=2):
                    pipe = build_pipeline(cv_X, clone(estimator))
                    pipe.fit(cv_X.iloc[fit_i], y_train.iloc[fit_i])
                    p = pipe.predict_proba(cv_X.iloc[hold_i])[:, 1]
                fold_seconds.append(time.perf_counter() - started)
                fold_scores.append(scores(y_train.iloc[hold_i], p))
                print(json.dumps({"cv_model": name, "fold": fold, **fold_scores[-1],
                                  "seconds": fold_seconds[-1]}, default=str), flush=True)
                del pipe, p
            cv_results.append({
                "model": name,
                "folds": args.cv_folds,
                "seed": SEED,
                "mean_roc_auc": float(np.mean([s["roc_auc"] for s in fold_scores])),
                "std_roc_auc": float(np.std([s["roc_auc"] for s in fold_scores], ddof=1)),
                "mean_average_precision": float(np.mean([s["average_precision"] for s in fold_scores])),
                "std_average_precision": float(np.std([s["average_precision"] for s in fold_scores], ddof=1)),
                "fold_seconds": fold_seconds,
            })

    report = {
        "data": "data/raw/application_train.csv; existing seed-42 stratified 60/20/20 split",
        "test_partition_evaluated_or_used_for_selection": False,
        "train_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "validation_positive_rate": float(y_val.mean()),
        "reference_validation": {"roc_auc": 0.755089, "average_precision": 0.243390},
        "seed": SEED,
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                      "lightgbm": __import__("lightgbm").__version__},
        "initial_rss_mib": round(rss_start / 1024**2, 1),
        "validation_results": results,
        "cross_validation": cv_results,
    }
    RESULTS_PATH.write_text(
        json.dumps(json_safe(report), indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
