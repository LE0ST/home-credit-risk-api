"""Fit the selected LightGBM model using only the development split."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from credit_features import ALL_ENGINEERED_FEATURES, HomeCreditFeatures

ROOT = Path(__file__).resolve().parent
PARAMS = dict(n_estimators=700, learning_rate=0.035, num_leaves=31, max_depth=-1,
              min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=8, n_jobs=2, random_state=42, verbosity=-1)


def build_pipeline(X):
    numeric = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical = X.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    numeric.extend(ALL_ENGINEERED_FEATURES)
    preprocess = ColumnTransformer([
        ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scaler', StandardScaler())]), numeric),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    return Pipeline([('feature_engineering', HomeCreditFeatures()), ('preprocessor', preprocess),
                     ('classifier', LGBMClassifier(**PARAMS))])


def main():
    df = pd.read_csv(ROOT / 'data/raw/application_train.csv')
    y = df.pop('TARGET').astype('int8')
    X = df.drop(columns=['SK_ID_CURR'])
    train, rest = train_test_split(np.arange(len(y)), test_size=.4, random_state=42, stratify=y)
    val_rel, _ = train_test_split(np.arange(len(rest)), test_size=.5, random_state=42, stratify=y.iloc[rest])
    val = rest[val_rel]
    with threadpool_limits(limits=2):
        model = build_pipeline(X.iloc[train])
        model.fit(X.iloc[train], y.iloc[train])
        probabilities = model.predict_proba(X.iloc[val])[:, 1]
        precision, recall, thresholds = precision_recall_curve(y.iloc[val], probabilities)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
        best = int(np.argmax(f1))
        threshold = float(thresholds[best])
        metrics = {'threshold': threshold, 'precision': float(precision[best]), 'recall': float(recall[best]),
                   'f1': float(f1[best]), 'selection': 'maximum F1 on fixed validation split',
                   'validation_rows': len(val), 'test_used': False}
        del model, probabilities
        dev = np.concatenate([train, val])
        model = build_pipeline(X.iloc[dev])
        started = time.perf_counter()
        model.fit(X.iloc[dev], y.iloc[dev])
        metrics['training_seconds'] = time.perf_counter() - started
        metrics['training_rows'] = len(dev)
        metrics['transformed_feature_count'] = len(model.named_steps['preprocessor'].get_feature_names_out())
    numeric = set(X.select_dtypes(include=['number', 'bool']).columns)
    bundle = {'model': model, 'threshold': threshold, 'feature_columns': list(X.columns),
              'feature_types': {c: 'numeric' if c in numeric else 'categorical' for c in X.columns},
              'model_name': 'LightGBM tuned', 'model_params': PARAMS}
    path = ROOT / 'models/lightgbm_home_credit.joblib'
    joblib.dump(bundle, path, compress=3)
    metrics['artifact_bytes'] = path.stat().st_size
    (ROOT / 'reports/lightgbm_validation.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
