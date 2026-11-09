"""Minimal direct-vs-API prediction consistency check using a train row."""
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from predict import BUNDLE, app, predict_record


def test_direct_and_api_prediction_match():
    df = pd.read_csv("data/raw/application_train.csv")
    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = df["TARGET"]
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    raw_row = X_train.iloc[0].to_dict()
    row = {key: (None if pd.isna(value) else (value.item() if hasattr(value, "item") else value))
           for key, value in raw_row.items()}

    direct = predict_record(row, BUNDLE)
    response = TestClient(app).post("/predict", json=row)
    assert response.status_code == 200
    assert response.json() == direct


def test_prediction_rejects_target_and_identifier():
    row = {column: None for column in BUNDLE["feature_columns"]}
    row["TARGET"] = 0
    response = TestClient(app).post("/predict", json=row)
    assert response.status_code == 422
