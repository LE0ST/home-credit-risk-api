"""Minimal direct-vs-API prediction consistency check using a synthetic request."""
from fastapi.testclient import TestClient

from predict import BUNDLE, app, predict_record


def test_direct_and_api_prediction_match():
    import json
    from pathlib import Path
    row = json.loads((Path(__file__).resolve().parents[1] / "examples/predict_request.json").read_text())

    direct = predict_record(row, BUNDLE)
    response = TestClient(app).post("/predict", json=row)
    assert response.status_code == 200
    assert response.json() == direct


def test_prediction_rejects_target_and_identifier():
    row = {column: None for column in BUNDLE["feature_columns"]}
    row["TARGET"] = 0
    response = TestClient(app).post("/predict", json=row)
    assert response.status_code == 422
