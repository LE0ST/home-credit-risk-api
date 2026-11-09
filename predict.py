"""FastAPI inference service for the trained Home Credit model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgb_home_credit.joblib"
if not MODEL_PATH.is_file():
    raise RuntimeError("Modelo no entrenado; ejecuta `uv run python train.py` primero.")
BUNDLE = joblib.load(MODEL_PATH)
FEATURE_COLUMNS: list[str] = BUNDLE["feature_columns"]
FEATURE_TYPES: dict[str, str] = BUNDLE["feature_types"]
THRESHOLD: float = BUNDLE["threshold"]

app = FastAPI(title="Home Credit Risk API", version="1.0.0")


class PredictionResponse(BaseModel):
    default_probability: float
    predicted_default: bool
    threshold: float


def validate_features(features: dict[str, Any]) -> dict[str, Any]:
    """Require the model's exact feature schema and reject IDs/target/invalid values."""
    forbidden = {"TARGET", "SK_ID_CURR"}.intersection(features)
    if forbidden:
        raise ValueError(f"No se admiten columnas objetivo o identificadores: {sorted(forbidden)}")
    missing = set(FEATURE_COLUMNS) - set(features)
    extra = set(features) - set(FEATURE_COLUMNS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"faltan columnas: {sorted(missing)}")
        if extra:
            details.append(f"columnas no esperadas: {sorted(extra)}")
        raise ValueError("; ".join(details))

    clean: dict[str, Any] = {}
    for column in FEATURE_COLUMNS:
        value = features[column]
        if value is None:
            clean[column] = np_nan()
        elif FEATURE_TYPES[column] == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{column} debe ser un número o null")
            if not math.isfinite(float(value)):
                raise ValueError(f"{column} debe ser finito")
            clean[column] = value
        elif isinstance(value, str):
            clean[column] = value
        else:
            raise ValueError(f"{column} debe ser texto o null")
    return clean


def np_nan() -> float:
    # DataFrame converts this sentinel into the missing values the pipeline imputes.
    return float("nan")


def predict_record(features: dict[str, Any], bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shared prediction path used by the API and parity test."""
    active_bundle = BUNDLE if bundle is None else bundle
    valid = validate_features(features)
    row = pd.DataFrame([valid], columns=active_bundle["feature_columns"])
    probability = float(active_bundle["model"].predict_proba(row)[0, 1])
    threshold = float(active_bundle["threshold"])
    return {
        "default_probability": probability,
        "predicted_default": probability >= threshold,
        "threshold": threshold,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": BUNDLE["model_name"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_record(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
