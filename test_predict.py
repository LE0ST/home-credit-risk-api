"""Small HTTP client example; payload contains features only, never TARGET or ID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--payload", default="examples/predict_request.json", help="JSON payload with the 120 features")
    parser.add_argument("--data", help="Optional: generate a payload from one row in a local CSV")
    args = parser.parse_args()

    if args.data:
        row = pd.read_csv(Path(args.data), nrows=1).iloc[0]
        raw_features = row.drop(labels=["TARGET", "SK_ID_CURR"]).to_dict()
        features = {
            key: (None if pd.isna(value) else (value.item() if hasattr(value, "item") else value))
            for key, value in raw_features.items()
        }
    else:
        features = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    if "TARGET" in features or "SK_ID_CURR" in features:
        raise ValueError("The prediction example must not include TARGET or SK_ID_CURR.")
    features = json.loads(json.dumps(features, allow_nan=False))
    response = requests.post(f"{args.url.rstrip('/')}/predict", json=features, timeout=60)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
