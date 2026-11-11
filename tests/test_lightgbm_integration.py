import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_features import HomeCreditFeatures
from predict import BUNDLE, app, predict_record

ROOT = Path(__file__).resolve().parents[1]


def example():
    return json.loads((ROOT / 'examples/predict_request.json').read_text())


def test_serialized_model_loads_in_fresh_process():
    code = "import joblib; b=joblib.load('models/lightgbm_home_credit.joblib'); print(b['model_name'])"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert 'LightGBM tuned' in result.stdout


def test_api_matches_loaded_model_and_probability_bounds():
    row = example()
    assert len(row) == 120
    response = TestClient(app).post('/predict', json=row)
    assert response.status_code == 200
    assert response.json() == predict_record(row, BUNDLE)
    assert 0 <= response.json()['default_probability'] <= 1


def test_feature_transform_is_stable_for_zero_missing_and_infinity():
    frame = pd.DataFrame([example(), example()])
    frame.loc[0, 'AMT_INCOME_TOTAL'] = 0
    frame.loc[0, 'AMT_ANNUITY'] = np.inf
    transformer = HomeCreditFeatures().fit(frame)
    a = transformer.transform(frame)
    b = transformer.transform(frame)
    pd.testing.assert_frame_equal(a, b)
    assert pd.isna(a.loc[0, 'CREDIT_INCOME_RATIO'])
    assert pd.isna(a.loc[0, 'CREDIT_ANNUITY_RATIO'])
    assert list(a.columns) == list(transformer.get_feature_names_out())


@pytest.mark.parametrize('field', ['TARGET', 'SK_ID_CURR'])
def test_forbidden_fields(field):
    row = example(); row[field] = 1
    assert TestClient(app).post('/predict', json=row).status_code == 422


def test_missing_invalid_and_unknown_category():
    client = TestClient(app)
    row = example(); row.pop(next(iter(row)))
    assert client.post('/predict', json=row).status_code == 422
    row = example(); row['AMT_CREDIT'] = 'invalid'
    assert client.post('/predict', json=row).status_code == 422
    row = example(); row['AMT_CREDIT'] = None; row['NAME_CONTRACT_TYPE'] = 'unseen category'
    assert client.post('/predict', json=row).status_code == 200


def test_published_schema_matches_both_models():
    import joblib
    schema = json.loads((ROOT / 'feature_schema.json').read_text())['features']
    for name in ['lightgbm', 'xgb']:
        bundle = joblib.load(ROOT / f'models/{name}_home_credit.joblib')
        assert list(schema) == bundle['feature_columns']
        assert schema == bundle['feature_types']


@pytest.mark.parametrize('selection,script', [('lightgbm','train_lightgbm.py'), ('xgboost','train.py')])
def test_missing_artifact_diagnostic(selection, script):
    import os
    code = "from unittest.mock import patch; import pathlib; p=patch.object(pathlib.Path,'is_file',return_value=False); p.start(); import predict"
    result = subprocess.run([sys.executable,'-c',code],cwd=ROOT,env={**os.environ,'HOME_CREDIT_MODEL':selection},capture_output=True,text=True)
    assert result.returncode != 0
    assert f'python {script}' in result.stderr


@pytest.mark.parametrize('selection', ['lightgbm','xgboost'])
def test_synthetic_example_for_each_model(selection):
    import os
    code = "import json; from fastapi.testclient import TestClient; from predict import app,predict_record; p=json.load(open('examples/predict_request_synthetic.json')); r=TestClient(app).post('/predict',json=p); assert r.status_code==200; assert r.json()==predict_record(p)"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,env={**os.environ,'HOME_CREDIT_MODEL':selection},check=True,capture_output=True)
