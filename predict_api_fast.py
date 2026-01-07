#connect it with_pranjals
"""
FastAPI app to serve state-level predicted demand using saved artifacts from the notebook.
Endpoints:
 - GET /health
 - GET /state_predictions?limit=10&rep=last
 - GET /top_state

Run:
    pip install -r requirements.txt
    uvicorn predict_api_fast:app --reload --host 0.0.0.0 --port 8000

Ensure `preprocess.joblib` and `demand_model.pkl` are in the same directory.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import logging
import json

app = FastAPI(title="Demand Prediction API (FastAPI)")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_CSV = 'PSP_Weather_Merged_EDA_Cleaned.csv'
PIPELINE_PATH = 'preprocess.joblib'
MODEL_PATH = 'demand_model.pkl'

pipeline = None
model = None


@app.on_event("startup")
def load_artifacts():
    global pipeline, model
    try:
        if os.path.exists(PIPELINE_PATH):
            pipeline = joblib.load(PIPELINE_PATH)
        else:
            logger.warning(f"Pipeline not found: {PIPELINE_PATH}")
            pipeline = None
    except Exception as e:
        pipeline = None
        logger.warning(f"Failed loading pipeline: {e}")

    try:
        if os.path.exists(MODEL_PATH):
            model = pickle.load(open(MODEL_PATH, 'rb'))
        else:
            logger.warning(f"Model not found: {MODEL_PATH}")
            model = None
    except Exception as e:
        model = None
        logger.warning(f"Failed loading model: {e}")


def build_state_rep(df_raw: pd.DataFrame, rep_method: str = 'last') -> pd.DataFrame:
    df = df_raw.dropna().reset_index(drop=True)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    if 'Date' in df.columns and rep_method == 'last':
        rep = df.sort_values('Date').groupby('State', as_index=False).last()
    else:
        if rep_method == 'median':
            num = df.select_dtypes(include=[np.number]).groupby(df['State']).median().reset_index()
        else:
            num = df.select_dtypes(include=[np.number]).groupby(df['State']).mean().reset_index()
        others = df.drop(columns=df.select_dtypes(include=[np.number]).columns).groupby('State', as_index=False).first()
        rep = pd.merge(others, num, on='State', how='left')
    if 'Max_Demand_Met_MW' in rep.columns:
        rep = rep.drop(columns=['Max_Demand_Met_MW'])
    if 'Date' in rep.columns:
        rep = rep.drop(columns=['Date'])
    return rep


def prepare_for_pipeline(rep_df: pd.DataFrame, pipeline) -> pd.DataFrame:
    X_state = rep_df.copy()
    cat_cols = []
    num_cols = []
    try:
        pre = pipeline.named_steps.get('preprocess', pipeline)
        for tname, trans, cols in getattr(pre, 'transformers_', []):
            if isinstance(cols, (list, tuple)):
                if tname == 'cat':
                    cat_cols = list(cols)
                elif tname == 'num':
                    num_cols = list(cols)
    except Exception:
        cat_cols = [c for c in X_state.columns if X_state[c].dtype == object]
        num_cols = [c for c in X_state.columns if X_state[c].dtype.kind in 'biufc']

    for c in cat_cols:
        if c not in X_state.columns:
            X_state[c] = 'Unknown'
    for c in num_cols:
        if c not in X_state.columns:
            X_state[c] = 0.0

    input_cols = None
    try:
        input_cols = list(pipeline.feature_names_in_)
    except Exception:
        if cat_cols or num_cols:
            input_cols = cat_cols + num_cols
    if input_cols is not None:
        missing = [c for c in input_cols if c not in X_state.columns]
        for c in missing:
            X_state[c] = 0 if (c not in cat_cols) else 'Unknown'
        X_state = X_state[input_cols]
    return X_state


def compute_state_predictions(limit: Optional[int], rep: str) -> List[Dict]:
    """Return a list of prediction dicts [{'State':..., 'predicted_demand':...}, ...]"""
    if pipeline is None:
        raise HTTPException(status_code=500, detail='preprocess.joblib not found or failed to load')
    if model is None:
        raise HTTPException(status_code=500, detail='demand_model.pkl not found or failed to load')
    if not os.path.exists(DATA_CSV):
        raise HTTPException(status_code=500, detail=f'data CSV not found: {DATA_CSV}')

    df_raw = pd.read_csv(DATA_CSV)
    rep_df = build_state_rep(df_raw, rep_method=rep)
    if rep_df.shape[0] == 0:
        raise HTTPException(status_code=400, detail='no states found in data')

    X_state = prepare_for_pipeline(rep_df, pipeline)
    try:
        Xp = pipeline.transform(X_state)
        if hasattr(Xp, 'toarray'):
            Xp = Xp.toarray()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'pipeline transform failed: {e}')

    try:
        preds = model.predict(Xp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'model predict failed: {e}')

    rep_df = rep_df.copy()
    rep_df['predicted_demand'] = preds
    rep_sorted = rep_df.sort_values('predicted_demand', ascending=False).reset_index(drop=True)
    out = rep_sorted[['State', 'predicted_demand']]
    if limit is not None:
        out = out.head(limit)
    return out.to_dict(orient='records')


@app.get('/health')
def health():
    return {'status': 'ok', 'pipeline_loaded': pipeline is not None, 'model_loaded': model is not None}


@app.get('/state_predictions')
def state_predictions(limit: Optional[int] = Query(None, ge=1), rep: str = 'last'):
    preds = compute_state_predictions(limit, rep)
    return JSONResponse({'count': len(preds), 'predictions': preds})


@app.get('/top_state')
def top_state(rep: str = 'last'):
    preds = compute_state_predictions(1, rep)
    if not preds:
        raise HTTPException(status_code=500, detail='no predictions available')
    top = preds[0]
    return {'state': top['State'], 'predicted_demand': top['predicted_demand']}


if __name__ == '__main__':
    try:
        import uvicorn  # type: ignore
    except Exception:
        logger.error('uvicorn is not installed. Install with `pip install uvicorn[standard]`')
        raise
    uvicorn.run('predict_api_fast:app', host='0.0.0.0', port=8000, reload=True)


