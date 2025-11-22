"""
Simple Flask API to expose state-level predicted demand based on saved pipeline and model.
Endpoints:
 - GET /top_state          -> returns JSON with the state that has highest predicted demand
 - GET /state_predictions  -> returns JSON list of states and their predicted demand (sorted desc). Optional ?limit=N
 - GET /health             -> simple health check

Run:
    pip install -r requirements.txt
    python predict_api.py

Notes: ensure `preprocess.joblib` and `demand_model.pkl` exist in the same folder (created by the notebook).
"""

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import pickle
import os

app = Flask(__name__)

DATA_CSV = 'PSP_Weather_Merged_EDA_Cleaned.csv'
PIPELINE_PATH = 'preprocess.joblib'
MODEL_PATH = 'demand_model.pkl'

# Load artifacts at startup (fail fast with clear error message)
pipeline = None
model = None

try:
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline file not found: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
except Exception as e:
    pipeline = None
    app.logger.warning(f"Pipeline load failed: {e}")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = pickle.load(open(MODEL_PATH, 'rb'))
except Exception as e:
    model = None
    app.logger.warning(f"Model load failed: {e}")


def build_state_rep(df_raw, rep_method='last'):
    """
    Build representative rows per State.
    rep_method: 'last' (default), 'median', or 'mean'
    Returns a DataFrame with one row per State (no target column, Date removed)
    """
    df = df_raw.dropna().reset_index(drop=True)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    if 'Date' in df.columns and rep_method == 'last':
        rep = df.sort_values('Date').groupby('State', as_index=False).last()
    else:
        # median/mean will aggregate numeric columns; categorical columns take first
        if rep_method == 'median':
            num = df.select_dtypes(include=[np.number]).groupby(df['State']).median().reset_index()
        else:
            num = df.select_dtypes(include=[np.number]).groupby(df['State']).mean().reset_index()
        # for non-numeric cols, take first occurrence per State
        others = df.drop(columns=df.select_dtypes(include=[np.number]).columns).groupby('State', as_index=False).first()
        rep = pd.merge(others, num, on='State', how='left')
    # drop target if present
    if 'Max_Demand_Met_MW' in rep.columns:
        rep = rep.drop(columns=['Max_Demand_Met_MW'])
    if 'Date' in rep.columns:
        rep = rep.drop(columns=['Date'])
    return rep


def prepare_for_pipeline(rep_df, pipeline):
    """
    Make rep_df safe for pipeline.transform: ensure expected input columns exist,
    fill missing columns with sensible defaults, and reorder columns if possible.
    """
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


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'pipeline_loaded': pipeline is not None, 'model_loaded': model is not None})


@app.route('/state_predictions')
def state_predictions():
    limit = request.args.get('limit', default=None, type=int)
    rep_method = request.args.get('rep', default='last', type=str)

    if pipeline is None:
        return jsonify({'error': 'preprocess.joblib not found or failed to load'}), 500
    if model is None:
        return jsonify({'error': 'demand_model.pkl not found or failed to load'}), 500

    if not os.path.exists(DATA_CSV):
        return jsonify({'error': f'data CSV not found: {DATA_CSV}'}), 500

    df_raw = pd.read_csv(DATA_CSV)
    rep = build_state_rep(df_raw, rep_method=rep_method)
    if rep.shape[0] == 0:
        return jsonify({'error': 'no states found in data'}), 400

    X_state = prepare_for_pipeline(rep, pipeline)
    try:
        Xp = pipeline.transform(X_state)
        if hasattr(Xp, 'toarray'):
            Xp = Xp.toarray()
    except Exception as e:
        return jsonify({'error': f'pipeline transform failed: {e}'}), 500

    try:
        preds = model.predict(Xp)
    except Exception as e:
        return jsonify({'error': f'model predict failed: {e}'}), 500

    rep = rep.copy()
    rep['predicted_demand'] = preds
    rep_sorted = rep.sort_values('predicted_demand', ascending=False).reset_index(drop=True)
    out = rep_sorted[['State', 'predicted_demand']]
    if limit is not None:
        out = out.head(limit)
    result = out.to_dict(orient='records')
    return jsonify({'count': len(result), 'predictions': result})


@app.route('/top_state')
def top_state():
    resp = state_predictions()
    # state_predictions returns a flask response tuple, so extract JSON
    if isinstance(resp, tuple):
        body, status = resp[0].get_json(), resp[1]
        if status != 200:
            return resp
    else:
        body = resp.get_json()
    preds = body.get('predictions', [])
    if not preds:
        return jsonify({'error': 'no predictions available'}), 500
    top = preds[0]
    return jsonify({'state': top['State'], 'predicted_demand': top['predicted_demand']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
