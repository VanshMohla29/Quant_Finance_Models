"""
India HMM Regime Detector — FastAPI Service
=============================================
Serves the trained HMM model as a REST API.
The React dashboard (or any client) can POST market features
and receive a real-time regime prediction.

Install:
    pip install fastapi uvicorn hmmlearn scikit-learn numpy pandas

Run:
    uvicorn regime_api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional
import pickle, os, json
import numpy as np
import pandas as pd

app = FastAPI(
    title="India HMM Regime Detector API",
    description="Probabilistic market regime detection for NIFTY 50 (Gaussian HMM)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACT_DIR = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(ARTIFACT_DIR, "hmm_model.pkl")
SCALER_PATH  = os.path.join(ARTIFACT_DIR, "hmm_scaler.pkl")
LABEL_PATH   = os.path.join(ARTIFACT_DIR, "label_map.pkl")
MIX_PATH     = os.path.join(ARTIFACT_DIR, "learned_sector_mix.json")

_model      = None
_scaler     = None
_label_map  = None
_learned_sector_mix = {}

def load_artifacts():
    global _model, _scaler, _label_map, _learned_sector_mix
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            _scaler = pickle.load(f)
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, 'rb') as f:
            _label_map = pickle.load(f)
    if os.path.exists(MIX_PATH):
        with open(MIX_PATH, 'r') as f:
            _learned_sector_mix = json.load(f)

@app.on_event("startup")
def startup():
    load_artifacts()

class MarketFeatures(BaseModel):
    ret_1d:         float = Field(..., description="1-day return")
    vol_20d:        float = Field(..., description="20-day annualized realized volatility")
    price_vs_ma200: float = Field(..., description="Distance from 200-day moving average")
    rsi14:          float = Field(..., description="14-day RSI (0-100)")
    vix:            float = Field(..., description="India VIX level")
    drawdown:       float = Field(..., description="Drawdown from all-time peak (negative)")

REGIME_EXPOSURE = {
    "Bull":     1.0,
    "Bear":     0.0,
    "HighVol":  0.2,
    "Sideways": 0.6,
}

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": _model is not None,
        "n_states": _model.n_components if _model else None,
    }

@app.post("/regime/predict")
def predict_regime(features: MarketFeatures):
    if _model is None or _scaler is None:
        raise HTTPException(503, "HMM model not loaded. Run train pipeline first.")

    raw = np.array([[
        features.ret_1d,
        features.vol_20d,
        features.price_vs_ma200,
        features.rsi14,
        features.vix,
        features.drawdown,
    ]])

    scaled = _scaler.transform(raw)
    state = int(_model.predict(scaled)[0])
    posteriors = _model.predict_proba(scaled)[0]

    s_name = _label_map.get(state, {}).get("name", "Sideways") if _label_map else f"State_{state}"
    color  = _label_map.get(state, {}).get("color", "#3b82f6") if _label_map else "#3b82f6"

    probs = {}
    if _label_map:
        for s_idx, prob in enumerate(posteriors):
            r = _label_map.get(s_idx, {}).get("name", f"State_{s_idx}")
            probs[r] = round(float(prob), 4)

    exp = sum(probs.get(r, 0.0) * REGIME_EXPOSURE.get(r, 0.5) for r in probs)
    return {
        "regime": s_name,
        "color": color,
        "market_exposure": round(exp, 4),
        "posteriors": probs,
        "recommended_sector_mix": _learned_sector_mix.get(s_name, {})
    }

@app.get("/regime/strategy")
def regime_strategy(regime: str):
    if regime not in REGIME_EXPOSURE:
        raise HTTPException(400, f"Unknown regime. Choose from: {list(REGIME_EXPOSURE.keys())}")
    eq = REGIME_EXPOSURE[regime]
    return {
        "regime": regime,
        "equity_exposure": eq,
        "cash_exposure": round(1.0 - eq, 2),
        "sector_mix": _learned_sector_mix.get(regime, {})
    }
