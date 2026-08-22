"""
India HMM Regime Detector — FastAPI Service
=============================================
Serves the trained HMM model as a REST API.
The React dashboard (or any client) can POST market features
and receive a real-time regime prediction.

Install:
    pip install fastapi uvicorn hmmlearn scikit-learn numpy pandas

Run:
    uvicorn regime_api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health           — liveness check
    GET  /regime/current   — latest regime from last prediction
    POST /regime/predict   — predict regime from feature dict
    GET  /regime/history   — last N regime transitions
    GET  /regime/strategy  — current regime strategy + sector weights
"""

import os, pickle, json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(
    title="India HMM Regime API",
    description="4-state Gaussian HMM market regime detector for NIFTY 50",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + scaler at startup
MODEL_PATH  = os.environ.get("HMM_MODEL_PATH",  "output/hmm_model.pkl")
SCALER_PATH = os.environ.get("HMM_SCALER_PATH", "output/hmm_scaler.pkl")

_model  = None
_scaler = None
_history = []


@app.on_event("startup")
def load_model():
    global _model, _scaler
    try:
        with open(MODEL_PATH,  "rb") as f: _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
        print("✓ HMM model and scaler loaded")
    except FileNotFoundError:
        print("⚠ Model not found — run train.py first to generate output/hmm_model.pkl")


class MarketFeatures(BaseModel):
    """
    16 features expected by the HMM (same order as select_features_for_hmm).
    Provide daily values for all fields.
    """
    ret_1d:          float   # daily return (e.g. 0.0042)
    vol_20d:         float   # 20-day annualised realised vol
    vol_ratio:       float   # vol_10d / vol_60d
    price_vs_ma200:  float   # (price - 200dMA) / 200dMA
    ma50_vs_ma200:   float   # (50dMA - 200dMA) / 200dMA
    rsi14:           float   # 14-day RSI (0-100)
    vix:             float   # India VIX level
    vix_ratio:       float   # VIX / 20d mean VIX
    rate_spread:     float   # 10Y G-Sec - Repo Rate
    fx_trend:        float   # 20d % change in USD/INR
    fii_ma20:        float   # 20d average FII flow (₹ crore / 1000)
    drawdown:        float   # (price - 252d peak) / 252d peak
    yield_curve:     float   # 10Y G-Sec - 2Y G-Sec (NEW)
    real_rate:       float   # Repo Rate - CPI (NEW)
    dxy_chg:         float   # 20d % change in DXY proxy (NEW)
    iip_level:       float   # IIP YoY % (NEW)

    class Config:
        schema_extra = {
            "example": {
                "ret_1d": 0.003, "vol_20d": 0.14, "vol_ratio": 0.9,
                "price_vs_ma200": 0.05, "ma50_vs_ma200": 0.02, "rsi14": 58,
                "vix": 14.5, "vix_ratio": 0.95, "rate_spread": 2.3,
                "fx_trend": 0.01, "fii_ma20": 1.2, "drawdown": -0.03,
                "yield_curve": 1.8, "real_rate": 0.5, "dxy_chg": 0.01,
                "iip_level": 5.2,
            }
        }


REGIME_STRATEGIES_API = {
    "Bull":     "Long NIFTY 50 Fut 40% | Bank ETF 20% | Infra 15% | PUT hedge 5% | Cash 20%",
    "Bear":     "Short NIFTY 25% | Gold ETF 25% | Pharma/FMCG 15% | S-T Debt 25% | Long USD 10%",
    "HighVol":  "ATM Straddle 20% | Gold+Silver 25% | Liquid Fund 40% | Bank PUTs 10% | GSec 5%",
    "Sideways": "Short Strangle 20% | IT ETF 20% | Div Yield 20% | M-D Debt 25% | NIFTY SIP 15%",
}

LABEL_MAP_DEMO = {0: "Bull", 1: "Bear", 2: "HighVol", 3: "Sideways"}


def _predict_raw(features: MarketFeatures):
    if _model is None or _scaler is None:
        raise HTTPException(503, "Model not loaded. Run training pipeline first.")
    x = np.array([[
        features.ret_1d, features.vol_20d, features.vol_ratio,
        features.price_vs_ma200, features.ma50_vs_ma200, features.rsi14,
        features.vix, features.vix_ratio, features.rate_spread,
        features.fx_trend, features.fii_ma20, features.drawdown,
        features.yield_curve, features.real_rate, features.dxy_chg,
        features.iip_level,
    ]])
    x_s = _scaler.transform(x)
    state = int(_model.predict(x_s)[0])
    proba = _model.predict_proba(x_s)[0].tolist()
    regime = LABEL_MAP_DEMO.get(state, "Sideways")
    return regime, state, proba


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/regime/predict")
def predict_regime(features: MarketFeatures):
    """Predict market regime from current feature values."""
    regime, state, proba = _predict_raw(features)
    result = {
        "regime":      regime,
        "hmm_state":   state,
        "probabilities": {
            "Bull":     round(proba[0], 4) if len(proba) > 0 else 0,
            "Bear":     round(proba[1], 4) if len(proba) > 1 else 0,
            "HighVol":  round(proba[2], 4) if len(proba) > 2 else 0,
            "Sideways": round(proba[3], 4) if len(proba) > 3 else 0,
        },
        "strategy":    REGIME_STRATEGIES_API[regime],
        "confidence":  round(max(proba), 4),
    }
    _history.append(result)
    return result


@app.get("/regime/current")
def current_regime():
    """Return the most recent prediction."""
    if not _history:
        return {"message": "No predictions yet. POST to /regime/predict first."}
    return _history[-1]


@app.get("/regime/history")
def regime_history(n: int = 20):
    """Return last N regime predictions."""
    return {"history": _history[-n:], "total": len(_history)}


@app.get("/regime/strategy")
def regime_strategy(regime: str = "Bull"):
    """Return strategy and sector weights for a given regime."""
    if regime not in REGIME_STRATEGIES_API:
        raise HTTPException(400, f"Unknown regime. Choose from: {list(REGIME_STRATEGIES_API)}")
    return {
        "regime":   regime,
        "strategy": REGIME_STRATEGIES_API[regime],
        "sector_weights": {
            "Bull":     {"NIFTY Bank":0.22,"NIFTY IT":0.18,"NIFTY Auto":0.15,"NIFTY Metal":0.12,"NIFTY Realty":0.10,"NIFTY Infra":0.10,"NIFTY Energy":0.08,"NIFTY FMCG":0.03,"NIFTY Pharma":0.02},
            "Bear":     {"NIFTY Pharma":0.30,"NIFTY FMCG":0.25,"NIFTY IT":0.15,"NIFTY Energy":0.15,"NIFTY Infra":0.05,"NIFTY Bank":0.05,"NIFTY Auto":0.05},
            "HighVol":  {"NIFTY Pharma":0.35,"NIFTY FMCG":0.30,"NIFTY Energy":0.20,"NIFTY IT":0.10,"NIFTY Bank":0.05},
            "Sideways": {"NIFTY IT":0.25,"NIFTY FMCG":0.20,"NIFTY Pharma":0.15,"NIFTY Bank":0.15,"NIFTY Auto":0.10,"NIFTY Energy":0.08,"NIFTY Infra":0.07},
        }.get(regime, {}),
    }
