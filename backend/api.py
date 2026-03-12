"""
FastAPI backend for Stock Price Prediction React frontend.
Exposes the existing data_handler.py and model.py logic via REST API.

Run with:
    uvicorn backend.api:app --reload --port 8000
(run from the project root: "Stock price prediction" folder)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Render/free instances are CPU-only. Force TensorFlow CPU to avoid CUDA init errors.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_handler import (
    fetch_stock_data, get_stock_info, compute_technical_indicators,
    prepare_multifeature_data, inverse_transform, FEATURE_COLUMNS,
)
from model import (
    build_lstm_model, predict, predict_future, WarmUpScheduler,
)
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ──────────────────────────────────────────────────────────────
app = FastAPI(title="Stock Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _safe(series):
    """Convert a pandas Series to a list, replacing NaN with None."""
    return [None if pd.isna(v) else float(v) for v in series]

def _col(df, col):
    return _safe(df[col]) if col in df.columns else []


# ──────────────────────────────────────────────────────────────
# GET /api/stock/{ticker}
# ──────────────────────────────────────────────────────────────

@app.get("/api/stock/{ticker}")
async def get_stock(ticker: str, period: str = "5y"):
    """Fetch OHLCV data + technical indicators + company info for a ticker."""
    t = ticker.upper()
    try:
        df = fetch_stock_data(t, period)
        df = compute_technical_indicators(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    info = {}
    try:
        info = get_stock_info(t) or {}
    except Exception:
        pass

    chart_df = df.tail(500)

    return {
        "info": info,
        "chart": {
            "dates":      [d.strftime("%Y-%m-%d") for d in chart_df.index],
            "open":       _safe(chart_df["Open"]),
            "high":       _safe(chart_df["High"]),
            "low":        _safe(chart_df["Low"]),
            "close":      _safe(chart_df["Close"]),
            "volume":     [int(v) if not pd.isna(v) else None for v in chart_df["Volume"]],
            "sma20":      _col(chart_df, "SMA_20"),
            "sma50":      _col(chart_df, "SMA_50"),
            "sma200":     _col(chart_df, "SMA_200"),
            "bbUpper":    _col(chart_df, "BB_Upper"),
            "bbLower":    _col(chart_df, "BB_Lower"),
            "rsi":        _col(chart_df, "RSI"),
            "macd":       _col(chart_df, "MACD"),
            "macdSignal": _col(chart_df, "MACD_Signal"),
        },
    }


# ──────────────────────────────────────────────────────────────
# POST /api/train
# ──────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    ticker: str
    period: str = "5y"
    lookback: int = 60
    epochs: int = 100
    forecast_days: int = 30


@app.post("/api/train")
async def train_endpoint(body: TrainRequest):
    """
    Train the BiLSTM model and return predictions + forecast.
    Note: training runs synchronously — the client should use a long timeout.
    """
    ticker = body.ticker.upper()

    # ── Fetch & prepare data ──
    try:
        df = fetch_stock_data(ticker, body.period)
        df = compute_technical_indicators(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data fetch failed: {e}")

    try:
        X_train, y_train, X_test, y_test, scaler, train_size, close_idx, num_features = \
            prepare_multifeature_data(df, lookback=body.lookback)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data preparation failed: {e}")

    # ── Build & train model ──
    model = build_lstm_model(input_shape=(body.lookback, num_features))

    callbacks = [
        WarmUpScheduler(target_lr=0.0005, warmup_epochs=5),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8, min_lr=1e-6, verbose=0),
    ]

    try:
        history = model.fit(
            X_train, y_train,
            epochs=body.epochs,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            shuffle=True,
            verbose=0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    # ── Evaluate on test set ──
    y_pred_scaled = predict(model, X_test)
    y_pred   = inverse_transform(scaler, y_pred_scaled, close_idx, num_features)
    y_actual = inverse_transform(scaler, y_test,        close_idx, num_features)

    rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae  = float(mean_absolute_error(y_actual, y_pred))
    r2   = float(r2_score(y_actual, y_pred))
    mape = float(np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-8))) * 100)

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    clean_df   = df[available_features].dropna()
    test_dates = clean_df.index[train_size:][:len(y_actual)]

    # ── Future forecast ──
    feature_data = df[available_features].dropna().values
    scaled_all   = scaler.transform(feature_data)
    last_seq     = scaled_all[-body.lookback:]

    forecast_result = predict_future(
        model, last_seq, scaler,
        days=body.forecast_days,
        close_idx=close_idx,
        num_features=num_features,
        n_simulations=50,
    )

    last_actual_price = float(df["Close"].iloc[-1])
    future_prices = forecast_result["median"]
    upper_band    = forecast_result["upper"]
    lower_band    = forecast_result["lower"]

    price_gap     = last_actual_price - future_prices[0]
    future_prices = future_prices + price_gap
    upper_band    = upper_band + price_gap
    lower_band    = lower_band + price_gap

    last_date    = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=body.forecast_days)
    recent_actual = df["Close"].tail(60)

    actual_epochs = len(history.history["loss"])

    return {
        "metrics": {
            "rmse": rmse,
            "mae":  mae,
            "r2":   r2,
            "mape": mape,
        },
        "prediction": {
            "actual":    [float(v) for v in y_actual],
            "predicted": [float(v) for v in y_pred],
            "dates":     [d.strftime("%Y-%m-%d") for d in test_dates],
        },
        "training": {
            "loss":     [float(v) for v in history.history["loss"]],
            "val_loss": [float(v) for v in history.history.get("val_loss", [])],
            "epochs":   actual_epochs,
        },
        "forecast": {
            "dates":         [last_date.strftime("%Y-%m-%d")] + [d.strftime("%Y-%m-%d") for d in future_dates],
            "prices":        [last_actual_price] + [float(v) for v in future_prices],
            "upper":         [last_actual_price] + [float(v) for v in upper_band],
            "lower":         [last_actual_price] + [float(v) for v in lower_band],
            "recent_dates":  [d.strftime("%Y-%m-%d") for d in recent_actual.index],
            "recent_prices": [float(v) for v in recent_actual.values],
        },
        "current_price": last_actual_price,
    }


# ──────────────────────────────────────────────────────────────
# Serve React frontend (production build)
# Must be registered AFTER all /api routes
# ──────────────────────────────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")

if os.path.exists(_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_dist, "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        file_path = os.path.join(_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_dist, "index.html"))
