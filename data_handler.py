"""
data_handler.py — Stock data fetching & preprocessing for LSTM.
Primary: Direct Yahoo Finance v8 API with browser headers (bypasses rate-limiting).
Fallback: yfinance library.
"""

import time
import json
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

# Try importing yfinance, but don't require it
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


# ── Browser-like headers to avoid rate-limiting ──
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://finance.yahoo.com/",
    "Origin": "https://finance.yahoo.com",
}

# Period mapping for Yahoo Finance v8 API
_PERIOD_MAP = {
    "1y": ("1y", "1d"),
    "2y": ("2y", "1d"),
    "5y": ("5y", "1d"),
    "10y": ("10y", "1wk"),
    "max": ("max", "1d"),
}

MAX_RETRIES = 3
BASE_DELAY = 3


def _fetch_via_direct_api(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch stock data directly from Yahoo Finance v8 chart API.
    Uses browser-like headers to avoid rate limiting.
    """
    range_val, interval = _PERIOD_MAP.get(period, ("5y", "1d"))

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={range_val}&interval={interval}&includeAdjustedClose=true"
    )

    session = requests.Session()
    session.headers.update(_HEADERS)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=15)

            if resp.status_code == 429:
                # Rate limited — wait and retry
                wait = BASE_DELAY * (2 ** attempt)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Parse the response
            result = data.get("chart", {}).get("result", [])
            if not result:
                raise ValueError(f"No data found for ticker '{ticker}'. Check if the symbol is valid.")

            chart = result[0]
            timestamps = chart.get("timestamp", [])
            quote = chart.get("indicators", {}).get("quote", [{}])[0]

            if not timestamps:
                raise ValueError(f"No data found for ticker '{ticker}'. Check if the symbol is valid.")

            df = pd.DataFrame({
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
            }, index=pd.to_datetime(timestamps, unit="s"))

            df.index.name = "Date"
            df = df.dropna()

            if df.empty:
                raise ValueError(f"No data found for ticker '{ticker}'. Check if the symbol is valid.")

            return df

        except ValueError:
            raise
        except requests.exceptions.HTTPError as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))

    raise Exception(f"Direct API failed after {MAX_RETRIES} retries: {last_error}")


def _fetch_via_yfinance(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fallback: use yfinance library."""
    if not HAS_YFINANCE:
        raise ImportError("yfinance not available")

    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check if the symbol is valid.")

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[available]
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    df = df.dropna()
    return df


def fetch_stock_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical OHLCV stock data.
    Strategy: Direct API first → yfinance fallback.
    """
    # Try direct Yahoo Finance API first (more reliable, avoids yfinance rate-limiting)
    try:
        return _fetch_via_direct_api(ticker, period)
    except Exception as direct_err:
        pass

    # Fallback to yfinance
    try:
        return _fetch_via_yfinance(ticker, period)
    except Exception as yf_err:
        pass

    raise Exception(
        f"Could not fetch data for '{ticker}'. "
        f"Direct API: {direct_err} | yfinance: {yf_err}"
    )


def get_stock_info(ticker: str) -> dict:
    """
    Fetch metadata for a stock ticker.
    Returns empty dict on any failure — app derives info from historical data.
    """
    info = {}

    # Try direct API for basic info
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1d"
        session = requests.Session()
        session.headers.update(_HEADERS)
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            if meta:
                info = {
                    "shortName": meta.get("shortName", ticker),
                    "longName": meta.get("longName", meta.get("shortName", ticker)),
                    "currentPrice": meta.get("regularMarketPrice"),
                    "regularMarketPrice": meta.get("regularMarketPrice"),
                    "previousClose": meta.get("previousClose") or meta.get("chartPreviousClose"),
                    "fiftyTwoWeekHigh": meta.get("fiftyTwoWeekHigh"),
                    "fiftyTwoWeekLow": meta.get("fiftyTwoWeekLow"),
                    "currency": meta.get("currency", "USD"),
                    "exchangeName": meta.get("exchangeName"),
                }
    except Exception:
        pass

    # Get market cap via Yahoo cookie-crumb authenticated API
    if not info.get("marketCap"):
        try:
            ys = requests.Session()
            ys.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            # Step 1: Get auth cookies from fc.yahoo.com
            ys.get("https://fc.yahoo.com", timeout=5)
            # Step 2: Get crumb
            crumb_resp = ys.get(
                "https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=5
            )
            crumb = crumb_resp.text
            if crumb and not crumb.startswith("{"):
                # Step 3: Use cookies + crumb for quoteSummary
                qs_resp = ys.get(
                    f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
                    f"?modules=price&crumb={crumb}",
                    timeout=8,
                )
                if qs_resp.status_code == 200:
                    qs_data = qs_resp.json()
                    price_info = (
                        qs_data.get("quoteSummary", {})
                        .get("result", [{}])[0]
                        .get("price", {})
                    )
                    mcap_raw = price_info.get("marketCap", {})
                    mcap_val = mcap_raw.get("raw") if isinstance(mcap_raw, dict) else mcap_raw
                    if mcap_val:
                        info["marketCap"] = mcap_val
                    if not info.get("longName"):
                        info["longName"] = price_info.get("longName", ticker)
                        info["shortName"] = price_info.get("shortName", ticker)
        except Exception:
            pass

    return info


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to the dataframe.
    """
    df = df.copy()

    # Simple Moving Averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # Exponential Moving Average
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Daily Returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Volatility (20-day rolling std of returns)
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()

    # RSI (Relative Strength Index) — 14-day
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std

    return df


# ── Features used for multi-feature LSTM ──
FEATURE_COLUMNS = ["Close", "Volume", "SMA_20", "SMA_50", "RSI", "MACD", "BB_Upper", "BB_Lower"]
TARGET_COLUMN = "Close"


def prepare_multifeature_data(df: pd.DataFrame, lookback: int = 60,
                              train_split: float = 0.8):
    """
    Prepare multi-feature sequences for LSTM training.
    Uses 8 features: Close, Volume, SMA_20, SMA_50, RSI, MACD, BB_Upper, BB_Lower.

    Returns:
        X_train, y_train, X_test, y_test, scaler, train_size, close_idx, num_features
    """
    # Select available features
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if TARGET_COLUMN not in available_features:
        available_features.insert(0, TARGET_COLUMN)

    close_idx = available_features.index(TARGET_COLUMN)
    num_features = len(available_features)

    # Drop rows with NaN (from indicators that need warmup period)
    feature_df = df[available_features].dropna()
    data = feature_df.values

    # Scale ALL features together
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * train_split)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, :])       # All features
        y.append(scaled_data[i, close_idx])             # Predict Close only

    X, y = np.array(X), np.array(y)

    # X shape: (samples, lookback, num_features)
    split_idx = train_size - lookback
    split_idx = max(split_idx, 1)  # Safety

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test, scaler, train_size, close_idx, num_features


def prepare_data(df: pd.DataFrame, feature: str = "Close", lookback: int = 60,
                 train_split: float = 0.8):
    """
    Legacy single-feature data preparation (kept for compatibility).
    """
    data = df[[feature]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * train_split)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split_idx = train_size - lookback
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, y_train, X_test, y_test, scaler, train_size


def inverse_transform(scaler, data: np.ndarray, close_idx: int = 0,
                      num_features: int = 1) -> np.ndarray:
    """
    Convert scaled predictions back to original price scale.
    Works with both single-feature and multi-feature scalers.
    """
    if num_features == 1:
        return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

    # Multi-feature: reconstruct full-width array, inverse-transform, extract Close
    dummy = np.zeros((len(data), num_features))
    dummy[:, close_idx] = data
    return scaler.inverse_transform(dummy)[:, close_idx]

