# Stock Market Price Prediction

An AI-powered stock price prediction app using **LSTM deep learning** with an interactive **Streamlit** web dashboard.

## Features

- **Live Data** — Fetches real-time stock data from Yahoo Finance
- **LSTM Neural Network** — Stacked LSTM with Dropout for time-series forecasting
- **Interactive Charts** — Candlestick, Moving Averages, Bollinger Bands, RSI, MACD
- **Future Forecasting** — Predict stock prices up to 90 days ahead with confidence bands
- **Performance Metrics** — RMSE, MAE, MAPE, R² Score
- **Premium Dark UI** — Modern glassmorphism design with gradient accents
- **Global Stocks** — Works with US (`AAPL`, `TSLA`) and Indian (`RELIANCE.NS`, `TCS.NS`) stocks

## Quick Start

### 1. Install Dependencies

```bash
cd "Stock price prediction"
pip install -r requirements.txt
```

### 2. Run the App

```bash
python -m streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### 3. Use It

1. Enter a stock ticker (e.g., `AAPL`) in the sidebar
2. Adjust parameters — lookback window, epochs, forecast days
3. Click **"Train Model & Predict"**
4. View the forecast chart and performance metrics

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit + Plotly |
| AI Model | LSTM (TensorFlow/Keras) |
| Data | Yahoo Finance (yfinance) |
| Preprocessing | scikit-learn (MinMaxScaler) |

## Project Structure

```
├── app.py              # Streamlit web app (entry point)
├── model.py            # LSTM model architecture & training
├── data_handler.py     # Data fetching & preprocessing
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Disclaimer

This project is for **educational purposes only**. Stock markets are inherently unpredictable. Do not use these predictions for actual trading or investment decisions.
