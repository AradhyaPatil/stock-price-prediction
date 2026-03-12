# Stock Market Price Prediction

An AI-powered stock price prediction platform using a **Bidirectional LSTM** deep learning model, a **FastAPI** backend, and a professional **React** trading-desk-style dashboard.

## Features

- **Live Data** — Fetches real-time OHLCV data and company info from Yahoo Finance
- **BiLSTM + Attention** — Bidirectional LSTM with custom attention layer and warmup LR scheduler
- **Technical Indicators** — SMA 20/50/200, Bollinger Bands, RSI, MACD computed server-side
- **Interactive Charts** — Candlestick + volume overlay via Lightweight Charts; RSI/MACD via Recharts
- **Future Forecasting** — Monte Carlo simulation for price forecast with confidence bands (up to 90 days)
- **Performance Metrics** — RMSE, MAE, MAPE, R² Score displayed alongside training loss curves
- **Professional Dark UI** — Market-terminal styled React dashboard with Tailwind CSS
- **Global Stocks** — Works with US (`AAPL`, `TSLA`, `NVDA`) and Indian (`RELIANCE.NS`, `TCS.NS`) stocks

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS |
| Charts | Lightweight Charts (price), Recharts (indicators/training) |
| Backend API | FastAPI + Uvicorn |
| AI Model | BiLSTM + Attention (TensorFlow / Keras) |
| Data | Yahoo Finance (`yfinance` + direct API) |
| Preprocessing | scikit-learn `MinMaxScaler` |
| Deployment | Render (single service — API + static frontend) |

## Project Structure

```
├── backend/
│   └── api.py              # FastAPI REST endpoints + serves React build
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main dashboard shell
│   │   └── components/
│   │       ├── Sidebar.jsx          # Control panel
│   │       ├── StockOverview.jsx    # KPI snapshot cards
│   │       ├── PriceChart.jsx       # Candlestick + overlay chart
│   │       ├── IndicatorCharts.jsx  # RSI / MACD panels
│   │       └── TrainingPanel.jsx    # Training loss, metrics, forecast
│   ├── package.json
│   └── vite.config.js
├── app.py              # Legacy Streamlit app (still functional)
├── model.py            # BiLSTM model architecture & training
├── data_handler.py     # Data fetching & feature engineering
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
└── Procfile            # Process start command
```

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Install backend dependencies

```bash
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
```

### 3. Start both servers

**Backend** (from project root):
```bash
python -m uvicorn backend.api:app --reload --port 8000
```

**Frontend** (in a second terminal):
```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` — the Vite dev server proxies all `/api` calls to the FastAPI backend automatically.

### 4. Use It

1. Enter a stock ticker (e.g., `AAPL`) in the left sidebar
2. Click **Load Data** to fetch historical data and view charts
3. Adjust lookback window, epochs, and forecast days
4. Click **Train Model** to run the BiLSTM pipeline
5. View predictions, training loss, performance metrics, and future forecast

## Production Build (single service)

Build the React app and serve everything from FastAPI:

```bash
cd frontend && npm run build && cd ..
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Deploy to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your repository — Render auto-detects `render.yaml`
4. Click **Deploy**

The build command installs Python deps, builds the React app, and the start command launches a single FastAPI service that serves both the API and the frontend.

> **Note:** Model training with TensorFlow requires ~1–2 GB RAM. Upgrade to Render's **Starter** plan if you hit memory limits on the free tier.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/stock/{ticker}?period=5y` | Fetch OHLCV + indicators + company info |
| `POST` | `/api/train` | Train model and return predictions + forecast |

## Disclaimer

This project is for **educational purposes only**. Stock markets are inherently unpredictable. Do not use these predictions for actual trading or investment decisions.
