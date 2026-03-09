"""
Stock Market Price Prediction AI
=================================
A Streamlit-powered web app for predicting stock prices using LSTM deep learning.
Fetches live data from Yahoo Finance, trains an LSTM neural network, and
visualizes future price forecasts with interactive Plotly charts.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_handler import (fetch_stock_data, get_stock_info, compute_technical_indicators,
                         prepare_multifeature_data, inverse_transform, FEATURE_COLUMNS)
from model import build_lstm_model, train_model, predict, predict_future


# ──────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Price Prediction ML Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header gradient ── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #b8b5ff;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
        font-weight: 400;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }

    /* ── Spacer between KPI rows and charts ── */
    .section-spacer {
        margin-top: 1.5rem;
    }
    .metric-label {
        color: #8892b0;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-value {
        color: #ccd6f6;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .metric-value.green { color: #64ffda; }
    .metric-value.red   { color: #ff6b6b; }
    .metric-value.blue  { color: #82aaff; }
    .metric-value.gold  { color: #ffd700; }

    /* ── Section headers ── */
    .section-header {
        color: #8892b0;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid rgba(130,170,255,0.3);
        letter-spacing: -0.3px;
    }

    /* ── Info box ── */
    .info-box {
        background: linear-gradient(135deg, #0d1b2a, #1b263b);
        border-left: 4px solid #82aaff;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #b8c5db;
        font-size: 0.9rem;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* Sidebar collapse arrow — color + always visible */
    [data-testid="stSidebar"] svg {
        color: #ccd6f6 !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ccd6f6 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown h5,
    [data-testid="stSidebar"] .stMarkdown h6 {
        color: #8892b0 !important;
    }

    /* ── Sidebar labels visibility fix ── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #e6f1ff !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Sidebar markdown text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #ccd6f6 !important;
    }

    /* Sidebar help text / tooltips */
    [data-testid="stSidebar"] .stTooltipIcon svg {
        fill: #8892b0 !important;
    }
    [data-testid="stSidebar"] small {
        color: #8892b0 !important;
    }

    /* Sidebar slider value text */
    [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #ccd6f6 !important;
    }

    /* Sidebar selectbox / dropdown text */
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
        color: #b8c5db !important;
    }

    /* Sidebar text input — black text on white background */
    [data-testid="stSidebar"] input[type="text"],
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stTextInput div[data-baseweb="input"] input {
        color: #000000 !important;
    }

    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: rgba(130,170,255,0.2) !important;
    }

    /* Sidebar button */
    [data-testid="stSidebar"] .stButton > button {
        color: #ffffff !important;
    }

    /* Sidebar footer text */
    .sidebar-footer {
        text-align: center;
        color: #8892b0;
        font-size: 0.75rem;
    }

    /* ── Training timer ── */
    .timer-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(130,170,255,0.2);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        text-align: center;
        color: #82aaff;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }

    /* ── Training progress ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #82aaff, #c792ea) !important;
    }

    /* ── Keep header mounted so the collapsed-sidebar control stays usable ── */
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="collapsedControl"] {
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="collapsedControl"] button,
    [data-testid="collapsedControl"] svg {
        color: #ccd6f6 !important;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>Stock Price Prediction ML Model</h1>
    <p>BiLSTM · Attention Mechanism · Layer Normalization · Dropout · Adam Optimizer · MSE Loss · EarlyStopping · LR Scheduling · MinMaxScaler </p>
    <p>SMA · EMA · RSI · MACD · Bollinger Bands · Monte Carlo Simulation · Real-time Yahoo Finance Data</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    # Popular stock recommendations — global, sorted by priority
    st.markdown("##### Popular Stocks")
    stock_options = [
        # Priority-ordered mix of global stocks by market cap / popularity
        "AAPL",           # 🇺🇸 Apple
        "MSFT",           # 🇺🇸 Microsoft
        "NVDA",           # 🇺🇸 NVIDIA
        "GOOGL",          # 🇺🇸 Alphabet
        "AMZN",           # 🇺🇸 Amazon
        "2222.SR",        # �� Saudi Aramco
        "RELIANCE.NS",    # �� Reliance Industries
        "META",           # 🇺🇸 Meta Platforms
        "TSM",            # 🇹🇼 TSMC (US-listed)
        "BRK-B",          # 🇺🇸 Berkshire Hathaway
        "TSLA",           # 🇺🇸 Tesla
        "NOVO-B.CO",      # �� Novo Nordisk
        "TCS.NS",         # �� Tata Consultancy
        "7203.T",         # 🇯🇵 Toyota Motor
        "ASML",           # 🇳🇱 ASML Holding
        "HDFCBANK.NS",    # 🇮🇳 HDFC Bank
        "SHEL.L",         # 🇬🇧 Shell
        "005930.KS",      # 🇰🇷 Samsung Electronics
        "LLY",            # 🇺🇸 Eli Lilly
        "JPM",            # 🇺🇸 JPMorgan Chase
        "INFY.NS",        # 🇮🇳 Infosys
        "V",              # 🇺🇸 Visa
        "UNH",            # 🇺🇸 UnitedHealth
        "NESN.SW",        # 🇨🇭 Nestlé
        "NVO",            # 🇩🇰 Novo Nordisk (US-listed)
        "ICICIBANK.NS",   # 🇮🇳 ICICI Bank
        "MC.PA",          # 🇫🇷 LVMH
        "WMT",            # 🇺🇸 Walmart
        "XOM",            # 🇺🇸 ExxonMobil
        "BHARTIARTL.NS",  # 🇮🇳 Bharti Airtel
        "MA",             # 🇺🇸 Mastercard
        "JNJ",            # 🇺🇸 Johnson & Johnson
        "0700.HK",        # 🇭🇰 Tencent
        "SBIN.NS",        # 🇮🇳 State Bank of India
        "AZN.L",          # 🇬🇧 AstraZeneca
        "NFLX",           # 🇺🇸 Netflix
        "SAP",            # 🇩🇪 SAP
        "RY.TO",          # 🇨🇦 Royal Bank of Canada
        "VALE",           # 🇧🇷 Vale (US-listed)
        "ITC.NS",         # 🇮🇳 ITC
        "AMD",            # 🇺🇸 AMD
        "9984.T",         # 🇯🇵 SoftBank Group
        "HINDUNILVR.NS",  # 🇮🇳 Hindustan Unilever
        "DIS",            # 🇺🇸 Disney
        "LT.NS",          # 🇮🇳 Larsen & Toubro
        "CBA.AX",         # 🇦🇺 Commonwealth Bank
        "OR.PA",          # 🇫🇷 L'Oréal
        "KO",             # 🇺🇸 Coca-Cola
        "RACE.MI",        # 🇮🇹 Ferrari
        "D05.SI",         # 🇸🇬 DBS Group (Singapore)
        "KOTAKBANK.NS",   # 🇮🇳 Kotak Mahindra Bank
        "INTC",           # 🇺🇸 Intel
        "BABA",           # 🇨🇳 Alibaba (US-listed)
        "ADANIENT.NS",    # 🇮🇳 Adani Enterprises
        "HSBA.L",         # 🇬🇧 HSBC
        "PEP",            # 🇺🇸 PepsiCo
        "CVX",            # 🇺🇸 Chevron
        "NPN.JO",         # 🇿🇦 Naspers (South Africa)
        "BAJFINANCE.NS",  # 🇮🇳 Bajaj Finance
        "SIE.DE",         # 🇩🇪 Siemens
        "6758.T",         # 🇯🇵 Sony Group
        "TATAMOTORS.NS",  # 🇮🇳 Tata Motors
        "CRM",            # 🇺🇸 Salesforce
        "GS",             # 🇺🇸 Goldman Sachs
        "EQNR",           # 🇳🇴 Equinor (Norway)
        "AXISBANK.NS",    # 🇮🇳 Axis Bank
        "BHP.AX",         # 🇦🇺 BHP Group
        "ORCL",           # 🇺🇸 Oracle
        "ITUB",           # 🇧🇷 Itaú Unibanco (US-listed)
        "SUNPHARMA.NS",   # 🇮🇳 Sun Pharma
        "SHOP.TO",        # 🇨🇦 Shopify
        "BA",             # 🇺🇸 Boeing
        "AMX",            # 🇲🇽 América Móvil (US-listed)
        "WIPRO.NS",       # 🇮🇳 Wipro
        "9988.HK",        # 🇭🇰 Alibaba (HK)
        "MARUTI.NS",      # 🇮🇳 Maruti Suzuki
        "ADBE",           # 🇺🇸 Adobe
        "ITX.MC",         # 🇪🇸 Inditex (Zara)
        "TITAN.NS",       # 🇮🇳 Titan Company
        "UBER",           # 🇺🇸 Uber
        "HCLTECH.NS",     # 🇮🇳 HCL Technologies
        "BP.L",           # 🇬🇧 BP
        "ABNB",           # 🇺🇸 Airbnb
        "MCD",            # 🇺🇸 McDonald's
        "DRREDDY.NS",     # 🇮🇳 Dr. Reddy's
        "MS",             # 🇺🇸 Morgan Stanley
        "6861.T",         # 🇯🇵 Keyence
        "PBR",            # 🇧🇷 Petrobras (US-listed)
        "TATASTEEL.NS",   # 🇮🇳 Tata Steel
        "PFE",            # 🇺🇸 Pfizer
        "SBUX",           # 🇺🇸 Starbucks
        "NTPC.NS",        # 🇮🇳 NTPC
        "VOLV-B.ST",      # 🇸🇪 Volvo (Sweden)
        "ENR.DE",         # 🇩🇪 Siemens Energy
        "CIPLA.NS",       # 🇮🇳 Cipla
        "ABI.BR",         # 🇧🇪 AB InBev (Belgium)
        "PYPL",           # 🇺🇸 PayPal
        "POWERGRID.NS",   # 🇮🇳 Power Grid
        "SAN.MC",         # 🇪🇸 Banco Santander (Spain)
        "TECHM.NS",       # 🇮🇳 Tech Mahindra
        "SQ",             # 🇺🇸 Block
        "NICE",           # 🇮🇱 NICE Ltd (Israel)
        "ASIANPAINT.NS",  # 🇮🇳 Asian Paints
        "NESTLEIND.NS",   # 🇮🇳 Nestlé India
        "MRNA",           # 🇺🇸 Moderna
        "NU",             # 🇧🇷 Nu Holdings (US-listed)
        "NOKIA",          # 🇫🇮 Nokia (Finland)
        "CYBR",           # 🇮🇱 CyberArk (Israel)
        "ONGC.NS",        # 🇮🇳 ONGC
        "COALINDIA.NS",   # 🇮🇳 Coal India
        "TD.TO",          # 🇨🇦 Toronto-Dominion Bank
        "HINDALCO.NS",    # 🇮🇳 Hindalco
        "JSWSTEEL.NS",    # 🇮🇳 JSW Steel
        "SPOT",           # 🇸🇪 Spotify (Sweden, US-listed)
        "EICHERMOT.NS",   # 🇮🇳 Eicher Motors
        "BRITANNIA.NS",   # 🇮🇳 Britannia
        "M&M.NS",         # 🇮🇳 Mahindra & Mahindra
        "COST",           # 🇺🇸 Costco
        "HEROMOTOCO.NS",  # 🇮🇳 Hero MotoCorp
        "TATAPOWER.NS",   # 🇮🇳 Tata Power
        "ABB",            # 🇨🇭 ABB (Switzerland, US-listed)
        "DIVISLAB.NS",    # 🇮🇳 Divi's Labs
        "BAJAJFINSV.NS",  # 🇮🇳 Bajaj Finserv
        "RIO.L",          # 🇬🇧 Rio Tinto
        "ULTRACEMCO.NS",  # 🇮🇳 UltraTech Cement
        "CSL.AX",         # 🇦🇺 CSL Limited (Australia)
        "MELI",           # 🇦🇷 MercadoLibre (Argentina, US-listed)
        "CRWD",           # 🇺🇸 CrowdStrike
        "SNOW",           # 🇺🇸 Snowflake
        "SE",             # 🇸🇬 Sea Limited (Singapore, US-listed)
        "GRAB",           # 🇸🇬 Grab Holdings (Singapore)
        "PLTR",           # 🇺🇸 Palantir
        "COIN",           # 🇺🇸 Coinbase
        "RIVN",           # 🇺🇸 Rivian
        "LCID",           # 🇺🇸 Lucid Motors
        "NIO",            # 🇨🇳 NIO (US-listed)
        "LI",             # 🇨🇳 Li Auto (US-listed)
        "XPEV",           # 🇨🇳 XPeng (US-listed)
        "JD",             # 🇨🇳 JD.com (US-listed)
        "PDD",            # 🇨🇳 PDD Holdings (US-listed)
        "BIDU",           # 🇨🇳 Baidu (US-listed)
    ]

    selected_stock = st.selectbox(
        "",
        options=["— Type your own below —"] + stock_options,
        index=0,
        help="Select from popular global stocks, or type your own ticker below"
    )

    # Ticker input
    default_ticker = selected_stock if selected_stock != "— Type your own below —" else "AAPL"
    ticker = st.text_input(
        "Stock Ticker",
        value=default_ticker,
        help="e.g. AAPL, TSLA, GOOGL, RELIANCE.NS, TCS.NS"
    ).upper().strip()

    # Data period
    period = st.selectbox(
        "Historical Data Period",
        options=["1y", "2y", "5y", "10y", "max"],
        index=2,
        help="How far back to fetch data"
    )

    st.markdown("---")
    st.markdown("### Model Parameters")

    # Lookback window
    lookback = st.slider(
        "Lookback Window (days)",
        min_value=20, max_value=120, value=60, step=10,
        help="Number of past days the model uses to predict the next day"
    )

    # Epochs
    epochs = st.slider(
        "Training Epochs",
        min_value=10, max_value=500, value=100, step=10,
        help="More epochs = better training (but slower). Recommended: 100-200"
    )

    # Forecast days
    forecast_days = st.slider(
        "Forecast Days",
        min_value=5, max_value=90, value=30, step=5,
        help="How many future days to predict"
    )

    st.markdown("---")

    # Train button
    train_btn = st.button("Train Model & Predict", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer">'
        'Built with ❤️ using Streamlit + TensorFlow<br>'
        '⚠️ Not financial advice'
        '</div>',
        unsafe_allow_html=True
    )


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def render_metric(label, value, css_class=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """


def format_large_number(n):
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"${n/1e12:.2f}T"
    if n >= 1e9:
        return f"${n/1e9:.2f}B"
    if n >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


# ──────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────

if not ticker:
    st.warning("Please enter a stock ticker symbol in the sidebar.")
    st.stop()

# ── Cached data fetching (prevents Yahoo Finance rate-limiting) ──
@st.cache_data(ttl=1800, show_spinner=False)  # Cache for 30 minutes
def _cached_fetch_data(ticker, period):
    """Cached wrapper — fetches OHLCV data (the important part)."""
    df = fetch_stock_data(ticker, period)
    df = compute_technical_indicators(df)
    return df


def _get_info_safe(ticker):
    """Best-effort stock info — never blocks the app."""
    try:
        return get_stock_info(ticker)
    except Exception:
        return {}


# ── Fetch data ──
try:
    with st.spinner(f"Fetching data for **{ticker}**..."):
        df = _cached_fetch_data(ticker, period)
        info = _get_info_safe(ticker)
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# ── Stock overview ──
st.markdown('<div class="section-header">Stock Overview</div>', unsafe_allow_html=True)

company_name = info.get("longName", info.get("shortName", ticker))
current_price = info.get("currentPrice", info.get("regularMarketPrice", df["Close"].iloc[-1]))
prev_close = info.get("previousClose", df["Close"].iloc[-2] if len(df) > 1 else current_price)
change = current_price - prev_close if current_price and prev_close else 0
change_pct = (change / prev_close * 100) if prev_close else 0
change_color = "green" if change >= 0 else "red"
change_sign = "+" if change >= 0 else ""

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(render_metric("Company", company_name), unsafe_allow_html=True)
with col2:
    price_str = f"${current_price:,.2f}" if current_price else "N/A"
    st.markdown(render_metric("Current Price", price_str, "blue"), unsafe_allow_html=True)
with col3:
    st.markdown(render_metric("Daily Change", f"{change_sign}{change_pct:.2f}%", change_color), unsafe_allow_html=True)
with col4:
    high52 = info.get("fiftyTwoWeekHigh", df["High"].max())
    st.markdown(render_metric("52W High", f"${high52:,.2f}" if high52 else "N/A", "green"), unsafe_allow_html=True)
with col5:
    mcap = info.get("marketCap")
    st.markdown(render_metric("Market Cap", format_large_number(mcap), "gold"), unsafe_allow_html=True)

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PRICE CHART (Candlestick + Volume + Moving Averages)
# ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Price Chart & Technical Indicators</div>', unsafe_allow_html=True)

chart_df = df.tail(min(500, len(df)))  # Show recent data for readability

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.75, 0.25],
    subplot_titles=("", "Volume")
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=chart_df.index,
        open=chart_df["Open"],
        high=chart_df["High"],
        low=chart_df["Low"],
        close=chart_df["Close"],
        name="OHLC",
        increasing_line_color="#64ffda",
        decreasing_line_color="#ff6b6b"
    ),
    row=1, col=1
)

# Moving Averages
for ma, color, dash in [
    ("SMA_20", "#82aaff", None),
    ("SMA_50", "#c792ea", None),
    ("SMA_200", "#ffcb6b", "dash")
]:
    if ma in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index, y=chart_df[ma],
                name=ma, line=dict(color=color, width=1.5, dash=dash),
                opacity=0.8
            ),
            row=1, col=1
        )

# Bollinger Bands
if "BB_Upper" in chart_df.columns:
    fig.add_trace(
        go.Scatter(x=chart_df.index, y=chart_df["BB_Upper"], name="BB Upper",
                   line=dict(color="rgba(255,215,0,0.3)", width=1, dash="dot")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=chart_df.index, y=chart_df["BB_Lower"], name="BB Lower",
                   line=dict(color="rgba(255,215,0,0.3)", width=1, dash="dot"),
                   fill="tonexty", fillcolor="rgba(255,215,0,0.04)"),
        row=1, col=1
    )

# Volume
colors = ["#64ffda" if c >= o else "#ff6b6b"
          for c, o in zip(chart_df["Close"], chart_df["Open"])]
fig.add_trace(
    go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume",
           marker_color=colors, opacity=0.6),
    row=2, col=1
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0a1a",
    plot_bgcolor="#0a0a1a",
    font=dict(family="Inter", color="#8892b0"),
    height=600,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0, font=dict(size=11, color="#ccd6f6")
    ),
    xaxis_rangeslider_visible=False,
    xaxis2=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(136,146,176,0.1)"),
    yaxis2=dict(showgrid=False),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": "hover"})


# ── RSI & MACD sub-charts ──
col_rsi, col_macd = st.columns(2)

with col_rsi:
    if "RSI" in chart_df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["RSI"], name="RSI",
            line=dict(color="#82aaff", width=2)
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", opacity=0.5,
                          annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#64ffda", opacity=0.5,
                          annotation_text="Oversold")
        fig_rsi.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            title=dict(text="RSI (14)", font=dict(size=14, color="#ccd6f6")),
            height=250, margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor="rgba(136,146,176,0.1)"),
            font=dict(family="Inter", color="#8892b0"),
            showlegend=False
        )
        st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar": "hover"})

with col_macd:
    if "MACD" in chart_df.columns:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["MACD"], name="MACD",
            line=dict(color="#82aaff", width=2)
        ))
        fig_macd.add_trace(go.Scatter(
            x=chart_df.index, y=chart_df["MACD_Signal"], name="Signal",
            line=dict(color="#c792ea", width=2)
        ))
        macd_hist = chart_df["MACD"] - chart_df["MACD_Signal"]
        hist_colors = ["#64ffda" if v >= 0 else "#ff6b6b" for v in macd_hist]
        fig_macd.add_trace(go.Bar(
            x=chart_df.index, y=macd_hist, name="Histogram",
            marker_color=hist_colors, opacity=0.4
        ))
        fig_macd.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            title=dict(text="MACD", font=dict(size=14, color="#ccd6f6")),
            height=250, margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(showgrid=True, gridcolor="rgba(136,146,176,0.1)"),
            font=dict(family="Inter", color="#8892b0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10, color="#ccd6f6"))
        )
        st.plotly_chart(fig_macd, use_container_width=True, config={"displayModeBar": "hover"})


# ──────────────────────────────────────────────────────────────
# LSTM TRAINING & PREDICTION
# ──────────────────────────────────────────────────────────────

if train_btn:
    st.markdown('<div class="section-header">LSTM Model Training & Forecast</div>', unsafe_allow_html=True)

    # Prepare multi-feature data
    with st.spinner("Preparing multi-feature data..."):
        try:
            X_train, y_train, X_test, y_test, scaler, train_size, close_idx, num_features = prepare_multifeature_data(
                df, lookback=lookback
            )
        except Exception as e:
            st.error(f"Data preparation failed: {e}")
            st.stop()

    # Show which features are being used
    used_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    features_str = " · ".join(used_features)
    st.markdown(f"""
    <div class="info-box">
        <strong>Dataset:</strong> {len(df)} total days &nbsp;|&nbsp;
        <strong>Training:</strong> {len(X_train)} samples &nbsp;|&nbsp;
        <strong>Testing:</strong> {len(X_test)} samples &nbsp;|&nbsp;
        <strong>Lookback:</strong> {lookback} days<br>
        <strong>Features ({num_features}):</strong> {features_str}
    </div>
    """, unsafe_allow_html=True)

    # Build model
    with st.spinner("Building BiLSTM + Attention model..."):
        model = build_lstm_model(input_shape=(lookback, num_features))

    # Train with live timer
    progress_bar = st.progress(0, text="Training LSTM model...")
    timer_placeholder = st.empty()
    status_text = st.empty()

    import threading

    train_result = {}

    def _run_training():
        try:
            train_result["history"] = train_model(
                model, X_train, y_train, epochs=epochs, batch_size=32
            )
        except Exception as e:
            train_result["error"] = e

    train_thread = threading.Thread(target=_run_training, daemon=True)
    train_start = time.time()
    train_thread.start()

    # Live timer — update every second while training runs
    while train_thread.is_alive():
        elapsed = time.time() - train_start
        mins, secs = divmod(int(elapsed), 60)
        timer_placeholder.markdown(
            f'<div class="timer-box">Training in progress... '
            f'<strong>{mins:02d}:{secs:02d}</strong></div>',
            unsafe_allow_html=True,
        )
        time.sleep(1)

    train_thread.join()
    train_elapsed = time.time() - train_start

    # Check for training errors
    if "error" in train_result:
        st.error(f"❌ Training failed: {train_result['error']}")
        st.stop()

    history = train_result["history"]

    actual_epochs = len(history.history["loss"])
    progress_bar.progress(100, text=f"Training complete — {actual_epochs} epochs")

    # Final timer — show total time + stats
    mins, secs = divmod(int(train_elapsed), 60)
    timer_placeholder.markdown(
        f'<div class="timer-box">Training complete in '
        f'<strong>{mins:02d}:{secs:02d}</strong> &nbsp;|&nbsp; '
        f'{actual_epochs} epochs &nbsp;|&nbsp; '
        f'{train_elapsed/actual_epochs:.1f}s per epoch</div>',
        unsafe_allow_html=True,
    )

    # Display training loss chart
    col_loss, col_empty = st.columns([1, 1])
    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history.history["loss"], name="Training Loss",
            line=dict(color="#82aaff", width=2)
        ))
        if "val_loss" in history.history:
            fig_loss.add_trace(go.Scatter(
                y=history.history["val_loss"], name="Validation Loss",
                line=dict(color="#c792ea", width=2)
            ))
        fig_loss.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
            title=dict(text="Training Loss", font=dict(size=14, color="#ccd6f6")),
            xaxis_title="Epoch", yaxis_title="MSE Loss",
            height=300, margin=dict(l=20, r=20, t=40, b=40),
            font=dict(family="Inter", color="#8892b0"),
            yaxis=dict(showgrid=True, gridcolor="rgba(136,146,176,0.1)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#ccd6f6"))
        )
        st.plotly_chart(fig_loss, use_container_width=True, config={"displayModeBar": "hover"})

    # ── Predictions on test set ──
    with st.spinner("Running predictions on test set..."):
        y_pred_scaled = predict(model, X_test)
        y_pred = inverse_transform(scaler, y_pred_scaled, close_idx, num_features)
        y_actual = inverse_transform(scaler, y_test, close_idx, num_features)

    # ── Metrics ──
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(render_metric("RMSE", f"${rmse:.2f}", "red"), unsafe_allow_html=True)
    with mc2:
        st.markdown(render_metric("MAE", f"${mae:.2f}", "blue"), unsafe_allow_html=True)
    with mc3:
        st.markdown(render_metric("MAPE", f"{mape:.2f}%", "gold"), unsafe_allow_html=True)
    with mc4:
        r2_color = "green" if r2 > 0.8 else ("gold" if r2 > 0.5 else "red")
        st.markdown(render_metric("R² Score", f"{r2:.4f}", r2_color), unsafe_allow_html=True)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # ── Actual vs Predicted chart ──
    # Use the NaN-dropped data index (since multi-feature prep drops warmup rows)
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    clean_df = df[available_features].dropna()
    test_dates = clean_df.index[train_size:][:len(y_actual)]

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=test_dates, y=y_actual, name="Actual Price",
        line=dict(color="#64ffda", width=2)
    ))
    fig_pred.add_trace(go.Scatter(
        x=test_dates, y=y_pred, name="Predicted Price",
        line=dict(color="#ff6b6b", width=2, dash="dot")
    ))
    fig_pred.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
        title=dict(text="Actual vs Predicted (Test Set)", font=dict(size=16, color="#ccd6f6")),
        xaxis_title="Date", yaxis_title="Price ($)",
        height=400, margin=dict(l=20, r=20, t=50, b=40),
        font=dict(family="Inter", color="#8892b0"),
        yaxis=dict(showgrid=True, gridcolor="rgba(136,146,176,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#ccd6f6"))
    )
    st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": "hover"})

    # ── Future Forecast ──
    st.markdown('<div class="section-header">Future Price Forecast</div>', unsafe_allow_html=True)

    with st.spinner(f"Forecasting next {forecast_days} days (Monte Carlo simulation)..."):
        # Get the last 'lookback' multi-feature scaled values
        available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
        feature_data = df[available_features].dropna().values
        scaled_all = scaler.transform(feature_data)
        last_seq = scaled_all[-lookback:]

        forecast_result = predict_future(
            model, last_seq, scaler, days=forecast_days,
            close_idx=close_idx, num_features=num_features,
            n_simulations=50
        )

    future_prices = forecast_result["median"]
    upper_band = forecast_result["upper"]
    lower_band = forecast_result["lower"]
    sim_paths = forecast_result["paths"]

    # ── Anchor forecast to last actual price ──
    last_actual_price = float(df["Close"].iloc[-1])
    price_gap = last_actual_price - future_prices[0]
    future_prices = future_prices + price_gap
    upper_band = upper_band + price_gap
    lower_band = lower_band + price_gap
    sim_paths = sim_paths + price_gap

    # Create future dates (skip weekends)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

    # Prepend last actual point so forecast line connects seamlessly
    connect_date = last_date
    forecast_dates_connected = [connect_date] + list(future_dates)
    forecast_prices_connected = np.concatenate([[last_actual_price], future_prices])
    upper_connected = np.concatenate([[last_actual_price], upper_band])
    lower_connected = np.concatenate([[last_actual_price], lower_band])

    # Recent actual + forecast
    recent_actual = df["Close"].tail(60)

    fig_future = go.Figure()

    # Recent actual prices
    fig_future.add_trace(go.Scatter(
        x=recent_actual.index, y=recent_actual.values, name="Recent Actual",
        line=dict(color="#82aaff", width=2)
    ))

    # Show a few sample simulation paths (faded, for visual richness)
    path_colors = ["rgba(255,100,100,0.2)", "rgba(100,255,200,0.2)", "rgba(200,150,255,0.2)"]
    for i in range(min(3, len(sim_paths))):
        path_connected = np.concatenate([[last_actual_price], sim_paths[i]])
        fig_future.add_trace(go.Scatter(
            x=forecast_dates_connected, y=path_connected,
            line=dict(color=path_colors[i % 3], width=1),
            showlegend=False, hoverinfo="skip"
        ))

    # Confidence band (10th–90th percentile)
    fig_future.add_trace(go.Scatter(
        x=forecast_dates_connected, y=upper_connected, name="Upper Bound",
        line=dict(width=0), showlegend=False
    ))
    fig_future.add_trace(go.Scatter(
        x=forecast_dates_connected, y=lower_connected, name="Confidence Band (10-90%)",
        line=dict(width=0), fill="tonexty",
        fillcolor="rgba(255,215,0,0.1)"
    ))

    # Main forecast line (median — with realistic ups and downs)
    fig_future.add_trace(go.Scatter(
        x=forecast_dates_connected, y=forecast_prices_connected,
        name=f"Forecast ({forecast_days}d)",
        line=dict(color="#ffd700", width=2.5),
        mode="lines+markers",
        marker=dict(size=3)
    ))

    # Vertical divider (using add_shape + add_annotation to avoid Plotly _mean bug)
    fig_future.add_shape(
        type="line",
        x0=str(last_date), x1=str(last_date),
        y0=0, y1=1, yref="paper",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
    )
    fig_future.add_annotation(
        x=str(last_date), y=1, yref="paper",
        text="Today", showarrow=False,
        font=dict(color="#8892b0", size=11),
        yshift=10,
    )

    fig_future.update_layout(
        template="plotly_dark", paper_bgcolor="#0a0a1a", plot_bgcolor="#0a0a1a",
        title=dict(text=f"{ticker} — {forecast_days}-Day Price Forecast (Monte Carlo)",
                   font=dict(size=16, color="#ccd6f6")),
        xaxis_title="Date", yaxis_title="Price ($)",
        height=450, margin=dict(l=20, r=20, t=50, b=40),
        font=dict(family="Inter", color="#8892b0"),
        yaxis=dict(showgrid=True, gridcolor="rgba(136,146,176,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#ccd6f6"))
    )
    st.plotly_chart(fig_future, use_container_width=True, config={"displayModeBar": "hover"})

    # Forecast summary
    forecast_change = float(future_prices[-1]) - float(df["Close"].iloc[-1])
    forecast_change_pct = (forecast_change / float(df["Close"].iloc[-1])) * 100
    fc_sign = "+" if forecast_change >= 0 else ""
    fc_color = "green" if forecast_change >= 0 else "red"
    direction = "📈 Bullish" if forecast_change >= 0 else "📉 Bearish"

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.markdown(render_metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}", "blue"), unsafe_allow_html=True)
    with fc2:
        st.markdown(render_metric(f"Predicted ({forecast_days}d)", f"${future_prices[-1]:,.2f}", fc_color), unsafe_allow_html=True)
    with fc3:
        st.markdown(render_metric("Expected Change", f"{fc_sign}{forecast_change_pct:.2f}%", fc_color), unsafe_allow_html=True)
    with fc4:
        st.markdown(render_metric("Signal", direction, fc_color), unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="info-box" style="border-left-color: #ff6b6b; margin-top: 2rem;">
        ⚠️ <strong>Disclaimer:</strong> This prediction is generated by a machine learning model for educational purposes only.
        Stock markets are inherently unpredictable. <strong>Do not use this for actual trading or investment decisions.</strong>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show instructions when not training
    st.markdown("""
    <div class="info-box" style="margin-top: 1rem;">
        <strong>Configure your parameters</strong> in the sidebar, then click
        <strong>"Train Model & Predict"</strong> to start the LSTM prediction engine.<br><br>
        <strong>Tips:</strong><br>
        • Use tickers like <code>AAPL</code>, <code>TSLA</code>, <code>GOOGL</code> for US stocks<br>
        • For Indian stocks, add <code>.NS</code> suffix — e.g. <code>RELIANCE.NS</code>, <code>TCS.NS</code><br>
        • More historical data + higher epochs = better predictions (but slower training)
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# DATA TABLE
# ──────────────────────────────────────────────────────────────

with st.expander("📋 View Raw Data", expanded=False):
    st.dataframe(
        df.tail(100).style.format({
            "Open": "${:.2f}", "High": "${:.2f}", "Low": "${:.2f}",
            "Close": "${:.2f}", "Volume": "{:,.0f}"
        }),
        use_container_width=True,
        height=400
    )
