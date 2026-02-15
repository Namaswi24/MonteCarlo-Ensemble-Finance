import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import yfinance as yf
import pickle
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings('ignore')

MODEL_DIR = "models"
SCALER_DIR = "scalers"

# ==========================================
# BACKEND FUNCTIONS
# ==========================================
def get_data(tickers, start_date=None, end_date=None):
    """Fetch market data from Yahoo Finance."""
    if start_date and end_date:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        raw = yf.download(tickers + ["SPY"], start=start_str, end=end_str, 
                         progress=False, auto_adjust=False)
    else:
        raw = yf.download(tickers + ["SPY"], period="5y", 
                         progress=False, auto_adjust=False)
    
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Adj Close'].dropna()
    else:
        prices = raw[['Adj Close']].dropna()
    
    returns = prices.pct_change().dropna()
    return prices, returns

def load_pretrained_model(ticker):
    """Load pre-trained model and scaler - FIXED for Keras 3.x"""
    model_path = os.path.join(MODEL_DIR, f"{ticker}_best.h5")
    scaler_path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    
    # FIX: Load with compile=False to avoid Keras 3.x serialization issues
    model = load_model(model_path, compile=False)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def prepare_data_for_prediction(prices, ticker, scaler):
    """Transform price data using pre-trained scaler."""
    ticker_prices = prices[ticker] if ticker in prices.columns else prices
    scaled_data = scaler.transform(ticker_prices.values.reshape(-1, 1))
    return scaled_data

def run_monte_carlo(prices, returns, ticker, model, scaler, scaled_data, num_simulations=200):
    """Run Monte Carlo simulation using pre-trained model."""
    ticker_prices = prices[ticker] if ticker in prices.columns else prices
    ticker_returns = returns[ticker] if ticker in returns.columns else returns
    
    last_seq = scaled_data[-60:].reshape(1, 60, 1)
    target_scaled = model.predict(last_seq, verbose=0)[0][0]
    target = scaler.inverse_transform([[target_scaled]])[0][0]
    
    current_price = ticker_prices.iloc[-1]
    mu = (target - current_price) / current_price
    vol = ticker_returns.std()
    
    paths = np.zeros((30, num_simulations))
    paths[0] = current_price
    
    for d in range(1, 30):
        z = np.random.normal(0, 1, num_simulations)
        paths[d] = paths[d-1] * np.exp((mu - 0.5*vol**2) + vol * z)
    
    return paths, target

def calculate_portfolio_metrics(returns, tickers, weights):
    """Calculate portfolio-level metrics."""
    expected_return = np.sum(returns[tickers].mean() * 252 * weights)
    cov_matrix = returns[tickers].cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        'expected_return': expected_return,
        'volatility': portfolio_volatility,
        'sharpe': sharpe_ratio
    }

def calculate_stock_metrics(returns, ticker):
    """Calculate individual stock metrics."""
    ticker_returns = returns[ticker] if ticker in returns.columns else returns
    
    mean_return = ticker_returns.mean() * 252
    volatility = ticker_returns.std() * np.sqrt(252)
    sharpe = (mean_return / (ticker_returns.std() * np.sqrt(252))) if ticker_returns.std() > 0 else 0
    
    return {
        'mean_return': mean_return,
        'volatility': volatility,
        'sharpe': sharpe
    }

def check_models_ready(tickers):
    """Check if all pre-trained models exist."""
    ready = 0
    missing = []
    
    for ticker in tickers:
        model_path = os.path.join(MODEL_DIR, f"{ticker}_best.h5")
        scaler_path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            ready += 1
        else:
            missing.append(ticker)
    
    return ready, missing

def verify_models_exist(tickers):
    """Verify all required models exist."""
    ready, missing = check_models_ready(tickers)
    
    if ready < len(tickers):
        error_msg = f"Missing models for: {', '.join(missing)}"
        raise FileNotFoundError(error_msg)
    
    return True

st.set_page_config(page_title="AI Bank Strategist", layout="wide", page_icon="🏦")

# Premium Black & Cyan Theme CSS - Toned Down Click
st.markdown("""
    <style>
    /* Main dark background */
    .main {
        background: #000000;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    
    /* Animated header with cyan accent */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin: 1.5rem 0 2rem 0;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 50%, #00d4ff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        letter-spacing: -2px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Dark glass metrics */
    [data-testid="stMetric"] {
        background: rgba(30, 30, 30, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 700;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Dark tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(20, 20, 20, 0.9);
        backdrop-filter: blur(10px);
        padding: 12px;
        border-radius: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: #1a1a1a;
        border-radius: 12px;
        padding: 0 32px;
        font-weight: 800;
        font-size: 1.1rem;
        color: #808080;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00b8d4 0%, #0088aa 100%);
        color: #000000 !important;
        box-shadow: 0 6px 20px rgba(0, 184, 212, 0.4);
        transform: translateY(-2px);
        border: 1px solid #00b8d4;
    }
    
    /* Premium cyan buttons - toned down on click */
    .stButton > button {
        border-radius: 12px;
        font-weight: 800;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        background: linear-gradient(135deg, #00b8d4 0%, #0088aa 100%);
        color: #000000;
        box-shadow: 0 4px 16px rgba(0, 184, 212, 0.3);
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        box-shadow: 0 6px 24px rgba(0, 212, 255, 0.4);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        background: linear-gradient(135deg, #0088aa 0%, #006688 100%);
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(0, 136, 170, 0.3);
    }
    
    /* Darker sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #000000 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00d4ff !important;
    }
    
    /* Alert boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.9) 0%, rgba(22, 163, 74, 0.9) 100%);
        color: white !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.9) 0%, rgba(0, 153, 255, 0.9) 100%);
        color: white !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.9) 0%, rgba(220, 38, 38, 0.9) 100%);
        color: white !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.9) 0%, rgba(217, 119, 6, 0.9) 100%);
        color: white !important;
    }
    
    /* Input styling */
    [data-testid="stNumberInput"] label {
        color: #00d4ff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stNumberInput"] input {
        border-radius: 8px;
        border: 2px solid rgba(0, 212, 255, 0.3);
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem;
        background: rgba(20, 20, 20, 0.6);
        color: #00d4ff;
    }
    
    [data-testid="stNumberInput"] input:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.2);
        background: rgba(20, 20, 20, 0.8);
    }
    
    /* Selectbox styling */
    [data-testid="stSelectbox"] label {
        color: #00d4ff !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stSelectbox"] > div > div {
        background: rgba(20, 20, 20, 0.6);
        border: 2px solid rgba(0, 212, 255, 0.3);
        color: #e0e0e0;
    }
    
    /* Date input styling */
    [data-testid="stDateInput"] label {
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
    
    /* Checkbox styling */
    [data-testid="stCheckbox"] label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    /* All text elements */
    p, span, div {
        color: #d0d0d0;
    }
    
    h1, h2, h3 {
        color: #00d4ff;
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        background: rgba(20, 20, 20, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        background: rgba(20, 20, 20, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Caption text */
    .stCaption {
        color: #a0a0a0 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">AI-Powered Bank Portfolio Strategist</h1>', unsafe_allow_html=True)

# ==========================================
# CONFIGURATION
# ==========================================
tickers = ["JPM", "BAC", "GS", "MS", "C", "WFC", "HSBC", "RY", "TD", "USB"]
ticker_names = {
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "C": "Citigroup", "WFC": "Wells Fargo",
    "HSBC": "HSBC Holdings", "RY": "Royal Bank of Canada",
    "TD": "TD Bank", "USB": "U.S. Bancorp"
}

# ==========================================
# VERIFY PRE-TRAINED MODELS EXIST
# ==========================================
try:
    verify_models_exist(tickers)
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Quick Setup: Run TRAIN_MODELS_COLAB.py in Google Colab, then download models/ and scalers/ folders")
    st.stop()

# ==========================================
# SIDEBAR: INVESTMENT INPUT
# ==========================================
st.sidebar.header("Portfolio Configuration")
st.sidebar.markdown("### Investment Capital")

investment = st.sidebar.number_input(
    "Total Investment Amount ($)", 
    min_value=100, 
    max_value=10000000, 
    value=None,
    step=1000,
    help="Enter your total investment capital",
    placeholder="e.g., 50000"
)

# Simplified date range - no expander, no extra text
st.sidebar.markdown("---")
use_custom = st.sidebar.checkbox("Custom Date Range", value=False)

if use_custom:
    from datetime import datetime, timedelta
    end_def = datetime.now()
    start_def = end_def - timedelta(days=5*365)
    
    c1, c2 = st.sidebar.columns(2)
    with c1:
        start_date = st.date_input("Start Date", value=start_def)
    with c2:
        end_date = st.date_input("End Date", value=end_def)
else:
    start_date = None
    end_date = None

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    return get_data(tickers, start, end)

with st.spinner("Loading market data..."):
    try:
        prices, returns = load_data(tickers, start_date, end_date)
        # REMOVED: st.sidebar.success("Data loaded successfully")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

# Show model status
st.sidebar.markdown("---")
st.sidebar.markdown("### AI Models Status")
ready, missing = check_models_ready(tickers)

# ==========================================
# REQUIRE INVESTMENT
# ==========================================
if investment is None:
    st.sidebar.warning("Enter investment amount")
    st.warning("Enter your investment amount in the sidebar to view portfolio")
    st.info("Models are pre-trained in Google Colab")
    st.stop()

st.sidebar.success(f"${investment:,.0f}")

# ==========================================
# CALCULATE PORTFOLIO
# ==========================================
sharpe_ratios = {t: calculate_stock_metrics(returns, t)['sharpe'] for t in tickers}
sharpes = pd.Series(sharpe_ratios)
weights = sharpes / sharpes.sum()
port_metrics = calculate_portfolio_metrics(returns, tickers, weights.values)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["Portfolio", "AI Forecasts", "Analytics"])

# ==========================================
# TAB 1: PORTFOLIO
# ==========================================
with tab1:
    st.header("Your Portfolio Strategy")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Investment", f"${investment:,.0f}")
    with c2:
        st.metric("Annual Return", f"{port_metrics['expected_return']*100:.2f}%")
    with c3:
        st.metric("Risk", f"{port_metrics['volatility']*100:.2f}%")
    with c4:
        st.metric("Sharpe Ratio", f"{port_metrics['sharpe']:.2f}")
    
    st.markdown("---")
    
    col_pie, col_table = st.columns([1.2, 1.8])
    
    with col_pie:
        st.subheader("Allocation")
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=tickers,
            values=weights * investment,
            hole=0.5,
            marker=dict(
                colors=px.colors.qualitative.Bold,
                line=dict(color='#1a1a1a', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=13, family='Arial Black', color='white'),
            pull=[0.03] * len(tickers)
        )])
        
        fig_pie.update_layout(
            height=500,
            margin=dict(t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(font=dict(color='#e0e0e0'))
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_table:
        st.subheader("Breakdown")
        
        data = []
        for t in tickers:
            ret = returns[t] if t in returns.columns else returns
            data.append({
                "Bank": t,
                "Name": ticker_names[t],
                "Weight": f"{weights[t]*100:.1f}%",
                "Amount": f"${weights[t]*investment:,.2f}",
                "Sharpe": f"{sharpe_ratios[t]:.2f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)
        
        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            f"portfolio_{investment}.csv",
            use_container_width=True
        )

# ==========================================
# TAB 2: AI FORECASTS
# ==========================================
with tab2:
    st.header("AI-Powered Forecasts")
    
    selected = st.selectbox(
        "Select Bank for Forecast",
        tickers,
        format_func=lambda x: f"{x} - {ticker_names[x]}"
    )
    
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Generating forecast for {selected}..."):
            try:
                # Load pre-trained model
                model, scaler = load_pretrained_model(selected)
                scaled_data = prepare_data_for_prediction(prices, selected, scaler)
                
                # Run Monte Carlo
                paths, target = run_monte_carlo(
                    prices, returns, selected, model, scaler, scaled_data, 200
                )
                
                st.success(f"Forecast complete for {selected}")
                
                # Metrics
                curr = prices[selected].iloc[-1]
                rets = (paths[-1] - paths[0]) / paths[0]
                var = np.percentile(rets, 5)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"${curr:.2f}")
                m2.metric("Target (30d)", f"${target:.2f}", f"{((target/curr-1)*100):.2f}%")
                m3.metric("VaR (5%)", f"{var*100:.2f}%")
                m4.metric("Best Case", f"{np.max(rets)*100:.2f}%")
                
                # Chart
                fig = go.Figure()
                
                for i in range(min(100, paths.shape[1])):
                    fig.add_trace(go.Scatter(
                        x=list(range(30)),
                        y=paths[:, i],
                        mode='lines',
                        line=dict(width=1, color='rgba(0,212,255,0.1)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig.add_trace(go.Scatter(
                    x=list(range(30)),
                    y=np.median(paths, axis=1),
                    name="Median",
                    line=dict(color='#00d4ff', width=5)
                ))
                
                fig.update_layout(
                    height=600,
                    xaxis_title="Days Ahead",
                    yaxis_title="Price ($)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(10,10,10,0.8)',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0'),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0'),
                    legend=dict(font=dict(color='#e0e0e0'))
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ==========================================
# TAB 3: ANALYTICS
# ==========================================
with tab3:
    st.header("Performance Analytics")
    
    norm = prices[tickers] / prices[tickers].iloc[0] * 100
    
    fig = go.Figure()
    colors = px.colors.qualitative.Bold
    for i, t in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=norm.index,
            y=norm[t],
            name=t,
            mode='lines',
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        height=600,
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base=100)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,10,10,0.8)',
        hovermode='x unified',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='#e0e0e0'),
        legend=dict(font=dict(color='#e0e0e0'))
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem; background: rgba(20,20,20,0.8); border-radius: 16px; border: 1px solid rgba(0,212,255,0.2);'>
        <p style='color: #00d4ff; margin: 0; font-weight: 700;'><strong>Educational Tool | Not Financial Advice</strong></p>
        <p style='color: #a0a0a0; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Models pre-trained in Google Colab | Yahoo Finance Data</p>
    </div>
""", unsafe_allow_html=True)