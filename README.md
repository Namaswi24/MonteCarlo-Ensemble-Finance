# 🏦 Bank Portfolio AI - Advanced Financial Analysis & Prediction System

Automated **stock price prediction** and **risk analysis** for major banking institutions using **ensemble machine learning** combining **LSTM Neural Networks** and **Random Forest** models.  
This project demonstrates the use of deep learning and reinforcement-weighted ensembles for **financial time series forecasting**, achieving superior prediction accuracy through multi-model integration and Monte Carlo simulations.

---

##  Team Members
- **Janapareddy Vidya Varshini** – 230041013  
- **Korubilli Vaishnavi** – 230041016  
- **Mullapudi Namaswi** – 230041023 


---

## 📋 Problem Statement

> Develop an intelligent automated system to analyze and predict stock performance of **10 major banking institutions** using ensemble machine learning techniques.  
> The goal is to build an accurate financial forecasting model capable of risk assessment, price prediction, and portfolio optimization with minimal human intervention.

Stock market analysis for banking institutions requires understanding complex temporal patterns, assessing multiple risk metrics, and generating reliable forecasts for investment decisions.  
Manual analysis is time-intensive and susceptible to cognitive biases — hence, an automated ML-powered solution is proposed.

---

## 📊 Dataset Structure

### Dataset Info
- **Data Source:** Yahoo Finance (yfinance API)
- **Time Period:** 5 years of historical data
- **Stocks Analyzed:** 10 major banks
  - JPM (JPMorgan Chase), BAC (Bank of America), GS (Goldman Sachs)
  - MS (Morgan Stanley), C (Citigroup), WFC (Wells Fargo)
  - HSBC (HSBC Holdings), RY (Royal Bank of Canada)
  - TD (Toronto-Dominion Bank), USB (U.S. Bancorp)
- **Benchmark:** SPY (S&P 500 ETF)


---

## ⚙️ Data Preprocessing & Augmentation

Implemented through comprehensive data pipeline:
- **Data Collection:** Fetch 5-year historical data via yfinance API
- **Normalization:** MinMaxScaler to [0, 1] range
- **Sequence Creation:** 60-day lookback windows for time series
- **Feature Engineering:**
  - Daily returns calculation
  - Volatility metrics
  - Beta calculation against SPY benchmark


---

## 🤖 Model Architecture — Ensemble Learning

The system combines two powerful ML models with reinforcement-based weighting for optimal predictions.

### 🔑 Key Components
1. **LSTM Neural Network:**
   - Two-layer LSTM architecture (64 and 32 units)
   - Dropout layer (0.2) to prevent overfitting
   - Captures long-term temporal dependencies in stock prices
   - Early stopping mechanism to optimize training

2. **Random Forest Regressor:**
   - Ensemble of 100 decision trees
   - Robust to outliers and non-linear patterns
   - Provides complementary predictions to LSTM

3. **Reinforcement-Weighted Ensemble:**
   - Dynamically combines LSTM and Random Forest predictions
   - Weights determined by inverse error (better model gets higher weight)
   - Adaptive performance across different stocks

4. **Monte Carlo Simulation:**
   - Generates 500 possible price paths for 30-day horizon
   - Uses Geometric Brownian Motion for realistic price modeling
   - Provides confidence intervals and risk distribution

---

## 🧪 Experimental Setup

| Parameter | Value |
|------------|--------|
| **Framework** | TensorFlow/Keras, scikit-learn |
| **Device** | CUDA (GPU) if available, else CPU |
| **Input Size** | 60 timesteps |
| **Training Epochs** | 20 (with early stopping) |
| **Batch Size** | 32 |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss Function** | Mean Squared Error (MSE) |
| **Metrics** | MAE, MSE, RMSE, VaR, CVaR, Sharpe, Rachev |

---

## 📈 Results & Observations

- Models trained successfully for all **10 banking stocks**
- Early stopping prevented overfitting (best models at epochs 15-19)
- Ensemble approach outperformed individual models
- Generated **150,000 price paths** (500 paths × 30 days × 10 stocks) for Monte Carlo analysis

### Performance Metrics

| Metric | Description |
|---------|--------------|
| **MAE (Mean Absolute Error)** | Average prediction error in normalized scale |
| **RMSE (Root Mean Squared Error)** | Standard error metric penalizing large deviations |
| **VaR (Value at Risk 5%)** | Maximum expected loss at 95% confidence |
| **CVaR (Conditional VaR)** | Expected loss beyond VaR threshold |
| **Sharpe Ratio** | Risk-adjusted return metric |
| **Rachev Ratio** | Upside potential vs downside risk |

### Key Findings
✅ LSTM-RF ensemble achieved **MAE of 0.01-0.03** (normalized scale)  
✅ High Sharpe ratios identified best risk-adjusted banking stocks  
✅ Monte Carlo simulations provided comprehensive risk assessment  
✅ Models generalized well across different market conditions  

---

## 📊 Visualizations

The system generates two main visualizations:

1. **30-Day Monte Carlo Simulations:**
   - 10 subplots (one per bank)
   - Sample paths showing price uncertainty
   - Median path highlighted for expected trajectory

2. **Risk-Reward Ratio Comparison:**
   - Bar chart comparing Sharpe and Rachev ratios
   - Identifies best risk-adjusted investment opportunities
   - Helps in portfolio optimization decisions

---





## 🚀 How to Run This Project

### Step 1: Clone or Download the Repository

```bash
# Clone the GitHub repository
git clone https://github.com/Namaswi24/MonteCarlo-Ensemble-Finance.git

# Navigate into the project folder
cd MonteCarlo-Ensemble-Finance

```

**Important:** Make sure you have the complete folder with:
- `app.py` file
- `models/` folder (with 10 .keras files)
- `scalers/` folder (with 10 .pkl files)
- `README.md` file

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt --break-system-packages
```

**Required packages:**
```txt
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
streamlit>=1.28.0
plotly>=5.14.0
pickle5
```

### Step 3: Run the Application

```bash
# Run Streamlit app
streamlit run app.py
```
 Streamlit will automatically open in your browser at `http://localhost:8501`

### What You Can Do:

✅ Select from 10 pre-trained banking stocks  
✅ View 30-day price predictions with Monte Carlo simulations  
✅ Analyze comprehensive risk metrics (VaR, CVaR, Sharpe, Rachev)  
✅ Interactive visualizations with real-time updates  
✅ Download prediction results and charts

---


## 📁 Project Structure

```
Bank-Portfolio-AI/
│
├── README.md                       # Project documentation
├── app.py                         # Streamlit web application (main file)
│
├── models/                        # Pre-trained LSTM models (10 files)
│   ├── JPM_model.keras
│   ├── BAC_model.keras
│   ├── GS_model.keras
│   ├── MS_model.keras
│   ├── C_model.keras
│   ├── WFC_model.keras
│   ├── HSBC_model.keras
│   ├── RY_model.keras
│   ├── TD_model.keras
│   └── USB_model.keras
│
└── scalers/                       # Data scalers (10 files)
    ├── JPM_scaler.pkl
    ├── BAC_scaler.pkl
    ├── GS_scaler.pkl
    ├── MS_scaler.pkl
    ├── C_scaler.pkl
    ├── WFC_scaler.pkl
    ├── HSBC_scaler.pkl
    ├── RY_scaler.pkl
    ├── TD_scaler.pkl
    └── USB_scaler.pkl
```



