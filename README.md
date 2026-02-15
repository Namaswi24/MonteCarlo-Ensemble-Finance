##  Team Members
- **Janapareddy Vidya Varshini** – 230041013  
- **Korubilli Vaishnavi** – 230041016  
- **Mullapudi Namaswi** – 230041023 

# 🏦 Bank Portfolio AI - Advanced Financial Analysis & Prediction System

A comprehensive machine learning system for analyzing and predicting stock performance of major banking institutions using ensemble learning, LSTM neural networks, and Monte Carlo simulations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models & Methodology](#models--methodology)
- [Output & Metrics](#output--metrics)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements an advanced AI-powered financial analysis system that:

1. **Analyzes 10 major banking stocks**: JPM, BAC, GS, MS, C, WFC, HSBC, RY, TD, USB
2. **Trains ensemble ML models** combining LSTM neural networks and Random Forest
3. **Performs risk analysis** using VaR, CVaR, Sharpe Ratio, and Rachev Ratio
4. **Generates 30-day forecasts** using Monte Carlo simulations (500 paths per stock)
5. **Saves trained models** for deployment in production applications

## ✨ Features

### 📊 Statistical Analysis
- Daily return statistics and volatility calculations
- Beta calculation against S&P 500 (SPY) benchmark
- Annual volatility estimates
- 5-year historical data analysis

### 🤖 Machine Learning Models
- **LSTM Neural Network**: Deep learning for time series prediction
  - 64-unit LSTM layer with dropout
  - 32-unit LSTM layer
  - Early stopping to prevent overfitting
  - 20 epochs with validation monitoring
  
- **Random Forest Regressor**: Ensemble tree-based model
  - 100 estimators
  - Robust to outliers
  
- **Reinforcement-Weighted Ensemble**: Combines both models using inverse error weighting

### 📈 Risk Analysis
- **Value at Risk (VaR)**: 5th percentile loss estimation
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold
- **Sharpe Ratio**: Risk-adjusted return metric
- **Rachev Ratio**: Upside potential vs downside risk

### 🎲 Monte Carlo Simulation
- 500 independent price paths per stock
- 30-day forward projections
- Median path tracking
- Confidence interval visualization

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Install Dependencies
```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow keras pickle5 --break-system-packages
```

Or using requirements.txt:
```bash
pip install -r requirements.txt --break-system-packages
```

### Create Requirements File
```txt
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
keras>=2.12.0
```

## 📁 Project Structure

```
bank-portfolio-ai/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── notebooks/
│   └── analysis.ipynb                 # Main analysis notebook
│
├── models/                            # Trained LSTM models (generated)
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
├── scalers/                           # Data scalers (generated)
│   ├── JPM_scaler.pkl
│   ├── BAC_scaler.pkl
│   ├── GS_scaler.pkl
│   ├── MS_scaler.pkl
│   ├── C_scaler.pkl
│   ├── WFC_scaler.pkl
│   ├── HSBC_scaler.pkl
│   ├── RY_scaler.pkl
│   ├── TD_scaler.pkl
│   └── USB_scaler.pkl
│
└── outputs/                           # Generated visualizations
    ├── monte_carlo_simulations.png
    └── risk_ratio_comparison.png
```

## 💻 Usage

### Option 1: Run Complete Analysis

```python
# Execute the entire analysis pipeline
python analysis_notebook.py
```

This will:
1. Download 5 years of historical data
2. Calculate statistical metrics
3. Train all ML models
4. Perform risk analysis
5. Generate Monte Carlo simulations
6. Create visualizations
7. Save trained models and scalers

### Option 2: Load Pre-trained Models

```python
import pickle
from tensorflow.keras.models import load_model

# Load a specific model
ticker = "JPM"
model = load_model(f"models/{ticker}_model.keras")

# Load corresponding scaler
with open(f"scalers/{ticker}_scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
# ... your prediction code here
```

### Option 3: Deploy in Application

```python
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

class BankStockPredictor:
    def __init__(self, model_dir="models", scaler_dir="scalers"):
        self.models = {}
        self.scalers = {}
        self.tickers = ["JPM", "BAC", "GS", "MS", "C", "WFC", "HSBC", "RY", "TD", "USB"]
        
        # Load all models
        for ticker in self.tickers:
            model_path = os.path.join(model_dir, f"{ticker}_model.keras")
            scaler_path = os.path.join(scaler_dir, f"{ticker}_scaler.pkl")
            
            self.models[ticker] = load_model(model_path)
            with open(scaler_path, 'rb') as f:
                self.scalers[ticker] = pickle.load(f)
    
    def predict(self, ticker, last_60_days):
        """
        Predict next day price
        
        Args:
            ticker: Stock ticker symbol
            last_60_days: Array of last 60 closing prices
        
        Returns:
            Predicted price for next day
        """
        scaled = self.scalers[ticker].transform(last_60_days.reshape(-1, 1))
        prediction = self.models[ticker].predict(scaled.reshape(1, 60, 1), verbose=0)
        return self.scalers[ticker].inverse_transform(prediction)[0][0]

# Usage
predictor = BankStockPredictor()
predicted_price = predictor.predict("JPM", last_60_prices)
```

## 🧮 Models & Methodology

### Data Processing
1. **Historical Data**: 5 years of adjusted closing prices
2. **Normalization**: MinMaxScaler (0-1 range)
3. **Sequence Creation**: 60-day lookback windows
4. **Train/Test Split**: 80/20 ratio

### LSTM Architecture
```
Input Layer: (60, 1) - 60 timesteps
│
├─ LSTM Layer 1: 64 units, return_sequences=True
├─ Dropout: 0.2
├─ LSTM Layer 2: 32 units
└─ Dense Output: 1 unit (predicted price)

Optimizer: Adam
Loss Function: MSE (Mean Squared Error)
```

### Ensemble Weighting
The final prediction combines LSTM and Random Forest using inverse error weighting:

```
Weight_LSTM = MAE_RF / (MAE_LSTM + MAE_RF)
Weight_RF = 1 - Weight_LSTM

Final_Prediction = (LSTM_pred × Weight_LSTM) + (RF_pred × Weight_RF)
```

### Monte Carlo Simulation
```python
# For each of 500 paths:
for day in range(1, 31):
    S[day] = S[day-1] × exp((μ - 0.5σ²) + σ × Z)
    
where:
    μ = expected return (from ML prediction)
    σ = historical volatility
    Z = random normal distribution
```

## 📊 Output & Metrics

### Section 1: Daily Return Statistics
- Mean Daily Return
- Daily Standard Deviation
- Annual Volatility
- Beta (vs SPY)

### Section 2: ML Performance Metrics
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- LSTM Weight in Ensemble

### Section 3: Risk & Ratio Summary
- VaR (5%): Value at Risk at 5th percentile
- CVaR (5%): Conditional VaR (expected shortfall)
- Sharpe Ratio: (Return - Risk_free) / Volatility
- Rachev Ratio: E[gain|95th%] / E[loss|5th%]

### Visualizations
1. **30-Day Monte Carlo Simulations** (10 subplots)
   - 10 sample paths per stock
   - Median path highlighted
   - Confidence bands

2. **Risk-Reward Ratio Comparison** (Bar Chart)
   - Sharpe Ratio comparison
   - Rachev Ratio comparison

## 📦 Requirements

### Core Dependencies
- **yfinance**: Financial data download
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **scikit-learn**: ML utilities and Random Forest
- **tensorflow/keras**: Deep learning (LSTM)

### System Requirements
- RAM: 8GB+ recommended
- Storage: 500MB+ for models and data
- GPU: Optional (speeds up training)

## 🔧 Troubleshooting

### Model Loading Errors
**Issue**: `Unable to load model.h5`
**Solution**: Models are now saved as `.keras` format (TensorFlow 2.12+)
```python
# Correct way to load
model = load_model("models/JPM_model.keras")
```

### Data Download Issues
**Issue**: `yfinance` fails to download data
**Solution**: 
```python
# Add retry logic
import time
for attempt in range(3):
    try:
        data = yf.download(tickers, period="5y")
        break
    except:
        time.sleep(5)
```

### Memory Errors
**Issue**: Large models cause OOM
**Solution**: Load models individually when needed
```python
# Instead of loading all at once
model = load_model(f"models/{ticker}_model.keras")
prediction = model.predict(data)
del model  # Free memory
```

### Pickle Version Issues
**Issue**: Can't load scaler files
**Solution**: Install `pickle5` for compatibility
```bash
pip install pickle5 --break-system-packages
```

## 📈 Performance Benchmarks

Typical training time (per model):
- CPU: ~3-5 minutes
- GPU: ~1-2 minutes

Expected accuracy metrics:
- MAE: 0.01 - 0.03 (normalized scale)
- RMSE: 0.015 - 0.04 (normalized scale)

## 🚀 Future Enhancements

- [ ] Add more technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Implement attention mechanisms in LSTM
- [ ] Add sentiment analysis from news/social media
- [ ] Real-time prediction API
- [ ] Portfolio optimization algorithms
- [ ] Backtesting framework
- [ ] Integration with trading platforms

## 📝 License

This project is for educational and research purposes. Not financial advice.

## ⚠️ Disclaimer

**This software is for educational purposes only. Do not use this for actual trading decisions without consulting a financial advisor. Past performance does not guarantee future results. The authors are not responsible for any financial losses.**

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ for quantitative finance enthusiasts**
