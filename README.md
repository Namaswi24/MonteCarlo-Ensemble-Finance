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


