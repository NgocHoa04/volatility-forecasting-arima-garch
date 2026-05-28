# Volatility Forecasting & Risk Analysis

A comprehensive project for forecasting S&P 500 return volatility and estimating Value-at-Risk using ARIMA-GARCH family models, OHLC-based realized volatility estimators, and statistical backtesting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This project implements a hybrid ARIMA-GARCH modeling pipeline on S&P 500 daily returns. The mean equation is modeled by ARIMA, the conditional variance by six GARCH variants, and VaR/ES are estimated and validated through formal backtesting. An advanced HAR-RV extension replaces the standard absolute-return proxy with OHLC-based volatility estimators (Parkinson, Garman-Klass, Yang-Zhang) and evaluates their impact on forecast accuracy across market regimes.

**Pipeline:**
1. Data collection via `yfinance` (closing prices + full OHLC)
2. Stationarity testing & exploratory analysis
3. ARIMA specification comparison (Constant, AR(1), ARMA(1,1), AR(4)) — selected by AIC/BIC/Ljung-Box
4. GARCH family fitting (6 variants: GARCH, GJR-GARCH, EGARCH × Normal/Student-t)
5. Restricted GJR-GARCH — LR test for symmetric vs asymmetric effects
6. HAR-RV benchmark — multi-horizon regression on realized volatility proxies
7. **Advanced HAR-RV** — OHLC-based estimators (Parkinson, Garman-Klass, Yang-Zhang) as superior RV proxies
8. VaR & ES estimation at 95%/99% confidence levels
9. Backtesting — Kupiec POF & Christoffersen tests; forecast accuracy (MSE, MAE, QLIKE, Diebold-Mariano with Holm correction, Model Confidence Set)

---

## Quick Start


### 1️⃣ Prerequisites & Installation

```bash
# Requires Python 3.9+
git clone https://github.com/NgocHoa04/volatility-forecasting-arima-garch.git
cd volatility-forecasting-arima-garch

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```
### 2️⃣ Quick Demo

```bash
python scripts/quick_demo.py
```

### 3️⃣ Full Analysis with Real Data

**Option A: Python Script**
```bash
python scripts/volatility_forecasting_analysis.py
```

**Option B: Interactive Jupyter Notebook**
```bash
jupyter notebook volatility_forecasting_VaR.ipynb
```

The notebook is self-contained and walks through the full pipeline from data download to backtesting results.

---

## Project Structure

```
volatility-forecasting-arima-garch/
│
├── volatility_forecasting_VaR.ipynb     # Main notebook — full pipeline
├── requirements.txt
├── .gitignore
│
├── dataset/                             # Input data (downloaded via yfinance)
│   └── price_data.csv                   # S&P 500 daily prices (Close + OHLC)
│
├── results/                             # Model outputs (auto-generated)
│   ├── model_comparison.csv             # GARCH variant AIC/BIC ranking
│   ├── volatility_forecasts.csv         # Rolling forecasts from all models
│   ├── var_forecasts.csv                # VaR & ES at 95%/99%
│   ├── results_backtesting.csv          # Kupiec POF & Christoffersen results
│   ├── results_har_vs_garch_accuracy.csv  # HAR vs GARCH (MSE/MAE/QLIKE)
│   ├── har_proxy_qlike_ranking_by_regime.csv  # OHLC proxy ranking by market regime
│   ├── har_proxy_mse_vs_qlike_agreement.csv   # MSE vs QLIKE stability check
│   └── appendix_dm_test_*.csv           # Diebold-Mariano pairwise tests
│
└── report/
    ├── figures/                         # Generated plots (auto-saved)
    └── time_series_report.pdf           # Project Report
```

---

## Models

| Model | Description |
|---|---|
| ARIMA | Mean equation — auto specification comparison (AIC/BIC/Ljung-Box) |
| GARCH(1,1) | Symmetric conditional variance — Normal & Student-t |
| GJR-GARCH(1,1) | Asymmetric response to negative shocks — Normal & Student-t |
| EGARCH(1,1) | Log-variance with leverage effect — Normal & Student-t |
| GJR-GARCH Restricted | LR test: α ≡ 0 (asymmetry only) vs full GJR |
| HAR-RV | Multi-horizon OLS regression on realized volatility proxies |

### OHLC-Based Volatility Estimators

Used as superior RV proxies in the advanced HAR-RV section, replacing the standard absolute-return baseline:

| Estimator | Formula basis | Key property |
|---|---|---|
| Absolute Return | `\|log return\|` | Baseline proxy (close-to-close) |
| Parkinson (1980) | High/Low range | More efficient than close-to-close |
| Garman-Klass (1980) | Open, High, Low, Close | Handles overnight gaps |
| Yang-Zhang (2000) | Overnight jump + intraday range | Minimum-variance unbiased estimator |

Each proxy is used to fit and generate rolling HAR-RV forecasts, with performance compared across MSE, QLIKE, and market regimes (full period, crisis, normal) via Diebold-Mariano tests and Model Confidence Set (MCS).

---

## Results

Outputs are saved to `results/` (CSV tables) and `report/figures/` (plots). VaR coverage rates are evaluated against nominal confidence levels (95%, 99%) via Kupiec and Christoffersen tests. Model forecasts are compared using MSE, MAE, QLIKE, and Diebold-Mariano tests with Holm correction. OHLC proxy rankings are assessed for stability across market regimes.

---

## Tech Stack

`Python 3.9+` · `Jupyter Notebook` · `arch` · `pmdarima` · `statsmodels` · `yfinance` · `scipy` · `scikit-learn` · `matplotlib` · `seaborn`
