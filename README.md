# Volatility Forecasting & Risk Analysis

**Production-grade Python package for forecasting S&P 500 volatility and computing Value-at-Risk using GARCH and HAR-RV models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This package implements two complementary volatility forecasting approaches:

- **GARCH Family**: 6 variants (GARCH, GJR-GARCH, EGARCH × Normal/Student-t)
- **HAR-RV**: Multi-horizon linear regression model

Features:
- ✅ 755-step expanding window rolling forecasts
- ✅ VaR/ES computation at 95%/99% confidence
- ✅ Kupiec POF + Christoffersen backtesting
- ✅ Model comparison and accuracy metrics
- ✅ Verified end-to-end execution

## Quick Start

### 1️⃣ Prerequisites & Installation

```bash
# Require Python 3.9+
python --version

# Clone repository
git clone <repo-url>
cd volatility-forecasting-arima-garch

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Quick Demo

```bash
python scripts/quick_demo.py
```
✅ Uses synthetic GARCH(1,1) data | ⏱️ ~2-3 minutes | 📊 Generates forecasts & backtests | 📁 Saves to `results/` & `report/figures/`

### 3️⃣ Full Analysis with Real Data

**Option A: Python Script (Recommended)**
```bash
python scripts/volatility_forecasting_analysis.py
```
✅ Downloads S&P 500 daily prices (2010-2024) | 📈 Generates 6 GARCH variants + HAR-RV comparison | 💾 Saves results & plots

**Option B: Interactive Jupyter Notebook**
```bash
jupyter notebook notebooks/00_volatility_forecasting_VaR.ipynb
```
✅ Full pipeline with step-by-step execution | 📊 Visualizations & analysis 

---

## Structure

```
notebooks/                            # Interactive analysis
├──00_volatility_forecasting_VaR.ipynb  # Original Full Pipeline notebook
├── 01_data_exploration.ipynb         # Data loading & EDA
├── 02_arima_modeling.ipynb           # ARIMA fitting & diagnostics
├── 03_garch_modeling.ipynb           # GARCH 6-variant comparison
├── 04_har_rv_modeling.ipynb          # HAR-RV multi-horizon model
├── 05_var_es_analysis.ipynb          # VaR/ES computation & plots
└── 06_backtesting.ipynb              # Kupiec & Christoffersen tests

src/volatility_forecasting/           # Main package
├── data/
│   ├── loader.py                     # Download S&P 500 data
│   └── preprocessor.py               # Log returns, validation
├── models/
│   ├── arima_model.py                # ARIMA for mean
│   ├── garch_models.py               # GARCH family (6 variants)
│   └── har_model.py                  # HAR-RV multi-horizon
├── analysis/
│   ├── var_analysis.py               # VaR & ES computation
│   └── backtesting.py                # Kupiec & Christoffersen tests
├── utils/
│   ├── plotting.py                   # Visualization
│   └── validation.py                 # Data validation
├── config.py                         # 100+ parameters
├── logger.py                         # Logging setup
└── __init__.py

scripts/                              # Standalone pipelines
├── volatility_forecasting_analysis.py # Production (real S&P 500)
├── quick_demo.py                     # Demo (synthetic data)
└── run_full_analysis.py

dataset/                              # Input data
results/                              # Output CSVs
report/figures/                       # Generated plots and project report
```

---

## Models

### GARCH Family (6 Variants)

```
├── GARCH(1,1)-Normal
├── GARCH(1,1)-Student-t
├── GJR-GARCH(1,1)-Normal
├── GJR-GARCH(1,1)-Student-t
├── EGARCH(1,1)-Normal
└── EGARCH(1,1)-Student-t
```

Selected by **AIC criterion**. Best model typically **Student-t GARCH** (captures fat tails).

### HAR-RV (Heterogeneous Autoregressive)

Linear regression on realized volatility proxies:
- **RV_d**: Daily (|return|)
- **RV_w**: 5-day average
- **RV_m**: 22-day average

---

## Configuration

Edit `src/volatility_forecasting/config.py`:

```python
GARCH_P = 1              # GARCH order
GARCH_Q = 1              # Q order
VAR_CONFIDENCE = [0.95, 0.99]  # VaR levels
HAR_LAGS_WEEKLY = 5
HAR_LAGS_MONTHLY = 22
```

---

## Usage (Python API)

```python
from src.volatility_forecasting.models.garch_models import rolling_volatility_forecast
from src.volatility_forecasting.analysis.var_analysis import compute_var_es

# Rolling GARCH forecast (expanding window)
vol_forecast = rolling_volatility_forecast(
    full_returns=returns,
    test_index=test.index,
    train_size=len(train),
    vol='Garch', p=1, o=1, q=1, dist='t'
)

# Compute VaR & ES
var_results = compute_var_es(vol_forecast, confidence_levels=[0.95, 0.99], dist='t')
print(f"VaR 95%: {var_results['VaR_95'].mean():.4f}%")
```

---

## Outputs

**CSV Results** (`results/`)
- `model_comparison.csv` — AIC/BIC ranking
- `volatility_forecasts.csv` — Forecasts all models
- `var_forecasts.csv` — VaR/ES values
- `results_backtesting.csv` — Kupiec + Christoffersen tests

**Plots** (`report/figures/`) - Generated when running with real data

---

## Testing

```bash
pytest tests/
python scripts/quick_demo.py  # End-to-end verification
```