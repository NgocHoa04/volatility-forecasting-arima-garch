# Volatility Forecasting & Risk Analysis

Production-grade Python package for S&P 500 volatility forecasting and Value-at-Risk computation using ARIMA, GARCH, and HAR-RV models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Overview

This package implements:

- **ARIMA**: Automatic order selection with specification comparison (Constant, AR(1), ARMA(1,1), AR(4))
- **GARCH Family**: 6 variants (GARCH, GJR-GARCH, EGARCH × Normal/Student-t)
- **GJR-GARCH Restricted**: LR test comparing p=0 (asymmetry only) vs p=1 (symmetric + asymmetric)
- **HAR-RV**: Multi-horizon regression on realized volatility proxies

Key features:
- ARIMA specification comparison with AIC/BIC/Ljung-Box metrics
- Restricted vs unrestricted GJR-GARCH hypothesis testing
- 755-step expanding window rolling forecasts
- VaR/ES computation at 95%/99% confidence levels
- Kupiec POF and Christoffersen backtesting
- Forecast accuracy comparison (MSE, MAE, QLIKE, Diebold-Mariano)

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
✅ Uses synthetic GARCH(1,1) data | 📊 Generates forecasts & backtests | 📁 Saves to `results/` & `report/figures/`

### 3️⃣ Full Analysis with Real Data

**Option A: Python Script**
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

## Project Structure

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

## Models

### ARIMA (Mean)

Auto-ARIMA with specification comparison:
- Constant Mean (0,0,0)
- AR(1) (1,0,0)
- ARMA(1,1) (1,0,1)
- AR(4) (4,0,0)

Evaluated on: AIC, BIC, Ljung-Box test (lag 10)

### GARCH (Volatility)

| Model | Normal | Student-t |
|-------|--------|-----------|
| GARCH(1,1) | ✓ | ✓ |
| GJR-GARCH(1,1) | ✓ | ✓ |
| EGARCH(1,1) | ✓ | ✓ |

Selected by AIC criterion. Student-t typically best (fat tails).

### GJR-GARCH Restricted

**Hypothesis:** Does symmetric GARCH effect (α) improve asymmetric-only model?

- **Unrestricted**: GJR(1,1) with p=1, o=1, q=1
- **Restricted**: GJR(0,1,1) with p=0, o=1, q=1 (α ≡ 0)
- **Test**: Likelihood Ratio test on ARIMA residuals
- **Interpretation**: p > 0.05 → asymmetry sufficient; p < 0.05 → both terms needed

### HAR-RV

Regression on realized volatility proxies:
- RV_d: Daily |return|
- RV_w: 5-day average
- RV_m: 22-day average

## Usage

### ARIMA Comparison

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

specs = {'Constant': (0,0,0), 'AR(1)': (1,0,0), 'ARMA': (1,0,1), 'AR(4)': (4,0,0)}
for name, order in specs.items():
    model = ARIMA(returns, order=order)
    res = model.fit()
    lb = acorr_ljungbox(res.resid, lags=[10], return_df=True)
    print(f"{name}: AIC={res.aic:.2f}, BIC={res.bic:.2f}, LB_p={lb['lb_pvalue'].values[0]:.4f}")
```

### Restricted GJR-GARCH

```python
from arch import arch_model
from scipy import stats

# Fit both models on ARIMA residuals
gjr_unr = arch_model(arima_residuals, mean='Zero', vol='GARCH', p=1, o=1, q=1, dist='studentst')
gjr_res = arch_model(arima_residuals, mean='Zero', vol='GARCH', p=0, o=1, q=1, dist='studentst')
res_unr = gjr_unr.fit(disp='off')
res_res = gjr_res.fit(disp='off')

# LR test
lr_stat = 2 * (res_unr.loglikelihood - res_res.loglikelihood)
lr_pval = 1 - stats.chi2.cdf(lr_stat, df=1)
print(f"LR: {lr_stat:.4f}, p-value: {lr_pval:.4f}")
```

### Rolling Forecast

```python
from src.volatility_forecasting.models.garch_models import rolling_volatility_forecast

vol_forecast = rolling_volatility_forecast(
    full_returns=returns, test_index=test.index, train_size=len(train),
    vol='Garch', p=1, o=1, q=1, dist='t'
)
```

### VaR & ES

```python
from src.volatility_forecasting.analysis.var_analysis import compute_var_es

results = compute_var_es(vol_forecast, confidence_levels=[0.95, 0.99], dist='t')
print(f"VaR 95%: {results['VaR_95'].mean():.4f}%")
print(f"ES 95%: {results['ES_95'].mean():.4f}%")
```

## Outputs

**CSV Results** (`results/`)
- `model_comparison.csv` — GARCH AIC/BIC ranking
- `appendix_table13_arima_params.csv` — ARIMA specification comparison
- `volatility_forecasts.csv` — Forecasts from all models
- `var_forecasts.csv` — VaR/ES at 95%/99%
- `results_backtesting.csv` — Kupiec POF and Christoffersen tests
- `results_har_vs_garch_accuracy.csv` — HAR vs GARCH comparison (MSE/MAE/QLIKE)
- `appendix_dm_test_*.csv` — Diebold-Mariano tests

**Plots** (`report/figures/`) — Generated visualizations

## Configuration

Edit `src/volatility_forecasting/config.py`:
```python
GARCH_P = 1
GARCH_Q = 1
VAR_CONFIDENCE = [0.95, 0.99]
HAR_LAGS_WEEKLY = 5
HAR_LAGS_MONTHLY = 22
```

## Testing

```bash
pytest tests/
python scripts/quick_demo.py
```