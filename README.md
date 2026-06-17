# Volatility Forecasting & Risk Analysis

A comprehensive project for forecasting S&P 500 return volatility and estimating Value-at-Risk using ARIMA-GARCH family models, OHLC-based realized volatility estimators, and statistical backtesting.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

This project implements a hybrid ARIMA-GARCH modeling pipeline on S&P 500 daily returns (2010–2024). The mean equation is modeled by ARIMA, the conditional variance by six GARCH variants, and VaR/ES are estimated and validated through formal backtesting. An advanced HAR-RV extension replaces the standard absolute-return proxy with OHLC-based volatility estimators (Parkinson, Garman-Klass, Yang-Zhang) and evaluates their impact on forecast accuracy across market regimes.

**Pipeline:**
1. Data collection via `yfinance` (closing prices + full OHLC)
2. Stationarity testing & exploratory analysis (ADF, KPSS, ACF/PACF)
3. ARIMA specification comparison (Constant, AR(1), ARMA(1,1), AR(4)) — selected by AIC/BIC/Ljung-Box
4. ARCH effect test on ARIMA residuals (ARCH-LM)
5. GARCH family fitting (6 variants: GARCH, GJR-GARCH, EGARCH × Normal/Student-t)
6. Restricted GJR-GARCH — LR test for symmetric vs asymmetric effects
7. HAR-RV benchmark — multi-horizon OLS regression on realized volatility proxies
8. **Advanced HAR-RV** — OHLC-based estimators (Parkinson, Garman-Klass, Yang-Zhang) as superior RV proxies; Diebold-Mariano + Holm correction; MCS; regime robustness
9. VaR & ES estimation at 95%/99% confidence levels
10. Backtesting — Kupiec POF & Christoffersen tests; subsample backtesting across market regimes
11. Diebold-Mariano tests (MSE, MAE, QLIKE) with Holm–Bonferroni correction and Model Confidence Set

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
├── dataset/                                    # Input data (downloaded via yfinance)
│   └── price_data.csv                          # S&P 500 daily prices (Close + OHLC)
│
├── notebooks/                                  # Jupyter notebook files
│   └── volatility_forecasting_VaR.ipynb        # Main notebook — full pipeline
│
├── src/
│   └── volatility_forecasting/
│       ├── models/
│       │   └── har_model.py                    # HARModel class — OLS fit, rolling forecast
│       └── utils/
│           ├── ohlc_estimators.py              # parkinson_volatility, garman_klass_volatility,
│           │                                   # yang_zhang_volatility, compute_multiple_proxies
│           └── statistical_tests.py            # diebold_mariano_test, holm_correction,
│                                               # model_confidence_set
│
├── results/                                    # Model outputs (auto-generated)
│   ├── har_proxy_accuracy.csv                  # HAR-RV proxy forecast accuracy (MSE/MAE/QLIKE)
│   ├── har_proxy_ranking_by_regime.csv         # Proxy MSE ranking by market regime
│   ├── har_proxy_qlike_ranking_by_regime.csv   # Proxy QLIKE ranking by market regime
│   ├── har_proxy_mse_vs_qlike_agreement.csv    # MSE vs QLIKE stability check
│   ├── cross_proxy_qlike_comparison.csv        # Cross-proxy QLIKE comparison
│   ├── cross_proxy_rank_comparison.csv         # Cross-proxy rank comparison
│   ├── unified_mcs_summary.csv                 # Model Confidence Set results
│   ├── unified_holm_dm_test.csv                # Holm-corrected DM tests (all models)
│   ├── proxies_train_cleaned.csv               # Cleaned OHLC proxies — train set
│   ├── proxies_test_cleaned.csv                # Cleaned OHLC proxies — test set
│   ├── dm_test_full_3loss.csv                  # DM test full results (MSE, MAE, QLIKE)
│   ├── table14_dm_test_3loss.csv               # DM test summary table
│   └── appendix_*.csv                          # Appendix tables (see Appendix section)
│
├── report/
│   ├── figures/                                # Generated plots (auto-saved)
│   └── time_series_report.pdf                  # Project Report
│
├── requirements.txt
└── .gitignore
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

### Appendix Tables

| Table | Contents | File |
|---|---|---|
| A | Full GARCH parameter table: ω, α, β, γ, ν, AIC, BIC, LogL, persistence | `appendix_table12_garch_params.csv` |
| B | ARIMA(4,0,0) coefficient table: estimate, SE, t-stat, p-value | `appendix_table13_arima_params.csv` |
| C | Average VaR 95%, VaR 99%, ES 99% + violation counts for all models | `appendix_table15_var_es_summary.csv` |
| D | Diebold-Mariano test (MSE and MAE) — HAR-RV vs each GARCH model | `appendix_dm_test_mse.csv`, `appendix_dm_test_mae.csv` |
| E | Expected Shortfall backtesting — McNeil & Frey (2000) exceedance residuals | `appendix_es_backtest.csv` |
| F | GARCH standardized residual diagnostics: ARCH-LM, Jarque-Bera, skewness, kurtosis | `appendix_garch_diagnostics.csv` |
| — | Subsample backtesting across market regimes | `appendix_subsample_backtesting.csv` |

---

## Tech Stack

`Python 3.9+` · `Jupyter Notebook` · `arch` · `pmdarima` · `statsmodels` · `yfinance` · `scipy` · `scikit-learn` · `matplotlib` · `seaborn`
