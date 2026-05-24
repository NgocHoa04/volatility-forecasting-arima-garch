"""
Configuration and constants for volatility forecasting project.
"""

import os
from pathlib import Path
from typing import Final

# ──────────────────────────────────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────────────────────────────────

PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent
DATA_RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "report" / "figures"
CONFIGS_DIR: Final[Path] = PROJECT_ROOT / "configs"
LOGS_DIR: Final[Path] = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# Data Parameters
# ──────────────────────────────────────────────────────────────────────────

DATA_TICKER: Final[str] = "^GSPC"  # S&P 500
DATA_START_DATE: Final[str] = "2010-01-01"
DATA_END_DATE: Final[str] = "2024-12-31"
DATA_RAW_FILE: Final[Path] = DATA_RAW_DIR / "price_data.csv"
DATA_PROCESSED_FILE: Final[Path] = DATA_PROCESSED_DIR / "returns.csv"

# ──────────────────────────────────────────────────────────────────────────
# Train/Test Split
# ──────────────────────────────────────────────────────────────────────────

TRAIN_SIZE_PCT: Final[float] = 0.8  # 80% train, 20% test
TEST_START_DATE: Final[str] = "2021-01-01"  # Alternative: specify explicit date

# ──────────────────────────────────────────────────────────────────────────
# Stationarity Tests
# ──────────────────────────────────────────────────────────────────────────

ADF_ALPHA: Final[float] = 0.05  # Significance level
KPSS_ALPHA: Final[float] = 0.05

# ──────────────────────────────────────────────────────────────────────────
# ARIMA Configuration
# ──────────────────────────────────────────────────────────────────────────

ARIMA_MAX_P: Final[int] = 5
ARIMA_MAX_D: Final[int] = 2
ARIMA_MAX_Q: Final[int] = 5
ARIMA_SEASONAL: Final[bool] = False
ARIMA_STEPWISE: Final[bool] = True
ARIMA_INFORMATION_CRITERION: Final[str] = "aic"

# ──────────────────────────────────────────────────────────────────────────
# GARCH Configuration
# ──────────────────────────────────────────────────────────────────────────

GARCH_MODELS: Final[list] = [
    "GARCH",
    "GJR-GARCH",
    "EGARCH",
]

GARCH_LAGS: Final[int] = 1
GARCH_P: Final[int] = 1
GARCH_Q: Final[int] = 1
GARCH_O: Final[int] = 1  # For GJR-GARCH
GARCH_POWER: Final[float] = 2.0  # For EGARCH

# ──────────────────────────────────────────────────────────────────────────
# HAR-RV Configuration
# ──────────────────────────────────────────────────────────────────────────

HAR_DAILY_LAG: Final[int] = 1
HAR_WEEKLY_LAG: Final[int] = 5
HAR_MONTHLY_LAG: Final[int] = 22

# ──────────────────────────────────────────────────────────────────────────
# Rolling Forecast Configuration
# ──────────────────────────────────────────────────────────────────────────

ROLLING_WINDOW_SIZE: Final[int] = 252  # ~1 year of trading days
FORECAST_HORIZON: Final[int] = 1  # 1-step-ahead forecast

# ──────────────────────────────────────────────────────────────────────────
# Value-at-Risk (VaR) & Expected Shortfall (ES)
# ──────────────────────────────────────────────────────────────────────────

VaR_CONFIDENCE_LEVELS: Final[list] = [0.95, 0.99]  # 95% and 99%
VAR_PORTFOLIO_VALUE: Final[float] = 1_000_000.0  # $1M for example

# ──────────────────────────────────────────────────────────────────────────
# Backtesting Configuration
# ──────────────────────────────────────────────────────────────────────────

KUPIEC_ALPHA: Final[float] = 0.05
CHRISTOFFERSEN_ALPHA: Final[float] = 0.05

# ──────────────────────────────────────────────────────────────────────────
# Visualization Configuration
# ──────────────────────────────────────────────────────────────────────────

FIGURE_DPI: Final[int] = 120
FIGURE_FACECOLOR: Final[str] = "white"
PLOT_STYLE: Final[str] = "seaborn-v0_8-darkgrid"
FONT_SIZE: Final[int] = 11

# ──────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────

LOG_LEVEL: Final[str] = "INFO"
LOG_FORMAT: Final[str] = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOG_FILE: Final[Path] = LOGS_DIR / "volatility_forecasting.log"

# ──────────────────────────────────────────────────────────────────────────
# Display Options
# ──────────────────────────────────────────────────────────────────────────

PANDAS_FLOAT_FORMAT: Final[str] = "{:.4f}"
PANDAS_MAX_ROWS: Final[int] = 100
PANDAS_MAX_COLUMNS: Final[int] = 20
