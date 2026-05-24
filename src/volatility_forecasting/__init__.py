"""
Volatility Forecasting and Value-at-Risk Estimation Package

Using ARIMA-GARCH models and HAR-RV benchmarks for stock market volatility.
"""

__version__ = "1.0.0"
__author__ = "Time Series Analysis Team"

from . import data, models, analysis, utils

__all__ = ["data", "models", "analysis", "utils"]
