"""Analysis module for VaR, backtesting, and diagnostics."""

from .var_analysis import VaRAnalysis, compute_var, compute_es
from .backtesting import Backtesting
from .proxy_validation import (
    validate_proxy_levels, detect_proxy_outliers, cap_proxy_outliers,
    validate_proxy_correlation, generate_proxy_report
)

__all__ = [
    "VaRAnalysis", "compute_var", "compute_es", "Backtesting",
    "validate_proxy_levels", "detect_proxy_outliers", "cap_proxy_outliers",
    "validate_proxy_correlation", "generate_proxy_report"
]
