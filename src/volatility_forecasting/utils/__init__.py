"""Utilities module."""

from .plotting import setup_plot_style, save_figure
from .validation import validate_returns, validate_data
from .volatility_proxies import (
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
    compute_multiple_proxies,
)
from .statistical_tests import (
    holm_correction,
    model_confidence_set,
    diebold_mariano_test,
    compare_forecasts,
)

__all__ = [
    "setup_plot_style",
    "save_figure",
    "validate_returns",
    "validate_data",
    "parkinson_volatility",
    "garman_klass_volatility",
    "yang_zhang_volatility",
    "compute_multiple_proxies",
    "holm_correction",
    "model_confidence_set",
    "diebold_mariano_test",
    "compare_forecasts",
]
