"""
Main analysis script for volatility forecasting.

Run this script to execute the full pipeline:
1. Download and preprocess data
2. Fit ARIMA and GARCH models
3. Compute VaR and Expected Shortfall
4. Backtest VaR models
5. Generate results and visualizations
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from volatility_forecasting.logger import setup_logger
from volatility_forecasting.config import (
    RESULTS_DIR, FIGURES_DIR,
    ROLLING_WINDOW_SIZE, FORECAST_HORIZON,
)
from volatility_forecasting.data import DataLoader, DataPreprocessor
from volatility_forecasting.models import (
    GARCHModel, GJRGARCHModel, EGARCHModel, HARModel, ARIMAModel
)
from volatility_forecasting.analysis import VaRAnalysis, Backtesting
from volatility_forecasting.utils import setup_plot_style, save_figure

logger = setup_logger(__name__)


def main():
    """Execute full analysis pipeline."""
    logger.info("=" * 80)
    logger.info("VOLATILITY FORECASTING AND VALUE-AT-RISK ANALYSIS")
    logger.info("=" * 80)
    
    # Setup
    setup_plot_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: Load and preprocess data
    # ─────────────────────────────────────────────────────────────
    
    logger.info("\n[1/5] Loading and preprocessing data...")
    
    loader = DataLoader()
    price_data = loader.download()
    
    preprocessor = DataPreprocessor(price_data)
    returns = preprocessor.compute_log_returns("Adj Close")
    returns = preprocessor.remove_na()
    
    logger.info(f"Data loaded: {len(returns)} observations")
    logger.info(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Summary statistics
    print("\n" + preprocessor.describe_returns().to_string())
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: Fit models
    # ─────────────────────────────────────────────────────────────
    
    logger.info("\n[2/5] Fitting volatility models...")
    
    # Convert to percentage
    returns_pct = returns * 100
    
    # Fit GARCH models
    models = {
        "GARCH": GARCHModel(),
        "GJR-GARCH": GJRGARCHModel(),
        "EGARCH": EGARCHModel(),
    }
    
    for name, model in models.items():
        model.fit(returns_pct, disp="off")
    
    # Fit HAR model
    har = HARModel()
    har.fit(returns)
    
    logger.info(f"HAR R²: {har.get_r_squared(returns):.4f}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: Rolling forecast
    # ─────────────────────────────────────────────────────────────
    
    logger.info("\n[3/5] Generating rolling forecasts...")
    
    test_size = len(returns) - ROLLING_WINDOW_SIZE
    forecasts = {name: [] for name in models.keys()}
    forecasts["HAR"] = []
    dates = []
    
    for i in range(ROLLING_WINDOW_SIZE, len(returns)):
        # Get training window
        train_window = returns_pct.iloc[i - ROLLING_WINDOW_SIZE:i]
        test_date = returns.index[i]
        
        # Fit and forecast each GARCH model
        for name, model in models.items():
            model_copy = model.__class__()
            model_copy.fit(train_window, disp="off")
            forecast = model_copy.forecast(horizon=FORECAST_HORIZON)
            forecasts[name].append(forecast["volatility"][0])
        
        # Forecast HAR
        train_returns = returns.iloc[i - ROLLING_WINDOW_SIZE:i]
        har_copy = HARModel()
        har_copy.fit(train_returns)
        forecast_har = har_copy.forecast(train_returns, steps=1)
        forecasts["HAR"].append(forecast_har * 100)  # Convert to percentage
        
        dates.append(test_date)
        
        if (i - ROLLING_WINDOW_SIZE + 1) % 100 == 0:
            logger.info(f"  Completed {i - ROLLING_WINDOW_SIZE + 1}/{test_size} forecasts")
    
    # Create results dataframe
    forecasts_df = pd.DataFrame(forecasts, index=dates)
    results_file = RESULTS_DIR / "results_volatility_forecasts.csv"
    forecasts_df.to_csv(results_file)
    logger.info(f"Saved forecasts to {results_file}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: VaR and Backtesting
    # ─────────────────────────────────────────────────────────────
    
    logger.info("\n[4/5] Computing VaR and backtesting...")
    
    # Use last window for VaR demo
    test_returns = returns_pct.iloc[-len(forecasts_df):]
    
    var_results = {}
    for model_name in forecasts_df.columns:
        var_analysis = VaRAnalysis(
            test_returns,
            forecasts_df[model_name],
            confidence_levels=[0.95, 0.99]
        )
        var_analysis.compute()
        var_results[model_name] = var_analysis.var_results[0.95]
    
    logger.info("VaR and ES computed")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 5: Summary and results
    # ─────────────────────────────────────────────────────────────
    
    logger.info("\n[5/5] Generating summary and visualizations...")
    
    # Model comparison
    comparison = pd.DataFrame({
        "Mean Volatility": forecasts_df.mean(),
        "Std Volatility": forecasts_df.std(),
    })
    
    comparison_file = RESULTS_DIR / "results_model_comparison.csv"
    comparison.to_csv(comparison_file)
    logger.info(f"Saved comparison to {comparison_file}")
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in forecasts_df.columns:
        ax.plot(forecasts_df.index, forecasts_df[col], label=col, alpha=0.7)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    ax.set_title("Volatility Forecasts: GARCH vs HAR-RV Models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = save_figure(fig, "volatility_comparison")
    logger.info(f"Saved figure to {fig_path}")
    plt.close(fig)
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info(f"Figures saved to: {FIGURES_DIR}")
    logger.info(f"Logs saved to: logs/volatility_forecasting.log")


if __name__ == "__main__":
    main()
