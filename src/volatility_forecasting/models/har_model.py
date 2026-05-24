"""HAR (Heterogeneous Autoregressive) model with realized volatility."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional

from ..config import (
    HAR_DAILY_LAG,
    HAR_WEEKLY_LAG,
    HAR_MONTHLY_LAG,
)
from ..logger import logger


class HARModel:
    """
    Heterogeneous Autoregressive (HAR) model.
    
    Combines three horizons of realized volatility:
    - Daily (1 day)
    - Weekly (5 days)
    - Monthly (22 days)
    """
    
    def __init__(
        self,
        daily_lag: int = HAR_DAILY_LAG,
        weekly_lag: int = HAR_WEEKLY_LAG,
        monthly_lag: int = HAR_MONTHLY_LAG,
    ):
        """
        Initialize HAR model.
        
        Parameters
        ----------
        daily_lag : int
            Daily lag (typically 1)
        weekly_lag : int
            Weekly lag (typically 5)
        monthly_lag : int
            Monthly lag (typically 22)
        """
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        
        self.model = None
        self.fitted = False
        self.coefficients = None
        self.intercept = None
    
    @staticmethod
    def compute_realized_volatility(returns: pd.Series, window: int = 1) -> pd.Series:
        """
        Compute realized volatility as rolling standard deviation.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        window : int
            Rolling window size
            
        Returns
        -------
        pd.Series
            Realized volatility
        """
        return returns.rolling(window).std()
    
    def prepare_features(self, returns: pd.Series) -> pd.DataFrame:
        """
        Prepare HAR features from returns.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
            
        Returns
        -------
        pd.DataFrame
            Features with columns [daily, weekly, monthly]
        """
        # Compute realized volatility
        rv = self.compute_realized_volatility(returns, window=1).abs()
        
        # Create feature dataframe
        features = pd.DataFrame({
            "daily": rv.shift(self.daily_lag),
            "weekly": rv.rolling(self.weekly_lag).mean().shift(1),
            "monthly": rv.rolling(self.monthly_lag).mean().shift(1),
        })
        
        return features
    
    def fit(self, returns: pd.Series) -> None:
        """
        Fit HAR model.
        
        Parameters
        ----------
        returns : pd.Series
            Return series for training
        """
        logger.info("Fitting HAR model...")
        
        # Prepare features and target
        features = self.prepare_features(returns)
        rv = self.compute_realized_volatility(returns, window=1).abs()
        
        # Remove NaN rows
        valid_idx = ~(features.isna().any(axis=1) | rv.isna())
        X = features[valid_idx].values
        y = rv[valid_idx].values
        
        # Fit linear regression
        self.model = LinearRegression()
        self.model.fit(X, y)
        
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        self.fitted = True
        
        logger.info(
            f"HAR model fitted. Coefficients: "
            f"daily={self.coefficients[0]:.4f}, "
            f"weekly={self.coefficients[1]:.4f}, "
            f"monthly={self.coefficients[2]:.4f}"
        )
    
    def forecast(self, returns: pd.Series, steps: int = 1) -> np.ndarray:
        """
        Forecast realized volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Return series (for feature computation)
        steps : int
            Number of steps ahead (1-step supported)
            
        Returns
        -------
        np.ndarray
            Forecasted volatility
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if steps != 1:
            logger.warning("Multi-step forecast not fully supported. Using 1-step.")
            steps = 1
        
        # Compute features from latest data
        features = self.prepare_features(returns)
        latest_features = features.iloc[-1:].values
        
        forecast = self.model.predict(latest_features)
        return forecast[0]
    
    def get_r_squared(self, returns: pd.Series) -> float:
        """
        Get R-squared on training data.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
            
        Returns
        -------
        float
            R-squared value
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        features = self.prepare_features(returns)
        rv = self.compute_realized_volatility(returns, window=1).abs()
        
        valid_idx = ~(features.isna().any(axis=1) | rv.isna())
        X = features[valid_idx].values
        y = rv[valid_idx].values
        
        return self.model.score(X, y)
