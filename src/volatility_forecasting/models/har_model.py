"""HAR (Heterogeneous Autoregressive) model with realized volatility."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Optional, Literal

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
    
    Supports multiple volatility proxies:
    - returns.abs() [default]
    - Parkinson
    - Garman-Klass
    - Yang-Zhang
    """
    
    def __init__(
        self,
        daily_lag: int = HAR_DAILY_LAG,
        weekly_lag: int = HAR_WEEKLY_LAG,
        monthly_lag: int = HAR_MONTHLY_LAG,
        proxy_type: Literal['abs', 'parkinson', 'garman_klass', 'yang_zhang'] = 'abs',
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
        proxy_type : str
            Volatility proxy type: 'abs', 'parkinson', 'garman_klass', 'yang_zhang'
        """
        self.daily_lag = daily_lag
        self.weekly_lag = weekly_lag
        self.monthly_lag = monthly_lag
        self.proxy_type = proxy_type
        
        self.model = None
        self.fitted = False
        self.coefficients = None
        self.intercept = None
        self.rv_series = None  # Store RV series for forecasting
    
    def compute_realized_volatility(self, 
                                    returns: pd.Series = None,
                                    ohlc_data: pd.DataFrame = None,
                                    window: int = 1) -> pd.Series:
        """
        Compute realized volatility proxy.
        
        Parameters
        ----------
        returns : pd.Series, optional
            Return series (for 'abs' proxy)
        ohlc_data : pd.DataFrame, optional
            OHLC data (for Parkinson, Garman-Klass, Yang-Zhang)
        window : int
            Rolling window size
            
        Returns
        -------
        pd.Series
            Realized volatility proxy
        """
        if self.proxy_type == 'abs':
            if returns is None:
                raise ValueError("returns required for 'abs' proxy")
            rv = returns.abs()
        
        elif self.proxy_type == 'parkinson':
            if ohlc_data is None:
                raise ValueError("ohlc_data required for Parkinson proxy")
            from ..utils.volatility_proxies import parkinson_volatility
            rv = parkinson_volatility(
                ohlc_data['High'],
                ohlc_data['Low'],
                periods=1
            )
        
        elif self.proxy_type == 'garman_klass':
            if ohlc_data is None:
                raise ValueError("ohlc_data required for Garman-Klass proxy")
            from ..utils.volatility_proxies import garman_klass_volatility
            rv = garman_klass_volatility(
                ohlc_data['High'],
                ohlc_data['Low'],
                ohlc_data['Open'],
                ohlc_data['Close'],
                periods=1
            )
        
        elif self.proxy_type == 'yang_zhang':
            if ohlc_data is None:
                raise ValueError("ohlc_data required for Yang-Zhang proxy")
            from ..utils.volatility_proxies import yang_zhang_volatility
            rv = yang_zhang_volatility(
                ohlc_data['Open'],
                ohlc_data['High'],
                ohlc_data['Low'],
                ohlc_data['Close'],
                periods=5
            )
        
        else:
            raise ValueError(f"Unknown proxy_type: {self.proxy_type}")
        
        return rv
    
    def prepare_features(self, rv: pd.Series) -> pd.DataFrame:
        """
        Prepare HAR features from realized volatility series.
        
        Parameters
        ----------
        rv : pd.Series
            Realized volatility series
            
        Returns
        -------
        pd.DataFrame
            Features with columns [daily, weekly, monthly]
        """
        # Create feature dataframe
        features = pd.DataFrame({
            "daily": rv.shift(self.daily_lag),
            "weekly": rv.rolling(self.weekly_lag).mean().shift(1),
            "monthly": rv.rolling(self.monthly_lag).mean().shift(1),
        })
        
        return features
    
    def fit(self, rv: pd.Series) -> None:
        """
        Fit HAR model.
        
        Parameters
        ----------
        rv : pd.Series
            Realized volatility series for training
        """
        logger.info(f"Fitting HAR model with proxy_type='{self.proxy_type}'...")
        
        # Store RV for later forecasting
        self.rv_series = rv.copy()
        
        # Prepare features and target
        features = self.prepare_features(rv)
        
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
    
    def forecast(self, rv: pd.Series = None, steps: int = 1) -> np.ndarray:
        """
        Forecast realized volatility.
        
        Parameters
        ----------
        rv : pd.Series, optional
            New RV series (for rolling forecast)
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
        
        # Use provided RV or stored series
        if rv is None:
            if self.rv_series is None:
                raise ValueError("No RV series provided and none stored from fit()")
            rv = self.rv_series
        
        # Compute features from latest data
        features = self.prepare_features(rv)
        latest_features = features.iloc[-1:].values
        
        forecast = self.model.predict(latest_features)
        return forecast[0]
    
    def get_r_squared(self, rv: pd.Series = None) -> float:
        """
        Get R-squared on training data.
        
        Parameters
        ----------
        rv : pd.Series, optional
            RV series (uses stored if not provided)
            
        Returns
        -------
        float
            R-squared value
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if rv is None:
            if self.rv_series is None:
                raise ValueError("No RV series provided and none stored from fit()")
            rv = self.rv_series
        
        features = self.prepare_features(rv)
        
        valid_idx = ~(features.isna().any(axis=1) | rv.isna())
        X = features[valid_idx].values
        y = rv[valid_idx].values
        
        return self.model.score(X, y)
